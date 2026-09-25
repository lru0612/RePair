"""Diagnosis, weighted issue clustering, and fixed-codebook reassignment."""

from __future__ import annotations

from collections import Counter

from repair.io import derive_seed, stable_json
from repair.models import parse_json_reply
from repair.pairs.prompts import (
    FREE_DIAGNOSER_SYSTEM,
    FREE_DIAGNOSER_USER_TEMPLATE,
    REDIAGNOSER_SYSTEM,
    REDIAGNOSER_USER_TEMPLATE,
)
from repair.pairs.schema import (
    Rubric,
    eligible_turns,
    question_anchors,
    read_codebook,
    specificity_error,
    trajectory_view,
)

REWRITE_PROMPT = """Rewrite the supplied cluster of diagnostic findings into general rubrics.
Each rubric has id, trigger, hint, and criterion. The trigger names an observable mistake.
The hint gives a reusable decision rule. The criterion can be checked using only the agent's
input and candidate turn. Separate distinct decision errors, merge equivalent ones, and remove
question-specific names, dates, numbers, identifiers, quoted phrases, and suggested answers.
Return JSON: {"rubrics": [{"id": "kebab-case", "trigger": "...", "hint": "...", "criterion": "..."}]}."""

MERGE_PROMPT = """Consolidate these draft rubrics into one codebook. Merge equivalent rules and
remove duplicates while preserving distinct actionable errors. Each trigger must identify an
observable mistake; each hint must teach a general decision rule; each criterion must be judged
from the current input and candidate turn. Do not introduce question-specific details.
Return JSON: {"rubrics": [{"id": "kebab-case", "trigger": "...", "hint": "...", "criterion": "..."}]}."""


def diagnose(trajectory: dict, client, config: dict) -> list[dict]:
    settings = config["induction"]
    data = parse_json_reply(
        client.generate(
            [
                {
                    "role": "system",
                    "content": FREE_DIAGNOSER_SYSTEM.format(max_findings=settings["max_findings"]),
                },
                {
                    "role": "user",
                    "content": FREE_DIAGNOSER_USER_TEMPLATE.format(**trajectory_view(trajectory)),
                },
            ],
            settings=settings["decode"],
            stage="induction",
            role="diagnoser",
            identity=trajectory["id"],
            seed=derive_seed(config["seed"], trajectory["id"], "diagnose"),
        )
    )
    rows = data.get("findings", []) if isinstance(data, dict) else None
    if not isinstance(rows, list):
        raise ValueError("Expected a findings list")
    valid = {turn["turn"] for turn in eligible_turns(trajectory)}
    anchors = question_anchors(trajectory["question"])
    result, seen = [], set()
    for finding in rows[: settings["max_findings"]]:
        if not isinstance(finding, dict) or type(finding.get("step")) is not int:
            continue
        if finding["step"] not in valid or finding["step"] in seen:
            continue
        try:
            rubric = Rubric.from_dict(
                {
                    "id": finding.get("issue"),
                    "trigger": finding.get("trigger"),
                    "hint": finding.get("hint"),
                    "criterion": finding.get("success_criteria"),
                }
            )
        except ValueError:
            continue
        if any(
            specificity_error(text, anchors)
            for text in (rubric.id, rubric.trigger, rubric.hint, rubric.criterion)
        ):
            continue
        result.append(
            {
                "trajectory_id": trajectory["id"],
                "turn": finding["step"],
                **rubric.as_dict(),
            }
        )
        seen.add(finding["step"])
    return result


def cluster_issues(
    findings: list[dict], settings: dict, seed: int
) -> tuple[list[list[dict]], dict]:
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion
    from sklearn.preprocessing import normalize

    frequency = Counter(row["id"] for row in findings)
    names = sorted(frequency)
    if len(names) < settings["clusters"]:
        raise ValueError(f"Need at least {settings['clusters']} distinct issues for clustering")
    representatives = {}
    for row in findings:
        representatives.setdefault(row["id"], row)
    texts = [name + " " + representatives[name]["hint"] for name in names]
    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
        ]
    ).fit_transform(texts)
    dimensions = min(settings["svd_dimensions"], min(features.shape) - 1)
    if dimensions < 1:
        raise ValueError("Insufficient TF-IDF features")
    embeddings = normalize(
        TruncatedSVD(n_components=dimensions, random_state=seed).fit_transform(features)
    )
    labels = KMeans(
        n_clusters=settings["clusters"], n_init=settings["n_init"], random_state=seed
    ).fit_predict(embeddings, sample_weight=np.array([frequency[name] for name in names]))
    assignment = dict(zip(names, labels))
    groups = [[] for _ in range(settings["clusters"])]
    for finding in findings:
        groups[int(assignment[finding["id"]])].append(finding)
    return groups, {"effective_svd_dimensions": dimensions, "issue_frequencies": dict(frequency)}


def induce_codebook(
    findings: list[dict], failures: list[dict], rewriter, consolidator, config: dict
) -> dict:
    settings = config["induction"]
    groups, stats = cluster_issues(findings, settings, config["seed"])
    drafts = []
    for number, group in enumerate(groups):
        data = parse_json_reply(
            rewriter.generate(
                [
                    {"role": "system", "content": REWRITE_PROMPT},
                    {"role": "user", "content": stable_json(group)},
                ],
                settings=settings["decode"],
                stage="induction",
                role="rewriter",
                identity=f"cluster:{number}",
                seed=derive_seed(config["seed"], "cluster", number),
            )
        )
        drafts.extend(r.as_dict() for r in read_codebook(data["rubrics"]))
    data = parse_json_reply(
        consolidator.generate(
            [
                {"role": "system", "content": MERGE_PROMPT},
                {"role": "user", "content": stable_json(drafts)},
            ],
            settings=settings["decode"],
            stage="induction",
            role="consolidator",
            identity="codebook",
            seed=derive_seed(config["seed"], "codebook"),
        )
    )
    anchors = set().union(*(question_anchors(row["question"]) for row in failures))
    rubrics = [
        r
        for r in read_codebook(data["rubrics"])
        if not any(
            specificity_error(text, anchors) for text in (r.id, r.trigger, r.hint, r.criterion)
        )
    ]
    if not rubrics:
        raise ValueError("No codebook rules passed the specificity check")
    return {"rubrics": [r.as_dict() for r in rubrics], "clustering": stats}


def reassign(trajectory: dict, codebook: list[Rubric], client, config: dict) -> list[dict]:
    codebook_text = "\n\n".join(stable_json(r.as_dict()) for r in codebook)
    data = parse_json_reply(
        client.generate(
            [
                {"role": "system", "content": REDIAGNOSER_SYSTEM.format()},
                {
                    "role": "user",
                    "content": REDIAGNOSER_USER_TEMPLATE.format(
                        n_families=len(codebook),
                        codebook=codebook_text,
                        **trajectory_view(trajectory),
                    ),
                },
            ],
            settings=config["induction"]["decode"],
            stage="reassignment",
            role="reassignment",
            identity=trajectory["id"],
            seed=derive_seed(config["seed"], trajectory["id"], "reassign"),
        )
    )
    valid = {turn["turn"] for turn in eligible_turns(trajectory)}
    names = {rubric.id for rubric in codebook}
    assignments, seen = [], set()
    for label in data.get("labels", []):
        if not isinstance(label, dict):
            continue
        step, family = label.get("step"), label.get("family")
        if type(step) is not int or step not in valid or family not in names:
            continue
        key = (step, family)
        if key not in seen:
            assignments.append(
                {"trajectory_id": trajectory["id"], "turn": step, "rubric_id": family}
            )
            seen.add(key)
    return assignments
