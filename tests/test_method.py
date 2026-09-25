import copy
import json

import pytest

from repair.harness.context import WorkingContext, parse_turn
from repair.pairs.datasets import successful_turns, top_k_pairs
from repair.pairs.induction import diagnose, reassign
from repair.pairs.repair import both_yes, candidate_error, grounded, repair_assignment
from repair.pairs.schema import Rubric, eligible_turns, failure_pool, specificity_error


def test_context_keeps_all_actions_and_bounds_cited_evidence():
    context = WorkingContext("question", "answer", max_evidence=2, history_chars=20)
    context.constraints = [
        {"cid": "c1", "text": "constraint", "satisfied": True, "support": ["d1", "d2", "d3"]}
    ]
    for i in range(3):
        context.note_document(
            {"tool": "get_document", "args": {"docid": f"d{i + 1}", "goal": "goal"}},
            ["x" * 500],
            "y" * 500,
        )
    assert len(context.evidence) == 2
    assert [item["docid"] for item in context.evidence] == ["d2", "d3"]
    assert all(len(item["fact"]) == 300 for item in context.evidence)
    assert len(context.history) == 3
    assert all(len(item["result"]) == 20 for item in context.history)
    rendered = context.render("a full latest observation")
    assert rendered.count("# ") == 8
    assert "a full latest observation" not in context.render("the next observation")


def test_uncited_evidence_is_evicted_first():
    context = WorkingContext("q", "a", max_evidence=2)
    context.constraints = [
        {"cid": "c1", "text": "constraint", "satisfied": True, "support": ["d1"]}
    ]
    for i in range(3):
        context.note_document(
            {"tool": "get_document", "args": {"docid": f"d{i + 1}", "goal": "goal"}},
            [f"fact {i}"],
            "result",
        )
    assert [row["docid"] for row in context.evidence] == ["d1", "d3"]


def test_turn_round_trip_and_invalid_structure(turn_factory):
    turn = turn_factory()
    assert parse_turn(turn.render()) == turn
    with pytest.raises(ValueError):
        parse_turn(turn.render() + "\n" + turn.render())
    with pytest.raises(ValueError):
        parse_turn('<state>[]</state><action>{"tool":"search","args":{"query":"x"}}</action>')


def test_failure_selection_requires_a_real_incorrect_verdict(trajectory):
    assert failure_pool([trajectory], []) == []
    assert (
        failure_pool(
            [trajectory], [{"id": trajectory["id"], "correct": False, "judge_error": True}]
        )
        == []
    )
    assert failure_pool([trajectory], [{"id": trajectory["id"], "correct": False}]) == [trajectory]
    trajectory["turns"][0]["forced_finish"] = True
    assert eligible_turns(trajectory) == []


def test_successful_sft_keeps_legal_forced_finish_turns(trajectory):
    trajectory["turns"][0]["forced_finish"] = True
    samples = successful_turns([trajectory], [{"id": trajectory["id"], "correct": True}])
    assert len(samples) == 1


def test_grounding_requires_at_least_half_including_odd_counts():
    assert not grounded("Northvale Westbridge Southbank", "Northvale")
    assert grounded("Northvale Westbridge Southbank", "Northvale and Southbank")
    assert not grounded("123", "123")


def test_nonterminal_checks_new_documents_and_duplicate_actions(trajectory, turn_factory):
    source = parse_turn(trajectory["turns"][0]["completion"])
    prompt = trajectory["turns"][0]["prompt"]
    assert (
        candidate_error(source, source, source.render(), prompt, "Northvale", 10000)
        == "identical_action"
    )
    new = turn_factory("get_document", docid="unknown_doc", goal="read")
    assert (
        candidate_error(new, source, new.render(), prompt, "Northvale", 10000)
        == "unseen_document_id"
    )
    new = turn_factory("search", query="Northvale")
    assert candidate_error(new, source, new.render(), prompt, "Northvale", 10000) == "answer_leak"


def test_first_accepted_candidate_and_information_isolation(
    recipe, trajectory, turn_factory, fake_client
):
    rubric = Rubric(
        "read-result", "A useful result is visible.", "Read the result.", "The result is read."
    )
    invalid = turn_factory("get_document", docid="unknown_doc", goal="read").render()
    valid = turn_factory("get_document", docid="doc_a", goal="read").render()
    repairer = fake_client([invalid, valid])
    validator = fake_client([json.dumps({"repairs_per_rubric": 1, "natural": 1})])
    assignment = {"trajectory_id": trajectory["id"], "turn": 0, "rubric_id": rubric.id}
    pair, attempts = repair_assignment(
        trajectory,
        assignment,
        rubric,
        repairer=repairer,
        validator=validator,
        reference="Hiddenanswer",
        config=recipe,
    )
    assert pair is not None and len(attempts) == 2
    assert len(validator.requests) == 1
    assert pair["prompt"] == trajectory["turns"][0]["prompt"]
    assert rubric.hint not in pair["prompt"]["system"]
    assert pair["rejected"] == trajectory["turns"][0]["completion"]
    for messages, _ in repairer.requests:
        assert "Hiddenanswer" not in str(messages)
        assert trajectory["turns"][0]["completion"] not in str(messages)
    assert "Hiddenanswer" not in str(validator.requests)
    assert rubric.hint in repairer.requests[0][0][0]["content"]


def test_no_rubric_skips_guidance_and_validator(recipe, trajectory, turn_factory, fake_client):
    candidate = turn_factory("get_document", docid="doc_a", goal="read").render()
    client = fake_client([candidate])
    pair, _ = repair_assignment(
        trajectory,
        {"trajectory_id": trajectory["id"], "turn": 0},
        None,
        repairer=client,
        validator=None,
        reference="Hiddenanswer",
        config=recipe,
    )
    assert pair is not None
    assert client.requests[0][0][0]["content"] == trajectory["turns"][0]["prompt"]["system"]


def test_validator_fails_closed():
    assert both_yes({"repairs_per_rubric": "yes", "natural": 1})
    assert not both_yes({"repairs_per_rubric": "yes", "natural": "probably"})
    assert not both_yes({"repairs_per_rubric": 1})
    assert not both_yes([])


def test_reassignment_preserves_multiple_rubrics(recipe, trajectory, fake_client):
    rubrics = [
        Rubric("a", "trigger", "hint", "criterion"),
        Rubric("b", "trigger", "hint", "criterion"),
    ]
    client = fake_client(
        [
            json.dumps(
                {
                    "labels": [
                        {"step": 0, "family": "a"},
                        {"step": 0, "family": "b"},
                        {"step": 0, "family": "a"},
                        {"step": 99, "family": "b"},
                    ]
                }
            )
        ]
    )
    assigned = reassign(trajectory, rubrics, client, recipe)
    assert [row["rubric_id"] for row in assigned] == ["a", "b"]


def test_diagnosis_has_no_reference_answer(recipe, trajectory, fake_client):
    client = fake_client(
        [
            json.dumps(
                {
                    "findings": [
                        {
                            "step": 0,
                            "issue": "unread-evidence",
                            "trigger": "Useful evidence has not been read.",
                            "hint": "Read the evidence.",
                            "success_criteria": "The next action reads the visible result.",
                        }
                    ]
                }
            )
        ]
    )
    assert len(diagnose(trajectory, client, recipe)) == 1
    assert trajectory["answer"] in str(client.requests)
    assert "Hiddenanswer" not in str(client.requests)
    assert specificity_error("Read a result from 2026.", set()) == "specific_number"


def test_top_k_uses_assignment_counts_not_pair_counts():
    rubrics = [Rubric("a", "t", "h", "c"), Rubric("b", "t", "h", "c")]
    assignments = [
        {"trajectory_id": "t", "turn": 0, "rubric_id": "a"},
        {"trajectory_id": "t", "turn": 1, "rubric_id": "a"},
        {"trajectory_id": "t", "turn": 2, "rubric_id": "b"},
    ]
    pairs = [
        {"rubric_id": "b", "id": "p1"},
        {"rubric_id": "b", "id": "p2"},
        {"rubric_id": "a", "id": "p3"},
    ]
    selected, names = top_k_pairs(pairs, assignments, rubrics, 1)
    assert names == ["a"] and selected == [pairs[2]]
    duplicate = copy.deepcopy(assignments) + [assignments[0]]
    with pytest.raises(ValueError):
        top_k_pairs(pairs, duplicate, rubrics, 1)
