import copy

from repair.costs import estimate_budget, summarize_usage
from repair.evaluation import summarize
from repair.harness.rollout import run_query
from repair.models import Reply


class Tokenizer:
    def apply_chat_template(self, *args, **kwargs):
        return [1, 2, 3]

    def encode(self, text, **kwargs):
        return list(text.encode())


class Search:
    def search(self, query):
        return [{"docid": "doc_a", "snippet": "Northvale"}]


def test_truncation_retries_change_seed_and_penalty(recipe, turn_factory, fake_client):
    broken = Reply("partial", "partial", "", "length", 3, 1)
    finish = turn_factory("finish", exact_answer="Northvale").render()
    client = fake_client([broken] * 5 + [finish])
    result = run_query(
        {"id": "q1", "question": "question"},
        0,
        policy=client,
        retrieval=Search(),
        tokenizer=Tokenizer(),
        config=recipe,
        evaluation=True,
        stage="evaluation",
    )
    assert result["status"] == "completed"
    requests = [kwargs for _, kwargs in client.requests]
    assert [r["settings"]["repetition_penalty"] for r in requests] == [1.1, 1.1, 1.1, 1.2, 1.3, 1.4]
    assert len({r["seed"] for r in requests}) == 6
    assert all("max_tokens" not in r["settings"] for r in requests)


def test_collection_cap_and_generation_threshold(recipe, turn_factory, fake_client):
    config = copy.deepcopy(recipe)
    config["rollout"]["force_finish_tokens"] = 8
    search = turn_factory().render()
    finish = turn_factory("finish", exact_answer="Northvale").render()
    client = fake_client([search, finish])
    result = run_query(
        {"id": "q1", "question": "question"},
        0,
        policy=client,
        retrieval=Search(),
        tokenizer=Tokenizer(),
        config=config,
        evaluation=False,
        stage="collection",
    )
    assert result["turns"][1]["forced_finish"]
    assert not result["natural_finish"]
    assert all(request[1]["settings"]["max_tokens"] == 4096 for request in client.requests)


def test_timeout_does_not_make_a_model_request(recipe, fake_client):
    client = fake_client([])
    result = run_query(
        {"id": "q1", "question": "question"},
        0,
        policy=client,
        retrieval=Search(),
        tokenizer=Tokenizer(),
        config=recipe,
        evaluation=True,
        stage="evaluation",
        deadline=0,
    )
    assert result["status"] == "timeout" and client.requests == []


def test_metric_denominators_include_failed_and_missing_runs(trajectory):
    timed_out = copy.deepcopy(trajectory)
    timed_out["id"], timed_out["sample"], timed_out["status"] = "q1:1", 1, "timeout"
    result = summarize(
        [{"id": "q1"}],
        [trajectory, timed_out],
        [{"id": "q1:0", "correct": True}, {"id": "q1:1", "correct": True}],
        3,
    )
    assert result["acc"] == 1 / 3
    assert result["pass_at_3"] == 1
    assert result["missing_runs"] == 1
    assert result["behavior_runs"] == 1


def test_cost_accounting_separates_reader_and_includes_judge():
    rows = [
        {
            "stage": "collection",
            "role": "policy",
            "identity": "q:0:0",
            "prompt_tokens": 10,
            "completion_tokens": 2,
        },
        {
            "stage": "collection",
            "role": "judge",
            "identity": "q:0",
            "prompt_tokens": 20,
            "completion_tokens": 3,
        },
        {
            "stage": "collection",
            "role": "reader",
            "identity": "q:0:0",
            "prompt_tokens": 100,
            "completion_tokens": 50,
        },
        {
            "stage": "repair",
            "role": "repairer",
            "identity": "a:0",
            "prompt_tokens": 4,
            "completion_tokens": 6,
        },
        {
            "stage": "repair",
            "role": "repairer",
            "identity": "a:1",
            "prompt_tokens": 4,
            "completion_tokens": 8,
        },
    ]
    report = summarize_usage(rows, [{"accelerator_hours": 2.0}])
    assert report["stages"]["failure_pool"]["prefill_tokens"] == 30
    assert report["mean_candidate_generated_tokens"] == 7
    assert report["reader_calls_excluded"] == 1
    assert report["mean_rollout_generated_tokens"] == 2
    estimate = estimate_budget(
        {
            "rollouts": 4,
            "mean_prefill_tokens": 10,
            "mean_generated_tokens": 20,
            "assumptions": "Synthetic arithmetic example.",
        }
    )
    assert estimate["prefill_tokens"] == 40 and estimate["estimated"]
