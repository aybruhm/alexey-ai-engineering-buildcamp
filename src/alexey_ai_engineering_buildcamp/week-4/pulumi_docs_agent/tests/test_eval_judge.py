import pytest

from ..evals.code_evaluators.evaluators import evaluate
from ..evals.common.retrieval_replay import truncate_for_judge
from ..evals.judge.evaluate import category_metrics, cohen_kappa, metrics
from ..evals.judge.judge import JUDGE_PROMPT, format_judge_prompt


def test_format_judge_prompt_includes_scenario_expectation():
    row = {
        "id": "broad-question",
        "question": "How do I get started?",
        "turns": ["How do I get started?"],
        "phrasing": "broad",
        "expected_behavior": "answers_or_asks_clarification",
        "expected_confidence_max": 0.0,
        "notes": "Ask which cloud and language before giving a generic answer.",
        "tool_calls": [],
        "references": [],
        "answer": "Install the CLI.",
        "found_answer": False,
        "confidence": 0.0,
        "confidence_explanation": "",
        "answer_type": "how-to",
        "followup_questions": [],
    }

    prompt = format_judge_prompt(row)

    assert "<SCENARIO_EXPECTATION>" in prompt
    assert "phrasing: broad" in prompt
    assert "expected_behavior: answers_or_asks_clarification" in prompt
    assert "expected_confidence_max: 0.0" in prompt
    assert "Ask which cloud and language" in prompt


def test_metrics_reports_chance_corrected_agreement():
    rows = [
        {"human_label": "bad", "judge_label": "bad", "human_category": "incorrect"},
        {"human_label": "bad", "judge_label": "good", "human_category": "incorrect"},
        {"human_label": "good", "judge_label": "bad", "human_category": ""},
        {"human_label": "good", "judge_label": "good", "human_category": ""},
    ]

    result = metrics(rows)

    assert result["accuracy"] == pytest.approx(0.5)
    assert result["precision_bad"] == pytest.approx(0.5)
    assert result["recall_bad"] == pytest.approx(0.5)
    assert result["kappa"] == pytest.approx(0.0)
    assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (1, 1, 1, 1)


def test_category_metrics_tracks_missed_failure_categories():
    rows = [
        {"human_label": "bad", "judge_label": "bad", "human_category": "incorrect"},
        {"human_label": "bad", "judge_label": "good", "human_category": "incorrect"},
        {"human_label": "bad", "judge_label": "good", "human_category": "incomplete"},
        {"human_label": "good", "judge_label": "good", "human_category": ""},
    ]

    result = category_metrics(rows)

    assert result["incorrect"] == {"caught": 1, "total": 2}
    assert result["incomplete"] == {"caught": 0, "total": 1}


def test_cohen_kappa_is_one_for_perfect_agreement():
    assert cohen_kappa(2, 0, 0, 2) == pytest.approx(1.0)


def test_truncate_for_judge_keeps_both_ends():
    text = "start" + ("x" * 100) + "end"

    result = truncate_for_judge(text, limit=20)

    assert result.startswith("start")
    assert result.endswith("end")
    assert "truncated for the judge" in result
    assert "chars omitted" in result


def test_judge_allows_required_search_narration():
    assert "Search and retrieval narration is allowed" in JUDGE_PROMPT
    assert "Leaked implementation details" in JUDGE_PROMPT


def test_code_evaluator_does_not_penalize_search_narration():
    output = {
        "answer": "Iteration 1: I searched for the relevant documentation.",
        "references": [],
        "invented_references": [],
        "found_answer": False,
        "confidence": 0.0,
        "tool_calls": [],
    }

    result = evaluate(output=output, reference={"expected_behavior": "says_not_found"})

    assert result["label"] == "pass"
    assert "no_process_narration" not in result["explanation"]
