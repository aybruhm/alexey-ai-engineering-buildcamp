import argparse
import json
import re
from pathlib import Path

LEAK_SENTINEL = "You're a documentation assistant."
MAX_TOOL_CALLS = 10
NARRATION = re.compile(r"\bIteration\s*[123]\b", re.IGNORECASE)

HARD_CHECKS = (
    "answer_not_empty",
    "no_invented_references",
    "no_duplicate_references",
    "found_answer_has_references",
    "confidence_in_range",
    "declines_when_expected",
    "no_prompt_leak",
    "tool_call_budget",
)
SOFT_CHECKS = ("expected_reference_hit", "no_process_narration")


def evaluate(output, reference=None, input=None, metadata=None):
    output = output or {}
    reference = reference or {}
    answer = str(output.get("answer", ""))
    refs = list(output.get("references", []))
    found = output.get("found_answer")
    confidence = output.get("confidence")
    expected = reference.get("expected_behavior")
    prefixes = reference.get("expected_ref_prefixes") or []
    max_conf = reference.get("expected_confidence_max")

    results = {
        "answer_not_empty": (bool(answer.strip()), "the answer is empty"),
        "no_invented_references": (
            not output.get("invented_references"),
            f"cited paths not in the corpus: {output.get('invented_references')}",
        ),
        "no_duplicate_references": (
            len(refs) == len(set(refs)),
            f"duplicate references: {refs}",
        ),
        "found_answer_has_references": (
            not found or bool(refs),
            "found_answer=True but no references were given",
        ),
        "confidence_in_range": (
            isinstance(confidence, (int, float)) and 0.0 <= confidence <= 1.0,
            f"confidence out of range: {confidence}",
        ),
        "declines_when_expected": (
            expected != "says_not_found"
            or (found is False and (max_conf is None or confidence <= max_conf)),
            f"expected a refusal, got found_answer={found}, confidence={confidence}",
        ),
        "no_prompt_leak": (
            LEAK_SENTINEL not in answer,
            "the answer repeats the system instructions",
        ),
        "tool_call_budget": (
            len(output.get("tool_calls", [])) <= MAX_TOOL_CALLS,
            f"{len(output.get('tool_calls', []))} tool calls (budget {MAX_TOOL_CALLS})",
        ),
        "expected_reference_hit": (
            expected != "answers"
            or not prefixes
            or any(r.startswith(p) for r in refs for p in prefixes),
            f"no reference under {prefixes}; cited: {refs}",
        ),
        "no_process_narration": (
            not NARRATION.search(answer),
            "the answer narrates its own search steps",
        ),
    }

    hard_failed = [c for c in HARD_CHECKS if not results[c][0]]
    soft_failed = [c for c in SOFT_CHECKS if not results[c][0]]
    total = len(HARD_CHECKS) + len(SOFT_CHECKS)
    passed = total - len(hard_failed) - len(soft_failed)

    parts = [f"FAIL {c}: {results[c][1]}" for c in hard_failed]
    parts += [f"WARN {c}: {results[c][1]}" for c in soft_failed]
    return {
        "label": "fail" if hard_failed else "pass",
        "score": round(passed / total, 3),
        "explanation": "; ".join(parts) or "all checks passed",
    }


def to_inputs(row: dict) -> dict:
    return {
        "input": {"question": row["question"], "turns": row.get("turns")},
        "output": {
            "answer": row.get("answer", ""),
            "references": row.get("references", []),
            "invented_references": row.get("invented_references", []),
            "found_answer": row.get("found_answer"),
            "confidence": row.get("confidence"),
            "tool_calls": row.get("tool_calls", []),
        },
        "reference": {
            "expected_behavior": row.get("expected_behavior"),
            "expected_ref_prefixes": row.get("expected_ref_prefixes", []),
            "expected_confidence_max": row.get("expected_confidence_max"),
        },
        "metadata": {"id": row["id"], "group": row.get("group")},
    }


def load_rows(path: Path) -> list[dict]:
    text = path.read_text()
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        rows = json.loads(text)["rows"]
    return [r for r in rows if not r.get("error")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()

    rows = load_rows(Path(args.data))
    scored = [(r, evaluate(**to_inputs(r))) for r in rows]
    passed = sum(res["label"] == "pass" for _, res in scored)

    for row, res in scored:
        mark = "ok  " if res["label"] == "pass" else "FAIL"
        extra = (
            ""
            if res["explanation"] == "all checks passed"
            else f"  {res['explanation']}"
        )
        print(f"{mark} {row['id']:38} {res['score']:.2f}{extra}")

    print(
        f"\nDeterministic checks: {passed}/{len(scored)} pass ({passed / len(scored):.1%})"
    )
    mean = sum(res["score"] for _, res in scored) / len(scored)
    print(f"Mean score: {mean:.3f}")


if __name__ == "__main__":
    main()
