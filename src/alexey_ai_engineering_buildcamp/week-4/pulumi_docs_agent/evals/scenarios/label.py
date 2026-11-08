import json
import sys
from pathlib import Path

import streamlit as st

WEEK_DIR = Path(__file__).resolve().parents[3]
if str(WEEK_DIR) not in sys.path:
    sys.path.insert(0, str(WEEK_DIR))

from pulumi_docs_agent.evals.common.paths import (  # noqa: E402
    OBSERVABILITY_DATA_DIR,
    SCENARIO_RESULTS_DIR,
)

GITHUB_DOCS = "https://github.com/pulumi/docs/blob/master/content/"
FAILURE_CATEGORIES = [
    "",
    "hallucination",
    "wrong-scope",
    "incorrect",
    "incomplete",
    "false-not-found",
    "mixes-own-knowledge",
    "wrong-language",
    "other",
]


def load_runs(path: Path) -> list[dict]:
    text = path.read_text()
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def labels_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_results.json")


def load_labels(path: Path) -> dict[str, dict]:
    p = labels_path(path)
    if not p.exists():
        return {}
    return {item["id"]: item for item in json.loads(p.read_text())}


def save_labels(path: Path, labels: dict[str, dict]) -> None:
    labels_path(path).write_text(json.dumps(list(labels.values()), indent=2))


def doc_url(file_path: str) -> str:
    return GITHUB_DOCS + file_path


def system_instructions() -> str:
    try:
        from pulumi_docs_agent.prompts import INSTRUCTIONS

        return INSTRUCTIONS
    except Exception as exc:  # the app should still work if the import fails
        return f"(could not import instructions: {exc})"


def first_unlabeled(runs: list[dict], labels: dict[str, dict]) -> int:
    for i, r in enumerate(runs):
        if r["id"] not in labels:
            return i
    return len(runs) - 1


def main() -> None:
    st.set_page_config(page_title="Label agent runs", layout="wide")
    st.title("Label agent runs")

    files = sorted(
        [
            *SCENARIO_RESULTS_DIR.glob("*.jsonl"),
            *SCENARIO_RESULTS_DIR.glob("*.json"),
            *OBSERVABILITY_DATA_DIR.glob("traces_*.jsonl"),
        ],
        reverse=True,
    )
    files = [f for f in files if not f.stem.endswith("_results")]
    if not files:
        st.error(f"No result files in {SCENARIO_RESULTS_DIR}. Run `make scenarios` first.")
        return
    path = st.sidebar.selectbox("Input file", files, format_func=lambda p: p.name)

    # Reload state when the file changes.
    if st.session_state.get("loaded_path") != str(path):
        st.session_state.loaded_path = str(path)
        st.session_state.runs = load_runs(path)
        st.session_state.labels = load_labels(path)
        st.session_state.idx = first_unlabeled(
            st.session_state.runs, st.session_state.labels
        )

    runs = st.session_state.runs
    labels = st.session_state.labels
    idx = st.session_state.idx

    # --- sidebar: progress and navigation ---------------------------------
    done = sum(1 for r in runs if r["id"] in labels)
    st.sidebar.progress(done / len(runs), text=f"{done}/{len(runs)} labeled")
    n_good = sum(1 for v in labels.values() if v["label"] == "good")
    st.sidebar.write(f"good: {n_good} · not good: {done - n_good}")
    options = list(range(len(runs)))
    choice = st.sidebar.selectbox(
        "Jump to",
        options,
        index=idx,
        format_func=lambda i: (
            f"{'✓' if runs[i]['id'] in labels else '·'} {runs[i]['id']}"
        ),
    )
    if choice != idx:
        st.session_state.idx = idx = choice
        st.rerun()
    col_prev, col_next = st.sidebar.columns(2)
    if col_prev.button("← Prev", disabled=idx == 0):
        st.session_state.idx -= 1
        st.rerun()
    if col_next.button("Next →", disabled=idx >= len(runs) - 1):
        st.session_state.idx += 1
        st.rerun()

    run = runs[idx]
    existing = labels.get(run["id"])
    st.caption(
        f"{idx + 1}/{len(runs)} · `{run['id']}` · {run.get('group')} · {run.get('phrasing')}"
    )

    if run.get("error"):
        st.error(f"The run failed: {run['error']}")

    # --- 1. question -------------------------------------------------------
    st.header("User question")
    turns = run.get("turns") or [run["question"]]
    if len(turns) > 1:
        for n, t in enumerate(turns, 1):
            st.markdown(f"**Turn {n}:** {t}")
    else:
        st.markdown(run["question"])

    with st.expander("System instructions"):
        st.code(system_instructions(), language="markdown", wrap_lines=True)

    # --- 2. answer ---------------------------------------------------------
    st.header("Answer")
    st.markdown(run.get("answer", "_no answer_"))

    # --- 3. structured fields ---------------------------------------------
    st.subheader("References")
    refs = run.get("references", [])
    if not refs:
        st.write("_none_")
    for ref in refs:
        bad = ref in run.get("invented_references", [])
        suffix = " ⚠️ **not in the corpus**" if bad else ""
        st.markdown(f"- [{ref}]({doc_url(ref)}){suffix}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("found_answer", str(run.get("found_answer")))
    c2.metric("confidence", run.get("confidence"))
    c3.metric("answer_type", run.get("answer_type"))
    c4.metric("tool calls", run.get("n_tool_calls"))
    st.caption(f"Confidence explanation: {run.get('confidence_explanation')}")

    if run.get("followup_questions"):
        with st.expander("Follow-up questions"):
            for q in run["followup_questions"]:
                st.markdown(f"- {q}")

    with st.expander("Self-checks"):
        for c in run.get("checks", []):
            st.markdown(
                f"{'✅' if c['passed'] else '❌'} **{c['rule']}**: {c['explanation']}"
            )

    with st.expander("Tool calls"):
        for n, tc in enumerate(run.get("tool_calls", []), 1):
            st.markdown(f"{n}. `{tc['name']}`")
            st.json(tc["args"], expanded=False)

    # Scenario context. Shown last so it doesn't bias reading the answer first.
    with st.expander("Scenario notes (designed expectation, may bias you)"):
        st.write(f"expected behavior: `{run.get('expected_behavior')}`")
        st.write(f"expected references: {run.get('expected_ref_prefixes')}")
        st.write(run.get("notes", ""))

    # --- 4. decision, at the bottom ---------------------------------------
    st.divider()
    st.header("Decision")
    st.caption(
        "Good = correct and grounded in the retrieved docs, or correctly says "
        "it is out of scope / not covered. Not good = hallucination, answers "
        "something it should have declined, wrong info, or false 'not found'."
    )
    key = run["id"]
    label_options = ["good", "not good"]
    label = st.radio(
        "Is this a good run?",
        label_options,
        index=label_options.index(existing["label"]) if existing else None,
        horizontal=True,
        key=f"label-{key}",
    )
    cat_default = existing.get("failure_category", "") if existing else ""
    category = st.selectbox(
        "Failure category (only if not good; free to leave blank)",
        FAILURE_CATEGORIES,
        index=FAILURE_CATEGORIES.index(cat_default)
        if cat_default in FAILURE_CATEGORIES
        else 0,
        key=f"cat-{key}",
    )
    comments = st.text_area(
        "Comments",
        value=existing.get("comments", "") if existing else "",
        key=f"comments-{key}",
    )

    if st.button("Save and next", type="primary", disabled=label is None):
        labels[key] = {
            "id": key,
            "label": label,
            "failure_category": category if label == "not good" else "",
            "comments": comments,
        }
        save_labels(path, labels)
        # Next unlabeled item after this one, wrapping around.
        order = list(range(idx + 1, len(runs))) + list(range(idx))
        nxt = next((i for i in order if runs[i]["id"] not in labels), None)
        if nxt is None:
            st.session_state.idx = idx
            st.success("All items labeled.")
        else:
            st.session_state.idx = nxt
            st.rerun()


main()
