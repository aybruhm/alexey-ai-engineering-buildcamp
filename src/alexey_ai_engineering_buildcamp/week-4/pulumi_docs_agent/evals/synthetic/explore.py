import json
import sys
from pathlib import Path

import streamlit as st

WEEK_DIR = Path(__file__).resolve().parents[3]
if str(WEEK_DIR) not in sys.path:
    sys.path.insert(0, str(WEEK_DIR))

from pulumi_docs_agent.evals.common.paths import SYNTHETIC_DATA_DIR

GITHUB_DOCS = "https://github.com/pulumi/docs/blob/master/content/"
CHECKS = {
    "judge_answer_correctness": "Answer correctness",
    "judge_instruction_following": "Instruction following",
    "judge_trajectory": "Trajectory",
}


@st.cache_data
def load(path: str) -> list[dict]:
    return json.loads(Path(path).read_text())


def agent_instructions() -> str:
    try:
        from pulumi_docs_agent.prompts import INSTRUCTIONS

        return INSTRUCTIONS
    except Exception as exc:
        return f"(could not import instructions: {exc})"


def badge(score: str) -> str:
    return "🟢 good" if score == "good" else "🔴 bad"


def main() -> None:
    st.set_page_config(page_title="Explore judge results", layout="wide")
    st.title("Explore judge results")

    files = sorted(SYNTHETIC_DATA_DIR.glob("*_judged.json"), reverse=True)
    if not files:
        st.error(
            f"No *_judged.json files in {SYNTHETIC_DATA_DIR}. Run make judge-synthetic first."
        )
        return
    path = st.sidebar.selectbox("Judged file", files, format_func=lambda p: p.name)
    entries = load(str(path))

    st.sidebar.subheader("Filters")
    selected = {}
    for key, title in CHECKS.items():
        good = sum(e[key]["score"] == "good" for e in entries)
        selected[key] = st.sidebar.radio(
            f"{title} ({good}/{len(entries)} good)",
            ["all", "good", "bad"],
            horizontal=True,
            key=f"filter-{key}",
        )
    groups = sorted({e["group"] for e in entries})
    group = st.sidebar.selectbox("Area", ["all", *groups])
    persona = st.sidebar.selectbox(
        "Persona", ["all", *sorted({e["phrasing"] for e in entries})]
    )

    def keep(e: dict) -> bool:
        if group != "all" and e["group"] != group:
            return False
        if persona != "all" and e["phrasing"] != persona:
            return False
        return all(v == "all" or e[k]["score"] == v for k, v in selected.items())

    filtered = [e for e in entries if keep(e)]
    st.sidebar.write(f"**{len(filtered)}** of {len(entries)} match")
    if not filtered:
        st.info("No entries match the filters.")
        return

    idx = st.sidebar.number_input("Entry", 1, len(filtered), 1) - 1
    e = filtered[idx]
    st.caption(
        f"{idx + 1}/{len(filtered)} · `{e['id']}` · {e['group']} · {e['phrasing']}"
    )

    cols = st.columns(3)
    for col, (key, title) in zip(cols, CHECKS.items()):
        col.metric(title, badge(e[key]["score"]))

    st.header("Question")
    st.markdown(e["question"])
    st.subheader("Reference answer")
    st.markdown(e["reference_answer"])
    st.markdown(
        f"Source: [{e['filename']}]({GITHUB_DOCS + e['filename']}) "
        f"(lines {e['line_number_start']}–{e['line_number_end']}, "
        f"answer/line overlap {e.get('answer_overlap')})"
    )

    st.header("Agent answer")
    st.markdown(e.get("answer", "_none_"))
    if e.get("references"):
        st.markdown(
            "**References:** "
            + ", ".join(f"[{r}]({GITHUB_DOCS + r})" for r in e["references"])
        )
        st.caption(f"Cited the source page: {e.get('ref_prefix_hit')}")

    st.header("Judge reasoning")
    for key, title in CHECKS.items():
        with st.expander(
            f"{title}: {badge(e[key]['score'])}", expanded=e[key]["score"] == "bad"
        ):
            st.write(e[key]["reasoning"])
            if e[key].get("suggestion"):
                st.markdown(f"**Suggestion:** {e[key]['suggestion']}")

    with st.expander(f"Tool calls ({len(e.get('tool_calls', []))})"):
        for n, tc in enumerate(e.get("tool_calls", []), 1):
            st.markdown(f"{n}. `{tc['name']}`")
            st.json(tc["args"], expanded=False)
    with st.expander("Agent self-checks"):
        for c in e.get("checks", []):
            st.markdown(
                f"{'✅' if c['passed'] else '❌'} **{c['rule']}**: {c['explanation']}"
            )
    with st.expander("Agent instructions (system prompt)"):
        st.code(agent_instructions(), language="markdown", wrap_lines=True)


main()
