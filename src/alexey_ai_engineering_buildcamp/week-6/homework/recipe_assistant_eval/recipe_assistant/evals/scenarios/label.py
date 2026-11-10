import sys
from pathlib import Path

import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parents[3]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from recipe_assistant.evals.common.labels import (
    FAILURE_CATEGORIES,
    LABELS,
    load_labels,
    load_recipes,
    load_results,
    run_id,
    save_labels,
)
from recipe_assistant.evals.common.paths import LABELS_FILE, RESULTS_FILE

RUBRIC = """
**Good**
- The answer is correct and based on the recipe data, or
- it correctly says it doesn't have a recipe for something not in the collection.

**Bad**
- Makes up a recipe or ingredients not in the dataset → `hallucination`
- Gives cooking advice the recipe data doesn't support (e.g. substitution tips) → `invented-advice`
- Says it can't help when a matching recipe exists → `false-not-found`
- Wrong details from the recipe (times, ingredients, tags) → `incorrect-details`
"""


def system_instructions() -> str:
    try:
        from recipe_assistant.prompts import INSTRUCTIONS

        return INSTRUCTIONS
    except Exception as exc:
        return f"(could not import instructions: {exc})"


def mentioned_recipes(text: str, recipes: list[dict]) -> list[str]:
    """Recipe names that appear in the answer, so they can be checked against the data."""
    text = text.lower()
    found = []
    for r in recipes:
        name = r["name"].lower()
        short = (
            name.split(" ", 1)[1] if " " in name else name
        )  # "lava cake", "green curry"
        if name in text or short in text:
            found.append(r["name"])
    return found


def show_recipe(r: dict) -> None:
    st.markdown(
        f"**[{r['id']}] {r['name']}**: {r['cuisine']}, {r['difficulty']} · "
        f"prep {r['prep_time']} min · cook {r['cook_time']} min · serves {r['servings']}"
    )
    st.markdown("**Ingredients:** " + ", ".join(r["ingredients"]))
    st.markdown("**Instructions:** " + r["instructions"])
    st.markdown("**Tags:** " + ", ".join(r["tags"]))


def first_unlabeled(runs: list[dict], labels: dict[str, dict]) -> int:
    for i, r in enumerate(runs):
        if run_id(r) not in labels:
            return i
    return len(runs) - 1


def main() -> None:
    st.set_page_config(page_title="Label recipe assistant runs", layout="wide")
    st.title("Label recipe assistant runs")

    if not RESULTS_FILE.exists():
        st.error(f"{RESULTS_FILE.name} not found. Run `make run_scenarios` first.")
        return

    if "runs" not in st.session_state:
        st.session_state.runs = load_results()
        st.session_state.labels = load_labels()
        st.session_state.idx = first_unlabeled(
            st.session_state.runs, st.session_state.labels
        )

    runs: list[dict] = st.session_state.runs
    labels: dict[str, dict] = st.session_state.labels
    idx: int = st.session_state.idx
    recipes = load_recipes()

    # --- sidebar: progress and navigation ---------------------------------
    done = sum(1 for r in runs if run_id(r) in labels)
    st.sidebar.progress(done / len(runs), text=f"{done}/{len(runs)} labeled")
    n_good = sum(1 for v in labels.values() if v["label"] == "good")
    st.sidebar.write(f"good: {n_good} · bad: {done - n_good}")

    categories: dict[str, int] = {}
    for v in labels.values():
        if v["label"] == "bad":
            cat = v.get("failure_category") or "(none)"
            categories[cat] = categories.get(cat, 0) + 1
    if categories:
        st.sidebar.caption("Failure categories")
        for cat, count in sorted(categories.items(), key=lambda kv: -kv[1]):
            st.sidebar.write(f"- {cat}: {count}")

    def option_label(i: int) -> str:
        lab = labels.get(run_id(runs[i]))
        mark = {"good": "✅", "bad": "❌"}.get(lab["label"], "·") if lab else "·"
        return f"{mark} {i + 1}. {runs[i]['question'][:50]}"

    choice = st.sidebar.selectbox(
        "Jump to", list(range(len(runs))), index=idx, format_func=option_label
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
    st.sidebar.caption(f"Saving to `{LABELS_FILE.name}`")

    run = runs[idx]
    key = run_id(run)
    existing = labels.get(key)
    st.caption(f"{idx + 1}/{len(runs)} · {run['category']} · {run['type']}")

    # --- 1. question and answer -------------------------------------------
    st.header("User question")
    st.markdown(run["question"])

    with st.expander("System instructions"):
        st.code(system_instructions(), language="markdown", wrap_lines=True)

    st.header("Response")
    st.markdown(run["output"])

    c1, c2, c3 = st.columns(3)
    c1.metric("time (s)", run["execution_time"])
    c2.metric("tokens", run["tokens"]["total_tokens"])
    c3.metric("cost ($)", run["cost"])

    # --- 2. ground truth to check against ---------------------------------
    st.subheader("Recipe data")
    names = [r["name"] for r in recipes]
    selected = st.multiselect(
        "Recipes to check against (auto-selected from names in the response)",
        names,
        default=mentioned_recipes(run["output"], recipes),
        key=f"recipes-{key}",
    )
    for r in recipes:
        if r["name"] in selected:
            with st.container(border=True):
                show_recipe(r)
    with st.expander("All recipe names in the collection"):
        st.write(", ".join(names))

    # --- 3. decision ------------------------------------------------------
    st.divider()
    st.header("Decision")
    with st.expander("Rubric", expanded=done == 0):
        st.markdown(RUBRIC)

    label = st.radio(
        "Is this response good?",
        LABELS,
        index=LABELS.index(existing["label"]) if existing else None,
        horizontal=True,
        key=f"label-{key}",
    )
    cat_default = existing.get("failure_category", "") if existing else ""
    category = st.selectbox(
        "Why did it fail? (only for bad)",
        FAILURE_CATEGORIES,
        index=FAILURE_CATEGORIES.index(cat_default)
        if cat_default in FAILURE_CATEGORIES
        else 0,
        disabled=label != "bad",
        key=f"cat-{key}",
    )
    comments = st.text_area(
        "Comments",
        value=existing.get("comments", "") if existing else "",
        key=f"comments-{key}",
    )

    missing_category = label == "bad" and not category
    if missing_category:
        st.warning("Pick a failure category for a bad response.")

    if st.button(
        "Save and next", type="primary", disabled=label is None or missing_category
    ):
        labels[key] = {
            "id": key,
            "label": label,
            "failure_category": category if label == "bad" else "",
            "comments": comments,
        }
        save_labels(labels)
        order = list(range(idx + 1, len(runs))) + list(range(idx))
        nxt = next((i for i in order if run_id(runs[i]) not in labels), None)
        if nxt is None:
            st.success("All items labeled. Run `make csv` to export to results.csv.")
        else:
            st.session_state.idx = nxt
            st.rerun()


main()
