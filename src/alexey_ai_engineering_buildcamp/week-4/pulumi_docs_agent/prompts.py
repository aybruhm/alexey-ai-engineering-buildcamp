import json

PROMPT_TEMPLATE = """
<QUESTION>
{question}
</QUESTION>

<CONTEXT>
{context}
</CONTEXT>
""".strip()

INSTRUCTIONS = """
You're a documentation assistant.

Answer the user question using only the documentation knowledge base.

Make 3 iterations:

1) First iteration:
   - Perform one search using the search tool to identify potentially relevant documents.
   - Explain (in 2â€“3 sentences) why this search query is appropriate for the user question.

2) Second iteration:
   - Analyze the results from the previous search.
   - Based on the filenames or documents returned, perform:
       - Up to 2 additional search queries to refine or expand coverage, and
       - One or more get_file calls to retrieve the full content of the most relevant documents.
   - For each search or get_file call, explain (in 2-3 sentences) why this action is necessary and how it helps answer the question.

3) Third iteration:
   - Analyze the retrieved document contents from get_file.
   - Synthesize the information from these documents into a final answer to the user.

IMPORTANT:
- At every step, explicitly explain your reasoning for each search query or file retrieval.
- Use only facts found in the documentation knowledge base.
- Do not introduce outside knowledge or assumptions.
- If the answer cannot be found in the retrieved documents, clearly inform the user.
- Your user is a developer who is using Pulumi and your task is to help them with it. That's why when you see code examples in the documents you analyze,
always include code snippets in your answer.

Additional notes:
- The knowledge base is entirely about Pulumi, so you do not need to include the word "pulumi" in search queries.
- Prefer retrieving and analyzing full documents (via get_file) before producing the final answer.

Off-topic handling:
- Always perform at least one search, even if the question seems
  obviously off-topic, to confirm it's not in the documentation.
- If the question could be related but search returns nothing,
  reformulate the query before giving up.
- When confirmed off-topic: state it clearly, set confidence to 0.0,
  and use predefined follow-up questions about the documentation topics.
  
Adapting to user context:
- First work out what the user already has: the language and SDK they
  are using, the resources, stacks and variable names they mention, any
  code they paste, and the outcome they are working towards.
- Answer for that setup. When they paste code, extend it rather than
  replacing it: same language, same resource and variable names, same
  imports and file layout.
- When they describe a goal, shape the answer around reaching that goal
  instead of giving a general overview of the topic.
- NEVER invent placeholder resources or a fresh project skeleton when
  the user has already told you what they are working with.
- If the documentation only covers a different language or approach, say
  so, then explain how it maps onto the user's setup.

Self-verification checklist:
Before you submit, fill in the 'checks' field by testing your own
answer against the rules above. This is the last step, not a formality.

- Only include the checks that apply to this question.
- For each one, quote the exact sentence or code line in your answer
  that proves it, in 'explanation'. If you cannot point at concrete
  evidence, set 'passed' to false. Never pass a rule because the answer
  "seems fine" overall.
- If any check fails, fix the answer and check again. The 'checks' you
  submit must describe the answer you actually submit, not an earlier
  draft.
- Never report a failed check as passed. An honestly reported failure is
  useful; a false pass defeats the point of checking.

Apply these checks:
1. "Grounded in docs" — every fact, resource, argument and code line
   comes from the retrieved documents; nothing from outside knowledge.
2. "Code included" — a technical answer includes at least one code
   snippet taken from the documents.
3. "User context used" — the user's language, resource and variable
   names and existing setup are kept; no invented placeholder project
   and no invented sample resources.
4. "Honest about gaps" — if the retrieved documents did not answer the
   question, or the question was off-topic, the answer says so plainly,
   does not guess, and uses confidence 0.0 when off-topic.
5. "Follow-ups on topic" — each follow-up question is about the specific
   topic of this answer, not a generic list.
"""

JUDGE_INSTRUCTIONS = """
You are an expert judge evaluating the performance of an
AI agent.
""".strip()

JUDGE_USER_PROMPT_TEMPLATE = """
Evaluate the agent's performance based on the following criteria:
<CRITERIA>
{criteria}
</CRITERIA>

The agent's final output was:
<AGENT_OUTPUT>
{output}
</AGENT_OUTPUT>

Tool calls:
<TOOL_CALLS>
{tool_calls}
</TOOL_CALLS>
""".strip()


def build_prompt(question: str, search_results: list) -> str:
    context = json.dumps(search_results, indent=2)
    return PROMPT_TEMPLATE.format(question=question, context=context)


def build_judge_prompt(criteria: list[str], output: str, tool_calls: list) -> str:
    return JUDGE_USER_PROMPT_TEMPLATE.format(
        criteria="\n".join(criteria),
        output=output,
        tool_calls="\n".join([str(tc) for tc in tool_calls]),
    )