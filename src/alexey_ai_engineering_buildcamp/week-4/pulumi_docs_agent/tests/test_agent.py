import pytest

from ..agent import run_agent_stream
from .utils import capture_usage, collect_tools, get_model_name


@pytest.mark.asyncio
async def test_agent_runs(create_agent):
    agent = create_agent

    user_prompt = "create s3 + cloudfront in aws"
    result = await run_agent_stream(agent, user_prompt)

    search_result = result.output  # type: ignore
    assert search_result.answer is not None  # type: ignore
    assert search_result.confidence >= 0.0  # type: ignore
    assert search_result.found_answer is True  # type: ignore
    assert len(search_result.followup_questions) > 0  # type: ignore


@pytest.mark.asyncio
async def test_agent_uses_tools(create_agent):
    agent = create_agent

    user_prompt = "create s3 + cloudfront in aws"
    result = await run_agent_stream(agent, user_prompt)

    usage = result.usage  # type: ignore
    capture_usage(get_model_name(agent.model), usage)

    messages = result.new_messages()  # type: ignore

    tool_calls = collect_tools(messages)
    assert len(tool_calls) >= 2

    # The model is free to call tools in any order, and to call others in
    # between, so assert on the set of names rather than on positions.
    tool_names = {call.name for call in tool_calls}
    assert {"search", "get_file"} <= tool_names, f"got: {sorted(tool_names)}"


@pytest.mark.asyncio
async def test_agent_includes_code_in_answer(create_agent):
    agent = create_agent

    user_prompt = "how do i make use of s3 and cloudfront in aws with python"
    result = await run_agent_stream(agent, user_prompt)

    usage = result.usage  # type: ignore
    capture_usage(get_model_name(agent.model), usage)

    search_result = result.output  # type: ignore
    assert "```python" in search_result.answer or "```python" in search_result.answer  # type: ignore


@pytest.mark.asyncio
async def test_agent_includes_references(create_agent):
    agent = create_agent

    user_prompt = "create ecs, fargate and alb in aws using python"
    result = await run_agent_stream(agent, user_prompt)

    usage = result.usage  # type: ignore
    capture_usage(get_model_name(agent.model), usage)

    search_result = result.output  # type: ignore

    references = search_result.references  # type: ignore
    assert len(references) >= 1
    assert references[0].file_path is not None
    assert references[0].explanation is not None

    used_references = {r.file_path: r for r in references}
    assert (
        "blog/get-started-with-docker-on-aws-fargate-using-pulumi/index.md"
        in used_references
        or "blog/managing-containers-on-aws-with-pulumi/index.md" in used_references
    )
