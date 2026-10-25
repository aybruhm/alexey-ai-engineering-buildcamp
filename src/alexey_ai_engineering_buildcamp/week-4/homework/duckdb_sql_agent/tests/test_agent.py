import pytest

from ..agent import init_agent, run_agent
from ..models import SQLAgentConfig
from ..tools import SQLTools
from ..utils import init_database
from .judge import assert_criteria
from .utils import collect_tools

EXPECTED_COUNT = 22413


@pytest.mark.asyncio
async def test_agent_counts_trips_with_more_than_five_passengers() -> None:
    sql_tools = SQLTools(db_path=init_database())
    agent = init_agent(SQLAgentConfig(), sql_tools)

    result = await run_agent(
        agent,
        "How many trips had more than 5 passengers?",
    )
    output = result.output

    assert isinstance(output.sql_query, str)
    assert output.sql_query.strip(), "the agent returned an empty SQL query"

    # The model may format the count with thousands separators.
    assert str(EXPECTED_COUNT) in output.result_text.replace(",", ""), (
        f"expected {EXPECTED_COUNT} in result_text: {output.result_text}"
    )


@pytest.mark.asyncio
async def test_agent_gets_schema_before_running_sql() -> None:
    sql_tools = SQLTools(db_path=init_database())
    agent = init_agent(SQLAgentConfig(), sql_tools)

    result = await run_agent(agent, "What is the most common payment type?")
    tool_calls = collect_tools(result.new_messages())
    names = [call.name for call in tool_calls]

    assert tool_calls, "the agent did not call any tools"
    assert names[0] == "get_schema", f"expected get_schema first, got: {names}"
    assert "run_sql" in names, f"expected run_sql to be called, got: {names}"


@pytest.mark.asyncio
async def test_agent_identifies_highest_average_fare_hour() -> None:
    sql_tools = SQLTools(db_path=init_database())
    agent = init_agent(SQLAgentConfig(), sql_tools)

    result = await run_agent(
        agent,
        "Which hour of the day has the highest average fare amount?",
    )

    await assert_criteria(
        result,
        [
            "the SQL query correctly calculates average fare by hour of day",
            "the result identifies a specific hour as having the highest average fare",
            "the result includes the actual average fare amount",
        ],
    )
