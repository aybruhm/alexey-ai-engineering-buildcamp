import pytest

from ..agent import init_agent, run_agent
from ..models import SQLAgentConfig
from ..tools import SQLTools
from ..utils import init_database
from .utils import collect_tools, extract_numbers

CASES = [
    {
        "id": "credit-card-tip",
        "question": "What is the average tip amount for credit card payments?",
        # payment_type = 1 is the credit-card code in the TLC data.
        "sql_terms": ["tip_amount", "payment_type"],
        "expected_number": 4.1697,
        "tolerance": 0.01,
    },
    {
        "id": "busiest-pickup-location",
        "question": "Which pickup location (PULocationID) has the most trips?",
        # Group by pickup location and count, then take the top row.
        "sql_terms": ["PULocationID", "COUNT"],
        "expected_number": 132,
        "tolerance": 0,
    },
    {
        "id": "long-trip-average-fare",
        "question": "What is the average fare for trips longer than 10 miles?",
        # "Longer than" means trip_distance > 10, not >= 10.
        "sql_terms": ["fare_amount", "trip_distance"],
        "expected_number": 62.8808,
        "tolerance": 0.01,
    },
    {
        "id": "zero-passenger-trips",
        "question": "How many trips had zero passengers recorded?",
        # Recorded zero is passenger_count = 0, which is different from NULL.
        "sql_terms": ["passenger_count", "COUNT"],
        "expected_number": 31465,
        "tolerance": 0,
    },
    {
        "id": "busiest-day-of-week",
        "question": "What is the busiest day of the week for taxi trips?",
        # Day of week comes from the pickup timestamp.
        "sql_terms": ["tpep_pickup_datetime", "COUNT"],
        "expected_text": "Wednesday",
    },
]


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
@pytest.mark.asyncio
async def test_agent_answers_data_question(case) -> None:
    sql_tools = SQLTools(db_path=init_database())
    agent = init_agent(SQLAgentConfig(), sql_tools)

    result = await run_agent(agent, case["question"])
    output = result.output

    tool_calls = collect_tools(result.new_messages())
    names = [call.name for call in tool_calls]
    assert names, "the agent did not call any tools"
    assert names[0] == "get_schema", f"expected get_schema first, got: {names}"
    assert "run_sql" in names, f"expected run_sql to be called, got: {names}"

    sql = output.sql_query.lower()
    for term in case["sql_terms"]:
        assert term.lower() in sql, (
            f"expected {term!r} in the SQL query: {output.sql_query}"
        )

    if "expected_number" in case:
        numbers = extract_numbers(output.result_text)
        expected = case["expected_number"]
        tolerance = case.get("tolerance", 0.0)
        assert any(abs(number - expected) <= tolerance for number in numbers), (
            f"expected {expected} in result_text: {output.result_text}"
        )
    else:
        assert case["expected_text"].lower() in output.result_text.lower(), (
            f"expected {case['expected_text']!r} in result_text: {output.result_text}"
        )
