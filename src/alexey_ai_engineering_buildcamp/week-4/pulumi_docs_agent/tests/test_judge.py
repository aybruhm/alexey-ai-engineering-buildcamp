import pytest

from ..agent import run_agent_stream
from .llm_judge import CRITERIA, assert_criteria

JUDGE_CASES = [
    {
        "id": "s3-cloudfront",
        "question": "create s3 + cloudfront in aws",
        "criteria": [
            "the answer explains how to create an S3 bucket with Pulumi",
            "the answer explains how the site is served, via static website hosting and/or a CDN such as CloudFront",
            "the answer includes a code example inside a fenced code block",
            "the answer does not invent Pulumi resources, properties or CLI flags that do not exist",
        ],
    },
    {
        "id": "fargate",
        "question": "run docker containers on aws fargate",
        "criteria": [
            "the answer explains how to run Docker containers on AWS Fargate using Pulumi",
            "the answer names the container image and the service the containers run in",
            "the answer includes a code example inside a fenced code block",
            "the answer does not invent Pulumi resources, properties or CLI flags that do not exist",
        ],
    },
    {
        "id": "secrets-management",
        "question": "what is secrets management",
        "criteria": [
            "the answer explains what secrets management means",
            "the answer is an explanation rather than step-by-step instructions",
            "the answer does not invent Pulumi resources, properties or CLI flags that do not exist",
        ],
    },
    {
        "id": "multi-hop",
        "question": "how do i store a secret and use it in a stack config",
        "criteria": [
            "the answer explains how to store a secret",
            "the answer explains how to consume that secret from stack configuration",
            "the answer addresses both halves of the question rather than only one",
            "the answer does not invent Pulumi resources, properties or CLI flags that do not exist",
        ],
    },
    {
        "id": "off-topic-capital",
        "question": "what is the capital of france",
        "criteria": [
            "the answer states that the documentation does not contain this information",
            "the answer does not answer the question using knowledge from outside the documents",
            "the answer does not present a citation as proof of an answer it did not find",
        ],
    },
]


@pytest.mark.asyncio
async def test_agent_uses_tools(create_agent):
    agent = create_agent

    user_prompt = "create s3 + cloudfront in aws"
    result = await run_agent_stream(agent, user_prompt)
    assert result is not None, "the agent run produced no result"

    await assert_criteria(
        result,
        CRITERIA,
    )


@pytest.mark.parametrize("case", JUDGE_CASES, ids=[case["id"] for case in JUDGE_CASES])
@pytest.mark.asyncio
async def test_agent_answer_quality(case, create_agent):
    result = await run_agent_stream(create_agent, case["question"])
    assert result is not None, "the agent run produced no result"

    await assert_criteria(result, case["criteria"])


@pytest.mark.asyncio
async def test_ambiguous_term_registry(create_agent):
    user_prompt = "registry"
    result = await run_agent_stream(create_agent, user_prompt)
    assert result is not None, "the agent run produced no result"

    await assert_criteria(
        result,
        [
            "the answer identifies which sense of 'registry' it is answering about",
            "the answer is about the Pulumi Registry (the catalogue of Pulumi packages) rather than about hosting container images",
            "follow-up questions are about Pulumi, not about generic container tooling",
        ],
    )


@pytest.mark.asyncio
async def test_adapts_to_user_context(create_agent):
    user_prompt = (
        "how do i create an s3 bucket? i already have a pulumi program "
        "in typescript with a Stack class in index.ts"
    )
    result = await run_agent_stream(create_agent, user_prompt)
    assert result is not None, "the agent run produced no result"

    await assert_criteria(
        result,
        [
            "the answer uses TypeScript, not Python",
            "the answer builds on the user's existing program (the Stack class in index.ts) instead of starting a new project",
            "the answer does not give a generic tutorial that ignores the user's stated context",
        ],
    )
