"""Tests for the run() method and RunResult model."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from opper_agents.core.agent import Agent
from opper_agents.base.context import RunResult, Usage


@pytest.fixture
def mock_opper_client():
    """Create a mock Opper client."""
    with patch("opper_agents.base.agent.Opper") as mock_opper_cls:
        mock_client = MagicMock()
        mock_client.sdk_configuration = MagicMock()
        mock_client.spans = MagicMock()
        mock_client.spans.create_async = AsyncMock(
            return_value=AsyncMock(id="span-123")
        )
        mock_client.spans.update_async = AsyncMock()
        mock_client.spans.get_async = AsyncMock()
        mock_client.call_async = AsyncMock(
            return_value=AsyncMock(
                json_payload={
                    "reasoning": "Done",
                    "tool_calls": [],
                    "user_message": "Complete",
                    "memory_updates": {},
                    "is_complete": True,
                    "final_result": "The answer is 42",
                },
                usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                span_id="span-456",
            )
        )
        mock_opper_cls.return_value = mock_client
        yield mock_client


@pytest.mark.asyncio
async def test_run_returns_run_result(mock_opper_client):
    """Test that run() returns a RunResult with result and usage."""
    agent = Agent(
        name="TestAgent",
        verbose=False,
        opper_api_key="test-key",
    )
    run_result = await agent.run("What is 6 * 7?")

    assert isinstance(run_result, RunResult)
    assert run_result.result == "The answer is 42"
    assert isinstance(run_result.usage, Usage)


@pytest.mark.asyncio
async def test_run_tracks_usage(mock_opper_client):
    """Test that run() tracks usage statistics."""
    agent = Agent(
        name="TestAgent",
        verbose=False,
        opper_api_key="test-key",
    )
    run_result = await agent.run("What is 6 * 7?")

    assert run_result.usage.requests >= 1
    assert run_result.usage.total_tokens > 0


@pytest.mark.asyncio
async def test_process_still_works(mock_opper_client):
    """Test that process() still returns just the result (backward compat)."""
    agent = Agent(
        name="TestAgent",
        verbose=False,
        opper_api_key="test-key",
    )
    result = await agent.process("What is 6 * 7?")

    # process() returns just the result, not RunResult
    assert result == "The answer is 42"
    assert not isinstance(result, RunResult)


@pytest.mark.asyncio
async def test_run_result_model():
    """Test RunResult model creation and access."""
    usage = Usage(requests=3, total_tokens=500)
    result = RunResult(result={"answer": "hello"}, usage=usage)

    assert result.result == {"answer": "hello"}
    assert result.usage.requests == 3
    assert result.usage.total_tokens == 500


@pytest.mark.asyncio
async def test_run_result_default_usage():
    """Test RunResult with default usage."""
    result = RunResult(result="hello")

    assert result.result == "hello"
    assert result.usage.requests == 0
    assert result.usage.total_tokens == 0


@pytest.mark.asyncio
async def test_run_tracks_cost_from_response():
    """Test that run() extracts cost from the Opper API response."""
    with patch("opper_agents.base.agent.Opper") as mock_opper_cls:
        mock_client = MagicMock()
        mock_client.sdk_configuration = MagicMock()
        mock_client.spans = MagicMock()
        mock_client.spans.create_async = AsyncMock(
            return_value=AsyncMock(id="span-123")
        )
        mock_client.spans.update_async = AsyncMock()
        mock_client.call_async = AsyncMock(
            return_value=AsyncMock(
                json_payload={
                    "reasoning": "Done",
                    "tool_calls": [],
                    "user_message": "Done",
                    "memory_updates": {},
                    "is_complete": True,
                    "final_result": "42",
                },
                usage={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
                cost={"generation": 0.003, "platform": 0.001, "total": 0.004},
                span_id="span-456",
            )
        )
        mock_opper_cls.return_value = mock_client

        agent = Agent(name="TestAgent", verbose=False, opper_api_key="test-key")
        run_result = await agent.run("Test")

        assert run_result.usage.cost.generation == 0.003
        assert run_result.usage.cost.platform == 0.001
        assert run_result.usage.cost.total == 0.004
