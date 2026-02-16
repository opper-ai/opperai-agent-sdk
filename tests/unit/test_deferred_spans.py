"""Tests for deferred span updates."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from opper_agents.core.agent import Agent
from opper_agents.base.context import PendingSpanUpdate


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
        mock_opper_cls.return_value = mock_client
        yield mock_client


def test_pending_span_update_model():
    """Test PendingSpanUpdate model creation."""
    update = PendingSpanUpdate(span_id="span-123", output="result", name="think")
    assert update.span_id == "span-123"
    assert update.output == "result"
    assert update.name == "think"
    assert update.error is None
    assert update.start_time is None


@pytest.mark.asyncio
async def test_span_updates_are_deferred(mock_opper_client):
    """Test that span updates are batched and flushed at end of run."""
    mock_opper_client.call_async = AsyncMock(
        side_effect=[
            # Think: done immediately
            AsyncMock(
                json_payload={
                    "reasoning": "Simple task",
                    "tool_calls": [],
                    "user_message": "Done",
                    "memory_updates": {},
                    "is_complete": True,
                    "final_result": "42",
                },
                usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                span_id="think-span-123",
            ),
        ]
    )

    agent = Agent(
        name="TestAgent",
        verbose=False,
        opper_api_key="test-key",
    )

    await agent.process("Test")

    # Span updates should have been called (via deferred flush)
    assert mock_opper_client.spans.update_async.called


@pytest.mark.asyncio
async def test_queue_and_flush_span_updates(mock_opper_client):
    """Test the _queue_span_update and _flush_pending_span_updates methods."""
    agent = Agent(
        name="TestAgent",
        verbose=False,
        opper_api_key="test-key",
    )

    # Initialize context manually
    from opper_agents.base.context import AgentContext

    agent.context = AgentContext(agent_name="TestAgent")

    # Queue some updates
    agent._queue_span_update(span_id="span-1", output="result-1")
    agent._queue_span_update(span_id="span-2", name="renamed")

    assert len(agent.context.pending_span_updates) == 2

    # Flush
    await agent._flush_pending_span_updates()

    assert len(agent.context.pending_span_updates) == 0
    assert mock_opper_client.spans.update_async.call_count == 2


@pytest.mark.asyncio
async def test_flush_handles_errors_gracefully(mock_opper_client):
    """Test that flush continues even if individual updates fail."""
    mock_opper_client.spans.update_async = AsyncMock(
        side_effect=[Exception("API error"), None]
    )

    agent = Agent(
        name="TestAgent",
        verbose=False,
        opper_api_key="test-key",
    )

    from opper_agents.base.context import AgentContext

    agent.context = AgentContext(agent_name="TestAgent")

    agent._queue_span_update(span_id="span-1", output="fail")
    agent._queue_span_update(span_id="span-2", output="succeed")

    # Should not raise despite first update failing
    await agent._flush_pending_span_updates()

    assert mock_opper_client.spans.update_async.call_count == 2
    assert len(agent.context.pending_span_updates) == 0
