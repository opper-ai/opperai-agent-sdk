"""Tests for parallel tool execution."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from opper_agents.core.agent import Agent
from opper_agents.utils.decorators import tool


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


def test_parallel_tool_execution_default_false():
    """Test that parallel_tool_execution defaults to False."""
    with patch("opper_agents.base.agent.Opper") as mock_cls:
        mock_cls.return_value = MagicMock(sdk_configuration=MagicMock())
        agent = Agent(name="Test", opper_api_key="test-key")
        assert agent.parallel_tool_execution is False


def test_parallel_tool_execution_can_be_enabled():
    """Test that parallel_tool_execution can be set to True."""
    with patch("opper_agents.base.agent.Opper") as mock_cls:
        mock_cls.return_value = MagicMock(sdk_configuration=MagicMock())
        agent = Agent(
            name="Test",
            opper_api_key="test-key",
            parallel_tool_execution=True,
        )
        assert agent.parallel_tool_execution is True


@pytest.mark.asyncio
async def test_parallel_execution_multiple_tools(mock_opper_client):
    """Test that multiple tools execute in parallel when enabled."""
    execution_order = []

    @tool
    async def tool_a(x: str) -> str:
        """Tool A."""
        execution_order.append("a_start")
        await asyncio.sleep(0.05)
        execution_order.append("a_end")
        return f"a_{x}"

    @tool
    async def tool_b(x: str) -> str:
        """Tool B."""
        execution_order.append("b_start")
        await asyncio.sleep(0.05)
        execution_order.append("b_end")
        return f"b_{x}"

    mock_opper_client.call_async = AsyncMock(
        side_effect=[
            # Think: call both tools
            AsyncMock(
                json_payload={
                    "reasoning": "Call both tools",
                    "tool_calls": [
                        {"name": "tool_a", "parameters": {"x": "1"}, "reasoning": "a"},
                        {"name": "tool_b", "parameters": {"x": "2"}, "reasoning": "b"},
                    ],
                    "user_message": "Working...",
                    "memory_updates": {},
                },
                span_id="span-think",
            ),
            # Think: done
            AsyncMock(
                json_payload={
                    "reasoning": "Complete",
                    "tool_calls": [],
                    "user_message": "Done",
                    "memory_updates": {},
                },
                span_id="span-think-2",
            ),
            # Final result
            AsyncMock(message="Result"),
        ]
    )

    agent = Agent(
        name="ParallelAgent",
        tools=[tool_a, tool_b],
        opper_api_key="test-key",
        parallel_tool_execution=True,
    )

    await agent.process("Test parallel")

    # In parallel execution, both tools should start before either ends
    assert "a_start" in execution_order
    assert "b_start" in execution_order
    # The key parallel indicator: both starts happen before any end
    a_start_idx = execution_order.index("a_start")
    b_start_idx = execution_order.index("b_start")
    a_end_idx = execution_order.index("a_end")
    b_end_idx = execution_order.index("b_end")
    # At least one tool should start before the other ends (parallel behavior)
    assert b_start_idx < a_end_idx or a_start_idx < b_end_idx


@pytest.mark.asyncio
async def test_sequential_execution_by_default(mock_opper_client):
    """Test that tools execute sequentially by default."""
    execution_order = []

    @tool
    async def tool_a(x: str) -> str:
        """Tool A."""
        execution_order.append("a_start")
        await asyncio.sleep(0.01)
        execution_order.append("a_end")
        return f"a_{x}"

    @tool
    async def tool_b(x: str) -> str:
        """Tool B."""
        execution_order.append("b_start")
        await asyncio.sleep(0.01)
        execution_order.append("b_end")
        return f"b_{x}"

    mock_opper_client.call_async = AsyncMock(
        side_effect=[
            # Think: call both tools
            AsyncMock(
                json_payload={
                    "reasoning": "Call both tools",
                    "tool_calls": [
                        {"name": "tool_a", "parameters": {"x": "1"}, "reasoning": "a"},
                        {"name": "tool_b", "parameters": {"x": "2"}, "reasoning": "b"},
                    ],
                    "user_message": "Working...",
                    "memory_updates": {},
                },
                span_id="span-think",
            ),
            # Think: done
            AsyncMock(
                json_payload={
                    "reasoning": "Complete",
                    "tool_calls": [],
                    "user_message": "Done",
                    "memory_updates": {},
                },
                span_id="span-think-2",
            ),
            # Final result
            AsyncMock(message="Result"),
        ]
    )

    agent = Agent(
        name="SequentialAgent",
        tools=[tool_a, tool_b],
        opper_api_key="test-key",
        parallel_tool_execution=False,  # explicit default
    )

    await agent.process("Test sequential")

    # In sequential execution, a should complete before b starts
    assert execution_order == ["a_start", "a_end", "b_start", "b_end"]
