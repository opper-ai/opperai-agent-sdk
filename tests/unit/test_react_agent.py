"""
Unit tests for ReactAgent.

Tests the ReAct pattern agent implementation.
"""

import pytest
from unittest.mock import AsyncMock
from opper_agents.agents.react import ReactAgent
from opper_agents.utils.decorators import tool


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
def multiply(x: int, y: int) -> int:
    """Multiply two numbers."""
    return x * y


def test_react_agent_initialization():
    """Test ReactAgent can be initialized with tools."""
    agent = ReactAgent(
        name="MathAgent",
        description="Math operations agent",
        tools=[add, multiply],
        verbose=False,
        opper_api_key="test-key",
    )
    assert agent.name == "MathAgent"
    assert len(agent.tools) == 2
    assert agent.get_tool("add") is not None
    assert agent.get_tool("multiply") is not None


@pytest.mark.asyncio
async def test_react_agent_simple_execution(mock_opper_client):
    """Test ReactAgent executes a simple task with one action."""
    # Mock the LLM to:
    # 1. First call: Decide to use add tool
    # 2. Second call: Mark task as complete
    # 3. Third call: Generate final result
    mock_opper_client.call_async = AsyncMock(
        side_effect=[
            # First reasoning call - take action
            AsyncMock(
                json_payload={
                    "reasoning": "I need to add 5 and 3",
                    "is_complete": False,
                    "action": {"tool_name": "add", "parameters": {"a": 5, "b": 3}},
                    "user_message": "Adding numbers...",
                }
            ),
            # Second reasoning call - task complete
            AsyncMock(
                json_payload={
                    "reasoning": "The result is 8, task complete",
                    "is_complete": True,
                    "action": None,
                    "user_message": "Done!",
                }
            ),
            # Generate final result
            AsyncMock(json_payload={"result": 8}, message="The result is 8"),
        ]
    )

    agent = ReactAgent(
        name="MathAgent",
        tools=[add, multiply],
        verbose=False,
        opper_api_key="test-key",
    )

    result = await agent.process("What is 5 + 3?")
    assert result is not None


@pytest.mark.asyncio
async def test_react_agent_multiple_actions(mock_opper_client):
    """Test ReactAgent handles multiple sequential actions."""
    # Mock the LLM to:
    # 1. Add 5 + 3
    # 2. Multiply result by 2
    # 3. Complete
    mock_opper_client.call_async = AsyncMock(
        side_effect=[
            # First action: add
            AsyncMock(
                json_payload={
                    "reasoning": "First add 5 and 3",
                    "is_complete": False,
                    "action": {"tool_name": "add", "parameters": {"a": 5, "b": 3}},
                    "user_message": "Adding...",
                }
            ),
            # Second action: multiply
            AsyncMock(
                json_payload={
                    "reasoning": "Now multiply 8 by 2",
                    "is_complete": False,
                    "action": {"tool_name": "multiply", "parameters": {"x": 8, "y": 2}},
                    "user_message": "Multiplying...",
                }
            ),
            # Complete
            AsyncMock(
                json_payload={
                    "reasoning": "Result is 16, done",
                    "is_complete": True,
                    "action": None,
                    "user_message": "Done!",
                }
            ),
            # Final result
            AsyncMock(json_payload={"result": 16}, message="The result is 16"),
        ]
    )

    agent = ReactAgent(
        name="MathAgent",
        tools=[add, multiply],
        verbose=False,
        opper_api_key="test-key",
    )

    result = await agent.process("What is (5 + 3) * 2?")
    assert result is not None

    # Verify we had 2 iterations (2 actions)
    assert agent.context.iteration == 2


@pytest.mark.asyncio
async def test_react_agent_tool_error_handling(mock_opper_client):
    """Test ReactAgent handles tool execution errors gracefully."""

    @tool
    def failing_tool() -> str:
        """A tool that always fails."""
        raise ValueError("Intentional failure")

    # Mock LLM to call failing tool then complete
    mock_opper_client.call_async = AsyncMock(
        side_effect=[
            # Call failing tool
            AsyncMock(
                json_payload={
                    "reasoning": "Try the failing tool",
                    "is_complete": False,
                    "action": {"tool_name": "failing_tool", "parameters": {}},
                    "user_message": "Trying...",
                }
            ),
            # Complete after error
            AsyncMock(
                json_payload={
                    "reasoning": "Tool failed, but I can still respond",
                    "is_complete": True,
                    "action": None,
                    "user_message": "Done",
                }
            ),
            # Final result
            AsyncMock(json_payload={"error": "Tool failed"}, message="Error occurred"),
        ]
    )

    agent = ReactAgent(
        name="TestAgent",
        tools=[failing_tool],
        verbose=False,
        opper_api_key="test-key",
    )

    result = await agent.process("Test error handling")
    assert result is not None

    # Check that error was recorded in context
    assert len(agent.context.execution_history) == 1
    cycle = agent.context.execution_history[0]
    assert not cycle.results[0].success
    assert "Intentional failure" in cycle.results[0].error


@pytest.mark.asyncio
async def test_react_agent_max_iterations(mock_opper_client):
    """Test ReactAgent respects max_iterations limit."""
    # Mock LLM to always return an action (never complete)
    mock_opper_client.call_async = AsyncMock(
        side_effect=[
            # Keep returning actions
            AsyncMock(
                json_payload={
                    "reasoning": "Keep going",
                    "is_complete": False,
                    "action": {"tool_name": "add", "parameters": {"a": 1, "b": 1}},
                    "user_message": "Working...",
                }
            )
            for _ in range(10)  # More than max_iterations
        ]
        + [
            # Final result after max iterations
            AsyncMock(json_payload={"result": "partial"}, message="Partial result")
        ]
    )

    agent = ReactAgent(
        name="MathAgent",
        tools=[add],
        max_iterations=3,  # Set low limit
        verbose=False,
        opper_api_key="test-key",
    )

    result = await agent.process("Never-ending task")
    assert result is not None

    # Should have stopped at max_iterations
    assert agent.context.iteration == 3


@pytest.mark.asyncio
async def test_react_agent_no_action_when_not_complete(mock_opper_client):
    """Test ReactAgent handles case where is_complete=False but no action provided."""
    mock_opper_client.call_async = AsyncMock(
        side_effect=[
            # Invalid state: not complete but no action
            AsyncMock(
                json_payload={
                    "reasoning": "Not sure what to do",
                    "is_complete": False,
                    "action": None,  # Invalid: should provide action if not complete
                    "user_message": "Thinking...",
                }
            ),
            # Final result
            AsyncMock(json_payload={"result": "incomplete"}, message="Task incomplete"),
        ]
    )

    agent = ReactAgent(
        name="TestAgent",
        tools=[add],
        verbose=False,
        opper_api_key="test-key",
    )

    result = await agent.process("Test invalid state")
    assert result is not None

    # Should have stopped immediately due to invalid state
    assert agent.context.iteration == 0


@pytest.mark.asyncio
async def test_react_agent_as_tool(mock_opper_client):
    """Test ReactAgent can be converted to a tool for use by other agents."""
    mock_opper_client.call_async = AsyncMock(
        side_effect=[
            # ReactAgent reasoning
            AsyncMock(
                json_payload={
                    "reasoning": "Completing task",
                    "is_complete": True,
                    "action": None,
                    "user_message": "Done",
                }
            ),
            # Final result
            AsyncMock(json_payload={"result": "success"}, message="Success"),
        ]
    )

    agent = ReactAgent(
        name="SubAgent",
        tools=[add],
        verbose=False,
        opper_api_key="test-key",
    )

    # Convert to tool
    agent_tool = agent.as_tool()
    assert agent_tool.name == "SubAgent_agent"
    assert "SubAgent" in agent_tool.description


def test_react_agent_inherits_from_agent():
    """Test that ReactAgent properly inherits from Agent."""
    from opper_agents.core.agent import Agent

    agent = ReactAgent(
        name="Test", tools=[add], verbose=False, opper_api_key="test-key"
    )
    assert isinstance(agent, Agent)
    assert hasattr(agent, "_execute_tool")
    assert hasattr(agent, "_generate_final_result")


@pytest.mark.asyncio
async def test_react_agent_with_verbose_mode(mock_opper_client, capsys):
    """Test ReactAgent verbose output."""
    mock_opper_client.call_async = AsyncMock(
        side_effect=[
            # One action
            AsyncMock(
                json_payload={
                    "reasoning": "Testing verbose mode",
                    "is_complete": False,
                    "action": {"tool_name": "add", "parameters": {"a": 1, "b": 1}},
                    "user_message": "Adding...",
                }
            ),
            # Complete
            AsyncMock(
                json_payload={
                    "reasoning": "Done",
                    "is_complete": True,
                    "action": None,
                    "user_message": "Complete",
                }
            ),
            # Final result
            AsyncMock(json_payload={"result": 2}, message="Result is 2"),
        ]
    )

    agent = ReactAgent(
        name="VerboseAgent",
        tools=[add],
        verbose=True,  # Enable verbose mode
        opper_api_key="test-key",
    )

    await agent.process("Test verbose")

    # Check that verbose output was printed
    captured = capsys.readouterr()
    assert "ReAct Iteration" in captured.out
    assert "Reasoning:" in captured.out
    assert "Action:" in captured.out
