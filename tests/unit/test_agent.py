"""
Unit tests for main Agent implementation.

Tests the think-act loop, tool execution, and memory integration.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from opper_agent.core.agent import Agent
from opper_agent.core.schemas import Thought, ToolCall
from opper_agent.utils.decorators import tool


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
def multiply(x: int, y: int) -> int:
    """Multiply two numbers."""
    return x * y


@tool
def failing_tool():
    """This tool always fails."""
    raise ValueError("Intentional failure")


@pytest.mark.asyncio
async def test_agent_initialization(mock_opper_client):
    """Test Agent initialization with basic parameters."""
    agent = Agent(
        name="TestAgent",
        description="Test agent",
        tools=[add, multiply],
        verbose=False,
        opper_api_key="test-key",
    )

    assert agent.name == "TestAgent"
    assert len(agent.tools) == 2
    assert agent.enable_memory is False


@pytest.mark.asyncio
async def test_agent_with_memory(mock_opper_client):
    """Test Agent initialization with memory enabled."""
    agent = Agent(
        name="MemoryAgent",
        tools=[add],
        enable_memory=True,
        opper_api_key="test-key",
    )

    assert agent.enable_memory is True


@pytest.mark.asyncio
async def test_agent_simple_execution(mock_opper_client):
    """Test basic agent execution flow."""
    # Mock responses
    mock_opper_client.call_async = AsyncMock(
        side_effect=[
            # First call: think - decide to use add tool
            AsyncMock(
                json_payload={
                    "reasoning": "I need to add 5 and 3",
                    "tool_calls": [
                        {
                            "name": "add",
                            "parameters": {"a": 5, "b": 3},
                            "reasoning": "Adding numbers",
                        }
                    ],
                    "user_message": "Calculating...",
                    "memory_updates": {},
                }
            ),
            # Second call: think - task complete
            AsyncMock(
                json_payload={
                    "reasoning": "Result is 8, task complete",
                    "tool_calls": [],
                    "user_message": "Done",
                    "memory_updates": {},
                }
            ),
            # Third call: generate final result
            AsyncMock(message="The sum of 5 and 3 is 8"),
        ]
    )

    agent = Agent(
        name="MathAgent", tools=[add], verbose=False, opper_api_key="test-key"
    )

    result = await agent.process("What is 5 + 3?")
    assert result == "The sum of 5 and 3 is 8"
    assert mock_opper_client.call_async.call_count == 3


@pytest.mark.asyncio
async def test_agent_multiple_tool_calls(mock_opper_client):
    """Test agent executing multiple tools in one iteration."""
    mock_opper_client.call_async = AsyncMock(
        side_effect=[
            # Think: call both add and multiply
            AsyncMock(
                json_payload={
                    "reasoning": "Need to add and multiply",
                    "tool_calls": [
                        {
                            "name": "add",
                            "parameters": {"a": 2, "b": 3},
                            "reasoning": "Add first",
                        },
                        {
                            "name": "multiply",
                            "parameters": {"x": 2, "y": 3},
                            "reasoning": "Multiply second",
                        },
                    ],
                    "user_message": "Calculating...",
                    "memory_updates": {},
                }
            ),
            # Think: task complete
            AsyncMock(
                json_payload={
                    "reasoning": "Done",
                    "tool_calls": [],
                    "user_message": "Complete",
                    "memory_updates": {},
                }
            ),
            # Generate result
            AsyncMock(message="Results: 5 and 6"),
        ]
    )

    agent = Agent(
        name="MathAgent", tools=[add, multiply], verbose=False, opper_api_key="test-key"
    )
    result = await agent.process("Calculate")

    # Check that context has the execution history
    assert agent.context.iteration == 1  # One full cycle
    assert len(agent.context.execution_history) == 1
    cycle = agent.context.execution_history[0]
    assert len(cycle.results) == 2
    assert cycle.results[0].success is True
    assert cycle.results[0].result == 5
    assert cycle.results[1].result == 6


@pytest.mark.asyncio
async def test_agent_tool_not_found(mock_opper_client):
    """Test agent handling of non-existent tool."""
    mock_opper_client.call_async = AsyncMock(
        side_effect=[
            # Think: try to call non-existent tool
            AsyncMock(
                json_payload={
                    "reasoning": "Calling unknown tool",
                    "tool_calls": [
                        {
                            "name": "unknown_tool",
                            "parameters": {},
                            "reasoning": "Test",
                        }
                    ],
                    "user_message": "Working...",
                    "memory_updates": {},
                }
            ),
            # Think: task complete
            AsyncMock(
                json_payload={
                    "reasoning": "Done",
                    "tool_calls": [],
                    "user_message": "Complete",
                    "memory_updates": {},
                }
            ),
            # Generate result
            AsyncMock(message="Completed with errors"),
        ]
    )

    agent = Agent(
        name="TestAgent", tools=[add], verbose=False, opper_api_key="test-key"
    )
    result = await agent.process("Test")

    # Check that tool error was recorded
    cycle = agent.context.execution_history[0]
    assert cycle.results[0].success is False
    assert "not found" in cycle.results[0].error


@pytest.mark.asyncio
async def test_agent_tool_execution_error(mock_opper_client):
    """Test agent handling of tool execution errors."""
    mock_opper_client.call_async = AsyncMock(
        side_effect=[
            # Think: call failing tool
            AsyncMock(
                json_payload={
                    "reasoning": "Calling tool",
                    "tool_calls": [
                        {"name": "failing_tool", "parameters": {}, "reasoning": "Test"}
                    ],
                    "user_message": "Working...",
                    "memory_updates": {},
                }
            ),
            # Think: task complete
            AsyncMock(
                json_payload={
                    "reasoning": "Done",
                    "tool_calls": [],
                    "user_message": "Complete",
                    "memory_updates": {},
                }
            ),
            # Generate result
            AsyncMock(message="Handled error"),
        ]
    )

    agent = Agent(
        name="TestAgent", tools=[failing_tool], verbose=False, opper_api_key="test-key"
    )
    result = await agent.process("Test")

    # Check that tool error was caught and recorded
    cycle = agent.context.execution_history[0]
    assert cycle.results[0].success is False
    assert "Intentional failure" in cycle.results[0].error


@pytest.mark.asyncio
async def test_agent_max_iterations(mock_opper_client):
    """Test agent respects max_iterations limit."""
    # Always return tool calls to force max iterations
    mock_opper_client.call_async = AsyncMock(
        return_value=AsyncMock(
            json_payload={
                "reasoning": "Keep going",
                "tool_calls": [
                    {"name": "add", "parameters": {"a": 1, "b": 1}, "reasoning": "Add"}
                ],
                "user_message": "Working...",
                "memory_updates": {},
            }
        )
    )

    agent = Agent(
        name="TestAgent",
        tools=[add],
        max_iterations=3,
        verbose=False,
        opper_api_key="test-key",
    )

    # This should stop after 3 iterations and generate final result
    # Mock the final result call
    mock_opper_client.call_async.side_effect = [
        AsyncMock(
            json_payload={
                "reasoning": "Iteration 1",
                "tool_calls": [
                    {"name": "add", "parameters": {"a": 1, "b": 1}, "reasoning": "Add"}
                ],
                "user_message": "Working...",
                "memory_updates": {},
            }
        ),
        AsyncMock(
            json_payload={
                "reasoning": "Iteration 2",
                "tool_calls": [
                    {"name": "add", "parameters": {"a": 2, "b": 2}, "reasoning": "Add"}
                ],
                "user_message": "Working...",
                "memory_updates": {},
            }
        ),
        AsyncMock(
            json_payload={
                "reasoning": "Iteration 3",
                "tool_calls": [
                    {"name": "add", "parameters": {"a": 3, "b": 3}, "reasoning": "Add"}
                ],
                "user_message": "Working...",
                "memory_updates": {},
            }
        ),
        AsyncMock(message="Stopped at max iterations"),
    ]

    result = await agent.process("Keep calculating")

    assert agent.context.iteration == 3
    assert len(agent.context.execution_history) == 3


@pytest.mark.asyncio
async def test_agent_with_memory_updates(mock_opper_client):
    """Test agent writing to memory."""
    mock_opper_client.call_async = AsyncMock(
        side_effect=[
            # Think: update memory
            AsyncMock(
                json_payload={
                    "reasoning": "Saving state",
                    "tool_calls": [],
                    "user_message": "Saving...",
                    "memory_updates": {
                        "project_status": {
                            "value": {"status": "in_progress"},
                            "description": "Project state",
                            "metadata": {"updated": True},
                        }
                    },
                }
            ),
            # Generate result
            AsyncMock(message="Memory updated"),
        ]
    )

    agent = Agent(
        name="MemoryAgent",
        tools=[],
        enable_memory=True,
        verbose=False,
        opper_api_key="test-key",
    )
    result = await agent.process("Save state")

    # Check memory was updated
    assert agent.context.memory.has_entries()
    payload = await agent.context.memory.read(["project_status"])
    assert payload["project_status"]["status"] == "in_progress"


@pytest.mark.asyncio
async def test_agent_context_tracking(mock_opper_client):
    """Test that agent properly tracks context during execution."""
    mock_opper_client.call_async = AsyncMock(
        side_effect=[
            AsyncMock(
                json_payload={
                    "reasoning": "First iteration",
                    "tool_calls": [
                        {
                            "name": "add",
                            "parameters": {"a": 1, "b": 1},
                            "reasoning": "Add",
                        }
                    ],
                    "user_message": "Working...",
                    "memory_updates": {},
                }
            ),
            AsyncMock(
                json_payload={
                    "reasoning": "Done",
                    "tool_calls": [],
                    "user_message": "Complete",
                    "memory_updates": {},
                }
            ),
            AsyncMock(message="Result"),
        ]
    )

    agent = Agent(
        name="TestAgent", tools=[add], verbose=False, opper_api_key="test-key"
    )
    await agent.process("Test")

    # Verify context state
    assert agent.context.agent_name == "TestAgent"
    assert agent.context.parent_span_id == "test-span-id"
    assert agent.context.iteration == 1
    assert len(agent.context.execution_history) == 1


@pytest.mark.asyncio
async def test_agent_span_hierarchy(mock_opper_client):
    """Test that agent creates proper span hierarchy with parent-child relationships."""
    # Mock span creation
    parent_span = AsyncMock(id="parent-span-123")
    tool_span = AsyncMock(id="tool-span-1")
    mock_opper_client.spans.create_async = AsyncMock(
        side_effect=[parent_span, tool_span]
    )
    mock_opper_client.spans.update_async = AsyncMock()

    # Mock LLM calls
    mock_opper_client.call_async = AsyncMock(
        side_effect=[
            # Think call
            AsyncMock(
                json_payload={
                    "reasoning": "Calculating",
                    "tool_calls": [
                        {
                            "name": "add",
                            "parameters": {"a": 2, "b": 3},
                            "reasoning": "Add",
                        }
                    ],
                    "user_message": "Working...",
                    "memory_updates": {},
                }
            ),
            # Think call (done)
            AsyncMock(
                json_payload={
                    "reasoning": "Complete",
                    "tool_calls": [],
                    "user_message": "Done",
                    "memory_updates": {},
                }
            ),
            # Generate final result
            AsyncMock(message="The answer is 5"),
        ]
    )

    agent = Agent(
        name="MathAgent", tools=[add], verbose=False, opper_api_key="test-key"
    )
    result = await agent.process("What is 2 + 3?")

    # Verify parent span was created (first call)
    assert mock_opper_client.spans.create_async.call_count == 2  # parent + 1 tool
    parent_create_call = mock_opper_client.spans.create_async.call_args_list[0]
    assert parent_create_call.kwargs["name"] == "MathAgent_execution"
    assert "2 + 3" in parent_create_call.kwargs["input"]

    # Verify context has parent span ID
    assert agent.context.parent_span_id == "parent-span-123"

    # Verify all LLM calls used parent_span_id
    for call in mock_opper_client.call_async.call_args_list:
        assert call.kwargs["parent_span_id"] == "parent-span-123"

    # Verify parent span was updated with final output (last update call)
    parent_update_calls = [
        call
        for call in mock_opper_client.spans.update_async.call_args_list
        if call.kwargs.get("span_id") == "parent-span-123"
    ]
    assert len(parent_update_calls) == 1
    assert "answer is 5" in parent_update_calls[0].kwargs["output"]


@pytest.mark.asyncio
async def test_tool_call_span_creation(mock_opper_client):
    """Test that each tool call creates its own span with proper hierarchy."""
    # Mock parent span
    parent_span = AsyncMock(id="parent-span-123")
    mock_opper_client.spans.create_async = AsyncMock(
        side_effect=[
            parent_span,  # Parent span for agent execution
            AsyncMock(id="tool-span-1"),  # Span for first tool call
            AsyncMock(id="tool-span-2"),  # Span for second tool call
        ]
    )
    mock_opper_client.spans.update_async = AsyncMock()

    # Mock LLM calls
    mock_opper_client.call_async = AsyncMock(
        side_effect=[
            # Think: call two tools
            AsyncMock(
                json_payload={
                    "reasoning": "Need to calculate",
                    "tool_calls": [
                        {
                            "name": "add",
                            "parameters": {"a": 2, "b": 3},
                            "reasoning": "Add first",
                        },
                        {
                            "name": "multiply",
                            "parameters": {"x": 5, "y": 4},
                            "reasoning": "Multiply second",
                        },
                    ],
                    "user_message": "Calculating...",
                    "memory_updates": {},
                }
            ),
            # Think: done
            AsyncMock(
                json_payload={
                    "reasoning": "Complete",
                    "tool_calls": [],
                    "user_message": "Done",
                    "memory_updates": {},
                }
            ),
            # Generate final result
            AsyncMock(message="Results: 5 and 20"),
        ]
    )

    agent = Agent(
        name="MathAgent",
        tools=[add, multiply],
        verbose=False,
        opper_api_key="test-key",
    )
    result = await agent.process("Calculate")

    # Verify spans were created: 1 parent + 2 tool spans
    assert mock_opper_client.spans.create_async.call_count == 3

    # Check parent span
    parent_call = mock_opper_client.spans.create_async.call_args_list[0]
    assert parent_call.kwargs["name"] == "MathAgent_execution"

    # Check first tool span (add)
    tool1_call = mock_opper_client.spans.create_async.call_args_list[1]
    assert tool1_call.kwargs["name"] == "tool_add"
    assert "a" in tool1_call.kwargs["input"]
    assert tool1_call.kwargs["parent_id"] == "parent-span-123"

    # Check second tool span (multiply)
    tool2_call = mock_opper_client.spans.create_async.call_args_list[2]
    assert tool2_call.kwargs["name"] == "tool_multiply"
    assert "x" in tool2_call.kwargs["input"]
    assert tool2_call.kwargs["parent_id"] == "parent-span-123"

    # Verify tool spans were updated with results
    assert mock_opper_client.spans.update_async.call_count == 3  # 2 tools + 1 parent

    # Check tool result updates (first two calls are for tools)
    tool1_update = mock_opper_client.spans.update_async.call_args_list[0]
    assert tool1_update.kwargs["span_id"] == "tool-span-1"
    assert "5" in tool1_update.kwargs["output"]

    tool2_update = mock_opper_client.spans.update_async.call_args_list[1]
    assert tool2_update.kwargs["span_id"] == "tool-span-2"
    assert "20" in tool2_update.kwargs["output"]


@pytest.mark.asyncio
async def test_tool_call_span_with_error(mock_opper_client):
    """Test that tool span captures errors properly."""
    # Mock parent span and tool span
    parent_span = AsyncMock(id="parent-span-123")
    tool_span = AsyncMock(id="tool-span-error")
    mock_opper_client.spans.create_async = AsyncMock(
        side_effect=[parent_span, tool_span]
    )
    mock_opper_client.spans.update_async = AsyncMock()

    # Mock LLM calls
    mock_opper_client.call_async = AsyncMock(
        side_effect=[
            # Think: call failing tool
            AsyncMock(
                json_payload={
                    "reasoning": "Testing error",
                    "tool_calls": [
                        {"name": "failing_tool", "parameters": {}, "reasoning": "Test"}
                    ],
                    "user_message": "Working...",
                    "memory_updates": {},
                }
            ),
            # Think: done
            AsyncMock(
                json_payload={
                    "reasoning": "Complete",
                    "tool_calls": [],
                    "user_message": "Done",
                    "memory_updates": {},
                }
            ),
            # Generate final result
            AsyncMock(message="Handled error"),
        ]
    )

    agent = Agent(
        name="TestAgent",
        tools=[failing_tool],
        verbose=False,
        opper_api_key="test-key",
    )
    result = await agent.process("Test")

    # Verify tool span was updated with error
    tool_update_calls = [
        call
        for call in mock_opper_client.spans.update_async.call_args_list
        if call.kwargs.get("span_id") == "tool-span-error"
    ]
    assert len(tool_update_calls) == 1
    assert tool_update_calls[0].kwargs["output"] is None
    assert "Intentional failure" in tool_update_calls[0].kwargs["error"]


@pytest.mark.asyncio
async def test_agent_input_schema_validation(mock_opper_client):
    """Test agent input validation with Pydantic schema."""
    from pydantic import BaseModel

    class TaskInput(BaseModel):
        task: str
        priority: int = 1

    mock_opper_client.call_async = AsyncMock(
        side_effect=[
            AsyncMock(
                json_payload={
                    "reasoning": "Processing task",
                    "tool_calls": [],
                    "user_message": "Done",
                    "memory_updates": {},
                }
            ),
            AsyncMock(message="Task completed"),
        ]
    )

    agent = Agent(
        name="TaskAgent",
        tools=[],
        input_schema=TaskInput,
        verbose=False,
        opper_api_key="test-key",
    )

    # Test with dict input (should be validated)
    result = await agent.process({"task": "test task", "priority": 5})
    assert agent.context.goal.task == "test task"
    assert agent.context.goal.priority == 5
