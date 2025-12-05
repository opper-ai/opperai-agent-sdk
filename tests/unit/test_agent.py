"""
Unit tests for main Agent implementation.

Tests the think-act loop, tool execution, and memory integration.
"""

import pytest
from unittest.mock import AsyncMock
from opper_agents.core.agent import Agent
from opper_agents.utils.decorators import tool


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
    await agent.process("Calculate")

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
    await agent.process("Test")

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
    await agent.process("Test")

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

    await agent.process("Keep calculating")

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
    await agent.process("Save state")

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
    await agent.process("What is 2 + 3?")

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
    mock_opper_client.spans.get_async = AsyncMock()  # For span existence check

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
    await agent.process("Calculate")

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
    # 2 tools + 1 parent + 2 think spans (renaming)
    assert mock_opper_client.spans.update_async.call_count == 5

    # Check tool result updates
    tool1_update = [
        c
        for c in mock_opper_client.spans.update_async.call_args_list
        if c.kwargs.get("span_id") == "tool-span-1"
    ][0]
    assert "5" in tool1_update.kwargs["output"]

    tool2_update = [
        c
        for c in mock_opper_client.spans.update_async.call_args_list
        if c.kwargs.get("span_id") == "tool-span-2"
    ][0]
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
    await agent.process("Test")

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
    await agent.process({"task": "test task", "priority": 5})
    assert agent.context.goal.task == "test task"
    assert agent.context.goal.priority == 5


@pytest.mark.asyncio
async def test_agent_usage_tracking(mock_opper_client):
    """Test that agent properly tracks token usage from LLM calls."""
    # Mock LLM calls with usage information
    mock_opper_client.call_async = AsyncMock(
        side_effect=[
            # First think call
            AsyncMock(
                json_payload={
                    "reasoning": "Need to add numbers",
                    "tool_calls": [
                        {
                            "name": "add",
                            "parameters": {"a": 5, "b": 3},
                            "reasoning": "Adding",
                        }
                    ],
                    "user_message": "Calculating...",
                    "memory_updates": {},
                },
                usage={
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "total_tokens": 150,
                },
            ),
            # Second think call
            AsyncMock(
                json_payload={
                    "reasoning": "Done",
                    "tool_calls": [],
                    "user_message": "Complete",
                    "memory_updates": {},
                },
                usage={
                    "input_tokens": 120,
                    "output_tokens": 30,
                    "total_tokens": 150,
                },
            ),
            # Final result call
            AsyncMock(
                message="The sum is 8",
                usage={
                    "input_tokens": 80,
                    "output_tokens": 20,
                    "total_tokens": 100,
                },
            ),
        ]
    )

    agent = Agent(
        name="MathAgent", tools=[add], verbose=False, opper_api_key="test-key"
    )
    await agent.process("What is 5 + 3?")

    # Verify usage was tracked
    assert agent.context.usage.requests == 3  # Three LLM calls
    assert agent.context.usage.input_tokens == 300  # 100 + 120 + 80
    assert agent.context.usage.output_tokens == 100  # 50 + 30 + 20
    assert agent.context.usage.total_tokens == 400  # 150 + 150 + 100


@pytest.mark.asyncio
async def test_agent_with_memory_reads(mock_opper_client):
    """Test agent writing and reading from memory."""
    # Mock spans for tool and memory operations
    parent_span = AsyncMock(id="parent-span-123")
    tool_span = AsyncMock(id="tool-span-1")
    memory_write_span = AsyncMock(id="memory-write-span-1")
    memory_read_span = AsyncMock(id="memory-read-span-1")

    mock_opper_client.spans.create_async = AsyncMock(
        side_effect=[parent_span, tool_span, memory_write_span, memory_read_span]
    )
    mock_opper_client.spans.update_async = AsyncMock()

    mock_opper_client.call_async = AsyncMock(
        side_effect=[
            # Think 1: Use a tool first (to continue loop)
            AsyncMock(
                json_payload={
                    "reasoning": "Need to calculate and save",
                    "tool_calls": [
                        {
                            "name": "add",
                            "parameters": {"a": 500, "b": 500},
                            "reasoning": "Calculate",
                        }
                    ],
                    "user_message": "Calculating...",
                    "memory_reads": [],
                    "memory_updates": {
                        "budget": {
                            "value": 1000,
                            "description": "Total budget",
                        }
                    },
                }
            ),
            # Think 2: Read from memory (loop continues because of memory_reads)
            AsyncMock(
                json_payload={
                    "reasoning": "Need to retrieve budget from memory",
                    "tool_calls": [],
                    "user_message": "Loading...",
                    "memory_reads": ["budget"],
                    "memory_updates": {},
                }
            ),
            # Think 3: Use loaded memory, task complete
            AsyncMock(
                json_payload={
                    "reasoning": "Budget loaded, task complete",
                    "tool_calls": [],
                    "user_message": "Done",
                    "memory_reads": [],
                    "memory_updates": {},
                }
            ),
            # Generate result
            AsyncMock(message="Budget is 1000"),
        ]
    )

    agent = Agent(
        name="MemoryAgent",
        tools=[add],  # Add the tool for the first iteration
        enable_memory=True,
        verbose=False,
        opper_api_key="test-key",
    )
    await agent.process("Check budget")

    # Verify memory was written
    assert agent.context.memory.has_entries()
    memory_data = await agent.context.memory.read(["budget"])
    assert memory_data["budget"] == 1000

    # Verify the loop continued through all 3 iterations as expected
    assert mock_opper_client.call_async.call_count == 4  # 3 thinks + 1 final


@pytest.mark.asyncio
async def test_agent_memory_loop_continuation(mock_opper_client):
    """Test that loop continues when memory_reads are present even with no tool_calls."""
    parent_span = AsyncMock(id="parent-span-123")
    memory_write_span = AsyncMock(id="memory-write-span-1")
    memory_read_span = AsyncMock(id="memory-read-span-1")

    mock_opper_client.spans.create_async = AsyncMock(
        side_effect=[parent_span, memory_write_span, memory_read_span]
    )
    mock_opper_client.spans.update_async = AsyncMock()

    mock_opper_client.call_async = AsyncMock(
        side_effect=[
            # Iteration 1: Write to memory AND signal intent to read (continues loop)
            AsyncMock(
                json_payload={
                    "reasoning": "Storing value and need to read it back",
                    "tool_calls": [],  # No tool calls
                    "user_message": "Saving...",
                    "memory_reads": ["key"],  # Signal to read on next iteration
                    "memory_updates": {"key": {"value": "data", "description": "Data"}},
                }
            ),
            # Iteration 2: Read from memory is done, now complete
            AsyncMock(
                json_payload={
                    "reasoning": "Data loaded from memory, task complete",
                    "tool_calls": [],  # No tool calls
                    "user_message": "Done",
                    "memory_reads": [],  # No more reads needed
                    "memory_updates": {},
                }
            ),
            # Generate result
            AsyncMock(message="Complete"),
        ]
    )

    agent = Agent(
        name="MemoryAgent",
        tools=[],
        enable_memory=True,
        verbose=True,  # Enable verbose to see what's happening
        opper_api_key="test-key",
    )
    await agent.process("Test memory continuation")

    # Should have 2 think calls (iteration 1 continues due to memory_reads)
    assert mock_opper_client.call_async.call_count == 3  # 2 thinks + 1 final result

    # Verify memory was written
    assert agent.context.memory.has_entries()
    memory_data = await agent.context.memory.read(["key"])
    assert memory_data["key"] == "data"
