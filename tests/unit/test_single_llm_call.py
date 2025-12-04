"""
Tests for the single LLM call pattern implementation.

This test file verifies that agents can return final results in a single LLM call
when the task is immediately completable, avoiding the redundant second call.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from opper_agents import Agent
from opper_agents.agents import ReactAgent
from opper_agents import tool
from opper_agents.core.schemas import Thought, ReactThought
from pydantic import BaseModel


# Test tools
@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
def multiply(x: int, y: int) -> int:
    """Multiply two numbers."""
    return x * y


# Output schemas for testing
class SimpleAnswer(BaseModel):
    answer: str
    confidence: float


class MathResult(BaseModel):
    result: int
    explanation: str


@pytest.mark.asyncio
async def test_agent_single_call_immediate_response(mock_opper_client, monkeypatch):
    """Test that simple questions return in a single LLM call."""
    # Set mock API key
    monkeypatch.setenv("OPPER_API_KEY", "test-key")

    # Mock the LLM to return a complete response immediately
    mock_opper_client.call_async.return_value = AsyncMock(
        json_payload={
            "reasoning": "This is a simple factual question I can answer directly",
            "is_complete": True,
            "final_result": "The capital of France is Paris",
            "tool_calls": [],
            "user_message": "Here's your answer",
            "memory_reads": [],
            "memory_updates": {},
        },
        usage=MagicMock(requests=1, total_tokens=100),
    )

    agent = Agent(name="TestAgent", tools=[])
    result = await agent.process("What is the capital of France?")

    # Should return the result directly
    assert result == "The capital of France is Paris"

    # Should only make ONE LLM call (no _generate_final_result call)
    assert mock_opper_client.call_async.call_count == 1

    # Verify the call was for thinking (dynamic name: think_{agent_name})
    call_args = mock_opper_client.call_async.call_args
    assert call_args.kwargs["name"] == "think_testagent"


@pytest.mark.asyncio
async def test_agent_single_call_with_output_schema(mock_opper_client, monkeypatch):
    """Test single call with structured output schema."""
    monkeypatch.setenv("OPPER_API_KEY", "test-key")
    mock_opper_client.call_async.return_value = AsyncMock(
        json_payload={
            "reasoning": "I can provide a direct answer with confidence",
            "is_complete": True,
            "final_result": {
                "answer": "Python is a high-level programming language",
                "confidence": 0.95,
            },
            "tool_calls": [],
            "user_message": "Here's your answer",
            "memory_reads": [],
            "memory_updates": {},
        },
        usage=MagicMock(requests=1, total_tokens=120),
    )

    agent = Agent(name="TestAgent", tools=[], output_schema=SimpleAnswer)
    result = await agent.process("What is Python?")

    # Should return validated Pydantic model instance
    assert isinstance(result, SimpleAnswer)
    assert result.answer == "Python is a high-level programming language"
    assert result.confidence == 0.95

    # Only one LLM call
    assert mock_opper_client.call_async.call_count == 1


@pytest.mark.asyncio
async def test_agent_tool_usage_then_final_result(mock_opper_client, monkeypatch):
    """Test that agent uses tools first, then provides final result in next think call."""
    monkeypatch.setenv("OPPER_API_KEY", "test-key")
    # First call: decide to use tool
    mock_opper_client.call_async.side_effect = [
        AsyncMock(
            json_payload={
                "reasoning": "I need to calculate this sum",
                "is_complete": False,
                "final_result": None,
                "tool_calls": [
                    {
                        "name": "add",
                        "parameters": {"a": 5, "b": 3},
                        "reasoning": "Adding the numbers",
                    }
                ],
                "user_message": "Calculating...",
                "memory_reads": [],
                "memory_updates": {},
            },
            usage=MagicMock(requests=1, total_tokens=80),
        ),
        # Second call: task complete with result
        AsyncMock(
            json_payload={
                "reasoning": "I have calculated the sum successfully",
                "is_complete": True,
                "final_result": {"result": 8, "explanation": "5 + 3 = 8"},
                "tool_calls": [],
                "user_message": "Calculation complete",
                "memory_reads": [],
                "memory_updates": {},
            },
            usage=MagicMock(requests=1, total_tokens=90),
        ),
    ]

    agent = Agent(name="MathAgent", tools=[add, multiply], output_schema=MathResult)
    result = await agent.process("What is 5 + 3?")

    # Should return validated Pydantic model instance
    assert isinstance(result, MathResult)
    assert result.result == 8
    assert result.explanation == "5 + 3 = 8"

    # Should make exactly 2 LLM calls (two think calls, no generate_final_result)
    assert mock_opper_client.call_async.call_count == 2

    # Both calls should be think calls (dynamic name: think_{agent_name})
    for call in mock_opper_client.call_async.call_args_list:
        assert call.kwargs["name"] == "think_mathagent"


@pytest.mark.asyncio
async def test_react_agent_single_call(mock_opper_client, monkeypatch):
    """Test ReactAgent with single LLM call pattern."""
    monkeypatch.setenv("OPPER_API_KEY", "test-key")
    mock_opper_client.call_async.return_value = AsyncMock(
        json_payload={
            "reasoning": "This is a straightforward question I can answer",
            "is_complete": True,
            "action": None,
            "user_message": "Task complete",
            "final_result": "React agents use a Reasoning + Acting pattern",
        },
        usage=MagicMock(requests=1, total_tokens=110),
    )

    agent = ReactAgent(name="ReactTestAgent", tools=[])
    result = await agent.process("What is a React agent?")

    # Should return the result directly
    assert result == "React agents use a Reasoning + Acting pattern"

    # Only one LLM call
    assert mock_opper_client.call_async.call_count == 1

    # Should be a reason call (dynamic name: reason_{agent_name})
    assert (
        mock_opper_client.call_async.call_args.kwargs["name"] == "reason_reacttestagent"
    )


@pytest.mark.asyncio
async def test_backward_compatibility_empty_tool_calls(mock_opper_client, monkeypatch):
    """Test backward compatibility when is_complete is False and tool_calls is empty."""
    monkeypatch.setenv("OPPER_API_KEY", "test-key")
    # First think call returns old-style response (no is_complete/final_result)
    mock_opper_client.call_async.side_effect = [
        AsyncMock(
            json_payload={
                "reasoning": "Task is done",
                "tool_calls": [],  # Empty means complete in old pattern
                "user_message": "Complete",
                "memory_reads": [],
                "memory_updates": {},
                # Note: is_complete defaults to False, final_result defaults to None
            },
            usage=MagicMock(requests=1, total_tokens=70),
        ),
        # Falls back to generate_final_result call
        MagicMock(
            json_payload="The task has been completed successfully",
            usage=MagicMock(requests=1, total_tokens=50),
            message="The task has been completed successfully",  # Some responses use .message
        ),
    ]

    agent = Agent(name="BackwardCompatAgent", tools=[])
    result = await agent.process("Do something")

    # Should still work and return result
    assert result == "The task has been completed successfully"

    # Should make 2 calls (think + generate_final_result for backward compat)
    assert mock_opper_client.call_async.call_count == 2

    # First call is think (dynamic name), second is generate_final_result (also dynamic)
    assert (
        mock_opper_client.call_async.call_args_list[0].kwargs["name"]
        == "think_backwardcompatagent"
    )
    assert (
        mock_opper_client.call_async.call_args_list[1].kwargs["name"]
        == "generate_final_result_backwardcompatagent"
    )


@pytest.mark.asyncio
async def test_max_iterations_without_completion(mock_opper_client, monkeypatch):
    """Test that agent handles max iterations gracefully."""
    monkeypatch.setenv("OPPER_API_KEY", "test-key")
    # Always return incomplete thoughts
    mock_opper_client.call_async.return_value = AsyncMock(
        json_payload={
            "reasoning": "Still working",
            "is_complete": False,
            "final_result": None,
            "tool_calls": [],  # No tools but not complete
            "user_message": "Processing...",
            "memory_reads": [],
            "memory_updates": {},
        },
        usage=MagicMock(requests=1, total_tokens=60),
    )

    agent = Agent(name="TestAgent", tools=[], max_iterations=3)

    # Should eventually hit max iterations and fall back
    await agent.process("Complex task")

    # When tool_calls is empty, loop exits even if is_complete=False
    # So it makes 1 think call + 1 generate_final_result as fallback
    assert mock_opper_client.call_async.call_count == 2


@pytest.mark.asyncio
async def test_single_call_validates_output_schema(mock_opper_client, monkeypatch):
    """Test that single LLM call validates and converts to output_schema."""
    monkeypatch.setenv("OPPER_API_KEY", "test-key")
    mock_opper_client.call_async.return_value = AsyncMock(
        json_payload={
            "reasoning": "I can answer this directly",
            "is_complete": True,
            "final_result": {"result": 42, "explanation": "The answer to everything"},
            "tool_calls": [],
            "user_message": "Done",
            "memory_reads": [],
            "memory_updates": {},
        },
        usage=MagicMock(requests=1, total_tokens=100),
    )

    agent = Agent(name="TestAgent", tools=[], output_schema=MathResult)
    result = await agent.process("What is the meaning of life?")

    # Result should be validated as MathResult Pydantic model
    assert isinstance(result, MathResult)
    assert result.result == 42
    assert result.explanation == "The answer to everything"

    # Only one LLM call
    assert mock_opper_client.call_async.call_count == 1


@pytest.mark.asyncio
async def test_thought_schema_new_fields():
    """Test that Thought schema has the new is_complete and final_result fields."""
    # Test default values
    thought = Thought(reasoning="Test reasoning")
    assert thought.is_complete is False
    assert thought.final_result is None
    assert thought.tool_calls == []

    # Test with is_complete and final_result set
    thought_complete = Thought(
        reasoning="Task done",
        is_complete=True,
        final_result={"answer": "42", "explanation": "The meaning of life"},
        tool_calls=[],
    )
    assert thought_complete.is_complete is True
    assert thought_complete.final_result == {
        "answer": "42",
        "explanation": "The meaning of life",
    }

    # Test ReactThought also has final_result
    react_thought = ReactThought(
        reasoning="React reasoning",
        is_complete=True,
        final_result="Direct answer",
    )
    assert react_thought.is_complete is True
    assert react_thought.final_result == "Direct answer"
