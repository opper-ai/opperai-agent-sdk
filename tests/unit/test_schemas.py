"""
Unit tests for core schemas.

Tests the Pydantic models used for agent reasoning and tool execution.
"""

import pytest
from opper_agent.core.schemas import ToolCall, Thought, MemoryDecision


def test_tool_call_creation():
    """Test ToolCall model creation."""
    tool_call = ToolCall(
        name="search",
        parameters={"query": "test", "limit": 10},
        reasoning="Need to search for information",
    )

    assert tool_call.name == "search"
    assert tool_call.parameters["query"] == "test"
    assert tool_call.parameters["limit"] == 10
    assert "search" in tool_call.reasoning


def test_tool_call_default_parameters():
    """Test ToolCall with default empty parameters."""
    tool_call = ToolCall(name="no_args_tool", reasoning="Simple tool")

    assert tool_call.name == "no_args_tool"
    assert tool_call.parameters == {}


def test_thought_creation():
    """Test Thought model creation with tool calls."""
    thought = Thought(
        reasoning="I need to gather information first",
        tool_calls=[
            ToolCall(
                name="search", parameters={"query": "test"}, reasoning="Search first"
            ),
            ToolCall(name="analyze", parameters={}, reasoning="Then analyze"),
        ],
        user_message="Working on it...",
    )

    assert "gather information" in thought.reasoning
    assert len(thought.tool_calls) == 2
    assert thought.tool_calls[0].name == "search"
    assert thought.user_message == "Working on it..."


def test_thought_empty_tool_calls():
    """Test Thought with empty tool calls (task complete signal)."""
    thought = Thought(
        reasoning="Task is complete, no more actions needed", tool_calls=[]
    )

    assert len(thought.tool_calls) == 0
    assert thought.reasoning == "Task is complete, no more actions needed"


def test_thought_default_values():
    """Test Thought default values."""
    thought = Thought(reasoning="Just reasoning")

    assert thought.reasoning == "Just reasoning"
    assert thought.tool_calls == []
    assert thought.user_message == "Working on it..."
    assert thought.memory_updates == {}


def test_thought_with_memory_updates():
    """Test Thought with memory updates."""
    thought = Thought(
        reasoning="Saving to memory",
        memory_updates={
            "project_status": {
                "value": {"status": "in_progress"},
                "description": "Project state",
                "metadata": {"updated_by": "agent"},
            }
        },
    )

    assert "project_status" in thought.memory_updates
    assert thought.memory_updates["project_status"]["value"]["status"] == "in_progress"


def test_memory_decision_creation():
    """Test MemoryDecision model creation."""
    decision = MemoryDecision(
        should_use_memory=True,
        selected_keys=["key1", "key2"],
        rationale="These keys are relevant to current task",
    )

    assert decision.should_use_memory is True
    assert len(decision.selected_keys) == 2
    assert "key1" in decision.selected_keys
    assert "relevant" in decision.rationale


def test_memory_decision_no_memory():
    """Test MemoryDecision when memory should not be used."""
    decision = MemoryDecision(
        should_use_memory=False, selected_keys=[], rationale="No relevant memory"
    )

    assert decision.should_use_memory is False
    assert len(decision.selected_keys) == 0


def test_memory_decision_default_keys():
    """Test MemoryDecision with default empty keys."""
    decision = MemoryDecision(should_use_memory=False, rationale="No memory needed")

    assert decision.should_use_memory is False
    assert decision.selected_keys == []
