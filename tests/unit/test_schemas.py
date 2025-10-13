"""
Unit tests for core schemas.

Tests the Pydantic models used for agent reasoning and tool execution.
"""

from opper_agents.core.schemas import ToolCall, Thought


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


def test_thought_with_memory_reads():
    """Test Thought with memory reads."""
    thought = Thought(
        reasoning="Loading from memory",
        memory_reads=["project_status", "user_preferences"],
        tool_calls=[],
    )

    assert len(thought.memory_reads) == 2
    assert "project_status" in thought.memory_reads
    assert "user_preferences" in thought.memory_reads


def test_thought_memory_reads_default():
    """Test Thought with default empty memory_reads."""
    thought = Thought(reasoning="No memory needed")

    assert thought.memory_reads == []
