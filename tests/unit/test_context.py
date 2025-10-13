"""
Unit tests for context module.

Tests for Usage, AgentContext, and ExecutionCycle data models.
"""

from opper_agents.base.context import Usage, AgentContext, ExecutionCycle


def test_usage_addition():
    """Test combining usage statistics."""
    u1 = Usage(requests=1, input_tokens=100, output_tokens=50, total_tokens=150)
    u2 = Usage(requests=2, input_tokens=200, output_tokens=100, total_tokens=300)
    combined = u1.add(u2)
    assert combined.requests == 3
    assert combined.total_tokens == 450
    assert combined.input_tokens == 300
    assert combined.output_tokens == 150


def test_usage_repr():
    """Test usage string representation."""
    u = Usage(requests=5, total_tokens=1000)
    assert "requests=5" in repr(u)
    assert "tokens=1000" in repr(u)


def test_agent_context_initialization():
    """Test AgentContext initialization with defaults."""
    ctx = AgentContext(agent_name="TestAgent")
    assert ctx.agent_name == "TestAgent"
    assert ctx.iteration == 0
    assert len(ctx.execution_history) == 0
    assert ctx.session_id is not None
    assert ctx.usage.requests == 0


def test_context_add_cycle():
    """Test adding execution cycles to context."""
    ctx = AgentContext(agent_name="Test")
    cycle = ExecutionCycle(iteration=1)
    ctx.add_cycle(cycle)
    assert ctx.iteration == 1
    assert len(ctx.execution_history) == 1
    assert ctx.execution_history[0] == cycle


def test_context_usage_tracking():
    """Test cumulative usage tracking."""
    ctx = AgentContext(agent_name="Test")
    ctx.update_usage(Usage(requests=1, total_tokens=100))
    ctx.update_usage(Usage(requests=1, total_tokens=200))
    assert ctx.usage.requests == 2
    assert ctx.usage.total_tokens == 300


def test_context_get_last_n_cycles():
    """Test retrieving last N cycles."""
    ctx = AgentContext(agent_name="Test")
    for i in range(5):
        ctx.add_cycle(ExecutionCycle(iteration=i))

    last_3 = ctx.get_last_n_cycles(3)
    assert len(last_3) == 3
    assert last_3[0].iteration == 2
    assert last_3[-1].iteration == 4


def test_context_get_last_n_cycles_empty():
    """Test get_last_n_cycles with empty history."""
    ctx = AgentContext(agent_name="Test")
    cycles = ctx.get_last_n_cycles(3)
    assert cycles == []


def test_context_clear_history():
    """Test clearing execution history."""
    ctx = AgentContext(agent_name="Test")
    ctx.add_cycle(ExecutionCycle(iteration=1))
    ctx.add_cycle(ExecutionCycle(iteration=2))
    assert len(ctx.execution_history) == 2

    ctx.clear_history()
    assert len(ctx.execution_history) == 0
    assert ctx.iteration == 2  # iteration counter is not reset


def test_context_metadata():
    """Test metadata storage."""
    ctx = AgentContext(agent_name="Test")
    ctx.metadata["custom_key"] = "custom_value"
    ctx.metadata["count"] = 42

    assert ctx.metadata["custom_key"] == "custom_value"
    assert ctx.metadata["count"] == 42


def test_execution_cycle_creation():
    """Test ExecutionCycle model creation."""
    cycle = ExecutionCycle(iteration=1, thought="test thought")
    assert cycle.iteration == 1
    assert cycle.thought == "test thought"
    assert cycle.timestamp > 0
    assert len(cycle.tool_calls) == 0
    assert len(cycle.results) == 0


def test_context_get_context_size():
    """Test getting context size in tokens."""
    ctx = AgentContext(agent_name="Test")
    ctx.update_usage(Usage(requests=1, total_tokens=500))
    assert ctx.get_context_size() == 500


def test_context_timestamps():
    """Test that timestamps are updated correctly."""
    ctx = AgentContext(agent_name="Test")
    initial_time = ctx.updated_at

    import time

    time.sleep(0.01)  # Small delay to ensure time difference
    ctx.update_usage(Usage(requests=1, total_tokens=100))

    assert ctx.updated_at > initial_time
