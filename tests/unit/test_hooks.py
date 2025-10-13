"""
Unit tests for hooks module.

Tests for HookManager, HookEvents, and hook execution.
"""

import pytest
from opper_agents.base.hooks import HookManager, HookEvents
from opper_agents.base.context import AgentContext


@pytest.mark.asyncio
async def test_hook_registration():
    """Test registering a hook."""
    manager = HookManager()

    async def test_hook(context: AgentContext):
        context.metadata["hook_called"] = True

    manager.register(HookEvents.AGENT_START, test_hook)
    assert manager.has_hooks(HookEvents.AGENT_START)


@pytest.mark.asyncio
async def test_hook_trigger():
    """Test triggering a hook."""
    manager = HookManager()
    context = AgentContext(agent_name="Test")

    async def test_hook(context: AgentContext):
        context.metadata["value"] = 42

    manager.register(HookEvents.AGENT_START, test_hook)
    await manager.trigger(HookEvents.AGENT_START, context)

    assert context.metadata["value"] == 42


@pytest.mark.asyncio
async def test_hook_failure_doesnt_break():
    """Test that hook failures don't break execution."""
    manager = HookManager()
    context = AgentContext(agent_name="Test")

    async def failing_hook(context: AgentContext):
        raise ValueError("Hook error")

    async def working_hook(context: AgentContext):
        context.metadata["success"] = True

    manager.register(HookEvents.AGENT_START, failing_hook)
    manager.register(HookEvents.AGENT_START, working_hook)

    # Should not raise, and working hook should execute
    await manager.trigger(HookEvents.AGENT_START, context)
    assert context.metadata.get("success") is True


@pytest.mark.asyncio
async def test_multiple_hooks():
    """Test multiple hooks for same event."""
    manager = HookManager()
    context = AgentContext(agent_name="Test")
    counter = {"count": 0}

    async def inc_hook(context: AgentContext):
        counter["count"] += 1

    manager.register(HookEvents.LOOP_START, inc_hook)
    manager.register(HookEvents.LOOP_START, inc_hook)

    await manager.trigger(HookEvents.LOOP_START, context)
    assert counter["count"] == 2


@pytest.mark.asyncio
async def test_sync_hook():
    """Test synchronous hook execution."""
    manager = HookManager()
    context = AgentContext(agent_name="Test")

    def sync_hook(context: AgentContext):
        context.metadata["sync"] = True

    manager.register(HookEvents.AGENT_START, sync_hook)
    await manager.trigger(HookEvents.AGENT_START, context)

    assert context.metadata["sync"] is True


@pytest.mark.asyncio
async def test_trigger_nonexistent_event():
    """Test triggering event with no registered hooks."""
    manager = HookManager()
    context = AgentContext(agent_name="Test")

    # Should not raise
    await manager.trigger("nonexistent_event", context)


def test_has_hooks():
    """Test checking if hooks are registered."""
    manager = HookManager()

    def test_hook(context: AgentContext):
        pass

    assert not manager.has_hooks(HookEvents.AGENT_START)

    manager.register(HookEvents.AGENT_START, test_hook)
    assert manager.has_hooks(HookEvents.AGENT_START)


def test_clear_specific_hooks():
    """Test clearing hooks for a specific event."""
    manager = HookManager()

    def hook1(context: AgentContext):
        pass

    def hook2(context: AgentContext):
        pass

    manager.register(HookEvents.AGENT_START, hook1)
    manager.register(HookEvents.AGENT_END, hook2)

    assert manager.has_hooks(HookEvents.AGENT_START)
    assert manager.has_hooks(HookEvents.AGENT_END)

    manager.clear_hooks(HookEvents.AGENT_START)

    assert not manager.has_hooks(HookEvents.AGENT_START)
    assert manager.has_hooks(HookEvents.AGENT_END)


def test_clear_all_hooks():
    """Test clearing all hooks."""
    manager = HookManager()

    def hook1(context: AgentContext):
        pass

    def hook2(context: AgentContext):
        pass

    manager.register(HookEvents.AGENT_START, hook1)
    manager.register(HookEvents.AGENT_END, hook2)

    assert manager.get_hook_count() == 2

    manager.clear_hooks()

    assert manager.get_hook_count() == 0
    assert not manager.has_hooks(HookEvents.AGENT_START)
    assert not manager.has_hooks(HookEvents.AGENT_END)


def test_register_multiple():
    """Test registering multiple hooks at once."""
    manager = HookManager()

    def hook1(context: AgentContext):
        pass

    def hook2(context: AgentContext):
        pass

    hooks = [
        (HookEvents.AGENT_START, hook1),
        (HookEvents.AGENT_END, hook2),
    ]

    manager.register_multiple(hooks)

    assert manager.has_hooks(HookEvents.AGENT_START)
    assert manager.has_hooks(HookEvents.AGENT_END)
    assert manager.get_hook_count() == 2


@pytest.mark.asyncio
async def test_hook_with_kwargs():
    """Test hooks receiving additional kwargs."""
    manager = HookManager()
    context = AgentContext(agent_name="Test")

    async def hook_with_args(context: AgentContext, **kwargs):
        context.metadata["extra_data"] = kwargs.get("data")

    manager.register(HookEvents.AGENT_START, hook_with_args)
    await manager.trigger(HookEvents.AGENT_START, context, data="test_value")

    assert context.metadata["extra_data"] == "test_value"


def test_hook_events_constants():
    """Test that all expected hook event constants exist."""
    assert hasattr(HookEvents, "AGENT_START")
    assert hasattr(HookEvents, "AGENT_END")
    assert hasattr(HookEvents, "AGENT_ERROR")
    assert hasattr(HookEvents, "LOOP_START")
    assert hasattr(HookEvents, "LOOP_END")
    assert hasattr(HookEvents, "TOOL_CALL")
    assert hasattr(HookEvents, "TOOL_RESULT")
    assert hasattr(HookEvents, "TOOL_ERROR")
    assert hasattr(HookEvents, "LLM_CALL")
    assert hasattr(HookEvents, "LLM_RESPONSE")
    assert hasattr(HookEvents, "THINK_END")
