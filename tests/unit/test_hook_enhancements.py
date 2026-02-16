"""Tests for hook system enhancements: on/once/off, memory hooks, tool_call_id."""

import pytest
from opper_agents.base.hooks import HookManager, HookEvents
from opper_agents.base.context import AgentContext


def test_memory_hook_events_exist():
    """Test that memory hook event constants exist."""
    assert HookEvents.MEMORY_READ == "memory_read"
    assert HookEvents.MEMORY_WRITE == "memory_write"
    assert HookEvents.MEMORY_ERROR == "memory_error"


def test_hook_manager_on():
    """Test on() registers a hook and returns cleanup function."""
    manager = HookManager()
    calls = []

    def handler(ctx, **kwargs):
        calls.append(kwargs)

    cleanup = manager.on("test_event", handler)

    assert callable(cleanup)
    assert manager.has_hooks("test_event")
    assert manager.get_hook_count() == 1


def test_hook_manager_on_cleanup():
    """Test that cleanup function from on() removes the hook."""
    manager = HookManager()

    def handler(ctx, **kwargs):
        pass

    cleanup = manager.on("test_event", handler)
    assert manager.has_hooks("test_event")

    cleanup()
    assert not manager.has_hooks("test_event")


def test_hook_manager_off():
    """Test off() removes a specific hook."""
    manager = HookManager()
    calls = []

    def handler_a(ctx, **kwargs):
        calls.append("a")

    def handler_b(ctx, **kwargs):
        calls.append("b")

    manager.on("test_event", handler_a)
    manager.on("test_event", handler_b)
    assert manager.get_hook_count() == 2

    manager.off("test_event", handler_a)
    assert manager.get_hook_count() == 1


def test_hook_manager_off_nonexistent():
    """Test off() doesn't fail for non-registered hooks."""
    manager = HookManager()

    def handler(ctx, **kwargs):
        pass

    # Should not raise
    manager.off("nonexistent_event", handler)


@pytest.mark.asyncio
async def test_hook_manager_once():
    """Test once() hook fires only once then auto-removes."""
    manager = HookManager()
    calls = []

    def handler(ctx, **kwargs):
        calls.append(1)

    manager.once("test_event", handler)
    assert manager.has_hooks("test_event")

    ctx = AgentContext(agent_name="Test")
    await manager.trigger("test_event", ctx)
    assert len(calls) == 1

    # Should auto-remove after first trigger
    await manager.trigger("test_event", ctx)
    assert len(calls) == 1  # Still 1, not called again


@pytest.mark.asyncio
async def test_hook_manager_once_async():
    """Test once() with async handler."""
    manager = HookManager()
    calls = []

    async def handler(ctx, **kwargs):
        calls.append(1)

    manager.once("test_event", handler)

    ctx = AgentContext(agent_name="Test")
    await manager.trigger("test_event", ctx)
    assert len(calls) == 1

    await manager.trigger("test_event", ctx)
    assert len(calls) == 1  # Not called again


@pytest.mark.asyncio
async def test_hook_manager_once_cleanup():
    """Test that cleanup from once() removes before trigger."""
    manager = HookManager()
    calls = []

    def handler(ctx, **kwargs):
        calls.append(1)

    cleanup = manager.once("test_event", handler)
    cleanup()  # Remove before triggering

    ctx = AgentContext(agent_name="Test")
    await manager.trigger("test_event", ctx)
    assert len(calls) == 0


@pytest.mark.asyncio
async def test_on_trigger_with_kwargs():
    """Test that hooks receive kwargs correctly."""
    manager = HookManager()
    received = []

    def handler(ctx, **kwargs):
        received.append(kwargs)

    manager.on("test_event", handler)

    ctx = AgentContext(agent_name="Test")
    await manager.trigger("test_event", ctx, tool_call_id="abc-123", extra="data")

    assert len(received) == 1
    assert received[0]["tool_call_id"] == "abc-123"
    assert received[0]["extra"] == "data"
