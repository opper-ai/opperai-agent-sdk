"""
Hook system for agent lifecycle events.

This module provides the hook manager and event system for tracking
and responding to agent execution events.
"""

from typing import Dict, List, Callable, Any, Optional
from .context import AgentContext
import asyncio
import logging

logger = logging.getLogger(__name__)

# Type alias for hook functions
HookFunction = Callable[..., Any]


class HookEvents:
    """Standard hook event names."""

    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    AGENT_ERROR = "agent_error"

    LOOP_START = "loop_start"
    LOOP_END = "loop_end"

    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    TOOL_ERROR = "tool_error"

    LLM_CALL = "llm_call"
    LLM_RESPONSE = "llm_response"
    THINK_END = "think_end"

    # Streaming events
    STREAM_START = "stream_start"
    STREAM_CHUNK = "stream_chunk"
    STREAM_END = "stream_end"
    STREAM_ERROR = "stream_error"


class HookManager:
    """
    Manages hook registration and execution.
    Supports both class-based and decorator-based hooks.
    """

    def __init__(self, verbose: bool = False):
        self.hooks: Dict[str, List[HookFunction]] = {}
        self.verbose = verbose

    def register(self, event: str, hook: HookFunction) -> None:
        """Register a hook function for an event."""
        if event not in self.hooks:
            self.hooks[event] = []
        self.hooks[event].append(hook)

        if self.verbose:
            logger.info(f"Registered hook for event: {event}")

    def register_multiple(self, hooks: List[tuple[str, HookFunction]]) -> None:
        """Register multiple hooks at once. Each tuple is (event, hook_func)."""
        for event, hook in hooks:
            self.register(event, hook)

    async def trigger(self, event: str, context: AgentContext, **kwargs: Any) -> None:
        """
        Trigger all hooks for an event.
        Hooks that fail don't stop execution.
        """
        if event not in self.hooks:
            return

        for hook_func in self.hooks[event]:
            try:
                # Handle both sync and async hooks
                if asyncio.iscoroutinefunction(hook_func):
                    await hook_func(context, **kwargs)
                else:
                    hook_func(context, **kwargs)

            except Exception as e:
                logger.warning(f"Hook '{event}' failed: {e}")
                # Don't break execution if hook fails

    def has_hooks(self, event: str) -> bool:
        """Check if any hooks are registered for an event."""
        return event in self.hooks and len(self.hooks[event]) > 0

    def clear_hooks(self, event: Optional[str] = None) -> None:
        """Clear hooks for a specific event, or all hooks if event is None."""
        if event:
            self.hooks.pop(event, None)
        else:
            self.hooks.clear()

    def get_hook_count(self) -> int:
        """Get total number of registered hooks."""
        return sum(len(hooks) for hooks in self.hooks.values())
