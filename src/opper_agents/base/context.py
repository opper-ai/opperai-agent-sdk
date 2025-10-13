"""
Core data models for agent execution context.

This module provides the data structures for tracking agent execution state,
token usage, and execution history.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, TYPE_CHECKING
import time

if TYPE_CHECKING:
    from ..memory.memory import Memory


class Usage(BaseModel):
    """Tracks token usage across agent execution."""

    requests: int = Field(default=0, description="Number of LLM requests")
    input_tokens: int = Field(default=0, description="Input tokens used")
    output_tokens: int = Field(default=0, description="Output tokens used")
    total_tokens: int = Field(default=0, description="Total tokens")

    def add(self, other: "Usage") -> "Usage":
        """Combine usage statistics."""
        return Usage(
            requests=self.requests + other.requests,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )

    def __repr__(self) -> str:
        return f"Usage(requests={self.requests}, tokens={self.total_tokens})"


class ExecutionCycle(BaseModel):
    """Represents one think-act cycle in agent execution."""

    iteration: int = Field(description="Iteration number")
    thought: Optional[Any] = Field(default=None, description="Agent's reasoning")
    tool_calls: List[Any] = Field(default=[], description="Tools called")
    results: List[Any] = Field(default=[], description="Tool results")
    timestamp: float = Field(default_factory=time.time)

    class Config:
        arbitrary_types_allowed = True


class AgentContext(BaseModel):
    """
    Maintains all state for an agent execution session.
    Single source of truth for execution state, history, and metadata.
    """

    # Identity
    agent_name: str = Field(description="Name of the agent")
    session_id: str = Field(default_factory=lambda: str(time.time()))

    # Tracing
    parent_span_id: Optional[str] = Field(
        default=None, description="Parent span ID for all calls in this agent execution"
    )

    # Execution state
    iteration: int = Field(default=0, description="Current iteration")
    goal: Optional[Any] = Field(default=None, description="Current goal")

    # History
    execution_history: List[ExecutionCycle] = Field(
        default_factory=list, description="History of execution cycles"
    )

    # Token tracking
    usage: Usage = Field(default_factory=Usage, description="Token usage stats")

    # Memory (optional, will be None if not enabled)
    memory: Optional[Memory] = Field(default=None, description="Agent memory store")

    # Metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional context metadata"
    )

    # Timestamps
    started_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    class Config:
        arbitrary_types_allowed = True

    def update_usage(self, usage: Usage) -> None:
        """Update cumulative usage statistics."""
        self.usage = self.usage.add(usage)
        self.updated_at = time.time()

    def add_cycle(self, cycle: ExecutionCycle) -> None:
        """Add an execution cycle to history."""
        self.execution_history.append(cycle)
        self.iteration += 1
        self.updated_at = time.time()

    def get_context_size(self) -> int:
        """Get current context size in tokens."""
        return self.usage.total_tokens

    def get_last_n_cycles(self, n: int = 3) -> List[ExecutionCycle]:
        """Get last N execution cycles for context."""
        return self.execution_history[-n:] if self.execution_history else []

    def get_last_iterations_summary(self, n: int = 2) -> List[Dict[str, Any]]:
        """Condensed view of recent iterations for LLM context."""
        summary: List[Dict[str, Any]] = []
        for cycle in self.execution_history[-n:]:
            summary.append(
                {
                    "iteration": cycle.iteration,
                    "thought": getattr(cycle.thought, "reasoning", str(cycle.thought)),
                    "tool_calls": [
                        call.name for call in getattr(cycle, "tool_calls", [])
                    ],
                    "results": [
                        {"tool": result.tool_name, "success": result.success}
                        for result in getattr(cycle, "results", [])
                    ],
                }
            )
        return summary

    def clear_history(self) -> None:
        """Clear execution history (useful for long-running agents)."""
        self.execution_history.clear()


# Defer model rebuilding until Memory is imported
def _rebuild_if_memory_available() -> None:
    """Rebuild AgentContext model once Memory is available."""
    try:
        from ..memory.memory import Memory  # noqa: F401

        AgentContext.model_rebuild()
    except ImportError:
        # Memory not yet available, will be rebuilt later
        pass


# Attempt rebuild when this module is imported
_rebuild_if_memory_available()
