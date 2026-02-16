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


class Cost(BaseModel):
    """Tracks cost breakdown for agent execution."""

    generation: float = Field(default=0.0, description="Generation cost")
    platform: float = Field(default=0.0, description="Platform cost")
    total: float = Field(default=0.0, description="Total cost")

    def add(self, other: "Cost") -> "Cost":
        """Combine cost statistics."""
        return Cost(
            generation=self.generation + other.generation,
            platform=self.platform + other.platform,
            total=self.total + other.total,
        )


class BaseUsage(BaseModel):
    """Base usage statistics without breakdown (used in breakdown values)."""

    requests: int = Field(default=0, description="Number of LLM requests")
    input_tokens: int = Field(default=0, description="Input tokens used")
    output_tokens: int = Field(default=0, description="Output tokens used")
    total_tokens: int = Field(default=0, description="Total tokens")
    cost: Cost = Field(default_factory=Cost, description="Cost breakdown")

    def add(self, other: "BaseUsage") -> "BaseUsage":
        """Combine usage statistics."""
        return BaseUsage(
            requests=self.requests + other.requests,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cost=self.cost.add(other.cost),
        )

    def __repr__(self) -> str:
        return f"BaseUsage(requests={self.requests}, tokens={self.total_tokens})"


class Usage(BaseModel):
    """Tracks token usage across agent execution."""

    requests: int = Field(default=0, description="Number of LLM requests")
    input_tokens: int = Field(default=0, description="Input tokens used")
    output_tokens: int = Field(default=0, description="Output tokens used")
    total_tokens: int = Field(default=0, description="Total tokens")
    cost: Cost = Field(default_factory=Cost, description="Cost breakdown")
    breakdown: Optional[Dict[str, BaseUsage]] = Field(
        default=None, description="Per-source usage breakdown for nested agents"
    )

    def add(self, other: "Usage") -> "Usage":
        """Combine usage statistics."""
        # Merge breakdowns
        merged_breakdown: Optional[Dict[str, BaseUsage]] = None
        if self.breakdown or other.breakdown:
            merged_breakdown = dict(self.breakdown or {})
            for key, value in (other.breakdown or {}).items():
                if key in merged_breakdown:
                    merged_breakdown[key] = merged_breakdown[key].add(value)
                else:
                    merged_breakdown[key] = value

        return Usage(
            requests=self.requests + other.requests,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cost=self.cost.add(other.cost),
            breakdown=merged_breakdown,
        )

    def to_base_usage(self) -> BaseUsage:
        """Convert to BaseUsage (without breakdown)."""
        return BaseUsage(
            requests=self.requests,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            total_tokens=self.total_tokens,
            cost=self.cost,
        )

    def __repr__(self) -> str:
        return f"Usage(requests={self.requests}, tokens={self.total_tokens})"


class PendingSpanUpdate(BaseModel):
    """A span update to be flushed at the end of the run."""

    span_id: str = Field(description="Span ID to update")
    output: Optional[str] = Field(default=None, description="Span output")
    error: Optional[str] = Field(default=None, description="Error message")
    start_time: Optional[Any] = Field(default=None, description="Start time")
    end_time: Optional[Any] = Field(default=None, description="End time")
    name: Optional[str] = Field(default=None, description="Span name override")

    class Config:
        arbitrary_types_allowed = True


class RunResult(BaseModel):
    """Result from agent.run() - includes both result and usage statistics."""

    result: Any = Field(description="Agent execution result")
    usage: Usage = Field(default_factory=Usage, description="Token usage statistics")

    class Config:
        arbitrary_types_allowed = True


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

    # Deferred span updates
    pending_span_updates: List[PendingSpanUpdate] = Field(
        default_factory=list, description="Batched span updates to flush at end of run"
    )

    class Config:
        arbitrary_types_allowed = True

    def update_usage(self, usage: Usage) -> None:
        """Update cumulative usage statistics."""
        self.usage = self.usage.add(usage)
        self.updated_at = time.time()

    def update_usage_with_source(self, source: str, delta: Usage) -> None:
        """Update usage and track per-source breakdown for nested agents."""
        self.update_usage(delta)
        if self.usage.breakdown is None:
            self.usage.breakdown = {}
        existing = self.usage.breakdown.get(source, BaseUsage())
        self.usage.breakdown[source] = existing.add(delta.to_base_usage())

    def cleanup_breakdown_if_only_parent(self, parent_agent_name: str) -> None:
        """Remove breakdown if only the parent agent is present."""
        if self.usage.breakdown is None:
            return
        keys = list(self.usage.breakdown.keys())
        if len(keys) <= 1 and (not keys or keys[0] == parent_agent_name):
            self.usage.breakdown = None

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
