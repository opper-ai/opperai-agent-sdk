"""
Core schemas for agent reasoning and tool execution.

This module defines the structured outputs used by agents during execution.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class ToolCall(BaseModel):
    """Represents a single tool invocation."""

    name: str = Field(description="Tool name to call")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Parameters to pass to tool"
    )
    reasoning: str = Field(description="Why this tool should be called")


class Thought(BaseModel):
    """
    Agent's reasoning and action plan.

    Key insight: Empty tool_calls list indicates task completion.
    """

    reasoning: str = Field(description="Analysis of current situation")
    tool_calls: List[ToolCall] = Field(
        default=[], description="Tools to call (empty means task is complete)"
    )
    user_message: str = Field(
        default="Working on it...", description="Status message for user"
    )
    memory_updates: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Memory writes the model wants to perform (key -> payload)",
    )


class MemoryDecision(BaseModel):
    """Structure returned by the memory router LLM call."""

    should_use_memory: bool = Field(description="Whether memory should be loaded")
    selected_keys: List[str] = Field(default_factory=list, description="Keys to load")
    rationale: str = Field(description="Short reason for the decision")
