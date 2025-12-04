"""
Core schemas for agent reasoning and tool execution.

This module defines the structured outputs used by agents during execution.
"""

from pydantic import BaseModel, Field, create_model
from typing import List, Dict, Any, Optional, Type


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
    When is_complete=True, final_result should contain the task output.
    """

    reasoning: str = Field(description="Analysis of current situation")
    tool_calls: List[ToolCall] = Field(
        default=[], description="Tools to call (empty means task is complete)"
    )
    user_message: str = Field(
        default="Working on it...", description="Status message for user"
    )
    memory_reads: List[str] = Field(
        default_factory=list,
        description="Memory keys to load for this iteration (optional)",
    )
    memory_updates: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Memory writes the model wants to perform (key -> payload with value, description, metadata)",
    )
    # New fields for single LLM call pattern
    is_complete: bool = Field(
        default=False,
        description="Whether the task is complete and final_result is available",
    )
    final_result: Optional[Any] = Field(
        default=None,
        description="The final result when is_complete=True (should match output_schema if specified)",
    )


# ReAct Agent Schemas


class Action(BaseModel):
    """
    Represents an action in the ReAct pattern.

    An action specifies a tool to call with its parameters.
    """

    tool_name: str = Field(description="Name of the tool to execute")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Parameters for the tool"
    )


class ReactThought(BaseModel):
    """
    ReAct pattern reasoning step.

    The agent reasons about the current situation and decides on an action.
    If is_complete is True, the agent has finished and no action is needed.
    When is_complete=True, final_result should contain the task output.
    """

    reasoning: str = Field(
        description="Detailed reasoning about the current situation and next steps"
    )
    is_complete: bool = Field(
        default=False,
        description="True if the task is complete and no further actions needed",
    )
    action: Optional[Action] = Field(
        default=None,
        description="The action to take (None if is_complete=True)",
    )
    user_message: str = Field(
        default="Thinking...", description="Status message for the user"
    )
    final_result: Optional[Any] = Field(
        default=None,
        description="The final result when is_complete=True (should match output_schema if specified)",
    )


# Cache for dynamically created schemas to avoid recreating them
_thought_schema_cache: Dict[str, Type[BaseModel]] = {}
_react_thought_schema_cache: Dict[str, Type[BaseModel]] = {}


def create_thought_with_output_schema(
    output_schema: Optional[Type[BaseModel]] = None,
) -> Type[Thought]:
    """
    Create a Thought schema with typed final_result field.

    When output_schema is provided, the final_result field will be typed
    to that schema, allowing Opper to enforce the correct structure.

    Args:
        output_schema: Optional Pydantic model for the final result

    Returns:
        Thought class (original or dynamically created subclass)
    """
    if output_schema is None:
        return Thought

    # Check cache first
    cache_key = output_schema.__name__
    if cache_key in _thought_schema_cache:
        return _thought_schema_cache[cache_key]  # type: ignore[return-value]

    # Create dynamic model with typed final_result
    # Use create_model to generate a new class that inherits Thought's fields
    # but with final_result typed to the output_schema
    ThoughtWithSchema = create_model(
        f"Thought_{output_schema.__name__}",
        __base__=Thought,
        final_result=(
            Optional[output_schema],
            Field(
                default=None,
                description=f"The final result when is_complete=True. Must be a {output_schema.__name__} object.",
            ),
        ),
    )

    # Cache for reuse
    _thought_schema_cache[cache_key] = ThoughtWithSchema
    return ThoughtWithSchema  # type: ignore[return-value]


def create_react_thought_with_output_schema(
    output_schema: Optional[Type[BaseModel]] = None,
) -> Type[ReactThought]:
    """
    Create a ReactThought schema with typed final_result field.

    When output_schema is provided, the final_result field will be typed
    to that schema, allowing Opper to enforce the correct structure.

    Args:
        output_schema: Optional Pydantic model for the final result

    Returns:
        ReactThought class (original or dynamically created subclass)
    """
    if output_schema is None:
        return ReactThought

    # Check cache first
    cache_key = output_schema.__name__
    if cache_key in _react_thought_schema_cache:
        return _react_thought_schema_cache[cache_key]  # type: ignore[return-value]

    # Create dynamic model with typed final_result
    ReactThoughtWithSchema = create_model(
        f"ReactThought_{output_schema.__name__}",
        __base__=ReactThought,
        final_result=(
            Optional[output_schema],
            Field(
                default=None,
                description=f"The final result when is_complete=True. Must be a {output_schema.__name__} object.",
            ),
        ),
    )

    # Cache for reuse
    _react_thought_schema_cache[cache_key] = ReactThoughtWithSchema
    return ReactThoughtWithSchema  # type: ignore[return-value]
