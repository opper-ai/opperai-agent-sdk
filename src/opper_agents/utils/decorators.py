"""
Decorators for creating tools and hooks.

This module provides convenient decorators for converting functions
into tools and marking hooks for lifecycle events.
"""

from typing import Callable, Optional, Dict, Any, List, Union, TypeVar, overload
from ..base.tool import FunctionTool

F = TypeVar("F", bound=Callable[..., Any])


@overload
def tool(
    func: None = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
    output_schema: Optional[Any] = None,
    examples: Optional[List[Dict[str, Any]]] = None,
) -> Callable[[F], FunctionTool]: ...


@overload
def tool(
    func: F,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
    output_schema: Optional[Any] = None,
    examples: Optional[List[Dict[str, Any]]] = None,
) -> FunctionTool: ...


def tool(
    func: Optional[F] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
    output_schema: Optional[Any] = None,
    examples: Optional[List[Dict[str, Any]]] = None,
) -> Union[FunctionTool, Callable[[F], FunctionTool]]:
    """
    Decorator to convert a function into a Tool.

    Usage:
        @tool
        def add(a: int, b: int) -> int:
            '''Add two numbers.'''
            return a + b

        @tool(name="custom_name", description="Custom desc")
        def my_func(x: str) -> str:
            return x.upper()

        @tool(output_schema=MyOutputModel, examples=[{"input": {...}, "output": {...}}])
        def search(query: str) -> dict:
            '''Search for something.'''
            return {"results": [...]}

    Args:
        func: Function to wrap (when used without arguments)
        name: Custom tool name (default: function name)
        description: Custom description (default: function docstring)
        parameters: Custom parameter schema (default: auto-extracted)
        output_schema: Pydantic model describing the output structure
        examples: List of example input/output pairs

    Returns:
        FunctionTool instance wrapping the function
    """

    def decorator(f: Callable[..., Any]) -> FunctionTool:
        return FunctionTool(f, name, description, parameters, output_schema, examples)

    if func is None:
        # Called with arguments: @tool(name="something")
        return decorator
    else:
        # Called without arguments: @tool
        return decorator(func)


def hook(event_name: str) -> Callable[[F], F]:
    """
    Decorator to mark a function as a hook for a specific event.

    Usage:
        @hook("agent_start")
        async def on_start(context: AgentContext, agent: Agent):
            print("Agent starting!")

        agent = Agent(name="Test", hooks=[on_start])

    Args:
        event_name: Name of the event to hook into (e.g., "agent_start")

    Returns:
        Decorated function with hook metadata
    """

    def decorator(func: F) -> F:
        # Mark the function with hook metadata
        setattr(func, "_hook_event", event_name)
        return func

    return decorator
