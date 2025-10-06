"""
Tool system for agent actions.

This module provides the abstraction for tools (actions) that agents can execute,
including function wrapping and execution handling.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Callable, Optional, List, Union, Sequence, Protocol, Tuple
from abc import ABC, abstractmethod
import inspect
import asyncio
import time


class ToolResult(BaseModel):
    """Standardized result from tool execution."""

    tool_name: str = Field(description="Name of the tool executed")
    success: bool = Field(description="Whether execution succeeded")
    result: Any = Field(description="Tool execution result")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    execution_time: float = Field(description="Execution time in seconds")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    class Config:
        arbitrary_types_allowed = True


class Tool(BaseModel, ABC):
    """
    Abstract base class for all tools.
    All tools must implement execute() method.
    """

    name: str = Field(description="Tool name")
    description: str = Field(description="Tool description")
    parameters: Dict[str, Any] = Field(description="Tool parameters schema")

    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        pass


class FunctionTool(Tool):
    """
    Tool that wraps a Python function.
    Handles both sync and async functions automatically.
    """

    func: Callable = Field(description="The wrapped function", exclude=True)

    def __init__(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        # Extract metadata from function
        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or f"Execute {func.__name__}"

        # Auto-extract parameters if not provided
        if parameters is None:
            parameters = self._extract_parameters(func)

        super().__init__(
            name=tool_name,
            description=tool_description,
            parameters=parameters,
            func=func,
        )

    def _extract_parameters(self, func: Callable) -> Dict[str, Any]:
        """Extract parameter information from function signature."""
        sig = inspect.signature(func)
        parameters = {}

        for param_name, param in sig.parameters.items():
            # Skip special parameters
            if param_name.startswith("_"):
                continue

            # Get type annotation
            param_type = "any"
            if param.annotation != inspect.Parameter.empty:
                if hasattr(param.annotation, "__name__"):
                    param_type = param.annotation.__name__
                else:
                    param_type = str(param.annotation)

            # Get default value
            default_info = ""
            if param.default != inspect.Parameter.empty:
                default_info = f" (default: {param.default})"

            parameters[param_name] = f"{param_type}{default_info}"

        return parameters

    async def execute(self, **kwargs) -> ToolResult:
        """Execute the wrapped function."""
        start_time = time.time()

        try:
            # Filter out special parameters
            filtered_kwargs = {
                k: v for k, v in kwargs.items() if not k.startswith("_")
            }

            # Check if function is async
            if asyncio.iscoroutinefunction(self.func):
                result = await self.func(**filtered_kwargs)
            else:
                # Run sync function in executor to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, lambda: self.func(**filtered_kwargs)
                )

            execution_time = time.time() - start_time

            return ToolResult(
                tool_name=self.name,
                success=True,
                result=result,
                execution_time=execution_time,
            )

        except Exception as e:
            execution_time = time.time() - start_time

            return ToolResult(
                tool_name=self.name,
                success=False,
                result=None,
                error=str(e),
                execution_time=execution_time,
            )


class ToolProvider(Protocol):
    """Expands into tools at runtime (e.g. MCP bundles)."""

    async def setup(self, agent: "BaseAgent") -> List[Tool]:
        """Connect resources and return concrete tools."""
        ...

    async def teardown(self) -> None:
        """Cleanup after the agent run finishes."""
        ...


def normalize_tools(
    raw: Sequence[Union[Tool, ToolProvider]],
) -> Tuple[List[Tool], List[ToolProvider]]:
    """Utility to split concrete tools from providers."""

    tools: List[Tool] = []
    providers: List[ToolProvider] = []
    for item in raw:
        if isinstance(item, Tool):
            tools.append(item)
        elif hasattr(item, "setup") and hasattr(item, "teardown"):
            providers.append(item)
        else:
            raise TypeError(f"Unsupported tool item: {item!r}")
    return tools, providers
