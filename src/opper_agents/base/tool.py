"""
Tool system for agent actions.

This module provides the abstraction for tools (actions) that agents can execute,
including function wrapping and execution handling.
"""

from pydantic import BaseModel, Field
from typing import (
    Dict,
    Any,
    Callable,
    Optional,
    List,
    Union,
    Sequence,
    Protocol,
    Tuple,
    TYPE_CHECKING,
)
from abc import ABC, abstractmethod
import inspect
import asyncio
import time

if TYPE_CHECKING:
    from .agent import BaseAgent


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
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with given parameters."""
        pass


class FunctionTool(Tool):
    """
    Tool that wraps a Python function.
    Handles both sync and async functions automatically.
    """

    func: Optional[Callable] = Field(
        default=None, description="The wrapped function", exclude=True
    )
    wrapped_agent: Optional[Any] = Field(
        default=None,
        description="Reference to wrapped agent if this tool wraps an agent",
        exclude=True,
    )

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

        # Initialize parent without func (it's excluded anyway)
        super().__init__(
            name=tool_name,
            description=tool_description,
            parameters=parameters,
        )
        # Set func directly after parent initialization
        object.__setattr__(self, "func", func)

    def _extract_parameters(self, func: Callable) -> Dict[str, Any]:
        """
        Extract parameter information from function signature.

        If a parameter is a Pydantic model, extracts its full schema.
        Otherwise, extracts type hints as string descriptions.
        """
        sig = inspect.signature(func)
        parameters = {}

        for param_name, param in sig.parameters.items():
            # Skip special parameters
            if param_name.startswith("_"):
                continue

            # Check if annotation is a Pydantic model
            if param.annotation != inspect.Parameter.empty:
                # Try to detect Pydantic BaseModel
                try:
                    # Check if it's a Pydantic model by looking for model_json_schema
                    if hasattr(param.annotation, "model_json_schema"):
                        # Extract full Pydantic schema
                        schema = param.annotation.model_json_schema()
                        # Use the properties from the schema for better LLM understanding
                        if "properties" in schema:
                            parameters[param_name] = {
                                "type": "object",
                                "properties": schema["properties"],
                                "required": schema.get("required", []),
                                "description": schema.get(
                                    "description", f"Structured input for {param_name}"
                                ),
                            }
                            continue
                except Exception:
                    # If schema extraction fails, fall through to simple type extraction
                    pass

            # Get simple type annotation for non-Pydantic types
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

            parameters[param_name] = {
                "type": param_type,
                "description": f"{param_type}{default_info}",
            }

        return parameters

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the wrapped function."""
        start_time = time.time()

        try:
            # func is always set in __init__, but satisfy type checker
            assert self.func is not None, "Function not initialized"

            # Extract special parameters (prefixed with _)
            parent_span_id = kwargs.get("_parent_span_id")

            # Filter out special parameters for function call
            filtered_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}

            # Convert dict parameters to Pydantic models if needed
            sig = inspect.signature(self.func)
            for param_name, param in sig.parameters.items():
                if (
                    param_name in filtered_kwargs
                    and param.annotation != inspect.Parameter.empty
                ):
                    # Check if parameter expects a Pydantic model
                    if hasattr(param.annotation, "model_validate"):
                        value = filtered_kwargs[param_name]
                        # If value is a dict, convert to Pydantic model
                        if isinstance(value, dict):
                            try:
                                filtered_kwargs[param_name] = (
                                    param.annotation.model_validate(value)
                                )
                            except Exception:
                                # If conversion fails, let the function handle it
                                pass

            # Pass parent_span_id if function accepts it
            if "_parent_span_id" in sig.parameters:
                filtered_kwargs["_parent_span_id"] = parent_span_id

            # Check if function is async
            func = self.func  # Capture for lambda/type narrowing
            if asyncio.iscoroutinefunction(func):
                result = await func(**filtered_kwargs)
            else:
                # Run sync function in executor to avoid blocking
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None, lambda: func(**filtered_kwargs)
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
