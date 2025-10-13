"""
Unit tests for tool module.

Tests for Tool, FunctionTool, ToolResult, and tool execution.
"""

import pytest
from opper_agents.base.tool import FunctionTool, ToolResult


def sync_add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


async def async_multiply(x: int, y: int) -> int:
    """Multiply two numbers."""
    return x * y


def failing_tool():
    """This tool always fails."""
    raise ValueError("Intentional failure")


def tool_with_defaults(name: str, count: int = 5) -> str:
    """Tool with default parameters."""
    return f"{name} * {count}"


@pytest.mark.asyncio
async def test_function_tool_sync():
    """Test FunctionTool with synchronous function."""
    tool = FunctionTool(sync_add)
    result = await tool.execute(a=2, b=3)
    assert result.success
    assert result.result == 5
    assert result.error is None
    assert result.execution_time >= 0


@pytest.mark.asyncio
async def test_function_tool_async():
    """Test FunctionTool with asynchronous function."""
    tool = FunctionTool(async_multiply)
    result = await tool.execute(x=3, y=4)
    assert result.success
    assert result.result == 12
    assert result.error is None
    assert result.execution_time >= 0


@pytest.mark.asyncio
async def test_function_tool_error():
    """Test FunctionTool error handling."""
    tool = FunctionTool(failing_tool)
    result = await tool.execute()
    assert not result.success
    assert result.error is not None
    assert "Intentional failure" in result.error
    assert result.result is None
    assert result.execution_time >= 0


def test_parameter_extraction():
    """Test automatic parameter extraction from function signature."""
    tool = FunctionTool(sync_add)
    assert "a" in tool.parameters
    assert "b" in tool.parameters
    # Parameters are now dicts with 'type' and 'description' keys
    assert tool.parameters["a"]["type"] == "int"
    assert tool.parameters["b"]["type"] == "int"


def test_parameter_extraction_with_defaults():
    """Test parameter extraction with default values."""
    tool = FunctionTool(tool_with_defaults)
    assert "name" in tool.parameters
    assert "count" in tool.parameters
    # Parameters are now dicts with default info in the description
    assert "default: 5" in tool.parameters["count"]["description"]
    assert tool.parameters["count"]["type"] == "int"


def test_tool_name_extraction():
    """Test that tool name is extracted from function name."""
    tool = FunctionTool(sync_add)
    assert tool.name == "sync_add"


def test_tool_custom_name():
    """Test custom tool name."""
    tool = FunctionTool(sync_add, name="custom_add")
    assert tool.name == "custom_add"


def test_tool_description_from_docstring():
    """Test that description is extracted from docstring."""
    tool = FunctionTool(sync_add)
    assert "Add two numbers" in tool.description


def test_tool_custom_description():
    """Test custom tool description."""
    tool = FunctionTool(sync_add, description="Custom description")
    assert tool.description == "Custom description"


@pytest.mark.asyncio
async def test_tool_filters_special_params():
    """Test that special parameters (starting with _) are filtered out."""

    def tool_func(normal_param: str, _special_param: str = "ignored") -> str:
        """Function with special parameter."""
        return normal_param

    tool = FunctionTool(tool_func)
    result = await tool.execute(normal_param="test", _special_param="should_ignore")
    assert result.success
    assert result.result == "test"


@pytest.mark.asyncio
async def test_tool_result_metadata():
    """Test ToolResult metadata field."""
    tool = FunctionTool(sync_add)
    result = await tool.execute(a=1, b=2)

    # Metadata should be empty by default
    assert isinstance(result.metadata, dict)
    assert len(result.metadata) == 0


def test_tool_result_creation():
    """Test ToolResult model creation."""
    result = ToolResult(
        tool_name="test_tool",
        success=True,
        result="output",
        execution_time=0.5,
        metadata={"key": "value"},
    )

    assert result.tool_name == "test_tool"
    assert result.success is True
    assert result.result == "output"
    assert result.execution_time == 0.5
    assert result.metadata["key"] == "value"


@pytest.mark.asyncio
async def test_function_tool_preserves_none_result():
    """Test that None results are preserved."""

    def returns_none() -> None:
        """Function that returns None."""
        return None

    tool = FunctionTool(returns_none)
    result = await tool.execute()
    assert result.success
    assert result.result is None
    assert result.error is None


@pytest.mark.asyncio
async def test_function_tool_execution_time():
    """Test that execution time is tracked."""
    import asyncio

    async def slow_func() -> str:
        """Function that takes time."""
        await asyncio.sleep(0.1)
        return "done"

    tool = FunctionTool(slow_func)
    result = await tool.execute()
    assert result.success
    assert result.execution_time >= 0.1
