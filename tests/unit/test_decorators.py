"""
Unit tests for decorators module.

Tests for @tool and @hook decorators.
"""

import pytest
from opper_agent.utils.decorators import tool, hook
from opper_agent.base.tool import FunctionTool


def test_tool_decorator_without_args():
    """Test @tool decorator without arguments."""

    @tool
    def add(a: int, b: int) -> int:
        return a + b

    assert isinstance(add, FunctionTool)
    assert add.name == "add"


def test_tool_decorator_with_args():
    """Test @tool decorator with custom arguments."""

    @tool(name="custom_add", description="Custom description")
    def add(a: int, b: int) -> int:
        return a + b

    assert isinstance(add, FunctionTool)
    assert add.name == "custom_add"
    assert add.description == "Custom description"


def test_tool_decorator_preserves_functionality():
    """Test that decorated tool can still be executed."""

    @tool
    async def multiply(x: int, y: int) -> int:
        """Multiply two numbers."""
        return x * y

    # Tool should be executable
    assert isinstance(multiply, FunctionTool)


def test_hook_decorator():
    """Test @hook decorator."""

    @hook("agent_start")
    async def on_start(context):
        pass

    assert hasattr(on_start, "_hook_event")
    assert on_start._hook_event == "agent_start"


def test_hook_decorator_preserves_function():
    """Test that @hook preserves the original function."""

    @hook("agent_end")
    async def on_end(context):
        """Hook docstring."""
        return "done"

    # Function should still be callable
    assert callable(on_end)
    assert on_end.__doc__ == "Hook docstring."


def test_tool_with_custom_parameters():
    """Test @tool with custom parameter schema."""

    @tool(parameters={"x": "number", "y": "number"})
    def add(x, y):
        return x + y

    assert isinstance(add, FunctionTool)
    assert add.parameters["x"] == "number"
    assert add.parameters["y"] == "number"


def test_multiple_tools():
    """Test creating multiple tools with decorator."""

    @tool
    def tool1() -> str:
        return "tool1"

    @tool
    def tool2() -> str:
        return "tool2"

    assert isinstance(tool1, FunctionTool)
    assert isinstance(tool2, FunctionTool)
    assert tool1.name == "tool1"
    assert tool2.name == "tool2"


def test_multiple_hooks():
    """Test creating multiple hooks with decorator."""

    @hook("event1")
    def hook1(context):
        pass

    @hook("event2")
    def hook2(context):
        pass

    assert hook1._hook_event == "event1"
    assert hook2._hook_event == "event2"


@pytest.mark.asyncio
async def test_decorated_tool_execution():
    """Test that decorated tool can be executed properly."""

    @tool
    def greet(name: str) -> str:
        """Greet someone."""
        return f"Hello, {name}!"

    result = await greet.execute(name="Alice")
    assert result.success
    assert result.result == "Hello, Alice!"


def test_tool_extracts_docstring():
    """Test that @tool extracts function docstring as description."""

    @tool
    def documented_func(x: int) -> int:
        """This is the documentation."""
        return x * 2

    assert "This is the documentation" in documented_func.description


def test_tool_without_docstring():
    """Test @tool on function without docstring."""

    @tool
    def no_doc(x: int) -> int:
        return x

    # Should have a default description
    assert no_doc.description is not None
    assert len(no_doc.description) > 0
