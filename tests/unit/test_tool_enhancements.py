"""Tests for tool output_schema, examples, and usage on ToolResult."""

from pydantic import BaseModel
from opper_agents.base.tool import FunctionTool, ToolResult
from opper_agents.base.context import Usage
from opper_agents.utils.decorators import tool


class WeatherOutput(BaseModel):
    city: str
    temp_c: float


def test_tool_result_with_usage():
    """Test ToolResult can carry usage from nested agents."""
    usage = Usage(requests=3, total_tokens=500)
    result = ToolResult(
        tool_name="test",
        success=True,
        result="hello",
        execution_time=0.1,
        usage=usage,
    )
    assert result.usage is not None
    assert result.usage.total_tokens == 500


def test_tool_result_without_usage():
    """Test ToolResult defaults to no usage."""
    result = ToolResult(
        tool_name="test",
        success=True,
        result="hello",
        execution_time=0.1,
    )
    assert result.usage is None


def test_function_tool_with_output_schema():
    """Test FunctionTool accepts output_schema."""

    def get_weather(city: str) -> dict:
        return {"city": city, "temp_c": 22.0}

    ft = FunctionTool(
        func=get_weather,
        name="get_weather",
        output_schema=WeatherOutput,
    )
    assert ft.output_schema == WeatherOutput


def test_function_tool_with_examples():
    """Test FunctionTool accepts examples."""
    examples = [
        {"input": {"city": "London"}, "output": {"city": "London", "temp_c": 15.0}}
    ]

    def get_weather(city: str) -> dict:
        return {"city": city, "temp_c": 22.0}

    ft = FunctionTool(
        func=get_weather,
        name="get_weather",
        examples=examples,
    )
    assert ft.examples == examples


def test_function_tool_defaults():
    """Test FunctionTool defaults for output_schema and examples."""

    def my_func(x: str) -> str:
        return x

    ft = FunctionTool(func=my_func)
    assert ft.output_schema is None
    assert ft.examples is None


def test_tool_decorator_with_output_schema():
    """Test @tool decorator passes output_schema."""

    @tool(output_schema=WeatherOutput)
    def get_weather(city: str) -> dict:
        """Get weather for a city."""
        return {"city": city, "temp_c": 22.0}

    assert isinstance(get_weather, FunctionTool)
    assert get_weather.output_schema == WeatherOutput


def test_tool_decorator_with_examples():
    """Test @tool decorator passes examples."""
    examples = [
        {"input": {"city": "London"}, "output": {"city": "London", "temp_c": 15.0}}
    ]

    @tool(examples=examples)
    def get_weather(city: str) -> dict:
        """Get weather for a city."""
        return {"city": city, "temp_c": 22.0}

    assert isinstance(get_weather, FunctionTool)
    assert get_weather.examples == examples


def test_tool_decorator_plain_still_works():
    """Test that @tool without args still works."""

    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    assert isinstance(add, FunctionTool)
    assert add.name == "add"
    assert add.output_schema is None
    assert add.examples is None
