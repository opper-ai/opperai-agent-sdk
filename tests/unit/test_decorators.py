"""
Unit tests for decorators module.

Tests for @tool and @hook decorators.
"""

import pytest
from typing import List
from pydantic import BaseModel, Field
from opper_agents.utils.decorators import tool, hook
from opper_agents.base.tool import FunctionTool


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


def test_tool_with_pydantic_parameter():
    """Test @tool extracts Pydantic model schema for parameters."""

    class ReportInput(BaseModel):
        title: str = Field(description="The title of the report")
        summary: str = Field(description="The summary of the report")
        key_findings: List[str] = Field(description="The key findings")
        detailed_analysis: str = Field(description="The detailed analysis")

    @tool
    def save_report(report: ReportInput) -> str:
        """Save a report to file."""
        return f"Report {report.title} saved"

    # Check that the tool was created
    assert isinstance(save_report, FunctionTool)
    assert save_report.name == "save_report"

    # Check that parameters contain the full schema
    assert "report" in save_report.parameters
    report_param = save_report.parameters["report"]

    # Should be a dict (schema) not a string
    assert isinstance(report_param, dict)
    assert report_param["type"] == "object"
    assert "properties" in report_param
    assert "title" in report_param["properties"]
    assert "summary" in report_param["properties"]
    assert "key_findings" in report_param["properties"]
    assert "detailed_analysis" in report_param["properties"]


def test_tool_with_mixed_parameters():
    """Test @tool with both Pydantic and simple type parameters."""

    class UserInput(BaseModel):
        name: str
        age: int

    @tool
    def process_user(user: UserInput, debug: bool = False) -> str:
        """Process user data."""
        return f"Processed {user.name}"

    assert isinstance(process_user, FunctionTool)

    # Check Pydantic parameter
    assert "user" in process_user.parameters
    user_param = process_user.parameters["user"]
    assert isinstance(user_param, dict)
    assert user_param["type"] == "object"

    # Check simple parameter - now also a dict with 'type' and 'description'
    assert "debug" in process_user.parameters
    debug_param = process_user.parameters["debug"]
    assert isinstance(debug_param, dict)
    assert debug_param["type"] == "bool"
    assert "bool" in debug_param["description"]


@pytest.mark.asyncio
async def test_tool_with_pydantic_execution():
    """Test that tool with Pydantic parameter can be executed."""

    class MathInput(BaseModel):
        a: int
        b: int
        operation: str

    @tool
    def calculate(input: MathInput) -> int:
        """Perform calculation."""
        if input.operation == "add":
            return input.a + input.b
        elif input.operation == "multiply":
            return input.a * input.b
        return 0

    # Execute with Pydantic instance
    math_input = MathInput(a=5, b=3, operation="add")
    result = await calculate.execute(input=math_input)

    assert result.success
    assert result.result == 8


@pytest.mark.asyncio
async def test_tool_with_pydantic_dict_conversion():
    """Test that tool automatically converts dict to Pydantic model."""

    class ReportInput(BaseModel):
        title: str
        summary: str
        key_findings: List[str]

    @tool
    def save_report(report: ReportInput) -> str:
        """Save report to file."""
        return f"Saved: {report.title} with {len(report.key_findings)} findings"

    # Execute with dict (as LLM would provide)
    report_dict = {
        "title": "Test Report",
        "summary": "This is a test",
        "key_findings": ["Finding 1", "Finding 2", "Finding 3"],
    }

    result = await save_report.execute(report=report_dict)

    assert result.success
    assert "Saved: Test Report" in result.result
    assert "3 findings" in result.result
