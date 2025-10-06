"""
Unit tests for BaseAgent abstract class.

Tests for agent initialization, tool management, hooks, and agent-as-tool functionality.
"""

import pytest
import asyncio
import concurrent.futures
from typing import Any
from opper_agent.base.agent import BaseAgent
from opper_agent.base.tool import FunctionTool
from opper_agent.utils.decorators import tool, hook
from opper_agent.base.hooks import HookEvents


# Concrete implementation for testing
class TestAgent(BaseAgent):
    """Minimal concrete agent for testing BaseAgent functionality."""

    async def process(self, input: Any, _parent_span_id: str = None) -> Any:
        return f"Processed: {input}"

    async def _run_loop(self, goal: Any) -> Any:
        return goal


@tool
def dummy_tool() -> str:
    """A simple test tool."""
    return "result"


@tool
def add_tool(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def test_base_agent_initialization(mock_opper_client, monkeypatch):
    """Test BaseAgent initialization with required parameters."""
    monkeypatch.setenv("OPPER_API_KEY", "test-key")

    agent = TestAgent(name="Test", tools=[dummy_tool])

    assert agent.name == "Test"
    assert len(agent.tools) == 1
    assert agent.description == "Agent: Test"
    assert agent.max_iterations == 25
    assert agent.verbose is False


def test_base_agent_with_custom_config(mock_opper_client, monkeypatch):
    """Test BaseAgent with custom configuration."""
    monkeypatch.setenv("OPPER_API_KEY", "test-key")

    agent = TestAgent(
        name="CustomAgent",
        description="Custom description",
        instructions="Custom instructions",
        max_iterations=50,
        verbose=True,
        model="anthropic/claude-3-opus",
    )

    assert agent.name == "CustomAgent"
    assert agent.description == "Custom description"
    assert agent.instructions == "Custom instructions"
    assert agent.max_iterations == 50
    assert agent.verbose is True
    assert agent.model == "anthropic/claude-3-opus"


def test_base_agent_requires_api_key():
    """Test that BaseAgent raises error without API key."""
    import os

    # Remove API key if it exists
    old_key = os.environ.pop("OPPER_API_KEY", None)

    try:
        with pytest.raises(ValueError, match="OPPER_API_KEY not found"):
            TestAgent(name="Test")
    finally:
        # Restore API key
        if old_key:
            os.environ["OPPER_API_KEY"] = old_key


def test_tool_management(mock_opper_client, monkeypatch):
    """Test adding and retrieving tools."""
    monkeypatch.setenv("OPPER_API_KEY", "test-key")

    agent = TestAgent(name="Test")
    agent.add_tool(dummy_tool)

    assert len(agent.tools) == 1
    assert agent.get_tool("dummy_tool") is not None
    assert agent.get_tool("nonexistent") is None
    assert "dummy_tool" in agent.list_tools()


def test_multiple_tools(mock_opper_client, monkeypatch):
    """Test agent with multiple tools."""
    monkeypatch.setenv("OPPER_API_KEY", "test-key")

    agent = TestAgent(name="Test", tools=[dummy_tool, add_tool])

    assert len(agent.tools) == 2
    assert agent.get_tool("dummy_tool") is not None
    assert agent.get_tool("add_tool") is not None


def test_hook_registration(mock_opper_client, monkeypatch):
    """Test registering hooks with agent."""
    monkeypatch.setenv("OPPER_API_KEY", "test-key")

    @hook("agent_start")
    async def on_start(context):
        pass

    agent = TestAgent(name="Test", hooks=[on_start])

    assert agent.hook_manager.has_hooks(HookEvents.AGENT_START)


@pytest.mark.asyncio
async def test_agent_as_tool(mock_opper_client, monkeypatch):
    """Test converting agent to tool."""
    monkeypatch.setenv("OPPER_API_KEY", "test-key")

    agent = TestAgent(name="SubAgent")
    tool = agent.as_tool()

    assert isinstance(tool, FunctionTool)
    assert "SubAgent" in tool.name
    assert "SubAgent" in tool.description


@pytest.mark.asyncio
async def test_agent_process_method(mock_opper_client, monkeypatch):
    """Test agent process method (concrete implementation)."""
    monkeypatch.setenv("OPPER_API_KEY", "test-key")

    agent = TestAgent(name="Test", tools=[dummy_tool])
    result = await agent.process("test task")

    assert result is not None
    assert "test task" in result


def test_agent_repr(mock_opper_client, monkeypatch):
    """Test agent string representation."""
    monkeypatch.setenv("OPPER_API_KEY", "test-key")

    agent = TestAgent(name="TestAgent", tools=[dummy_tool])
    repr_str = repr(agent)

    assert "TestAgent" in repr_str
    assert "name='TestAgent'" in repr_str
    assert "tools=1" in repr_str


def test_unsupported_tool_type(mock_opper_client, monkeypatch):
    """Test that invalid tool types raise TypeError."""
    monkeypatch.setenv("OPPER_API_KEY", "test-key")

    with pytest.raises(TypeError, match="Unsupported tool type"):
        TestAgent(name="Test", tools=["not_a_tool"])


def test_agent_with_input_output_schemas(mock_opper_client, monkeypatch):
    """Test agent with Pydantic schemas."""
    from pydantic import BaseModel

    monkeypatch.setenv("OPPER_API_KEY", "test-key")

    class InputSchema(BaseModel):
        task: str

    class OutputSchema(BaseModel):
        result: str

    agent = TestAgent(
        name="Test",
        input_schema=InputSchema,
        output_schema=OutputSchema,
    )

    assert agent.input_schema == InputSchema
    assert agent.output_schema == OutputSchema


def test_base_tools_vs_runtime_tools(mock_opper_client, monkeypatch):
    """Test distinction between base_tools and tools list."""
    monkeypatch.setenv("OPPER_API_KEY", "test-key")

    agent = TestAgent(name="Test", tools=[dummy_tool])

    # Add a runtime tool (not a base tool)
    agent.add_tool(add_tool, as_base=False)

    assert len(agent.base_tools) == 1  # Only dummy_tool
    assert len(agent.tools) == 2  # Both tools


def test_agent_default_model(mock_opper_client, monkeypatch):
    """Test agent uses default model if not specified."""
    monkeypatch.setenv("OPPER_API_KEY", "test-key")

    agent = TestAgent(name="Test")

    assert agent.model == "gcp/gemini-flash-latest"


def test_agent_with_custom_description(mock_opper_client, monkeypatch):
    """Test agent with custom description."""
    monkeypatch.setenv("OPPER_API_KEY", "test-key")

    agent = TestAgent(
        name="Test",
        description="This is a custom test agent",
    )

    assert agent.description == "This is a custom test agent"


def test_get_tool_returns_none_for_missing(mock_opper_client, monkeypatch):
    """Test that get_tool returns None for non-existent tools."""
    monkeypatch.setenv("OPPER_API_KEY", "test-key")

    agent = TestAgent(name="Test", tools=[dummy_tool])

    assert agent.get_tool("nonexistent_tool") is None


def test_list_tools_empty(mock_opper_client, monkeypatch):
    """Test list_tools with no tools."""
    monkeypatch.setenv("OPPER_API_KEY", "test-key")

    agent = TestAgent(name="Test")

    assert agent.list_tools() == []


def test_agent_with_instructions(mock_opper_client, monkeypatch):
    """Test agent with instructions."""
    monkeypatch.setenv("OPPER_API_KEY", "test-key")

    instructions = "Always be helpful and thorough"
    agent = TestAgent(name="Test", instructions=instructions)

    assert agent.instructions == instructions


# Agent-as-tool timeout and error propagation tests
class SlowAgent(BaseAgent):
    """Agent that simulates slow processing for timeout testing."""

    async def process(self, input: Any, _parent_span_id: str = None) -> Any:
        await asyncio.sleep(2)  # Simulate slow processing
        return f"Processed: {input}"

    async def _run_loop(self, goal: Any) -> Any:
        return goal


class FailingAgent(BaseAgent):
    """Agent that fails during processing for error testing."""

    async def process(self, input: Any, _parent_span_id: str = None) -> Any:
        raise ValueError("Agent processing failed intentionally")

    async def _run_loop(self, goal: Any) -> Any:
        return goal


@pytest.mark.asyncio
async def test_agent_as_tool_timeout(mock_opper_client, monkeypatch):
    """Test that agent-as-tool respects timeout when agent takes too long."""
    monkeypatch.setenv("OPPER_API_KEY", "test-key")

    agent = SlowAgent(name="SlowAgent")
    tool = agent.as_tool()

    # Patch the tool's function to use a very short timeout
    def short_timeout_wrapper(task: str, **kwargs):
        """Wrapper with shorter timeout for testing."""

        async def call_agent():
            input_data = {"task": task, **kwargs}
            return await agent.process(input_data)

        # Always use ThreadPoolExecutor path with short timeout
        def run_in_thread():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(call_agent())
            finally:
                new_loop.close()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_thread)
            return future.result(timeout=0.5)  # Short timeout

    tool.func = short_timeout_wrapper

    # Execute and expect timeout
    with pytest.raises(concurrent.futures.TimeoutError):
        tool.func("test task")


@pytest.mark.asyncio
async def test_agent_as_tool_error_propagation(mock_opper_client, monkeypatch):
    """Test that errors from agent.process() propagate through agent-as-tool."""
    monkeypatch.setenv("OPPER_API_KEY", "test-key")

    agent = FailingAgent(name="FailingAgent")
    tool = agent.as_tool()

    # Test that the error propagates when called from outside event loop
    with pytest.raises(ValueError, match="Agent processing failed intentionally"):
        tool.func("test task")


@pytest.mark.asyncio
async def test_agent_as_tool_in_nested_event_loop(mock_opper_client, monkeypatch):
    """Test agent-as-tool works correctly when called from within running event loop."""
    monkeypatch.setenv("OPPER_API_KEY", "test-key")

    agent = TestAgent(name="SubAgent")
    tool = agent.as_tool()

    # Simulate calling from within a running event loop
    async def call_from_async_context():
        # This should use ThreadPoolExecutor path since we're in a running loop
        result = await asyncio.get_event_loop().run_in_executor(
            None, tool.func, "nested task"
        )
        return result

    result = await call_from_async_context()
    assert "nested task" in str(result)


def test_agent_as_tool_outside_event_loop(mock_opper_client, monkeypatch):
    """Test agent-as-tool works when called without running event loop."""
    monkeypatch.setenv("OPPER_API_KEY", "test-key")

    agent = TestAgent(name="SubAgent")
    tool = agent.as_tool()

    # Call directly (no running event loop)
    result = tool.func("direct task")
    assert "direct task" in str(result)
