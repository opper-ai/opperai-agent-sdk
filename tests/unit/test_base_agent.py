"""
Unit tests for BaseAgent abstract class.

Tests for agent initialization, tool management, hooks, and agent-as-tool functionality.
"""

import pytest
import asyncio
import concurrent.futures
from typing import Any
from opper_agents.base.agent import BaseAgent
from opper_agents.base.tool import FunctionTool
from opper_agents.utils.decorators import tool, hook
from opper_agents.base.hooks import HookEvents


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


def test_agent_as_tool_with_input_schema(mock_opper_client, monkeypatch):
    """Test that agent.as_tool() exposes input_schema fields as parameters."""
    from pydantic import BaseModel, Field
    from typing import List

    monkeypatch.setenv("OPPER_API_KEY", "test-key")

    class ResearchRequest(BaseModel):
        topic: str = Field(description="The topic to research")
        depth: int = Field(description="Depth of research (1-5)")
        sources: List[str] = Field(description="Preferred sources")

    agent = TestAgent(
        name="ResearchAgent",
        input_schema=ResearchRequest,
    )

    tool = agent.as_tool()

    # Check that parameters contain the schema fields
    assert isinstance(tool.parameters, dict)
    assert "topic" in tool.parameters
    assert "depth" in tool.parameters
    assert "sources" in tool.parameters

    # Should not have the generic "task" parameter anymore
    # (or if it does, should also have the schema fields)
    assert len(tool.parameters) >= 3


def test_agent_as_tool_without_input_schema(mock_opper_client, monkeypatch):
    """Test that agent.as_tool() without input_schema has default task parameter."""
    monkeypatch.setenv("OPPER_API_KEY", "test-key")

    agent = TestAgent(name="SimpleAgent")
    tool = agent.as_tool()

    # Should have simple task parameter
    assert "task" in tool.parameters
    assert isinstance(tool.parameters["task"], str)
    assert (
        "delegate" in tool.parameters["task"].lower()
        or "str" in tool.parameters["task"]
    )


@pytest.mark.asyncio
async def test_agent_as_tool_with_schema_execution(mock_opper_client, monkeypatch):
    """Test that agent-as-tool with input_schema can be executed with structured data."""
    from pydantic import BaseModel

    monkeypatch.setenv("OPPER_API_KEY", "test-key")

    class UserData(BaseModel):
        name: str
        age: int

    class DataAgent(BaseAgent):
        """Agent that processes structured data."""

        async def process(self, input: Any, _parent_span_id: str = None) -> Any:
            if isinstance(input, dict):
                return f"Processed {input.get('name', 'unknown')}, age {input.get('age', 0)}"
            return f"Processed: {input}"

        async def _run_loop(self, goal: Any) -> Any:
            return goal

    agent = DataAgent(
        name="DataAgent",
        input_schema=UserData,
    )

    tool = agent.as_tool()

    # Execute with structured parameters
    result = tool.func(name="Alice", age=30)
    assert "Alice" in str(result)
    assert "30" in str(result)


@pytest.mark.asyncio
async def test_agent_as_tool_receives_kwargs_as_dict(mock_opper_client, monkeypatch):
    """Test that agent-as-tool receives kwargs properly when called via execute()."""
    from pydantic import BaseModel

    monkeypatch.setenv("OPPER_API_KEY", "test-key")

    class PersonInfo(BaseModel):
        name: str
        age: int
        city: str

    class PersonAgent(BaseAgent):
        """Agent that processes person info."""

        async def process(self, input: Any, _parent_span_id: str = None) -> Any:
            # Should receive input as dict with all kwargs
            if isinstance(input, dict):
                name = input.get("name", "unknown")
                age = input.get("age", 0)
                city = input.get("city", "unknown")
                return f"{name}, {age} years old, from {city}"
            return "Invalid input"

        async def _run_loop(self, goal: Any) -> Any:
            return goal

    agent = PersonAgent(
        name="PersonAgent",
        input_schema=PersonInfo,
    )

    tool = agent.as_tool()

    # Execute via FunctionTool.execute() (as parent agent would)
    result = await tool.execute(name="Bob", age=25, city="NYC")

    assert result.success
    assert "Bob" in result.result
    assert "25" in result.result
    assert "NYC" in result.result


def test_visualize_flow_basic(mock_opper_client, monkeypatch):
    """Test basic visualization without tools."""
    monkeypatch.setenv("OPPER_API_KEY", "test-key")

    agent = TestAgent(name="BasicAgent")
    diagram = agent.visualize_flow()

    assert "```mermaid" in diagram
    assert "graph TB" in diagram
    assert "BasicAgent" in diagram
    assert "```" in diagram


def test_visualize_flow_with_tools(mock_opper_client, monkeypatch):
    """Test visualization with regular tools."""
    monkeypatch.setenv("OPPER_API_KEY", "test-key")

    agent = TestAgent(name="ToolAgent", tools=[dummy_tool, add_tool])
    diagram = agent.visualize_flow()

    assert "BasicAgent" in diagram or "ToolAgent" in diagram
    assert "dummy_tool" in diagram
    assert "add_tool" in diagram
    assert "⚙️" in diagram or "tool" in diagram.lower()


def test_visualize_flow_with_agent_tool(mock_opper_client, monkeypatch):
    """Test visualization with agent-as-tool."""
    monkeypatch.setenv("OPPER_API_KEY", "test-key")

    sub_agent = TestAgent(name="SubAgent")
    main_agent = TestAgent(name="MainAgent", tools=[sub_agent.as_tool()])

    diagram = main_agent.visualize_flow()

    assert "MainAgent" in diagram
    assert "agent" in diagram.lower()
    # Should detect agent tool pattern
    assert "SubAgent" in diagram or "_agent" in diagram


def test_visualize_flow_with_schemas(mock_opper_client, monkeypatch):
    """Test visualization with input/output schemas."""
    from pydantic import BaseModel

    monkeypatch.setenv("OPPER_API_KEY", "test-key")

    class InputSchema(BaseModel):
        query: str

    class OutputSchema(BaseModel):
        result: str

    agent = TestAgent(
        name="SchemaAgent",
        input_schema=InputSchema,
        output_schema=OutputSchema,
    )
    diagram = agent.visualize_flow()

    assert "InputSchema" in diagram
    assert "OutputSchema" in diagram
    assert "📥" in diagram or "Input" in diagram
    assert "📤" in diagram or "Output" in diagram


def test_visualize_flow_with_hooks(mock_opper_client, monkeypatch):
    """Test visualization with hooks."""
    monkeypatch.setenv("OPPER_API_KEY", "test-key")

    @hook("agent_start")
    async def on_start(context):
        pass

    agent = TestAgent(name="HookAgent", hooks=[on_start])
    diagram = agent.visualize_flow()

    assert "HookAgent" in diagram
    assert "Hook" in diagram or "🪝" in diagram


def test_visualize_flow_save_to_file(mock_opper_client, monkeypatch, tmp_path):
    """Test saving visualization to file."""
    monkeypatch.setenv("OPPER_API_KEY", "test-key")

    agent = TestAgent(name="FileAgent")
    output_path = tmp_path / "agent_flow.md"

    returned_path = agent.visualize_flow(output_path=str(output_path))

    assert returned_path == str(output_path)
    assert output_path.exists()

    content = output_path.read_text()
    assert "# Agent Flow: FileAgent" in content
    assert "```mermaid" in content
    assert "FileAgent" in content


def test_visualize_flow_auto_add_md_extension(mock_opper_client, monkeypatch, tmp_path):
    """Test that .md extension is automatically added if missing."""
    monkeypatch.setenv("OPPER_API_KEY", "test-key")

    agent = TestAgent(name="ExtAgent")
    output_path = tmp_path / "agent_flow"  # No .md extension

    returned_path = agent.visualize_flow(output_path=str(output_path))

    assert returned_path.endswith(".md")
    assert (tmp_path / "agent_flow.md").exists()


def test_visualize_flow_sanitizes_node_ids(mock_opper_client, monkeypatch):
    """Test that node IDs are properly sanitized."""
    monkeypatch.setenv("OPPER_API_KEY", "test-key")

    agent = TestAgent(name="My-Agent.Name With Spaces")
    diagram = agent.visualize_flow()

    # Should work without syntax errors
    assert "```mermaid" in diagram
    # Spaces and special chars should be handled
    assert "My" in diagram


def test_agent_tool_timeout_parameter(mock_opper_client, monkeypatch):
    """Test agent_tool_timeout parameter configuration."""
    monkeypatch.setenv("OPPER_API_KEY", "test-key")

    # Default timeout
    agent = TestAgent(name="DefaultTimeout")
    assert agent.agent_tool_timeout == 120.0

    # Custom timeout
    agent = TestAgent(name="CustomTimeout", agent_tool_timeout=300.0)
    assert agent.agent_tool_timeout == 300.0

    # No timeout
    agent = TestAgent(name="NoTimeout", agent_tool_timeout=None)
    assert agent.agent_tool_timeout is None


@pytest.mark.asyncio
async def test_agent_as_tool_uses_configured_timeout(mock_opper_client, monkeypatch):
    """Test that agent-as-tool respects the configured agent_tool_timeout."""
    monkeypatch.setenv("OPPER_API_KEY", "test-key")

    # Agent with short timeout
    agent = SlowAgent(name="SlowAgent", agent_tool_timeout=0.5)
    tool = agent.as_tool()

    # Execute from async context and expect timeout (SlowAgent sleeps for 2 seconds)
    # When called from running event loop, it uses ThreadPoolExecutor with timeout
    with pytest.raises(concurrent.futures.TimeoutError):
        # Direct call from async context triggers ThreadPoolExecutor path
        tool.func("test task")
