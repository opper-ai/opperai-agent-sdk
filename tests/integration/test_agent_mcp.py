"""
Integration tests for Agent with MCP tool providers.

These tests verify that agents can:
- Use MCP servers as tool providers
- Execute MCP tools during agent loops
- Handle multiple MCP servers
- Properly tear down MCP connections
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from types import SimpleNamespace
from contextlib import asynccontextmanager

from opper_agents import Agent, tool
from opper_agents.mcp.config import MCPServerConfig
from opper_agents.mcp.provider import mcp

import mcp as mcp_module


class AgentMCPFakeSession:
    """Mock MCP session for agent integration tests."""

    def __init__(self, tools=None):
        self.tools = tools or []
        self.initialize = AsyncMock(
            return_value=SimpleNamespace(
                serverInfo=mcp_module.Implementation(name="test-server", version="1.0"),
            )
        )
        self.list_tools = AsyncMock(return_value=SimpleNamespace(tools=self.tools))
        self.call_tool_mock = AsyncMock()

    async def call_tool(self, name, args):
        """Mock call_tool with proper response structure."""
        result = await self.call_tool_mock(name, args)
        # Return proper MCP response structure
        if isinstance(result, dict):
            return SimpleNamespace(content=[{"text": str(result)}])
        return SimpleNamespace(content=[{"text": str(result)}])

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.fixture
def mock_transports():
    """Mock all MCP transports."""

    @asynccontextmanager
    async def fake_transport(*args, **kwargs):
        yield object(), object()

    return {
        "stdio": fake_transport,
        "sse": fake_transport,
    }


def make_response(json_payload):
    """Helper to create a mock response with json_payload."""
    response = MagicMock()
    response.json_payload = json_payload
    return response


@pytest.mark.asyncio
async def test_agent_with_single_mcp_server(
    mock_acompletion, mock_transports, vcr_cassette, opper_api_key, mock_opper_client
):
    """Test agent using tools from a single MCP server."""
    # Setup MCP tools
    mcp_tools = [
        SimpleNamespace(
            name="search_docs",
            description="Search documentation",
            inputSchema={"type": "object", "properties": {"query": {"type": "string"}}},
            outputSchema=None,
        ),
    ]

    session = AgentMCPFakeSession(tools=mcp_tools)
    session.call_tool_mock = AsyncMock(
        return_value={"results": "Found documentation about testing"}
    )

    config = MCPServerConfig(
        name="docs",
        transport="stdio",
        command="python",
    )

    # Mock LLM response to use MCP tool
    mock_opper_client.call_async = AsyncMock(
        return_value=make_response(
            {
                "reasoning": "I need to search the documentation",
                "tool_calls": [
                    {
                        "name": "docs:search_docs",
                        "parameters": {"query": "testing"},
                        "reasoning": "Search for testing docs",
                    }
                ],
                "user_message": "Working on it...",
            }
        )
    )

    with (
        patch("opper_agents.mcp.client.stdio_client", mock_transports["stdio"]),
        patch("opper_agents.mcp.client.mcp.ClientSession", lambda *a, **kw: session),
    ):
        agent = Agent(
            name="DocAgent",
            tools=[mcp(config)],
            opper_api_key=opper_api_key,
            max_iterations=1,
        )

        # Process goal
        await agent.process("Search for testing documentation")

        # Verify MCP tool was called
        assert session.call_tool_mock.called
        call_args = session.call_tool_mock.call_args[0]
        assert call_args[0] == "search_docs"
        assert call_args[1]["query"] == "testing"


@pytest.mark.asyncio
async def test_agent_with_multiple_mcp_servers(
    mock_acompletion, mock_transports, vcr_cassette, opper_api_key, mock_opper_client
):
    """Test agent using tools from multiple MCP servers."""
    # Server 1 tools
    tools_server1 = [
        SimpleNamespace(
            name="read_file",
            description="Read a file",
            inputSchema={"type": "object", "properties": {"path": {"type": "string"}}},
            outputSchema=None,
        ),
    ]

    # Server 2 tools
    tools_server2 = [
        SimpleNamespace(
            name="search_web",
            description="Search the web",
            inputSchema={"type": "object", "properties": {"query": {"type": "string"}}},
            outputSchema=None,
        ),
    ]

    session1 = AgentMCPFakeSession(tools=tools_server1)
    session1.call_tool_mock = AsyncMock(return_value="File contents")

    session2 = AgentMCPFakeSession(tools=tools_server2)
    session2.call_tool_mock = AsyncMock(return_value="Search results")

    config1 = MCPServerConfig(name="filesystem", transport="stdio", command="python")
    config2 = MCPServerConfig(name="web", transport="stdio", command="python")

    def session_factory(*args, **kwargs):
        if not hasattr(session_factory, "call_count"):
            session_factory.call_count = 0
        session_factory.call_count += 1
        return session1 if session_factory.call_count == 1 else session2

    # Mock LLM to use both tools
    call_count = [0]

    def mock_completion_side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            # First call: use filesystem tool
            return make_response(
                {
                    "reasoning": "Read the file first",
                    "tool_calls": [
                        {
                            "name": "filesystem:read_file",
                            "parameters": {"path": "/test.txt"},
                            "reasoning": "Read file",
                        }
                    ],
                    "user_message": "Working on it...",
                }
            )
        else:
            # Second call: finish
            return make_response(
                {
                    "reasoning": "Task complete",
                    "tool_calls": [],
                    "user_message": "Done",
                }
            )

    mock_opper_client.call_async.side_effect = mock_completion_side_effect

    with (
        patch("opper_agents.mcp.client.stdio_client", mock_transports["stdio"]),
        patch("opper_agents.mcp.client.mcp.ClientSession", session_factory),
    ):
        agent = Agent(
            name="MultiAgent",
            tools=[mcp(config1, config2)],
            opper_api_key=opper_api_key,
            max_iterations=5,
        )

        await agent.process("Read test.txt")

        # Verify tool from first server was called
        assert session1.call_tool_mock.called


@pytest.mark.asyncio
async def test_agent_with_mcp_and_regular_tools(
    mock_acompletion, mock_transports, vcr_cassette, opper_api_key, mock_opper_client
):
    """Test agent using both MCP tools and regular function tools."""
    # MCP tools
    mcp_tools = [
        SimpleNamespace(
            name="mcp_tool",
            description="MCP tool",
            inputSchema={},
            outputSchema=None,
        ),
    ]

    session = AgentMCPFakeSession(tools=mcp_tools)
    session.call_tool_mock = AsyncMock(return_value="MCP result")

    # Regular tool
    @tool
    def regular_tool(x: int) -> int:
        """A regular Python tool."""
        return x * 2

    config = MCPServerConfig(name="test", transport="stdio", command="python")

    # Mock LLM to use regular tool
    mock_opper_client.call_async = AsyncMock(
        return_value=make_response(
            {
                "reasoning": "Use regular tool",
                "tool_calls": [
                    {
                        "name": "regular_tool",
                        "parameters": {"x": 5},
                        "reasoning": "Calculate",
                    }
                ],
                "user_message": "Working on it...",
            }
        )
    )

    with (
        patch("opper_agents.mcp.client.stdio_client", mock_transports["stdio"]),
        patch("opper_agents.mcp.client.mcp.ClientSession", lambda *a, **kw: session),
    ):
        agent = Agent(
            name="HybridAgent",
            tools=[mcp(config), regular_tool],
            opper_api_key=opper_api_key,
            max_iterations=1,
        )

        await agent.process("Calculate 5 * 2")

        # Regular tool should work
        # Note: We can't easily verify execution without running full agent loop
        # But we've verified the agent accepts both tool types


@pytest.mark.asyncio
async def test_agent_mcp_teardown(
    mock_acompletion, mock_transports, vcr_cassette, opper_api_key, mock_opper_client
):
    """Test that MCP connections are properly closed after agent execution."""
    mcp_tools = [
        SimpleNamespace(
            name="tool",
            description="Tool",
            inputSchema={},
            outputSchema=None,
        ),
    ]

    session = AgentMCPFakeSession(tools=mcp_tools)
    session.call_tool_mock = AsyncMock(return_value="result")

    # Track disconnect calls
    disconnect_called = []

    original_disconnect = None

    async def track_disconnect(self):
        disconnect_called.append(True)
        if original_disconnect:
            await original_disconnect()

    config = MCPServerConfig(name="test", transport="stdio", command="python")

    mock_opper_client.call_async = AsyncMock(
        return_value=make_response(
            {
                "reasoning": "Done",
                "tool_calls": [],
                "user_message": "Finished",
            }
        )
    )

    with (
        patch("opper_agents.mcp.client.stdio_client", mock_transports["stdio"]),
        patch("opper_agents.mcp.client.mcp.ClientSession", lambda *a, **kw: session),
        patch(
            "opper_agents.mcp.client.MCPClient.disconnect",
            track_disconnect,
        ),
    ):
        agent = Agent(
            name="TestAgent",
            tools=[mcp(config)],
            opper_api_key=opper_api_key,
        )

        await agent.process("Do something")

        # Verify disconnect was called
        assert len(disconnect_called) > 0


@pytest.mark.asyncio
async def test_agent_mcp_tool_error_handling(
    mock_acompletion, mock_transports, vcr_cassette, opper_api_key, mock_opper_client
):
    """Test that agent handles MCP tool errors gracefully."""
    mcp_tools = [
        SimpleNamespace(
            name="failing_tool",
            description="Tool that fails",
            inputSchema={},
            outputSchema=None,
        ),
    ]

    session = AgentMCPFakeSession(tools=mcp_tools)
    # Make tool fail
    session.call_tool_mock = AsyncMock(side_effect=RuntimeError("Tool failed"))

    config = MCPServerConfig(name="test", transport="stdio", command="python")

    call_count = [0]

    def mock_completion_side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            # Try to use the failing tool
            return make_response(
                {
                    "reasoning": "Try the tool",
                    "tool_calls": [
                        {
                            "name": "test:failing_tool",
                            "parameters": {},
                            "reasoning": "Test",
                        }
                    ],
                    "user_message": "Working on it...",
                }
            )
        else:
            # Agent should recover and finish
            return make_response(
                {
                    "reasoning": "Tool failed, giving up",
                    "tool_calls": [],
                    "user_message": "Could not complete task",
                }
            )

    mock_opper_client.call_async.side_effect = mock_completion_side_effect

    with (
        patch("opper_agents.mcp.client.stdio_client", mock_transports["stdio"]),
        patch("opper_agents.mcp.client.mcp.ClientSession", lambda *a, **kw: session),
    ):
        agent = Agent(
            name="ErrorAgent",
            tools=[mcp(config)],
            opper_api_key=opper_api_key,
            max_iterations=2,
        )

        # Should not raise, should handle error gracefully
        await agent.process("Do something")

        # Verify we tried to call the tool
        assert session.call_tool_mock.called


@pytest.mark.asyncio
async def test_agent_mcp_connection_failure(
    mock_acompletion, mock_transports, vcr_cassette, opper_api_key, mock_opper_client
):
    """Test agent handles MCP connection failure gracefully."""

    @asynccontextmanager
    async def failing_transport(*args, **kwargs):
        raise RuntimeError("Connection failed")
        yield  # This line won't be reached

    config = MCPServerConfig(name="test", transport="stdio", command="python")

    mock_opper_client.call_async = AsyncMock(
        return_value=make_response(
            {
                "reasoning": "Done",
                "tool_calls": [],
                "user_message": "No tools available",
            }
        )
    )

    with patch("opper_agents.mcp.client.stdio_client", failing_transport):
        # Agent should start even if MCP connection fails
        agent = Agent(
            name="TestAgent",
            tools=[mcp(config)],
            opper_api_key=opper_api_key,
        )

        # Should still be able to process (without MCP tools)
        result = await agent.process("Do something")

        # Agent should have handled the connection failure
        # and continued without MCP tools
        assert result is not None


@pytest.mark.asyncio
async def test_agent_final_result_generation_with_mcp_cleanup(
    mock_acompletion, mock_transports, vcr_cassette, opper_api_key, mock_opper_client
):
    """
    Test that agent can generate final results after MCP cleanup.

    This test verifies that the anyio cancel scope shielding in
    _generate_final_result() prevents CancelledError when making
    HTTP calls to Opper after MCP stdio clients have disconnected.

    Regression test for: asyncio.exceptions.CancelledError during
    final result generation after MCP stdio disconnect.
    """
    mcp_tools = [
        SimpleNamespace(
            name="test_tool",
            description="Test tool",
            inputSchema={},
            outputSchema=None,
        ),
    ]

    session = AgentMCPFakeSession(tools=mcp_tools)
    session.call_tool_mock = AsyncMock(return_value="Tool executed")

    config = MCPServerConfig(name="test", transport="stdio", command="python")

    call_count = [0]

    def mock_completion_side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            # First call: use the MCP tool
            return make_response(
                {
                    "reasoning": "Using MCP tool",
                    "tool_calls": [
                        {
                            "name": "test:test_tool",
                            "parameters": {},
                            "reasoning": "Execute tool",
                        }
                    ],
                    "user_message": "Working...",
                }
            )
        elif call_count[0] == 2:
            # Second call: finish execution
            return make_response(
                {
                    "reasoning": "Done",
                    "tool_calls": [],
                    "user_message": "Complete",
                }
            )
        else:
            # Third call: generate final result (this is the critical one)
            # This happens AFTER MCP disconnect and should not raise CancelledError
            return make_response("Final result generated successfully")

    mock_opper_client.call_async.side_effect = mock_completion_side_effect

    with (
        patch("opper_agents.mcp.client.stdio_client", mock_transports["stdio"]),
        patch("opper_agents.mcp.client.mcp.ClientSession", lambda *a, **kw: session),
    ):
        agent = Agent(
            name="TestAgent",
            tools=[mcp(config)],
            opper_api_key=opper_api_key,
            max_iterations=3,
        )

        # This should complete without CancelledError
        result = await agent.process("Test task")

        # Verify the final result was generated
        assert result is not None
        # Verify all three LLM calls were made (think, think, generate_final_result)
        assert call_count[0] == 3
