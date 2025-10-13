"""
Integration tests for MCP client and provider.

These tests verify the full lifecycle of MCP connections including:
- Connection establishment
- Tool discovery
- Tool execution
- Proper cleanup
"""

import pytest
from unittest.mock import AsyncMock, patch
from types import SimpleNamespace
from contextlib import asynccontextmanager

from opper_agents.mcp.config import MCPServerConfig
from opper_agents.mcp.client import MCPClient
from opper_agents.mcp.provider import mcp
from opper_agents.base.tool import ToolResult

import mcp as mcp_module


class IntegrationFakeSession:
    """Mock session for integration tests."""

    def __init__(self, tools=None):
        self.tools = tools or []
        self.initialize = AsyncMock(
            return_value=SimpleNamespace(
                serverInfo=mcp_module.Implementation(name="test-server", version="1.0"),
            )
        )
        self.list_tools = AsyncMock(return_value=SimpleNamespace(tools=self.tools))
        self.call_tool = AsyncMock()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.fixture
def mock_stdio_transport():
    """Mock stdio transport."""

    @asynccontextmanager
    async def fake_stdio(*args, **kwargs):
        yield object(), object()

    return fake_stdio


@pytest.fixture
def mock_sse_transport():
    """Mock SSE transport."""

    @asynccontextmanager
    async def fake_sse(*args, **kwargs):
        yield object(), object()

    return fake_sse


@pytest.mark.asyncio
async def test_full_mcp_lifecycle_stdio(mock_stdio_transport):
    """Test complete MCP lifecycle with stdio transport."""
    # Create fake session with tools
    test_tools = [
        SimpleNamespace(
            name="read_file",
            description="Read a file",
            inputSchema={"type": "object", "properties": {"path": {"type": "string"}}},
            outputSchema=None,
        ),
        SimpleNamespace(
            name="write_file",
            description="Write to a file",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
            },
            outputSchema=None,
        ),
    ]

    session = IntegrationFakeSession(tools=test_tools)
    session.call_tool = AsyncMock(
        return_value=SimpleNamespace(content=[{"text": "File content here"}])
    )

    config = MCPServerConfig(
        name="filesystem",
        transport="stdio",
        command="uvx",
        args=["mcp-server-filesystem", "/tmp"],
    )

    with (
        patch("opper_agents.mcp.client.stdio_client", mock_stdio_transport),
        patch("opper_agents.mcp.client.mcp.ClientSession", lambda *a, **kw: session),
    ):
        # Step 1: Connect
        client = MCPClient(config)
        await client.connect()
        assert client.connected

        # Step 2: List tools
        tools = await client.list_tools()
        assert len(tools) == 2
        assert tools[0].name == "read_file"
        assert tools[1].name == "write_file"

        # Step 3: Call tool
        result = await client.call_tool("read_file", {"path": "/tmp/test.txt"})
        assert result is not None
        session.call_tool.assert_awaited_with("read_file", {"path": "/tmp/test.txt"})

        # Step 4: Disconnect
        await client.disconnect()
        assert not client.connected


@pytest.mark.asyncio
async def test_mcp_provider_full_lifecycle(mock_stdio_transport):
    """Test MCPToolProvider full lifecycle."""
    test_tools = [
        SimpleNamespace(
            name="search",
            description="Search for something",
            inputSchema={"type": "object", "properties": {"query": {"type": "string"}}},
            outputSchema=None,
        ),
    ]

    session = IntegrationFakeSession(tools=test_tools)
    session.call_tool = AsyncMock(
        return_value=SimpleNamespace(content=[{"text": "Search results"}])
    )

    config = MCPServerConfig(
        name="search-server",
        transport="stdio",
        command="python",
    )

    with (
        patch("opper_agents.mcp.client.stdio_client", mock_stdio_transport),
        patch("opper_agents.mcp.client.mcp.ClientSession", lambda *a, **kw: session),
    ):
        # Step 1: Setup provider
        provider = mcp(config)
        tools = await provider.setup(None)

        assert len(tools) == 1
        assert tools[0].name == "search-server:search"

        # Step 2: Execute tool
        result = await tools[0].execute(query="test query")
        assert result.success
        assert result.result is not None

        # Step 3: Teardown
        await provider.teardown()


@pytest.mark.asyncio
async def test_multiple_mcp_servers(mock_stdio_transport):
    """Test provider with multiple MCP servers."""
    tools_server1 = [
        SimpleNamespace(
            name="tool1",
            description="Tool from server 1",
            inputSchema={},
            outputSchema=None,
        ),
    ]

    tools_server2 = [
        SimpleNamespace(
            name="tool2",
            description="Tool from server 2",
            inputSchema={},
            outputSchema=None,
        ),
    ]

    session1 = IntegrationFakeSession(tools=tools_server1)
    session1.call_tool = AsyncMock(
        return_value=SimpleNamespace(content=[{"text": "Result 1"}])
    )

    session2 = IntegrationFakeSession(tools=tools_server2)
    session2.call_tool = AsyncMock(
        return_value=SimpleNamespace(content=[{"text": "Result 2"}])
    )

    config1 = MCPServerConfig(name="server1", transport="stdio", command="python")
    config2 = MCPServerConfig(name="server2", transport="stdio", command="python")

    def session_factory(*args, **kwargs):
        # Return different sessions for different calls
        if not hasattr(session_factory, "call_count"):
            session_factory.call_count = 0
        session_factory.call_count += 1
        return session1 if session_factory.call_count == 1 else session2

    with (
        patch("opper_agents.mcp.client.stdio_client", mock_stdio_transport),
        patch("opper_agents.mcp.client.mcp.ClientSession", session_factory),
    ):
        provider = mcp(config1, config2)
        tools = await provider.setup(None)

        # Should have tools from both servers
        assert len(tools) == 2
        assert tools[0].name == "server1:tool1"
        assert tools[1].name == "server2:tool2"

        # Execute each tool
        result1 = await tools[0].execute()
        result2 = await tools[1].execute()

        assert result1.success
        assert result2.success

        await provider.teardown()


@pytest.mark.asyncio
async def test_mcp_tool_execution_with_parameters(mock_stdio_transport):
    """Test MCP tool execution with various parameter types."""
    test_tools = [
        SimpleNamespace(
            name="complex_tool",
            description="Tool with complex parameters",
            inputSchema={
                "type": "object",
                "properties": {
                    "string_param": {"type": "string"},
                    "number_param": {"type": "number"},
                    "bool_param": {"type": "boolean"},
                    "array_param": {"type": "array"},
                    "object_param": {"type": "object"},
                },
            },
            outputSchema=None,
        ),
    ]

    session = IntegrationFakeSession(tools=test_tools)
    session.call_tool = AsyncMock(
        return_value=SimpleNamespace(content=[{"text": "Success"}])
    )

    config = MCPServerConfig(name="test", transport="stdio", command="python")

    with (
        patch("opper_agents.mcp.client.stdio_client", mock_stdio_transport),
        patch("opper_agents.mcp.client.mcp.ClientSession", lambda *a, **kw: session),
    ):
        provider = mcp(config)
        tools = await provider.setup(None)

        # Execute with various parameter types
        result = await tools[0].execute(
            string_param="hello",
            number_param=42,
            bool_param=True,
            array_param=[1, 2, 3],
            object_param={"key": "value"},
        )

        assert result.success

        # Verify parameters were passed correctly
        session.call_tool.assert_awaited_once()
        call_args = session.call_tool.call_args[0]
        assert call_args[0] == "complex_tool"
        assert call_args[1]["string_param"] == "hello"
        assert call_args[1]["number_param"] == 42
        assert call_args[1]["bool_param"] is True
        assert call_args[1]["array_param"] == [1, 2, 3]
        assert call_args[1]["object_param"] == {"key": "value"}

        await provider.teardown()


@pytest.mark.asyncio
async def test_mcp_reconnection(mock_stdio_transport):
    """Test that MCP client can disconnect and reconnect."""
    session = IntegrationFakeSession()

    config = MCPServerConfig(name="test", transport="stdio", command="python")

    with (
        patch("opper_agents.mcp.client.stdio_client", mock_stdio_transport),
        patch("opper_agents.mcp.client.mcp.ClientSession", lambda *a, **kw: session),
    ):
        client = MCPClient(config)

        # First connection
        await client.connect()
        assert client.connected

        await client.disconnect()
        assert not client.connected

        # Second connection
        await client.connect()
        assert client.connected

        await client.disconnect()
        assert not client.connected


@pytest.mark.asyncio
async def test_mcp_tool_result_structure(mock_stdio_transport):
    """Test that MCP tool results have correct structure."""
    test_tools = [
        SimpleNamespace(
            name="test_tool",
            description="Test",
            inputSchema={},
            outputSchema=None,
        ),
    ]

    session = IntegrationFakeSession(tools=test_tools)
    session.call_tool = AsyncMock(
        return_value=SimpleNamespace(content=[{"text": "Result text", "type": "text"}])
    )

    config = MCPServerConfig(name="test", transport="stdio", command="python")

    with (
        patch("opper_agents.mcp.client.stdio_client", mock_stdio_transport),
        patch("opper_agents.mcp.client.mcp.ClientSession", lambda *a, **kw: session),
    ):
        provider = mcp(config)
        tools = await provider.setup(None)

        result = await tools[0].execute()

        # Verify ToolResult structure
        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.tool_name == "test:test_tool"
        assert result.result is not None
        assert result.error is None
        assert result.execution_time >= 0

        await provider.teardown()


@pytest.mark.asyncio
async def test_mcp_disconnect_with_cancel_scope_shielding(mock_stdio_transport):
    """
    Test that MCP client disconnect is shielded from AnyIO cancel scopes.

    This test verifies that the disconnect operation completes successfully
    even when called within an AnyIO cancel scope, preventing CancelledError
    from propagating to subsequent operations.
    """
    import anyio

    session = IntegrationFakeSession()
    config = MCPServerConfig(name="test", transport="stdio", command="python")

    with (
        patch("opper_agents.mcp.client.stdio_client", mock_stdio_transport),
        patch("opper_agents.mcp.client.mcp.ClientSession", lambda *a, **kw: session),
    ):
        client = MCPClient(config)
        await client.connect()
        assert client.connected

        # Simulate a cancel scope that might occur during cleanup
        with anyio.CancelScope() as scope:
            scope.cancel()
            # Disconnect should still work despite the cancelled scope
            # The shield in disconnect() protects it
            await client.disconnect()
            assert not client.connected

        # Verify we can continue operations after disconnect
        # (simulates making HTTP calls to Opper after MCP cleanup)
        test_value = None
        with anyio.CancelScope(shield=True):
            # This simulates the final result generation
            await anyio.sleep(0)
            test_value = "completed"

        assert test_value == "completed"
