"""
Tests for MCP tool provider.
"""

import pytest
from unittest.mock import AsyncMock, patch
from opper_agents.mcp.config import MCPServerConfig
from opper_agents.mcp.provider import MCPToolProvider, mcp
from opper_agents.mcp.client import MCPTool
from opper_agents.base.tool import FunctionTool


def test_mcp_helper_function():
    """Test mcp() helper function."""
    config1 = MCPServerConfig(
        name="server1",
        transport="stdio",
        command="python",
    )
    config2 = MCPServerConfig(
        name="server2",
        transport="http-sse",
        url="https://example.com",
    )

    provider = mcp(config1, config2)

    assert isinstance(provider, MCPToolProvider)
    assert len(provider.configs) == 2
    assert provider.configs[0] == config1
    assert provider.configs[1] == config2


def test_mcp_helper_with_name_prefix():
    """Test mcp() helper with custom name prefix."""
    config = MCPServerConfig(
        name="server1",
        transport="stdio",
        command="python",
    )

    provider = mcp(config, name_prefix="custom")

    assert provider.name_prefix == "custom"


def test_mcp_helper_no_configs():
    """Test mcp() fails without configs."""
    with pytest.raises(ValueError, match="At least one MCPServerConfig"):
        mcp()


@pytest.mark.asyncio
async def test_provider_setup_single_server():
    """Test provider setup with single server."""
    config = MCPServerConfig(
        name="test",
        transport="stdio",
        command="python",
    )

    provider = MCPToolProvider([config])

    # Mock client
    mock_client = AsyncMock()
    mock_client.connect = AsyncMock()
    mock_client.list_tools = AsyncMock(
        return_value=[
            MCPTool(
                name="tool1",
                description="Test tool 1",
                parameters={"arg": "string"},
            ),
            MCPTool(
                name="tool2",
                description="Test tool 2",
                parameters={},
            ),
        ]
    )

    # Mock MCPClient.from_config
    with patch(
        "opper_agents.mcp.provider.MCPClient.from_config", return_value=mock_client
    ):
        tools = await provider.setup(None)

        assert len(tools) == 2
        assert all(isinstance(tool, FunctionTool) for tool in tools)

        # Check tool names have server prefix
        assert tools[0].name == "test:tool1"
        assert tools[1].name == "test:tool2"

        # Check descriptions
        assert tools[0].description == "Test tool 1"
        assert tools[1].description == "Test tool 2"

        # Verify client was connected
        mock_client.connect.assert_called_once()


@pytest.mark.asyncio
async def test_provider_setup_multiple_servers():
    """Test provider setup with multiple servers."""
    config1 = MCPServerConfig(name="server1", transport="stdio", command="python")
    config2 = MCPServerConfig(
        name="server2", transport="http-sse", url="https://example.com"
    )

    provider = MCPToolProvider([config1, config2])

    # Mock clients
    mock_client1 = AsyncMock()
    mock_client1.connect = AsyncMock()
    mock_client1.list_tools = AsyncMock(
        return_value=[
            MCPTool(name="tool1", description="Tool from server1", parameters={})
        ]
    )

    mock_client2 = AsyncMock()
    mock_client2.connect = AsyncMock()
    mock_client2.list_tools = AsyncMock(
        return_value=[
            MCPTool(name="tool2", description="Tool from server2", parameters={})
        ]
    )

    # Mock factory to return different clients
    def mock_from_config(config):
        if config.name == "server1":
            return mock_client1
        return mock_client2

    with patch(
        "opper_agents.mcp.provider.MCPClient.from_config", side_effect=mock_from_config
    ):
        tools = await provider.setup(None)

        assert len(tools) == 2
        assert tools[0].name == "server1:tool1"
        assert tools[1].name == "server2:tool2"

        # Both clients should be connected
        mock_client1.connect.assert_called_once()
        mock_client2.connect.assert_called_once()


@pytest.mark.asyncio
async def test_provider_setup_with_custom_prefix():
    """Test provider setup with custom name prefix."""
    config = MCPServerConfig(name="test", transport="stdio", command="python")

    provider = MCPToolProvider([config], name_prefix="custom")

    # Mock client
    mock_client = AsyncMock()
    mock_client.connect = AsyncMock()
    mock_client.list_tools = AsyncMock(
        return_value=[MCPTool(name="tool1", description="Test", parameters={})]
    )

    with patch(
        "opper_agents.mcp.provider.MCPClient.from_config", return_value=mock_client
    ):
        tools = await provider.setup(None)

        # Tool should use custom prefix
        assert tools[0].name == "custom:tool1"


@pytest.mark.asyncio
async def test_provider_setup_failure_handling():
    """Test provider handles setup failures gracefully."""
    config1 = MCPServerConfig(name="good", transport="stdio", command="python")
    config2 = MCPServerConfig(name="bad", transport="stdio", command="python")

    provider = MCPToolProvider([config1, config2])

    # Mock clients - one succeeds, one fails
    mock_client_good = AsyncMock()
    mock_client_good.connect = AsyncMock()
    mock_client_good.list_tools = AsyncMock(
        return_value=[MCPTool(name="tool1", description="Good tool", parameters={})]
    )

    mock_client_bad = AsyncMock()
    mock_client_bad.connect = AsyncMock(side_effect=RuntimeError("Connection failed"))

    def mock_from_config(config):
        if config.name == "good":
            return mock_client_good
        return mock_client_bad

    with patch(
        "opper_agents.mcp.provider.MCPClient.from_config", side_effect=mock_from_config
    ):
        # Should not raise, should continue with working server
        tools = await provider.setup(None)

        # Only tools from good server should be returned
        assert len(tools) == 1
        assert tools[0].name == "good:tool1"


@pytest.mark.asyncio
async def test_provider_teardown():
    """Test provider teardown disconnects all clients."""
    config1 = MCPServerConfig(name="server1", transport="stdio", command="python")
    config2 = MCPServerConfig(name="server2", transport="stdio", command="python")

    provider = MCPToolProvider([config1, config2])

    # Mock clients
    mock_client1 = AsyncMock()
    mock_client1.disconnect = AsyncMock()

    mock_client2 = AsyncMock()
    mock_client2.disconnect = AsyncMock()

    # Manually populate clients
    provider.clients = {"server1": mock_client1, "server2": mock_client2}

    await provider.teardown()

    # Both clients should be disconnected
    mock_client1.disconnect.assert_called_once()
    mock_client2.disconnect.assert_called_once()

    # Clients dict should be cleared
    assert len(provider.clients) == 0


@pytest.mark.asyncio
async def test_provider_teardown_handles_errors():
    """Test provider teardown continues even if disconnect fails."""
    config = MCPServerConfig(name="test", transport="stdio", command="python")

    provider = MCPToolProvider([config])

    # Mock client that fails to disconnect
    mock_client = AsyncMock()
    mock_client.disconnect = AsyncMock(side_effect=RuntimeError("Disconnect failed"))

    provider.clients = {"test": mock_client}

    # Should not raise
    await provider.teardown()

    # Clients should still be cleared
    assert len(provider.clients) == 0


@pytest.mark.asyncio
async def test_wrapped_tool_execution():
    """Test wrapped MCP tool can be executed."""
    config = MCPServerConfig(name="test", transport="stdio", command="python")

    provider = MCPToolProvider([config])

    # Mock client
    mock_client = AsyncMock()
    mock_client.connected = True
    mock_client.call_tool = AsyncMock(return_value=["result data"])

    # Create wrapped tool
    mcp_tool = MCPTool(name="test_tool", description="Test", parameters={})
    wrapped_tool = provider._wrap_tool("test", mcp_tool)

    # Add client to provider
    provider.clients["test"] = mock_client

    # Execute wrapped tool
    result = await wrapped_tool.execute(arg1="value1", arg2="value2")

    # Verify result
    assert result.success
    assert result.result == ["result data"]
    assert result.tool_name == "test:test_tool"

    # Verify client was called with correct parameters
    mock_client.call_tool.assert_called_once_with(
        "test_tool", {"arg1": "value1", "arg2": "value2"}
    )


@pytest.mark.asyncio
async def test_wrapped_tool_execution_error():
    """Test wrapped MCP tool handles execution errors."""
    config = MCPServerConfig(name="test", transport="stdio", command="python")

    provider = MCPToolProvider([config])

    # Mock client that fails
    mock_client = AsyncMock()
    mock_client.connected = True
    mock_client.call_tool = AsyncMock(side_effect=RuntimeError("Tool failed"))

    # Create wrapped tool
    mcp_tool = MCPTool(name="test_tool", description="Test", parameters={})
    wrapped_tool = provider._wrap_tool("test", mcp_tool)

    provider.clients["test"] = mock_client

    # Execute wrapped tool - should handle error gracefully
    result = await wrapped_tool.execute(arg="value")

    # Result should indicate failure
    assert not result.success
    assert result.error is not None
    assert "Tool failed" in result.error


@pytest.mark.asyncio
async def test_wrapped_tool_disconnected_client():
    """Test wrapped tool handles disconnected client by reconnecting."""
    config = MCPServerConfig(name="test", transport="stdio", command="python")

    provider = MCPToolProvider([config])

    # Mock disconnected client that can reconnect
    mock_client = AsyncMock()
    mock_client.connected = False
    mock_client.connect = AsyncMock()  # Should be called to reconnect

    async def mock_connect():
        mock_client.connected = True

    mock_client.connect.side_effect = mock_connect
    mock_client.call_tool = AsyncMock(return_value="result after reconnect")

    # Create wrapped tool
    mcp_tool = MCPTool(name="test_tool", description="Test", parameters={})
    wrapped_tool = provider._wrap_tool("test", mcp_tool)

    provider.clients["test"] = mock_client

    # Execute wrapped tool - should reconnect automatically
    result = await wrapped_tool.execute(arg="value")

    # Should succeed after reconnecting
    assert result.success
    mock_client.connect.assert_called_once()  # Verify reconnect was attempted


@pytest.mark.asyncio
async def test_provider_setup_all_servers_fail():
    """Test provider handles all servers failing gracefully."""
    config1 = MCPServerConfig(name="fail1", transport="stdio", command="python")
    config2 = MCPServerConfig(name="fail2", transport="stdio", command="python")

    provider = MCPToolProvider([config1, config2])

    # Both clients fail to connect
    mock_client1 = AsyncMock()
    mock_client1.connect = AsyncMock(side_effect=RuntimeError("Connection failed"))

    mock_client2 = AsyncMock()
    mock_client2.connect = AsyncMock(side_effect=RuntimeError("Connection failed"))

    def mock_from_config(config):
        if config.name == "fail1":
            return mock_client1
        return mock_client2

    with patch(
        "opper_agents.mcp.provider.MCPClient.from_config", side_effect=mock_from_config
    ):
        # Should not raise, should return empty list
        tools = await provider.setup(None)

        # No tools should be available
        assert len(tools) == 0


@pytest.mark.asyncio
async def test_provider_setup_timeout():
    """Test provider handles connection timeouts."""
    import asyncio

    config = MCPServerConfig(name="slow", transport="stdio", command="python")

    provider = MCPToolProvider([config])

    # Mock client that times out
    mock_client = AsyncMock()

    async def slow_connect():
        await asyncio.sleep(1)  # Simulate slow connection (1 second is enough for test)

    mock_client.connect = slow_connect

    with patch(
        "opper_agents.mcp.provider.MCPClient.from_config", return_value=mock_client
    ):
        # Even with timeout, should handle gracefully
        await provider.setup(None)

        # In practice, this test would need proper timeout handling
        # For now, we're just verifying the structure


@pytest.mark.asyncio
async def test_wrapped_tool_missing_client():
    """Test wrapped tool handles missing client."""
    config = MCPServerConfig(name="test", transport="stdio", command="python")

    provider = MCPToolProvider([config])

    # Create wrapped tool but don't add client
    mcp_tool = MCPTool(name="test_tool", description="Test", parameters={})
    wrapped_tool = provider._wrap_tool("test", mcp_tool)

    # Don't add client to provider.clients

    # Execute should handle missing client
    result = await wrapped_tool.execute()

    assert not result.success
    assert result.error is not None


@pytest.mark.asyncio
async def test_provider_teardown_partial_failure():
    """Test provider teardown continues even if some disconnects fail."""
    config1 = MCPServerConfig(name="server1", transport="stdio", command="python")
    config2 = MCPServerConfig(name="server2", transport="stdio", command="python")

    provider = MCPToolProvider([config1, config2])

    # First client fails to disconnect
    mock_client1 = AsyncMock()
    mock_client1.disconnect = AsyncMock(side_effect=RuntimeError("Disconnect failed"))

    # Second client disconnects successfully
    mock_client2 = AsyncMock()
    mock_client2.disconnect = AsyncMock()

    provider.clients = {"server1": mock_client1, "server2": mock_client2}

    # Should not raise, should continue
    await provider.teardown()

    # Both disconnects should have been attempted
    mock_client1.disconnect.assert_called_once()
    mock_client2.disconnect.assert_called_once()

    # Clients should be cleared
    assert len(provider.clients) == 0


@pytest.mark.asyncio
async def test_provider_empty_tool_list():
    """Test provider handles servers with no tools."""
    config = MCPServerConfig(name="empty", transport="stdio", command="python")

    provider = MCPToolProvider([config])

    # Mock client with no tools
    mock_client = AsyncMock()
    mock_client.connect = AsyncMock()
    mock_client.list_tools = AsyncMock(return_value=[])  # Empty list

    with patch(
        "opper_agents.mcp.provider.MCPClient.from_config", return_value=mock_client
    ):
        tools = await provider.setup(None)

        # Should return empty list without error
        assert len(tools) == 0
        assert isinstance(tools, list)
