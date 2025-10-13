"""
Tests for MCP configuration.
"""

import pytest
from pydantic import ValidationError
from opper_agents.mcp.config import MCPServerConfig


def test_stdio_config_valid():
    """Test valid stdio configuration."""
    config = MCPServerConfig(
        name="filesystem",
        transport="stdio",
        command="uvx",
        args=["mcp-server-filesystem", "/tmp"],
        env={"DEBUG": "1"},
        timeout=10.0,
    )

    assert config.name == "filesystem"
    assert config.transport == "stdio"
    assert config.command == "uvx"
    assert config.args == ["mcp-server-filesystem", "/tmp"]
    assert config.env == {"DEBUG": "1"}
    assert config.timeout == 10.0
    assert config.url is None


def test_stdio_config_minimal():
    """Test minimal stdio configuration."""
    config = MCPServerConfig(
        name="test",
        transport="stdio",
        command="python",
    )

    assert config.name == "test"
    assert config.command == "python"
    assert config.args == []
    assert config.env == {}
    assert config.timeout == 30.0


def test_stdio_config_missing_command():
    """Test stdio config fails without command."""
    with pytest.raises(ValidationError, match="command is required"):
        MCPServerConfig(
            name="test",
            transport="stdio",
        )


def test_http_sse_config_valid():
    """Test valid HTTP-SSE configuration."""
    config = MCPServerConfig(
        name="search",
        transport="http-sse",
        url="https://mcp-server.example.com/sse",
        timeout=15.0,
    )

    assert config.name == "search"
    assert config.transport == "http-sse"
    assert config.url == "https://mcp-server.example.com/sse"
    assert config.timeout == 15.0
    assert config.command is None
    assert config.headers == {}


def test_http_sse_config_with_headers():
    """Headers are accepted for HTTP-SSE transport."""
    config = MCPServerConfig(
        name="search",
        transport="http-sse",
        url="https://mcp-server.example.com/sse",
        headers={"Authorization": "Bearer token"},
    )

    assert config.headers == {"Authorization": "Bearer token"}


def test_http_sse_config_missing_url():
    """Test HTTP-SSE config fails without URL."""
    with pytest.raises(ValidationError, match="url is required"):
        MCPServerConfig(
            name="test",
            transport="http-sse",
        )


def test_invalid_transport():
    """Test invalid transport type fails."""
    with pytest.raises(ValidationError):
        MCPServerConfig(
            name="test",
            transport="invalid",
            command="python",
        )


def test_config_defaults():
    """Test default values are applied."""
    config = MCPServerConfig(
        name="test",
        transport="stdio",
        command="python",
    )

    assert config.args == []
    assert config.env == {}
    assert config.timeout == 30.0


def test_streamable_http_config_valid():
    """Test valid streamable HTTP configuration."""
    config = MCPServerConfig(
        name="composio",
        transport="streamable-http",
        url="https://backend.composio.dev/v3/mcp/test/mcp",
        timeout=60.0,
    )

    assert config.name == "composio"
    assert config.transport == "streamable-http"
    assert config.url == "https://backend.composio.dev/v3/mcp/test/mcp"
    assert config.timeout == 60.0
    assert config.command is None


def test_streamable_http_config_missing_url():
    """Test streamable HTTP config fails without URL."""
    with pytest.raises(ValidationError, match="url is required"):
        MCPServerConfig(
            name="test",
            transport="streamable-http",
        )


def test_http_sse_method_default():
    """Test HTTP-SSE method defaults to GET."""
    config = MCPServerConfig(
        name="test",
        transport="http-sse",
        url="https://example.com/sse",
    )

    assert config.method == "GET"


def test_http_sse_method_post():
    """Test HTTP-SSE can use POST method."""
    config = MCPServerConfig(
        name="test",
        transport="http-sse",
        url="https://example.com/sse",
        method="POST",
    )

    assert config.method == "POST"


def test_http_sse_invalid_method():
    """Test HTTP-SSE rejects invalid methods."""
    with pytest.raises(ValidationError):
        MCPServerConfig(
            name="test",
            transport="http-sse",
            url="https://example.com/sse",
            method="PUT",
        )


def test_streamable_http_with_headers():
    """Test streamable HTTP accepts headers."""
    config = MCPServerConfig(
        name="test",
        transport="streamable-http",
        url="https://example.com/mcp",
        headers={"Authorization": "Bearer token"},
    )

    assert config.headers == {"Authorization": "Bearer token"}


def test_stdio_ignores_url():
    """Test stdio transport ignores URL parameter."""
    config = MCPServerConfig(
        name="test",
        transport="stdio",
        command="python",
        url="https://example.com",  # Should be ignored
    )

    assert config.transport == "stdio"
    assert config.url == "https://example.com"  # Stored but not used


def test_config_with_all_fields():
    """Test config with all fields specified."""
    config = MCPServerConfig(
        name="full-config",
        transport="http-sse",
        url="https://example.com/sse",
        headers={"X-API-Key": "secret"},
        method="POST",
        command=None,
        args=[],
        env={},
        timeout=45.0,
    )

    assert config.name == "full-config"
    assert config.transport == "http-sse"
    assert config.url == "https://example.com/sse"
    assert config.headers == {"X-API-Key": "secret"}
    assert config.method == "POST"
    assert config.timeout == 45.0
