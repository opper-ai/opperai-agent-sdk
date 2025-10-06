"""
Tests for MCP configuration.
"""

import pytest
from pydantic import ValidationError
from opper_agent.mcp.config import MCPServerConfig


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
