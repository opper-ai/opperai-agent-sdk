"""
Model Context Protocol (MCP) integration.

This module provides seamless integration with MCP servers, allowing them to be
used as tool providers in agents. It supports multiple transport types (stdio,
http-sse, streamable-http) and handles connection lifecycle management automatically.

The main components are:
- MCPServerConfig: Declarative configuration for MCP servers
- MCPClient: Low-level async client for MCP communication
- MCPToolProvider: High-level provider that exposes MCP tools to agents
- mcp(): Helper function for clean agent configuration syntax

Examples:
    # Basic usage with stdio transport
    from opper_agent import Agent
    from opper_agent.mcp import MCPServerConfig, mcp

    filesystem = MCPServerConfig(
        name="filesystem",
        transport="stdio",
        command="uvx",
        args=["mcp-server-filesystem", "/tmp"]
    )

    agent = Agent(
        name="FileAgent",
        tools=[mcp(filesystem)]
    )

    # HTTP-SSE transport
    search = MCPServerConfig(
        name="search",
        transport="http-sse",
        url="https://api.example.com/mcp",
        headers={"Authorization": "Bearer token"}
    )

    agent = Agent(
        name="SearchAgent",
        tools=[mcp(search)]
    )

    # Multiple MCP servers
    agent = Agent(
        name="MultiAgent",
        tools=[mcp(filesystem, search), my_local_tool]
    )
"""

from .config import MCPServerConfig
from .client import MCPClient, MCPTool
from .provider import MCPToolProvider, mcp

__all__ = [
    "MCPServerConfig",
    "MCPClient",
    "MCPTool",
    "MCPToolProvider",
    "mcp",
]
