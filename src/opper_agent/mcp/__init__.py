"""
Model Context Protocol (MCP) integration.

Provides clean integration with MCP servers as tool providers.
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
