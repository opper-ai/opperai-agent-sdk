"""
MCP server configuration.

Provides declarative configuration for Model Context Protocol servers.
"""

from pydantic import BaseModel, Field, model_validator
from typing import Dict, List, Literal, Optional


class MCPServerConfig(BaseModel):
    """
    Declarative configuration for an MCP server.

    Supports two transport types:
    - stdio: Local subprocess communication
    - http-sse: HTTP Server-Sent Events

    Examples:
        # Local stdio server
        config = MCPServerConfig(
            name="filesystem",
            transport="stdio",
            command="uvx",
            args=["mcp-server-filesystem", "/path/to/dir"]
        )

        # Remote HTTP-SSE server
        config = MCPServerConfig(
            name="search",
            transport="http-sse",
            url="https://mcp-server.example.com/sse"
        )
    """

    name: str = Field(description="Unique identifier for this MCP server")
    transport: Literal["stdio", "http-sse"] = Field(
        description="Transport protocol to use"
    )

    # HTTP-SSE specific
    url: Optional[str] = Field(
        default=None, description="URL for HTTP-SSE transport (required if http-sse)"
    )
    headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Optional HTTP headers for HTTP-SSE transport",
    )

    # stdio specific
    command: Optional[str] = Field(
        default=None,
        description="Command to execute for stdio transport (required if stdio)",
    )
    args: List[str] = Field(
        default_factory=list, description="Arguments for the command"
    )
    env: Dict[str, str] = Field(
        default_factory=dict, description="Environment variables for the subprocess"
    )

    # Common
    timeout: float = Field(
        default=30.0, description="Timeout for operations in seconds"
    )

    @model_validator(mode="after")
    def validate_transport_requirements(self):
        """Validate transport-specific requirements."""
        if self.transport == "stdio" and not self.command:
            raise ValueError("command is required for stdio transport")
        if self.transport == "http-sse" and not self.url:
            raise ValueError("url is required for http-sse transport")
        return self
