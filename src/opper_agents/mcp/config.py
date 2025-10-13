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
    transport: Literal["stdio", "http-sse", "streamable-http"] = Field(
        description="Transport protocol to use: stdio (local), http-sse (deprecated SSE), or streamable-http (new 2025 protocol)"
    )

    # HTTP-SSE specific
    url: Optional[str] = Field(
        default=None, description="URL for HTTP-SSE transport (required if http-sse)"
    )
    headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Optional HTTP headers for HTTP-SSE transport",
    )
    method: Literal["GET", "POST"] = Field(
        default="GET",
        description="HTTP method for SSE connection (GET or POST). Some servers like Composio require POST.",
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
    def validate_transport_requirements(self) -> "MCPServerConfig":
        """
        Validate transport-specific requirements.

        Ensures that required fields are present based on the selected transport type.
        For stdio, a command is required. For http-sse and streamable-http, a URL is required.

        Returns:
            Validated config instance.

        Raises:
            ValueError: If transport-specific requirements are not met.
        """
        if self.transport == "stdio" and not self.command:
            raise ValueError("command is required for stdio transport")
        if self.transport in ("http-sse", "streamable-http") and not self.url:
            raise ValueError(f"url is required for {self.transport} transport")
        return self
