"""
MCP client integration built on top of the official python-sdk.

Provides thin wrappers that translate between the Opper Agent abstractions
and the Model Context Protocol ClientSession implementation.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

import mcp
from mcp.client.sse import sse_client
from mcp.client.session_group import SseServerParameters
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamablehttp_client

from .config import MCPServerConfig
from .custom_sse import sse_client_post

logger = logging.getLogger(__name__)


class MCPTool(BaseModel):
    """
    Metadata for a tool exposed by an MCP server.

    This class represents a tool's schema and metadata as advertised by an MCP server.
    It includes the tool's name, description, input parameters schema, and optional
    output schema.

    Attributes:
        name: Unique identifier for the tool within the server.
        description: Human-readable description of what the tool does.
        parameters: JSON schema describing the tool's input parameters.
        output_schema: Optional JSON schema describing structured output format.
    """

    name: str = Field(description="Tool name")
    description: str = Field(default="", description="Human readable description")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="JSON schema describing tool inputs"
    )
    output_schema: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional JSON schema describing structured output"
    )


class MCPClient:
    """
    Asynchronous client wrapper around an MCP ClientSession.

    This client manages the lifecycle of a connection to an MCP server, supporting
    multiple transport types (stdio, http-sse, streamable-http). It handles connection
    management, tool discovery, and tool execution.

    The client maintains a connection pool and tool cache for efficiency. Tools are
    cached after the first list_tools() call to avoid redundant server queries.

    Attributes:
        config: Server configuration including transport and connection details.
        server_info: Information about the connected server (available after connect()).

    Examples:
        # stdio transport (local subprocess)
        config = MCPServerConfig(
            name="filesystem",
            transport="stdio",
            command="uvx",
            args=["mcp-server-filesystem", "/tmp"]
        )
        client = MCPClient(config)
        await client.connect()
        tools = await client.list_tools()
        result = await client.call_tool("read_file", {"path": "/tmp/test.txt"})
        await client.disconnect()

        # http-sse transport (remote server)
        config = MCPServerConfig(
            name="search",
            transport="http-sse",
            url="https://api.example.com/mcp"
        )
        client = MCPClient(config)
        await client.connect()
        tools = await client.list_tools()
        await client.disconnect()
    """

    def __init__(self, config: MCPServerConfig) -> None:
        """
        Initialize MCP client with configuration.

        Args:
            config: Server configuration including transport type and connection details.
        """
        self.config = config
        self._exit_stack: Optional[AsyncExitStack] = None
        self._session: Optional[mcp.ClientSession] = None
        self._connected = False
        self._tool_cache: List[MCPTool] = []
        self.server_info: Optional[mcp.Implementation] = None

    @classmethod
    def from_config(cls, config: MCPServerConfig) -> "MCPClient":
        """
        Create a client for the given server configuration.

        Args:
            config: MCP server configuration.

        Returns:
            Initialized MCPClient instance (not yet connected).
        """
        return cls(config)

    async def connect(self) -> None:
        """
        Establish a connection to the configured MCP server.

        This method initializes the appropriate transport (stdio, http-sse, or
        streamable-http), creates a ClientSession, and performs the MCP initialization
        handshake. The connection is idempotent - calling connect() multiple times
        has no effect if already connected.

        Raises:
            ValueError: If transport configuration is invalid or missing required fields.
            RuntimeError: If connection fails or server initialization fails.
        """
        if self._connected:
            return

        exit_stack = AsyncExitStack()
        try:
            read_stream: Any
            write_stream: Any

            if self.config.transport == "stdio":
                if not self.config.command:
                    raise ValueError("stdio transport requires a command")
                params = StdioServerParameters(
                    command=self.config.command,
                    args=self.config.args,
                    env=self.config.env or None,
                )
                read_stream, write_stream = await exit_stack.enter_async_context(
                    stdio_client(params)
                )
            elif self.config.transport == "http-sse":
                if not self.config.url:
                    raise ValueError("HTTP-SSE transport requires a URL")

                # Choose SSE client based on method
                if self.config.method == "POST":
                    # Use custom POST SSE client
                    read_stream, write_stream = await exit_stack.enter_async_context(
                        sse_client_post(
                            url=self.config.url,
                            headers=self.config.headers or None,
                            timeout=self.config.timeout,
                            sse_read_timeout=self.config.timeout,
                        )
                    )
                else:
                    # Use default GET SSE client from MCP SDK
                    sse_params = SseServerParameters(
                        url=self.config.url,
                        headers=self.config.headers or None,
                        timeout=self.config.timeout,
                        sse_read_timeout=self.config.timeout,
                    )
                    read_stream, write_stream = await exit_stack.enter_async_context(
                        sse_client(
                            url=sse_params.url,
                            headers=sse_params.headers,
                            timeout=sse_params.timeout,
                            sse_read_timeout=sse_params.sse_read_timeout,
                        )
                    )
            elif self.config.transport == "streamable-http":
                if not self.config.url:
                    raise ValueError("Streamable HTTP transport requires a URL")

                # Use new Streamable HTTP transport (MCP 2025)
                (
                    read_stream,
                    write_stream,
                    _get_session_id,
                ) = await exit_stack.enter_async_context(
                    streamablehttp_client(
                        url=self.config.url,
                        headers=self.config.headers or None,
                        timeout=self.config.timeout,
                        sse_read_timeout=self.config.timeout,
                    )
                )
            else:
                raise ValueError(f"Unsupported transport '{self.config.transport}'")

            session = mcp.ClientSession(read_stream, write_stream)
            self._session = await exit_stack.enter_async_context(session)
            init_result = await self._session.initialize()
            self.server_info = init_result.serverInfo
            self._connected = True
            self._exit_stack = exit_stack
            logger.debug(
                "Connected to MCP server %s via %s",
                self.config.name,
                self.config.transport,
            )
        except Exception as e:
            # Shield cleanup from AnyIO cancel scopes to preserve the original error
            # Without this, CancelledError can mask the real connection error
            import anyio

            try:
                with anyio.CancelScope(shield=True):
                    await exit_stack.aclose()
            except Exception as cleanup_error:
                logger.debug(
                    f"Error during cleanup after connection failure: {cleanup_error}"
                )

            # Extract meaningful error from ExceptionGroup if present
            # MCP SDK uses AnyIO TaskGroups which wrap errors in ExceptionGroups
            original_error = e
            if hasattr(e, "exceptions"):  # ExceptionGroup or BaseExceptionGroup
                # Find the first non-cancellation error
                for exc in e.exceptions:
                    if not isinstance(exc, asyncio.CancelledError):
                        original_error = exc
                        break

            # Re-raise the original/extracted connection error with context
            raise RuntimeError(
                f"Failed to connect to MCP server '{self.config.name}' "
                f"via {self.config.transport}: {original_error}"
            ) from original_error

    async def disconnect(self) -> None:
        """
        Terminate the underlying MCP session.

        This method gracefully closes the connection to the MCP server, cleaning up
        all resources including streams, sessions, and cached data. The method is
        idempotent - calling it multiple times has no effect if already disconnected.

        A short delay is included before closing to prevent EPIPE errors if the server
        is still writing data. The entire cleanup is shielded from cancellation to
        ensure proper resource cleanup.
        """
        if not self._connected:
            return

        assert self._exit_stack is not None

        try:
            # Import anyio for shielding
            import anyio

            # Shield the entire cleanup from cancellation to ensure proper resource cleanup
            # This prevents AnyIO cancel scope issues from affecting subsequent operations
            with anyio.CancelScope(shield=True):
                # Close the exit stack which will terminate the subprocess
                await self._exit_stack.aclose()

                # Give the subprocess a moment to fully terminate
                # This prevents buffered stderr/stdout from the subprocess (like npm notices)
                # from appearing in the terminal after disconnect
                await asyncio.sleep(0.1)
        except Exception as e:
            logger.debug(f"Error during MCP disconnect for '{self.config.name}': {e}")
        finally:
            self._exit_stack = None
            self._session = None
            self._tool_cache = []
            self._connected = False
            logger.debug("Disconnected from MCP server %s", self.config.name)

    async def list_tools(self) -> List[MCPTool]:
        """
        Return the tools advertised by the server.

        This method queries the server for its available tools and caches the result.
        Subsequent calls return the cached list without making additional server requests.

        Returns:
            List of MCPTool objects describing available tools.

        Raises:
            RuntimeError: If client is not connected.
        """
        session = self._ensure_session()
        if self._tool_cache:
            return self._tool_cache

        response = await session.list_tools()
        tools: List[MCPTool] = []
        for tool in response.tools:
            tools.append(
                MCPTool(
                    name=tool.name,
                    description=tool.description or "",
                    parameters=tool.inputSchema,
                    output_schema=tool.outputSchema,
                )
            )

        self._tool_cache = tools
        return tools

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Invoke a tool on the server and return its result.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Dictionary of arguments to pass to the tool.

        Returns:
            Tool execution result from the server.

        Raises:
            RuntimeError: If client is not connected.
            Exception: If tool execution fails on the server.
        """
        session = self._ensure_session()
        result = await session.call_tool(tool_name, arguments or {})
        return result

    @property
    def connected(self) -> bool:
        """
        Check if client is currently connected to the server.

        Returns:
            True if connected, False otherwise.
        """
        return self._connected

    def _ensure_session(self) -> mcp.ClientSession:
        """
        Ensure client is connected and return the session.

        Returns:
            Active MCP ClientSession.

        Raises:
            RuntimeError: If client is not connected.
        """
        if not self._connected or self._session is None:
            raise RuntimeError(
                f"MCP client for server '{self.config.name}' is not connected"
            )
        return self._session


__all__ = ["MCPClient", "MCPTool"]
