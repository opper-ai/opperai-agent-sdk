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

from .config import MCPServerConfig

logger = logging.getLogger(__name__)


class MCPTool(BaseModel):
    """Metadata for a tool exposed by an MCP server."""

    name: str = Field(description="Tool name")
    description: str = Field(default="", description="Human readable description")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="JSON schema describing tool inputs"
    )
    output_schema: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional JSON schema describing structured output"
    )


class MCPClient:
    """Asynchronous client wrapper around an MCP ClientSession."""

    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config
        self._exit_stack: Optional[AsyncExitStack] = None
        self._session: Optional[mcp.ClientSession] = None
        self._connected = False
        self._tool_cache: List[MCPTool] = []
        self.server_info: Optional[mcp.Implementation] = None

    @classmethod
    def from_config(cls, config: MCPServerConfig) -> "MCPClient":
        """Create a client for the given server configuration."""
        return cls(config)

    async def connect(self) -> None:
        """Establish a connection to the configured MCP server."""
        if self._connected:
            return

        exit_stack = AsyncExitStack()
        try:
            read_stream: Any
            write_stream: Any

            if self.config.transport == "stdio":
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
                params = SseServerParameters(
                    url=self.config.url,
                    headers=self.config.headers or None,
                    timeout=self.config.timeout,
                    sse_read_timeout=self.config.timeout,
                )
                read_stream, write_stream = await exit_stack.enter_async_context(
                    sse_client(
                        url=params.url,
                        headers=params.headers,
                        timeout=params.timeout,
                        sse_read_timeout=params.sse_read_timeout,
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
        except Exception:
            await exit_stack.aclose()
            raise

    async def disconnect(self) -> None:
        """Terminate the underlying MCP session."""
        if not self._connected:
            return

        assert self._exit_stack is not None

        try:
            # Give the server a moment to finish any pending writes
            # This prevents EPIPE errors when the server is still writing
            await asyncio.sleep(0.2)

            await self._exit_stack.aclose()
        except Exception as e:
            logger.debug(f"Error during MCP disconnect for '{self.config.name}': {e}")
        finally:
            self._exit_stack = None
            self._session = None
            self._tool_cache = []
            self._connected = False
            logger.debug("Disconnected from MCP server %s", self.config.name)

    async def list_tools(self) -> List[MCPTool]:
        """Return the tools advertised by the server."""
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
        """Invoke a tool on the server and return its result."""
        session = self._ensure_session()
        result = await session.call_tool(tool_name, arguments or {})
        return result

    @property
    def connected(self) -> bool:
        """Expose connection state for provider bookkeeping."""
        return self._connected

    def _ensure_session(self) -> mcp.ClientSession:
        if not self._connected or self._session is None:
            raise RuntimeError(
                f"MCP client for server '{self.config.name}' is not connected"
            )
        return self._session


__all__ = ["MCPClient", "MCPTool"]
