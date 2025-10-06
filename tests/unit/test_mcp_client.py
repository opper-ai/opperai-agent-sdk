"""Unit tests for MCP client wrapper built on top of python-sdk."""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, Tuple
from unittest.mock import AsyncMock, patch

import pytest

import mcp

from opper_agent.mcp.client import MCPClient, MCPTool
import opper_agent.mcp.client as client_module
from opper_agent.mcp.config import MCPServerConfig


class FakeSession:
    """Minimal async context manager mimicking mcp.ClientSession."""

    def __init__(self) -> None:
        self.initialize = AsyncMock(
            return_value=SimpleNamespace(
                serverInfo=mcp.Implementation(name="fake-server", version="1.0"),
            )
        )
        self.list_tools = AsyncMock(
            return_value=SimpleNamespace(
                tools=[
                    SimpleNamespace(
                        name="alpha",
                        description="Alpha tool",
                        inputSchema={"type": "object"},
                        outputSchema=None,
                    )
                ]
            )
        )
        self.call_tool = AsyncMock(return_value="call-result")
        self.exit_called = False

    async def __aenter__(self) -> "FakeSession":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        self.exit_called = True
        return False


@asynccontextmanager
async def fake_stdio_client(*_: Any, **__: Any) -> Tuple[Any, Any]:
    yield object(), object()


@asynccontextmanager
async def fake_sse_client(*_: Any, **__: Any) -> Tuple[Any, Any]:
    yield object(), object()


@pytest.mark.asyncio
async def test_mcp_client_connect_stdio() -> None:
    """Client establishes stdio connection and stores server info."""
    config = MCPServerConfig(
        name="filesystem",
        transport="stdio",
        command="uvx",
        args=["mcp-server-filesystem", "/tmp"],
    )

    session = FakeSession()

    with (
        patch("opper_agent.mcp.client.stdio_client", fake_stdio_client),
        patch("opper_agent.mcp.client.mcp.ClientSession", lambda *a, **kw: session),
    ):
        client = MCPClient(config)
        await client.connect()

        session.initialize.assert_awaited()
        assert client.server_info is not None
        assert client.server_info.name == "fake-server"

        # Disconnect should close the underlying session
        await client.disconnect()
        assert session.exit_called is True


@pytest.mark.asyncio
async def test_mcp_client_list_tools_caches_results() -> None:
    """Tools are cached after the first list operation."""
    config = MCPServerConfig(
        name="filesystem",
        transport="stdio",
        command="uvx",
    )

    session = FakeSession()

    with (
        patch("opper_agent.mcp.client.stdio_client", fake_stdio_client),
        patch("opper_agent.mcp.client.mcp.ClientSession", lambda *a, **kw: session),
    ):
        client = MCPClient(config)
        await client.connect()

        tools_first = await client.list_tools()
        tools_second = await client.list_tools()

        assert session.list_tools.await_count == 1
        assert tools_first == tools_second
        assert isinstance(tools_first[0], MCPTool)
        assert tools_first[0].name == "alpha"


@pytest.mark.asyncio
async def test_mcp_client_call_tool() -> None:
    """Client proxies tool invocations to the session."""
    config = MCPServerConfig(
        name="search",
        transport="http-sse",
        url="https://example.com/mcp",
    )

    session = FakeSession()

    with (
        patch("opper_agent.mcp.client.sse_client", fake_sse_client),
        patch("opper_agent.mcp.client.mcp.ClientSession", lambda *a, **kw: session),
    ):
        client = MCPClient(config)
        await client.connect()

        result = await client.call_tool("alpha", {"query": "hello"})

        session.call_tool.assert_awaited_with("alpha", {"query": "hello"})
        assert result == "call-result"


@pytest.mark.asyncio
async def test_mcp_client_requires_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Operations before connect raise informative errors."""
    config = MCPServerConfig(
        name="filesystem",
        transport="stdio",
        command="uvx",
    )

    client = MCPClient(config)

    with pytest.raises(RuntimeError, match="not connected"):
        await client.list_tools()

    with pytest.raises(RuntimeError, match="not connected"):
        await client.call_tool("alpha", {})
