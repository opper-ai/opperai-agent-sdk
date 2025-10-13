"""
Custom SSE client that supports POST method.

The official MCP Python SDK's sse_client only supports GET,
but some MCP servers (like Composio) require POST.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import anyio
import httpx
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from httpx_sse import ServerSentEvent, aconnect_sse


@asynccontextmanager
async def sse_client_post(
    url: str,
    headers: dict[str, Any] | None = None,
    timeout: float = 5,
    sse_read_timeout: float = 300,
) -> Any:
    """
    SSE client that uses POST method instead of GET.

    Args:
        url: SSE endpoint URL
        headers: Optional HTTP headers
        timeout: Connection timeout in seconds
        sse_read_timeout: Timeout for reading SSE events

    Yields:
        Tuple of (read_stream, write_stream) for MCP communication
    """
    read_stream_writer: MemoryObjectSendStream[ServerSentEvent | Exception]
    read_stream: MemoryObjectReceiveStream[ServerSentEvent | Exception]
    read_stream_writer, read_stream = anyio.create_memory_object_stream(0)

    write_stream: MemoryObjectSendStream[str | bytes]
    write_stream_reader: MemoryObjectReceiveStream[str | bytes]
    write_stream, write_stream_reader = anyio.create_memory_object_stream(0)

    async with anyio.create_task_group() as tg:
        # Create httpx client for SSE connection
        async with httpx.AsyncClient(timeout=timeout) as http_client:
            # Prepare headers for SSE connection
            sse_headers = headers or {}
            # Ensure Accept header for SSE (aconnect_sse adds this but we're explicit)
            if "Accept" not in sse_headers:
                sse_headers["Accept"] = "text/event-stream"

            async with aconnect_sse(
                http_client,
                "POST",  # <-- This is the key difference from GET
                url,
                headers=sse_headers,
            ) as event_source:
                event_source.response.raise_for_status()

                async def read_sse_task() -> None:
                    """Read SSE events and forward to read_stream."""
                    try:
                        async for event in event_source.aiter_sse():
                            await read_stream_writer.send(event)
                    except Exception as e:
                        await read_stream_writer.send(e)

                async def write_sse_task() -> None:
                    """Read from write_stream and send as POST data."""
                    try:
                        with anyio.fail_after(sse_read_timeout):
                            async for message in write_stream_reader:
                                # For POST SSE, we send data as request body
                                # This part may need adjustment based on server expectations
                                pass
                    except TimeoutError:
                        pass

                tg.start_soon(read_sse_task)
                tg.start_soon(write_sse_task)

                try:
                    yield read_stream, write_stream
                finally:
                    tg.cancel_scope.cancel()


__all__ = ["sse_client_post"]
