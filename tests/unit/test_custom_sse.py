"""
Tests for custom SSE client with POST support.
"""

import pytest
from unittest.mock import MagicMock, patch
from contextlib import asynccontextmanager
from httpx_sse import ServerSentEvent

from opper_agents.mcp.custom_sse import sse_client_post


@pytest.fixture
def mock_sse_response():
    """Mock SSE response."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    return mock_response


@pytest.fixture
def mock_event_source(mock_sse_response):
    """Mock SSE event source."""

    @asynccontextmanager
    async def _mock_event_source(*args, **kwargs):
        event_source = MagicMock()
        event_source.response = mock_sse_response

        async def mock_aiter():
            # Yield a few test events
            yield ServerSentEvent(event="message", data='{"test": "data1"}')
            yield ServerSentEvent(event="message", data='{"test": "data2"}')

        event_source.aiter_sse = mock_aiter
        yield event_source

    return _mock_event_source


@pytest.mark.asyncio
async def test_sse_client_post_basic_connection(mock_event_source):
    """Test POST SSE client establishes connection."""
    with patch("opper_agents.mcp.custom_sse.aconnect_sse", mock_event_source):
        async with sse_client_post(
            url="https://example.com/sse",
            headers={"Authorization": "Bearer token"},
            timeout=5,
        ) as (read_stream, write_stream):
            assert read_stream is not None
            assert write_stream is not None


@pytest.mark.asyncio
async def test_sse_client_post_reads_events(mock_event_source):
    """Test POST SSE client reads events from server."""
    with patch("opper_agents.mcp.custom_sse.aconnect_sse", mock_event_source):
        async with sse_client_post(
            url="https://example.com/sse",
            timeout=5,
        ) as (read_stream, write_stream):
            # Read first event
            event1 = await read_stream.receive()
            assert isinstance(event1, ServerSentEvent)
            assert event1.data == '{"test": "data1"}'

            # Read second event
            event2 = await read_stream.receive()
            assert isinstance(event2, ServerSentEvent)
            assert event2.data == '{"test": "data2"}'


@pytest.mark.asyncio
async def test_sse_client_post_default_headers():
    """Test POST SSE client adds default Accept header."""
    captured_headers = {}

    @asynccontextmanager
    async def capture_headers_sse(*args, **kwargs):
        captured_headers.update(kwargs.get("headers", {}))
        event_source = MagicMock()
        event_source.response = MagicMock()
        event_source.response.raise_for_status = MagicMock()

        async def empty_aiter():
            if False:
                yield

        event_source.aiter_sse = empty_aiter
        yield event_source

    with patch("opper_agents.mcp.custom_sse.aconnect_sse", capture_headers_sse):
        async with sse_client_post(url="https://example.com/sse", timeout=5):
            pass

    assert "Accept" in captured_headers
    assert captured_headers["Accept"] == "text/event-stream"


@pytest.mark.asyncio
async def test_sse_client_post_preserves_custom_headers():
    """Test POST SSE client preserves custom headers."""
    captured_headers = {}

    @asynccontextmanager
    async def capture_headers_sse(*args, **kwargs):
        captured_headers.update(kwargs.get("headers", {}))
        event_source = MagicMock()
        event_source.response = MagicMock()
        event_source.response.raise_for_status = MagicMock()

        async def empty_aiter():
            if False:
                yield

        event_source.aiter_sse = empty_aiter
        yield event_source

    with patch("opper_agents.mcp.custom_sse.aconnect_sse", capture_headers_sse):
        async with sse_client_post(
            url="https://example.com/sse",
            headers={"Authorization": "Bearer token", "X-Custom": "value"},
            timeout=5,
        ):
            pass

    assert captured_headers["Authorization"] == "Bearer token"
    assert captured_headers["X-Custom"] == "value"
    assert captured_headers["Accept"] == "text/event-stream"


@pytest.mark.asyncio
async def test_sse_client_post_uses_post_method():
    """Test POST SSE client uses POST method."""
    captured_method = None

    @asynccontextmanager
    async def capture_method_sse(*args, **kwargs):
        nonlocal captured_method
        # args[1] should be the method
        if len(args) > 1:
            captured_method = args[1]
        event_source = MagicMock()
        event_source.response = MagicMock()
        event_source.response.raise_for_status = MagicMock()

        async def empty_aiter():
            if False:
                yield

        event_source.aiter_sse = empty_aiter
        yield event_source

    with patch("opper_agents.mcp.custom_sse.aconnect_sse", capture_method_sse):
        async with sse_client_post(url="https://example.com/sse", timeout=5):
            pass

    assert captured_method == "POST"


@pytest.mark.asyncio
async def test_sse_client_post_handles_errors():
    """Test POST SSE client handles connection errors."""

    @asynccontextmanager
    async def failing_sse(*args, **kwargs):
        raise ConnectionError("Failed to connect")
        yield

    with patch("opper_agents.mcp.custom_sse.aconnect_sse", failing_sse):
        # Error is wrapped in ExceptionGroup by anyio task group
        with pytest.raises(BaseExceptionGroup) as exc_info:
            async with sse_client_post(url="https://example.com/sse", timeout=5):
                pass

        # Verify the original error is in the exception group
        assert len(exc_info.value.exceptions) == 1
        assert isinstance(exc_info.value.exceptions[0], ConnectionError)
        assert "Failed to connect" in str(exc_info.value.exceptions[0])


@pytest.mark.asyncio
async def test_sse_client_post_handles_none_headers():
    """Test POST SSE client handles None headers gracefully."""
    captured_headers = {}

    @asynccontextmanager
    async def capture_headers_sse(*args, **kwargs):
        captured_headers.update(kwargs.get("headers", {}))
        event_source = MagicMock()
        event_source.response = MagicMock()
        event_source.response.raise_for_status = MagicMock()

        async def empty_aiter():
            if False:
                yield

        event_source.aiter_sse = empty_aiter
        yield event_source

    with patch("opper_agents.mcp.custom_sse.aconnect_sse", capture_headers_sse):
        async with sse_client_post(
            url="https://example.com/sse", headers=None, timeout=5
        ):
            pass

    # Should have default Accept header even with None headers
    assert "Accept" in captured_headers
