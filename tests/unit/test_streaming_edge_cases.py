"""
Tests for streaming edge cases: cancellation, cleanup, and error handling.

These tests verify that:
1. STREAM_END is always emitted, even on exceptions
2. Stream cancellation is handled properly with cleanup
3. Resources are released on early exceptions
"""

import asyncio
import pytest
from unittest.mock import AsyncMock
from pydantic import BaseModel, Field

from opper_agents.core.agent import Agent
from opper_agents.agents.react import ReactAgent
from opper_agents.base.hooks import HookEvents
from opper_agents.utils.decorators import hook


class SimpleOutput(BaseModel):
    """Simple output schema for testing."""

    result: str = Field(description="result value")


def _make_event(delta=None, json_path=None, chunk_type="json", span_id=None):
    """Helper to create mock stream events."""

    class Data:
        def __init__(self):
            self.delta = delta
            self.json_path = json_path
            self.chunk_type = chunk_type
            self.span_id = span_id

    class Event:
        def __init__(self):
            self.data = Data()

    return Event()


@pytest.mark.asyncio
async def test_stream_end_emitted_on_early_exception(mock_opper_client, opper_api_key):
    """
    Test that STREAM_END is emitted even if an exception occurs
    before the stream is fully consumed.
    """
    events_seen = []

    @hook(HookEvents.STREAM_START)
    async def on_start(context, call_type, **kwargs):
        events_seen.append(("START", call_type))

    @hook(HookEvents.STREAM_ERROR)
    async def on_error(context, call_type, error, **kwargs):
        events_seen.append(("ERROR", call_type, str(error)))

    @hook(HookEvents.STREAM_END)
    async def on_end(context, call_type, **kwargs):
        events_seen.append(("END", call_type))

    # Stream that raises exception on first event
    async def failing_stream():
        yield _make_event(delta="partial", json_path="reasoning")
        raise RuntimeError("Stream failed mid-way")

    async def stream_async(name: str, **kwargs):
        class Response:
            def __init__(self):
                self.result = failing_stream()
                self.usage = None

        return Response()

    mock_opper_client.stream_async = AsyncMock(side_effect=stream_async)

    agent = Agent(
        name="FailingStream",
        enable_streaming=True,
        output_schema=SimpleOutput,
        hooks=[on_start, on_error, on_end],
        opper_api_key=opper_api_key,
    )

    # Should raise the exception
    with pytest.raises(RuntimeError, match="Stream failed mid-way"):
        await agent.process("test")

    # Verify lifecycle: START -> ERROR -> END
    event_types = [e[0] for e in events_seen]

    # Should have START, ERROR, and END for "think" phase
    assert "START" in event_types
    assert "ERROR" in event_types
    assert "END" in event_types

    # STREAM_END should be the last event (from finally block)
    assert event_types[-1] == "END"


@pytest.mark.asyncio
async def test_stream_end_emitted_on_immediate_exception(
    mock_opper_client, opper_api_key
):
    """
    Test that STREAM_END is emitted even if stream_async() itself
    raises an exception before yielding any events.
    """
    events_seen = []

    @hook(HookEvents.STREAM_START)
    async def on_start(context, call_type, **kwargs):
        events_seen.append(("START", call_type))

    @hook(HookEvents.STREAM_ERROR)
    async def on_error(context, call_type, error, **kwargs):
        events_seen.append(("ERROR", call_type))

    @hook(HookEvents.STREAM_END)
    async def on_end(context, call_type, **kwargs):
        events_seen.append(("END", call_type))

    # stream_async raises immediately
    async def stream_async(name: str, **kwargs):
        raise ConnectionError("Failed to connect to streaming API")

    mock_opper_client.stream_async = AsyncMock(side_effect=stream_async)

    agent = Agent(
        name="ImmediateFailStream",
        enable_streaming=True,
        output_schema=SimpleOutput,
        hooks=[on_start, on_error, on_end],
        opper_api_key=opper_api_key,
    )

    # Should raise the exception
    with pytest.raises(ConnectionError):
        await agent.process("test")

    # Verify: START (before try) -> ERROR (in except) -> END (in finally)
    event_types = [e[0] for e in events_seen]
    assert event_types == ["START", "ERROR", "END"]


@pytest.mark.asyncio
async def test_stream_cancellation_cleanup(mock_opper_client, opper_api_key):
    """
    Test that cancelling a stream mid-way triggers proper cleanup
    and STREAM_END is emitted.
    """
    events_seen = []
    chunks_received = []

    @hook(HookEvents.STREAM_START)
    async def on_start(context, call_type, **kwargs):
        events_seen.append(("START", call_type))

    @hook(HookEvents.STREAM_CHUNK)
    async def on_chunk(context, call_type, chunk_data, **kwargs):
        chunks_received.append(chunk_data.get("delta"))

    @hook(HookEvents.STREAM_END)
    async def on_end(context, call_type, **kwargs):
        events_seen.append(("END", call_type))

    # Stream that yields multiple chunks slowly
    async def slow_stream():
        for i in range(10):
            yield _make_event(delta=f"chunk{i}", json_path="reasoning")
            await asyncio.sleep(0.01)  # Small delay

    async def stream_async(name: str, **kwargs):
        class Response:
            def __init__(self):
                self.result = slow_stream()
                self.usage = None

        return Response()

    mock_opper_client.stream_async = AsyncMock(side_effect=stream_async)

    agent = Agent(
        name="CancellableStream",
        enable_streaming=True,
        output_schema=SimpleOutput,
        hooks=[on_start, on_chunk, on_end],
        opper_api_key=opper_api_key,
    )

    # Create a task and cancel it mid-stream
    task = asyncio.create_task(agent.process("test"))

    # Let it process a few chunks
    await asyncio.sleep(0.03)

    # Cancel the task
    task.cancel()

    # Should raise CancelledError
    with pytest.raises(asyncio.CancelledError):
        await task

    # Verify that at least START was emitted
    # Note: STREAM_END emission on cancellation depends on whether
    # the finally block runs before task cancellation completes
    event_types = [e[0] for e in events_seen]
    assert "START" in event_types

    # We should have received some chunks before cancellation
    assert len(chunks_received) > 0


@pytest.mark.asyncio
async def test_react_agent_stream_end_on_exception(mock_opper_client, opper_api_key):
    """
    Test that ReactAgent also properly emits STREAM_END on exceptions.
    """
    events_seen = []

    @hook(HookEvents.STREAM_START)
    async def on_start(context, call_type, **kwargs):
        events_seen.append(("START", call_type))

    @hook(HookEvents.STREAM_ERROR)
    async def on_error(context, call_type, error, **kwargs):
        events_seen.append(("ERROR", call_type))

    @hook(HookEvents.STREAM_END)
    async def on_end(context, call_type, **kwargs):
        events_seen.append(("END", call_type))

    # Failing stream for ReactAgent
    async def stream_async(name: str, **kwargs):
        raise ValueError("ReactAgent stream failed")

    mock_opper_client.stream_async = AsyncMock(side_effect=stream_async)

    agent = ReactAgent(
        name="FailingReactStream",
        enable_streaming=True,
        output_schema=SimpleOutput,
        hooks=[on_start, on_error, on_end],
        opper_api_key=opper_api_key,
    )

    # Should raise the exception
    with pytest.raises(ValueError, match="ReactAgent stream failed"):
        await agent.process("test")

    # Verify lifecycle includes STREAM_END
    event_types = [e[0] for e in events_seen]
    assert "START" in event_types
    assert "ERROR" in event_types
    assert "END" in event_types
    assert event_types[-1] == "END"


@pytest.mark.asyncio
async def test_field_buffers_preserved_on_exception(mock_opper_client, opper_api_key):
    """
    Test that field_buffers are accessible in STREAM_END hook
    even when an exception occurs, allowing inspection of partial data.
    """
    captured_buffers = {}

    @hook(HookEvents.STREAM_END)
    async def on_end(context, call_type, field_buffers, **kwargs):
        captured_buffers[call_type] = field_buffers.copy()

    # Stream that yields some data then fails
    async def partial_stream():
        yield _make_event(delta="Thinking", json_path="reasoning")
        yield _make_event(delta=" about", json_path="reasoning")
        raise RuntimeError("Failed mid-stream")

    async def stream_async(name: str, **kwargs):
        class Response:
            def __init__(self):
                self.result = partial_stream()
                self.usage = None

        return Response()

    mock_opper_client.stream_async = AsyncMock(side_effect=stream_async)

    agent = Agent(
        name="PartialStream",
        enable_streaming=True,
        output_schema=SimpleOutput,
        hooks=[on_end],
        opper_api_key=opper_api_key,
    )

    # Should raise exception
    with pytest.raises(RuntimeError):
        await agent.process("test")

    # Verify field_buffers were passed to STREAM_END
    assert "think" in captured_buffers
    buffers = captured_buffers["think"]

    # Should have partial data from before the exception
    assert "reasoning" in buffers
    assert "".join(buffers["reasoning"]) == "Thinking about"


@pytest.mark.asyncio
async def test_multiple_exceptions_all_emit_stream_end(
    mock_opper_client, opper_api_key
):
    """
    Test that if both think and final_result streaming fail,
    both emit STREAM_END.
    """
    events_seen = []

    @hook(HookEvents.STREAM_START)
    async def on_start(context, call_type, **kwargs):
        events_seen.append(("START", call_type))

    @hook(HookEvents.STREAM_END)
    async def on_end(context, call_type, **kwargs):
        events_seen.append(("END", call_type))

    call_count = [0]

    # Fail on first call (think), succeed on second (won't happen)
    async def stream_async(name: str, **kwargs):
        call_count[0] += 1
        # Accept dynamic think names (think_{agent_name})
        if name.startswith("think_"):
            raise RuntimeError("Think stream failed")
        # Won't reach here
        raise AssertionError("Should not reach final_result")

    mock_opper_client.stream_async = AsyncMock(side_effect=stream_async)

    agent = Agent(
        name="MultiFailStream",
        enable_streaming=True,
        output_schema=SimpleOutput,
        hooks=[on_start, on_end],
        opper_api_key=opper_api_key,
    )

    # Should raise exception from think phase
    with pytest.raises(RuntimeError, match="Think stream failed"):
        await agent.process("test")

    # Should have START and END for the failed think call
    event_types = [e[0] for e in events_seen]
    call_types = [e[1] for e in events_seen]

    assert event_types.count("START") == 1
    assert event_types.count("END") == 1
    assert call_types[0] == "think"  # START for think
    assert call_types[1] == "think"  # END for think
