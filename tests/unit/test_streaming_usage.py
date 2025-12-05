import pytest
from unittest.mock import AsyncMock
from pydantic import BaseModel, Field

from opper_agents.core.agent import Agent


class Out(BaseModel):
    text: str = Field(description="out")


def make_evt(delta=None, json_path=None, span_id=None, chunk_type="json"):
    class Data:
        def __init__(self):
            self.delta = delta
            self.json_path = json_path
            self.span_id = span_id
            self.chunk_type = chunk_type

    class Event:
        def __init__(self):
            self.data = Data()

    return Event()


@pytest.mark.asyncio
async def test_streaming_usage_via_span_lookup(mock_opper_client, opper_api_key):
    think_span_id = "span-think-1"
    final_span_id = "span-final-2"

    async def gen_think():
        # First chunk only carries span id
        yield make_evt(span_id=think_span_id)
        # Then stream reasoning
        yield make_evt(delta="All ", json_path="reasoning")
        yield make_evt(delta="good", json_path="reasoning")

    async def gen_final():
        yield make_evt(span_id=final_span_id)
        yield make_evt(delta="ok", json_path="text")

    async def stream_async(name: str, **kwargs):
        class Resp:
            def __init__(self, result):
                self.result = result
                # Simulate no direct usage on streaming response
                self.usage = None

        # Accept dynamic function names: think_{agent_name}, generate_final_result_{agent_name}
        if name.startswith("think_"):
            return Resp(gen_think())
        elif name.startswith("generate_final_result_"):
            return Resp(gen_final())
        else:
            raise AssertionError(f"unexpected stream call name: {name}")

    mock_opper_client.stream_async = AsyncMock(side_effect=stream_async)

    # Mock spans.get_async -> returns object with trace_id
    # Called for: usage tracking (think), span rename (think), usage tracking (final)
    mock_opper_client.spans.get_async = AsyncMock(
        side_effect=[
            AsyncMock(trace_id="trace-1"),  # usage tracking for think
            AsyncMock(trace_id="trace-1"),  # span rename check for think
            AsyncMock(trace_id="trace-1"),  # usage tracking for final_result
        ]
    )

    # Mock traces.get_async -> returns object with spans including data.total_tokens
    class SpanData:
        def __init__(self, total_tokens):
            self.total_tokens = total_tokens

    class Span:
        def __init__(self, id, total_tokens):
            self.id = id
            self.data = SpanData(total_tokens)

    mock_opper_client.traces.get_async = AsyncMock(
        side_effect=[
            AsyncMock(spans=[Span(think_span_id, 37)]),
            AsyncMock(spans=[Span(final_span_id, 21)]),
        ]
    )

    agent = Agent(
        name="StreamUsage",
        instructions="Test",
        output_schema=Out,
        enable_streaming=True,
        opper_api_key=opper_api_key,
    )

    out = await agent.process("go")
    assert isinstance(out, Out)
    assert agent.context.usage.requests == 2  # think + final_result
    assert agent.context.usage.total_tokens == 58  # 37 + 21
