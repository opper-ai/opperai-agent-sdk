import pytest
from unittest.mock import AsyncMock
from pydantic import BaseModel, Field

from opper_agents.agents.react import ReactAgent
from opper_agents.base.hooks import HookEvents
from opper_agents.utils.decorators import hook


class FinalOut(BaseModel):
    summary: str = Field(description="Final summary")


def make_stream_event(delta=None, json_path=None, chunk_type="json", span_id=None):
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
async def test_react_agent_streaming(mock_opper_client, opper_api_key):
    async def gen_reason():
        # Provide a full ReactThought via streaming
        yield make_stream_event(delta="Analyzing situation", json_path="reasoning")
        yield make_stream_event(delta="true", json_path="is_complete")

    async def gen_final():
        yield make_stream_event(delta="Done", json_path="summary")

    async def stream_async(name: str, **kwargs):
        class Response:
            def __init__(self, result):
                self.result = result
                self.usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 1}

        # Function names are now dynamic: reason_{agent_name}, generate_final_result_{agent_name}
        if name == "reason_reactstream":
            return Response(gen_reason())
        elif name == "generate_final_result_reactstream":
            return Response(gen_final())
        else:
            raise AssertionError(f"Unexpected stream call: {name}")

    mock_opper_client.stream_async = AsyncMock(side_effect=stream_async)

    reason_chunks = []
    final_chunks = []
    parsed_reason = []

    @hook(HookEvents.STREAM_CHUNK)
    async def on_chunk(
        context, agent, call_type, chunk_data, accumulated, field_buffers, **kwargs
    ):
        if call_type == "reason":
            reason_chunks.append(accumulated)
        if call_type == "final_result":
            final_chunks.append(accumulated)

    @hook(HookEvents.LLM_RESPONSE)
    async def on_llm_response(context, agent, call_type, parsed=None, **kwargs):
        if call_type == "reason" and parsed is not None:
            parsed_reason.append(parsed)

    agent = ReactAgent(
        name="ReActStream",
        instructions="Test",
        output_schema=FinalOut,
        enable_streaming=True,
        hooks=[on_chunk, on_llm_response],
        opper_api_key=opper_api_key,
    )

    result = await agent.process("task")

    assert result.summary == "Done"
    assert parsed_reason and parsed_reason[0].is_complete is True
    assert reason_chunks and any(
        c.endswith("Analyzing situation") for c in reason_chunks
    )
    assert final_chunks and final_chunks[-1] == "Done"

    # Ensure non-streaming call not used
    assert not mock_opper_client.call_async.called
