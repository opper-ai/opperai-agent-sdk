import pytest
from unittest.mock import AsyncMock
from pydantic import BaseModel, Field

from opper_agents.core.agent import Agent
from opper_agents.base.hooks import HookEvents
from opper_agents.utils.decorators import hook


class TestOutput(BaseModel):
    text: str = Field(description="Final output text")


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
async def test_agent_streaming_flow(mock_opper_client, opper_api_key):
    # Arrange streaming responses for think and final result
    async def gen_think():
        # Stream a simple Thought with only reasoning and no tool_calls
        yield make_stream_event(delta="All ", json_path="reasoning")
        yield make_stream_event(delta="done", json_path="reasoning")

    async def gen_final():
        # Stream the final output schema field
        yield make_stream_event(delta="Hello", json_path="text")
        yield make_stream_event(delta=" ", json_path="text")
        yield make_stream_event(delta="world", json_path="text")

    async def stream_async(name: str, **kwargs):
        class Response:
            def __init__(self, result):
                self.result = result
                self.usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 1}

        # Function names are now dynamic: think_{agent_name}, generate_final_result_{agent_name}
        if name == "think_streamtest":
            return Response(gen_think())
        elif name == "generate_final_result_streamtest":
            return Response(gen_final())
        else:
            raise AssertionError(f"Unexpected stream call name: {name}")

    mock_opper_client.stream_async = AsyncMock(side_effect=stream_async)

    # Capture hooks
    think_chunks = []
    final_chunks = []
    llm_parsed_think = []
    llm_parsed_final = []

    @hook(HookEvents.STREAM_CHUNK)
    async def on_stream_chunk(
        context, agent, call_type, chunk_data, accumulated, field_buffers, **kwargs
    ):
        if call_type == "think":
            think_chunks.append(accumulated)
        elif call_type == "final_result":
            final_chunks.append(accumulated)

    @hook(HookEvents.LLM_RESPONSE)
    async def on_llm_response(
        context, agent, call_type, response=None, parsed=None, **kwargs
    ):
        if call_type == "think" and parsed is not None:
            llm_parsed_think.append(parsed)
        if call_type == "final_result" and parsed is not None:
            llm_parsed_final.append(parsed)

    agent = Agent(
        name="StreamTest",
        instructions="Test",
        output_schema=TestOutput,
        enable_streaming=True,
        hooks=[on_stream_chunk, on_llm_response],
        opper_api_key=opper_api_key,
    )

    # Act
    result = await agent.process("do it")

    # Assert
    assert isinstance(result, TestOutput)
    assert result.text == "Hello world"

    # Hooks captured
    assert think_chunks, "Expected STREAM_CHUNK events during think"
    assert final_chunks, "Expected STREAM_CHUNK events during final_result"
    assert think_chunks[-1].endswith("done")
    assert final_chunks[-1].endswith("world")

    # LLM_RESPONSE carried parsed values
    assert llm_parsed_think and llm_parsed_think[0].reasoning.startswith("All ")
    assert llm_parsed_final and isinstance(llm_parsed_final[0], TestOutput)

    # Streaming path should avoid non-streaming call_async
    assert not mock_opper_client.call_async.called
