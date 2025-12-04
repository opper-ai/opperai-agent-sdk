import pytest
from unittest.mock import AsyncMock
from pydantic import BaseModel, Field

from opper_agents.core.agent import Agent
from opper_agents.base.hooks import HookEvents
from opper_agents.utils.decorators import hook


class OutModel(BaseModel):
    value: str = Field(description="output value")


def _evt(delta=None, json_path=None, chunk_type="json", span_id=None):
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
async def test_stream_error_hook_on_exception(mock_opper_client, opper_api_key):
    # Make stream_async raise during think phase
    async def stream_async(name: str, **kwargs):
        raise RuntimeError("boom")

    mock_opper_client.stream_async = AsyncMock(side_effect=stream_async)

    errors = []

    @hook(HookEvents.STREAM_ERROR)
    async def on_stream_error(context, agent, call_type, error, **kwargs):
        errors.append((call_type, str(error)))

    agent = Agent(
        name="ErrStream",
        enable_streaming=True,
        output_schema=OutModel,
        hooks=[on_stream_error],
        opper_api_key=opper_api_key,
    )

    with pytest.raises(RuntimeError):
        await agent.process("x")

    # Ensure error hook fired for think
    assert errors and errors[0][0] in ("think", "final_result")
    assert "boom" in errors[0][1]


@pytest.mark.asyncio
async def test_llm_response_includes_response_and_parsed(
    mock_opper_client, opper_api_key
):
    # Prepare streams for think and final_result
    async def gen_think():
        # Only reasoning; Thought schema fills defaults
        yield _evt(delta="Reasoning", json_path="reasoning")

    async def gen_final():
        # Output schema field
        yield _evt(delta="ok", json_path="value")

    async def stream_async(name: str, **kwargs):
        class Response:
            def __init__(self, result, marker):
                self.result = result
                self.usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 1}
                self.marker = marker

        # Accept dynamic function names: think_{agent_name}, generate_final_result_{agent_name}
        if name.startswith("think_"):
            return Response(gen_think(), marker="think_resp")
        elif name.startswith("generate_final_result_"):
            return Response(gen_final(), marker="final_resp")
        else:
            raise AssertionError(f"unexpected stream name: {name}")

    mock_opper_client.stream_async = AsyncMock(side_effect=stream_async)

    seen = {"think": None, "final_result": None}

    @hook(HookEvents.LLM_RESPONSE)
    async def on_llm_response(
        context, agent, call_type, response=None, parsed=None, **kwargs
    ):
        # Ensure both response and parsed are provided
        assert response is not None and parsed is not None
        seen[call_type] = (getattr(response, "marker", None), parsed)

    agent = Agent(
        name="PayloadStream",
        enable_streaming=True,
        output_schema=OutModel,
        hooks=[on_llm_response],
        opper_api_key=opper_api_key,
    )

    out = await agent.process("task")
    assert isinstance(out, OutModel)
    assert out.value == "ok"

    # Think hook contains our marker and a Thought-like parsed object
    assert seen["think"][0] == "think_resp"
    assert hasattr(seen["think"][1], "reasoning")

    # Final result hook contains our marker and the parsed OutModel
    assert seen["final_result"][0] == "final_resp"
    assert isinstance(seen["final_result"][1], OutModel)
