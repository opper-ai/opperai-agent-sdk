"""
Concurrency stress test for as_tool() usage propagation.

Verifies that _usage_lock correctly isolates usage across concurrent calls
to the same agent-as-tool, preventing cross-contamination of _last_run_usage.
"""

import asyncio
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from opper_agents.base.agent import BaseAgent
from opper_agents.base.context import RunResult, Usage
from opper_agents.base.tool import ToolResult


class StubAgent(BaseAgent):
    """Minimal agent that returns distinguishable usage per call."""

    def __init__(self, delay: float = 0.01) -> None:
        self.name = "StubAgent"
        self.description = "stub"
        self.instructions = None
        self.input_schema = None
        self.output_schema = None
        self.max_iterations = 1
        self.verbose = False
        self.logger = None
        self.model = "test"
        self.enable_streaming = False
        self.agent_tool_timeout = None
        self.parallel_tool_execution = False
        self.base_tools = []
        self.tool_providers = []
        self.active_provider_tools = {}
        self.tools = []
        self.context = None
        self._call_counter = 0
        self._counter_lock = asyncio.Lock()
        self._delay = delay

        # Minimal hook manager
        self.hook_manager = MagicMock()
        self.hook_manager.trigger = AsyncMock()
        self.hook_manager.hooks = {}
        self.hook_manager.get_hook_count = MagicMock(return_value=0)

        # Minimal opper mock
        self.opper = MagicMock()

    async def process(self, input: Any, _parent_span_id: Optional[str] = None) -> Any:
        return "result"

    async def _run_loop(self, goal: Any) -> Any:
        return "result"

    async def run(self, input: Any, _parent_span_id: Optional[str] = None) -> RunResult:
        # Assign a unique call index under lock
        async with self._counter_lock:
            self._call_counter += 1
            call_index = self._call_counter

        # Simulate work with a small delay to force interleaving
        await asyncio.sleep(self._delay)

        # Return unique usage per call: requests = call_index * 100
        usage = Usage(
            requests=call_index * 100,
            input_tokens=call_index * 10,
            output_tokens=call_index * 5,
            total_tokens=call_index * 15,
        )
        return RunResult(result=f"result_{call_index}", usage=usage)


@pytest.mark.asyncio
async def test_as_tool_concurrent_usage_isolation() -> None:
    """
    Fire N concurrent execute() calls on the same agent-as-tool and verify:
    1. All N results have non-None usage
    2. All N usage.requests values are unique (no cross-contamination)
    """
    agent = StubAgent(delay=0.01)
    tool = agent.as_tool()

    n = 10

    # Fire N concurrent calls
    tasks = [tool.execute(task=f"task_{i}", _parent_span_id=None) for i in range(n)]
    results: list[ToolResult] = await asyncio.gather(*tasks)

    # All results should have non-None usage
    for i, result in enumerate(results):
        assert result.usage is not None, f"Result {i} has None usage"

    # All usage.requests values should be unique (no cross-contamination)
    request_counts = [r.usage.requests for r in results]
    assert len(set(request_counts)) == n, (
        f"Expected {n} unique request counts, got {len(set(request_counts))}: "
        f"{request_counts}"
    )

    # Each request count should be a multiple of 100 (call_index * 100)
    for count in request_counts:
        assert count % 100 == 0, f"Unexpected request count: {count}"
        assert 100 <= count <= n * 100, f"Request count out of range: {count}"
