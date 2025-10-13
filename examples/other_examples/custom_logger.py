"""
Example: Creating a custom logger.

This example shows how to build a custom logger by extending AgentLogger.
You can customize logging to:
- Write to files
- Send to logging services
- Format output in your own style
- Add custom metrics/monitoring
"""

import asyncio
from datetime import datetime
from contextlib import contextmanager
from typing import Any, Optional, Dict, List
from opper_agents import Agent, tool, AgentLogger


class FileLogger(AgentLogger):
    """
    Custom logger that writes to a file.

    This is a simple example - you could extend this to:
    - Use Python's logging module
    - Send logs to a monitoring service (Datadog, New Relic, etc.)
    - Store structured logs in JSON format
    - Add performance metrics
    """

    def __init__(self, filename: str = "agent_log.txt"):
        self.filename = filename
        # Clear the file on init
        with open(self.filename, "w") as f:
            f.write(f"=== Agent Log Started at {datetime.now()} ===\n\n")

    def _write(self, message: str) -> None:
        """Write a message to the log file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.filename, "a") as f:
            f.write(f"[{timestamp}] {message}\n")

    def log_iteration(self, iteration: int, max_iterations: int) -> None:
        self._write(f">>> Iteration {iteration}/{max_iterations}")

    @contextmanager
    def log_thinking(self) -> Any:
        self._write("Thinking...")
        yield
        self._write("Thought complete.")

    def log_thought(self, reasoning: str, tool_count: int) -> None:
        self._write(f"Reasoning: {reasoning}")
        self._write(f"Planned tool calls: {tool_count}")

    def log_tool_call(self, tool_name: str, parameters: Dict[str, Any]) -> None:
        params_str = ", ".join(f"{k}={v}" for k, v in parameters.items())
        self._write(f"Calling tool: {tool_name}({params_str})")

    def log_tool_result(
        self, tool_name: str, success: bool, result: Any, error: Optional[str] = None
    ) -> None:
        if success:
            self._write(f"Tool result: SUCCESS - {result}")
        else:
            self._write(f"Tool result: FAILED - {error}")

    def log_memory_read(self, keys: List[str]) -> None:
        self._write(f"Reading memory keys: {keys}")

    def log_memory_loaded(self, data: Dict[str, Any]) -> None:
        self._write(f"Loaded memory: {data}")

    def log_memory_write(self, keys: List[str]) -> None:
        self._write(f"Writing to memory: {keys}")

    def log_final_result(self) -> None:
        self._write("Generating final result...")

    def log_warning(self, message: str) -> None:
        self._write(f"WARNING: {message}")

    def log_error(self, message: str) -> None:
        self._write(f"ERROR: {message}")


class MetricsLogger(AgentLogger):
    """
    Custom logger that tracks metrics without printing.

    Useful for production environments where you want to:
    - Track performance metrics
    - Monitor tool usage
    - Measure iteration counts
    - Send data to analytics services
    """

    def __init__(self) -> None:
        self.metrics = {
            "iterations": 0,
            "tool_calls": 0,
            "successful_tools": 0,
            "failed_tools": 0,
            "memory_reads": 0,
            "memory_writes": 0,
        }

    def log_iteration(self, iteration: int, max_iterations: int) -> None:
        self.metrics["iterations"] = iteration

    @contextmanager
    def log_thinking(self) -> Any:
        yield

    def log_thought(self, reasoning: str, tool_count: int) -> None:
        pass  # Could track reasoning length, sentiment, etc.

    def log_tool_call(self, tool_name: str, parameters: Dict[str, Any]) -> None:
        self.metrics["tool_calls"] += 1

    def log_tool_result(
        self, tool_name: str, success: bool, result: Any, error: Optional[str] = None
    ) -> None:
        if success:
            self.metrics["successful_tools"] += 1
        else:
            self.metrics["failed_tools"] += 1

    def log_memory_read(self, keys: List[str]) -> None:
        self.metrics["memory_reads"] += len(keys)

    def log_memory_loaded(self, data: Dict[str, Any]) -> None:
        pass

    def log_memory_write(self, keys: List[str]) -> None:
        self.metrics["memory_writes"] += len(keys)

    def log_final_result(self) -> None:
        pass

    def log_warning(self, message: str) -> None:
        pass

    def log_error(self, message: str) -> None:
        pass

    def get_metrics(self) -> Dict[str, int]:
        """Get collected metrics."""
        return self.metrics.copy()


@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


async def main() -> None:
    print("=" * 80)
    print("EXAMPLE 1: FILE LOGGER")
    print("=" * 80)

    file_logger = FileLogger("agent_execution.log")

    agent_file = Agent(
        name="MathAgent",
        description="Agent that performs math operations",
        tools=[add, multiply],
        logger=file_logger,
    )

    result = await agent_file.process("Calculate (5 + 3) * 2")
    print(f"Result: {result}")
    print(f"\nLog written to: {file_logger.filename}")
    print("\nLog contents:")
    with open(file_logger.filename, "r") as f:
        print(f.read())

    print("\n" + "=" * 80)
    print("EXAMPLE 2: METRICS LOGGER")
    print("=" * 80)

    metrics_logger = MetricsLogger()

    agent_metrics = Agent(
        name="MathAgent",
        description="Agent that performs math operations",
        tools=[add, multiply],
        logger=metrics_logger,
    )

    result = await agent_metrics.process("Calculate 10 + 20, then multiply by 3")
    print(f"Result: {result}")
    print("\nCollected metrics:")
    for key, value in metrics_logger.get_metrics().items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
