# Other Examples

Additional examples demonstrating various features of the Opper Agent SDK.

## Logging Examples

### logging_comparison.py

Demonstrates the three built-in logging options:

1. **SimpleLogger** (default with `verbose=True`)
   - Plain text output with basic formatting
   - Good for simple debugging
   - Minimal overhead

2. **RichLogger** (fancy console output with Opper colors)
   - Beautiful console output with Opper brand colors
   - Spinner while thinking
   - Formatted tool calls and results
   - Color-coded success/error messages

3. **Silent mode** (`verbose=False`)
   - No logging output
   - Best for production when you have external logging

**Run it:**
```bash
uv run examples/other_examples/logging_comparison.py
```

### custom_logger.py

Shows how to create custom loggers by extending `AgentLogger`:

1. **FileLogger** - writes logs to a file
2. **MetricsLogger** - collects execution metrics without printing

Use cases for custom loggers:
- Writing to log files
- Sending to monitoring services (Datadog, New Relic, etc.)
- Storing structured logs in JSON
- Tracking performance metrics
- Custom formatting for your team's needs

**Run it:**
```bash
uv run examples/other_examples/custom_logger.py
```

## Creating Your Own Logger

To create a custom logger, extend `AgentLogger` and implement these methods:

```python
from opper_agent import AgentLogger

class MyLogger(AgentLogger):
    def log_iteration(self, iteration: int, max_iterations: int):
        # Called at the start of each iteration
        pass

    def log_thinking(self):
        # Context manager for thinking phase
        yield

    def log_thought(self, reasoning: str, tool_count: int):
        # Called after the agent decides what to do
        pass

    def log_tool_call(self, tool_name: str, parameters: dict):
        # Called before executing a tool
        pass

    def log_tool_result(self, tool_name: str, success: bool, result: Any, error: str):
        # Called after tool execution
        pass

    def log_memory_read(self, keys: List[str]):
        # Called when reading from memory
        pass

    def log_memory_loaded(self, data: dict):
        # Called after memory is loaded
        pass

    def log_memory_write(self, keys: List[str]):
        # Called when writing to memory
        pass

    def log_final_result(self):
        # Called before generating final result
        pass

    def log_warning(self, message: str):
        # Called for warnings
        pass

    def log_error(self, message: str):
        # Called for errors
        pass
```

Then use it:

```python
agent = Agent(
    name="MyAgent",
    logger=MyLogger(),
    tools=[...],
)
```
