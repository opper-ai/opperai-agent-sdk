# Claude Code Instructions for Opper Agent SDK Rebuild

## Context
You are helping rebuild the Opper Agent SDK with clean OOP architecture, proper separation of concerns, and extensibility. This is a ground-up rebuild focused on maintainability, testing, and developer experience.

## Key Documents
- `IMPLEMENTATION_PLAN.md` - Detailed step-by-step implementation with complete code examples
- `REBUILD_SPEC.md` - High-level architecture overview and design decisions
- `.cursor/rules/opper.mdc` - **Opper SDK/API reference** - When in doubt about how Opper works, check this file
- **Always reference these documents** before implementing features

## Development Principles

### 1. Code Quality Standards
- **Follow the class hierarchy strictly**: `BaseAgent` (abstract) → `Agent`, `ReactAgent`, custom agents
- **Use Pydantic models** for all data structures (context, schemas, configurations)
- **Type hints everywhere** - no `Any` unless absolutely necessary
- **Keep methods focused** - single responsibility principle (max ~30-40 lines per method)
- **Async by default** - all agent operations are async, including `opper.call()`
- No emojis

### 2. Testing First Approach
- Write tests before or alongside implementation (TDD preferred)
- Target **>90% coverage** across the codebase
- Test both success and failure paths explicitly
- Use `mock_acompletion` to mock LLM responses (very convenient)
- Use pytest with asyncio support and VCR for HTTP mocking
- **NEVER remove tests without asking**
- **NEVER remove test assertions to make tests pass**
- Write proper tests, not scripts when trying things out
- Tests live in `tests/` folder - put them in the appropriate subfolder

### 3. Opper Integration Patterns
- Use `await self.opper.call()` for all LLM calls (**async**)
- Create spans for traceability: `self.opper.spans.create()`
- Always track token usage: `self.context.update_usage(usage)`
- Parent spans properly: pass `parent_span_id` to nested calls
- Use structured outputs: pass `output_schema=SomePydanticModel`
- **When in doubt about Opper SDK/API**, check `.cursor/rules/opper.mdc`

### 4. Hook System Rules
- Trigger hooks at **all** lifecycle points (see `HookEvents` class)
- Hooks should **never** break execution - wrap in try/except
- Support both sync and async hooks (check with `asyncio.iscoroutinefunction`)
- Log hook failures as warnings, never raise

### 5. Error Handling Strategy
- **Agent-level**: Retry LLM calls with exponential backoff
- **Tool-level**: Return `ToolResult` with `success=False` and error message
- **Hook-level**: Catch, log, continue (never break flow)
- **Memory-level**: Degrade gracefully (continue without memory if it fails)
- See Phase 6 in `REBUILD_SPEC.md` for comprehensive strategy

## Environment Setup

This project uses **UV** for fast, reliable Python package management.

### Initial Setup
```bash
# Install dependencies and create virtual environment
uv sync
```

### Testing and Code Quality
```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov

# Format code (ALWAYS run this at the end of your changes)
uv run ruff format

# Lint code
uv run ruff check
```

### Development Workflow
1. Make changes
2. Write/update tests
3. Run tests: `uv run pytest`
4. Format code: `uv run ruff format` (**required before committing**)
5. Check linting: `uv run ruff check`

## Implementation Checklist

### Before Writing Code
- [ ] Read the relevant section in `IMPLEMENTATION_PLAN.md` completely
- [ ] Understand how it fits into the overall architecture
- [ ] Check if tests are specified in the plan
- [ ] Review related code already implemented

### While Writing Code
- [ ] Follow the **exact** structure and signatures from the plan
- [ ] Add comprehensive docstrings (Google style)
- [ ] Use the specified imports (don't add unnecessary ones)
- [ ] Keep `self.verbose` checks for optional logging
- [ ] Add type hints to all parameters and return values

### After Writing Code
- [ ] Write/run the specified tests (`uv run pytest`)
- [ ] Verify all imports resolve correctly
- [ ] Check integration with other components
- [ ] Update examples if the API changed
- [ ] **Run formatter: `uv run ruff format`** (required)
- [ ] Run linter: `uv run ruff check`

## Common Patterns

### Tool Definition
```python
@tool
def search_web(query: str, max_results: int = 10) -> dict:
    """
    Search the web for a query.

    Args:
        query: Search query string
        max_results: Maximum number of results to return

    Returns:
        Dictionary with search results
    """
    return {"results": [...]}
```

### Hook Definition
```python
@hook("agent_start")
async def log_agent_start(context: AgentContext, agent: BaseAgent):
    """Log when agent starts execution."""
    print(f"Starting agent: {agent.name}")
    context.metadata["start_time"] = time.time()
```

### Agent Creation
```python
agent = Agent(
    name="ResearchAgent",
    description="Agent that researches topics",
    instructions="Be thorough and cite sources",
    tools=[search_tool, summarize_tool, mcp(mcp_config)],
    input_schema=ResearchRequest,
    output_schema=ResearchResult,
    max_iterations=25,
    verbose=True,
    enable_memory=True
)
```

### Opper Call Pattern
```python
response = await self.opper.call(
    name="think",
    instructions="Analyze and decide next action",
    input=context_dict,
    output_schema=Thought,
    model=self.model,
    parent_span_id=self.context.trace_id
)
```

## Testing Patterns

This project uses **pytest** with asyncio support and **VCR** for HTTP mocking. Always use `mock_acompletion` for mocking LLM responses.

### Test Organization
```
tests/
├── unit/              # Unit tests (isolated, no external calls)
├── integration/       # Integration tests (with mocked HTTP)
└── e2e/              # End-to-end tests (real API calls)
```

### Unit Test Example (No API Calls)
```python
import pytest
from opper_agent.base.context import AgentContext, Usage

@pytest.mark.asyncio
async def test_agent_context_usage_tracking():
    """Test usage tracking without any API calls."""
    ctx = AgentContext(agent_name="Test")
    usage1 = Usage(requests=1, total_tokens=100)
    usage2 = Usage(requests=1, total_tokens=200)

    ctx.update_usage(usage1)
    ctx.update_usage(usage2)

    assert ctx.usage.requests == 2
    assert ctx.usage.total_tokens == 300
```

### Integration Test with mock_acompletion
```python
import pytest
from unittest.mock import AsyncMock
from opper_agent.core.agent import Agent
from opper_agent.utils.decorators import tool

@pytest.mark.asyncio
async def test_agent_with_tools(mock_acompletion):
    """Test agent execution with mocked LLM responses."""
    # Mock the LLM response
    mock_acompletion.return_value = AsyncMock(
        json_payload={
            "reasoning": "I need to add 5 and 3",
            "tool_calls": [
                {"name": "add", "parameters": {"a": 5, "b": 3}, "reasoning": "Adding numbers"}
            ],
            "user_message": "Calculating..."
        }
    )

    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    agent = Agent(
        name="MathAgent",
        tools=[add],
        opper_api_key="test-key"
    )

    result = await agent.process("What is 5 + 3?")
    assert result is not None

    # Verify the mock was called
    assert mock_acompletion.called
```

### VCR for HTTP Mocking
```python
import pytest
from opper_agent.core.agent import Agent

@pytest.mark.vcr  # Decorator to record/replay HTTP interactions
@pytest.mark.asyncio
async def test_agent_with_real_opper_api(vcr):
    """Test with VCR recording actual HTTP calls."""
    agent = Agent(
        name="TestAgent",
        tools=[...],
        opper_api_key=os.getenv("OPPER_API_KEY")
    )

    # First run records HTTP interactions
    # Subsequent runs replay from cassette
    result = await agent.process("test task")
    assert result is not None
```

### Testing Error Handling
```python
import pytest
from opper_agent.base.tool import FunctionTool

@pytest.mark.asyncio
async def test_tool_execution_error():
    """Test that tool errors are handled gracefully."""
    def failing_tool():
        raise ValueError("Intentional failure")

    tool = FunctionTool(failing_tool)
    result = await tool.execute()

    # Tool should return ToolResult with success=False
    assert result.success is False
    assert "Intentional failure" in result.error
    assert result.execution_time >= 0
```

## Testing Rules (CRITICAL)

**These rules are non-negotiable:**

### ❌ NEVER Do This
1. **NEVER remove tests without explicit permission** - Always ask first
2. **NEVER remove test assertions to make tests pass** - Fix the code, not the test
3. **NEVER skip running `uv run ruff format`** - Required before committing
4. **NEVER write scripts instead of proper tests** - If you're testing something, write a pytest test

### ✅ ALWAYS Do This
1. **ALWAYS put tests in the `tests/` folder** - Organized by type (unit/integration/e2e)
2. **ALWAYS use `mock_acompletion` for mocking LLM responses** - It's very convenient
3. **ALWAYS run `uv run pytest` before pushing** - Ensure tests pass
4. **ALWAYS format with `uv run ruff format`** - Last step before committing

### Test Quality Standards
- Tests must be isolated (unit tests shouldn't make external calls)
- Tests must be reproducible (same input → same output)
- Tests must have clear assertions (what exactly are you testing?)
- Tests must have descriptive names (`test_agent_handles_tool_errors` not `test_1`)


## Questions to Ask Before Implementing

- Does this follow the `BaseAgent` contract?
- Will this work with custom agent subclasses?
- Is error handling appropriate for this layer?
- Are hooks triggered at the right points?
- Is this code testable in isolation?
- Does this match the spec in `IMPLEMENTATION_PLAN.md`?

## Priority Order

When making tradeoffs, follow this priority:

1. **Correctness** - Must work reliably and predictably
2. **Simplicity** - Easy to understand, use, and debug
3. **Extensibility** - Easy to build custom agents and tools
4. **Performance** - Fast enough (optimize only if needed)



## File Organization

When creating new files, follow this structure:

```
src/opper_agent/
├── __init__.py              # Main exports (Agent, tool, hook, etc.)
├── base/                    # Core abstractions
│   ├── agent.py            # BaseAgent
│   ├── context.py          # AgentContext, Usage, ExecutionCycle
│   ├── tool.py             # Tool, FunctionTool, ToolResult
│   └── hooks.py            # HookManager, HookEvents
├── core/                    # Main implementations
│   ├── agent.py            # Agent (while tools > 0)
│   └── schemas.py          # Thought, ToolCall, MemoryDecision
├── agents/                  # Specialized agent types
│   ├── react.py            # ReactAgent
│   └── chat.py             # ChatAgent
├── memory/                  # Memory system
│   └── memory.py           # Memory, MemoryEntry
├── mcp/                     # MCP integration
│   ├── provider.py         # MCPToolProvider
│   ├── client.py           # MCP clients (stdio, http-sse)
│   └── config.py           # MCPServerConfig
└── utils/                   # Utilities
    ├── decorators.py       # @tool, @hook
    └── opper.py            # Opper helpers
```

## Success Criteria

Before marking a phase complete:

- [ ] All tasks in `IMPLEMENTATION_PLAN.md` for that phase are done
- [ ] Unit tests pass: `uv run pytest --cov` (>90% coverage)
- [ ] Integration tests pass (if applicable)
- [ ] Type checking passes: `uv run mypy src/`
- [ ] Linting passes: `uv run ruff check`
- [ ] Code formatted: `uv run ruff format` (**required**)
- [ ] Documentation is updated
