# Agents Code Instructions for Opper Agent SDK Rebuild

## Context
You are helping maintain and extend the Opper Agent SDK - a Python framework for building AI agents with clean OOP architecture, proper separation of concerns, and extensibility.

## Key Documents
- `SPEC.md` - Architecture overview, component descriptions, and design decisions
- `.cursor/rules/opper.mdc` - Opper SDK/API reference
- **Always reference these documents** when working on the codebase

## Code Quality Standards

### 1. Architecture
- **Class hierarchy**: `BaseAgent` (abstract) → `Agent`, `ReactAgent`, `ChatAgent`, custom agents
- **Type safety**: Pydantic models for all data structures
- **Type hints**: Required on all functions/methods
- **Async first**: All agent operations are async
- **Single responsibility**: Keep methods focused (~30-40 lines max)

### 2. Opper Integration
- Use `await self.opper.call()` for LLM calls (always async)
- Create spans for traceability: `self.opper.spans.create()`
- Track usage: `self.context.update_usage(usage)`
- Parent spans properly: pass `parent_span_id`
- Use structured outputs: `output_schema=SomePydanticModel`
- **Check `.cursor/rules/opper.mdc` for Opper API details**

### 3. Hook System
- Trigger hooks at all lifecycle points (see `HookEvents`)
- Hooks never break execution (wrap in try/except)
- Support both sync and async hooks
- Log failures as warnings

### 4. Error Handling
- **Agent-level**: Graceful degradation, max iterations protection
- **Tool-level**: Return `ToolResult` with `success=False` and error message
- **Hook-level**: Catch, log, continue
- **Memory-level**: Degrade gracefully

## Development Workflow

### Environment Setup
This project uses **UV** for package management.

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov

# Format code (REQUIRED before committing)
uvx ruff format

# Lint code
uvx ruff check
```

### Workflow
1. Make changes
2. Write/update tests
3. Run tests: `uv run pytest`
4. **Format code: `uvx ruff format`** (required)
5. Check linting: `uvx ruff check`

## Testing Rules (Critical)

### ❌ NEVER Do This
1. **NEVER remove tests without asking** - Always get permission first
2. **NEVER remove assertions to make tests pass** - Fix the code, not the test
3. **NEVER skip `uvx ruff format`** - Required before committing
4. **NEVER write scripts instead of tests** - Write proper pytest tests

### ✅ ALWAYS Do This
1. **ALWAYS put tests in `tests/`** - Organized by unit/integration/e2e
2. **ALWAYS use `mock_acompletion`** - For mocking LLM responses
3. **ALWAYS run `uv run pytest`** - Before pushing
4. **ALWAYS format with `uvx ruff format`** - Last step before committing

### Test Organization
```
tests/
├── unit/              # Unit tests (isolated, no external calls)
├── integration/       # Integration tests (with mocked HTTP)
└── e2e/              # End-to-end tests (real API calls)
```

### Testing Patterns

**Unit Test (No API Calls)**
```python
import pytest
from opper_agent.base.context import AgentContext, Usage

@pytest.mark.asyncio
async def test_context_usage_tracking():
    ctx = AgentContext(agent_name="Test")
    ctx.update_usage(Usage(requests=1, total_tokens=100))
    assert ctx.usage.requests == 1
```

**Integration Test (Mock LLM)**
```python
@pytest.mark.asyncio
async def test_agent_with_tools(mock_acompletion):
    mock_acompletion.return_value = AsyncMock(
        json_payload={"reasoning": "...", "tool_calls": [...]}
    )

    agent = Agent(name="Test", tools=[my_tool])
    result = await agent.process("test input")

    assert result is not None
    assert mock_acompletion.called
```

## Common Patterns

### Tool Definition
```python
@tool
def search_web(query: str, max_results: int = 10) -> dict:
    """
    Search the web for a query.

    Args:
        query: Search query string
        max_results: Maximum number of results

    Returns:
        Search results dictionary
    """
    return {"results": [...]}
```

### Hook Definition
```python
@hook("agent_start")
async def log_start(context: AgentContext, agent: BaseAgent):
    """Log when agent starts."""
    print(f"Starting {agent.name}")
```

### Agent Creation
```python
agent = Agent(
    name="ResearchAgent",
    description="Researches topics",
    instructions="Be thorough and cite sources",
    tools=[search_tool, summarize_tool],
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

## File Organization

```
src/opper_agent/
├── __init__.py              # Main exports
├── base/                    # Core abstractions (BaseAgent, Context, Tool, Hooks)
├── core/                    # Main implementations (Agent, schemas)
├── agents/                  # Specialized agents (ReactAgent, ChatAgent)
├── memory/                  # Memory system
├── mcp/                     # MCP integration (provider, client, config)
└── utils/                   # Utilities (decorators, logging)
```

## Questions to Ask Before Implementing

- Does this follow the `BaseAgent` contract?
- Will this work with custom agent subclasses?
- Is error handling appropriate?
- Are hooks triggered correctly?
- Is this testable in isolation?
- Does this match the architecture in `SPEC.md`?

## Priority Order

When making tradeoffs:

1. **Correctness** - Must work reliably
2. **Simplicity** - Easy to understand and debug
3. **Extensibility** - Easy to build custom agents
4. **Performance** - Fast enough (optimize only if needed)

## Code Style

- Pydantic models for all data structures
- Type hints everywhere
- Async by default
- Google-style docstrings
- Single responsibility principle
- No emojis unless explicitly requested

## Success Criteria

Before marking work complete:

- [ ] Tests pass: `uv run pytest --cov`
- [ ] Type checking passes: `uv run mypy src/`
- [ ] Linting passes: `uvx ruff check`
- [ ] **Code formatted: `uvx ruff format`** (required)
- [ ] Documentation updated if APIs changed
