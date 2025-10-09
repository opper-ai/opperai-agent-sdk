# Opper Agent SDK - Architecture & Implementation

## Overview

The Opper Agent SDK is a Python framework for building AI agents with clean OOP architecture, proper separation of concerns, and extensibility. It provides a reliable foundation for creating agents that use tools-based reasoning loops with dynamic tool selection.

## Core Architecture

### Design Principles

1. **Clean OOP Design**: Proper class hierarchies with clear separation of concerns
2. **Native Opper Integration**: Traces, spans, and calls handled elegantly
3. **Extensibility**: Easy to build custom agents through inheritance
4. **Simplicity**: Keep integrations simple and maintainable
5. **Type Safety**: Pydantic models throughout

### Class Hierarchy

```
BaseAgent (Abstract)
├── Agent (Main implementation - while tools>0 loop)
├── ReactAgent (ReAct pattern)
├── ChatAgent (Conversational agent)
└── [Custom Agents] (User-defined)
```

---

## Core Components

### Base Layer (`base/`)

#### `BaseAgent` (Abstract Base Class)
The foundation for all agents. Defines the contract that all agents must follow:

- **Lifecycle management**: Initialization, cleanup, process entry point
- **Opper integration**: Client management, traces, spans
- **Hook system**: Event triggering at key lifecycle points
- **Tool management**: Add tools, convert agents to tools
- **State management**: AgentContext for execution state

Key abstract methods:
- `async def process(input: Any) -> Any`: Main entry point
- `async def _run_loop(goal: Any) -> Any`: Core reasoning loop (agent-specific)

#### `AgentContext`
Maintains all state for an agent execution:

```python
class AgentContext(BaseModel):
    agent_name: str
    session_id: str
    trace_id: str
    iteration: int = 0
    goal: Optional[Any] = None
    execution_history: List[ExecutionCycle] = []
    memory: Optional[Memory] = None
    usage: Usage = Usage()
    metadata: Dict[str, Any] = {}
```

Tracks token usage, execution history, and provides clean state access.

#### `Tool` System
- `Tool`: Base class for all tools
- `FunctionTool`: Wraps Python functions as tools
- `ToolResult`: Standardized tool execution result
- `@tool` decorator: Converts functions to tools with automatic parameter extraction

#### `HookManager`
Manages lifecycle hooks for agents. Hooks never break execution - they're wrapped in try/except and logged on failure.

**Standard Hook Events:**
- `agent_start`: Before agent begins
- `agent_end`: After agent completes
- `agent_error`: When agent errors
- `loop_start`: Start of each iteration
- `loop_end`: End of each iteration
- `think_end`: After the think step returns a Thought
- `tool_call`: Before tool execution
- `tool_result`: After tool execution
- `llm_call`: Before LLM call
- `llm_response`: After LLM response

---

### Core Implementation (`core/`)

#### `Agent` Class
Main agent implementation using the "while tools > 0" loop pattern:

**Loop Logic:**
1. **Think**: Call LLM to decide next action (returns `Thought` with tool calls)
2. **If tool calls > 0**: Execute tools, add to history, continue loop
3. **If tool calls == 0**: Generate final response, exit loop

Key features:
- Token usage tracking
- Automatic span creation for traceability
- Memory integration (optional)
- Clean tool result generation
- Input/output schema validation

#### Schemas
- `Thought`: AI's reasoning and action plan with tool calls
- `ToolCall`: Single tool call with parameters and reasoning
- `ExecutionCycle`: One think-act cycle with results

---

### Memory System (`memory/`)

**LLM-Controlled Memory**: The LLM directly controls memory reads/writes through the `Thought` schema.

```python
class Memory(BaseModel):
    store: Dict[str, MemoryEntry] = Field(default_factory=dict)

    async def list_entries() -> List[Dict[str, Any]]  # Catalog for LLM
    async def read(keys: List[str]) -> Dict[str, Any]
    async def write(key: str, value: Any, ...) -> None
    async def clear() -> None
```

**Memory Integration:**
- Memory catalog provided to LLM in each `_think()` call
- LLM requests reads via `Thought.memory_reads`
- LLM writes via `Thought.memory_updates`
- Memory operations create spans for tracing
- Memory reads extend iterations (agent continues if memory requested)

Memory objects persist across `process()` calls for the lifetime of the agent instance.

---

### MCP Integration (`mcp/`)

Simplified MCP integration using the **ToolProvider** pattern:

#### `MCPToolProvider`
Implements `ToolProvider` protocol:
- `async setup()`: Connects to MCP servers, returns wrapped tools
- `async teardown()`: Disconnects from servers
- Automatic tool naming with prefix

#### `MCPServerConfig`
Configuration for MCP servers with support for:
- **stdio transport**: Docker containers, local binaries
- **http-sse transport**: HTTP-based MCP servers

#### Usage Pattern
```python
from opper_agent import mcp, MCPServerConfig

filesystem_server = MCPServerConfig(
    name="filesystem",
    transport="stdio",
    command="docker",
    args=[...]
)

agent = Agent(
    name="FileAgent",
    tools=[mcp(filesystem_server), my_local_tool]
)
```

MCP tools are activated during `process()` and cleaned up automatically.

---

### Specialized Agents (`agents/`)

#### `ReactAgent`
ReAct (Reasoning + Acting) pattern agent with custom loop:
1. Reason: Analyze situation
2. Act: Take action
3. Observe: Review result
4. Repeat or complete

#### `ChatAgent`
Conversational agent that maintains conversation state across multiple turns:
- Conversation history management
- Save/load conversation state
- Context-aware responses

---

### Utilities (`utils/`)

#### Decorators
- `@tool`: Convert functions to tools (with automatic schema extraction)
- `@hook`: Register hook functions for lifecycle events

#### Logging
- `AgentLogger`: Abstract base for custom loggers
- `RichLogger`: Fancy console output with colors and formatting
- `SimpleLogger`: Clean, minimal logging output

---

## Directory Structure

```
src/opper_agent/
├── __init__.py              # Main exports
├── base/                    # Core abstractions
│   ├── agent.py            # BaseAgent
│   ├── context.py          # AgentContext, Usage, ExecutionCycle
│   ├── tool.py             # Tool, FunctionTool, ToolResult
│   └── hooks.py            # HookManager, HookEvents
├── core/                    # Main implementations
│   ├── agent.py            # Agent (while tools > 0)
│   └── schemas.py          # Thought, ToolCall
├── agents/                  # Specialized agent types
│   ├── react.py            # ReactAgent
│   └── chat.py             # ChatAgent
├── memory/                  # Memory system
│   └── memory.py           # Memory, MemoryEntry
├── mcp/                     # MCP integration
│   ├── provider.py         # MCPToolProvider
│   ├── client.py           # MCP clients (stdio, http-sse)
│   ├── config.py           # MCPServerConfig
│   └── custom_sse.py       # Custom SSE client
└── utils/                   # Utilities
    ├── decorators.py       # @tool, @hook
    └── logging.py          # Logging system
```

---

## Key Design Decisions

### 1. Abstract Base Class Pattern
- Forces consistent interface across all agent types
- Prevents direct instantiation of `BaseAgent`
- Clear contract for custom agents
- Python's ABC provides compile-time checks

### 2. "While Tools > 0" Loop
- Natural stopping condition (no tools = done)
- Clear signal when agent completes task
- Allows flexible tool execution patterns
- Works well with LLM tool calling

### 3. Separate HookManager
- Clean separation of concerns
- Easy to test hooks independently
- Can evolve hook system without touching agent logic
- Supports both class-based and decorator hooks

### 4. AgentContext State Object
- Single source of truth for execution state
- Easy to serialize/deserialize
- Clean API for state access
- Simplifies testing (inject mock context)

### 5. LLM-Controlled Memory
- No separate "memory router" call (fewer LLM calls)
- LLM decides when to read/write memory
- Memory operations visible in traces
- More flexible and transparent

### 6. ToolProvider Protocol for MCP
- Clean lifecycle management (setup/teardown)
- Lazy tool resolution (connect only when needed)
- Consistent pattern for all tool sources
- Easy to extend for other tool providers

---

## Testing

### Test Structure
```
tests/
├── unit/              # 17+ test files, isolated tests
├── integration/       # MCP integration tests
└── e2e/              # End-to-end tests (planned)
```

### Coverage
- **Unit tests**: Core components, agents, tools, hooks, memory, MCP
- **Integration tests**: MCP stdio/http-sse, agent with real tools
- **Target**: >90% test coverage

### Testing Patterns
- Mock LLM responses with `mock_acompletion`
- VCR for HTTP recording/replay
- Async test support with pytest-asyncio
- Error handling for all failure scenarios

---

## Multi-Agent Support

Agents can be used as tools by other agents via `agent.as_tool()`:

```python
math_agent = Agent(name="Math", tools=[...])
research_agent = Agent(name="Research", tools=[...])

coordinator = Agent(
    name="Coordinator",
    tools=[math_agent.as_tool(), research_agent.as_tool()]
)
```

This enables delegation and specialization patterns.

---

## Error Handling

### Principles
- **Agent-level**: Graceful degradation, max iterations protection
- **Tool-level**: Return `ToolResult` with `success=False` and error message
- **Hook-level**: Catch, log, continue (never break execution)
- **Memory-level**: Degrade gracefully if memory operations fail

### Async Error Handling
All async operations have proper error handling with context managers and cleanup.

---

## Opper Integration

### Tracing
- Agent-level traces for entire execution
- Spans for each think cycle
- Spans for tool execution
- Spans for memory operations
- All spans properly parented for hierarchy

### Usage Tracking
- Track tokens per LLM call
- Cumulative usage in `AgentContext`
- Accessible for monitoring and cost tracking

---

## Future Enhancements

### Potential Features
- Tool result cleaning/summarization middleware
- Advanced retry logic with exponential backoff
- Context window management
- Streaming support for long-running agents
- Additional specialized agent types

### Extension Points
- Custom `BaseAgent` subclasses
- Custom `ToolProvider` implementations
- Custom `AgentLogger` implementations
- Custom memory backends

---

## Development Workflow

### Environment Setup
```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Format code
uvx ruff format

# Lint code
uvx ruff check
```

### Code Style
- Pydantic models for all data structures
- Type hints everywhere
- Async by default
- Google-style docstrings
- Single responsibility principle

---

## Success Criteria

1. ✅ Clean, maintainable codebase with proper OOP
2. ✅ Comprehensive test coverage (>90%)
3. ✅ Clear Opper traces/spans for observability
4. ✅ Easy to build custom agents through inheritance
5. ✅ Simple MCP integration with clean lifecycle
6. ✅ Multi-agent support via `as_tool()`
7. ✅ Flexible hook system for all lifecycle events
8. ✅ Memory system with LLM control
9. ✅ Type safety throughout with Pydantic

---

## Resources

- **Examples**: See `examples/` directory for working code
- **Tests**: See `tests/` for usage patterns
- **Opper Docs**: [docs.opper.ai](https://docs.opper.ai)
- **Opper API**: See `.cursor/rules/opper.mdc` for SDK reference
