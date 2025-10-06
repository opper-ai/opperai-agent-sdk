# Opper Agent SDK - Rebuild Specification

## Executive Summary

This document provides a comprehensive specification and phased implementation plan for rebuilding the Opper Agent SDK from scratch. The rebuild focuses on clean architecture, proper OOP principles, seamless Opper integration, and extensibility.

## Architecture Overview

### Core Principles

1. **Clean OOP Design**: Proper class hierarchies with clear separation of concerns
2. **Native Opper Integration**: Traces, spans, and calls handled elegantly
3. **Extensibility**: Easy to build custom agents through inheritance
4. **Simplicity**: Keep integrations simple and maintainable
5. **Type Safety**: Pydantic models throughout

### Class Hierarchy

```
BaseAgent (Abstract)
├── Agent (Main implementation - while tools>0 loop)
├── ReactAgent (Custom: REACT pattern)
├── ChatAgent (Future: Conversation management)
└── [Custom Agents] (User-defined)
```

---

## Phase 1: Core Architecture (Foundation)

### 1.1 Base Classes & Interfaces

#### `BaseAgent` (Abstract Base Class)
```python
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """
    Abstract base class for all agents. Defines the contract that all agents must follow.

    Key Responsibilities:
    - Manage agent lifecycle (initialization, cleanup)
    - Provide Opper client integration
    - Define hook system interface
    - Manage tracing/spans
    - Tool management interface
    """

    # Required attributes
    name: str
    description: str
    instructions: Optional[str]
    tools: Sequence[Union[Tool, ToolProvider]]
    input_schema: Optional[Type[BaseModel]]
    output_schema: Optional[Type[BaseModel]]

    # Configuration
    max_iterations: int
    verbose: bool
    model: Optional[str]

    # State
    opper: Opper  # Opper client
    context: AgentContext  # Current execution context

    @abstractmethod
    async def process(self, input: Any) -> Any:
        """Main entry point - must be implemented by subclasses"""
        pass

    @abstractmethod
    async def _run_loop(self, goal: Any) -> Any:
        """Core run loop - custom for each agent type"""
        pass

    # Standard methods (implemented in BaseAgent)
    def start_trace(self, name: str, input: Any) -> Span:
        """Start a new trace/span"""
        pass

    def add_tool(self, tool: Tool) -> None:
        """Add tool to agent"""
        pass

    def as_tool(self, **kwargs) -> FunctionTool:
        """Convert agent to tool for multi-agent systems"""
        pass
```

#### `AgentContext`
```python
class AgentContext(BaseModel):
    """
    Maintains all state for an agent execution.
    Tracks usage, history, memory, and provides clean access to execution data.
    """
    agent_name: str
    session_id: str
    trace_id: str

    # Execution state
    iteration: int = 0
    goal: Optional[Any] = None

    # History
    execution_history: List[ExecutionCycle] = []

    # Memory (optional)
    memory: Optional[Memory] = None

    # Token tracking
    usage: Usage = Usage()

    # Metadata
    metadata: Dict[str, Any] = {}

    def update_usage(self, usage: Usage) -> None:
        """Track cumulative token usage"""
        self.usage.add(usage)

    def add_cycle(self, cycle: ExecutionCycle) -> None:
        """Add execution cycle to history"""
        self.execution_history.append(cycle)

    def get_context_size(self) -> int:
        """Get current context size in tokens"""
        return self.usage.total_tokens
```

#### `Usage`
```python
class Usage(BaseModel):
    """Token usage tracking"""
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def add(self, other: Usage) -> Usage:
        """Combine usage stats"""
        return Usage(
            requests=self.requests + other.requests,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens
        )
```

### 1.2 Tool System

#### `Tool` (Base Class)
```python
class Tool(BaseModel):
    """Base tool class"""
    name: str
    description: str
    parameters: Dict[str, Any]

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool"""
        pass
```

#### `FunctionTool`
```python
class FunctionTool(Tool):
    """Wraps Python functions as tools"""
    func: Callable

    async def execute(self, **kwargs) -> ToolResult:
        """Execute wrapped function"""
        # Handle sync/async functions
        # Return standardized ToolResult
        pass
```

#### `ToolResult`
```python
class ToolResult(BaseModel):
    """Standardized tool execution result"""
    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time: float
    metadata: Dict[str, Any] = {}
```

#### `@tool` Decorator
```python
def tool(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None
) -> FunctionTool:
    """
    Decorator to convert functions to tools.
    Auto-extracts parameters from function signature.
    """
    pass
```

#### Tool Providers (Lazy Tool Resolution)
```python
class ToolProvider(Protocol):
    """Expand into tools at runtime."""

    async def setup(self, agent: "BaseAgent") -> List[Tool]:
        """Prepare provider, connect resources, and return tools."""

    async def teardown(self) -> None:
        """Cleanup after the agent run finishes."""


def normalize_tools(
    raw: Sequence[Union[Tool, ToolProvider]]
) -> Tuple[List[Tool], List[ToolProvider]]:
    """Split concrete tools from providers so lifecycle can be managed."""

    tools: List[Tool] = []
    providers: List[ToolProvider] = []
    for item in raw:
        if isinstance(item, Tool):
            tools.append(item)
        else:
            providers.append(item)
    return tools, providers
```

In `BaseAgent.__init__` we store the raw list, then:

```python
self.base_tools, self.tool_providers = normalize_tools(tools or [])
self.tools: List[Tool] = list(self.base_tools)
self.active_provider_tools: Dict[ToolProvider, List[Tool]] = {}
```

During `process()`:

```python
await self._activate_tool_providers()
try:
    ...  # existing execution flow
finally:
    await self._deactivate_tool_providers()
```

### 1.3 Hook System

#### `HookManager`
```python
class HookManager:
    """
    Manages all hooks for an agent.
    Supports both class-based and decorator-based hooks.
    """

    hooks: Dict[str, List[HookFunction]] = {}

    async def trigger(self, event: str, context: AgentContext, **kwargs) -> None:
        """Trigger all hooks for an event"""
        for hook_func in self.hooks.get(event, []):
            try:
                await hook_func(context, **kwargs)
            except Exception as e:
                logger.warning(f"Hook {event} failed: {e}")
                # Don't break execution on hook failure

    def register(self, event: str, hook: HookFunction) -> None:
        """Register a hook for an event"""
        if event not in self.hooks:
            self.hooks[event] = []
        self.hooks[event].append(hook)
```

#### Standard Hook Events
```python
# Must-have hooks:
- agent_start: Before agent begins
- agent_end: After agent completes
- agent_error: When agent errors
- loop_start: Start of each iteration
- loop_end: End of each iteration
- think_end: After the think step returns a Thought (before tool execution)
- tool_call: Before tool execution
- tool_result: After tool execution
- llm_call: Before LLM call
- llm_response: After LLM response
```

#### `@hook` Decorator
```python
def hook(event_name: str) -> Callable:
    """
    Decorator to create hook functions.

    Usage:
        @hook("agent_start")
        async def my_hook(context: AgentContext, agent: Agent):
            print(f"Starting {agent.name}")
    """
    pass
```

---

## Phase 2: Main Agent Implementation

### 2.1 `Agent` Class (Primary Implementation)

```python
class Agent(BaseAgent):
    """
    Main agent implementation using while tools>0 loop.

    Loop Logic:
    1. Think: Decide next action
    2. If tool calls > 0: Execute tools and continue
    3. If tool calls == 0: Generate final response
    """

    async def _run_loop(self, goal: Any) -> Any:
        """
        Main reasoning loop: while tools > 0
        """
        iteration = 0

        while iteration < self.max_iterations:
            # Hook: loop_start
            await self.hooks.trigger("loop_start", self.context)

            # Think
            thought = await self._think(goal)

            # Check if done (no tools to call)
            if not thought.tool_calls or len(thought.tool_calls) == 0:
                # Generate final response
                result = await self._generate_final_result(goal)
                break

            # Execute tools
            for tool_call in thought.tool_calls:
                tool_result = await self._execute_tool(tool_call)
                self.context.add_cycle(
                    ExecutionCycle(
                        thought=thought,
                        tool_call=tool_call,
                        result=tool_result
                    )
                )

            # Hook: loop_end
            await self.hooks.trigger("loop_end", self.context)

            iteration += 1

        return result

    async def _think(self, goal: Any) -> Thought:
        """
        LLM call to decide next action.
        Returns thought with tool calls (or empty if done).
        """
        # Optional: quick memory probe so model can opt-in
        memory_snapshot = None
        if self.context.memory and self.context.memory.has_entries():
            memory_snapshot = await self._prepare_memory_snapshot(goal)

        # Hook: llm_call

        # Build context for LLM
        context = self._build_llm_context(goal, memory=memory_snapshot)

        # Call Opper
        response = await self.opper.call(
            name="think",
            instructions=self._build_think_instructions(),
            input=context,
            output_schema=Thought,
            model=self.model,
            parent_span_id=self.context.trace_id
        )

        # Track usage
        self.context.update_usage(response.usage)

        # Hook: llm_response

        thought = Thought(**response.json_payload)

        # Hook: think_end
        await self.hooks.trigger("think_end", self.context, thought=thought)

        return thought

    async def _execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a single tool call"""
        # Hook: tool_call

        tool = self.get_tool(tool_call.name)

        # Create span for tool execution
        with self.opper.spans.create(
            name=f"tool_{tool_call.name}",
            parent_id=self.context.trace_id
        ) as span:
            result = await tool.execute(**tool_call.parameters)

            # Optional: Clean tool result
            if self.clean_tool_results:
                result = await self._clean_tool_result(result)

        # Hook: tool_result

        return result
```

### 2.2 Thought Schema

```python
class ToolCall(BaseModel):
    """Represents a single tool call"""
    name: str
    parameters: Dict[str, Any]
    reasoning: str

class Thought(BaseModel):
    """
    AI's reasoning and action plan.
    Empty tool_calls indicates completion.
    """
    reasoning: str = Field(description="Analysis of situation")
    tool_calls: List[ToolCall] = Field(
        default=[],
        description="Tools to call (empty if task complete)"
    )
    memory_updates: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Keyed updates the agent wants to persist to memory"
    )
    user_message: str = Field(description="Status update for user")
```

### 2.3 Execution Cycle

```python
class ExecutionCycle(BaseModel):
    """Represents one think-act cycle"""
    iteration: int
    thought: Thought
    tool_calls: List[ToolCall]
    results: List[ToolResult]
    timestamp: float
```

---

## Phase 3: Optional Features

### 3.1 Memory System

```python
class MemoryEntry(BaseModel):
    """Metadata + payload for a single memory slot."""

    key: str
    description: str
    value: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)
    last_accessed: float = Field(default_factory=time.time)


class Memory(BaseModel):
    """
    Fast memory system for agents.
    Stores MemoryEntry objects and surfaces a lightweight catalog to the LLM.
    """

    store: Dict[str, MemoryEntry] = Field(default_factory=dict)

    def has_entries(self) -> bool:
        return len(self.store) > 0

    async def list_entries(self) -> List[Dict[str, Any]]:
        """
        Return summaries so model can decide what to load.
        Each summary includes key, description, and optional tags/metadata.
        """
        catalog = []
        for entry in self.store.values():
            catalog.append(
                {
                    "key": entry.key,
                    "description": entry.description,
                    "metadata": entry.metadata,
                }
            )
        return catalog

    async def read(self, keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """Read specific memory entries (or all when keys is None)."""
        if not keys:
            keys = list(self.store.keys())
        payload = {}
        for key in keys:
            if key in self.store:
                entry = self.store[key]
                entry.last_accessed = time.time()
                payload[key] = entry.value
        return payload

    async def write(
        self,
        key: str,
        value: Any,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write/update memory and keep catalog metadata fresh."""
        if key in self.store:
            entry = self.store[key]
            entry.value = value
            if description:
                entry.description = description
            if metadata:
                entry.metadata.update(metadata)
            entry.last_accessed = time.time()
        else:
            self.store[key] = MemoryEntry(
                key=key,
                description=description or key,
                value=value,
                metadata=metadata or {},
            )

    async def clear(self) -> None:
        """Clear memory."""
        self.store.clear()
```

#### Memory Integration
```python
class MemoryDecision(BaseModel):
    should_use_memory: bool
    selected_keys: List[str] = []
    rationale: str


async def _prepare_memory_snapshot(self, goal: Any) -> Optional[Dict[str, Any]]:
    """
    Lightweight LLM call that lets the model opt-in to loading memory.
    Returns hydrated entries or None when memory is skipped.
    """

    catalog = await self.context.memory.list_entries()
    decision_response = await self.opper.call(
        name="memory_router",
        instructions=self._memory_router_instructions(),
        input={
            "goal": str(goal),
            "memory_catalog": catalog,
            "recent_history": self.context.get_last_iterations_summary(2),
        },
        output_schema=MemoryDecision,
        model=self.model,
        parent_span_id=self.context.trace_id,
    )

    decision = MemoryDecision(**decision_response.json_payload)
    if not decision.should_use_memory or not decision.selected_keys:
        return None

    return await self.context.memory.read(decision.selected_keys)


def _memory_router_instructions(self) -> str:
    return """Decide if any memory should be loaded for the next reasoning step.
- Return should_use_memory=true only when a listed memory clearly helps.
- selected_keys must only contain keys from memory_catalog.
- Keep rationale concise.
"""


# After tool execution: optionally update memory
if self.context.memory and thought.memory_updates:
    for key, update in thought.memory_updates.items():
        await self.context.memory.write(**update)
```

### 3.2 Tool Result Cleaning

```python
class ToolResultCleaner:
    """
    Optional middleware to clean/summarize tool results.
    Uses small LLM to extract relevant information.
    """

    threshold: int = 1000  # chars
    model: str = "groq/llama-3.1-8b-instant"  # Default fast model, user-configurable

    async def clean(
        self,
        result: ToolResult,
        goal: str,
        opper: Opper,
        parent_span_id: str
    ) -> ToolResult:
        """
        Clean tool result if it exceeds threshold.
        """
        if len(str(result.result)) < self.threshold:
            return result

        # Call small LLM to summarize
        cleaned = await opper.call(
            name="clean_tool_result",
            instructions="Extract only relevant information for the goal",
            input={
                "goal": goal,
                "tool_result": str(result.result)
            },
            model=self.model,  # User-configurable model
            parent_span_id=parent_span_id
        )

        result.result = cleaned.message
        result.metadata["cleaned"] = True

        return result
```

---

## Phase 4: MCP Integration

### 4.1 Simplified MCP Architecture

```python
class MCPServerConfig(BaseModel):
    """Configuration for an MCP server."""

    name: str
    transport: Literal["stdio", "http-sse"]
    url: Optional[str] = None  # For http-sse
    command: Optional[str] = None  # For stdio
    args: List[str] = Field(default_factory=list)
    env: Dict[str, str] = Field(default_factory=dict)
    timeout: float = 30.0


class MCPToolProvider(ToolProvider):
    """ToolProvider that encapsulates MCP servers."""

    def __init__(
        self,
        configs: Sequence[MCPServerConfig],
        *,
        name_prefix: Optional[str] = None,
    ) -> None:
        self.configs = list(configs)
        self.name_prefix = name_prefix
        self.clients: Dict[str, MCPClient] = {}

    async def setup(self, agent: "BaseAgent") -> List[Tool]:
        """Connect to servers and return wrapped MCP tools."""
        tools: List[Tool] = []
        for config in self.configs:
            client = self._create_client(config)
            await client.connect()
            self.clients[config.name] = client

            for mcp_tool in await client.list_tools():
                tools.append(self._wrap_tool(config.name, mcp_tool))

        return tools

    async def teardown(self) -> None:
        """Disconnect from all servers."""
        for client in self.clients.values():
            await client.disconnect()
        self.clients.clear()

    def _create_client(self, config: MCPServerConfig) -> MCPClient:
        return MCPClient.from_config(config)

    def _wrap_tool(self, server_name: str, mcp_tool: MCPTool) -> FunctionTool:
        prefix = self.name_prefix or server_name
        tool_name = f"{prefix}:{mcp_tool.name}"

        async def tool_func(**kwargs: Any) -> Any:
            client = self.clients[server_name]
            return await client.call_tool(mcp_tool.name, kwargs)

        return FunctionTool(
            func=tool_func,
            name=tool_name,
            description=mcp_tool.description,
            parameters=mcp_tool.parameters,
        )


def mcp(*configs: MCPServerConfig, name_prefix: Optional[str] = None) -> MCPToolProvider:
    """Helper so users can write tools = [mcp(config), my_tool]."""
    if not configs:
        raise ValueError("At least one MCPServerConfig is required")
    return MCPToolProvider(configs, name_prefix=name_prefix)
```

### 4.2 Usage Pattern

```python
search = MCPServerConfig(
    name="search",
    transport="http-sse",
    url="https://mcp-server.com/endpoint",
)

agent = Agent(
    name="SearchAgent",
    tools=[
        mcp(search),
        my_local_tool,
    ],
    ...,
)
# BaseAgent will activate providers during process():
# - connect MCP servers
# - surface FunctionTools
# - disconnect on teardown
```

---

## Phase 5: Advanced Features

### 5.1 Custom Agents (REACT Example)

```python
class ReactAgent(BaseAgent):
    """
    REACT pattern agent: Reasoning + Acting in cycles.

    Loop:
    1. Reason: Analyze situation
    2. Act: Take action
    3. Observe: Review result
    4. Repeat or complete
    """

    async def _run_loop(self, goal: Any) -> Any:
        """Custom REACT loop"""

        observation = "Initial task received"

        while self.context.iteration < self.max_iterations:
            # REASON
            reasoning = await self._reason(goal, observation)

            # ACT
            if reasoning.is_complete:
                return await self._generate_final_result(goal)

            action = reasoning.action
            result = await self._execute_tool(action)

            # OBSERVE
            observation = result.result

            self.context.iteration += 1

        return await self._generate_final_result(goal)
```

### 5.2 Multi-Agent Support

Already implemented via `agent.as_tool()` - keeps current pattern which works well.

```python
math_agent = Agent(name="Math", tools=[...])
research_agent = Agent(name="Research", tools=[...])

coordinator = Agent(
    name="Coordinator",
    tools=[
        math_agent.as_tool(),
        research_agent.as_tool()
    ]
)
```

### 5.3 Chat Agent (Placeholder)

```python
class Conversation:
    """
    Manages conversation state across multiple turns.
    """

    messages: List[Message] = []

    def add_message(self, role: str, content: str) -> None:
        """Add message to conversation"""
        self.messages.append(Message(role=role, content=content))

    def to_dict(self) -> Dict:
        """Serialize conversation"""
        return {"messages": [m.dict() for m in self.messages]}

    @classmethod
    def from_dict(cls, data: Dict) -> "Conversation":
        """Deserialize conversation"""
        conv = cls()
        conv.messages = [Message(**m) for m in data["messages"]]
        return conv

class ChatAgent(BaseAgent):
    """
    Agent that handles multi-turn conversations.
    Maintains conversation state and context.
    """

    conversation: Conversation

    async def process(self, message: str) -> str:
        """Process user message in conversation"""
        # Add user message
        self.conversation.add_message("user", message)

        # Generate response with full conversation context
        response = await self._generate_response(
            self.conversation.to_dict()
        )

        # Add assistant message
        self.conversation.add_message("assistant", response)

        return response

    def save_conversation(self, path: str) -> None:
        """Save conversation to file"""
        with open(path, 'w') as f:
            json.dump(self.conversation.to_dict(), f)

    def load_conversation(self, path: str) -> None:
        """Load conversation from file"""
        with open(path, 'r') as f:
            data = json.load(f)
        self.conversation = Conversation.from_dict(data)
```

---

## Phase 6: Error Handling Strategy

### 6.1 Error Handling Principles

The SDK should gracefully handle errors at multiple levels:

#### Agent-Level Errors
- LLM call failures: Retry with exponential backoff (configurable)
- Tool provider setup/teardown failures: Log and continue with available tools
- Max iterations exceeded: Return partial result with clear error message
- Invalid input/output schema: Validate early, fail fast with helpful messages

#### Tool-Level Errors
- Tool execution failures: Return ToolResult with success=False and error message
- Invalid tool calls from LLM: Log warning, skip tool, continue execution
- Tool timeouts: Configurable per-tool, fail gracefully

#### Hook-Level Errors
- Hook failures never break execution
- Log warnings but continue agent flow
- Provide hook error callback option for monitoring

#### Memory-Level Errors
- Memory read/write failures: Degrade gracefully, continue without memory
- Invalid memory keys: Log warning, skip operation

### 6.2 Testing Strategy

Each error scenario should have explicit test coverage:
- Unit tests for error conditions in isolation
- Integration tests for error propagation
- E2E tests for graceful degradation under failure
- Timeout tests for all async operations

**Key Test Scenarios:**
- Opper API unavailable
- Tool provider connection failure
- Malformed LLM responses
- Infinite loops / max iterations
- Memory corruption
- Nested agent failures

---

## Phase 7: Testing & Documentation

### 7.1 Test Structure

```
tests/
├── unit/
│   ├── test_base_agent.py
│   ├── test_agent.py
│   ├── test_tools.py
│   ├── test_hooks.py
│   ├── test_memory.py
│   └── test_context.py
├── integration/
│   ├── test_opper_integration.py
│   ├── test_mcp_integration.py
│   └── test_multi_agent.py
└── e2e/
    ├── test_math_agent.py
    ├── test_research_agent.py
    └── test_react_agent.py
```

### 7.2 Test Coverage Requirements

- Unit tests: >90% coverage
- All public APIs tested
- Error conditions tested
- Async operations tested properly

---

## Implementation Phases

### Phase 1: Foundation (Week 1-2)
- [ ] `BaseAgent` abstract class
- [ ] `AgentContext` and state management
- [ ] `Tool` and `FunctionTool` classes
- [ ] `@tool` decorator
- [ ] `HookManager` and `@hook` decorator
- [ ] Unit tests for core classes

### Phase 2: Main Agent (Week 2-3)
- [ ] `Agent` class with while-loop logic
- [ ] `Thought` and `ToolCall` schemas
- [ ] Opper integration (traces, spans)
- [ ] Token usage tracking
- [ ] Integration tests

### Phase 3: Optional Features (Week 3-4)
- [ ] `Memory` system
- [ ] `ToolResultCleaner` middleware
- [ ] Verbosity flag support
- [ ] Context size warnings
- [ ] Unit tests

### Phase 4: MCP Integration (Week 4-5)
- [ ] Simplified `MCPToolProvider`
- [ ] HTTP-SSE and stdio support
- [ ] Tool conversion logic
- [ ] MCP integration tests

### Phase 5: Advanced Features (Week 5-6)
- [ ] `ReactAgent` example
- [ ] `ChatAgent` placeholder
- [ ] Multi-agent improvements
- [ ] E2E tests

### Phase 6: Error Handling (Week 6)
- [ ] Implement error handling strategy
- [ ] Add retry logic for LLM calls
- [ ] Graceful degradation tests
- [ ] Error propagation tests

### Phase 7: Polish & Docs (Week 7)
- [ ] Comprehensive testing
- [ ] Documentation
- [ ] Examples
- [ ] Migration guide
- [ ] Performance optimization

---

## Directory Structure

```
src/opper_agent/
├── __init__.py
├── base/
│   ├── __init__.py
│   ├── agent.py          # BaseAgent
│   ├── context.py        # AgentContext, Usage
│   ├── tool.py           # Tool, FunctionTool
│   └── hooks.py          # HookManager, decorators
├── core/
│   ├── __init__.py
│   └── agent.py          # Agent (main implementation)
├── agents/
│   ├── __init__.py
│   ├── react.py          # ReactAgent
│   └── chat.py           # ChatAgent (placeholder)
├── memory/
│   ├── __init__.py
│   └── memory.py         # Memory system
├── middleware/
│   ├── __init__.py
│   └── cleaner.py        # ToolResultCleaner
├── mcp/
│   ├── __init__.py
│   ├── provider.py       # MCPToolProvider
│   ├── client.py         # MCP clients
│   └── config.py         # MCPServerConfig
└── utils/
    ├── __init__.py
    ├── decorators.py     # @tool, @hook
    └── opper.py          # Opper helpers
```

---

## Key Design Decisions

### 1. Why Abstract Base Class?
- Forces consistent interface across all agent types
- Makes it impossible to instantiate BaseAgent directly
- Clear contract for what agents must implement
- Python's ABC provides compile-time checks

### 2. Why `while tools > 0` Loop?
- Natural stopping condition
- Clear signal when agent is done
- Allows flexible tool execution
- Works well with LLM tool calling patterns

### 3. Why Separate HookManager?
- Cleaner separation of concerns
- Easier to test hooks independently
- Can evolve hook system without touching agent logic
- Supports both class-based and decorator hooks

### 4. Why AgentContext?
- Single source of truth for execution state
- Easy to serialize/deserialize
- Clean API for state access
- Simplifies testing (inject mock context)

### 5. Why Optional Memory/Cleaner?
- Not all agents need these features
- Keeps base implementation simple
- Easy to add when needed
- Can be configured per-agent

---

## Migration Path

### From Old to New

1. **Tool definitions**: No changes needed
   ```python
   @tool
   def my_tool(x: int) -> str:
       return str(x)
   ```

2. **Agent creation**: Minimal changes
   ```python
   # Old
   agent = Agent(name="Test", tools=[...])

   # New (same API)
   agent = Agent(name="Test", tools=[...])
   ```

3. **Hooks**: Same decorator syntax
   ```python
   @hook("agent_start")
   async def on_start(context, agent):
       pass
   ```

4. **MCP**: Cleaner API
   ```python
   # Old: Implicit connections, unclear lifecycle

   # New: Declarative provider bundled as a tool
   tools = [
       mcp(config),  # handles connect/disconnect automatically
       local_tool,
   ]
   agent = Agent(name="Search", tools=tools)
   ```

---

## Success Criteria

1. ✅ All existing examples work with new SDK
2. ✅ Clean, maintainable codebase
3. ✅ >90% test coverage
4. ✅ Clear Opper traces/spans
5. ✅ Easy to build custom agents
6. ✅ Simple MCP integration
7. ✅ Comprehensive documentation
8. ✅ Performance: ≥ old implementation

---

## Notes

- Keep all existing features (multi-agent, as_tool, etc.)
- Add placeholders for future features (ChatAgent)
- Focus on clean code and good tests
- Document everything
- Maintain backward compatibility where possible
