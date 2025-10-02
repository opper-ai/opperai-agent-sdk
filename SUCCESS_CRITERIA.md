# Success Criteria for Opper Agent SDK Rebuild

## Primary Goals

The rebuild is considered successful when:

1. ✅ **Code Quality**: Clean, maintainable, well-documented OOP code
2. ✅ **Feature Parity**: All current functionality preserved (or better)
3. ✅ **Examples Work**: All examples run successfully (or with minimal changes)
4. ✅ **Extensibility**: Easy to build custom agents
5. ✅ **Test Coverage**: Comprehensive test suite

---

## 1. Core Functionality Must Work

### 1.1 Basic Agent Creation & Execution
```python
# MUST WORK (exactly or very similar)
from opper_agent import Agent, tool

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

agent = Agent(
    name="MathAgent",
    description="Does math",
    tools=[add]
)

result = await agent.process("Calculate 5 + 3")
assert result is not None
```

**Success Criteria:**
- Agent initializes without errors
- Tool decorator works
- Process method executes
- Result is returned

### 1.2 Structured Input/Output
```python
# MUST WORK (exactly or very similar)
from pydantic import BaseModel, Field

class MathProblem(BaseModel):
    problem: str = Field(description="The math problem")

class MathSolution(BaseModel):
    answer: float = Field(description="The answer")
    reasoning: str = Field(description="How we got it")

agent = Agent(
    name="MathAgent",
    input_schema=MathProblem,
    output_schema=MathSolution,
    tools=[add]
)

result = await agent.process(MathProblem(problem="What is 5 + 3?"))
assert isinstance(result, MathSolution)
assert hasattr(result, 'answer')
```

**Success Criteria:**
- Input validation works
- Output validation works
- Pydantic models integrate cleanly

### 1.3 Hooks System
```python
# MUST WORK (exactly or very similar)
from opper_agent import hook, RunContext

@hook("agent_start")
async def on_start(context: RunContext, agent: Agent):
    print(f"Agent {agent.name} starting")

@hook("tool_call")
async def on_tool(context: RunContext, agent: Agent, tool, **kwargs):
    print(f"Calling tool: {tool.name}")

agent = Agent(
    name="Test",
    tools=[add],
    hooks=[on_start, on_tool]
)

await agent.process("test")
# Should print hook messages
```

**Success Criteria:**
- Hook decorator works
- Multiple hooks can be registered
- Hooks execute at right times
- Hook failures don't crash agent

---

## 2. MCP Integration Must Work

```python
# MUST WORK (pattern can be slightly different but should be simple)
from opper_agent.mcp import MCPToolManager, MCPServerConfig

# Setup
mcp = MCPToolManager()

# Add servers
mcp.add_server(MCPServerConfig(
    name="filesystem",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    transport="stdio"
))

mcp.add_server(MCPServerConfig(
    name="search",
    url="https://mcp-server.com/endpoint",
    transport="http-sse"
))

# Connect
await mcp.connect_all()

# Get tools
tools = mcp.get_all_tools()
print(f"Got {len(tools)} tools")

# Use with agent
agent = Agent(name="Test", tools=tools)

# Cleanup
await mcp.disconnect_all()
```

**Success Criteria:**
- Multiple transports work (stdio, http-sse)
- Multiple servers can be used simultaneously
- Tool conversion works correctly
- Connection lifecycle is clean (no hanging connections)
- Error handling is graceful

---

## 3. Custom Agent Extensibility

### 3.1 Custom Agent (REACT Example)

```python
# SHOULD WORK (new pattern for custom agents)
from opper_agent import BaseAgent

class ReactAgent(BaseAgent):
    """Custom REACT-pattern agent."""

    async def _run_loop(self, goal):
        observation = "Starting"

        for i in range(self.max_iterations):
            # REASON
            reasoning = await self._reason(goal, observation)

            # ACT
            if reasoning.is_complete:
                break

            action = reasoning.next_action
            result = await self._execute_tool(action)

            # OBSERVE
            observation = result.result

        return await self._generate_final_result(goal)

    async def _reason(self, goal, observation):
        # Custom LLM call for reasoning
        ...

# Should work
react_agent = ReactAgent(
    name="ReactAgent",
    tools=[...]
)

result = await react_agent.process("solve this problem")
```

**Success Criteria:**
- BaseAgent provides proper foundation
- Custom loop logic is easy to implement
- All BaseAgent features work (hooks, tools, etc.)
- Custom agents integrate with multi-agent patterns

---

## 4. Examples Must Run

### 4.1 Deep Research Agent Example

The research agent from `deep_research_agent.py` should work with minimal changes:

```python
# Original pattern (should work or be very similar)
from opper_agent import Agent, hook, tool
from opper_agent.mcp import MCPServerConfig, MCPToolManager

# Define schemas
class ResearchRequest(BaseModel):
    topic: str
    depth: str = "comprehensive"
    sources_required: int = 10

class ResearchFindings(BaseModel):
    executive_summary: str
    key_findings: List[str]
    detailed_analysis: str

# Setup MCP
mcp = MCPToolManager()
mcp.add_server(MCPServerConfig(
    name="composio-search",
    url="https://apollo.composio.dev/...",
    transport="http-sse"
))

await mcp.connect_all()
tools = mcp.get_all_tools()

# Custom tool
@tool
def save_report(report: str) -> str:
    with open("report.md", "w") as f:
        f.write(report)
    return "Report saved"

tools.append(save_report)

# Hook
@hook("agent_start")
async def on_start(context, agent):
    print(f"Starting research on: {context.goal}")

# Create agent
agent = Agent(
    name="ResearchAgent",
    instructions="You are a comprehensive research agent...",
    input_schema=ResearchRequest,
    output_schema=ResearchFindings,
    tools=tools,
    hooks=[on_start],
    model="groq/gpt-oss-120b"
)

# Run
result = await agent.process(
    ResearchRequest(
        topic="What is Opper AI?",
        sources_required=15
    )
)

await mcp.disconnect_all()
```

**Success Criteria:**
- Agent initializes with all components
- MCP tools integrate smoothly
- Custom tools work alongside MCP tools
- Hooks execute properly
- Result matches output schema

### 4.2 Multi-Agent Example

From `multi_agent_example.py`:

```python
# MUST WORK (exactly or very similar)
from opper_agent import Agent, tool

# Specialized agents
@tool
def calculate(expr: str) -> float:
    return eval(expr)

math_agent = Agent(
    name="MathAgent",
    description="Handles math",
    instructions="Show your work step by step",
    tools=[calculate]
)

@tool
def translate(text: str) -> str:
    return text  # simplified

swedish_agent = Agent(
    name="SwedishAgent",
    description="Handles Swedish",
    tools=[translate]
)

# Coordinator using agent.as_tool()
coordinator = Agent(
    name="Coordinator",
    description="Routes tasks",
    tools=[
        math_agent.as_tool(tool_name="delegate_to_math"),
        swedish_agent.as_tool(tool_name="delegate_to_swedish")
    ]
)

# Should route correctly
result = await coordinator.process("Calculate 15 * 8")
```

**Success Criteria:**
- `agent.as_tool()` works
- Agent delegation functions properly
- Coordinator routes to correct sub-agent
- Results flow back correctly

### 4.3 Math Agent Example

From `math_agent_example.py`:

```python
# MUST WORK (exactly or very similar)
from opper_agent import Agent, tool, hook

@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

@tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

@hook("agent_start")
async def on_start(context, agent):
    print(f"🧮 Math Agent started")

@hook("tool_result")
async def on_result(context, agent, tool, result):
    print(f"Tool {tool.name} returned: {result.result}")

agent = Agent(
    name="MathAgent",
    description="Mathematical operations",
    tools=[add_numbers, multiply_numbers],
    hooks=[on_start, on_result],
    verbose=True
)

result = await agent.process("Calculate (12 * 8) + (45 / 9)")
```

**Success Criteria:**
- Multiple tools work
- Hooks fire correctly
- Verbose mode shows execution
- Math operations execute in sequence

---

## 5. Advanced Features

### 5.1 Memory System (Optional Feature)

```python
# SHOULD WORK (new feature, can be different)
agent = Agent(
    name="MemoryAgent",
    tools=[...],
    enable_memory=True  # Enable memory
)

# Memory is automatically managed
result = await agent.process("Remember: my name is Jose")
# Agent stores to memory

result = await agent.process("What's my name?")
# Agent reads from memory
assert "Jose" in str(result)
```

**Success Criteria:**
- Memory can be enabled/disabled
- Agent reads/writes memory automatically
- Memory persists across loop iterations
- Memory doesn't break when disabled

---

## 6. Developer Experience Requirements

### 6.1 Easy Custom Tools
```python
# Creating tools should be trivial
@tool
def my_tool(x: int, y: str) -> dict:
    """Tool description."""
    return {"x": x, "y": y}

# Parameters auto-extracted from signature
# Description from docstring
# Type hints respected
```

### 6.2 Clear Error Messages
```python
# When things go wrong, errors should be helpful
try:
    agent = Agent(name="Test")  # No tools
    await agent.process("do something")
except Exception as e:
    # Error should clearly indicate the problem
    assert "no tools" in str(e).lower() or similar helpful message
```

### 6.3 Good Tracing
```python
# Opper platform should show:
agent = Agent(name="TestAgent", tools=[...])
await agent.process("test")

# In Opper UI:
# - Clear agent trace
# - Individual tool spans
# - LLM call spans
# - Token usage per call
# - Clear parent-child relationships
```

**Success Criteria:**
- Traces appear in Opper dashboard
- Span hierarchy is clear
- Tool executions are visible
- Token usage is tracked
- Easy to debug failures

---

## 7. Code Quality Requirements

### 7.1 Clean Architecture
```python
# Code should follow clear patterns:
from opper_agent.base import BaseAgent, Tool, AgentContext
from opper_agent.core import Agent
from opper_agent.mcp import MCPToolManager

# Imports are logical
# No circular dependencies
# Clear module boundaries
```

### 7.2 Type Safety
```python
# Everything should be properly typed
def process(self, input: Any) -> Any:  # ✅
    ...

# Pydantic models throughout
class AgentContext(BaseModel):  # ✅
    ...

# Type hints help IDEs
agent.tools  # IDE knows this is List[Tool]
```

### 7.3 Documentation
```python
# Every public API documented
class Agent(BaseAgent):
    """
    Main agent implementation.

    Uses while-loop pattern: continues until no tools to call.

    Args:
        name: Agent name
        tools: List of available tools
        ...

    Example:
        >>> agent = Agent(name="Test", tools=[...])
        >>> result = await agent.process("goal")
    """
```

**Success Criteria:**
- Every public class documented
- Every public method documented
- Examples in docstrings
- Clear parameter descriptions

---

## 8. Testing Requirements

### 8.1 Unit Tests
```python
# Core components tested in isolation
def test_tool_execution():
    tool = FunctionTool(lambda x: x * 2)
    result = await tool.execute(x=5)
    assert result.success
    assert result.result == 10

def test_context_usage_tracking():
    ctx = AgentContext(agent_name="Test")
    ctx.update_usage(Usage(total_tokens=100))
    assert ctx.usage.total_tokens == 100
```

**Coverage Target:** >90% for core modules

### 8.2 Integration Tests
```python
# Test components working together
@pytest.mark.integration
async def test_agent_with_tools():
    @tool
    def add(a, b):
        return a + b

    agent = Agent(name="Test", tools=[add])
    result = await agent.process("add 2 and 3")
    assert result is not None

@pytest.mark.integration
async def test_mcp_integration():
    mcp = MCPToolManager()
    # ... test MCP workflow
```

### 8.3 E2E Tests
```python
# Test complete workflows
@pytest.mark.e2e
async def test_research_agent_workflow():
    # Similar to deep_research_agent.py
    agent = Agent(...)
    result = await agent.process(ResearchRequest(...))
    assert isinstance(result, ResearchFindings)
    assert len(result.key_findings) > 0
```

---

## 9. Performance Requirements

### 9.1 No Performance Regression
```python
# New implementation should be >= old performance
# Measure: Time to complete standard benchmark
old_time = 5.2  # seconds
new_time = measure_benchmark()
assert new_time <= old_time * 1.1  # Allow 10% variance
```

### 9.2 Memory Efficiency
```python
# No memory leaks
# Especially important for:
# - MCP connections (cleanup properly)
# - Long-running agents
# - Multiple agent instances
```

### 9.3 Token Efficiency
```python
# Track and optimize token usage
agent = Agent(...)
result = await agent.process("task")

print(f"Tokens used: {agent.context.usage.total_tokens}")
# Should be reasonable for the task
```

---

## 10. Migration Path

### 10.1 Backward Compatibility
```python
# Most existing code should work with minimal changes
# Old pattern:
from opper_agent_old import Agent, tool

# New pattern (same or very similar):
from opper_agent import Agent, tool

# If syntax must change, it should be obvious and well-documented
```

### 10.2 Migration Guide
```markdown
# Migration Guide should explain:
1. What stayed the same
2. What changed and why
3. How to update code
4. Common issues and solutions
```

---

## Acceptance Test Checklist

### Before Considering Rebuild Complete:

- [ ] **Basic Functionality**
  - [ ] Simple agent with tools runs
  - [ ] Hooks work as expected
  - [ ] Input/output schemas work

- [ ] **MCP Integration**
  - [ ] HTTP-SSE transport works
  - [ ] stdio transport works
  - [ ] Multiple servers simultaneously
  - [ ] Clean connection lifecycle

- [ ] **Custom Agents**
  - [ ] BaseAgent enables custom implementations
  - [ ] REACT example works
  - [ ] Custom agents work with as_tool()

- [ ] **Examples**
  - [ ] Deep research agent works (with MCP)
  - [ ] Multi-agent example works
  - [ ] Math agent example works

- [ ] **Advanced Features**
  - [ ] Memory system works (optional)
  - [ ] Tool result cleaning works (optional)
  - [ ] Verbose mode works

- [ ] **Code Quality**
  - [ ] All public APIs documented
  - [ ] Type hints throughout
  - [ ] Clean module structure
  - [ ] No circular dependencies

- [ ] **Testing**
  - [ ] >90% unit test coverage
  - [ ] Integration tests pass
  - [ ] E2E tests pass
  - [ ] No memory leaks

- [ ] **Tracing**
  - [ ] Traces visible in Opper dashboard
  - [ ] Clear span hierarchy
  - [ ] Token usage tracked
  - [ ] Tool executions visible

- [ ] **Documentation**
  - [ ] README updated
  - [ ] API docs complete
  - [ ] Migration guide written
  - [ ] Examples updated

---

## Non-Goals (Out of Scope)

These are NOT required for success:

- ❌ 100% identical API to old version (close is fine)
- ❌ Supporting Python < 3.10
- ❌ Synchronous API (async-only is fine)
- ❌ Every edge case from old implementation
- ❌ Streaming responses (can be added later)
- ❌ ChatAgent full implementation (placeholder is fine)

---

## Summary

**The rebuild is successful when:**

1. Core functionality works perfectly
2. MCP integration is seamless and simple
3. Custom agents are easy to build
4. Examples work (exactly or with minor, obvious changes)
5. Code is clean and maintainable
6. Tests provide confidence

**The key insight:** It's okay if some syntax changes slightly, as long as:
- The changes make the code cleaner
- The changes are well-documented
- The migration path is clear
- The core patterns remain familiar

**Priority order:**
1. Core functionality (Agent, tools, hooks)
2. MCP integration (clean and reliable)
3. Custom agents (extensibility via BaseAgent)
4. Examples work (research, multi-agent, math)
5. Advanced features (memory, cleaning, etc.)
