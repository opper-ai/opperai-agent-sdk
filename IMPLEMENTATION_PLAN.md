# Opper Agent SDK - Detailed Implementation Plan

## Overview

This document provides a step-by-step implementation plan for rebuilding the Opper Agent SDK. Each phase includes specific tasks, code snippets, and testing requirements.

---

## Phase 1: Foundation & Base Classes (Week 1-2)

### Task 1.1: Project Setup & Structure

**Files to create:**
```
src/opper_agent/
├── __init__.py                    # Main exports
├── base/
│   ├── __init__.py               # Base exports
│   ├── agent.py                  # BaseAgent
│   ├── context.py                # AgentContext, Usage, ExecutionCycle
│   ├── tool.py                   # Tool, FunctionTool, ToolResult
│   └── hooks.py                  # HookManager, hook events
└── utils/
    ├── __init__.py
    └── decorators.py             # @tool, @hook decorators
```

**Implementation Steps:**

1. Create directory structure
2. Set up `__init__.py` files with proper exports
3. Create placeholder files with docstrings

**Testing:**
```bash
# Verify imports work
python -c "from opper_agent.base import BaseAgent"
```

### Task 1.2: Data Models (`context.py`)

**Implementation:**

```python
# src/opper_agent/base/context.py
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import time

class Usage(BaseModel):
    """Tracks token usage across agent execution."""

    requests: int = Field(default=0, description="Number of LLM requests")
    input_tokens: int = Field(default=0, description="Input tokens used")
    output_tokens: int = Field(default=0, description="Output tokens used")
    total_tokens: int = Field(default=0, description="Total tokens")

    def add(self, other: 'Usage') -> 'Usage':
        """Combine usage statistics."""
        return Usage(
            requests=self.requests + other.requests,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens
        )

    def __repr__(self) -> str:
        return f"Usage(requests={self.requests}, tokens={self.total_tokens})"


class ExecutionCycle(BaseModel):
    """Represents one think-act cycle in agent execution."""

    iteration: int = Field(description="Iteration number")
    thought: Optional[Any] = Field(default=None, description="Agent's reasoning")
    tool_calls: List[Any] = Field(default=[], description="Tools called")
    results: List[Any] = Field(default=[], description="Tool results")
    timestamp: float = Field(default_factory=time.time)

    class Config:
        arbitrary_types_allowed = True


class AgentContext(BaseModel):
    """
    Maintains all state for an agent execution session.
    Single source of truth for execution state, history, and metadata.
    """

    # Identity
    agent_name: str = Field(description="Name of the agent")
    session_id: str = Field(default_factory=lambda: str(time.time()))

    # Tracing
    trace_id: Optional[str] = Field(default=None, description="Root trace ID")
    span_id: Optional[str] = Field(default=None, description="Current span ID")

    # Execution state
    iteration: int = Field(default=0, description="Current iteration")
    goal: Optional[Any] = Field(default=None, description="Current goal")

    # History
    execution_history: List[ExecutionCycle] = Field(
        default_factory=list,
        description="History of execution cycles"
    )

    # Token tracking
    usage: Usage = Field(default_factory=Usage, description="Token usage stats")

    # Memory (optional, will be None if not enabled)
    memory: Optional["Memory"] = Field(
        default=None,
        description="Agent memory store"
    )

    # Metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context metadata"
    )

    # Timestamps
    started_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    class Config:
        arbitrary_types_allowed = True

    def update_usage(self, usage: Usage) -> None:
        """Update cumulative usage statistics."""
        self.usage = self.usage.add(usage)
        self.updated_at = time.time()

    def add_cycle(self, cycle: ExecutionCycle) -> None:
        """Add an execution cycle to history."""
        self.execution_history.append(cycle)
        self.iteration += 1
        self.updated_at = time.time()

    def get_context_size(self) -> int:
        """Get current context size in tokens."""
        return self.usage.total_tokens

    def get_last_n_cycles(self, n: int = 3) -> List[ExecutionCycle]:
        """Get last N execution cycles for context."""
        return self.execution_history[-n:] if self.execution_history else []

    def get_last_iterations_summary(self, n: int = 2) -> List[Dict[str, Any]]:
        """Condensed view of recent iterations for LLM context."""
        summary: List[Dict[str, Any]] = []
        for cycle in self.execution_history[-n:]:
            summary.append(
                {
                    "iteration": cycle.iteration,
                    "thought": getattr(cycle.thought, "reasoning", str(cycle.thought)),
                    "tool_calls": [call.name for call in getattr(cycle, "tool_calls", [])],
                    "results": [
                        {"tool": result.tool_name, "success": result.success}
                        for result in getattr(cycle, "results", [])
                    ],
                }
            )
        return summary

    def clear_history(self) -> None:
        """Clear execution history (useful for long-running agents)."""
        self.execution_history.clear()
```

**Tests:**

```python
# tests/unit/test_context.py
import pytest
from opper_agent.base.context import Usage, AgentContext, ExecutionCycle

def test_usage_addition():
    u1 = Usage(requests=1, input_tokens=100, output_tokens=50, total_tokens=150)
    u2 = Usage(requests=2, input_tokens=200, output_tokens=100, total_tokens=300)
    combined = u1.add(u2)
    assert combined.requests == 3
    assert combined.total_tokens == 450

def test_agent_context_initialization():
    ctx = AgentContext(agent_name="TestAgent")
    assert ctx.agent_name == "TestAgent"
    assert ctx.iteration == 0
    assert len(ctx.execution_history) == 0

def test_context_add_cycle():
    ctx = AgentContext(agent_name="Test")
    cycle = ExecutionCycle(iteration=1)
    ctx.add_cycle(cycle)
    assert ctx.iteration == 1
    assert len(ctx.execution_history) == 1

def test_context_usage_tracking():
    ctx = AgentContext(agent_name="Test")
    ctx.update_usage(Usage(requests=1, total_tokens=100))
    ctx.update_usage(Usage(requests=1, total_tokens=200))
    assert ctx.usage.requests == 2
    assert ctx.usage.total_tokens == 300
```

### Task 1.3: Tool System (`tool.py`)

**Implementation:**

```python
# src/opper_agent/base/tool.py
from pydantic import BaseModel, Field
from typing import Dict, Any, Callable, Optional, List, Union, Sequence, Protocol, Tuple
from abc import ABC, abstractmethod
import inspect
import asyncio
import time

class ToolResult(BaseModel):
    """Standardized result from tool execution."""

    tool_name: str = Field(description="Name of the tool executed")
    success: bool = Field(description="Whether execution succeeded")
    result: Any = Field(description="Tool execution result")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    execution_time: float = Field(description="Execution time in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        arbitrary_types_allowed = True


class Tool(BaseModel, ABC):
    """
    Abstract base class for all tools.
    All tools must implement execute() method.
    """

    name: str = Field(description="Tool name")
    description: str = Field(description="Tool description")
    parameters: Dict[str, Any] = Field(description="Tool parameters schema")

    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        pass


class FunctionTool(Tool):
    """
    Tool that wraps a Python function.
    Handles both sync and async functions automatically.
    """

    func: Callable = Field(description="The wrapped function", exclude=True)

    def __init__(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        # Extract metadata from function
        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or f"Execute {func.__name__}"

        # Auto-extract parameters if not provided
        if parameters is None:
            parameters = self._extract_parameters(func)

        super().__init__(
            name=tool_name,
            description=tool_description,
            parameters=parameters,
            func=func,
        )

    def _extract_parameters(self, func: Callable) -> Dict[str, Any]:
        """Extract parameter information from function signature."""
        sig = inspect.signature(func)
        parameters = {}

        for param_name, param in sig.parameters.items():
            # Skip special parameters
            if param_name.startswith("_"):
                continue

            # Get type annotation
            param_type = "any"
            if param.annotation != inspect.Parameter.empty:
                if hasattr(param.annotation, "__name__"):
                    param_type = param.annotation.__name__
                else:
                    param_type = str(param.annotation)

            # Get default value
            default_info = ""
            if param.default != inspect.Parameter.empty:
                default_info = f" (default: {param.default})"

            parameters[param_name] = f"{param_type}{default_info}"

        return parameters

    async def execute(self, **kwargs) -> ToolResult:
        """Execute the wrapped function."""
        start_time = time.time()

        try:
            # Filter out special parameters
            filtered_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}

            # Check if function is async
            if asyncio.iscoroutinefunction(self.func):
                result = await self.func(**filtered_kwargs)
            else:
                # Run sync function in executor to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: self.func(**filtered_kwargs))

            execution_time = time.time() - start_time

            return ToolResult(
                tool_name=self.name,
                success=True,
                result=result,
                execution_time=execution_time,
            )

        except Exception as e:
            execution_time = time.time() - start_time

            return ToolResult(
                tool_name=self.name,
                success=False,
                result=None,
                error=str(e),
                execution_time=execution_time,
            )
```

#### Tool Providers
```python
from typing import Protocol, Sequence, Tuple

class ToolProvider(Protocol):
    """Expands into tools at runtime (e.g. MCP bundles)."""

    async def setup(self, agent: "BaseAgent") -> List[Tool]:
        """Connect resources and return concrete tools."""

    async def teardown(self) -> None:
        """Cleanup after the agent run finishes."""


def normalize_tools(
    raw: Sequence[Union[Tool, ToolProvider]]
) -> Tuple[List[Tool], List[ToolProvider]]:
    """Utility to split concrete tools from providers."""

    tools: List[Tool] = []
    providers: List[ToolProvider] = []
    for item in raw:
        if isinstance(item, Tool):
            tools.append(item)
        elif hasattr(item, "setup") and hasattr(item, "teardown"):
            providers.append(item)
        else:
            raise TypeError(f"Unsupported tool item: {item!r}")
    return tools, providers
```

**Tests:**

```python
# tests/unit/test_tool.py
import pytest
from opper_agent.base.tool import Tool, FunctionTool, ToolResult

def sync_add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

async def async_multiply(x: int, y: int) -> int:
    """Multiply two numbers."""
    return x * y

def failing_tool():
    """This tool always fails."""
    raise ValueError("Intentional failure")

@pytest.mark.asyncio
async def test_function_tool_sync():
    tool = FunctionTool(sync_add)
    result = await tool.execute(a=2, b=3)
    assert result.success
    assert result.result == 5

@pytest.mark.asyncio
async def test_function_tool_async():
    tool = FunctionTool(async_multiply)
    result = await tool.execute(x=3, y=4)
    assert result.success
    assert result.result == 12

@pytest.mark.asyncio
async def test_function_tool_error():
    tool = FunctionTool(failing_tool)
    result = await tool.execute()
    assert not result.success
    assert result.error is not None
    assert "Intentional failure" in result.error

def test_parameter_extraction():
    tool = FunctionTool(sync_add)
    assert "a" in tool.parameters
    assert "b" in tool.parameters
    assert "int" in tool.parameters["a"]
```

### Task 1.4: Hook System (`hooks.py`)

**Implementation:**

```python
# src/opper_agent/base/hooks.py
from typing import Dict, List, Callable, Any
from .context import AgentContext
import asyncio
import logging

logger = logging.getLogger(__name__)

# Type alias for hook functions
HookFunction = Callable[[AgentContext, ...], Any]

# Standard hook events
class HookEvents:
    """Standard hook event names."""

    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    AGENT_ERROR = "agent_error"

    LOOP_START = "loop_start"
    LOOP_END = "loop_end"

    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    TOOL_ERROR = "tool_error"

    LLM_CALL = "llm_call"
    LLM_RESPONSE = "llm_response"
    THINK_END = "think_end"


class HookManager:
    """
    Manages hook registration and execution.
    Supports both class-based and decorator-based hooks.
    """

    def __init__(self, verbose: bool = False):
        self.hooks: Dict[str, List[HookFunction]] = {}
        self.verbose = verbose

    def register(self, event: str, hook: HookFunction) -> None:
        """Register a hook function for an event."""
        if event not in self.hooks:
            self.hooks[event] = []
        self.hooks[event].append(hook)

        if self.verbose:
            logger.info(f"Registered hook for event: {event}")

    def register_multiple(self, hooks: List[tuple]) -> None:
        """Register multiple hooks at once. Each tuple is (event, hook_func)."""
        for event, hook in hooks:
            self.register(event, hook)

    async def trigger(self, event: str, context: AgentContext, **kwargs) -> None:
        """
        Trigger all hooks for an event.
        Hooks that fail don't stop execution.
        """
        if event not in self.hooks:
            return

        for hook_func in self.hooks[event]:
            try:
                # Handle both sync and async hooks
                if asyncio.iscoroutinefunction(hook_func):
                    await hook_func(context, **kwargs)
                else:
                    hook_func(context, **kwargs)

            except Exception as e:
                logger.warning(f"Hook '{event}' failed: {e}")
                # Don't break execution if hook fails

    def has_hooks(self, event: str) -> bool:
        """Check if any hooks are registered for an event."""
        return event in self.hooks and len(self.hooks[event]) > 0

    def clear_hooks(self, event: Optional[str] = None) -> None:
        """Clear hooks for a specific event, or all hooks if event is None."""
        if event:
            self.hooks.pop(event, None)
        else:
            self.hooks.clear()

    def get_hook_count(self) -> int:
        """Get total number of registered hooks."""
        return sum(len(hooks) for hooks in self.hooks.values())
```

**Tests:**

```python
# tests/unit/test_hooks.py
import pytest
from opper_agent.base.hooks import HookManager, HookEvents
from opper_agent.base.context import AgentContext

@pytest.mark.asyncio
async def test_hook_registration():
    manager = HookManager()

    async def test_hook(context: AgentContext):
        context.metadata["hook_called"] = True

    manager.register(HookEvents.AGENT_START, test_hook)
    assert manager.has_hooks(HookEvents.AGENT_START)

@pytest.mark.asyncio
async def test_hook_trigger():
    manager = HookManager()
    context = AgentContext(agent_name="Test")

    async def test_hook(context: AgentContext):
        context.metadata["value"] = 42

    manager.register(HookEvents.AGENT_START, test_hook)
    await manager.trigger(HookEvents.AGENT_START, context)

    assert context.metadata["value"] == 42

@pytest.mark.asyncio
async def test_hook_failure_doesnt_break():
    manager = HookManager()
    context = AgentContext(agent_name="Test")

    async def failing_hook(context: AgentContext):
        raise ValueError("Hook error")

    async def working_hook(context: AgentContext):
        context.metadata["success"] = True

    manager.register(HookEvents.AGENT_START, failing_hook)
    manager.register(HookEvents.AGENT_START, working_hook)

    # Should not raise, and working hook should execute
    await manager.trigger(HookEvents.AGENT_START, context)
    assert context.metadata.get("success") == True

@pytest.mark.asyncio
async def test_multiple_hooks():
    manager = HookManager()
    context = AgentContext(agent_name="Test")
    counter = {"count": 0}

    async def inc_hook(context: AgentContext):
        counter["count"] += 1

    manager.register(HookEvents.LOOP_START, inc_hook)
    manager.register(HookEvents.LOOP_START, inc_hook)

    await manager.trigger(HookEvents.LOOP_START, context)
    assert counter["count"] == 2
```

### Task 1.5: Decorators (`utils/decorators.py`)

**Implementation:**

```python
# src/opper_agent/utils/decorators.py
from typing import Callable, Optional, Dict, Any
from ..base.tool import FunctionTool

def tool(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
) -> FunctionTool:
    """
    Decorator to convert a function into a Tool.

    Usage:
        @tool
        def add(a: int, b: int) -> int:
            '''Add two numbers.'''
            return a + b

        @tool(name="custom_name", description="Custom desc")
        def my_func(x: str) -> str:
            return x.upper()
    """

    def decorator(f: Callable) -> FunctionTool:
        return FunctionTool(f, name, description, parameters)

    if func is None:
        # Called with arguments: @tool(name="something")
        return decorator
    else:
        # Called without arguments: @tool
        return decorator(func)


def hook(event_name: str) -> Callable:
    """
    Decorator to mark a function as a hook for a specific event.

    Usage:
        @hook("agent_start")
        async def on_start(context: AgentContext, agent: Agent):
            print("Agent starting!")

        agent = Agent(name="Test", hooks=[on_start])
    """

    def decorator(func: Callable) -> Callable:
        # Mark the function with hook metadata
        func._hook_event = event_name
        return func

    return decorator
```

**Tests:**

```python
# tests/unit/test_decorators.py
import pytest
from opper_agent.utils.decorators import tool, hook
from opper_agent.base.tool import FunctionTool

def test_tool_decorator_without_args():
    @tool
    def add(a: int, b: int) -> int:
        return a + b

    assert isinstance(add, FunctionTool)
    assert add.name == "add"

def test_tool_decorator_with_args():
    @tool(name="custom_add", description="Custom description")
    def add(a: int, b: int) -> int:
        return a + b

    assert isinstance(add, FunctionTool)
    assert add.name == "custom_add"
    assert add.description == "Custom description"

def test_hook_decorator():
    @hook("agent_start")
    async def on_start(context):
        pass

    assert hasattr(on_start, "_hook_event")
    assert on_start._hook_event == "agent_start"
```

---

## Phase 2: BaseAgent Implementation (Week 2-3)

### Task 2.1: BaseAgent Abstract Class

**Implementation:**

```python
# src/opper_agent/base/agent.py
from abc import ABC, abstractmethod
from typing import List, Optional, Any, Type, Union, Sequence, Dict, Callable
from pydantic import BaseModel
from opperai import Opper
import os

from .context import AgentContext, Usage
from .tool import Tool, FunctionTool
from .hooks import HookManager, HookEvents

class BaseAgent(ABC):
    """
    Abstract base class for all agents.

    All agents must implement:
    - process(): Main entry point
    - _run_loop(): Core execution logic

    Provides:
    - Opper client integration
    - Hook system
    - Tool management
    - Tracing support
    """

    def __init__(
        self,
        name: str,
        description: Optional[str] = None,
        instructions: Optional[str] = None,
        tools: Optional[Sequence[Union[Tool, "ToolProvider"]]] = None,
        input_schema: Optional[Type[BaseModel]] = None,
        output_schema: Optional[Type[BaseModel]] = None,
        hooks: Optional[Union[List[Callable], HookManager]] = None,
        max_iterations: int = 25,
        verbose: bool = False,
        model: Optional[str] = None,
        opper_api_key: Optional[str] = None,
    ):
        """
        Initialize base agent.

        Args:
            name: Agent name
            description: Agent description
            instructions: Behavior instructions
            tools: List of tools available to agent
            input_schema: Pydantic model for input validation
            output_schema: Pydantic model for output validation
            hooks: Hook functions or HookManager
            max_iterations: Maximum execution iterations
            verbose: Enable verbose logging
            model: Default model for LLM calls
            opper_api_key: Opper API key (or from env)
        """
        # Basic config
        self.name = name
        self.description = description or f"Agent: {name}"
        self.instructions = instructions
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.model = model or "anthropic/claude-3.5-sonnet"

        # Schemas
        self.input_schema = input_schema
        self.output_schema = output_schema

        # Tools
        self.base_tools: List[Tool] = []
        self.tool_providers: List["ToolProvider"] = []
        self.active_provider_tools: Dict["ToolProvider", List[Tool]] = {}
        self.tools: List[Tool] = []
        self._initialize_tools(tools or [])

        # Opper client
        api_key = opper_api_key or os.getenv("OPPER_API_KEY")
        if not api_key:
            raise ValueError("OPPER_API_KEY not found in environment or parameters")
        self.opper = Opper(http_bearer=api_key)

        # Hook system
        self.hook_manager = self._setup_hooks(hooks)

        # Context (initialized per execution)
        self.context: Optional[AgentContext] = None

    def _setup_hooks(
        self,
        hooks: Optional[Union[List[Callable], HookManager]]
    ) -> HookManager:
        """Setup hook manager from hooks parameter."""
        if isinstance(hooks, HookManager):
            return hooks

        manager = HookManager(verbose=self.verbose)

        if hooks:
            for hook_func in hooks:
                if hasattr(hook_func, "_hook_event"):
                    event = hook_func._hook_event
                    manager.register(event, hook_func)

        return manager

    def _initialize_tools(self, raw_tools: Sequence[Union[Tool, "ToolProvider"]]) -> None:
        """Separate concrete tools from providers."""
        for item in raw_tools:
            if isinstance(item, Tool):
                self.base_tools.append(item)
                self.tools.append(item)
            elif hasattr(item, "setup") and hasattr(item, "teardown"):
                self.tool_providers.append(item)
            else:
                raise TypeError(f"Unsupported tool type: {item!r}")

    async def _activate_tool_providers(self) -> None:
        """Connect providers and register their tools."""
        for provider in self.tool_providers:
            provided = await provider.setup(self)
            self.active_provider_tools[provider] = provided
            for tool in provided:
                self.add_tool(tool, as_base=False)

    async def _deactivate_tool_providers(self) -> None:
        """Disconnect providers and remove their tools."""
        for provider, tools in self.active_provider_tools.items():
            for tool in tools:
                if tool in self.tools:
                    self.tools.remove(tool)
            await provider.teardown()
        self.active_provider_tools.clear()

    # Tool management
    def add_tool(self, tool: Tool, *, as_base: bool = True) -> None:
        """Add a tool to the agent."""
        if as_base:
            self.base_tools.append(tool)
        if tool not in self.tools:
            self.tools.append(tool)

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get tool by name."""
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

    def list_tools(self) -> List[str]:
        """Get list of tool names."""
        return [tool.name for tool in self.tools]

    # Tracing
    def start_trace(self, name: str, input_data: Any) -> Any:
        """Start a new Opper trace."""
        return self.opper.spans.create(
            name=name,
            input=str(input_data) if input_data else None
        )

    # Abstract methods - must be implemented by subclasses
    @abstractmethod
    async def process(self, input: Any) -> Any:
        """
        Main entry point for agent execution.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    async def _run_loop(self, goal: Any) -> Any:
        """
        Core execution loop.
        Must be implemented by subclasses.
        """
        pass

    # Multi-agent support
    def as_tool(
        self,
        tool_name: Optional[str] = None,
        description: Optional[str] = None
    ) -> FunctionTool:
        """
        Convert this agent into a tool for use by other agents.

        Args:
            tool_name: Custom tool name (default: agent name)
            description: Custom description (default: agent description)

        Returns:
            FunctionTool that wraps this agent
        """
        import asyncio
        import concurrent.futures

        tool_name = tool_name or f"{self.name}_agent"
        description = description or f"Delegate to {self.name}: {self.description}"

        def agent_tool(task: str, **kwargs) -> Any:
            """Tool function that delegates to agent."""

            async def call_agent():
                input_data = {"task": task, **kwargs}
                if self.instructions:
                    input_data["instructions"] = self.instructions
                return await self.process(input_data)

            # Handle event loop
            try:
                loop = asyncio.get_running_loop()

                def run_in_thread():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(call_agent())
                    finally:
                        new_loop.close()

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_in_thread)
                    return future.result(timeout=60)

            except RuntimeError:
                return asyncio.run(call_agent())

        return FunctionTool(
            func=agent_tool,
            name=tool_name,
            description=description,
            parameters={"task": "str - Task to delegate to agent"}
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', tools={len(self.tools)})"
```

**Tests:**

```python
# tests/unit/test_base_agent.py
import pytest
from opper_agent.base.agent import BaseAgent
from opper_agent.base.tool import FunctionTool
from opper_agent.utils.decorators import tool

# Concrete implementation for testing
class TestAgent(BaseAgent):
    async def process(self, input: Any) -> Any:
        return f"Processed: {input}"

    async def _run_loop(self, goal: Any) -> Any:
        return goal

@tool
def dummy_tool() -> str:
    return "result"

def test_base_agent_initialization():
    agent = TestAgent(name="Test", tools=[dummy_tool])
    assert agent.name == "Test"
    assert len(agent.tools) == 1

def test_tool_management():
    agent = TestAgent(name="Test")
    agent.add_tool(dummy_tool)
    assert len(agent.tools) == 1
    assert agent.get_tool("dummy_tool") is not None

@pytest.mark.asyncio
async def test_agent_as_tool():
    agent = TestAgent(name="SubAgent")
    tool = agent.as_tool()
    assert isinstance(tool, FunctionTool)
    assert "SubAgent" in tool.name

@pytest.mark.asyncio
async def test_nested_agent_execution():
    """Test agent-as-tool in nested event loop scenarios"""
    sub_agent = TestAgent(name="SubAgent", tools=[dummy_tool])
    main_agent = TestAgent(name="MainAgent", tools=[sub_agent.as_tool()])

    result = await main_agent.process("test task")
    assert result is not None

@pytest.mark.asyncio
async def test_agent_as_tool_timeout():
    """Test timeout handling for agent tools"""
    # Test that agent-as-tool respects timeout and fails gracefully
    pass

@pytest.mark.asyncio
async def test_agent_as_tool_error_propagation():
    """Test error propagation from sub-agent to parent"""
    # Ensure errors bubble up correctly through agent hierarchy
    pass
```

**Note on `as_tool()` Implementation:**

The current event loop handling approach (using ThreadPoolExecutor for nested event loops) is the most robust solution for our multi-agent requirements. This ensures:
- Sub-agents can be called from within async agent execution
- Proper isolation between parent/child agent contexts
- No event loop conflicts

Critical testing areas:
- Nested agent calls (agent calling agent calling agent)
- Error propagation through agent hierarchy
- Timeout behavior at each level
- Context isolation between parent and child agents

---

## Phase 3: Main Agent Implementation (Week 3-4)

### Task 3.1: Thought Schema

```python
# src/opper_agent/core/schemas.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ToolCall(BaseModel):
    """Represents a single tool invocation."""

    name: str = Field(description="Tool name to call")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters to pass to tool"
    )
    reasoning: str = Field(description="Why this tool should be called")


class Thought(BaseModel):
    """
    Agent's reasoning and action plan.

    Key insight: Empty tool_calls list indicates task completion.
    """

    reasoning: str = Field(description="Analysis of current situation")
    tool_calls: List[ToolCall] = Field(
        default=[],
        description="Tools to call (empty means task is complete)"
    )
    user_message: str = Field(
        default="Working on it...",
        description="Status message for user"
    )
    memory_updates: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Memory writes the model wants to perform (key -> payload)"
    )
```

### Task 3.2: Agent Class

```python
# src/opper_agent/core/agent.py
from typing import Any, List, Optional
import time

from ..base.agent import BaseAgent
from ..base.context import AgentContext, ExecutionCycle
from ..memory.memory import Memory
from ..base.hooks import HookEvents
from ..base.tool import ToolResult
from .schemas import Thought, ToolCall

class Agent(BaseAgent):
    """
    Main agent implementation using 'while tools > 0' loop.

    Loop Logic:
    - Think: Decide next actions
    - If tool_calls > 0: Execute tools and continue
    - If tool_calls == 0: Generate final result
    """

    def __init__(self, *args, **kwargs):
        """Initialize agent with additional options."""
        # Extract Agent-specific options
        self.clean_tool_results = kwargs.pop("clean_tool_results", False)
        self.enable_memory = kwargs.pop("enable_memory", False)

        super().__init__(*args, **kwargs)

    async def process(self, input: Any) -> Any:
        """
        Main entry point for agent execution.

        Args:
            input: Goal/task to process (validated against input_schema)

        Returns:
            Result (validated against output_schema if specified)
        """
        # Validate input
        if self.input_schema:
            if isinstance(input, dict):
                input = self.input_schema(**input)
            elif not isinstance(input, self.input_schema):
                input = self.input_schema(input=input)

        # Initialize context
        self.context = AgentContext(
            agent_name=self.name,
            goal=input,
            memory=Memory() if self.enable_memory else None
        )

        trace = None

        try:
            await self._activate_tool_providers()

            # Start trace
            trace = self.start_trace(
                name=f"{self.name}_execution",
                input_data=input
            )
            self.context.trace_id = trace.id

            # Trigger: agent_start
            await self.hook_manager.trigger(
                HookEvents.AGENT_START,
                self.context,
                agent=self
            )

            # Run main loop
            result = await self._run_loop(input)

            # Trigger: agent_end
            await self.hook_manager.trigger(
                HookEvents.AGENT_END,
                self.context,
                agent=self,
                result=result
            )

            if trace:
                # Update trace output after successful completion
                self.opper.spans.update(
                    span_id=trace.id,
                    output=str(result)
                )

            return result

        except Exception as e:
            # Trigger: agent_error
            await self.hook_manager.trigger(
                HookEvents.AGENT_ERROR,
                self.context,
                agent=self,
                error=e
            )
            raise
        finally:
            await self._deactivate_tool_providers()

    async def _run_loop(self, goal: Any) -> Any:
        """
        Main execution loop: while tools > 0

        Returns when thought.tool_calls is empty.
        """
        iteration = 0

        while iteration < self.max_iterations:
            # Trigger: loop_start
            await self.hook_manager.trigger(
                HookEvents.LOOP_START,
                self.context,
                agent=self
            )

            if self.verbose:
                print(f"\n--- Iteration {iteration + 1}/{self.max_iterations} ---")

            # Think
            thought = await self._think(goal)

            if self.verbose:
                print(f"💭 Reasoning: {thought.reasoning}")
                print(f"🔧 Tool calls: {len(thought.tool_calls)}")

            # Check if done
            if not thought.tool_calls or len(thought.tool_calls) == 0:
                if self.verbose:
                    print("✅ No more tool calls - generating final result")
                break

            # Execute tool calls
            results = []
            for tool_call in thought.tool_calls:
                result = await self._execute_tool(tool_call)
                results.append(result)

            # Record cycle
            cycle = ExecutionCycle(
                iteration=iteration,
                thought=thought,
                tool_calls=thought.tool_calls,
                results=results
            )
            self.context.add_cycle(cycle)

            # Update memory if needed
            if self.enable_memory and thought.memory_updates:
                for key, update in thought.memory_updates.items():
                    await self.context.memory.write(
                        key=key,
                        value=update.get("value"),
                        description=update.get("description"),
                        metadata=update.get("metadata"),
                    )

            # Trigger: loop_end
            await self.hook_manager.trigger(
                HookEvents.LOOP_END,
                self.context,
                agent=self
            )

            iteration += 1

        # Generate final result
        result = await self._generate_final_result(goal)
        return result

    async def _think(self, goal: Any) -> Thought:
        """Call LLM to reason about next actions."""

        # Optional: build lightweight memory snapshot for LLM
        memory_snapshot = None
        if self.enable_memory and self.context.memory and self.context.memory.has_entries():
            memory_snapshot = await self._prepare_memory_snapshot(goal)

        # Build context
        context = {
            "goal": str(goal),
            "agent_description": self.description,
            "instructions": self.instructions or "No specific instructions.",
            "available_tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
                for tool in self.tools
            ],
            "execution_history": [
                {
                    "iteration": cycle.iteration,
                    "thought": cycle.thought.reasoning if hasattr(cycle.thought, "reasoning") else str(cycle.thought),
                    "results": [
                        {"tool": r.tool_name, "success": r.success}
                        for r in cycle.results
                    ]
                }
                for cycle in self.context.get_last_n_cycles(3)
            ],
            "current_iteration": self.context.iteration + 1,
            "max_iterations": self.max_iterations,
            "memory": memory_snapshot
        }

        instructions = """You are in a Think-Act reasoning loop.

YOUR TASK:
1. Analyze the current situation
2. Decide if the goal is complete or more actions are needed
3. If more actions needed: specify tools to call
4. If goal complete: return empty tool_calls list

IMPORTANT:
- Return empty tool_calls array when task is COMPLETE
- Only use available tools
- Provide clear reasoning for each decision
"""

        # Trigger: llm_call
        await self.hook_manager.trigger(
            HookEvents.LLM_CALL,
            self.context,
            agent=self,
            call_type="think"
        )

        # Call Opper
        response = await self.opper.call(
            name="think",
            instructions=instructions,
            input=context,
            output_schema=Thought,
            model=self.model,
            parent_span_id=self.context.span_id
        )

        # Track usage
        # (Opper returns usage in response)
        # self.context.update_usage(...)

        # Trigger: llm_response
        await self.hook_manager.trigger(
            HookEvents.LLM_RESPONSE,
            self.context,
            agent=self,
            call_type="think",
            response=response
        )

        thought = Thought(**response.json_payload)

        # Trigger: think_end
        await self.hook_manager.trigger(
            HookEvents.THINK_END,
            self.context,
            agent=self,
            thought=thought
        )

        return thought

    async def _execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a single tool call."""

        if self.verbose:
            print(f"  ⚡ Calling {tool_call.name} with {tool_call.parameters}")

        tool = self.get_tool(tool_call.name)
        if not tool:
            return ToolResult(
                tool_name=tool_call.name,
                success=False,
                result=None,
                error=f"Tool '{tool_call.name}' not found",
                execution_time=0.0
            )

        # Trigger: tool_call
        await self.hook_manager.trigger(
            HookEvents.TOOL_CALL,
            self.context,
            agent=self,
            tool=tool,
            parameters=tool_call.parameters
        )

        # Execute
        result = await tool.execute(**tool_call.parameters)

        # Trigger: tool_result
        await self.hook_manager.trigger(
            HookEvents.TOOL_RESULT,
            self.context,
            agent=self,
            tool=tool,
            result=result
        )

        if self.verbose:
            status = "✅" if result.success else "❌"
            print(f"  {status} Result: {result.result}")

        return result

    async def _generate_final_result(self, goal: Any) -> Any:
        """Generate final structured result."""

        if self.verbose:
            print("\n🎯 Generating final result...")

        context = {
            "goal": str(goal),
            "instructions": self.instructions,
            "execution_history": [
                {
                    "iteration": cycle.iteration,
                    "actions_taken": [r.tool_name for r in cycle.results],
                    "results": [
                        {"tool": r.tool_name, "result": str(r.result)[:200]}
                        for r in cycle.results if r.success
                    ]
                }
                for cycle in self.context.execution_history
            ],
            "total_iterations": self.context.iteration
        }

        instructions = """Generate the final result based on the execution history.
Follow any instructions provided for formatting and style."""

        response = await self.opper.call(
            name="generate_final_result",
            instructions=instructions,
            input=context,
            output_schema=self.output_schema,
            model=self.model,
            parent_span_id=self.context.trace_id
        )

        if self.output_schema:
            return self.output_schema(**response.json_payload)
        return response.message
```

### Task 3.3: Memory System

> **NOTE (To be reviewed in future):** The memory system was implemented with a **simpler LLM-driven approach** instead of the planned "memory router" pattern. The LLM directly requests memory reads/writes through the Thought schema, making it more flexible and transparent.

**Actual Implementation:**

The memory system uses a catalog-based approach where:
1. The LLM sees a **memory catalog** (keys + descriptions) in context
2. The LLM **requests specific keys** via `Thought.memory_reads` field
3. Requested memory is **loaded into context** for the next iteration
4. The LLM **writes memory** via `Thought.memory_updates` field

**Memory Core (`src/opper_agent/memory/memory.py`):**

```python
# Implementation matches the planned version (no changes needed)
class MemoryEntry(BaseModel):
    """Single memory slot with metadata."""
    key: str
    description: str
    value: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)
    last_accessed: float = Field(default_factory=time.time)

class Memory(BaseModel):
    """Fast in-memory store that exposes a catalog to the LLM."""
    store: Dict[str, MemoryEntry] = Field(default_factory=dict)
    # ... (methods: has_entries, list_entries, read, write, clear)
```

**Thought Schema Extension (`src/opper_agent/core/schemas.py`):**

```python
class Thought(BaseModel):
    """Agent's reasoning and action plan."""
    reasoning: str
    tool_calls: List[ToolCall] = []
    user_message: str = "Working on it..."

    # Memory fields (direct LLM control)
    memory_reads: List[str] = Field(
        default_factory=list,
        description="Memory keys to load for this iteration"
    )
    memory_updates: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Memory writes (key -> {value, description, metadata})"
    )
```

**Agent Integration (`src/opper_agent/core/agent.py`):**

```python
# In _think() - provide memory catalog to LLM
context = {
    # ... other context fields
    "memory_catalog": await self.context.memory.list_entries() if self.enable_memory else None,
    "loaded_memory": self.context.metadata.get("current_memory", None),
}

# In _run_loop() - handle memory reads
if self.enable_memory and thought.memory_reads:
    memory_data = await self.context.memory.read(thought.memory_reads)
    self.context.metadata["current_memory"] = memory_data
    # (with proper span tracking)

# In _run_loop() - handle memory writes
if self.enable_memory and thought.memory_updates:
    for key, update in thought.memory_updates.items():
        await self.context.memory.write(
            key=key,
            value=update.get("value"),
            description=update.get("description"),
            metadata=update.get("metadata"),
        )
    # (with proper span tracking)

# Exit condition updated
has_memory_reads = self.enable_memory and thought.memory_reads and len(thought.memory_reads) > 0
if not has_tool_calls and not has_memory_reads:
    break  # Done
```

**Key Differences from Original Plan:**

1. **No MemoryDecision schema** - LLM controls memory directly through Thought
2. **No _prepare_memory_snapshot() method** - Memory catalog always provided
3. **No _memory_router_instructions()** - Instructions integrated into main think prompt
4. **Simpler flow** - LLM decides what to read/write in the same reasoning step
5. **Memory reads extend iteration** - Agent continues if memory_reads present (even without tool_calls)

**Tests:**

```python
# tests/unit/test_memory.py - Basic memory tests (unchanged)
@pytest.mark.asyncio
async def test_memory_read_write():
    memory = Memory()
    await memory.write("project", {"status": "in_progress"}, description="Project snapshot")
    catalog = await memory.list_entries()
    assert catalog[0]["key"] == "project"
    payload = await memory.read(["project"])
    assert payload["project"]["status"] == "in_progress"

# Additional integration tests needed for:
# - LLM requesting memory reads through Thought.memory_reads
# - LLM writing memory through Thought.memory_updates
# - Memory persisting across iterations
# - Span tracking for memory operations
```

**Future Review Topics:**

- Should we add back the "memory router" pattern for more selective loading?
- Should memory_reads and tool_calls be mutually exclusive?
- Should loaded memory be cleared between iterations or persist?
- Should we add memory pruning/expiration strategies?

### Task 4.1: MCP Tool Provider

**Implementation:**

```python
# src/opper_agent/mcp/config.py
from pydantic import BaseModel, Field
from typing import Dict, List, Literal, Optional

class MCPServerConfig(BaseModel):
    """Declarative configuration for an MCP server."""

    name: str
    transport: Literal["stdio", "http-sse"]
    url: Optional[str] = None
    command: Optional[str] = None
    args: List[str] = Field(default_factory=list)
    env: Dict[str, str] = Field(default_factory=dict)
    timeout: float = 30.0
```

```python
# src/opper_agent/mcp/provider.py
from typing import Sequence, Optional, Dict, List, Any

from ..base.tool import FunctionTool, Tool, ToolProvider
from ..base.agent import BaseAgent
from .config import MCPServerConfig
from .client import MCPClient, MCPTool

class MCPToolProvider(ToolProvider):
    """ToolProvider wrapper around one or more MCP servers."""

    def __init__(self, configs: Sequence[MCPServerConfig], *, name_prefix: Optional[str] = None) -> None:
        self.configs = list(configs)
        self.name_prefix = name_prefix
        self.clients: Dict[str, MCPClient] = {}

    async def setup(self, agent: BaseAgent) -> List[Tool]:
        tools: List[Tool] = []
        for config in self.configs:
            client = MCPClient.from_config(config)
            await client.connect()
            self.clients[config.name] = client

            for mcp_tool in await client.list_tools():
                tools.append(self._wrap_tool(config.name, mcp_tool))
        return tools

    async def teardown(self) -> None:
        for client in self.clients.values():
            await client.disconnect()
        self.clients.clear()

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
    """Helper so callers can write tools = [mcp(config), local_tool]."""
    if not configs:
        raise ValueError("At least one MCPServerConfig is required")
    return MCPToolProvider(configs, name_prefix=name_prefix)
```

**Usage Example:**

```python
from opper_agent.mcp.config import MCPServerConfig
from opper_agent.mcp.provider import mcp

search = MCPServerConfig(name="search", transport="http-sse", url="https://mcp.example.com")

agent = Agent(
    name="SearchAgent",
    tools=[
        mcp(search),
        my_local_tool,
    ],
)
```

During `process()` the agent will activate the provider, connect to MCP servers, expose wrapped tools, and automatically disconnect during teardown.
