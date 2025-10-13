"""
Base agent abstract class.

This module provides the abstract base class that all agents inherit from,
defining the core interface and common functionality.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Any, Type, Union, Sequence, Dict, Callable
from pydantic import BaseModel
from opperai import Opper
import os
import asyncio
import concurrent.futures

from .context import AgentContext
from .tool import Tool, FunctionTool
from .hooks import HookManager
from ..utils.logging import AgentLogger, SimpleLogger


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
        logger: Optional[AgentLogger] = None,
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
            verbose: Enable verbose logging (legacy, use logger instead)
            logger: Custom logger instance (defaults to SimpleLogger if verbose=True)
            model: Default model for LLM calls
            opper_api_key: Opper API key (or from env)
        """
        # Basic config
        self.name = name
        self.description = description or f"Agent: {name}"
        self.instructions = instructions
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.model = model or "gcp/gemini-flash-latest"

        # Logger setup
        if logger is not None:
            self.logger = logger
        elif verbose:
            self.logger = SimpleLogger()
        else:
            self.logger = None

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
        self, hooks: Optional[Union[List[Callable], HookManager]]
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

    def _initialize_tools(
        self, raw_tools: Sequence[Union[Tool, "ToolProvider"]]
    ) -> None:
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

    # Abstract methods - must be implemented by subclasses
    @abstractmethod
    async def process(self, input: Any, _parent_span_id: Optional[str] = None) -> Any:
        """
        Main entry point for agent execution.
        Must be implemented by subclasses.

        Args:
            input: Goal/task to process
            _parent_span_id: Optional parent span ID for nested agent calls
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
        self, tool_name: Optional[str] = None, description: Optional[str] = None
    ) -> FunctionTool:
        """
        Convert this agent into a tool for use by other agents.

        If the agent has an input_schema, the schema's fields will be exposed
        as tool parameters for better LLM understanding.

        Args:
            tool_name: Custom tool name (default: agent name)
            description: Custom description (default: agent description)

        Returns:
            FunctionTool that wraps this agent
        """
        tool_name = tool_name or f"{self.name}_agent"
        description = description or f"Delegate to {self.name}: {self.description}"

        # Extract parameters from input_schema if available
        parameters = {"task": "str - Task to delegate to agent"}
        if self.input_schema:
            try:
                # Get the full schema from the Pydantic model
                schema = self.input_schema.model_json_schema()
                if "properties" in schema:
                    # Use the schema properties directly for better structure
                    parameters = schema["properties"]
            except Exception:
                # Fall back to simple task parameter if schema extraction fails
                pass

        def agent_tool(
            task: str = None, _parent_span_id: Optional[str] = None, **kwargs
        ) -> Any:
            """Tool function that delegates to agent."""

            async def call_agent():
                # If input_schema exists and we have kwargs, use them directly
                if self.input_schema and kwargs:
                    input_data = kwargs
                    # Add task if provided and not already in kwargs
                    if task and "task" not in kwargs:
                        input_data["task"] = task
                else:
                    # Fall back to simple task-based input
                    input_data = {"task": task or "", **kwargs}
                    if self.instructions:
                        input_data["instructions"] = self.instructions

                return await self.process(input_data, _parent_span_id=_parent_span_id)

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
            parameters=parameters,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', tools={len(self.tools)})"
