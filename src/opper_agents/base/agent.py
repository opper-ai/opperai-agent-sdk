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
from .tool import Tool, FunctionTool, ToolProvider
from .hooks import HookManager
from ..utils.logging import AgentLogger, SimpleLogger
from ..utils.version import get_user_agent


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
        tools: Optional[Sequence[Union[Tool, ToolProvider]]] = None,
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
            self.logger: Optional[AgentLogger] = logger
        elif verbose:
            self.logger = SimpleLogger()
        else:
            self.logger = None

        # Schemas
        self.input_schema = input_schema
        self.output_schema = output_schema

        # Tools
        self.base_tools: List[Tool] = []
        self.tool_providers: List[ToolProvider] = []
        self.active_provider_tools: Dict[ToolProvider, List[Tool]] = {}
        self.tools: List[Tool] = []
        self._initialize_tools(tools or [])

        # Opper client with custom User-Agent
        api_key = opper_api_key or os.getenv("OPPER_API_KEY")
        if not api_key:
            raise ValueError("OPPER_API_KEY not found in environment or parameters")

        # Create Opper client
        self.opper = Opper(http_bearer=api_key)

        # Override the User-Agent in the SDK configuration
        # The Opper SDK ignores client headers and uses its own user_agent from SDKConfiguration
        self.opper.sdk_configuration.user_agent = get_user_agent()

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

    def _initialize_tools(self, raw_tools: Sequence[Union[Tool, ToolProvider]]) -> None:
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
            task: Optional[str] = None,
            _parent_span_id: Optional[str] = None,
            **kwargs: Any,
        ) -> Any:
            """Tool function that delegates to agent."""

            async def call_agent() -> Any:
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
                asyncio.get_running_loop()

                def run_in_thread() -> Any:
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

        tool = FunctionTool(
            func=agent_tool,
            name=tool_name,
            description=description,
            parameters=parameters,
        )

        # Store reference to wrapped agent for visualization
        tool._wrapped_agent = self

        return tool

    def visualize_flow(
        self, output_path: Optional[str] = None, _visited: Optional[set] = None
    ) -> str:
        """
        Generate a Mermaid diagram visualizing the agent's flow and structure.

        Shows:
        - Agent as main node
        - Tools (with distinction between function tools and sub-agents)
        - Sub-agent structures recursively
        - Input/output schemas if defined
        - Hooks if registered
        - Tool providers (MCP, etc.)

        Args:
            output_path: Optional path to save the diagram markdown file.
                        If provided, saves to file and returns the path.
                        If None, returns the Mermaid markdown string.
            _visited: Internal parameter to track visited agents (prevents cycles)

        Returns:
            Mermaid diagram as markdown string, or file path if saved.

        Example:
            ```python
            agent = Agent(name="MyAgent", tools=[search_tool])
            diagram = agent.visualize_flow()
            print(diagram)  # View the Mermaid markdown

            # Or save to file
            agent.visualize_flow(output_path="agent_flow.md")
            ```
        """
        if _visited is None:
            _visited = set()

        lines = []
        is_root = len(_visited) == 0

        if is_root:
            lines.extend(["```mermaid", "graph TB"])

        # Sanitize functions
        def sanitize_id(name: str) -> str:
            """Sanitize node IDs for Mermaid."""
            return (
                name.replace(" ", "_")
                .replace("-", "_")
                .replace(".", "_")
                .replace("(", "")
                .replace(")", "")
            )

        def sanitize_label(text: str) -> str:
            """Sanitize labels for Mermaid (remove problematic characters)."""
            # Remove or replace characters that break Mermaid
            text = text.replace('"', "'")
            text = text.replace("\n", " ")
            text = text.replace("`", "")
            # Take only first line if multiline
            text = text.split(".")[0].strip()
            # Limit length
            if len(text) > 45:
                text = text[:42] + "..."
            return text

        agent_id = sanitize_id(self.name)

        # Avoid cycles
        if agent_id in _visited:
            return ""
        _visited.add(agent_id)

        # Main agent node with description
        agent_label = f"🤖 {self.name}"
        if self.description and self.description != f"Agent: {self.name}":
            desc = sanitize_label(self.description)
            agent_label += f"<br/><i>{desc}</i>"
        lines.append(f'    {agent_id}["{agent_label}"]:::agent')

        # Input/Output schemas
        if self.input_schema:
            schema_id = f"{agent_id}_input"
            schema_name = self.input_schema.__name__
            lines.append(f'    {schema_id}["Input: {schema_name}"]:::schema')
            lines.append(f"    {schema_id} --> {agent_id}")

        if self.output_schema:
            schema_id = f"{agent_id}_output"
            schema_name = self.output_schema.__name__
            lines.append(f'    {schema_id}["Output: {schema_name}"]:::schema')
            lines.append(f"    {agent_id} --> {schema_id}")

        # Hooks
        if self.hook_manager.get_hook_count() > 0:
            hook_id = f"{agent_id}_hooks"
            hook_events = list(self.hook_manager.hooks.keys())
            # Show which hooks are registered
            hook_names = ", ".join(hook_events[:3])  # Show first 3
            if len(hook_events) > 3:
                hook_names += f" +{len(hook_events) - 3}"
            lines.append(f'    {hook_id}["🪝 {hook_names}"]:::hook')
            lines.append(f"    {agent_id} -.-> {hook_id}")

        # Tool providers (MCP, etc.)
        if self.tool_providers:
            for i, provider in enumerate(self.tool_providers):
                provider_id = f"{agent_id}_provider_{i}"
                provider_name = provider.__class__.__name__
                lines.append(f'    {provider_id}["{provider_name}"]:::provider')
                lines.append(f"    {agent_id} --> {provider_id}")

        # Tools
        for tool in self.base_tools:
            tool_id = sanitize_id(f"{agent_id}_{tool.name}")

            # Check if this tool wraps an agent
            wrapped_agent = getattr(tool, "_wrapped_agent", None)

            if wrapped_agent:
                # Recursively visualize sub-agent
                sub_agent_id = sanitize_id(wrapped_agent.name)
                lines.append(f"    {agent_id} --> {sub_agent_id}")

                # Recursively add sub-agent structure
                sub_lines = wrapped_agent.visualize_flow(_visited=_visited).split("\n")
                # Filter out mermaid wrapper lines
                sub_lines = [
                    line
                    for line in sub_lines
                    if line
                    and "```mermaid" not in line
                    and "```" not in line.strip()
                    and "%% Styling" not in line
                    and "classDef" not in line
                ]
                lines.extend(sub_lines)
            else:
                # Regular function tool
                tool_label = f"⚙️ {sanitize_label(tool.name)}"
                if tool.description:
                    desc = sanitize_label(tool.description)
                    tool_label += f"<br/><i>{desc}</i>"
                lines.append(f'    {tool_id}["{tool_label}"]:::tool')
                lines.append(f"    {agent_id} --> {tool_id}")

        # Add styling (only for root call)
        if is_root:
            lines.append("")
            lines.append("    %% Styling - Opper Brand Colors")
            lines.append(
                "    classDef agent fill:#8CF0DC,stroke:#1B2E40,stroke-width:3px,color:#1B2E40"
            )
            lines.append(
                "    classDef tool fill:#FFD7D7,stroke:#3C3CAF,stroke-width:2px,color:#1B2E40"
            )
            lines.append(
                "    classDef schema fill:#F8F8F8,stroke:#3C3CAF,stroke-width:2px,color:#1B2E40"
            )
            lines.append(
                "    classDef hook fill:#FFB186,stroke:#3C3CAF,stroke-width:2px,color:#1B2E40"
            )
            lines.append(
                "    classDef provider fill:#8CECF2,stroke:#1B2E40,stroke-width:2px,color:#1B2E40"
            )
            lines.append("```")

        mermaid_markdown = "\n".join(lines)

        # Save to file if path provided (only for root call)
        if output_path and is_root:
            # Ensure the output path has .md extension
            if not output_path.endswith(".md"):
                output_path += ".md"

            with open(output_path, "w") as f:
                f.write(f"# Agent Flow: {self.name}\n\n")
                f.write(mermaid_markdown)

            return output_path

        return mermaid_markdown

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', tools={len(self.tools)})"
