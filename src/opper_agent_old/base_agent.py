"""
Base Agent class for building AI agents with Opper.
Provides a foundation for agents with a core reasoning loop (Think -> Act).
"""

from typing import Any, Dict, List, Optional, Callable, Union
from pydantic import BaseModel, Field
from opperai import Opper
import time
import inspect
import os


class Usage(BaseModel):
    """Usage statistics for agent operations."""

    requests: int = Field(default=0, description="Number of requests made")
    input_tokens: int = Field(default=0, description="Total input tokens used")
    output_tokens: int = Field(default=0, description="Total output tokens used")
    total_tokens: int = Field(default=0, description="Total tokens used")

    def add(self, other: "Usage") -> "Usage":
        """Add usage statistics from another Usage object."""
        return Usage(
            requests=self.requests + other.requests,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


class RunContext(BaseModel):
    """Context information for agent runs."""

    agent_name: str = Field(description="Name of the agent")
    timestamp: float = Field(description="Unix timestamp when context was created")
    iteration: Optional[int] = Field(
        default=None, description="Current iteration number"
    )
    goal: Optional[str] = Field(
        default=None, description="Current goal being processed"
    )
    usage: Usage = Field(default_factory=Usage, description="Usage statistics")

    # Additional context data
    extra: Dict[str, Any] = Field(
        default_factory=dict, description="Additional context data"
    )


class AgentHooks:
    """Base class for agent hooks. Override methods to handle specific events."""

    async def on_agent_start(self, context: RunContext, agent: "Agent") -> None:
        """Called when the agent starts processing a goal."""
        pass

    async def on_agent_end(
        self, context: RunContext, agent: "Agent", output: Any
    ) -> None:
        """Called when the agent finishes processing a goal."""
        pass

    async def on_agent_error(
        self, context: RunContext, agent: "Agent", error: Exception
    ) -> None:
        """Called when the agent encounters an error."""
        pass

    async def on_iteration_start(self, context: RunContext, agent: "Agent") -> None:
        """Called at the start of each reasoning iteration."""
        pass

    async def on_iteration_end(self, context: RunContext, agent: "Agent") -> None:
        """Called at the end of each reasoning iteration."""
        pass

    async def on_think_start(self, context: RunContext, agent: "Agent") -> None:
        """Called before the agent starts thinking/reasoning."""
        pass

    async def on_think_end(
        self, context: RunContext, agent: "Agent", thought: Any
    ) -> None:
        """Called after the agent finishes thinking/reasoning."""
        pass

    async def on_tool_start(
        self, context: RunContext, agent: "Agent", tool: "Tool"
    ) -> None:
        """Called before a tool is executed."""
        pass

    async def on_tool_end(
        self, context: RunContext, agent: "Agent", tool: "Tool", result: Any
    ) -> None:
        """Called after a tool is executed successfully."""
        pass

    async def on_tool_error(
        self, context: RunContext, agent: "Agent", tool: "Tool", error: Exception
    ) -> None:
        """Called when a tool execution fails."""
        pass

    async def on_llm_start(
        self, context: RunContext, agent: "Agent", call_name: str, input_data: Any
    ) -> None:
        """Called before an LLM call is made."""
        pass

    async def on_llm_end(
        self,
        context: RunContext,
        agent: "Agent",
        call_name: str,
        input_data: Any,
        output: Any,
    ) -> None:
        """Called after an LLM call completes."""
        pass


class Tool(BaseModel):
    """Represents a tool that an agent can use."""

    name: str = Field(description="The name of the tool")
    description: str = Field(description="Description of what the tool does")
    parameters: Dict[str, Any] = Field(description="Parameters the tool accepts")

    def execute(self, _parent_span_id: Optional[str] = None, **kwargs) -> Any:
        """
        Execute the tool with given parameters.

        Args:
            _parent_span_id: Optional parent span ID for tracing AI calls within tools
            **kwargs: Tool-specific parameters

        Returns:
            Tool execution result
        """
        # This is a placeholder - subclasses should implement actual tool logic
        raise NotImplementedError(f"Tool {self.name} execution not implemented")

    def make_ai_call(
        self,
        opper_client,
        name: str,
        instructions: str,
        input_data: Any = None,
        output_schema: Optional[type] = None,
        parent_span_id: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Any:
        """
        Helper method for tools to make AI calls with proper tracing.

        Args:
            opper_client: The Opper client instance
            name: Name of the AI call for tracking
            instructions: Instructions for the AI
            input_data: Input data for the call
            output_schema: Optional output schema
            parent_span_id: Parent span ID for tracing
            model: Optional model to use (defaults to "groq/gpt-oss-120b")

        Returns:
            AI call result
        """
        return opper_client.call(
            name=name,
            instructions=instructions,
            input=input_data,
            output_schema=output_schema,
            parent_span_id=parent_span_id,
            model=model or "groq/gpt-oss-120b",
        )


class FunctionTool(Tool):
    """A Tool that wraps a regular Python function."""

    func: Callable = Field(description="The function to execute", exclude=True)

    def __init__(
        self,
        func: Callable,
        name: str = None,
        description: str = None,
        parameters: Dict[str, str] = None,
    ):
        # Extract function metadata
        func_name = name or func.__name__
        func_description = description or func.__doc__ or f"Execute {func.__name__}"

        # Auto-generate parameters from function signature if not provided
        if parameters is None:
            parameters = self._extract_parameters(func)

        super().__init__(
            name=func_name,
            description=func_description,
            parameters=parameters,
            func=func,
        )

    def _extract_parameters(self, func: Callable) -> Dict[str, str]:
        """Extract parameter information from function signature."""
        sig = inspect.signature(func)
        parameters = {}

        for param_name, param in sig.parameters.items():
            # Skip special parameters
            if param_name.startswith("_"):
                continue

            # Build parameter description
            param_type = "any"
            if param.annotation != inspect.Parameter.empty:
                if hasattr(param.annotation, "__name__"):
                    param_type = param.annotation.__name__
                else:
                    param_type = str(param.annotation)

            # Add default value info if present
            default_info = ""
            if param.default != inspect.Parameter.empty:
                default_info = f" (default: {param.default})"

            parameters[param_name] = f"{param_type}{default_info}"

        return parameters

    def execute(self, _parent_span_id: Optional[str] = None, **kwargs) -> Any:
        """Execute the wrapped function with the provided arguments."""
        try:
            # Filter out the special _parent_span_id parameter
            filtered_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}

            # Check if function accepts _parent_span_id for AI calls
            sig = inspect.signature(self.func)
            if "_parent_span_id" in sig.parameters:
                filtered_kwargs["_parent_span_id"] = _parent_span_id

            result = self.func(**filtered_kwargs)

            # Return in standard format
            return {
                "success": True,
                "result": result,
                "data": result if isinstance(result, dict) else {"output": result},
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "result": f"Error executing {self.name}: {str(e)}",
            }


def hook(event_name: str):
    """
    Decorator to create a hook function that can be passed to Agent constructor.

    Usage:
        @hook("on_agent_start")
        async def on_start(context, agent):
            print("Agent started!")

        agent = Agent(name="Test", tools=[...], hooks=[on_start])
    """

    def decorator(func):
        func._hook_event = event_name
        return func

    return decorator


def tool(
    func: Callable = None,
    *,
    name: str = None,
    description: str = None,
    parameters: Dict[str, str] = None,
):
    """
    Decorator to convert a function into a Tool.

    Args:
        func: The function to wrap (when used without parentheses)
        name: Optional custom name for the tool (defaults to function name)
        description: Optional custom description (defaults to function docstring)
        parameters: Optional custom parameter descriptions (auto-extracted if not provided)

    Usage:
        @tool
        def add_numbers(a: int, b: int) -> int:
            \"\"\"Add two numbers together.\"\"\"
            return a + b

        @tool(name="custom_name", description="Custom description")
        def my_function(x: str) -> str:
            return x.upper()

        @tool(parameters={"query": "str - Search query", "limit": "int - Max results"})
        def search(query: str, limit: int = 10) -> list:
            # Implementation here
            pass
    """

    def decorator(f: Callable) -> FunctionTool:
        return FunctionTool(f, name, description, parameters)

    if func is None:
        # Called with arguments: @tool(name="something")
        return decorator
    else:
        # Called without arguments: @tool
        return decorator(func)


class Thought(BaseModel):
    """Represents the thinking step that combines planning and reflection."""

    reasoning: str = Field(
        description="Analysis of current situation, previous results, and what needs to be done"
    )
    goal_achieved: bool = Field(
        description="Whether everything needed to complete the request has been done"
    )
    task_list: str = Field(
        description="A markdown list of small atomic tasks remaining and completed in direction of the goal"
    )
    tool_name: Optional[str] = Field(
        default=None,
        description="Name of the tool to use, or 'direct_response' for direct completion, or 'none' if goal achieved",
    )
    tool_parameters: Optional[Dict[str, Any]] = Field(
        default=None, description="Parameters to pass to the tool"
    )
    iteration: Optional[int] = Field(
        default=None, description="The iteration number when this thought was generated"
    )
    expected_outcome: Optional[str] = Field(
        default=None, description="What we expect to happen from this action"
    )
    user_message: str = Field(
        description="A note to the user on what you are about to do"
    )


class ActionResult(BaseModel):
    """Represents the result of executing an action."""

    result: str = Field(description="The result or output from the action")
    tool_name: str = Field(description="Name of the tool that was executed")
    parameters: Dict[str, Any] = Field(
        description="Parameters that were passed to the tool"
    )
    execution_time: float = Field(description="Time taken to execute the action")


class Agent:
    """
    AI agent class using Opper with tool-based operation.

    This class provides a foundation for building agents with:
    - A core reasoning loop (Think -> Act)
    - Tool management and execution
    - Integration with Opper for LLM calls
    - Structured state management
    - Input/output schema validation
    - Flexible hook system

    Can be used in two ways:
    1. Direct instantiation: provide tools and description in constructor
    2. Subclassing: override get_tools(), get_agent_description(), is_goal_achieved()
    """

    def __init__(
        self,
        name: str,
        opper_api_key: Optional[str] = None,
        max_iterations: int = 25,
        verbose: bool = False,
        tools: Optional[List[Tool]] = None,
        description: Optional[str] = None,
        instructions: Optional[str] = None,
        stream: bool = False,
        callback: Optional[callable] = None,
        model: Optional[str] = None,
        input_schema: Optional[type] = None,
        output_schema: Optional[type] = None,
        hooks: Optional[Union[AgentHooks, List[callable]]] = None,
        clean_tool_results: bool = True,
        tool_result_clean_threshold: int = 1000,
    ):
        """
        Initialize the base agent.

        Args:
            name: The name of the agent
            opper_api_key: Optional API key for Opper (will use env var if not provided)
            max_iterations: Maximum number of reasoning loop iterations
            verbose: Whether to print detailed execution logs
            tools: Optional list of tools (if provided, get_tools() doesn't need to be implemented)
            description: Optional description (if provided, get_agent_description() doesn't need to be implemented)
            instructions: Optional instructions for how this agent should behave when used as a tool
            stream: Whether to use streaming output for LLM calls (defaults to False)
            callback: Optional callback function to receive status updates (event_type, data)
            model: Optional default model to use for all LLM calls
            input_schema: Optional Pydantic model for input validation (defaults to str)
            output_schema: Optional Pydantic model for output validation (defaults to str)
            hooks: Optional AgentHooks instance or list of hook functions decorated with @hook() for handling events
            clean_tool_results: Whether to clean tool results using LLM when they exceed threshold (defaults to True)
            tool_result_clean_threshold: Character threshold above which tool results are cleaned (defaults to 500)
        """
        self.name = name
        # try to read api key from environment variable
        if opper_api_key is None:
            opper_api_key = os.getenv("OPPER_API_KEY")
        if opper_api_key is None:
            raise ValueError("OPPER_API_KEY is not set")

        self.opper = Opper(http_bearer=opper_api_key)
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.input_schema = input_schema or None
        self.output_schema = output_schema or None
        self.instructions = instructions
        self.stream = stream
        self.callback = callback
        self.model = model
        self.clean_tool_results = clean_tool_results
        self.tool_result_clean_threshold = tool_result_clean_threshold

        # Agent mode is always tools-based
        self.mode = "tools"

        # Initialize agent state
        self.current_thought: Optional[Thought] = None
        self.execution_history: List[Dict[str, Any]] = []
        self.current_goal: Optional[str] = None
        self.execution_context: Dict[
            str, Any
        ] = {}  # Context data shared between iterations

        # Initialize hook system
        self._decorator_hooks = {}  # Store decorator-based hooks

        if hooks is None:
            self.hooks = None
        elif isinstance(hooks, AgentHooks):
            # Class-based hooks
            self.hooks = hooks
        elif isinstance(hooks, list):
            # Decorator-based hooks (list of functions)
            self.hooks = None
            for hook_func in hooks:
                if hasattr(hook_func, "_hook_event"):
                    event_name = hook_func._hook_event
                    if event_name not in self._decorator_hooks:
                        self._decorator_hooks[event_name] = []
                    self._decorator_hooks[event_name].append(hook_func)
        else:
            raise ValueError(
                "hooks must be either an AgentHooks instance or a list of hook functions"
            )

        self.current_iteration: int = 0
        self.run_context = RunContext(agent_name=self.name, timestamp=time.time())

        # Get tools and description (from constructor or subclass)
        self.tools = tools if tools is not None else self.get_tools()
        self.description = (
            description if description is not None else self.get_agent_description()
        )

        # Initialize agent state
        self._initialize_agent()

    def _initialize_agent(self):
        """Initialize agent-specific state. Override in subclasses if needed."""
        pass

    # Hook system methods
    def hook(self, *event_names):
        """
        Decorator for registering hooks.

        Usage:
            @agent.hook("on_agent_start")
            async def on_start(context, agent):
                print("Agent started!")

            @agent.hook("on_tool_start", "on_tool_end")
            async def tool_monitor(context, agent, tool, result=None):
                if result is None:
                    print(f"Tool {tool.name} started")
                else:
                    print(f"Tool {tool.name} completed")
        """

        def decorator(func):
            for event_name in event_names:
                if event_name not in self._decorator_hooks:
                    self._decorator_hooks[event_name] = []
                self._decorator_hooks[event_name].append(func)
            return func

        return decorator

    async def _call_hook(self, method_name: str, *args, **kwargs):
        """Call both class-based and decorator-based hooks."""
        # Call class-based hooks
        if self.hooks and hasattr(self.hooks, method_name):
            try:
                method = getattr(self.hooks, method_name)
                await method(self.run_context, self, *args, **kwargs)
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Class hook error in {method_name}: {e}")
                # Continue execution even if hook fails

        # Call decorator-based hooks
        if method_name in self._decorator_hooks:
            for hook_func in self._decorator_hooks[method_name]:
                try:
                    # Handle different parameter signatures
                    sig = inspect.signature(hook_func)
                    bound_args = sig.bind(self.run_context, self, *args, **kwargs)
                    bound_args.apply_defaults()
                    await hook_func(*bound_args.args, **bound_args.kwargs)
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️  Decorator hook error in {method_name}: {e}")
                    # Continue execution even if hook fails

    def _emit_status(self, event_type: str, data: Any):
        """Emit a status update through the callback if one is provided."""
        if self.callback:
            try:
                self.callback(event_type, data)
            except Exception as e:
                # Don't let callback errors break the agent
                if self.verbose:
                    print(f"⚠️  Callback error for {event_type}: {e}")

    def get_tools(self) -> List[Tool]:
        """
        Return the list of tools available to this agent.
        Override this method if not providing tools in constructor.

        Returns:
            List of Tool instances this agent can use
        """
        return []

    def get_agent_description(self) -> str:
        """
        Return a description of what this agent does.
        Override this method if not providing description in constructor.

        Returns:
            String description of the agent's purpose and capabilities
        """
        return f"AI agent named {self.name}"

    def is_goal_achieved(
        self, goal: str, execution_history: List[Dict[str, Any]]
    ) -> bool:
        """
        Check if the goal has been achieved based on execution history.
        Override this method for custom goal achievement logic.

        Default implementation: checks if the last thought indicates goal achievement.

        Args:
            goal: The goal to check
            execution_history: History of think-act cycles

        Returns:
            True if the goal is achieved, False otherwise
        """
        if not execution_history:
            return False

        # Default implementation: check if last thought indicates goal achievement
        last_cycle = execution_history[-1]
        if hasattr(last_cycle, "goal_achieved"):
            return last_cycle.goal_achieved
        elif isinstance(last_cycle, dict):
            return last_cycle.get("goal_achieved", False)
        else:
            return False

    def add_tool(self, tool: Tool):
        """Add a tool to the agent's toolkit."""
        self.tools.append(tool)

    def remove_tool(self, tool_name: str):
        """Remove a tool from the agent's toolkit."""
        self.tools = [tool for tool in self.tools if tool.name != tool_name]

    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """Get a specific tool by name."""
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None

    def list_tools(self) -> List[str]:
        """Get a list of available tool names."""
        return [tool.name for tool in self.tools]

    async def _think(self, goal: str, parent_span_id: str) -> Thought:
        """Think about the current situation and decide what to do next."""
        current_iteration = len(self.execution_history) + 1
        context = {
            "goal": goal,
            "agent_description": self.description,
            "agent_mode": self.mode,
            "agent_instructions": self.instructions
            or "No specific instructions provided.",
            "available_tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
                for tool in self.tools
            ],
            "execution_history": self.execution_history[-3:]
            if self.execution_history
            else [],  # Last 3 cycles for context
            "execution_context": self.execution_context,
            "current_iteration": current_iteration,
            "max_iterations": self.max_iterations,
            "iterations_remaining": self.max_iterations - current_iteration,
        }

        # Build the call instructions that reference the context
        instructions = f"""You are implementing the THINK step in a Think -> Act reasoning loop.

YOUR RESPONSIBILITIES:
1. ANALYZE the current situation toward achieving the goal
2. REVIEW previous action results and update context if needed
3. PLAN the next series of *small atomic tasks* to complete the goal
4. DECIDE if the goal has been achieved or if another action is needed
5. CHOOSE the specific tool and parameters for the next action

THINKING PROCESS:
- Consider the main goal and current progress
- Analyze the last action result (if any) and what it tells you
- Review execution_context for important data from previous actions
- Be AWARE of your iteration count: you are on iteration {current_iteration} of {self.max_iterations} maximum
- With {self.max_iterations - current_iteration} iterations remaining, plan efficiently to complete the goal
- If approaching iteration limit, prioritize the most important parts of the goal
- Determine if goal is achieved or what action is needed next

CONTEXT UPDATES:
If the last action produced useful results, extract important data into context_updates:
- Use descriptive keys (e.g., 'email_count', 'calculation_result', 'user_preference')
- Store IDs, references, computed values, or state information
- This data will be available in future think cycles

GOAL ACHIEVEMENT:
Set goal_achieved=true only if the main goal is completely accomplished.

ACTION SELECTION:
If goal_achieved is False, then:
- Choose the most appropriate tool for the next step
- Always use tools to complete the goal if possible
- Provide ALL required parameters
- Use data from execution_context when available
- Set tool_name to 'direct_response' only if no tool makes sense AND you can complete the goal without tools
- Set tool_name to 'none' if goal is achieved

AGENT BEHAVIOR:
Follow the agent_instructions provided in the context object to guide your reasoning and action selection.

Be thorough in your reasoning and decisive in your action selection."""

        # Internal think calls should never use streaming - only final responses should stream
        think_call = await self.call_llm(
            name="think",
            instructions=instructions,
            input_data=context,
            output_schema=Thought,
            model=self.model,
            parent_span_id=parent_span_id,
            stream=False,  # Force non-streaming for internal calls
        )

        # Always return the structured Thought object for internal calls
        thought_data = think_call.json_payload
        thought_data["iteration"] = current_iteration
        return Thought(**thought_data)

    async def _execute_action(
        self, thought: Thought, parent_span_id: Optional[str] = None
    ) -> ActionResult:
        """Execute the action determined by the thought."""
        import time

        start_time = time.time()

        if thought.tool_name == "none" or not thought.tool_name:
            # No action needed - goal achieved
            execution_time = time.time() - start_time
            return ActionResult(
                success=True,
                result="No action needed - goal achieved",
                tool_name="none",
                parameters={},
                execution_time=execution_time,
            )

        if thought.tool_name == "direct_response":
            # Direct response without tools - goal achieved
            execution_time = time.time() - start_time
            return ActionResult(
                success=True,
                result="Direct response provided",
                tool_name="direct_response",
                parameters=thought.tool_parameters or {},
                execution_time=execution_time,
            )

        # Find and execute the tool
        tool = self.get_tool(thought.tool_name)
        if not tool:
            execution_time = time.time() - start_time
            return ActionResult(
                result=f"Tool '{thought.tool_name}' not found",
                tool_name=thought.tool_name,
                parameters=thought.tool_parameters,
                execution_time=execution_time,
            )

        try:
            # Create a span for this tool execution
            tool_span = None
            if parent_span_id:
                tool_span = self.opper.spans.create(
                    name=f"tool_{thought.tool_name}",
                    input=f"Tool: {thought.tool_name}, Parameters: {thought.tool_parameters}",
                    parent_id=parent_span_id,
                )

            # Trigger on_tool_start hook
            await self._call_hook("on_tool_start", tool)

            # Execute the tool with tracing context
            result = tool.execute(
                _parent_span_id=tool_span.id if tool_span else None,
                **thought.tool_parameters,
            )

            # Trigger on_tool_end hook
            await self._call_hook("on_tool_end", tool, result)

            execution_time = time.time() - start_time

            # Clean the tool result if it's too long
            cleaned_result = await self.clean_tool_result(
                tool_result=str(result),
                tool_name=thought.tool_name,
                goal=str(self.current_goal) if self.current_goal else "Unknown goal",
                parent_span_id=tool_span.id if tool_span else None,
            )

            # Update the tool span with results
            if tool_span:
                self.opper.spans.update(
                    span_id=tool_span.id, output=f"Success: {cleaned_result[:500]}..."
                )

            return ActionResult(
                result=cleaned_result,
                tool_name=thought.tool_name,
                parameters=thought.tool_parameters,
                execution_time=execution_time,
            )
        except Exception as e:
            execution_time = time.time() - start_time

            # Trigger on_tool_error hook
            await self._call_hook("on_tool_error", tool, e)

            # Update the tool span with error
            if tool_span:
                self.opper.spans.update(span_id=tool_span.id, output=f"Error: {str(e)}")

            return ActionResult(
                success=False,
                result=f"Error executing tool: {str(e)}",
                tool_name=thought.tool_name,
                parameters=thought.tool_parameters or {},
                execution_time=execution_time,
            )

    async def _generate_final_result(
        self, goal: Any, execution_history: List[Dict[str, Any]], parent_span_id: str
    ) -> Any:
        """Generate the final structured result based on the output schema."""

        # Generate structured result using the output schema
        context = {
            "goal": goal,
            "execution_history": execution_history,
            "agent_description": self.description,
            "agent_instructions": self.instructions
            or "No specific instructions provided.",
            "goal_achieved": self.is_goal_achieved(goal, execution_history),
            "iterations": len(execution_history),
        }

        # Build instructions for final result generation that reference context
        final_instructions = """Given the goal and the information collected in execution_history generate a final result according to the output schema.

Follow the agent_instructions provided in the context object to guide how you format and present the final result."""

        result_call = await self.call_llm(
            name="generate_final_result",
            instructions=final_instructions,
            input_data=context,
            output_schema=self.output_schema,
            model=self.model,
            parent_span_id=parent_span_id,
            stream=self.stream,  # Use agent's streaming setting for final result
        )

        if self.stream:
            # Handle streaming output
            if self.output_schema:
                # For streaming with output schema, collect the streamed chunks
                result = {}
                current_path = None

                for chunk in result_call.result.generator:
                    if hasattr(chunk, "data") and chunk.data:
                        # Check for json_path to know which field is being streamed
                        if hasattr(chunk.data, "json_path") and chunk.data.json_path:
                            current_path = chunk.data.json_path

                        # Handle delta content
                        if hasattr(chunk.data, "delta") and chunk.data.delta:
                            delta_content = ""
                            if isinstance(chunk.data.delta, str):
                                delta_content = chunk.data.delta
                            elif (
                                hasattr(chunk.data.delta, "content")
                                and chunk.data.delta.content
                            ):
                                delta_content = chunk.data.delta.content

                            if delta_content:
                                # If we have a path, accumulate content for that specific field
                                if current_path:
                                    if current_path not in result:
                                        result[current_path] = ""
                                    result[current_path] += delta_content
                                    print(
                                        f"[{current_path}] {delta_content}",
                                        end="",
                                        flush=True,
                                    )
                                else:
                                    # Fallback to concatenating all content
                                    if "content" not in result:
                                        result["content"] = ""
                                    result["content"] += delta_content
                                    print(delta_content, end="", flush=True)

                print()  # New line after streaming

                # Try to parse the structured result
                try:
                    import json

                    if current_path and len(result) == 1:
                        # Single field result
                        parsed_result = json.loads(list(result.values())[0])
                        return parsed_result
                    elif "content" in result:
                        # Fallback to parsing content as JSON
                        parsed_result = json.loads(result["content"])
                        return parsed_result
                    else:
                        # Try to construct JSON from structured fields
                        return result
                except json.JSONDecodeError:
                    # If parsing fails, try to parse markdown format
                    content = result.get("content", "") or str(result)
                    return self._parse_markdown_to_json(content)
            else:
                # For streaming without output schema, just stream the content
                result = ""
                current_path = None

                for chunk in result_call.result.generator:
                    if hasattr(chunk, "data") and chunk.data:
                        # Check for json_path to know which field is being streamed
                        if hasattr(chunk.data, "json_path") and chunk.data.json_path:
                            current_path = chunk.data.json_path

                        # Handle delta content
                        if hasattr(chunk.data, "delta") and chunk.data.delta:
                            delta_content = ""
                            if isinstance(chunk.data.delta, str):
                                delta_content = chunk.data.delta
                            elif (
                                hasattr(chunk.data.delta, "content")
                                and chunk.data.delta.content
                            ):
                                delta_content = chunk.data.delta.content

                            if delta_content:
                                result += delta_content
                                if current_path:
                                    print(
                                        f"[{current_path}] {delta_content}",
                                        end="",
                                        flush=True,
                                    )
                                else:
                                    print(delta_content, end="", flush=True)

                print()  # New line after streaming
                return result
        else:
            # Handle non-streaming output
            if self.output_schema:
                return result_call.json_payload
            return result_call.message

    def _parse_markdown_to_json(self, markdown_content: str) -> dict:
        """Parse markdown content to extract structured data as JSON."""
        try:
            import re

            # Initialize result dictionary
            result = {}

            # Extract title
            title_match = re.search(r'\*\*Title:\*\*\s*"([^"]+)"', markdown_content)
            if title_match:
                result["title"] = title_match.group(1)

            # Extract summary
            summary_match = re.search(
                r"\*\*Summary:\*\*\s*(.+?)(?=\n\n|\n\*\*|$)",
                markdown_content,
                re.DOTALL,
            )
            if summary_match:
                result["summary"] = summary_match.group(1).strip()

            # If we have an output schema, try to extract other common fields
            if self.output_schema and hasattr(self.output_schema, "model_fields"):
                for field_name in self.output_schema.model_fields.keys():
                    if field_name not in result:  # Skip if already extracted
                        # Look for **FieldName:** pattern
                        field_pattern = (
                            rf"\*\*{field_name.title()}:\*\*\s*(.+?)(?=\n\n|\n\*\*|$)"
                        )
                        field_match = re.search(
                            field_pattern, markdown_content, re.DOTALL
                        )
                        if field_match:
                            result[field_name] = field_match.group(1).strip()

            # If no structured data found, return the raw content
            if not result:
                return {"message": markdown_content}

            return result

        except Exception as e:
            # Fallback to returning the raw content
            return {"message": markdown_content}

    def get_tools_summary(self) -> str:
        """Get a formatted summary of available tools."""
        tool_names = [tool.name for tool in self.tools]
        return f"Agent '{self.name}': {', '.join(tool_names)}"

    def get_context_summary(self) -> str:
        """Get a formatted summary of current execution context."""
        if not self.execution_context:
            return "Execution context is empty"

        summary = f"Execution context ({len(self.execution_context)} items):\n"
        for key, value in self.execution_context.items():
            value_str = str(value)
            if len(value_str) > 100:
                value_str = value_str[:97] + "..."
            summary += f"  {key}: {value_str}\n"
        return summary.strip()

    def get_execution_summary(self) -> str:
        """Get a formatted summary of execution history (thoughts and actions)."""
        if not self.execution_history:
            return "No execution history available"

        summary = f"Execution history ({len(self.execution_history)} iterations):\n"
        for cycle in self.execution_history:
            iteration = cycle.get("iteration", "?")
            tool = cycle.get("action_tool", "unknown")
            success = "SUCCESS" if cycle.get("action_success", False) else "FAILED"
            goal_achieved = "GOAL_ACHIEVED" if cycle.get("goal_achieved", False) else ""

            summary += f"  {iteration}. Think→{tool} {success}{goal_achieved}"

            if "error_details" in cycle:
                error = cycle["error_details"].get("action_error", "Unknown error")
                summary += f" - Error: {error[:50]}..."

            summary += "\n"

        return summary.strip()

    def clear_context(self):
        """Clear the execution context. Useful for testing or manual control."""
        self.execution_context = {}

    async def clean_tool_result(
        self,
        tool_result: str,
        tool_name: str,
        goal: str,
        parent_span_id: Optional[str] = None,
    ) -> str:
        """
        Clean tool results using an LLM to remove unnecessary information.

        This method uses an LLM to analyze tool results and remove bogus or unnecessary
        information that doesn't contribute to achieving the goal.

        Args:
            tool_result: The raw tool result to clean
            tool_name: Name of the tool that produced the result
            goal: The current goal being worked towards
            parent_span_id: Optional parent span ID for tracing

        Returns:
            Cleaned tool result with unnecessary information removed
        """
        if (
            not self.clean_tool_results
            or len(tool_result) <= self.tool_result_clean_threshold
        ):
            return tool_result

        if self.verbose:
            print(
                f"🧹 Cleaning tool result from {tool_name} ({len(tool_result)} chars)"
            )

        # Create a span for the cleaning operation
        clean_span = None
        if parent_span_id:
            clean_span = self.opper.spans.create(
                name=f"clean_tool_result_{tool_name}",
                input=f"Cleaning result from {tool_name}",
                parent_id=parent_span_id,
            )

        try:
            # Build instructions for tool result cleaning that reference context
            clean_instructions = f"""You are tasked with cleaning a tool result to remove unnecessary information.

CONTEXT:
- Tool: {tool_name}
- Goal: {goal}
- Original result length: {len(tool_result)} characters

TASK:
Analyze the tool result and remove any bogus, irrelevant, or unnecessary information that doesn't contribute to achieving the goal. Keep only the essential information that is directly relevant to the goal.

GUIDELINES:
- Remove error messages, debug information, or technical details that don't help with the goal
- Remove redundant or repetitive information
- Keep key data, results, and actionable information
- Maintain the original structure and format when possible
- If the result is already clean and relevant, return it as-is
- Be conservative - only remove clearly unnecessary information

AGENT BEHAVIOR:
Follow the agent_instructions provided in the context object to determine what information is relevant and should be kept in the cleaned result.

Return the cleaned result directly without any additional commentary or explanation."""

            # Use LLM to clean the tool result - internal calls should not use streaming
            clean_call = await self.call_llm(
                name="clean_tool_result",
                instructions=clean_instructions,
                input_data={
                    "tool_name": tool_name,
                    "goal": goal,
                    "original_result": tool_result,
                    "agent_instructions": self.instructions
                    or "No specific instructions provided.",
                },
                model=self.model,
                parent_span_id=clean_span.id if clean_span else None,
                stream=False,  # Force non-streaming for internal calls
            )

            # Extract the cleaned result - internal calls should not use streaming
            cleaned_result = clean_call.message

            # Update the clean span with results
            if clean_span:
                self.opper.spans.update(
                    span_id=clean_span.id,
                    output=f"Cleaned from {len(tool_result)} to {len(cleaned_result)} characters",
                )

            if self.verbose:
                print(
                    f"✅ Cleaned result: {len(tool_result)} → {len(cleaned_result)} chars"
                )

            return cleaned_result.strip()

        except Exception as e:
            if self.verbose:
                print(f"⚠️  Error cleaning tool result: {e}")

            # Update the clean span with error
            if clean_span:
                self.opper.spans.update(
                    span_id=clean_span.id, output=f"Error: {str(e)}"
                )

            # Return original result if cleaning fails
            return tool_result

    def process(self, goal: Any):
        """
        Process a goal using the reasoning loop: Think -> Act -> Repeat.

        This is the main method that implements the core processing logic.
        It continues until the goal is achieved or max iterations are reached.

        Args:
            goal: The goal to achieve (will be validated against input_schema)

        Returns:
            For streaming mode: Async generator that yields chunks
            For non-streaming mode: Coroutine that returns final result
        """
        # Validate input against input_schema
        if self.input_schema and self.input_schema != str:
            if isinstance(goal, str) and hasattr(self.input_schema, "model_validate"):
                # If goal is a string but input_schema expects a Pydantic model, wrap it
                goal = self.input_schema.model_validate({"goal": goal})
            elif hasattr(self.input_schema, "model_validate"):
                # If input_schema is a Pydantic model, validate the goal
                goal = self.input_schema.model_validate(goal)

        if self.stream:
            # For streaming mode, return an async generator
            return self._process_with_streaming(goal)
        else:
            # For non-streaming mode, return a coroutine
            return self._process_with_tools(goal)

    async def _process_with_streaming(self, goal: Any):
        """
        Process a goal with streaming output that yields chunks in real-time.

        This method runs the same logic as _process_with_tools but yields
        streaming chunks during the final result generation.

        Args:
            goal: The goal to achieve

        Yields:
            Streaming chunks with json_path and delta information
        """
        # Run the main processing logic up to the final result generation
        self.current_goal = goal
        self.execution_history = []
        self.execution_context = {}  # Reset context for new goal
        self.current_iteration = 0

        # Update run context
        self.run_context.goal = str(goal)
        self.run_context.timestamp = time.time()

        # Trigger on_agent_start hook
        await self._call_hook("on_agent_start")

        # Start a trace for this goal processing session
        trace = self.start_trace(name=f"{self.name}_tools", input_data=goal)

        # Emit goal start event
        self._emit_status(
            "goal_start",
            {
                "goal": goal,
                "agent_name": self.name,
                "mode": "tools",
                "available_tools": [tool.name for tool in self.tools],
            },
        )

        if self.verbose:
            print(f"Starting goal: {goal}")
            print(f"🤖 Agent: {self.name}")
            print(f"🔧 Available tools: {[tool.name for tool in self.tools]}")
            print(f"Max iterations: {self.max_iterations}")

        iteration = 0
        goal_achieved_early = False
        while iteration < self.max_iterations:
            iteration += 1
            self.current_iteration = iteration

            # Update run context with current iteration
            self.run_context.iteration = iteration

            # Trigger on_iteration_start hook
            await self._call_hook("on_iteration_start")

            # Create a span for this iteration
            iteration_span = self.opper.spans.create(
                name=f"iteration_{iteration}",
                input=f"Iteration {iteration} of goal: {goal}",
                parent_id=trace.id,
            )

            if self.verbose:
                print(f"\n--- Iteration {iteration} ---")

            # Step 1: Think
            await self._call_hook("on_think_start")
            thought = await self._think(goal, iteration_span.id)
            await self._call_hook("on_think_end", thought)

            # Add thought to execution history
            self.execution_history.append(thought)

            if self.verbose:
                print(f"💭 Thought: {thought.reasoning}")
                if thought.tool_name:
                    print(f"🔧 Tool: {thought.tool_name}")
                    print(f"📝 Parameters: {thought.tool_parameters}")

            # Check if goal is achieved
            if thought.goal_achieved:
                goal_achieved_early = True
                if self.verbose:
                    print("✅ Goal achieved!")
                break

            # Check if this is a direct response - skip to final result immediately
            if thought.tool_name == "direct_response":
                if self.verbose:
                    print("✅ Direct response - skipping to final result")

                # Generate the final structured result with streaming immediately
                async for chunk in self._generate_final_result_streaming(
                    goal, self.execution_history, trace.id
                ):
                    yield chunk

                # Emit goal completion event
                self._emit_status(
                    "goal_completed",
                    {
                        "goal": goal,
                        "achieved": True,
                        "iterations": iteration,
                        "final_result": "streamed",
                    },
                )

                # Trigger on_agent_end hook
                await self._call_hook("on_agent_end")
                return

            # Step 2: Act (if tool is selected)
            if thought.tool_name:
                await self._call_hook("on_tool_start", tool=thought.tool_name)

                # Handle special cases
                if thought.tool_name == "none":
                    # No action needed - goal achieved
                    if self.verbose:
                        print("✅ No action needed - goal achieved")

                    # Add successful result to execution history
                    self.execution_history.append(
                        ActionResult(
                            tool_name=thought.tool_name,
                            success=True,
                            result="No action needed - goal achieved",
                            parameters={},
                            execution_time=0.0,  # Placeholder for now
                        )
                    )
                else:
                    # Find the tool
                    tool = next(
                        (t for t in self.tools if t.name == thought.tool_name), None
                    )
                    if not tool:
                        error_msg = f"Tool '{thought.tool_name}' not found"
                        if self.verbose:
                            print(f"❌ {error_msg}")

                        # Add error to execution history
                        self.execution_history.append(
                            ActionResult(
                                tool_name=thought.tool_name,
                                success=False,
                                result=error_msg,
                                error=error_msg,
                                parameters=thought.tool_parameters or {},
                                execution_time=0.0,  # Placeholder for now
                            )
                        )
                    else:
                        # Execute the tool
                        try:
                            if self.verbose:
                                print(f"🔧 Executing {thought.tool_name}...")

                            tool_result = await tool.execute(
                                _parent_span_id=iteration_span.id,
                                **thought.tool_parameters,
                            )

                            if self.verbose:
                                print(f"✅ Tool result: {tool_result}")

                            # Extract the actual result from the tool response
                            if isinstance(tool_result, dict) and tool_result.get(
                                "success"
                            ):
                                actual_result = tool_result.get("result", "")
                            else:
                                actual_result = str(tool_result)

                            # Clean the tool result using LLM
                            cleaned_result = await self.clean_tool_result(
                                actual_result,
                                thought.tool_name,
                                thought.goal,
                                iteration_span.id,
                            )

                            # Add successful result to execution history
                            self.execution_history.append(
                                ActionResult(
                                    tool_name=thought.tool_name,
                                    success=True,
                                    result=cleaned_result,
                                    data=tool_result,
                                    parameters=thought.tool_parameters or {},
                                    execution_time=0.0,  # Placeholder for now
                                )
                            )

                        except Exception as e:
                            error_msg = f"Error executing {thought.tool_name}: {str(e)}"
                            if self.verbose:
                                print(f"❌ {error_msg}")

                            # Add error to execution history
                            self.execution_history.append(
                                ActionResult(
                                    tool_name=thought.tool_name,
                                    success=False,
                                    result=error_msg,
                                    error=str(e),
                                    parameters=thought.tool_parameters or {},
                                    execution_time=0.0,  # Placeholder for now
                                )
                            )

                await self._call_hook("on_tool_end", tool=thought.tool_name)

            # Update current thought
            self.current_thought = thought

            # Trigger on_iteration_end hook
            await self._call_hook("on_iteration_end")

        # Generate the final structured result with streaming
        async for chunk in self._generate_final_result_streaming(
            goal, self.execution_history, trace.id
        ):
            yield chunk

        # Emit goal completion event
        self._emit_status(
            "goal_completed",
            {
                "goal": goal,
                "achieved": self.is_goal_achieved(goal, self.execution_history),
                "iterations": iteration,
                "final_result": "streamed",
            },
        )

        # Trigger on_agent_end hook
        await self._call_hook("on_agent_end")

    async def _generate_final_result_streaming(
        self,
        goal: Any,
        execution_history: List[Any],
        parent_span_id: Optional[str] = None,
    ):
        """
        Generate the final structured result with streaming output.

        This method is similar to _generate_final_result but yields streaming chunks
        instead of printing them directly.

        Args:
            goal: The original goal
            execution_history: List of thoughts and actions taken
            parent_span_id: Optional parent span ID for tracing

        Yields:
            Streaming chunks with json_path and delta information
        """
        # Create a span for final result generation (defer to avoid blocking streaming)
        final_span = None
        if parent_span_id:
            try:
                final_span = self.opper.spans.create(
                    name="generate_final_result",
                    input=f"Generate final result for goal: {goal}",
                    parent_id=parent_span_id,
                )
            except Exception:
                # If span creation fails, continue without it to avoid blocking streaming
                pass

        # Prepare context for the final result generation
        # For direct responses, use minimal context to speed up processing
        if (
            execution_history
            and len(execution_history) == 1
            and execution_history[0].tool_name == "direct_response"
        ):
            context = {
                "goal": goal,
                "agent_instructions": self.instructions
                or "No specific instructions provided.",
            }
        else:
            context = {
                "goal": goal,
                "execution_history": execution_history,
                "agent_instructions": self.instructions
                or "No specific instructions provided.",
                "available_tools": [tool.name for tool in self.tools],
                "current_iteration": self.current_iteration,
            }

        # Create instructions for final result generation
        if (
            execution_history
            and len(execution_history) == 1
            and execution_history[0].tool_name == "direct_response"
        ):
            # Simplified instructions for direct responses
            final_instructions = """Generate a friendly, helpful response to the user's input according to the output schema.

Follow the agent_instructions provided in the context object to guide your response style."""
        else:
            # Full instructions for complex responses
            final_instructions = """Given the goal and the information collected in execution_history generate a final result according to the output schema.

Follow the agent_instructions provided in the context object to guide how you format and present the final result."""

        # Make the LLM call with streaming
        result_call = await self.call_llm(
            name="generate_final_result",
            instructions=final_instructions,
            input_data=context,
            output_schema=self.output_schema,
            model=self.model,
            parent_span_id=parent_span_id,
            stream=True,  # Force streaming for this method
        )

        # Handle streaming output
        if self.output_schema:
            # For streaming with output schema, collect the streamed chunks by field
            result = {}
            current_path = None

            for chunk in result_call.result.generator:
                if hasattr(chunk, "data") and chunk.data:
                    # Check for json_path to know which field is being streamed
                    if hasattr(chunk.data, "json_path") and chunk.data.json_path:
                        current_path = chunk.data.json_path

                    # Handle delta content
                    if hasattr(chunk.data, "delta") and chunk.data.delta:
                        delta_content = ""
                        if isinstance(chunk.data.delta, str):
                            delta_content = chunk.data.delta
                        elif (
                            hasattr(chunk.data.delta, "content")
                            and chunk.data.delta.content
                        ):
                            delta_content = chunk.data.delta.content

                        if delta_content:
                            # Yield the chunk with streaming information
                            yield {
                                "type": "streaming_chunk",
                                "json_path": current_path,
                                "delta": delta_content,
                                "content": delta_content,
                            }

                            # If we have a path, accumulate content for that specific field
                            if current_path:
                                if current_path not in result:
                                    result[current_path] = ""
                                result[current_path] += delta_content
                            else:
                                # Fallback to concatenating all content
                                if "content" not in result:
                                    result["content"] = ""
                                result["content"] += delta_content

            # Yield final result
            try:
                import json

                if current_path and len(result) == 1:
                    # Single field result
                    parsed_result = json.loads(list(result.values())[0])
                    yield {
                        "type": "final_result",
                        "json_path": None,
                        "delta": None,
                        "content": parsed_result,
                    }
                elif "content" in result:
                    # Fallback to parsing content as JSON
                    parsed_result = json.loads(result["content"])
                    yield {
                        "type": "final_result",
                        "json_path": None,
                        "delta": None,
                        "content": parsed_result,
                    }
                else:
                    # Try to construct JSON from structured fields
                    yield {
                        "type": "final_result",
                        "json_path": None,
                        "delta": None,
                        "content": result,
                    }
            except json.JSONDecodeError:
                # If parsing fails, try to parse markdown format
                content = result.get("content", "") or str(result)
                parsed_result = self._parse_markdown_to_json(content)
                yield {
                    "type": "final_result",
                    "json_path": None,
                    "delta": None,
                    "content": parsed_result,
                }
        else:
            # For streaming without output schema, just stream the content
            result = ""
            current_path = None

            for chunk in result_call.result.generator:
                if hasattr(chunk, "data") and chunk.data:
                    # Check for json_path to know which field is being streamed
                    if hasattr(chunk.data, "json_path") and chunk.data.json_path:
                        current_path = chunk.data.json_path

                    # Handle delta content
                    if hasattr(chunk.data, "delta") and chunk.data.delta:
                        delta_content = ""
                        if isinstance(chunk.data.delta, str):
                            delta_content = chunk.data.delta
                        elif (
                            hasattr(chunk.data.delta, "content")
                            and chunk.data.delta.content
                        ):
                            delta_content = chunk.data.delta.content

                        if delta_content:
                            result += delta_content
                            yield {
                                "type": "streaming_chunk",
                                "json_path": current_path,
                                "delta": delta_content,
                                "content": delta_content,
                            }

            # Yield final result
            yield {
                "type": "final_result",
                "json_path": None,
                "delta": None,
                "content": result,
            }

        # Update the span with the final result
        self.opper.spans.update(span_id=final_span.id, output="Final result generated")

    async def _process_with_tools(self, goal: Any) -> Any:
        """
        Process a goal using the reasoning loop: Think -> Act -> Repeat.

        This implements the traditional tool-based reasoning loop.
        It continues until the goal is achieved or max iterations are reached.

        Args:
            goal: The goal to achieve

        Returns:
            Dictionary containing the final result and execution history
        """
        self.current_goal = goal
        self.execution_history = []
        self.execution_context = {}  # Reset context for new goal
        self.current_iteration = 0

        # Update run context
        self.run_context.goal = str(goal)
        self.run_context.timestamp = time.time()

        # Trigger on_agent_start hook
        await self._call_hook("on_agent_start")

        # Start a trace for this goal processing session
        trace = self.start_trace(name=f"{self.name}_tools", input_data=goal)

        # Emit goal start event
        self._emit_status(
            "goal_start",
            {
                "goal": goal,
                "agent_name": self.name,
                "mode": "tools",
                "available_tools": [tool.name for tool in self.tools],
            },
        )

        if self.verbose:
            print(f"Starting goal: {goal}")
            print(f"🤖 Agent: {self.name}")
            print(f"🔧 Available tools: {[tool.name for tool in self.tools]}")
            print(f"Max iterations: {self.max_iterations}")

        iteration = 0
        goal_achieved_early = False
        while iteration < self.max_iterations:
            iteration += 1
            self.current_iteration = iteration

            # Update run context with current iteration
            self.run_context.iteration = iteration

            # Trigger on_iteration_start hook
            await self._call_hook("on_iteration_start")

            # Create a span for this iteration
            iteration_span = self.opper.spans.create(
                name=f"iteration_{iteration}",
                input=f"Iteration {iteration} of goal: {goal}",
                parent_id=trace.id,
            )

            if self.verbose:
                print(f"\n--- Iteration {iteration} ---")

            # Step 1: Think
            await self._call_hook("on_think_start")
            thought = await self._think(goal, iteration_span.id)
            await self._call_hook("on_think_end", thought)

            # Emit think event
            self._emit_status(
                "thought_created", {"iteration": iteration, "thought": thought.dict()}
            )

            if self.verbose:
                print(f"🧠 Thought: {thought.reasoning}")
                print(f"Goal achieved: {thought.goal_achieved}")
                print(f"Next action: {thought.tool_name}")

            # Check if goal is achieved
            if thought.goal_achieved:
                goal_achieved_early = True
                if self.verbose:
                    print("Goal achieved!")
                break

            # Check if this is a direct response - skip to final result immediately
            if thought.tool_name == "direct_response":
                if self.verbose:
                    print("✅ Direct response - skipping to final result")

                # Generate the final structured result immediately
                final_result = await self._generate_final_result(
                    goal, self.execution_history, trace.id
                )

                # Emit goal completion event
                self._emit_status(
                    "goal_completed",
                    {
                        "goal": goal,
                        "achieved": True,
                        "iterations": iteration,
                        "final_result": final_result,
                    },
                )

                # Trigger on_agent_end hook
                await self._call_hook("on_agent_end")
                return final_result

            # Step 2: Act (if action is needed)
            if thought.tool_name:
                action_result = await self._execute_action(thought, iteration_span.id)

                # Action result is stored in execution_history

                # Emit action result event
                self._emit_status(
                    "action_executed",
                    {
                        "iteration": iteration,
                        "thought": thought.dict(),
                        "action_result": action_result.dict(),
                    },
                )

                if self.verbose:
                    print(
                        f"Action: {action_result.tool_name} with {thought.tool_parameters}"
                    )
                    print(f"Result: {action_result.result}")
                    print(
                        f"Success: True"
                    )  # ActionResult always represents successful execution
            else:
                # No action needed
                action_result = ActionResult(
                    result="No action needed in this iteration",
                    tool_name="none",
                    parameters={},
                    execution_time=0.0,
                )

            # Store this iteration's cycle
            cycle = {
                "iteration": iteration,
                "thought_reasoning": thought.reasoning[:200] + "..."
                if len(thought.reasoning) > 200
                else thought.reasoning,
                "goal_achieved": thought.goal_achieved,
                "action_tool": action_result.tool_name,
                "action_parameters": action_result.parameters,
                "action_success": True,  # ActionResult always represents successful execution
                "action_result": action_result.result,
                "execution_time": action_result.execution_time,
                "timestamp": time.time(),
            }

            # Note: ActionResult always represents successful execution
            # Error handling is done at the tool execution level

            self.execution_history.append(cycle)

            # Update the iteration span with the results
            self.opper.spans.update(
                span_id=iteration_span.id,
                output=f"Thought: {thought.reasoning[:100]}... | Action: {action_result.tool_name} | Success: True",
            )

            # Update current thought
            self.current_thought = thought

            # Trigger on_iteration_end hook
            await self._call_hook("on_iteration_end")

        # Generate the final structured result
        final_result = await self._generate_final_result(
            goal, self.execution_history, trace.id
        )

        # Emit goal completion event
        self._emit_status(
            "goal_completed",
            {
                "goal": goal,
                "achieved": self.is_goal_achieved(goal, self.execution_history),
                "iterations": iteration,
                "final_result": final_result,
            },
        )

        self.opper.spans.update(span_id=trace.id, output=str(final_result))

        if self.verbose:
            print(f"\nCompleted in {iteration} iterations")

            # Check if we reached max iterations without achieving goal
            if not goal_achieved_early and iteration >= self.max_iterations:
                print(
                    f"Warning: Reached maximum iterations ({self.max_iterations}) without achieving goal"
                )
                print(
                    "Consider increasing max_iterations or breaking down the goal into smaller steps"
                )

            # Handle both structured and unstructured results
            if isinstance(final_result, dict) and "achieved" in final_result:
                print(f"Goal achieved: {final_result['achieved']}")
            else:
                goal_achieved = self.is_goal_achieved(goal, self.execution_history)
                print(f"Goal achieved: {goal_achieved}")
                if not goal_achieved and iteration >= self.max_iterations:
                    print(
                        "Agent stopped due to iteration limit, goal may be partially complete"
                    )
                print(f"📄 Structured result: {final_result}")

        # Trigger on_agent_end hook
        await self._call_hook("on_agent_end", final_result)

        return final_result

    async def call_llm(
        self,
        name: str,
        instructions: str,
        input_schema: Optional[type] = None,
        output_schema: Optional[type] = None,
        input_data: Any = None,
        model: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        stream: Optional[bool] = None,
    ):
        """
        Make a call to the LLM using Opper.

        This method provides a convenient way for agents to interact with LLMs
        while maintaining proper tracing and schema validation.

        Args:
            name: Name of the call for tracking
            instructions: Instructions for the LLM
            input_schema: Optional Pydantic model for input validation
            output_schema: Optional Pydantic model for output validation
            input_data: Input data for the call
            model: Optional specific model to use (defaults to "groq/gpt-oss-120b")
            parent_span_id: Optional parent span ID for tracing
            stream: Optional override for streaming (defaults to self.stream)

        Returns:
            The result of the LLM call (streaming or non-streaming based on stream parameter)
        """
        # Trigger on_llm_start hook
        if self.hooks:
            await self._call_hook("on_llm_start", name, input_data)

        # Use stream parameter if provided, otherwise use agent's stream setting
        use_streaming = stream if stream is not None else self.stream

        if use_streaming:
            # Use streaming call
            result = self.opper.stream(
                name=name,
                instructions=instructions,
                input_schema=input_schema,
                output_schema=output_schema,
                input=input_data,
                model=model or "groq/gpt-oss-120b",
                parent_span_id=parent_span_id,
            )
        else:
            # Use regular call
            result = self.opper.call(
                name=name,
                instructions=instructions,
                input_schema=input_schema,
                output_schema=output_schema,
                input=input_data,
                model=model or "groq/gpt-oss-120b",
                parent_span_id=parent_span_id,
            )

        # Trigger on_llm_end hook
        if self.hooks:
            await self._call_hook("on_llm_end", name, input_data, result)

        return result

    def start_trace(self, name: str, input_data: Any = None):
        """
        Start a new trace for the agent's operations.

        Args:
            name: Name of the trace
            input_data: Input data for the trace

        Returns:
            The created span
        """
        return self.opper.spans.create(
            name=name, input=str(input_data) if input_data else None
        )

    def as_tool(
        self, tool_name: Optional[str] = None, description: Optional[str] = None
    ) -> FunctionTool:
        """
        Convert this agent into a tool that can be used by other agents.

        This allows agents to be used as tools in other agents' tool lists,
        enabling multi-agent systems where agents can delegate tasks to each other.

        Args:
            tool_name: Optional custom name for the tool (defaults to agent name)
            description: Optional custom description for the tool (defaults to agent description)

        Returns:
            FunctionTool that can be added to another agent's tools list

        Example:
            >>> math_agent = Agent(name="MathAgent", instructions="Always show your work step by step", tools=[...])
            >>> routing_agent = Agent(
            ...     name="RoutingAgent",
            ...     tools=[math_agent.as_tool()]
            ... )
        """
        import asyncio
        import concurrent.futures
        import time
        from typing import Any, Dict

        tool_name = tool_name or f"{self.name}_agent"
        description = description or f"Delegate to {self.name}: {self.description}"

        def agent_tool(task: str, **kwargs) -> Any:
            """Tool function that delegates to the agent."""
            start_time = time.time()

            try:
                # Create a task for the agent
                async def call_agent():
                    # Prepare the input data
                    input_data = {"task": task, **kwargs}

                    # If agent has instructions, prepend them to the task
                    if self.instructions:
                        input_data["task"] = f"{self.instructions}\n\n{task}"

                    return await self.process(input_data)

                # Run the async call in a new event loop
                def run_in_thread():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(call_agent())
                    finally:
                        loop.close()

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_in_thread)
                    result = future.result(timeout=60)  # 60 second timeout

                # Ensure the result is in the expected format
                if isinstance(result, dict):
                    # If it's a dict, return it as-is (it should have the right structure)
                    return result
                else:
                    # If it's a string or other type, wrap it in the expected format
                    return {
                        "message": str(result),
                        "success": True,
                        "agent_used": self.name,
                    }

            except Exception as e:
                return {
                    "error": f"Agent {self.name} failed: {str(e)}",
                    "success": False,
                    "agent_used": self.name,
                }

        # Create delegation parameters for the tool
        parameters = {
            "task": f"The task to be processed by {self.name}",
        }

        # Add additional parameters from the agent's input schema if available
        if self.input_schema and hasattr(self.input_schema, "model_fields"):
            for field_name, field_info in self.input_schema.model_fields.items():
                if field_name != "task":  # Skip task as it's already defined
                    field_type = field_info.annotation
                    field_description = (
                        field_info.description or f"Parameter {field_name}"
                    )

                    # Convert type to string for documentation
                    if hasattr(field_type, "__name__"):
                        type_str = field_type.__name__
                    else:
                        type_str = str(field_type)

                    parameters[field_name] = f"{field_description} (Type: {type_str})"

        # Add common delegation parameters
        parameters.update(
            {
                "user_id": "ID of the user making the request (optional)",
                "priority": "Priority level 1-5 (optional, default: 1)",
            }
        )

        return FunctionTool(
            func=agent_tool,
            name=tool_name,
            description=description,
            parameters=parameters,
        )

    def __str__(self) -> str:
        """String representation of the agent."""
        return f"Agent(name='{self.name}', mode='tools', tools={len(self.tools)})"

    def __repr__(self) -> str:
        """Detailed representation of the agent."""
        return f"Agent(name='{self.name}', mode='tools', description='{self.description}', tools={[tool.name for tool in self.tools]})"
