"""
Base Agent class for building AI agents with Opper.
Provides a foundation for agents with a core reasoning loop (Think -> Act).
"""

# No longer need ABC since BaseAgent is now a concrete class
from typing import Any, Dict, List, Optional, Callable
from pydantic import BaseModel, Field
from opperai import Opper
import time
import inspect
from .workflows import FinalizedWorkflow, InMemoryStorage
import os


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
    goal_achieved: bool = Field(description="Whether the main goal has been achieved")
    todo_list: str = Field(description="A markdown list of tasks checked of and todo")
    next_action_needed: bool = Field(
        description="Whether an action is needed to make progress"
    )
    tool_name: str = Field(
        description="Name of the tool to use, or 'direct_response' for direct completion, or 'none' if goal achieved"
    )
    tool_parameters: Dict[str, Any] = Field(
        description="Parameters to pass to the tool"
    )
    expected_outcome: str = Field(
        description="What we expect to happen from this action"
    )
    user_message: str = Field(
        description="A note to the user on what you are about to do"
    )


class ActionResult(BaseModel):
    """Represents the result of executing an action."""

    success: bool = Field(description="Whether the action was executed successfully")
    result: str = Field(description="The result or output from the action")
    tool_name: str = Field(description="Name of the tool that was executed")
    parameters: Dict[str, Any] = Field(
        description="Parameters that were passed to the tool"
    )
    execution_time: float = Field(description="Time taken to execute the action")


class Agent:
    """
    AI agent class using Opper with dual-mode operation support.

    This class provides a foundation for building agents with:
    - A core reasoning loop (Think -> Act)
    - Tool management and execution OR flow-based workflow execution
    - Integration with Opper for LLM calls
    - Structured state management

    Supports two modes of operation:
    1. Tools mode: Uses tools with the traditional reasoning loop (Think -> Act)
    2. Flow mode: Uses structured workflows with FinalizedWorkflow

    Can be used in two ways:
    1. Direct instantiation: provide tools/flow and description in constructor
    2. Subclassing: override get_tools(), get_agent_description(), is_goal_achieved()
    """

    def __init__(
        self,
        name: str,
        opper_api_key: Optional[str] = None,
        max_iterations: int = 25,
        verbose: bool = False,
        output_schema: Optional[type] = None,
        tools: Optional[List[Tool]] = None,
        flow: Optional[FinalizedWorkflow] = None,
        description: Optional[str] = None,
        callback: Optional[callable] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize the base agent.

        Args:
            name: The name of the agent
            opper_api_key: Optional API key for Opper (will use env var if not provided)
            max_iterations: Maximum number of reasoning loop iterations
            verbose: Whether to print detailed execution logs
            output_schema: Optional Pydantic model for structuring the final result
            tools: Optional list of tools (if provided, get_tools() doesn't need to be implemented)
            flow: Optional FinalizedWorkflow for flow-based operations (mutually exclusive with tools)
            description: Optional description (if provided, get_agent_description() doesn't need to be implemented)
            callback: Optional callback function to receive status updates (event_type, data)
            model: Optional default model to use for all LLM calls (can be overridden in individual steps)
        """
        # Validate mode selection
        if tools is not None and flow is not None:
            raise ValueError(
                "Cannot specify both 'tools' and 'flow' parameters. Choose one mode of operation."
            )

        self.name = name
        # try to read api key from environment variable
        if opper_api_key is None:
            opper_api_key = os.getenv("OPPER_API_KEY")
        if opper_api_key is None:
            raise ValueError("OPPER_API_KEY is not set")

        self.opper = Opper(http_bearer=opper_api_key)
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.output_schema = output_schema
        self.callback = callback
        self.model = model

        # Mode selection
        self.mode = "flow" if flow is not None else "tools"
        self.flow = flow

        # Initialize agent state
        self.current_thought: Optional[Thought] = None
        self.execution_history: List[Dict[str, Any]] = []
        self.current_goal: Optional[str] = None
        self.execution_context: Dict[
            str, Any
        ] = {}  # Context data shared between iterations
        self.last_action_result: Optional[ActionResult] = None

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
        return last_cycle.get("goal_achieved", False)

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

    def _think(self, goal: str, parent_span_id: str) -> Thought:
        """Think about the current situation and decide what to do next."""
        current_iteration = len(self.execution_history) + 1
        context = {
            "goal": goal,
            "agent_description": self.description,
            "agent_mode": self.mode,
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
            "last_action_result": self.last_action_result.dict()
            if self.last_action_result
            else None,
            "current_iteration": current_iteration,
            "max_iterations": self.max_iterations,
            "iterations_remaining": self.max_iterations - current_iteration,
        }

        think_call = self.call_llm(
            name="think",
            instructions=f"""You are implementing the THINK step in a Think -> Act reasoning loop.

YOUR RESPONSIBILITIES:
1. ANALYZE the current situation toward achieving the goal
2. REVIEW previous action results and update context if needed
3. DECIDE if the goal has been achieved or if another action is needed
4. CHOOSE the specific tool and parameters for the next action

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
If an action is needed (next_action_needed=true):
- Choose the most appropriate tool for the next step
- Always use tools to complete the goal if possible
- Provide ALL required parameters
- Use data from execution_context when available
- Set tool_name to 'direct_response' only if no tool makes sense AND you can complete the goal without tools
- Set tool_name to 'none' if goal is achieved

Be thorough in your reasoning and decisive in your action selection.""",
            input_data=context,
            output_schema=Thought,
            parent_span_id=parent_span_id,
        )

        return Thought(**think_call.json_payload)

    def _execute_action(
        self, thought: Thought, parent_span_id: Optional[str] = None
    ) -> ActionResult:
        """Execute the action determined by the thought."""
        import time

        start_time = time.time()

        if thought.tool_name == "none" or not thought.next_action_needed:
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
            # Direct completion without tool usage
            execution_time = time.time() - start_time
            return ActionResult(
                success=True,
                result="Task completed directly without tool usage",
                tool_name="direct_response",
                parameters=thought.tool_parameters,
                execution_time=execution_time,
            )

        # Find and execute the tool
        tool = self.get_tool(thought.tool_name)
        if not tool:
            execution_time = time.time() - start_time
            return ActionResult(
                success=False,
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

            # Execute the tool with tracing context
            result = tool.execute(
                _parent_span_id=tool_span.id if tool_span else None,
                **thought.tool_parameters,
            )

            execution_time = time.time() - start_time

            # Update the tool span with results
            if tool_span:
                self.opper.spans.update(
                    span_id=tool_span.id, output=f"Success: {str(result)[:200]}..."
                )

            return ActionResult(
                success=True,
                result=str(result),
                tool_name=thought.tool_name,
                parameters=thought.tool_parameters,
                execution_time=execution_time,
            )
        except Exception as e:
            execution_time = time.time() - start_time

            # Update the tool span with error
            if tool_span:
                self.opper.spans.update(span_id=tool_span.id, output=f"Error: {str(e)}")

            return ActionResult(
                success=False,
                result=f"Error executing tool: {str(e)}",
                tool_name=thought.tool_name,
                parameters=thought.tool_parameters,
                execution_time=execution_time,
            )

    def _generate_final_result(
        self, goal: str, execution_history: List[Dict[str, Any]], parent_span_id: str
    ) -> Any:
        """Generate the final structured result based on the output schema."""
        if not self.output_schema:
            # Return default result format if no schema specified
            return {
                "goal": goal,
                "achieved": self.is_goal_achieved(goal, execution_history),
                "iterations": len(execution_history),
                "execution_history": execution_history,
            }

        # Generate structured result using the output schema
        context = {
            "goal": goal,
            "execution_history": execution_history,
            "agent_description": self.description,
            "goal_achieved": self.is_goal_achieved(goal, execution_history),
            "iterations": len(execution_history),
        }

        result_call = self.call_llm(
            name="generate_final_result",
            instructions="You are a result formatter. Based on the goal and execution history, generate a structured final result. Extract the key information and format it according to the required schema. Focus on the main outcomes and insights from the agent's work.",
            input_data=context,
            output_schema=self.output_schema,
            parent_span_id=parent_span_id,
        )

        return result_call.json_payload

    def get_tools_summary(self) -> str:
        """Get a formatted summary of available tools."""
        if self.mode == "flow":
            return f"Agent '{self.name}' (flow mode): {self.flow.id if self.flow else 'No flow'}"
        else:
            tool_names = [tool.name for tool in self.tools]
            return f"Agent '{self.name}' (tools mode): {', '.join(tool_names)}"

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

    async def _execute_flow(self, goal: str) -> Any:
        """Execute the flow-based workflow with the given goal as input."""
        if not self.flow:
            raise ValueError("Flow mode requires a flow to be provided")

        # Start a trace for this goal processing session
        trace = self.start_trace(name=f"{self.name}_flow", input_data=goal)

        # Emit goal start event
        self._emit_status(
            "goal_start", {"goal": goal, "agent_name": self.name, "mode": "flow"}
        )

        if self.verbose:
            print(f"Starting flow-based goal: {goal}")
            print(f"Agent: {self.name}")
            print(f"Flow: {self.flow.id}")

        try:
            # Create event callback for workflow events
            def workflow_event_callback(event_type: str, data: Any):
                """Forward workflow events through the agent's callback system."""
                if self.callback:
                    # Enhance event with agent context
                    enhanced_data = {
                        "agent_name": self.name,
                        "mode": "flow",
                        "flow_id": self.flow.id,
                        **data,
                    }
                    self.callback(f"workflow_{event_type}", enhanced_data)

            # Create workflow run
            workflow_run = self.flow.create_run(
                opper=self.opper,
                storage=InMemoryStorage(),
                tools={},  # Tools are handled differently in flows
                event_callback=workflow_event_callback,
                default_model=self.model,
            )
            workflow_run.parent_span_id = trace.id

            # Convert goal to the flow's input model
            if hasattr(self.flow.input_model, "model_validate"):
                # If it's a Pydantic model, try to create it from the goal
                try:
                    # Check if the model has a 'goal' field
                    if (
                        hasattr(self.flow.input_model, "__annotations__")
                        and "goal" in self.flow.input_model.__annotations__
                    ):
                        flow_input = self.flow.input_model(goal=goal)
                    else:
                        # For models without a 'goal' field, we need to use AI to convert the goal to structured input
                        # This is a more intelligent conversion using the model's schema
                        conversion_result = self.call_llm(
                            name="convert_goal_to_input",
                            instructions=(
                                f"Convert the user's goal into the required input format for the workflow. "
                                f"Extract the relevant information from the goal and structure it according to the schema. "
                                f"Goal: {goal}"
                            ),
                            input_data={
                                "goal": goal,
                                "required_fields": list(
                                    self.flow.input_model.__annotations__.keys()
                                ),
                            },
                            output_schema=self.flow.input_model,
                            parent_span_id=trace.id,
                        )
                        flow_input = self.flow.input_model.model_validate(
                            conversion_result.json_payload
                        )
                except Exception as e:
                    # If AI conversion fails, try simple parsing for common cases
                    try:
                        # Enhanced fallback parsing for different model types
                        if "ingredients" in self.flow.input_model.__annotations__:
                            # Try to extract ingredients from the goal string
                            import re

                            # Look for patterns like "1 egg", "200g bacon", etc.
                            ingredient_patterns = re.findall(
                                r"(?:\d+\w*\s+)?[a-zA-Z][a-zA-Z\s]+(?=,|$|and)", goal
                            )
                            if ingredient_patterns:
                                ingredients = [
                                    ing.strip() for ing in ingredient_patterns
                                ]
                                flow_input = self.flow.input_model(
                                    ingredients=ingredients
                                )
                            else:
                                # Fallback: split by commas and common separators
                                parts = re.split(r"[,;]|using:|with:", goal.lower())
                                ingredients = [
                                    part.strip()
                                    for part in parts
                                    if part.strip() and not part.startswith("create")
                                ]
                                if ingredients:
                                    flow_input = self.flow.input_model(
                                        ingredients=ingredients
                                    )
                                else:
                                    raise ValueError(
                                        "Cannot extract ingredients from goal"
                                    )
                        elif all(
                            field in self.flow.input_model.__annotations__
                            for field in ["goal", "priority", "category"]
                        ):
                            # Handle UserRequest-style models with structured goal strings
                            import re

                            # Try to parse structured goal like "Priority: HIGH | Category: processing | Goal: ..."
                            priority_match = re.search(
                                r"Priority:\s*(\w+)", goal, re.IGNORECASE
                            )
                            category_match = re.search(
                                r"Category:\s*(\w+)", goal, re.IGNORECASE
                            )
                            goal_match = re.search(
                                r"Goal:\s*(.+?)(?:\s*\||$)", goal, re.IGNORECASE
                            )

                            if priority_match and category_match and goal_match:
                                flow_input = self.flow.input_model(
                                    goal=goal_match.group(1).strip(),
                                    priority=priority_match.group(1).lower(),
                                    category=category_match.group(1).lower(),
                                )
                            else:
                                # Fallback: treat entire string as goal with default priority/category
                                flow_input = self.flow.input_model(
                                    goal=goal, priority="medium", category="processing"
                                )
                        else:
                            # Try to create with default values
                            flow_input = self.flow.input_model()
                    except Exception:
                        # Last resort: raise informative error
                        raise ValueError(
                            f"Cannot convert goal '{goal}' to flow input model {self.flow.input_model.__name__}. "
                            f"The goal could not be parsed into the required format. "
                            f"Required fields: {list(self.flow.input_model.__annotations__.keys())}. "
                            f"Original error: {e}"
                        )
            else:
                flow_input = goal

            # Execute the workflow
            result = await workflow_run.start(input_data=flow_input)

            # Update trace with result
            self.opper.spans.update(span_id=trace.id, output=str(result))

            # Emit goal completion event
            self._emit_status(
                "goal_completed",
                {
                    "goal": goal,
                    "achieved": True,  # Flow completion implies success
                    "mode": "flow",
                    "final_result": result,
                },
            )

            if self.verbose:
                print("Flow completed successfully")
                print(f"Result: {result}")

            return result

        except Exception as e:
            if self.verbose:
                print(f"Flow execution failed: {str(e)}")

            # Update trace with error
            self.opper.spans.update(span_id=trace.id, output=f"Error: {str(e)}")

            # Emit error event
            self._emit_status(
                "goal_completed",
                {"goal": goal, "achieved": False, "mode": "flow", "error": str(e)},
            )

            raise

    def process(self, goal: str) -> Any:
        """
        Process a goal using either the reasoning loop (tools mode) or workflow execution (flow mode).

        This is the main method that implements the core processing logic.
        The behavior depends on the agent's mode:
        - Tools mode: Uses Think -> Act -> Repeat reasoning loop
        - Flow mode: Executes the provided workflow

        Args:
            goal: The goal to achieve

        Returns:
            Final result (format depends on mode and output_schema)
        """
        if self.mode == "flow":
            # Import asyncio here to avoid issues if not needed
            import asyncio

            return asyncio.run(self._execute_flow(goal))
        else:
            return self._process_with_tools(goal)

    def _process_with_tools(self, goal: str) -> Dict[str, Any]:
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
        self.last_action_result = None  # Reset last action result

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

            # Create a span for this iteration
            iteration_span = self.opper.spans.create(
                name=f"iteration_{iteration}",
                input=f"Iteration {iteration} of goal: {goal}",
                parent_id=trace.id,
            )

            if self.verbose:
                print(f"\n--- Iteration {iteration} ---")

            # Step 1: Think
            thought = self._think(goal, iteration_span.id)

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

            # Step 2: Act (if action is needed)
            if thought.next_action_needed:
                action_result = self._execute_action(thought, iteration_span.id)

                # Store the action result for the next think cycle
                self.last_action_result = action_result

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
                    print(f"Success: {action_result.success}")
            else:
                # No action needed
                action_result = ActionResult(
                    success=True,
                    result="No action needed in this iteration",
                    tool_name="none",
                    parameters={},
                    execution_time=0.0,
                )
                self.last_action_result = action_result

            # Store this iteration's cycle
            cycle = {
                "iteration": iteration,
                "thought_reasoning": thought.reasoning[:200] + "..."
                if len(thought.reasoning) > 200
                else thought.reasoning,
                "goal_achieved": thought.goal_achieved,
                "action_tool": action_result.tool_name,
                "action_parameters": action_result.parameters,
                "action_success": action_result.success,
                "action_result": action_result.result,
                "execution_time": action_result.execution_time,
                "timestamp": time.time(),
            }

            # Add error details if action failed
            if not action_result.success:
                cycle["error_details"] = {
                    "action_error": action_result.result,
                    "tool_name": action_result.tool_name,
                    "parameters": action_result.parameters,
                }

            self.execution_history.append(cycle)

            # Update the iteration span with the results
            self.opper.spans.update(
                span_id=iteration_span.id,
                output=f"Thought: {thought.reasoning[:100]}... | Action: {action_result.tool_name} | Success: {action_result.success}",
            )

            # Update current thought
            self.current_thought = thought

        # Generate the final structured result
        final_result = self._generate_final_result(
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

        return final_result

    def call_llm(
        self,
        name: str,
        instructions: str,
        input_schema: Optional[type] = None,
        output_schema: Optional[type] = None,
        input_data: Any = None,
        model: Optional[str] = None,
        parent_span_id: Optional[str] = None,
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

        Returns:
            The result of the LLM call
        """
        return self.opper.call(
            name=name,
            instructions=instructions,
            input_schema=input_schema,
            output_schema=output_schema,
            input=input_data,
            model=model or "groq/gpt-oss-120b",
            parent_span_id=parent_span_id,
        )

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

    def __str__(self) -> str:
        """String representation of the agent."""
        if self.mode == "flow":
            return f"Agent(name='{self.name}', mode='flow', flow='{self.flow.id if self.flow else None}')"
        else:
            return f"Agent(name='{self.name}', mode='tools', tools={len(self.tools)})"

    def __repr__(self) -> str:
        """Detailed representation of the agent."""
        if self.mode == "flow":
            return f"Agent(name='{self.name}', mode='flow', description='{self.description}', flow='{self.flow.id if self.flow else None}')"
        else:
            return f"Agent(name='{self.name}', mode='tools', description='{self.description}', tools={[tool.name for tool in self.tools]})"
