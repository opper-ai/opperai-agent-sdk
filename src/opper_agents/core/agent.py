"""
Main Agent implementation using 'while tools > 0' loop.

This module contains the primary Agent class that implements the think-act loop.
"""

from typing import Any, Optional, List

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

        self.memory: Optional[Memory] = None
        if self.enable_memory:
            try:
                self.memory = Memory()
            except Exception as exc:  # pragma: no cover - defensive guard
                # Disable memory if initialization fails to keep agent operational
                self.enable_memory = False
                if self.logger:
                    self.logger.log_warning(
                        f"Memory disabled due to initialization error: {exc}"
                    )

    async def process(self, input: Any, _parent_span_id: Optional[str] = None) -> Any:
        """
        Main entry point for agent execution.

        Args:
            input: Goal/task to process (validated against input_schema)
            _parent_span_id: Optional parent span ID for nested agent calls

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
            memory=self.memory if self.enable_memory else None,
        )

        parent_span = None

        try:
            await self._activate_tool_providers()

            # Create parent span for this agent execution
            # If _parent_span_id is provided, this span will be nested under it
            parent_span = await self.opper.spans.create_async(
                name=f"{self.name}_execution",
                input=str(input),
                parent_id=_parent_span_id,
            )
            self.context.parent_span_id = parent_span.id

            # Trigger: agent_start
            await self.hook_manager.trigger(
                HookEvents.AGENT_START, self.context, agent=self
            )

            # Run main loop
            result = await self._run_loop(input)

            # Trigger: agent_end
            await self.hook_manager.trigger(
                HookEvents.AGENT_END, self.context, agent=self, result=result
            )

            # Disconnect MCP servers before span updates
            # This prevents issues with stdio pipes during final operations
            await self._deactivate_tool_providers()

            if parent_span:
                # Update parent span with final output
                # Shield from AnyIO cancel scopes that may have been left by MCP cleanup
                import anyio

                with anyio.CancelScope(shield=True):
                    await self.opper.spans.update_async(
                        span_id=parent_span.id, output=str(result)
                    )

            return result

        except Exception as e:
            # Trigger: agent_error
            await self.hook_manager.trigger(
                HookEvents.AGENT_ERROR, self.context, agent=self, error=e
            )
            raise
        finally:
            # Ensure tool providers are deactivated even if an error occurred
            # This is idempotent, safe to call multiple times
            await self._deactivate_tool_providers()

    async def _run_loop(self, goal: Any) -> Any:
        """
        Main execution loop: while tools > 0

        Returns when thought.tool_calls is empty.
        """
        while self.context.iteration < self.max_iterations:
            await self.hook_manager.trigger(
                HookEvents.LOOP_START, self.context, agent=self
            )

            if self.logger:
                self.logger.log_iteration(
                    self.context.iteration + 1, self.max_iterations
                )

            thought: Optional[Thought] = None
            results: List[ToolResult] = []
            loop_complete = False

            try:
                # Show spinner while thinking
                if self.logger:
                    with self.logger.log_thinking():
                        thought = await self._think(goal)
                else:
                    thought = await self._think(goal)

                # Log the thought
                if self.logger and thought is not None:
                    self.logger.log_thought(thought.reasoning, len(thought.tool_calls))

                memory_reads_performed = False
                memory_writes_performed = False

                if (
                    self.enable_memory
                    and self.context.memory is not None
                    and thought is not None
                    and thought.memory_reads
                ):
                    if self.logger:
                        self.logger.log_memory_read(thought.memory_reads)

                    memory_read_span = await self.opper.spans.create_async(
                        name="memory_read",
                        input=str(thought.memory_reads),
                        parent_id=self.context.parent_span_id,
                    )

                    memory_data = await self.context.memory.read(thought.memory_reads)

                    await self.opper.spans.update_async(
                        span_id=memory_read_span.id,
                        output=str(memory_data),
                    )

                    self.context.metadata["current_memory"] = memory_data
                    memory_reads_performed = True
                    if self.logger:
                        self.logger.log_memory_loaded(memory_data)

                if (
                    self.enable_memory
                    and self.context.memory is not None
                    and thought is not None
                    and thought.memory_updates
                ):
                    if self.logger:
                        self.logger.log_memory_write(
                            list(thought.memory_updates.keys())
                        )

                    memory_write_span = await self.opper.spans.create_async(
                        name="memory_write",
                        input=str(list(thought.memory_updates.keys())),
                        parent_id=self.context.parent_span_id,
                    )

                    for key, update in thought.memory_updates.items():
                        await self.context.memory.write(
                            key=key,
                            value=update.get("value"),
                            description=update.get("description"),
                            metadata=update.get("metadata"),
                        )

                    await self.opper.spans.update_async(
                        span_id=memory_write_span.id,
                        output=f"Successfully wrote {len(thought.memory_updates)} keys",
                    )

                    memory_writes_performed = True

                if thought is not None:
                    for tool_call in thought.tool_calls:
                        result = await self._execute_tool(tool_call)
                        results.append(result)

                    cycle = ExecutionCycle(
                        iteration=self.context.iteration,
                        thought=thought,
                        tool_calls=thought.tool_calls,
                        results=results,
                    )
                    activity_occurred = (
                        bool(results)
                        or memory_reads_performed
                        or memory_writes_performed
                    )

                    if activity_occurred:
                        self.context.add_cycle(cycle)

                    has_tool_calls = len(thought.tool_calls) > 0
                    has_memory_reads = (
                        self.enable_memory and len(thought.memory_reads) > 0
                    )
                    loop_complete = not has_tool_calls and not has_memory_reads

            finally:
                await self.hook_manager.trigger(
                    HookEvents.LOOP_END, self.context, agent=self
                )

            if loop_complete:
                if self.logger:
                    self.logger.log_final_result()
                break

        result = await self._generate_final_result(goal)
        return result

    async def _think(self, goal: Any) -> Thought:
        """Call LLM to reason about next actions."""

        # Build memory catalog if memory is enabled
        memory_catalog = None
        if (
            self.enable_memory
            and self.context.memory
            and self.context.memory.has_entries()
        ):
            memory_catalog = await self.context.memory.list_entries()

        # Build context
        context = {
            "goal": str(goal),
            "agent_description": self.description,
            "instructions": self.instructions or "No specific instructions.",
            "available_tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
                for tool in self.tools
            ],
            "execution_history": [
                {
                    "iteration": cycle.iteration,
                    "thought": (
                        cycle.thought.reasoning
                        if hasattr(cycle.thought, "reasoning")
                        else str(cycle.thought)
                    ),
                    "results": [
                        {
                            "tool": r.tool_name,
                            "success": r.success,
                            "result": str(r.result),
                        }
                        for r in cycle.results
                    ],
                }
                for cycle in self.context.get_last_n_cycles(3)
            ],
            "current_iteration": self.context.iteration + 1,
            "max_iterations": self.max_iterations,
            "memory_catalog": memory_catalog,
            "loaded_memory": self.context.metadata.get("current_memory", None),
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

        # Add memory instructions if enabled
        if self.enable_memory:
            instructions += """

MEMORY SYSTEM:
You have access to a persistent memory system that works across iterations.

Memory Operations:
1. READ: Use memory_reads field to load specific keys (e.g., ["trip_budget", "favorite_city"])
2. WRITE: Use memory_updates field to save information for later use
   Example: {"trip_budget": {"value": 1250.0, "description": "Total trip budget calculated"}}

When to use memory:
- Save important calculations, decisions, or user preferences
- Load memory when you need information from earlier in the conversation
- Check memory_catalog to see what's available before requesting keys
- Use descriptive keys like "budget_total", "user_favorite_city", etc.

The memory you write persists across all process() calls on this agent.
"""

        # Trigger: llm_call
        await self.hook_manager.trigger(
            HookEvents.LLM_CALL, self.context, agent=self, call_type="think"
        )

        # Call Opper (use call_async for async)
        response = await self.opper.call_async(
            name="think",
            instructions=instructions,
            input=context,
            output_schema=Thought,
            model=self.model,
            parent_span_id=self.context.parent_span_id,
        )

        # Track usage
        self._track_usage(response)

        # Trigger: llm_response
        await self.hook_manager.trigger(
            HookEvents.LLM_RESPONSE,
            self.context,
            agent=self,
            call_type="think",
            response=response,
        )

        thought = Thought(**response.json_payload)

        # Trigger: think_end
        await self.hook_manager.trigger(
            HookEvents.THINK_END, self.context, agent=self, thought=thought
        )

        return thought

    async def _execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a single tool call and create a span for it."""

        if self.logger:
            self.logger.log_tool_call(tool_call.name, tool_call.parameters)

        tool = self.get_tool(tool_call.name)
        if not tool:
            return ToolResult(
                tool_name=tool_call.name,
                success=False,
                result=None,
                error=f"Tool '{tool_call.name}' not found",
                execution_time=0.0,
            )

        # Create span for this tool call
        tool_span = await self.opper.spans.create_async(
            name=f"tool_{tool_call.name}",
            input=str(tool_call.parameters),
            parent_id=self.context.parent_span_id,
        )

        # Trigger: tool_call
        await self.hook_manager.trigger(
            HookEvents.TOOL_CALL,
            self.context,
            agent=self,
            tool=tool,
            parameters=tool_call.parameters,
        )

        # Execute - pass tool span as parent for nested operations (like agents-as-tools)
        result = await tool.execute(
            **tool_call.parameters, _parent_span_id=tool_span.id
        )

        # Update tool span with result
        await self.opper.spans.update_async(
            span_id=tool_span.id,
            output=str(result.result) if result.success else None,
            error=result.error if not result.success else None,
        )

        # Trigger: tool_result
        await self.hook_manager.trigger(
            HookEvents.TOOL_RESULT, self.context, agent=self, tool=tool, result=result
        )

        if self.logger:
            self.logger.log_tool_result(
                tool_call.name, result.success, result.result, result.error
            )

        return result

    async def _generate_final_result(self, goal: Any) -> Any:
        """
        Generate final structured result.

        This method is shielded from AnyIO cancel scopes to prevent issues when
        MCP stdio clients have been disconnected (which can leave cancel scopes active).
        """
        import anyio

        context = {
            "goal": str(goal),
            "instructions": self.instructions,
            "execution_history": [
                {
                    "iteration": cycle.iteration,
                    "actions_taken": [r.tool_name for r in cycle.results],
                    "results": [
                        {"tool": r.tool_name, "result": str(r.result)}
                        for r in cycle.results
                        if r.success
                    ],
                }
                for cycle in self.context.execution_history
            ],
            "total_iterations": self.context.iteration,
        }

        instructions = """Generate the final result based on the execution history.
Follow any instructions provided for formatting and style."""

        # Shield this from AnyIO cancel scopes that may have been left by MCP stdio cleanup
        with anyio.CancelScope(shield=True):
            response = await self.opper.call_async(
                name="generate_final_result",
                instructions=instructions,
                input=context,
                output_schema=self.output_schema,
                model=self.model,
                parent_span_id=self.context.parent_span_id,
            )

            # Track usage
            self._track_usage(response)

            # Serialize the response
            if self.output_schema:
                return self.output_schema(**response.json_payload)
            return response.message

    def _track_usage(self, response: Any) -> None:
        """
        Track token usage from an Opper response.

        Safely extracts usage info if available, otherwise skips tracking.
        """
        if not hasattr(response, "usage") or not response.usage:
            return

        try:
            from ..base.context import Usage

            usage_dict = response.usage
            if isinstance(usage_dict, dict):
                usage = Usage(
                    requests=1,
                    input_tokens=usage_dict.get("input_tokens", 0),
                    output_tokens=usage_dict.get("output_tokens", 0),
                    total_tokens=usage_dict.get("total_tokens", 0),
                )
                self.context.update_usage(usage)
        except Exception as e:
            # Don't break execution if usage tracking fails
            if self.logger:
                self.logger.log_warning(f"Could not track usage: {e}")
