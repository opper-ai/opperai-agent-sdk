"""
Main Agent implementation using 'while tools > 0' loop.

This module contains the primary Agent class that implements the think-act loop.
"""

from typing import Any, Optional, Dict

from ..base.agent import BaseAgent
from ..base.context import AgentContext, ExecutionCycle
from ..memory.memory import Memory
from ..base.hooks import HookEvents
from ..base.tool import ToolResult
from .schemas import Thought, ToolCall, MemoryDecision


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
            memory=Memory() if self.enable_memory else None,
        )

        trace = None

        try:
            await self._activate_tool_providers()

            # Start trace
            trace = await self.start_trace(
                name=f"{self.name}_execution", input_data=input
            )
            self.context.trace_id = trace.id

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

            if trace:
                # Update trace output after successful completion
                await self.opper.spans.update_async(
                    span_id=trace.id, output=str(result)
                )

            return result

        except Exception as e:
            # Trigger: agent_error
            await self.hook_manager.trigger(
                HookEvents.AGENT_ERROR, self.context, agent=self, error=e
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
                HookEvents.LOOP_START, self.context, agent=self
            )

            if self.verbose:
                print(f"\n--- Iteration {iteration + 1}/{self.max_iterations} ---")

            # Think
            thought = await self._think(goal)

            if self.verbose:
                print(f"💭 Reasoning: {thought.reasoning}")
                print(f"🔧 Tool calls: {len(thought.tool_calls)}")

            # Update memory if needed (do this before breaking on empty tool_calls)
            if self.enable_memory and thought.memory_updates:
                for key, update in thought.memory_updates.items():
                    await self.context.memory.write(
                        key=key,
                        value=update.get("value"),
                        description=update.get("description"),
                        metadata=update.get("metadata"),
                    )

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
                results=results,
            )
            self.context.add_cycle(cycle)

            # Trigger: loop_end
            await self.hook_manager.trigger(
                HookEvents.LOOP_END, self.context, agent=self
            )

            iteration += 1

        # Generate final result
        result = await self._generate_final_result(goal)
        return result

    async def _think(self, goal: Any) -> Thought:
        """Call LLM to reason about next actions."""

        # Optional: build lightweight memory snapshot for LLM
        memory_snapshot = None
        if (
            self.enable_memory
            and self.context.memory
            and self.context.memory.has_entries()
        ):
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
                        {"tool": r.tool_name, "success": r.success}
                        for r in cycle.results
                    ],
                }
                for cycle in self.context.get_last_n_cycles(3)
            ],
            "current_iteration": self.context.iteration + 1,
            "max_iterations": self.max_iterations,
            "memory": memory_snapshot,
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
            HookEvents.LLM_CALL, self.context, agent=self, call_type="think"
        )

        # Call Opper (use call_async for async)
        response = await self.opper.call_async(
            name="think",
            instructions=instructions,
            input=context,
            output_schema=Thought,
            model=self.model,
            parent_span_id=self.context.span_id,
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
            response=response,
        )

        thought = Thought(**response.json_payload)

        # Trigger: think_end
        await self.hook_manager.trigger(
            HookEvents.THINK_END, self.context, agent=self, thought=thought
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
                execution_time=0.0,
            )

        # Trigger: tool_call
        await self.hook_manager.trigger(
            HookEvents.TOOL_CALL,
            self.context,
            agent=self,
            tool=tool,
            parameters=tool_call.parameters,
        )

        # Execute
        result = await tool.execute(**tool_call.parameters)

        # Trigger: tool_result
        await self.hook_manager.trigger(
            HookEvents.TOOL_RESULT, self.context, agent=self, tool=tool, result=result
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

        response = await self.opper.call_async(
            name="generate_final_result",
            instructions=instructions,
            input=context,
            output_schema=self.output_schema,
            model=self.model,
            parent_span_id=self.context.trace_id,
        )

        if self.output_schema:
            return self.output_schema(**response.json_payload)
        return response.message

    async def _prepare_memory_snapshot(self, goal: Any) -> Optional[Dict[str, Any]]:
        """Let the LLM decide whether to hydrate memory before think."""

        catalog = await self.context.memory.list_entries()
        if not catalog:
            return None

        response = await self.opper.call_async(
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

        decision = MemoryDecision(**response.json_payload)
        if not decision.should_use_memory or not decision.selected_keys:
            return None

        return await self.context.memory.read(decision.selected_keys)

    def _memory_router_instructions(self) -> str:
        """Instructions for the memory router LLM call."""
        return """Decide if any memory entries should be loaded for the next reasoning step.
- Only set should_use_memory=true when an entry clearly helps.
- selected_keys must come from memory_catalog.
- Keep the rationale concise."""
