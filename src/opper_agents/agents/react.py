"""
ReAct Agent implementation.

This module implements the ReAct (Reasoning + Acting) pattern agent.
"""

from typing import Any

from ..core.agent import Agent
from ..core.schemas import ReactThought, ToolCall
from ..base.context import ExecutionCycle
from ..base.hooks import HookEvents


class ReactAgent(Agent):
    """
    ReAct pattern agent: Reasoning + Acting in cycles.

    Loop:
    1. Reason: Analyze situation and decide on action
    2. Act: Execute the chosen tool
    3. Observe: Review result
    4. Repeat or complete

    The ReAct pattern is simpler than the default Agent:
    - Only one tool call per iteration (not multiple)
    - Explicit observation step
    - Clear separation between reasoning and acting
    """

    async def _run_loop(self, goal: Any) -> Any:
        """
        Custom ReAct loop implementation.

        This overrides the default Agent loop to implement the ReAct pattern.
        """
        observation = "Task received. Ready to begin."

        while self.context.iteration < self.max_iterations:
            # Trigger: loop_start
            await self.hook_manager.trigger(
                HookEvents.LOOP_START, self.context, agent=self
            )

            if self.verbose:
                print(
                    f"\n--- ReAct Iteration {self.context.iteration + 1}/{self.max_iterations} ---"
                )
                print(f"Observation: {observation}")

            # REASON: Analyze situation and decide on action
            thought = await self._reason(goal, observation)

            if self.verbose:
                print(f"Reasoning: {thought.reasoning}")

            # Check if task is complete
            if thought.is_complete:
                if self.verbose:
                    print("Task complete - generating final result")
                break

            # ACT: Execute the action
            if not thought.action:
                if self.verbose:
                    print(
                        "Warning: No action specified but task not complete. Ending loop."
                    )
                break

            if self.verbose:
                print(
                    f"Action: {thought.action.tool_name}({thought.action.parameters})"
                )

            # Convert Action to ToolCall for execution
            tool_call = ToolCall(
                name=thought.action.tool_name,
                parameters=thought.action.parameters,
                reasoning=thought.reasoning,
            )

            result = await self._execute_tool(tool_call)

            # OBSERVE: Update observation with result
            if result.success:
                observation = (
                    f"Tool '{result.tool_name}' succeeded with result: {result.result}"
                )
            else:
                observation = (
                    f"Tool '{result.tool_name}' failed with error: {result.error}"
                )

            # Record cycle
            cycle = ExecutionCycle(
                iteration=self.context.iteration,
                thought=thought,
                tool_calls=[tool_call],
                results=[result],
            )
            self.context.add_cycle(cycle)

            # Trigger: loop_end
            await self.hook_manager.trigger(
                HookEvents.LOOP_END, self.context, agent=self
            )

        # Generate final result
        result = await self._generate_final_result(goal)
        return result

    async def _reason(self, goal: Any, observation: str) -> ReactThought:
        """
        Reason about the current situation and decide on next action.

        Args:
            goal: The original goal
            observation: Current observation from last action

        Returns:
            ReactThought with reasoning and action decision
        """
        # Build context for reasoning
        context = {
            "goal": str(goal),
            "agent_description": self.description,
            "instructions": self.instructions or "No specific instructions.",
            "current_observation": observation,
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
                    "reasoning": (
                        cycle.thought.reasoning
                        if hasattr(cycle.thought, "reasoning")
                        else str(cycle.thought)
                    ),
                    "action": (cycle.tool_calls[0].name if cycle.tool_calls else None),
                    "result": (
                        "success"
                        if cycle.results and cycle.results[0].success
                        else "failure"
                    ),
                }
                for cycle in self.context.get_last_n_cycles(3)
            ],
            "current_iteration": self.context.iteration + 1,
            "max_iterations": self.max_iterations,
        }

        instructions = """You are using the ReAct (Reasoning + Acting) pattern.

YOUR TASK:
1. Reason about the current observation and situation
2. Decide if the goal is complete or if you need to take an action
3. If complete: set is_complete=True and action=None
4. If not complete: set is_complete=False and specify the action to take

IMPORTANT:
- You can only call ONE tool per iteration
- Analyze the observation carefully before deciding
- Use available tools to accomplish the goal
- Set is_complete=True when you have enough information to answer the goal
"""

        # Trigger: llm_call
        await self.hook_manager.trigger(
            HookEvents.LLM_CALL,
            self.context,
            agent=self,
            call_type="reason",
        )

        # Call Opper
        response = await self.opper.call_async(
            name="react_reason",
            instructions=instructions,
            input=context,
            output_schema=ReactThought,
            model=self.model,
            parent_span_id=self.context.parent_span_id,
        )

        # Trigger: llm_response
        await self.hook_manager.trigger(
            HookEvents.LLM_RESPONSE,
            self.context,
            agent=self,
            call_type="reason",
            response=response,
        )

        thought = ReactThought(**response.json_payload)

        # Trigger: think_end (for consistency with base Agent)
        await self.hook_manager.trigger(
            HookEvents.THINK_END,
            self.context,
            agent=self,
            thought=thought,
        )

        return thought
