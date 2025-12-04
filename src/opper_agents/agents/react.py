"""
ReAct Agent implementation.

This module implements the ReAct (Reasoning + Acting) pattern agent.
"""

from typing import Any, Optional, Type

from ..core.agent import Agent
from ..core.schemas import (
    ReactThought,
    ToolCall,
    create_react_thought_with_output_schema,
)
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
        assert self.context is not None, "Context must be initialized"
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
                # Check for final result in thought (single LLM call pattern)
                if thought.final_result is not None:
                    final_result = thought.final_result

                    # If output_schema is specified, validate the final_result
                    if self.output_schema:
                        # Already validated by dynamic schema (Pydantic did it during ReactThought construction)
                        if isinstance(final_result, self.output_schema):
                            pass  # Already correct type
                        elif isinstance(final_result, dict):
                            try:
                                final_result = self.output_schema(**final_result)
                            except Exception:
                                # Validation failed - fall back to _generate_final_result
                                if self.verbose:
                                    print(
                                        "Final result validation failed - generating structured result"
                                    )
                                break
                        else:
                            # final_result is not a dict and not the expected type
                            # Fall back to _generate_final_result
                            if self.verbose:
                                print(
                                    "Final result not structured - generating structured result"
                                )
                            break

                    if self.verbose:
                        print(
                            f"✓ Task completed in {self.context.iteration + 1} iteration(s)"
                        )

                    # Trigger loop end before returning
                    await self.hook_manager.trigger(
                        HookEvents.LOOP_END, self.context, agent=self
                    )

                    return final_result

                # Fallback to old behavior
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
        assert self.context is not None, "Context must be initialized"
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
                        if cycle.thought and hasattr(cycle.thought, "reasoning")
                        else str(cycle.thought)
                        if cycle.thought
                        else ""
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
3. If complete:
   - Set is_complete=True
   - Provide the complete answer/output in final_result
   - Set action=None
4. If not complete:
   - Set is_complete=False
   - Specify the action to take
   - Leave final_result as None

IMPORTANT:
- When task is COMPLETE, you MUST set is_complete=True AND provide final_result
- The final_result should be a complete, well-structured answer based on all work done
- You can only call ONE tool per iteration
- Analyze the observation carefully before deciding
- Use available tools to accomplish the goal
- If an output_schema was specified, ensure final_result matches that schema
"""

        # Trigger: llm_call
        await self.hook_manager.trigger(
            HookEvents.LLM_CALL,
            self.context,
            agent=self,
            call_type="reason",
        )

        # Create dynamic ReactThought schema with typed final_result if output_schema is specified
        thought_schema = create_react_thought_with_output_schema(self.output_schema)

        # Generate function name: reason_{agent_name} (sanitized)
        sanitized_name = self.name.lower().replace(" ", "_").replace("-", "_")
        function_name = f"reason_{sanitized_name}"

        # Choose streaming or non-streaming path
        if getattr(self, "enable_streaming", False):
            thought = await self._reason_streaming(
                context, instructions, thought_schema, function_name
            )
        else:
            # Call Opper (non-streaming)
            response = await self.opper.call_async(
                name=function_name,
                instructions=instructions,
                input=context,
                output_schema=thought_schema,  # type: ignore[arg-type]
                model=self.model,
                parent_span_id=self.context.parent_span_id,
            )

            # Track usage (non-streaming)
            try:
                # reuse Agent helper
                from ..core.agent import Agent as AgentImpl

                AgentImpl._track_usage(self, response)
            except Exception:
                pass

            # Trigger: llm_response
            await self.hook_manager.trigger(
                HookEvents.LLM_RESPONSE,
                self.context,
                agent=self,
                call_type="reason",
                response=response,
            )

            thought = thought_schema(**response.json_payload)

        # Trigger: think_end (for consistency with base Agent)
        await self.hook_manager.trigger(
            HookEvents.THINK_END,
            self.context,
            agent=self,
            thought=thought,
        )

        return thought

    async def _reason_streaming(
        self,
        context: dict,
        instructions: str,
        thought_schema: Type[ReactThought],
        function_name: str,
    ) -> ReactThought:
        """Streaming version of _reason() for ReAct agent."""
        assert self.context is not None, "Context must be initialized"

        # Trigger: stream_start
        await self.hook_manager.trigger(
            HookEvents.STREAM_START,
            self.context,
            agent=self,
            call_type="reason",
        )

        field_buffers: dict[str, list[str]] = {}
        stream_span_id: Optional[str] = None

        try:
            stream_response = await self.opper.stream_async(
                name=function_name,
                instructions=instructions,
                input=context,
                output_schema=thought_schema,  # type: ignore[arg-type]
                model=self.model,
                parent_span_id=self.context.parent_span_id,
            )

            async for event in stream_response.result:
                if not hasattr(event, "data"):
                    continue

                data = event.data
                # Capture span id if provided
                if (
                    hasattr(data, "span_id")
                    and getattr(data, "span_id")
                    and not stream_span_id
                ):
                    stream_span_id = getattr(data, "span_id")
                if not hasattr(data, "delta") or data.delta is None:
                    continue

                # Handle Unset/None values by defaulting to "_root"
                json_path = getattr(data, "json_path", None)
                if json_path is None or not isinstance(json_path, str):
                    json_path = "_root"

                if json_path not in field_buffers:
                    field_buffers[json_path] = []
                field_buffers[json_path].append(str(data.delta))

                accumulated = "".join(field_buffers[json_path])

                await self.hook_manager.trigger(
                    HookEvents.STREAM_CHUNK,
                    self.context,
                    agent=self,
                    call_type="reason",
                    chunk_data={
                        "delta": data.delta,
                        "json_path": json_path,
                        "chunk_type": getattr(data, "chunk_type", None),
                    },
                    accumulated=accumulated,
                    field_buffers=field_buffers.copy(),
                )

            # If no json_path was provided, fall back to creating a basic ReactThought
            if set(field_buffers.keys()) == {"_root"}:
                thought_text = "".join(field_buffers.get("_root", []))
                thought = thought_schema(reasoning=thought_text, is_complete=True)
            else:
                # Reconstruct full JSON response using Agent helper
                from ..core.agent import Agent as BaseAgentImpl

                json_response = BaseAgentImpl._reconstruct_json_from_buffers(
                    self, field_buffers, thought_schema
                )
                thought = thought_schema(**json_response)

            # Track usage for streaming via span lookup if direct usage missing
            try:
                if hasattr(stream_response, "usage") and getattr(
                    stream_response, "usage"
                ):
                    from ..core.agent import Agent as AgentImpl

                    AgentImpl._track_usage(self, stream_response)
                elif stream_span_id:
                    span = await self.opper.spans.get_async(span_id=stream_span_id)
                    trace_id = getattr(span, "trace_id", None)
                    if trace_id:
                        trace = await self.opper.traces.get_async(trace_id=trace_id)
                        spans = getattr(trace, "spans", None) or []
                        total_tokens = None
                        for s in spans:
                            if getattr(s, "id", None) == stream_span_id:
                                data = getattr(s, "data", None)
                                total_tokens = (
                                    getattr(data, "total_tokens", None)
                                    if data
                                    else None
                                )
                                break
                        if total_tokens is not None:
                            from ..base.context import Usage as _Usage

                            self.context.update_usage(
                                _Usage(
                                    requests=1,
                                    input_tokens=0,
                                    output_tokens=0,
                                    total_tokens=int(total_tokens),
                                )
                            )
            except Exception:
                pass

            await self.hook_manager.trigger(
                HookEvents.LLM_RESPONSE,
                self.context,
                agent=self,
                call_type="reason",
                response=stream_response,
                parsed=thought,
            )

            return thought

        except Exception as e:
            await self.hook_manager.trigger(
                HookEvents.STREAM_ERROR,
                self.context,
                agent=self,
                call_type="reason",
                error=e,
            )
            raise
        finally:
            # Always emit STREAM_END, even if exception occurred
            # This ensures consumers know the stream lifecycle is complete
            await self.hook_manager.trigger(
                HookEvents.STREAM_END,
                self.context,
                agent=self,
                call_type="reason",
                field_buffers=field_buffers.copy(),
            )
