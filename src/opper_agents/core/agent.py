"""
Main Agent implementation using 'while tools > 0' loop.

This module contains the primary Agent class that implements the think-act loop.
"""

from typing import Any, Optional, List, Type
from datetime import datetime, timezone
from pydantic import BaseModel

from ..base.agent import BaseAgent
from ..base.context import AgentContext, ExecutionCycle, Usage
from ..memory.memory import Memory
from ..base.hooks import HookEvents
from ..base.tool import ToolResult
from .schemas import Thought, ToolCall, create_thought_with_output_schema


class Agent(BaseAgent):
    """
    Main agent implementation using 'while tools > 0' loop.

    Loop Logic:
    - Think: Decide next actions
    - If tool_calls > 0: Execute tools and continue
    - If tool_calls == 0: Generate final result
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
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

            start_time = datetime.now(timezone.utc)

            # Create parent span for this agent execution
            # If _parent_span_id is provided, this span will be nested under it
            parent_span = await self.opper.spans.create_async(
                name=f"{self.name}_execution",
                input=str(input),
                parent_id=_parent_span_id,
                type="agent 🤖",  # Feature 2
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

                end_time = datetime.now(timezone.utc)

                with anyio.CancelScope(shield=True):
                    await self.opper.spans.update_async(
                        span_id=parent_span.id,
                        output=str(result),
                        start_time=start_time,
                        end_time=end_time,
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
        assert self.context is not None, "Context must be initialized"
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

                    mem_read_start = datetime.now(timezone.utc)
                    memory_read_span = await self.opper.spans.create_async(
                        name="memory_read",
                        input=str(thought.memory_reads),
                        parent_id=self.context.parent_span_id,
                        type="memory 🧠",  # Feature 2
                    )

                    memory_data = await self.context.memory.read(thought.memory_reads)

                    mem_read_end = datetime.now(timezone.utc)

                    await self.opper.spans.update_async(
                        span_id=memory_read_span.id,
                        output=str(memory_data),
                        start_time=mem_read_start,
                        end_time=mem_read_end,
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

                    mem_write_start = datetime.now(timezone.utc)
                    memory_write_span = await self.opper.spans.create_async(
                        name="memory_write",
                        input=str(list(thought.memory_updates.keys())),
                        parent_id=self.context.parent_span_id,
                        type="memory 🧠",  # Feature 2
                    )

                    for key, update in thought.memory_updates.items():
                        await self.context.memory.write(
                            key=key,
                            value=update.get("value"),
                            description=update.get("description"),
                            metadata=update.get("metadata"),
                        )

                    mem_write_end = datetime.now(timezone.utc)

                    await self.opper.spans.update_async(
                        span_id=memory_write_span.id,
                        output=f"Successfully wrote {len(thought.memory_updates)} keys",
                        start_time=mem_write_start,
                        end_time=mem_write_end,
                    )

                    memory_writes_performed = True

                if thought is not None:
                    # Check for immediate completion with final result (single LLM call pattern)
                    if thought.is_complete and thought.final_result is not None:
                        final_result = thought.final_result

                        # If output_schema is specified, validate the final_result
                        if self.output_schema:
                            # Already validated by dynamic schema (Pydantic did it during Thought construction)
                            if isinstance(final_result, self.output_schema):
                                pass  # Already correct type
                            elif isinstance(final_result, dict):
                                try:
                                    final_result = self.output_schema(**final_result)
                                except Exception:
                                    # Validation failed - fall back to _generate_final_result
                                    break
                            else:
                                # final_result is not a dict and not the expected type
                                # Fall back to _generate_final_result
                                break

                        # Trigger loop end before returning
                        await self.hook_manager.trigger(
                            HookEvents.LOOP_END, self.context, agent=self
                        )

                        if self.verbose:
                            print(
                                f"✓ Task completed in {self.context.iteration + 1} iteration(s)"
                            )

                        return final_result

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

        # Fallback to old behavior if no final_result was provided (backward compatibility)
        result = await self._generate_final_result(goal)
        return result

    async def _think(self, goal: Any) -> Thought:
        """Call LLM to reason about next actions."""
        assert self.context is not None, "Context must be initialized"

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
                        if cycle.thought and hasattr(cycle.thought, "reasoning")
                        else str(cycle.thought)
                        if cycle.thought
                        else ""
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
4. If goal complete:
   - Set is_complete=true
   - Provide the complete answer/output in final_result
   - Leave tool_calls empty

IMPORTANT:
- When task is COMPLETE, you MUST set is_complete=true AND provide final_result
- The final_result should be a complete, well-structured answer based on all work done
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

        # Create dynamic Thought schema with typed final_result if output_schema is specified
        thought_schema = create_thought_with_output_schema(self.output_schema)

        # Generate function name: think_{agent_name} (sanitized)
        sanitized_name = self.name.lower().replace(" ", "_").replace("-", "_")
        function_name = f"think_{sanitized_name}"

        if self.enable_streaming:
            # STREAMING PATH
            thought = await self._think_streaming(
                context, instructions, thought_schema, function_name
            )
        else:
            # NON-STREAMING PATH (existing code)
            response = await self.opper.call_async(
                name=function_name,
                instructions=instructions,
                input=context,
                output_schema=thought_schema,  # type: ignore[arg-type]
                model=self.model,
                parent_span_id=self.context.parent_span_id,
            )

            # Rename span to simpler "think" (best-effort, may fail silently)
            # First ensure span exists by fetching it (like Node SDK does)
            if hasattr(response, "span_id") and response.span_id:
                try:
                    await self.opper.spans.get_async(span_id=response.span_id)
                    await self.opper.spans.update_async(
                        span_id=response.span_id, name="think"
                    )
                except Exception:
                    pass  # Span may not be updatable

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

            thought = thought_schema(**response.json_payload)

        # Trigger: think_end (same for both paths)
        await self.hook_manager.trigger(
            HookEvents.THINK_END, self.context, agent=self, thought=thought
        )

        return thought

    async def _execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a single tool call and create a span for it."""
        assert self.context is not None, "Context must be initialized"

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
        tool_start_time = datetime.now(timezone.utc)
        tool_span = await self.opper.spans.create_async(
            name=f"tool_{tool_call.name}",
            input=str(tool_call.parameters),
            parent_id=self.context.parent_span_id,
            type="tool 🔧",  # Feature 2
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

        tool_end_time = datetime.now(timezone.utc)

        # Update tool span with result
        await self.opper.spans.update_async(
            span_id=tool_span.id,
            output=str(result.result) if result.success else None,
            error=result.error if not result.success else None,
            start_time=tool_start_time,
            end_time=tool_end_time,
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
        assert self.context is not None, "Context must be initialized"
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
            if self.enable_streaming:
                # STREAMING PATH
                result = await self._generate_final_result_streaming(
                    context, instructions
                )
            else:
                # NON-STREAMING PATH (existing code)
                # Generate dynamic function name for better traceability
                sanitized_name = self.name.lower().replace(" ", "_").replace("-", "_")
                function_name = f"generate_final_result_{sanitized_name}"

                response = await self.opper.call_async(
                    name=function_name,
                    instructions=instructions,
                    input=context,
                    output_schema=self.output_schema,  # type: ignore[arg-type]
                    model=self.model,
                    parent_span_id=self.context.parent_span_id,
                )

                # Track usage
                self._track_usage(response)

                # Serialize the response
                if self.output_schema:
                    result = self.output_schema(**response.json_payload)
                else:
                    result = response.message

            return result

    def _track_usage(self, response: Any) -> None:
        """
        Track token usage from an Opper response.

        Safely extracts usage info if available, otherwise skips tracking.
        """
        if not hasattr(response, "usage") or not response.usage:
            return

        try:
            assert self.context is not None, "Context must be initialized"
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

    async def _think_streaming(
        self,
        context: dict,
        instructions: str,
        thought_schema: Type[Thought],
        function_name: str,
    ) -> Thought:
        """
        Streaming version of _think().

        Accumulates chunks, emits hooks, then parses final result.

        Args:
            context: Context dict to send to LLM
            instructions: Instructions for the LLM
            thought_schema: The Thought schema to use (may include typed final_result)
            function_name: The function name for Opper call

        Returns:
            Parsed Thought model
        """
        # Ensure context is initialized for type checkers
        assert self.context is not None, "Context must be initialized"

        # Trigger: stream_start
        await self.hook_manager.trigger(
            HookEvents.STREAM_START, self.context, agent=self, call_type="think"
        )

        # Buffer for accumulating response
        field_buffers: dict[str, list[str]] = {}  # json_path -> [deltas]
        stream_span_id: Optional[str] = None

        try:
            # Call Opper streaming API
            stream_response = await self.opper.stream_async(
                name=function_name,
                instructions=instructions,
                input=context,
                output_schema=thought_schema,  # type: ignore[arg-type]
                model=self.model,
                parent_span_id=self.context.parent_span_id,
            )

            # Consume stream
            async for event in stream_response.result:
                if not hasattr(event, "data"):
                    continue

                data = event.data

                # Capture span id if provided (usually first chunk)
                if (
                    hasattr(data, "span_id")
                    and getattr(data, "span_id")
                    and not stream_span_id
                ):
                    stream_span_id = getattr(data, "span_id")

                # Skip chunks without delta (but after capturing span id)
                if not hasattr(data, "delta") or data.delta is None:
                    continue

                # Accumulate by json_path (for structured) or root (for text)
                # Handle Unset/None values by defaulting to "_root"
                json_path = getattr(data, "json_path", None)
                if json_path is None or not isinstance(json_path, str):
                    json_path = "_root"

                if json_path not in field_buffers:
                    field_buffers[json_path] = []
                field_buffers[json_path].append(str(data.delta))

                # Calculate accumulated text for this field
                accumulated = "".join(field_buffers[json_path])

                # Trigger: stream_chunk hook
                await self.hook_manager.trigger(
                    HookEvents.STREAM_CHUNK,
                    self.context,
                    agent=self,
                    call_type="think",
                    chunk_data={
                        "delta": data.delta,
                        "json_path": json_path,
                        "chunk_type": getattr(data, "chunk_type", None),
                    },
                    accumulated=accumulated,
                    field_buffers=field_buffers.copy(),
                )

            # If no json_path was provided, fall back to putting all text in reasoning
            if set(field_buffers.keys()) == {"_root"}:
                thought_text = "".join(field_buffers.get("_root", []))
                thought = thought_schema(reasoning=thought_text)
            else:
                # Reconstruct full JSON response
                json_response = self._reconstruct_json_from_buffers(
                    field_buffers, thought_schema
                )

                # Parse with Pydantic (same as non-streaming)
                thought = thought_schema(**json_response)

            # Track usage if available directly, otherwise via span lookup
            if hasattr(stream_response, "usage") and getattr(stream_response, "usage"):
                self._track_usage(stream_response)
            elif stream_span_id:
                try:
                    # Fetch span -> trace -> span data to get total_tokens
                    span = await self.opper.spans.get_async(span_id=stream_span_id)
                    trace_id = getattr(span, "trace_id", None)
                    if trace_id:
                        trace = await self.opper.traces.get_async(trace_id=trace_id)
                        spans = getattr(trace, "spans", None) or []
                        total_tokens: Optional[int] = None
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
                            usage = Usage(
                                requests=1,
                                input_tokens=0,
                                output_tokens=0,
                                total_tokens=int(total_tokens),
                            )
                            self.context.update_usage(usage)
                except Exception as usage_exc:  # pragma: no cover - defensive
                    if self.logger:
                        self.logger.log_warning(
                            f"Could not fetch streaming usage: {usage_exc}"
                        )

            # Rename span to simpler "think" (best-effort, may fail silently)
            # First ensure span exists by fetching it (like Node SDK does)
            if stream_span_id:
                try:
                    # Fetch span first to ensure it's committed in the database
                    await self.opper.spans.get_async(span_id=stream_span_id)
                    await self.opper.spans.update_async(
                        span_id=stream_span_id, name="think"
                    )
                except Exception:
                    pass  # Span may not be updatable

            # Trigger: llm_response (include raw stream response and parsed model)
            await self.hook_manager.trigger(
                HookEvents.LLM_RESPONSE,
                self.context,
                agent=self,
                call_type="think",
                response=stream_response,
                parsed=thought,
            )

            return thought

        except Exception as e:
            # Trigger: stream_error
            await self.hook_manager.trigger(
                HookEvents.STREAM_ERROR,
                self.context,
                agent=self,
                call_type="think",
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
                call_type="think",
                field_buffers=field_buffers.copy(),
            )

    def _reconstruct_json_from_buffers(
        self, field_buffers: dict[str, list[str]], schema: Type[BaseModel]
    ) -> dict:
        """
        Reconstruct JSON object from streaming buffers.

        Takes the accumulated field buffers from streaming and reconstructs
        a complete JSON dict that can be parsed by Pydantic.

        Args:
            field_buffers: Dict mapping json_path to list of delta strings
                          e.g., {"reasoning": ["I", " need", " to"],
                                 "tool_calls[0].name": ["search", "_web"]}
            schema: Pydantic model for structure hints

        Returns:
            Complete JSON dict ready for Pydantic parsing
            e.g., {"reasoning": "I need to",
                   "tool_calls": [{"name": "search_web"}]}
        """
        result: dict[str, Any] = {}

        # If only root text exists, return it as text
        if set(field_buffers.keys()) == {"_root"}:
            return {"text": "".join(field_buffers.get("_root", []))}

        for json_path, deltas in field_buffers.items():
            if json_path == "_root":
                # Ignore stray root chunks when structured fields are present
                continue

            # Reconstruct nested structure from json_path
            # e.g., "reasoning" -> result["reasoning"]
            # e.g., "tool_calls[0].name" -> result["tool_calls"][0]["name"]
            full_value = "".join(deltas)
            self._set_nested_value(result, json_path, full_value, schema)

        # Normalize any dicts that actually represent indexed arrays
        normalized = self._normalize_indexed(result)

        # Special repair for Thought.tool_calls to ensure proper list-of-dicts shape
        # Also handle dynamic schemas like Thought_Story that inherit from Thought
        try:
            from .schemas import Thought as ThoughtSchema

            # Use issubclass to handle dynamic Thought_{OutputSchema} variants
            if issubclass(schema, ThoughtSchema) and isinstance(normalized, dict):
                tc = normalized.get("tool_calls")
                # Convert dict of indexed items to list
                if isinstance(tc, dict):
                    ordered: list[Any] = []
                    for idx, v in sorted(
                        ((int(k), val) for k, val in tc.items()), key=lambda x: x[0]
                    ):
                        # Flatten nested single-item lists
                        if isinstance(v, list) and v and isinstance(v[0], dict):
                            v = v[0]
                        ordered.append(v)
                    normalized["tool_calls"] = ordered
                elif isinstance(tc, list):
                    fixed: list[Any] = []
                    for v in tc:
                        if isinstance(v, list) and v and isinstance(v[0], dict):
                            fixed.append(v[0])
                        else:
                            fixed.append(v)
                    normalized["tool_calls"] = fixed

                # Handle empty string final_result - should be None for Optional field
                # This happens when streaming doesn't produce a final_result value
                fr = normalized.get("final_result")
                if fr == "" or fr == {}:
                    normalized["final_result"] = None
        except Exception:
            # Best-effort; do not break if repair fails
            pass

        from typing import cast, Dict

        return cast(Dict[str, Any], normalized)

    def _normalize_indexed(self, obj: Any) -> Any:
        """Return a deep-normalized version where dicts with integer-like
        keys are converted into lists in numeric order."""
        if isinstance(obj, dict):
            # Normalize children first
            normalized_items = {k: self._normalize_indexed(v) for k, v in obj.items()}
            keys = list(normalized_items.keys())
            if keys and all(
                isinstance(k, int) or (isinstance(k, str) and str(k).isdigit())
                for k in keys
            ):
                max_index = max(int(k) for k in keys)
                new_list: list[Any] = [None] * (max_index + 1)
                for k, v in normalized_items.items():
                    new_list[int(k)] = v
                return new_list
            return normalized_items
        if isinstance(obj, list):
            return [self._normalize_indexed(v) for v in obj]
        return obj

    def _set_nested_value(
        self, obj: dict, path: str, value: Any, schema: Type[BaseModel]
    ) -> None:
        """
        Set nested value in dict using dot notation path.

        Handles both object nesting and array indices.

        Examples:
            path="reasoning", value="Think about it"
            -> obj["reasoning"] = "Think about it"

            path="tool_calls[0].name", value="search_web"
            -> obj["tool_calls"][0]["name"] = "search_web"

            path="metadata.score", value="95"
            -> obj["metadata"]["score"] = 95 (coerced to int)

        Args:
            obj: Target dict to modify
            path: Dot notation path with optional array indices
            value: Value to set (will be type-coerced)
            schema: Pydantic model for type inference
        """
        import re

        # Parse path supporting both bracket and dot index styles
        # Examples:
        #   "tool_calls[0].name" -> ["tool_calls", 0, "name"]
        #   "tool_calls.0.name"  -> ["tool_calls", 0, "name"]
        parts: list[object] = []
        for raw in path.split("."):
            match = re.match(r"(\w+)\[(\d+)\]", raw)
            if match:
                parts.append(match.group(1))
                parts.append(int(match.group(2)))
            elif raw.isdigit():
                parts.append(int(raw))
            else:
                parts.append(raw)

        # Navigate to parent and create structure as needed
        current = obj
        last_list_container: Optional[list] = None
        for i, part in enumerate(parts[:-1]):
            next_part = parts[i + 1]

            if isinstance(next_part, int):
                # Next part is an array index
                if part not in current:
                    current[part] = []
                # Ensure array is long enough
                while len(current[part]) <= next_part:
                    current[part].append({})
                last_list_container = current[part]
                current = current[part][next_part]
            else:
                # Next part is an object key
                if part not in current:
                    current[part] = {}
                current = current[part]

        # Set final value with type coercion
        final_key = parts[-1]
        coerced = self._coerce_type(value, schema, path)
        if isinstance(final_key, int):
            # Path ends with an array index: set the list element directly
            if last_list_container is None:
                # Create a new list if somehow missing
                last_list_container = []
            while len(last_list_container) <= final_key:
                last_list_container.append({})
            last_list_container[final_key] = coerced
        else:
            current[final_key] = coerced

    def _coerce_type(self, value: str, schema: Type[BaseModel], json_path: str) -> Any:
        """
        Coerce string value to correct type based on Pydantic schema.

        The API sends all deltas as strings during streaming, but the final
        JSON needs proper types (int, float, bool, etc.) for Pydantic validation.

        This method uses heuristics and schema inspection to determine the
        correct type for each field.

        Args:
            value: String value from streaming
            schema: Pydantic model to inspect for type hints
            json_path: Path to the field (for schema lookup)

        Returns:
            Value coerced to appropriate type

        Examples:
            "true" -> True
            "false" -> False
            "123" -> 123
            "45.67" -> 45.67
            "hello" -> "hello"
        """
        # Simple heuristic for common types

        # Boolean
        if value.lower() in ("true", "false"):
            return value.lower() == "true"

        # Number
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            pass

        # TODO: Could inspect Pydantic schema for better type inference
        # For now, heuristics are good enough for most cases

        return value  # Keep as string

    async def _generate_final_result_streaming(
        self, context: dict, instructions: str
    ) -> Any:
        """
        Streaming version of _generate_final_result().

        Similar to _think_streaming() but uses self.output_schema
        instead of Thought schema.

        Args:
            context: Context dict to send to LLM
            instructions: Instructions for the LLM

        Returns:
            Parsed model (if output_schema) or text string
        """
        # Ensure context is initialized for type checkers
        assert self.context is not None, "Context must be initialized"

        # Trigger: stream_start
        await self.hook_manager.trigger(
            HookEvents.STREAM_START,
            self.context,
            agent=self,
            call_type="final_result",
        )

        field_buffers: dict[str, list[str]] = {}
        stream_span_id: Optional[str] = None

        # Generate dynamic function name for better traceability
        sanitized_name = self.name.lower().replace(" ", "_").replace("-", "_")
        function_name = f"generate_final_result_{sanitized_name}"

        try:
            stream_response = await self.opper.stream_async(
                name=function_name,
                instructions=instructions,
                input=context,
                output_schema=self.output_schema,  # type: ignore[arg-type]
                model=self.model,
                parent_span_id=self.context.parent_span_id,
            )

            async for event in stream_response.result:
                if not hasattr(event, "data"):
                    continue

                data = event.data

                # Capture span id if provided (usually first chunk)
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
                    call_type="final_result",
                    chunk_data={
                        "delta": data.delta,
                        "json_path": json_path,
                        "chunk_type": getattr(data, "chunk_type", None),
                    },
                    accumulated=accumulated,
                    field_buffers=field_buffers.copy(),
                )

            # Handle case where no output_schema is defined
            result: Any
            if self.output_schema:
                json_response = self._reconstruct_json_from_buffers(
                    field_buffers, self.output_schema
                )
                result = self.output_schema(**json_response)
            else:
                # Text mode: just concatenate all deltas
                result = "".join(
                    delta for deltas in field_buffers.values() for delta in deltas
                )

            if hasattr(stream_response, "usage") and getattr(stream_response, "usage"):
                self._track_usage(stream_response)
            elif stream_span_id:
                try:
                    span = await self.opper.spans.get_async(span_id=stream_span_id)
                    trace_id = getattr(span, "trace_id", None)
                    if trace_id:
                        trace = await self.opper.traces.get_async(trace_id=trace_id)
                        spans = getattr(trace, "spans", None) or []
                        total_tokens: Optional[int] = None
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
                            usage = Usage(
                                requests=1,
                                input_tokens=0,
                                output_tokens=0,
                                total_tokens=int(total_tokens),
                            )
                            self.context.update_usage(usage)
                except Exception as usage_exc:  # pragma: no cover - defensive
                    if self.logger:
                        self.logger.log_warning(
                            f"Could not fetch streaming usage: {usage_exc}"
                        )

            await self.hook_manager.trigger(
                HookEvents.LLM_RESPONSE,
                self.context,
                agent=self,
                call_type="final_result",
                response=stream_response,
                parsed=result,
            )

            return result

        except Exception as e:
            await self.hook_manager.trigger(
                HookEvents.STREAM_ERROR,
                self.context,
                agent=self,
                call_type="final_result",
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
                call_type="final_result",
                field_buffers=field_buffers.copy(),
            )
