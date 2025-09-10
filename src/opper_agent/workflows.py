from __future__ import annotations

import asyncio
import inspect
import time
import uuid
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Tuple, Type, TypeVar, Union, Generic, get_type_hints

from pydantic import BaseModel


I = TypeVar("I", bound=BaseModel)
O = TypeVar("O", bound=BaseModel)


class ExecutionContext:
    """Simplified execution context for the new data + context pattern."""
    
    def __init__(self, step_context):
        """Initialize from a full StepContext."""
        self._step_context = step_context
    
    @property
    def state(self) -> Dict[str, Any]:
        """Access to step state."""
        return self._step_context.state
    
    @property
    def run_id(self) -> str:
        """Workflow run ID."""
        return self._step_context.run_id
    
    @property
    def step_id(self) -> str:
        """Current step ID."""
        return self._step_context.step_id
    
    @property
    def opper(self):
        """Access to Opper client."""
        return self._step_context.opper
    
    @property
    def tools(self) -> Dict[str, Any]:
        """Available tools."""
        return self._step_context.tools
    
    @property
    def memory(self) -> Any:
        """Workflow memory."""
        return self._step_context.memory
    
    async def llm(self, *, name: str, instructions: str, input_schema: Optional[Type[BaseModel]] = None, 
                 output_schema: Optional[Type[BaseModel]] = None, input: Optional[BaseModel] = None, 
                 model: Optional[str] = None) -> BaseModel:
        """Call AI model using opper.call syntax - delegates to full StepContext."""
        return await self._step_context.call_model(
            name=name, instructions=instructions, input_schema=input_schema,
            output_schema=output_schema, input_obj=input, model=model
        )
    
    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit event - delegates to full StepContext."""
        return self._step_context._emit_event(event_type, data)


def _serialize_model(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    return obj


async def _maybe_await(value: Any) -> Any:
    if asyncio.iscoroutine(value):
        return await value
    return value


class StepContext(Generic[I, O]):
    """Execution context passed to each step run function."""

    def __init__(
        self,
        *,
        input_data: I,
        state: Dict[str, Any],
        run_id: str,
        step_id: str,
        opper,
        parent_span_id: Optional[str],
        tools: Dict[str, Any],
        memory: Any,
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]],
        emit: Callable[[Dict[str, Any]], None],
        checkpoint: Callable[[Any], Awaitable[None]],
        default_model: Optional[str] = None,
    ) -> None:
        self.input_data = input_data
        self.state = state
        self.run_id = run_id
        self.step_id = step_id
        self.opper = opper
        self.parent_span_id = parent_span_id
        self.tools = tools
        self.memory = memory
        self.event_callback = event_callback
        self.emit = emit
        self.checkpoint = checkpoint
        self.default_model = default_model

    def _emit_event(self, event_type: str, data: Dict[str, Any] = None):
        """Emit an event if callback is available."""
        if self.event_callback:
            event_data = {
                "run_id": self.run_id,
                "step_id": self.step_id,
                "timestamp": time.time(),
                **(data or {})
            }
            self.event_callback(event_type, event_data)

    async def call_model(
        self,
        *,
        name: str,
        instructions: str,
        input_schema: Optional[Type[BaseModel]] = None,
        output_schema: Optional[Type[BaseModel]] = None,
        input_obj: Optional[BaseModel] = None,
        model: Optional[str] = None,
    ) -> BaseModel:
        """Helper for structured LLM calls via Opper using schemas.

        Dynamic variables are passed through input, not interpolated into the prompt.
        """
        # Use provided model or fall back to default model
        effective_model = model or self.default_model
        
        # Emit model call start event
        self._emit_event("model_call_start", {
            "call_name": name,
            "model": effective_model,
            "input_schema": input_schema.__name__ if input_schema else None,
            "output_schema": output_schema.__name__ if output_schema else None
        })
        
        try:
            result = self.opper.call(
                name=name,
                instructions=instructions,
                input_schema=input_schema,
                output_schema=output_schema,
                input=input_obj,
                model=effective_model,
                parent_span_id=self.parent_span_id,
            )
            
            # Emit model call success event
            self._emit_event("model_call_success", {
                "call_name": name,
                "model": effective_model
            })
            
            return result.json_payload  # Expecting dict compatible with output_schema
            
        except Exception as e:
            # Emit model call error event
            self._emit_event("model_call_error", {
                "call_name": name,
                "model": effective_model,
                "error": str(e)
            })
            raise


class StepDef(Generic[I, O]):
    def __init__(
        self,
        *,
        id: str,
        input_model: Type[I],
        output_model: Type[O],
        run: Callable[[StepContext[I, O]], Awaitable[O]],
        description: Optional[str] = None,
        retry: Optional[Dict[str, Any]] = None,
        timeout_ms: Optional[int] = None,
        on_error: str = "fail",
        map_in: Optional[Callable[[Any], Any]] = None,
        map_out: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        self.id = id
        self.input_model = input_model
        self.output_model = output_model
        self.run = run
        self.description = description
        self.retry = retry or {"attempts": 1}
        self.timeout_ms = timeout_ms
        self.on_error = on_error
        self.map_in = map_in
        self.map_out = map_out


class Step(Generic[I, O]):
    def __init__(self, defn: StepDef[I, O]) -> None:
        self.defn = defn


def create_step(
    *,
    id: str,
    input_model: Type[I],
    output_model: Type[O],
    run: Callable[[StepContext[I, O]], Awaitable[O]],
    description: Optional[str] = None,
    retry: Optional[Dict[str, Any]] = None,
    timeout_ms: Optional[int] = None,
    on_error: str = "fail",
    map_in: Optional[Callable[[Any], Any]] = None,
    map_out: Optional[Callable[[Any], Any]] = None,
) -> Step[I, O]:
    return Step(
        StepDef(
            id=id,
            input_model=input_model,
            output_model=output_model,
            run=run,
            description=description,
            retry=retry,
            timeout_ms=timeout_ms,
            on_error=on_error,
            map_in=map_in,
            map_out=map_out,
        )
    )


def step(
    func: Callable = None,
    *,
    id: str = None,
    description: str = None,
    retry: Optional[Dict[str, Any]] = None,
    timeout_ms: Optional[int] = None,
    on_error: str = "fail",
    map_in: Optional[Callable[[Any], Any]] = None,
    map_out: Optional[Callable[[Any], Any]] = None,
):
    """
    Decorator to convert a function into a Step.
    
    Automatically extracts input and output models from function type hints.
    Supports two patterns:
    
    1. Context-only (legacy): func(ctx: StepContext[InputModel, OutputModel]) -> OutputModel
    2. Data + Context (preferred): func(data: InputModel, ctx: StepContext) -> OutputModel
    
    Args:
        func: The function to wrap (when used without parentheses)
        id: Optional custom ID for the step (defaults to function name)
        description: Optional custom description (defaults to function docstring)
        retry: Optional retry configuration
        timeout_ms: Optional timeout in milliseconds
        on_error: Error handling strategy ("fail", "skip", or "continue")
        map_in: Optional input transformation function
        map_out: Optional output transformation function
    
    Usage:
        # Preferred: Data + Context pattern
        @step
        async def process_data(data: InputModel, ctx: StepContext) -> OutputModel:
            \"\"\"Process some data.\"\"\"
            result = await ctx.call_model(...)
            return OutputModel(...)
        
        # Legacy: Context-only pattern (still supported)
        @step
        async def process_data(ctx: StepContext[InputModel, OutputModel]) -> OutputModel:
            data = ctx.input_data
            return OutputModel(...)
    """
    def decorator(f: Callable) -> Step:
        # Extract step ID
        step_id = id or f.__name__
        
        # Extract description
        step_description = description or f.__doc__ or f"Execute {f.__name__}"
        
        # Extract type hints and determine pattern
        try:
            type_hints = get_type_hints(f)
            sig = inspect.signature(f)
            params = list(sig.parameters.values())
            
            if not params:
                raise ValueError(f"Step function {f.__name__} must take at least one parameter")
            
            # Get the return type (output model)
            return_type = type_hints.get('return')
            if not return_type:
                raise ValueError(f"Step function {f.__name__} must have a return type annotation")
            
            # Determine pattern based on parameter count and types
            if len(params) == 1:
                # Legacy pattern: func(ctx: StepContext[I, O]) -> O
                first_param = params[0]
                if first_param.name != 'ctx':
                    raise ValueError(f"Step function {f.__name__} single parameter should be named 'ctx'")
                
                # Extract input and output types from StepContext[I, O]
                ctx_type = type_hints.get('ctx') or first_param.annotation
                
                if hasattr(ctx_type, '__args__') and len(ctx_type.__args__) >= 2:
                    input_model = ctx_type.__args__[0]
                    output_model = ctx_type.__args__[1]
                else:
                    # Fallback: try to extract from return type
                    input_model = BaseModel
                    output_model = return_type
                
                # Use original function as-is for legacy pattern
                step_function = f
                
            elif len(params) == 2:
                # New pattern: func(data: InputModel, ctx: StepContext) -> OutputModel
                data_param = params[0]
                ctx_param = params[1]
                
                if ctx_param.name != 'ctx':
                    raise ValueError(f"Step function {f.__name__} second parameter should be named 'ctx'")
                
                # Extract input model from first parameter
                input_model = type_hints.get(data_param.name) or data_param.annotation
                output_model = return_type
                
                # Create wrapper function that adapts new pattern to legacy StepContext pattern
                async def step_function(ctx):
                    return await f(ctx.input_data, ExecutionContext(ctx))
                
            else:
                raise ValueError(f"Step function {f.__name__} must take 1 or 2 parameters, got {len(params)}")
            
        except Exception as e:
            raise ValueError(f"Could not extract type information from step function {f.__name__}: {e}")
        
        return create_step(
            id=step_id,
            input_model=input_model,
            output_model=output_model,
            run=step_function,
            description=step_description,
            retry=retry,
            timeout_ms=timeout_ms,
            on_error=on_error,
            map_in=map_in,
            map_out=map_out,
        )
    
    if func is None:
        # Called with arguments: @step(id="something")
        return decorator
    else:
        # Called without arguments: @step
        return decorator(func)


class Workflow(Generic[I, O]):
    def __init__(self, *, id: str, input_model: Type[I], output_model: Type[O]) -> None:
        self.id = id
        self.input_model = input_model
        self.output_model = output_model
        self.pipeline: List[Tuple[str, Any]] = []

    def then(self, next_item: Union["Step[O, Any]", "Workflow[O, Any]"]) -> "Workflow[I, Any]":
        self.pipeline.append(("then", next_item))
        return self  # type: ignore[return-value]

    def parallel(
        self,
        items: List[Union["Step[O, Any]", "Workflow[O, Any]"]],
        *,
        concurrency: Optional[int] = None,
    ) -> "Workflow[I, List[Any]]":
        self.pipeline.append(("parallel", (items, concurrency)))
        return self  # type: ignore[return-value]

    def branch(
        self,
        cases: List[Tuple[Callable[[Any], Union[bool, Awaitable[bool]]], Union["Step[O, Any]", "Workflow[O, Any]"]]],
    ) -> "Workflow[I, List[Any]]":
        self.pipeline.append(("branch", cases))
        return self  # type: ignore[return-value]

    def dowhile(
        self,
        item: Union["Step[Any, Any]", "Workflow[Any, Any]"],
        pred: Callable[[Any], Union[bool, Awaitable[bool]]],
    ) -> "Workflow[I, Any]":
        self.pipeline.append(("dowhile", (item, pred)))
        return self  # type: ignore[return-value]

    def dountil(
        self,
        item: Union["Step[Any, Any]", "Workflow[Any, Any]"],
        pred: Callable[[Any], Union[bool, Awaitable[bool]]],
    ) -> "Workflow[I, Any]":
        self.pipeline.append(("dountil", (item, pred)))
        return self  # type: ignore[return-value]

    def foreach(
        self,
        item: Union["Step[Any, Any]", "Workflow[Any, Any]"],
        *,
        concurrency: Optional[int] = None,
        map_func: Optional[Callable[[Any], Iterable[Any]]] = None,
    ) -> "Workflow[I, List[Any]]":
        self.pipeline.append(("foreach", (item, concurrency, map_func)))
        return self  # type: ignore[return-value]

    def map(self, mapper: Callable[[Any], Any]) -> "Workflow[I, Any]":
        self.pipeline.append(("map", mapper))
        return self  # type: ignore[return-value]

    def commit(self) -> "FinalizedWorkflow[I, O]":
        return FinalizedWorkflow(self)


def clone_workflow(wf: "FinalizedWorkflow[I, O]", *, id: Optional[str] = None) -> Workflow[I, O]:
    new = Workflow(id=id or f"{wf.id}-clone", input_model=wf.input_model, output_model=wf.output_model)
    new.pipeline = list(wf.pipeline)
    return new


class Storage:
    async def save(self, run_id: str, data: Any) -> None:
        raise NotImplementedError

    async def load(self, run_id: str) -> Any:
        raise NotImplementedError


class InMemoryStorage(Storage):
    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}

    async def save(self, run_id: str, data: Any) -> None:
        self._store[run_id] = data

    async def load(self, run_id: str) -> Any:
        return self._store.get(run_id)


class FinalizedWorkflow(Generic[I, O]):
    def __init__(self, wf: Workflow[I, O]) -> None:
        self.id = wf.id
        self.input_model = wf.input_model
        self.output_model = wf.output_model
        self.pipeline = wf.pipeline

    def create_run(
        self,
        *,
        opper,
        storage: Storage,
        tools: Optional[Dict[str, Any]] = None,
        memory: Any = None,
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        default_model: Optional[str] = None,
    ) -> "WorkflowRun[I, O]":
        return WorkflowRun(self, opper=opper, storage=storage, tools=tools or {}, memory=memory, event_callback=event_callback, default_model=default_model)


class WorkflowRun(Generic[I, O]):
    def __init__(
        self,
        wf: FinalizedWorkflow[I, O],
        *,
        opper,
        storage: Storage,
        tools: Dict[str, Any],
        memory: Any,
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]],
        default_model: Optional[str] = None,
    ) -> None:
        self.wf = wf
        self.opper = opper
        self.storage = storage
        self.tools = tools
        self.memory = memory
        self.event_callback = event_callback
        self.default_model = default_model
        self.run_id = str(uuid.uuid4())
        self.parent_span_id: Optional[str] = None

    def _emit_event(self, event_type: str, data: Dict[str, Any] = None):
        """Emit an event if callback is available."""
        if self.event_callback:
            event_data = {
                "run_id": self.run_id,
                "workflow_id": self.wf.id,
                "timestamp": time.time(),
                **(data or {})
            }
            self.event_callback(event_type, event_data)

    async def start(self, *, input_data: I) -> O:
        # Emit workflow start event
        self._emit_event("workflow_start", {
            "input_data": _serialize_model(input_data)
        })
        
        # Create a run-level span; associate to parent if provided
        if self.parent_span_id:
            session_span = self.opper.spans.create(name=self.wf.id, input=_serialize_model(input_data), parent_id=self.parent_span_id)
        else:
            session_span = self.opper.spans.create(name=self.wf.id, input=_serialize_model(input_data))
        self.parent_span_id = getattr(session_span, "id", None)
        out: Any = input_data
        try:
            for kind, payload in self.wf.pipeline:
                if kind == "then":
                    out = await self._exec_item(payload, out)
                elif kind == "map":
                    out = await _maybe_await(payload(out))
                elif kind == "parallel":
                    items, conc = payload
                    semaphore = asyncio.Semaphore(conc) if conc else None

                    async def _run_parallel(itm):
                        if semaphore is None:
                            return await self._exec_item(itm, out)
                        async with semaphore:
                            return await self._exec_item(itm, out)

                    out = await asyncio.gather(*[_run_parallel(i) for i in items])
                elif kind == "branch":
                    tasks: List[Awaitable[Any]] = []
                    for cond, itm in payload:
                        ok = await _maybe_await(cond(out))
                        if ok:
                            tasks.append(self._exec_item(itm, out))
                    out = await asyncio.gather(*tasks) if tasks else []
                elif kind == "dowhile":
                    itm, pred = payload
                    cur = out
                    while await _maybe_await(pred(cur)):
                        cur = await self._exec_item(itm, cur)
                    out = cur
                elif kind == "dountil":
                    itm, pred = payload
                    cur = out
                    while not await _maybe_await(pred(cur)):
                        cur = await self._exec_item(itm, cur)
                    out = cur
                elif kind == "foreach":
                    itm, conc, map_func = payload
                    src_iter = list(map_func(out)) if map_func else list(out)
                    semaphore = asyncio.Semaphore(conc) if conc else None

                    async def _run_foreach(x):
                        if semaphore is None:
                            return await self._exec_item(itm, x)
                        async with semaphore:
                            return await self._exec_item(itm, x)

                    out = await asyncio.gather(*[_run_foreach(x) for x in src_iter])
                await self._checkpoint(out)
            
            # Emit workflow success event
            self._emit_event("workflow_success", {
                "output_data": _serialize_model(out)
            })
            
            return out  # type: ignore[return-value]
        except Exception as e:
            # Emit workflow error event
            self._emit_event("workflow_error", {
                "error": str(e)
            })
            raise
        finally:
            try:
                self.opper.spans.update(
                    span_id=self.parent_span_id,
                    output=_serialize_model(out),
                )
            except Exception:
                pass

    async def _exec_item(self, item: Any, input_obj: Any) -> Any:
        if isinstance(item, FinalizedWorkflow):
            sub_run = WorkflowRun(item, opper=self.opper, storage=self.storage, tools=self.tools, memory=self.memory, event_callback=self.event_callback, default_model=self.default_model)
            # Chain sub-run under the same parent run span
            sub_run.parent_span_id = self.parent_span_id
            return await sub_run.start(input_data=input_obj)
        if isinstance(item, Step):
            return await self._exec_step(item, input_obj)
        raise TypeError("Unsupported item in workflow pipeline")

    async def _exec_step(self, step: Step[Any, Any], input_obj: Any) -> Any:
        if step.defn.map_in is not None:
            mapped_in = step.defn.map_in(input_obj)
            model_in = step.defn.input_model.model_validate(_serialize_model(mapped_in))
        else:
            payload = _serialize_model(input_obj)
            model_in = step.defn.input_model.model_validate(payload)

        # Emit step start event
        self._emit_event("step_start", {
            "step_id": step.defn.id,
            "input_data": _serialize_model(model_in)
        })
        
        # Do not create per-step spans; Opper calls will create their own child spans.
        ctx = StepContext(
            input_data=model_in,
            state={},
            run_id=self.run_id,
            step_id=step.defn.id,
            opper=self.opper,
            parent_span_id=self.parent_span_id,
            tools=self.tools,
            memory=self.memory,
            event_callback=self.event_callback,
            emit=lambda e: None,
            checkpoint=self._checkpoint,
            default_model=self.default_model,
        )

        attempts = int(step.defn.retry.get("attempts", 1) if step.defn.retry else 1)
        backoff = step.defn.retry.get("backoff_ms", 0) if step.defn.retry else 0
        last_exc: Optional[BaseException] = None
        for i in range(attempts):
            try:
                if step.defn.timeout_ms:
                    result = await asyncio.wait_for(step.defn.run(ctx), timeout=step.defn.timeout_ms / 1000.0)
                else:
                    result = await step.defn.run(ctx)
                out_obj = result
                
                # Emit step success event
                self._emit_event("step_success", {
                    "step_id": step.defn.id,
                    "output_data": _serialize_model(result)
                })
                break
            except Exception as ex:  # noqa: BLE001
                last_exc = ex
                
                # Emit step retry event if not the last attempt
                if i < attempts - 1:
                    self._emit_event("step_retry", {
                        "step_id": step.defn.id,
                        "attempt": i + 1,
                        "max_attempts": attempts,
                        "error": str(ex)
                    })
                    delay = backoff(i) if callable(backoff) else backoff
                    if delay:
                        await asyncio.sleep(delay / 1000.0)
                    continue
                
                # Emit step error event for final failure
                self._emit_event("step_error", {
                    "step_id": step.defn.id,
                    "error": str(ex),
                    "on_error": step.defn.on_error
                })
                
                if step.defn.on_error == "fail":
                    raise
                if step.defn.on_error in ("skip", "continue"):
                    out_obj = input_obj
                    break
        if last_exc and step.defn.on_error == "fail":
            raise last_exc

        if step.defn.map_out is not None:
            out_obj = step.defn.map_out(out_obj)

        model_out = step.defn.output_model.model_validate(_serialize_model(out_obj))
        return model_out

    async def _checkpoint(self, data: Any) -> None:
        await self.storage.save(run_id=self.run_id, data=_serialize_model(data))


__all__ = [
    "StepContext",
    "ExecutionContext",
    "StepDef",
    "Step",
    "create_step",
    "step",
    "Workflow",
    "FinalizedWorkflow",
    "clone_workflow",
    "Storage",
    "InMemoryStorage",
    "WorkflowRun",
]
