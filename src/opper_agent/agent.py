from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel

from .workflows import FinalizedWorkflow, Storage, WorkflowRun


class Agent:
    def __init__(
        self,
        *,
        name: str,
        instructions: str,
        flow: FinalizedWorkflow[Any, Any],
        model: Optional[str] = None,
        memory: Optional[Any] = None,
    ) -> None:
        self.name = name
        self.instructions = instructions
        self.flow = flow
        self.model = model
        self.memory = memory

    def get_description(self) -> str:
        return self.instructions

    def get_flow(self) -> FinalizedWorkflow[Any, Any]:
        return self.flow

    def create_run(
        self,
        *,
        opper,
        storage: Storage,
        tools: Optional[Dict[str, Any]] = None,
        logger: Optional[Any] = None,
    ) -> WorkflowRun[Any, Any]:
        return self.flow.create_run(opper=opper, storage=storage, tools=tools or {}, memory=self.memory, logger=logger)
