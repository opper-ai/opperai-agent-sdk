"""
Base abstractions for the Opper Agent SDK.

This module contains the core abstract classes and data models
that all agent implementations build upon.
"""

from .agent import BaseAgent
from .context import AgentContext, Usage, ExecutionCycle
from .tool import Tool, FunctionTool, ToolResult
from .hooks import HookManager, HookEvents

__all__ = [
    "BaseAgent",
    "AgentContext",
    "Usage",
    "ExecutionCycle",
    "Tool",
    "FunctionTool",
    "ToolResult",
    "HookManager",
    "HookEvents",
]
