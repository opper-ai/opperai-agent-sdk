"""
Memory system for agents.

This module provides in-memory storage with LLM-friendly catalog access.
"""

from .memory import Memory, MemoryEntry

# Rebuild AgentContext now that Memory is defined
from ..base.context import AgentContext

AgentContext.model_rebuild()

__all__ = [
    "Memory",
    "MemoryEntry",
]
