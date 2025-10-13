"""
Agent memory system.

This module provides in-memory storage with LLM-friendly catalog access.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import time


class MemoryEntry(BaseModel):
    """Single memory slot with metadata."""

    key: str
    description: str
    value: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)
    last_accessed: float = Field(default_factory=time.time)

    class Config:
        arbitrary_types_allowed = True


class Memory(BaseModel):
    """Fast in-memory store that exposes a catalog to the LLM."""

    store: Dict[str, MemoryEntry] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    def has_entries(self) -> bool:
        """Check if memory has any entries."""
        return len(self.store) > 0

    async def list_entries(self) -> List[Dict[str, Any]]:
        """Summaries so the LLM can decide whether to load anything."""
        catalog: List[Dict[str, Any]] = []
        for entry in self.store.values():
            catalog.append(
                {
                    "key": entry.key,
                    "description": entry.description,
                    "metadata": entry.metadata,
                }
            )
        return catalog

    async def read(self, keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """Return payloads for the selected keys (or all when keys is None)."""
        if not keys:
            keys = list(self.store.keys())
        snapshot: Dict[str, Any] = {}
        for key in keys:
            if key in self.store:
                entry = self.store[key]
                entry.last_accessed = time.time()
                snapshot[key] = entry.value
        return snapshot

    async def write(
        self,
        key: str,
        value: Any,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Insert or update a memory entry."""
        if key in self.store:
            entry = self.store[key]
            entry.value = value
            if description:
                entry.description = description
            if metadata:
                entry.metadata.update(metadata)
            entry.last_accessed = time.time()
        else:
            self.store[key] = MemoryEntry(
                key=key,
                description=description or key,
                value=value,
                metadata=metadata or {},
            )

    async def clear(self) -> None:
        """Clear all memory entries."""
        self.store.clear()
