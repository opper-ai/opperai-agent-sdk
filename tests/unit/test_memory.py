"""
Unit tests for agent memory system.

Tests the Memory and MemoryEntry classes for storing and retrieving
agent state across execution cycles.
"""

import pytest
from opper_agents.memory.memory import Memory


@pytest.mark.asyncio
async def test_memory_initialization():
    """Test Memory initialization."""
    memory = Memory()
    assert memory.has_entries() is False
    assert len(memory.store) == 0


@pytest.mark.asyncio
async def test_memory_write():
    """Test writing to memory."""
    memory = Memory()
    await memory.write(
        key="project",
        value={"status": "in_progress"},
        description="Project snapshot",
        metadata={"priority": "high"},
    )

    assert memory.has_entries() is True
    assert "project" in memory.store
    assert memory.store["project"].value["status"] == "in_progress"
    assert memory.store["project"].description == "Project snapshot"


@pytest.mark.asyncio
async def test_memory_read_single_key():
    """Test reading a single key from memory."""
    memory = Memory()
    await memory.write(
        "project", {"status": "in_progress"}, description="Project state"
    )

    payload = await memory.read(["project"])
    assert "project" in payload
    assert payload["project"]["status"] == "in_progress"


@pytest.mark.asyncio
async def test_memory_read_all():
    """Test reading all memory entries."""
    memory = Memory()
    await memory.write("key1", "value1", description="First")
    await memory.write("key2", "value2", description="Second")

    payload = await memory.read()  # Read all
    assert len(payload) == 2
    assert payload["key1"] == "value1"
    assert payload["key2"] == "value2"


@pytest.mark.asyncio
async def test_memory_read_nonexistent_key():
    """Test reading a key that doesn't exist."""
    memory = Memory()
    await memory.write("exists", "value", description="Exists")

    payload = await memory.read(["exists", "doesnt_exist"])
    assert "exists" in payload
    assert "doesnt_exist" not in payload


@pytest.mark.asyncio
async def test_memory_update():
    """Test updating an existing memory entry."""
    memory = Memory()
    await memory.write("counter", 1, description="Counter")
    await memory.write("counter", 2, description="Updated counter")

    payload = await memory.read(["counter"])
    assert payload["counter"] == 2
    assert memory.store["counter"].description == "Updated counter"


@pytest.mark.asyncio
async def test_memory_update_with_metadata():
    """Test updating memory entry with metadata."""
    memory = Memory()
    await memory.write("data", {"value": 1}, description="Data", metadata={"v": 1})
    await memory.write("data", {"value": 2}, metadata={"v": 2})

    entry = memory.store["data"]
    assert entry.value["value"] == 2
    assert entry.metadata["v"] == 2


@pytest.mark.asyncio
async def test_memory_list_entries():
    """Test listing memory catalog."""
    memory = Memory()
    await memory.write(
        "project",
        {"status": "active"},
        description="Project state",
        metadata={"team": "A"},
    )
    await memory.write(
        "config", {"env": "prod"}, description="Configuration", metadata={"version": 1}
    )

    catalog = await memory.list_entries()
    assert len(catalog) == 2

    # Check catalog structure (should not include values)
    project_entry = next(e for e in catalog if e["key"] == "project")
    assert project_entry["key"] == "project"
    assert project_entry["description"] == "Project state"
    assert project_entry["metadata"]["team"] == "A"
    assert "value" not in project_entry  # Values not in catalog


@pytest.mark.asyncio
async def test_memory_clear():
    """Test clearing all memory."""
    memory = Memory()
    await memory.write("key1", "value1", description="First")
    await memory.write("key2", "value2", description="Second")

    assert memory.has_entries() is True
    await memory.clear()
    assert memory.has_entries() is False
    assert len(memory.store) == 0


@pytest.mark.asyncio
async def test_memory_entry_last_accessed():
    """Test that last_accessed is updated on read."""
    memory = Memory()
    await memory.write("test", "value", description="Test")

    initial_access = memory.store["test"].last_accessed

    # Small delay to ensure time difference
    import asyncio

    await asyncio.sleep(0.01)

    await memory.read(["test"])
    updated_access = memory.store["test"].last_accessed

    assert updated_access > initial_access


@pytest.mark.asyncio
async def test_memory_complex_values():
    """Test storing complex nested data structures."""
    memory = Memory()
    complex_data = {
        "users": [{"name": "Alice", "role": "admin"}, {"name": "Bob", "role": "user"}],
        "config": {"timeout": 30, "retries": 3, "endpoints": ["api1", "api2"]},
    }

    await memory.write("app_state", complex_data, description="Application state")

    payload = await memory.read(["app_state"])
    assert len(payload["app_state"]["users"]) == 2
    assert payload["app_state"]["users"][0]["name"] == "Alice"
    assert payload["app_state"]["config"]["timeout"] == 30


@pytest.mark.asyncio
async def test_memory_default_description():
    """Test that key is used as default description."""
    memory = Memory()
    await memory.write("my_key", "value")  # No description

    entry = memory.store["my_key"]
    assert entry.description == "my_key"
