"""
Tests for the agent logging system.
"""

import pytest
from io import StringIO
import sys
from opper_agents.utils.logging import SimpleLogger, RichLogger
from opper_agents import Agent, tool


def test_simple_logger_basic():
    """Test SimpleLogger basic functionality."""
    logger = SimpleLogger()

    # Capture stdout
    captured = StringIO()
    sys.stdout = captured

    logger.log_iteration(1, 5)
    logger.log_thought("Testing the thought process", 2)
    logger.log_tool_call("test_tool", {"param": "value"})
    logger.log_tool_result("test_tool", True, "Success result", None)
    logger.log_warning("Test warning")
    logger.log_error("Test error")

    sys.stdout = sys.__stdout__

    output = captured.getvalue()
    assert "Iteration 1/5" in output
    assert "Testing the thought process" in output
    assert "Tool calls: 2" in output
    assert "test_tool" in output
    assert "SUCCESS" in output
    assert "Warning: Test warning" in output
    assert "Error: Test error" in output


def test_simple_logger_memory():
    """Test SimpleLogger memory operations."""
    logger = SimpleLogger()

    captured = StringIO()
    sys.stdout = captured

    logger.log_memory_read(["key1", "key2"])
    logger.log_memory_loaded({"key1": "value1", "key2": "value2"})
    logger.log_memory_write(["key1"])

    sys.stdout = sys.__stdout__

    output = captured.getvalue()
    assert "Loading memory keys" in output
    assert "key1" in output
    assert "Loaded memory" in output
    assert "Writing to memory" in output


def test_simple_logger_thinking_context():
    """Test SimpleLogger thinking context manager."""
    logger = SimpleLogger()

    # Should not raise any errors
    with logger.log_thinking():
        pass


def test_rich_logger_basic():
    """Test RichLogger basic functionality."""
    logger = RichLogger()

    # Test that it doesn't crash (output is hard to test with Rich)
    logger.log_iteration(1, 5)
    logger.log_thought("Testing with Rich", 3)
    logger.log_tool_call("rich_tool", {"param": "value"})
    logger.log_tool_result("rich_tool", True, "Success", None)
    logger.log_warning("Rich warning")
    logger.log_error("Rich error")


def test_rich_logger_thinking_spinner():
    """Test RichLogger thinking spinner context manager."""
    logger = RichLogger()

    # Should not raise any errors
    with logger.log_thinking():
        pass


@pytest.mark.asyncio
async def test_agent_with_simple_logger_default(mock_opper_client):
    """Test Agent with SimpleLogger enabled via verbose=True (default)."""

    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    # Mock LLM responses
    from unittest.mock import AsyncMock

    mock_opper_client.call_async = AsyncMock(
        return_value=AsyncMock(
            json_payload={
                "reasoning": "I need to add the numbers",
                "tool_calls": [],
                "memory_reads": [],
                "memory_updates": {},
            },
            message="Final result",
            usage={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        )
    )

    agent = Agent(
        name="TestAgent",
        tools=[add],
        verbose=True,  # Should enable SimpleLogger by default
        opper_api_key="test-key",
    )

    # Verify logger is set and is SimpleLogger
    assert agent.logger is not None
    assert isinstance(agent.logger, SimpleLogger)

    result = await agent.process("Test task")
    assert result is not None


@pytest.mark.asyncio
async def test_agent_with_rich_logger_custom(mock_opper_client):
    """Test Agent with RichLogger explicitly set."""

    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    # Mock LLM responses
    from unittest.mock import AsyncMock

    mock_opper_client.call_async = AsyncMock(
        return_value=AsyncMock(
            json_payload={
                "reasoning": "I need to add the numbers",
                "tool_calls": [],
                "memory_reads": [],
                "memory_updates": {},
            },
            message="Final result",
            usage={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        )
    )

    rich_logger = RichLogger()

    agent = Agent(
        name="TestAgent",
        tools=[add],
        logger=rich_logger,
        opper_api_key="test-key",
    )

    # Verify RichLogger is used
    assert agent.logger is rich_logger
    assert isinstance(agent.logger, RichLogger)

    result = await agent.process("Test task")
    assert result is not None


@pytest.mark.asyncio
async def test_agent_with_custom_logger(mock_opper_client):
    """Test Agent with custom SimpleLogger."""

    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    # Mock LLM responses
    from unittest.mock import AsyncMock

    mock_opper_client.call_async = AsyncMock(
        return_value=AsyncMock(
            json_payload={
                "reasoning": "I need to add the numbers",
                "tool_calls": [],
                "memory_reads": [],
                "memory_updates": {},
            },
            message="Final result",
            usage={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        )
    )

    custom_logger = SimpleLogger()

    agent = Agent(
        name="TestAgent",
        tools=[add],
        logger=custom_logger,
        opper_api_key="test-key",
    )

    # Verify custom logger is used
    assert agent.logger is custom_logger
    assert isinstance(agent.logger, SimpleLogger)

    result = await agent.process("Test task")
    assert result is not None


@pytest.mark.asyncio
async def test_agent_no_logger_when_verbose_false(mock_opper_client):
    """Test Agent has no logger when verbose=False and no logger provided."""

    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    # Mock LLM responses
    from unittest.mock import AsyncMock

    mock_opper_client.call_async = AsyncMock(
        return_value=AsyncMock(
            json_payload={
                "reasoning": "I need to add the numbers",
                "tool_calls": [],
                "memory_reads": [],
                "memory_updates": {},
            },
            message="Final result",
            usage={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        )
    )

    agent = Agent(
        name="TestAgent",
        tools=[add],
        verbose=False,
        opper_api_key="test-key",
    )

    # Verify no logger is set
    assert agent.logger is None

    result = await agent.process("Test task")
    assert result is not None
