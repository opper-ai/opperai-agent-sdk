"""
Pytest configuration and shared fixtures.

This module provides common fixtures for all tests, including:
- Event loop management for async tests
- Mock fixtures for Opper API calls
- VCR configuration (coming in Phase 3)
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture(scope="session")
def event_loop():
    """
    Create a session-scoped event loop for all async tests.

    This prevents issues with nested event loops and ensures consistent
    async behavior across all tests, especially for agent-as-tool scenarios.
    """
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture
def mock_acompletion(monkeypatch):
    """
    Mock opper.call() responses for testing agents.

    Usage in tests:
        @pytest.mark.asyncio
        async def test_agent(mock_acompletion):
            mock_acompletion.return_value = AsyncMock(
                json_payload={"reasoning": "...", "tool_calls": []}
            )
            # Test code here

    This is the standard way to mock LLM responses per CLAUDE.md.
    """
    mock = AsyncMock()
    return mock


@pytest.fixture
def mock_opper_client():
    """
    Mock Opper client for agent initialization.

    Returns a mock client with common methods stubbed out.
    Useful for testing agent logic without actual API calls.
    """
    mock = MagicMock()
    mock.call = AsyncMock()
    mock.spans = MagicMock()
    mock.spans.create = MagicMock(return_value=MagicMock(id="test-span-id"))
    mock.spans.update = MagicMock()
    return mock


# TODO: Add VCR configuration for Phase 3 integration tests
# @pytest.fixture(scope="module")
# def vcr_config():
#     """VCR configuration for recording/replaying HTTP interactions."""
#     return {
#         "filter_headers": ["authorization", "x-api-key"],
#         "record_mode": "once",
#     }

# TODO: Add API key fixture for Phase 3 e2e tests
# @pytest.fixture
# def opper_api_key():
#     """Get Opper API key from environment for e2e tests."""
#     import os
#     return os.getenv("OPPER_API_KEY", "test-key")
