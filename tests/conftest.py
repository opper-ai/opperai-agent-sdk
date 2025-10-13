"""
Pytest configuration and shared fixtures.

This module provides common fixtures for all tests, including:
- Event loop management for async tests
- Mock fixtures for Opper API calls
- VCR configuration for integration tests
"""

import asyncio
import os
import pytest
import vcr
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
def mock_opper_client(monkeypatch):
    """
    Mock Opper client for agent initialization.

    Returns a mock client with common methods stubbed out.
    Useful for testing agent logic without actual API calls.
    """
    mock = MagicMock()
    mock.call = AsyncMock()
    mock.call_async = AsyncMock()  # Add call_async for async operations
    mock.spans = MagicMock()
    mock.spans.create = MagicMock(return_value=MagicMock(id="test-span-id"))
    mock.spans.create_async = AsyncMock(return_value=MagicMock(id="test-span-id"))
    mock.spans.update = MagicMock()
    mock.spans.update_async = AsyncMock()

    # Patch the Opper class constructor to return our mock
    def mock_opper_init(*args, **kwargs):
        return mock

    monkeypatch.setattr("opper_agents.base.agent.Opper", mock_opper_init)
    return mock


@pytest.fixture
def vcr_cassette(request):
    """
    VCR fixture for recording/replaying HTTP interactions in integration tests.

    This fixture automatically creates cassettes named after the test function
    and stores them in tests/fixtures/vcr_cassettes/{module}/{test_name}.yaml

    Usage:
        @pytest.mark.asyncio
        async def test_my_integration(vcr_cassette):
            # HTTP calls will be recorded/replayed automatically
            pass
    """
    test_name = request.node.name
    module_file = request.module.__file__
    file_name = os.path.splitext(os.path.basename(module_file))[0]

    cassette_name = f"{file_name}/{test_name}.yaml"

    my_vcr = vcr.VCR(
        cassette_library_dir="tests/fixtures/vcr_cassettes",
        path_transformer=vcr.VCR.ensure_suffix(".yaml"),
        filter_headers=[
            "authorization",
            "api-key",
            "x-opper-api-key",
        ],
        ignore_hosts=["testserver"],
        ignore_localhost=True,
    )

    with my_vcr.use_cassette(cassette_name):
        yield


@pytest.fixture
def opper_api_key():
    """
    Get Opper API key from environment for integration/e2e tests.

    Falls back to 'test-key' for local development.
    """
    return os.getenv("OPPER_API_KEY", "test-key")
