"""Tests for User-Agent header configuration in agents."""

import pytest
from opper_agents import Agent
from opper_agents.utils.version import get_user_agent


@pytest.mark.asyncio
async def test_agent_sets_custom_user_agent():
    """Test that Agent properly sets the custom User-Agent in Opper SDK."""
    agent = Agent(
        name="TestAgent",
        description="Test agent for User-Agent verification",
        opper_api_key="test-api-key",
    )

    # Verify the User-Agent is set correctly in the Opper SDK configuration
    expected_user_agent = get_user_agent()
    actual_user_agent = agent.opper.sdk_configuration.user_agent

    assert actual_user_agent == expected_user_agent
    assert "opper-agents-python" in actual_user_agent
    assert "/" in actual_user_agent  # Should have format: name/version


@pytest.mark.asyncio
async def test_user_agent_format():
    """Test that User-Agent follows the expected format."""
    agent = Agent(name="TestAgent", opper_api_key="test-api-key")

    user_agent = agent.opper.sdk_configuration.user_agent

    # Should match format: opper-agents-python/x.y.z
    parts = user_agent.split("/")
    assert len(parts) == 2
    assert parts[0] == "opper-agents-python"
    assert parts[1]  # Version should exist

    # Version should look like semver (at least x.y.z)
    version_parts = parts[1].split(".")
    assert len(version_parts) >= 3
