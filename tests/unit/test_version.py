"""Tests for version utilities."""

import re
from opper_agents.utils.version import (
    SDK_NAME,
    SDK_VERSION,
    SDK_PLATFORM,
    get_user_agent,
)


def test_sdk_name():
    """Test SDK_NAME constant."""
    assert SDK_NAME == "opper-agents"


def test_sdk_platform():
    """Test SDK_PLATFORM constant."""
    assert SDK_PLATFORM == "python"


def test_sdk_version():
    """Test SDK_VERSION is a valid semver string."""
    assert SDK_VERSION is not None
    assert isinstance(SDK_VERSION, str)
    # Should match semver pattern (e.g., 0.1.0, 1.2.3, etc.)
    assert re.match(r"^\d+\.\d+\.\d+", SDK_VERSION)


def test_get_user_agent():
    """Test get_user_agent returns correct format."""
    user_agent = get_user_agent()
    expected = f"{SDK_NAME}-{SDK_PLATFORM}/{SDK_VERSION}"

    assert user_agent == expected
    # Should match pattern: opper-agents-python/0.1.0
    assert re.match(r"^opper-agents-python/\d+\.\d+\.\d+", user_agent)
