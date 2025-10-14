"""
SDK version information.

This module provides version metadata that is included in the User-Agent header
for all Opper API requests made by this SDK.
"""

import importlib.metadata

# SDK package name
SDK_NAME = "opper-agents"

# SDK version from package metadata
try:
    SDK_VERSION = importlib.metadata.version("opper-agents")
except importlib.metadata.PackageNotFoundError:
    # Fallback for development/editable installs
    SDK_VERSION = "0.1.0"

# Platform identifier (python for Python)
SDK_PLATFORM = "python"


def get_user_agent() -> str:
    """
    Returns the formatted User-Agent string for this SDK.

    Format: opper-agents-python/0.1.0

    Returns:
        str: The formatted User-Agent string
    """
    return f"{SDK_NAME}-{SDK_PLATFORM}/{SDK_VERSION}"
