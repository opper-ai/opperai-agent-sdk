"""
Opper Agent SDK - Build reliable AI agents with clean architecture.

Main exports:
    - Agent: Main agent implementation
    - ReactAgent: ReAct pattern agent
    - tool: Decorator to create tools from functions
    - hook: Decorator to create lifecycle hooks
    - AgentContext: Execution context manager
    - Memory: Agent memory system
"""

# Version
__version__ = "0.1.0"

# Core exports
from .core.agent import Agent
from .utils.decorators import tool, hook
from .base.context import AgentContext
from .memory.memory import Memory

# from .agents.react import ReactAgent  # Phase 4
# from .mcp.provider import mcp  # Phase 4
# from .mcp.config import MCPServerConfig  # Phase 4

__all__ = [
    "__version__",
    "Agent",
    "tool",
    "hook",
    "AgentContext",
    "Memory",
    # "ReactAgent",
    # "mcp",
    # "MCPServerConfig",
]
