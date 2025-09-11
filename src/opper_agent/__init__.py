from .workflows import (
    StepContext,
    ExecutionContext,
    StepDef,
    Step,
    create_step,
    step,
    Workflow,
    FinalizedWorkflow,
    clone_workflow,
    Storage,
    InMemoryStorage,
    WorkflowRun,
)
from .base_agent import Agent, tool

# MCP support (optional import to avoid breaking existing code)
try:
    from .mcp_client import MCPClient, MCPServerConfig, MCPServers
    from .mcp_tools import MCPToolAdapter, MCPToolManager, create_mcp_tools, mcp_tools

    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False
    MCPClient = None
    MCPServerConfig = None
    MCPServers = None
    MCPToolAdapter = None
    MCPToolManager = None
    create_mcp_tools = None
    mcp_tools = None

__all__ = [
    "StepContext",
    "ExecutionContext",
    "StepDef",
    "Step",
    "create_step",
    "step",
    "Workflow",
    "FinalizedWorkflow",
    "clone_workflow",
    "Storage",
    "InMemoryStorage",
    "WorkflowRun",
    "Agent",
    "tool",
]

# Add MCP exports if available
if _MCP_AVAILABLE:
    __all__.extend(
        [
            "MCPClient",
            "MCPServerConfig",
            "MCPServers",
            "MCPToolAdapter",
            "MCPToolManager",
            "create_mcp_tools",
            "mcp_tools",
        ]
    )
