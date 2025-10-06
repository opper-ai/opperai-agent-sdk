from .base_agent import Agent, tool, hook, AgentHooks, RunContext, Usage, FunctionTool

# MCP support (optional import to avoid breaking existing code)
try:
    from .mcp import (
        MCPClient,
        MCPServerConfig,
        MCPServers,
        MCPToolAdapter,
        MCPToolManager,
        create_mcp_tools,
        create_mcp_tools_async,
        mcp_tools,
    )

    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False
    MCPClient = None
    MCPServerConfig = None
    MCPServers = None
    MCPToolAdapter = None
    MCPToolManager = None
    create_mcp_tools = None
    create_mcp_tools_async = None
    mcp_tools = None

__all__ = [
    "Agent",
    "tool",
    "hook",
    "AgentHooks",
    "RunContext",
    "Usage",
    "FunctionTool",
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
            "create_mcp_tools_async",
            "mcp_tools",
        ]
    )
