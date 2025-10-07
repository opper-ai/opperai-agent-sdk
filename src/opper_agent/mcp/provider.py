"""
MCP Tool Provider.

Wraps MCP servers as ToolProviders that can be used seamlessly
with agents alongside regular tools.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
import logging

if TYPE_CHECKING:
    from ..base.agent import BaseAgent

from ..base.tool import FunctionTool, Tool
from .client import MCPClient, MCPTool
from .config import MCPServerConfig

logger = logging.getLogger(__name__)


class MCPToolProvider:
    """
    ToolProvider wrapper around one or more MCP servers.

    This class implements the ToolProvider protocol, allowing MCP servers to be
    used seamlessly as tools in agent configurations. It manages the lifecycle of
    connections to multiple MCP servers and exposes their tools to agents.

    The provider handles:
    - Automatic connection/disconnection to MCP servers
    - Tool discovery and wrapping as FunctionTool objects
    - Error handling and graceful degradation if servers fail
    - Tool name prefixing to avoid conflicts between servers

    Attributes:
        configs: List of MCP server configurations to connect to.
        name_prefix: Optional prefix for all tool names (defaults to server name).
        clients: Dictionary mapping server names to their MCPClient instances.

    Examples:
        # Single MCP server
        filesystem = MCPServerConfig(
            name="filesystem",
            transport="stdio",
            command="uvx",
            args=["mcp-server-filesystem", "/tmp"]
        )

        agent = Agent(
            name="FileAgent",
            tools=[mcp(filesystem), my_local_tool]
        )

        # Multiple MCP servers
        search_config = MCPServerConfig(...)
        filesystem_config = MCPServerConfig(...)

        agent = Agent(
            name="MultiAgent",
            tools=[mcp(search_config, filesystem_config)]
        )
    """

    def __init__(
        self,
        configs: Sequence[MCPServerConfig],
        *,
        name_prefix: Optional[str] = None,
    ) -> None:
        """
        Initialize MCP tool provider.

        Args:
            configs: One or more MCP server configurations
            name_prefix: Optional prefix for tool names (default: server name)
        """
        self.configs = list(configs)
        self.name_prefix = name_prefix
        self.clients: Dict[str, MCPClient] = {}

    async def setup(self, agent: "BaseAgent") -> List[Tool]:
        """
        Connect to MCP servers and return wrapped tools.

        This is called by the agent before execution starts.

        Args:
            agent: The agent that will use these tools

        Returns:
            List of FunctionTool objects wrapping MCP tools
        """
        tools: List[Tool] = []

        for config in self.configs:
            try:
                # Create and connect client
                client = MCPClient.from_config(config)
                await client.connect()
                self.clients[config.name] = client

                # List available tools
                mcp_tools = await client.list_tools()

                # Wrap each MCP tool as a FunctionTool
                for mcp_tool in mcp_tools:
                    wrapped_tool = self._wrap_tool(config.name, mcp_tool)
                    tools.append(wrapped_tool)

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"MCP server '{config.name}' provided {len(mcp_tools)} tools"
                    )

            except Exception as e:
                logger.error(
                    f"Failed to setup MCP server '{config.name}': {e}. Skipping.",
                    exc_info=True,
                )
                # Continue with other servers - don't break agent initialization

        return tools

    async def teardown(self) -> None:
        """
        Disconnect from all MCP servers.

        This is called by the agent after execution completes.
        """
        for server_name, client in list(self.clients.items()):
            try:
                await client.disconnect()
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Disconnected from MCP server: {server_name}")
            except Exception as e:
                logger.warning(
                    f"Error disconnecting from MCP server '{server_name}': {e}"
                )

        self.clients.clear()

    def _wrap_tool(self, server_name: str, mcp_tool: MCPTool) -> FunctionTool:
        """
        Wrap an MCP tool as a FunctionTool.

        Creates a FunctionTool that delegates to the MCP server when called.
        The tool name is prefixed with the server name (or custom prefix) to avoid
        conflicts when using multiple MCP servers.

        Args:
            server_name: Name of the MCP server providing this tool.
            mcp_tool: MCP tool metadata to wrap.

        Returns:
            FunctionTool that calls the MCP tool when executed.
        """
        # Build tool name with prefix
        prefix = self.name_prefix or server_name
        tool_name = f"{prefix}:{mcp_tool.name}"

        # Create async function that calls the MCP tool
        async def tool_func(**kwargs: Any) -> Any:
            """Dynamically created tool function that calls MCP server."""
            client = self.clients[server_name]

            if not client.connected:
                try:
                    await client.connect()
                except Exception as connect_error:
                    logger.error(
                        "Failed to reconnect to MCP server '%s': %s",
                        server_name,
                        connect_error,
                    )
                    raise RuntimeError(
                        f"MCP server '{server_name}' is not connected"
                    ) from connect_error

            try:
                result = await client.call_tool(mcp_tool.name, kwargs)
                return result
            except Exception as e:
                logger.error(
                    f"Error calling MCP tool '{mcp_tool.name}' on server '{server_name}': {e}"
                )
                raise

        # Set function metadata for better introspection
        tool_func.__name__ = tool_name
        tool_func.__doc__ = mcp_tool.description

        # Create FunctionTool wrapper
        return FunctionTool(
            func=tool_func,
            name=tool_name,
            description=mcp_tool.description,
            parameters=mcp_tool.parameters,
        )


def mcp(
    *configs: MCPServerConfig,
    name_prefix: Optional[str] = None,
) -> MCPToolProvider:
    """
    Helper function to create MCPToolProvider.

    Allows clean syntax in agent tool lists:
    ```python
    agent = Agent(
        name="MyAgent",
        tools=[
            mcp(filesystem_config, search_config),
            my_local_tool,
        ]
    )
    ```

    Args:
        *configs: One or more MCPServerConfig objects
        name_prefix: Optional prefix for tool names

    Returns:
        MCPToolProvider instance

    Raises:
        ValueError: If no configs provided
    """
    if not configs:
        raise ValueError("At least one MCPServerConfig is required")

    return MCPToolProvider(configs, name_prefix=name_prefix)
