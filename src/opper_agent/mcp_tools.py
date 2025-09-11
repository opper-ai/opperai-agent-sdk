"""
MCP tool integration for Opper Agent SDK.

This module provides adapters to use MCP tools as Opper Agent tools,
enabling seamless integration of external MCP servers with the agent system.
"""

import asyncio
from typing import Any, Dict, List, Optional, Callable, Type
from pydantic import BaseModel, Field, create_model
import logging

from .mcp_client import MCPClient, MCPServerConfig, MCPToolSchema, MCPError
from .base_agent import FunctionTool

logger = logging.getLogger(__name__)


class MCPToolAdapter:
    """
    Adapter that converts MCP tools into Opper Agent tools.

    This class manages MCP client connections and provides a bridge
    between MCP tools and the Opper Agent tool system.
    """

    def __init__(
        self, server_config: MCPServerConfig, tool_prefix: Optional[str] = None
    ):
        """
        Initialize MCP tool adapter.

        Args:
            server_config: Configuration for the MCP server
            tool_prefix: Optional prefix for tool names (e.g., "mcp_fs_")
        """
        self.server_config = server_config
        self.tool_prefix = tool_prefix or f"mcp_{server_config.name}_"
        self.client: Optional[MCPClient] = None
        self._tools_cache: List[FunctionTool] = []
        self._connected = False

    async def connect(self) -> None:
        """Connect to the MCP server and load tools."""
        if self._connected:
            return

        try:
            self.client = self.server_config.create_client()
            await self.client.connect()
            self._connected = True

            # Load and convert tools
            await self._load_tools()

            logger.info(
                f"Connected to MCP server '{self.server_config.name}' with {len(self._tools_cache)} tools"
            )

        except Exception as e:
            logger.error(
                f"Failed to connect to MCP server '{self.server_config.name}': {e}"
            )
            raise

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        if self.client:
            await self.client.disconnect()
            self.client = None
            self._connected = False
            self._tools_cache.clear()

    async def _load_tools(self) -> None:
        """Load MCP tools and convert them to Opper Agent tools."""
        if not self.client:
            return

        mcp_tools = self.client.get_tools()
        self._tools_cache = []

        for mcp_tool in mcp_tools:
            try:
                agent_tool = self._convert_mcp_tool(mcp_tool)
                self._tools_cache.append(agent_tool)
            except Exception as e:
                logger.warning(f"Failed to convert MCP tool '{mcp_tool.name}': {e}")

    def _convert_mcp_tool(self, mcp_tool: MCPToolSchema) -> FunctionTool:
        """Convert an MCP tool schema to an Opper Agent tool."""
        # Create Pydantic model from MCP input schema
        input_model = self._create_pydantic_model_from_schema(
            mcp_tool.inputSchema, f"{mcp_tool.name.title()}Input"
        )

        # Create tool function (synchronous wrapper around async MCP call)
        def tool_function(**kwargs) -> Dict[str, Any]:
            try:
                # Validate input using Pydantic model
                validated_input = input_model(**kwargs)

                # Create a fresh MCP client for each call to avoid event loop issues
                async def _async_call():
                    client = self.server_config.create_client()
                    try:
                        await client.connect()
                        result = await client.call_tool(
                            mcp_tool.name, validated_input.model_dump()
                        )
                        return result
                    finally:
                        await client.disconnect()

                # Run the async call with a fresh event loop
                result = asyncio.run(_async_call())

                return result

            except Exception as e:
                logger.error(f"MCP tool '{mcp_tool.name}' execution failed: {e}")
                raise MCPError(f"Tool execution failed: {e}")

        # Create FunctionTool instance
        tool_name = f"{self.tool_prefix}{mcp_tool.name}"

        # Extract parameters from the Pydantic model for the Tool
        parameters = {}
        if hasattr(input_model, "__fields__"):
            for field_name, field_info in input_model.__fields__.items():
                parameters[field_name] = (
                    field_info.description or f"Parameter {field_name}"
                )

        return FunctionTool(
            func=tool_function,
            name=tool_name,
            description=mcp_tool.description,
            parameters=parameters,
        )

    def _create_pydantic_model_from_schema(
        self, schema: Dict[str, Any], model_name: str
    ) -> Type[BaseModel]:
        """Create a Pydantic model from a JSON schema."""
        if not schema or "properties" not in schema:
            # Return a simple model with no fields
            return create_model(model_name)

        properties = schema["properties"]
        required_fields = set(schema.get("required", []))

        fields = {}
        for field_name, field_schema in properties.items():
            field_type = self._json_schema_to_python_type(field_schema)
            field_description = field_schema.get("description", "")

            if field_name in required_fields:
                fields[field_name] = (field_type, Field(description=field_description))
            else:
                fields[field_name] = (
                    Optional[field_type],
                    Field(default=None, description=field_description),
                )

        return create_model(model_name, **fields)

    def _json_schema_to_python_type(self, schema: Dict[str, Any]) -> type:
        """Convert JSON schema type to Python type."""
        schema_type = schema.get("type", "string")

        if schema_type == "string":
            return str
        elif schema_type == "integer":
            return int
        elif schema_type == "number":
            return float
        elif schema_type == "boolean":
            return bool
        elif schema_type == "array":
            # For arrays, try to determine item type
            items_schema = schema.get("items", {})
            if items_schema:
                item_type = self._json_schema_to_python_type(items_schema)
                return List[item_type]
            return List[Any]
        elif schema_type == "object":
            return Dict[str, Any]
        else:
            return Any

    def get_tools(self) -> List[FunctionTool]:
        """Get list of converted Opper Agent tools."""
        return self._tools_cache.copy()

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()


class MCPToolManager:
    """
    Manager for multiple MCP tool adapters.

    This class helps manage multiple MCP servers and their tools,
    providing a unified interface for the Opper Agent system.
    """

    def __init__(self):
        """Initialize MCP tool manager."""
        self.adapters: Dict[str, MCPToolAdapter] = {}
        self._all_tools: List[FunctionTool] = []

    def add_server(
        self, server_config: MCPServerConfig, tool_prefix: Optional[str] = None
    ) -> MCPToolAdapter:
        """
        Add an MCP server to the manager.

        Args:
            server_config: Configuration for the MCP server
            tool_prefix: Optional prefix for tool names

        Returns:
            The created MCP tool adapter
        """
        if server_config.name in self.adapters:
            raise ValueError(f"MCP server '{server_config.name}' already added")

        adapter = MCPToolAdapter(server_config, tool_prefix)
        self.adapters[server_config.name] = adapter
        return adapter

    async def connect_all(self) -> None:
        """Connect to all MCP servers."""
        tasks = []
        for adapter in self.adapters.values():
            tasks.append(adapter.connect())

        # Connect to all servers concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log any connection failures
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                server_name = list(self.adapters.keys())[i]
                logger.error(
                    f"Failed to connect to MCP server '{server_name}': {result}"
                )

        # Update tools cache
        await self._update_tools_cache()

    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers."""
        tasks = []
        for adapter in self.adapters.values():
            tasks.append(adapter.disconnect())

        await asyncio.gather(*tasks, return_exceptions=True)
        self._all_tools.clear()

    async def _update_tools_cache(self) -> None:
        """Update the cache of all available tools."""
        self._all_tools = []
        for adapter in self.adapters.values():
            if adapter._connected:
                self._all_tools.extend(adapter.get_tools())

    def get_all_tools(self) -> List[FunctionTool]:
        """Get all tools from all connected MCP servers."""
        return self._all_tools.copy()

    def get_server_tools(self, server_name: str) -> List[FunctionTool]:
        """Get tools from a specific MCP server."""
        if server_name not in self.adapters:
            raise ValueError(f"MCP server '{server_name}' not found")

        return self.adapters[server_name].get_tools()

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect_all()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect_all()


def create_mcp_tools(
    server_configs: List[MCPServerConfig],
) -> Callable[[], List[FunctionTool]]:
    """
    Create a function that returns MCP tools for use with Opper Agent.

    This is a helper function that creates and manages MCP connections
    and returns tools in a format compatible with the Agent constructor.

    Args:
        server_configs: List of MCP server configurations

    Returns:
        Function that returns list of MCP tools

    Example:
        >>> from opper_agent import Agent
        >>> from opper_agent.mcp_tools import create_mcp_tools
        >>> from opper_agent.mcp_client import MCPServerConfig
        >>>
        >>> # Create MCP tools with Opper docs server
        >>> opper_docs_server = MCPServerConfig(
        ...     name="opper",
        ...     url="https://docs.opper.ai/mcp",
        ...     transport="http-sse",
        ...     enabled=True
        ... )
        >>> mcp_tools = create_mcp_tools([opper_docs_server])
        >>>
        >>> # Create agent with MCP tools
        >>> agent = Agent(
        ...     name="MCPAgent",
        ...     description="Agent with access to Opper documentation",
        ...     tools=mcp_tools(),
        ...     model="anthropic/claude-3.5-sonnet"
        ... )
    """
    manager = MCPToolManager()

    # Add all server configurations
    for config in server_configs:
        manager.add_server(config)

    def get_tools() -> List[FunctionTool]:
        """Get MCP tools (synchronous wrapper)."""
        # This is a bit tricky since we need async operations
        # We'll need to run the async operations in an event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an event loop, we can't use run_until_complete
                # This is a limitation - MCP tools need to be set up outside the event loop
                logger.warning(
                    "Cannot initialize MCP tools inside an event loop. Initialize MCP tools before creating the agent."
                )
                return []
            else:
                return loop.run_until_complete(_async_get_tools())
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(_async_get_tools())

    async def _async_get_tools() -> List[FunctionTool]:
        """Async version of get_tools."""
        # Connect to all servers and keep connections alive
        await manager.connect_all()
        return manager.get_all_tools()

    return get_tools


# Decorator for easy MCP tool integration
def mcp_tools(*server_configs: MCPServerConfig):
    """
    Decorator to add MCP tools to an agent.

    Args:
        *server_configs: MCP server configurations

    Example:
        >>> from opper_agent import Agent
        >>> from opper_agent.mcp_tools import mcp_tools
        >>> from opper_agent.mcp_client import MCPServerConfig
        >>>
        >>> # Create Opper docs MCP server config
        >>> opper_docs_server = MCPServerConfig(
        ...     name="opper",
        ...     url="https://docs.opper.ai/mcp",
        ...     transport="http-sse",
        ...     enabled=True
        ... )
        >>>
        >>> @mcp_tools(opper_docs_server)
        >>> class MyAgent(Agent):
        ...     def __init__(self):
        ...         super().__init__(
        ...             name="MCPAgent",
        ...             description="Agent with access to Opper documentation"
        ...         )
    """

    def decorator(agent_class):
        original_init = agent_class.__init__

        def new_init(self, *args, **kwargs):
            # Get MCP tools
            mcp_tool_list = create_mcp_tools(list(server_configs))()

            # Add to existing tools if any
            existing_tools = kwargs.get("tools", [])
            if existing_tools:
                kwargs["tools"] = existing_tools + mcp_tool_list
            else:
                kwargs["tools"] = mcp_tool_list

            original_init(self, *args, **kwargs)

        agent_class.__init__ = new_init
        return agent_class

    return decorator
