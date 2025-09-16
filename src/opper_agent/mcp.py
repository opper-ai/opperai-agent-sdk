"""
MCP (Model Context Protocol) integration for Opper Agent SDK.

This module provides a unified interface for using MCP tools with Opper Agent,
combining both the low-level MCP client functionality and high-level tool integration.
"""

import asyncio
import json
import subprocess
from typing import Any, Dict, List, Optional, Callable, Type
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel, Field, create_model
import logging

from .base_agent import FunctionTool

logger = logging.getLogger(__name__)


# ============================================================================
# MCP Protocol Types and Exceptions
# ============================================================================

class MCPTransportType(Enum):
    """Supported MCP transport types."""
    STDIO = "stdio"
    SSE = "sse"
    HTTP = "http"


@dataclass
class MCPToolSchema:
    """MCP tool schema definition."""
    name: str
    description: str
    inputSchema: Dict[str, Any]


@dataclass
class MCPResource:
    """MCP resource definition."""
    uri: str
    name: str
    description: Optional[str] = None
    mimeType: Optional[str] = None


@dataclass
class MCPPrompt:
    """MCP prompt definition."""
    name: str
    description: str
    arguments: Optional[List[Dict[str, Any]]] = None


class MCPError(Exception):
    """Base exception for MCP-related errors."""
    pass


class MCPConnectionError(MCPError):
    """Exception raised when MCP connection fails."""
    pass


class MCPToolError(MCPError):
    """Exception raised when MCP tool execution fails."""
    pass


# ============================================================================
# MCP Client Implementations
# ============================================================================

class MCPClient:
    """
    Model Context Protocol client for connecting to MCP servers.
    
    Supports stdio transport for connecting to MCP servers that communicate
    via standard input/output streams.
    """
    
    def __init__(
        self,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
    ):
        """
        Initialize MCP client.
        
        Args:
            command: Command to start the MCP server
            args: Arguments for the server command
            env: Environment variables for the server process
            timeout: Timeout for server operations in seconds
        """
        self.command = command
        self.args = args or []
        self.env = env
        self.timeout = timeout
        
        self._process: Optional[subprocess.Popen] = None
        self._request_id = 0
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._capabilities: Optional[Dict[str, Any]] = None
        self._tools: Dict[str, MCPToolSchema] = {}
        self._resources: Dict[str, MCPResource] = {}
        self._prompts: Dict[str, MCPPrompt] = {}
        self._connected = False
    
    async def connect(self) -> None:
        """Connect to the MCP server."""
        try:
            # Start the MCP server process
            self._process = subprocess.Popen(
                [self.command] + self.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=self.env,
            )
            
            # Start background task to read responses
            asyncio.create_task(self._read_responses())
            
            # Initialize the connection
            await self._initialize()
            self._connected = True
            
            logger.info(f"Connected to MCP server: {self.command}")
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            raise MCPConnectionError(f"Failed to connect to MCP server: {e}")
    
    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        if self._process:
            self._process.terminate()
            try:
                await asyncio.wait_for(
                    asyncio.create_task(self._wait_for_process()), timeout=5.0
                )
            except asyncio.TimeoutError:
                self._process.kill()
            finally:
                self._process = None
                self._connected = False
                logger.info("Disconnected from MCP server")
    
    async def _wait_for_process(self) -> None:
        """Wait for the process to terminate."""
        if self._process:
            while self._process.poll() is None:
                await asyncio.sleep(0.1)
    
    async def _initialize(self) -> None:
        """Initialize the MCP connection."""
        # Send initialize request
        response = await self._send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}, "resources": {}, "prompts": {}},
                "clientInfo": {"name": "opper-agent-sdk", "version": "0.1.0"},
            },
        )
        
        self._capabilities = response.get("capabilities", {})
        
        # Send initialized notification
        await self._send_notification("notifications/initialized")
        
        # Load available tools, resources, and prompts
        await self._load_capabilities()
    
    async def _load_capabilities(self) -> None:
        """Load available tools, resources, and prompts from the server."""
        # Load tools
        if self._capabilities and "tools" in self._capabilities:
            try:
                tools_response = await self._send_request("tools/list")
                tools = tools_response.get("tools", [])
                for tool in tools:
                    schema = MCPToolSchema(
                        name=tool["name"],
                        description=tool["description"],
                        inputSchema=tool.get("inputSchema", {}),
                    )
                    self._tools[tool["name"]] = schema
                logger.info(f"Loaded {len(self._tools)} MCP tools")
            except Exception as e:
                logger.warning(f"Failed to load MCP tools: {e}")
        
        # Load resources
        if self._capabilities and "resources" in self._capabilities:
            try:
                resources_response = await self._send_request("resources/list")
                resources = resources_response.get("resources", [])
                for resource in resources:
                    res = MCPResource(
                        uri=resource["uri"],
                        name=resource["name"],
                        description=resource.get("description"),
                        mimeType=resource.get("mimeType"),
                    )
                    self._resources[resource["uri"]] = res
                logger.info(f"Loaded {len(self._resources)} MCP resources")
            except Exception as e:
                logger.warning(f"Failed to load MCP resources: {e}")
        
        # Load prompts
        if self._capabilities and "prompts" in self._capabilities:
            try:
                prompts_response = await self._send_request("prompts/list")
                prompts = prompts_response.get("prompts", [])
                for prompt in prompts:
                    p = MCPPrompt(
                        name=prompt["name"],
                        description=prompt["description"],
                        arguments=prompt.get("arguments"),
                    )
                    self._prompts[prompt["name"]] = p
                logger.info(f"Loaded {len(self._prompts)} MCP prompts")
            except Exception as e:
                logger.warning(f"Failed to load MCP prompts: {e}")
    
    async def _send_request(
        self, method: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Send a JSON-RPC request to the MCP server."""
        if not self._process or not self._process.stdin:
            raise MCPConnectionError("Not connected to MCP server")
        
        self._request_id += 1
        request_id = str(self._request_id)
        
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {},
        }
        
        # Create future for response
        future = asyncio.Future()
        self._pending_requests[request_id] = future
        
        try:
            # Send request
            request_json = json.dumps(request) + "\n"
            self._process.stdin.write(request_json)
            self._process.stdin.flush()
            
            # Wait for response
            response = await asyncio.wait_for(future, timeout=self.timeout)
            return response
            
        except asyncio.TimeoutError:
            self._pending_requests.pop(request_id, None)
            raise MCPError(f"Request timeout for method: {method}")
        except Exception as e:
            self._pending_requests.pop(request_id, None)
            raise MCPError(f"Request failed for method {method}: {e}")
    
    async def _send_notification(
        self, method: str, params: Optional[Dict[str, Any]] = None
    ) -> None:
        """Send a JSON-RPC notification to the MCP server."""
        if not self._process or not self._process.stdin:
            raise MCPConnectionError("Not connected to MCP server")
        
        notification = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        
        notification_json = json.dumps(notification) + "\n"
        self._process.stdin.write(notification_json)
        self._process.stdin.flush()
    
    async def _read_responses(self) -> None:
        """Background task to read responses from the MCP server."""
        if not self._process or not self._process.stdout:
            return
        
        try:
            while self._process and self._process.poll() is None:
                line = self._process.stdout.readline()
                if not line:
                    break
                
                try:
                    response = json.loads(line.strip())
                    await self._handle_response(response)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse MCP response: {e}")
                except Exception as e:
                    logger.error(f"Error handling MCP response: {e}")
                    
        except Exception as e:
            logger.error(f"Error reading MCP responses: {e}")
    
    async def _handle_response(self, response: Dict[str, Any]) -> None:
        """Handle a response from the MCP server."""
        if "id" in response:
            # This is a response to a request
            request_id = str(response["id"])
            future = self._pending_requests.pop(request_id, None)
            if future and not future.done():
                if "error" in response:
                    error = response["error"]
                    future.set_exception(MCPError(f"MCP error: {error}"))
                else:
                    future.set_result(response.get("result", {}))
        else:
            # This is a notification - log it
            method = response.get("method", "unknown")
            logger.debug(f"Received MCP notification: {method}")
    
    def get_tools(self) -> List[MCPToolSchema]:
        """Get list of available tools."""
        return list(self._tools.values())
    
    def get_resources(self) -> List[MCPResource]:
        """Get list of available resources."""
        return list(self._resources.values())
    
    def get_prompts(self) -> List[MCPPrompt]:
        """Get list of available prompts."""
        return list(self._prompts.values())
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a tool on the MCP server.
        
        Args:
            name: Name of the tool to call
            arguments: Arguments to pass to the tool
            
        Returns:
            Tool execution result
            
        Raises:
            MCPToolError: If tool execution fails
        """
        if not self._connected:
            raise MCPConnectionError("Not connected to MCP server")
        
        if name not in self._tools:
            raise MCPToolError(f"Tool not found: {name}")
        
        try:
            response = await self._send_request(
                "tools/call", {"name": name, "arguments": arguments}
            )
            
            return response
            
        except Exception as e:
            logger.error(f"MCP tool call failed for {name}: {e}")
            raise MCPToolError(f"Tool call failed for {name}: {e}")
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to MCP server."""
        return self._connected
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()


class MCPHTTPClient:
    """
    HTTP-based Model Context Protocol client for connecting to MCP servers.
    
    Supports HTTP and HTTP-SSE transports for connecting to MCP servers that
    communicate via HTTP endpoints.
    """
    
    def __init__(
        self,
        url: str,
        transport: str = "http-sse",
        timeout: float = 30.0,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize HTTP MCP client.
        
        Args:
            url: Base URL of the MCP server
            transport: Transport type ("http" or "http-sse")
            timeout: Timeout for server operations in seconds
            headers: Additional HTTP headers
        """
        self.url = url.rstrip("/")
        self.transport = transport
        self.timeout = timeout
        self.headers = headers or {}
        
        self._request_id = 0
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._capabilities: Optional[Dict[str, Any]] = None
        self._tools: Dict[str, MCPToolSchema] = {}
        self._resources: Dict[str, MCPResource] = {}
        self._prompts: Dict[str, MCPPrompt] = {}
        self._connected = False
        self._session = None
    
    async def connect(self) -> None:
        """Connect to the HTTP MCP server."""
        try:
            import aiohttp
            
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout), headers=self.headers
            )
            
            # Initialize the connection
            await self._initialize()
            self._connected = True
            
            logger.info(f"Connected to HTTP MCP server: {self.url}")
            
        except ImportError:
            raise MCPConnectionError(
                "aiohttp is required for HTTP MCP clients. Install with: pip install aiohttp"
            )
        except Exception as e:
            logger.error(f"Failed to connect to HTTP MCP server: {e}")
            raise MCPConnectionError(f"Failed to connect to HTTP MCP server: {e}")
    
    async def disconnect(self) -> None:
        """Disconnect from the HTTP MCP server."""
        if self._session:
            await self._session.close()
            self._session = None
            self._connected = False
            logger.info("Disconnected from HTTP MCP server")
    
    async def _initialize(self) -> None:
        """Initialize the HTTP MCP connection."""
        # Send initialize request
        response = await self._send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}, "resources": {}, "prompts": {}},
                "clientInfo": {"name": "opper-agent-sdk", "version": "0.1.0"},
            },
        )
        
        self._capabilities = response.get("capabilities", {})
        
        # Load available tools, resources, and prompts
        await self._load_capabilities()
    
    async def _load_capabilities(self) -> None:
        """Load available tools, resources, and prompts from the server."""
        # Load tools
        if self._capabilities and "tools" in self._capabilities:
            try:
                tools_response = await self._send_request("tools/list")
                tools = tools_response.get("tools", [])
                for tool in tools:
                    schema = MCPToolSchema(
                        name=tool["name"],
                        description=tool["description"],
                        inputSchema=tool.get("inputSchema", {}),
                    )
                    self._tools[tool["name"]] = schema
                logger.info(f"Loaded {len(self._tools)} HTTP MCP tools")
            except Exception as e:
                logger.warning(f"Failed to load HTTP MCP tools: {e}")
        
        # Load resources
        if self._capabilities and "resources" in self._capabilities:
            try:
                resources_response = await self._send_request("resources/list")
                resources = resources_response.get("resources", [])
                for resource in resources:
                    res = MCPResource(
                        uri=resource["uri"],
                        name=resource["name"],
                        description=resource.get("description"),
                        mimeType=resource.get("mimeType"),
                    )
                    self._resources[resource["uri"]] = res
                logger.info(f"Loaded {len(self._resources)} HTTP MCP resources")
            except Exception as e:
                logger.warning(f"Failed to load HTTP MCP resources: {e}")
        
        # Load prompts
        if self._capabilities and "prompts" in self._capabilities:
            try:
                prompts_response = await self._send_request("prompts/list")
                prompts = prompts_response.get("prompts", [])
                for prompt in prompts:
                    p = MCPPrompt(
                        name=prompt["name"],
                        description=prompt["description"],
                        arguments=prompt.get("arguments"),
                    )
                    self._prompts[prompt["name"]] = p
                logger.info(f"Loaded {len(self._prompts)} HTTP MCP prompts")
            except Exception as e:
                logger.warning(f"Failed to load HTTP MCP prompts: {e}")
    
    async def _send_request(
        self, method: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Send a JSON-RPC request to the HTTP MCP server."""
        if not self._session:
            raise MCPConnectionError("Not connected to HTTP MCP server")
        
        self._request_id += 1
        request_id = str(self._request_id)
        
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {},
        }
        
        try:
            # Try different endpoint patterns for MCP over HTTP
            endpoints_to_try = [
                self.url,  # Try the base URL directly
                f"{self.url}/rpc",  # Standard RPC endpoint
                f"{self.url}/mcp",  # MCP-specific endpoint
            ]
            
            last_error = None
            for endpoint in endpoints_to_try:
                try:
                    # Send HTTP POST request with proper headers for SSE transport
                    headers = {
                        "Content-Type": "application/json",
                        "Accept": "application/json, text/event-stream",
                    }
                    async with self._session.post(
                        endpoint, json=request, headers=headers
                    ) as response:
                        if response.status == 200:
                            # Handle SSE response format
                            response_text = await response.text()
                            
                            # Parse SSE format: look for "data: " lines
                            response_data = None
                            for line in response_text.split("\n"):
                                if line.startswith("data: "):
                                    try:
                                        response_data = json.loads(
                                            line[6:]
                                        )  # Remove "data: " prefix
                                        break
                                    except json.JSONDecodeError:
                                        continue
                            
                            if not response_data:
                                # If no SSE data found, try parsing as regular JSON
                                try:
                                    response_data = json.loads(response_text)
                                except json.JSONDecodeError:
                                    last_error = (
                                        f"Could not parse response: {response_text}"
                                    )
                                    continue
                            
                            if "error" in response_data:
                                error = response_data["error"]
                                raise MCPError(f"MCP error: {error}")
                            
                            return response_data.get("result", {})
                        else:
                            last_error = (
                                f"HTTP error {response.status}: {await response.text()}"
                            )
                            continue
                
                except Exception as e:
                    last_error = str(e)
                    continue
            
            # If all endpoints failed, raise the last error
            raise MCPError(f"All endpoints failed. Last error: {last_error}")
            
        except asyncio.TimeoutError:
            raise MCPError(f"Request timeout for method: {method}")
        except MCPError:
            raise
        except Exception as e:
            raise MCPError(f"Request failed for method {method}: {e}")
    
    def get_tools(self) -> List[MCPToolSchema]:
        """Get list of available tools."""
        return list(self._tools.values())
    
    def get_resources(self) -> List[MCPResource]:
        """Get list of available resources."""
        return list(self._resources.values())
    
    def get_prompts(self) -> List[MCPPrompt]:
        """Get list of available prompts."""
        return list(self._prompts.values())
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a tool on the HTTP MCP server.
        
        Args:
            name: Name of the tool to call
            arguments: Arguments to pass to the tool
            
        Returns:
            Tool execution result
            
        Raises:
            MCPToolError: If tool execution fails
        """
        if not self._connected:
            raise MCPConnectionError("Not connected to HTTP MCP server")
        
        if name not in self._tools:
            raise MCPToolError(f"Tool not found: {name}")
        
        try:
            response = await self._send_request(
                "tools/call", {"name": name, "arguments": arguments}
            )
            
            return response
            
        except Exception as e:
            logger.error(f"HTTP MCP tool call failed for {name}: {e}")
            raise MCPToolError(f"Tool call failed for {name}: {e}")
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to HTTP MCP server."""
        return self._connected
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()


# ============================================================================
# MCP Server Configuration
# ============================================================================

class MCPServerConfig:
    """Configuration for MCP servers."""
    
    def __init__(
        self,
        name: str,
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
        auto_connect: bool = True,
        url: Optional[str] = None,
        transport: str = "stdio",
        enabled: bool = True,
    ):
        """
        Initialize MCP server configuration.
        
        Args:
            name: Friendly name for the server
            command: Command to start the server (for stdio transport)
            args: Command arguments (for stdio transport)
            env: Environment variables (for stdio transport)
            timeout: Request timeout in seconds
            auto_connect: Whether to auto-connect on first use
            url: URL for HTTP-based servers
            transport: Transport type ("stdio", "http-sse", "http")
            enabled: Whether the server is enabled
        """
        self.name = name
        self.command = command
        self.args = args or []
        self.env = env
        self.timeout = timeout
        self.auto_connect = auto_connect
        self.url = url
        self.transport = transport
        self.enabled = enabled
        
        # Validate configuration
        if transport == "stdio" and not command:
            raise ValueError("Command is required for stdio transport")
        if transport in ["http-sse", "http"] and not url:
            raise ValueError("URL is required for HTTP-based transports")
    
    def create_client(self) -> "MCPClient":
        """Create an MCP client from this configuration."""
        if self.transport == "stdio":
            return MCPClient(
                command=self.command, args=self.args, env=self.env, timeout=self.timeout
            )
        elif self.transport in ["http-sse", "http"]:
            return MCPHTTPClient(
                url=self.url, transport=self.transport, timeout=self.timeout
            )
        else:
            raise ValueError(f"Unsupported transport type: {self.transport}")


# ============================================================================
# MCP Tool Integration for Opper Agent
# ============================================================================

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
                        # MCP tools expect parameters to be wrapped in a 'params' object
                        result = await client.call_tool(
                            mcp_tool.name, {"params": validated_input.model_dump()}
                        )
                        return result
                    finally:
                        await client.disconnect()
                
                # Check if we're in an event loop
                try:
                    loop = asyncio.get_running_loop()
                    # We're in an event loop, need to use a different approach
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, _async_call())
                        result = future.result()
                except RuntimeError:
                    # No event loop running, safe to use asyncio.run
                    result = asyncio.run(_async_call())
                
                # Ensure proper cleanup of any remaining connections
                import gc
                gc.collect()
                
                return result
                
            except Exception as e:
                logger.error(f"MCP tool '{mcp_tool.name}' execution failed: {e}")
                raise MCPError(f"Tool execution failed: {e}")
        
        # Create FunctionTool instance
        tool_name = f"{self.tool_prefix}{mcp_tool.name}"
        
        # Extract parameters from the MCP tool's input schema for better documentation
        parameters = self._extract_parameters_from_schema(mcp_tool.inputSchema)
        
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
    
    def _extract_parameters_from_schema(self, schema: Dict[str, Any]) -> Dict[str, str]:
        """Extract parameters from MCP tool input schema for FunctionTool."""
        if not schema or "properties" not in schema:
            return {}
        
        parameters = {}
        properties = schema["properties"]
        required_fields = set(schema.get("required", []))
        
        for field_name, field_schema in properties.items():
            # Get field description
            description = field_schema.get("description", f"Parameter {field_name}")
            
            # Get field type information
            field_type = field_schema.get("type", "string")
            if field_type == "array":
                items = field_schema.get("items", {})
                item_type = items.get("type", "string")
                type_info = f"array of {item_type}"
            elif field_type == "object":
                # For objects, try to extract nested properties for better documentation
                nested_props = self._extract_nested_object_properties(field_schema)
                if nested_props:
                    type_info = f"object with properties: {nested_props}"
                else:
                    type_info = "object"
            else:
                type_info = field_type
            
            # Add required indicator
            required_indicator = " (required)" if field_name in required_fields else " (optional)"
            
            # Combine description with type info
            full_description = f"{description} - Type: {type_info}{required_indicator}"
            
            parameters[field_name] = full_description
        
        return parameters
    
    def _extract_nested_object_properties(self, object_schema: Dict[str, Any]) -> str:
        """Extract property names from a nested object schema for documentation."""
        if not object_schema or "properties" not in object_schema:
            return ""
        
        properties = object_schema["properties"]
        required_fields = set(object_schema.get("required", []))
        
        prop_descriptions = []
        for prop_name, prop_schema in properties.items():
            prop_type = prop_schema.get("type", "string")
            required_marker = " (required)" if prop_name in required_fields else " (optional)"
            prop_descriptions.append(f"{prop_name}: {prop_type}{required_marker}")
        
        return ", ".join(prop_descriptions)
    
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
    
    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers."""
        tasks = []
        for adapter in self.adapters.values():
            tasks.append(adapter.disconnect())
        
        # Disconnect from all servers concurrently
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Clear tools cache
        self._all_tools.clear()
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect_all()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect_all()


# ============================================================================
# High-level Helper Functions
# ============================================================================

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
        >>> from opper_agent.mcp import create_mcp_tools, MCPServerConfig
        >>>
        >>> # Create MCP tools with Gmail server
        >>> gmail_server = MCPServerConfig(
        ...     name="gmail",
        ...     url="https://mcp.composio.dev/partner/composio/gmail/mcp",
        ...     transport="http-sse",
        ...     enabled=True
        ... )
        >>> mcp_tools = create_mcp_tools([gmail_server])
        >>>
        >>> # Create agent with MCP tools
        >>> agent = Agent(
        ...     name="MCPAgent",
        ...     description="Agent with access to Gmail",
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
                # If we're already in an event loop, we need to use a different approach
                # We'll create a task and wait for it to complete
                import concurrent.futures
                import threading
                
                def run_in_thread():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(_async_get_tools())
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_in_thread)
                    return future.result(timeout=30)  # 30 second timeout
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


async def create_mcp_tools_async(
    server_configs: List[MCPServerConfig],
) -> List[FunctionTool]:
    """
    Create MCP tools asynchronously for use with Opper Agent.
    
    This is the async version that should be used when already in an event loop.
    
    Args:
        server_configs: List of MCP server configurations
        
    Returns:
        List of MCP tools
        
    Example:
        >>> import asyncio
        >>> from opper_agent.mcp import create_mcp_tools_async, MCPServerConfig
        >>>
        >>> async def main():
        ...     gmail_server = MCPServerConfig(
        ...         name="gmail",
        ...         url="https://mcp.composio.dev/partner/composio/gmail/mcp",
        ...         transport="http-sse",
        ...         enabled=True
        ...     )
        ...     tools = await create_mcp_tools_async([gmail_server])
        ...     return tools
        >>>
        >>> tools = asyncio.run(main())
    """
    manager = MCPToolManager()
    
    # Add all server configurations
    for config in server_configs:
        manager.add_server(config)
    
    # Connect to all servers and keep connections alive
    await manager.connect_all()
    return manager.get_all_tools()


# Decorator for easy MCP tool integration
def mcp_tools(*server_configs: MCPServerConfig):
    """
    Decorator to add MCP tools to an agent.
    
    Args:
        *server_configs: MCP server configurations
        
    Example:
        >>> from opper_agent import Agent
        >>> from opper_agent.mcp import mcp_tools, MCPServerConfig
        >>>
        >>> # Create Gmail MCP server config
        >>> gmail_server = MCPServerConfig(
        ...     name="gmail",
        ...     url="https://mcp.composio.dev/partner/composio/gmail/mcp",
        ...     transport="http-sse",
        ...     enabled=True
        ... )
        >>>
        >>> @mcp_tools(gmail_server)
        >>> class MyAgent(Agent):
        ...     def __init__(self):
        ...         super().__init__(
        ...             name="GmailAgent",
        ...             description="Agent with access to Gmail"
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


# ============================================================================
# Pre-configured MCP Servers
# ============================================================================

class MCPServers:
    """Pre-configured MCP server configurations for common tools."""
    
    @staticmethod
    def filesystem(path: str = "/tmp") -> MCPServerConfig:
        """Filesystem MCP server for file operations."""
        return MCPServerConfig(
            name="filesystem",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", path],
        )
    
    @staticmethod
    def git(repository_path: str = ".") -> MCPServerConfig:
        """Git MCP server for Git operations."""
        return MCPServerConfig(
            name="git",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-git", repository_path],
        )
    
    @staticmethod
    def sqlite(db_path: str) -> MCPServerConfig:
        """SQLite MCP server for database operations."""
        return MCPServerConfig(
            name="sqlite",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-sqlite", db_path],
        )
    
    @staticmethod
    def web_search(api_key: Optional[str] = None) -> MCPServerConfig:
        """Web search MCP server."""
        env = {"API_KEY": api_key} if api_key else None
        return MCPServerConfig(
            name="web_search",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-web-search"],
            env=env,
        )