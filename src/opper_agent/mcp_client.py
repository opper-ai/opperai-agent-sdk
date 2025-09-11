"""
Model Context Protocol (MCP) client implementation for Opper Agent SDK.

This module provides MCP client functionality to connect to external tools
and data sources through the standardized MCP protocol.
"""

import asyncio
import json
import subprocess
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


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

    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """
        Read a resource from the MCP server.

        Args:
            uri: URI of the resource to read

        Returns:
            Resource content
        """
        if not self._connected:
            raise MCPConnectionError("Not connected to MCP server")

        try:
            response = await self._send_request("resources/read", {"uri": uri})

            return response

        except Exception as e:
            logger.error(f"MCP resource read failed for {uri}: {e}")
            raise MCPError(f"Resource read failed for {uri}: {e}")

    async def get_prompt(
        self, name: str, arguments: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get a prompt from the MCP server.

        Args:
            name: Name of the prompt
            arguments: Arguments for the prompt

        Returns:
            Prompt content
        """
        if not self._connected:
            raise MCPConnectionError("Not connected to MCP server")

        if name not in self._prompts:
            raise MCPError(f"Prompt not found: {name}")

        try:
            response = await self._send_request(
                "prompts/get", {"name": name, "arguments": arguments or {}}
            )

            return response

        except Exception as e:
            logger.error(f"MCP prompt get failed for {name}: {e}")
            raise MCPError(f"Prompt get failed for {name}: {e}")

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

    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """
        Read a resource from the HTTP MCP server.

        Args:
            uri: URI of the resource to read

        Returns:
            Resource content
        """
        if not self._connected:
            raise MCPConnectionError("Not connected to HTTP MCP server")

        try:
            response = await self._send_request("resources/read", {"uri": uri})

            return response

        except Exception as e:
            logger.error(f"HTTP MCP resource read failed for {uri}: {e}")
            raise MCPError(f"Resource read failed for {uri}: {e}")

    async def get_prompt(
        self, name: str, arguments: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get a prompt from the HTTP MCP server.

        Args:
            name: Name of the prompt
            arguments: Arguments for the prompt

        Returns:
            Prompt content
        """
        if not self._connected:
            raise MCPConnectionError("Not connected to HTTP MCP server")

        if name not in self._prompts:
            raise MCPError(f"Prompt not found: {name}")

        try:
            response = await self._send_request(
                "prompts/get", {"name": name, "arguments": arguments or {}}
            )

            return response

        except Exception as e:
            logger.error(f"HTTP MCP prompt get failed for {name}: {e}")
            raise MCPError(f"Prompt get failed for {name}: {e}")

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


# Pre-configured MCP servers for common use cases
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
