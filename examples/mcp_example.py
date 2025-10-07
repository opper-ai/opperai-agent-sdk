"""
Example: Using MCP servers with agents.

This demonstrates how to integrate MCP servers seamlessly with agents
using the clean `mcp()` helper function.

PREREQUISITES:
--------------
You need Node.js/npm installed to run these MCP servers.

Install the MCP servers globally (one-time setup):
    npm install -g @modelcontextprotocol/server-filesystem
    npm install -g @modelcontextprotocol/server-everything

Or use npx with -y flag to auto-install (shown in examples below).

AVAILABLE MCP SERVERS:
----------------------
- @modelcontextprotocol/server-filesystem - File operations
- @modelcontextprotocol/server-everything - Kitchen sink (time, echo, add, etc.)
- @modelcontextprotocol/server-memory - Simple memory/storage
- @modelcontextprotocol/server-sqlite - SQLite database operations

See: https://github.com/modelcontextprotocol/servers
"""

import asyncio
from opper_agent import Agent, tool, mcp, MCPServerConfig


# Define local tools as usual
@tool
def calculate(expression: str) -> str:
    """
    Calculate a mathematical expression.

    Args:
        expression: Mathematical expression to evaluate

    Returns:
        Result of the calculation
    """
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


async def main():
    """Example: Agent with MCP servers and local tools."""

    # OPTION 1: Filesystem server (requires read/write access)
    # filesystem_server = MCPServerConfig(
    #     name="filesystem",
    #     transport="stdio",
    #     command="npx",
    #     args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    # )

    # OPTION 2: Everything server (safe, no file access needed)
    # Provides tools like: get_time, echo, add_numbers
    everything_server = MCPServerConfig(
        name="everything",
        transport="stdio",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-everything"],
    )

    # Create agent with MCP servers and local tools mixed together
    agent = Agent(
        name="TestAgent",
        description="Agent that can use MCP tools and local tools",
        instructions="Use available tools to help answer questions",
        tools=[
            mcp(everything_server),  # MCP server as a tool provider
            calculate,  # Local tool
        ],
        max_iterations=10,
        verbose=True,
    )

    # Use the agent - it will automatically:
    # 1. Connect to MCP servers before execution
    # 2. Expose MCP tools alongside local tools
    # 3. Disconnect from MCP servers after execution
    result = await agent.process("What time is it? Also calculate 42 * 137")

    print("\n" + "=" * 60)
    print("RESULT:")
    print("=" * 60)
    print(result)


async def filesystem_example():
    """Example: Using filesystem MCP server."""

    # Filesystem server - change path as needed
    filesystem = MCPServerConfig(
        name="filesystem",
        transport="stdio",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    )

    agent = Agent(
        name="FileAgent",
        description="Agent that can read and write files",
        instructions="Help users with file operations",
        tools=[mcp(filesystem)],
        verbose=True,
    )

    result = await agent.process("List files in the directory and count them")
    print(result)


async def custom_prefix_example():
    """Example: Custom tool name prefixes."""

    everything = MCPServerConfig(
        name="everything",
        transport="stdio",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-everything"],
    )

    # Use custom prefix for tool names
    agent = Agent(
        name="CustomPrefixAgent",
        tools=[
            mcp(everything, name_prefix="mcp"),  # Tools: "mcp:echo", "mcp:get_time"
            calculate,
        ],
        verbose=True,
    )

    result = await agent.process("Echo 'hello world' and tell me the time")
    print(result)


if __name__ == "__main__":
    # Run the basic example (uses @modelcontextprotocol/server-everything)
    asyncio.run(main())

    # Uncomment to try other examples:
    # asyncio.run(filesystem_example())
    # asyncio.run(custom_prefix_example())
