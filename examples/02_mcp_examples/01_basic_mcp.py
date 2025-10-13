"""Example of using local MCP servers.
This example uses a container and UVX to run the MCP servers.

Prerequisites:
1. Filesystem MCP server (via Docker):
   docker run -i --rm -v "$PWD:/workspace" node:20 \
     npx -y @modelcontextprotocol/server-filesystem /workspace

2. SQLite MCP server (via UVX):
   uvx mcp-server-sqlite --db-path path/to/db
"""

import asyncio
import os
from opper_agents import Agent, mcp, MCPServerConfig


async def main() -> None:
    # Filesystem MCP server configuration (via Docker)
    # Mounts current directory to /workspace in container
    # Note: npm notices are suppressed via NPM_CONFIG_UPDATE_NOTIFIER=false
    filesystem = MCPServerConfig(
        name="filesystem",
        transport="stdio",
        command="docker",
        args=[
            "run",
            "-i",
            "--rm",
            "-v",
            f"{os.getcwd()}:/workspace",
            "-e",
            "NPM_CONFIG_UPDATE_NOTIFIER=false",
            "node:20",
            "npx",
            "-y",
            "@modelcontextprotocol/server-filesystem",
            "/workspace",
        ],
        env={"NPM_CONFIG_UPDATE_NOTIFIER": "false"},
    )

    # SQLite MCP server configuration (via UVX)
    sqlite = MCPServerConfig(
        name="sqlite",
        transport="stdio",
        command="uvx",
        args=[
            "mcp-server-sqlite",
            "--db-path",
            "./test.db",
        ],
    )

    # Create agent with both MCP servers as tool providers
    agent = Agent(
        name="LocalStdIO",
        description="Agent with filesystem and SQLite capabilities via MCP",
        instructions="Use the filesystem and SQLite tools to help complete tasks",
        tools=[mcp(filesystem), mcp(sqlite)],
        max_iterations=10,
        verbose=True,
    )

    # Example task covering both tools
    result = await agent.process(
        "Create a file /workspace/hello.txt with the content 'hi', "
        "then execute 'SELECT 1 as one' on the SQLite database."
    )

    print("\n" + "=" * 60)
    print("RESULT:")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    # Note: You may see npm notices from Docker containers after the agent completes.
    # These are harmless stderr messages from the subprocess cleanup and don't affect functionality.
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except SystemExit:
        pass

    # cleanup delete hello.txt and test.db
    os.remove("hello.txt")
    os.remove("test.db")
