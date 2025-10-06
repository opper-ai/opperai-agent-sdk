"""
Debug MCP connection to see what's happening.
"""

import asyncio
import logging
from opper_agent.mcp.config import MCPServerConfig
from opper_agent.mcp.client import MCPClient

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


async def test_mcp_connection():
    """Test MCP connection with full debug output."""
    print("=" * 60)
    print("Testing MCP Connection")
    print("=" * 60)

    # Configure filesystem server
    config = MCPServerConfig(
        name="filesystem",
        transport="stdio",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/Users"],
        timeout=10.0,  # Longer timeout for debugging
    )

    print(f"\nConnecting to: {config.command} {' '.join(config.args)}")
    print()

    client = MCPClient.from_config(config)

    try:
        # Connect
        print("Step 1: Connecting...")
        await client.connect()
        print("✓ Connected successfully")

        # List tools
        print("\nStep 2: Listing tools...")
        tools = await client.list_tools()
        print(f"✓ Found {len(tools)} tools:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")

        # Try calling a tool
        if tools:
            print(f"\nStep 3: Testing tool '{tools[0].name}'...")
            result = await client.call_tool(tools[0].name, {})
            print(f"✓ Tool result: {result}")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Disconnect
        print("\nStep 4: Disconnecting...")
        await client.disconnect()
        print("✓ Disconnected")


if __name__ == "__main__":
    asyncio.run(test_mcp_connection())
