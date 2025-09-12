#!/usr/bin/env python3
"""
MCP (Model Context Protocol) Example - Demonstrates using the Opper docs MCP server with Opper Agent SDK.

This example shows how to integrate the Opper documentation MCP server to provide
agents with access to comprehensive Opper documentation and examples.
"""

import os
import sys
import asyncio
from typing import List, Dict, Any

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from opper_agent import Agent, MCPServers, create_mcp_tools, mcp_tools
    from opper_agent.mcp_client import MCPServerConfig
except ImportError:
    print(
        "ERROR: MCP support not available. This might be due to missing dependencies."
    )
    print("   MCP functionality requires additional setup.")
    sys.exit(1)


def demo_mcp_tools_basic():
    """Demonstrate basic MCP tool integration."""
    print("\nMCP Tools - Basic Integration")
    print("=" * 50)

    try:
        # Create MCP tools from Opper docs server
        opper_docs_server = MCPServerConfig(
            name="opper",
            url="https://docs.opper.ai/mcp",
            transport="http-sse",
            enabled=True,
        )
        mcp_tools_func = create_mcp_tools([opper_docs_server])

        # Get the tools (this will connect to MCP servers)
        tools = mcp_tools_func()

        print(f"Successfully loaded {len(tools)} MCP tools:")
        for tool in tools:
            print(f"   • {tool.name}: {tool.description}")

        # Create agent with MCP tools
        agent = Agent(
            name="MCPDocsAgent",
            description="An agent that can access Opper documentation using MCP",
            tools=tools,
            model="anthropic/claude-3.5-sonnet",
            verbose=True,
        )

        # Test the agent
        test_goals = [
            "What is Opper and what does it do?",
            "How do I create a call in Opper?",
            "What are the best practices for building AI applications with Opper?",
        ]

        for goal in test_goals:
            print(f"\n--- Testing Goal ---")
            print(f"Goal: {goal}")

            try:
                result = agent.process(goal)
                print(f"Result: {result}")
            except Exception as e:
                print(f"Error: {str(e)}")

    except Exception as e:
        print(f"❌ Error setting up MCP tools: {e}")
        print(
            "   Make sure you have network connectivity to access the Opper docs MCP server."
        )


# Create Opper docs server config for decorator
opper_docs_server = MCPServerConfig(
    name="opper", url="https://docs.opper.ai/mcp", transport="http-sse", enabled=True
)


@mcp_tools(opper_docs_server)
class MCPAgent(Agent):
    """Example agent class with MCP tools using decorator."""

    def __init__(self):
        super().__init__(
            name="DecoratedMCPAgent",
            description="An agent with Opper docs MCP tools added via decorator",
            model="anthropic/claude-3.5-sonnet",
            verbose=True,
        )


def demo_mcp_decorator():
    """Demonstrate MCP integration using decorator pattern."""
    print("\nMCP Tools - Decorator Pattern")
    print("=" * 50)

    try:
        agent = MCPAgent()

        print(f"Created agent with MCP tools")
        print(f"   Available tools: {len(agent.get_tools())}")

        # Test the decorated agent
        goal = "What are the key concepts in Opper and how do they work together?"
        print(f"\n--- Testing Decorated Agent ---")
        print(f"Goal: {goal}")

        try:
            result = agent.process(goal)
            print(f"✅ Result: {result}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")

    except Exception as e:
        print(f"Error creating decorated agent: {e}")


def demo_custom_mcp_server():
    """Demonstrate using custom MCP server configuration."""
    print("\nMCP Tools - Custom Server Configuration")
    print("=" * 50)

    try:
        # Create custom MCP server configuration for Opper docs
        custom_server = MCPServerConfig(
            name="opper-custom",
            url="https://docs.opper.ai/mcp",
            transport="http-sse",
            enabled=True,
        )

        # Create tools from custom server
        mcp_tools_func = create_mcp_tools([custom_server])
        tools = mcp_tools_func()

        print(f"Loaded {len(tools)} tools from custom MCP server")

        # Create agent
        agent = Agent(
            name="CustomMCPAgent",
            description="Agent with custom Opper docs MCP server configuration",
            tools=tools,
            model="anthropic/claude-3.5-sonnet",
            verbose=True,
        )

        # Test with Opper documentation queries
        goal = "Explain how to use datasets in Opper and provide an example"
        print(f"\n--- Testing Custom MCP Server ---")
        print(f"Goal: {goal}")

        try:
            result = agent.process(goal)
            print(f"✅ Result: {result}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")

    except Exception as e:
        print(f"Error with custom MCP server: {e}")
        print(
            "   Make sure you have network connectivity to access the Opper docs MCP server."
        )


async def demo_mcp_async():
    """Demonstrate async MCP operations."""
    print("\nMCP Tools - Async Operations")
    print("=" * 50)

    try:
        from opper_agent.mcp_tools import MCPToolManager

        # Create MCP tool manager
        manager = MCPToolManager()

        # Add Opper docs server
        opper_server = MCPServerConfig(
            name="opper",
            url="https://docs.opper.ai/mcp",
            transport="http-sse",
            enabled=True,
        )
        manager.add_server(opper_server)

        # Connect to all servers
        async with manager:
            tools = manager.get_all_tools()
            print(f"Connected to MCP servers with {len(tools)} total tools")

            # List tools by server
            for server_name in manager.adapters.keys():
                server_tools = manager.get_server_tools(server_name)
                print(f"   • {server_name}: {len(server_tools)} tools")

        print("Successfully disconnected from all MCP servers")

    except Exception as e:
        print(f"Error with async MCP operations: {e}")


def check_mcp_prerequisites():
    """Check if MCP prerequisites are available."""
    print("\nChecking MCP Prerequisites")
    print("=" * 40)

    # For Opper docs MCP server, we mainly need network connectivity
    print("INFO: Opper docs MCP server uses HTTP-SSE transport")
    print("   No Node.js/npm installation required")

    # Check network connectivity for Opper docs MCP server
    try:
        import urllib.request

        urllib.request.urlopen("https://docs.opper.ai", timeout=5)
        print("Network connectivity to Opper docs available")
    except Exception as e:
        print(f"WARNING: Could not reach Opper docs: {e}")
        print("   Network connectivity required for Opper docs MCP server")
        return False

    return True


def main():
    """Main function to run MCP examples."""
    # Get API key
    api_key = os.getenv("OPPER_API_KEY")
    if not api_key:
        print("⚠️  No OPPER_API_KEY found in environment.")
        print("   MCP examples will show structure but may fail on actual LLM calls.")
        print("")

    print("Model Context Protocol (MCP) Integration Examples")
    print("=" * 60)
    print("MCP enables agents to connect to external tools and data sources")
    print("through a standardized protocol developed by Anthropic.")
    print("This example uses the Opper documentation MCP server.")
    print("")

    # Check prerequisites
    if not check_mcp_prerequisites():
        print("\nERROR: MCP prerequisites not met.")
        print("Please check your network connectivity.")
        return

    # Run examples
    demo_mcp_tools_basic()
    demo_mcp_decorator()
    demo_custom_mcp_server()

    # Run async example
    print("\nRunning async MCP example...")
    asyncio.run(demo_mcp_async())

    print("\n" + "=" * 70)
    print("MCP Integration Examples Complete!")
    print("\nKey Takeaways:")
    print("   • MCP enables standardized integration with external tools")
    print("   • Opper provides MCP server for documentation access")
    print("   • Both functional and decorator patterns supported")
    print("   • Async operations supported for advanced use cases")
    print("   • Tools automatically converted to Opper Agent format")
    print("\nOpper MCP Server Features:")
    print("   • Access to comprehensive Opper documentation")
    print("   • Real-time information about Opper features")
    print("   • Examples and best practices")
    print("   • API reference and guides")
    print("   • Integration examples and tutorials")


if __name__ == "__main__":
    main()
