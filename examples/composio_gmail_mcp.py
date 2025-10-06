#!/usr/bin/env python3
"""
Gmail MCP Integration with Opper Agent SDK

This example shows how to use MCP (Model Context Protocol) tools with Opper Agent
to create a Gmail agent that can read and manage emails.
"""

import os
import sys
import asyncio
from typing import List, Dict, Any
from pydantic import BaseModel, Field

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from opper_agent_old import Agent, hook, RunContext, create_mcp_tools_async
from opper_agent_old.mcp import MCPServerConfig, MCPToolManager


class GmailResults(BaseModel):
    """Results from the Gmail MCP tools."""
    emails: List[Dict[str, Any]] = Field(description="List of emails with ID, Subject, and Sender")
    total_count: int = Field(description="Total number of emails")
    success: bool = Field(description="Whether the operation was successful")
    message: str = Field(description="Status message")

@hook("on_think_end")
async def on_think_end(context: RunContext, agent: Agent, thought: Any):
    """Post-thinking hook to analyze the agent's reasoning."""
    print(thought.user_message)
    print(thought.todo_list)
    print(thought.tool_name)

async def main():
    """Run the Gmail agent with MCP integration."""
    if not os.getenv("OPPER_API_KEY"):
        print("❌ Set OPPER_API_KEY environment variable")
        return

    print("🚀 Gmail MCP Integration Example")
    print("=" * 50)
    
    # Create MCP server configuration for Gmail
    gmail_server = MCPServerConfig(
        name="composio-gmail",
        url="https://mcp.composio.dev/partner/composio/gmail/mcp?customerId=4857bb13-e7a5-47f4-8c6f-ae220779eec4&agent=cursor",
        transport="http-sse",
        enabled=True
    )
    
    # Create MCP tool manager for proper connection management
    mcp_manager = MCPToolManager()
    mcp_manager.add_server(gmail_server)
    
    try:
        # Create MCP tools with proper connection management
        print("📡 Creating MCP tools...")
        await mcp_manager.connect_all()
        tools = mcp_manager.get_all_tools()
        print(f"✅ Created {len(tools)} MCP tools")
        
        # Create agent with MCP tools
        agent = Agent(
            name="GmailAgent",
            description="Gmail agent with MCP integration for reading and managing emails",
            tools=tools,
            hooks=[on_think_end],
            output_schema=GmailResults,
            #verbose=True
        )
        
        # Example: List recent unread emails
        goal = "List my last 5 unread emails and show their subjects and senders."
        print(f"\n🎯 Goal: {goal}")
        print("🚀 Running Gmail Agent...")
        
        try:
            result = await agent.process(goal)
            print(f"✅ Result: {result}")
        except Exception as e:
            print(f"❌ Error: {e}")
            print("💡 This might be due to network issues or authentication problems")
    
    finally:
        # Properly disconnect from all MCP servers
        print("\n🔌 Cleaning up MCP connections...")
        await mcp_manager.disconnect_all()
        print("✅ MCP connections cleaned up")


if __name__ == "__main__":
    asyncio.run(main())