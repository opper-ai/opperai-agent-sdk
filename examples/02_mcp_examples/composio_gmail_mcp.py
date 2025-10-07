#!/usr/bin/env python3
"""
Gmail MCP Integration with Opper Agent SDK

This example shows how to use MCP (Model Context Protocol) tools with Opper Agent
to create a Gmail agent that can read and manage emails.

PREREQUISITES:
--------------
1. Set your OPPER_API_KEY environment variable
2. Get valid Composio credentials from https://app.composio.dev/
3. Set COMPOSIO_GMAIL_MCP_URL environment variable with your Composio MCP endpoint

NOTE: You'll need to get your own Composio MCP endpoint URL and set it as an environment variable.
"""

import asyncio
import os
from typing import List, Dict, Any
from pydantic import BaseModel, Field

from opper_agent import Agent, hook, mcp, MCPServerConfig
from opper_agent.base.context import AgentContext


class GmailResults(BaseModel):
    """Results from the Gmail MCP tools."""

    emails: List[Dict[str, Any]] = Field(
        description="List of emails with ID, Subject, and Sender"
    )
    total_count: int = Field(description="Total number of emails")
    success: bool = Field(description="Whether the operation was successful")
    message: str = Field(description="Status message")


@hook("loop_end")
async def on_loop_end(context: AgentContext, agent: Agent):
    """Print agent's reasoning after each iteration."""
    if context.execution_history:
        latest = context.execution_history[-1]
        if latest.thought:
            print(f"\n[Iteration {latest.iteration}] {latest.thought.reasoning}\n")


async def main():
    """Run the Gmail agent with MCP integration."""
    print("Gmail MCP Integration Example")
    print("=" * 50)

    # Configure Composio Gmail MCP server
    # IMPORTANT: Replace this URL with your own Composio credentials
    gmail_config = MCPServerConfig(
        name="composio-gmail",
        transport="streamable-http",
        url="Your composio url",
    )

    print("Creating Gmail Agent with MCP tools...")

    # Create agent with MCP tools
    # MCP connection/disconnection is handled automatically
    agent = Agent(
        name="GmailAgent",
        description="Gmail agent with MCP integration for reading and managing emails",
        instructions="Use Gmail tools to help users manage their emails. Be concise and helpful.",
        tools=[
            mcp(gmail_config),  # Composio Gmail MCP tools
        ],
        output_schema=GmailResults,
        max_iterations=20,
        verbose=True,
    )

    # Example: List recent unread emails
    goal = "List my last 5 unread emails and show their subjects and senders."
    print(f"\nGoal: {goal}")
    print("Running Gmail Agent...\n")

    try:
        result = await agent.process(goal)
        print("\n" + "=" * 50)
        print("RESULT:")
        print("=" * 50)
        print(result)
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("   - Make sure you have valid Composio credentials")
        print("   - Check that the MCP endpoint URL is correct")
        print("   - Verify your OPPER_API_KEY is set")
        print("   - Check network connectivity")


if __name__ == "__main__":
    asyncio.run(main())
