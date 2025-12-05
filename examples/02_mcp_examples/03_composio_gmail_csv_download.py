#!/usr/bin/env python3
"""
Gmail MCP Integration - Download CSV from Specific Email

This example shows how to use MCP (Model Context Protocol) tools with Opper Agent
to find an email with subject "test_csv" and download its CSV attachment.

PREREQUISITES:
--------------
1. Set your OPPER_API_KEY environment variable
2. Get valid Composio credentials from https://app.composio.dev/
3. Set COMPOSIO_GMAIL_MCP_URL environment variable with your Composio MCP endpoint
4. Have an email with subject "test_csv" containing a CSV attachment in your inbox

The agent will:
- Search for emails with subject "test_csv"
- Find CSV attachments in that email
- Get attachment metadata (including S3 URL) using Gmail MCP tools
- Fetch the actual CSV content from the S3 URL using a custom fetch_url tool
- Return the complete CSV file content
"""

import asyncio
import os
import sys
from typing import Optional
from pydantic import BaseModel, Field

from opper_agents import Agent, hook, mcp, MCPServerConfig, tool
from opper_agents.base.context import AgentContext
import aiohttp


class CSVDownloadResult(BaseModel):
    """Results from the CSV download operation."""

    success: bool = Field(description="Whether the operation was successful")
    message: str = Field(description="Status message")
    email_subject: Optional[str] = Field(default=None, description="Email subject")
    email_sender: Optional[str] = Field(default=None, description="Email sender")
    csv_filename: Optional[str] = Field(
        default=None, description="Name of the CSV file"
    )
    csv_content: Optional[str] = Field(
        default=None, description="Content of the CSV file"
    )


@hook("loop_end")
async def on_loop_end(context: AgentContext, agent: Agent) -> None:
    """Print agent's reasoning after each iteration."""
    if context.execution_history:
        latest = context.execution_history[-1]
        if latest.thought:
            print(f"\n[Iteration {latest.iteration}] {latest.thought.reasoning}\n")


@tool
async def fetch_url(url: str) -> str:
    """
    Fetch content from a URL (S3, HTTP, HTTPS).

    Args:
        url: The URL to fetch content from

    Returns:
        The text content from the URL
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(
            url, timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            response.raise_for_status()
            return await response.text()


async def main() -> None:
    """Run the Gmail agent to download CSV from email with subject 'test_csv'."""
    print("Gmail MCP Integration - Download CSV from 'test_csv' Email")
    print("=" * 60)

    # Get Composio URL from environment variable
    composio_url = os.getenv("COMPOSIO_GMAIL_MCP_URL")
    composio_api_key = os.getenv("COMPOSIO_API_KEY")

    if not composio_url:
        print("\nERROR: COMPOSIO_GMAIL_MCP_URL environment variable is not set!")
        print("\nTo get your Composio MCP URL:")
        print("1. Go to https://app.composio.dev/")
        print("2. Sign up or log in to your account")
        print("3. Navigate to the MCP section")
        print("4. Get your Gmail MCP endpoint URL")
        print("\nThen set the environment variable:")
        print('  export COMPOSIO_GMAIL_MCP_URL="your-composio-url-here"')
        print("\nOptionally set COMPOSIO_API_KEY if needed:")
        print('  export COMPOSIO_API_KEY="your-api-key"')
        sys.exit(1)

    # Configure Composio Gmail MCP server
    gmail_config = MCPServerConfig(
        name="composio-gmail",
        transport="streamable-http",
        url=composio_url,
        headers=(
            {"Authorization": f"Bearer {composio_api_key}"} if composio_api_key else {}
        ),
    )

    print("Creating Gmail Agent with MCP tools...")

    # Create agent with MCP tools
    agent = Agent(
        name="GmailCSVDownloader",
        description="Gmail agent that downloads CSV attachments from the email with subject 'test_csv'",
        instructions="""You are a helpful assistant that can access Gmail and download CSV attachments.

Your task is to:
1. Search for the email with subject "test_csv"
2. Get the email details and find any CSV attachments
3. Download the CSV attachment from that email
4. If the download returns an S3 URL or remote URL, you MUST fetch the actual file content from that URL
5. Return the COMPLETE CSV file content (all rows) along with email information

Use the tools available to you:
- Gmail MCP tools: Search for emails, retrieve details, get attachment metadata
- fetch_url tool: Fetch the actual content from S3 URLs or any HTTP/HTTPS URL

IMPORTANT: The final result must contain the COMPLETE CSV file content, not just a URL reference.
Be thorough and provide clear information about what you find.""",
        tools=[
            mcp(gmail_config),  # Composio Gmail MCP tools
            fetch_url,  # HTTP fetch tool for downloading S3 content
        ],
        output_schema=CSVDownloadResult,
        max_iterations=20,
        verbose=True,
    )

    # Goal: Download CSV from email with subject "test_csv"
    goal = """Find the email with subject "test_csv" and download the CSV file attachment from it.

    IMPORTANT:
    - If you receive an S3 URL or any remote URL for the attachment, you MUST fetch the actual file content from that URL
    - Do not just return the URL - get the actual CSV content

    Return:
    - Email subject and sender
    - CSV filename
    - COMPLETE CSV file content (all rows, not truncated)"""

    print('\nGoal: Find email with subject "test_csv" and download CSV attachment')
    print("Running Gmail Agent...\n")

    try:
        result = await agent.process(goal)

        print("\n" + "=" * 60)
        print("RESULT:")
        print("=" * 60)
        print(f"\nSuccess: {result.success}")
        print(f"Message: {result.message}")

        if result.email_subject:
            print("\nEmail Details:")
            print(f"  Subject: {result.email_subject}")
            print(f"  Sender: {result.email_sender}")

        if result.csv_filename:
            print("\nCSV File:")
            print(f"  Filename: {result.csv_filename}")

            if result.csv_content:
                # Save the CSV file locally
                output_path = f"./{result.csv_filename}"
                with open(output_path, "w") as f:
                    f.write(result.csv_content)
                print(f"  ✓ Saved to: {output_path}")

                # Show first few lines of CSV
                lines = result.csv_content.split("\n")[:5]
                print("\nFirst few lines of CSV:")
                for line in lines:
                    print(f"  {line}")
        else:
            print("\nNo CSV file found in the email with subject 'test_csv'")

        print("\n" + "=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("   - Make sure you have valid Composio credentials")
        print("   - Check that the MCP endpoint URL is correct")
        print("   - Verify your OPPER_API_KEY is set")
        print("   - Check network connectivity")
        print('   - Ensure you have an email with subject "test_csv" in your inbox')
        print("   - Check that the email has a CSV attachment")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
