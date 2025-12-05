#!/usr/bin/env python3
"""
Gmail MCP Integration - Download Excel from Specific Email

This example shows how to use MCP (Model Context Protocol) tools with Opper Agent
to find an email with subject "test_excel" and download its Excel attachment.

PREREQUISITES:
--------------
1. Set your OPPER_API_KEY environment variable
2. Get valid Composio credentials from https://app.composio.dev/
3. Set COMPOSIO_GMAIL_MCP_URL environment variable with your Composio MCP endpoint
4. Have an email with subject "test_excel" containing an Excel attachment in your inbox

The agent will:
- Search for emails with subject "test_excel"
- Find Excel attachments in that email
- Get attachment metadata (including S3 URL) using Gmail MCP tools
- Fetch the actual Excel file from the S3 URL using a custom fetch_url tool
- Save the Excel file locally
"""

import asyncio
import os
import sys
from typing import Optional
from pydantic import BaseModel, Field

from opper_agents import Agent, hook, mcp, MCPServerConfig, tool
from opper_agents.base.context import AgentContext
import aiohttp


class ExcelDownloadResult(BaseModel):
    """Results from the Excel download operation."""

    success: bool = Field(description="Whether the operation was successful")
    message: str = Field(description="Status message")
    email_subject: Optional[str] = Field(default=None, description="Email subject")
    email_sender: Optional[str] = Field(default=None, description="Email sender")
    excel_filename: Optional[str] = Field(
        default=None, description="Name of the Excel file"
    )
    file_url: Optional[str] = Field(
        default=None, description="S3 URL of the Excel file"
    )


@hook("loop_end")
async def on_loop_end(context: AgentContext, agent: Agent) -> None:
    """Print agent's reasoning after each iteration."""
    if context.execution_history:
        latest = context.execution_history[-1]
        if latest.thought:
            print(f"\n[Iteration {latest.iteration}] {latest.thought.reasoning}\n")


@tool
async def fetch_binary_file(url: str, filename: str) -> str:
    """
    Fetch binary content from a URL (S3, HTTP, HTTPS) and save it locally.

    Args:
        url: The URL to fetch content from
        filename: The local filename to save the content to

    Returns:
        Success message with the saved file path
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(
            url, timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            response.raise_for_status()
            content = await response.read()

            # Save the binary content
            with open(filename, "wb") as f:
                f.write(content)

            return f"Successfully downloaded and saved file to {filename} ({len(content)} bytes)"


async def main() -> None:
    """Run the Gmail agent to download Excel from email with subject 'test_excel'."""
    print("Gmail MCP Integration - Download Excel from 'test_excel' Email")
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
        name="GmailExcelDownloader",
        description="Gmail agent that downloads Excel attachments from the email with subject 'test_excel'",
        instructions="""You are a helpful assistant that can access Gmail and download Excel attachments.

Your task is to:
1. Search for the email with subject "test_excel"
2. Get the email details and find any Excel attachments (.xlsx, .xls)
3. Download the Excel attachment from that email
4. If the download returns an S3 URL or remote URL, you MUST use the fetch_binary_file tool to download and save it locally
5. Return the email information and file details

Use the tools available to you:
- Gmail MCP tools: Search for emails, retrieve details, get attachment metadata
- fetch_binary_file tool: Download the Excel file from S3 URLs or any HTTP/HTTPS URL and save it locally

IMPORTANT:
- Excel files are binary files, so use fetch_binary_file (not fetch_url)
- Make sure to save the file with the correct filename
- The final result must confirm the file was downloaded successfully
Be thorough and provide clear information about what you find.""",
        tools=[
            mcp(gmail_config),  # Composio Gmail MCP tools
            fetch_binary_file,  # HTTP fetch tool for downloading binary S3 content
        ],
        output_schema=ExcelDownloadResult,
        max_iterations=20,
        verbose=True,
    )

    # Goal: Download Excel from email with subject "test_excel"
    goal = """Find the email with subject "test_excel" and download the Excel file attachment from it.

    IMPORTANT:
    - If you receive an S3 URL or any remote URL for the attachment, you MUST use fetch_binary_file to download it
    - Save the file with its original filename
    - Do not just return the URL - actually download the file

    Return:
    - Email subject and sender
    - Excel filename
    - File URL (S3 URL)"""

    print('\nGoal: Find email with subject "test_excel" and download Excel attachment')
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

        if result.excel_filename:
            print("\nExcel File:")
            print(f"  Filename: {result.excel_filename}")

            if result.file_url:
                print(f"  Source URL: {result.file_url}")

            # Check if file exists locally
            if os.path.exists(f"./{result.excel_filename}"):
                file_size = os.path.getsize(f"./{result.excel_filename}")
                print(f"  ✓ Downloaded to: ./{result.excel_filename}")
                print(f"  Size: {file_size:,} bytes")
            else:
                print("  ⚠ File not found locally")
        else:
            print("\nNo Excel file found in the email with subject 'test_excel'")

        print("\n" + "=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("   - Make sure you have valid Composio credentials")
        print("   - Check that the MCP endpoint URL is correct")
        print("   - Verify your OPPER_API_KEY is set")
        print("   - Check network connectivity")
        print('   - Ensure you have an email with subject "test_excel" in your inbox')
        print("   - Check that the email has an Excel attachment")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
