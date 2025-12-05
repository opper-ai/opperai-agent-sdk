#!/usr/bin/env python3
"""
Daily Digest Agent with Multi-MCP Integration

This example demonstrates a comprehensive daily digest agent that:
1. Fetches emails from the last 24 hours via Gmail MCP
2. Gets top 10 Hacker News posts from the last day via HN MCP
3. Creates a structured Notion page with:
   - Main News last 24h
   - Main Email last 24h
   - Actions for today and links
4. On Thursday/Friday, adds Stockholm restaurant recommendations for the weekend

PREREQUISITES:
--------------
1. Set your OPPER_API_KEY environment variable
2. Get valid Composio credentials from https://app.composio.dev/
3. Create a .env file with the following variables:
   - COMPOSIO_GMAIL_MCP_URL
   - COMPOSIO_HACKERNEWS_MCP_URL
   - COMPOSIO_NOTION_MCP_URL
   - COMPOSIO_SEARCH_MCP_URL

Example .env file:
------------------
COMPOSIO_GMAIL_MCP_URL="https://backend.composio.dev/api/v1/connectedAccounts/{id}/mcp/..."
COMPOSIO_HACKERNEWS_MCP_URL="https://backend.composio.dev/api/v1/connectedAccounts/{id}/mcp/..."
COMPOSIO_NOTION_MCP_URL="https://backend.composio.dev/api/v1/connectedAccounts/{id}/mcp/..."
COMPOSIO_SEARCH_MCP_URL="https://backend.composio.dev/api/v1/connectedAccounts/{id}/mcp/..."
"""

import asyncio
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from opper_agents import Agent, hook, mcp, MCPServerConfig
from opper_agents.base.context import AgentContext
from opper_agents.utils.logging import RichLogger

# Load environment variables from .env file
load_dotenv()


class EmailSummary(BaseModel):
    """Summary of an important email."""

    sender: str = Field(description="Email sender")
    subject: str = Field(description="Email subject")
    summary: str = Field(description="Brief summary of the email content")
    action_required: bool = Field(
        description="Whether this email requires action from the user"
    )
    priority: str = Field(description="Priority level: high, medium, or low")


class NewsItem(BaseModel):
    """A news item from Hacker News."""

    title: str = Field(description="News title")
    url: Optional[str] = Field(description="Link to the article")
    points: Optional[int] = Field(description="Number of points")
    summary: str = Field(description="Brief summary of what this is about")
    relevance: str = Field(description="Why this might be interesting")


class ActionItem(BaseModel):
    """An action item for today."""

    action: str = Field(description="The action to take")
    source: str = Field(description="Source: email, news, or other")
    link: Optional[str] = Field(description="Related link if available")
    priority: str = Field(description="Priority: high, medium, or low")


class WeekendRecommendation(BaseModel):
    """A restaurant recommendation for the weekend."""

    name: str = Field(description="Restaurant name")
    cuisine: str = Field(description="Type of cuisine")
    description: str = Field(description="Brief description")
    reason: str = Field(description="Why this is recommended")


class DailyDigest(BaseModel):
    """Complete daily digest output."""

    date: str = Field(description="Date of the digest")
    emails: List[EmailSummary] = Field(
        description="Summary of important emails from last 24h"
    )
    news: List[NewsItem] = Field(description="Top news items from Hacker News")
    actions: List[ActionItem] = Field(description="Action items for today")
    weekend_recommendations: Optional[List[WeekendRecommendation]] = Field(
        default=None, description="Weekend restaurant recommendations (Thu/Fri only)"
    )
    notion_page_created: bool = Field(
        description="Whether the Notion page was created successfully"
    )
    notion_page_url: Optional[str] = Field(
        default=None, description="URL to the created Notion page"
    )


@hook("loop_end")
async def on_loop_end(context: AgentContext, agent: Agent) -> None:
    """Print agent's reasoning after each iteration."""
    if context.execution_history:
        latest = context.execution_history[-1]
        if latest.thought:
            print(f"\n[Iteration {latest.iteration}] {latest.thought.reasoning}\n")


def check_environment_variables() -> Dict[str, str]:
    """Check and validate all required environment variables."""
    required_vars = {
        "COMPOSIO_GMAIL_MCP_URL": "Gmail",
        "COMPOSIO_HACKERNEWS_MCP_URL": "Hacker News",
        "COMPOSIO_NOTION_MCP_URL": "Notion",
        "COMPOSIO_SEARCH_MCP_URL": "Search (TripAdvisor)",
    }

    missing_vars = []
    env_vars = {}

    for var_name, service_name in required_vars.items():
        value = os.getenv(var_name)
        if not value:
            missing_vars.append(f"  - {var_name} (for {service_name})")
        else:
            env_vars[var_name] = value

    if missing_vars:
        print("\nERROR: Missing required environment variables!")
        print("\nMissing variables:")
        for var in missing_vars:
            print(var)
        print("\nTo get your Composio MCP URLs:")
        print("1. Go to https://app.composio.dev/")
        print("2. Sign up or log in to your account")
        print("3. Navigate to the MCP section")
        print("4. Get your MCP endpoint URLs for each service")
        print("\nCreate a .env file in the project root with:")
        print("COMPOSIO_GMAIL_MCP_URL=your-gmail-url")
        print("COMPOSIO_HACKERNEWS_MCP_URL=your-hackernews-url")
        print("COMPOSIO_NOTION_MCP_URL=your-notion-url")
        print("COMPOSIO_SEARCH_MCP_URL=your-search-url")
        sys.exit(1)

    return env_vars


def is_weekend_planning_day() -> bool:
    """Check if today is Thursday or Friday."""
    today = datetime.now().weekday()
    return today in [3, 4]  # Thursday=3, Friday=4


async def main() -> None:
    """Run the daily digest agent."""
    print("Daily Digest Agent")
    print("=" * 60)

    # Check environment variables
    env_vars = check_environment_variables()

    # Get current date and day info
    today = datetime.now()
    date_str = today.strftime("%Y-%m-%d")
    day_name = today.strftime("%A")
    is_weekend_prep = is_weekend_planning_day()

    print(f"\nDate: {date_str} ({day_name})")
    if is_weekend_prep:
        print(
            "Weekend planning mode: Will include Stockholm restaurant recommendations"
        )

    # Configure MCP servers
    gmail_config = MCPServerConfig(
        name="composio-gmail",
        transport="streamable-http",
        url=env_vars["COMPOSIO_GMAIL_MCP_URL"],
    )

    hackernews_config = MCPServerConfig(
        name="composio-hackernews",
        transport="streamable-http",
        url=env_vars["COMPOSIO_HACKERNEWS_MCP_URL"],
    )

    notion_config = MCPServerConfig(
        name="composio-notion",
        transport="streamable-http",
        url=env_vars["COMPOSIO_NOTION_MCP_URL"],
    )

    search_config = MCPServerConfig(
        name="composio-search",
        transport="streamable-http",
        url=env_vars["COMPOSIO_SEARCH_MCP_URL"],
    )

    # Build instructions dynamically based on day
    base_instructions = """
    You are a personal productivity assistant that creates comprehensive daily digests.

    Your workflow:
    1. Fetch emails from the last 24 hours using Gmail tools
       - Focus on important/unread emails
       - Identify emails that require action
       - Summarize key content

    2. Fetch top 10 Hacker News posts from the last day
       - Get the most upvoted posts
       - Summarize what makes them interesting
       - Note technical relevance

    3. Generate action items based on emails and news
       - Extract actionable tasks from emails
       - Create follow-up actions
       - Include relevant links

    4. Create a well-structured Notion page with sections in the automated notes:
       - Page title: "Daily Digest - {date}"
       - Section: "Main News last 24h" (Hacker News posts)
       - Section: "Main Email last 24h" (Email summaries)
       - Section: "Actions for today" (Action items with links)
    """

    weekend_instructions = """
    5. WEEKEND PLANNING (Thursday/Friday only):
       - Search TripAdvisor for Stockholm restaurant recommendations
       - Focus on highly-rated restaurants with good reviews
       - Include variety of cuisines
       - Add a "Weekend Recommendations" section to the Notion page
    """

    instructions = base_instructions
    if is_weekend_prep:
        instructions += weekend_instructions

    instructions += """
    Guidelines:
    - Be concise and actionable
    - Prioritize important information
    - Use clear formatting in Notion
    - Include links where relevant - ALWAYS use descriptive titles for links (e.g., "Read the full article on TechCrunch" instead of just "link")
    - Use bullet points and headings effectively
    - When adding URLs, make them clickable with meaningful anchor text that describes the destination
    """

    print("\nCreating Daily Digest Agent with multiple MCP tools...")

    # Create agent with all MCP tools
    tools = [
        mcp(gmail_config),
        mcp(hackernews_config),
        mcp(notion_config),
    ]

    # Add search tools if it's weekend planning day
    if is_weekend_prep:
        tools.append(mcp(search_config))

    agent = Agent(
        name="DailyDigestAgent",
        description="Personal productivity assistant that creates comprehensive daily digests from emails, news, and generates actionable insights",
        instructions=instructions,
        output_schema=DailyDigest,
        tools=tools,
        max_iterations=30,  # More iterations for complex multi-step workflow
        verbose=True,
        logger=RichLogger(),
    )

    # Build the goal dynamically
    goal = f"""
    Create my daily digest for {date_str} ({day_name}):

    1. Get emails from the last 24 hours (focus on important/unread)
    2. Get top 10 Hacker News posts from the last day
    3. Generate action items from emails and any relevant news
    4. Create a Notion page titled "Daily Digest - {date_str}" with:
       - Section: "Main News last 24h"
       - Section: "Main Email last 24h"
       - Section: "Actions for today"
    """

    if is_weekend_prep:
        goal += """
    5. Search for Stockholm restaurant recommendations for the weekend
       - Use TripAdvisor or similar sources
       - Focus on highly-rated restaurants
       - Include variety of cuisines
       - Add a "Weekend Recommendations" section to the Notion page
    """

    print(f"\nGoal: {goal}")
    print("\nRunning Daily Digest Agent...")
    print("=" * 60)

    try:
        result = await agent.process(goal)
        print("\n" + "=" * 60)
        print("DAILY DIGEST CREATED")
        print("=" * 60)
        print(f"\nDate: {result.date}")
        print(f"\nEmails processed: {len(result.emails)}")
        print(f"News items: {len(result.news)}")
        print(f"Action items: {len(result.actions)}")

        if result.weekend_recommendations:
            print(f"Weekend recommendations: {len(result.weekend_recommendations)}")

        if result.notion_page_created:
            print("\nNotion page created successfully!")
            if result.notion_page_url:
                print(f"URL: {result.notion_page_url}")
        else:
            print("\nWarning: Notion page creation may have failed")

        print("\n" + "=" * 60)
        print("FULL RESULT:")
        print("=" * 60)
        print(result.model_dump_json(indent=2))

        # Print token usage summary
        if agent.context:
            print("\n" + "=" * 60)
            print("TOKEN USAGE SUMMARY:")
            print("=" * 60)
            print(f"  LLM requests:  {agent.context.usage.requests}")
            print(f"  Input tokens:  {agent.context.usage.input_tokens:,}")
            print(f"  Output tokens: {agent.context.usage.output_tokens:,}")
            print(f"  Total tokens:  {agent.context.usage.total_tokens:,}")

    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("   - Verify all Composio credentials are valid")
        print("   - Check that MCP endpoint URLs are correct")
        print("   - Ensure OPPER_API_KEY is set")
        print("   - Verify network connectivity")
        print("   - Check that you have connected the required apps in Composio:")
        print("     * Gmail account")
        print("     * Notion workspace")
        print("     * Hacker News (if required)")
        print("     * Search/TripAdvisor access")
        raise


if __name__ == "__main__":
    asyncio.run(main())
