# Lets build a deep research style agent with Opper Agent SDK and Composio via MCP

import asyncio
import os
import sys

# Lets start with importing the Opper Agent SDK
from opper_agents import Agent, hook, tool, mcp, MCPServerConfig
from opper_agents.base.context import AgentContext
from opper_agents.utils.logging import RichLogger

# deps
from pydantic import BaseModel, Field
from typing import List, Optional


# Lets add an input schema so we can structure the input to the agent
class ResearchRequest(BaseModel):
    topic: str = Field(description="The main topic or question to research")
    depth: str = Field(
        default="comprehensive",
        description="Research depth: 'quick', 'standard', or 'comprehensive'",
    )
    focus_areas: Optional[List[str]] = Field(
        default=None, description="Specific areas or subtopics to focus on"
    )
    sources_required: int = Field(
        default=10, description="Minimum number of sources to gather"
    )


# Lets also add an output schema so we can structure the output from the agent
class ResearchFindings(BaseModel):
    thoughts: str = Field(description="Research process thoughts and methodology")
    topic: str = Field(description="The researched topic")
    executive_summary: str = Field(description="High-level summary of findings")
    key_findings: List[str] = Field(description="Main findings from the research")
    detailed_analysis: str = Field(description="In-depth analysis of the topic")


# Get Composio URL from environment variable
composio_url = os.getenv("COMPOSIO_SEARCH_MCP_URL")

if not composio_url:
    print("\nERROR: COMPOSIO_SEARCH_MCP_URL environment variable is not set!")
    print("\nTo get your Composio MCP URL:")
    print("1. Go to https://app.composio.dev/")
    print("2. Sign up or log in to your account")
    print("3. Navigate to the MCP section")
    print("4. Get your Search MCP endpoint URL")
    print("   (should include 'include_composio_helper_actions=true')")
    print("\nThen set the environment variable:")
    print('  export COMPOSIO_SEARCH_MCP_URL="your-composio-url-here"')
    print("\nOr run this script with:")
    print('  COMPOSIO_SEARCH_MCP_URL="your-url" python deep_research_agent_composio.py')
    sys.exit(1)

# Configure Composio MCP server for search tools
composio_config = MCPServerConfig(
    name="composio-search",
    transport="streamable-http",
    url=composio_url,
)


# Actually lets add a hook so we can peek into the agents reasoning as it is working
@hook("loop_end")
async def on_loop_end(context: AgentContext, agent: Agent) -> None:
    """Print agent's reasoning after each iteration."""
    if context.execution_history:
        latest = context.execution_history[-1]
        if latest.thought:
            print(f"\n[Iteration {latest.iteration}] {latest.thought.reasoning}\n")


# Lets add a custom tool so that the agent can save the report to a file
@tool
def save_report(report: str) -> str:
    """
    Save the comprehensive report to a file in markdown format.

    Args:
        report: The markdown-formatted report content to save

    Returns:
        Confirmation message
    """
    with open("report.md", "w", encoding="utf-8") as f:
        f.write(report)
    return "Report saved to report.md"


async def main() -> None:
    # Lets add some more detailed instructions to the agent
    instructions = """
    You are a comprehensive research agent that can use the search tools to find information on the web and the content tools to extract content and build a comprehensive report on the topic.

    Guidelines:
    - Always use the search tools to find information on the web
    - Always use the content tools to extract content and build a comprehensive report on the topic
    - Always use multiple sources to get a comprehensive understanding of the topic
    - Don't stop until you have a very broad understanding of the topic
    - When done, always save the report to a file using the save_report tool
    """

    # Create agent with MCP tools and custom tool
    agent = Agent(
        name="ComprehensiveResearchAgent",
        description="A comprehensive research agent that can use the search tools to find information on the web and the content tools to extract content and build a comprehensive report on the topic",
        instructions=instructions,
        input_schema=ResearchRequest,
        output_schema=ResearchFindings,
        tools=[
            mcp(composio_config),  # MCP tools from Composio
            save_report,  # Custom local tool
        ],
        model="groq/gpt-oss-120b",  # Fast and cheap! But you can use any model you want.
        max_iterations=10,  # Give it plenty of iterations for deep research
        verbose=True,
        logger=RichLogger(),
    )

    result = await agent.process(
        ResearchRequest(
            topic="What is Opper AI?",
            depth="comprehensive",
            focus_areas=["traction", "team", "product", "innovation"],
            sources_required=3,
        )
    )

    # Pretty print the structured result
    print("\n" + "=" * 60)
    print("RESEARCH FINDINGS")
    print("=" * 60)
    print(f"\nTopic: {result.topic}\n")
    print("Executive Summary:")
    print(f"   {result.executive_summary}\n")
    print("Key Findings:")
    for i, finding in enumerate(result.key_findings, 1):
        print(f"   {i}. {finding}")
    print("\nDetailed Analysis:")
    print(f"   {result.detailed_analysis}\n")


if __name__ == "__main__":
    asyncio.run(main())
