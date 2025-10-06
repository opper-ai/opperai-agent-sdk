# Lets build a deep research style agent with Opper Agent SDK and Composio via MCP

import asyncio

# Lets start with importing the Opper Agent SDK
from opper_agent_old import Agent, hook, RunContext, tool
from opper_agent_old.mcp import MCPServerConfig, MCPToolManager

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


# We need to give it some tools. Lets connect to Composio and access search tools
mcp = MCPToolManager()
mcp.add_server(
    MCPServerConfig(
        name="composio-search",
        url="https://apollo.composio.dev/v3/mcp/6f548f6b-1eeb-400d-9a75-59a6a7788b41/mcp?user_id=pg-test-afc92e70-3424-4c48-8444-deb49953260c",
        transport="http-sse",
        enabled=True,
    )
)


# Actually lets add a hookk so we can peak into the agents reasoning as it is working
@hook("on_think_end")
async def on_think_end(context: RunContext, agent: Agent, thought: any):
    print(thought.user_message)


# Lets add a custom tool so that the agent can save the report to a file
@tool(description="Save the comprehensive report to a file in markdown format")
def save_report(report: str):
    with open("report.md", "w", encoding="utf-8") as f:
        f.write(report)
    return f"Report saved to report.md"


async def main():
    # Lets connect to the mcp servers and get the tools
    await mcp.connect_all()
    tools = mcp.get_all_tools()
    tools.append(save_report)

    print(f"Got {len(tools)} tools\n")

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

    # Basic agent
    agent = Agent(
        name="ComprehensiveResearchAgent",
        description="A comprehensive research agent that can use the search tools to find information on the web and the content tools to extract content and build a comprehensive report on the topic",
        instructions=instructions,
        input_schema=ResearchRequest,
        output_schema=ResearchFindings,
        tools=tools,
        hooks=[on_think_end],
        model="groq/gpt-oss-120b",  # Fast and cheap! But you can use any model you want.
    )

    result = await agent.process(
        ResearchRequest(
            topic="What is Opper AI?",
            depth="comprehensive",
            focus_areas=["traction", "team", "product", "innovation"],
            sources_required=15,
        )
    )
    print(result)

    await mcp.disconnect_all()


if __name__ == "__main__":
    asyncio.run(main())
