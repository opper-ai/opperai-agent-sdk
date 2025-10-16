"""
Example: Visualizing Agent Flow with Mermaid Diagrams

This example demonstrates the visualize_flow() method which creates
Mermaid diagrams showing agent structure, tools, schemas, and relationships.
"""

import os
from pydantic import BaseModel, Field
from opper_agents import Agent, tool, hook, mcp, MCPServerConfig


# Define some example tools
@tool
def search_web(query: str, max_results: int = 10) -> dict:
    """
    Search the web for information.

    Args:
        query: Search query string
        max_results: Maximum number of results to return

    Returns:
        Dictionary with search results
    """
    return {"results": [f"Result for: {query}"], "count": max_results}


@tool
def calculate(expression: str) -> float:
    """
    Perform mathematical calculations.

    Args:
        expression: Mathematical expression to evaluate

    Returns:
        Result of the calculation
    """
    return eval(expression)


@tool
def save_to_file(content: str, filename: str) -> str:
    """
    Save content to a file.

    Args:
        content: Content to save
        filename: Name of the file

    Returns:
        Success message
    """
    return f"Saved to {filename}"


# Define schemas
class ResearchRequest(BaseModel):
    """Input schema for research requests."""

    topic: str = Field(description="The topic to research")
    depth: int = Field(default=1, description="Research depth (1-5)")
    output_format: str = Field(default="markdown", description="Output format")


class ResearchResult(BaseModel):
    """Output schema for research results."""

    summary: str = Field(description="Summary of findings")
    sources: list[str] = Field(description="List of sources used")
    confidence: float = Field(description="Confidence score (0-1)")


# Define hooks
@hook("agent_start")
def log_start(context, **kwargs):
    """Log when agent starts."""
    print(f"[Hook] Agent {context.agent_name} starting...")


@hook("agent_end")
def log_end(context, **kwargs):
    """Log when agent completes."""
    print(f"[Hook] Agent {context.agent_name} completed!")


def main():
    """Create various agents and visualize their flows."""

    # Example 1: Simple agent with tools
    print("=" * 70)
    print("Example 1: Simple Agent with Tools")
    print("=" * 70)

    simple_agent = Agent(
        name="SimpleAgent",
        description="A simple agent with basic tools",
        instructions="Use tools to help answer user questions",
        tools=[search_web, calculate],
        opper_api_key=os.getenv("OPPER_API_KEY"),
    )

    diagram = simple_agent.visualize_flow()
    print(diagram)
    print()

    # Example 2: Agent with schemas and hooks
    print("=" * 70)
    print("Example 2: Agent with Schemas and Hooks")
    print("=" * 70)

    research_agent = Agent(
        name="ResearchAgent",
        description="Conducts in-depth research on topics",
        instructions="Research thoroughly and cite sources",
        tools=[search_web, save_to_file],
        input_schema=ResearchRequest,
        output_schema=ResearchResult,
        hooks=[log_start, log_end],
        opper_api_key=os.getenv("OPPER_API_KEY"),
    )

    diagram = research_agent.visualize_flow(output_path="research_agent_flow.md")
    print(diagram)
    print()

    # Example 3: Multi-agent system with agent-as-tool
    print("=" * 70)
    print("Example 3: Multi-Agent System")
    print("=" * 70)

    # Create specialized sub-agents
    math_agent = Agent(
        name="MathAgent",
        description="Performs mathematical calculations",
        tools=[calculate],
        opper_api_key=os.getenv("OPPER_API_KEY"),
    )

    search_agent = Agent(
        name="SearchAgent",
        description="Searches for information online",
        tools=[search_web],
        opper_api_key=os.getenv("OPPER_API_KEY"),
    )

    # Create coordinator that uses other agents as tools
    coordinator = Agent(
        name="Coordinator",
        description="Orchestrates multiple specialized agents",
        instructions="Delegate to specialized agents as needed",
        tools=[
            math_agent.as_tool(),
            search_agent.as_tool(),
            save_to_file,
        ],
        hooks=[log_start, log_end],
        opper_api_key=os.getenv("OPPER_API_KEY"),
    )

    diagram = coordinator.visualize_flow()
    print(diagram)
    print()

    # Save to file example
    print("=" * 70)
    print("Example 4: Save to File")
    print("=" * 70)

    output_path = coordinator.visualize_flow(output_path="coordinator_flow.md")
    print(f"Diagram saved to: {output_path}")
    print()

    # Also visualize sub-agents
    math_agent.visualize_flow(output_path="math_agent_flow.md")
    search_agent.visualize_flow(output_path="search_agent_flow.md")

    print("Sub-agent diagrams also saved!")
    print()

    # Example 5: Agent with MCP Tools
    print("=" * 70)
    print("Example 5: Agent with MCP Tool Provider")
    print("=" * 70)

    # Create a mock MCP server config (won't actually connect, just for visualization)
    # In a real scenario, this would connect to an actual MCP server
    filesystem_mcp = MCPServerConfig(
        name="filesystem",
        transport="stdio",
        command="docker",
        args=[
            "run",
            "-i",
            "--rm",
            "-v",
            f"{os.getcwd()}:/workspace",
            "node:20",
            "npx",
            "-y",
            "@modelcontextprotocol/server-filesystem",
            "/workspace",
        ],
    )

    # Create agent with MCP provider
    mcp_agent = Agent(
        name="MCPAgent",
        description="Agent with filesystem tools via MCP",
        instructions="Use filesystem tools to read and write files",
        tools=[
            mcp(filesystem_mcp),  # MCP provider
            calculate,  # Regular tool
        ],
        opper_api_key=os.getenv("OPPER_API_KEY"),
    )

    # Show with provider node only (default)
    diagram = mcp_agent.visualize_flow(output_path="mcp_agent_flow.md")
    print(diagram)
    print()

    # Example 6: MCP Agent with actual tools shown (connects to MCP server)
    print("=" * 70)
    print("Example 6: MCP Agent with Actual Tools (connects to server)")
    print("=" * 70)
    print("Note: This will try to connect to the MCP server to discover tools.")
    print("If connection fails, it will fall back to showing the provider node.")
    print()

    diagram_with_tools = mcp_agent.visualize_flow(
        output_path="mcp_agent_with_tools_flow.md"
    )
    print(diagram_with_tools)
    print()

    print("=" * 70)
    print("All visualizations complete!")
    print("=" * 70)
    print("You can view these markdown files in:")
    print("- GitHub")
    print("- VS Code (with Markdown Preview or Mermaid extension)")
    print("- Any Mermaid-compatible viewer")


if __name__ == "__main__":
    main()
