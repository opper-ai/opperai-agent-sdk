#!/usr/bin/env python3
"""
Example demonstrating Pydantic models in tool parameters.

This example shows how to use structured Pydantic models as tool parameters.
The SDK automatically extracts the full schema and exposes it to the LLM,
allowing for rich structured inputs.
"""

import asyncio
import os
from typing import List
from pydantic import BaseModel, Field
from opper_agents import Agent, tool


class ReportInput(BaseModel):
    """Structured input for report generation."""

    title: str = Field(description="The title of the report")
    summary: str = Field(description="The summary of the report")
    key_findings: List[str] = Field(description="The key findings of the report")
    detailed_analysis: str = Field(description="The detailed analysis of the report")


@tool(description="Save the comprehensive report to a file in markdown format")
def save_report(report: ReportInput) -> str:
    """
    Save a structured report to a markdown file.

    Args:
        report: The report data with title, summary, findings, and analysis

    Returns:
        Success message with file path
    """
    filename = f"{report.title.replace(' ', '_').lower()}.md"

    # Create markdown content
    content = f"# {report.title}\n\n"
    content += f"## Summary\n\n{report.summary}\n\n"
    content += "## Key Findings\n\n"
    for i, finding in enumerate(report.key_findings, 1):
        content += f"{i}. {finding}\n"
    content += f"\n## Detailed Analysis\n\n{report.detailed_analysis}\n"

    # Save to file
    with open(filename, "w") as f:
        f.write(content)

    return f"Report successfully saved to {filename}"


@tool
def analyze_topic(topic: str) -> dict:
    """
    Analyze a topic and return structured data.

    This simulates research/analysis that would be used to create a report.
    """
    # Simulated analysis
    analyses = {
        "AI Safety": {
            "summary": "AI Safety focuses on ensuring artificial intelligence systems behave as intended and remain beneficial to humanity.",
            "findings": [
                "Alignment problem remains a critical challenge",
                "Interpretability research is gaining traction",
                "Regulatory frameworks are emerging globally",
            ],
            "analysis": "The field of AI Safety has seen significant growth in recent years. "
            "Major concerns include ensuring AI systems remain aligned with human values, "
            "developing robust testing methodologies, and creating effective governance structures. "
            "International cooperation is essential for addressing these challenges.",
        },
        "Climate Change": {
            "summary": "Climate change represents one of the most pressing challenges facing humanity.",
            "findings": [
                "Global temperatures continue to rise",
                "Renewable energy adoption is accelerating",
                "Extreme weather events are increasing in frequency",
            ],
            "analysis": "Scientific consensus indicates urgent action is needed to mitigate climate impacts. "
            "While technological solutions exist, implementation requires coordinated global effort. "
            "The transition to sustainable energy systems is both an environmental and economic imperative.",
        },
    }

    return analyses.get(
        topic,
        {
            "summary": f"Analysis of {topic} topic.",
            "findings": [
                "Research indicates growing interest in this area",
                "Multiple perspectives exist on this topic",
                "Further investigation is recommended",
            ],
            "analysis": f"The topic of {topic} requires comprehensive examination across multiple dimensions. "
            f"Current understanding suggests this is an evolving field with significant implications.",
        },
    )


async def main() -> None:
    """Demonstrate Pydantic model tool parameters."""

    if not os.getenv("OPPER_API_KEY"):
        print("Error: Set OPPER_API_KEY environment variable")
        print("Example: export OPPER_API_KEY=your-key-here")
        return

    print("\n" + "=" * 70)
    print("Pydantic Tool Parameters Example")
    print("=" * 70)

    # Create an agent with tools that accept structured inputs
    agent = Agent(
        name="ReportGenerator",
        description="An agent that researches topics and generates comprehensive reports",
        instructions="""
You are a research assistant that creates comprehensive reports.

When asked to create a report:
1. Use analyze_topic to gather information about the topic
2. Structure the information appropriately
3. Use save_report with the structured ReportInput to save the report

Be thorough and ensure all report fields are properly filled with relevant content.
        """,
        tools=[analyze_topic, save_report],
        max_iterations=10,
        verbose=True,
    )

    # Test 1: Generate a report on AI Safety
    print("\n" + "-" * 70)
    print("Test 1: Generate Report on AI Safety")
    print("-" * 70 + "\n")

    task = "Create a comprehensive report on AI Safety and save it to a file."
    result = await agent.process(task)

    print("\n" + "=" * 70)
    print(f"Result: {result}")
    print("=" * 70)

    # Test 2: Generate a report on Climate Change
    print("\n" + "-" * 70)
    print("Test 2: Generate Report on Climate Change")
    print("-" * 70 + "\n")

    task = "Research Climate Change and create a detailed report file."
    result = await agent.process(task)

    print("\n" + "=" * 70)
    print(f"Result: {result}")
    print("=" * 70)

    # Show that the agent understands the structured schema
    print("\n" + "-" * 70)
    print("Tool Schema Information")
    print("-" * 70)
    print("\nThe save_report tool exposes this schema to the LLM:")
    print(f"  Tool: {save_report.name}")
    print(f"  Parameters: {save_report.parameters}")
    print("\nThe LLM sees the full Pydantic schema with:")
    print("  - title (str)")
    print("  - summary (str)")
    print("  - key_findings (List[str])")
    print("  - detailed_analysis (str)")
    print("\nThis allows the LLM to call the tool with properly structured data!")


if __name__ == "__main__":
    asyncio.run(main())
