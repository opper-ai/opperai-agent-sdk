"""
Example: Using an Agent as a Tool

This example demonstrates using as_tool() to compose agents — a coordinator
agent delegates work to specialized sub-agents.

Run with: uv run python examples/01_getting_started/02_agent_as_tool.py
"""

import asyncio
import os

from opper_agents import Agent


# Create specialized agents
math_agent = Agent(
    name="MathAgent",
    description="Performs mathematical calculations",
    instructions=(
        "You are a math expert. When given a calculation request, "
        "perform it and return just the numeric result. "
        "Be precise and show your work briefly."
    ),
    opper_api_key=os.getenv("OPPER_API_KEY"),
)

research_agent = Agent(
    name="ResearchAgent",
    description="Researches and explains concepts",
    instructions=(
        "You are a research expert. When given a topic, provide "
        "a clear, concise explanation with relevant details."
    ),
    opper_api_key=os.getenv("OPPER_API_KEY"),
)

# Create a coordinator that delegates to sub-agents via as_tool()
coordinator = Agent(
    name="Coordinator",
    description="Coordinates specialist agents to solve problems",
    instructions=(
        "You help users solve problems by delegating to specialized agents. "
        "Use MathAgent_agent for calculations and ResearchAgent_agent for explanations."
    ),
    tools=[
        math_agent.as_tool(),
        research_agent.as_tool(),
    ],
    verbose=True,
    opper_api_key=os.getenv("OPPER_API_KEY"),
)


async def main() -> None:
    """Run the example."""

    if not os.getenv("OPPER_API_KEY"):
        print("Error: Set OPPER_API_KEY environment variable")
        return

    print("=" * 60)
    print("Agent as Tool Example")
    print("=" * 60)

    task = (
        "I need to calculate 37 * 25 * 1.15, then explain what compound interest means"
    )
    print(f"\nTask: {task}\n")

    run_result = await coordinator.run(task)

    print("\n" + "=" * 60)
    print(f"Result: {run_result.result}")
    print("=" * 60)

    # Show usage from RunResult (includes sub-agent token usage)
    usage = run_result.usage
    print(f"\nUsage: {usage.requests} requests, {usage.total_tokens} tokens")

    # Show per-agent token breakdown
    if usage.breakdown:
        print("\nPer-agent breakdown:")
        for name, stats in usage.breakdown.items():
            print(f"  {name}: {stats.total_tokens} tokens")


if __name__ == "__main__":
    asyncio.run(main())
