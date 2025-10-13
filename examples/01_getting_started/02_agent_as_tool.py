"""
Example: Using an Agent as a Tool

This example demonstrates TWO ways to use one agent as a tool in another agent:
1. Using the built-in as_tool() method (recommended)
2. Manually wrapping with @tool decorator (more control)

We'll create a MathAgent and a ResearchAgent, then show both approaches.
"""

import asyncio
import os

from opper_agents import Agent, tool


# Create specialized agents
math_agent = Agent(
    name="MathAgent",
    description="Performs mathematical calculations",
    instructions="""
    You are a math expert. When given a calculation request,
    perform it and return just the numeric result.
    Be precise and show your work briefly.
    """,
    verbose=True,
    opper_api_key=os.getenv("OPPER_API_KEY"),
)

research_agent = Agent(
    name="ResearchAgent",
    description="Researches and explains concepts",
    instructions="""
    You are a research expert. When given a topic, provide
    a clear, concise explanation with relevant details.
    """,
    verbose=True,
    opper_api_key=os.getenv("OPPER_API_KEY"),
)


# APPROACH 1: Using as_tool() - Simple and recommended
# This is the built-in way to convert an agent to a tool
coordinator_agent_v1 = Agent(
    name="CoordinatorV1",
    description="Coordinator using as_tool() method",
    instructions="""
    You help users solve problems by delegating to specialized agents.
    Use MathAgent_agent for calculations and ResearchAgent_agent for explanations.
    """,
    tools=[
        math_agent.as_tool(),
        research_agent.as_tool(),
    ],
    verbose=True,
    opper_api_key=os.getenv("OPPER_API_KEY"),
)


# APPROACH 2: Manual wrapping with @tool - More control over interface
# This gives you full control over the tool's signature and behavior
# Note: We create fresh agent instances inside the tool to avoid event loop issues
@tool
async def calculate(expression: str) -> str:
    """
    Perform a mathematical calculation.

    Args:
        expression: A math expression or question (e.g., "What is 15 * 23?")

    Returns:
        The calculated result
    """
    # Create a fresh agent instance for this call
    agent = Agent(
        name="MathAgent",
        description="Performs mathematical calculations",
        instructions="""
        You are a math expert. When given a calculation request,
        perform it and return just the numeric result.
        Be precise and show your work briefly.
        """,
        verbose=False,
        opper_api_key=os.getenv("OPPER_API_KEY"),
    )
    result = await agent.process(expression)
    return str(result)


@tool
async def research_topic(topic: str, focus: str = "general") -> str:
    """
    Research and explain a topic.

    Args:
        topic: The topic to research
        focus: Specific aspect to focus on (e.g., "history", "technical")

    Returns:
        A detailed explanation
    """
    query = f"Explain {topic}"
    if focus != "general":
        query += f" with focus on {focus}"

    # Create a fresh agent instance for this call
    agent = Agent(
        name="ResearchAgent",
        description="Researches and explains concepts",
        instructions="""
        You are a research expert. When given a topic, provide
        a clear, concise explanation with relevant details.
        """,
        verbose=False,
        opper_api_key=os.getenv("OPPER_API_KEY"),
    )
    result = await agent.process(query)
    return str(result)


coordinator_agent_v2 = Agent(
    name="CoordinatorV2",
    description="Coordinator using manual @tool wrappers",
    instructions="""
    You help users solve problems by delegating to specialized tools.
    Use calculate() for math and research_topic() for explanations.
    """,
    tools=[calculate, research_topic],
    verbose=True,
    opper_api_key=os.getenv("OPPER_API_KEY"),
)


async def main() -> None:
    """Run the example."""

    # Example 1: Using as_tool() approach
    print("\n" + "=" * 70)
    print("APPROACH 1: Using as_tool() - Simple and Recommended")
    print("=" * 70)

    result = await coordinator_agent_v1.process(
        "I need to calculate 37 * 25 * 1.15, then explain what compound interest means"
    )

    print(f"\nFinal Result: {result}")

    # Example 2: Using manual @tool wrapper approach
    print("\n" + "=" * 70)
    print("APPROACH 2: Using @tool Wrapper - More Control")
    print("=" * 70)

    result = await coordinator_agent_v2.process(
        "Calculate the area of a rectangle with length 12.5 and width 8.3, "
        "then research the Pythagorean theorem with focus on history"
    )

    print(f"\nFinal Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
