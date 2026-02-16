"""
Quick test to verify the agent works.

Run this with: uv run python examples/01_getting_started/01_first_agent.py
"""

import asyncio
import os
from opper_agents import Agent, tool
from opper_agents.utils.logging import RichLogger


@tool
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    print(f"Tool called: add({a}, {b})")
    return a + b


@tool
def multiply(x: int, y: int) -> int:
    """Multiply two numbers together."""
    print(f"Tool called: multiply({x}, {y})")
    return x * y


async def main() -> None:
    """Run a quick test of the agent."""

    # Check for API key
    if not os.getenv("OPPER_API_KEY"):
        print("Error: OPPER_API_KEY environment variable not set")
        print("\nPlease set it with:")
        print("  export OPPER_API_KEY='your-key-here'")
        return

    print("OPPER_API_KEY found")
    print("\n" + "=" * 60)
    print("Testing Agent with Simple Math Task")
    print("=" * 60 + "\n")

    # Create agent
    agent = Agent(
        name="MathAgent",
        description="An agent that performs math operations",
        instructions="Solve the math problem using the available tools.",
        tools=[add, multiply],
        max_iterations=5,
        verbose=True,  # Show detailed execution
        logger=RichLogger(),  # Default logger is SimpleLogger (normal print statements)
    )

    # Run a simple task
    task = "What is (5 + 3) * 2?"
    print(f"Task: {task}\n")

    try:
        run_result = await agent.run(task)
        print("\n" + "=" * 60)
        print(f"Final Result: {run_result.result}")
        print("=" * 60)

        # Show usage from RunResult
        usage = run_result.usage
        print("\nUsage Statistics:")
        print(f"  - Requests:     {usage.requests}")
        print(f"  - Input tokens: {usage.input_tokens}")
        print(f"  - Output tokens:{usage.output_tokens}")
        print(f"  - Total tokens: {usage.total_tokens}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
