"""
Quick test to verify the agent works.

Run this with: uv run python examples/quick_test.py
"""

import asyncio
import os
from opper_agent import Agent, tool
from pydantic import BaseModel, Field


class MathProblem(BaseModel):
    problem: str = Field(description="The math problem")

class MathSolution(BaseModel):
    answer: float = Field(description="The answer")
    reasoning: str = Field(description="How we got it")


@tool
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    print(f"🔧 Tool called: add({a}, {b})")
    return a + b


@tool
def multiply(x: int, y: int) -> int:
    """Multiply two numbers together."""
    print(f"🔧 Tool called: multiply({x}, {y})")
    return x * y


async def main():
    """Run a quick test of the agent."""

    # Check for API key
    if not os.getenv("OPPER_API_KEY"):
        print("❌ Error: OPPER_API_KEY environment variable not set")
        print("\nPlease set it with:")
        print("  export OPPER_API_KEY='your-key-here'")
        return

    print("✅ OPPER_API_KEY found")
    print("\n" + "=" * 60)
    print("Testing Agent with Simple Math Task")
    print("=" * 60 + "\n")

    # Create agent
    agent = Agent(
        name="MathAgent",
        description="An agent that performs math operations",
        instructions="Solve the math problem using the available tools.",
        tools=[add, multiply],
        input_schema=MathProblem,
        output_schema=MathSolution,
        max_iterations=5,
        verbose=True,  # Show detailed execution
    )

    # Run a simple task
    # Option 1: Pass as dict (input_schema will validate)
    task_dict = {"problem": "What is (5 + 3) * 2?"}

    # Option 2: Pass as Pydantic model directly
    # task = MathProblem(problem="What is (5 + 3) * 2?")

    print(f"Task: {task_dict['problem']}\n")

    try:
        result = await agent.process(task_dict)
        print("\n" + "=" * 60)
        print("✅ Final Result (Structured Output)")
        print("=" * 60)
        print(f"Type: {type(result).__name__}")
        print(f"Answer: {result.answer}")
        print(f"Reasoning: {result.reasoning}")
        print("=" * 60)

        # Show execution stats
        if agent.context:
            print(f"\n📊 Execution Stats:")
            print(f"  - Iterations: {agent.context.iteration}")
            print(
                f"  - Tool calls: {sum(len(c.tool_calls) for c in agent.context.execution_history)}"
            )
            print(f"  - Token usage: {agent.context.usage}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
