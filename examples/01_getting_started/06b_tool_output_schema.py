"""
Example 06b: Tools with Output Schemas and Examples

This example demonstrates:
- Defining tools with output_schema to describe what they return
- Providing examples to help the LLM understand tool behavior
- Using run() to get both result and usage statistics

Run with: uv run python examples/01_getting_started/06b_tool_output_schema.py
"""

import asyncio
import os
from pydantic import BaseModel, Field
from opper_agents import Agent, tool


# =============================================================================
# Tool with simple output_schema
# =============================================================================


class AddResult(BaseModel):
    """Result of adding two numbers."""

    sum: float = Field(description="The sum of the two numbers")


@tool(
    output_schema=AddResult,
    examples=[
        {
            "input": {"a": 2, "b": 3},
            "output": {"sum": 5},
            "description": "Basic addition of positive numbers",
        },
        {
            "input": {"a": -5, "b": 5},
            "output": {"sum": 0},
            "description": "Adding opposite numbers results in zero",
        },
    ],
)
def add(a: float, b: float) -> float:
    """Add two numbers together.

    Args:
        a: First number
        b: Second number

    Returns:
        The sum of the two numbers
    """
    print(f"  Tool called: add({a}, {b})")
    return a + b


# =============================================================================
# Tool with structured output_schema
# =============================================================================


class DivisionResult(BaseModel):
    """Result of dividing two numbers."""

    quotient: float = Field(description="The result of the division")
    remainder: float = Field(description="The remainder after integer division")


@tool(
    output_schema=DivisionResult,
    examples=[
        {
            "input": {"numerator": 10, "denominator": 3},
            "output": {"quotient": 3.333, "remainder": 1},
            "description": "Division with remainder",
        },
        {
            "input": {"numerator": 15, "denominator": 5},
            "output": {"quotient": 3.0, "remainder": 0},
            "description": "Clean division with no remainder",
        },
    ],
)
def divide(numerator: float, denominator: float) -> dict:
    """Divide first number by second number, returns quotient and remainder.

    Args:
        numerator: Number to divide
        denominator: Number to divide by

    Returns:
        Dictionary with quotient and remainder
    """
    print(f"  Tool called: divide({numerator}, {denominator})")
    if denominator == 0:
        raise ValueError("Cannot divide by zero")
    quotient = numerator / denominator
    remainder = numerator % denominator
    return {"quotient": quotient, "remainder": remainder}


# =============================================================================
# Simple tool without output_schema (for comparison)
# =============================================================================


@tool
def multiply(x: float, y: float) -> float:
    """Multiply two numbers together.

    Args:
        x: First number
        y: Second number

    Returns:
        The product of the two numbers
    """
    print(f"  Tool called: multiply({x}, {y})")
    return x * y


async def main() -> None:
    if not os.getenv("OPPER_API_KEY"):
        print("Error: Set OPPER_API_KEY environment variable")
        return

    print("=" * 60)
    print("Tool Output Schemas & Examples")
    print("=" * 60)

    agent = Agent(
        name="MathAgent",
        description="An agent that performs mathematical operations",
        instructions=(
            "You are a math expert. Solve the problem using the available tools. "
            "The tools provide output schemas and examples to help you understand "
            "their behavior."
        ),
        tools=[add, multiply, divide],
        max_iterations=5,
        verbose=True,
    )

    task = "What is (5 + 3) * 2? Also, what is 17 divided by 5?"
    print(f"\nTask: {task}\n")

    try:
        run_result = await agent.run(task)

        print("\n" + "=" * 60)
        print(f"Result: {run_result.result}")
        print("=" * 60)

        # Show usage stats from RunResult
        usage = run_result.usage
        print("\nUsage Statistics:")
        print(f"  Requests:     {usage.requests}")
        print(f"  Input tokens: {usage.input_tokens}")
        print(f"  Output tokens:{usage.output_tokens}")
        print(f"  Total tokens: {usage.total_tokens}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
