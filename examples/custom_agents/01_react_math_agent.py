#!/usr/bin/env python3
"""
Example demonstrating a ReactAgent for solving math problems.

This example shows how the ReactAgent works with the ReAct (Reasoning + Acting) pattern:
1. REASON: The agent analyzes the problem and decides on an action
2. ACT: The agent executes ONE tool at a time
3. OBSERVE: The agent reviews the result
4. Repeat until the problem is solved

Key differences from standard Agent:
- ReactAgent calls ONE tool per iteration (not multiple)
- Clear separation between reasoning, acting, and observing
- More explicit step-by-step problem solving
"""

import os
import sys
import asyncio
from pydantic import BaseModel, Field

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from opper_agents import ReactAgent, tool, hook
from typing import Any


# --- Input/Output Schemas ---
class MathProblem(BaseModel):
    problem: str = Field(description="The math problem to solve")
    show_work: bool = Field(
        default=True, description="Whether to show step-by-step work"
    )


class MathSolution(BaseModel):
    problem: str = Field(description="The original problem")
    answer: float = Field(description="The numerical answer")
    steps: list[str] = Field(default_factory=list, description="Step-by-step solution")


# --- Math Tools ---
@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


@tool
def subtract_numbers(a: float, b: float) -> float:
    """Subtract the second number from the first number."""
    return a - b


@tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b


@tool
def divide_numbers(a: float, b: float) -> float:
    """Divide the first number by the second number."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


@tool
def calculate_power(base: float, exponent: float) -> float:
    """Calculate base raised to the power of exponent."""
    return float(base**exponent)


@tool
def calculate_square_root(number: float) -> float:
    """Calculate the square root of a number."""
    if number < 0:
        raise ValueError("Cannot calculate square root of negative number")
    return float(number**0.5)


# --- Event Hooks ---
@hook("agent_start")
async def on_agent_start(context: Any, agent: Any) -> None:
    """Called when the agent starts processing."""
    print("\nReactAgent Math Solver Started")
    print(f"   Problem: {context.goal}")
    print("-" * 60)


@hook("think_end")
async def on_think_end(context: Any, agent: Any, thought: Any) -> None:
    """Called after each reasoning step."""
    print(f"\n💭 Reasoning: {thought.reasoning}")
    if thought.is_complete:
        print("    Task complete!")
    elif thought.action:
        print(f"   🔧 Action: {thought.action.tool_name}({thought.action.parameters})")


@hook("tool_result")
async def on_tool_result(context: Any, agent: Any, tool: Any, result: Any) -> None:
    """Called after each tool execution."""
    if result.success:
        print(f"   ✓ Result: {result.result}")
    else:
        print(f"   ✗ Error: {result.error}")


@hook("agent_end")
async def on_agent_end(context: Any, agent: Any, result: Any) -> None:
    """Called when the agent finishes."""
    print("-" * 60)
    print(f"\n Final Answer: {result}")
    print(f" Total iterations: {context.iteration}")
    print(f" Token usage: {context.usage.total_tokens}")


async def main() -> None:
    if not os.getenv("OPPER_API_KEY"):
        print(" Error: Set OPPER_API_KEY environment variable")
        print("   Example: export OPPER_API_KEY=your-key-here")
        sys.exit(1)

    # Create ReactAgent for math problem solving
    agent = ReactAgent(
        name="ReactMathAgent",
        description="A mathematical agent using the ReAct pattern to solve problems step by step.",
        instructions="""
You are a math problem solver using the ReAct (Reasoning + Acting) pattern.

For each step:
1. REASON about what needs to be done
2. Choose ONE tool to call (or mark complete if done)
3. Review the OBSERVATION from the tool
4. Repeat until the problem is solved

Solve problems systematically, one operation at a time.
Follow order of operations (PEMDAS): Parentheses, Exponents, Multiplication/Division, Addition/Subtraction.
        """,
        tools=[
            add_numbers,
            subtract_numbers,
            multiply_numbers,
            divide_numbers,
            calculate_power,
            calculate_square_root,
        ],
        input_schema=MathProblem,
        output_schema=MathSolution,
        max_iterations=15,
        verbose=False,  # We use hooks for custom output
        hooks=[on_agent_start, on_think_end, on_tool_result, on_agent_end],
    )

    # --- Example Problems ---

    print("\n" + "=" * 60)
    print("EXAMPLE 1: Multi-step arithmetic")
    print("=" * 60)
    problem1 = MathProblem(
        problem="Calculate (12 * 8) + (45 / 9) - 7",
        show_work=True,
    )
    _result1 = await agent.process(problem1)

    print("\n" + "=" * 60)
    print("EXAMPLE 2: Powers and roots")
    print("=" * 60)
    problem2 = MathProblem(
        problem="What is the square root of (2^8)?",
        show_work=True,
    )
    _result2 = await agent.process(problem2)

    print("\n" + "=" * 60)
    print("EXAMPLE 3: Complex expression")
    print("=" * 60)
    problem3 = MathProblem(
        problem="Calculate ((5 + 3) * 4) - (10 / 2)",
        show_work=True,
    )
    _result3 = await agent.process(problem3)


if __name__ == "__main__":
    asyncio.run(main())
