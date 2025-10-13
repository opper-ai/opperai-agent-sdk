#!/usr/bin/env python3
"""
Example demonstrating a math agent with event hooks for basic arithmetic operations.
Shows how to use structured input/output schemas with hooks for monitoring.
"""

import os
import sys
import asyncio
from typing import Any
from pydantic import BaseModel, Field

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from opper_agents import Agent, tool, hook
from opper_agents.base.context import AgentContext


# --- Input/Output Schemas ---
class MathProblem(BaseModel):
    problem: str = Field(description="The math problem to solve")
    show_work: bool = Field(
        default=False, description="Whether to show step-by-step work"
    )


class MathSolution(BaseModel):
    problem: str = Field(description="The original problem")
    answer: float = Field(description="The numerical answer")


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
@hook("on_agent_start")
async def on_agent_start(context: AgentContext, agent: Agent) -> None:
    print("🧮 Math Agent started")
    print(f"   Problem: {context.goal}")


@hook("on_think_end")
async def on_think_end(context: AgentContext, agent: Agent, thought: Any) -> None:
    """Post-thinking hook to analyze the agent's reasoning."""
    print(f"{thought.user_message}")


async def main() -> None:
    if not os.getenv("OPPER_API_KEY"):
        print("❌ Set OPPER_API_KEY environment variable")
        sys.exit(1)

    # Create math agent
    agent = Agent(
        name="MathAgent",
        description="A mathematical agent that can perform basic arithmetic operations and solve math problems step by step.",
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
        model="groq/gpt-oss-120b",
        verbose=False,
        hooks=[
            on_agent_start,
            on_think_end,
        ],
    )

    # --- Test Case ---

    problem = MathProblem(
        problem="Calculate (12 * 8) + (45 / 9) - 7",
    )
    result = await agent.process(problem)
    print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
