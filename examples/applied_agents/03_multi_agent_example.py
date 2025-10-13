#!/usr/bin/env python3
"""
Multi-Agent System Example using agent.as_tool()

This example demonstrates how to use agent.as_tool() to create a clean
multi-agent system where agents can delegate tasks to each other.
"""

import os
import sys
import asyncio
from typing import Any
from pydantic import BaseModel, Field

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from opper_agents import Agent, tool, hook
from opper_agents.base.context import AgentContext


# ============================================================================
# Data Models
# ============================================================================


class Request(BaseModel):
    """Request for task processing."""

    message: str = Field(description="The task to be performed")


class Response(BaseModel):
    """Response from task processing."""

    message: str = Field(description="The response or result of the task")


# ============================================================================
# Specialized Agents
# ============================================================================


# Math Agent
@tool
def calculate(expression: str) -> float:
    """Calculate a mathematical expression safely."""
    try:
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            raise ValueError("Expression contains invalid characters")
        result: float = eval(expression)
        return result
    except Exception as e:
        raise ValueError(f"Calculation error: {e}")


@tool
def solve_equation(equation: str) -> str:
    """Solve a simple algebraic equation."""
    if "x" in equation:
        return "x = 5 (simplified solution)"
    return "No variable found in equation"


math_agent = Agent(
    name="MathAgent",
    description="Handles mathematical calculations and problem solving",
    instructions="Always show your work step by step and explain your reasoning.",
    tools=[calculate, solve_equation],
    input_schema=Request,
    output_schema=Response,
)


# Swedish Agent
@tool
def translate_to_swedish(text: str) -> str:
    """Translate English text to Swedish."""
    translations = {
        "hello": "hej",
        "goodbye": "hej då",
        "thank you": "tack",
        "yes": "ja",
        "no": "nej",
        "please": "snälla",
    }

    text_lower = text.lower()
    for english, swedish in translations.items():
        if english in text_lower:
            text_lower = text_lower.replace(english, swedish)
    return text_lower.title()


@tool
def swedish_grammar_check(text: str) -> str:
    """Check Swedish grammar and provide corrections."""
    if "är" in text.lower():
        return f"Grammar check: '{text}' - looks good!"
    return f"Grammar check: '{text}' - consider adding 'är' for proper Swedish"


swedish_agent = Agent(
    name="SwedishAgent",
    description="Handles Swedish language tasks, translation, and grammar",
    instructions="Provide both the Swedish translation and a brief explanation of the grammar.",
    tools=[translate_to_swedish, swedish_grammar_check],
    input_schema=Request,
    output_schema=Response,
)


# Physics Agent
@tool
def calculate_force(mass: float, acceleration: float) -> float:
    """Calculate force using F = ma."""
    return mass * acceleration


@tool
def explain_physics_concept(concept: str) -> str:
    """Explain a physics concept in simple terms."""
    explanations = {
        "velocity": "Velocity is how fast something is moving in a specific direction. It's speed with direction!",
        "force": "Force is a push or pull that can make things move, stop, or change direction.",
        "gravity": "Gravity is the invisible force that pulls things toward the Earth.",
        "energy": "Energy is the ability to do work. It can be stored or moving.",
    }

    concept_lower = concept.lower()
    for key, explanation in explanations.items():
        if key in concept_lower:
            return explanation
    return f"Physics concept '{concept}': This is a fundamental concept in physics."


physics_agent = Agent(
    name="PhysicsAgent",
    description="Handles physics calculations and explanations",
    instructions="Explain the concept in simple terms and provide a real-world example.",
    tools=[calculate_force, explain_physics_concept],
    input_schema=Request,
    output_schema=Response,
)


# ============================================================================
# Routing Assistant Agent using agent.as_tool()
# ============================================================================


# Event Hooks
@hook("on_agent_start")
async def on_routing_start(context: AgentContext, agent: Agent) -> None:
    print(f"🎯 Routing Assistant started - Task: {context.goal}")


@hook("on_think_end")
async def on_routing_think(context: AgentContext, agent: Agent, thought: Any) -> None:
    print(f"🤔 {thought.user_message}")


# Create the routing assistant using agent.as_tool()
routing_assistant = Agent(
    name="RoutingAssistant",
    description="Routes tasks to specialized agents (Math, Swedish, Physics)",
    instructions="Given a user message or task, respond or use subagents to handle the task.",
    tools=[
        math_agent.as_tool(tool_name="delegate_to_math"),
        swedish_agent.as_tool(tool_name="delegate_to_swedish"),
        physics_agent.as_tool(tool_name="delegate_to_physics"),
    ],
    hooks=[on_routing_start, on_routing_think],
    input_schema=Request,
    output_schema=Response,
    verbose=False,
)


# ============================================================================
# Example Usage
# ============================================================================


async def main() -> None:
    """Run the multi-agent system example."""
    if not os.getenv("OPPER_API_KEY"):
        print("❌ Set OPPER_API_KEY environment variable")
        return

    print("🤖 Multi-Agent System with agent.as_tool()")
    print("=" * 50)

    # Example tasks
    test_tasks = [
        "Hello, how are you?",
        "Calculate 15 * 8 + 42",
        "Translate 'hello' to Swedish",
        "Explain what velocity means in physics",
        "Solve the equation 2x + 5 = 15",
        "Check the Swedish grammar in 'Jag är glad'",
        "Calculate the force if mass is 10kg and acceleration is 5m/s²",
    ]

    print(f"🧪 Running {len(test_tasks)} test tasks...\n")

    # Process each task
    for i, task in enumerate(test_tasks, 1):
        print(f"\n--- Task {i} ---")
        print(f"Task: {task}")

        try:
            # Create task request
            task_request = Request(message=task)

            # Process through routing assistant
            result = await routing_assistant.process(task_request)

            # Handle both dict and Pydantic model results
            if isinstance(result, dict):
                if "message" in result:
                    print(f"✅ Result: {result['message']}")
                else:
                    print(f"✅ Result: {result}")
            elif hasattr(result, "message"):
                print(f"✅ Result: {result.message}")
            else:
                print(f"✅ Result: {result}")

        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n🎉 Multi-agent system complete!")
    print("\n💡 Key Benefits:")
    print("   • Clean syntax: agent.as_tool()")
    print("   • Automatic parameter extraction")
    print("   • Proper async handling")
    print("   • Easy multi-agent composition")


if __name__ == "__main__":
    asyncio.run(main())
