#!/usr/bin/env python3
"""
Example demonstrating a weather agent with conversation input/output and post-thinking hooks.
Shows how to use structured input/output schemas with hooks for monitoring.

This example has been updated to work with the new Opper Agent SDK.
"""

import os
import asyncio
import time
from typing import Any, List, Optional
from pydantic import BaseModel, Field
from opper_agents.utils.logging import RichLogger

from opper_agents import Agent, tool, hook
from opper_agents.base.context import AgentContext


# --- Input/Output Schemas ---
class ConversationMessage(BaseModel):
    role: str = Field(
        description="The role of the message sender (user, assistant, system)"
    )
    content: str = Field(description="The content of the message")


class ConversationInput(BaseModel):
    messages: List[ConversationMessage] = Field(
        description="List of conversation messages"
    )
    location: Optional[str] = Field(
        default=None, description="Location for weather queries"
    )


class AgentMessage(BaseModel):
    """Agent's response in conversation format."""

    role: str = Field(default="assistant", description="Role is always assistant")
    content: str = Field(description="The agent's response to the user")


# --- Weather Tool ---
@tool
def get_weather(location: str) -> dict:
    """Get current weather information for a location."""
    time.sleep(0.5)  # Simulate API call
    from datetime import datetime

    # Simulate weather data with today's date information
    today_date = datetime.now().strftime("%Y-%m-%d")
    weather_data = {
        "New York": {"date": today_date, "weather": "Sunny, 72°F, light winds"},
        "London": {"date": today_date, "weather": "Cloudy, 15°C, light rain"},
        "Tokyo": {"date": today_date, "weather": "Partly cloudy, 22°C, humid"},
        "Paris": {"date": today_date, "weather": "Overcast, 18°C, gentle breeze"},
        "Sydney": {"date": today_date, "weather": "Clear, 25°C, strong winds"},
    }
    result = weather_data.get(
        location,
        {"date": today_date, "weather": f"Weather data not available for {location}"},
    )
    return result


@tool
def get_current_time(location: str) -> str:
    """Get current time information for a location."""
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# --- Hooks ---
@hook("agent_start")
async def on_agent_start(context: AgentContext, agent: Agent) -> None:
    """Hook triggered when agent starts execution."""
    print("🤖 Weather Agent started")
    print(f"   Input: {context.goal}")


@hook("think_end")
async def on_think_end(context: AgentContext, agent: Agent, thought: Any) -> None:
    """Post-thinking hook to analyze the agent's reasoning."""
    print("💭 Agent thinking:")
    print(f"   Reasoning: {thought.reasoning[:100]}...")
    print(f"   Tool calls planned: {len(thought.tool_calls)}")
    print(f"   User message: {thought.user_message}")


@hook("agent_end")
async def on_agent_end(context: AgentContext, agent: Agent, result: Any) -> None:
    """Hook triggered when agent completes execution."""
    print("✅ Agent completed")
    print(f"   Iterations: {context.iteration}")
    print(f"   Result type: {type(result).__name__}")


# --- Main Demo Function ---
async def main() -> None:
    if not os.getenv("OPPER_API_KEY"):
        print("❌ Set OPPER_API_KEY environment variable")
        return

    print("🌤️  Weather Conversation Agent Demo")
    print("=" * 50)

    # Create weather agent
    agent = Agent(
        name="WeatherAgent",
        description="A conversational agent that can provide weather information and engage in natural conversation.",
        instructions="You are a helpful weather assistant. Respond naturally to user queries and provide weather information when asked.",
        tools=[get_weather, get_current_time],
        input_schema=ConversationInput,
        output_schema=AgentMessage,
        verbose=True,
        logger=RichLogger(),
        hooks=[on_agent_start, on_think_end, on_agent_end],
    )

    # --- Test Cases ---

    print("\n--- Test Case 1: Weather Query ---")
    conversation1 = ConversationInput(
        messages=[
            ConversationMessage(
                role="user", content="What's the weather like in New York?"
            )
        ],
        location="New York",
    )
    result1 = await agent.process(conversation1)
    print("\n📤 Final Result 1:")
    print(f"   Role: {result1.role}")
    print(f"   Content: {result1.content}")

    print("\n" + "=" * 50)
    print("\n--- Test Case 2: General Conversation ---")
    conversation2 = ConversationInput(
        messages=[ConversationMessage(role="user", content="Hello! How are you today?")]
    )
    result2 = await agent.process(conversation2)
    print("\n📤 Final Result 2:")
    print(f"   Role: {result2.role}")
    print(f"   Content: {result2.content}")

    print("\n" + "=" * 50)
    print("\n--- Test Case 3: Multi-turn Conversation ---")
    conversation3 = ConversationInput(
        messages=[
            ConversationMessage(
                role="user", content="I'm planning a trip to London next week."
            ),
            ConversationMessage(
                role="assistant",
                content="That sounds exciting! London is a great city to visit.",
            ),
            ConversationMessage(
                role="user", content="What should I expect for weather there?"
            ),
        ],
        location="London",
    )
    result3 = await agent.process(conversation3)
    print("\n📤 Final Result 3:")
    print(f"   Role: {result3.role}")
    print(f"   Content: {result3.content}")

    print("\n" + "=" * 50)
    print("\n✅ Weather Conversation Demo Complete!")


if __name__ == "__main__":
    asyncio.run(main())
