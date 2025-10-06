#!/usr/bin/env python3
"""
Example demonstrating a weather agent with conversation input/output and post-thinking hooks.
Shows how to use structured input/output schemas with hooks for monitoring.
"""

import os
import sys
import asyncio
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from opper_agent_old import Agent, tool, hook, RunContext, Usage


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


class AgentMessage(ConversationMessage):
    content: str = Field(description="The agent's response to the user")


# --- Weather Tool ---
@tool
def get_weather(location: str) -> str:
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
    return weather_data.get(location, f"Weather data not available for {location}")


@tool
def get_current_time(location: str) -> str:
    """Get current time information for a location."""

    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# --- Hooks ---
@hook("on_agent_start")
async def on_agent_start(context: RunContext, agent: Agent):
    print(f"🤖 Weather Agent started")
    print(f"   Input: {context.goal}")


@hook("on_think_end")
async def on_think_end(context: RunContext, agent: Agent, thought: Any):
    """Post-thinking hook to analyze the agent's reasoning."""
    print(thought.user_message)
    print(thought.todo_list)
    print(thought.tool_name)


# --- Main Demo Function ---
async def main():
    if not os.getenv("OPPER_API_KEY"):
        print("❌ Set OPPER_API_KEY environment variable")
        sys.exit(1)

    print("🌤️  Weather Conversation Agent Demo")
    print("=" * 50)

    # Create weather agent
    agent = Agent(
        name="WeatherAgent",
        description="A conversational agent that can provide weather information and engage in natural conversation.",
        tools=[get_weather, get_current_time],
        # model="anthropic/claude-3.5-sonnet",
        input_schema=ConversationInput,
        output_schema=AgentMessage,
        verbose=False,
        hooks=[
            on_agent_start,
            on_think_end,
        ],
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
    print(f"\nFinal Result 1: {result1}")

    print("\n--- Test Case 2: General Conversation ---")
    conversation2 = ConversationInput(
        messages=[ConversationMessage(role="user", content="Hello! How are you today?")]
    )
    result2 = await agent.process(conversation2)
    print(f"\nFinal Result 2: {result2}")

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
    print(f"\nFinal Result 3: {result3}")

    print("\n✅ Weather Conversation Demo Complete!")


if __name__ == "__main__":
    asyncio.run(main())
