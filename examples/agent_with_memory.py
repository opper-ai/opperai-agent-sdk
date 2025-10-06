"""
Example demonstrating the Agent with memory across multiple tasks.

This example shows:
- Agent with memory enabled
- Multiple sequential tasks that build on each other
- Memory-aware decision making across tasks
- Agent accessing previous task results from memory
"""

import asyncio
from opper_agent import Agent, tool


@tool
def calculate(expression: str) -> float:
    """
    Calculate a mathematical expression.

    Args:
        expression: Math expression to evaluate (e.g., "2 + 2", "10 * 5")

    Returns:
        Result of the calculation
    """
    try:
        # NOTE: eval() only safe in controlled environment
        result = eval(expression, {"__builtins__": {}}, {})
        return float(result)
    except Exception as e:
        raise ValueError(f"Invalid expression: {e}")


@tool
def get_weather(city: str) -> str:
    """
    Get weather information for a city (simulated).

    Args:
        city: Name of the city

    Returns:
        Weather description
    """
    # Simulated weather data
    weather_data = {
        "san francisco": "Sunny, 72°F",
        "new york": "Cloudy, 65°F",
        "london": "Rainy, 58°F",
        "tokyo": "Clear, 68°F",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


@tool
def save_user_preference(key: str, value: str) -> str:
    """
    Save a user preference.

    Args:
        key: Preference key (e.g., "favorite_city", "preferred_unit")
        value: Preference value

    Returns:
        Confirmation message
    """
    return f"Saved preference: {key} = {value}"


async def main():
    """Run a multi-task agent example with memory."""

    # Create agent with memory enabled
    agent = Agent(
        name="AssistantAgent",
        description="A helpful assistant that remembers context across tasks",
        instructions="""You are a helpful assistant with memory capabilities.

MEMORY USAGE:
- When you calculate something important, save it to memory
- When a user mentions they like something, remember their preferences
- Before starting a new task, check if you have relevant information in memory
- Use memory to provide context-aware responses

IMPORTANT:
- Store calculation results with descriptive keys like "budget_total", "distance_km"
- Store user preferences with keys like "favorite_city", "preferred_temperature_unit"
- When asked about something you previously calculated or saved, retrieve it from memory first
""",
        tools=[calculate, get_weather, save_user_preference],
        enable_memory=True,
        max_iterations=15,
        verbose=True,
    )

    print("=" * 70)
    print("Multi-Task Memory Example")
    print("=" * 70)
    print()

    # Task 1: Do some calculations and remember them
    print("🎯 TASK 1: Planning a trip budget")
    print("-" * 70)
    result1 = await agent.process(
        """Calculate my trip budget:
        - Flight: $450
        - Hotel for 5 nights at $120/night
        - Daily food budget: $80/day for 5 days

        Calculate the total and remember it as my trip budget."""
    )
    print(f"\n✅ Result: {result1}\n")

    # Check memory state after task 1
    if agent.context and agent.context.memory:
        print("📝 Memory after Task 1:")
        catalog = await agent.context.memory.list_entries()
        for entry in catalog:
            print(f"  - {entry['key']}: {entry['description']}")
        print()

    # Task 2: Save a preference
    print("🎯 TASK 2: Save travel preferences")
    print("-" * 70)
    result2 = await agent.process(
        "I love San Francisco! Save it as my favorite city and check the weather there."
    )
    print(f"\n✅ Result: {result2}\n")

    # Check memory state after task 2
    if agent.context and agent.context.memory:
        print("📝 Memory after Task 2:")
        catalog = await agent.context.memory.list_entries()
        for entry in catalog:
            print(f"  - {entry['key']}: {entry['description']}")
        print()

    # Task 3: Use memory from previous tasks (THIS IS THE KEY PART!)
    print("🎯 TASK 3: Use information from memory")
    print("-" * 70)
    result3 = await agent.process(
        """Based on what you remember about my trip budget and favorite city:
        1. What was my total trip budget?
        2. What's the weather like in my favorite city?
        3. Can I add $200 for activities? What would be my new total?

        Use the information you saved earlier to answer these questions."""
    )
    print(f"\n✅ Result: {result3}\n")

    # Final memory state
    if agent.context and agent.context.memory:
        print("=" * 70)
        print("📋 Final Memory State:")
        print("=" * 70)
        catalog = await agent.context.memory.list_entries()
        memory_data = await agent.context.memory.read()

        for entry in catalog:
            key = entry['key']
            print(f"\n{key}:")
            print(f"  Description: {entry['description']}")
            print(f"  Value: {memory_data.get(key, 'N/A')}")
            if entry.get('metadata'):
                print(f"  Metadata: {entry['metadata']}")

    print("\n" + "=" * 70)
    print("Example Complete!")
    print("=" * 70)
    print("\nNotice how the agent:")
    print("  1. Calculated the budget and saved it to memory")
    print("  2. Saved your favorite city preference")
    print("  3. Retrieved both pieces of information to answer the final questions")
    print("\nThis demonstrates memory working across multiple agent.process() calls!")


if __name__ == "__main__":
    asyncio.run(main())
