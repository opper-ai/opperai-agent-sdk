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


async def main():
    """Run a multi-task agent example with memory."""

    # Create agent with memory enabled
    agent = Agent(
        name="AssistantAgent",
        description="A helpful assistant that remembers context across tasks",
        instructions="You are a helpful assistant",
        tools=[calculate, get_weather],
        enable_memory=True,
        max_iterations=15,
        verbose=True,
    )

    print("=" * 70)
    print("Agent Memory Example - Memory Within Single Process")
    print("=" * 70)
    print()
    print("This example shows memory working WITHIN a single process() call.")
    print("The agent will use memory as a 'scratch pad' during multi-step tasks.")
    print()

    # Task: Multi-step task that uses memory WITHIN the same process() call
    print("🎯 TASK: Multi-step vacation planning with memory")
    print("-" * 70)
    result = await agent.process(
        """I'm planning a vacation to Tokyo. Please:
        1. Calculate the base costs: Flight $800 + Hotel for 7 nights at $150/night + Food budget $100/day for 7 days
        2. Save this as 'tokyo_base_budget' in memory with a clear description
        3. Check the weather in Tokyo
        4. Now retrieve the base budget from memory and add $500 for activities
        5. Tell me the final total budget

        Important: Use memory to store the base budget after step 1, then retrieve it in step 4."""
    )
    print(f"\n✅ Result: {result}\n")

    # Final memory state
    if agent.context and agent.context.memory:
        print("=" * 70)
        print("📋 Final Memory State:")
        print("=" * 70)
        catalog = await agent.context.memory.list_entries()
        memory_data = await agent.context.memory.read()

        for entry in catalog:
            key = entry["key"]
            print(f"\n{key}:")
            print(f"  Description: {entry['description']}")
            print(f"  Value: {memory_data.get(key, 'N/A')}")
            if entry.get("metadata"):
                print(f"  Metadata: {entry['metadata']}")

    print("\n" + "=" * 70)
    print("Example Complete!")
    print("=" * 70)
    print("\nNotice how the agent:")
    print("  1. Calculated the base budget and saved it to memory")
    print("  2. Performed other tasks (weather check)")
    print("  3. Retrieved the base budget from memory later in the same task")
    print("  4. Used the retrieved value to calculate the final total")
    print("\nMemory acts as a 'scratch pad' for complex multi-step reasoning!")
    print("It persists within a single process() call, enabling the agent to")
    print("store intermediate results and retrieve them when needed.")


if __name__ == "__main__":
    asyncio.run(main())
