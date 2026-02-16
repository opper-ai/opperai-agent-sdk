"""
Example 09a: Parallel Tool Execution

This example demonstrates:
- Using parallel_tool_execution=True to run independent tools concurrently
- How parallel execution reduces total latency when tools are independent
- Using run() to get both result and usage statistics

Each tool simulates a 500ms API call. With 3 tools:
- Sequential: ~1500ms for tool execution
- Parallel:   ~500ms for tool execution

Run with: uv run python examples/01_getting_started/09a_parallel_tool_execution.py
"""

import asyncio
import os
import time
from opper_agents import Agent, tool


@tool
async def fetch_weather(city: str) -> dict:
    """Fetch current weather for a city.

    Args:
        city: City name

    Returns:
        Weather information
    """
    print(f"  Fetching weather for {city}...")
    start = time.time()
    await asyncio.sleep(0.5)  # Simulate API latency
    elapsed = int((time.time() - start) * 1000)
    print(f"  Weather for {city} done ({elapsed}ms)")
    return {"city": city, "temperature": 22, "condition": "Sunny"}


@tool
async def fetch_news(topic: str) -> dict:
    """Fetch latest news headlines about a topic.

    Args:
        topic: News topic to search for

    Returns:
        News headlines
    """
    print(f"  Fetching news about {topic}...")
    start = time.time()
    await asyncio.sleep(0.5)  # Simulate API latency
    elapsed = int((time.time() - start) * 1000)
    print(f"  News about {topic} done ({elapsed}ms)")
    return {
        "topic": topic,
        "headlines": [
            f"Breaking: New developments in {topic}",
            f"{topic} trends upward this quarter",
        ],
    }


@tool
async def fetch_stock(symbol: str) -> dict:
    """Fetch current stock price for a ticker symbol.

    Args:
        symbol: Stock ticker symbol

    Returns:
        Stock price information
    """
    print(f"  Fetching stock price for {symbol}...")
    start = time.time()
    await asyncio.sleep(0.5)  # Simulate API latency
    elapsed = int((time.time() - start) * 1000)
    print(f"  Stock {symbol} done ({elapsed}ms)")
    return {"symbol": symbol, "price": 185.42, "change": "+1.2%"}


async def main() -> None:
    if not os.getenv("OPPER_API_KEY"):
        print("Error: Set OPPER_API_KEY environment variable")
        return

    print("=" * 60)
    print("Parallel Tool Execution Example")
    print("=" * 60)

    # Create agent with parallel tool execution enabled
    agent = Agent(
        name="DashboardAgent",
        description="An agent that gathers data from multiple sources",
        instructions=(
            "You are a dashboard assistant. When asked for a briefing, "
            "gather ALL the requested data using the available tools. "
            "Use all relevant tools to collect comprehensive information."
        ),
        tools=[fetch_weather, fetch_news, fetch_stock],
        parallel_tool_execution=True,  # Enable parallel execution
        max_iterations=5,
        verbose=True,
    )

    task = (
        "Give me a quick briefing: weather in Tokyo, "
        "latest tech news, and AAPL stock price."
    )
    print(f"\nTask: {task}\n")

    try:
        start = time.time()
        run_result = await agent.run(task)
        elapsed = int((time.time() - start) * 1000)

        print("\n" + "=" * 60)
        print(f"Result: {run_result.result}")
        print("=" * 60)

        # Show timing
        print(f"\nTotal time: {elapsed}ms")
        print(
            "Note: With 3 tools each taking ~500ms, sequential would be "
            "~1500ms for tool execution alone."
        )

        # Show usage
        usage = run_result.usage
        print(f"\nUsage: {usage.requests} requests, {usage.total_tokens} tokens")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
