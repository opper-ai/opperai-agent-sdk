"""
Example 09b: Parallel Agent-as-Tool Execution

This example demonstrates:
- Composing specialized agents using as_tool()
- Running multiple sub-agents in parallel with parallel_tool_execution
- Using run() to get RunResult with usage breakdown per agent
- How parallel execution speeds up multi-agent coordination

Run with: uv run python examples/01_getting_started/09b_parallel_agent_as_tool.py
"""

import asyncio
import os
import time
from opper_agents import Agent


# =============================================================================
# Specialized sub-agents (each handles one domain)
# =============================================================================

weather_agent = Agent(
    name="WeatherAgent",
    description="Provides weather information for a given location",
    instructions=(
        "You are a weather expert. When asked about the weather:\n"
        "1. Provide a realistic weather description for the location\n"
        "2. Include temperature, conditions, and any notable details\n"
        "Keep responses concise (under 50 words)."
    ),
    max_iterations=2,
    opper_api_key=os.getenv("OPPER_API_KEY"),
)

finance_agent = Agent(
    name="FinanceAgent",
    description="Provides stock market and financial information",
    instructions=(
        "You are a finance expert. When asked about stocks or markets:\n"
        "1. Provide a brief market summary or stock info\n"
        "2. Include a realistic price and trend\n"
        "Keep responses concise (under 50 words)."
    ),
    max_iterations=2,
    opper_api_key=os.getenv("OPPER_API_KEY"),
)

news_agent = Agent(
    name="NewsAgent",
    description="Provides latest news headlines on any topic",
    instructions=(
        "You are a news analyst. When asked about news:\n"
        "1. Provide 2-3 realistic headline summaries\n"
        "2. Keep it brief and informative\n"
        "Keep responses concise (under 50 words)."
    ),
    max_iterations=2,
    opper_api_key=os.getenv("OPPER_API_KEY"),
)

# =============================================================================
# Coordinator agent: runs sub-agents in parallel
# =============================================================================

coordinator = Agent(
    name="BriefingCoordinator",
    description="Coordinates multiple specialist agents to produce a briefing",
    instructions=(
        "You are a briefing coordinator. When the user asks for a briefing:\n"
        "- Delegate to ALL specialist agents simultaneously to gather information\n"
        "- Use weather_agent for weather, finance_agent for markets, news_agent for news\n"
        "- Combine their outputs into a concise executive briefing"
    ),
    tools=[
        weather_agent.as_tool("weather_agent", "Get weather for a location"),
        finance_agent.as_tool("finance_agent", "Get stock/market information"),
        news_agent.as_tool("news_agent", "Get latest news on a topic"),
    ],
    parallel_tool_execution=True,  # Run sub-agents concurrently
    max_iterations=5,
    verbose=True,
    opper_api_key=os.getenv("OPPER_API_KEY"),
)


async def main() -> None:
    if not os.getenv("OPPER_API_KEY"):
        print("Error: Set OPPER_API_KEY environment variable")
        return

    print("=" * 60)
    print("Parallel Agent-as-Tool Example")
    print("=" * 60)

    task = (
        "Give me a morning briefing: weather in San Francisco, "
        "how's the S&P 500 doing, and top AI news."
    )
    print(f"\nTask: {task}\n")

    try:
        start = time.time()
        run_result = await coordinator.run(task)
        elapsed = int((time.time() - start) * 1000)

        print("\n" + "=" * 60)
        print("Final Briefing:")
        print("=" * 60)
        print(run_result.result)

        print(f"\nTotal time: {elapsed}ms")
        print(
            "Note: All 3 sub-agents ran in parallel, each making their own LLM calls."
        )

        # Show usage from RunResult
        usage = run_result.usage
        print("\n" + "=" * 60)
        print("Usage Statistics:")
        print("=" * 60)
        print(f"  Total requests: {usage.requests}")
        print(f"  Total tokens:   {usage.total_tokens}")

        # Show per-agent breakdown
        if usage.breakdown:
            print("\n  Per-agent breakdown:")
            for name, stats in usage.breakdown.items():
                print(f"    {name}: {stats.total_tokens} tokens")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
