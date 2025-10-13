"""
Example: Comparing different logger styles.

This example demonstrates the different logging options available:
1. SimpleLogger (default when verbose=True) - basic print-based logging
2. RichLogger - fancy console output with Opper brand colors
3. No logging (verbose=False)
"""

import asyncio
from opper_agents import Agent, tool, RichLogger


@tool
def calculate(expression: str) -> float:
    """
    Safely evaluate a mathematical expression.

    Args:
        expression: Math expression to evaluate (e.g., "2 + 2")

    Returns:
        Result of the calculation
    """
    # Simple safe evaluation for demo purposes
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return float(result)
    except Exception as e:
        raise ValueError(f"Invalid expression: {e}")


@tool
def format_currency(amount: float, currency: str = "USD") -> str:
    """
    Format a number as currency.

    Args:
        amount: Amount to format
        currency: Currency code (default: USD)

    Returns:
        Formatted currency string
    """
    return f"{currency} {amount:,.2f}"


async def main() -> None:
    print("=" * 80)
    print("1. SIMPLELOGGER (DEFAULT WITH VERBOSE=TRUE)")
    print("=" * 80)

    # SimpleLogger is used by default when verbose=True
    agent_simple = Agent(
        name="CalculatorAgent",
        description="Agent that performs calculations",
        tools=[calculate, format_currency],
        verbose=True,  # Automatically uses SimpleLogger
    )

    result = await agent_simple.process(
        "Calculate 123.45 * 67.89 and format it as USD currency"
    )
    print(f"\nFinal result: {result}\n")

    print("\n" + "=" * 80)
    print("2. RICHLOGGER (FANCY WITH OPPER COLORS)")
    print("=" * 80)

    # Use RichLogger for colorful, formatted logging
    agent_rich = Agent(
        name="CalculatorAgent",
        description="Agent that performs calculations",
        tools=[calculate, format_currency],
        logger=RichLogger(),  # Explicitly use RichLogger with Opper colors
    )

    result = await agent_rich.process(
        "Calculate 100 + 200 and format it as EUR currency"
    )
    print(f"\nFinal result: {result}\n")

    print("\n" + "=" * 80)
    print("3. NO LOGGING (SILENT MODE)")
    print("=" * 80)

    # No logging at all
    agent_silent = Agent(
        name="CalculatorAgent",
        description="Agent that performs calculations",
        tools=[calculate, format_currency],
        verbose=False,  # No logging
    )

    result = await agent_silent.process(
        "Calculate 50 * 2 and format it as GBP currency"
    )
    print(f"Final result (with silent execution): {result}\n")


if __name__ == "__main__":
    asyncio.run(main())
