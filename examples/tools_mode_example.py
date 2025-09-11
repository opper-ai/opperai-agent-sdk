#!/usr/bin/env python3
"""
Tools Mode Example - Comprehensive demonstration of Agent operating in tools mode.

This example showcases how an Agent can dynamically reason through problems by
selecting and using appropriate tools in an iterative Think -> Act loop.
"""

import os
import sys
import math
from datetime import datetime
from typing import List, Dict

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from opper_agent import Agent, tool


# === MATHEMATICAL TOOLS ===


@tool
def calculate(expression: str) -> float:
    """
    Safely evaluate mathematical expressions.

    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 3 * 4")

    Returns:
        The calculated result
    """
    try:
        # Basic safety: only allow mathematical operations
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            raise ValueError("Expression contains invalid characters")

        result = eval(expression)
        return float(result)
    except Exception as e:
        raise ValueError(f"Cannot evaluate expression '{expression}': {e}")


@tool
def square_root(number: float) -> float:
    """
    Calculate the square root of a number.

    Args:
        number: The number to calculate square root for

    Returns:
        The square root of the number
    """
    if number < 0:
        raise ValueError("Cannot calculate square root of negative number")
    return math.sqrt(number)


@tool
def factorial(n: int) -> int:
    """
    Calculate the factorial of a number.

    Args:
        n: Non-negative integer to calculate factorial for

    Returns:
        The factorial of n
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    if n > 20:
        raise ValueError("Factorial too large (n > 20)")

    result = 1
    for i in range(1, n + 1):
        result *= i
    return result


# === TEXT PROCESSING TOOLS ===


@tool
def count_words(text: str) -> int:
    """
    Count the number of words in a text.

    Args:
        text: The text to analyze

    Returns:
        Number of words in the text
    """
    return len(text.split())


@tool
def count_characters(text: str, include_spaces: bool = True) -> int:
    """
    Count characters in text with option to include/exclude spaces.

    Args:
        text: The text to analyze
        include_spaces: Whether to include spaces in the count

    Returns:
        Number of characters
    """
    if include_spaces:
        return len(text)
    else:
        return len(text.replace(" ", ""))


@tool
def reverse_text(text: str) -> str:
    """
    Reverse the order of characters in text.

    Args:
        text: The text to reverse

    Returns:
        The reversed text
    """
    return text[::-1]


@tool
def find_longest_word(text: str) -> str:
    """
    Find the longest word in a text.

    Args:
        text: The text to analyze

    Returns:
        The longest word found
    """
    words = text.split()
    if not words:
        return ""

    longest = max(words, key=len)
    return longest


# === DATA PROCESSING TOOLS ===


@tool
def sort_numbers(numbers: List[float], ascending: bool = True) -> List[float]:
    """
    Sort a list of numbers.

    Args:
        numbers: List of numbers to sort
        ascending: Whether to sort in ascending order (True) or descending (False)

    Returns:
        Sorted list of numbers
    """
    return sorted(numbers, reverse=not ascending)


@tool
def find_statistics(numbers: List[float]) -> Dict[str, float]:
    """
    Calculate basic statistics for a list of numbers.

    Args:
        numbers: List of numbers to analyze

    Returns:
        Dictionary with mean, median, min, max, and sum
    """
    if not numbers:
        raise ValueError("Cannot calculate statistics for empty list")

    sorted_nums = sorted(numbers)
    n = len(numbers)

    # Calculate median
    if n % 2 == 0:
        median = (sorted_nums[n // 2 - 1] + sorted_nums[n // 2]) / 2
    else:
        median = sorted_nums[n // 2]

    return {
        "count": n,
        "sum": sum(numbers),
        "mean": sum(numbers) / n,
        "median": median,
        "min": min(numbers),
        "max": max(numbers),
    }


@tool
def filter_numbers(
    numbers: List[float], condition: str, threshold: float
) -> List[float]:
    """
    Filter numbers based on a condition.

    Args:
        numbers: List of numbers to filter
        condition: Condition to apply ("greater", "less", "equal", "greater_equal", "less_equal")
        threshold: The threshold value for comparison

    Returns:
        Filtered list of numbers
    """
    if condition == "greater":
        return [n for n in numbers if n > threshold]
    elif condition == "less":
        return [n for n in numbers if n < threshold]
    elif condition == "equal":
        return [n for n in numbers if n == threshold]
    elif condition == "greater_equal":
        return [n for n in numbers if n >= threshold]
    elif condition == "less_equal":
        return [n for n in numbers if n <= threshold]
    else:
        raise ValueError(f"Unknown condition: {condition}")


# === TIME AND DATE TOOLS ===


@tool
def get_current_time() -> str:
    """
    Get the current date and time.

    Returns:
        Current timestamp as a formatted string
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def days_between_dates(date1: str, date2: str) -> int:
    """
    Calculate the number of days between two dates.

    Args:
        date1: First date in YYYY-MM-DD format
        date2: Second date in YYYY-MM-DD format

    Returns:
        Number of days between the dates (positive if date2 > date1)
    """
    try:
        d1 = datetime.strptime(date1, "%Y-%m-%d")
        d2 = datetime.strptime(date2, "%Y-%m-%d")
        return (d2 - d1).days
    except ValueError as e:
        raise ValueError(f"Invalid date format. Use YYYY-MM-DD. Error: {e}")


# === FILE SYSTEM TOOLS ===


@tool
def list_files_in_directory(directory: str = ".") -> List[str]:
    """
    List files in a directory.

    Args:
        directory: Path to the directory (default: current directory)

    Returns:
        List of filenames in the directory
    """
    try:
        import os

        files = [
            f
            for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f))
        ]
        return sorted(files)
    except Exception as e:
        raise ValueError(f"Cannot list files in '{directory}': {e}")


@tool
def read_text_file(filepath: str, max_lines: int = 50) -> str:
    """
    Read content from a text file.

    Args:
        filepath: Path to the file to read
        max_lines: Maximum number of lines to read (safety limit)

    Returns:
        File content as string
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = []
            for i, line in enumerate(f):
                if i >= max_lines:
                    lines.append(f"... (truncated after {max_lines} lines)")
                    break
                lines.append(line.rstrip())
            return "\n".join(lines)
    except Exception as e:
        raise ValueError(f"Cannot read file '{filepath}': {e}")


# === DEMONSTRATION SCENARIOS ===


def create_math_agent(api_key=None):
    """Create an agent specialized in mathematical problem solving."""
    return Agent(
        name="MathWizard",
        description="An intelligent agent that solves mathematical problems using various calculation tools",
        tools=[
            calculate,
            square_root,
            factorial,
            sort_numbers,
            find_statistics,
            filter_numbers,
        ],
        opper_api_key=api_key,
        verbose=True,
    )


def create_text_analyst(api_key=None):
    """Create an agent specialized in text analysis."""
    return Agent(
        name="TextAnalyst",
        description="An intelligent agent that analyzes and processes text using various text tools",
        tools=[count_words, count_characters, reverse_text, find_longest_word],
        opper_api_key=api_key,
        verbose=True,
    )


def create_general_assistant(api_key=None):
    """Create a general-purpose agent with access to all tools."""
    all_tools = [
        # Math tools
        calculate,
        square_root,
        factorial,
        # Text tools
        count_words,
        count_characters,
        reverse_text,
        find_longest_word,
        # Data tools
        sort_numbers,
        find_statistics,
        filter_numbers,
        # Time tools
        get_current_time,
        days_between_dates,
        # File tools
        list_files_in_directory,
        read_text_file,
    ]

    return Agent(
        name="GeneralAssistant",
        description="A versatile agent that can solve various problems using mathematical, text processing, data analysis, and file system tools",
        tools=all_tools,
        opper_api_key=api_key,
        verbose=True,
    )


def demo_math_problems(api_key=None):
    """Demonstrate mathematical problem solving."""
    print("=== MATHEMATICAL PROBLEM SOLVING ===")

    agent = create_math_agent(api_key)

    problems = [
        "Calculate the area of a circle with radius 5.5",
        "Find the factorial of 8 and then calculate its square root",
        "I have these test scores: [85, 92, 78, 96, 88, 91, 84]. What are the mean, median, and which scores are above 90?",
        "Sort these numbers in descending order and find the difference between max and min: [15, 3, 9, 27, 12, 6]",
    ]

    for i, problem in enumerate(problems, 1):
        print(f"\n--- Problem {i} ---")
        print(f"Question: {problem}")
        try:
            result = agent.process(problem)
            print(f"Answer: {result}")
        except Exception as e:
            print(f"Error: {e}")

    return agent


def demo_text_analysis(api_key=None):
    """Demonstrate text analysis capabilities."""
    print("\n=== TEXT ANALYSIS ===")

    agent = create_text_analyst(api_key)

    tasks = [
        "Analyze this sentence: 'The quick brown fox jumps over the lazy dog.' Tell me the word count, character count, and longest word.",
        "Take the phrase 'Hello World' and reverse it, then count how many characters it has without spaces.",
        "What's the longest word in this text: 'Supercalifragilisticexpialidocious is a fantastically long word from Mary Poppins'?",
    ]

    for i, task in enumerate(tasks, 1):
        print(f"\n--- Task {i} ---")
        print(f"Request: {task}")
        try:
            result = agent.process(task)
            print(f"Analysis: {result}")
        except Exception as e:
            print(f"Error: {e}")

    return agent


def demo_complex_reasoning(api_key=None):
    """Demonstrate complex multi-step reasoning with tool combinations."""
    print("\n=== COMPLEX REASONING ===")

    agent = create_general_assistant(api_key)

    complex_tasks = [
        "I need to analyze my project files. First, list the files in the current directory, then read the README.md file if it exists and count how many words it contains.",
        "Help me with this data analysis: I have sales figures [1250, 980, 1100, 1350, 890, 1200, 1450, 1050]. Calculate the average, find which values are above average, and tell me how many days have passed since 2024-01-01.",
        "I'm planning a word puzzle. Take the phrase 'ARTIFICIAL INTELLIGENCE', reverse it, count the characters (excluding spaces), then calculate the square root of that number.",
        "Calculate the factorial of 6, then find the square root of that result, and finally tell me what percentage this represents of the number 100.",
    ]

    for i, task in enumerate(complex_tasks, 1):
        print(f"\n--- Complex Task {i} ---")
        print(f"Challenge: {task}")
        try:
            result = agent.process(task)
            print(f"Solution: {result}")
        except Exception as e:
            print(f"Error: {e}")

    return agent


def demo_tool_discovery(api_key=None):
    """Demonstrate how the agent discovers and uses appropriate tools."""
    print("\n=== TOOL DISCOVERY & SELECTION ===")

    agent = create_general_assistant(api_key)

    print("Available tools:")
    for tool_func in agent.tools:
        print(f"  - {tool_func.name}: {tool_func.description.split('.')[0]}")

    discovery_tasks = [
        "I need to do some math - what tools do you have for calculations?",
        "Can you help me analyze text? What text processing capabilities do you have?",
        "What can you tell me about the current time and date calculations?",
        "Do you have any file system tools? I might need to work with files.",
    ]

    for task in discovery_tasks:
        print(f"\nUser: {task}")
        try:
            result = agent.process(task)
            print(f"Agent: {result}")
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main demonstration function."""
    print("🔧 Tools Mode Example - Agent with Dynamic Tool Selection")
    print("=" * 60)

    # Check if API key is available
    api_key = os.getenv("OPPER_API_KEY")
    if not api_key:
        print("⚠️  OPPER_API_KEY not set. Using dummy key for demonstration.")
        print("   Set OPPER_API_KEY environment variable to run with real AI.")
        api_key = "dummy-key-for-demo"

    print(
        f"🔑 Using API key: {'*' * 20 if api_key != 'dummy-key-for-demo' else 'dummy-key-for-demo'}"
    )

    try:
        # Run different demonstration scenarios
        demo_math_problems(api_key)
        demo_text_analysis(api_key)
        demo_complex_reasoning(api_key)
        demo_tool_discovery(api_key)

        print("\n" + "=" * 60)
        print("✅ Tools Mode Demonstration Complete!")
        print("\n🎯 Key Takeaways:")
        print("   • Agents dynamically select appropriate tools for each task")
        print("   • Tools can be combined for complex multi-step reasoning")
        print("   • @tool decorator makes function integration seamless")
        print("   • Type hints provide automatic parameter validation")
        print("   • Agents can discover and explain their own capabilities")

    except Exception as e:
        if "dummy-key" in str(e).lower() or "api" in str(e).lower():
            print(f"\n⚠️  API Error (expected with dummy key): {str(e)[:100]}...")
            print("   Set OPPER_API_KEY to run with real AI capabilities")
        else:
            print(f"\n❌ Unexpected error: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
