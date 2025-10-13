#!/usr/bin/env python3
"""
Example demonstrating a ReactAgent for research tasks.

This example shows how ReactAgent works well for sequential research tasks where
each step depends on the previous one. The ReAct pattern is ideal for:
- Information gathering (search → read → summarize)
- Multi-step analysis (analyze → synthesize → conclude)
- Iterative refinement (try → evaluate → retry)

Unlike the standard Agent which can call multiple tools in parallel,
ReactAgent focuses on one action at a time with explicit reasoning.
"""

import os
import sys
import asyncio
from pydantic import BaseModel, Field
from typing import List

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from opper_agents import ReactAgent, tool, hook
from typing import Any


# --- Schemas ---
class ResearchRequest(BaseModel):
    topic: str = Field(description="Topic to research")
    depth: str = Field(
        default="basic", description="Research depth: basic, detailed, or comprehensive"
    )


class ResearchResult(BaseModel):
    topic: str = Field(description="Research topic")
    summary: str = Field(description="Summary of findings")
    key_points: List[str] = Field(description="Key points discovered")
    sources_used: List[str] = Field(description="Information sources consulted")


# --- Research Tools (Simulated) ---
@tool
def search_knowledge_base(query: str) -> str:
    """
    Search internal knowledge base for information.
    Returns relevant information about the query.
    """
    # Simulated knowledge base
    knowledge = {
        "python": "Python is a high-level programming language known for readability. Created by Guido van Rossum in 1991. Popular for web development, data science, AI/ML, and automation.",
        "react": "React is a JavaScript library for building user interfaces, developed by Meta. Uses component-based architecture and virtual DOM for efficient updates. Very popular for web applications.",
        "agents": "AI agents are autonomous systems that perceive their environment and take actions to achieve goals. They use reasoning, planning, and tool use to complete complex tasks.",
        "default": "Information not found in knowledge base. Try a different search term.",
    }

    query_lower = query.lower()
    for key in knowledge:
        if key in query_lower:
            return knowledge[key]

    return knowledge["default"]


@tool
def extract_key_facts(text: str, count: int = 3) -> List[str]:
    """
    Extract key facts from given text.
    Returns a list of the most important facts.
    """
    # Simple simulation - in reality would use NLP
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    return sentences[: min(count, len(sentences))]


@tool
def summarize_text(text: str, max_words: int = 50) -> str:
    """
    Summarize text to a specified maximum word count.
    Returns a concise summary.
    """
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


@tool
def verify_information(claim: str) -> dict:
    """
    Verify a claim or piece of information.
    Returns verification status and confidence level.
    """
    # Simulated verification
    return {
        "claim": claim,
        "verified": True,
        "confidence": 0.85,
        "note": "Cross-referenced with knowledge base",
    }


# --- Event Hooks for Progress Tracking ---
@hook("agent_start")
async def on_start(context: Any, agent: Any) -> None:
    """Track research start."""
    print("\nReactAgent Research Assistant Started")
    print(f"   Topic: {context.goal}")
    print("-" * 70)


@hook("loop_start")
async def on_iteration_start(context: Any, agent: Any) -> None:
    """Track each iteration."""
    print(f"\n[Iteration {context.iteration + 1}]")


@hook("think_end")
async def on_reasoning(context: Any, agent: Any, thought: Any) -> None:
    """Display agent's reasoning."""
    print(f"Thought: {thought.reasoning}")
    if thought.action:
        print(f"Action: {thought.action.tool_name}")


@hook("tool_result")
async def on_result(context: Any, agent: Any, tool: Any, result: Any) -> None:
    """Display tool results."""
    if result.success:
        result_preview = str(result.result)[:100]
        if len(str(result.result)) > 100:
            result_preview += "..."
        print(f"Result: {result_preview}")
    else:
        print(f"Error: {result.error}")


@hook("agent_end")
async def on_complete(context: Any, agent: Any, result: Any) -> None:
    """Display final results."""
    print("-" * 70)
    print("\nResearch Complete!")
    print(f"   Iterations used: {context.iteration}")
    print(f"   Tokens used: {context.usage.total_tokens}")


async def main() -> None:
    if not os.getenv("OPPER_API_KEY"):
        print("Error: Set OPPER_API_KEY environment variable")
        sys.exit(1)

    # Create ReactAgent for research
    agent = ReactAgent(
        name="ReactResearchAgent",
        description="A research agent using the ReAct pattern to gather and analyze information step by step.",
        instructions="""
You are a research assistant using the ReAct (Reasoning + Acting) pattern.

Your approach:
1. REASON: Think about what information you need
2. ACT: Use ONE tool to gather or process information
3. OBSERVE: Review what you learned
4. REPEAT: Continue until you have comprehensive findings

Available tools:
- search_knowledge_base: Find information on a topic
- extract_key_facts: Pull out important facts from text
- summarize_text: Create concise summaries
- verify_information: Check accuracy of claims

Research systematically, building on previous findings.
        """,
        tools=[
            search_knowledge_base,
            extract_key_facts,
            summarize_text,
            verify_information,
        ],
        input_schema=ResearchRequest,
        output_schema=ResearchResult,
        max_iterations=10,
        verbose=False,
        hooks=[on_start, on_iteration_start, on_reasoning, on_result, on_complete],
    )

    # --- Example Research Tasks ---

    print("\n" + "=" * 70)
    print("EXAMPLE: Research Python Programming Language")
    print("=" * 70)

    request = ResearchRequest(topic="Python programming language", depth="detailed")

    result = await agent.process(request)

    print("\nFinal Research Report:")
    print(f"   Topic: {result.topic}")
    print(f"   Summary: {result.summary}")
    print("   Key Points:")
    for i, point in enumerate(result.key_points, 1):
        print(f"      {i}. {point}")
    print(f"   Sources: {', '.join(result.sources_used)}")


if __name__ == "__main__":
    asyncio.run(main())
