"""
Advanced streaming example - Monitor agent reasoning and tool selection in real-time.

This example demonstrates:
- Streaming hooks for intermediate agent steps (think/reasoning)
- Field-specific streaming (watch "reasoning", "tool_calls", etc.)
- Using accumulated data to show progress
- Real-time visibility into agent decision-making

Run: uv run python examples/other_examples/advanced_streaming_hooks.py
"""

import asyncio
import os
from opper_agents import Agent, tool, hook, HookEvents
from opper_agents.base.context import AgentContext
from pydantic import BaseModel, Field


class Analysis(BaseModel):
    """Analysis result."""

    topic: str = Field(description="What was analyzed")
    key_findings: list[str] = Field(description="Main findings")
    conclusion: str = Field(description="Final conclusion")


@tool
def research_topic(topic: str) -> dict:
    """Research a topic and return key facts.

    This helper performs a simple keyword match so that broader topic
    descriptions (e.g., "recent developments in AI") still return data.
    """
    tl = topic.lower()

    if any(k in tl for k in ["ai", "artificial intelligence", "machine learning"]):
        facts = [
            "LLMs advancing rapidly",
            "Multimodal models emerging",
            "Agentic AI growing",
        ]
    elif any(k in tl for k in ["climate", "global warming", "climate change"]):
        facts = ["Global temps rising", "Ice caps melting", "Sea levels up"]
    elif any(k in tl for k in ["space", "astronomy", "mars", "telescope"]):
        facts = [
            "Mars missions planned",
            "Webb telescope discoveries",
            "Private spaceflight",
        ]
    else:
        facts = ["No data available"]

    return {"facts": facts}


@tool
def analyze_data(data: str) -> dict:
    """Analyze data and extract insights."""
    return {
        "insights": [
            f"Pattern detected in: {data[:30]}...",
            "Trend shows increasing importance",
            "Correlation with recent developments",
        ]
    }


# Track streaming state
class StreamState:
    """Track which fields we've seen."""

    reasoning_seen = False
    tool_calls_seen = False
    current_call_type = None


state = StreamState()


@hook(HookEvents.STREAM_START)
async def on_stream_start(context: AgentContext, call_type: str, **kwargs) -> None:
    """Mark the start of a streaming call."""
    state.current_call_type = call_type
    state.reasoning_seen = False
    state.tool_calls_seen = False

    if call_type == "think":
        print("\n🤔 Agent thinking...")
    elif call_type == "final_result":
        print("\n📝 Generating final analysis...")


@hook(HookEvents.STREAM_CHUNK)
async def on_chunk(
    context: AgentContext, chunk_data: dict, accumulated: str, **kwargs
) -> None:
    """React to specific fields as they stream."""
    json_path = chunk_data.get("json_path", "")
    call_type = kwargs.get("call_type", "")

    # Show reasoning as it streams (thinking process)
    if json_path == "reasoning" and call_type == "think":
        if not state.reasoning_seen:
            print("   💭 Reasoning: ", end="", flush=True)
            state.reasoning_seen = True
        print(chunk_data.get("delta", ""), end="", flush=True)

    # Detect tool selection
    elif (
        "tool_calls" in json_path and not state.tool_calls_seen and call_type == "think"
    ):
        state.tool_calls_seen = True
        print("\n   🔧 Selecting tools...")

    # Show tool names as they arrive
    elif json_path.endswith(".name") and "tool_calls" in json_path:
        print(f"      → {accumulated}")

    # Show final output fields streaming
    elif call_type == "final_result":
        if json_path == "topic" and accumulated:
            print(f"   Topic: {accumulated}")
        elif json_path == "conclusion":
            if chunk_data.get("delta"):
                if not hasattr(state, "conclusion_shown"):
                    print("   Conclusion: ", end="", flush=True)
                    state.conclusion_shown = True
                print(chunk_data.get("delta", ""), end="", flush=True)


@hook(HookEvents.STREAM_END)
async def on_stream_end(context: AgentContext, call_type: str, **kwargs) -> None:
    """Clean up after stream ends."""
    if call_type == "think":
        print("\n   ✓ Thinking complete\n")
    elif call_type == "final_result":
        print("\n   ✓ Analysis complete\n")
        if hasattr(state, "conclusion_shown"):
            delattr(state, "conclusion_shown")


async def main() -> None:
    """Run advanced streaming example."""

    if not os.getenv("OPPER_API_KEY"):
        print("Error: OPPER_API_KEY environment variable not set")
        print("\nPlease set it with:")
        print("  export OPPER_API_KEY='your-key-here'")
        return

    print("=" * 60)
    print("Advanced Streaming: Real-time Agent Monitoring")
    print("=" * 60)
    print("\nThis shows streaming of both:")
    print("  • Intermediate reasoning (agent's thought process)")
    print("  • Final output generation")
    print("=" * 60)

    # Create agent with streaming
    agent = Agent(
        name="AnalystAgent",
        description="Research and analysis agent",
        instructions=(
            "Research the given topic thoroughly. "
            "Use available tools to gather facts and analyze them. "
            "Provide a comprehensive analysis with key findings."
        ),
        tools=[research_topic, analyze_data],
        output_schema=Analysis,
        enable_streaming=True,
        hooks=[on_stream_start, on_chunk, on_stream_end],
        max_iterations=5,
        verbose=False,
        # logger=RichLogger(),
    )

    # Run analysis
    try:
        task = "Analyze recent developments in AI"
        print(f"\n🎯 Task: {task}\n")

        result = await agent.process(task)

        # Show final structured result
        print("=" * 60)
        print("FINAL RESULT:")
        print(f"  Topic: {result.topic}")
        print(f"  Findings: {len(result.key_findings)} key points")
        for i, finding in enumerate(result.key_findings, 1):
            print(f"    {i}. {finding}")
        print(f"  Conclusion: {result.conclusion}")
        print("=" * 60)

        # Stats
        if agent.context:
            print(
                f"\nStats: {agent.context.iteration} iterations, {agent.context.usage.total_tokens} total tokens"
            )

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
