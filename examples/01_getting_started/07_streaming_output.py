"""
Simple streaming example - Watch the agent's final output stream in real-time.

This example demonstrates:
- Enabling streaming with enable_streaming=True
- Using STREAM_CHUNK hook to see output as it's generated
- Clean, readable real-time output

Run: uv run python examples/01_getting_started/07_streaming_output.py
"""

import asyncio
import os
from opper_agents import Agent, tool, hook, HookEvents
from opper_agents.base.context import AgentContext
from pydantic import BaseModel, Field


class Story(BaseModel):
    """A creative story response."""

    title: str = Field(description="Story title")
    content: str = Field(description="Story content")
    moral: str = Field(description="Story moral/lesson")


@tool
def get_random_word() -> str:
    """Get a random word for story inspiration."""
    import random

    words = ["dragon", "castle", "wizard", "forest", "treasure"]
    return random.choice(words)


# Hook to display content as it streams
@hook(HookEvents.STREAM_CHUNK)
async def on_chunk(context: AgentContext, chunk_data: dict, **kwargs) -> None:
    """Print story content as it streams."""
    json_path = chunk_data.get("json_path", "")

    # Only show content field streaming (most interesting for stories)
    if json_path == "content":
        print(chunk_data.get("delta", ""), end="", flush=True)


@hook(HookEvents.STREAM_END)
async def on_stream_end(context: AgentContext, **kwargs) -> None:
    """Add spacing after stream completes."""
    print("\n")  # Clean line break after streaming content


async def main() -> None:
    """Run streaming story generation example."""

    if not os.getenv("OPPER_API_KEY"):
        print("Error: OPPER_API_KEY environment variable not set")
        print("\nPlease set it with:")
        print("  export OPPER_API_KEY='your-key-here'")
        return

    print("=" * 60)
    print("Streaming Story Generator")
    print("=" * 60 + "\n")

    # Create agent with streaming enabled
    agent = Agent(
        name="StorytellerAgent",
        description="An agent that creates short stories",
        instructions=(
            "Create a creative short story (2-3 paragraphs) using the random word. "
            "Make it engaging and include a clear moral lesson."
        ),
        tools=[get_random_word],
        output_schema=Story,
        enable_streaming=True,  # Enable streaming!
        hooks=[on_chunk, on_stream_end],
        max_iterations=3,
        verbose=False,
    )

    # Run the agent
    try:
        print("Generating story...\n")
        print("-" * 60)
        print()

        result = await agent.process("Write me a story")

        print("-" * 60)
        print(f"\n📖 Title: {result.title}")
        print(f"💡 Moral: {result.moral}\n")

        # Show stats
        if agent.context:
            print(f"Stats: {agent.context.iteration} iterations, {agent.context.usage}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
