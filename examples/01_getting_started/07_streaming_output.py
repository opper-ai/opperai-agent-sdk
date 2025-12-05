"""
Simple streaming example — watch the agent's final output stream in real-time.

This example demonstrates:
- Enabling streaming with enable_streaming=True
- Using STREAM_START/STREAM_CHUNK/STREAM_END hooks to show streaming lifecycle
- Printing streamed content for a specific field (content)

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


# Minimal streaming state (for clean run)
class _MinimalStreamState:
    def __init__(self) -> None:
        self.think_started = False
        self.final_started = False


_state_min = _MinimalStreamState()


# Verbose hooks: labeled lifecycle and chunks
@hook(HookEvents.STREAM_START)
async def on_stream_start_verbose(
    context: AgentContext, call_type: str, **kwargs
) -> None:
    """Announce the start of a streaming call (from hook)."""
    print(f"[hook STREAM_START] call_type={call_type}")


@hook(HookEvents.STREAM_CHUNK)
async def on_chunk_verbose(context: AgentContext, chunk_data: dict, **kwargs) -> None:
    """Print streaming info and stream content field in-place (from hook)."""
    call_type = kwargs.get("call_type")
    json_path = chunk_data.get("json_path", "")
    delta = chunk_data.get("delta", "")

    # Check if this is a content field we want to stream inline
    # With single LLM call pattern, final_result fields come during "think"
    # with paths like "final_result.content" or "user_message"
    is_content_field = (
        json_path == "content"
        or json_path == "final_result.content"
        or json_path == "user_message"
    )

    # Log hook event with path for non-content fields
    if not is_content_field:
        if isinstance(delta, str) and delta.strip():
            print(
                f"[hook STREAM_CHUNK] call_type={call_type} path={json_path} delta={delta}"
            )
        return

    # Handle user_message specifically for verbose mode
    if json_path == "user_message" and isinstance(delta, str) and delta.strip():
        print(
            f"\n[User Message] {delta}", end="", flush=True
        )  # Print user message on a new line
        return

    # Content field: stream inline, but skip empty/whitespace-only deltas
    if isinstance(delta, str) and delta.strip():
        print(delta, end="", flush=True)


@hook(HookEvents.STREAM_END)
async def on_stream_end_verbose(
    context: AgentContext, call_type: str, **kwargs
) -> None:
    """Announce end of streaming and add a clean break (from hook)."""
    if call_type == "final_result":
        print()  # newline after streaming content
    print(f"[hook STREAM_END] call_type={call_type}")


# Minimal hooks: content-only streaming without labels
@hook(HookEvents.STREAM_CHUNK)
async def on_chunk_minimal(context: AgentContext, chunk_data: dict, **kwargs) -> None:
    call_type = kwargs.get("call_type")
    json_path = chunk_data.get("json_path", "")
    delta = chunk_data.get("delta", "")
    # Show minimal thinking (reasoning) without labels
    if (
        call_type == "think"
        and json_path == "reasoning"
        and isinstance(delta, str)
        and delta.strip()
    ):
        if not _state_min.think_started:
            print("Thinking...")
            _state_min.think_started = True
        print(delta, end="", flush=True)
        return

    # Show user_message updates minimally
    if (
        call_type == "think"
        and json_path == "user_message"
        and isinstance(delta, str)
        and delta.strip()
    ):
        print(f"\nStatus: {delta}", end="", flush=True)
        return

    # Show final content inline - with single LLM call pattern, final_result
    # is streamed during "think" with json_path="final_result.content"
    # Also support legacy "final_result" call_type for backward compatibility
    is_final_content = (
        call_type == "think" and json_path == "final_result.content"
    ) or (call_type == "final_result" and json_path == "content")
    if is_final_content and isinstance(delta, str) and delta.strip():
        if not _state_min.final_started:
            print("\nFinal output...")
            _state_min.final_started = True
        print(delta, end="", flush=True)


@hook(HookEvents.STREAM_END)
async def on_stream_end_minimal(
    context: AgentContext, call_type: str, **kwargs
) -> None:
    if call_type == "think":
        print()
    if call_type == "final_result":
        print()


async def main() -> None:
    """Run streaming story generation example."""

    if not os.getenv("OPPER_API_KEY"):
        print("Error: OPPER_API_KEY environment variable not set")
        print("\nPlease set it with:")
        print("  export OPPER_API_KEY='your-key-here'")
        return

    print("=" * 60)
    print("Streaming Story Generator")
    print("=" * 60)

    # Run 1: Verbose streaming with hook outputs
    print("Run 1: Verbose streaming with hook outputs")
    agent_verbose = Agent(
        name="StorytellerAgentVerbose",
        description="An agent that creates short stories",
        instructions=(
            "Create a creative short story (2-3 paragraphs) using the random word. "
            "Make it engaging and include a clear moral lesson."
        ),
        tools=[get_random_word],
        output_schema=Story,
        enable_streaming=True,
        hooks=[on_stream_start_verbose, on_chunk_verbose, on_stream_end_verbose],
        max_iterations=3,
        verbose=False,
    )

    try:
        print("Generating story...")
        print("-" * 60)
        result = await agent_verbose.process("Write me a story")
        print("-" * 60)
        print(f"Title: {result.title}")
        print(f"Moral: {result.moral}")
        if agent_verbose.context:
            print(
                f"Stats: {agent_verbose.context.iteration} iterations, {agent_verbose.context.usage}"
            )
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)

    # Run 2: Clean streaming (content only)
    print("Run 2: Clean streaming (content only)")
    # Reset minimal streaming state
    global _state_min
    _state_min = _MinimalStreamState()
    agent_minimal = Agent(
        name="StorytellerAgentClean",
        description="An agent that creates short stories",
        instructions=(
            "Create a creative short story (2-3 paragraphs) using the random word. "
            "Make it engaging and include a clear moral lesson."
        ),
        tools=[get_random_word],
        output_schema=Story,
        enable_streaming=True,
        hooks=[on_chunk_minimal, on_stream_end_minimal],
        max_iterations=3,
        verbose=False,
    )

    try:
        print("Generating story...")
        print("-" * 60)
        result2 = await agent_minimal.process("Write me a story")
        print("-" * 60)
        print(f"Title: {result2.title}")
        print(f"Moral: {result2.moral}")
        if agent_minimal.context:
            print(
                f"Stats: {agent_minimal.context.iteration} iterations, {agent_minimal.context.usage.total_tokens} total tokens"
            )
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
