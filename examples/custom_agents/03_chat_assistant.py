#!/usr/bin/env python3
"""
Example demonstrating a ChatAgent for conversational interactions.

This example shows how ChatAgent differs from standard Agent and ReactAgent:
- Maintains conversation state across multiple turns
- Remembers context from previous messages
- Can save and load conversation history
- Ideal for chatbot and assistant applications

Key differences:
- ChatAgent.process() takes individual messages (not complete tasks)
- Automatically manages conversation history
- Includes save/load functionality for persistence
"""

import os
import sys
import asyncio
from typing import Any

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from opper_agents import ChatAgent, tool, hook


# --- Helper Tools ---
@tool
def get_current_time() -> str:
    """Get the current time."""
    import datetime

    return datetime.datetime.now().strftime("%I:%M %p")


@tool
def get_current_date() -> str:
    """Get today's date."""
    import datetime

    return datetime.datetime.now().strftime("%B %d, %Y")


@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


@tool
def search_knowledge(query: str) -> str:
    """
    Search a simple knowledge base.
    Returns information about common topics.
    """
    knowledge_base = {
        "python": "Python is a high-level programming language known for its simplicity and readability. Created by Guido van Rossum in 1991.",
        "opper": "Opper is a platform for building AI agents with structured outputs, tracing, and tool integration.",
        "react": "ReAct (Reasoning + Acting) is a pattern for AI agents that alternates between reasoning about the task and taking actions.",
        "default": "I don't have specific information about that topic in my knowledge base.",
    }

    query_lower = query.lower()
    for key, value in knowledge_base.items():
        if key in query_lower:
            return value

    return knowledge_base["default"]


# --- Event Hooks for Visibility ---
@hook("agent_start")
async def on_conversation_start(context: Any, agent: Any) -> None:
    """Track conversation turns."""
    turn_number = len(agent.conversation.messages) // 2 + 1
    print(f"\n[Turn {turn_number}]")


@hook("agent_end")
async def on_conversation_end(context: Any, agent: Any, result: Any) -> None:
    """Show token usage for the turn."""
    tokens = context.usage.total_tokens if context else 0
    print(f"   (Tokens: {tokens})")


async def interactive_chat() -> None:
    """
    Interactive chat session with the ChatAgent.

    Demonstrates:
    - Multi-turn conversations
    - Context retention across turns
    - Tool usage in conversations
    - Conversation save/load
    """
    if not os.getenv("OPPER_API_KEY"):
        print("❌ Error: Set OPPER_API_KEY environment variable")
        print("   Example: export OPPER_API_KEY=your-key-here")
        sys.exit(1)

    # Create ChatAgent
    agent = ChatAgent(
        name="Assistant",
        model="gcp/gemini-flash-lite-latest",
        description="A helpful AI assistant that can answer questions and help with tasks.",
        instructions="""
You are a friendly and helpful AI assistant.

Guidelines:
- Be conversational and remember context from earlier in the conversation
- Use tools when appropriate to provide accurate information
- Be concise but thorough
- Ask clarifying questions if needed

When users ask about:
- Time or date: Use the get_current_time/get_current_date tools
- Calculations: Use the add_numbers tool
- General knowledge: Use the search_knowledge tool
        """,
        tools=[get_current_time, get_current_date, add_numbers, search_knowledge],
        max_history_messages=20,  # Keep last 20 messages
        verbose=False,
        hooks=[on_conversation_start, on_conversation_end],
    )

    print("\n" + "=" * 60)
    print("ChatAgent Interactive Demo")
    print("=" * 60)
    print("\nType 'exit' to quit")
    print("Type 'save' to save conversation")
    print("Type 'summary' to see conversation statistics")
    print("Type 'clear' to start a new conversation")
    print("\n" + "-" * 60)

    conversation_file = "/tmp/chat_conversation.json"

    while True:
        # Get user input
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue

        # Handle special commands
        if user_input.lower() == "exit":
            print("\nGoodbye!")
            break

        elif user_input.lower() == "save":
            agent.save_conversation(conversation_file)
            print(f"✓ Conversation saved to {conversation_file}")
            continue

        elif user_input.lower() == "summary":
            summary = agent.get_conversation_summary()
            print("\n📊 Conversation Summary:")
            print(f"   Total messages: {summary['total_messages']}")
            print(f"   Your messages: {summary['user_messages']}")
            print(f"   Assistant messages: {summary['assistant_messages']}")
            continue

        elif user_input.lower() == "clear":
            agent.clear_conversation()
            print("✓ Conversation cleared")
            continue

        # Process message with ChatAgent
        try:
            response = await agent.process(user_input)
            print(f"\nAssistant: {response}")
        except Exception as e:
            print(f"\n❌ Error: {e}")


async def programmatic_chat() -> None:
    """
    Programmatic chat example (non-interactive).

    Demonstrates how to use ChatAgent in a non-interactive application.
    """
    if not os.getenv("OPPER_API_KEY"):
        print("❌ Error: Set OPPER_API_KEY environment variable")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("ChatAgent Programmatic Demo")
    print("=" * 60)

    agent = ChatAgent(
        name="InfoBot",
        description="A bot that provides information",
        instructions="You are a helpful information assistant. Be concise and accurate.",
        tools=[get_current_time, get_current_date, search_knowledge],
        verbose=False,
        model="gcp/gemini-flash-lite-latest",
    )

    # Predefined conversation
    messages = [
        "Hello! What can you help me with?",
        "What time is it?",
        "And what's today's date?",
        "Can you tell me about Python?",
        "Thanks!",
    ]

    print("\nHaving a conversation with predefined messages...\n")

    for msg in messages:
        print(f"User: {msg}")
        response = await agent.process(msg)
        print(f"Assistant: {response}\n")

    # Show conversation summary
    print("-" * 60)
    summary = agent.get_conversation_summary()
    print("\nConversation Summary:")
    print(f"  Total turns: {summary['total_messages'] // 2}")
    print(f"  Messages: {summary['total_messages']}")

    # Save conversation
    save_path = "/tmp/programmatic_chat.json"
    agent.save_conversation(save_path)
    print(f"\n✓ Conversation saved to {save_path}")


async def main() -> None:
    """Main entry point - choose demo mode."""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--programmatic":
        await programmatic_chat()
    else:
        await interactive_chat()


if __name__ == "__main__":
    asyncio.run(main())
