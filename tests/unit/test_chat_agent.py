"""
Unit tests for ChatAgent.

Tests the conversational agent implementation.
"""

import pytest
import tempfile
import os
from unittest.mock import AsyncMock
from opper_agents.agents.chat import ChatAgent, Conversation, Message
from opper_agents.utils.decorators import tool


@tool
def get_time() -> str:
    """Get the current time."""
    return "10:30 AM"


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def test_chat_agent_initialization():
    """Test ChatAgent can be initialized."""
    agent = ChatAgent(
        name="ChatBot",
        description="A friendly chatbot",
        tools=[get_time, add],
        opper_api_key="test-key",
    )
    assert agent.name == "ChatBot"
    assert len(agent.tools) == 2
    assert len(agent.conversation.messages) == 0


def test_conversation_initialization():
    """Test Conversation can be created empty."""
    conv = Conversation()
    assert len(conv.messages) == 0


def test_conversation_add_message():
    """Test adding messages to conversation."""
    conv = Conversation()
    conv.add_message("user", "Hello")
    conv.add_message("assistant", "Hi there!")

    assert len(conv.messages) == 2
    assert conv.messages[0].role == "user"
    assert conv.messages[0].content == "Hello"
    assert conv.messages[1].role == "assistant"


def test_conversation_get_recent_messages():
    """Test retrieving recent messages."""
    conv = Conversation()
    for i in range(10):
        conv.add_message("user", f"Message {i}")

    recent = conv.get_recent_messages(n=3)
    assert len(recent) == 3
    assert recent[0].content == "Message 7"
    assert recent[2].content == "Message 9"


def test_conversation_to_dict():
    """Test conversation serialization."""
    conv = Conversation()
    conv.add_message("user", "Hello", timestamp=123.456)
    conv.add_message("assistant", "Hi!")

    data = conv.to_dict()
    assert "messages" in data
    assert len(data["messages"]) == 2
    assert data["messages"][0]["role"] == "user"
    assert data["messages"][0]["content"] == "Hello"


def test_conversation_from_dict():
    """Test conversation deserialization."""
    data = {
        "messages": [
            {"role": "user", "content": "Hello", "metadata": {}},
            {"role": "assistant", "content": "Hi!", "metadata": {}},
        ]
    }

    conv = Conversation.from_dict(data)
    assert len(conv.messages) == 2
    assert conv.messages[0].role == "user"
    assert conv.messages[1].content == "Hi!"


@pytest.mark.asyncio
async def test_chat_agent_simple_conversation(mock_opper_client):
    """Test ChatAgent handles a simple conversation."""
    # Mock responses for conversation turns
    mock_opper_client.call_async = AsyncMock(
        side_effect=[
            # Turn 1: User says hello
            AsyncMock(
                json_payload={
                    "reasoning": "User greeted me",
                    "tool_calls": [],
                    "user_message": "Responding to greeting",
                    "memory_updates": {},
                }
            ),
            AsyncMock(message="Hello! How can I help you today?"),
            # Turn 2: User asks for time
            AsyncMock(
                json_payload={
                    "reasoning": "User wants the time",
                    "tool_calls": [
                        {
                            "name": "get_time",
                            "parameters": {},
                            "reasoning": "Get current time",
                        }
                    ],
                    "user_message": "Getting time",
                    "memory_updates": {},
                }
            ),
            AsyncMock(
                json_payload={
                    "reasoning": "Got the time",
                    "tool_calls": [],
                    "user_message": "Done",
                    "memory_updates": {},
                }
            ),
            AsyncMock(message="The current time is 10:30 AM"),
        ]
    )

    agent = ChatAgent(
        name="ChatBot",
        tools=[get_time],
        verbose=False,
        opper_api_key="test-key",
    )

    # Turn 1
    response1 = await agent.process("Hello!")
    assert response1 == "Hello! How can I help you today?"
    assert len(agent.conversation.messages) == 2  # user + assistant

    # Turn 2
    response2 = await agent.process("What time is it?")
    assert "10:30 AM" in response2
    assert len(agent.conversation.messages) == 4  # 2 more messages


@pytest.mark.asyncio
async def test_chat_agent_multi_turn_context(mock_opper_client):
    """Test that ChatAgent maintains context across turns."""
    mock_opper_client.call_async = AsyncMock(
        side_effect=[
            # Turn 1
            AsyncMock(
                json_payload={
                    "reasoning": "User introduced themselves",
                    "tool_calls": [],
                    "user_message": "Noted",
                    "memory_updates": {},
                }
            ),
            AsyncMock(message="Nice to meet you, Alice!"),
            # Turn 2 - should remember the name
            AsyncMock(
                json_payload={
                    "reasoning": "User asking about previous context",
                    "tool_calls": [],
                    "user_message": "Responding",
                    "memory_updates": {},
                }
            ),
            AsyncMock(message="Yes, you told me your name is Alice."),
        ]
    )

    agent = ChatAgent(
        name="MemoryBot",
        tools=[],
        verbose=False,
        opper_api_key="test-key",
    )

    await agent.process("My name is Alice")
    response = await agent.process("Do you remember my name?")

    assert "Alice" in response


@pytest.mark.asyncio
async def test_chat_agent_save_load_conversation(mock_opper_client):
    """Test saving and loading conversation."""
    mock_opper_client.call_async = AsyncMock(
        side_effect=[
            AsyncMock(
                json_payload={
                    "reasoning": "User greeted",
                    "tool_calls": [],
                    "user_message": "Responding",
                    "memory_updates": {},
                }
            ),
            AsyncMock(message="Hello!"),
        ]
    )

    agent = ChatAgent(
        name="ChatBot",
        tools=[],
        verbose=False,
        opper_api_key="test-key",
    )

    # Have a conversation
    await agent.process("Hi there!")
    assert len(agent.conversation.messages) == 2

    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        temp_path = f.name

    try:
        agent.save_conversation(temp_path)

        # Create new agent and load
        agent2 = ChatAgent(
            name="ChatBot2",
            tools=[],
            verbose=False,
            opper_api_key="test-key",
        )
        agent2.load_conversation(temp_path)

        # Check conversation was restored
        assert len(agent2.conversation.messages) == 2
        assert agent2.conversation.messages[0].content == "Hi there!"
        assert agent2.conversation.messages[1].content == "Hello!"
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_chat_agent_clear_conversation():
    """Test clearing conversation history."""
    agent = ChatAgent(
        name="ChatBot",
        tools=[],
        verbose=False,
        opper_api_key="test-key",
    )

    # Add some messages
    agent.conversation.add_message("user", "Hello")
    agent.conversation.add_message("assistant", "Hi")
    assert len(agent.conversation.messages) == 2

    # Clear
    agent.clear_conversation()
    assert len(agent.conversation.messages) == 0


def test_chat_agent_conversation_summary():
    """Test getting conversation summary."""
    agent = ChatAgent(
        name="ChatBot",
        tools=[],
        verbose=False,
        opper_api_key="test-key",
    )

    # Add messages
    agent.conversation.add_message("user", "Message 1")
    agent.conversation.add_message("assistant", "Response 1")
    agent.conversation.add_message("user", "Message 2")
    agent.conversation.add_message("assistant", "Response 2")

    summary = agent.get_conversation_summary()
    assert summary["total_messages"] == 4
    assert summary["user_messages"] == 2
    assert summary["assistant_messages"] == 2
    assert len(summary["recent_messages"]) == 4


def test_chat_agent_max_history_trimming():
    """Test that conversation history is trimmed to max_history_messages."""
    agent = ChatAgent(
        name="ChatBot",
        tools=[],
        max_history_messages=10,
        verbose=False,
        opper_api_key="test-key",
    )

    # Add more than max messages
    for i in range(15):
        agent.conversation.add_message("user", f"Message {i}")

    # Manually trigger trim
    agent._trim_conversation_history()

    # Should keep only the last 10
    assert len(agent.conversation.messages) == 10
    assert agent.conversation.messages[0].content == "Message 5"
    assert agent.conversation.messages[-1].content == "Message 14"


@pytest.mark.asyncio
async def test_chat_agent_with_tools(mock_opper_client):
    """Test ChatAgent can use tools in conversation."""
    mock_opper_client.call_async = AsyncMock(
        side_effect=[
            # User asks for calculation
            AsyncMock(
                json_payload={
                    "reasoning": "User wants to add numbers",
                    "tool_calls": [
                        {
                            "name": "add",
                            "parameters": {"a": 5, "b": 3},
                            "reasoning": "Adding 5 and 3",
                        }
                    ],
                    "user_message": "Calculating",
                    "memory_updates": {},
                }
            ),
            AsyncMock(
                json_payload={
                    "reasoning": "Got result",
                    "tool_calls": [],
                    "user_message": "Done",
                    "memory_updates": {},
                }
            ),
            AsyncMock(message="5 + 3 equals 8"),
        ]
    )

    agent = ChatAgent(
        name="MathBot",
        tools=[add],
        verbose=False,
        opper_api_key="test-key",
    )

    response = await agent.process("What is 5 + 3?")
    assert "8" in response


def test_message_with_metadata():
    """Test Message can store metadata."""
    msg = Message(
        role="user", content="Hello", metadata={"timestamp": 123, "ip": "127.0.0.1"}
    )
    assert msg.role == "user"
    assert msg.content == "Hello"
    assert msg.metadata["timestamp"] == 123
    assert msg.metadata["ip"] == "127.0.0.1"


def test_chat_agent_inherits_from_agent():
    """Test that ChatAgent properly inherits from Agent."""
    from opper_agents.core.agent import Agent

    agent = ChatAgent(name="Test", tools=[], verbose=False, opper_api_key="test-key")
    assert isinstance(agent, Agent)
    assert hasattr(agent, "_execute_tool")
    assert hasattr(agent, "_generate_final_result")


def test_conversation_add_message_with_metadata():
    """Test adding messages with metadata."""
    conv = Conversation()
    conv.add_message("user", "Hello", timestamp=123.456, source="web")

    assert len(conv.messages) == 1
    assert conv.messages[0].metadata["timestamp"] == 123.456
    assert conv.messages[0].metadata["source"] == "web"
