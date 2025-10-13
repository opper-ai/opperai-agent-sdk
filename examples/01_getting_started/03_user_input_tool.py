"""
Basic example showing how to create a tool that gets user input.

This example demonstrates:
- Creating a tool that requests user input
- Using the tool within an agent workflow
- Building interactive agents that can ask questions
"""

import asyncio
from opper_agents import Agent, tool
from pydantic import BaseModel, Field


class ConversationGoal(BaseModel):
    topic: str = Field(description="The topic to discuss with the user")


class ConversationResult(BaseModel):
    summary: str = Field(description="A summary of the conversation")
    user_responses: list[str] = Field(description="List of user responses collected")


@tool
def get_user_input(question: str) -> str:
    """
    Ask the user a question and get their response.

    Args:
        question: The question to ask the user

    Returns:
        The user's response as a string
    """
    print(f"\n[AGENT QUESTION]: {question}")
    user_response = input("> ")
    return user_response


async def main() -> None:
    """Run a simple interactive agent."""

    # Create an agent that will have a conversation with the user
    agent = Agent(
        name="InterviewAgent",
        description="An agent that conducts a brief interview with the user",
        instructions="""
        You are a friendly interviewer. Your task is to:
        1. Ask the user 2-3 questions about the given topic
        2. Listen to their responses
        3. Provide a summary of what you learned
        
        Use the get_user_input tool to ask questions one at a time.
        Be conversational and friendly.
        """,
        tools=[get_user_input],
        input_schema=ConversationGoal,
        output_schema=ConversationResult,
        max_iterations=10,
        verbose=True,
    )

    # Start the conversation
    topic = "their favorite hobbies"

    print("=" * 60)
    print(f"Starting conversation about: {topic}")
    print("=" * 60)

    try:
        result = await agent.process({"topic": topic})

        # Display the results
        print("\n" + "=" * 60)
        print("CONVERSATION SUMMARY")
        print("=" * 60)
        print(f"\nSummary: {result.summary}")
        print(f"\nUser responses collected: {len(result.user_responses)}")
        for i, response in enumerate(result.user_responses, 1):
            print(f"  {i}. {response}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nConversation interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
