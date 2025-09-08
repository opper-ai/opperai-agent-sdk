import asyncio
import os
import sys
from typing import Literal

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from pydantic import BaseModel, Field

from opper_agent import Agent, InMemoryStorage, Workflow, create_step
from opperai import Opper


class UserMessage(BaseModel):
    text: str = Field(description="The user's message")


class IntentResult(BaseModel):
    thoughts: str = Field(description="Reasoning when classifying intent")
    intent: Literal["small_talk", "linux_question", "goodbye", "unsupported"] = Field(description="Detected user intent")
    text: str = Field(description="Echo of the original user message for downstream steps")


class ChatResponse(BaseModel):
    thoughts: str = Field(description="Reasoning when crafting the reply")
    reply: str = Field(description="Assistant reply to the user")


async def detect_intent_step(ctx):
    msg: UserMessage = ctx.input_data
    result = await ctx.call_model(
        name="chat_detect_intent",
        instructions=(
            "Classify the user message into exactly one intent: 'small_talk', 'linux_question', 'goodbye', or 'unsupported'. "
            "If the message is about Linux (shell, commands, filesystems, services, packages, permissions, networking, logs), choose 'linux_question'. "
            "If it doesn't fit, choose 'unsupported'. Echo the original text in 'text'."
        ),
        input_schema=UserMessage,
        output_schema=IntentResult,
        input_obj=msg,
    )
    return result


detect_intent = create_step(
    id="detect_intent",
    input_model=UserMessage,
    output_model=IntentResult,
    run=detect_intent_step,
)


async def respond_small_talk_step(ctx):
    intent: IntentResult = ctx.input_data
    result = await ctx.call_model(
        name="chat_respond_small_talk",
        instructions=(
            "Write a warm, concise small-talk response to the user message in 'text'."
        ),
        input_schema=IntentResult,
        output_schema=ChatResponse,
        input_obj=intent,
    )
    return result


respond_small_talk = create_step(
    id="respond_small_talk",
    input_model=IntentResult,
    output_model=ChatResponse,
    run=respond_small_talk_step,
)


async def respond_linux_step(ctx):
    intent: IntentResult = ctx.input_data
    result = await ctx.call_model(
        name="chat_respond_linux",
        instructions=(
            "Answer the Linux-related question found in 'text' accurately and concisely. "
            "Provide clear steps and include exact commands with brief explanations."
        ),
        input_schema=IntentResult,
        output_schema=ChatResponse,
        input_obj=intent,
    )
    return result


respond_linux = create_step(
    id="respond_linux",
    input_model=IntentResult,
    output_model=ChatResponse,
    run=respond_linux_step,
)


async def respond_goodbye_step(ctx):
    intent: IntentResult = ctx.input_data
    result = await ctx.call_model(
        name="chat_respond_goodbye",
        instructions=(
            "Respond with a brief, friendly farewell to wrap up the conversation."
        ),
        input_schema=IntentResult,
        output_schema=ChatResponse,
        input_obj=intent,
    )
    return result


respond_goodbye = create_step(
    id="respond_goodbye",
    input_model=IntentResult,
    output_model=ChatResponse,
    run=respond_goodbye_step,
)


async def respond_unsupported_step(ctx):
    intent: IntentResult = ctx.input_data
    result = await ctx.call_model(
        name="chat_respond_unsupported",
        instructions=(
            "Politely explain that the request isn't supported and suggest asking a Linux-related question, simple question, or making small talk."
        ),
        input_schema=IntentResult,
        output_schema=ChatResponse,
        input_obj=intent,
    )
    return result


respond_unsupported = create_step(
    id="respond_unsupported",
    input_model=IntentResult,
    output_model=ChatResponse,
    run=respond_unsupported_step,
)


chat_agent = Agent(
    name="Chatbot Agent",
    instructions=(
        "You are a friendly linux expert"
    ),
    tools=[],
    flow=Workflow(id="chatbot-flow", input_model=UserMessage, output_model=ChatResponse)
        .then(detect_intent)
        .branch([
            (lambda r: r.intent == "small_talk", respond_small_talk),
            (lambda r: r.intent == "linux_question", respond_linux),
            (lambda r: r.intent == "goodbye", respond_goodbye),
            (lambda r: r.intent == "unsupported", respond_unsupported),
        ])
        .map(lambda arr: arr[0]) # Extract the first (and only) result from the branch operation
        .commit(),
)


async def main() -> None:
    api_key = os.getenv("OPPER_API_KEY")
    if not api_key:
        raise RuntimeError("OPPER_API_KEY is required")
    opper = Opper(http_bearer=api_key)

    storage = InMemoryStorage()

    # Example user inputs: create a new run per message so each is traced as a single run
    for text in [
        "Hi there!",
        "What is the capital of France?",
        "Thanks, bye!",
        "How do I check disk usage on Linux?",
    ]:
        run = chat_agent.create_run(opper=opper, storage=storage, tools={}, logger=print)
        result = await run.start(input_data=UserMessage(text=text))
        print({"input": text, "reply": result.reply})


if __name__ == "__main__":
    asyncio.run(main())
