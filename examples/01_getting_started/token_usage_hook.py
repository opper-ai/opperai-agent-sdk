"""
Quick example of using hooks to check token consumption.
"""

import asyncio
from opper_agents import Agent, tool, hook
from opper_agents.base.context import AgentContext
from opper_agents.base.agent import BaseAgent
from opper_agents.base.tool import Tool, ToolResult
from pydantic import BaseModel, Field


class MathProblem(BaseModel):
    problem: str = Field(description="The math problem")


class MathSolution(BaseModel):
    answer: float = Field(description="The answer")
    reasoning: str = Field(description="How we got it")


@tool
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@tool
def multiply(x: int, y: int) -> int:
    """Multiply two numbers together."""
    return x * y


@tool
def get_user_input(query: str) -> str:
    """Get user input."""
    user_response = input("\n\n[USER INPUT REQUESTED]\n\n" + query + "\n")
    return user_response


@hook("loop_start")
async def on_loop_start(context: AgentContext, agent: BaseAgent) -> None:
    """Called at the start of each iteration loop - shows usage so far."""
    print(f"\nLoop iteration {context.iteration + 1} starting")
    print("  Token usage so far:")
    print(f"    - LLM requests: {context.usage.requests}")
    print(f"    - Input tokens:  {context.usage.input_tokens}")
    print(f"    - Output tokens: {context.usage.output_tokens}")
    print(f"    - Total tokens:  {context.usage.total_tokens}")


@hook("loop_end")
async def on_loop_end(context: AgentContext, agent: BaseAgent) -> None:
    """Called at the end of each iteration loop - shows updated usage."""
    print(f"\nLoop iteration {context.iteration} completed")
    print("  Token usage after this loop:")
    print(f"    - LLM requests: {context.usage.requests}")
    print(f"    - Input tokens:  {context.usage.input_tokens}")
    print(f"    - Output tokens: {context.usage.output_tokens}")
    print(f"    - Total tokens:  {context.usage.total_tokens}")


@hook("tool_call")
async def on_tool_call(
    context: AgentContext, agent: BaseAgent, tool: Tool, parameters: dict
) -> None:
    """Called before executing a tool."""
    print(f"\nCalling tool '{tool.name}'")
    print(f"   Parameters: {parameters}")


@hook("tool_result")
async def on_tool_result(
    context: AgentContext, agent: BaseAgent, tool: Tool, result: ToolResult
) -> None:
    """Called after tool execution."""
    status = "SUCCESS" if result.success else "FAILED"
    print(f"   [{status}] Tool '{tool.name}' result: {result.result}")
    print(f"   Execution time: {result.execution_time:.3f}s")


async def main() -> None:
    """Run a quick test of the agent."""

    # Create agent with all hooks
    agent = Agent(
        name="MathAgent",
        description="An agent that performs math operations",
        instructions="Solve the math problem using the available tools. Before concluding ask the user if any other operations are needed.",
        tools=[add, multiply, get_user_input],
        hooks=[
            on_loop_start,  # Shows usage at start of each loop
            on_loop_end,  # Shows usage at end of each loop
            on_tool_call,
            on_tool_result,
        ],
        input_schema=MathProblem,
        output_schema=MathSolution,
        max_iterations=5,
        verbose=False,  # Show detailed execution
    )

    # Run a simple task
    task_dict = {"problem": "What is (5 + 3) * 2?"}

    print(f"Task: {task_dict['problem']}\n")

    try:
        result = await agent.process(task_dict)

        print("\n" + "=" * 60)
        print("FINAL RESULT:")
        print(f"Answer: {result.answer}")
        print(f"Reasoning: {result.reasoning}")
        print("=" * 60)

        # Show execution stats
        if agent.context:
            print("\nExecution Stats:")
            print(f"  - Iterations: {agent.context.iteration}")
            print(
                f"  - Tool calls: {sum(len(c.tool_calls) for c in agent.context.execution_history)}"
            )
            print(f"  - Parent span ID: {agent.context.parent_span_id}")
            print(f"  - Session ID: {agent.context.session_id}")
            print("\n  Token Usage:")
            print(f"    - LLM requests: {agent.context.usage.requests}")
            print(f"    - Input tokens:  {agent.context.usage.input_tokens}")
            print(f"    - Output tokens: {agent.context.usage.output_tokens}")
            print(f"    - Total tokens:  {agent.context.usage.total_tokens}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
