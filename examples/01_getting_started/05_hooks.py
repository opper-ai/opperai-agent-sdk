"""
Quick test to show how the Agent works with all hooks.
Hooks are ways of running code at specific points in the agent's lifecycle.

The current hooks available are:
- on_agent_start
- on_agent_end
- on_agent_error
- on_loop_start
- on_loop_end
- on_llm_call
- on_llm_response
- on_think_end
- on_tool_call
- on_tool_result
"""

import asyncio
from typing import Any
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
    user_response = input("[USER INPUT REQUESTED]\n" + query + "\n")
    return user_response


# Hook definitions - demonstrating all available hooks
@hook("agent_start")
async def on_agent_start(context: AgentContext, agent: BaseAgent) -> None:
    """Called when agent execution starts."""
    print(f"\nHOOK [on_agent_start]: Agent '{agent.name}' starting execution")
    print(f"   Goal: {context.goal}")
    # context.metadata["start_timestamp"] = asyncio.get_event_loop().time()


@hook("agent_end")
async def on_agent_end(context: AgentContext, agent: BaseAgent, result: object) -> None:
    """Called when agent execution ends successfully."""
    elapsed = asyncio.get_event_loop().time() - context.metadata.get(
        "start_timestamp", 0
    )
    print(f"\nHOOK [on_agent_end]: Agent '{agent.name}' completed successfully")
    print(f"   Execution time: {elapsed:.2f}s")
    print(f"   Total iterations: {context.iteration}")


@hook("agent_error")
async def on_agent_error(
    context: AgentContext, agent: BaseAgent, error: Exception
) -> None:
    """Called when agent encounters an error."""
    print(f"\nHOOK [on_agent_error]: Agent '{agent.name}' encountered error: {error}")


@hook("loop_start")
async def on_loop_start(context: AgentContext, agent: BaseAgent) -> None:
    """Called at the start of each iteration loop."""
    print(f"\nHOOK [on_loop_start]: Loop iteration {context.iteration + 1} starting")


@hook("loop_end")
async def on_loop_end(context: AgentContext, agent: BaseAgent) -> None:
    """Called at the end of each iteration loop."""
    cycle = context.execution_history[-1] if context.execution_history else None
    if cycle:
        print(f"   [on_loop_end] Cycle: {cycle}")
        print(f"   Loop iteration {cycle.iteration + 1} completed")
        print(f"   Tools executed: {len(cycle.results)}")


@hook("llm_call")
async def on_llm_call(context: AgentContext, agent: BaseAgent, call_type: str) -> None:
    """Called before making an LLM call."""
    print(f"\nHOOK [on_llm_call]: Making LLM call (type: {call_type})")


@hook("llm_response")
async def on_llm_response(
    context: AgentContext, agent: BaseAgent, call_type: str, response: object
) -> None:
    """Called after receiving LLM response."""
    print(f"   [on_llm_response] LLM response received (type: {call_type})")


@hook("think_end")
async def on_think_end(context: AgentContext, agent: BaseAgent, thought: Any) -> None:
    """Called after the think/reasoning step."""
    print("\nHOOK [on_think_end]: Thought completed")
    print(f"   Reasoning: {thought.reasoning[:100]}...")
    print(f"   Tool calls planned: {len(thought.tool_calls)}")


@hook("tool_call")
async def on_tool_call(
    context: AgentContext, agent: BaseAgent, tool: Tool, parameters: dict
) -> None:
    """Called before executing a tool."""
    print(f"\nHOOK [on_tool_call]: Calling tool '{tool.name}'")
    print(f"   Parameters: {parameters}")


@hook("tool_result")
async def on_tool_result(
    context: AgentContext, agent: BaseAgent, tool: Tool, result: ToolResult
) -> None:
    """Called after tool execution."""
    status = "✓" if result.success else "✗"
    print(f"   [on_tool_result] {status} Tool '{tool.name}' result: {result.result}")
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
            on_agent_start,
            on_agent_end,
            on_agent_error,
            on_loop_start,
            on_loop_end,
            on_llm_call,
            on_llm_response,
            on_think_end,
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

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
