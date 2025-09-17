# Opper Agent SDK

A Python SDK for building AI agents with [Opper Task Completion API](https://opper.ai). Create intelligent agents that use tools-based reasoning loops with dynamic tool selection, event tracking, and MCP integration.

## Features

- **Reasoning with customizable model**: Think → Act reasoning loop with dynamic tool selection
- **Extendable tool support**: Support for MCP or custom tools
- **Event Hooks**: Flexible hook system for accessing any internal Agent event
- **Composable interface**: Agent supports structured input and output schema for ease of integration
- **Multi-agent support**: Agents can be used as tools for other agents to allow for delegation
- **Type Safety internals**: Pydantic model validation throughout execution
- **Error Handling**: Robust error handling with retry mechanisms
- **Tracing & Monitoring**: Full observability with Opper's tracing system

## Agent Architecture

The `Agent` class provides a reasoning loop for dynamic problem solving:

```python
from opper_agent import Agent, tool

@tool
def calculate_area(length: float, width: float) -> float:
    """Calculate the area of a rectangle."""
    return length * width

agent = Agent(
    name="ReasoningAgent",
    description="Solves problems through iterative thinking",
    tools=[calculate_area], 
)
```

## Installation

### Prerequisites

- Python >= 3.10
- Git for cloning the repository

### Install from Source

1. **Clone the repository:**
```bash
git clone https://github.com/opper-ai/opperai-agent-sdk.git
cd opperai-agent-sdk
```

2. **Install the package:**

**Option 1: Editable Installation (Recommended for Development)**
```bash
pip install -e .
```

**Option 2: Regular Installation**
```bash
pip install .
```

**Option 3: Using UV (Modern Package Manager)**
```bash
uv pip install -e .
```

**Option 4: Using UV with uv run (No Installation Required)**
```bash
uv run python examples/math_agent_example.py
```

3. **Verify Installation:**
```bash
python -c "from opper_agent import Agent; print('opper-agent-sdk installed successfully!')"
```

### Dependencies

The package automatically installs these dependencies:
- `pydantic >= 2.6.0` - Data validation and parsing
- `opperai >= 0.3.0` - Opper API client
- `rich >= 13.7.0` - Rich text formatting
- `typing-extensions >= 4.9.0` - Extended typing support
- `aiohttp >= 3.8.0` - HTTP client for MCP integration

## Quick Start

### 1. Set up your environment

```bash
export OPPER_API_KEY="your-opper-api-key"
```

Go to https://platform.opper.ai to generate an api key.

### 2. Example Math Agent

Here's a complete example of a math agent with event hooks and structured input/output:

```python
import asyncio
from opper_agent import Agent, tool, hook, RunContext
from pydantic import BaseModel, Field

# Define input/output schemas
class MathProblem(BaseModel):
    problem: str = Field(description="The math problem to solve")
    show_work: bool = Field(default=True, description="Whether to show step-by-step work")

class MathSolution(BaseModel):
    problem: str = Field(description="The original problem")
    answer: float = Field(description="The numerical answer")

# Define math tools
@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

@tool
def subtract_numbers(a: float, b: float) -> float:
    """Subtract the second number from the first number."""
    return a - b

@tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b

@tool
def divide_numbers(a: float, b: float) -> float:
    """Divide the first number by the second number."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

# Event hook to log agent startup
@hook("on_agent_start")
async def on_agent_start(context: RunContext, agent: Agent):
    print(f"🧮 Math Agent started")
    print(f"   Problem: {context.goal}")

# Event hook to log agent actions
@hook("on_think_end")
async def on_think_end(context: RunContext, agent: Agent, thought: Any):
    """Post-thinking hook to analyze the agent's reasoning."""
    print(f"{thought.user_message}")

# Create the math agent
agent = Agent(
    name="MathAgent",
    description="A mathematical agent that can perform basic arithmetic operations and solve math problems step by step.",
    tools=[add_numbers, subtract_numbers, multiply_numbers, divide_numbers],
    hooks=[on_agent_start, on_think_end],
    input_schema=MathProblem,
    output_schema=MathSolution,
    verbose=False
)

# Use the agent
async def main():
    problem = MathProblem(
        problem="Calculate (12 * 8) + (45 / 9) - 7"
    )
    result = await agent.process(problem)
    print(f"Result: {result}")

# Run the example
asyncio.run(main())
```

## Model Selection

You can specify AI models at the **agent level** to control which model is used for reasoning. The SDK supports all models available through the Opper API:

```python
# Create agent with specific model
agent = Agent(
    name="ClaudeAgent",
    description="An agent that uses Claude for reasoning",
    tools=[my_tools],
    model="anthropic/claude-4-sonnet"  # Model for reasoning and tool selection
)

# Or use a different model
gpt_agent = Agent(
    name="GPTAgent", 
    description="An agent that uses GPT-4 for reasoning",
    tools=[my_tools],
    model="openai/gpt-4o"  # Model for reasoning
)
```

*If no model is specified, Opper uses a default model optimized for agent reasoning.*

## MCP (Model Context Protocol) Integration

Connect your agents to external tools and data sources using the standardized Model Context Protocol developed by Anthropic. Fully supports HTTP-SSE transport for modern MCP servers.

### Quick MCP Setup

```python
import asyncio
from opper_agent import Agent, create_mcp_tools_async
from opper_agent.mcp import MCPServerConfig

async def main():
    mcp_server = MCPServerConfig(
        name="my_server",
        url="https://your-mcp-server.com/mcp",
        transport="http-sse",
        enabled=True
    )

    tools = await create_mcp_tools_async([mcp_server])
    agent = Agent(
        name="MCPAgent",
        description="Agent with access to external tools via MCP",
        tools=tools,
        model="anthropic/claude-3.5-sonnet"
    )

    result = await agent.process("Your goal here")
    print(result)

asyncio.run(main())
```

### MCPToolManager for Advanced Usage

For better connection management and multiple MCP servers:

```python
import asyncio
from opper_agent import Agent
from opper_agent.mcp import MCPServerConfig, MCPToolManager

async def main():
    # Create MCP tool manager
    mcp_manager = MCPToolManager()
    
    # Add multiple MCP servers
    gmail_server = MCPServerConfig(
        name="gmail",
        url="https://mcp.composio.dev/partner/composio/gmail/mcp",
        transport="http-sse"
    )
    
    filesystem_server = MCPServerConfig(
        name="filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    )
    
    mcp_manager.add_server(gmail_server)
    mcp_manager.add_server(filesystem_server)
    
    try:
        # Connect to all servers
        await mcp_manager.connect_all()
        tools = mcp_manager.get_all_tools()
        
        # Create agent with all MCP tools
        agent = Agent(
            name="MultiMCPAgent",
            description="Agent with access to multiple MCP servers",
            tools=tools
        )
        
        result = await agent.process("Your goal here")
        print(result)
        
    finally:
        # Properly cleanup connections
        await mcp_manager.disconnect_all()

asyncio.run(main())
```

### Custom MCP Servers

```python
from opper_agent.mcp import MCPServerConfig

custom_server = MCPServerConfig(
    name="my_custom_server",
    command="node",
    args=["my-mcp-server.js", "--config", "config.json"],
    timeout=30.0
)

mcp_tools = create_mcp_tools([custom_server])
agent = Agent(name="CustomAgent", tools=mcp_tools())
```

**Transport Support:**
- **HTTP-SSE**: Modern web-based MCP servers
- **stdio**: Traditional subprocess-based MCP servers (requires Node.js/npm)

## Multi-Agent Systems

Create sophisticated multi-agent systems where agents can delegate tasks to each other using the `agent.as_tool()` method.

### Simple Multi-Agent Example

```python
import asyncio
from opper_agent import Agent, tool
from pydantic import BaseModel, Field

# Define specialized agents with instructions
@tool
def calculate(expression: str) -> float:
    """Calculate a mathematical expression."""
    return eval(expression)

math_agent = Agent(
    name="MathAgent",
    description="Handles mathematical calculations",
    instructions="Always show your work step by step and explain your reasoning.",
    tools=[calculate]
)

@tool
def translate_to_swedish(text: str) -> str:
    """Translate English to Swedish."""
    return "hej" if "hello" in text.lower() else text

swedish_agent = Agent(
    name="SwedishAgent", 
    description="Handles Swedish language tasks",
    instructions="Provide both the Swedish translation and a brief explanation of the grammar.",
    tools=[translate_to_swedish]
)

# Create routing agent using agent.as_tool()
routing_agent = Agent(
    name="RoutingAgent",
    description="Routes tasks to specialized agents",
    tools=[
        math_agent.as_tool(tool_name="delegate_to_math"),
        swedish_agent.as_tool(tool_name="delegate_to_swedish")
    ]
)

# Use the multi-agent system
async def main():
    result = await routing_agent.process("Calculate 15 * 8 + 42")
    print(result)

asyncio.run(main())
```

### Additional multi-Agent Features

- **Agent Hierarchies**: Agents can delegate to other agents that delegate to more agents
- **Custom Tool Names**: `agent.as_tool(tool_name="custom_name")`
- **Custom Descriptions**: `agent.as_tool(description="Custom description")`
- **Agent Instructions**: Define instructions in the Agent constructor for consistent behavior
- **Timeout Protection**: 60-second timeout prevents hanging operations
- **Structured Responses**: Consistent response format across all agents

The instructions are automatically prepended to the task when delegating, ensuring consistent behavior across your multi-agent system.

## Example agents

```bash
# Run the main math agent example (featured)
python examples/math_agent_example.py

# Test weather conversation agent with hooks
python examples/weather_conversation_example.py

# Test MCP integration with Gmail (requires setup)
python examples/composio_gmail_mcp.py

# Test multi-agent system with agent.as_tool()
python examples/multi_agent_example.py

# Test Opper docs integration (currently has server issues)
python examples/opper_docs_example.py

# All examples require OPPER_API_KEY environment variable
export OPPER_API_KEY="your-api-key"
```

## Available Hooks

Available hook methods: `on_agent_start`, `on_agent_end`, `on_agent_error`, `on_iteration_start`, `on_iteration_end`, `on_think_start`, `on_think_end`, `on_tool_start`, `on_tool_end`, `on_tool_error`, `on_llm_start`, `on_llm_end`.

## Monitoring and Tracing

The agent provides comprehensive observability for production deployments:

### Agent Tracing
- **Agent-level spans**: Track entire reasoning sessions
- **Thought cycles**: Monitor think-act iterations  
- **Tool execution**: Performance metrics for each tool call
- **Model interactions**: AI reasoning and decision making

View your traces in the [Opper Dashboard](https://platform.opper.ai) 

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [Opper Documentation](https://docs.opper.ai)
- **Issues**: [GitHub Issues](https://github.com/opper-ai/opperai-agent-sdk/issues)
- **Community**: [Opper Discord](https://discord.gg/opper)

---

Built with [Opper](https://opper.ai)

