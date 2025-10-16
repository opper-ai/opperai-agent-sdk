# <img src="assets/opper-logo.png" alt="Opper Logo" width="32" align="center"/> Opper Agent SDK

A Python SDK for building AI agents with [Opper Task Completion API](https://opper.ai). Create intelligent agents that use tools-based reasoning loops with dynamic tool selection, event tracking, and MCP integration.

## Table of Contents

1. [Features](#1-features)
2. [Getting Started](#2-getting-started)
3. [Installation](#3-installation)
4. [Quick Start](#4-quick-start)
   - [Set up your environment](#41-set-up-your-environment)
   - [Explore the Examples](#42-explore-the-examples)
   - [Model Selection](#43-model-selection)
5. [Agent as a Tool](#5-agent-as-a-tool)
6. [MCP (Model Context Protocol) Integration](#6-mcp-model-context-protocol-integration)
7. [Hooks](#7-hooks)
8. [Visualizing Agent Flow](#8-visualizing-agent-flow)
9. [Monitoring and Tracing](#9-monitoring-and-tracing)
10. [License](#10-license)
11. [Support](#11-support)

## 1. Features

- **Reasoning with customizable model**: Think → Act reasoning loop with dynamic tool selection
- **Extendable tool support**: Support for MCP or custom tools
- **Event Hooks**: Flexible hook system for accessing any internal Agent event
- **Composable interface**: Agent supports structured input and output schema for ease of integration
- **Multi-agent support**: Agents can be used as tools for other agents to allow for delegation
- **Type Safety internals**: Pydantic model validation throughout execution
- **Error Handling**: Robust error handling with retry mechanisms
- **Tracing & Monitoring**: Full observability with Opper's tracing system

## 2. Getting Started

Building an agent takes three steps:

```python
from opper_agent import Agent, tool

# 1. Define your tools
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny"

# 2. Create the agent
agent = Agent(
    name="WeatherBot",
    description="Helps with weather queries",
    tools=[get_weather]
)

# 3. Run it
result = await agent.process("What's the weather in Paris?")
```

## 3. Installation

### Prerequisites

- Python >= 3.11

### Install from PyPI

```bash
pip install opper-agents
```

Or using UV:
```bash
uv pip install opper-agents
```

### Install from Source (For Development)

If you want to contribute or modify the SDK:

1. **Clone the repository:**
```bash
git clone https://github.com/opper-ai/opperai-agent-sdk.git
cd opperai-agent-sdk
```

2. **Install in editable mode:**
```bash
# Using pip
pip install -e .

# Or using UV (recommended)
uv pip install -e .
```

## 4. Quick Start

### 4.1. Set up your environment

```bash
export OPPER_API_KEY="your-opper-api-key"
```

Get your API key at [platform.opper.ai](https://platform.opper.ai).

### 4.2. Explore the Examples

Check out the `examples/` directory for working examples:

- **Getting Started** (`examples/01_getting_started/`): Basic agent usage, memory, hooks
- **MCP Integration** (`examples/02_mcp_examples/`): Connect to MCP servers
- **Applied Agents** (`examples/applied_agents/`): Real-world examples like multi-agent systems
- **Custom Agents** (`examples/custom_agents/`): Build specialized agent types (React, Chat)

Run any example:
```bash
python examples/01_getting_started/01_first_agent.py
```

### 4.3. Model Selection

You can specify AI models at the **agent level** to control which model is used for reasoning. The SDK supports all models available through the Opper API:

```python
# Create agent with specific model
agent = Agent(
    name="ClaudeAgent",
    description="An agent that uses Claude for reasoning",
    tools=[my_tools],
    model="anthropic/claude-4-sonnet"  # Model for reasoning and tool selection
)
```

*If no model is specified, Opper uses a default model optimized for agent reasoning.*

## 5. Agent as a Tool

Agents can be used as tools by other agents for delegation and specialization:

```python
# Create specialized agents
math_agent = Agent(name="MathAgent", description="Performs calculations")
research_agent = Agent(name="ResearchAgent", description="Explains concepts")

# Use them as tools in a coordinator agent
coordinator = Agent(
    name="Coordinator",
    tools=[math_agent.as_tool(), research_agent.as_tool()]
)
```

See `examples/01_getting_started/02_agent_as_tool.py` for a complete example.

## 6. MCP (Model Context Protocol) Integration

The SDK supports MCP servers as tool providers, allowing agents to connect to external services. Both `stdio` and HTTP-SSE transports are supported.

```python
from opper_agent import mcp, MCPServerConfig

# Configure an MCP server
filesystem_server = MCPServerConfig(
    name="filesystem",
    transport="stdio",
    command="docker",
    args=["run", "-i", "--rm", "..."]
)

# Use MCP tools in your agent
agent = Agent(
    name="FileAgent",
    tools=[mcp(filesystem_server)],
)
```

See `examples/02_mcp_examples/` for working examples with filesystem, SQLite, and Composio servers.

## 7. Hooks

Hooks let you run code at specific points in the agent's lifecycle for logging, monitoring, or custom behavior:

**Available hooks**: `agent_start`, `agent_end`, `agent_error`, `loop_start`, `loop_end`, `llm_call`, `llm_response`, `think_end`, `tool_call`, `tool_result`

```python
from opper_agent import hook
from opper_agent.base.context import AgentContext
from opper_agent.base.agent import BaseAgent

@hook("agent_start")
async def log_start(context: AgentContext, agent: BaseAgent):
    print(f"Agent {agent.name} starting with goal: {context.goal}")

agent = Agent(
    name="MyAgent",
    hooks=[log_start],
    tools=[...]
)
```

See `examples/01_getting_started/05_hooks.py` for all available hooks with detailed examples.

## 8. Visualizing Agent Flow

Generate Mermaid diagrams of your agent's structure showing tools, sub-agents, schemas, and hooks. Perfect for documentation and understanding complex multi-agent systems.

```python
agent.visualize_flow(output_path="agent_flow.md")
```

![Agent Flow Diagram](assets/image.png)

## 9. Monitoring and Tracing

The agent provides comprehensive observability for production deployments:

### Agent Tracing
- **Agent-level spans**: Track entire reasoning sessions
- **Thought cycles**: Monitor think-act iterations  
- **Tool execution**: Performance metrics for each tool call
- **Model interactions**: AI reasoning and decision making

View your traces in the [Opper Dashboard](https://platform.opper.ai)

## 10. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 11. Support

- **Documentation**: [Opper Documentation](https://docs.opper.ai)
- **Issues**: [GitHub Issues](https://github.com/opper-ai/opperai-agent-sdk/issues)
- **Community**: [Opper Discord](https://discord.gg/opper)

---

Built with [Opper](https://opper.ai)

