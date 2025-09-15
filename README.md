# Opper Agent SDK
[![Sign Up and Start Using Opper](https://img.shields.io/badge/Sign%20Up-Start%20Using%20Opper-blue?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAzMTUgMzE1Ij48dGl0bGU+T3BwZXI8L3RpdGxlPjxnIGNsaXAtcGF0aD0idXJsKCNjbGlwMCkiPjxwYXRoIGZpbGwtcnVsZT0iZXZlbm9kZCIgY2xpcC1ydWxlPSJldmVub2RkIiBkPSJNMTA1LjA0IDE1Ny41MkMxMjAuNiAxNTcuMzQgMTYwLjUyIDE1MS41NCAxNjAuNTIgOTYuNTYwMUMxNjAuNTIgMTUxLjU0IDIwMC40NCAxNTcuMzQgMjE2IDE1Ny41MkMxNjQuMSAxNjEuNjMgMTYwLjUyIDIxNy45OCAxNjAuNTIgMjE3Ljk4QzE2MC41MiAyMTcuOTggMTU2Ljk0IDE2MS42NSAxMDUuMDQgMTU3LjUyWk0xNTkuNzggMzE1QzcxLjUzIDMxNSAwIDI0NC40OSAwIDE1Ny41QzAgLTE4Ljk0OTkgMTU5Ljc4IDAuNjUwMDc1IDE1OS43OCAwLjY1MDA3NUMxNTkuNzggODcuMjIgODguMzYgMTU3LjQgMC4yIDE1Ny41QzE0OS44IDE1Ny42NCAxNTkuNzggMzE1IDE1OS43OCAzMTVaIiBmaWxsPSJ1cmwoI2dyYWQpIi8+PC9nPjxkZWZzPjxsaW5lYXJHcmFkaWVudCBpZD0iZ3JhZCIgeDE9IjcyLjM4IiB5MT0iNTQuNzgwMSIgeDI9IjEzOS41MiIgeTI9IjI3MC44IiBncmFkaWVudFVuaXRzPSJ1c2VyU3BhY2VPblVzZSI+PHN0b3Agc3RvcC1jb2xvcj0iIzhDRUNGMiIvPjxzdG9wIG9mZnNldD0iMSIgc3RvcC1jb2xvcj0iI0Y5QjU4QyIvPjwvbGluZWFyR3JhZGllbnQ+PC9kZWZzPjwvc3ZnPg==
)](https://opper.ai/)
A powerful Python SDK for building AI agents with [Opper Task Completion API](https://opper.ai).  Agents comes with event tracking for real-time UI integration. 

## Table of Contents

1. [Features](#1-features)
2. [Agent Modes](#2-agent-modes)
   - [Tools Mode](#21-tools-mode)
   - [Flow Mode - Structured workflows](#22-flow-mode---structured-workflows)
   - [Combined Agents](#23-combined-agents)
3. [Installation](#3-installation)
4. [Quick Start](#4-quick-start)
5. [Model Selection](#5-model-selection)
6. [Examples](#6-examples)
7. [Monitoring and Tracing](#7-monitoring-and-tracing)
8. [License](#8-license)
9. [Support](#9-support)

## 1. Features

- **Dual Mode Operation**: Choose between tools-based reasoning or structured workflows
- **Real-time Events**: Comprehensive event system for UI progress tracking
- **Error Handling**: Robust error handling with retry mechanisms and fallback strategies
- **Tracing & Monitoring**: Full observability with Opper's tracing system
- **Type Safety**: Full Pydantic model validation throughout execution


## 2. Agent Modes

The `Agent` supports two operational modes that can be used independently or combined. Ultimately it's up to the user to chose the type of agent to build (and many tasks can be solved with both methods), here are some suggestions on where to use which:

- **Tools Mode**: For dynamic reasoning where the AI decides which tools to use
- **Flow Mode**: For structured workflows with predefined steps
- **Combined**: Mix both approaches in a single agent


### 2.1 Tools Mode
[Tools Mode Docs](docs_tools.md)
The agent dynamically selects and calls your Python functions based on the task. Great for exploratory tasks and complex reasoning.
```python
from opper_agent import Agent, tool

@tool
def tool1(place: str, date: str) -> str:
    return "The weather in " + place + " on " + date + " is sunny" # dummy tool implementation

tools_agent = Agent(
    name="WeatherAgent",
    description="Given a place and a date returns the expected weather",
    tools=[tool1], # Provided tools
)
```

### 2.2 Flow Mode - Structured workflows 
[Flow Mode Docs](flow.md)
Define a sequence of typed steps that execute in order. Perfect for predictable processes and when you need reliable, repeatable outcomes.
Basic flow structure:
```python
@step
def step_1():
    #Implementation goes here
    return "Output step 1"

def step_2():
    return "Output step 2"

workflow = (
    Workflow(
        id="travel-itinerary-workflow",
        input_model=...,
        output_model=...,
    )
    .then(step_1)  
    .then(step_2)  # Outputs of step 1 feed step 2
    .commit()
)

```

Complete travel planning example:
```python
from opper_agent import Agent, step, Workflow, ExecutionContext

class TravelRequest(BaseModel):
    request: str = Field(description="The user's travel request")

class TravelItinerary(BaseModel):
    daily_plan: str = Field(description="Day-by-day itinerary with activities and attractions")
    must_see: str = Field(description="Top must-see attractions for this duration")
    food_recommendations: str = Field(description="Local food and restaurant suggestions")

# Step 1: Extract travel details from user request
@step
async def extract_travel_details(data: TravelRequest, ctx: ExecutionContext) -> TravelDetails:
    """Extract destination city and duration from the user's travel request."""
    result = await ctx.llm(
        name="travel_parser",
        instructions="Extract the destination city and number of stay days from the user's travel request.",
        input_schema=TravelRequest,
        output_schema=TravelDetails,
        input=data,
    )
    return result

# Step 2: Create travel itinerary based on extracted details
@step
async def create_itinerary(data: TravelDetails, ctx: ExecutionContext) -> TravelItinerary:
    """Create a detailed travel itinerary for the specified destination and duration."""
    result = await ctx.llm(
        name="itinerary_planner",
        instructions="""Create a detailed travel itinerary based on the destination and
        duration. Include day-by-day activities, must-see attractions, food
        recommendations, and practical tips""",
        input_schema=TravelDetails,
        output_schema=TravelItinerary,
        input=data,
        model="anthropic/claude-3.5-sonnet",  # Optional: specify model
    )
    return result


# Create workflow - sequential steps where output of step 1 feeds into step 2
workflow = (
    Workflow(
        id="travel-itinerary-workflow",
        input_model=TravelRequest,
        output_model=TravelItinerary,
    )
    .then(extract_travel_details)  # Step 1: TravelRequest -> TravelDetails
    .then(create_itinerary)  # Step 2: TravelDetails -> TravelItinerary
    .commit()
)

# Create agent with workflow
agent = Agent(
    name="TravelPlannerAgent",
    description="Sequential workflow that extracts travel details then creates an itinerary",
    flow=workflow,
    verbose=True,
    model="berget/gpt-oss-120b", # You can define the LLM here
)

# Process goal - executes sequential workflow
result = agent.process("I'm visiting Rome for 3 days, can you help me plan my trip?")
print(result)
```

### 2.3 Combined Agents
You can call tools-mode agents from within flow steps. For example, add weather checking to the travel itinerary workflow by calling the weather agent as an intermediate step.

```python
workflow = (
    Workflow(
        id="travel-itinerary-workflow",
        input_model=TravelRequest,
        output_model=TravelItinerary,
    )
    .then(extract_travel_details) 
    .then(get_weather_info)  # Add this new step that calls your tools agent
    .then(create_itinerary)        
    .commit()
)
```
Full implementation of the combined agents [here](./examples/combined_tool_flow.py)



## 3. Installation

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
*Changes to the code are immediately reflected - perfect for development and testing.*

**Option 2: Regular Installation**
```bash
pip install .
```
*Installs as a regular package - more stable for production use.*

**Option 3: Using UV (Modern Package Manager)**
```bash
uv pip install -e .
# or
uv sync #if you cloned the package
```
*Faster installation with better dependency resolution.*

3. **Verify Installation:**
```bash
python -c "from opper_agent import Agent; print('✅ opper-agent-sdk installed successfully!')"
```


## 4. Quick Start

### 1. Set up your environment

```bash
export OPPER_API_KEY="your-opper-api-key"
```

Go to https://platform.opper.ai to generate an api key.


## 5. Model Selection

You can specify AI models at the **agent level** (default for all steps) or **step level** (override for specific steps). The SDK supports all models available through the Opper API:

### **Agent-Level Default Model:**
```python
# Set default model for all workflow steps
agent = Agent(
    name="ClaudeAgent",
    description="An agent that uses Claude by default",
    flow=my_workflow,
    model="anthropic/claude-3.5-sonnet"  # Default model for all steps
)

# Or for tools mode
tools_agent = Agent(
    name="GPTAgent", 
    description="An agent that uses GPT-4 by default",
    tools=[my_tools],
    model="groq/gpt-oss-120b"  # Default model for reasoning
)
```



*If no model is specified at agent or step level, Opper uses a default model.*


## 6. Examples

See many implementation examples under the examples directory

## 7. Monitoring and Tracing

Both agent modes provide comprehensive observability:

### Tools Mode Tracing
- **Agent-level spans**: Track entire reasoning sessions
- **Thought cycles**: Monitor think-act iterations
- **Tool execution**: Performance metrics for each tool call
- **Model interactions**: AI reasoning and decision making

### Flow Mode Tracing  
- **Workflow-level spans**: Track entire workflow execution
- **Step-level performance**: Monitor individual step performance
- **Model call logging**: Detailed logging of all AI interactions
- **Parallel execution**: Visibility into concurrent operations

### Real-time Events
Both modes emit comprehensive events for:
- **Progress tracking**: Real-time status updates
- **Performance monitoring**: Execution times and metrics
- **Error handling**: Detailed error information and retries
- **UI integration**: Perfect for building responsive interfaces

View your traces in the [Opper Dashboard](https://platform.opper.ai) 


## 8. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 9. Support

- **Documentation**: [Opper Documentation](https://docs.opper.ai)
- **Issues**: [GitHub Issues](https://github.com/opper-ai/opperai-agent-sdk/issues)
- **Community**: [Opper Discord](https://discord.gg/opper)

---

Built with ❤️ using [Opper](https://opper.ai)

