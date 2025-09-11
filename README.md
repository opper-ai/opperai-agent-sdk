# Opper Agent SDK

A powerful Python SDK for building AI agents with [Opper Task Completion API](https://opper.ai). Create intelligent agents that can operate in two modes: **tools-based reasoning loops** or **structured workflow execution**, with the possibility to mix modes for more advanced multi agent systems. Agents comes with event tracking for real-time UI integration. 

## 🚀 Features

- **Dual Mode Operation**: Choose between tools-based reasoning or structured workflows
- **Tools Mode**: Traditional Think → Act reasoning loop with dynamic tool selection
- **Flow Mode**: Structured workflows with parallel processing and branching
- **Real-time Events**: Comprehensive event system for UI progress tracking
- **Error Handling**: Robust error handling with retry mechanisms and fallback strategies
- **Tracing & Monitoring**: Full observability with Opper's tracing system
- **Type Safety**: Full Pydantic model validation throughout execution

## 🧠 Two modes

The `Agent` is the core class that supports both operational modes:

```python
from opper_agent import Agent

# Tools Mode - Dynamic reasoning
tools_agent = Agent(
    name="ReasoningAgent",
    description="Solves problems through iterative thinking",
    tools=[tool1, tool2, tool3], # You implement tools
    callback=event_handler  # Optional real-time events
)

# Flow Mode - Structured workflows  
flow_agent = Agent(
    name="WorkflowAgent", 
    description="Executes predefined workflows",
    flow=my_workflow, # You implement steps
    callback=event_handler  # Optional real-time events
)
```

## 📦 Installation

### Prerequisites

- **Python >= 3.10**
- **Git** for cloning the repository

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
```
*Faster installation with better dependency resolution.*

3. **Verify Installation:**
```bash
python -c "from opper_agent import Agent; print('✅ opper-agent-sdk installed successfully!')"
```

### Dependencies

The package automatically installs these dependencies:
- `pydantic >= 2.6.0` - Data validation and parsing
- `opperai >= 0.3.0` - Opper API client
- `rich >= 13.7.0` - Rich text formatting
- `typing-extensions >= 4.9.0` - Extended typing support

## 🏃 Quick Start

### 1. Set up your environment

```bash
export OPPER_API_KEY="your-opper-api-key"
```

Go to https://platform.opper.ai to generate an api key.

## 2. Choose Your Agent Mode

The Agent supports two modes of operation:

### 🔧 Tools Mode - Reasoning Loop

Perfect for complex problem-solving with dynamic tool selection:

```python
from opper_agent import Agent, tool

# Define tools
@tool
def calculate_area(length: float, width: float) -> float:
    """Calculate the area of a rectangle."""
    return length * width

@tool
def calculate_perimeter(length: float, width: float) -> float:
    """Calculate the perimeter of a rectangle."""
    return 2 * (length + width)

# Create agent with tools
agent = Agent(
    name="MathAgent",
    description="An agent that helps with geometry calculations",
    tools=[calculate_area, calculate_perimeter],
    verbose=True
)

# Process a goal - agent will think and act iteratively
result = agent.process("Calculate the area and perimeter of a 5x3 rectangle")
print(result)
```

### 🔄 Flow Mode - Structured Workflows

Perfect for predictable, multi-step processes using the elegant `@step` decorator:

```python
from opper_agent import Agent, step, Workflow, ExecutionContext
from pydantic import BaseModel, Field

class MathInput(BaseModel):
    goal: str = Field(description="The calculation goal")
    length: float = Field(default=5.0)
    width: float = Field(default=3.0)

class AreaResult(BaseModel):
    thoughts: str = Field(description="Reasoning for area calculation")
    area: float = Field(description="Calculated area")

class PerimeterResult(BaseModel):
    thoughts: str = Field(description="Reasoning for perimeter calculation")
    perimeter: float = Field(description="Calculated perimeter")

# Define steps using the @step decorator - clean and intuitive!
@step
async def calculate_area(data: MathInput, ctx: ExecutionContext) -> AreaResult:
    """Calculate area using AI with direct data access."""
    result = await ctx.llm(
        name="area_calculator",
        instructions="Calculate area of rectangle with given dimensions",
        input_schema=MathInput,
        output_schema=AreaResult,
        input=data
    )
    return result

@step
async def calculate_perimeter(data: MathInput, ctx: ExecutionContext) -> PerimeterResult:
    """Calculate perimeter using AI with direct data access."""
    result = await ctx.llm(
        name="perimeter_calculator",
        instructions="Calculate perimeter of rectangle with given dimensions",
        input_schema=MathInput,
        output_schema=PerimeterResult,
        input=data,
        model="anthropic/claude-3.5-sonnet"  # Optional: specify model
    )
    return result

# Create workflow - steps are automatically configured!
workflow = (Workflow(id="math-workflow", input_model=MathInput, output_model=(AreaResult, PerimeterResult))
    .then(calculate_area)
    .then(calculate_perimeter)
    .commit())

# Create agent with workflow
agent = Agent(
    name="FlowMathAgent",
    description="Structured workflow for calculations", 
    flow=workflow,
    verbose=True
)

# Process goal - executes structured workflow
result = agent.process("Calculate area and perimeter of a 5x3 rectangle")
print(result)
```

## 🤖 Model Selection

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

### **Step-Level Model Override:**
```python
# Agent with default model
agent = Agent(
    name="MixedModelAgent",
    flow=my_workflow,
    model="openai/gpt-4o-mini"  # Default: fast and cost-effective
)

@step
async def analyze_with_default(data: InputModel, ctx: ExecutionContext) -> OutputModel:
    """Uses agent's default model (gpt-4o-mini)."""
    return await ctx.llm(
        name="analyzer",
        instructions="Analyze this data",
        input_schema=InputModel,
        output_schema=OutputModel,
        input=data
        # No model specified - uses agent default
    )

@step  
async def complex_reasoning(data: InputModel, ctx: ExecutionContext) -> OutputModel:
    """Override default for complex reasoning."""
    return await ctx.llm(
        name="reasoner", 
        instructions="Perform complex reasoning on this data",
        input_schema=InputModel,
        output_schema=OutputModel,
        input=data,
        model="anthropic/claude-4.1-opus"  # Override agent default
    )
```


*If no model is specified at agent or step level, Opper uses a default model.*

## 🔗 MCP (Model Context Protocol) Integration

Connect your agents to external tools and data sources using the standardized Model Context Protocol developed by Anthropic. **Fully supports HTTP-SSE transport** for modern MCP servers including the Opper documentation server.

### **Quick MCP Setup:**
```python
from opper_agent import Agent, create_mcp_tools
from opper_agent.mcp_client import MCPServerConfig

# Create Opper docs MCP server configuration
opper_docs_server = MCPServerConfig(
    name="opper",
    url="https://docs.opper.ai/mcp",
    transport="http-sse",
    enabled=True
)

# Create MCP tools from Opper docs server
mcp_tools = create_mcp_tools([opper_docs_server])

# Create agent with MCP tools
agent = Agent(
    name="MCPAgent",
    description="Agent with access to Opper documentation",
    tools=mcp_tools(),
    model="anthropic/claude-3.5-sonnet"
)

# Agent can now access live Opper documentation and provide guidance
result = agent.process("What are the best practices for building AI applications with Opper?")
# Returns comprehensive information from the Opper knowledge base
```

### **Custom MCP Servers:**
```python
from opper_agent.mcp_client import MCPServerConfig

# Configure custom MCP server
custom_server = MCPServerConfig(
    name="my_custom_server",
    command="node",
    args=["my-mcp-server.js", "--config", "config.json"],
    timeout=30.0
)

# Use with agent
mcp_tools = create_mcp_tools([custom_server])
agent = Agent(name="CustomAgent", tools=mcp_tools())
```

**Transport Support:**
- **HTTP-SSE**: Modern web-based MCP servers (like Opper docs)
- **stdio**: Traditional subprocess-based MCP servers (requires Node.js/npm)

## 📡 Real-time Status with Event Callbacks

Track agent progress in real-time for UI integration:

```python
def progress_handler(event_type: str, data: dict):
    """Handle real-time events from agent execution."""
    if event_type == "goal_start":
        print(f"🎯 Starting: {data['goal']}")
    elif event_type == "thought_created":
        print(f"🧠 Thinking: {data['thought']['reasoning'][:50]}...")
    elif event_type == "action_executed":
        result = data['action_result']
        status = "✅" if result['success'] else "❌"
        print(f"⚡ Action: {result['tool_name']} {status}")
    elif event_type.startswith("workflow_"):
        workflow_event = event_type.replace("workflow_", "")
        if workflow_event == "step_start":
            print(f"📝 Step: {data['step_id']}")
        elif workflow_event == "model_call_start":
            print(f"🤖 AI Call: {data['call_name']}")

# Use callback with any agent mode
agent = Agent(
    name="TrackedAgent",
    tools=[...],  # or flow=workflow
    callback=progress_handler,  # Real-time events
    verbose=False  # Let events handle the output
)

result = agent.process("Your goal here")
```

## 📚 Examples

```bash
# Test comprehensive tools mode demonstration
python examples/tools_mode_example.py

# Test comprehensive flow mode demonstration  
python examples/flow_mode_example.py

# Test MCP integration with Opper docs server
python examples/mcp_example.py

# All examples require OPPER_API_KEY environment variable
export OPPER_API_KEY="your-api-key"
```

**Note:** The MCP example demonstrates live integration with the Opper documentation server, allowing agents to access real-time information about Opper features, APIs, and best practices.

## 🏗️ Core Concepts

### Tools - Building Blocks for Reasoning

Create tools that agents can dynamically select and use:

```python
from opper_agent import tool

@tool
def web_search(query: str, limit: int = 5) -> list:
    """Search the web for information."""
    # Your implementation here
    return search_results

@tool  
def calculate(expression: str) -> float:
    """Evaluate a mathematical expression."""
    return eval(expression)  # Use safely in production!

# Tools are automatically described to the agent
agent = Agent(name="Helper", tools=[web_search, calculate])
```

### Flows - Structured Execution Paths

Create sophisticated workflows with the `@step` decorator and advanced control flow:

```python
from opper_agent import Agent, step, Workflow, ExecutionContext
from pydantic import BaseModel, Field
from typing import List

# Define data models
class DocumentInput(BaseModel):
    content: str = Field(description="Document content to analyze")
    priority: str = Field(description="Processing priority: high, normal, low")

class Analysis(BaseModel):
    sentiment: str = Field(description="Sentiment analysis result")
    topics: List[str] = Field(description="Key topics identified")
    summary: str = Field(description="Document summary")

class ProcessedDoc(BaseModel):
    analysis: Analysis
    enhanced_content: str = Field(description="Enhanced document content")
    metadata: dict = Field(description="Processing metadata")

# Define steps using @step decorator with data + context pattern
@step
async def analyze_document(doc: DocumentInput, ctx: ExecutionContext) -> Analysis:
    """Analyze document content for sentiment and topics."""
    result = await ctx.llm(
        name="document_analyzer",
        instructions="Analyze the document for sentiment and key topics",
        input_schema=DocumentInput,
        output_schema=Analysis,
        input=doc
    )
    return result

@step(retry={"attempts": 3}, timeout_ms=30000)
async def enhance_content(analysis: Analysis, ctx: ExecutionContext) -> str:
    """Enhance content based on analysis."""
    result = await ctx.llm(
        name="content_enhancer",
        instructions=f"Enhance content focusing on {', '.join(analysis.topics)}",
        input_schema=Analysis,
        output_schema=str,
        input=analysis
    )
    return result

@step(on_error="continue")
async def add_metadata(analysis: Analysis, ctx: ExecutionContext) -> dict:
    """Add processing metadata."""
    return {
        "processed_at": "2024-01-01T00:00:00Z",
        "topics_count": len(analysis.topics),
        "sentiment_confidence": 0.95
    }

# Conditional processing based on priority
def is_high_priority(data):
    return data.priority == "high"

def is_normal_priority(data):
    return data.priority == "normal"

@step
async def priority_processing(doc: DocumentInput, ctx: ExecutionContext) -> DocumentInput:
    """Special processing for high-priority documents."""
    # Add priority-specific processing
    return doc

@step
async def batch_processing(doc: DocumentInput, ctx: ExecutionContext) -> DocumentInput:
    """Batch processing for normal/low priority documents."""
    # Add batch-specific processing
    return doc

# Build complex workflow with branching and parallelism
workflow = (Workflow(id="document-processor", input_model=DocumentInput, output_model=ProcessedDoc)
    # Conditional branching based on priority
    .branch([
        (is_high_priority, priority_processing),
        (is_normal_priority, batch_processing),
    ])
    
    # Parallel processing of analysis tasks
    .parallel([
        analyze_document,
        # Process multiple analysis types in parallel
        Workflow(id="parallel-analysis", input_model=DocumentInput, output_model=Analysis)
            .then(analyze_document)
            .commit()
    ])
    
    # Sequential processing of results
    .then(enhance_content)
    .then(add_metadata)
    
    # Final combination step
    .map(lambda results: ProcessedDoc(
        analysis=results[0],  # From analyze_document
        enhanced_content=results[1],  # From enhance_content  
        metadata=results[2]   # From add_metadata
    ))
    .commit())
```

### Advanced Flow Patterns

## 🔄 Advanced Features

### Mode Comparison

| Feature | Tools Mode | Flow Mode |
|---------|------------|-----------|
| **Execution Style** | Think → Act loop | Structured workflow |
| **Flexibility** | High (dynamic decisions) | Medium (predefined steps) |
| **Best For** | Complex reasoning, exploration | Predictable processes, pipelines |
| **Tool Selection** | AI-driven, contextual | Predefined in workflow |
| **Error Handling** | Built-in retry logic | Configurable per step |
| **Parallelization** | Sequential by default | Built-in parallel execution |

### Tools Mode Advanced Features

**Dynamic Tool Selection:**
```python
# Agent automatically chooses appropriate tools based on context
@tool
def web_search(query: str) -> str: ...

@tool  
def calculate(expression: str) -> float: ...

@tool
def send_email(to: str, subject: str, body: str) -> bool: ...

agent = Agent(name="Assistant", tools=[web_search, calculate, send_email])
# Agent will intelligently select tools as needed
```

**Custom Output Schemas:**
```python
from pydantic import BaseModel

class TaskResult(BaseModel):
    completed: bool
    summary: str
    actions_taken: list[str]

agent = Agent(
    name="TaskAgent",
    tools=[...],
    output_schema=TaskResult  # Structured final output
)
```

### Flow Mode Advanced Features

**Parallel Processing with `foreach`:**
```python
.foreach(
    process_item_step,
    concurrency=3,  # Process 3 items simultaneously
    map_func=lambda data: data.items  # Extract items to process
)
```

**Conditional Branching:**
```python
.branch([
    (lambda data: data.priority == "urgent", urgent_handler),
    (lambda data: data.priority == "normal", normal_handler),
])
```

**Error Handling with Retries:**
```python
create_step(
    id="robust_step",
    retry={"attempts": 3, "backoff_ms": 1000},
    on_error="continue",  # "fail", "skip", or "continue"
    run=step_function,
)
```

## 📊 Monitoring and Tracing

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

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

1. **Clone the repository**
```bash
git clone https://github.com/opper-ai/opperai-agent-sdk.git
cd opperai-agent-sdk
```

2. **Create a virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install in development mode**
```bash
pip install -e .
```

4. **Set up your environment**
```bash
export OPPER_API_KEY="your-opper-api-key"
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [Opper Documentation](https://docs.opper.ai)
- **Issues**: [GitHub Issues](https://github.com/opper-ai/opperai-agent-sdk/issues)
- **Community**: [Opper Discord](https://discord.gg/opper)

---

Built with ❤️ using [Opper](https://opper.ai)

