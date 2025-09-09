# Opper Agent SDK

A powerful Python SDK for building AI agents using [Opper](https://opper.ai). Create intelligent agents that can operate in two modes: **tools-based reasoning loops** or **structured workflow execution**, with comprehensive event tracking for real-time UI integration.

## 🚀 Features

- **Dual Mode Operation**: Choose between tools-based reasoning or structured workflows
- **Tools Mode**: Traditional Think → Act reasoning loop with dynamic tool selection
- **Flow Mode**: Structured workflows with parallel processing and branching
- **Real-time Events**: Comprehensive event system for UI progress tracking
- **Error Handling**: Robust error handling with retry mechanisms and fallback strategies
- **Tracing & Monitoring**: Full observability with Opper's tracing system
- **Type Safety**: Full Pydantic model validation throughout execution

## 📦 Installation

```bash
pip install opper-agent-sdk
```

Or install from source:

```bash
git clone https://github.com/gsandahl/wip-agent.git
cd wip-agent
pip install -e .
```

## 🏃 Quick Start

### 1. Set up your environment

```bash
export OPPER_API_KEY="your-opper-api-key"
```

### 2. Choose Your Agent Mode

The Agent supports two modes of operation:

## 🔧 Tools Mode - Reasoning Loop

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

## 🔄 Flow Mode - Structured Workflows

Perfect for predictable, multi-step processes:

```python
import asyncio
from opper_agent import Agent
from opper_agent.workflows import Workflow, create_step
from pydantic import BaseModel, Field

class MathInput(BaseModel):
    goal: str = Field(description="The calculation goal")
    length: float = Field(default=5.0)
    width: float = Field(default=3.0)

class MathResult(BaseModel):
    thoughts: str = Field(description="Reasoning for calculations")
    area: float = Field(description="Calculated area")
    perimeter: float = Field(description="Calculated perimeter")

async def calculate_step(ctx):
    """Step that performs calculations using AI."""
    data = ctx.input_data
    
    result = await ctx.call_model(
        name="math_calculator",
        instructions="Calculate area and perimeter of rectangle with given dimensions",
        input_schema=MathInput,
        output_schema=MathResult,
        input_obj=data
    )
    return result

# Create workflow
step = create_step(id="calculate", input_model=MathInput, output_model=MathResult, run=calculate_step)
workflow = Workflow(id="math-workflow", input_model=MathInput, output_model=MathResult).then(step).commit()

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

This repository includes comprehensive examples for both agent modes:

### 🔧 Tools Mode Examples

**Dual Mode Demo (`examples/dual_mode_example.py`)**
- Compare tools vs flow modes solving the same problem
- Shows dynamic tool selection vs structured execution
- Perfect for understanding the differences

**Event Tracking (`examples/event_callback_example.py`)**
- Real-time progress tracking with event callbacks
- UI integration patterns for both modes
- Comprehensive event handling demonstrations

```bash
python examples/dual_mode_example.py
python examples/event_callback_example.py
```

### 🔄 Flow Mode Examples

**👨‍🍳 Chef Agent (`examples/chef_agent.py`)**
A cooking assistant using structured workflows:
- Sequential recipe generation pipeline
- Meal ideas → detailed recipes → quantities → shopping lists
- Shows linear workflow progression

**🤖 Chatbot Agent (`examples/chatbot_agent.py`)**  
Conversational agent with branching workflows:
- Intent detection and classification
- Conditional workflow paths based on user input
- Demonstrates workflow branching patterns

**📋 RFP Agent (`examples/rfp_agent.py`)**
Sophisticated document processor:
- **Knowledge base integration** with company information
- **Parallel processing** using `foreach` workflows  
- **Advanced workflow patterns** with error handling

```bash
python examples/chef_agent.py
python examples/chatbot_agent.py
python examples/rfp_agent.py
```

## 🏗️ Core Concepts

### Agent - Dual Mode Operation

The `Agent` is the core class that supports both operational modes:

```python
from opper_agent import Agent

# Tools Mode - Dynamic reasoning
tools_agent = Agent(
    name="ReasoningAgent",
    description="Solves problems through iterative thinking",
    tools=[tool1, tool2, tool3],
    callback=event_handler  # Optional real-time events
)

# Flow Mode - Structured workflows  
flow_agent = Agent(
    name="WorkflowAgent", 
    description="Executes predefined workflows",
    flow=my_workflow,
    callback=event_handler  # Optional real-time events
)
```

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

### Workflows - Structured Execution Paths

Define sequences of steps with various control flow patterns:

```python
from opper_agent.workflows import Workflow, create_step

workflow = Workflow(id="my-workflow", input_model=InputModel, output_model=OutputModel)
    .then(step1)                    # Sequential execution
    .parallel([step2, step3])       # Parallel execution
    .foreach(step4, concurrency=3)  # Process items in parallel
    .branch([                       # Conditional branching
        (condition1, step5),
        (condition2, step6),
    ])
    .map(transform_function)        # Data transformation
    .commit()
```

### Steps - Workflow Building Blocks

Create reusable steps with full context access:

```python
async def my_step_function(ctx):
    """Step function with access to context and events"""
    input_data = ctx.input_data
    
    # Emit custom events (forwarded through BaseAgent callback)
    ctx._emit_event("custom_progress", {"status": "processing"})
    
    # Call AI models with structured input/output
    result = await ctx.call_model(
        name="step_name",
        instructions="Your instructions",
        input_schema=InputSchema,
        output_schema=OutputSchema,
        input_obj=input_data,
    )
    
    # Access Opper client directly
    knowledge_results = ctx.opper.knowledge.query(...)
    
    return OutputModel(...)

my_step = create_step(
    id="my_step",
    input_model=InputModel,
    output_model=OutputModel,
    run=my_step_function,
)
```

### Event System - Real-time Progress Tracking

Monitor agent execution with comprehensive events:

```python
def ui_event_handler(event_type: str, data: dict):
    """Handle events for UI updates"""
    timestamp = data.get('timestamp', 0)
    
    # Tools mode events
    if event_type == "thought_created":
        update_thinking_display(data['thought']['reasoning'])
    elif event_type == "action_executed":
        update_action_log(data['action_result'])
    
    # Flow mode events  
    elif event_type == "workflow_step_start":
        update_progress_bar(f"Starting {data['step_id']}")
    elif event_type == "workflow_model_call_start":
        show_ai_indicator(data['call_name'])
    
    # Universal events
    elif event_type == "goal_completed":
        show_completion_status(data['achieved'])

# Events work with both modes
agent = Agent(name="TrackedAgent", tools=[...], callback=ui_event_handler)
```

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

### Knowledge Base Integration

Works with both modes for enhanced AI capabilities:

```python
# Setup knowledge base
knowledge_base = opper.knowledge.create(name="my-kb")
opper.knowledge.add(
    knowledge_base_id=kb.id, 
    content="Your knowledge content", 
    metadata={"category": "docs"}
)

# In Tools Mode - tools can query knowledge
@tool
def search_docs(query: str, _parent_span_id: str = None) -> str:
    """Search internal documentation."""
    # Access through opper client in tool
    results = opper.knowledge.query(knowledge_base_id=kb_id, query=query)
    return format_results(results)

# In Flow Mode - steps can query knowledge
async def knowledge_step(ctx):
    results = ctx.opper.knowledge.query(
        knowledge_base_id=kb_id, 
        query="What information do I need?",
        top_k=5
    )
    return process_knowledge(results)
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

View your traces in the [Opper Dashboard](https://platform.opper.ai) and consume events in real-time with the callback system.

## 🧪 Testing

Run the examples to verify your setup:

```bash
# Test dual mode comparison
python examples/dual_mode_example.py

# Test event callback system
python examples/event_callback_example.py

# Test flow mode examples
python examples/chef_agent.py
python examples/chatbot_agent.py
python examples/rfp_agent.py

# All examples require OPPER_API_KEY environment variable
export OPPER_API_KEY="your-api-key"
```

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

1. **Clone the repository**
```bash
git clone https://github.com/gsandahl/wip-agent.git
cd wip-agent
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
- **Issues**: [GitHub Issues](https://github.com/gsandahl/wip-agent/issues)
- **Community**: [Opper Discord](https://discord.gg/opper)

## 🗺️ Roadmap

- [x] **Dual Mode BaseAgent**: Tools-based reasoning and workflow execution
- [x] **Event Callback System**: Real-time progress tracking for UIs
- [ ] Additional workflow control structures (while loops, try/catch)
- [ ] Built-in tool integrations (web search, file operations, etc.)
- [ ] Visual workflow designer and debugging tools
- [ ] Performance optimization and caching features
- [ ] More example agents and industry-specific use cases
- [ ] Integration with popular UI frameworks (React, Vue, etc.)

---

Built with ❤️ using [Opper](https://opper.ai)
