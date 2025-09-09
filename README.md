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
from opper_agent import Agent, step, Workflow, StepContext
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
async def calculate_area(ctx: StepContext[MathInput, AreaResult]) -> AreaResult:
    """Calculate area using AI with automatic type inference."""
    data = ctx.input_data
    
    result = await ctx.call_model(
        name="area_calculator",
        instructions="Calculate area of rectangle with given dimensions",
        input_schema=MathInput,
        output_schema=AreaResult,
        input_obj=data
    )
    return result

@step
async def calculate_perimeter(ctx: StepContext[MathInput, PerimeterResult]) -> PerimeterResult:
    """Calculate perimeter using AI with automatic type inference."""
    data = ctx.input_data
    
    result = await ctx.call_model(
        name="perimeter_calculator",
        instructions="Calculate perimeter of rectangle with given dimensions",
        input_schema=MathInput,
        output_schema=PerimeterResult,
        input_obj=data
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

# All examples require OPPER_API_KEY environment variable
export OPPER_API_KEY="your-api-key"
```

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

### Steps - Workflow Building Blocks

Create elegant workflow steps with the `@step` decorator:

```python
from opper_agent import step, StepContext

@step
async def analyze_sentiment(ctx: StepContext[TextInput, SentimentResult]) -> SentimentResult:
    """Analyze text sentiment with automatic type inference."""
    text = ctx.input_data.text
    
    result = await ctx.call_model(
        name="sentiment_analyzer",
        instructions="Analyze the sentiment of the provided text",
        input_schema=TextInput,
        output_schema=SentimentResult,
        input_obj=ctx.input_data
    )
    return result

# Configure advanced options with decorator parameters
@step(retry={"attempts": 3}, timeout_ms=30000, on_error="continue")
async def robust_processing(ctx: StepContext[Input, Output]) -> Output:
    """Step with retry logic and error handling."""
    # Implementation with automatic fallbacks
    return await ctx.call_model(...)
```

### Workflows - Structured Execution Paths

Create sophisticated workflows with the `@step` decorator and advanced control flow:

```python
from opper_agent import Agent, step, Workflow, StepContext
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

# Define steps using @step decorator
@step
async def analyze_document(ctx: StepContext[DocumentInput, Analysis]) -> Analysis:
    """Analyze document content for sentiment and topics."""
    doc = ctx.input_data
    
    result = await ctx.call_model(
        name="document_analyzer",
        instructions="Analyze the document for sentiment and key topics",
        input_schema=DocumentInput,
        output_schema=Analysis,
        input_obj=doc
    )
    return result

@step(retry={"attempts": 3}, timeout_ms=30000)
async def enhance_content(ctx: StepContext[Analysis, str]) -> str:
    """Enhance content based on analysis."""
    analysis = ctx.input_data
    
    result = await ctx.call_model(
        name="content_enhancer",
        instructions=f"Enhance content focusing on {', '.join(analysis.topics)}",
        input_schema=Analysis,
        output_schema=str,
        input_obj=analysis
    )
    return result

@step(on_error="continue")
async def add_metadata(ctx: StepContext[Analysis, dict]) -> dict:
    """Add processing metadata."""
    analysis = ctx.input_data
    
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
async def priority_processing(ctx: StepContext[DocumentInput, DocumentInput]) -> DocumentInput:
    """Special processing for high-priority documents."""
    doc = ctx.input_data
    # Add priority-specific processing
    return doc

@step
async def batch_processing(ctx: StepContext[DocumentInput, DocumentInput]) -> DocumentInput:
    """Batch processing for normal/low priority documents."""
    doc = ctx.input_data
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

### Advanced Workflow Patterns

**Parallel Processing with `foreach`:**
```python
@step
async def process_item(ctx: StepContext[Item, ProcessedItem]) -> ProcessedItem:
    """Process individual items in parallel."""
    item = ctx.input_data
    # Process each item
    return ProcessedItem(...)

workflow = (Workflow(id="batch-processor", input_model=BatchInput, output_model=BatchOutput)
    .foreach(
        process_item,
        concurrency=5,  # Process 5 items simultaneously
        map_func=lambda batch: batch.items  # Extract items to process
    )
    .commit())
```

**Conditional Branching:**
```python
@step
async def urgent_handler(ctx: StepContext[Task, Result]) -> Result:
    """Handle urgent tasks with special processing."""
    return await ctx.call_model(name="urgent_processor", ...)

@step  
async def normal_handler(ctx: StepContext[Task, Result]) -> Result:
    """Handle normal tasks with standard processing."""
    return await ctx.call_model(name="normal_processor", ...)

workflow = (Workflow(id="task-router", input_model=Task, output_model=Result)
    .branch([
        (lambda task: task.priority == "urgent", urgent_handler),
        (lambda task: task.priority == "normal", normal_handler),
    ])
    .commit())
```

**Error Handling and Retries:**
```python
@step(retry={"attempts": 3, "backoff_ms": 1000}, on_error="continue")
async def robust_step(ctx: StepContext[Input, Output]) -> Output:
    """Step with automatic retries and graceful error handling."""
    try:
        result = await ctx.call_model(name="api_call", ...)
        return result
    except Exception as e:
        # Log error and return fallback
        ctx._emit_event("step_fallback", {"error": str(e)})
        return Output(fallback=True)
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

View your traces in the [Opper Dashboard](https://platform.opper.ai) 


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

---

Built with ❤️ using [Opper](https://opper.ai)

