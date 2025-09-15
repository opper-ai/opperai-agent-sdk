# Flow Mode Documentation

Build structured, predictable AI workflows with typed steps and powerful orchestration patterns.

## Table of Contents

1. [Introduction](#1-introduction)
2. [Core Concepts](#2-core-concepts)
   - [Steps](#21-steps)
   - [Workflows](#22-workflows)
   - [ExecutionContext](#23-executioncontext)
3. [Getting Started](#3-getting-started)
4. [Advanced Features](#4-advanced-features)
   - [Parallel Processing](#41-parallel-processing)
   - [Conditional Branching](#42-conditional-branching)
   - [Error Handling](#43-error-handling)
   - [State Management](#44-state-management)
5. [Model Selection](#5-model-selection)
6. [Event Callbacks](#6-event-callbacks)
7. [Best Practices](#7-best-practices)

## 1. Introduction

Flow mode provides structured, sequential execution of AI tasks through predefined steps. Perfect for:

- **Predictable processes**: Multi-step workflows with consistent outputs
- **Data pipelines**: Transform data through a series of AI operations  
- **Complex orchestration**: Parallel processing, branching, and error handling
- **Reusable workflows**: Share and compose workflows across projects

## 2. Core Concepts

### 2.1 Steps

Steps are the building blocks of workflows. Each step receives typed input and produces typed output.

```python
from opper_agent import step, ExecutionContext
from pydantic import BaseModel, Field

class UserInput(BaseModel):
    text: str = Field(description="User's raw input")

class Analysis(BaseModel):
    sentiment: str = Field(description="Sentiment: positive, negative, neutral")
    topics: list[str] = Field(description="Key topics identified")

@step
async def analyze_text(data: UserInput, ctx: ExecutionContext) -> Analysis:
    """Analyze user input for sentiment and topics."""
    result = await ctx.llm(
        name="text_analyzer",
        instructions="Analyze the text for sentiment and key topics",
        input_schema=UserInput,
        output_schema=Analysis,
        input=data,
    )
    return result
```

**Step Features:**
- **Type safety**: Pydantic models for inputs and outputs
- **Automatic validation**: Input/output validation at runtime
- **Retry logic**: Built-in retry with exponential backoff
- **Error handling**: Configurable error strategies

### 2.2 Workflows

Workflows orchestrate steps into complete processes using a fluent API.

```python
from opper_agent import Workflow

workflow = (
    Workflow(
        id="sentiment-analyzer",
        input_model=UserInput,
        output_model=Analysis,
    )
    .then(analyze_text)  # Sequential execution
    .commit()  # Finalize workflow
)
```

**Workflow Patterns:**
- **Sequential**: `.then(step)` - Steps execute in order
- **Parallel**: `.parallel([step1, step2])` - Steps execute simultaneously  
- **Conditional**: `.branch([(condition, step)])` - Conditional execution
- **Loops**: `.foreach(step, map_func)` - Process collections

### 2.3 ExecutionContext

The context provides access to AI models, state, and utilities within steps.

```python
@step
async def enhanced_step(data: InputModel, ctx: ExecutionContext) -> OutputModel:
    # Make AI calls
    result = await ctx.llm(name="processor", ...)
    
    # Access state
    previous_data = ctx.get_state("key", default_value)
    ctx.set_state("result", result)
    
    # Get workflow metadata
    workflow_id = ctx.workflow_id
    step_name = ctx.current_step
    
    return result
```

## 3. Getting Started

### Basic Workflow Example

```python
from opper_agent import Agent, step, Workflow, ExecutionContext
from pydantic import BaseModel, Field

# Define data models
class TaskInput(BaseModel):
    description: str = Field(description="Task description")

class TaskAnalysis(BaseModel):
    complexity: str = Field(description="Task complexity: simple, medium, complex")
    estimated_time: int = Field(description="Estimated time in minutes")

class TaskPlan(BaseModel):
    analysis: TaskAnalysis
    steps: list[str] = Field(description="Ordered list of execution steps")

# Define workflow steps
@step
async def analyze_task(data: TaskInput, ctx: ExecutionContext) -> TaskAnalysis:
    """Analyze task complexity and time requirements."""
    return await ctx.llm(
        name="task_analyzer",
        instructions="Analyze the task complexity and estimate completion time",
        input_schema=TaskInput,
        output_schema=TaskAnalysis,
        input=data,
    )

@step
async def create_plan(analysis: TaskAnalysis, ctx: ExecutionContext) -> list[str]:
    """Create detailed execution steps based on analysis."""
    result = await ctx.llm(
        name="plan_creator",
        instructions="Create a detailed step-by-step plan based on the analysis",
        input_schema=TaskAnalysis,
        output_schema=list[str],
        input=analysis,
    )
    return result

# Build workflow
workflow = (
    Workflow(
        id="task-planner",
        input_model=TaskInput,
        output_model=TaskPlan,
    )
    .then(analyze_task)
    .then(create_plan)
    .map(lambda results: TaskPlan(
        analysis=results[0],
        steps=results[1]
    ))
    .commit()
)

# Create and use agent
agent = Agent(
    name="TaskPlannerAgent",
    description="Analyzes tasks and creates execution plans",
    flow=workflow,
)

# Execute workflow
result = agent.process("Build a web scraper for product prices")
print(f"Complexity: {result.analysis.complexity}")
print(f"Steps: {result.steps}")
```

## 4. Advanced Features

### 4.1 Parallel Processing

Execute multiple steps simultaneously for better performance.

```python
# Parallel step execution
workflow = (
    Workflow(id="parallel-analysis", input_model=DocumentInput, output_model=Analysis)
    .parallel([
        sentiment_analysis_step,
        topic_extraction_step,
        summary_generation_step,
    ])
    .map(lambda results: Analysis(
        sentiment=results[0],
        topics=results[1], 
        summary=results[2]
    ))
    .commit()
)

# Process collections in parallel
workflow = (
    Workflow(id="batch-processor", input_model=BatchInput, output_model=BatchOutput)
    .foreach(
        process_item_step,
        concurrency=5,  # Process 5 items simultaneously
        map_func=lambda data: data.items  # Extract items to process
    )
    .commit()
)
```

### 4.2 Conditional Branching

Route execution based on data conditions.

```python
# Define conditions
def is_urgent(data):
    return data.priority == "urgent"

def is_normal(data):
    return data.priority == "normal"

# Conditional workflow
workflow = (
    Workflow(id="priority-handler", input_model=TaskInput, output_model=TaskOutput)
    .branch([
        (is_urgent, urgent_processing_step),
        (is_normal, normal_processing_step),
    ])
    .then(finalize_step)
    .commit()
)
```

### 4.3 Error Handling

Configure robust error handling and retry strategies.

```python
# Step-level error handling
@step(
    retry={"attempts": 3, "backoff_ms": 1000},  # Retry with exponential backoff
    timeout_ms=30000,  # 30 second timeout
    on_error="continue"  # Options: "fail", "skip", "continue"
)
async def robust_step(data: InputModel, ctx: ExecutionContext) -> OutputModel:
    """Step with comprehensive error handling."""
    try:
        result = await ctx.llm(name="processor", ...)
        return result
    except Exception as e:
        ctx.set_state("error", str(e))
        # Return fallback result
        return OutputModel(status="error", message=str(e))
```

### 4.4 State Management

Share data between steps using workflow state.

```python
@step
async def load_data(input_data: InputModel, ctx: ExecutionContext) -> ProcessedData:
    """Load and cache data for subsequent steps."""
    data = await load_from_source(input_data.source)
    
    # Store in state for other steps
    ctx.set_state("raw_data", data)
    ctx.set_state("timestamp", datetime.now())
    
    return ProcessedData(data=data)

@step  
async def analyze_cached_data(processed: ProcessedData, ctx: ExecutionContext) -> Analysis:
    """Analyze using cached data."""
    # Retrieve from state
    raw_data = ctx.get_state("raw_data")
    timestamp = ctx.get_state("timestamp")
    
    # Use cached data in analysis
    return await ctx.llm(
        name="analyzer",
        instructions=f"Analyze data loaded at {timestamp}",
        input={"processed": processed, "raw": raw_data},
        output_schema=Analysis,
    )
```

## 5. Model Selection

Override models at the agent or step level for optimal performance.

```python
# Agent-level default
agent = Agent(
    name="FlowAgent",
    flow=workflow,
    model="openai/gpt-4o-mini"  # Default for all steps
)

# Step-level override
@step
async def complex_reasoning(data: InputModel, ctx: ExecutionContext) -> OutputModel:
    """Use a more powerful model for complex reasoning."""
    return await ctx.llm(
        name="reasoner",
        instructions="Perform complex multi-step reasoning",
        input_schema=InputModel,
        output_schema=OutputModel,
        input=data,
        model="anthropic/claude-3.5-sonnet"  # Override agent default
    )
```

## 6. Event Callbacks

Monitor workflow execution in real-time.

```python
def workflow_handler(event_type: str, data: dict):
    """Handle workflow events for UI integration."""
    if event_type == "workflow_start":
        print(f"🚀 Starting workflow: {data['workflow_id']}")
    elif event_type == "workflow_step_start":
        print(f"📝 Step: {data['step_id']}")
    elif event_type == "workflow_model_call_start":
        print(f"🤖 AI Call: {data['call_name']}")
    elif event_type == "workflow_step_complete":
        duration = data.get('duration_ms', 0)
        print(f"✅ Completed {data['step_id']} in {duration}ms")
    elif event_type == "workflow_complete":
        print(f"🎉 Workflow completed successfully")

agent = Agent(
    name="MonitoredAgent",
    flow=workflow,
    callback=workflow_handler,
    verbose=False  # Let events handle output
)
```

## 7. Best Practices

### Design Principles
- **Single Responsibility**: Each step should have one clear purpose
- **Type Safety**: Always use Pydantic models for inputs and outputs  
- **Error Handling**: Plan for failures with retries and fallbacks
- **State Management**: Use workflow state sparingly for shared data

### Performance Tips
- **Parallel Processing**: Use `.parallel()` for independent operations
- **Model Selection**: Use faster models for simple tasks, powerful models for complex reasoning
- **Batching**: Process collections efficiently with `.foreach()`
- **Caching**: Store expensive computations in workflow state

### Testing Workflows
```python
# Test individual steps
test_input = UserInput(text="Test message")
result = await analyze_text(test_input, mock_context)

# Test complete workflows
agent = Agent(name="TestAgent", flow=workflow)
result = agent.process("Test input")
assert result.status == "success"
```

### Common Patterns

**Data Pipeline:**
```python
workflow = (
    Workflow(id="data-pipeline", input_model=RawData, output_model=ProcessedData)
    .then(extract_step)
    .then(transform_step) 
    .then(validate_step)
    .then(load_step)
    .commit()
)
```

**Analysis & Action:**
```python
workflow = (
    Workflow(id="analyze-act", input_model=Input, output_model=Result)
    .then(analyze_step)
    .branch([
        (needs_action, action_step),
        (needs_escalation, escalation_step),
    ])
    .then(finalize_step)
    .commit()
)
```

**Fan-out/Fan-in:**
```python
workflow = (
    Workflow(id="fan-out-in", input_model=Input, output_model=CombinedResult)
    .then(prepare_step)
    .parallel([
        process_option_a,
        process_option_b,
        process_option_c,
    ])
    .then(combine_results)
    .commit()
)
```

---

For complete examples, see the `examples/` directory in the repository.