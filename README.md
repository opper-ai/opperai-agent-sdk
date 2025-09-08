# Opper Agent SDK

A powerful Python SDK for building AI agents with structured workflows using [Opper](https://opper.ai). This SDK provides a framework for creating complex, multi-step AI workflows with proper error handling, parallel processing, and comprehensive tracing.

## 🚀 Features

- **Structured Workflows**: Build complex AI pipelines with clear step definitions
- **Parallel Processing**: Execute steps concurrently with `foreach` and `parallel` operations
- **Error Handling**: Robust error handling with retry mechanisms and fallback strategies
- **Tracing & Monitoring**: Full observability with Opper's tracing system
- **Type Safety**: Full Pydantic model validation throughout workflows
- **Flexible Control Flow**: Support for branching, loops, and conditional execution

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

### 2. Create your first agent

```python
import asyncio
from opper_agent import Agent, Workflow, create_step, InMemoryStorage
from opperai import Opper
from pydantic import BaseModel, Field

class UserMessage(BaseModel):
    text: str = Field(description="The user's message")

class Response(BaseModel):
    reply: str = Field(description="The agent's response")

async def process_message_step(ctx):
    """Process a user message and generate a response"""
    message = ctx.input_data
    
    result = await ctx.call_model(
        name="simple_chat",
        instructions="You are a helpful assistant. Respond to the user's message.",
        input_schema=UserMessage,
        output_schema=Response,
        input_obj=message,
    )
    
    return Response(reply=result.get('reply', ''))

# Create the step
process_message = create_step(
    id="process_message",
    input_model=UserMessage,
    output_model=Response,
    run=process_message_step,
)

# Create the agent
simple_agent = Agent(
    name="Simple Chat Agent",
    instructions="You are a helpful assistant",
    flow=Workflow(id="simple-chat", input_model=UserMessage, output_model=Response)
        .then(process_message)
        .commit(),
)

# Use the agent
async def main():
    opper = Opper(http_bearer="your-api-key")
    storage = InMemoryStorage()
    
    run = simple_agent.create_run(opper=opper, storage=storage, tools={})
    result = await run.start(input_data=UserMessage(text="Hello!"))
    print(result.reply)

if __name__ == "__main__":
    asyncio.run(main())
```

## 📚 Examples

This repository includes several comprehensive examples:

### 🤖 Chatbot Agent (`examples/chatbot_agent.py`)
A conversational agent that demonstrates:
- Intent detection and classification
- Branching workflows based on user input
- Handling different conversation types (small talk, technical questions, etc.)

```bash
python examples/chatbot_agent.py
```

### 👨‍🍳 Chef Agent (`examples/chef_agent.py`)
A cooking assistant that shows sequential workflow processing:
- Generates meal ideas from available ingredients
- Creates detailed recipes with step-by-step instructions
- Adds precise quantities and measurements
- Generates organized shopping lists

```bash
python examples/chef_agent.py
```

### 📋 RFP Agent (`examples/rfp_agent.py`)
A sophisticated RFP (Request for Proposal) processor featuring:
- **Knowledge base integration** with company information
- **Parallel question processing** using `foreach` workflows
- **Professional response generation** with confidence scoring
- **Advanced workflow patterns** demonstrating the full power of the SDK

```bash
python examples/rfp_agent.py
```

## 🏗️ Core Concepts

### Workflows
Define sequences of steps with various control flow patterns:

```python
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

### Steps
Building blocks of workflows with full context access:

```python
async def my_step_function(ctx):
    """Step function with access to context"""
    input_data = ctx.input_data
    
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

### Agents
Combine workflows with configuration and instructions:

```python
agent = Agent(
    name="My Agent",
    instructions="System instructions for the agent",
    flow=workflow,
)
```

## 🔄 Advanced Features

### Parallel Processing with `foreach`
Process collections of items concurrently:

```python
.foreach(
    process_item_step,
    concurrency=3,  # Process 3 items simultaneously
    map_func=lambda data: data.items  # Extract items to process
)
```

### Conditional Branching
Execute different paths based on conditions:

```python
.branch([
    (lambda data: data.type == "urgent", urgent_handler),
    (lambda data: data.type == "normal", normal_handler),
])
```

### Error Handling
Configure robust error handling with retries:

```python
create_step(
    id="robust_step",
    retry={"attempts": 3, "backoff_ms": 1000},
    on_error="continue",  # "fail", "skip", or "continue"
    run=step_function,
)
```

### Knowledge Base Integration
Seamlessly integrate with Opper's knowledge bases:

```python
# Setup knowledge base
knowledge_base = opper.knowledge.create(name="my-kb")
opper.knowledge.add(
    knowledge_base_id=kb.id, 
    content="Your knowledge content", 
    metadata={"category": "docs"}
)

# Query in workflow steps
results = ctx.opper.knowledge.query(
    knowledge_base_id=kb_id, 
    query="What information do I need?",
    top_k=5
)
```

## 📊 Monitoring and Tracing

All workflow executions are automatically traced in Opper with:
- **Workflow-level spans**: Track entire workflow execution
- **Step-level performance metrics**: Monitor individual step performance
- **Model call logging**: Detailed logging of all AI interactions
- **Error tracking**: Automatic error capture and reporting

View your traces in the [Opper Dashboard](https://platform.opper.ai).

## 🧪 Testing

Run the examples to verify your setup:

```bash
# Test the chatbot agent
python examples/chatbot_agent.py

# Test the chef agent  
python examples/chef_agent.py

# Test the RFP agent (requires OPPER_API_KEY)
python examples/rfp_agent.py
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

- [ ] Additional workflow control structures (while loops, try/catch)
- [ ] Built-in tool integrations
- [ ] Workflow visualization tools
- [ ] Performance optimization features
- [ ] More example agents and use cases

---

Built with ❤️ using [Opper](https://opper.ai)
