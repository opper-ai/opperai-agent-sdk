# Tools Mode Documentation

Build dynamic AI agents that intelligently select and execute Python functions based on context and goals.

## Table of Contents

1. [Introduction](#1-introduction)
2. [Core Concepts](#2-core-concepts)
   - [Tools](#21-tools)
   - [Agent Reasoning](#22-agent-reasoning)
   - [Tool Selection](#23-tool-selection)
3. [Getting Started](#3-getting-started)
4. [Advanced Features](#4-advanced-features)
   - [Custom Output Schemas](#41-custom-output-schemas)
   - [Tool Error Handling](#42-tool-error-handling)
   - [Complex Tool Chains](#43-complex-tool-chains)
5. [MCP Integration](#5-mcp-integration)
6. [Best Practices](#6-best-practices)

## 1. Introduction

Tools mode enables agents to dynamically reason about problems and select appropriate tools to solve them. Perfect for:

- **Exploratory tasks**: When you don't know the exact sequence of steps needed
- **Complex reasoning**: Multi-step problems requiring adaptive decision-making
- **Interactive workflows**: Agents that respond to dynamic user requests
- **General-purpose assistants**: Versatile agents with broad capabilities

## 2. Core Concepts

### 2.1 Tools

Tools are Python functions decorated with `@tool` that agents can discover and execute. Each tool should have a single, clear purpose.

```python
from opper_agent import tool

@tool
def web_search(query: str, limit: int = 5) -> list[dict]:
    """Search the web for information about a topic.
    
    Args:
        query: Search terms to look for
        limit: Maximum number of results to return
        
    Returns:
        List of search results with title, url, and snippet
    """
    # Your implementation here
    results = perform_web_search(query, limit)
    return results

@tool
def calculate(expression: str) -> float:
    """Evaluate a mathematical expression safely.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 2 * 3")
        
    Returns:
        Numerical result of the expression
    """
    # Use safe evaluation in production
    return eval(expression)

@tool
def send_email(to: str, subject: str, body: str) -> bool:
    """Send an email to a recipient.
    
    Args:
        to: Email address of recipient
        subject: Email subject line
        body: Email content
        
    Returns:
        True if email sent successfully, False otherwise
    """
    try:
        # Your email implementation
        send_email_implementation(to, subject, body)
        return True
    except Exception:
        return False
```

**Tool Requirements:**
- **Clear docstrings**: Help the agent understand when and how to use the tool
- **Type hints**: Provide clear input/output types
- **Single purpose**: Each tool should do one thing well
- **Error handling**: Handle failures gracefully and return meaningful results

### 2.2 Agent Reasoning

Tools mode agents follow a think-act cycle:

1. **Think**: Analyze the goal and current situation
2. **Plan**: Decide which tool(s) to use and in what order
3. **Act**: Execute the selected tool with appropriate parameters
4. **Observe**: Process the tool results
5. **Repeat**: Continue until the goal is achieved

```python
from opper_agent import Agent

agent = Agent(
    name="ResearchAssistant",
    description="Helps users research topics by searching and analyzing information",
    tools=[web_search, calculate, send_email],
    model="anthropic/claude-3.5-sonnet"  # Optional: specify reasoning model
)

# Agent will automatically plan and execute tool sequence
result = agent.process("Research the population of major cities in Japan and calculate the average")
```

### 2.3 Tool Selection

Agents intelligently select tools based on:

- **Goal analysis**: Understanding what the user wants to achieve
- **Context awareness**: Considering previous actions and their results  
- **Tool capabilities**: Matching required functionality to available tools
- **Parameter requirements**: Ensuring tools can be called with available data

## 3. Getting Started

### Simple Assistant Example

```python
from opper_agent import Agent, tool
import requests
import json

@tool
def get_weather(city: str) -> dict:
    """Get current weather information for a city.
    
    Args:
        city: Name of the city to get weather for
        
    Returns:
        Weather information including temperature, conditions, humidity
    """
    # Simplified weather API call
    api_key = "your_api_key"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    
    try:
        response = requests.get(url)
        data = response.json()
        return {
            "city": city,
            "temperature": data["main"]["temp"],
            "conditions": data["weather"][0]["description"],
            "humidity": data["main"]["humidity"]
        }
    except Exception as e:
        return {"error": f"Could not get weather for {city}: {str(e)}"}

@tool
def unit_converter(value: float, from_unit: str, to_unit: str) -> dict:
    """Convert between different units of measurement.
    
    Args:
        value: Numerical value to convert
        from_unit: Source unit (celsius, fahrenheit, km, miles, etc.)
        to_unit: Target unit to convert to
        
    Returns:
        Conversion result with original and converted values
    """
    conversions = {
        ("celsius", "fahrenheit"): lambda x: (x * 9/5) + 32,
        ("fahrenheit", "celsius"): lambda x: (x - 32) * 5/9,
        ("km", "miles"): lambda x: x * 0.621371,
        ("miles", "km"): lambda x: x / 0.621371,
    }
    
    key = (from_unit.lower(), to_unit.lower())
    if key in conversions:
        converted = conversions[key](value)
        return {
            "original": f"{value} {from_unit}",
            "converted": f"{converted:.2f} {to_unit}",
            "value": converted
        }
    else:
        return {"error": f"Conversion from {from_unit} to {to_unit} not supported"}

# Create agent with tools
agent = Agent(
    name="WeatherAssistant",
    description="Provides weather information and unit conversions",
    tools=[get_weather, unit_converter]
)

# Agent will use tools as needed
result = agent.process("What's the weather in Tokyo? Convert the temperature to Fahrenheit.")
print(result)
```

## 4. Advanced Features

### 4.1 Custom Output Schemas

Structure the agent's final response using Pydantic models.

```python
from pydantic import BaseModel, Field
from typing import List

class ResearchResult(BaseModel):
    summary: str = Field(description="Summary of findings")
    sources: List[str] = Field(description="URLs of sources used")
    key_facts: List[str] = Field(description="Important facts discovered")
    confidence: float = Field(description="Confidence score 0-1")

agent = Agent(
    name="ResearchAgent",
    description="Conducts thorough research on topics",
    tools=[web_search, calculate],
    output_schema=ResearchResult  # Structured output
)

# Agent will format results according to schema
result = agent.process("Research renewable energy adoption rates globally")
print(f"Summary: {result.summary}")
print(f"Sources: {result.sources}")
```

### 4.2 Tool Error Handling

Build robust tools that handle errors gracefully.

```python
@tool
def safe_api_call(endpoint: str, params: dict = None) -> dict:
    """Make a safe API call with error handling.
    
    Args:
        endpoint: API endpoint URL
        params: Optional parameters to include
        
    Returns:
        API response data or error information
    """
    try:
        response = requests.get(endpoint, params=params, timeout=10)
        response.raise_for_status()
        return {
            "success": True,
            "data": response.json(),
            "status_code": response.status_code
        }
    except requests.exceptions.Timeout:
        return {"success": False, "error": "API call timed out"}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"API error: {str(e)}"}
    except json.JSONDecodeError:
        return {"success": False, "error": "Invalid JSON response"}
```

### 4.3 Complex Tool Chains

Tools can be designed to work together for complex workflows.

```python
@tool
def analyze_sentiment(text: str) -> dict:
    """Analyze the sentiment of text."""
    # Implementation here
    return {"sentiment": "positive", "confidence": 0.85}

@tool
def generate_response(sentiment: str, context: str) -> str:
    """Generate appropriate response based on sentiment."""
    templates = {
        "positive": "Thank you for your positive feedback about {context}!",
        "negative": "I understand your concerns about {context}. Let me help.",
        "neutral": "I appreciate your input about {context}."
    }
    return templates.get(sentiment, "Thank you for your feedback.").format(context=context)

@tool
def log_interaction(sentiment: str, response: str) -> bool:
    """Log customer interaction for analysis."""
    # Implementation here
    return True

# Agent will chain these tools together automatically
agent = Agent(
    name="CustomerService",
    description="Analyzes customer feedback and generates appropriate responses",
    tools=[analyze_sentiment, generate_response, log_interaction]
)
```

## 5. MCP Integration

Connect to external systems using the Model Context Protocol (MCP) for expanded tool capabilities.

### Quick MCP Setup

```python
from opper_agent import Agent, create_mcp_tools
from opper_agent.mcp_client import MCPServerConfig

# Connect to Opper documentation server
opper_docs_server = MCPServerConfig(
    name="opper",
    url="https://docs.opper.ai/mcp",
    transport="http-sse",
    enabled=True
)

# Create MCP tools
mcp_tools = create_mcp_tools([opper_docs_server])

# Combine with custom tools
agent = Agent(
    name="OpperExpert",
    description="Expert assistant with access to live Opper documentation",
    tools=[*mcp_tools(), web_search, calculate],  # Mix MCP and custom tools
    model="anthropic/claude-3.5-sonnet"
)

# Agent can now access live documentation
result = agent.process("How do I implement retry logic in Opper workflows?")
```

### Custom MCP Servers

```python
# Connect to custom subprocess-based MCP server
custom_server = MCPServerConfig(
    name="database_tools",
    command="node",
    args=["database-mcp-server.js"],
    timeout=30.0
)

db_tools = create_mcp_tools([custom_server])
agent = Agent(name="DataAgent", tools=db_tools())
```

**Transport Support:**
- **HTTP-SSE**: Modern web-based servers (recommended)
- **stdio**: Traditional subprocess-based servers

## 6. Best Practices

### Tool Design
- **Clear Purpose**: Each tool should have one clear, specific function
- **Comprehensive Docstrings**: Help the agent understand when and how to use tools
- **Error Handling**: Return meaningful error information instead of throwing exceptions
- **Type Safety**: Use proper type hints for all parameters and return values

### Agent Configuration
- **Descriptive Names**: Use clear, specific agent names and descriptions
- **Model Selection**: Choose appropriate models for your reasoning complexity
- **Tool Organization**: Group related tools together, avoid tool overload

### Performance Tips
- **Tool Efficiency**: Optimize frequently-used tools for speed
- **Error Recovery**: Design tools to fail gracefully and provide alternatives
- **Caching**: Cache expensive operations when possible
- **Timeouts**: Set appropriate timeouts for external API calls

### Testing Tools
```python
# Test tools independently
def test_weather_tool():
    result = get_weather("London")
    assert "temperature" in result
    assert "city" in result

# Test agent behavior
def test_agent_reasoning():
    agent = Agent(name="TestAgent", tools=[get_weather])
    result = agent.process("What's the weather like?")
    # Agent should ask for a city or use a default
    assert "city" in result.lower() or "location" in result.lower()
```

### Common Patterns

**Information Gathering:**
```python
tools = [web_search, summarize_text, fact_check]
agent = Agent(name="Researcher", tools=tools)
```

**Data Processing:**
```python
tools = [load_data, clean_data, analyze_data, generate_report]
agent = Agent(name="DataProcessor", tools=tools)
```

**Task Automation:**
```python
tools = [read_email, categorize_content, send_notification, update_database]
agent = Agent(name="Automator", tools=tools)
```

---

Tools mode provides powerful flexibility for building adaptive AI agents. Start with simple tools and gradually add complexity as your use cases evolve.