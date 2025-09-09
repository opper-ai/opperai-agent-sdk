#!/usr/bin/env python3
"""
Example demonstrating the @step decorator for defining workflow steps.
Shows how to use the decorator with type hints for automatic model extraction.
"""

import asyncio
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from pydantic import BaseModel, Field
from opper_agent import Agent, step, Workflow, StepContext
from opperai import Opper


# Define input and output models
class TextInput(BaseModel):
    text: str = Field(description="Input text to process")
    operation: str = Field(description="Operation to perform", default="uppercase")

class TextOutput(BaseModel):
    thoughts: str = Field(description="Reasoning about the operation")
    result: str = Field(description="Processed text result")
    operation_used: str = Field(description="The operation that was performed")


# Define steps using the @step decorator
@step
async def process_text(ctx: StepContext[TextInput, TextOutput]) -> TextOutput:
    """Process text based on the specified operation."""
    data = ctx.input_data
    
    # Use AI to process the text
    result = await ctx.call_model(
        name="text_processor",
        instructions=f"Process the text using the '{data.operation}' operation. Explain your reasoning.",
        input_schema=TextInput,
        output_schema=TextOutput,
        input_obj=data
    )
    
    return result

@step(id="validate_result", retry={"attempts": 2})
async def validate_output(ctx: StepContext[TextOutput, TextOutput]) -> TextOutput:
    """Validate and potentially enhance the text processing result."""
    data = ctx.input_data
    
    # Simple validation - in practice this could be more sophisticated
    if len(data.result) < len(data.thoughts):
        # Result seems too short, enhance it
        enhanced = await ctx.call_model(
            name="text_enhancer",
            instructions="The result seems brief. Enhance it while maintaining the original operation intent.",
            input_schema=TextOutput,
            output_schema=TextOutput,
            input_obj=data
        )
        return enhanced
    
    return data

@step(description="Final formatting step", on_error="continue")
async def format_result(ctx: StepContext[TextOutput, TextOutput]) -> TextOutput:
    """Apply final formatting to the result."""
    data = ctx.input_data
    
    # Add some formatting
    formatted_result = f"✨ {data.result} ✨"
    
    return TextOutput(
        thoughts=data.thoughts + " | Applied final formatting.",
        result=formatted_result,
        operation_used=data.operation_used
    )


def main():
    """Main function demonstrating the @step decorator."""
    api_key = os.getenv("OPPER_API_KEY")
    if not api_key:
        print("⚠️  OPPER_API_KEY environment variable is not set!")
        print("This example will show the decorator usage but won't execute the workflow.")
        api_key = "dummy-key"
    
    print("=== @step Decorator Example ===")
    
    # Create workflow using decorator-defined steps
    workflow = (Workflow(id="text-processor", input_model=TextInput, output_model=TextOutput)
        .then(process_text)
        .then(validate_output)
        .then(format_result)
        .commit())
    
    # Create agent with workflow
    agent = Agent(
        name="TextProcessor",
        description="An agent that processes text using decorated steps",
        flow=workflow,
        verbose=True,
        opper_api_key=api_key
    )
    
    print(f"Agent: {agent}")
    print(f"Workflow steps: {len(workflow.pipeline)}")
    
    # Show step information
    print("\n📋 Step Information:")
    for i, (kind, payload) in enumerate(workflow.pipeline):
        if kind == "then" and hasattr(payload, 'defn'):
            step_def = payload.defn
            print(f"  {i+1}. {step_def.id}")
            print(f"     Description: {step_def.description}")
            print(f"     Input: {step_def.input_model.__name__}")
            print(f"     Output: {step_def.output_model.__name__}")
            print(f"     Error handling: {step_def.on_error}")
            if step_def.retry.get("attempts", 1) > 1:
                print(f"     Retry attempts: {step_def.retry['attempts']}")
    
    if api_key != "dummy-key":
        try:
            # Process some text
            result = agent.process("Make this text EXCITING and BOLD: Hello world")
            print(f"\n✅ Result: {result}")
        except Exception as e:
            print(f"❌ Execution failed: {e}")
    else:
        print("\n💡 To run the workflow, set OPPER_API_KEY environment variable")
    
    print("\n=== Decorator Benefits ===")
    print("✅ Automatic type extraction from function annotations")
    print("✅ Clean, readable step definitions")
    print("✅ Same configuration options as create_step()")
    print("✅ IDE support with type hints")
    print("✅ Docstring becomes step description automatically")


if __name__ == "__main__":
    main()
