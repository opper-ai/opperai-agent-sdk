#!/usr/bin/env python3
"""
Example demonstrating Agent in both tools mode and flow mode.
This shows how the same agent class can operate in two different ways.
"""

import asyncio
import os
import sys
from typing import Any

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from pydantic import BaseModel, Field
from opper_agent import Agent, tool
from opper_agent.workflows import Workflow, create_step
from opperai import Opper


# === TOOLS MODE EXAMPLE ===

@tool
def calculate_area(length: float, width: float) -> float:
    """Calculate the area of a rectangle."""
    return length * width

@tool  
def calculate_perimeter(length: float, width: float) -> float:
    """Calculate the perimeter of a rectangle."""
    return 2 * (length + width)

def tools_mode_example():
    """Example using Agent in tools mode."""
    print("=== TOOLS MODE EXAMPLE ===")
    
    # Create agent with tools
    agent = Agent(
        name="MathAgent",
        description="An agent that helps with basic geometry calculations",
        tools=[calculate_area, calculate_perimeter],
        verbose=True,
        opper_api_key=os.getenv("OPPER_API_KEY")
    )
    
    print(f"Agent: {agent}")
    print(f"Mode: {agent.mode}")
    print(f"Tools: {agent.get_tools_summary()}")
    
    # Process a goal
    result = agent.process("Calculate the area and perimeter of a rectangle that is 5 meters long and 3 meters wide")
    print(f"Result: {result}")
    return result


# === FLOW MODE EXAMPLE ===

class RectangleInput(BaseModel):
    goal: str = Field(description="The calculation goal")
    length: float = Field(description="Length of the rectangle", default=5.0)
    width: float = Field(description="Width of the rectangle", default=3.0)

class CalculationResult(BaseModel):
    thoughts: str = Field(description="Reasoning for the calculations")
    area: float = Field(description="Calculated area")
    perimeter: float = Field(description="Calculated perimeter")
    summary: str = Field(description="Summary of the calculations")

async def calculate_step(ctx):
    """Step that performs rectangle calculations."""
    rect: RectangleInput = ctx.input_data
    
    # Use the AI to structure the calculation results
    result = await ctx.call_model(
        name="rectangle_calculator",
        instructions=(
            "Calculate the area and perimeter of a rectangle with the given dimensions. "
            "Area = length × width, Perimeter = 2 × (length + width). "
            "Provide the calculations and a summary."
        ),
        input_schema=RectangleInput,
        output_schema=CalculationResult,
        input_obj=rect
    )
    return result

calculate_rectangle = create_step(
    id="calculate_rectangle",
    input_model=RectangleInput,
    output_model=CalculationResult,
    run=calculate_step
)

def flow_mode_example():
    """Example using Agent in flow mode."""
    print("\n=== FLOW MODE EXAMPLE ===")
    
    # Create workflow
    workflow = Workflow(
        id="rectangle-calculator",
        input_model=RectangleInput,
        output_model=CalculationResult
    ).then(calculate_rectangle).commit()
    
    # Create agent with flow
    agent = Agent(
        name="FlowMathAgent", 
        description="An agent that uses workflows for geometry calculations",
        flow=workflow,
        verbose=True,
        opper_api_key=os.getenv("OPPER_API_KEY")
    )
    
    print(f"Agent: {agent}")
    print(f"Mode: {agent.mode}")
    print(f"Flow: {agent.get_tools_summary()}")
    
    # Process a goal (will be converted to RectangleInput)
    result = agent.process("Calculate the area and perimeter of a rectangle that is 5 meters long and 3 meters wide")
    print(f"Result: {result}")
    return result


def main():
    """Main function that demonstrates both modes."""
    api_key = os.getenv("OPPER_API_KEY")
    if not api_key:
        print("⚠️  OPPER_API_KEY environment variable is not set!")
        print("Please set your Opper API key to run this example.")
        return
    
    try:
        # Run tools mode example
        tools_result = tools_mode_example()
        
        # Run flow mode example  
        flow_result = flow_mode_example()
        
        print("\n=== COMPARISON ===")
        print("Tools mode result type:", type(tools_result))
        print("Flow mode result type:", type(flow_result))
        
        print("\n✅ Both modes completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
