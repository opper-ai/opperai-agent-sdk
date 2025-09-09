#!/usr/bin/env python3
"""
Example demonstrating Agent event callback system for UI progress tracking.
Shows comprehensive events from both tools mode and flow mode.
"""

import asyncio
import os
import sys
from typing import Any, Dict

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from pydantic import BaseModel, Field
from opper_agent import Agent, tool
from opper_agent.workflows import Workflow, create_step
from opperai import Opper


# === EVENT TRACKING ===

class EventTracker:
    """Tracks and displays events from agent execution."""
    
    def __init__(self):
        self.events = []
        self.event_counts = {}
    
    def event_callback(self, event_type: str, data: Dict[str, Any]):
        """Callback function to receive agent events."""
        # Track event
        self.events.append({
            "type": event_type,
            "data": data,
            "timestamp": data.get("timestamp", 0)
        })
        
        # Count events
        self.event_counts[event_type] = self.event_counts.get(event_type, 0) + 1
        
        # Display event (for UI, you'd update your interface here)
        self._display_event(event_type, data)
    
    def _display_event(self, event_type: str, data: Dict[str, Any]):
        """Display event in a user-friendly format."""
        timestamp = data.get("timestamp", 0)
        
        if event_type == "goal_start":
            print(f"🎯 Starting: {data.get('goal', 'Unknown goal')}")
            print(f"   Mode: {data.get('mode', 'unknown')}")
            if data.get('available_tools'):
                print(f"   Tools: {', '.join(data['available_tools'])}")
        
        elif event_type == "thought_created":
            print(f"🧠 Thinking (iteration {data.get('iteration', '?')})")
            thought = data.get('thought', {})
            if thought.get('reasoning'):
                print(f"   Reasoning: {thought['reasoning'][:80]}...")
            print(f"   Next action: {thought.get('tool_name', 'unknown')}")
        
        elif event_type == "action_executed":
            result = data.get('action_result', {})
            success = "✅" if result.get('success') else "❌"
            print(f"⚡ Action: {result.get('tool_name', 'unknown')} {success}")
            if result.get('result'):
                print(f"   Result: {str(result['result'])[:80]}...")
        
        elif event_type.startswith("workflow_"):
            workflow_event = event_type.replace("workflow_", "")
            self._display_workflow_event(workflow_event, data)
        
        elif event_type == "goal_completed":
            achieved = "✅" if data.get('achieved') else "❌"
            print(f"🏁 Goal completed {achieved}")
            if data.get('error'):
                print(f"   Error: {data['error']}")
    
    def _display_workflow_event(self, event_type: str, data: Dict[str, Any]):
        """Display workflow-specific events."""
        if event_type == "workflow_start":
            print(f"🔄 Workflow started: {data.get('workflow_id', 'unknown')}")
        
        elif event_type == "step_start":
            print(f"📝 Step started: {data.get('step_id', 'unknown')}")
        
        elif event_type == "step_success":
            print(f"✅ Step completed: {data.get('step_id', 'unknown')}")
        
        elif event_type == "step_error":
            print(f"❌ Step failed: {data.get('step_id', 'unknown')}")
            print(f"   Error: {data.get('error', 'Unknown error')}")
        
        elif event_type == "step_retry":
            attempt = data.get('attempt', '?')
            max_attempts = data.get('max_attempts', '?')
            print(f"🔄 Step retry {attempt}/{max_attempts}: {data.get('step_id', 'unknown')}")
        
        elif event_type == "model_call_start":
            print(f"🤖 AI call started: {data.get('call_name', 'unknown')}")
        
        elif event_type == "model_call_success":
            print(f"✅ AI call completed: {data.get('call_name', 'unknown')}")
        
        elif event_type == "model_call_error":
            print(f"❌ AI call failed: {data.get('call_name', 'unknown')}")
            print(f"   Error: {data.get('error', 'Unknown error')}")
        
        elif event_type == "workflow_success":
            print(f"🎉 Workflow completed successfully")
        
        elif event_type == "workflow_error":
            print(f"💥 Workflow failed: {data.get('error', 'Unknown error')}")
    
    def get_summary(self):
        """Get a summary of all events."""
        total_events = len(self.events)
        print(f"\n📊 Event Summary ({total_events} total events):")
        for event_type, count in sorted(self.event_counts.items()):
            print(f"   {event_type}: {count}")


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
    """Example using Agent in tools mode with event tracking."""
    print("=== TOOLS MODE WITH EVENT TRACKING ===")
    
    # Create event tracker
    tracker = EventTracker()
    
    # Create agent with tools and event callback
    agent = Agent(
        name="MathAgent",
        description="An agent that helps with basic geometry calculations",
        tools=[calculate_area, calculate_perimeter],
        callback=tracker.event_callback,  # This is where events are sent
        verbose=False,  # Turn off verbose to see just events
        opper_api_key=os.getenv("OPPER_API_KEY")
    )
    
    # Process a goal
    result = agent.process("Calculate the area and perimeter of a rectangle that is 5 meters long and 3 meters wide")
    
    # Show event summary
    tracker.get_summary()
    return result, tracker


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
    """Example using Agent in flow mode with event tracking."""
    print("\n=== FLOW MODE WITH EVENT TRACKING ===")
    
    # Create event tracker
    tracker = EventTracker()
    
    # Create workflow
    workflow = Workflow(
        id="rectangle-calculator",
        input_model=RectangleInput,
        output_model=CalculationResult
    ).then(calculate_rectangle).commit()
    
    # Create agent with flow and event callback
    agent = Agent(
        name="FlowMathAgent", 
        description="An agent that uses workflows for geometry calculations",
        flow=workflow,
        callback=tracker.event_callback,  # This is where events are sent
        verbose=False,  # Turn off verbose to see just events
        opper_api_key=os.getenv("OPPER_API_KEY")
    )
    
    # Process a goal
    result = agent.process("Calculate the area and perimeter of a rectangle that is 5 meters long and 3 meters wide")
    
    # Show event summary
    tracker.get_summary()
    return result, tracker


def main():
    """Main function demonstrating event tracking in both modes."""
    api_key = os.getenv("OPPER_API_KEY")
    if not api_key:
        print("⚠️  OPPER_API_KEY environment variable is not set!")
        print("Please set your Opper API key to run this example.")
        return
    
    try:
        # Run tools mode example
        tools_result, tools_tracker = tools_mode_example()
        
        # Run flow mode example  
        flow_result, flow_tracker = flow_mode_example()
        
        print("\n=== EVENT COMPARISON ===")
        print(f"Tools mode events: {len(tools_tracker.events)}")
        print(f"Flow mode events: {len(flow_tracker.events)}")
        
        print("\n✅ Both modes completed with comprehensive event tracking!")
        print("\n💡 In a real UI application, you would:")
        print("   - Update progress bars based on step events")
        print("   - Show real-time status based on workflow events")
        print("   - Display errors and retries for better UX")
        print("   - Track performance metrics from event timestamps")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
