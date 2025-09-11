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
    verbose=True,
)

# Process a goal - agent will think and act iteratively
result = agent.process("Calculate the area and perimeter of a 5x3 rectangle")
print(result)
