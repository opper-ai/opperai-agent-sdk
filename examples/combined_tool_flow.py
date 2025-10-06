from opper_agent_old import Agent, step, Workflow, ExecutionContext, tool
from pydantic import BaseModel, Field


@tool
def tool1(place: str, date: str) -> str:
    return "The weather in " + place + " on " + date + " is sunny"


tools_agent = Agent(
    name="WeatherAgent",
    description="Given a place and a date returns the expected weather",
    tools=[tool1],  # You implement tools
)

weather = tools_agent.process("What's the weather in Paris today?")
print("Weather agent result:")
print(weather)


class TravelRequest(BaseModel):
    request: str = Field(description="The user's travel request")


class TravelDetails(BaseModel):
    city: str = Field(description="The destination city")
    days_of_stay: int = Field(description="Number of days the user will stay")


class TravelWithWeather(BaseModel):
    city: str = Field(description="The destination city")
    days_of_stay: int = Field(description="Number of days the user will stay")
    weather_info: str = Field(description="Weather information for the destination")


class TravelItinerary(BaseModel):
    daily_plan: str = Field(
        description="Day-by-day itinerary with activities and attractions"
    )
    must_see: str = Field(description="Top must-see attractions for this duration")
    food_recommendations: str = Field(
        description="Local food and restaurant suggestions"
    )


# Step 1: Extract travel details from user request
@step
async def extract_travel_details(
    data: TravelRequest, ctx: ExecutionContext
) -> TravelDetails:
    """Extract destination city and duration from the user's travel request."""
    result = await ctx.llm(
        name="travel_parser",
        instructions="Extract the destination city, country, and number of days from the user's travel request. Also infer their travel style based on any hints in their message.",
        input_schema=TravelRequest,
        output_schema=TravelDetails,
        input=data,
    )
    return result


# Step 2: Get weather information for the destination
@step
async def get_weather_info(
    data: TravelDetails, ctx: ExecutionContext
) -> TravelWithWeather:
    """Get weather information for the destination using the weather agent."""
    # Use the weather agent to get weather info for the specific city
    weather_query = f"What's the weather like in {data.city} today?"
    weather_result = tools_agent.process(weather_query)["execution_history"][-1][
        "action_result"
    ]

    return TravelWithWeather(
        city=data.city, days_of_stay=data.days_of_stay, weather_info=weather_result
    )


# Step 3: Create travel itinerary based on travel details AND weather
@step
async def create_itinerary(
    data: TravelWithWeather, ctx: ExecutionContext
) -> TravelItinerary:
    """Create a detailed travel itinerary considering destination, duration, and weather."""
    result = await ctx.llm(
        name="itinerary_planner",
        instructions="Create a detailed travel itinerary based on the destination, duration, and current weather conditions. Use the weather information to suggest appropriate indoor/outdoor activities and clothing recommendations.",
        input_schema=TravelWithWeather,
        output_schema=TravelItinerary,
        input=data,
        model="anthropic/claude-3.5-sonnet",  # Optional: specify model
    )
    return result


# Create workflow - now with 3 sequential steps including weather
workflow = (
    Workflow(
        id="travel-itinerary-workflow",
        input_model=TravelRequest,
        output_model=TravelItinerary,
    )
    .then(extract_travel_details)  # Step 1: TravelRequest -> TravelDetails
    .then(get_weather_info)  # Step 2: TravelDetails -> TravelWithWeather
    .then(create_itinerary)  # Step 3: TravelWithWeather -> TravelItinerary
    .commit()
)

# Create agent with workflow
agent = Agent(
    name="TravelPlannerAgent",
    description="Sequential workflow that extracts travel details then creates an itinerary",
    flow=workflow,
    verbose=True,
    model="berget/gpt-oss-120b",
)

# Process goal - executes sequential workflow
result = agent.process(
    "I'm visiting Rome for 3 days next weekend, can you help me plan my trip?"
)
print(result)
