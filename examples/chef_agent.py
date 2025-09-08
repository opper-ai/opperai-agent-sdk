import asyncio
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from pydantic import BaseModel, Field

from opper_agent import Agent, InMemoryStorage, Workflow, create_step
from opperai import Opper


class MealRequest(BaseModel):
    ingredients: list[str] = Field(description="Available ingredients")
    dietary: str | None = Field(default=None, description="Dietary preference")


class ChefIdea(BaseModel):
    thoughts: str = Field(description="Reasoning for the meal idea")
    meal: str = Field(description="Proposed meal")


class RecipeSteps(BaseModel):
    thoughts: str = Field(description="Reasoning when composing steps")
    steps: list[str] = Field(description="Polished, step-by-step recipe text (without quantities)")


class QuantifiedRecipe(BaseModel):
    thoughts: str = Field(description="Reasoning for quantifying ingredients")
    ingredients_with_quantities: list[str] = Field(description="List of ingredients with quantities and units")


class ShoppingListOut(BaseModel):
    thoughts: str = Field(description="Reasoning for the shopping list")
    items: list[str] = Field(description="Shopping list items for the recipe")


class CombinedRecipeOut(BaseModel):
    thoughts: str = Field(description="Reasoning for the final presentation")
    description: str = Field(description="A compelling description that sells the dish")
    steps: list[str] = Field(description="Polished, step-by-step recipe text with quantities")
    shopping_list: list[str] = Field(description="Final shopping list items")


async def generate_idea_step(ctx):
    req = ctx.input_data
    result = await ctx.call_model(
        name="chef_generate_idea",
        instructions=(
            "Propose a specific, appetizing meal idea based on the available ingredients and dietary needs. "
            "Return only the idea in the 'meal' field (e.g., 'Spaghetti Carbonara')."
        ),
        input_schema=MealRequest,
        output_schema=ChefIdea,
        input_obj=req,
    )
    return result


generate_idea = create_step(
    id="generate_idea",
    input_model=MealRequest,
    output_model=ChefIdea,
    run=generate_idea_step,
)


async def generate_steps_step(ctx):
    idea: ChefIdea = ctx.input_data
    result = await ctx.call_model(
        name="chef_generate_steps",
        instructions=(
            "Write a clear, step-by-step recipe for the meal in 'meal'. "
            "Do not include quantities; focus on the technique and ordering."
        ),
        input_schema=ChefIdea,
        output_schema=RecipeSteps,
        input_obj=idea,
    )
    return result


generate_steps = create_step(
    id="generate_steps",
    input_model=ChefIdea,
    output_model=RecipeSteps,
    run=generate_steps_step,
)


async def generate_quantities_step(ctx):
    steps: RecipeSteps = ctx.input_data
    result = await ctx.call_model(
        name="chef_generate_quantities",
        instructions=(
            "Given a home-cooked recipe (without quantities), add exact, practical quantities and units for each ingredient. "
            "Return a list of ingredients with quantities and a refined recipe text including quantities."
        ),
        input_schema=RecipeSteps,
        output_schema=QuantifiedRecipe,
        input_obj=steps,
    )
    return result


generate_quantities = create_step(
    id="generate_quantities",
    input_model=RecipeSteps,
    output_model=QuantifiedRecipe,
    run=generate_quantities_step,
)


async def generate_shopping_list_step(ctx):
    q: QuantifiedRecipe = ctx.input_data
    result = await ctx.call_model(
        name="chef_generate_shopping_list",
        instructions=(
            "Given a quantified recipe (with ingredients and units), generate a concise shopping list. "
            "Group similar items and avoid duplicates."
        ),
        input_schema=QuantifiedRecipe,
        output_schema=ShoppingListOut,
        input_obj=q,
    )
    return result


generate_shopping_list = create_step(
    id="generate_shopping_list",
    input_model=QuantifiedRecipe,
    output_model=ShoppingListOut,
    run=generate_shopping_list_step,
)


async def combine_recipe_step(ctx):
    shopping: ShoppingListOut = ctx.input_data
    result = await ctx.call_model(
        name="chef_combine_recipe",
        instructions=(
            "You are Chef Michel. Combine the provided recipe and shopping list into a polished deliverable. "
            "Write a short, enticing description (1-3 sentences) that sells the dish, followed by a clear, "
            "well-structured recipe (ingredients and steps), and include the shopping list unchanged."
        ),
        input_schema=ShoppingListOut,
        output_schema=CombinedRecipeOut,
        input_obj=shopping,
    )
    return result


combine_recipe = create_step(
    id="combine_recipe",
    input_model=ShoppingListOut,
    output_model=CombinedRecipeOut,
    run=combine_recipe_step,
)


chef_agent = Agent(
    name="Chef Agent",
    instructions=(
        "You are Michel, a practical and experienced home chef who helps people cook great meals."
    ),
    flow=Workflow(id="chef-workflow", input_model=MealRequest, output_model=CombinedRecipeOut)
        .then(generate_idea)
        .then(generate_steps)
        .then(generate_quantities)
        .then(generate_shopping_list)
        .then(combine_recipe)
        .commit(),
)


async def main() -> None:
    api_key = os.getenv("OPPER_API_KEY")
    if not api_key:
        raise RuntimeError("OPPER_API_KEY is required")
    opper = Opper(http_bearer=api_key)

    storage = InMemoryStorage()
    run = chef_agent.create_run(opper=opper, storage=storage, tools={}, logger=print)
    result = await run.start(
        input_data=MealRequest(ingredients=["1 egg", "200g bacon", "100g spaghetti"], 
        dietary=None))
    print(result.model_dump())


if __name__ == "__main__":
    asyncio.run(main())
