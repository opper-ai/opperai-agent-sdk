import asyncio
import os
import sys
from typing import List, Literal

# Allow running from repo root without install
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from pydantic import BaseModel, Field

from opper_agent import (
    InMemoryStorage,
    Workflow,
    create_step,
)

try:
    from opperai import Opper
except Exception:  # pragma: no cover
    Opper = None  # type: ignore


# Input: raw candidate resume text
class ResumeIn(BaseModel):
    text: str = Field(description="Raw resume text for a job candidate")


# Parsed candidate profile
class CandidateProfile(BaseModel):
    thoughts: str = Field(description="Reasoning when extracting profile fields")
    role: str = Field(description="Primary role title for the candidate")
    experience_years: int = Field(description="Years of professional experience")
    skills: List[str] = Field(description="List of top candidate skills")
    category: Literal["technical", "behavioral"] = Field(description="Interview track suggested by profile")


# Output of question generation step
class QuestionOut(BaseModel):
    thoughts: str = Field(description="Reasoning for question selection")
    question: str = Field(description="Interview question to ask")
    question_type: Literal["technical", "behavioral"] = Field(description="Type of question")
    profile: CandidateProfile = Field(description="Echo of the candidate profile for downstream steps")


# Final formatted output
class FinalOut(BaseModel):
    thoughts: str = Field(description="Reasoning for final formatting")
    profile: CandidateProfile = Field(description="Candidate profile")
    selected_question: str = Field(description="Chosen question to ask the candidate")


async def _parse_resume_run(ctx):
    text = ctx.input_data.text
    # Use Opper if available; fallback to heuristic parser for local runs
    if hasattr(ctx.opper, "call"):
        result = await ctx.call_model(
            name="parse_candidate_profile",
            instructions="Extract a structured candidate profile from the resume text.",
            input_schema=ResumeIn,
            output_schema=CandidateProfile,
            input_obj=ctx.input_data,
        )
        return result
    # Heuristic fallback
    lower = text.lower()
    role = "Software Engineer" if any(k in lower for k in ["engineer", "developer"]) else "Project Manager"
    experience_years = 5 if any(k in lower for k in ["5 years", "five years"]) else 2
    skills = [s for s in ["python", "typescript", "sql", "leadership", "communication"] if s in lower]
    category: Literal["technical", "behavioral"] = "technical" if any(s in skills for s in ["python", "typescript", "sql"]) else "behavioral"
    return {
        "thoughts": "Parsed via heuristic fallback",
        "role": role,
        "experience_years": experience_years,
        "skills": skills,
        "category": category,
    }


parse_resume = create_step(
    id="parse_resume",
    input_model=ResumeIn,
    output_model=CandidateProfile,
    run=_parse_resume_run,
)


async def _ask_technical_run(ctx):
    profile = ctx.input_data
    if hasattr(ctx.opper, "call"):
        result = await ctx.call_model(
            name="generate_technical_question",
            instructions="Given a candidate profile, produce one technical interview question.",
            input_schema=CandidateProfile,
            output_schema=QuestionOut,
            input_obj=profile,
        )
        return result
    # Fallback
    question = f"Describe how you would design a scalable API, {profile.role}."
    return {
        "thoughts": "Local fallback technical question",
        "question": question,
        "question_type": "technical",
        "profile": profile,
    }


ask_technical = create_step(
    id="ask_technical",
    input_model=CandidateProfile,
    output_model=QuestionOut,
    run=_ask_technical_run,
)


async def _ask_behavioral_run(ctx):
    profile = ctx.input_data
    if hasattr(ctx.opper, "call"):
        result = await ctx.call_model(
            name="generate_behavioral_question",
            instructions="Given a candidate profile, produce one behavioral interview question.",
            input_schema=CandidateProfile,
            output_schema=QuestionOut,
            input_obj=profile,
        )
        return result
    # Fallback
    question = f"Tell me about a time you resolved a team conflict, {profile.role}."
    return {
        "thoughts": "Local fallback behavioral question",
        "question": question,
        "question_type": "behavioral",
        "profile": profile,
    }


ask_behavioral = create_step(
    id="ask_behavioral",
    input_model=CandidateProfile,
    output_model=QuestionOut,
    run=_ask_behavioral_run,
)


async def _format_output_run(ctx):
    q: QuestionOut = ctx.input_data
    return FinalOut(
        thoughts="Format final output for UI",
        profile=q.profile,
        selected_question=q.question,
    )


format_output = create_step(
    id="format_output",
    input_model=QuestionOut,
    output_model=FinalOut,
    run=_format_output_run,
)


# Workflow: parse resume -> branch (technical or behavioral) -> format
wf = (
    Workflow(id="recruiter-workflow", input_model=ResumeIn, output_model=FinalOut)
    .then(parse_resume)
    .branch([
        (lambda p: p.category == "technical", ask_technical),
        (lambda p: p.category == "behavioral", ask_behavioral),
    ])
    .map(lambda arr: arr[0])
    .then(format_output)
    .commit()
)


def _dummy_opper():
    class _Span:
        def __init__(self):
            self.id = None

    class _Spans:
        def create(self, **kwargs):
            return _Span()
        def update(self, **kwargs):
            return None

    class _Opper:
        def __init__(self):
            self.spans = _Spans()

    return _Opper()


async def main() -> None:
    api_key = os.getenv("OPPER_API_KEY")
    if Opper is not None and api_key:
        opper = Opper(http_bearer=api_key)
    else:
        if Opper is not None and not api_key:
            print("OPPER_API_KEY not set; using local dummy spans. Set it to send traces to Opper.")
        opper = _dummy_opper()

    storage = InMemoryStorage()
    run = wf.create_run(opper=opper, storage=storage, tools={}, memory=None, logger=print)

    resume_text = (
        "Senior Software Engineer with 5 years experience in Python and SQL. "
        "Built scalable APIs and mentored junior developers."
    )
    result = await run.start(input_data=ResumeIn(text=resume_text))
    print(result.model_dump())


if __name__ == "__main__":
    asyncio.run(main())
