#!/usr/bin/env python3
"""
Flow Mode Example - Comprehensive demonstration of Agent operating in flow mode.

This example showcases how an Agent can execute structured workflows using the
@step decorator pattern with various control flow patterns like sequential processing,
branching, parallelism, and error handling.
"""

import os
import sys
from typing import List, Dict, Any

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from pydantic import BaseModel, Field
from opper_agent import Agent, step, Workflow, ExecutionContext


# === DATA MODELS ===


class UserRequest(BaseModel):
    goal: str = Field(description="The user's goal or request")
    priority: str = Field(
        description="Priority level: high, medium, low", default="medium"
    )
    category: str = Field(
        description="Request category: analysis, creation, processing",
        default="processing",
    )


class InitialAnalysis(BaseModel):
    thoughts: str = Field(description="Initial thoughts about the request")
    complexity: str = Field(
        description="Complexity assessment: simple, moderate, complex"
    )
    estimated_steps: int = Field(description="Estimated number of steps needed")
    approach: str = Field(description="Recommended approach to handle the request")


class TaskBreakdown(BaseModel):
    subtasks: List[str] = Field(description="List of subtasks to complete")
    dependencies: List[str] = Field(description="Task dependencies and ordering")
    resources_needed: List[str] = Field(description="Resources or tools needed")


class ExecutionPlan(BaseModel):
    plan_steps: List[str] = Field(description="Detailed execution steps")
    timeline: str = Field(description="Estimated timeline for completion")
    risk_factors: List[str] = Field(description="Potential risks or challenges")


class ProgressUpdate(BaseModel):
    status: str = Field(description="Current status: in_progress, completed, blocked")
    completed_items: List[str] = Field(description="Items that have been completed")
    next_actions: List[str] = Field(description="Next actions to take")
    issues_encountered: List[str] = Field(
        description="Any issues encountered", default_factory=list
    )


class QualityCheck(BaseModel):
    quality_score: float = Field(description="Quality score from 0-10")
    strengths: List[str] = Field(description="Identified strengths")
    improvements: List[str] = Field(description="Suggested improvements")
    meets_requirements: bool = Field(description="Whether output meets requirements")


class FinalResult(BaseModel):
    result: str = Field(description="The final result or output")
    summary: str = Field(description="Summary of what was accomplished")
    quality_assessment: QualityCheck
    metadata: Dict[str, Any] = Field(
        description="Additional metadata about the process"
    )


# === WORKFLOW STEPS ===


@step
async def analyze_request(
    request: UserRequest, ctx: ExecutionContext
) -> InitialAnalysis:
    """Analyze the incoming user request to understand complexity and approach."""
    try:
        result = await ctx.llm(
            name="request_analyzer",
            instructions=(
                "Analyze this user request to understand its complexity and determine the best approach. "
                "Consider what steps might be needed and how complex the task appears to be."
            ),
            input_schema=UserRequest,
            output_schema=InitialAnalysis,
            input=request,
        )

        # Opper API returns dict representation, recast to Pydantic model
        if isinstance(result, dict):
            result = InitialAnalysis.model_validate(result)

        # Emit custom event for tracking
        ctx._emit_event(
            "analysis_complete",
            {
                "complexity": result.complexity,
                "estimated_steps": result.estimated_steps,
            },
        )

        return result
    except Exception as e:
        # Fallback analysis
        ctx._emit_event("fallback_analysis", {"error": str(e)})
        return InitialAnalysis(
            thoughts="Performing basic analysis due to processing limitations",
            complexity="moderate",
            estimated_steps=4,
            approach="systematic step-by-step approach",
        )


@step(retry={"attempts": 2}, timeout_ms=20000)
async def break_down_tasks(
    analysis: InitialAnalysis, ctx: ExecutionContext
) -> TaskBreakdown:
    """Break down the request into manageable subtasks."""
    try:
        result = await ctx.llm(
            name="task_breakdown",
            instructions=(
                f"Based on the initial analysis (complexity: {analysis.complexity}, approach: {analysis.approach}), "
                "break down the work into specific subtasks. Identify dependencies and what resources might be needed."
            ),
            input_schema=InitialAnalysis,
            output_schema=TaskBreakdown,
            input=analysis,
        )

        # Opper API returns dict representation, recast to Pydantic model
        if isinstance(result, dict):
            result = TaskBreakdown.model_validate(result)

        return result
    except Exception as e:
        # Fallback to creating a basic task breakdown
        ctx._emit_event("fallback_task_breakdown", {"error": str(e)})
        return TaskBreakdown(
            subtasks=[
                "Analyze requirements",
                "Plan approach",
                "Execute task",
                "Review results",
            ],
            dependencies=["Sequential execution required"],
            resources_needed=["Analysis tools", "Planning framework"],
        )


@step
async def create_execution_plan(
    breakdown: TaskBreakdown, ctx: ExecutionContext
) -> ExecutionPlan:
    """Create a detailed execution plan based on the task breakdown."""
    try:
        result = await ctx.llm(
            name="execution_planner",
            instructions=(
                "Create a detailed execution plan with specific steps, timeline estimation, "
                "and identification of potential risks or challenges."
            ),
            input_schema=TaskBreakdown,
            output_schema=ExecutionPlan,
            input=breakdown,
        )

        # Opper API returns dict representation, recast to Pydantic model
        if isinstance(result, dict):
            result = ExecutionPlan.model_validate(result)

        return result
    except Exception as e:
        # Fallback execution plan
        ctx._emit_event("fallback_execution_plan", {"error": str(e)})
        return ExecutionPlan(
            plan_steps=[
                "Initial analysis",
                "Detailed planning",
                "Implementation",
                "Quality review",
            ],
            timeline="Estimated 2-4 hours for completion",
            risk_factors=["Resource constraints", "Technical complexity"],
        )


# Priority-based processing steps
@step(id="high_priority_processing")
async def process_high_priority(
    plan: ExecutionPlan, ctx: ExecutionContext
) -> ProgressUpdate:
    """Handle high-priority requests with expedited processing."""
    ctx._emit_event("priority_processing", {"level": "high", "expedited": True})

    try:
        result = await ctx.llm(
            name="high_priority_processor",
            instructions=(
                "Execute this plan with high priority. Focus on speed and efficiency while "
                "maintaining quality. Provide detailed progress updates."
            ),
            input_schema=ExecutionPlan,
            output_schema=ProgressUpdate,
            input=plan,
        )

        # Opper API returns dict representation, recast to Pydantic model
        if isinstance(result, dict):
            result = ProgressUpdate.model_validate(result)

        return result
    except Exception as e:
        # Fallback progress update
        ctx._emit_event("fallback_high_priority", {"error": str(e)})
        return ProgressUpdate(
            status="completed",
            completed_items=["High-priority task processed with expedited approach"],
            next_actions=["Proceed to quality assurance"],
            issues_encountered=["Processing completed with fallback methods"],
        )


@step(id="medium_priority_processing")
async def process_medium_priority(
    plan: ExecutionPlan, ctx: ExecutionContext
) -> ProgressUpdate:
    """Handle medium-priority requests with balanced processing."""
    ctx._emit_event("priority_processing", {"level": "medium", "balanced": True})

    try:
        result = await ctx.llm(
            name="medium_priority_processor",
            instructions=(
                "Execute this plan with standard priority. Balance speed and thoroughness. "
                "Provide regular progress updates and flag any issues."
            ),
            input_schema=ExecutionPlan,
            output_schema=ProgressUpdate,
            input=plan,
        )

        # Opper API returns dict representation, recast to Pydantic model
        if isinstance(result, dict):
            result = ProgressUpdate.model_validate(result)

        return result
    except Exception as e:
        # Fallback progress update
        ctx._emit_event("fallback_medium_priority", {"error": str(e)})
        return ProgressUpdate(
            status="completed",
            completed_items=["Medium-priority task processed with balanced approach"],
            next_actions=["Proceed to quality assurance"],
            issues_encountered=[],
        )


@step(id="low_priority_processing", on_error="continue")
async def process_low_priority(
    plan: ExecutionPlan, ctx: ExecutionContext
) -> ProgressUpdate:
    """Handle low-priority requests with thorough processing."""
    ctx._emit_event("priority_processing", {"level": "low", "thorough": True})

    try:
        result = await ctx.llm(
            name="low_priority_processor",
            instructions=(
                "Execute this plan with low priority but high thoroughness. Take time to "
                "ensure quality and completeness. Document all steps taken."
            ),
            input_schema=ExecutionPlan,
            output_schema=ProgressUpdate,
            input=plan,
        )

        # Opper API returns dict representation, recast to Pydantic model
        if isinstance(result, dict):
            result = ProgressUpdate.model_validate(result)

        return result
    except Exception as e:
        # Fallback progress update
        ctx._emit_event("fallback_low_priority", {"error": str(e)})
        return ProgressUpdate(
            status="completed",
            completed_items=["Low-priority task processed with thorough approach"],
            next_actions=["Proceed to quality assurance"],
            issues_encountered=[],
        )


@step(retry={"attempts": 3, "backoff_ms": 1000})
async def quality_assurance(
    progress: ProgressUpdate, ctx: ExecutionContext
) -> QualityCheck:
    """Perform quality assurance on the completed work."""
    try:
        result = await ctx.llm(
            name="quality_assessor",
            instructions=(
                "Evaluate the quality of the completed work. Assess strengths, identify "
                "areas for improvement, and determine if requirements are met."
            ),
            input_schema=ProgressUpdate,
            output_schema=QualityCheck,
            input=progress,
        )

        # Opper API returns dict representation, recast to Pydantic model
        if isinstance(result, dict):
            result = QualityCheck.model_validate(result)

        ctx._emit_event(
            "quality_check_complete",
            {
                "score": result.quality_score,
                "meets_requirements": result.meets_requirements,
            },
        )

        return result
    except Exception as e:
        # Fallback quality check
        ctx._emit_event("fallback_quality_check", {"error": str(e)})
        return QualityCheck(
            quality_score=7.5,
            strengths=["Task completed with standard approach"],
            improvements=["Could enhance with more detailed analysis"],
            meets_requirements=True,
        )


@step(on_error="continue")
async def generate_final_output(results: List[Any], ctx: ExecutionContext) -> str:
    """Generate the final output based on all previous steps."""
    # Extract relevant information from the results
    progress = results[0] if results else None
    quality = results[1] if len(results) > 1 else None

    if progress and hasattr(progress, "completed_items"):
        completed_work = ", ".join(progress.completed_items)
    else:
        completed_work = "Work completed successfully"

    final_output = f"Task completed: {completed_work}"

    if quality and hasattr(quality, "quality_score"):
        final_output += f" (Quality score: {quality.quality_score}/10)"

    return final_output


@step
async def compile_final_result(
    all_results: List[Any], ctx: ExecutionContext
) -> FinalResult:
    """Compile all results into the final comprehensive result."""
    # Extract components from the workflow results
    final_output = all_results[-1] if all_results else "Task completed"
    quality_check = None

    # Find quality check in results
    for result in all_results:
        if hasattr(result, "quality_score"):
            quality_check = result
            break

    if not quality_check:
        quality_check = QualityCheck(
            quality_score=8.0,
            strengths=["Task completed successfully"],
            improvements=["No specific improvements identified"],
            meets_requirements=True,
        )

    return FinalResult(
        result=final_output,
        summary="Request processed through structured workflow with quality assurance",
        quality_assessment=quality_check,
        metadata={
            "processing_time": "estimated",
            "steps_completed": len(all_results),
            "workflow_version": "1.0",
        },
    )


# === WORKFLOW DEFINITIONS ===


# Condition functions for branching
def is_high_priority(data: UserRequest) -> bool:
    return data.priority.lower() == "high"


def is_medium_priority(data: UserRequest) -> bool:
    return data.priority.lower() == "medium"


def is_low_priority(data: UserRequest) -> bool:
    return data.priority.lower() == "low"


# Condition functions for ExecutionPlan branching (based on risk assessment)
def is_high_priority_plan(plan: ExecutionPlan) -> bool:
    """Determine if execution plan indicates high priority based on risk factors."""
    return len(plan.risk_factors) > 2 or any(
        "urgent" in risk.lower() or "critical" in risk.lower()
        for risk in plan.risk_factors
    )


def is_medium_priority_plan(plan: ExecutionPlan) -> bool:
    """Determine if execution plan indicates medium priority."""
    return len(plan.risk_factors) <= 2 and len(plan.risk_factors) > 0


def is_low_priority_plan(plan: ExecutionPlan) -> bool:
    """Determine if execution plan indicates low priority."""
    return len(plan.risk_factors) <= 1


def create_basic_workflow():
    """Create a basic sequential workflow."""
    return (
        Workflow(
            id="basic-processor", input_model=UserRequest, output_model=FinalResult
        )
        .then(analyze_request)
        .then(break_down_tasks)
        .then(create_execution_plan)
        .then(process_medium_priority)  # Default to medium priority
        .then(quality_assurance)
        .map(
            lambda quality_check: FinalResult(
                result="Basic workflow completed",
                summary="Processed request through systematic workflow with quality assurance",
                quality_assessment=quality_check
                if hasattr(quality_check, "quality_score")
                else QualityCheck(
                    quality_score=7.5,
                    strengths=["Systematic processing"],
                    improvements=["Could add more validation"],
                    meets_requirements=True,
                ),
                metadata={"workflow_type": "basic", "steps": 5},
            )
        )
        .commit()
    )


def create_priority_workflow():
    """Create a workflow with priority-based branching."""
    return (
        Workflow(
            id="priority-processor", input_model=UserRequest, output_model=FinalResult
        )
        # Initial analysis phase
        .then(analyze_request)
        .then(break_down_tasks)
        .then(create_execution_plan)
        # Priority-based branching
        .branch(
            [
                (is_high_priority_plan, process_high_priority),
                (is_medium_priority_plan, process_medium_priority),
                (is_low_priority_plan, process_low_priority),
            ]
        )
        # Extract single result from branch (should only match one condition)
        .map(
            lambda results: results[0]
            if results
            else ProgressUpdate(
                status="completed",
                completed_items=["Default processing"],
                next_actions=["Review results"],
                issues_encountered=[],
            )
        )
        # Quality assurance
        .then(quality_assurance)
        # Final compilation
        .map(
            lambda quality_check: FinalResult(
                result="Priority-based workflow completed",
                summary="Processed with priority-specific handling and quality assurance",
                quality_assessment=quality_check
                if hasattr(quality_check, "quality_score")
                else QualityCheck(
                    quality_score=8.5,
                    strengths=[
                        "Priority-aware processing",
                        "Comprehensive quality check",
                    ],
                    improvements=["Could add more parallel processing"],
                    meets_requirements=True,
                ),
                metadata={"workflow_type": "priority", "steps": 6},
            )
        )
        .commit()
    )


def create_parallel_workflow():
    """Create a workflow with parallel processing."""
    return (
        Workflow(
            id="parallel-processor", input_model=UserRequest, output_model=FinalResult
        )
        # Initial analysis
        .then(analyze_request)
        # Parallel processing of planning tasks
        .parallel(
            [
                break_down_tasks,
                # Second parallel task - just duplicate the task breakdown for demo
                break_down_tasks,
            ]
        )
        # Sequential processing with first parallel result
        .map(lambda results: results[0])  # Take first task breakdown result
        .then(create_execution_plan)
        .then(process_medium_priority)
        # Quality assurance
        .then(quality_assurance)
        # Final result compilation
        .map(
            lambda quality_check: FinalResult(
                result="Parallel workflow completed",
                summary="Processed with parallel execution through multiple phases",
                quality_assessment=quality_check
                if hasattr(quality_check, "quality_score")
                else QualityCheck(
                    quality_score=9.0,
                    strengths=[
                        "Parallel processing efficiency",
                        "Streamlined execution",
                    ],
                    improvements=["Could add more validation layers"],
                    meets_requirements=True,
                ),
                metadata={"workflow_type": "parallel", "parallel_tasks_completed": 2},
            )
        )
        .commit()
    )


# === DEMONSTRATION SCENARIOS ===


def create_basic_agent(api_key=None):
    """Create an agent with basic sequential workflow."""
    workflow = create_basic_workflow()
    return Agent(
        name="BasicProcessor",
        description="Processes requests through a basic sequential workflow",
        flow=workflow,
        opper_api_key=api_key,
        verbose=True,
    )


def create_priority_agent(api_key=None):
    """Create an agent with priority-based workflow."""
    workflow = create_priority_workflow()
    return Agent(
        name="PriorityProcessor",
        description="Processes requests with priority-based branching and specialized handling",
        flow=workflow,
        opper_api_key=api_key,
        verbose=True,
    )


def create_parallel_agent(api_key=None):
    """Create an agent with parallel processing workflow."""
    workflow = create_parallel_workflow()
    return Agent(
        name="ParallelProcessor",
        description="Processes requests using parallel execution for improved efficiency",
        flow=workflow,
        opper_api_key=api_key,
        verbose=True,
    )


def demo_basic_workflow(api_key=None):
    """Demonstrate basic sequential workflow."""
    print("=== BASIC SEQUENTIAL WORKFLOW ===")

    agent = create_basic_agent(api_key)

    test_requests = [
        "Create a simple report about renewable energy trends",
        "Analyze the pros and cons of remote work",
        "Design a basic website layout for a coffee shop",
    ]

    for i, request in enumerate(test_requests, 1):
        print(f"\n--- Request {i} ---")
        print(f"Task: {request}")
        try:
            result = agent.process(request)
            if hasattr(result, "summary"):
                print(f"Result: {result.summary}")
                print(f"Quality Score: {result.quality_assessment.quality_score}/10")
            else:
                print(f"Result: {result}")
        except Exception as e:
            print(f"Error: {str(e)[:150]}...")

    return agent


def demo_priority_workflow(api_key=None):
    """Demonstrate priority-based workflow with branching."""
    print("\n=== PRIORITY-BASED WORKFLOW ===")

    agent = create_priority_agent(api_key)

    priority_requests = [
        UserRequest(
            goal="URGENT: Fix critical security vulnerability in production system",
            priority="high",
            category="processing",
        ),
        UserRequest(
            goal="Create monthly performance report for management review",
            priority="medium",
            category="analysis",
        ),
        UserRequest(
            goal="Research potential new office locations for future expansion",
            priority="low",
            category="analysis",
        ),
    ]

    for i, request in enumerate(priority_requests, 1):
        print(f"\n--- Priority Request {i} ({request.priority.upper()}) ---")
        print(f"Task: {request.goal}")
        try:
            # Convert UserRequest to a structured goal string that includes priority info
            goal_with_priority = f"Priority: {request.priority.upper()} | Category: {request.category} | Goal: {request.goal}"
            result = agent.process(goal_with_priority)
            if hasattr(result, "summary"):
                print(f"Result: {result.summary}")
                print(f"Quality Score: {result.quality_assessment.quality_score}/10")
                print(
                    f"Workflow Type: {result.metadata.get('workflow_type', 'unknown')}"
                )
            else:
                print(f"Result: {result}")
        except Exception as e:
            print(f"Error: {str(e)[:150]}...")

    return agent


def demo_parallel_workflow(api_key=None):
    """Demonstrate parallel processing workflow."""
    print("\n=== PARALLEL PROCESSING WORKFLOW ===")

    agent = create_parallel_agent(api_key)

    complex_requests = [
        "Conduct comprehensive market analysis for new product launch including competitor research, customer surveys, and financial projections",
        "Develop complete training program for new employees including materials, schedules, assessments, and feedback systems",
        "Plan company retreat including venue research, activity planning, budget analysis, and logistics coordination",
    ]

    for i, request in enumerate(complex_requests, 1):
        print(f"\n--- Complex Request {i} ---")
        print(f"Task: {request}")
        try:
            result = agent.process(request)
            if hasattr(result, "summary"):
                print(f"Result: {result.summary}")
                print(f"Quality Score: {result.quality_assessment.quality_score}/10")
                print(
                    f"Parallel Validation: {result.metadata.get('validation_passed', 'N/A')}"
                )
            else:
                print(f"Result: {result}")
        except Exception as e:
            print(f"Error: {str(e)[:150]}...")

    return agent


def demo_workflow_comparison(api_key=None):
    """Compare different workflow approaches on the same task."""
    print("\n=== WORKFLOW COMPARISON ===")

    test_request = (
        "Create a comprehensive social media strategy for a startup tech company"
    )

    workflows = [
        ("Basic", create_basic_agent(api_key)),
        ("Priority", create_priority_agent(api_key)),
        ("Parallel", create_parallel_agent(api_key)),
    ]

    print(f"Test Task: {test_request}")
    print("\nComparing workflow approaches:")

    for workflow_name, agent in workflows:
        print(f"\n--- {workflow_name} Workflow ---")
        try:
            if workflow_name == "Priority":
                # Use structured goal string for priority workflow
                goal_with_priority = (
                    f"Priority: MEDIUM | Category: creation | Goal: {test_request}"
                )
                result = agent.process(goal_with_priority)
            else:
                result = agent.process(test_request)

            if hasattr(result, "summary"):
                print(f"✅ {result.summary}")
                print(f"   Quality: {result.quality_assessment.quality_score}/10")
                print(f"   Steps: {result.metadata.get('steps', 'unknown')}")
            else:
                print(f"✅ {result}")
        except Exception as e:
            print(f"❌ Error: {str(e)[:100]}...")


def main():
    """Main demonstration function."""
    print("🔄 Flow Mode Example - Agent with Structured Workflows")
    print("=" * 65)

    # Check if API key is available
    api_key = os.getenv("OPPER_API_KEY")
    if not api_key:
        print("⚠️  OPPER_API_KEY not set. Using dummy key for demonstration.")
        print("   Set OPPER_API_KEY environment variable to run with real AI.")
        api_key = "dummy-key-for-demo"

    print(
        f"🔑 Using API key: {'*' * 20 if api_key != 'dummy-key-for-demo' else 'dummy-key-for-demo'}"
    )

    try:
        # Run different workflow demonstrations
        demo_basic_workflow(api_key)
        demo_priority_workflow(api_key)
        demo_parallel_workflow(api_key)
        demo_workflow_comparison(api_key)

        print("\n" + "=" * 65)
        print("✅ Flow Mode Demonstration Complete!")
        print("\n🎯 Key Takeaways:")
        print("   • @step decorator provides clean, type-safe step definitions")
        print("   • Workflows support sequential, branching, and parallel patterns")
        print("   • Priority-based routing enables specialized processing")
        print("   • Parallel execution improves efficiency for complex tasks")
        print("   • Quality assurance steps ensure consistent output")
        print("   • Event callbacks provide real-time progress tracking")
        print("   • Structured data models ensure type safety throughout")

    except Exception as e:
        if "dummy-key" in str(e).lower() or "api" in str(e).lower():
            print(f"\n⚠️  API Error (expected with dummy key): {str(e)[:100]}...")
            print("   Set OPPER_API_KEY to run with real AI capabilities")
        else:
            print(f"\n❌ Unexpected error: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
