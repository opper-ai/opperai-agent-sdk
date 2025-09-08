import asyncio
import os
import sys
import time
from typing import List, Optional, Dict, Any

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from pydantic import BaseModel, Field

from opper_agent import Agent, InMemoryStorage, Workflow, create_step
from opperai import Opper


# Data models for the RFP pipeline
class RFPQuestion(BaseModel):
    question_id: str = Field(description="Unique identifier for the question")
    question_text: str = Field(description="The actual question text")
    category: str = Field(description="Category of the question (e.g., technical, financial, operational)")
    required: bool = Field(description="Whether this question is mandatory")
    max_length: Optional[int] = Field(description="Maximum character length for the answer", default=None)


class RFP(BaseModel):
    rfp_id: str = Field(description="Unique identifier for the RFP")
    title: str = Field(description="Title of the RFP")
    description: str = Field(description="Description of the RFP")
    questions: List[RFPQuestion] = Field(description="List of questions in the RFP")
    submission_deadline: str = Field(description="Deadline for submission")
    fund_type: str = Field(description="Type of fund being requested")


class RFPProcessingInput(BaseModel):
    rfp: RFP = Field(description="The RFP to process")
    knowledge_base_ready: bool = Field(description="Whether knowledge base is ready", default=False)


class AnswerGenerationInput(BaseModel):
    question: RFPQuestion = Field(description="The RFP question to answer")
    knowledge_result: "KnowledgeQueryResult" = Field(description="Retrieved knowledge for the question")
    rfp_context: str = Field(description="Context about the RFP and fund")


class KnowledgeQueryResult(BaseModel):
    thoughts: str = Field(description="Reasoning for knowledge retrieval")
    question_id: str = Field(description="ID of the question")
    relevant_knowledge: List[str] = Field(description="List of relevant knowledge chunks")
    knowledge_sources: List[str] = Field(description="Sources of the knowledge")


class RFPAnswer(BaseModel):
    thoughts: str = Field(description="The AI's reasoning process for crafting this answer")
    question_id: str = Field(description="ID of the question being answered")
    answer_text: str = Field(description="The drafted answer to the RFP question as one paragraph")
    confidence_score: float = Field(description="Confidence score for the answer (0-1)")
    knowledge_sources: List[str] = Field(description="Sources from knowledge base used")
    suggested_length: int = Field(description="Suggested character length for the answer")


class RFPResponse(BaseModel):
    thoughts: str = Field(description="Final thoughts on the RFP response compilation")
    rfp_id: str = Field(description="ID of the RFP being responded to")
    answers: List[RFPAnswer] = Field(description="List of answers to RFP questions")
    total_answers: int = Field(description="Total number of answers provided")
    completion_rate: float = Field(description="Percentage of questions answered")


# Global knowledge base storage
KNOWLEDGE_BASE = None


def create_mock_rfp() -> RFP:
    """Create a mock RFP for a fund management service"""
    return RFP(
        rfp_id="RFP-2024-001",
        title="Fund Management Services for Technology Growth Fund",
        description="We are seeking proposals for comprehensive fund management services for our new $50M technology growth fund. The fund will focus on Series A and B investments in SaaS, AI, and fintech companies.",
        submission_deadline="2024-12-31",
        fund_type="Technology Growth Fund",
        questions=[
            RFPQuestion(
                question_id="Q1",
                question_text="Describe your firm's experience in managing technology growth funds, including track record, team expertise, and relevant case studies.",
                category="experience",
                required=True,
                max_length=2000
            ),
            RFPQuestion(
                question_id="Q2",
                question_text="What is your investment philosophy and approach to Series A and B investments in the technology sector?",
                category="strategy",
                required=True,
                max_length=1500
            ),
            RFPQuestion(
                question_id="Q3",
                question_text="Detail your fee structure, including management fees, carried interest, and any additional costs.",
                category="financial",
                required=True,
                max_length=1000
            ),
            RFPQuestion(
                question_id="Q4",
                question_text="How do you approach ESG considerations in your investment process?",
                category="esg",
                required=False,
                max_length=1000
            ),
            RFPQuestion(
                question_id="Q5",
                question_text="What technology and tools do you use for portfolio management, reporting, and investor communications?",
                category="operational",
                required=True,
                max_length=1200
            )
        ]
    )


async def setup_knowledge_base(opper):
    """Set up the knowledge base with relevant information for RFP responses"""
    global KNOWLEDGE_BASE
    
    if KNOWLEDGE_BASE is not None:
        return KNOWLEDGE_BASE
        
    knowledge_base_name = "rfp-knowledge-base"
    
    # Try to get existing knowledge base
    try:
        KNOWLEDGE_BASE = opper.knowledge.get_by_name(knowledge_base_name=knowledge_base_name)
        print(f"Using existing knowledge base: {knowledge_base_name}")
        return KNOWLEDGE_BASE
    except Exception:
        pass
    
    # Create new knowledge base
    KNOWLEDGE_BASE = opper.knowledge.create(name=knowledge_base_name)
    print(f"Created new knowledge base: {knowledge_base_name}")
    
    # Wait for knowledge base to be ready
    time.sleep(3)
    
    # Knowledge base entries
    knowledge_entries = [
        {
            "content": "TechVentures Capital is a leading venture capital firm with 15+ years of experience in technology investing. We have successfully managed 8 funds totaling $2.5B in assets under management. Our team consists of 25 investment professionals with deep expertise in SaaS, AI, and fintech sectors. Notable exits include CloudFlow (acquired by Microsoft for $1.2B) and DataSense (IPO at $500M valuation).",
            "metadata": {"category": "firm_info", "source": "internal", "key": "firm-overview"}
        },
        {
            "content": "Our investment philosophy centers on backing exceptional founders building category-defining companies. We focus on Series A and B rounds where we can provide both capital and strategic value. We look for companies with strong product-market fit, scalable business models, and teams capable of executing rapid growth. We typically invest $5-15M per company and take board seats to provide hands-on support.",
            "metadata": {"category": "strategy", "source": "internal", "key": "investment-philosophy"}
        },
        {
            "content": "Our standard fee structure includes: 2% annual management fee on committed capital for the first 5 years, 1.5% for years 6-10, and 1% thereafter. Carried interest is 20% after an 8% preferred return hurdle. We charge no additional fees for standard services. Fund expenses are capped at 0.5% of committed capital annually.",
            "metadata": {"category": "financial", "source": "internal", "key": "fee-structure"}
        },
        {
            "content": "ESG considerations are integrated throughout our investment process. We evaluate potential investments on environmental impact, social responsibility, and governance practices. We have a dedicated ESG committee that reviews all investments and provides ongoing oversight. We also work with portfolio companies to improve their ESG practices and reporting.",
            "metadata": {"category": "esg", "source": "internal", "key": "esg-approach"}
        },
        {
            "content": "We use a comprehensive technology stack for portfolio management: Salesforce for CRM and deal flow management, Carta for cap table and equity management, Tableau for data visualization and reporting, Slack for team communication, and custom dashboards for real-time portfolio monitoring. All systems are cloud-based with enterprise-grade security.",
            "metadata": {"category": "operational", "source": "internal", "key": "technology-stack"}
        }
    ]
    
    # Add entries to the knowledge base
    for entry in knowledge_entries:
        opper.knowledge.add(
            knowledge_base_id=KNOWLEDGE_BASE.id,
            content=entry["content"],
            metadata=entry["metadata"]
        )
        print(f"Added knowledge entry: {entry['metadata']['key']}")
    
    return KNOWLEDGE_BASE


# Step 1: Setup knowledge base
class KnowledgeSetupResult(BaseModel):
    thoughts: str = Field(description="Thoughts on knowledge base setup")
    rfp: RFP = Field(description="The RFP being processed")
    knowledge_base_id: str = Field(description="ID of the knowledge base")
    setup_successful: bool = Field(description="Whether setup was successful")


async def setup_knowledge_step(ctx):
    """Setup the knowledge base for RFP processing"""
    input_data: RFPProcessingInput = ctx.input_data
    rfp = input_data.rfp
    
    try:
        knowledge_base = await setup_knowledge_base(ctx.opper)
        return KnowledgeSetupResult(
            thoughts=f"Successfully set up knowledge base for RFP {rfp.rfp_id}",
            rfp=rfp,
            knowledge_base_id=knowledge_base.id,
            setup_successful=True
        )
    except Exception as e:
        return KnowledgeSetupResult(
            thoughts=f"Failed to setup knowledge base: {str(e)}",
            rfp=rfp,
            knowledge_base_id="",
            setup_successful=False
        )


setup_knowledge = create_step(
    id="setup_knowledge",
    input_model=RFPProcessingInput,
    output_model=KnowledgeSetupResult,
    run=setup_knowledge_step,
)


# Step 2a: Prepare questions for foreach processing
class QuestionContext(BaseModel):
    question: RFPQuestion = Field(description="The RFP question to process")
    knowledge_base_id: str = Field(description="ID of the knowledge base")
    rfp_context: str = Field(description="Context about the RFP")
    rfp_id: str = Field(description="ID of the RFP being processed")


class QuestionsPreparationResult(BaseModel):
    thoughts: str = Field(description="Thoughts on question preparation")
    question_contexts: List[QuestionContext] = Field(description="List of question contexts for foreach processing")
    rfp: RFP = Field(description="The original RFP")


async def prepare_questions_step(ctx):
    """Prepare questions for foreach processing"""
    input_data: KnowledgeSetupResult = ctx.input_data
    rfp = input_data.rfp
    
    if not input_data.setup_successful:
        return QuestionsPreparationResult(
            thoughts="Cannot prepare questions without knowledge base",
            question_contexts=[],
            rfp=rfp
        )
    
    rfp_context = f"RFP: {rfp.title} - {rfp.description}"
    question_contexts = [
        QuestionContext(
            question=question,
            knowledge_base_id=input_data.knowledge_base_id,
            rfp_context=rfp_context,
            rfp_id=rfp.rfp_id
        )
        for question in rfp.questions
    ]
    
    return QuestionsPreparationResult(
        thoughts=f"Prepared {len(question_contexts)} questions for parallel processing",
        question_contexts=question_contexts,
        rfp=rfp
    )


prepare_questions = create_step(
    id="prepare_questions",
    input_model=KnowledgeSetupResult,
    output_model=QuestionsPreparationResult,
    run=prepare_questions_step,
)


# Step 2b: Process a single question (used in foreach)
async def process_single_question_step(ctx):
    """Process a single RFP question and generate an answer"""
    input_data: QuestionContext = ctx.input_data
    question = input_data.question
    
    # Query knowledge base for this specific question
    query_input = QuestionKnowledgeInput(
        question=question,
        knowledge_base_id=input_data.knowledge_base_id,
        rfp_context=input_data.rfp_context
    )
    
    # Actually query the knowledge base first
    try:
        knowledge_results = ctx.opper.knowledge.query(
            knowledge_base_id=input_data.knowledge_base_id,
            query=question.question_text,
            top_k=3
        )
        
        relevant_knowledge = [result.content for result in knowledge_results]
        knowledge_sources = []
        
        for result in knowledge_results:
            if hasattr(result, 'metadata') and result.metadata and 'key' in result.metadata:
                knowledge_sources.append(result.metadata['key'])
            else:
                knowledge_sources.append(f"source_{hash(result.content) % 10000}")
        
        # Create the knowledge result with actual data
        knowledge_result = KnowledgeQueryResult(
            thoughts=f"Retrieved {len(relevant_knowledge)} relevant knowledge chunks for question {question.question_id}",
            question_id=question.question_id,
            relevant_knowledge=relevant_knowledge,
            knowledge_sources=knowledge_sources
        )
        
    except Exception as e:
        # Create fallback knowledge result
        knowledge_result = KnowledgeQueryResult(
            thoughts=f"Failed to query knowledge base: {str(e)}",
            question_id=question.question_id,
            relevant_knowledge=[],
            knowledge_sources=[]
        )
    
    # Generate answer for this question
    answer_input = AnswerGenerationInput(
        question=question,
        knowledge_result=knowledge_result,
        rfp_context=input_data.rfp_context
    )
    
    answer_dict = await ctx.call_model(
        name="generate_rfp_answer",
        instructions=(
            "You are an expert fund manager responding to an RFP. Use the provided knowledge context to craft a professional, "
            "comprehensive answer to the RFP question. Ensure the answer is well-structured, addresses all aspects of the question, "
            "and stays within any length constraints. Be specific and provide concrete examples where appropriate. "
            "Always provide a confidence score between 0.7 and 1.0 based on how well the knowledge matches the question."
        ),
        input_schema=AnswerGenerationInput,
        output_schema=RFPAnswer,
        input_obj=answer_input,
    )
    
    # Convert dictionary to RFPAnswer model and ensure correct data is set
    answer = RFPAnswer(
        thoughts=answer_dict.get('thoughts', ''),
        question_id=question.question_id,  # Ensure correct question_id
        answer_text=answer_dict.get('answer_text', ''),
        confidence_score=answer_dict.get('confidence_score', 0.0),
        knowledge_sources=knowledge_result.knowledge_sources,  # Use actual knowledge sources
        suggested_length=len(answer_dict.get('answer_text', ''))
    )
    
    return answer


process_single_question = create_step(
    id="process_single_question",
    input_model=QuestionContext,
    output_model=RFPAnswer,
    run=process_single_question_step,
)


# Step 2c: Input model for collecting answers from foreach
class AnswerListInput(BaseModel):
    answers: List[RFPAnswer] = Field(description="List of answers from foreach processing")


# Step 2c: Collect answers from foreach processing
class AnswerCollectionResult(BaseModel):
    thoughts: str = Field(description="Thoughts on answer collection")
    rfp: RFP = Field(description="The RFP being processed")
    answers: List[RFPAnswer] = Field(description="All generated answers")
    questions_processed: int = Field(description="Number of questions processed")


async def collect_answers_step(ctx):
    """Collect answers from foreach processing"""
    input_data: AnswerListInput = ctx.input_data
    answers = input_data.answers
    
    # Create a minimal RFP object from the answers
    # We'll extract the RFP ID from the first answer if available
    rfp_id = "RFP-2024-001"  # Default for our mock RFP
    if answers:
        # Try to extract RFP ID from question IDs or use default
        first_answer = answers[0]
        if hasattr(first_answer, 'question_id'):
            rfp_id = "RFP-2024-001"  # Our mock RFP ID
    
    # Create a minimal RFP object for the final step
    mock_rfp = RFP(
        rfp_id=rfp_id,
        title="Fund Management Services for Technology Growth Fund",
        description="Mock RFP for technology growth fund management services",
        questions=[],  # We don't need the full questions list at this point
        submission_deadline="2024-12-31",
        fund_type="Technology Growth Fund"
    )
    
    return AnswerCollectionResult(
        thoughts=f"Collected {len(answers)} answers from parallel processing using foreach workflow",
        rfp=mock_rfp,
        answers=answers,
        questions_processed=len(answers)
    )


collect_answers = create_step(
    id="collect_answers",
    input_model=AnswerListInput,
    output_model=AnswerCollectionResult,
    run=collect_answers_step,
)


# Step 3: Compile final response
async def compile_final_response_step(ctx):
    """Compile the final RFP response with quality assessment"""
    input_data: AnswerCollectionResult = ctx.input_data
    rfp = input_data.rfp
    answers = input_data.answers
    
    completion_rate = len(answers) / len(rfp.questions) * 100 if rfp.questions else 0
    
    # Use AI to provide final thoughts on the overall response quality
    compile_input = CompileResponseInput(
        rfp=rfp,
        answers=answers,
        completion_rate=completion_rate
    )
    
    result_dict = await ctx.call_model(
        name="compile_rfp_response",
        instructions=(
            "You are reviewing a completed RFP response. Analyze the quality and completeness of all answers. "
            "Provide thoughtful feedback on whether this response would be competitive and compelling to the RFP issuer. "
            "Consider the coherence across answers, coverage of requirements, and professional presentation."
        ),
        input_schema=CompileResponseInput,
        output_schema=RFPResponse,
        input_obj=compile_input,
    )
    
    # Convert dictionary to RFPResponse model and ensure correct data is preserved
    result = RFPResponse(
        thoughts=result_dict.get('thoughts', 'Compiled RFP response successfully'),
        rfp_id=rfp.rfp_id,
        answers=answers,
        total_answers=len(answers),
        completion_rate=completion_rate
    )
    
    return result


compile_final_response = create_step(
    id="compile_final_response",
    input_model=AnswerCollectionResult,
    output_model=RFPResponse,
    run=compile_final_response_step,
)


# Additional models needed for the workflow
class QuestionKnowledgeInput(BaseModel):
    question: RFPQuestion = Field(description="The RFP question to find knowledge for")
    knowledge_base_id: str = Field(description="ID of the knowledge base to query")
    rfp_context: str = Field(description="Context about the RFP")


class CompileResponseInput(BaseModel):
    rfp: RFP = Field(description="The original RFP")
    answers: List[RFPAnswer] = Field(description="All generated answers")
    completion_rate: float = Field(description="Percentage of questions answered")


# Create the RFP processing agent with foreach workflow
rfp_agent = Agent(
    name="RFP Processing Agent",
    instructions=(
        "You are an expert fund management consultant specializing in responding to RFPs for investment services. "
        "You have deep knowledge of venture capital, fund management, and institutional investing practices."
    ),
    flow=Workflow(id="rfp-processing-flow", input_model=RFPProcessingInput, output_model=RFPResponse)
        .then(setup_knowledge)           # Step 1: Setup knowledge base
        .then(prepare_questions)         # Step 2a: Prepare questions for foreach
        .foreach(                        # Step 2b: Process each question in parallel
            process_single_question,
            concurrency=3,               # Process up to 3 questions concurrently
            map_func=lambda prep_result: prep_result.question_contexts
        )
        .map(lambda answers_list: AnswerListInput(answers=answers_list))  # Convert list to proper input model
        .then(collect_answers)           # Step 2c: Collect all answers
        .then(compile_final_response)    # Step 3: Compile final response
        .commit(),
)


async def main() -> None:
    api_key = os.getenv("OPPER_API_KEY")
    if not api_key:
        print("Warning: OPPER_API_KEY not found. Using fallback for demo purposes.")
        print("Please set OPPER_API_KEY environment variable for full functionality.")
        return
        
    opper = Opper(http_bearer=api_key)
    storage = InMemoryStorage()
    
    print("🚀 Starting RFP Processing Pipeline")
    print("=" * 50)
    
    # Create mock RFP
    print("\n📋 Step 1: Creating mock RFP...")
    rfp = create_mock_rfp()
    print(f"Created RFP: {rfp.title}")
    print(f"Questions: {len(rfp.questions)}")
    
    # Process RFP using the agent workflow with foreach
    print("\n🤖 Step 2: Processing RFP with foreach workflow...")
    print("   → Step 1: Setting up knowledge base...")
    print("   → Step 2a: Preparing questions for parallel processing...")
    print("   → Step 2b: Processing questions in parallel using foreach...")
    print("   → Step 2c: Collecting answers from parallel processing...")
    print("   → Step 3: Compiling final response...")
    
    try:
        def workflow_logger(message):
            print(message)
            if isinstance(message, str):
                print(f"   [Workflow] {message}")
            else:
                print(f"   [Workflow] {message}")
        
        run = rfp_agent.create_run(opper=opper, storage=storage, tools={}, logger=workflow_logger)
        final_response = await run.start(
            input_data=RFPProcessingInput(rfp=rfp, knowledge_base_ready=False)
        )
        
        # Display results
        print("\n📊 Results Summary:")
        print(f"RFP ID: {final_response.rfp_id}")
        print(f"Total Questions: {len(rfp.questions)}")
        print(f"Answers Generated: {final_response.total_answers}")
        print(f"Completion Rate: {final_response.completion_rate:.1f}%")
        
        # Display sample answers
        print("\n📝 Sample Answers:")
        for answer in final_response.answers[:2]:  # Show first 2 answers
            print(f"\n--- Question {answer.question_id} ---")
            print(f"Answer: {answer.answer_text[:200]}...")
            print(f"Confidence: {answer.confidence_score:.2f}")
            print(f"Sources: {', '.join(answer.knowledge_sources)}")
        
        print(f"\n🎯 Final Thoughts: {final_response.thoughts}")
        print("\n🎉 RFP Pipeline completed successfully!")
        
    except Exception as e:
        print(f"❌ Error processing RFP: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
