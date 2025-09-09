#!/usr/bin/env python3
"""
Advanced workflow example demonstrating branching, parallelism, and error handling
using the @step decorator with comprehensive control flow patterns.
"""

import asyncio
import os
import sys
from typing import List, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from pydantic import BaseModel, Field
from opper_agent import Agent, step, Workflow, StepContext


# === DATA MODELS ===

class DocumentInput(BaseModel):
    content: str = Field(description="Document content to process")
    priority: str = Field(description="Processing priority: high, normal, low", default="normal")
    document_type: str = Field(description="Type of document: article, report, email", default="article")

class SentimentAnalysis(BaseModel):
    sentiment: str = Field(description="Sentiment: positive, negative, neutral")
    confidence: float = Field(description="Confidence score 0-1")
    reasoning: str = Field(description="Reasoning for sentiment classification")

class TopicAnalysis(BaseModel):
    topics: List[str] = Field(description="Key topics identified")
    categories: List[str] = Field(description="Document categories")
    reasoning: str = Field(description="Reasoning for topic identification")

class CombinedAnalysis(BaseModel):
    sentiment: SentimentAnalysis
    topics: TopicAnalysis
    summary: str = Field(description="Document summary")

class ProcessingMetadata(BaseModel):
    processing_time: float = Field(description="Processing time in seconds")
    priority_level: str = Field(description="Priority level used")
    steps_completed: List[str] = Field(description="List of completed steps")
    errors_encountered: List[str] = Field(description="Any errors encountered", default_factory=list)

class ProcessedDocument(BaseModel):
    analysis: CombinedAnalysis
    enhanced_content: str = Field(description="Enhanced document content")
    metadata: ProcessingMetadata
    recommendations: List[str] = Field(description="Content improvement recommendations")


# === STEP DEFINITIONS ===

@step
async def analyze_sentiment(ctx: StepContext[DocumentInput, SentimentAnalysis]) -> SentimentAnalysis:
    """Analyze document sentiment with confidence scoring."""
    doc = ctx.input_data
    
    result = await ctx.call_model(
        name="sentiment_analyzer",
        instructions="Analyze the sentiment of the document content. Provide confidence score and reasoning.",
        input_schema=DocumentInput,
        output_schema=SentimentAnalysis,
        input_obj=doc
    )
    return result

@step(retry={"attempts": 2}, timeout_ms=15000)
async def analyze_topics(ctx: StepContext[DocumentInput, TopicAnalysis]) -> TopicAnalysis:
    """Extract topics and categorize the document."""
    doc = ctx.input_data
    
    result = await ctx.call_model(
        name="topic_analyzer", 
        instructions="Extract key topics and categorize the document. Focus on main themes and subjects.",
        input_schema=DocumentInput,
        output_schema=TopicAnalysis,
        input_obj=doc
    )
    return result

@step
async def create_summary(ctx: StepContext[DocumentInput, str]) -> str:
    """Generate a concise summary of the document."""
    doc = ctx.input_data
    
    result = await ctx.call_model(
        name="summarizer",
        instructions="Create a concise, informative summary of the document content.",
        input_schema=DocumentInput,
        output_schema=str,
        input_obj=doc
    )
    return result

@step
async def combine_analysis(ctx: StepContext[List, CombinedAnalysis]) -> CombinedAnalysis:
    """Combine parallel analysis results."""
    results = ctx.input_data
    
    # Extract results from parallel processing
    sentiment_analysis = results[0]  # From analyze_sentiment
    topic_analysis = results[1]      # From analyze_topics  
    summary = results[2]             # From create_summary
    
    return CombinedAnalysis(
        sentiment=sentiment_analysis,
        topics=topic_analysis,
        summary=summary
    )

# Priority-based processing steps
@step(id="high_priority_processing")
async def process_high_priority(ctx: StepContext[DocumentInput, DocumentInput]) -> DocumentInput:
    """Special processing for high-priority documents."""
    doc = ctx.input_data
    
    # Add urgency markers and fast-track processing
    enhanced_content = f"[URGENT] {doc.content}"
    
    return DocumentInput(
        content=enhanced_content,
        priority=doc.priority,
        document_type=doc.document_type
    )

@step(id="normal_priority_processing", on_error="continue")
async def process_normal_priority(ctx: StepContext[DocumentInput, DocumentInput]) -> DocumentInput:
    """Standard processing for normal priority documents."""
    doc = ctx.input_data
    
    # Standard processing
    return doc

@step(id="batch_processing")
async def process_low_priority(ctx: StepContext[DocumentInput, DocumentInput]) -> DocumentInput:
    """Batch processing for low priority documents."""
    doc = ctx.input_data
    
    # Add batch processing marker
    enhanced_content = f"[BATCH] {doc.content}"
    
    return DocumentInput(
        content=enhanced_content,
        priority=doc.priority,
        document_type=doc.document_type
    )

@step(retry={"attempts": 3, "backoff_ms": 1000})
async def enhance_content(ctx: StepContext[CombinedAnalysis, str]) -> str:
    """Enhance content based on analysis results."""
    analysis = ctx.input_data
    
    # Use analysis to enhance content
    topic_focus = ", ".join(analysis.topics.topics[:3])  # Top 3 topics
    
    result = await ctx.call_model(
        name="content_enhancer",
        instructions=f"Enhance the content focusing on these topics: {topic_focus}. "
                    f"Consider the {analysis.sentiment.sentiment} sentiment.",
        input_schema=CombinedAnalysis,
        output_schema=str,
        input_obj=analysis
    )
    return result

@step(on_error="continue")
async def generate_recommendations(ctx: StepContext[CombinedAnalysis, List[str]]) -> List[str]:
    """Generate content improvement recommendations."""
    analysis = ctx.input_data
    
    # Generate recommendations based on analysis
    recommendations = []
    
    if analysis.sentiment.confidence < 0.7:
        recommendations.append("Consider clarifying emotional tone for better sentiment clarity")
    
    if len(analysis.topics.topics) < 3:
        recommendations.append("Consider expanding topic coverage for richer content")
    
    if len(analysis.summary) < 100:
        recommendations.append("Summary could be more detailed")
    
    # Add AI-generated recommendations
    try:
        ai_recs = await ctx.call_model(
            name="recommendation_generator",
            instructions="Generate 2-3 specific recommendations for improving this content.",
            input_schema=CombinedAnalysis,
            output_schema=List[str],
            input_obj=analysis
        )
        recommendations.extend(ai_recs)
    except Exception:
        # Fallback recommendations
        recommendations.append("Review content for clarity and engagement")
    
    return recommendations

@step
async def create_metadata(ctx: StepContext[DocumentInput, ProcessingMetadata]) -> ProcessingMetadata:
    """Create processing metadata."""
    doc = ctx.input_data
    
    return ProcessingMetadata(
        processing_time=2.5,  # Simulated processing time
        priority_level=doc.priority,
        steps_completed=["sentiment_analysis", "topic_analysis", "summarization", "enhancement"],
        errors_encountered=[]
    )


# === WORKFLOW DEFINITION ===

# Condition functions for branching
def is_high_priority(data: DocumentInput) -> bool:
    return data.priority == "high"

def is_normal_priority(data: DocumentInput) -> bool:
    return data.priority == "normal"

def is_low_priority(data: DocumentInput) -> bool:
    return data.priority == "low"

# Create the complex workflow
def create_advanced_workflow():
    """Create a workflow demonstrating advanced patterns."""
    
    return (Workflow(id="advanced-document-processor", 
                    input_model=DocumentInput, 
                    output_model=ProcessedDocument)
        
        # 1. BRANCHING: Route based on priority
        .branch([
            (is_high_priority, process_high_priority),
            (is_normal_priority, process_normal_priority),
            (is_low_priority, process_low_priority),
        ])
        
        # 2. PARALLEL PROCESSING: Run analysis tasks concurrently
        .parallel([
            analyze_sentiment,
            analyze_topics,
            create_summary,
        ])
        
        # 3. SEQUENTIAL PROCESSING: Combine results
        .then(combine_analysis)
        
        # 4. PARALLEL PROCESSING: Generate enhancements and metadata
        .parallel([
            enhance_content,
            generate_recommendations,
            # Transform input for metadata step
            Workflow(id="metadata-sub", input_model=CombinedAnalysis, output_model=ProcessingMetadata)
                .map(lambda analysis: DocumentInput(
                    content="", 
                    priority=getattr(analysis, 'priority', 'normal')
                ))
                .then(create_metadata)
                .commit()
        ])
        
        # 5. FINAL COMBINATION: Assemble final result
        .map(lambda results: ProcessedDocument(
            analysis=results[0],  # From combine_analysis (previous step)
            enhanced_content=results[1],  # From enhance_content
            recommendations=results[2],   # From generate_recommendations
            metadata=results[3] if len(results) > 3 else ProcessingMetadata(
                processing_time=1.0,
                priority_level="normal", 
                steps_completed=["basic_processing"],
                errors_encountered=[]
            )
        ))
        
        .commit())


def main():
    """Demonstrate the advanced workflow patterns."""
    print("=== Advanced Workflow Example ===")
    
    # Create the workflow
    workflow = create_advanced_workflow()
    
    print(f"📋 Workflow: {workflow.id}")
    print(f"🔀 Pipeline steps: {len(workflow.pipeline)}")
    
    # Show workflow structure
    print("\n🏗️  Workflow Structure:")
    for i, (kind, payload) in enumerate(workflow.pipeline):
        if kind == "branch":
            print(f"  {i+1}. BRANCH: {len(payload)} conditions")
        elif kind == "parallel":
            items, concurrency = payload if isinstance(payload, tuple) else (payload, None)
            print(f"  {i+1}. PARALLEL: {len(items)} steps" + 
                  (f" (concurrency: {concurrency})" if concurrency else ""))
        elif kind == "then":
            step_name = getattr(payload.defn, 'id', 'unknown') if hasattr(payload, 'defn') else 'unknown'
            print(f"  {i+1}. SEQUENTIAL: {step_name}")
        elif kind == "map":
            print(f"  {i+1}. TRANSFORM: data mapping")
        else:
            print(f"  {i+1}. {kind.upper()}: {type(payload).__name__}")
    
    # Test with different priorities
    test_documents = [
        DocumentInput(
            content="This is an urgent report about system failures that need immediate attention!",
            priority="high",
            document_type="report"
        ),
        DocumentInput(
            content="Here's a regular article about machine learning trends in 2024.",
            priority="normal", 
            document_type="article"
        ),
        DocumentInput(
            content="Low priority email about upcoming team meeting next week.",
            priority="low",
            document_type="email"
        )
    ]
    
    api_key = os.getenv("OPPER_API_KEY", "dummy-key")
    
    for i, doc in enumerate(test_documents, 1):
        print(f"\n📄 Test Document {i} ({doc.priority} priority):")
        print(f"   Content: {doc.content[:60]}...")
        print(f"   Type: {doc.document_type}")
        
        if api_key != "dummy-key":
            try:
                # Create agent and process
                agent = Agent(
                    name="AdvancedProcessor",
                    description="Advanced document processor with branching and parallelism",
                    flow=workflow,
                    opper_api_key=api_key
                )
                
                result = agent.process(f"Process this {doc.priority} priority {doc.document_type}: {doc.content}")
                print(f"   ✅ Processed successfully")
                print(f"   📊 Topics: {len(result.analysis.topics.topics) if hasattr(result, 'analysis') else 'N/A'}")
                print(f"   🎯 Sentiment: {result.analysis.sentiment.sentiment if hasattr(result, 'analysis') else 'N/A'}")
                
            except Exception as e:
                print(f"   ❌ Processing failed: {str(e)[:100]}...")
        else:
            print(f"   ⏭️  Skipped (set OPPER_API_KEY to run)")
    
    print("\n=== Workflow Benefits ===")
    print("✅ Conditional branching based on document priority")
    print("✅ Parallel processing for independent analysis tasks")
    print("✅ Error handling with retries and fallbacks")
    print("✅ Type-safe step definitions with @step decorator")
    print("✅ Automatic event emission for UI progress tracking")
    print("✅ Flexible workflow composition and reuse")


if __name__ == "__main__":
    main()
