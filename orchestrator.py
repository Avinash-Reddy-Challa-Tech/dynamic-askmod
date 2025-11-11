"""
Orchestrator for AskMod RAG System

This module implements an intelligent orchestrator that:
1. Takes a user query
2. Generates more targeted sub-questions
3. Sends these sub-questions to AskMod in parallel
4. Enhances responses with actual code from citation links
5. Evaluates the responses for quality and clarity
6. Generates follow-up questions if needed
7. Synthesizes all responses into a comprehensive answer
"""

import asyncio
import json
import logging
import os
from typing import List, Dict, Any, Optional, Tuple

import aiohttp
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from pydantic.v1 import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from query_decomposer import QueryDecomposer
from enhanced_response_evaluator import EnhancedResponseEvaluator
from response_synthesizer import ResponseSynthesizer
from askmod_client import AskModClient
from code_extractor import CodeExtractor  

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Orchestrator:
    """
    Orchestrates the process of breaking down queries, getting responses, and synthesizing a final answer.
    """
    def __init__(self, askmod_client: AskModClient, decomposer: QueryDecomposer,
                 evaluator: EnhancedResponseEvaluator, synthesizer: ResponseSynthesizer,
                 llm: Optional[ChatGoogleGenerativeAI] = None, max_iterations: int = 2,
                 code_extractor: Optional[CodeExtractor] = None):
        """
        Initialize the Orchestrator.
        
        Args:
            askmod_client: Client for interacting with AskMod API
            decomposer: Component for breaking down queries into sub-questions
            evaluator: Enhanced evaluator for response quality
            synthesizer: Component for synthesizing responses into a final answer
            llm: Language model for generating questions (if None, a default model is used)
            max_iterations: Maximum number of iterations for resolving ambiguities
            code_extractor: Component for extracting code from citation links (if None, a default is created)
        """
        self.askmod_client = askmod_client
        self.decomposer = decomposer
        self.evaluator = evaluator
        self.synthesizer = synthesizer
        self.llm = llm or ChatGoogleGenerativeAI(temperature=0.1)
        self.max_iterations = max_iterations
        self.code_extractor = code_extractor or CodeExtractor()
        
        # Create a directory for storing responses
        os.makedirs("responses", exist_ok=True)
        
    async def process_query(self, user_query: str) -> Dict[str, Any]:
        """
        Process a user query through the orchestration pipeline.
        
        Args:
            user_query: The original user query
            
        Returns:
            Dict containing the final synthesized answer
        """
        logger.info(f"Processing user query: {user_query}")
        
        # Step 1: Decompose the original query into sub-questions
        sub_questions = await self.decomposer.generate_sub_questions(user_query)
        logger.info(f"Generated {len(sub_questions)} sub-questions")
        
        # Initialize containers for our process
        all_qa_pairs = []
        iteration = 0
        
        while iteration < self.max_iterations:
            # Step 2: Process sub-questions in parallel
            responses = await self._process_sub_questions(sub_questions)
            
            # Step 2.5 (NEW): Enhance responses with code from citation links
            enhanced_responses = []
            for i, response in enumerate(responses):
                # Skip enhancement for error responses or None responses
                if response is None or (isinstance(response, str) and response.startswith("Error:")):
                    enhanced_responses.append(response)
                    continue
                
                try:
                    # Save original response to a file for reference
                    with open(f"responses/original_response_{i}_{iteration}.txt", "w", encoding="utf-8") as f:
                        f.write(response)
                    
                    # Enhance the response with code
                    enhanced_response = await self.code_extractor.enhance_response_with_code(response)
                    
                    enhanced_responses.append(enhanced_response)
                    
                except Exception as e:
                    logger.error(f"Error enhancing response {i}: {str(e)}", exc_info=True)
                    # Fall back to original response
                    enhanced_responses.append(response)
            
            # Step 3: Evaluate responses for quality and clarity using enhanced evaluator
            evaluation_results = await self._evaluate_responses(user_query, sub_questions, enhanced_responses)
            
            # Add the current Q&A pairs to our collection with enhanced evaluation metrics
            for q, r, eval_result in zip(sub_questions, enhanced_responses, evaluation_results):
                # Extract metrics from the enhanced evaluation
                qa_pair = {
                    "question": q,
                    "answer": r,
                    # Map enhanced evaluator metrics to the format expected by existing code
                    "clarity_score": eval_result.get("combined_score", 0),
                    "relevance_score": eval_result.get("statistical_metrics", {}).get("query_term_overlap", 0),
                    "is_clear": not eval_result.get("is_ambiguous", True),
                    # Store additional metrics that might be useful
                    "code_citations": eval_result.get("statistical_metrics", {}).get("code_citations", 0),
                    "evaluation": eval_result  # Store full evaluation for reference
                }
                all_qa_pairs.append(qa_pair)
            
            # Step 4: Identify any responses that need clarification using enhanced criteria
            unclear_responses = [(q, r, e) for q, r, e in 
                               zip(sub_questions, enhanced_responses, evaluation_results) 
                               if e.get("is_ambiguous", True)]
            
            if not unclear_responses:
                # All responses are clear, break the loop
                logger.info("All responses are clear. Proceeding to synthesize.")
                break
                
            # Step 5: Generate follow-up questions for unclear responses
            # Use the enhanced evaluator's follow-up questions if available
            logger.info(f"Found {len(unclear_responses)} unclear responses. Generating follow-up questions.")
            sub_questions = await self._generate_followup_questions(unclear_responses, user_query)
            
            if not sub_questions:
                # No follow-up questions could be generated
                logger.info("No follow-up questions generated. Breaking loop.")
                break
                
            iteration += 1
            logger.info(f"Starting iteration {iteration}/{self.max_iterations} with {len(sub_questions)} follow-up questions")
        
        # Step 6: Synthesize a final answer from all collected Q&A pairs
        final_answer = await self.synthesizer.synthesize(user_query, all_qa_pairs)
        logger.info("Final answer synthesized successfully")
        
        # Save the final answer to a file
        with open("responses/final_answer.txt", "w", encoding="utf-8") as f:
            f.write(final_answer)
        
        return {
            "result": {
                "answer": final_answer
            }
        }
    
    async def _process_sub_questions(self, sub_questions: List[str]) -> List[str]:
        """
        Process multiple sub-questions in parallel.
        
        Args:
            sub_questions: List of sub-questions to process
            
        Returns:
            List of responses from AskMod
        """
        tasks = [self.askmod_client.send_query(question) for question in sub_questions]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions that occurred during processing
        processed_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(f"Error processing question '{sub_questions[i]}': {str(response)}")
                processed_responses.append(f"Error: Could not get a response for this question. {str(response)}")
            else:
                processed_responses.append(response)
                
        return processed_responses
    
    async def _evaluate_responses(self, original_query: str, questions: List[str], 
                                 responses: List[str]) -> List[Dict[str, Any]]:
        """
        Evaluate the quality of responses using the enhanced evaluator.
        
        Args:
            original_query: The original user query
            questions: The sub-questions that were asked
            responses: The responses received from AskMod
            
        Returns:
            List of enhanced evaluation results for each response
        """
        evaluation_results = []
        
        for question, response in zip(questions, responses):
            try:
                # Use the enhanced evaluator which requires all three parameters
                result = await self.evaluator.evaluate_response(
                    response_text=response,
                    original_query=original_query,
                    sub_question=question
                )
                evaluation_results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating response for question '{question}': {str(e)}")
                # Create a fallback evaluation result
                evaluation_results.append({
                    "combined_score": 0.5,
                    "is_ambiguous": True,
                    "issues": [f"Evaluation error: {str(e)}"],
                    "follow_up_questions": [],
                    "confidence": 0.0,
                    "statistical_metrics": {"query_term_overlap": 0.5}
                })
        
        return evaluation_results
    
    async def _generate_followup_questions(self, unclear_responses: List[Tuple[str, str, Dict[str, Any]]], 
                                         original_query: str) -> List[str]:
        """
        Generate follow-up questions for unclear responses.
        
        Args:
            unclear_responses: List of tuples containing (question, response, evaluation)
            original_query: The original user query
            
        Returns:
            List of follow-up questions
        """
        follow_up_questions = []
        
        for question, response, evaluation in unclear_responses:
            # First, check if the enhanced evaluator provided follow-up questions directly
            if "follow_up_questions" in evaluation and evaluation["follow_up_questions"]:
                # Use pre-generated questions from enhanced evaluator
                follow_up_questions.extend(evaluation["follow_up_questions"][:2])  # Limit to 2 per response
                logger.info(f"Using pre-generated follow-up questions from enhanced evaluator for: {question}")
                continue
                
            # Fall back to original approach if no pre-generated questions
            issues = evaluation.get("issues", [])
            issue_str = "; ".join(issues)
            
            # Create a prompt for generating a follow-up question
            prompt = PromptTemplate(
                template="""Based on the following:
- Original User Query: {original_query}
- Specific Question: {question}
- Response: {response}
- Issues: {issues}

Generate a more specific follow-up question that would help clarify the ambiguities or address the issues identified.
The follow-up question should be focused, precise, and directly related to resolving the issues in the context of the original query.

Follow-up question:""",
                input_variables=["original_query", "question", "response", "issues"]
            )
            
            # Generate the follow-up question
            chain = prompt | self.llm
            follow_up = await chain.ainvoke({
                "original_query": original_query,
                "question": question,
                "response": response,
                "issues": issue_str
            })
            
            # Extract the text from the response
            if hasattr(follow_up, 'content'):
                follow_up_text = follow_up.content
            else:
                follow_up_text = str(follow_up)
            
            # Clean up the response
            follow_up_text = follow_up_text.strip()
            
            follow_up_questions.append(follow_up_text)
        
        return follow_up_questions