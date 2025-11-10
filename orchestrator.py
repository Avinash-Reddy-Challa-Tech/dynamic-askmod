"""
Orchestrator for AskMod RAG System

This module implements an intelligent orchestrator that:
1. Takes a user query
2. Generates more targeted sub-questions
3. Sends these sub-questions to AskMod in parallel
4. Evaluates the responses for quality and clarity
5. Generates follow-up questions if needed
6. Synthesizes all responses into a comprehensive answer
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, Tuple

import aiohttp
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from pydantic.v1 import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from query_decomposer import QueryDecomposer
from response_evaluator import ResponseEvaluator
from response_synthesizer import ResponseSynthesizer
from askmod_client import AskModClient

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Orchestrator:
    """
    Orchestrates the process of breaking down queries, getting responses, and synthesizing a final answer.
    """
    def __init__(self, askmod_client: AskModClient, decomposer: QueryDecomposer,
                 evaluator: ResponseEvaluator, synthesizer: ResponseSynthesizer,
                 llm: Optional[ChatGoogleGenerativeAI] = None, max_iterations: int = 2):
        """
        Initialize the Orchestrator.
        
        Args:
            askmod_client: Client for interacting with AskMod API
            decomposer: Component for breaking down queries into sub-questions
            evaluator: Component for evaluating response quality
            synthesizer: Component for synthesizing responses into a final answer
            llm: Language model for generating questions (if None, a default model is used)
            max_iterations: Maximum number of iterations for resolving ambiguities
        """
        self.askmod_client = askmod_client
        self.decomposer = decomposer
        self.evaluator = evaluator
        self.synthesizer = synthesizer
        self.llm = llm or ChatGoogleGenerativeAI(temperature=0.1)
        self.max_iterations = max_iterations
        
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
            
            # Step 3: Evaluate responses for quality and clarity
            evaluation_results = await self._evaluate_responses(sub_questions, responses)
            
            # Add the current Q&A pairs to our collection
            for q, r, eval_result in zip(sub_questions, responses, evaluation_results):
                all_qa_pairs.append({
                    "question": q,
                    "answer": r,
                    "clarity_score": eval_result.get("clarity_score", 0),
                    "relevance_score": eval_result.get("relevance_score", 0),
                    "is_clear": eval_result.get("is_clear", False)
                })
            
            # Step 4: Identify any responses that need clarification
            unclear_responses = [(q, r, e) for q, r, e in 
                               zip(sub_questions, responses, evaluation_results) 
                               if not e.get("is_clear", False)]
            
            if not unclear_responses:
                # All responses are clear, break the loop
                logger.info("All responses are clear. Proceeding to synthesize.")
                break
                
            # Step 5: Generate follow-up questions for unclear responses
            logger.info(f"Found {len(unclear_responses)} unclear responses. Generating follow-up questions.")
            sub_questions = await self._generate_followup_questions(unclear_responses)
            
            if not sub_questions:
                # No follow-up questions could be generated
                logger.info("No follow-up questions generated. Breaking loop.")
                break
                
            iteration += 1
            logger.info(f"Starting iteration {iteration}/{self.max_iterations} with {len(sub_questions)} follow-up questions")
        
        # Step 6: Synthesize a final answer from all collected Q&A pairs
        final_answer = await self.synthesizer.synthesize(user_query, all_qa_pairs)
        logger.info("Final answer synthesized successfully")
        
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
    
    async def _evaluate_responses(self, questions: List[str], responses: List[str]) -> List[Dict[str, Any]]:
        """
        Evaluate the quality of responses.
        
        Args:
            questions: The questions that were asked
            responses: The responses received from AskMod
            
        Returns:
            List of evaluation results for each response
        """
        tasks = [self.evaluator.evaluate(q, r) for q, r in zip(questions, responses)]
        return await asyncio.gather(*tasks)
    
    async def _generate_followup_questions(self, unclear_responses: List[Tuple[str, str, Dict[str, Any]]]) -> List[str]:
        """
        Generate follow-up questions for unclear responses.
        
        Args:
            unclear_responses: List of tuples containing (question, response, evaluation)
            
        Returns:
            List of follow-up questions
        """
        follow_up_questions = []
        
        for question, response, evaluation in unclear_responses:
            issues = evaluation.get("issues", [])
            issue_str = "; ".join(issues)
            
            # Create a prompt for generating a follow-up question
            prompt = PromptTemplate(
                template="""Based on the following:
- Original Question: {question}
- Response: {response}
- Issues: {issues}

Generate a more specific follow-up question that would help clarify the ambiguities or address the issues identified.
The follow-up question should be focused, precise, and directly related to the original question but designed to elicit a clearer response.

Follow-up question:""",
                input_variables=["question", "response", "issues"]
            )
            
            # Generate the follow-up question
            chain = prompt | self.llm
            follow_up = await chain.ainvoke({
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