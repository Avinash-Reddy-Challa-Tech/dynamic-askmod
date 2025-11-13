"""
Intelligent Orchestrator for AskMod RAG System

This module implements a truly intelligent orchestrator that dynamically decides the next steps:
1. Takes a user query
2. Makes initial RAG call to get context
3. Dynamically decides whether to:
   - Generate sub-questions
   - Fetch specific code citations
   - Generate follow-up questions
   - Synthesize a final answer
4. Uses structured JSON decision making to determine next steps
"""

import asyncio
import json
import logging
import os
import re
from typing import List, Dict, Any, Optional, Tuple, Set

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
# from appmod_rag_tool import AppModRagTool

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OrchestratorDecision(BaseModel):
    """Schema for orchestrator decision based on response analysis."""
    decision: str = Field(
        description="Next action to take: 'fetch_code', 'generate_questions', 'follow_up', or 'synthesize'"
    )
    citations_to_fetch: List[str] = Field(
        default_factory=list,
        description="List of specific code citations to fetch (file paths)"
    )
    reason: str = Field(
        description="Reasoning behind the decision"
    )
    confidence: float = Field(
        description="Confidence score for this decision (0-1)"
    )
    generate_questions_from_rag: bool = Field(
        default=False, 
        description="Whether to generate new questions based on RAG context"
    )
    follow_up_questions: List[str] = Field(
        default_factory=list,
        description="List of follow-up questions if decision is 'follow_up'"
    )
    is_complete: bool = Field(
        default=False,
        description="Whether we have enough information to synthesize a final answer"
    )

class IntelligentOrchestrator:
    """
    An intelligent orchestrator that dynamically decides the next steps in the RAG process.
    """
    def __init__(self, 
                askmod_client: AskModClient, 
                decomposer: QueryDecomposer,
                evaluator: EnhancedResponseEvaluator, 
                synthesizer: ResponseSynthesizer,
                appmod_rag_tool: Optional[Any] = None,
                llm: Optional[ChatGoogleGenerativeAI] = None, 
                max_iterations: int = 3,
                code_extractor: Optional[CodeExtractor] = None):
        """
        Initialize the IntelligentOrchestrator.
        
        Args:
            askmod_client: Client for interacting with AskMod API
            decomposer: Component for breaking down queries into sub-questions
            evaluator: Enhanced evaluator for response quality
            synthesizer: Component for synthesizing responses into a final answer
            appmod_rag_tool: Tool for performing initial RAG context gathering
            llm: Language model for decision making (if None, a default model is used)
            max_iterations: Maximum number of iterations for resolving ambiguities
            code_extractor: Component for extracting code from citation links (if None, a default is created)
        """
        self.askmod_client = askmod_client
        self.decomposer = decomposer
        self.evaluator = evaluator
        self.synthesizer = synthesizer
        self.appmod_rag_tool = appmod_rag_tool
        self.llm = llm or ChatGoogleGenerativeAI(temperature=0.1)
        self.max_iterations = max_iterations
        self.code_extractor = code_extractor or CodeExtractor()
        
        # Create directories for storing intermediate data
        os.makedirs("responses", exist_ok=True)
        os.makedirs("decisions", exist_ok=True)
        os.makedirs("rag_context", exist_ok=True)
        os.makedirs("code_citations", exist_ok=True)
        
        # Initialize the decision parser
        self.decision_parser = JsonOutputParser(pydantic_object=OrchestratorDecision)
        
        # Create the decision prompt template
        self.decision_prompt_template = PromptTemplate(
            template="""You are an intelligent orchestrator that decides the next steps in a code documentation RAG system.

Original User Query: {original_query}

CONTEXT INFORMATION:
{context_information}

CURRENT RESPONSES:
{current_responses}

RETRIEVED CODE (if any):
{retrieved_code}

Your task is to decide the next step based on a careful analysis of the above information.

Possible decisions:
1. "fetch_code" - If you need to examine specific code files to better understand the responses
2. "generate_questions" - If the current information is insufficient and you need more context
3. "follow_up" - If you need clarification on specific aspects of the responses
4. "synthesize" - If you have enough information to provide a complete answer

For each decision, provide:
- A list of specific code citations to fetch (if decision is "fetch_code")
- Whether to generate new questions from RAG context (if decision is "generate_questions")
- A list of follow-up questions (if decision is "follow_up")
- Whether the information is complete enough to synthesize a final answer

Respond with ONLY a JSON object in the following format:
{format_instructions}""",
            input_variables=["original_query", "context_information", "current_responses", "retrieved_code"],
            partial_variables={"format_instructions": "{\"decision\": \"fetch_code|generate_questions|follow_up|synthesize\", \"citations_to_fetch\": [\"file/path1\", \"file/path2\"], \"reason\": \"Explanation of why this decision was made\", \"confidence\": 0.85, \"generate_questions_from_rag\": true|false, \"follow_up_questions\": [\"question1\", \"question2\"], \"is_complete\": true|false}"}
        )

    async def _get_rag_context(self, user_query: str, task_id: str, user_id: str, database_index: str = None) -> Dict[str, Any]:
        """
        Get initial RAG context for the user query.
        
        Args:
            user_query: The original user query
            task_id: Task ID for the RAG API call
            user_id: User ID for the RAG API call
            database_index: Database index (optional)
            
        Returns:
            Dictionary containing RAG context information
        """
        logger.info(f"Getting RAG context for query: {user_query}")
        if 0>1:
        # if self.appmod_rag_tool:
            # Format the query as a question dictionary
            question_dict = [{"question": user_query}]
            
            # try:
            #     # Call the AppModRagTool
            #     rag_responses, header_content, assistant_documents = self.appmod_rag_tool.run(
            #         image_to_questions=question_dict,
            #         user_id=user_id,
            #         git_url="",  # These can be configured based on your specific requirements
            #         task_id=task_id,
            #         feature_name="",
            #         git_token="",
            #         send_updates=False
            #     )
                
            #     # Extract the reference content from the response
            #     if rag_responses and isinstance(rag_responses, list) and len(rag_responses) > 0:
            #         reference = rag_responses[0].get("reference", "")
                    
            #         # Save RAG context to a file for reference
            #         with open(f"rag_context/rag_context.txt", "w", encoding="utf-8") as f:
            #             f.write(str(reference))
                    
            #         return {
            #             "rag_response": reference,
            #             "header_content": header_content,
            #             "assistant_documents": assistant_documents
            #         }
            #     else:
            #         logger.warning("No valid RAG responses received")
            #         return {"rag_response": "No context available", "header_content": "", "assistant_documents": []}
                    
            # except Exception as e:
            #     logger.error(f"Error getting RAG context: {str(e)}", exc_info=True)
            #     return {"rag_response": f"Error: {str(e)}", "header_content": "", "assistant_documents": []}
        else:
            # Fallback to direct AskMod query if AppModRagTool is not available
            try:
                response = await self.askmod_client.send_query(f"Provide comprehensive context for: {user_query}")
                
                # Save RAG context to a file for reference
                with open(f"rag_context/rag_context.txt", "w", encoding="utf-8") as f:
                    f.write(response)
                
                return {"rag_response": response, "header_content": "", "assistant_documents": []}
            except Exception as e:
                logger.error(f"Error getting AskMod context: {str(e)}", exc_info=True)
                return {"rag_response": f"Error: {str(e)}", "header_content": "", "assistant_documents": []}

    async def _make_orchestrator_decision(self, 
                                        original_query: str, 
                                        context_information: str, 
                                        current_responses: List[Dict[str, Any]], 
                                        retrieved_code: Dict[str, str]) -> OrchestratorDecision:
        """
        Make a decision about the next step in the orchestration process.
        
        Args:
            original_query: The original user query
            context_information: RAG context information
            current_responses: Current Q&A pairs with evaluation metrics
            retrieved_code: Code that has been retrieved so far
            
        Returns:
            OrchestratorDecision object containing the next action to take
        """
        logger.info("Making orchestrator decision based on current information")
        
        # Format current responses for the prompt
        formatted_responses = ""
        for i, response in enumerate(current_responses, 1):
            formatted_responses += f"Response {i}:\nQuestion: {response.get('question', 'N/A')}\n"
            formatted_responses += f"Answer: {response.get('answer', 'N/A')}\n"
            
            # Include evaluation metrics if available
            if 'evaluation' in response:
                eval_info = response['evaluation']
                formatted_responses += f"Clarity Score: {eval_info.get('combined_score', 'N/A')}\n"
                formatted_responses += f"Is Ambiguous: {eval_info.get('is_ambiguous', 'N/A')}\n"
                
                # Include identified issues
                issues = eval_info.get('issues', [])
                if issues:
                    formatted_responses += "Issues:\n"
                    for issue in issues:
                        formatted_responses += f"- {issue}\n"
            
            formatted_responses += "\n" + "-" * 40 + "\n\n"
        
        # Format retrieved code for the prompt
        formatted_code = ""
        for file_path, code in retrieved_code.items():
            formatted_code += f"File: {file_path}\n"
            # Limit code length for the prompt
            code_snippet = code
            formatted_code += f"```\n{code_snippet}\n```\n\n"
        
        if not formatted_code:
            formatted_code = "No code has been retrieved yet."
        
        try:
            # Prepare the variables to pass to the prompt template
            variables = {
                "original_query": original_query,
                "context_information": context_information,  # Limit context length
                "current_responses": formatted_responses,
                "retrieved_code": formatted_code
            }
            
            # Format the prompt using the template
            prompt_text = self.decision_prompt_template.format(**variables)
            
            # Call the LLM directly
            response = await self.llm.ainvoke(prompt_text)
            
            # Extract the text from the response
            if hasattr(response, 'content'):
                decision_text = response.content
            else:
                decision_text = str(response)
            
            # Save the decision for debugging
            with open(f"decisions/decision_{len(current_responses)}.json", "w", encoding="utf-8") as f:
                f.write(decision_text)
            
            # Parse the decision
            try:
                # First, try to extract JSON from markdown code blocks
                json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', decision_text, re.DOTALL)
                if json_match:
                    decision_json = json.loads(json_match.group(1).strip())
                else:
                    # Try direct JSON parsing
                    decision_json = json.loads(decision_text.strip())
                
                # Create OrchestratorDecision object
                decision = OrchestratorDecision(**decision_json)
                logger.info(f"Decision: {decision.decision}, Confidence: {decision.confidence}, Complete: {decision.is_complete}")
                return decision
                
            except (json.JSONDecodeError, Exception) as e:
                logger.error(f"Error parsing decision JSON: {str(e)}", exc_info=True)
                # Fall back to a default decision
                return OrchestratorDecision(
                    decision="generate_questions",
                    citations_to_fetch=[],
                    reason=f"Error parsing decision: {str(e)}",
                    confidence=0.5,
                    generate_questions_from_rag=True,
                    follow_up_questions=[],
                    is_complete=False
                )
                
        except Exception as e:
            logger.error(f"Error making orchestrator decision: {str(e)}", exc_info=True)
            # Fall back to a default decision
            return OrchestratorDecision(
                decision="generate_questions",
                citations_to_fetch=[],
                reason=f"Error making decision: {str(e)}",
                confidence=0.5,
                generate_questions_from_rag=True,
                follow_up_questions=[],
                is_complete=False
            )

    async def _fetch_code_for_citations(self, citations: List[str]) -> Dict[str, str]:
        """
        Fetch code for specific file citations.
        
        Args:
            citations: List of file paths to fetch
            
        Returns:
            Dictionary mapping file paths to their code content
        """
        logger.info(f"Fetching code for {len(citations)} citations")
        
        code_files = {}
        
        for file_path in citations:
            try:
                # Call the API to get file content
                # This is a placeholder - replace with actual code to fetch files from your system
                # For example, you might use a GitHub API client or a custom endpoint
                
                # Placeholder implementation
                code_content = await self._get_file_content(file_path)
                
                if code_content:
                    code_files[file_path] = code_content
                    
                    # Save code to file for reference
                    file_name = file_path.split("/")[-1]
                    with open(f"code_citations/{file_name}", "w", encoding="utf-8") as f:
                        f.write(code_content)
                    
                    logger.info(f"Successfully fetched code for {file_path}")
                else:
                    logger.warning(f"No code content returned for {file_path}")
            except Exception as e:
                logger.error(f"Error fetching code for {file_path}: {str(e)}")
        
        return code_files

    async def _get_file_content(self, file_path: str) -> str:
        """
        Get the content of a code file using the CodeExtractor.
        
        Args:
            file_path: Path to the file
            
        Returns:
            String containing the file content
        """
        logger.info(f"Getting file content for {file_path}")
        
        try:
            # Create a URL tuple format that CodeExtractor expects (display_text, url)
            display_text = file_path.split('/')[-1]  # Use filename as display text
            url_tuple = (display_text, file_path)
            
            # Use the CodeExtractor's fetch_code_content method
            result = await self.code_extractor.fetch_code_content(url_tuple)
            
            if result and "content" in result and result["content"]:
                logger.info(f"Successfully retrieved content for {file_path}")
                return result["content"]
            else:
                error_msg = result.get("error", "Unknown error") if result else "No result returned"
                logger.warning(f"Failed to get content for {file_path}: {error_msg}")
                return f"// Could not retrieve file content: {error_msg}"
                
        except Exception as e:
            logger.error(f"Error getting file content for {file_path}: {str(e)}", exc_info=True)
            return f"// Error fetching file content: {str(e)}"
    
    async def _generate_questions_from_rag(self, rag_context: str, original_query: str) -> List[str]:
        """
        Generate questions based on RAG context.
        
        Args:
            rag_context: RAG context information
            original_query: The original user query
            
        Returns:
            List of generated questions
        """
        logger.info("Generating questions based on RAG context")
        
        # Create a prompt for generating questions
        prompt_template = """You are an expert at generating targeted questions about code and software systems.

Original User Query: {original_query}

Context Information from RAG:
{rag_context}

Based on this information, generate 3 specific questions that will help gather detailed information to answer the original query.
The questions should:
1. Target different aspects of the original query
2. Be specific about code components, functions, or structures mentioned in the context
3. Help fill gaps in the RAG context
4. Use technical terminology appropriate to the codebase
5. Lead to concrete, factual answers about the code's functionality

Generate ONLY the list of 3 questions, each prefixed with "TRIGGER DOMAIN KNOWLEDGE AGENT:", as follows:

TRIGGER DOMAIN KNOWLEDGE AGENT: [Question 1]
TRIGGER DOMAIN KNOWLEDGE AGENT: [Question 2]
TRIGGER DOMAIN KNOWLEDGE AGENT: [Question 3]"""
        
        # Format the prompt with the variables
        prompt = prompt_template.format(
            original_query=original_query,
            rag_context=rag_context  # Limit context length
        )
        
        try:
            # Call the LLM directly with the formatted prompt
            response = await self.llm.ainvoke(prompt)
            
            # Extract the text from the response
            if hasattr(response, 'content'):
                questions_text = response.content
            else:
                questions_text = str(response)
                
            # Extract questions from the response
            questions = []
            for line in questions_text.strip().split('\n'):
                line = line.strip()
                if line.startswith("TRIGGER DOMAIN KNOWLEDGE AGENT:"):
                    question = line[len("TRIGGER DOMAIN KNOWLEDGE AGENT:"):].strip()
                    questions.append(f"TRIGGER DOMAIN KNOWLEDGE AGENT: {question}")
            
            # If we couldn't extract questions properly, try a fallback approach
            if not questions:
                # Just split by line and take up to 3 non-empty lines
                questions = [line.strip() for line in questions_text.split('\n') 
                            if line.strip() and not line.strip().startswith('#')]
                questions = [f"TRIGGER DOMAIN KNOWLEDGE AGENT: {q}" if not q.startswith("TRIGGER DOMAIN KNOWLEDGE AGENT:") else q 
                            for q in questions[:3]]
            
            logger.info(f"Generated {len(questions)} questions from RAG context")
            return questions
            
        except Exception as e:
            logger.error(f"Error generating questions from RAG: {str(e)}", exc_info=True)
            # Fall back to basic questions
            return [
                f"TRIGGER DOMAIN KNOWLEDGE AGENT: What are the main components involved in {original_query}?",
                f"TRIGGER DOMAIN KNOWLEDGE AGENT: How is the functionality of {original_query} implemented in the code?",
                f"TRIGGER DOMAIN KNOWLEDGE AGENT: What are the key interactions and dependencies for {original_query}?"
            ]

    async def process_query(self, user_query: str, task_id: str, user_id: str, database_index: str = None) -> Dict[str, Any]:
        """
        Process a user query through the intelligent orchestration pipeline.
        
        Args:
            user_query: The original user query
            task_id: Task ID for the RAG API call
            user_id: User ID for the RAG API call
            database_index: Database index (optional)
            
        Returns:
            Dict containing the final synthesized answer
        """
        logger.info(f"Processing user query: {user_query}")
        
        # Step 1: Get initial RAG context for the query
        rag_result = await self._get_rag_context(user_query, task_id, user_id, database_index)
        rag_context = rag_result.get("rag_response", "")
        header_content = rag_result.get("header_content", "")
        assistant_documents = rag_result.get("assistant_documents", [])
        
        logger.info(f"Received RAG context (length: {len(str(rag_context))})")
        
        # Initialize containers for our process
        all_qa_pairs = []
        retrieved_code = {}
        fetched_citations = set()
        iteration = 0
        
        while iteration < self.max_iterations:
            # Step 2: Make orchestrator decision
            decision = await self._make_orchestrator_decision(
                original_query=user_query,
                context_information=rag_context,
                current_responses=all_qa_pairs,
                retrieved_code=retrieved_code
            )
            
            # Log the decision
            logger.info(f"Iteration {iteration} - Decision: {decision.decision}")
            logger.info(f"Reason: {decision.reason}")
            
            # Step 3: Execute the decision
            if decision.decision == "fetch_code" and decision.citations_to_fetch:
                # Filter out citations we've already fetched
                new_citations = [c for c in decision.citations_to_fetch if c not in fetched_citations]
                
                if new_citations:
                    logger.info(f"Fetching code for {len(new_citations)} new citations")
                    code_files = await self._fetch_code_for_citations(new_citations)
                    retrieved_code.update(code_files)
                    fetched_citations.update(new_citations)
                else:
                    logger.info("No new citations to fetch")
                
            elif decision.decision == "generate_questions":
                # Generate questions based on RAG context
                if decision.generate_questions_from_rag:
                    logger.info("Generating questions from RAG context")
                    sub_questions = await self._generate_questions_from_rag(rag_context, user_query)
                else:
                    logger.info("Decomposing query into sub-questions")
                    sub_questions = await self.decomposer.generate_sub_questions(user_query)
                
                logger.info(f"Generated {len(sub_questions)} questions")
                
                # Process the sub-questions in parallel
                responses = await self._process_sub_questions(sub_questions)
                
                # Evaluate the responses
                evaluation_results = await self._evaluate_responses(user_query, sub_questions, responses)
                
                # Add the Q&A pairs to our collection
                for q, r, eval_result in zip(sub_questions, responses, evaluation_results):
                    qa_pair = {
                        "question": q,
                        "answer": r,
                        "clarity_score": eval_result.get("combined_score", 0),
                        "relevance_score": eval_result.get("statistical_metrics", {}).get("query_term_overlap", 0),
                        "is_clear": not eval_result.get("is_ambiguous", True),
                        "code_citations": eval_result.get("statistical_metrics", {}).get("code_citations", 0),
                        "evaluation": eval_result
                    }
                    all_qa_pairs.append(qa_pair)
                
            elif decision.decision == "follow_up" and decision.follow_up_questions:
                logger.info(f"Processing {len(decision.follow_up_questions)} follow-up questions")
                
                # Process the follow-up questions
                responses = await self._process_sub_questions(decision.follow_up_questions)
                
                # Evaluate the responses
                evaluation_results = await self._evaluate_responses(
                    user_query, 
                    decision.follow_up_questions, 
                    responses
                )
                
                # Add the Q&A pairs to our collection
                for q, r, eval_result in zip(decision.follow_up_questions, responses, evaluation_results):
                    qa_pair = {
                        "question": q,
                        "answer": r,
                        "clarity_score": eval_result.get("combined_score", 0),
                        "relevance_score": eval_result.get("statistical_metrics", {}).get("query_term_overlap", 0),
                        "is_clear": not eval_result.get("is_ambiguous", True),
                        "code_citations": eval_result.get("statistical_metrics", {}).get("code_citations", 0),
                        "evaluation": eval_result
                    }
                    all_qa_pairs.append(qa_pair)
            
            # Step 4: Check if we can synthesize a final answer
            if decision.is_complete or decision.decision == "synthesize":
                logger.info("Orchestrator has determined we have enough information to synthesize a final answer")
                break
            
            # Increment iteration counter
            iteration += 1
            logger.info(f"Completed iteration {iteration}/{self.max_iterations}")
            
            # Break if we've reached the maximum number of iterations
            if iteration >= self.max_iterations:
                logger.info(f"Reached maximum iterations ({self.max_iterations}), proceeding to synthesis")
                break
        
        # Step 5: Synthesize a final answer from all collected Q&A pairs and RAG context
        logger.info(f"Synthesizing final answer from {len(all_qa_pairs)} Q&A pairs and RAG context")
        
        # Enhance the QA pairs with RAG context
        enhanced_qa_pairs = all_qa_pairs.copy()
        if rag_context and rag_context != "No context available" and not rag_context.startswith("Error:"):
            # Add RAG context as an additional "response" to consider during synthesis
            rag_qa_pair = {
                "question": "What is the overall context for this query?",
                "answer": rag_context,
                "clarity_score": 0.9,  # Assume RAG context is clear
                "relevance_score": 0.9,  # Assume RAG context is relevant
                "is_clear": True,
                "code_citations": 0,
                "evaluation": {
                    "combined_score": 0.9,
                    "is_ambiguous": False,
                    "issues": [],
                    "follow_up_questions": [],
                    "confidence": 0.9
                }
            }
            enhanced_qa_pairs.append(rag_qa_pair)
        
        # Include retrieved code in the synthesis process
        for file_path, code_content in retrieved_code.items():
            code_qa_pair = {
                "question": f"What is the content of {file_path}?",
                "answer": f"```\n{code_content}\n```",
                "clarity_score": 0.95,  # Code is inherently clear as it's the actual source
                "relevance_score": 0.9,  # Assume the code is relevant (we only fetched relevant files)
                "is_clear": True,
                "code_citations": 1,
                "evaluation": {
                    "combined_score": 0.95,
                    "is_ambiguous": False,
                    "issues": [],
                    "follow_up_questions": [],
                    "confidence": 0.95
                }
            }
            enhanced_qa_pairs.append(code_qa_pair)
        
        # Synthesize the final answer
        final_answer = await self.synthesizer.synthesize(user_query, enhanced_qa_pairs)
        logger.info("Final answer synthesized successfully")
        
        # Save the final answer to a file
        with open("responses/final_answer.txt", "w", encoding="utf-8") as f:
            f.write(final_answer)
        
        return {
            "result": {
                "answer": final_answer,
                "rag_context": rag_context,
                "qa_pairs_count": len(all_qa_pairs),
                "iterations": iteration,
                "retrieved_code_count": len(retrieved_code),
                "header_content": header_content,
                "assistant_documents": assistant_documents
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