"""
Intelligent Orchestrator for AskMod RAG System with Source and Target Repository Support

This module implements an intelligent orchestrator that dynamically decides the next steps:
1. Takes a user query about implementing a feature from a source repository to a target repository
2. Makes initial RAG calls to get context from both repositories
3. Dynamically generates paired questions for source and target repositories
4. Synthesizes a comprehensive answer comparing the implementations
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
    target_repo: bool = Field(
        default=False,
        description="Whether the next action targets the target repository"
    )

class RepoConfig(BaseModel):
    """Configuration for a repository."""
    repo_url: str = Field(
        description="The URL of the repository"
    )
    assistant_name: str = Field(
        description="The AskMod assistant name for this repository"
    )
    database_index: str = Field(
        description="The database index for this repository"
    )
    organization_name: str = Field(
        description="The organization name for this repository"
    )
    task_id: str = Field(
        description="The task ID for this repository"
    )
    description: str = Field(
        default="",
        description="Description of the repository"
    )

class SubQuestionPair(BaseModel):
    """Schema for paired source and target questions."""
    source_question: str = Field(
        description="Question for the source repository"
    )
    target_question: str = Field(
        description="Question for the target repository"
    )
    context: str = Field(
        default="",
        description="Context for the paired questions"
    )

class IntelligentOrchestrator:
    """
    An intelligent orchestrator that works with source and target repositories.
    """
    def __init__(self, 
                askmod_client: AskModClient, 
                decomposer: QueryDecomposer,
                evaluator: EnhancedResponseEvaluator, 
                synthesizer: ResponseSynthesizer,
                appmod_rag_tool: Optional[Any] = None,
                llm: Optional[ChatGoogleGenerativeAI] = None, 
                max_iterations: int = 3,
                code_extractor: Optional[CodeExtractor] = None,
                source_repo_config: Optional[Dict[str, Any]] = None,
                target_repo_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the IntelligentOrchestrator with source and target repository support.
        
        Args:
            askmod_client: Client for interacting with AskMod API
            decomposer: Component for breaking down queries into sub-questions
            evaluator: Enhanced evaluator for response quality
            synthesizer: Component for synthesizing responses into a final answer
            appmod_rag_tool: Tool for performing initial RAG context gathering
            llm: Language model for decision making (if None, a default model is used)
            max_iterations: Maximum number of iterations for resolving ambiguities
            code_extractor: Component for extracting code from citation links (if None, a default is created)
            source_repo_config: Configuration for the source repository
            target_repo_config: Configuration for the target repository
        """
        self.askmod_client = askmod_client
        self.decomposer = decomposer
        self.evaluator = evaluator
        self.synthesizer = synthesizer
        self.appmod_rag_tool = appmod_rag_tool
        self.llm = llm or ChatGoogleGenerativeAI(temperature=0.1)
        self.max_iterations = max_iterations
        self.code_extractor = code_extractor or CodeExtractor()
        
        # Default source repository configuration (creative-workspace-backend)
        self.source_repo = RepoConfig(
            repo_url=source_repo_config.get("repo_url", "https://github.com/Techolution/creative-workspace-backend"),
            assistant_name=source_repo_config.get("assistant_name", "appmod2a5bd9f2dev"),
            database_index=source_repo_config.get("database_index", "appmod2a5bd9f2dev"),
            organization_name=source_repo_config.get("organization_name", "84lumber"),
            task_id=source_repo_config.get("task_id", "88bb18aa-2a7d-42bb-9a66-bf6282ae44a3"),
            description=source_repo_config.get("description", "Creative Workspace Backend")
        ) if source_repo_config else RepoConfig(
            repo_url="https://github.com/Techolution/creative-workspace-backend",
            assistant_name="appmod2a5bd9f2dev",
            database_index="appmod2a5bd9f2dev",
            organization_name="84lumber",
            task_id="88bb18aa-2a7d-42bb-9a66-bf6282ae44a3",
            description="Creative Workspace Backend"
        )
        
        # Default target repository configuration (userstory dashboard)
        self.target_repo = RepoConfig(
            repo_url=target_repo_config.get("repo_url", "https://github.com/Avinash-Reddy-Challa-Tech/dashboard"),
            assistant_name=target_repo_config.get("assistant_name", "appmod7c0dcde3dev"),
            database_index=target_repo_config.get("database_index", "appmod7c0dcde3dev"),
            organization_name=target_repo_config.get("organization_name", "techolution"),
            task_id=target_repo_config.get("task_id", "74f6cb95-a616-44d2-bb82-04731a1beefe"),
            description=target_repo_config.get("description", "Userstory Dashboard")
        ) if target_repo_config else RepoConfig(
            repo_url="https://github.com/Avinash-Reddy-Challa-Tech/dashboard",
            assistant_name="appmod7c0dcde3dev",
            database_index="appmod7c0dcde3dev",
            organization_name="techolution",
            task_id="74f6cb95-a616-44d2-bb82-04731a1beefe",
            description="Userstory Dashboard"
        )
        
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

SOURCE REPOSITORY CONTEXT:
{source_context}

TARGET REPOSITORY CONTEXT:
{target_context}

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
- Whether the next action targets the target repository

Respond with ONLY a JSON object in the following format:
{format_instructions}""",
            input_variables=["original_query", "source_context", "target_context", "current_responses", "retrieved_code"],
            partial_variables={"format_instructions": "{\"decision\": \"fetch_code|generate_questions|follow_up|synthesize\", \"citations_to_fetch\": [\"file/path1\", \"file/path2\"], \"reason\": \"Explanation of why this decision was made\", \"confidence\": 0.85, \"generate_questions_from_rag\": true|false, \"follow_up_questions\": [\"question1\", \"question2\"], \"is_complete\": true|false, \"target_repo\": true|false}"}
        )

        # Create the paired questions generator prompt template
        self.paired_questions_prompt_template = PromptTemplate(
            template="""You are an expert at generating paired questions about code to understand how to port features from one repository to another.

Original User Query: {original_query}

SOURCE REPOSITORY: {source_repo_desc}
TARGET REPOSITORY: {target_repo_desc}

Your task is to generate {num_pairs} pairs of targeted questions where:
1. Each source question focuses on understanding a feature in the source repository
2. Each target question focuses on how to implement that same feature in the target repository
3. The questions are simple, clear and directly address specific aspects of the original query
4. The questions are designed to get direct answers about code implementation, not general explanations

IMPORTANT GUIDELINES:
- Make questions VERY SIMPLE and focused on one specific aspect at a time
- Focus on HOW code is implemented, not WHY
- Ask about specific methods, functions, routes, or components
- Questions must be phrased to get factual responses about code implementation
- AVOID open-ended questions that would result in lengthy explanations
- All questions must start with "TRIGGER DOMAIN KNOWLEDGE AGENT: "

Example paired questions:
1. Source: "User Query: TRIGGER DOMAIN KNOWLEDGE AGENT: How is the authentication implemented in the source repository?"
   Target: "User Query: TRIGGER DOMAIN KNOWLEDGE AGENT: What authentication mechanisms exist in the target repository?"

2. Source: "User Query: TRIGGER DOMAIN KNOWLEDGE AGENT: What route handles PDF downloads in the source repository?"
   Target: "User Query: TRIGGER DOMAIN KNOWLEDGE AGENT: Are there any existing download routes in the target repository?"

Please generate {num_pairs} pairs of questions that will help understand how to implement a feature from the source repository in the target repository.

Format your response as JSON:
{{
  "question_pairs": [
    {{
      "source_question": "QUERY: TRIGGER DOMAIN KNOWLEDGE AGENT: [question for source repo]",
      "target_question": "QUERY: TRIGGER DOMAIN KNOWLEDGE AGENT: [question for target repo]",
      "context": "Brief explanation of what this pair is trying to determine"
    }},
    {{
      "source_question": "QUERY: TRIGGER DOMAIN KNOWLEDGE AGENT: [question for source repo]",
      "target_question": "QUERY: TRIGGER DOMAIN KNOWLEDGE AGENT: [question for target repo]",
      "context": "Brief explanation of what this pair is trying to determine"
    }},
    ...
  ]
}}
""",
            input_variables=["original_query", "source_repo_desc", "target_repo_desc", "num_pairs"]
        )

    def _build_askmod_payload(self, repo_config: RepoConfig, is_target: bool = False) -> Dict[str, Any]:
        """
        Build the payload for the AskMod client with repository-specific configuration.
        
        Args:
            repo_config: Repository configuration
            is_target: Whether this is for the target repository
            
        Returns:
            Dictionary containing the AskMod payload parameters
        """
        return {
            "organization_name": repo_config.organization_name,
            "task_id": repo_config.task_id,
            "database_index": repo_config.database_index,
            "assistant_name": repo_config.assistant_name,
            "repo_url": repo_config.repo_url,
            "is_target_repo": is_target
        }

    async def _get_rag_context(self, user_query: str, is_target_repo: bool = False) -> Dict[str, Any]:
        """
        Get initial RAG context for the user query using the RAG tool.
        Does NOT make an AskMod call, returns empty context if RAG tool is not available.
        
        Args:
            user_query: The original user query
            is_target_repo: Whether to use the target repository
            
        Returns:
            Dictionary containing RAG context information with empty values if no RAG tool available
        """
        # Determine repository configuration
        if is_target_repo:
            repo_config = {
                "repo_url": self.target_repo.repo_url,
                "taskId": self.target_repo.task_id,
                "organization_name": self.target_repo.organization_name
            }
        else:
            repo_config = {
                "repo_url": self.source_repo.repo_url,
                "taskId": self.source_repo.task_id,
                "organization_name": self.source_repo.organization_name
            }
        
        logger.info(f"Getting RAG context for {'target' if is_target_repo else 'source'} repository")
        
        # Use the AppModRagTool if available
        if self.appmod_rag_tool:
            try:
                # Format the query as a question dictionary
                question_dict = [{"question": user_query}]
                
                # Call the AppModRagTool
                rag_responses, header_content, assistant_documents = self.appmod_rag_tool.run(
                    image_to_questions=question_dict,
                    user_id=self.askmod_client.current_user_id,  # Use current_user_id instead of user_id
                    git_url=repo_config.get("repo_url", ""),
                    task_id=repo_config.get("taskId", ""),  # Use taskId instead of task_id
                    feature_name="",
                    git_token=self.askmod_client.github_token,
                    send_updates=False
                )
                
                # Extract the reference content from the response
                if rag_responses and isinstance(rag_responses, list) and len(rag_responses) > 0:
                    reference = rag_responses[0].get("reference", "")
                    
                    # Save RAG context to a file for reference
                    os.makedirs("rag_context", exist_ok=True)
                    context_file = f"rag_context/{'target' if is_target_repo else 'source'}_context.txt"
                    with open(context_file, "w", encoding="utf-8") as f:
                        f.write(str(reference))
                    
                    logger.info(f"Successfully retrieved RAG context for {'target' if is_target_repo else 'source'} repository")
                    return {
                        "context": reference,
                        "header_content": header_content,
                        "assistant_documents": assistant_documents
                    }
                else:
                    logger.warning("No valid RAG responses received")
                    return {
                        "context": "",
                        "header_content": {},
                        "assistant_documents": []
                    }
                    
            except Exception as e:
                logger.error(f"Error getting RAG context: {str(e)}", exc_info=True)
                return {
                    "context": "",
                    "header_content": {},
                    "assistant_documents": []
                }
        else:
            # No AppModRagTool available, return empty context
            logger.info("No RAG tool available, returning empty context")
            return {
                "context": "",
                "header_content": {},
                "assistant_documents": []
            }

    async def _generate_paired_questions(self, 
                                        user_query: str, 
                                        source_context: str, 
                                        target_context: str, 
                                        num_pairs: int = 3) -> List[SubQuestionPair]:
        """
        Generate paired questions for source and target repositories.
        
        Args:
            user_query: The original user query
            source_context: RAG context from the source repository
            target_context: RAG context from the target repository
            num_pairs: Number of question pairs to generate
            
        Returns:
            List of paired questions
        """
        logger.info(f"Generating {num_pairs} paired questions for source and target repositories")
        
        # Create the prompt with repository descriptions and contexts
        prompt = self.paired_questions_prompt_template.format(
            original_query=user_query,
            source_repo_desc=f"{self.source_repo.description}\nContext: {source_context[:500]}...",
            target_repo_desc=f"{self.target_repo.description}\nContext: {target_context[:500]}...",
            num_pairs=num_pairs
        )
        
        try:
            # Call the LLM to generate paired questions
            response = await self.llm.ainvoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Extract JSON from the response
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON without the markdown code block
                json_match = re.search(r'({[\s\S]*})', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    logger.error("Could not extract JSON from LLM response")
                    return self._fallback_paired_questions(user_query, num_pairs)
            
            # Parse the JSON
            data = json.loads(json_str)
            
            # Create SubQuestionPair objects
            question_pairs = []
            for pair in data["question_pairs"]:
                question_pairs.append(SubQuestionPair(
                    source_question=pair["source_question"],
                    target_question=pair["target_question"],
                    context=pair.get("context", "")
                ))
            
            logger.info(f"Successfully generated {len(question_pairs)} paired questions")
            return question_pairs
            
        except Exception as e:
            logger.error(f"Error generating paired questions: {str(e)}")
            return self._fallback_paired_questions(user_query, num_pairs)
    
    def _fallback_paired_questions(self, user_query: str, num_pairs: int = 3) -> List[SubQuestionPair]:
        """
        Provide fallback paired questions if generation fails.
        
        Args:
            user_query: The original user query
            num_pairs: Number of question pairs to generate
            
        Returns:
            List of fallback paired questions
        """
        logger.info(f"Using fallback paired questions generation")
        
        # Extract key feature from the query (simple heuristic)
        feature_match = re.search(r'feature\s+(?:in|of)\s+([^.]+)', user_query, re.IGNORECASE)
        feature = feature_match.group(1) if feature_match else "the requested functionality"
        
        # Create basic fallback question pairs
        fallback_pairs = [
            SubQuestionPair(
                source_question=f"User Query: TRIGGER DOMAIN KNOWLEDGE AGENT: How is {feature} implemented in the code?",
                target_question=f"User Query: TRIGGER DOMAIN KNOWLEDGE AGENT: What existing components in the codebase could help implement {feature}?",
                context=f"Understanding the basic implementation of {feature}"
            ),
            SubQuestionPair(
                source_question=f"User Query: TRIGGER DOMAIN KNOWLEDGE AGENT: What API routes or endpoints handle {feature}?",
                target_question=f"User Query: TRIGGER DOMAIN KNOWLEDGE AGENT: What is the API structure for implementing new routes or endpoints?",
                context=f"Understanding the API structure for {feature}"
            ),
            SubQuestionPair(
                source_question=f"User Query: TRIGGER DOMAIN KNOWLEDGE AGENT: What database models or schemas are used for {feature}?",
                target_question=f"User Query: TRIGGER DOMAIN KNOWLEDGE AGENT: What database models or schemas exist in the target repository?",
                context=f"Understanding the data model for {feature}"
            )
        ]
        
        return fallback_pairs[:num_pairs]

    async def _process_paired_questions(self, 
                                       user_query: str,
                                       source_context: str,
                                       target_context: str,
                                       question_pairs: List[SubQuestionPair]) -> List[Dict[str, Any]]:
        """
        Process paired questions for source and target repositories.
        
        Args:
            user_query: The original user query
            source_context: RAG context from the source repository
            target_context: RAG context from the target repository
            question_pairs: List of paired questions to process
            
        Returns:
            List of results containing questions, answers, and evaluations
        """
        logger.info(f"Processing {len(question_pairs)} paired questions")
        
        all_qa_pairs = []
        
        for pair_index, pair in enumerate(question_pairs):
            logger.info(f"Processing pair {pair_index + 1}/{len(question_pairs)}")
            
            # Process source question
            source_question = pair.source_question
            source_enhanced = f"{source_question}\n\nContext Information:\n{source_context}"
            
            # FIXED: Use is_target=False for source repository
            source_response = await self.askmod_client.send_query(source_enhanced, is_target=False)
            print(f"Source Response: {source_response}")
            
            # Evaluate source response
            source_eval = await self.evaluator.evaluate_response(
                response_text=source_response,
                original_query=user_query,
                sub_question=source_question
            )
            
            # Create source QA pair
            source_qa_pair = {
                "question": source_question,
                "answer": source_response,
                "clarity_score": source_eval.get("combined_score", 0),
                "relevance_score": source_eval.get("statistical_metrics", {}).get("query_term_overlap", 0),
                "is_clear": not source_eval.get("is_ambiguous", True),
                "code_citations": source_eval.get("statistical_metrics", {}).get("code_citations", 0),
                "evaluation": source_eval,
                "repository": "source",
                "context": pair.context
            }
            all_qa_pairs.append(source_qa_pair)
            
            # Process target question with context from source response
            target_question = pair.target_question
            target_enhanced = f"{target_question}\n\nSource Repository Information:\n{source_response}\n\nContext Information:\n{target_context}"
            
            # FIXED: Use is_target=True for target repository
            target_response = await self.askmod_client.send_query(target_enhanced, is_target=True)
            print(f"Target Response: {target_response}")
            
            # Evaluate target response
            target_eval = await self.evaluator.evaluate_response(
                response_text=target_response,
                original_query=user_query,
                sub_question=target_question
            )
            
            # Create target QA pair
            target_qa_pair = {
                "question": target_question,
                "answer": target_response,
                "clarity_score": target_eval.get("combined_score", 0),
                "relevance_score": target_eval.get("statistical_metrics", {}).get("query_term_overlap", 0),
                "is_clear": not target_eval.get("is_ambiguous", True),
                "code_citations": target_eval.get("statistical_metrics", {}).get("code_citations", 0),
                "evaluation": target_eval,
                "repository": "target",
                "context": pair.context,
                "source_question": source_question,
                "source_answer": source_response
            }
            all_qa_pairs.append(target_qa_pair)
        
        return all_qa_pairs

    async def _extract_code_citations(self, qa_pairs: List[Dict[str, Any]]) -> Set[str]:
        """
        Extract code file citations from QA pairs.
        
        Args:
            qa_pairs: List of QA pairs to extract citations from
            
        Returns:
            Set of code file citations
        """
        citations = set()
        
        for qa_pair in qa_pairs:
            answer = qa_pair.get("answer", "")
            
            # Extract Markdown-style links
            markdown_links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', answer)
            for link_text, url in markdown_links:
                if any(ext in url for ext in [".py", ".js", ".ts", ".jsx", ".tsx", ".java"]):
                    citations.add(url)
            
            # Extract file paths mentioned in the text
            file_paths = re.findall(r'(?:file|module|path)[\s:]+[\'"`]([\w\-./]+\.(py|js|ts|jsx|tsx|java))[\'"`]', answer)
            for path, _ in file_paths:
                citations.add(path)
        
        return citations

    async def _fetch_code_for_citations(self, citations: Set[str]) -> Dict[str, str]:
        """
        Fetch code content for a set of citations.
        
        Args:
            citations: Set of code citations to fetch
            
        Returns:
            Dictionary mapping citation URLs to code content
        """
        code_files = {}
        
        for citation in citations:
            try:
                # Use the code extractor to fetch the content
                citation_url = ("https://dev-appmod.techo.camp/analyzer/get_file_details?file_path=" + 
                               citation if not citation.startswith("http") else citation)
                
                # Create a dummy display text from the citation
                display_text = os.path.basename(citation)
                
                # Fetch the code content
                result = await self.code_extractor.fetch_code_content((display_text, citation_url))
                
                if result and "content" in result and result["content"]:
                    code_files[citation] = result["content"]
                    
                    # Save the code to a file
                    safe_filename = re.sub(r'[\\/*?:"<>|]', "_", os.path.basename(citation))
                    code_file_path = os.path.join("code_citations", safe_filename)
                    with open(code_file_path, "w", encoding="utf-8") as f:
                        f.write(result["content"])
            except Exception as e:
                logger.error(f"Error fetching code for citation {citation}: {str(e)}")
        
        return code_files

    async def _make_orchestration_decision(self,
                                          user_query: str, 
                                          source_context: str, 
                                          target_context: str,
                                          qa_pairs: List[Dict[str, Any]],
                                          retrieved_code: Dict[str, str]) -> OrchestratorDecision:
        """
        Make a decision about the next step in the orchestration process.
        
        Args:
            user_query: The original user query
            source_context: RAG context from the source repository
            target_context: RAG context from the target repository
            qa_pairs: List of QA pairs collected so far
            retrieved_code: Dictionary of retrieved code files
            
        Returns:
            OrchestratorDecision object
        """
        logger.info("Making orchestration decision based on current state")
        
        # Format the current responses for inclusion in the prompt
        current_responses = ""
        for i, qa in enumerate(qa_pairs, 1):
            repo_type = qa.get("repository", "unknown")
            current_responses += f"Question {i} ({repo_type}): {qa['question']}\n"
            current_responses += f"Answer {i}: {qa['answer'][:500]}...\n"
            current_responses += f"Clarity Score: {qa['clarity_score']}, Relevance Score: {qa['relevance_score']}\n\n"
            current_responses += "-" * 40 + "\n\n"
        
        # Format the retrieved code
        retrieved_code_str = ""
        for path, content in retrieved_code.items():
            # Truncate long code files
            truncated_content = content[:500] + "..." if len(content) > 500 else content
            retrieved_code_str += f"File: {path}\n```\n{truncated_content}\n```\n\n"
        
        # Create the decision prompt
        prompt = self.decision_prompt_template.format(
            original_query=user_query,
            source_context=source_context[:500] + "..." if len(source_context) > 500 else source_context,
            target_context=target_context[:500] + "..." if len(target_context) > 500 else target_context,
            current_responses=current_responses,
            retrieved_code=retrieved_code_str
        )
        
        try:
            # Call the LLM to make a decision
            response = await self.llm.ainvoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            print(f"Orchestration Decision Response: {response_text}")
            # Save the decision to a file for debugging
            with open("decisions/decision.json", "w", encoding="utf-8") as f:
                f.write(response_text)
            
            # Parse the decision with improved error handling
            try:
                # First try to parse as JSON manually to handle extra fields
                import json
                import re
                
                # Extract JSON from the response
                json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # Try to find JSON without markdown wrapper
                    json_match = re.search(r'({[\s\S]*})', response_text)
                    if json_match:
                        json_str = json_match.group(1)
                    else:
                        raise ValueError("No JSON found in response")
                
                # Parse the JSON
                decision_data = json.loads(json_str)
                
                # Handle follow_up_questions field properly
                follow_up_questions = decision_data.get("follow_up_questions", [])
                if isinstance(follow_up_questions, list) and len(follow_up_questions) > 0:
                    # Check if it's the complex format (list of objects with 'query' field)
                    if isinstance(follow_up_questions[0], dict) and 'query' in follow_up_questions[0]:
                        # Extract just the query text from each object
                        follow_up_questions = [q.get("query", str(q)) for q in follow_up_questions]
                
                # Create OrchestratorDecision with all the fields
                decision = OrchestratorDecision(
                    decision=decision_data.get("decision", "synthesize"),
                    citations_to_fetch=decision_data.get("citations_to_fetch", []),
                    reason=decision_data.get("reason", "No reason provided"),
                    confidence=decision_data.get("confidence", 0.5),
                    generate_questions_from_rag=decision_data.get("generate_questions_from_rag", False),
                    follow_up_questions=follow_up_questions,
                    is_complete=decision_data.get("is_complete", False),
                    target_repo=decision_data.get("target_repo", False)
                )
                
                logger.info(f"Orchestration decision: {decision.decision} (confidence: {decision.confidence:.2f})")
                
                return decision
                
            except Exception as parse_error:
                logger.error(f"JSON parsing failed: {str(parse_error)}")
                # Try the original parser as fallback
                decision = self.decision_parser.parse(response_text)
            logger.info(f"Orchestration decision: {decision.decision} (confidence: {decision.confidence:.2f})")
            
            return decision
            
        except Exception as e:
            logger.error(f"Error making orchestration decision: {str(e)}")
            
            # Create a fallback decision based on heuristics
            all_clear = all(not qa.get("is_ambiguous", True) for qa in qa_pairs)
            enough_qa_pairs = len(qa_pairs) >= 4  # At least 2 pairs
            
            if all_clear and enough_qa_pairs:
                logger.info("Fallback decision: synthesize (all responses are clear and we have enough QA pairs)")
                return OrchestratorDecision(
                    decision="synthesize",
                    citations_to_fetch=[],
                    reason="All responses are clear and we have enough information",
                    confidence=0.7,
                    generate_questions_from_rag=False,
                    follow_up_questions=[],
                    is_complete=True,
                    target_repo=False
                )
            elif len(qa_pairs) < 2:
                logger.info("Fallback decision: generate_questions (not enough QA pairs)")
                return OrchestratorDecision(
                    decision="generate_questions",
                    citations_to_fetch=[],
                    reason="Not enough information collected",
                    confidence=0.6,
                    generate_questions_from_rag=True,
                    follow_up_questions=[],
                    is_complete=False,
                    target_repo=False
                )
            else:
                # If there are code citations but we haven't fetched code yet
                citations = await self._extract_code_citations(qa_pairs)
                unfetched_citations = citations - set(retrieved_code.keys())
                
                if unfetched_citations:
                    logger.info(f"Fallback decision: fetch_code ({len(unfetched_citations)} unfetched citations)")
                    return OrchestratorDecision(
                        decision="fetch_code",
                        citations_to_fetch=list(unfetched_citations)[:3],
                        reason="Need to fetch code to better understand the implementation",
                        confidence=0.5,
                        generate_questions_from_rag=False,
                        follow_up_questions=[],
                        is_complete=False,
                        target_repo=False
                    )
                else:
                    logger.info("Fallback decision: synthesize (no unfetched citations)")
                    return OrchestratorDecision(
                        decision="synthesize",
                        citations_to_fetch=[],
                        reason="No more code to fetch and we have some information",
                        confidence=0.4,
                        generate_questions_from_rag=False,
                        follow_up_questions=[],
                        is_complete=True,
                        target_repo=False
                    )

    async def _process_follow_up_questions(self,
                                        user_query: str,
                                        source_context: str,
                                        target_context: str, 
                                        follow_up_questions: List[str],
                                        is_target: bool) -> List[Dict[str, Any]]:
        """
        Process follow-up questions for either the source or target repository.
        
        Args:
            user_query: The original user query
            source_context: RAG context from the source repository
            target_context: RAG context from the target repository
            follow_up_questions: List of follow-up questions to process
            is_target: Whether these questions are for the target repository
            
        Returns:
            List of QA pairs for the follow-up questions
        """
        logger.info(f"Processing {len(follow_up_questions)} follow-up questions for {'target' if is_target else 'source'} repository")
        
        follow_up_qa_pairs = []
        
        # FIXED: Set appropriate context for the repository
        if is_target:
            context = target_context
        else:
            context = source_context
            
        for question in follow_up_questions:
            # Enhance question with context
            enhanced_question = f"{question}\n\nContext Information:\n{context}"
            
            # FIXED: Get response using is_target parameter
            response = await self.askmod_client.send_query(enhanced_question, is_target=is_target)
            
            # Evaluate response
            evaluation = await self.evaluator.evaluate_response(
                response_text=response,
                original_query=user_query,
                sub_question=question
            )
            
            # Create QA pair
            qa_pair = {
                "question": question,
                "answer": response,
                "clarity_score": evaluation.get("combined_score", 0),
                "relevance_score": evaluation.get("statistical_metrics", {}).get("query_term_overlap", 0),
                "is_clear": not evaluation.get("is_ambiguous", True),
                "code_citations": evaluation.get("statistical_metrics", {}).get("code_citations", 0),
                "evaluation": evaluation,
                "repository": "target" if is_target else "source",
                "context": "Follow-up question"
            }
            follow_up_qa_pairs.append(qa_pair)
        
        return follow_up_qa_pairs

    async def process_query(self,
                          user_query: str, 
                          task_id: str = None, 
                          user_id: str = None, 
                          database_index: str = None,
                          source_repo_config: Dict[str, Any] = None,
                          target_repo_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a user query through the orchestration system.
        
        Args:
            user_query: The user's query
            task_id: Optional task ID (overrides the default)
            user_id: Optional user ID (overrides the default)
            database_index: Optional database index (overrides the default)
            source_repo_config: Optional source repository configuration
            target_repo_config: Optional target repository configuration
            
        Returns:
            Dictionary containing the final answer
        """
        logger.info(f"Processing query: {user_query}")
        
        # Update repository configurations if provided
        if source_repo_config:
            for key, value in source_repo_config.items():
                if hasattr(self.source_repo, key) and value:
                    setattr(self.source_repo, key, value)
        
        if target_repo_config:
            for key, value in target_repo_config.items():
                if hasattr(self.target_repo, key) and value:
                    setattr(self.target_repo, key, value)
        
        # Set task_id and database_index if provided
        if task_id:
            self.source_repo.task_id = task_id
        
        if database_index:
            self.source_repo.database_index = database_index
            
        # Get initial RAG context for both repositories using the AppMod RAG tool
        logger.info("Getting RAG context for both repositories")
        
        # Get RAG context for both repositories
        source_rag_data = await self._get_rag_context(user_query, is_target_repo=False)
        target_rag_data = await self._get_rag_context(user_query, is_target_repo=True)
        
        # Extract context strings for backward compatibility
        source_context = source_rag_data.get("context", "")
        target_context = target_rag_data.get("context", "")
        
        # Initialize variables to track orchestration state
        all_qa_pairs = []
        retrieved_code = {}
        fetched_citations = set()
        iteration = 0
        
        # Main orchestration loop
        while iteration < self.max_iterations:
            logger.info(f"Starting iteration {iteration + 1}/{self.max_iterations}")
            
            # If we don't have any QA pairs yet, generate initial paired questions
            if not all_qa_pairs:
                # Generate paired questions
                question_pairs = await self._generate_paired_questions(
                    user_query, source_context, target_context
                )
                
                # Process the paired questions
                new_qa_pairs = await self._process_paired_questions(
                    user_query, source_context, target_context, question_pairs
                )
                
                all_qa_pairs.extend(new_qa_pairs)
                logger.info(f"Generated and processed {len(question_pairs)} question pairs, resulting in {len(new_qa_pairs)} QA pairs")
                
            else:
                # Make a decision about the next step
                decision = await self._make_orchestration_decision(
                    user_query, source_context, target_context, all_qa_pairs, retrieved_code
                )
                
                if decision.decision == "fetch_code":
                    # Extract new citations that we haven't fetched yet
                    new_citations = set(decision.citations_to_fetch) - fetched_citations
                    
                    if new_citations:
                        logger.info(f"Fetching code for {len(new_citations)} new citations")
                        code_files = await self._fetch_code_for_citations(new_citations)
                        retrieved_code.update(code_files)
                        fetched_citations.update(new_citations)
                    else:
                        logger.info("No new citations to fetch")
                        
                elif decision.decision == "follow_up":
                    if decision.follow_up_questions:
                        logger.info(f"Processing {len(decision.follow_up_questions)} follow-up questions")
                        
                        # Process follow-up questions for the appropriate repository
                        follow_up_qa_pairs = await self._process_follow_up_questions(
                            user_query, source_context, target_context,
                            decision.follow_up_questions, decision.target_repo
                        )
                        
                        all_qa_pairs.extend(follow_up_qa_pairs)
                
                # Check if we can synthesize a final answer
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
        
        # Prepare for synthesis
        logger.info(f"Synthesizing final answer from {len(all_qa_pairs)} QA pairs")
        
        # Enhance QA pairs with repository contexts
        enhanced_qa_pairs = all_qa_pairs.copy()
        
        # Add source context as an additional QA pair
        if source_context and source_context != "No context available" and not source_context.startswith("Error:"):
            source_context_pair = {
                "question": "What is the overall context for the source repository?",
                "answer": source_context,
                "clarity_score": 0.9,
                "relevance_score": 0.9,
                "is_clear": True,
                "code_citations": 0,
                "repository": "source",
                "evaluation": {
                    "combined_score": 0.9,
                    "is_ambiguous": False,
                    "issues": [],
                    "follow_up_questions": [],
                    "confidence": 0.9
                }
            }
            enhanced_qa_pairs.append(source_context_pair)
        
        # Add target context as an additional QA pair
        if target_context and target_context != "No context available" and not target_context.startswith("Error:"):
            target_context_pair = {
                "question": "What is the overall context for the target repository?",
                "answer": target_context,
                "clarity_score": 0.9,
                "relevance_score": 0.9,
                "is_clear": True,
                "code_citations": 0,
                "repository": "target",
                "evaluation": {
                    "combined_score": 0.9,
                    "is_ambiguous": False,
                    "issues": [],
                    "follow_up_questions": [],
                    "confidence": 0.9
                }
            }
            enhanced_qa_pairs.append(target_context_pair)
        
        # Include retrieved code in the synthesis process
        for file_path, code_content in retrieved_code.items():
            code_qa_pair = {
                "question": f"What is the content of {file_path}?",
                "answer": f"```\n{code_content}\n```",
                "clarity_score": 0.95,
                "relevance_score": 0.9,
                "is_clear": True,
                "code_citations": 1,
                "repository": "source" if "creative-workspace-backend" in file_path else "target",
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
        
        # Get source and target repository descriptions
        source_repo_desc = self.source_repo.repo_url.split('/')[-1] if self.source_repo.repo_url else "source repository"
        target_repo_desc = self.target_repo.repo_url.split('/')[-1] if self.target_repo.repo_url else "target repository"
        
        return {
            "result": {
                "answer": final_answer,
                "source_context": source_context,
                "target_context": target_context,
                "qa_pairs_count": len(all_qa_pairs),
                "iterations": iteration,
                "retrieved_code_count": len(retrieved_code),
                "source_repository": source_repo_desc,
                "target_repository": target_repo_desc
            }
        }