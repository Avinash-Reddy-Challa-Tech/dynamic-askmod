"""
Query Decomposer with Source-Target Repository Support

This module is responsible for breaking down complex user queries into more specific
sub-questions for both source and target repositories to facilitate feature porting.
"""

import logging
import json
import re
from typing import List, Dict, Any, Optional, Tuple

from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from pydantic.v1 import BaseModel, Field

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the output schema for the query decomposition
class SubQuestions(BaseModel):
    """Schema for the sub-questions generated from the original query."""
    source_questions: List[str] = Field(
        description="List of 2-3 targeted sub-questions for the source repository"
    )
    target_questions: List[str] = Field(
        description="List of 2-3 targeted sub-questions for the target repository that align with the source questions"
    )
    reasoning: str = Field(
        description="Explanation of how these sub-questions will help answer the original query"
    )

class QueryDecomposer:
    """
    Component for breaking down complex queries into more specific sub-questions
    for both source and target repositories.
    """
    
    def __init__(self, llm: Optional[ChatGoogleGenerativeAI] = None, max_questions: int = 3):
        """
        Initialize the QueryDecomposer.
        
        Args:
            llm: Language model for generating sub-questions
            max_questions: Maximum number of sub-questions to generate per repository
        """
        self.llm = llm or ChatGoogleGenerativeAI(temperature=0.2)
        self.max_questions = max_questions
        self.parser = JsonOutputParser(pydantic_object=SubQuestions)
        
        # Create the prompt template for source-target query decomposition
        self.prompt_template = PromptTemplate(
            template="""You are an expert at breaking down complex questions about code and software into more specific, targeted sub-questions for both source and target repositories.

Original Query: {query}

Source Repository Description: {source_repo_desc}
Target Repository Description: {target_repo_desc}

Your task is to break down this query into {max_questions} pairs of specific sub-questions where:
1. Each source question focuses on understanding a specific aspect of the feature in the source repository
2. Each target question focuses on how to implement that same aspect in the target repository
3. The questions are more focused and precise than the original query
4. Together, they will provide comprehensive information to understand how to port the feature

IMPORTANT GUIDELINES:
- ALL questions must start with "TRIGGER DOMAIN KNOWLEDGE AGENT: " to ensure they activate the correct component
- Each question should be self-contained and not reference other questions
- Questions should focus on specific aspects of code implementation, such as routes, controllers, models, etc.
- Keep questions very simple and focused on one aspect at a time
- Ensure questions are direct and specific, not open-ended
- Questions should seek factual information about code implementation, not opinions or explanations

Example pairs:
1. Source: "TRIGGER DOMAIN KNOWLEDGE AGENT: What API route handles the PDF download feature in the source repository?"
   Target: "TRIGGER DOMAIN KNOWLEDGE AGENT: What is the API structure for implementing new download routes in the target repository?"

2. Source: "TRIGGER DOMAIN KNOWLEDGE AGENT: What database models are used for document storage in the source repository?"
   Target: "TRIGGER DOMAIN KNOWLEDGE AGENT: What database models exist in the target repository for storing files or documents?"

Respond with ONLY a JSON object that contains:
1. A list of {max_questions} source questions
2. A corresponding list of {max_questions} target questions
3. A brief reasoning for why these question pairs will help answer the original query

{format_instructions}""",
            input_variables=["query", "source_repo_desc", "target_repo_desc", "max_questions"],
            partial_variables={"format_instructions": ""}
        )
        
        # Create the prompt template for standard query decomposition (backward compatibility)
        self.standard_prompt_template = PromptTemplate(
            template="""You are an expert at breaking down complex questions about code and software into more specific, targeted sub-questions.

Original Query: {query}

Your task is to break down this query into {max_questions} specific sub-questions that:
1. Are more focused and precise than the original query
2. Cover different aspects of the original query
3. Are likely to elicit clear, specific responses from a code documentation system
4. Together, will provide comprehensive information to fully answer the original query
5. Are specific to programming and code concepts mentioned in the original query

Example:
If the original query is "How does the authentication system work?", good sub-questions might be:
- "TRIGGER DOMAIN KNOWLEDGE AGENT: What are the main components of the authentication system?"
- "TRIGGER DOMAIN KNOWLEDGE AGENT: What is the authentication flow from login to session creation?"
- "TRIGGER DOMAIN KNOWLEDGE AGENT: What security measures are implemented in the authentication system?"

IMPORTANT: NEED TO START THE QUERY WITH "TRIGGER DOMAIN KNOWLEDGE AGENT:" FOR EACH SUB-QUESTION.

Respond with ONLY a JSON object that contains:
1. A list of {max_questions} sub-questions
2. A brief reasoning for why these sub-questions will help answer the original query

{format_instructions}""",
            input_variables=["query", "max_questions"],
            partial_variables={"format_instructions": ""}
        )
    
    async def _format_prompt(self, template: PromptTemplate, **kwargs) -> str:
        """
        Format a prompt template with the provided kwargs.
        
        Args:
            template: The prompt template
            **kwargs: Keyword arguments for formatting
            
        Returns:
            Formatted prompt string
        """
        # Add format instructions to the kwargs
        if hasattr(self, 'parser'):
            kwargs['format_instructions'] = self.parser.get_format_instructions()
        
        # Format the template
        return template.format(**kwargs)
        
    async def generate_source_target_questions(self, 
                                              query: str, 
                                              source_repo_desc: str, 
                                              target_repo_desc: str) -> Tuple[List[str], List[str]]:
        """
        Generate paired sub-questions for source and target repositories.
        
        Args:
            query: The original user query
            source_repo_desc: Description of the source repository
            target_repo_desc: Description of the target repository
            
        Returns:
            Tuple containing (source_questions, target_questions)
        """
        logger.info(f"Generating source and target questions for query: {query}")
        
        # Format the prompt
        prompt = await self._format_prompt(
            self.prompt_template,
            query=query,
            source_repo_desc=source_repo_desc,
            target_repo_desc=target_repo_desc,
            max_questions=self.max_questions
        )
        
        try:
            # Generate the response
            response = await self.llm.ainvoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse the response
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
            
            # Parse the JSON
            result = json.loads(json_str)
            
            source_questions = result.get("source_questions", [])
            target_questions = result.get("target_questions", [])
            
            # Ensure questions start with the trigger phrase
            source_questions = [
                q if q.startswith("TRIGGER DOMAIN KNOWLEDGE AGENT:") 
                else f"TRIGGER DOMAIN KNOWLEDGE AGENT: {q}" 
                for q in source_questions
            ]
            
            target_questions = [
                q if q.startswith("TRIGGER DOMAIN KNOWLEDGE AGENT:") 
                else f"TRIGGER DOMAIN KNOWLEDGE AGENT: {q}" 
                for q in target_questions
            ]
            
            logger.info(f"Generated {len(source_questions)} source questions and {len(target_questions)} target questions")
            
            return source_questions, target_questions
            
        except Exception as e:
            logger.error(f"Error generating source-target questions: {str(e)}")
          
    async def generate_sub_questions(self, query: str) -> List[str]:
        """
        Generate targeted sub-questions from the original user query.
        This method is maintained for backward compatibility.
        
        Args:
            query: The original user query
            
        Returns:
            List of generated sub-questions
        """
        logger.info(f"Generating standard sub-questions for query: {query}")
        
        # Format the prompt
        prompt = await self._format_prompt(
            self.standard_prompt_template,
            query=query,
            max_questions=self.max_questions
        )
        
        try:
            # Generate the response
            response = await self.llm.ainvoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse the response
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
            
            # Parse the JSON
            result = json.loads(json_str)
            
            questions = result.get("questions", [])
            
            # Ensure questions start with the trigger phrase
            questions = [
                q if q.startswith("TRIGGER DOMAIN KNOWLEDGE AGENT:") 
                else f"TRIGGER DOMAIN KNOWLEDGE AGENT: {q}" 
                for q in questions
            ]
            
            logger.info(f"Generated {len(questions)} sub-questions")
            
            return questions
            
        except Exception as e:
            logger.error(f"Error generating sub-questions: {str(e)}")
    