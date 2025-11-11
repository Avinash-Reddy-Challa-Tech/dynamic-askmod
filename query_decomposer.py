"""
Query Decomposer

This module is responsible for breaking down complex user queries into more specific
sub-questions that can be answered more precisely by the AskMod system.
"""

import logging
from typing import List, Dict, Any, Optional

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
    questions: List[str] = Field(
        description="List of 2-3 targeted sub-questions that break down the original query"
    )
    reasoning: str = Field(
        description="Explanation of how these sub-questions will help answer the original query"
    )

class QueryDecomposer:
    """
    Component for breaking down complex queries into more specific sub-questions.
    """
    
    def __init__(self, llm: Optional[ChatGoogleGenerativeAI] = None, max_questions: int = 3):
        """
        Initialize the QueryDecomposer.
        
        Args:
            llm: Language model for generating sub-questions
            max_questions: Maximum number of sub-questions to generate
        """
        self.llm = llm or ChatGoogleGenerativeAI(temperature=0.2)
        self.max_questions = max_questions
        self.parser = JsonOutputParser(pydantic_object=SubQuestions)
        
        # Create the prompt template for query decomposition
        self.prompt_template = PromptTemplate(
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
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
    async def generate_sub_questions(self, query: str) -> List[str]:
        """
        Generate targeted sub-questions from the original user query.
        
        Args:
            query: The original user query
            
        Returns:
            List of generated sub-questions
        """
        logger.info(f"Generating sub-questions for query: {query}")
        
        # Create and run the chain to generate sub-questions
        chain = self.prompt_template | self.llm | self.parser
        
        try:
            result = await chain.ainvoke({
                "query": query,
                "max_questions": self.max_questions
            })
            
            logger.info(f"Generated {len(result['questions'])} sub-questions")
            logger.info(f"Reasoning: {result['reasoning']}")
            
            return result["questions"]
            
        except Exception as e:
            logger.error(f"Error generating sub-questions: {str(e)}")
            # Fall back to a simple decomposition strategy
            return self._fallback_decomposition(query)
    
    def _fallback_decomposition(self, query: str) -> List[str]:
        """
        Provide a simple fallback mechanism when the main decomposition fails.
        
        Args:
            query: The original user query
            
        Returns:
            List of basic sub-questions
        """
        logger.info("Using fallback decomposition strategy")
        
        # Basic fallback questions that work for most code queries
        fallback_questions = [
            f"What is the purpose and functionality of {query.strip().rstrip('?')}?",
            f"What are the main components or methods involved in {query.strip().rstrip('?')}?",
            f"How is {query.strip().rstrip('?')} implemented in the codebase?"
        ]
        
        return fallback_questions[:self.max_questions]