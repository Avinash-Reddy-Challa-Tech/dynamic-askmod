"""
Response Evaluator

This module is responsible for evaluating the quality and clarity of responses
received from the AskMod system, identifying any ambiguities or issues.
"""

import logging
from typing import Dict, Any, Optional, List

from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from pydantic.v1 import BaseModel, Field

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the output schema for response evaluation
class EvaluationResult(BaseModel):
    """Schema for evaluating the quality of a response."""
    clarity_score: int = Field(
        description="Score from 1-10 indicating how clear the response is (10 being perfectly clear)"
    )
    relevance_score: int = Field(
        description="Score from 1-10 indicating how relevant the response is to the question (10 being perfectly relevant)"
    )
    is_clear: bool = Field(
        description="Boolean indicating if the response clearly answers the question without significant ambiguities"
    )
    issues: List[str] = Field(
        description="List of specific issues or ambiguities in the response, if any"
    )
    reasoning: str = Field(
        description="Reasoning behind the evaluation scores and identified issues"
    )

class ResponseEvaluator:
    """
    Component for evaluating the quality and clarity of responses.
    """
    
    def __init__(self, llm: Optional[ChatGoogleGenerativeAI] = None, clarity_threshold: int = 7):
        """
        Initialize the ResponseEvaluator.
        
        Args:
            llm: Language model for evaluating responses
            clarity_threshold: Threshold for considering a response clear (1-10)
        """
        self.llm = llm or ChatGoogleGenerativeAI(temperature=0.1)
        self.clarity_threshold = clarity_threshold
        self.parser = JsonOutputParser(pydantic_object=EvaluationResult)
        
        # Create the prompt template for response evaluation
        self.prompt_template = PromptTemplate(
            template="""You are an expert at evaluating responses to technical questions about code and software.

Question: {question}

Response: {response}

Your task is to evaluate the quality of this response based on:
1. Clarity: How clear and understandable is the response?
2. Relevance: How directly does it address the question?
3. Completeness: Does it fully answer what was asked?
4. Specificity: Does it provide concrete details rather than vague generalities?

Specifically for code-related responses, look for:
- Whether specific functions, methods, or components are named
- Whether the implementation details are explained clearly
- Whether any ambiguous technical terms are properly defined
- Whether code flow or logic is explained in a step-by-step manner

{format_instructions}""",
            input_variables=["question", "response"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
    async def evaluate(self, question: str, response: str) -> Dict[str, Any]:
        """
        Evaluate the quality and clarity of a response.
        
        Args:
            question: The question that was asked
            response: The response to evaluate
            
        Returns:
            Evaluation results including clarity and relevance scores
        """
        logger.info(f"Evaluating response for question: {question[:50]}...")
        
        # Create and run the chain to evaluate the response
        chain = self.prompt_template | self.llm | self.parser
        
        try:
            result = await chain.ainvoke({
                "question": question,
                "response": response
            })
            
            # Determine if the response is clear based on our threshold
            result["is_clear"] = (
                result["clarity_score"] >= self.clarity_threshold and
                result["relevance_score"] >= self.clarity_threshold and
                len(result["issues"]) == 0
            )
            
            logger.info(f"Evaluation complete - Clarity: {result['clarity_score']}, "
                       f"Relevance: {result['relevance_score']}, Clear: {result['is_clear']}")
            
            if not result["is_clear"]:
                logger.info(f"Issues identified: {result['issues']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating response: {str(e)}")
            # Return a basic evaluation when the main evaluation fails
            return self._fallback_evaluation(question, response)
    
    def _fallback_evaluation(self, question: str, response: str) -> Dict[str, Any]:
        """
        Provide a simple fallback evaluation when the main evaluation fails.
        
        Args:
            question: The question that was asked
            response: The response to evaluate
            
        Returns:
            Basic evaluation results
        """
        logger.info("Using fallback evaluation strategy")
        
        # Check for some basic indicators of unclear responses
        issues = []
        
        if "I don't know" in response or "I'm not sure" in response:
            issues.append("Response expresses uncertainty")
            
        if len(response.split()) < 20:
            issues.append("Response is very brief")
            
        if "error" in response.lower() or "exception" in response.lower():
            issues.append("Response mentions errors")
        
        # Determine clarity based on identified issues
        is_clear = len(issues) == 0
        
        return {
            "clarity_score": 8 if is_clear else 5,
            "relevance_score": 8 if is_clear else 5,
            "is_clear": is_clear,
            "issues": issues,
            "reasoning": "Fallback evaluation based on basic heuristics"
        }