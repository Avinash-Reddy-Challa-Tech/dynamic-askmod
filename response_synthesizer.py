"""
Response Synthesizer

This module is responsible for synthesizing multiple responses into a comprehensive
final answer that addresses the original user query.
"""

import logging
from typing import Dict, Any, List, Optional

from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResponseSynthesizer:
    """
    Component for synthesizing multiple responses into a final answer.
    """
    
    def __init__(self, llm: Optional[ChatGoogleGenerativeAI] = None):
        """
        Initialize the ResponseSynthesizer.
        
        Args:
            llm: Language model for synthesizing responses
        """
        self.llm = llm or ChatGoogleGenerativeAI(temperature=0.3)
        
        # Create the prompt template for response synthesis
        self.prompt_template = PromptTemplate(
            template="""You are an expert at synthesizing multiple pieces of information into comprehensive, clear explanations about code and software systems.

Original User Query: {original_query}

I have collected the following information (questions and answers) related to the user's query:

{qa_pairs}

Your task is to synthesize this information into a single, comprehensive answer that:
1. Directly addresses the original user query
2. Integrates all relevant information from the collected Q&A pairs
3. Is well-structured, clear, and easy to understand
4. Includes specific code references, function names, and technical details where available
5. Resolves any contradictions between the different answers
6. Provides a complete picture of how the system or feature works

Make your answer technical but accessible, and ensure it provides a complete response to the original query. 
Don't simply list or enumerate the different answers, but integrate them into a cohesive explanation.
If there are gaps in the information provided, acknowledge them briefly.

Synthesized answer:""",
            input_variables=["original_query", "qa_pairs"]
        )
        
    async def synthesize(self, original_query: str, qa_pairs: List[Dict[str, Any]]) -> str:
        """
        Synthesize multiple Q&A pairs into a comprehensive answer.
        
        Args:
            original_query: The original user query
            qa_pairs: List of dictionaries containing questions, answers, and evaluation results
            
        Returns:
            Synthesized final answer
        """
        logger.info(f"Synthesizing final answer from {len(qa_pairs)} Q&A pairs")
        
        # Format the Q&A pairs for inclusion in the prompt
        formatted_qa_pairs = ""
        for i, qa in enumerate(qa_pairs, 1):
            formatted_qa_pairs += f"Question {i}: {qa['question']}\n"
            formatted_qa_pairs += f"Answer {i}: {qa['answer']}\n"
            formatted_qa_pairs += f"Clarity Score: {qa['clarity_score']}, Relevance Score: {qa['relevance_score']}\n\n"
        
        # Create and run the chain to synthesize the answer
        chain = self.prompt_template | self.llm
        
        try:
            result = await chain.ainvoke({
                "original_query": original_query,
                "qa_pairs": formatted_qa_pairs
            })
            
            # Extract the text from the response
            if hasattr(result, 'content'):
                final_answer = result.content
            else:
                final_answer = str(result)
            
            logger.info("Successfully synthesized final answer")
            
            return final_answer
            
        except Exception as e:
            logger.error(f"Error synthesizing answer: {str(e)}")
            # Fall back to a simpler synthesis approach
            return self._fallback_synthesis(original_query, qa_pairs)
    
    def _fallback_synthesis(self, original_query: str, qa_pairs: List[Dict[str, Any]]) -> str:
        """
        Provide a simple fallback synthesis when the main synthesis fails.
        
        Args:
            original_query: The original user query
            qa_pairs: List of dictionaries containing questions, answers, and evaluation results
            
        Returns:
            Basic synthesized answer
        """
        logger.info("Using fallback synthesis strategy")
        
        # Start with an introduction
        answer = f"In response to your query about '{original_query}':\n\n"
        
        # Add each Q&A pair, prioritizing those with higher clarity scores
        sorted_qa_pairs = sorted(qa_pairs, key=lambda qa: qa['clarity_score'], reverse=True)
        
        for i, qa in enumerate(sorted_qa_pairs, 1):
            answer += f"Regarding '{qa['question']}':\n"
            answer += f"{qa['answer']}\n\n"
        
        # Add a conclusion
        answer += "This information should help you understand the topic better. If you need more specific details, please ask a follow-up question."
        
        return answer