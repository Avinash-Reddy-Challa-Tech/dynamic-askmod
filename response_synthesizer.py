"""
Response Synthesizer with Source-Target Support

This module is responsible for synthesizing multiple responses into a comprehensive
final answer that addresses how to port a feature from a source repository to a target repository.
"""

import logging
import re
import json
from typing import Dict, Any, List, Optional, Tuple

from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResponseSynthesizer:
    """
    Component for synthesizing multiple responses into a final answer,
    with special support for source and target repository information.
    """
    
    def __init__(self, llm: Optional[ChatGoogleGenerativeAI] = None):
        """
        Initialize the ResponseSynthesizer.
        
        Args:
            llm: Language model for synthesizing responses
        """
        self.llm = llm or ChatGoogleGenerativeAI(temperature=0.3)
        
        # Create the prompt template for source-target response synthesis
        self.source_target_prompt_template = PromptTemplate(
            template="""You are an expert at synthesizing information about code implementations and explaining how to port features from one repository to another.

Original User Query: {original_query}

I have collected information about both the source and target repositories. Your task is to synthesize this information into a comprehensive guide for implementing the feature from the source repository in the target repository.

SOURCE REPOSITORY INFORMATION:
{source_info}

TARGET REPOSITORY INFORMATION:
{target_info}

CODE EXAMPLES (if available):
{code_examples}

Your task is to synthesize this information into a single, comprehensive answer that:
1. Clearly explains how the feature is implemented in the source repository
2. Identifies the key components, routes, and models involved in the feature
3. Analyzes the target repository's structure and capabilities
4. Provides a step-by-step plan for implementing the feature in the target repository
5. Highlights any potential challenges or modifications needed
6. Includes specific code references and examples where available

Format your response with these sections:
1. Summary: Brief overview of the feature and implementation approach
2. Source Repository Implementation: How the feature is currently implemented
3. Target Repository Analysis: Relevant components and structure in the target
4. Implementation Plan: Step-by-step approach to port the feature
5. Code Examples: Key code snippets that would need to be adapted
6. Potential Challenges: Issues that might arise during implementation

Make your answer technical but accessible, and ensure it provides a complete roadmap for implementing the feature in the target repository.
Include specific file paths, function names, and code patterns where possible.
If there are gaps in the information provided, acknowledge them briefly.

IMPORTANT GUIDELINES FOR CODE SNIPPETS:
- Include only the most relevant code snippets that directly address the implementation
- When including code, format it properly with markdown code blocks using the appropriate language
- For large code blocks, show just the most important parts (core functions, key logic, etc.)
- Ensure your explanation references and explains the included code

Synthesized answer:""",
            input_variables=["original_query", "source_info", "target_info", "code_examples"]
        )
        
        # Create the standard prompt template for backward compatibility
        self.standard_prompt_template = PromptTemplate(
            template="""You are an expert at synthesizing multiple pieces of information into comprehensive, clear explanations about code and software systems.

Original User Query: {original_query}

I have collected the following information (questions and answers) related to the user's query. 
Some answers include code blocks from cited files to provide more context:

{qa_pairs}

Your task is to synthesize this information into a single, comprehensive answer that:
1. Directly addresses the original user query
2. Integrates all relevant information from the collected Q&A pairs
3. Is well-structured, clear, and easy to understand
4. Includes specific code references, function names, and technical details where available
5. Incorporates relevant code snippets when they help illustrate key points (but avoid including redundant or overly large code blocks)
6. Resolves any contradictions between the different answers
7. Provides a complete picture of how the system or feature works

Make your answer technical but accessible, and ensure it provides a complete response to the original query. 
Don't simply list or enumerate the different answers, but integrate them into a cohesive explanation.
If there are gaps in the information provided, acknowledge them briefly.

IMPORTANT GUIDELINES FOR CODE SNIPPETS:
- Include only the most relevant code snippets that directly address the user's query
- When including code, format it properly with markdown code blocks using the appropriate language
- For large code blocks, consider showing just the most important parts (core functions, key logic, etc.)
- Ensure your explanation references and explains the included code

Synthesized answer:""",
            input_variables=["original_query", "qa_pairs"]
        )
        
    def _extract_code_blocks(self, text: str) -> List[str]:
        """
        Extract code blocks from text.
        
        Args:
            text: Text containing markdown code blocks
            
        Returns:
            List of extracted code blocks
        """
        pattern = r"```(?:typescript|javascript|python|java|[a-zA-Z]+)?\n(.*?)\n```"
        code_blocks = re.findall(pattern, text, re.DOTALL)
        return code_blocks
        
    def _organize_qa_pairs(self, qa_pairs: List[Dict[str, Any]]) -> Tuple[str, str, str]:
        """
        Organize QA pairs into source info, target info, and code examples.
        
        Args:
            qa_pairs: List of dictionaries containing questions, answers, and metadata
            
        Returns:
            Tuple of (source_info, target_info, code_examples)
        """
        source_qa_pairs = []
        target_qa_pairs = []
        code_examples = []
        
        for qa in qa_pairs:
            repository = qa.get("repository", "").lower()
            
            # Extract code blocks
            code_blocks = self._extract_code_blocks(qa.get("answer", ""))
            
            # Check if this is primarily a code answer
            is_code_answer = len(code_blocks) > 0 and len(''.join(code_blocks)) > len(qa.get("answer", "")) / 2
            
            if is_code_answer:
                # Add to code examples
                code_entry = f"### {qa.get('question', 'Code Example')}\n\n"
                code_entry += f"Repository: {repository}\n\n"
                code_entry += qa.get("answer", "")
                code_examples.append(code_entry)
            elif repository == "source":
                # Add to source info
                source_entry = f"### Q: {qa.get('question', '')}\n\n"
                source_entry += f"A: {qa.get('answer', '')}\n\n"
                source_qa_pairs.append(source_entry)
            elif repository == "target":
                # Add to target info
                target_entry = f"### Q: {qa.get('question', '')}\n\n"
                target_entry += f"A: {qa.get('answer', '')}\n\n"
                target_qa_pairs.append(target_entry)
            else:
                # If repository is not specified, add to both
                entry = f"### Q: {qa.get('question', '')}\n\n"
                entry += f"A: {qa.get('answer', '')}\n\n"
                source_qa_pairs.append(entry)
                target_qa_pairs.append(entry)
        
        # Join the entries
        source_info = "\n".join(source_qa_pairs)
        target_info = "\n".join(target_qa_pairs)
        code_examples_info = "\n".join(code_examples)
        
        return source_info, target_info, code_examples_info
        
    async def synthesize(self, original_query: str, qa_pairs: List[Dict[str, Any]]) -> str:
        """
        Synthesize multiple Q&A pairs into a comprehensive answer.
        
        Args:
            original_query: The original user query
            qa_pairs: List of dictionaries containing questions, answers, and metadata
            
        Returns:
            Synthesized final answer
        """
        logger.info(f"Synthesizing final answer from {len(qa_pairs)} Q&A pairs")
        
        # Check if we have repository information
        has_repo_info = any("repository" in qa for qa in qa_pairs)
        
        if has_repo_info:
            # Use source-target synthesis
            logger.info("Using source-target synthesis approach")
            
            # Organize QA pairs by repository
            source_info, target_info, code_examples = self._organize_qa_pairs(qa_pairs)
            
            # Create the prompt
            prompt = self.source_target_prompt_template.format(
                original_query=original_query,
                source_info=source_info,
                target_info=target_info,
                code_examples=code_examples
            )
        else:
            # Use standard synthesis (backward compatibility)
            logger.info("Using standard synthesis approach")
            
            # Format the Q&A pairs for inclusion in the prompt
            formatted_qa_pairs = ""
            for i, qa in enumerate(qa_pairs, 1):
                formatted_qa_pairs += f"Question {i}: {qa['question']}\n"
                
                # Include the full answer with code blocks
                formatted_qa_pairs += f"Answer {i}:\n{qa['answer']}\n"
                
                # Include evaluation metrics if available
                if 'clarity_score' in qa and 'relevance_score' in qa:
                    formatted_qa_pairs += f"Clarity Score: {qa['clarity_score']}, Relevance Score: {qa['relevance_score']}\n\n"
                
                # Add separator for better readability
                formatted_qa_pairs += "-" * 40 + "\n\n"
            
            # Create the prompt
            prompt = self.standard_prompt_template.format(
                original_query=original_query,
                qa_pairs=formatted_qa_pairs
            )
        
        try:
            # Generate the response
            response = await self.llm.ainvoke(prompt)
            
            # Extract the text from the response
            if hasattr(response, 'content'):
                final_answer = response.content
            else:
                final_answer = str(response)
            
            logger.info("Successfully synthesized final answer")
            
            return final_answer
            
        except Exception as e:
            logger.error(f"Error synthesizing answer: {str(e)}")
       