"""
Enhanced response evaluation metrics for AskMod orchestrator.
This handles responses with code citations and improves quality assessment.
Compatible with different LLM interfaces (OpenAI, Google, etc.)
"""

import re
import json
import asyncio
import logging
from typing import List, Dict, Any, Tuple

# Set up logging
logger = logging.getLogger(__name__)

class EnhancedResponseEvaluator:
    """Enhanced evaluator for AskMod responses that include code citations."""
    
    def __init__(self, llm):
        """Initialize the EnhancedResponseEvaluator.
        
        Args:
            llm: LangChain LLM instance
        """
        self.llm = llm
        logger.info(f"Initialized EnhancedResponseEvaluator with LLM type: {type(llm).__name__}")
        
        # Updated prompt to handle responses with code citations
        self.evaluation_prompt_template = """
        You are evaluating the quality of a response from a code documentation system.
        
        Original User Query: {original_query}
        Specific Question: {sub_question}
        Response: {response}
        
        This response may contain code citations in the format [identifier](path/to/file) or other link formats.
        These citations are an important quality indicator - they show evidence that the response is grounded in
        the actual codebase.
        
        Evaluate the response on the following criteria:
        
        1. Clarity (0-10): 
           - Is the response clear and easy to understand?
           - Are technical concepts explained well?
           - Is the language precise and unambiguous?
        
        2. Completeness (0-10):
           - Does it fully answer the specific question?
           - Are there any obvious gaps or missing information?
           - Does it address all aspects of the question?
        
        3. Code Grounding (0-10):
           - Does the response reference specific code files or functions?
           - Are the code citations relevant to the question?
           - Does it provide specific code locations rather than general statements?
        
        4. Structure (0-10):
           - Is the response well-organized with clear sections?
           - Does it use appropriate formatting (bullet points, paragraphs)?
           - Is there a logical flow to the explanation?
        
        5. Accuracy (0-10):
           - Based on the internal consistency, does the response appear accurate?
           - Are there any contradictions or suspicious claims?
           - Does the technical explanation make sense?
        
        For each criterion, provide a score from 0-10 and a brief explanation.
        
        Then provide an overall assessment with:
        - Overall score (0-1, calculated as average of criteria scores divided by 10)
        - Whether the response is ambiguous (true/false)
        - List of specific issues (if any)
        - Specific follow-up questions that would clarify any ambiguities
        - Confidence in this evaluation (0-1)
        
        Format your response as JSON:
        ```json
        {{
            "criteria": {{
                "clarity": {{"score": X, "explanation": "..."}},
                "completeness": {{"score": X, "explanation": "..."}},
                "code_grounding": {{"score": X, "explanation": "..."}},
                "structure": {{"score": X, "explanation": "..."}},
                "accuracy": {{"score": X, "explanation": "..."}}
            }},
            "overall": {{
                "score": X.X,
                "is_ambiguous": true/false,
                "issues": ["issue1", "issue2"],
                "follow_up_questions": ["question1", "question2"],
                "confidence": X.X
            }}
        }}
        ```
        """
    
    def _count_code_citations(self, response_text: str) -> int:
        """Count code citations in the response.
        
        Args:
            response_text: The response text
            
        Returns:
            Number of code citations
        """
        # Match patterns like [identifier](file/path) or <file/path> or any variant
        citation_patterns = [
            r'\[([^\]]+)\]\(([^)]+)\)',  # Markdown link [text](url)
            r'<([^>]+)>',                # <url> style
            r'\(([^)]+\.(?:cs|tsx?|js|py|java))\)'  # (filename.ext) for code files
        ]
        
        total_citations = 0
        for pattern in citation_patterns:
            citations = re.findall(pattern, response_text)
            total_citations += len(citations)
        
        logger.debug(f"Found {total_citations} code citations in response")
        return total_citations
    
    def _extract_code_paths(self, response_text: str) -> List[str]:
        """Extract code file paths from the response.
        
        Args:
            response_text: The response text
            
        Returns:
            List of code file paths
        """
        # Extract paths from markdown links [text](path)
        markdown_paths = re.findall(r'\[[^\]]+\]\(([^)]+)\)', response_text)
        
        # Extract paths enclosed in angle brackets
        angle_paths = re.findall(r'<([^>]+\.(cs|tsx?|js|py|java))>', response_text)
        angle_paths = [p[0] for p in angle_paths]  # Extract just the path
        
        # Extract file references that look like paths
        file_refs = re.findall(r'(?:in|from|at) [`\'"]([^\'"`]+\.(cs|tsx?|js|py|java))[\'"`]', response_text)
        file_refs = [p[0] for p in file_refs]  # Extract just the path
        
        all_paths = markdown_paths + angle_paths + file_refs
        logger.debug(f"Extracted {len(all_paths)} code file paths: {all_paths[:5]}" + ("..." if len(all_paths) > 5 else ""))
        return all_paths
    
    def _check_section_formatting(self, response_text: str) -> bool:
        """Check if the response has well-formatted sections.
        
        Args:
            response_text: The response text
            
        Returns:
            True if well-formatted, False otherwise
        """
        # Check for section headers (bold, h1, h2, etc.)
        has_headers = bool(re.search(r'(\*\*[^*]+\*\*|#+\s+[^#\n]+)', response_text))
        
        # Check for bullet points or numbered lists
        has_lists = bool(re.search(r'(\n\s*[-*]\s+|\n\s*\d+\.\s+)', response_text))
        
        # Check for code blocks
        has_code_blocks = bool(re.search(r'```[\s\S]+?```', response_text))
        
        # Check for paragraphs with reasonable length
        paragraphs = [p for p in re.split(r'\n\s*\n', response_text) if p.strip()]
        has_good_paragraphs = len(paragraphs) > 1 and all(len(p.strip()) < 500 for p in paragraphs)
        
        # Score based on formatting features
        formatting_score = sum([has_headers, has_lists, has_code_blocks, has_good_paragraphs])
        formatting_features = {
            "has_headers": has_headers,
            "has_lists": has_lists,
            "has_code_blocks": has_code_blocks,
            "has_good_paragraphs": has_good_paragraphs,
            "total_score": formatting_score
        }
        logger.debug(f"Section formatting check: {formatting_features}")
        return formatting_score >= 2  # At least 2 formatting features for good structure
    
    def _detect_contradiction(self, response_text: str) -> bool:
        """Detect potential contradictions in the response.
        
        Args:
            response_text: The response text
            
        Returns:
            True if contradictions detected, False otherwise
        """
        contradiction_patterns = [
            r'however, (?:this|it) is not',
            r'although (?:initially|previously).*but',
            r'(?:contrary|opposed) to what.*mentioned',
            r'this contradicts',
            r'inconsistent with'
        ]
        
        for pattern in contradiction_patterns:
            if re.search(pattern, response_text, re.IGNORECASE):
                logger.debug(f"Detected contradiction with pattern: {pattern}")
                return True
        
        return False
    
    def _detect_uncertainty(self, response_text: str) -> bool:
        """Detect expressions of uncertainty in the response.
        
        Args:
            response_text: The response text
            
        Returns:
            True if uncertainty detected, False otherwise
        """
        uncertainty_patterns = [
            r'(?:might|may|could|possibly) be',
            r'(?:unclear|ambiguous|uncertain)',
            r'not (?:certain|sure|clear)',
            r'(?:appears|seems) to',
            r'(?:couldn\'t|can\'t|cannot) (?:find|determine|confirm)'
        ]
        
        uncertainty_count = 0
        for pattern in uncertainty_patterns:
            matches = re.findall(pattern, response_text, re.IGNORECASE)
            uncertainty_count += len(matches)
        
        # More than 2 expressions of uncertainty indicates an ambiguous response
        logger.debug(f"Detected {uncertainty_count} expressions of uncertainty")
        return uncertainty_count > 2
    
    def _evaluate_response_statistically(self, response_text: str, query: str) -> Dict[str, Any]:
        """Perform statistical evaluation of the response.
        
        Args:
            response_text: The response text
            query: The query text
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Performing statistical evaluation of response")
        
        # Initialize metrics
        metrics = {}
        
        # Check length
        response_length = len(response_text)
        metrics["length"] = response_length
        metrics["length_score"] = min(1.0, response_length / 1000)  # Normalize to 0-1
        logger.debug(f"Response length: {response_length} chars, score: {metrics['length_score']:.2f}")
        
        # Count code citations
        code_citations = self._count_code_citations(response_text)
        metrics["code_citations"] = code_citations
        metrics["code_citation_score"] = min(1.0, code_citations / 5)  # Normalize to 0-1
        logger.debug(f"Code citations: {code_citations}, score: {metrics['code_citation_score']:.2f}")
        
        # Extract code paths
        code_paths = self._extract_code_paths(response_text)
        metrics["code_paths"] = code_paths
        metrics["has_code_paths"] = len(code_paths) > 0
        
        # Check section formatting
        metrics["well_formatted"] = self._check_section_formatting(response_text)
        logger.debug(f"Response is well formatted: {metrics['well_formatted']}")
        
        # Check for contradictions
        metrics["has_contradictions"] = self._detect_contradiction(response_text)
        logger.debug(f"Response has contradictions: {metrics['has_contradictions']}")
        
        # Check for uncertainty
        metrics["has_uncertainty"] = self._detect_uncertainty(response_text)
        logger.debug(f"Response has uncertainty: {metrics['has_uncertainty']}")
        
        # Query-response relevance (basic check for query terms in response)
        query_terms = set(re.findall(r'\b\w{3,}\b', query.lower()))
        response_terms = set(re.findall(r'\b\w{3,}\b', response_text.lower()))
        query_term_overlap = len(query_terms.intersection(response_terms)) / max(1, len(query_terms))
        metrics["query_term_overlap"] = query_term_overlap
        logger.debug(f"Query term overlap: {query_term_overlap:.2f}")
        
        # Calculate an overall statistical score
        overall_score = (
            metrics["length_score"] * 0.2 +
            metrics["code_citation_score"] * 0.4 +
            (1.0 if metrics["well_formatted"] else 0.5) * 0.2 +
            (0.3 if not metrics["has_contradictions"] else 0.0) * 0.1 +
            (0.3 if not metrics["has_uncertainty"] else 0.0) * 0.1 +
            query_term_overlap * 0.3
        )
        
        # Normalize to 0-1
        metrics["statistical_score"] = min(1.0, max(0.0, overall_score))
        logger.info(f"Overall statistical score: {metrics['statistical_score']:.2f}")
        
        # Determine if ambiguous based on statistical metrics
        metrics["is_statistically_ambiguous"] = (
            metrics["statistical_score"] < 0.6 or
            metrics["has_contradictions"] or
            (metrics["has_uncertainty"] and metrics["code_citation_score"] < 0.5)
        )
        logger.info(f"Response is statistically ambiguous: {metrics['is_statistically_ambiguous']}")
        
        return metrics
    
    async def _call_llm_async(self, prompt: str) -> str:
        """Call the LLM asynchronously, handling different LLM interfaces.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            The LLM response as a string
        """
        llm_type = type(self.llm).__name__
        logger.info(f"Calling LLM ({llm_type}) for evaluation")
        
        try:
            # Try the OpenAI-style interface
            if hasattr(self.llm, "apredict"):
                logger.debug(f"Using 'apredict' interface for {llm_type}")
                return await self.llm.apredict(prompt)
            
            # Try the ChatGoogleGenerativeAI-style interface
            elif hasattr(self.llm, "ainvoke"):
                logger.debug(f"Using 'ainvoke' interface for {llm_type}")
                result = await self.llm.ainvoke({"content": prompt})
                if hasattr(result, "content"):
                    return result.content
                return str(result)
            
            # Try the old LangChain interface
            elif hasattr(self.llm, "agenerate"):
                logger.debug(f"Using 'agenerate' interface for {llm_type}")
                # Use the 'prompts' parameter instead of 'messages'
                result = await self.llm.agenerate(prompts=[prompt])
                if hasattr(result, "generations") and result.generations:
                    return result.generations[0][0].text
                return str(result)
            
            # Fall back to synchronous call if no async methods available
            elif hasattr(self.llm, "predict"):
                logger.warning(f"Falling back to synchronous 'predict' interface for {llm_type}")
                return self.llm.predict(prompt)
            
            elif hasattr(self.llm, "invoke"):
                logger.warning(f"Falling back to synchronous 'invoke' interface for {llm_type}")
                result = self.llm.invoke({"content": prompt})
                if hasattr(result, "content"):
                    return result.content
                return str(result)
            
            else:
                # Direct call as a last resort
                logger.warning(f"Falling back to direct call interface for {llm_type}")
                return str(self.llm(prompt))
                
        except Exception as e:
            # Log the error and return a fallback response
            logger.error(f"Error calling LLM: {e}", exc_info=True)
            return json.dumps({
                "criteria": {
                    "clarity": {"score": 5, "explanation": "Error evaluating"},
                    "completeness": {"score": 5, "explanation": "Error evaluating"},
                    "code_grounding": {"score": 5, "explanation": "Error evaluating"},
                    "structure": {"score": 5, "explanation": "Error evaluating"},
                    "accuracy": {"score": 5, "explanation": "Error evaluating"}
                },
                "overall": {
                    "score": 0.5,
                    "is_ambiguous": True,
                    "issues": [f"LLM evaluation error: {str(e)}"],
                    "follow_up_questions": ["Could you provide more specific details?"],
                    "confidence": 0.3
                }
            })
    
    async def evaluate_response(self, response_text: str, original_query: str, sub_question: str) -> Dict[str, Any]:
        """Evaluate a response comprehensively using both LLM and statistical metrics.
        
        Args:
            response_text: The response to evaluate
            original_query: The original user query
            sub_question: The specific sub-question
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating response for sub-question: {sub_question[:50]}...")
        
        # First do a statistical evaluation
        statistical_metrics = self._evaluate_response_statistically(response_text, original_query)
        
        # Then do an LLM-based evaluation
        prompt = self.evaluation_prompt_template.format(
            original_query=original_query,
            sub_question=sub_question,
            response=response_text
        )
        
        llm_text = await self._call_llm_async(prompt)
        
        # Extract JSON from LLM response
        try:
            match = re.search(r'```json\s*(.*?)\s*```', llm_text, re.DOTALL)
            if match:
                logger.debug("Found JSON in code block")
                json_str = match.group(1)
                llm_metrics = json.loads(json_str)
            else:
                # Try direct JSON parsing if no code block
                logger.debug("Attempting direct JSON parsing")
                llm_metrics = json.loads(llm_text)
            
            logger.info(f"Successfully parsed LLM evaluation metrics")
            
        except (json.JSONDecodeError, AttributeError) as e:
            logger.error(f"Error parsing LLM response: {e}", exc_info=True)
            # Fallback metrics if parsing fails
            llm_metrics = {
                "criteria": {
                    "clarity": {"score": 5, "explanation": "Unable to evaluate clarity"},
                    "completeness": {"score": 5, "explanation": "Unable to evaluate completeness"},
                    "code_grounding": {"score": 5, "explanation": "Unable to evaluate code grounding"},
                    "structure": {"score": 5, "explanation": "Unable to evaluate structure"},
                    "accuracy": {"score": 5, "explanation": "Unable to evaluate accuracy"}
                },
                "overall": {
                    "score": 0.5,
                    "is_ambiguous": statistical_metrics["is_statistically_ambiguous"],
                    "issues": ["Failed to parse LLM evaluation"],
                    "follow_up_questions": [],
                    "confidence": 0.3
                }
            }
        
        # Combine statistical and LLM metrics
        combined_score = (
            statistical_metrics["statistical_score"] * 0.4 +
            llm_metrics["overall"]["score"] * 0.6
        )
        
        logger.info(f"Combined evaluation score: {combined_score:.2f} (statistical: {statistical_metrics['statistical_score']:.2f}, LLM: {llm_metrics['overall']['score']:.2f})")
        
        # Determine if response is ambiguous based on both evaluations
        is_ambiguous = (
            llm_metrics["overall"]["is_ambiguous"] or 
            (statistical_metrics["is_statistically_ambiguous"] and combined_score < 0.7)
        )
        
        logger.info(f"Response is ambiguous: {is_ambiguous}")
        
        # Create combined evaluation result
        evaluation_result = {
            "statistical_metrics": statistical_metrics,
            "llm_metrics": llm_metrics,
            "combined_score": combined_score,
            "is_ambiguous": is_ambiguous,
            "issues": llm_metrics["overall"].get("issues", []),
            "follow_up_questions": llm_metrics["overall"].get("follow_up_questions", []),
            "confidence": (
                statistical_metrics["statistical_score"] * 0.3 +
                llm_metrics["overall"].get("confidence", 0.5) * 0.7
            )
        }
        
        if evaluation_result["issues"]:
            logger.info(f"Identified {len(evaluation_result['issues'])} issues: {evaluation_result['issues']}")
            
        if evaluation_result["follow_up_questions"]:
            logger.info(f"Generated {len(evaluation_result['follow_up_questions'])} follow-up questions")
            
        return evaluation_result
    
    # For backward compatibility with the original interface
    async def evaluate(self, question: str, response: str, original_query: str = None) -> Dict[str, Any]:
        """Backward-compatible evaluation method.
        
        Args:
            question: The question that was asked
            response: The response to evaluate
            original_query: The original user query (optional)
            
        Returns:
            Evaluation result dictionary
        """
        logger.info("Using backward-compatible evaluate() method")
        
        # If original_query is not provided, use the question as the original query
        if original_query is None:
            logger.debug("No original_query provided, using question as original_query")
            original_query = question
            
        return await self.evaluate_response(response, original_query, question)
    
    def generate_clarification_questions(self, evaluation_result: Dict[str, Any]) -> List[str]:
        """Generate clarification questions based on the evaluation result.
        
        Args:
            evaluation_result: The evaluation result
            
        Returns:
            List of clarification questions
        """
        logger.info("Generating clarification questions from evaluation result")
        
        # First check if we have LLM-generated follow-up questions
        if evaluation_result.get("follow_up_questions") and len(evaluation_result["follow_up_questions"]) > 0:
            logger.info(f"Using {len(evaluation_result['follow_up_questions'])} pre-generated follow-up questions")
            return evaluation_result["follow_up_questions"][:2]  # Return top 2 questions
        
        # Otherwise, generate questions based on issues
        questions = []
        issues = evaluation_result.get("issues", [])
        logger.info(f"Generating questions from {len(issues)} issues")
        
        # Map issues to question templates
        issue_to_question = {
            "unclear": "Could you provide a more clear explanation of {issue_detail}?",
            "incomplete": "Could you provide more details about {issue_detail}?",
            "missing code reference": "Could you provide specific code file or function references for {issue_detail}?",
            "contradiction": "There seems to be a contradiction regarding {issue_detail}. Could you clarify this?",
            "ambiguous": "The explanation about {issue_detail} is ambiguous. Could you be more specific?",
            "lacks detail": "Could you provide more specific details about {issue_detail}?",
            "technical": "Could you explain the technical aspect of {issue_detail} in more detail?"
        }
        
        # Generate questions from issues
        for issue in issues:
            # Try to identify the type of issue and extract details
            for issue_type, template in issue_to_question.items():
                if issue_type in issue.lower():
                    # Extract the detail part (after "is unclear", "lacks", etc.)
                    parts = re.split(f"{issue_type}[s]?\\s+(?:about|regarding|concerning|in|of|with)?\\s*", issue.lower())
                    if len(parts) > 1:
                        issue_detail = parts[1].strip()
                    else:
                        issue_detail = "this aspect"
                    
                    questions.append(template.format(issue_detail=issue_detail))
                    logger.debug(f"Generated question for issue type '{issue_type}': {template.format(issue_detail=issue_detail)}")
                    break
            else:
                # If no specific issue type found, use a generic template
                generic_question = f"Could you clarify this issue: {issue}?"
                questions.append(generic_question)
                logger.debug(f"Generated generic question for unrecognized issue: {generic_question}")
        
        # If we couldn't generate questions from issues, use generic ones based on statistical metrics
        if not questions and evaluation_result.get("statistical_metrics"):
            logger.info("No questions generated from issues, using statistical metrics")
            stats = evaluation_result["statistical_metrics"]
            
            if stats.get("code_citation_score", 1.0) < 0.5:
                questions.append("Could you provide specific code file or function references for this explanation?")
            
            if stats.get("has_uncertainty", False):
                questions.append("There seems to be uncertainty in the explanation. Could you provide more definitive information?")
            
            if stats.get("length_score", 1.0) < 0.5:
                questions.append("Could you provide a more detailed explanation?")
        
        # Add a generic question if we still have no questions
        if not questions:
            logger.info("No questions generated, adding generic fallback questions")
            questions.append("Could you provide more specific details about this functionality?")
            questions.append("Could you explain which exact function or method handles this operation?")
        
        logger.info(f"Generated {len(questions)} clarification questions")
        return questions[:2]  # Return top 2 questions