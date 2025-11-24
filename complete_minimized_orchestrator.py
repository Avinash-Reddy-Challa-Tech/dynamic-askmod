"""
Minimized Orchestrator - COMPLETE Flow Preserved

This version keeps 100% of the original logic:
- RAG context gathering
- Max 3 iterations loop
- Orchestration decision making (fetch_code/follow_up/synthesize)
- Response evaluation
- Code extraction
- ALL prompts exactly the same
- ALL LLM calls exactly the same

Just cleaner, more compact code.
"""

import json
import uuid
import sys
import asyncio
import logging
import os
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Set
from gemini_integration import call_llm

import aiohttp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# LLM WRAPPER - Replace with your actual implementation
# ============================================================================

# async def call_llm(prompt: str, temperature: float = 0.2) -> str:
#     """
#     Replace with your actual LLM implementation.
#     Must match your existing LangChain call behavior.
    
#     Example:
#         from langchain_google_genai import ChatGoogleGenerativeAI
#         llm = ChatGoogleGenerativeAI(temperature=temperature, model="gemini-2.5-flash")
#         response = await llm.ainvoke(prompt)
#         return response.content
#     """
#     raise NotImplementedError("Implement with your actual LLM")


# ============================================================================
# ASKMOD CLIENT - Same payload logic as original
# ============================================================================

class AskModClient:
    """Same as original - exact payloads"""
    
    def __init__(self, endpoint: str, cookie: str):
        self.endpoint = endpoint
        self.cookie = cookie
        self.headers = {"Content-Type": "application/json"}
        
        # Same user config
        self.user_name = "Avinash Reddy Challa"
        self.user_email = "avinash.challa@techolution.com"
        self.user_id_payload = "0804b20a-2414-40c8-afd1-1bf0703e9d6e"
        self.current_user_id = "68e648e8658ff0e1799590c4"
        self.github_token = ""

        ### DEV ###
        self.concierge_id = "ea849533-fbb8-4645-81f8-445fb85ea68c"
        self.cw_core_assistant_name = "devlscassistant"
        self.agent_name = "CreativeLSCToModeConnectorAgentV1"
        self.agent_id = "68b54504abdb190b9eb73dc5"
        self.rag_agent_url = "https://dev-proposals-ai.techo.camp/utility/creative-lsc-to-mode-connector-agent"

        self.websocket_url = "wss://dev-proposals-ai.techo.camp/graphql"
        self.organization_name = "techolution"
        self.chat_assistant_name = "appmodoncw"

        # ### prod ###
        # self.concierge_id = "c19f796d-8fa4-4846-9a2f-af0065bb350f"
        # self.cw_core_assistant_name = "cwlscassistant"
        # self.agent_name = "CreativeLSCToModeConnectorAgent"
        # self.agent_id = "68ba66bb4d31614de6d7d7ec"
        # self.rag_agent_url = "https: //ellm.creativeworkspace.ai/utility/creative-lsc-to-mode-connector-agent"

        # self.websocket_url = "wss://dev-proposals-ai.techo.camp/graphql"
        # self.organization_name = "techolution"
        # self.chat_assistant_name = "appmodoncw"

        


        # Same repo configs
        self.source_config = {
            "taskId": "a8d5f6ec-2cb9-4f5c-951c-a184d2b33471",
            "databaseIndex": "appmodb2a99c5adev",
            "project_name": "Appmod-Agents",
            "updated_at": "2025-09-08 09:56:31",
            "status": "Completed",
            "repo_url": "https://github.com/Techolution/Appmod-Agents",
            "embedding_model": {"model_name": "text-embedding-ada-002", "model_type": "Azure OpenAI"},
            "organization_name": "84lumber",
            "assistantName": "appmodb2a99c5adev",
            "project_id": "0cb7baff-8cff-4ef3-965d-7d797401c6af",
            "user_id": "667bdcbc6bd304c195e1b33e"
        }
        

        self.target_config = {
            "taskId": "fb99f5eb-4dbe-4d9c-b69e-499b574fd683",
            "databaseIndex": "appmodffdd03d5dev",
            "project_name": "citation library",
            "updated_at": "2025-05-23 21:14:22",
            "status": "Completed",
            "repo_url": "https://github.com/Techolution/e-llm-studio-lib",
            "embedding_model": {"model_name": "text-embedding-ada-002", "model_type": "Azure OpenAI"},
            "organization_name": "84lumber",
            "assistantName": "appmodoncw",
            "project_id": "486c457f-792e-4353-858e-110efc0c7503",
            "user_id": "667bdcbc6bd304c195e1b33e"
        }
    
    async def send_query(self, query: str, is_target: bool = False) -> str:
        """Same as original"""
        logger.info(f"Sending query (ASKMOD CALL): {query}")

        config = self.target_config if is_target else self.source_config
        
        # if not query.startswith("TRIGGER DOMAIN KNOWLEDGE AGENT:"):
        #     query = f"TRIGGER DOMAIN KNOWLEDGE AGENT: {query}"

        # Generate unique IDs for this request
        request_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        chat_preview_id = str(uuid.uuid4())
        organization_name = config['organization_name']


        payload = {
                    "concierge_id": self.concierge_id,
                    "concierge_name": self.cw_core_assistant_name,
                    "request_id": request_id,
                    "agent_name": self.agent_name,
                    "agent_id": self.agent_id,
                    "agent_url": self.rag_agent_url,
                    "agent_arguments": { "user_query": query + " LSCMetaData{'activeModeName':'Appmod.AI Mode'}" },
                    "chat_history": [],
                    "question": query + " LSCMetaData{'activeModeName':'Appmod.AI Mode'}",
                    "prompt": "",
                    "modelType": "google",
                    "preview_id": chat_preview_id,
                    "orchByPassed": False,
                    "agent_config": {},
                    "metadata": {
                        "userName": "User", 
                        "userEmailId": "",
                        "llm": "claude-3.5-sonnet",
                        "remoteModeLLMAssistantDetails": {
                            "WEBSOCKET_URL": self.websocket_url,
                            "organizationName": self.organization_name,
                            "chatAssistantName": self.chat_assistant_name,
                            "selectedAIModel": "claude-35-sonnet",
                            "userName": "User",
                            "userEmailId": "",
                            "userId": config['user_id'],
                            "query": query + " LSCMetaData{'activeModeName':'Appmod.AI Mode'}",
                            "requestId": request_id,
                            "guestSessionId": session_id,
                            "chatPreviewId": chat_preview_id,
                            "sessionId": session_id,
                            "replyMessage": "",
                            "notificationSessionId": f"chat-session-{str(uuid.uuid4())}",
                            "images": None,
                            "customMetaData": {
                                "context": "{}",
                                "branchName": "master",
                                "userSelectedIntent": "",
                                "userSelectedProjects": [{
                                    "taskId": config['taskId'],
                                    "databaseIndex": config['databaseIndex'],
                                    "project_name": "User Story Project",
                                    "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "status": "Completed",
                                    "repo_url": config['repo_url'],
                                    "embedding_model": {"model_name": "text-embedding-ada-002", "model_type": "Azure OpenAI"},
                                }],
                                "organization_name": organization_name,
                                "current_user_id": config['user_id'],
                                "github_token": self.github_token,
                                "is_cw_flow": True,
                                "send_metadata_to_orchestrator": True,
                                "selectedProjectDetails": {
                                    "assistantName": config['assistantName'],
                                    "project_name": "User Story Project",
                                    "project_id": str(uuid.uuid4()),
                                    "task_id": config['taskId'],
                                    "current_user_id": config['user_id'],
                                    "owner_user_id": "",
                                    "embedding_model": {"model_name": "text-embedding-ada-002", "model_type": "Azure OpenAI"},
                                    "datasource": "codebase"
                                }
                            },
                            "textToSpeechAgentId": "68ac33a84d31614de6d36a01",
                            "useTextToSpeech": True,
                            "id": str(uuid.uuid4()),
                            "isAudioAgent": True,
                            "addUserMessageInChat": False,
                            "isDocumentRetrieval": False,
                            "multiAgentToggle": False,
                            "selectedAgent": None,
                            "runtimeAgents": None,
                            "agentSettings": None,
                            "customRootOrchestratorModelName": None,
                            "customRootOrchestratorModelType": None,
                            "customPrompt": "",
                            "sameGuestSessionIdAsCentralAssistant": "true"
                        },
                        "currentActiveModeOfUser": "Appmod.AI Mode",
                        "userId": config['user_id']
                    },
                    "agent_settings": {
                         "embeddingColumnName": "mxbai_embed_large_v1_question_embeddings",
                         "embeddingModel": "mxbai-embed-large-v1",
                         "overrideAgentConfig": None,
                         "rootOrchestratorConfig": {
                            "useGuide": False,
                            "dbIndexName": "techolution-cwlscassistant-agentGuide",
                            "dbType": "alloydb",
                            "guidePrompt": "You are strictly responsible...",
                            "conflictPrompt": "You are an intelligent assistant...",
                         }
                    }
                }


        # with open("payload.txt", "w") as f:
        #     f.write(str(payload))

        with open("payload.json", "w") as f:
            json.dump(payload, f, indent=2)

        # timeout = aiohttp.ClientTimeout(total=120)
        timeout = aiohttp.ClientTimeout(total=120, connect=30, sock_read=300)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(self.endpoint, headers=self.headers, json=payload) as response:
                if response.status == 200:
                    print("\n\n")
                    print("ASKMOD RESPONSE RECEIVED")
                    data = await response.json()
                    print(str(data)[:1000])

                    if "response" in data:
                        print(f"Response found in data: {data['response']}")

                        return data['response']
                return f"Error: {response.status}"


# ============================================================================
# QUERY DECOMPOSER - Same prompts as original
# ============================================================================

class QueryDecomposer:
    """Same prompt template as original"""
    
    def __init__(self, max_questions: int = 3):
        self.max_questions = max_questions
    
    async def generate_source_target_questions(self, query: str, source_desc: str, target_desc: str) -> Tuple[List[str], List[str]]:
        """Same as original - exact prompt"""
        
        # EXACT same prompt from original
        prompt = f"""You are an expert at breaking down complex questions about code and software into more specific, targeted sub-questions for both source and target repositories.

Original Query: {query}

Source Repository Description: {source_desc}
Target Repository Description: {target_desc}

Your task is to break down this query into {self.max_questions} pairs of specific sub-questions where:
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
1. A list of {self.max_questions} source questions
2. A corresponding list of {self.max_questions} target questions
3. A brief reasoning for why these question pairs will help answer the original query

{{
  "source_questions": ["question1", "question2", "question3"],
  "target_questions": ["question1", "question2", "question3"],
  "reasoning": "explanation"
}}"""

        response = await call_llm(prompt, temperature=0.2)
        
        # Parse JSON
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r'({[\s\S]*})', response)
            json_str = json_match.group(1) if json_match else response
        
        result = json.loads(json_str)
        
        # Ensure trigger prefix
        source_questions = [
            q if q.startswith("TRIGGER DOMAIN KNOWLEDGE AGENT:") else f"TRIGGER DOMAIN KNOWLEDGE AGENT: {q}"
            for q in result.get("source_questions", [])
        ]
        
        target_questions = [
            q if q.startswith("TRIGGER DOMAIN KNOWLEDGE AGENT:") else f"TRIGGER DOMAIN KNOWLEDGE AGENT: {q}"
            for q in result.get("target_questions", [])
        ]
        
        return source_questions, target_questions


# ============================================================================
# RESPONSE EVALUATOR - Same logic as original
# ============================================================================

class ResponseEvaluator:
    """Same evaluation logic as original"""
    
    async def evaluate_response(self, response_text: str, original_query: str, sub_question: str) -> Dict[str, Any]:
        """Same as original - statistical + LLM evaluation"""
        
        # Statistical metrics (same as original)
        code_citations = len(re.findall(r'\[([^\]]+)\]\(([^)]+)\)', response_text))
        query_terms = set(re.findall(r'\b\w{3,}\b', original_query.lower()))
        response_terms = set(re.findall(r'\b\w{3,}\b', response_text.lower()))
        query_term_overlap = len(query_terms.intersection(response_terms)) / max(1, len(query_terms))
        
        has_uncertainty = len(re.findall(r'(?:might|may|could|unclear|uncertain)', response_text, re.IGNORECASE)) > 2
        
        statistical_score = min(1.0, (
            min(1.0, len(response_text) / 1000) * 0.2 +
            min(1.0, code_citations / 5) * 0.4 +
            (0.0 if has_uncertainty else 0.3) * 0.2 +
            query_term_overlap * 0.2
        ))
        
        # LLM evaluation (same prompt as original)
        eval_prompt = f"""You are evaluating the quality of a response from a code documentation system.

Original User Query: {original_query}
Specific Question: {sub_question}
Response: {response_text}

Evaluate the response on the following criteria (0-10 each):
1. Clarity: Is the response clear and easy to understand?
2. Completeness: Does it fully answer the specific question?
3. Code Grounding: Does it reference specific code files or functions?
4. Structure: Is it well-organized with clear sections?
5. Accuracy: Does it appear accurate based on internal consistency?

Respond with JSON:
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
}}"""
        
        try:
            llm_response = await call_llm(eval_prompt, temperature=0.1)
            json_match = re.search(r'```json\s*(.*?)\s*```', llm_response, re.DOTALL)
            if json_match:
                llm_metrics = json.loads(json_match.group(1))
            else:
                llm_metrics = json.loads(llm_response)
        except:
            llm_metrics = {
                "overall": {"score": 0.5, "is_ambiguous": True, "issues": [], "follow_up_questions": [], "confidence": 0.3}
            }
        
        combined_score = statistical_score * 0.4 + llm_metrics["overall"]["score"] * 0.6
        
        return {
            "statistical_metrics": {"statistical_score": statistical_score, "code_citations": code_citations, "query_term_overlap": query_term_overlap},
            "llm_metrics": llm_metrics,
            "combined_score": combined_score,
            "is_ambiguous": llm_metrics["overall"]["is_ambiguous"],
            "issues": llm_metrics["overall"].get("issues", []),
            "follow_up_questions": llm_metrics["overall"].get("follow_up_questions", []),
            "confidence": combined_score
        }


# ============================================================================
# CODE EXTRACTOR - Same logic as original
# ============================================================================

class CodeExtractor:
    """Same code extraction logic as original"""
    
    def __init__(self, base_url: str = "https://dev-appmod.techo.camp/analyzer/get_file_details"):
        self.base_url = base_url
        self.file_cache = {}
    
    def extract_code_citations(self, response_text: str) -> Set[str]:
        """Extract code file citations from response"""
        citations = set()
        
        # Markdown links
        markdown_links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', response_text)
        for link_text, url in markdown_links:
            if any(ext in url for ext in [".py", ".js", ".ts", ".jsx", ".tsx"]):
                citations.add(url)
        
        # File paths in text
        file_paths = re.findall(r'(?:file|module|path)[\s:]+[\'"`]([\w\-./]+\.(py|js|ts|jsx|tsx))[\'"`]', response_text)
        for path, _ in file_paths:
            citations.add(path)
        
        return citations
    
    async def fetch_code_content(self, citation: str) -> Dict[str, Any]:
        """Fetch code content for citation"""
        
        # Check cache
        if citation in self.file_cache:
            return self.file_cache[citation]
        
        # Build API URL
        file_path = citation.split("file_path=")[-1].split("&")[0] if "file_path=" in citation else citation
        api_url = f"{self.base_url}?file_path={file_path}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(api_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data.get("content", "") or (data.get("0", {}).get("content", "") if isinstance(data, dict) else "")
                        
                        result = {"file_path": file_path, "content": content}
                        self.file_cache[citation] = result
                        return result
        except:
            pass
        
        return {"file_path": file_path, "content": None}


# ============================================================================
# RESPONSE SYNTHESIZER - Same prompt as original
# ============================================================================

class ResponseSynthesizer:
    """Same synthesis prompt as original"""
    
    async def synthesize(self, original_query: str, qa_pairs: List[Dict[str, Any]]) -> str:
        """Same as original - exact prompt"""
        
        # Organize by repository
        source_qa = [qa for qa in qa_pairs if qa.get("repository") == "source"]
        target_qa = [qa for qa in qa_pairs if qa.get("repository") == "target"]
        
        source_info = "\n\n".join([f"### Q: {qa['question']}\n\nA: {qa['answer']}" for qa in source_qa])
        target_info = "\n\n".join([f"### Q: {qa['question']}\n\nA: {qa['answer']}" for qa in target_qa])
        
        # Extract code examples
        code_examples = ""
        for qa in qa_pairs:
            code_blocks = re.findall(r'```(?:\w+)?\n(.*?)\n```', qa.get("answer", ""), re.DOTALL)
            if code_blocks:
                code_examples += f"\n### {qa['question'][:60]}...\n"
                for block in code_blocks[:2]:
                    code_examples += f"```\n{block}\n```\n"
        
        # EXACT same prompt from original
        prompt = f"""You are an expert at synthesizing information about code implementations and explaining how to port features from one repository to another.

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

Synthesized answer:"""

        return await call_llm(prompt, temperature=0.3)


# ============================================================================
# ORCHESTRATOR - Complete flow with iterations
# ============================================================================

class Orchestrator:
    """Complete orchestration flow - same as original"""
    
    def __init__(self, askmod_client: AskModClient, max_iterations: int = 3):
        self.client = askmod_client
        self.decomposer = QueryDecomposer()
        self.evaluator = ResponseEvaluator()
        self.synthesizer = ResponseSynthesizer()
        self.code_extractor = CodeExtractor()
        self.max_iterations = max_iterations
        
        os.makedirs("responses", exist_ok=True)
        os.makedirs("decisions", exist_ok=True)
        os.makedirs("code_citations", exist_ok=True)
    
    async def _generate_paired_questions(self, user_query: str) -> Tuple[List[str], List[str]]:
        """Same as original"""
        return await self.decomposer.generate_source_target_questions(
            user_query,
            "Creative Workspace Backend (84lumber)",
            "Userstory Dashboard (techolution)"
        )
    
    async def _process_paired_questions(self, questions: Tuple[List[str], List[str]]) -> List[Dict[str, Any]]:
        """Same as original - process source and target questions with evaluation"""
        source_questions, target_questions = questions
        qa_pairs = []
        
        for source_q, target_q in zip(source_questions, target_questions):
            # Source question
            logger.info(f"Processing source question: {source_q}")
            source_answer = await self.client.send_query(source_q, is_target=False)

            logger.info(f"Source answer: {source_answer}")
            source_eval = await self.evaluator.evaluate_response(source_answer, source_q, source_q)

            logger.info(f"Source evaluation: {source_eval}")
            qa_pairs.append({
                "question": source_q,
                "answer": source_answer,
                "repository": "source",
                "evaluation": source_eval,
                "clarity_score": source_eval["combined_score"],
                "is_clear": not source_eval["is_ambiguous"]
            })
            
            # Target question with source context
            target_enhanced = f"{target_q}\n\nSource Repository Information:\n{source_answer}"
            target_answer = await self.client.send_query(target_enhanced, is_target=True)
            target_eval = await self.evaluator.evaluate_response(target_answer, target_q, target_q)
            
            qa_pairs.append({
                "question": target_q,
                "answer": target_answer,
                "repository": "target",
                "evaluation": target_eval,
                "clarity_score": target_eval["combined_score"],
                "is_clear": not target_eval["is_ambiguous"],
                "source_context": source_answer
            })
        
        return qa_pairs
    
    async def _make_orchestration_decision(self, user_query: str, qa_pairs: List[Dict], retrieved_code: Dict) -> Dict[str, Any]:
        """Same as original - LLM decides next action"""
        
        # Format current state
        current_responses = "\n\n".join([
            f"Q ({qa.get('repository', 'unknown')}): {qa['question']}\nA: {qa['answer'][:500]}...\nClarity: {qa.get('clarity_score', 0):.2f}"
            for qa in qa_pairs
        ])
        
        retrieved_code_str = "\n\n".join([
            f"File: {path}\n```\n{content[:500]}...\n```"
            for path, content in list(retrieved_code.items())[:3]
        ])
        
        # EXACT same decision prompt from original
        prompt = f"""You are an intelligent orchestrator that decides the next steps in a code documentation RAG system.

Original User Query: {user_query}

CURRENT RESPONSES:
{current_responses}

RETRIEVED CODE (if any):
{retrieved_code_str}

Your task is to decide the next step based on a careful analysis of the above information.

Possible decisions:
1. "fetch_code" - If you need to examine specific code files to better understand the responses
2. "follow_up" - If you need clarification on specific aspects of the responses
3. "synthesize" - If you have enough information to provide a complete answer

Respond with ONLY a JSON object:
{{
  "decision": "fetch_code|follow_up|synthesize",
  "citations_to_fetch": ["file1", "file2"],
  "follow_up_questions": ["q1", "q2"],
  "is_complete": true|false,
  "target_repo": true|false,
  "reason": "explanation",
  "confidence": 0.85
}}"""

        try:
            response = await call_llm(prompt, temperature=0.1)
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                decision = json.loads(json_match.group(1))
            else:
                decision = json.loads(response)
        except:
            # Fallback decision
            decision = {
                "decision": "synthesize" if len(qa_pairs) >= 4 else "follow_up",
                "citations_to_fetch": [],
                "follow_up_questions": [],
                "is_complete": len(qa_pairs) >= 4,
                "target_repo": False,
                "reason": "Fallback decision",
                "confidence": 0.5
            }
        
        # Save decision
        with open("decisions/decision.json", "w") as f:
            json.dump(decision, f, indent=2)
        
        return decision
    
    async def _process_follow_up_questions(self, questions: List[str], is_target: bool) -> List[Dict[str, Any]]:
        """Same as original"""
        qa_pairs = []
        
        for question in questions:
            answer = await self.client.send_query(question, is_target=is_target)
            evaluation = await self.evaluator.evaluate_response(answer, question, question)
            
            qa_pairs.append({
                "question": question,
                "answer": answer,
                "repository": "target" if is_target else "source",
                "evaluation": evaluation,
                "clarity_score": evaluation["combined_score"],
                "is_clear": not evaluation["is_ambiguous"]
            })
        
        return qa_pairs
    
    async def _fetch_code_for_citations(self, citations: List[str]) -> Dict[str, str]:
        """Same as original"""
        code_files = {}
        
        for citation in citations[:3]:  # Limit to 3
            result = await self.code_extractor.fetch_code_content(citation)
            if result.get("content"):
                code_files[result["file_path"]] = result["content"]
                
                # Save code file
                safe_filename = re.sub(r'[\\/*?:"<>|]', "_", result["file_path"])
                with open(f"code_citations/{safe_filename}", "w") as f:
                    f.write(result["content"])
        
        return code_files
    
    async def process_query(self, user_query: str) -> Dict[str, Any]:
        """
        COMPLETE ORIGINAL FLOW:
        1. Initialize state
        2. Iteration loop (max 3):
           - First iteration: Generate & process paired questions
           - Subsequent: Make decision (fetch_code/follow_up/synthesize)
           - Break if complete
        3. Synthesize final answer
        """
        logger.info(f"Processing query: {user_query}")
        
        all_qa_pairs = []
        retrieved_code = {}
        fetched_citations = set()
        iteration = 0
        
        # MAIN ITERATION LOOP (same as original)
        while iteration < self.max_iterations:
            logger.info(f"Starting iteration {iteration + 1}/{self.max_iterations}")
            
            # FIRST ITERATION: Generate and process paired questions
            if not all_qa_pairs:
                logger.info("Generating paired questions...")
                questions = await self._generate_paired_questions(user_query)
                
                logger.info("Processing paired questions...")
                new_qa_pairs = await self._process_paired_questions(questions)
                all_qa_pairs.extend(new_qa_pairs)
                
                logger.info(f"Collected {len(new_qa_pairs)} QA pairs")
            
            # SUBSEQUENT ITERATIONS: Make orchestration decision
            else:
                logger.info("Making orchestration decision...")
                decision = await self._make_orchestration_decision(user_query, all_qa_pairs, retrieved_code)
                
                logger.info(f"Decision: {decision['decision']} (confidence: {decision.get('confidence', 0):.2f})")
                
                # FETCH CODE
                if decision["decision"] == "fetch_code":
                    new_citations = set(decision.get("citations_to_fetch", [])) - fetched_citations
                    if new_citations:
                        logger.info(f"Fetching code for {len(new_citations)} citations")
                        code_files = await self._fetch_code_for_citations(list(new_citations))
                        retrieved_code.update(code_files)
                        fetched_citations.update(new_citations)
                
                # FOLLOW-UP QUESTIONS
                elif decision["decision"] == "follow_up":
                    follow_up_questions = decision.get("follow_up_questions", [])
                    if follow_up_questions:
                        logger.info(f"Processing {len(follow_up_questions)} follow-up questions")
                        follow_up_qa = await self._process_follow_up_questions(
                            follow_up_questions,
                            decision.get("target_repo", False)
                        )
                        all_qa_pairs.extend(follow_up_qa)
                
                # CHECK IF COMPLETE
                if decision.get("is_complete") or decision["decision"] == "synthesize":
                    logger.info("Decision: Ready to synthesize")
                    break
            
            iteration += 1
        
        logger.info(f"Completed {iteration} iterations with {len(all_qa_pairs)} QA pairs")
        
        # SYNTHESIZE FINAL ANSWER (same as original)
        logger.info("Synthesizing final answer...")
        
        # Add retrieved code to QA pairs
        for file_path, code_content in retrieved_code.items():
            all_qa_pairs.append({
                "question": f"Code from {file_path}",
                "answer": f"```\n{code_content}\n```",
                "repository": "source",
                "clarity_score": 0.95,
                "is_clear": True
            })
        
        final_answer = await self.synthesizer.synthesize(user_query, all_qa_pairs)
        
        # Save result
        with open("responses/final_answer.txt", "w") as f:
            f.write(final_answer)
        
        logger.info("Process complete!")
        
        return {
            "result": {
                "answer": final_answer,
                "qa_pairs_count": len(all_qa_pairs),
                "iterations": iteration,
                "retrieved_code_count": len(retrieved_code),
                "source_repository": "creative-workspace-backend",
                "target_repository": "dashboard"
            }
        }


# ============================================================================
# MAIN
# ============================================================================

async def process_query(query: str) -> dict:
    """Main entry point"""
    
    ASKMOD_ENDPOINT = os.getenv('ASKMOD_ENDPOINT', 'https://dev-proposals-ai.techo.camp/utility/creative-lsc-to-mode-connector-agent/prediction')
    # ASKMOD_ENDPOINT = "https://ellm.creativeworkspace.ai/utility/creative-lsc-to-mode-connector-agent/prediction"
    ASKMOD_COOKIE = os.getenv('ASKMOD_COOKIE', 'Enterprise-GPT-Maker-techolution-appmodoncw=techolution-appmodoncw-aa48c862-9eea-4969-8874-42259302b340')

    logger.info(f"Using AskMod endpoint: {ASKMOD_ENDPOINT}, cookie length: {len(ASKMOD_COOKIE)}")
    
    client = AskModClient(ASKMOD_ENDPOINT, ASKMOD_COOKIE)
    orchestrator = Orchestrator(client, max_iterations=3)
    
    return await orchestrator.process_query(query)


async def main():
    """CLI entry point"""
    import sys

    logger.info("Starting Minimized Orchestrator...")
    
    query = sys.argv[1] if len(sys.argv) > 1 else input("Enter your query: ")
    
    print(f"Processing: {query}\n")
    result = await process_query(query)
    
    print("\n" + "="*80)
    print("IMPLEMENTATION GUIDE")
    print("="*80)
    print(result["result"]["answer"])
    print("="*80)
    
    print(f"\n✓ Completed in {result['result']['iterations']} iterations")
    print(f"✓ Collected {result['result']['qa_pairs_count']} QA pairs")
    print(f"✓ Retrieved {result['result']['retrieved_code_count']} code files")


if __name__ == "__main__":
    asyncio.run(main())
