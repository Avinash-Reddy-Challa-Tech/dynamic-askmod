"""
AskMod Client - Complete Fix

This version matches your working cURL request exactly and handles timeouts properly.
Key fixes:
1. Matches the exact payload structure from your working cURL
2. Uses correct user IDs and configurations  
3. Proper timeout handling
4. Simplified query processing
5. Better error handling
"""

import json
import logging
import uuid
import asyncio
from typing import Dict, Any

import aiohttp

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AskModClient:
    """
    Client for interacting with the AskMod API.
    Fixed to match the exact working cURL request structure.
    """
    
    def __init__(self, endpoint: str, cookie: str, 
                 task_id: str = "88bb18aa-2a7d-42bb-9a66-bf6282ae44a3",
                 database_index: str = "appmod2a5bd9f2dev",
                 user_id: str = "68e648e8658ff0e1799590c4"):
        """
        Initialize the AskMod client with the expected parameters.
        
        Args:
            endpoint: The AskMod API endpoint
            cookie: Authentication cookie for the API
            task_id: The task ID to use in requests (for backward compatibility)
            database_index: The database index to use in requests (for backward compatibility)
            user_id: The user ID to use in requests
        """
        self.endpoint = endpoint
        self.cookie = cookie
        
        # User configuration - using exact values from working cURL
        self.user_name = "Avinash Reddy Challa"
        self.user_email = "avinash.challa@techolution.com"
        self.user_id_payload = "0804b20a-2414-40c8-afd1-1bf0703e9d6e"  # From working cURL
        self.current_user_id = user_id  # The one passed in constructor
        
        # Headers for the API request
        self.headers = {
            "Content-Type": "application/json",
            "Cookie": self.cookie
        }
        
        # Repository configurations that exactly match your working cURL
        self.source_repo_config = {
            "taskId": "88bb18aa-2a7d-42bb-9a66-bf6282ae44a3",
            "databaseIndex": "appmod2a5bd9f2dev",
            "project_name": "Test-cw-backend",
            "updated_at": "2025-10-24 08:02:16",
            "status": "Completed",
            "repo_url": "https://github.com/Techolution/creative-workspace-backend",
            "embedding_model": {
                "model_name": "text-embedding-ada-002",
                "model_type": "Azure OpenAI"
            },
            "organization_name": "84lumber",
            "assistantName": "appmod2a5bd9f2dev",
            "project_id": "13a5ae97-6cc5-4e25-b1b4-c32eae44abbd"
        }
        
        self.target_repo_config = {
            "taskId": "74f6cb95-a616-44d2-bb82-04731a1beefe",
            "databaseIndex": "appmod7c0dcde3dev",
            "project_name": "Userstory dashboard",
            "updated_at": "2025-11-13 13:50:26",
            "status": "Completed",
            "repo_url": "https://github.com/Avinash-Reddy-Challa-Tech/dashboard",
            "embedding_model": {
                "model_name": "text-embedding-ada-002",
                "model_type": "Azure OpenAI"
            },
            "organization_name": "techolution",
            "assistantName": "appmod7c0dcde3dev",
            "project_id": "11545074-e967-44be-a591-91f6becbcd81"
        }
        
        # GitHub token from working config
        self.github_token = "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"
        
    def _prepare_exact_payload(self, query: str, is_target: bool = False) -> Dict[str, Any]:
        """
        Prepare the exact payload that matches your working cURL request.
        
        Args:
            query: The query to send to AskMod
            is_target: Whether this query is for the target repository
            
        Returns:
            Dict containing the exact payload structure
        """
        # Select the appropriate configuration
        config = self.target_repo_config if is_target else self.source_repo_config
        
        # Generate unique IDs for this request
        request_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        
        # Ensure query has the trigger prefix
        if not query.startswith("TRIGGER DOMAIN KNOWLEDGE AGENT:"):
            query = f"TRIGGER DOMAIN KNOWLEDGE AGENT: {query}"
        
        # Create the exact payload structure from your working cURL
        return {
            "name": "appmodoncw",  # Exactly as in working cURL
            "organizationName": "techolution",  # Exactly as in working cURL
            "aiModel": "claude-35-sonnet",
            "prompt": f"User Query:  {query}",  # Note the extra space after "Query:"
            "userName": self.user_name,
            "userEmailId": self.user_email,
            "userId": self.user_id_payload,  # Using the working user ID
            "requestId": f"requestId-{request_id}",
            "replyMessage": "",
            "notificationSessionId": f"chat-session-{session_id}",
            "images": "[]",
            "isAudioAgent": False,
            "metadata": {
                "context": "{}",
                "userSelectedIntent": "",
                "userSelectedProjects": [
                    {
                        "taskId": config["taskId"],
                        "databaseIndex": config["databaseIndex"],
                        "project_name": config["project_name"],
                        "updated_at": config["updated_at"],
                        "status": config["status"],
                        "repo_url": config["repo_url"],
                        "embedding_model": config["embedding_model"],
                        "organization_name": config["organization_name"]
                    }
                ],
                "current_user_id": self.current_user_id,  # Using constructor user_id
                "github_token": self.github_token,
                "is_cw_flow": True,
                "send_metadata_to_orchestrator": True,
                "organization_name": config["organization_name"],  # Repository specific org
                "selectedProjectDetails": {
                    "assistantName": config["assistantName"],
                    "project_name": config["project_name"],
                    "project_id": config["project_id"],
                    "task_id": config["taskId"],
                    "current_user_id": self.current_user_id,
                    "owner_user_id": self.current_user_id,
                    "embedding_model": config["embedding_model"],
                    "datasource": ""
                }
            }
        }
    

    async def send_query(self, query: str, is_target: bool = False) -> str:
        """
        Send a query to the AskMod API with proper error handling and timeouts.
        
        Args:
            query: The query to send
            is_target: Whether this query is for the target repository
            
        Returns:
            The answer from AskMod response
        """
        repo_type = "target" if is_target else "source"
        config = self.target_repo_config if is_target else self.source_repo_config
        
        logger.info(f"Sending query to AskMod for {repo_type} repository: {query[:100]}...")
        logger.info(f"Using {repo_type} repository configuration:")
        logger.info(f"  Task ID: {config['taskId']}")
        logger.info(f"  Database Index: {config['databaseIndex']}")
        logger.info(f"  Organization: {config['organization_name']}")
        logger.info(f"  Project: {config['project_name']}")
        
        # Prepare the exact payload
        payload = self._prepare_exact_payload(query, is_target)
        
        # Log the payload for debugging (first 500 chars)
        logger.debug(f"Payload preview: {json.dumps(payload, indent=2)[:500]}...")
        
        # Set timeout to 120 seconds to match your successful requests
        timeout = aiohttp.ClientTimeout(total=120, connect=30, sock_read=120)
        
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self.endpoint, 
                    headers=self.headers,
                    json=payload
                ) as response:
                    
                    logger.info(f"Received response status: {response.status}")
                    
                    if response.status == 200:
                        try:
                            # Get the raw response text
                            response_text = await response.text()
                            logger.debug(f"Raw response (first 200 chars): {response_text[:200]}...")
                            
                            # Parse as JSON
                            data = json.loads(response_text)
                            
                            # Extract the answer using the exact structure from your image
                            if "result" in data and isinstance(data["result"], dict):
                                if "Answer" in data["result"]:
                                    answer = data["result"]["Answer"]
                                    if answer and isinstance(answer, str) and len(answer.strip()) > 0:
                                        # Check if it's a domain rejection
                                        logger.info(f"Successfully received answer from {repo_type} repository (length: {len(answer)})")
                                        return answer
                                    else:
                                        logger.warning("Empty or invalid answer received")
                                else:
                                    logger.warning("No 'Answer' field in result")
                            else:
                                logger.warning("Invalid response structure")
                                
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON parsing error: {str(e)}")
                            
                        except Exception as e:
                            logger.error(f"Error processing response: {str(e)}")
                    else:
                        error_text = await response.text()
                        logger.error(f"HTTP error {response.status}: {error_text[:200]}...")
                        
        except asyncio.TimeoutError:
            logger.error(f"Request timed out for {repo_type} repository")
        except Exception as e:
            logger.error(f"Unexpected error during API request: {str(e)}")

    async def send_query_with_context(self, query: str, source_context: str = "", is_target: bool = False) -> str:
        """
        Send a query with additional context from the source repository.
        
        Args:
            query: The main query to send
            source_context: Context from the source repository
            is_target: Whether this query is for the target repository
            
        Returns:
            The answer from AskMod
        """
        # If we have source context, include it in the query
        if source_context and is_target:
            enhanced_query = f"{query}\n\nContext from source repository:\n{source_context[:500]}..."
            return await self.send_query(enhanced_query, is_target)
        else:
            return await self.send_query(query, is_target)