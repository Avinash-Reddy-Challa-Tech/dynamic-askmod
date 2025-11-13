"""
AskMod Client

This module provides a client for interacting with the AskMod API, handling authentication,
request formatting, and response parsing.
"""

import json
import logging
import uuid
from typing import Dict, Any, Optional

import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AskModClient:
    """
    Client for interacting with the AskMod API.
    """
    
    def __init__(self, endpoint: str, cookie: str, 
                 organization_name: str = "techolution",
                 task_id: str = "88bb18aa-2a7d-42bb-9a66-bf6282ae44a3",
                 database_index: str = "appmod2a5bd9f2dev",
                 user_id: str = "68e648e8658ff0e1799590c4",
                 user_name: str = "Avinash Reddy Challa",
                 user_email: str = "avinash.challa@techolution.com",
                 github_token: str = "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"):
        """
        Initialize the AskMod client.
        
        Args:
            endpoint: The AskMod API endpoint
            cookie: Authentication cookie for the API
            organization_name: The organization name to use in requests
            task_id: The task ID to use in requests
            database_index: The database index to use in requests
            user_id: The user ID to use in requests
            user_name: User's full name
            user_email: User's email address
            github_token: GitHub access token
        """
        self.endpoint = endpoint
        self.cookie = cookie
        self.organization_name = organization_name
        self.task_id = task_id
        self.database_index = database_index
        self.user_id = user_id
        self.user_name = user_name
        self.user_email = user_email
        self.github_token = github_token
        
        # Headers for the API request
        self.headers = {
            "Content-Type": "application/json",
            "Cookie": self.cookie
        }
        
    def _prepare_payload(self, query: str) -> Dict[str, Any]:
        """
        Prepare the payload for the AskMod API request based on the Postman example.
        
        Args:
            query: The query to send to AskMod
            
        Returns:
            Dict containing the formatted payload
        """
        # Generate unique IDs for this request
        request_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        
        return {
            "name": "appmodoncw",
            "organizationName": self.organization_name,
            "aiModel": "claude-35-sonnet",
            "prompt": f"User Query: {query}",
            "userName": self.user_name,
            "userEmailId": self.user_email,
            "userId": self.user_id,
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
                        "taskId": self.task_id,
                        "databaseIndex": self.database_index,
                        "project_name": "Test-cw-backend",
                        "updated_at": "2025-10-24 08:02:16",
                        "status": "Completed",
                        "repo_url": "https://github.com/Techolution/creative-workspace-backend",
                        "embedding_model": {
                            "model_name": "text-embedding-ada-002",
                            "model_type": "Azure OpenAI"
                        },
                        "organization_name": "84lumber"
                    }
                ],
                "current_user_id": self.user_id,
                "github_token": self.github_token,
                "is_cw_flow": True,
                "send_metadata_to_orchestrator": True,
                "organization_name": "84lumber",
                "selectedProjectDetails": {
                    "assistantName": "appmod2a5bd9f2dev",
                    "project_name": "Test-cw-backend",
                    "project_id": "13a5ae97-6cc5-4e25-b1b4-c32eae44abbd",
                    "task_id": self.task_id,
                    "current_user_id": self.user_id,
                    "owner_user_id": self.user_id,
                    "embedding_model": {
                        "model_name": "text-embedding-ada-002",
                        "model_type": "Azure OpenAI"
                    },
                    "datasource": ""
                }
            }
        }
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, json.JSONDecodeError))
    )
    async def send_query(self, query: str) -> str:
        """
        Send a query to the AskMod API and get the response.
        
        Args:
            query: The query to send
            
        Returns:
            The answer from AskMod
        """
        payload = self._prepare_payload(query)
        logger.info(f"Sending query to AskMod: {query}")
        
        # Log the full payload for debugging
        logger.debug(f"Full payload: {json.dumps(payload, indent=2)}")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.endpoint, 
                                   headers=self.headers,
                                   json=payload) as response:
                
                # Check if the request was successful
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Error from AskMod API: {response.status} - {error_text}")
                    return f"Error: AskMod API returned status code {response.status}"
                
                # Parse the response
                try:
                    # Get the raw response text first for debugging
                    response_text = await response.text()
                    logger.debug(f"Raw response: {response_text[:500]}...")
                    
                    # Parse as JSON
                    data = json.loads(response_text)
                    
                    # Based on the example response, look for result.Answer specifically
                    if "result" in data and isinstance(data["result"], dict) and "Answer" in data["result"]:
                        answer = data["result"]["Answer"]
                        if answer:
                            logger.info(f"Received answer from AskMod (length: {len(str(answer))})")
                            logger.info(f"response: {str(answer)[:100]}...")
                            return answer
                    
                    # Fallback parsing for different response structures
                    logger.warning(f"Expected result.Answer not found. Trying alternative parsing.")
                    
                    if "result" in data:
                        if isinstance(data["result"], dict):
                            # Try other possible field names in the result object
                            for field in ["answer", "response", "content", "text"]:
                                if field in data["result"]:
                                    answer = data["result"][field]
                                    logger.info(f"Found answer in result.{field} (length: {len(str(answer))})")
                                    return answer
                            
                            # Log what we found in the result if we couldn't find the answer
                            logger.warning(f"Could not find answer in result. Available fields: {list(data['result'].keys())}")
                            return str(data["result"])
                        elif isinstance(data["result"], str):
                            logger.info(f"Found string result (length: {len(data['result'])})")
                            return data["result"]
                    
                    # If we still couldn't extract an answer, log the response structure and return a fallback
                    logger.warning(f"Could not extract answer. Response structure: {list(data.keys())}")
                    logger.debug(f"Full response: {json.dumps(data)[:1000]}...")
                    
                    # Last resort: return a generic message
                    return "Unable to extract answer from API response. Please check the logs for details."
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing JSON response: {str(e)}")
                    response_text = await response.text()
                    logger.error(f"Raw response: {response_text[:200]}...")
                    # Return the raw response if we can't parse it
                    return f"Error parsing response: {response_text}"
                except Exception as e:
                    logger.error(f"Unexpected error processing response: {str(e)}")
                    return f"Error processing response: {str(e)}"