"""
AskMod Client

This module provides a client for interacting with the AskMod API, handling authentication,
request formatting, and response parsing.
"""

import json
import logging
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
                 task_id: str = "7c377dee-a767-43bc-ac52-63b16187391c",
                 database_index: str = "appmoda9c40dev",
                 user_id: str = "66d977791e9c242063fd3a1e"):
        """
        Initialize the AskMod client.
        
        Args:
            endpoint: The AskMod API endpoint
            cookie: Authentication cookie for the API
            organization_name: The organization name to use in requests
            task_id: The task ID to use in requests
            database_index: The database index to use in requests
            user_id: The user ID to use in requests
        """
        self.endpoint = endpoint
        self.cookie = cookie
        self.organization_name = organization_name
        self.task_id = task_id
        self.database_index = database_index
        self.user_id = user_id
        
        # Headers for the API request
        self.headers = {
            "Content-Type": "application/json",
            "Cookie": self.cookie
        }
        
    def _prepare_payload(self, query: str) -> Dict[str, Any]:
        """
        Prepare the payload for the AskMod API request.
        
        Args:
            query: The query to send to AskMod
            
        Returns:
            Dict containing the formatted payload
        """
        return {
            "name": "appmodoncw",
            "organizationName": self.organization_name,
            "prompt": f"User Query:{query}",
            "metadata": {
                "context": "{}",
                "userSelectedIntent": "",
                "userSelectedProjects": [
                    {
                        "taskId": self.task_id,
                        "databaseIndex": self.database_index,
                        "project_name": "",
                        "updated_at": "2025-07-01 16:08:26",
                        "status": "Completed",
                        "repo_url": "",
                        "embedding_model": {
                            "model_name": "text-embedding-ada-002",
                            "model_type": "Azure Openai"
                        }
                    }
                ],
                "current_user_id": self.user_id,
                "github_token": "",
                "is_cw_flow": True,
                "send_metadata_to_orchestrator": True,
                "selectedProjectDetails": {
                    "assistantName": "",
                    "project_name": "",
                    "project_id": "",
                    "task_id": self.task_id,
                    "current_user_id": "",
                    "owner_user_id": "",
                    "embedding_model": {
                        "model_name": "text-embedding-ada-002",
                        "model_type": "Azure Openai"
                    }
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
        logger.info(f"Sending query to AskMod: {query[:50]}...")
        
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
                    data = await response.json()
                    answer = data.get("result", {}).get("Answer", "No answer found in response")
                    logger.info(f"Received answer from AskMod (length: {len(answer)})")
                    logger.info(f"response: {answer}")
                    # logger.info(f"Full response data: {data}")
                    return answer
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing JSON response: {str(e)}")
                    response_text = await response.text()
                    logger.error(f"Raw response: {response_text[:200]}...")
                    raise