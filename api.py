"""
AskMod Orchestrator API with Gemini

This module provides a FastAPI server implementation for the AskMod Orchestrator using Gemini.
"""

import os
import logging
from dotenv import load_dotenv
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Depends, Header, Request
from pydantic import BaseModel, Field
from main import create_orchestrator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Create the FastAPI app
app = FastAPI(
    title="AskMod Orchestrator API with Gemini",
    description="An intelligent orchestration layer for the AskMod RAG system using Google's Gemini LLM",
    version="1.0.0"
)

# Define the request model
class QueryRequest(BaseModel):
    query: str = Field(..., description="The user's query about the codebase")
    cookie: Optional[str] = Field(None, description="Optional authentication cookie for AskMod")
    task_id: Optional[str] = Field(None, description="Optional task ID for AskMod")
    database_index: Optional[str] = Field(None, description="Optional database index for AskMod")
    organization_name: Optional[str] = Field(None, description="Optional organization name for AskMod")
    user_id: Optional[str] = Field(None, description="Optional user ID for AskMod")
    credentials_path: Optional[str] = Field(None, description="Optional path to Google Cloud credentials file")
    model_id: Optional[str] = Field(None, description="Optional Gemini model ID to use")

# Define the response model
class QueryResponse(BaseModel):
    answer: str = Field(..., description="The synthesized answer to the user's query")

# Function to extract and validate cookies
async def get_askmod_cookie(cookie: Optional[str] = Header(None)):
    """
    Extract and validate the AskMod cookie from the request headers.
    
    Args:
        cookie: The Cookie header from the request
        
    Returns:
        The extracted cookie or the default from environment variables
    """
    if cookie:
        return cookie
        
    # Fall back to the environment variable
    env_cookie = os.getenv('ASKMOD_COOKIE', '')
    if not env_cookie:
        logger.warning("No AskMod cookie provided in request or environment")
    
    return env_cookie

@app.post("/api/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    askmod_cookie: str = Depends(get_askmod_cookie)
):
    """
    Process a user query through the orchestration system.
    
    Args:
        request: The query request
        askmod_cookie: The AskMod authentication cookie
        
    Returns:
        The synthesized answer
    """
    logger.info(f"Received query request: {request.query}")
    
    try:
        # Override environment variables with request parameters if provided
        if request.cookie:
            os.environ['ASKMOD_COOKIE'] = request.cookie
        elif askmod_cookie:
            os.environ['ASKMOD_COOKIE'] = askmod_cookie
            
        if request.task_id:
            os.environ['TASK_ID'] = request.task_id
            
        if request.database_index:
            os.environ['DATABASE_INDEX'] = request.database_index
            
        if request.organization_name:
            os.environ['ORGANIZATION_NAME'] = request.organization_name
            
        if request.user_id:
            os.environ['USER_ID'] = request.user_id
            
        if request.credentials_path:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = request.credentials_path
            
        if request.model_id:
            os.environ['MODEL_ID'] = request.model_id
        
        # Verify that we have the necessary credentials
        if not os.path.exists(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '')):
            return {
                "answer": "Error: Google Cloud credentials file not found. "
                          "Please provide a valid credentials_path parameter or set the "
                          "GOOGLE_APPLICATION_CREDENTIALS environment variable."
            }
        
        # Create the orchestrator and process the query
        orchestrator = create_orchestrator()
        result = await orchestrator.process_query(request.query)
        
        # Return the response
        return {"answer": result["result"]["answer"]}
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Dict containing the status and version information
    """
    return {
        "status": "ok",
        "gemini_model": os.environ.get('MODEL_ID', 'gemini-2.5-flash'),
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    """
    Root endpoint with API information.
    
    Returns:
        Dict containing API information
    """
    return {
        "name": "AskMod Orchestrator API with Gemini",
        "version": "1.0.0",
        "description": "An intelligent orchestration layer for the AskMod RAG system using Google's Gemini LLM",
        "gemini_model": os.environ.get('MODEL_ID', 'gemini-2.5-flash'),
        "endpoints": [
            {"path": "/api/query", "method": "POST", "description": "Process a query"},
            {"path": "/health", "method": "GET", "description": "Health check"}
        ]
    }

# Run the FastAPI app with uvicorn when executed directly
if __name__ == "__main__":
    import uvicorn
    
    # Check if the required environment variables are set
    if not os.path.exists(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '')):
        logger.error("GOOGLE_APPLICATION_CREDENTIALS environment variable is not set or file does not exist!")
        print("Please set the GOOGLE_APPLICATION_CREDENTIALS environment variable to the path of your service account key file")
    
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("api:app", host=host, port=port, reload=True)