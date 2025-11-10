"""
AskMod Orchestrator

This is the main entry point for the AskMod Orchestrator application.
"""

import os
import asyncio
import logging
from dotenv import load_dotenv
from typing import Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from orchestrator import Orchestrator
from query_decomposer import QueryDecomposer
from response_evaluator import ResponseEvaluator
from response_synthesizer import ResponseSynthesizer
from askmod_client import AskModClient

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get configuration from environment variables
ASKMOD_ENDPOINT = os.getenv('ASKMOD_ENDPOINT', 'https://dev-proposals-ai.techo.camp/api/chat/chatResponse')
ASKMOD_COOKIE = os.getenv('ASKMOD_COOKIE', '')  # This should be set in the .env file
ORGANIZATION_NAME = os.getenv('ORGANIZATION_NAME', 'techolution')
TASK_ID = os.getenv('TASK_ID', '7c377dee-a767-43bc-ac52-63b16187391c')
DATABASE_INDEX = os.getenv('DATABASE_INDEX', 'appmoda9c40dev')
USER_ID = os.getenv('USER_ID', '66d977791e9c242063fd3a1e')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')  # This should be set in the .env file

def create_orchestrator() -> Orchestrator:
    """
    Create and configure the orchestrator with all its components.
    
    Returns:
        Configured Orchestrator instance
    """
    # Create the language model
    llm = ChatGoogleGenerativeAI(
        google_api_key=GEMINI_API_KEY,
        temperature=0.2,
        model="gemini-2.5-flash"
    )
    
    # Create the AskMod client
    askmod_client = AskModClient(
        endpoint=ASKMOD_ENDPOINT,
        cookie=ASKMOD_COOKIE,
        organization_name=ORGANIZATION_NAME,
        task_id=TASK_ID,
        database_index=DATABASE_INDEX,
        user_id=USER_ID
    )
    
    # Create the components
    decomposer = QueryDecomposer(llm=llm)
    evaluator = ResponseEvaluator(llm=llm)
    synthesizer = ResponseSynthesizer(llm=llm)
    
    # Create and return the orchestrator
    return Orchestrator(
        askmod_client=askmod_client,
        decomposer=decomposer,
        evaluator=evaluator,
        synthesizer=synthesizer,
        llm=llm
    )

async def process_query(query: str) -> Dict[str, Any]:
    """
    Process a user query through the orchestration system.
    
    Args:
        query: The user's query
        
    Returns:
        Dict containing the final answer
    """
    orchestrator = create_orchestrator()
    return await orchestrator.process_query(query)

async def main():
    """
    Main entry point for command-line usage.
    """
    # Check if the required environment variables are set
    if not ASKMOD_COOKIE:
        logger.error("ASKMOD_COOKIE environment variable is not set!")
        print("Please set the ASKMOD_COOKIE environment variable in a .env file")
        return
        
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY environment variable is not set!")
        print("Please set the GEMINI_API_KEY environment variable in a .env file")
        return
    
    # Get the query from command-line argument or input
    import sys
    if len(sys.argv) > 1:
        query = sys.argv[1]
    else:
        query = input("Enter your query: ")
    
    # Process the query
    print(f"Processing query: {query}")
    result = await process_query(query)
    
    # Print the result
    print("\nResult:")
    print(result["result"]["answer"])

if __name__ == "__main__":
    asyncio.run(main())