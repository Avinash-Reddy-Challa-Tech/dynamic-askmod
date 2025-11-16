"""
Main module for AskMod Orchestrator with compatibility fixes
"""

import os
import asyncio
import logging
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from orchestrator import IntelligentOrchestrator
from query_decomposer import QueryDecomposer
from enhanced_response_evaluator import EnhancedResponseEvaluator
from response_synthesizer import ResponseSynthesizer
from askmod_client import AskModClient
from code_extractor import CodeExtractor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get configuration from environment variables
ASKMOD_ENDPOINT = os.getenv('ASKMOD_ENDPOINT', 'https://dev-proposals-ai.techo.camp/api/chat/chatResponse')
ASKMOD_COOKIE = os.getenv('ASKMOD_COOKIE', '')  

# Source repository configuration
SOURCE_TASK_ID = os.getenv('SOURCE_TASK_ID', '88bb18aa-2a7d-42bb-9a66-bf6282ae44a3')
SOURCE_DATABASE_INDEX = os.getenv('SOURCE_DATABASE_INDEX', 'appmod2a5bd9f2dev')
SOURCE_ORGANIZATION_NAME = os.getenv('SOURCE_ORGANIZATION_NAME', '84lumber')

# Target repository configuration
TARGET_TASK_ID = os.getenv('TARGET_TASK_ID', '74f6cb95-a616-44d2-bb82-04731a1beefe')
TARGET_DATABASE_INDEX = os.getenv('TARGET_DATABASE_INDEX', 'appmod7c0dcde3dev')
TARGET_ORGANIZATION_NAME = os.getenv('TARGET_ORGANIZATION_NAME', 'techolution')

# User configuration
USER_ID = os.getenv('USER_ID', '68e648e8658ff0e1799590c4')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')

def create_orchestrator() -> IntelligentOrchestrator:
    """Create and configure the orchestrator with all its components."""
    # Create the language model
    llm = ChatGoogleGenerativeAI(
        google_api_key=GEMINI_API_KEY,
        temperature=0.2,
        model="gemini-2.5-flash"
    )
    
    # Create the AskMod client without the problematic parameter
    askmod_client = AskModClient(
        endpoint=ASKMOD_ENDPOINT,
        cookie=ASKMOD_COOKIE,
        # organization_name parameter removed
        task_id=SOURCE_TASK_ID,
        database_index=SOURCE_DATABASE_INDEX,
        user_id=USER_ID
    )
    
    # Create the components
    decomposer = QueryDecomposer(llm=llm)
    evaluator = EnhancedResponseEvaluator(llm=llm)
    synthesizer = ResponseSynthesizer(llm=llm)
    
    # Set up source and target configurations
    source_config = {
        "organization_name": SOURCE_ORGANIZATION_NAME,
        "task_id": SOURCE_TASK_ID,
        "database_index": SOURCE_DATABASE_INDEX,
        "repo_url": "https://github.com/Techolution/creative-workspace-backend",
        "assistant_name": "appmod2a5bd9f2dev",
        "description": "Creative Workspace Backend"
    }
    
    target_config = {
        "organization_name": TARGET_ORGANIZATION_NAME,
        "task_id": TARGET_TASK_ID,
        "database_index": TARGET_DATABASE_INDEX,
        "repo_url": "https://github.com/Avinash-Reddy-Challa-Tech/dashboard",
        "assistant_name": "appmod7c0dcde3dev",
        "description": "Userstory Dashboard"
    }
    
    # Create and return the orchestrator
    return IntelligentOrchestrator(
        askmod_client=askmod_client,
        decomposer=decomposer,
        evaluator=evaluator,
        synthesizer=synthesizer,
        llm=llm,
        code_extractor=CodeExtractor(),
        source_repo_config=source_config,
        target_repo_config=target_config
    )

async def process_query(query: str) -> dict:
    """Process a user query through the orchestrator."""
    orchestrator = create_orchestrator()
    
    # Try to process the query with simplified approach
    try:
        logger.info(f"Processing query: {query}")
        result = await orchestrator.process_query(
            user_query=query,
            user_id=USER_ID
        )
        return result
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        # Return a minimal result with error info
        return {
            "result": {
                "answer": f"Error processing query: {str(e)}. Please check the logs for more details.",
                "error": str(e)
            }
        }

async def main():
    """Main entry point."""
    import sys
    
    # Get the query from command line args
    if len(sys.argv) > 1:
        query = sys.argv[1]
    else:
        query = input("Enter your query: ")
    
    # Process the query
    print(f"Processing query: {query}")
    print("This may take a few minutes as we analyze both repositories...\n")
    
    result = await process_query(query)
    
    # Print the result
    print("\n" + "="*80)
    print("IMPLEMENTATION GUIDE:")
    print("="*80)
    print(result["result"]["answer"])
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())