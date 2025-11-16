#!/usr/bin/env python3
"""
Multi-purpose runner script for AskMod Orchestrator with source-target repository support.
"""

import sys
import os
import json
import asyncio
from main import process_query

# Define configs here instead of importing them
SOURCE_REPO_CONFIG = {
    "organization_name": os.getenv('SOURCE_ORGANIZATION_NAME', '84lumber'),
    "task_id": os.getenv('SOURCE_TASK_ID', '88bb18aa-2a7d-42bb-9a66-bf6282ae44a3'),
    "database_index": os.getenv('SOURCE_DATABASE_INDEX', 'appmod2a5bd9f2dev'),
    "repo_url": os.getenv('SOURCE_REPO_URL', 'https://github.com/Techolution/creative-workspace-backend'),
    "assistant_name": os.getenv('SOURCE_ASSISTANT_NAME', 'appmod2a5bd9f2dev'),
    "description": os.getenv('SOURCE_DESCRIPTION', 'Creative Workspace Backend')
}

TARGET_REPO_CONFIG = {
    "organization_name": os.getenv('TARGET_ORGANIZATION_NAME', 'techolution'),
    "task_id": os.getenv('TARGET_TASK_ID', '74f6cb95-a616-44d2-bb82-04731a1beefe'),
    "database_index": os.getenv('TARGET_DATABASE_INDEX', 'appmod7c0dcde3dev'),
    "repo_url": os.getenv('TARGET_REPO_URL', 'https://github.com/Avinash-Reddy-Challa-Tech/dashboard'),
    "assistant_name": os.getenv('TARGET_ASSISTANT_NAME', 'appmod7c0dcde3dev'),
    "description": os.getenv('TARGET_DESCRIPTION', 'Userstory Dashboard')
}

async def run_query(query: str, max_iterations=3, simple_mode=False):
    """
    Run a query through the orchestrator.
    
    Args:
        query: The query to process
        max_iterations: Maximum number of iterations in the orchestration process
        simple_mode: Whether to use simplified processing
    """
    print(f"Processing query: {query}")
    print("This may take a few minutes as we analyze both repositories...\n")
    
    # Add parameters if simple mode is enabled
    kwargs = {}
    if simple_mode:
        kwargs["simple_mode"] = True
        kwargs["max_iterations"] = max_iterations
    
    try:
        result = await process_query(query, **kwargs)
        
        # Print the result
        print("\n" + "="*80)
        print("IMPLEMENTATION GUIDE:")
        print("="*80)
        print(result["result"]["answer"])
        print("="*80)
        
        # Save the result to a file
        with open("implementation_guide.md", "w", encoding="utf-8") as f:
            f.write(result["result"]["answer"])
        print(f"\nThe implementation guide has been saved to 'implementation_guide.md'")
        
    except Exception as e:
        print(f"\n⚠️ ERROR: {str(e)}")
        print("\nTry using a simpler query or check your configuration.")

def main():
    """
    Main entry point with command-line argument parsing.
    """
    import argparse
    
    # Create argument parser
    parser = argparse.ArgumentParser(description='AskMod Orchestrator with Source-Target Repository Support')
    
    # Add arguments
    parser.add_argument('text', nargs='?', help='The query text')
    parser.add_argument('--max-iterations', type=int, default=3, 
                      help='Maximum number of iterations in the orchestration process')
    parser.add_argument('--simple', action='store_true',
                      help='Run in simplified mode with fewer API calls')
    parser.add_argument('--source-only', action='store_true',
                      help='Only query the source repository')
    parser.add_argument('--target-only', action='store_true',
                      help='Only query the target repository')
    
    # Parse arguments
    args = parser.parse_args()
    
    # If no text is provided, show a message
    if not args.text:
        print("\nAskMod Orchestrator with Source-Target Repository Support")
        print("--------------------------------------------------------")
        print(f"Source Repository: {SOURCE_REPO_CONFIG['description']} ({SOURCE_REPO_CONFIG['repo_url']})")
        print(f"Target Repository: {TARGET_REPO_CONFIG['description']} ({TARGET_REPO_CONFIG['repo_url']})")
        print("--------------------------------------------------------")
        print("Example query: 'How can I implement the PDF download feature from the source repository in the target repository?'")
        print("--------------------------------------------------------\n")
        args.text = input("Enter your query about porting features between repositories: ")
    
    # Run the query
    asyncio.run(run_query(args.text, args.max_iterations, args.simple))

if __name__ == "__main__":
    main()