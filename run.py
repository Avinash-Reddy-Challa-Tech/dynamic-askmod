#!/usr/bin/env python3
"""
Multi-purpose runner script for AskMod Orchestrator.

Usage:
    python run.py query "Your query here"  # Run a query through the orchestrator
    python run.py test                     # Run unit tests
"""

import sys
import asyncio
import unittest
from main import process_query

async def run_query(query: str):
    """
    Run a query through the orchestrator.
    """
    print(f"Processing query: {query}")
    print("This may take a minute or two depending on the complexity of the query...\n")
    
    result = await process_query(query)
    
    # Print the result in a formatted way
    print("\n" + "="*80)
    print("RESULT:")
    print("="*80)
    print(result["result"]["answer"])
    print("="*80)

def run_tests():
    """
    Run the unit tests.
    """
    tests = unittest.TestLoader().discover('.', pattern='test_*.py')
    unittest.TextTestRunner(verbosity=2).run(tests)

def main():
    """
    Main entry point.
    """
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} [query|test] [query text if 'query' selected]")
        sys.exit(1)
        
    command = sys.argv[1].lower()
    
    if command == "query":
        if len(sys.argv) < 3:
            query = input("Enter your query: ")
        else:
            query = " ".join(sys.argv[2:])
            
        asyncio.run(run_query(query))
        
    elif command == "test":
        run_tests()
        
    else:
        print(f"Unknown command: {command}")
        print(f"Usage: python {sys.argv[0]} [query|test] [query text if 'query' selected]")
        sys.exit(1)

if __name__ == "__main__":
    main()