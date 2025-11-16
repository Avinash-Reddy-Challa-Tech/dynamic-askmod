#!/usr/bin/env python3
"""
Test script for the fixed AskMod client
This script tests the exact functionality that was working in your cURL request
"""

import os
import asyncio
import logging
from dotenv import load_dotenv

# Import the fixed client
from askmod_client import AskModClient

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

async def test_askmod_client():
    """Test the fixed AskMod client with the exact query from your working cURL"""
    
    # Configuration from your environment
    endpoint = os.getenv('ASKMOD_ENDPOINT', 'https://dev-proposals-ai.techo.camp/api/chat/chatResponse')
    cookie = os.getenv('ASKMOD_COOKIE', '')
    
    if not cookie:
        print("❌ ASKMOD_COOKIE environment variable not set!")
        return
    
    print("🚀 Testing Fixed AskMod Client")
    print("=" * 50)
    
    # Create the client
    client = AskModClient(
        endpoint=endpoint,
        cookie=cookie
    )
    
    # Test 1: Exact query from your working cURL
    print("\n📝 Test 1: Exact query from working cURL")
    test_query = "TRIGGER DOMAIN KNOWLEDGE AGENT: explain me about email functionality"
    
    try:
        print(f"Query: {test_query}")
        print("Sending to source repository...")
        
        source_response = await client.send_query(test_query, is_target=False)
        
        print(f"✅ Source Response (length: {len(source_response)}):")
        print(f"First 200 chars: {source_response[:200]}...")
        
        if "fallback response" not in source_response.lower():
            print("🎉 SUCCESS: Received actual response from AskMod API!")
        else:
            print("⚠️  WARNING: Received fallback response (API call may have failed)")
            
    except Exception as e:
        print(f"❌ ERROR in source query: {str(e)}")
    
    # Test 2: Target repository
    print("\n📝 Test 2: Query to target repository")
    
    try:
        print("Sending to target repository...")
        
        target_response = await client.send_query(test_query, is_target=True)
        
        print(f"✅ Target Response (length: {len(target_response)}):")
        print(f"First 200 chars: {target_response[:200]}...")
        
        if "fallback response" not in target_response.lower():
            print("🎉 SUCCESS: Received actual response from AskMod API!")
        else:
            print("⚠️  WARNING: Received fallback response (API call may have failed)")
            
    except Exception as e:
        print(f"❌ ERROR in target query: {str(e)}")
    
    # Test 3: Repository switching verification
    print("\n📝 Test 3: Repository switching verification")
    
    print("Source repo config:")
    print(f"  - Task ID: {client.source_repo_config['taskId']}")
    print(f"  - Organization: {client.source_repo_config['organization_name']}")
    print(f"  - Project: {client.source_repo_config['project_name']}")
    
    print("Target repo config:")
    print(f"  - Task ID: {client.target_repo_config['taskId']}")
    print(f"  - Organization: {client.target_repo_config['organization_name']}")
    print(f"  - Project: {client.target_repo_config['project_name']}")
    
    print("\n🔧 Configuration Summary:")
    print(f"  - Endpoint: {endpoint}")
    print(f"  - Cookie: {'✅ Set' if cookie else '❌ Missing'}")
    print(f"  - User ID: {client.current_user_id}")
    print(f"  - Payload User ID: {client.user_id_payload}")

async def test_simplified_query():
    """Test with a simpler query to ensure basic functionality"""
    
    endpoint = os.getenv('ASKMOD_ENDPOINT', 'https://dev-proposals-ai.techo.camp/api/chat/chatResponse')
    cookie = os.getenv('ASKMOD_COOKIE', '')
    
    client = AskModClient(endpoint=endpoint, cookie=cookie)
    
    print("\n📝 Test 4: Simplified query")
    simple_query = "TRIGGER DOMAIN KNOWLEDGE AGENT: What is the main structure of the codebase?"
    
    try:
        response = await client.send_query(simple_query, is_target=False)
        
        print(f"✅ Response received (length: {len(response)}):")
        print(f"Content preview: {response[:300]}...")
        
        return "fallback response" not in response.lower()
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🔍 AskMod Client Fix Verification")
    print("This test uses the exact same payload structure as your working cURL")
    print("=" * 70)
    
    # Run the tests
    asyncio.run(test_askmod_client())
    
    print("\n" + "=" * 70)
    print("💡 Next Steps:")
    print("1. If you see 'SUCCESS' messages, the fix is working!")
    print("2. If you see 'WARNING' or 'ERROR' messages, check:")
    print("   - ASKMOD_COOKIE environment variable")
    print("   - Network connectivity")
    print("   - API endpoint availability")
    print("3. Enable DEBUG logging for more details:")
    print("   - Set logging.basicConfig(level=logging.DEBUG)")

if __name__ == "__main__":
    main()