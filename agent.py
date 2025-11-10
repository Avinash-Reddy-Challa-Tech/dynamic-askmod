import os
import concurrent.futures
import logging
from google import genai
from google.oauth2.service_account import Credentials # <-- Needed for JSON creds
from typing import List, Dict, Any, Union

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
MODEL_ID = "gemini-2.5-flash" 
MAX_WORKERS = 5 

# 🛑 ACTION: Set the path to your credentials file (e.g., in the environment)
CREDENTIALS_FILE_PATH = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "creds.json")
PROJECT_ID = os.environ.get("PROJECT_ID", "proposal-auto-ai-internal")
LOCATION = os.environ.get("GCP_LOCATION", 'us-central1')
SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


def _call_gemini_api(contents: str) -> str:
    """
    Internal function to handle the authenticated Gemini API call using URL Context with Vertex AI.
    Loads credentials explicitly for thread-safety and service environment use.
    """
    if PROJECT_ID == "YOUR_GCP_PROJECT_ID" or not os.path.exists(CREDENTIALS_FILE_PATH):
         return (
            "ERROR: FATAL_CONFIG: PROJECT_ID not set or creds.json not found. "
            f"Expected project: {PROJECT_ID}, creds path: {CREDENTIALS_FILE_PATH}"
        )
    print(f"conntext: {contents}")
    try:
        # 1. Load Credentials explicitly from the JSON file
        creds = Credentials.from_service_account_file(
            CREDENTIALS_FILE_PATH, 
            scopes=SCOPES
        )
        print(f"cerds: {creds}")
        
        # 2. Initialize the Vertex AI Client with explicit credentials
        client = genai.Client(
            vertexai=True, 
            project=PROJECT_ID, 
            location=LOCATION,
            credentials=creds
        )
        print(f"client: {client}")

        System_Prompt = """
        You are an intelligent assistant that must ensure you fully understand the user's intent before answering.

        Your goal is to provide the most accurate and relevant answer possible — but ONLY after confirming that you have enough information to do so.

        Follow these rules carefully:

        1. **Analyze the user query** for ambiguity, missing context, or multiple possible interpretations.
        2. If anything is unclear or underspecified (e.g., missing goals, inputs, formats, frameworks, constraints, or use-cases), ask up to **3 targeted clarification questions** that would help you give a more precise and useful response.
        3. If the query is already clear and unambiguous, skip clarifications and answer directly.
        4. Ask all clarification questions **together** in one message — do not answer the query until the user responds.
        5. Keep clarification questions concise, context-aware, and relevant to what’s missing.
        6. After getting clarifications, provide a complete and high-quality final answer.
        7. Give the response in a python list format, like list of question

        Example:
        User: "I want to build a chatbot."
        Assistant: 
        ["What platform or language will you use (e.g., Python, Node.js)?",
         "Should the chatbot use predefined rules or an LLM backend like Gemini or GPT?",
         "Do you need a web or messaging interface?"]

        Once you receive the clarifications, proceed with your final detailed answer.
        """

        contents = System_Prompt + contents 
        # Make the API call
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=contents
        )

        # Return text output
        return response.text
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client for Vertex AI. Error: {e}")
        return f"ERROR: Client initialization failed: {e}"
    
if __name__ == "__main__":
    query = "I want to create a rag tool with langchain, give me the pipline to do it!"
    query = "Explain me how does update bin works and what is the exact function used"
    response = _call_gemini_api(query)
    response = response[1:-2]
    print(response)