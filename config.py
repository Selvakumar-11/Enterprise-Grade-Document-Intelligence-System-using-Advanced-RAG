# ============================================================
#  config.py
#  Loads environment variables and initializes Azure OpenAI LLM
# ============================================================

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

load_dotenv()

def get_llm() -> AzureChatOpenAI:
    """Initialize and return the Azure OpenAI LLM."""
    headers = {
        'x-service-line': os.getenv('SERVICE_LINE'),
        'x-brand': os.getenv('BRAND'),
        'x-project': os.getenv('PROJECT'),
        'Content-Type': 'application/json',
        'Cache-Control': 'no-cache',
        'api-version': os.getenv('HEADER_API_VERSION'),
        'Ocp-Apim-Subscription-Key': os.getenv('API_KEY'),
    }

    return AzureChatOpenAI(
        model="GPT4o128k",
        api_version=os.getenv('API_VERSION'),
        azure_endpoint=os.getenv('END_POINT'),
        api_key=os.getenv('API_KEY'),
        deployment_name=os.getenv('DEPLOYMENT_ID'),
        default_headers=headers,
    )