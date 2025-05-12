import pytest
import sys

# Add the root directory to Python path
sys.path.append('/')

import os
import requests
import logging


@pytest.fixture(autouse=True)
def require_ollama_url():
    logger = logging.getLogger("ai_research_agent.test_ollama_connectivity")
    
    ollama_url = os.getenv("OLLAMA_BASE_URL")
    
    assert ollama_url, "OLLAMA_BASE_URL must be set"
    
    logger.info(f"Using OLLAMA_BASE_URL: {ollama_url}")

def test_tags_endpoint():
    logger = logging.getLogger("ai_research_agent.test_ollama_connectivity")
    logger.info("Testing connection to Ollama tags endpoint")
    
    url = os.getenv("OLLAMA_BASE_URL").rstrip("/") + "/api/tags"
    logger.info(f"Making GET request to {url}")
    
    resp = requests.get(url, timeout=5)
    assert resp.status_code == 200, f"Expected status code 200, got {resp.status_code}"
    
    j = resp.json()
    assert "models" in j, "Expected 'models' in response JSON"
    
    # Log available models for debugging
    model_count = len(j.get("models", []))
    logger.info(f"Server has {model_count} models available")
    logger.info(f"Connection to Ollama tags endpoint successful")