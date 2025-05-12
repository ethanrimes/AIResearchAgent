import pytest
import os
import openai
import logging

@pytest.fixture(autouse=True)
def require_openai_key():
    logger = logging.getLogger("ai_research_agent.test_openai_connectivity")
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "OPENAI_API_KEY must be set"
    logger.info("OpenAI API key is set")

def test_openai_connectivity():
    logger = logging.getLogger("ai_research_agent.test_openai_connectivity")
    logger.info("Testing OpenAI API connectivity")
    
    try:
        # Set up OpenAI client
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        # Test listing models
        logger.info("Listing OpenAI models")
        response = openai.models.list()
        
        assert response is not None, "OpenAI API response should not be None"
        assert len(response.data) > 0, "OpenAI API should return at least one model"
        
        # Log a few models for verification
        models = [model.id for model in response.data[:3]]
        logger.info(f"Available models (sample): {models}")
        
        logger.info("OpenAI API is accessible")
        
    except Exception as e:
        logger.error(f"Error connecting to OpenAI API: {e}")
        raise