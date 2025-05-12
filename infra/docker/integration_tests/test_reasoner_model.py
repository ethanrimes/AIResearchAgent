import pytest
import sys
import os
import logging

# Add the root directory to Python path
sys.path.append('/')
from read_only.models.ollama_client import OllamaClient, GenerationParameters

@pytest.fixture
def client():
    logger = logging.getLogger("ai_research_agent.test_reasoner_model")
    logger.info("Creating OllamaClient instance for reasoner model testing")
    return OllamaClient()

def test_generate_text(client):
    logger = logging.getLogger("ai_research_agent.test_reasoner_model")
    logger.info("Testing text generation with reasoner model")
    
    # Create generation parameters for a simple test
    test_prompt = "What is 2+2?"
    logger.info(f"Using test prompt: '{test_prompt}'")
    
    params = GenerationParameters(
        model="ignored",  # Will be replaced by client's default model
        prompt=test_prompt,
        temperature=0.0,  # Deterministic for testing
        num_predict=100,  # Limit output length for test
        stream=False
    )
    
    # Generate text
    logger.info(f"Generating text with model: {client.default_model}")
    response = client.generate(params)
    
    assert response is not None, "Response should not be None"
    assert "response" in response, "Response should contain a 'response' key"
    assert isinstance(response["response"], str), "Response text should be a string"
    assert len(response["response"]) > 0, "Response text should not be empty"
    
    # Log the response for verification
    logger.info(f"Generated response: {response['response'][:100]}...")
    logger.info(f"Performance metrics: {response['metrics']}")
    
    # Check if the answer contains "4" or "four" (for basic math check)
    answer_text = response["response"].lower()
    contains_correct_answer = "4" in answer_text or "four" in answer_text
    
    if contains_correct_answer:
        logger.info("Response contains the expected answer (4)")
    else:
        logger.warning(f"Response might not contain the expected answer: {answer_text}")
    
    logger.info("Reasoner model test completed successfully")