import os
import requests
import pytest
import sys
sys.path.append('/app')
from sandbox.read_only.models.ollama_client import OllamaClient, GenerationParameters

@pytest.fixture(autouse=True)
def require_ollama_url():
    assert os.getenv("OLLAMA_BASE_URL"), "OLLAMA_BASE_URL must be set"
    assert os.getenv("OLLAMA_DEFAULT_MODEL"), "OLLAMA_DEFAULT_MODEL must be set"

def test_tags_endpoint():
    url = os.getenv("OLLAMA_BASE_URL").rstrip("/") + "/api/tags"
    resp = requests.get(url, timeout=5)
    assert resp.status_code == 200
    j = resp.json()
    assert "models" in j

def test_generate_simple_text():
    """Test that the OllamaClient can successfully generate a simple text response."""
    # Create an instance of OllamaClient
    client = OllamaClient()
    
    # Create generation parameters for a simple prompt
    params = GenerationParameters(
        model=client.default_model,  # This will be ignored and replaced by the client
        prompt="Say hello world",
        temperature=0.0,  # Set to 0 for deterministic output
        num_predict=2048,  # Limit the number of tokens to generate
        stream=False
    )
    
    # Generate text
    response = client.generate(params)
    
    # Verify the response
    assert response is not None, "Response should not be None"
    assert "response" in response, "Response should contain a 'response' key"
    assert isinstance(response["response"], str), "Response text should be a string"
    assert len(response["response"]) > 0, "Response text should not be empty"
    
    # Print the response for debugging
    print(f"Generated response: {response['response']}")
    
    # Verify metrics are present
    assert "metrics" in response, "Response should contain metrics"
    assert "inference_time" in response["metrics"], "Metrics should include inference_time"
    assert "tokens_per_second" in response["metrics"], "Metrics should include tokens_per_second"