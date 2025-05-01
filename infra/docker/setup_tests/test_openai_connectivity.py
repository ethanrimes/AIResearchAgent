import os
import openai
import pytest
from openai import OpenAI

@pytest.fixture(autouse=True)
def require_api_key():
    assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY must be set"

def test_list_models():
    # Create a client instance (new API style)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Use the client to list models (new API style)
    models = client.models.list()
    
    # Check models was returned successfully
    assert hasattr(models, "data") and isinstance(models.data, list)