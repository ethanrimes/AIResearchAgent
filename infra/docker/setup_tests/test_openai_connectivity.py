# infra/docker/setup_tests/test_openai_connectivity.py
import os
import openai
import pytest

@pytest.fixture(autouse=True)
def require_api_key():
    assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY must be set"
    openai.api_key = os.getenv("OPENAI_API_KEY")

def test_list_models():
    # will raise if the key is invalid or network fails
    models = openai.Model.list()
    assert hasattr(models, "data") and isinstance(models.data, list)