# infra/docker/setup_tests/test_ollama_connectivity.py
import os
import requests
import pytest

@pytest.fixture(autouse=True)
def require_ollama_url():
    assert os.getenv("OLLAMA_BASE_URL"), "OLLAMA_BASE_URL must be set"

def test_tags_endpoint():
    url = os.getenv("OLLAMA_BASE_URL").rstrip("/") + "/api/tags"
    resp = requests.get(url, timeout=5)
    assert resp.status_code == 200
    j = resp.json()
    assert "models" in j