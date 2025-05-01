# infra/docker/setup_tests/test_dataloader.py
import pytest
from core.rstar_data_loader import rStarDataLoader

@pytest.fixture
def loader():
    # Data is now located in /app/data/rstar directory
    path = "/app/data/rstar/amc23_test.json"
    return rStarDataLoader(path)

def test_load_amc23(loader):
    problems = loader.load()
    assert isinstance(problems, list)
    assert len(problems) == 40
    first = problems[0]
    assert first["id"] == "0"
    assert "question" in first and first["question"]
    assert first["answer"] == "27"