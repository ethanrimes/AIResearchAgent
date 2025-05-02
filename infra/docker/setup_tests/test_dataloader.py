import pytest
from pathlib import Path
import sys
import os

# Add the app directory to Python path
sys.path.append('/app')
from sandbox.read_only.dataloader.rstar_data_loader import rStarDataLoader

@pytest.fixture
def loader():
    # Set environment variable for the dataset path
    os.environ["DATASET_PATH"] = "/app/data/rstar/amc23_test.json"
    # Create the loader instance
    return rStarDataLoader()

def test_load_amc23(loader):
    problems = loader.load()
    assert isinstance(problems, list)
    assert len(problems) == 40
    first = problems[0]
    assert first["id"] == "0"
    assert "question" in first and first["question"]
    assert first["answer"] == "27"