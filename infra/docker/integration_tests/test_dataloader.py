import pytest

# Add the root directory to Python path
sys.path.append('/')
from read_only.dataloader.rstar_data_loader import rStarDataLoader


import sys
import os
import logging



@pytest.fixture
def loader():
    logger = logging.getLogger("ai_research_agent.test_dataloader")
    logger.info("Creating rStarDataLoader instance")
    return rStarDataLoader()

def test_load_data(loader):
    logger = logging.getLogger("ai_research_agent.test_dataloader")
    logger.info("Testing data loading functionality")
    
    problems = loader.load()
    logger.info(f"Loaded {len(problems)} problems from dataset")
    
    assert isinstance(problems, list), "Loaded data should be a list"
    assert len(problems) > 0, "Dataset should contain at least one problem"
    
    # Log the first problem for verification
    if problems:
        first = problems[0]
        question_preview = first["question"][:100] + "..." if len(first["question"]) > 100 else first["question"]
        logger.info(f"First problem - ID: {first['id']}")
        logger.info(f"Question: {question_preview}")
        logger.info(f"Answer: {first['answer']}")
        
        assert "id" in first, "Problem should have an 'id' field"
        assert "question" in first, "Problem should have a 'question' field"
        assert "answer" in first, "Problem should have an 'answer' field"
        
    logger.info("Data loading test completed successfully")