# File: infra/docker/integration_tests/run_integration_tests.py
import logging
import pytest
from pathlib import Path

def run_additional_tests(agent):
    """
    Run additional tests after agent initialization.
    
    Args:
        agent: The initialized Agent instance
    """
    logger = logging.getLogger("ai_research_agent.run_experiment")
    logger.info("Running additional component tests after agent initialization...")
    
    # Set up test directory path
    test_dir = Path("/app/infra/docker/integration_tests")
    
    # Run the dataloader test
    logger.info("Running dataloader test")
    try:
        test_result = pytest.main(["-v", "-p", "no:cacheprovider", str(test_dir / "test_dataloader.py")])
        if test_result != 0:
            logger.warning("Dataloader test failed with exit code: %d", test_result)
        else:
            logger.info("Dataloader test passed successfully")
    except Exception as e:
        logger.error("Error running dataloader test: %s", e)
    
    # Run the reasoner model test
    logger.info("Running reasoner model test")
    try:
        test_result = pytest.main(["-v", "-p", "no:cacheprovider", str(test_dir / "test_reasoner_model.py")])
        if test_result != 0:
            logger.warning("Reasoner model test failed with exit code: %d", test_result)
        else:
            logger.info("Reasoner model test passed successfully")
    except Exception as e:
        logger.error("Error running reasoner model test: %s", e)
    
    # Run the answer evaluation test
    logger.info("Running answer evaluation test")
    try:
        test_result = pytest.main(["-v", "-p", "no:cacheprovider", str(test_dir / "test_answer_evaluation.py")])
        if test_result != 0:
            logger.warning("Answer evaluation test failed with exit code: %d", test_result)
        else:
            logger.info("Answer evaluation test passed successfully")
    except Exception as e:
        logger.error("Error running answer evaluation test: %s", e)
    
    logger.info("All additional component tests completed")