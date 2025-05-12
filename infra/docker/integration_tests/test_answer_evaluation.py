import pytest
import sys
import os
import logging

# Add the root directory to Python path
sys.path.append('/')
from read_only.evaluator.answer_evaluation import AnswerEvaluator

@pytest.fixture
def evaluator():
    logger = logging.getLogger("ai_research_agent.test_answer_evaluation")
    logger.info("Creating AnswerEvaluator instance")
    return AnswerEvaluator()

def test_extract_answer(evaluator):
    logger = logging.getLogger("ai_research_agent.test_answer_evaluation")
    logger.info("Testing answer extraction functionality")
    
    # Test with a simple boxed answer
    test_text = "The solution is \\boxed{42}."
    logger.info(f"Test text: '{test_text}'")
    
    extracted = evaluator.extract_answer(test_text)
    
    assert extracted is not None, "Should extract an answer"
    logger.info(f"Extracted answer: '{extracted}'")
    assert extracted == "42", f"Expected '42', got '{extracted}'"
    
    # Test with non-boxed answer
    test_text_2 = "Therefore, the answer is 24."
    logger.info(f"Test text 2: '{test_text_2}'")
    
    extracted_2 = evaluator.extract_answer(test_text_2)
    
    assert extracted_2 is not None, "Should extract an answer"
    logger.info(f"Extracted answer: '{extracted_2}'")
    
    logger.info("Answer extraction test completed successfully")

def test_evaluate_correctness(evaluator):
    logger = logging.getLogger("ai_research_agent.test_answer_evaluation")
    logger.info("Testing answer evaluation functionality")
    
    # Test with a simple numerical answer
    test_text = "After calculating, I get \\boxed{42}."
    ground_truth = "42"
    logger.info(f"Test text: '{test_text}', Ground truth: '{ground_truth}'")
    
    extracted, is_correct = evaluator.evaluate(test_text, ground_truth)
    
    assert extracted is not None, "Should extract an answer"
    assert is_correct is True, f"Answer should be correct, got {is_correct}"
    logger.info(f"Extracted answer: '{extracted}', Correct: {is_correct}")
    
    # Test with a wrong answer
    test_text = "After calculating, I get \\boxed{43}."
    ground_truth = "42"
    logger.info(f"Test text: '{test_text}', Ground truth: '{ground_truth}'")
    
    extracted, is_correct = evaluator.evaluate(test_text, ground_truth)
    
    assert extracted is not None, "Should extract an answer"
    assert is_correct is False, f"Answer should be incorrect, got {is_correct}"
    logger.info(f"Extracted answer: '{extracted}', Correct: {is_correct}")
    
    # Test with a fraction answer
    test_text = "The probability is \\boxed{\\frac{1}{2}}."
    ground_truth = "\\frac{1}{2}"
    logger.info(f"Test text: '{test_text}', Ground truth: '{ground_truth}'")
    
    extracted, is_correct = evaluator.evaluate(test_text, ground_truth)
    
    assert extracted is not None, "Should extract an answer"
    logger.info(f"Extracted answer: '{extracted}', Correct: {is_correct}")
    
    logger.info("Answer evaluation test completed successfully")