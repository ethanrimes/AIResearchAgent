import pytest
import sys
import asyncio

# Check if pytest-asyncio is available
try:
    import pytest_asyncio
    has_pytest_asyncio = True
except ImportError:
    has_pytest_asyncio = False

# Add the root directory to Python path
sys.path.append('/')
from read_only.models.ollama_client import OllamaClient, GenerationParameters

import os
import logging
import time

@pytest.fixture
def client():
    logger = logging.getLogger("ai_research_agent.test_reasoner_model")
    logger.info("Creating OllamaClient instance for reasoner model testing")
    client = OllamaClient()
    logger.info(f"Created client with batch_size={client.batch_size}, max_num_predict={client.max_num_predict}")
    return client


def test_single_prompt_generation(client):
    """Test the generate method with a single prompt (non-batched)."""
    logger = logging.getLogger("ai_research_agent.test_reasoner_model")
    logger.info("Testing single prompt generation with reasoner model")
    
    # Create a single test prompt
    test_prompt = "What is the capital of Japan?"
    logger.info(f"Testing with prompt: {test_prompt}")
    
    # Create generation parameters
    params = GenerationParameters(
        model="ignored",  # Will be replaced by client's default model
        prompt=test_prompt,
        temperature=0.0,  # Deterministic for testing
        num_predict=100   # Limit output length for test
    )
    
    # Generate text for the prompt
    logger.info(f"Generating text for prompt with model: {client.default_model}")
    
    # Start timing
    start_time = time.time()
    
    # Generate with non-batched call
    response = client.generate(params)
    
    # End timing
    total_elapsed = time.time() - start_time
    
    # Basic assertions
    assert response is not None, "Response should not be None"
    assert isinstance(response, dict), "Response should be a dictionary"
    assert "response" in response, "Response should contain a 'response' key"
    assert isinstance(response["response"], str), "Response text should be a string"
    assert len(response["response"]) > 0, "Response text should not be empty"
    
    # Metrics assertions
    assert "metrics" in response, "Response should contain a 'metrics' key"
    assert "inference_time" in response["metrics"], "Metrics should include inference_time"
    assert "tokens_per_second" in response["metrics"], "Metrics should include tokens_per_second"
    assert "total_tokens" in response["metrics"], "Metrics should include total_tokens"
    
    # Check for expected content
    answer_text = response["response"].lower()
    contains_correct_answer = "tokyo" in answer_text
    if contains_correct_answer:
        logger.info("Response contains the expected answer (Tokyo)")
    else:
        logger.warning(f"Response might not contain the expected answer: {answer_text}")
    
    # Log response
    logger.info(f"Response ({response['metrics']['inference_time']:.2f}s): {response['response'][:100]}...")
    
    # Log statistics
    tokens = response["metrics"]["total_tokens"]
    logger.info(f"Generation statistics:")
    logger.info(f"  Total time: {total_elapsed:.2f} seconds")
    logger.info(f"  Total tokens: {tokens}")
    logger.info(f"  Tokens per second: {tokens / total_elapsed:.2f}")
    
    # Verify inference time is reasonable
    assert response["metrics"]["inference_time"] > 0, "Inference time should be positive"
    assert response["metrics"]["inference_time"] < 30, "Inference time should be less than 30 seconds"
    
    logger.info("Single prompt generation test completed successfully")


def test_sync_batched_generation(client):
    """Test synchronous batch processing of multiple prompts."""
    logger = logging.getLogger("ai_research_agent.test_reasoner_model")
    logger.info("Testing synchronous batched text generation with reasoner model")
    
    # Create test prompts
    test_prompts = [
        "What is 2+2?",
        "What is the capital of France?"
    ]
    
    # Log test prompts
    logger.info(f"Testing with {len(test_prompts)} prompts:")
    for i, prompt in enumerate(test_prompts):
        logger.info(f"  Prompt {i+1}: {prompt}")
    
    # Create generation parameters for each prompt
    params_list = [
        GenerationParameters(
            model="ignored",  # Will be replaced by client's default model
            prompt=prompt,
            temperature=0.0,  # Deterministic for testing
            num_predict=100   # Limit output length for test
        )
        for prompt in test_prompts
    ]
    
    # Generate text for all prompts in batch
    logger.info(f"Generating text for {len(params_list)} prompts with model: {client.default_model}")
    logger.info(f"Using client batch size: {client.batch_size}")
    
    start_time = time.time()
    responses = client.generate(params_list)
    total_elapsed = time.time() - start_time
    
    # Basic assertions
    assert responses is not None, "Responses should not be None"
    assert isinstance(responses, list), "Responses should be a list"
    assert len(responses) == len(params_list), "Number of responses should match number of prompts"
    
    # Check individual responses
    total_tokens = 0
    for i, response in enumerate(responses):
        assert "response" in response, f"Response {i} should contain a 'response' key"
        assert isinstance(response["response"], str), f"Response {i} text should be a string"
        assert len(response["response"]) > 0, f"Response {i} text should not be empty"
        
        # Metrics assertions
        assert "metrics" in response, f"Response {i} should contain a 'metrics' key"
        assert "inference_time" in response["metrics"], f"Metrics for response {i} should include inference_time"
        assert "tokens_per_second" in response["metrics"], f"Metrics for response {i} should include tokens_per_second"
        assert "total_tokens" in response["metrics"], f"Metrics for response {i} should include total_tokens"
        
        # Log responses
        logger.info(f"Response {i+1} ({response['metrics']['inference_time']:.2f}s): {response['response'][:100]}...")
        total_tokens += response["metrics"]["total_tokens"]
    
    # Log aggregate statistics
    avg_time_per_prompt = total_elapsed / len(params_list)
    logger.info(f"Sync batch processing statistics:")
    logger.info(f"  Total time: {total_elapsed:.2f} seconds")
    logger.info(f"  Total tokens: {total_tokens}")
    logger.info(f"  Avg time per prompt: {avg_time_per_prompt:.2f} seconds")
    logger.info(f"  Overall tokens per second: {total_tokens / total_elapsed:.2f}")
    
    # Verify results contain useful content
    for i, response in enumerate(responses):
        answer_text = response["response"].lower()
        if i == 0:  # First prompt is "What is 2+2?"
            contains_correct_answer = "4" in answer_text or "four" in answer_text
            if contains_correct_answer:
                logger.info("First response contains the expected answer (4)")
            else:
                logger.warning(f"First response might not contain the expected answer: {answer_text}")
                
        elif i == 1:  # Second prompt is "What is the capital of France?"
            contains_correct_answer = "paris" in answer_text
            if contains_correct_answer:
                logger.info("Second response contains the expected answer (Paris)")
            else:
                logger.warning(f"Second response might not contain the expected answer: {answer_text}")
    
    logger.info("Synchronous batched prompt test completed successfully")


# This test only runs if pytest-asyncio is available
@pytest.mark.skipif(not has_pytest_asyncio, reason="pytest-asyncio is not installed")
@pytest.mark.asyncio
async def test_async_batched_generation(client):
    """Test asynchronous batch processing of multiple prompts."""
    logger = logging.getLogger("ai_research_agent.test_reasoner_model")
    logger.info("Testing asynchronous batched text generation with reasoner model")
    
    # Create test prompts
    test_prompts = [
        "What is the square root of 9?",
        "What is the capital of Italy?"
    ]
    
    # Log test prompts
    logger.info(f"Testing with {len(test_prompts)} prompts:")
    for i, prompt in enumerate(test_prompts):
        logger.info(f"  Prompt {i+1}: {prompt}")
    
    # Create generation parameters for each prompt
    params_list = [
        GenerationParameters(
            model="ignored",  # Will be replaced by client's default model
            prompt=prompt,
            temperature=0.0,  # Deterministic for testing
            num_predict=100   # Limit output length for test
        )
        for prompt in test_prompts
    ]
    
    # Generate text for all prompts asynchronously
    logger.info(f"Generating text asynchronously for {len(params_list)} prompts with model: {client.default_model}")
    logger.info(f"Using client batch size: {client.batch_size}")
    
    start_time = time.time()
    responses = await client.generate_async(params_list)
    total_elapsed = time.time() - start_time
    
    # Basic assertions
    assert responses is not None, "Responses should not be None"
    assert isinstance(responses, list), "Responses should be a list"
    assert len(responses) == len(params_list), "Number of responses should match number of prompts"
    
    # Check individual responses
    total_tokens = 0
    for i, response in enumerate(responses):
        assert "response" in response, f"Response {i} should contain a 'response' key"
        assert isinstance(response["response"], str), f"Response {i} text should be a string"
        assert len(response["response"]) > 0, f"Response {i} text should not be empty"
        
        # Metrics assertions
        assert "metrics" in response, f"Response {i} should contain a 'metrics' key"
        assert "inference_time" in response["metrics"], f"Metrics for response {i} should include inference_time"
        assert "tokens_per_second" in response["metrics"], f"Metrics for response {i} should include tokens_per_second"
        assert "total_tokens" in response["metrics"], f"Metrics for response {i} should include total_tokens"
        
        # Check specific answers (basic validation)
        if i == 0:  # First prompt about square root of 9
            answer_text = response["response"].lower()
            contains_correct_answer = "3" in answer_text or "three" in answer_text
            if contains_correct_answer:
                logger.info("First response contains the expected answer (3)")
            else:
                logger.warning(f"First response might not contain the expected answer: {answer_text}")
        
        elif i == 1:  # Second prompt about capital of Italy
            answer_text = response["response"].lower()
            contains_correct_answer = "rome" in answer_text
            if contains_correct_answer:
                logger.info("Second response contains the expected answer (Rome)")
            else:
                logger.warning(f"Second response might not contain the expected answer: {answer_text}")
        
        # Log responses
        logger.info(f"Response {i+1} ({response['metrics']['inference_time']:.2f}s): {response['response'][:100]}...")
        total_tokens += response["metrics"]["total_tokens"]
    
    # Log aggregate statistics
    avg_time_per_prompt = total_elapsed / len(params_list)
    logger.info(f"Async batch processing statistics:")
    logger.info(f"  Total time: {total_elapsed:.2f} seconds")
    logger.info(f"  Total tokens: {total_tokens}")
    logger.info(f"  Avg time per prompt: {avg_time_per_prompt:.2f} seconds")
    logger.info(f"  Overall tokens per second: {total_tokens / total_elapsed:.2f}")
    
    # Verify that batch processing is faster than sequential would be
    total_inference_time = sum(r["metrics"]["inference_time"] for r in responses)
    logger.info(f"  Sum of individual inference times: {total_inference_time:.2f} seconds")
    
    # Calculate speedup ratio
    speedup = total_inference_time / total_elapsed
    logger.info(f"  Speedup from async batching: {speedup:.2f}x")
    
    # Make a meaningful assertion about concurrent processing
    assert speedup > 1.0, "Async batch processing should be faster than sequential processing"
    
    logger.info("Asynchronous batched prompt test completed successfully")