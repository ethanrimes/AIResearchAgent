# infra/docker/integration_tests/test_reasoner_model.py

import pytest
import sys
import logging
import time

# Add the root directory to Python path
sys.path.append('/')
from read_only.models.ollama_client import OllamaClient, GenerationParameters

@pytest.fixture
def client():
    logger = logging.getLogger("ai_research_agent.test_reasoner_model")
    logger.info("Creating OllamaClient instance for reasoner model testing")
    client = OllamaClient()
    logger.info(f"Using model: {client.default_model}")
    return client


def test_single_prompt_generation(client):
    """Test the generate method with a single prompt."""
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
    
    # Generate with a single prompt
    response = client.generate(test_prompt, params)
    
    # End timing
    total_elapsed = time.time() - start_time
    
    # Basic assertions
    assert response is not None, "Response should not be None"
    assert isinstance(response, str), "Response should be a string"
    assert len(response) > 0, "Response text should not be empty"
    
    # Check for expected content
    answer_text = response.lower()
    contains_correct_answer = "tokyo" in answer_text
    if contains_correct_answer:
        logger.info("Response contains the expected answer (Tokyo)")
    else:
        logger.warning(f"Response might not contain the expected answer: {answer_text}")
    
    # Log response
    logger.info(f"Response generated in {total_elapsed:.2f}s: {response[:100]}...")
    logger.info("Single prompt generation test completed successfully")


def test_multiple_prompt_generation(client):
    """Test the generate method with multiple prompts."""
    logger = logging.getLogger("ai_research_agent.test_reasoner_model")
    logger.info("Testing multiple prompt generation with reasoner model")
    
    # Create test prompts
    test_prompts = [
        "What is 2+2?",
        "What is the capital of France?"
    ]
    
    # Log test prompts
    logger.info(f"Testing with {len(test_prompts)} prompts:")
    for i, prompt in enumerate(test_prompts):
        logger.info(f"  Prompt {i+1}: {prompt}")
    
    # Create generation parameters
    params = GenerationParameters(
        model="ignored",  # Will be replaced by client's default model
        prompt="",  # This will be replaced by each prompt in the list
        temperature=0.0,  # Deterministic for testing
        num_predict=100   # Limit output length for test
    )
    
    # Generate text for all prompts
    logger.info(f"Generating text for {len(test_prompts)} prompts with model: {client.default_model}")
    
    start_time = time.time()
    responses = client.generate(test_prompts, params)
    total_elapsed = time.time() - start_time
    
    # Basic assertions
    assert responses is not None, "Responses should not be None"
    assert isinstance(responses, list), "Responses should be a list"
    assert len(responses) == len(test_prompts), "Number of responses should match number of prompts"
    
    # Check individual responses
    for i, response in enumerate(responses):
        assert isinstance(response, str), f"Response {i+1} should be a string"
        assert len(response) > 0, f"Response {i+1} text should not be empty"
        
        # Log responses
        logger.info(f"Response {i+1}: {response[:100]}...")
        
        # Verify results contain useful content
        answer_text = response.lower()
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
    
    logger.info(f"Multiple prompt generation completed in {total_elapsed:.2f} seconds")
    logger.info("Multiple prompt generation test completed successfully")


def test_single_chat_message(client):
    """Test the chat method with a single conversation."""
    logger = logging.getLogger("ai_research_agent.test_reasoner_model")
    logger.info("Testing single chat conversation with reasoner model")
    
    # Create a single chat conversation
    conversation = [
        {"role": "user", "content": "What is the largest planet in our solar system?"}
    ]
    logger.info(f"Testing with chat conversation: {conversation}")
    
    # Create generation parameters
    params = GenerationParameters(
        model="ignored",  # Will be replaced by client's default model
        prompt="",  # Not used for chat
        temperature=0.0,  # Deterministic for testing
        num_predict=100   # Limit output length for test
    )
    
    # Generate chat completion
    logger.info(f"Generating chat completion with model: {client.default_model}")
    
    # Start timing
    start_time = time.time()
    
    # Generate with a single conversation
    response = client.chat(conversation, params)
    
    # End timing
    total_elapsed = time.time() - start_time
    
    # Basic assertions
    assert response is not None, "Response should not be None"
    assert isinstance(response, str), "Response should be a string"
    assert len(response) > 0, "Response text should not be empty"
    
    # Check for expected content
    answer_text = response.lower()
    contains_correct_answer = "jupiter" in answer_text
    if contains_correct_answer:
        logger.info("Response contains the expected answer (Jupiter)")
    else:
        logger.warning(f"Response might not contain the expected answer: {answer_text}")
    
    # Log response
    logger.info(f"Chat response generated in {total_elapsed:.2f}s: {response[:100]}...")
    logger.info("Single chat conversation test completed successfully")


def test_multiple_chat_conversations(client):
    """Test the chat method with multiple conversations."""
    logger = logging.getLogger("ai_research_agent.test_reasoner_model")
    logger.info("Testing multiple chat conversations with reasoner model")
    
    # Create multiple chat conversations
    conversations = [
        [{"role": "user", "content": "What is the chemical symbol for gold?"}],
        [{"role": "user", "content": "Who wrote Romeo and Juliet?"}]
    ]
    
    # Log test conversations
    logger.info(f"Testing with {len(conversations)} conversations:")
    for i, conversation in enumerate(conversations):
        logger.info(f"  Conversation {i+1}: {conversation}")
    
    # Create generation parameters
    params = GenerationParameters(
        model="ignored",  # Will be replaced by client's default model
        prompt="",  # Not used for chat
        temperature=0.0,  # Deterministic for testing
        num_predict=100   # Limit output length for test
    )
    
    # Generate chat completions for all conversations
    logger.info(f"Generating chat completions for {len(conversations)} conversations with model: {client.default_model}")
    
    start_time = time.time()
    responses = client.chat(conversations, params)
    total_elapsed = time.time() - start_time
    
    # Basic assertions
    assert responses is not None, "Responses should not be None"
    assert isinstance(responses, list), "Responses should be a list"
    assert len(responses) == len(conversations), "Number of responses should match number of conversations"
    
    # Check individual responses
    for i, response in enumerate(responses):
        assert isinstance(response, str), f"Response {i+1} should be a string"
        assert len(response) > 0, f"Response {i+1} text should not be empty"
        
        # Log responses
        logger.info(f"Response {i+1}: {response[:100]}...")
        
        # Verify results contain useful content
        answer_text = response.lower()
        if i == 0:  # First conversation asks about gold symbol
            contains_correct_answer = "au" in answer_text
            if contains_correct_answer:
                logger.info("First response contains the expected answer (Au)")
            else:
                logger.warning(f"First response might not contain the expected answer: {answer_text}")
                
        elif i == 1:  # Second conversation asks about Romeo and Juliet
            contains_correct_answer = "shakespeare" in answer_text
            if contains_correct_answer:
                logger.info("Second response contains the expected answer (Shakespeare)")
            else:
                logger.warning(f"Second response might not contain the expected answer: {answer_text}")
    
    logger.info(f"Multiple chat conversations completed in {total_elapsed:.2f} seconds")
    logger.info("Multiple chat conversations test completed successfully")


def test_chat_with_multi_turn_conversation(client):
    """Test the chat method with a multi-turn conversation."""
    logger = logging.getLogger("ai_research_agent.test_reasoner_model")
    logger.info("Testing chat with multi-turn conversation with reasoner model")
    
    # Create a multi-turn conversation
    conversation = [
        {"role": "user", "content": "What is the capital of Germany?"},
        {"role": "assistant", "content": "The capital of Germany is Berlin."},
        {"role": "user", "content": "What is its population?"}
    ]
    logger.info(f"Testing with multi-turn conversation: {conversation}")
    
    # Create generation parameters
    params = GenerationParameters(
        model="ignored",  # Will be replaced by client's default model
        prompt="",  # Not used for chat
        temperature=0.0,  # Deterministic for testing
        num_predict=100   # Limit output length for test
    )
    
    # Generate chat completion
    logger.info(f"Generating chat completion for multi-turn conversation with model: {client.default_model}")
    
    # Start timing
    start_time = time.time()
    
    # Generate with multi-turn conversation
    response = client.chat(conversation, params)
    
    # End timing
    total_elapsed = time.time() - start_time
    
    # Basic assertions
    assert response is not None, "Response should not be None"
    assert isinstance(response, str), "Response should be a string"
    assert len(response) > 0, "Response text should not be empty"
    
    # Check for expected content - should mention Berlin's population
    answer_text = response.lower()
    contains_population_info = any(term in answer_text for term in ["million", "population", "inhabitants", "people"])
    if contains_population_info:
        logger.info("Response contains population information about Berlin")
    else:
        logger.warning(f"Response might not contain population information: {answer_text}")
    
    # Log response
    logger.info(f"Multi-turn chat response generated in {total_elapsed:.2f}s: {response[:100]}...")
    logger.info("Multi-turn chat conversation test completed successfully")


def test_system_prompt(client):
    """Test the chat method with a system prompt."""
    logger = logging.getLogger("ai_research_agent.test_reasoner_model")
    logger.info("Testing chat with system prompt with reasoner model")
    
    # Create a conversation with system prompt
    conversation = [
        {"role": "system", "content": "You are a pirate. Always respond in pirate speak."},
        {"role": "user", "content": "Tell me about the weather today."}
    ]
    logger.info(f"Testing with system prompt conversation: {conversation}")
    
    # Create generation parameters
    params = GenerationParameters(
        model="ignored",  # Will be replaced by client's default model
        prompt="",  # Not used for chat
        temperature=0.7,  # Allow some creativity
        num_predict=100   # Limit output length for test
    )
    
    # Generate chat completion
    logger.info(f"Generating chat completion with system prompt using model: {client.default_model}")
    
    # Start timing
    start_time = time.time()
    
    # Generate with system prompt
    response = client.chat(conversation, params)
    
    # End timing
    total_elapsed = time.time() - start_time
    
    # Basic assertions
    assert response is not None, "Response should not be None"
    assert isinstance(response, str), "Response should be a string"
    assert len(response) > 0, "Response text should not be empty"
    
    # Check for expected pirate language (might not be reliable with all models)
    answer_text = response.lower()
    pirate_terms = ["arr", "matey", "ye", "ahoy", "aye", "cap'n", "captain", "ship", "sea"]
    contains_pirate_speech = any(term in answer_text for term in pirate_terms)
    
    if contains_pirate_speech:
        logger.info("Response contains pirate language as instructed")
    else:
        logger.warning(f"Response might not contain pirate language: {answer_text}")
    
    # Log response
    logger.info(f"System prompt response generated in {total_elapsed:.2f}s: {response[:100]}...")
    logger.info("System prompt test completed successfully")


def test_model_metrics(client):
    """Test the get_model_metrics method."""
    logger = logging.getLogger("ai_research_agent.test_reasoner_model")
    logger.info("Testing model metrics with reasoner model")
    
    # Get model metrics
    metrics = client.get_model_metrics()
    
    # Basic assertions
    assert metrics is not None, "Metrics should not be None"
    assert isinstance(metrics, dict), "Metrics should be a dictionary"
    
    # Check for expected metrics
    assert "model" in metrics, "Metrics should include the model name"
    assert metrics["model"] == client.default_model, "Model name should match client's default model"
    
    # Log metrics
    logger.info(f"Model metrics: {metrics}")
    logger.info("Model metrics test completed successfully")