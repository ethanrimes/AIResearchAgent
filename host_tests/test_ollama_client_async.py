#!/usr/bin/env python3
import logging
import json
import time
import sys
import ollama
from ollama import Client
from typing import List, Dict, Any, Union, Optional

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ollama_client_test.log')
    ]
)
logger = logging.getLogger("ollama_test")

class StandaloneOllamaClient:
    """
    Simplified Ollama client using the official Ollama Python client.
    
    Features:
    - Hard-coded model name (qwen3:0.6b)
    - Hard-coded batch size (4)
    - Hard-coded max token length (2048)
    - Simplified generate and chat functions
    """
    
    def __init__(self):
        # Hard-coded values
        self._default_model = "qwen3:0.6b"
        self._batch_size = 4
        self._max_num_predict = 2048
        
        # Set up logging
        self.logger = logging.getLogger("standalone_ollama_client")
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.propagate = False
        
        # Initialize the Ollama client
        self.client = Client()
        self.logger.info(f"Initialized StandaloneOllamaClient with model: {self._default_model}")
        self.logger.info(f"Batch size: {self._batch_size}, Max tokens: {self._max_num_predict}")
    
    def _process_batch(self, items, process_func):
        """
        Helper function to process items in batches.
        
        Args:
            items: List of items to process
            process_func: Function to process each item
            
        Returns:
            List of results
        """
        results = []
        total_items = len(items)
        
        for i in range(0, total_items, self._batch_size):
            batch = items[i:i + self._batch_size]
            batch_num = i // self._batch_size + 1
            total_batches = (total_items + self._batch_size - 1) // self._batch_size
            
            self.logger.info(f"Processing batch {batch_num}/{total_batches} with {len(batch)} items")
            
            for j, item in enumerate(batch):
                item_num = i + j + 1
                start_time = time.time()
                
                try:
                    # Process the item
                    result = process_func(item)
                    elapsed = time.time() - start_time
                    
                    self.logger.info(
                        f"Item {item_num}/{total_items} completed in {elapsed:.2f}s"
                    )
                    
                    results.append(result)
                    
                except Exception as e:
                    self.logger.error(f"Error processing item {item_num}: {e}")
                    raise
            
        return results
    
    def generate(self, prompts, options=None):
        """
        Generate completions for prompts.
        
        Args:
            prompts: String prompt or list of string prompts
            options: Optional dictionary of generation options
            
        Returns:
            Generated completion(s) in the format provided by the Ollama client
        """
        options = options or {}
        
        # Ensure num_predict doesn't exceed the maximum
        if "num_predict" in options:
            options["num_predict"] = min(options["num_predict"], self._max_num_predict)
        else:
            options["num_predict"] = self._max_num_predict
        
        # Convert single prompt to list for uniform processing
        is_single = not isinstance(prompts, list)
        prompts_list = [prompts] if is_single else prompts
        
        def process_prompt(prompt):
            prompt_preview = str(prompt)[:100] + ("..." if len(str(prompt)) > 100 else "")
            self.logger.info(f"Generating completion for prompt: {prompt_preview}")
            return self.client.generate(
                model=self._default_model,
                prompt=prompt,
                options=options,
                stream=False  # Ensure we get a complete response
            )
        
        results = self._process_batch(prompts_list, process_prompt)
        
        # Return single result for single prompt
        return results[0] if is_single else results
    
    def chat(self, messages, options=None):
        """
        Generate chat completions for messages.
        
        Args:
            messages: One of:
                     - Single message dict with 'role' and 'content'
                     - List of message dicts forming a conversation
                     - List of conversations (list of lists of message dicts)
            options: Optional dictionary of generation options
            
        Returns:
            Generated chat completion(s) in the format provided by the Ollama client
        """
        options = options or {}
        
        # Ensure num_predict doesn't exceed the maximum
        if "num_predict" in options:
            options["num_predict"] = min(options["num_predict"], self._max_num_predict)
        else:
            options["num_predict"] = self._max_num_predict
        
        # Determine input format and normalize
        if isinstance(messages, dict) and "role" in messages:
            # Single message -> Single conversation
            messages_list = [[messages]]
            is_single_or_conversation = True
        elif (isinstance(messages, list) and len(messages) > 0 and 
              isinstance(messages[0], dict) and "role" in messages[0]):
            # Single conversation
            messages_list = [messages]
            is_single_or_conversation = True
        else:
            # List of conversations
            messages_list = messages
            is_single_or_conversation = False
        
        def process_conversation(conversation):
            # Log first message of conversation
            if conversation and isinstance(conversation[0], dict) and "content" in conversation[0]:
                content_preview = str(conversation[0].get("content", ""))[:100]
                content_preview += "..." if len(str(conversation[0].get("content", ""))) > 100 else ""
                self.logger.info(f"Generating chat completion for conversation starting with: {content_preview}")
            else:
                self.logger.info(f"Generating chat completion for conversation")
            
            return self.client.chat(
                model=self._default_model,
                messages=conversation,
                options=options,
                stream=False  # Ensure we get a complete response
            )
        
        results = self._process_batch(messages_list, process_conversation)
        
        # Return single result for single message or conversation
        return results[0] if is_single_or_conversation else results


def extract_chat_response_content(response):
    """Extract content from ChatResponse objects from the Ollama library"""
    try:
        # Check if it's an object with a message attribute
        if hasattr(response, 'message') and hasattr(response.message, 'content'):
            return response.message.content
        
        # Check if it's a dict with a message key
        elif isinstance(response, dict) and 'message' in response:
            if isinstance(response['message'], dict) and 'content' in response['message']:
                return response['message']['content']
        
        # If all else fails, convert to string
        return str(response)
    except Exception as e:
        logger.error(f"Error extracting content from response: {e}")
        return f"[Error extracting content: {e}]"


def extract_metrics(response):
    """Extract performance metrics from response"""
    metrics = {}
    
    try:
        # For ChatResponse objects
        if hasattr(response, 'eval_count'):
            metrics['tokens_generated'] = response.eval_count
            if hasattr(response, 'eval_duration'):
                # Convert nanoseconds to seconds
                eval_duration_sec = response.eval_duration / 1_000_000_000
                if eval_duration_sec > 0:
                    metrics['tokens_per_second'] = response.eval_count / eval_duration_sec
            if hasattr(response, 'total_duration'):
                # Convert nanoseconds to seconds
                metrics['total_time'] = response.total_duration / 1_000_000_000
            
        # For dictionary responses
        elif isinstance(response, dict):
            if 'eval_count' in response:
                metrics['tokens_generated'] = response['eval_count']
                if 'eval_duration' in response:
                    # Convert nanoseconds to seconds
                    eval_duration_sec = response['eval_duration'] / 1_000_000_000
                    if eval_duration_sec > 0:
                        metrics['tokens_per_second'] = response['eval_count'] / eval_duration_sec
                if 'total_duration' in response:
                    # Convert nanoseconds to seconds
                    metrics['total_time'] = response['total_duration'] / 1_000_000_000
                    
    except Exception as e:
        logger.error(f"Error extracting metrics: {e}")
    
    return metrics


def print_response(title, response):
    """Helper function to print a response in a nicely formatted way"""
    print(f"\n{'='*10} {title} {'='*10}")
    
    try:
        # For list of responses
        if isinstance(response, list):
            print(f"Got {len(response)} responses:")
            
            for i, resp in enumerate(response):
                print(f"\n  [{i+1}] Response:")
                
                # Extract content
                content = None
                if hasattr(resp, 'response'):  # GenerateResponse
                    content = resp.response
                else:  # ChatResponse or other
                    content = extract_chat_response_content(resp)
                
                # Print content preview
                if content:
                    print(f"    Content: {content[:150]}...")
                else:
                    print(f"    Content: [No content available]")
                
                # Print metrics
                metrics = extract_metrics(resp)
                if metrics:
                    for key, value in metrics.items():
                        if isinstance(value, float):
                            print(f"    {key}: {value:.2f}")
                        else:
                            print(f"    {key}: {value}")
                else:
                    print(f"    [No metrics available]")
        
        # For single response
        else:
            # Extract content
            content = None
            if hasattr(response, 'response'):  # GenerateResponse
                content = response.response
            else:  # ChatResponse or other
                content = extract_chat_response_content(response)
            
            # Print content
            if content:
                print(f"Content: {content[:300]}...")
            else:
                print(f"Content: [No content available]")
            
            # Print metrics
            metrics = extract_metrics(response)
            if metrics:
                print("\nMetrics:")
                for key, value in metrics.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.2f}")
                    else:
                        print(f"  {key}: {value}")
            else:
                print("\n[No metrics available]")
                
    except Exception as e:
        print(f"Error printing response: {e}")
        logger.error(f"Error in print_response: {e}")
        print(f"Raw response type: {type(response)}")


def test_single_generate():
    """Test generating a completion for a single prompt"""
    logger.info("Testing single prompt generation")
    client = StandaloneOllamaClient()
    
    prompt = "Explain the concept of recursion in programming"
    options = {"temperature": 0.7, "num_predict": 100}
    
    try:
        response = client.generate(prompt, options)
        print_response("SINGLE GENERATE RESULT", response)
        return True
    except Exception as e:
        logger.error(f"Error in single generate test: {e}")
        return False


def test_batch_generate():
    """Test generating completions for multiple prompts"""
    logger.info("Testing batch prompt generation")
    client = StandaloneOllamaClient()
    
    prompts = [
        "What is the capital of France?",
        "Explain the difference between machine learning and deep learning",
        "What are the best practices for writing clean code?",
        "Describe the process of photosynthesis"
    ]
    options = {"temperature": 0.7, "num_predict": 100}
    
    try:
        responses = client.generate(prompts, options)
        print_response("BATCH GENERATE RESULTS", responses)
        return True
    except Exception as e:
        logger.error(f"Error in batch generate test: {e}")
        return False


def test_single_chat():
    """Test generating a chat completion for a single message"""
    logger.info("Testing single chat message")
    client = StandaloneOllamaClient()
    
    message = {"role": "user", "content": "What are three interesting facts about dolphins?"}
    options = {"temperature": 0.7, "num_predict": 100}
    
    try:
        response = client.chat(message, options)
        print_response("SINGLE CHAT RESULT", response)
        return True
    except Exception as e:
        logger.error(f"Error in single chat test: {e}")
        return False


def test_batch_chat():
    """Test generating chat completions for multiple conversations"""
    logger.info("Testing batch chat conversations")
    client = StandaloneOllamaClient()
    
    conversations = [
        [{"role": "user", "content": "What's the best way to learn programming?"}],
        [{"role": "user", "content": "Tell me a joke about computers"}],
        [{"role": "user", "content": "What's the difference between a CPU and a GPU?"}],
        [{"role": "user", "content": "How do I make a good pasta sauce?"}]
    ]
    options = {"temperature": 0.7, "num_predict": 100}
    
    try:
        responses = client.chat(conversations, options)
        print_response("BATCH CHAT RESULTS", responses)
        return True
    except Exception as e:
        logger.error(f"Error in batch chat test: {e}")
        return False


def run_tests():
    """Run all tests and report results"""
    print("\n==== OLLAMA CLIENT TEST SCRIPT ====\n")
    print("This script tests the StandaloneOllamaClient with both chat and generate functions")
    print("Testing single and batch (4) inputs for each function\n")
    
    # Run all tests
    tests = [
        ("Single Generate", test_single_generate),
        ("Batch Generate", test_batch_generate),
        ("Single Chat", test_single_chat),
        ("Batch Chat", test_batch_chat)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n---- Running Test: {name} ----")
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            logger.error(f"Test '{name}' failed with exception: {e}")
            results.append((name, False))
    
    # Print summary
    print("\n==== TEST SUMMARY ====")
    all_passed = True
    for name, success in results:
        status = "PASSED" if success else "FAILED"
        print(f"{name}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        print("\nAll tests passed successfully!")
        return 0
    else:
        print("\nSome tests failed. Check the logs for details.")
        return 1


if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)