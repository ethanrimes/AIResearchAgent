from __future__ import annotations

import json
import os
import sys
import time
import logging
from typing import Dict, List, Any, Optional, Union

import ollama
from ollama import Client
from dataclasses import dataclass

from core.agent.agent_config import AgentConfig

# ---------------------------------------------------------
#  Dataclasses
# ---------------------------------------------------------
@dataclass
class GenerationParameters:
    """
    Parameters for text generation.

    NOTE: The `model` field is **ignored** by OllamaClient; the client will
    replace it with its own default model before sending the request.
    """
    model: str
    prompt: str
    system: Optional[str] = None
    template: Optional[str] = None
    context: Optional[List[int]] = None
    stream: bool = False  # This will be forced to False
    raw: bool = False
    format: Optional[str] = None

    # generation options
    num_predict: Optional[int] = 32768
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repeat_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    seed: Optional[int] = None
    mirostat: Optional[int] = None
    mirostat_tau: Optional[float] = None
    mirostat_eta: Optional[float] = None
    stop: Optional[List[str]] = None
    tfs_z: Optional[float] = None
    num_gpu: Optional[int] = None
    num_thread: Optional[int] = None
    num_ctx: Optional[int] = 32768
    num_batch: Optional[int] = None
    num_keep: Optional[int] = None

    def to_options_dict(self) -> Dict[str, Any]:
        """Convert generation parameters to an options dictionary for the Ollama client."""
        options = {}
        
        # Add generation options
        if self.num_predict is not None:    options["num_predict"] = self.num_predict
        if self.temperature is not None:    options["temperature"] = self.temperature
        if self.top_k is not None:          options["top_k"] = self.top_k
        if self.top_p is not None:          options["top_p"] = self.top_p
        if self.repeat_penalty is not None: options["repeat_penalty"] = self.repeat_penalty
        if self.presence_penalty is not None: options["presence_penalty"] = self.presence_penalty
        if self.frequency_penalty is not None: options["frequency_penalty"] = self.frequency_penalty
        if self.seed is not None:           options["seed"] = self.seed
        if self.mirostat is not None:       options["mirostat"] = self.mirostat
        if self.mirostat_tau is not None:   options["mirostat_tau"] = self.mirostat_tau
        if self.mirostat_eta is not None:   options["mirostat_eta"] = self.mirostat_eta
        if self.stop is not None:           options["stop"] = self.stop
        if self.tfs_z is not None:          options["tfs_z"] = self.tfs_z
        if self.num_gpu is not None:        options["num_gpu"] = self.num_gpu
        if self.num_thread is not None:     options["num_thread"] = self.num_thread
        if self.num_ctx is not None:        options["num_ctx"] = self.num_ctx
        if self.num_batch is not None:      options["num_batch"] = self.num_batch
        if self.num_keep is not None:       options["num_keep"] = self.num_keep
        
        # Add other parameters
        if self.system is not None:        options["system"] = self.system
        if self.template is not None:      options["template"] = self.template
        if self.format is not None:        options["format"] = self.format
        
        return options


# ---------------------------------------------------------
#  Logging Configuration
# ---------------------------------------------------------
def setup_module_logger(logger_name: str) -> logging.Logger:
    """
    Set up a module-specific logger with appropriate handlers.
    
    Args:
        logger_name: The name for the logger
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(logger_name)
    
    # Only configure if the logger doesn't already have handlers
    if not logger.handlers:
        # Set the log level
        logger.setLevel(logging.INFO)
        
        # Create console handler that logs to stdout
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        # Add the handler to the logger
        logger.addHandler(console_handler)
        
        # Prevent propagation to root logger to avoid duplicate messages
        logger.propagate = False
        
        logger.debug(f"Logger {logger_name} has been configured")
    
    return logger


# ---------------------------------------------------------
#  Main Client
# ---------------------------------------------------------
class OllamaClient:
    """
    Immutable Ollama API client for text generation.

    After construction the base URL and the default model **cannot** be
    modified from the outside, and every request sent through this client
    will automatically contain these enforced defaults.
    """

    # ---------- construction & immutability helpers ----------
    def __init__(self) -> None:
        # Set base URL from environment with fallback to default
        base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
        
        # Set immutable attributes using object.__setattr__
        object.__setattr__(self, "_base_url", base_url)
        object.__setattr__(self, "_default_model", AgentConfig.reasoner_model())
        
        if not self._default_model:
            raise ValueError("Reasoner model not set in configuration")
        
        # Configure the Ollama client
        object.__setattr__(self, "_client", Client(host=base_url))
        
        # Configure module-specific logger
        logger_name = "ai_research_agent.ollama_client"
        object.__setattr__(self, "logger", setup_module_logger(logger_name))
        self.logger.info(f"Initializing OllamaClient with base URL: {self._base_url}")
        self.logger.info(f"Using model from configuration: {self._default_model}")

        # Verify connectivity
        self._check_connection()

        # *freeze* -- disallow changes to critical attrs after this point
        object.__setattr__(self, "_immutable", True)

    def __setattr__(self, name: str, value: Any) -> None:
        # prevent external mutation of critical fields once initialized
        if getattr(self, "_immutable", False) and name in {
            "_base_url", "_default_model", "base_url", "default_model"
        }:
            raise AttributeError(f"{name} is read-only after initialization.")
        super().__setattr__(name, value)

    # read-only public accessors
    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def default_model(self) -> str:
        return self._default_model

    # ---------- internal helpers ----------
    def _check_connection(self) -> None:
        try:
            # Get available models to check connectivity
            models = self._client.list()
            self.logger.info(f"Connected to Ollama server at {self._base_url}")
        except Exception as exc:
            self.logger.error(f"Failed to connect to Ollama server: {exc}")
            raise ConnectionError(f"Unable to connect to Ollama at {self._base_url}") from exc

    def _extract_text(self, response) -> str:
        """Extract plain text from a response object."""
        if hasattr(response, 'response'):
            return response.response
        elif hasattr(response, 'message') and hasattr(response.message, 'content'):
            return response.message.content
        elif isinstance(response, dict):
            if 'response' in response:
                return response['response']
            elif 'message' in response and isinstance(response['message'], dict):
                return response['message'].get('content', '')
        return str(response)

    def _get_tokens_generated(self, response) -> int:
        """Extract the number of tokens generated from a response object."""
        if hasattr(response, 'eval_count'):
            return response.eval_count
        elif isinstance(response, dict) and 'eval_count' in response:
            return response['eval_count']
        return 0

    # ---------- public API ----------
    def generate(self, 
                prompts: Union[str, List[str]], 
                params: Optional[Union[GenerationParameters, Dict[str, Any]]] = None) -> Union[str, List[str]]:
        """
        Generate text using the default model.
        
        Args:
            prompts: String prompt or list of string prompts
            params: Optional generation parameters
            
        Returns:
            Generated text as a string or list of strings
        """
        # Convert params to options dict if provided
        options = {}
        if params:
            if isinstance(params, GenerationParameters):
                options = params.to_options_dict()
            else:
                options = dict(params)
        
        # Always set stream to False
        options["stream"] = False
        
        # Ensure we have reasonable num_predict if not specified
        if "num_predict" not in options:
            options["num_predict"] = 32768
        
        # Convert single prompt to list for uniform processing
        is_single = not isinstance(prompts, list)
        prompts_list = [prompts] if is_single else prompts
        
        self.logger.info(f"Generating text for {len(prompts_list)} prompts with model {self._default_model}")
        
        results = []
        for idx, prompt in enumerate(prompts_list):
            start_time = time.time()
            prompt_preview = str(prompt)[:100] + ("..." if len(str(prompt)) > 100 else "")
            self.logger.info(f"Generating completion for prompt {idx+1}/{len(prompts_list)}: {prompt_preview}")
            
            try:
                response = self._client.generate(
                    model=self._default_model,
                    prompt=prompt,
                    options=options
                )
                
                elapsed = time.time() - start_time
                result_text = self._extract_text(response)
                tokens_generated = self._get_tokens_generated(response)
                
                tokens_per_sec = 0
                if elapsed > 0 and tokens_generated > 0:
                    tokens_per_sec = tokens_generated / elapsed
                
                preview = result_text[:100] + ("..." if len(result_text) > 100 else "")
                self.logger.info(
                    f"Response {idx+1}/{len(prompts_list)} completed in {elapsed:.2f}s, "
                    f"tokens: {tokens_generated}, tokens/sec: {tokens_per_sec:.2f}"
                )
                self.logger.info(f"Preview: {preview}")
                
                results.append(result_text)
                
            except Exception as e:
                self.logger.error(f"Error processing prompt {idx+1}/{len(prompts_list)}: {e}")
                raise
        
        # Return single result for single prompt
        return results[0] if is_single else results

    def chat(self, 
            messages: Union[Dict[str, str], List[Dict[str, str]], List[List[Dict[str, str]]]], 
            params: Optional[Union[GenerationParameters, Dict[str, Any]]] = None) -> Union[str, List[str]]:
        """
        Generate chat completions.
        
        Args:
            messages: One of:
                - Single message dict with 'role' and 'content'
                - List of message dicts forming a conversation
                - List of conversations (list of lists of message dicts)
            params: Optional generation parameters
            
        Returns:
            Generated text as a string or list of strings
        """
        # Convert params to options dict if provided
        options = {}
        if params:
            if isinstance(params, GenerationParameters):
                options = params.to_options_dict()
            else:
                options = dict(params)
        
        # Always set stream to False
        options["stream"] = False
        
        # Ensure we have reasonable num_predict if not specified
        if "num_predict" not in options:
            options["num_predict"] = 32768
        
        # Determine input format and normalize
        if isinstance(messages, dict) and "role" in messages:
            # Single message -> Single conversation
            messages_list = [[messages]]
            is_single_message = True
        elif (isinstance(messages, list) and len(messages) > 0 and 
              isinstance(messages[0], dict) and "role" in messages[0]):
            # Single conversation
            messages_list = [messages]
            is_single_message = True
        else:
            # List of conversations
            messages_list = messages
            is_single_message = False
        
        self.logger.info(f"Generating chat completions for {len(messages_list)} conversations with model {self._default_model}")
        
        results = []
        for idx, conversation in enumerate(messages_list):
            start_time = time.time()
            
            # Log first message of conversation
            content_preview = ""
            if conversation and isinstance(conversation[0], dict) and "content" in conversation[0]:
                content_preview = str(conversation[0].get("content", ""))[:100]
                content_preview += "..." if len(str(conversation[0].get("content", ""))) > 100 else ""
            
            self.logger.info(f"Generating chat completion {idx+1}/{len(messages_list)} for conversation starting with: {content_preview}")
            
            try:
                response = self._client.chat(
                    model=self._default_model,
                    messages=conversation,
                    options=options
                )
                
                elapsed = time.time() - start_time
                result_text = self._extract_text(response)
                tokens_generated = self._get_tokens_generated(response)
                
                tokens_per_sec = 0
                if elapsed > 0 and tokens_generated > 0:
                    tokens_per_sec = tokens_generated / elapsed
                
                preview = result_text[:100] + ("..." if len(result_text) > 100 else "")
                self.logger.info(
                    f"Response {idx+1}/{len(messages_list)} completed in {elapsed:.2f}s, "
                    f"tokens: {tokens_generated}, tokens/sec: {tokens_per_sec:.2f}"
                )
                self.logger.info(f"Preview: {preview}")
                
                results.append(result_text)
                
            except Exception as e:
                self.logger.error(f"Error processing conversation {idx+1}/{len(messages_list)}: {e}")
                raise
        
        # Return single result for single message or conversation
        return results[0] if is_single_message else results

    def get_model_metrics(self) -> Dict[str, Any]:
        """
        Quick performance probe for the default model.
        
        Returns:
            Dictionary with performance metrics
        """
        self.logger.info(f"Performing performance probe for model: {self._default_model}")
        
        t0 = time.time()
        response = self._client.generate(
            model=self._default_model,
            prompt="Hello",
            options={"num_predict": 1, "temperature": 0, "stream": False}
        )
        elapsed = time.time() - t0
        
        tokens_per_second = 0
        if hasattr(response, 'eval_count') and hasattr(response, 'eval_duration'):
            # Convert nanoseconds to seconds if necessary
            eval_duration_sec = response.eval_duration / 1_000_000_000 if response.eval_duration > 1000 else response.eval_duration
            if eval_duration_sec > 0:
                tokens_per_second = response.eval_count / eval_duration_sec
        
        metrics = {
            "model": self._default_model,
            "startup_time": elapsed,
            "first_token_time": elapsed,
            "tokens_per_second": tokens_per_second,
            "memory_used": getattr(response, 'prompt_eval_count', 0)
        }
        
        self.logger.info(f"Model metrics: {metrics}")
        return metrics