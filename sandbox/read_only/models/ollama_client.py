"""
Immutable client for text generation with an Ollama server.

All requests are forced to use:
  • the base-URL chosen at construction time
  • the default model chosen at construction time

Client code cannot override these settings.
"""

from __future__ import annotations

import json
import os
import sys
import time
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, Sequence

import aiohttp
import requests
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
    format: Optional[str] = None
    raw: bool = False

    # generation options
    num_predict: Optional[int] = None
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
    num_ctx: Optional[int] = None
    num_batch: Optional[int] = None
    num_keep: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "prompt": self.prompt,          # model will be injected later
        }
        if self.system is not None:     d["system"]   = self.system
        if self.template is not None:   d["template"] = self.template
        if self.context is not None:    d["context"]  = self.context
        if self.raw:                    d["raw"]      = True
        if self.format is not None:     d["format"]   = self.format

        # collect generation-options (excluding stream which should be top-level)
        base_fields = set(d.keys()) | {"model", "prompt", "raw"}
        options = {k: v for k, v in self.__dict__.items()
                   if k not in base_fields and v is not None}
        if options:
            d["options"] = options
        return d


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
    # Endpoint for generation
    GENERATE_ENDPOINT = "/api/generate"

    # ---------- construction & immutability helpers ----------
    def __init__(self) -> None:
        # Get configuration from the registry using class method
        # This properly uses the AgentConfig as a singleton with class methods
        
        # Set base URL from environment with fallback to default
        base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
        
        # Set immutable attributes using object.__setattr__
        object.__setattr__(self, "_base_url", base_url)
        object.__setattr__(self, "_default_model", AgentConfig.reasoner_model())
        
        # Get batch size and max num predict from config
        config = AgentConfig.config()
        batch_size = config.get("batch_size", 4)  # Default to 4 if not specified
        max_num_predict = config.get("max_num_predict", 2048)  # Default to 2048 if not specified
        
        object.__setattr__(self, "_batch_size", batch_size)
        object.__setattr__(self, "_max_num_predict", max_num_predict)
        
        if not self._default_model:
            raise ValueError("Reasoner model not set in configuration")
        
        # Configure module-specific logger
        logger_name = "ai_research_agent.ollama_client"
        object.__setattr__(self, "logger", setup_module_logger(logger_name))
        self.logger.info(f"Initializing OllamaClient with base URL: {self._base_url}")
        self.logger.info(f"Using model from configuration: {self._default_model}")
        self.logger.info(f"Batch size: {self._batch_size}, Max num predict: {self._max_num_predict}")

        # Verify connectivity
        self._check_connection()

        # *freeze* – disallow changes to critical attrs after this point
        object.__setattr__(self, "_immutable", True)

    def __setattr__(self, name: str, value: Any) -> None:
        # prevent external mutation of critical fields once initialized
        if getattr(self, "_immutable", False) and name in {
            "_base_url", "_default_model", "_batch_size", "_max_num_predict",
            "base_url", "default_model", "batch_size", "max_num_predict"
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
        
    @property
    def batch_size(self) -> int:
        return self._batch_size
        
    @property
    def max_num_predict(self) -> int:
        return self._max_num_predict

    # ---------- internal helpers ----------
    def _get_url(self, endpoint: str) -> str:
        return f"{self._base_url}{endpoint}"

    def _check_connection(self) -> None:
        try:
            # Using a simple GET to check if server is responsive
            r = requests.get(f"{self._base_url}/api/tags", timeout=5)
            r.raise_for_status()
            self.logger.info("Connected to Ollama server at %s", self._base_url)
        except requests.RequestException as exc:
            self.logger.error(f"Failed to connect to Ollama server: {exc}")
            raise ConnectionError(f"Unable to connect to Ollama at {self._base_url}") from exc

    def _enforce_defaults(self, params: Union[GenerationParameters, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Convert `params` to a dict and inject the enforced default model and cap num_predict.
        """
        if hasattr(params, "to_dict"):
            data = params.to_dict()            # dataclass instance
        else:
            data = dict(params)               # shallow copy of user dict
            
        data["model"] = self._default_model   # hard-overwrite model
        
        # Cap num_predict if it exceeds max_num_predict
        if "options" in data and "num_predict" in data["options"]:
            data["options"]["num_predict"] = min(
                data["options"]["num_predict"], 
                self._max_num_predict
            )
        elif "num_predict" in data:
            data["num_predict"] = min(data["num_predict"], self._max_num_predict)
        else:
            # Set default num_predict if not provided
            if "options" not in data:
                data["options"] = {}
            data["options"]["num_predict"] = self._max_num_predict
            
        return data

    async def _make_request_async(self, endpoint: str, data: Dict[str, Any], *, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Asynchronously make a request to the Ollama API.
        
        Args:
            endpoint: API endpoint to call
            data: Request data
            timeout: Request timeout in seconds
            
        Returns:
            Response dictionary
        """
        url = self._get_url(endpoint)
        self.logger.debug(f"Making async request to {url}")
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=data, timeout=timeout) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        self.logger.error(f"Request failed with status {resp.status}: {error_text}")
                        raise aiohttp.ClientResponseError(
                            request_info=resp.request_info,
                            history=resp.history,
                            status=resp.status,
                            message=error_text
                        )
                    
                    response_text = await resp.text()
                    if not response_text:
                        self.logger.warning("Received empty response")
                        return {"response": ""}
                    
                    try:
                        return json.loads(response_text)
                    except json.JSONDecodeError:
                        self.logger.warning("Failed to parse JSON response, returning raw text")
                        return {"response": response_text}
            except asyncio.TimeoutError:
                self.logger.error(f"Request timed out after {timeout}s")
                raise
            except Exception as e:
                self.logger.error(f"Error during async request: {e}")
                raise

    # ---------- public API ----------
    async def generate_async(self,
                   params: Union[GenerationParameters, Dict[str, Any], List[Union[GenerationParameters, Dict[str, Any]]]],
                   timeout: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Generate text asynchronously using the default model, handling batched prompts.
        
        Args:
            params: Generation parameters or list of parameters (model will be overridden)
            timeout: Request timeout in seconds
            
        Returns:
            List of response dictionaries containing generated text and metrics
        """
        # Convert single parameter to list for uniform processing
        if not isinstance(params, list):
            params_list = [params]
        else:
            params_list = params
            
        self.logger.info(f"Processing {len(params_list)} prompts with batch size {self._batch_size}")
        
        # Process in batches
        results = []
        for i in range(0, len(params_list), self._batch_size):
            batch = params_list[i:i + self._batch_size]
            self.logger.info(f"Processing batch {i//self._batch_size + 1} with {len(batch)} prompts")
            
            # Prepare all requests in the batch
            requests = []
            for j, param in enumerate(batch):
                data = self._enforce_defaults(param)
                prompt_preview = data["prompt"][:100] + ("..." if len(data["prompt"]) > 100 else "")
                self.logger.info(f"Batch {i//self._batch_size + 1}, Prompt {j+1}: {prompt_preview}")
                
                # Log generation parameters
                gen_params = {k: v for k, v in data.items() if k != "prompt"}
                gen_params["options"] = data.get("options", {})
                self.logger.info(f"Generation parameters: {json.dumps(gen_params)}")
                
                requests.append((data, time.time()))
                
            # Process all requests in the batch concurrently
            batch_results = await asyncio.gather(
                *[self._make_request_async(self.GENERATE_ENDPOINT, data, timeout=timeout) 
                  for data, _ in requests],
                return_exceptions=True
            )
            
            # Process results and add metrics
            for j, ((data, start_time), result) in enumerate(zip(requests, batch_results)):
                if isinstance(result, Exception):
                    self.logger.error(f"Error in batch {i//self._batch_size + 1}, prompt {j+1}: {result}")
                    results.append({
                        "error": str(result),
                        "metrics": {
                            "inference_time": time.time() - start_time,
                            "tokens_per_second": 0
                        }
                    })
                else:
                    elapsed = time.time() - start_time
                    response_text = result.get("response", "")
                    tokens = len(response_text.split())
                    
                    result["metrics"] = {
                        "inference_time": elapsed,
                        "tokens_per_second": tokens / elapsed if elapsed else 0,
                        "total_tokens": tokens
                    }
                    
                    # Log completed generation info
                    self.logger.info(
                        f"Batch {i//self._batch_size + 1}, Prompt {j+1} completed: "
                        f"{len(response_text)} chars, {tokens} tokens, "
                        f"{result['metrics']['tokens_per_second']:.2f} tokens/sec"
                    )
                    self.logger.info(
                        f"Response preview: {response_text[:100]}{'...' if len(response_text) > 100 else ''}"
                    )
                    
                    results.append(result)
                    
        return results

    def generate(self,
             params: Union[GenerationParameters, Dict[str, Any], List[Union[GenerationParameters, Dict[str, Any]]]],
             timeout: Optional[float] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Generate text using the default model, with support for batched prompts.
        
        Args:
            params: Generation parameters or list of parameters (model will be overridden)
            timeout: Request timeout in seconds
            
        Returns:
            Single response dictionary or list of response dictionaries
        """
        # Determine if this is a single or batch request
        is_single = not isinstance(params, list)
        
        # Run the async method in a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(self.generate_async(params, timeout))
        finally:
            loop.close()
            
        # Return single result or list based on input type
        if is_single:
            return results[0]
        return results

    def get_model_metrics(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Quick performance probe for the default model.
        
        Returns:
            Dictionary with performance metrics
        """
        self.logger.info(f"Performing performance probe for model: {self._default_model}")
        gen_params = GenerationParameters(
            model=self._default_model,
            prompt="Hello",
            num_predict=1,
            temperature=0
        )
        t0 = time.time()
        resp = self.generate(gen_params, timeout=timeout)
        elapsed = time.time() - t0
        metrics = {
            "model": self._default_model,
            "startup_time": elapsed,
            "first_token_time": elapsed,
            "tokens_per_second": resp["metrics"]["tokens_per_second"],
            "memory_usage": resp.get("memory_used", 0)
        }
        self.logger.info(f"Model metrics: {metrics}")
        return metrics