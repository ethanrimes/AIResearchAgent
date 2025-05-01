"""
Modular interface for querying the Ollama server on the host machine.

This module provides a comprehensive client for interacting with all
Ollama API endpoints, with support for different models, generation
parameters, and model management operations.
"""

import os
import json
import time
import logging
import requests
from typing import Dict, List, Any, Optional, Union, Generator, Tuple
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaEndpoints(Enum):
    """Enum for Ollama API endpoints."""
    GENERATE = "/api/generate"
    CHAT = "/api/chat"
    EMBEDDINGS = "/api/embeddings"
    TAGS = "/api/tags"
    SHOW = "/api/show"
    PULL = "/api/pull"
    PUSH = "/api/push"
    CREATE = "/api/create"
    DELETE = "/api/delete"
    COPY = "/api/copy"


@dataclass
class GenerationParameters:
    """Parameters for text generation."""
    model: str
    prompt: str
    system: Optional[str] = None
    template: Optional[str] = None
    context: Optional[List[int]] = None
    stream: bool = False
    raw: bool = False
    format: Optional[str] = None
    
    # Generation options
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
        """Convert parameters to dictionary format for API request."""
        result = {
            "model": self.model,
            "prompt": self.prompt,
        }
        
        # Add optional parameters if they are set
        if self.system is not None:
            result["system"] = self.system
        if self.template is not None:
            result["template"] = self.template
        if self.context is not None:
            result["context"] = self.context
        if self.stream is not False:
            result["stream"] = self.stream
        if self.raw is not False:
            result["raw"] = self.raw
        if self.format is not None:
            result["format"] = self.format
            
        # Add generation options if they are set
        options = {}
        for key, value in self.__dict__.items():
            if key not in ["model", "prompt", "system", "template", "context", "stream", "raw", "format"] and value is not None:
                options[key] = value
                
        if options:
            result["options"] = options
            
        return result


@dataclass
class ChatMessage:
    """A message in a chat conversation."""
    role: str  # "user", "assistant", or "system"
    content: str
    images: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chat message to dictionary format for API request."""
        result = {
            "role": self.role,
            "content": self.content
        }
        
        if self.images:
            result["images"] = self.images
            
        return result


@dataclass
class ChatParameters:
    """Parameters for chat completion."""
    model: str
    messages: List[ChatMessage]
    stream: bool = False
    format: Optional[str] = None
    
    # Generation options (same as for generate)
    options: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary format for API request."""
        result = {
            "model": self.model,
            "messages": [msg.to_dict() for msg in self.messages],
        }
        
        if self.stream:
            result["stream"] = self.stream
        if self.format:
            result["format"] = self.format
        if self.options:
            result["options"] = self.options
            
        return result


@dataclass
class EmbeddingParameters:
    """Parameters for generating embeddings."""
    model: str
    prompt: str
    options: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary format for API request."""
        result = {
            "model": self.model,
            "prompt": self.prompt,
        }
        
        if self.options:
            result["options"] = self.options
            
        return result


class OllamaClient:
    """Client for interacting with the Ollama server."""
    
    def __init__(self, base_url: Optional[str] = None):
        """
        Initialize the Ollama client.
        
        Args:
            base_url: Base URL for the Ollama server.
                      Defaults to OLLAMA_BASE_URL environment variable
                      or http://host.docker.internal:11434 if not set.
        """
        self.base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
        self.session = requests.Session()
        
        # Verify connection to Ollama server
        self._check_connection()
        
    def _check_connection(self) -> bool:
        """
        Check if the Ollama server is accessible.
        
        Returns:
            bool: True if connection is successful, False otherwise.
        
        Raises:
            ConnectionError: If unable to connect to the Ollama server.
        """
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            logger.info(f"Successfully connected to Ollama server at {self.base_url}")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to Ollama server at {self.base_url}: {e}")
            raise ConnectionError(f"Failed to connect to Ollama server: {e}")
    
    def _get_url(self, endpoint: Union[str, OllamaEndpoints]) -> str:
        """
        Get the full URL for an API endpoint.
        
        Args:
            endpoint: API endpoint to access.
        
        Returns:
            str: Full URL for the endpoint.
        """
        if isinstance(endpoint, OllamaEndpoints):
            endpoint = endpoint.value
        return f"{self.base_url}{endpoint}"
    
    def _make_request(self, 
                     method: str, 
                     endpoint: Union[str, OllamaEndpoints],
                     data: Optional[Dict[str, Any]] = None,
                     stream: bool = False,
                     timeout: Optional[float] = None) -> Union[requests.Response, Dict[str, Any]]:
        """
        Make a request to the Ollama API.
        
        Args:
            method: HTTP method (GET, POST, DELETE).
            endpoint: API endpoint to access.
            data: Request data for POST requests.
            stream: Whether to stream the response.
            timeout: Request timeout in seconds.
        
        Returns:
            Union[requests.Response, Dict[str, Any]]: Response from the API.
        
        Raises:
            requests.exceptions.RequestException: If the request fails.
        """
        url = self._get_url(endpoint)
        
        try:
            if method.upper() == "GET":
                response = self.session.get(url, timeout=timeout)
            elif method.upper() == "POST":
                if stream:
                    return self.session.post(url, json=data, stream=True, timeout=timeout)
                response = self.session.post(url, json=data, timeout=timeout)
            elif method.upper() == "DELETE":
                response = self.session.delete(url, json=data, timeout=timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            
            # For non-streaming responses, parse JSON response
            if response.content and not stream:
                return response.json()
            return response
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request to {url} failed: {e}")
            raise
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all available models on the Ollama server.
        
        Returns:
            List[Dict[str, Any]]: List of available models with metadata.
        """
        try:
            response = self._make_request("GET", OllamaEndpoints.TAGS)
            return response.get("models", [])
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            raise
    
    def generate(self, 
                params: Union[GenerationParameters, Dict[str, Any]],
                timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Generate text using the specified model.
        
        Args:
            params: Generation parameters.
            timeout: Request timeout in seconds.
        
        Returns:
            Dict[str, Any]: Generated text and metadata.
        """
        if isinstance(params, GenerationParameters):
            data = params.to_dict()
        else:
            data = params
            
        # If streaming is enabled, handle accordingly
        if data.get("stream", False):
            return self.generate_stream(params, timeout)
            
        try:
            start_time = time.time()
            response = self._make_request("POST", OllamaEndpoints.GENERATE, data, timeout=timeout)
            
            # Add performance metrics
            elapsed_time = time.time() - start_time
            response["metrics"] = {
                "inference_time": elapsed_time,
                "tokens_per_second": len(response.get("response", "").split()) / elapsed_time if elapsed_time > 0 else 0
            }
            
            return response
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def generate_stream(self, 
                       params: Union[GenerationParameters, Dict[str, Any]],
                       timeout: Optional[float] = None) -> Generator[Dict[str, Any], None, None]:
        """
        Stream generated text from the model.
        
        Args:
            params: Generation parameters.
            timeout: Request timeout in seconds.
        
        Yields:
            Dict[str, Any]: Chunks of generated text and metadata.
        """
        if isinstance(params, GenerationParameters):
            data = params.to_dict()
        else:
            data = params.copy()
            
        # Ensure streaming is enabled
        data["stream"] = True
        
        try:
            response = self._make_request("POST", OllamaEndpoints.GENERATE, data, stream=True, timeout=timeout)
            start_time = time.time()
            total_tokens = 0
            
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    total_tokens += 1
                    
                    # Add performance metrics
                    elapsed_time = time.time() - start_time
                    chunk["metrics"] = {
                        "inference_time": elapsed_time,
                        "tokens_per_second": total_tokens / elapsed_time if elapsed_time > 0 else 0
                    }
                    
                    yield chunk
                    
                    # If done, break the loop
                    if chunk.get("done", False):
                        break
                        
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise
    
    def chat(self, 
            params: Union[ChatParameters, Dict[str, Any]],
            timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Generate a chat completion.
        
        Args:
            params: Chat parameters.
            timeout: Request timeout in seconds.
        
        Returns:
            Dict[str, Any]: Chat completion and metadata.
        """
        if isinstance(params, ChatParameters):
            data = params.to_dict()
        else:
            data = params
            
        # If streaming is enabled, handle accordingly
        if data.get("stream", False):
            return self.chat_stream(params, timeout)
            
        try:
            start_time = time.time()
            response = self._make_request("POST", OllamaEndpoints.CHAT, data, timeout=timeout)
            
            # Add performance metrics
            elapsed_time = time.time() - start_time
            response_content = response.get("message", {}).get("content", "")
            response["metrics"] = {
                "inference_time": elapsed_time,
                "tokens_per_second": len(response_content.split()) / elapsed_time if elapsed_time > 0 else 0
            }
            
            return response
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            raise
    
    def chat_stream(self, 
                  params: Union[ChatParameters, Dict[str, Any]],
                  timeout: Optional[float] = None) -> Generator[Dict[str, Any], None, None]:
        """
        Stream chat completion from the model.
        
        Args:
            params: Chat parameters.
            timeout: Request timeout in seconds.
        
        Yields:
            Dict[str, Any]: Chunks of chat completion and metadata.
        """
        if isinstance(params, ChatParameters):
            data = params.to_dict()
        else:
            data = params.copy()
            
        # Ensure streaming is enabled
        data["stream"] = True
        
        try:
            response = self._make_request("POST", OllamaEndpoints.CHAT, data, stream=True, timeout=timeout)
            start_time = time.time()
            total_tokens = 0
            
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    total_tokens += 1
                    
                    # Add performance metrics
                    elapsed_time = time.time() - start_time
                    chunk["metrics"] = {
                        "inference_time": elapsed_time,
                        "tokens_per_second": total_tokens / elapsed_time if elapsed_time > 0 else 0
                    }
                    
                    yield chunk
                    
                    # If done, break the loop
                    if chunk.get("done", False):
                        break
                        
        except Exception as e:
            logger.error(f"Streaming chat completion failed: {e}")
            raise
    
    def get_embeddings(self, 
                     params: Union[EmbeddingParameters, Dict[str, Any]],
                     timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Get embeddings for a text prompt.
        
        Args:
            params: Embedding parameters.
            timeout: Request timeout in seconds.
        
        Returns:
            Dict[str, Any]: Embeddings and metadata.
        """
        if isinstance(params, EmbeddingParameters):
            data = params.to_dict()
        else:
            data = params
            
        try:
            response = self._make_request("POST", OllamaEndpoints.EMBEDDINGS, data, timeout=timeout)
            return response
        except Exception as e:
            logger.error(f"Embeddings generation failed: {e}")
            raise
    
    def show_model_info(self, model_name: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Show information about a model.
        
        Args:
            model_name: Name of the model to show.
            timeout: Request timeout in seconds.
        
        Returns:
            Dict[str, Any]: Model information.
        """
        try:
            data = {"name": model_name}
            response = self._make_request("POST", OllamaEndpoints.SHOW, data, timeout=timeout)
            return response
        except Exception as e:
            logger.error(f"Failed to get model info for {model_name}: {e}")
            raise
    
    def pull_model(self, 
                 model_name: str, 
                 insecure: bool = False,
                 stream: bool = True,
                 timeout: Optional[float] = None) -> Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]:
        """
        Pull a model from the Ollama library.
        
        Args:
            model_name: Name of the model to pull.
            insecure: Allow insecure connections.
            stream: Whether to stream the pull progress.
            timeout: Request timeout in seconds.
        
        Returns:
            Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]: 
                Pull status or generator yielding progress updates.
        """
        data = {
            "name": model_name,
            "insecure": insecure,
            "stream": stream
        }
        
        try:
            if stream:
                response = self._make_request("POST", OllamaEndpoints.PULL, data, stream=True, timeout=timeout)
                
                for line in response.iter_lines():
                    if line:
                        yield json.loads(line)
            else:
                return self._make_request("POST", OllamaEndpoints.PULL, data, timeout=timeout)
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            raise
    
    def delete_model(self, model_name: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Delete a model from the Ollama server.
        
        Args:
            model_name: Name of the model to delete.
            timeout: Request timeout in seconds.
        
        Returns:
            Dict[str, Any]: Deletion status.
        """
        try:
            data = {"name": model_name}
            response = self._make_request("DELETE", OllamaEndpoints.DELETE, data, timeout=timeout)
            return response
        except Exception as e:
            logger.error(f"Failed to delete model {model_name}: {e}")
            raise
    
    def copy_model(self, source: str, destination: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Copy a model to create a new one.
        
        Args:
            source: Source model name.
            destination: Destination model name.
            timeout: Request timeout in seconds.
        
        Returns:
            Dict[str, Any]: Copy status.
        """
        try:
            data = {"source": source, "destination": destination}
            response = self._make_request("POST", OllamaEndpoints.COPY, data, timeout=timeout)
            return response
        except Exception as e:
            logger.error(f"Failed to copy model from {source} to {destination}: {e}")
            raise
    
    def create_model(self, 
                   model_name: str, 
                   modelfile: str,
                   stream: bool = True,
                   timeout: Optional[float] = None) -> Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]:
        """
        Create a model from a Modelfile.
        
        Args:
            model_name: Name for the new model.
            modelfile: Content of the Modelfile.
            stream: Whether to stream the creation progress.
            timeout: Request timeout in seconds.
        
        Returns:
            Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]: 
                Creation status or generator yielding progress updates.
        """
        data = {
            "name": model_name,
            "modelfile": modelfile,
            "stream": stream
        }
        
        try:
            if stream:
                response = self._make_request("POST", OllamaEndpoints.CREATE, data, stream=True, timeout=timeout)
                
                for line in response.iter_lines():
                    if line:
                        yield json.loads(line)
            else:
                return self._make_request("POST", OllamaEndpoints.CREATE, data, timeout=timeout)
        except Exception as e:
            logger.error(f"Failed to create model {model_name}: {e}")
            raise
    
    def push_model(self, 
                 model_name: str, 
                 insecure: bool = False,
                 stream: bool = True,
                 timeout: Optional[float] = None) -> Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]:
        """
        Push a model to the Ollama library.
        
        Args:
            model_name: Name of the model to push.
            insecure: Allow insecure connections.
            stream: Whether to stream the push progress.
            timeout: Request timeout in seconds.
        
        Returns:
            Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]: 
                Push status or generator yielding progress updates.
        """
        data = {
            "name": model_name,
            "insecure": insecure,
            "stream": stream
        }
        
        try:
            if stream:
                response = self._make_request("POST", OllamaEndpoints.PUSH, data, stream=True, timeout=timeout)
                
                for line in response.iter_lines():
                    if line:
                        yield json.loads(line)
            else:
                return self._make_request("POST", OllamaEndpoints.PUSH, data, timeout=timeout)
        except Exception as e:
            logger.error(f"Failed to push model {model_name}: {e}")
            raise
    
    def get_model_metrics(self, model_name: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Get performance metrics for a model.
        
        Args:
            model_name: Name of the model.
            timeout: Request timeout in seconds.
        
        Returns:
            Dict[str, Any]: Model performance metrics.
        """
        try:
            # Run a simple generation to get metrics
            params = GenerationParameters(
                model=model_name,
                prompt="Hello",
                num_predict=1,
                temperature=0
            )
            
            start_time = time.time()
            response = self.generate(params, timeout=timeout)
            elapsed_time = time.time() - start_time
            
            # Extract and return metrics
            return {
                "model": model_name,
                "startup_time": elapsed_time,
                "first_token_time": elapsed_time,
                "tokens_per_second": response["metrics"]["tokens_per_second"] if "metrics" in response else 0,
                "memory_usage": response.get("memory_used", 0) if "memory_used" in response else 0
            }
        except Exception as e:
            logger.error(f"Failed to get metrics for model {model_name}: {e}")
            raise


# Example usage
def test_ollama_client():
    """Test the Ollama client with a simple generation."""
    client = OllamaClient()
    
    # List available models
    models = client.list_models()
    print(f"Available models: {len(models)}")
    for model in models:
        print(f"  - {model['name']}")
    
    # Test generation with Qwen model if available
    qwen_models = [m for m in models if "qwen" in m["name"].lower()]
    if qwen_models:
        model_name = qwen_models[0]["name"]
        print(f"\nTesting generation with {model_name}...")
        
        params = GenerationParameters(
            model=model_name,
            prompt="Explain how test-time compute optimization works in LLMs in one paragraph.",
            temperature=0.7,
            num_predict=100
        )
        
        try:
            result = client.generate(params)
            print("\nGeneration result:")
            print(f"Response: {result['response']}")
            print(f"Generated in {result['metrics']['inference_time']:.2f} seconds")
            print(f"Tokens per second: {result['metrics']['tokens_per_second']:.2f}")
        except Exception as e:
            print(f"Generation test failed: {e}")
    else:
        print("\nNo Qwen models found for testing.")


if __name__ == "__main__":
    # This will only run when the file is executed directly
    test_ollama_client()