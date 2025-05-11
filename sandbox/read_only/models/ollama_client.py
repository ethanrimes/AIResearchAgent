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
import time
import logging
from typing import Dict, List, Any, Optional, Union, Generator

import requests
from dataclasses import dataclass

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
    stream: bool = False
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

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "prompt": self.prompt,          # model will be injected later
        }
        if self.system is not None:     d["system"]   = self.system
        if self.template is not None:   d["template"] = self.template
        if self.context is not None:    d["context"]  = self.context
        if self.stream:                 d["stream"]   = True
        if self.raw:                    d["raw"]      = True
        if self.format is not None:     d["format"]   = self.format

        # collect generation-options (excluding stream which should be top-level)
        base_fields = set(d.keys()) | {"model", "prompt", "stream", "raw"}
        options = {k: v for k, v in self.__dict__.items()
                   if k not in base_fields and v is not None}
        if options:
            d["options"] = options
        return d


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
        object.__setattr__(self, "_base_url",
                        os.getenv("OLLAMA_BASE_URL",
                                                "http://host.docker.internal:11434"))
        object.__setattr__(self, "_default_model",
                        os.getenv("OLLAMA_DEFAULT_MODEL"))
        if not self._default_model:
            raise ValueError("A default model must be supplied via "
                            "OLLAMA_DEFAULT_MODEL environment variable.")
        object.__setattr__(self, "_session", requests.Session())
        
        # Create logger before freezing the object
        object.__setattr__(self, "logger", logging.getLogger("ai_research_agent.ollama_client"))

        # immediately verify connectivity
        self._check_connection()

        # *freeze* – disallow changes to critical attrs after this point
        object.__setattr__(self, "_immutable", True)

    def __setattr__(self, name: str, value: Any) -> None:
        # prevent external mutation of critical fields once initialised
        if getattr(self, "_immutable", False) and name in {
            "_base_url", "_default_model", "base_url", "default_model"
        }:
            raise AttributeError(f"{name} is read-only after initialisation.")
        super().__setattr__(name, value)

    # read-only public accessors
    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def default_model(self) -> str:
        return self._default_model

    # ---------- internal helpers ----------
    def _get_url(self, endpoint: str) -> str:
        return f"{self._base_url}{endpoint}"

    def _check_connection(self) -> None:
        try:
            # Using a simple GET to check if server is responsive
            r = self._session.get(f"{self._base_url}/api/tags", timeout=5)
            r.raise_for_status()
            self.logger.info("Connected to Ollama server at %s", self._base_url)
        except requests.RequestException as exc:
            raise ConnectionError(f"Unable to connect to Ollama at {self._base_url}") from exc

    def _enforce_defaults(self, params: Union[GenerationParameters, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Convert `params` to a dict and inject the enforced default model.
        """
        if hasattr(params, "to_dict"):
            data = params.to_dict()            # dataclass instance
        else:
            data = dict(params)               # shallow copy of user dict
        data["model"] = self._default_model   # hard-overwrite
        return data

    def _make_request(self, endpoint: str, data: Dict[str, Any], *, stream: bool = False, timeout: Optional[float] = None) -> Union[requests.Response, Dict[str, Any]]:
        url = self._get_url(endpoint)

        # Double-check stream setting between function param and data to ensure consistency
        if "stream" in data:
            is_streaming = data["stream"]
        else:
            is_streaming = stream

        if is_streaming:
            return self._session.post(url, json=data, stream=True, timeout=timeout)
        
        # Explicitly ensure stream is set to False in data for non-streaming requests
        data_copy = data.copy()
        data_copy["stream"] = False
        
        # For non-streaming requests, use requests.post directly with the URL
        resp = requests.post(url, json=data_copy, timeout=timeout)
        resp.raise_for_status()
        
        if not resp.content:
            return resp

        # Handle different response formats based on content type
        content_type = resp.headers.get('Content-Type', '')
        
        # If we're getting ndjson even though we requested non-streaming
        if 'application/x-ndjson' in content_type:
            # Combine all the chunks into a single response
            try:
                lines = resp.text.strip().split("\n")
                if not lines:
                    return {"response": ""}
                
                # Extract and concatenate all responses
                full_response = ""
                for line in lines:
                    try:
                        chunk = json.loads(line)
                        if "response" in chunk:
                            full_response += chunk["response"]
                    except json.JSONDecodeError:
                        continue
                
                # Get metadata from the last chunk if possible
                try:
                    last_chunk = json.loads(lines[-1])
                    result = {**last_chunk, "response": full_response}
                    return result
                except (json.JSONDecodeError, IndexError):
                    return {"response": full_response}
            except Exception as e:
                self.logger.error(f"Error processing NDJSON response: {e}")
                return {"response": resp.text}
        
        # Standard JSON response processing
        try:
            return resp.json()
        except json.JSONDecodeError:
            return {"response": resp.text}

    # ---------- public API ----------
    def generate(self,
                 params: Union[GenerationParameters, Dict[str, Any]],
                 timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Generate text using the default model.
        
        Args:
            params: Generation parameters (the model will be overridden)
            timeout: Request timeout in seconds
            
        Returns:
            Response dictionary containing the generated text and metrics
        """
        data = self._enforce_defaults(params)

        # Ensure stream parameter is at the top level, not inside options
        is_streaming = False
        if isinstance(params, GenerationParameters):
            is_streaming = params.stream
        elif isinstance(params, dict):
            is_streaming = params.get("stream", False)
        
        # Override stream setting in data
        if "options" in data and "stream" in data["options"]:
            # If stream was in options, move it to top level
            data["stream"] = data["options"].pop("stream")
            # Remove options if empty after popping stream
            if not data["options"]:
                data.pop("options")
        
        if is_streaming or data.get("stream", False):
            # Set stream flag directly in data
            data["stream"] = True
            self.logger.info("Request set to streaming mode")
            # Redirect to streaming helper
            return {   # wrap generator in dict for symmetric API
                "stream": self.generate_stream(data, timeout=timeout)
            }
        
        # Ensure stream is explicitly set to False for non-streaming requests
        data["stream"] = False
        self.logger.info("Request set to non-streaming mode")

        t0 = time.time()
        resp = self._make_request(self.GENERATE_ENDPOINT, data, stream=False, timeout=timeout)
        elapsed = time.time() - t0
        resp["metrics"] = {
            "inference_time": elapsed,
            "tokens_per_second": len(resp.get("response", "").split()) / elapsed
                                if elapsed else 0
        }
        return resp

    def generate_stream(self,
                        params: Union[GenerationParameters, Dict[str, Any]],
                        timeout: Optional[float] = None
                        ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream text generation using the default model.
        
        Args:
            params: Generation parameters (the model will be overridden)
            timeout: Request timeout in seconds
            
        Returns:
            Generator yielding response chunks as they become available
        """
        data = self._enforce_defaults(params)
        data["stream"] = True

        resp = self._make_request(self.GENERATE_ENDPOINT, data,
                                 stream=True, timeout=timeout)

        t0 = time.time()
        tokens = 0
        for line in resp.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            tokens += 1
            elapsed = time.time() - t0
            chunk["metrics"] = {
                "inference_time": elapsed,
                "tokens_per_second": tokens / elapsed if elapsed else 0
            }
            yield chunk
            if chunk.get("done"):
                break

    def get_model_metrics(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Quick performance probe for the default model.
        
        Returns:
            Dictionary with performance metrics
        """
        gen_params = GenerationParameters(
            model=self._default_model,
            prompt="Hello",
            num_predict=1,
            temperature=0
        )
        t0 = time.time()
        resp = self.generate(gen_params, timeout=timeout)
        elapsed = time.time() - t0
        return {
            "model": self._default_model,
            "startup_time": elapsed,
            "first_token_time": elapsed,
            "tokens_per_second": resp["metrics"]["tokens_per_second"],
            "memory_usage": resp.get("memory_used", 0)
        }