"""
Immutable client for querying an Ollama server running on the host.

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
from enum import Enum

# ---------------------------------------------------------
#  Logging
# ---------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
#  Enumerations & Dataclasses (unchanged except docstrings)
# ---------------------------------------------------------
class OllamaEndpoints(Enum):
    GENERATE    = "/api/generate"
    CHAT        = "/api/chat"
    EMBEDDINGS  = "/api/embeddings"
    TAGS        = "/api/tags"
    SHOW        = "/api/show"
    PULL        = "/api/pull"
    PUSH        = "/api/push"
    CREATE      = "/api/create"
    DELETE      = "/api/delete"
    COPY        = "/api/copy"


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
        if self.stream:                 d["stream"]   = True
        if self.raw:                    d["raw"]      = True
        if self.format is not None:     d["format"]   = self.format

        # collect generation-options
        base_fields = set(d.keys()) | {"model", "prompt"}
        options = {k: v for k, v in self.__dict__.items()
                   if k not in base_fields and v is not None}
        if options:
            d["options"] = options
        return d


@dataclass
class ChatMessage:
    role: str
    content: str
    images: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {"role": self.role, "content": self.content}
        if self.images:
            d["images"] = self.images
        return d


@dataclass
class ChatParameters:
    model: str
    messages: List[ChatMessage]
    stream: bool = False
    format: Optional[str] = None
    options: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "messages": [m.to_dict() for m in self.messages],
        }
        if self.stream:   d["stream"] = True
        if self.format:   d["format"] = self.format
        if self.options:  d["options"] = self.options
        return d


@dataclass
class EmbeddingParameters:
    model: str
    prompt: str
    options: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {"prompt": self.prompt}
        if self.options:
            d["options"] = self.options
        return d


# ---------------------------------------------------------
#  Main Client
# ---------------------------------------------------------
class OllamaClient:
    """
    Immutable Ollama API client.

    After construction the base URL and the default model **cannot** be
    modified from the outside, and every request sent through this client
    will automatically contain these enforced defaults.
    """

    # ---------- construction & immutability helpers ----------
    def __init__(self,
                 base_url: Optional[str] = None,
                 default_model: Optional[str] = None) -> None:
        object.__setattr__(self, "_base_url",
                           base_url or os.getenv("OLLAMA_BASE_URL",
                                                 "http://host.docker.internal:11434"))
        object.__setattr__(self, "_default_model",
                           default_model or os.getenv("OLLAMA_DEFAULT_MODEL"))
        if not self._default_model:
            raise ValueError("A default model must be supplied via argument or "
                             "OLLAMA_DEFAULT_MODEL environment variable.")
        object.__setattr__(self, "_session", requests.Session())

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
    def _get_url(self, endpoint: Union[str, OllamaEndpoints]) -> str:
        path = endpoint.value if isinstance(endpoint, OllamaEndpoints) else endpoint
        return f"{self._base_url}{path}"

    def _check_connection(self) -> None:
        try:
            r = self._session.get(self._get_url(OllamaEndpoints.TAGS), timeout=5)
            r.raise_for_status()
            logger.info("Connected to Ollama server at %s", self._base_url)
        except requests.RequestException as exc:
            raise ConnectionError(f"Unable to connect to Ollama at {self._base_url}") from exc

    def _enforce_defaults(self,
                          params: Union[GenerationParameters,
                                        ChatParameters,
                                        EmbeddingParameters,
                                        Dict[str, Any]]) -> Dict[str, Any]:
        """
        Convert `params` to a dict and inject the enforced default model.
        """
        if hasattr(params, "to_dict"):
            data = params.to_dict()            # dataclass instance
        else:
            data = dict(params)               # shallow copy of user dict
        data["model"] = self._default_model   # hard-overwrite
        return data

    def _make_request(self,
                      method: str,
                      endpoint: OllamaEndpoints,
                      data: Optional[Dict[str, Any]] = None,
                      *,
                      stream: bool = False,
                      timeout: Optional[float] = None
                      ) -> Union[requests.Response, Dict[str, Any]]:
        url = self._get_url(endpoint)

        if method == "GET":
            resp = self._session.get(url, timeout=timeout)
        elif method == "POST":
            if stream:
                return self._session.post(url, json=data, stream=True, timeout=timeout)
            resp = self._session.post(url, json=data, timeout=timeout)
        elif method == "DELETE":
            resp = self._session.delete(url, json=data, timeout=timeout)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

        resp.raise_for_status()
        return resp.json() if resp.content and not stream else resp

    # ---------- public high-level API ----------
    # Text generation --------------------------------------------------
    def generate(self,
                 params: Union[GenerationParameters, Dict[str, Any]],
                 timeout: Optional[float] = None) -> Dict[str, Any]:
        data = self._enforce_defaults(params)

        if data.get("stream"):
            # redirect to streaming helper
            return {   # wrap generator in dict for symmetric API
                "stream": self.generate_stream(data, timeout=timeout)
            }

        t0 = time.time()
        resp = self._make_request("POST", OllamaEndpoints.GENERATE, data,
                                  timeout=timeout)
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
        data = self._enforce_defaults(params)
        data["stream"] = True

        resp = self._make_request("POST", OllamaEndpoints.GENERATE, data,
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

    # Chat --------------------------------------------------------------
    def chat(self,
             params: Union[ChatParameters, Dict[str, Any]],
             timeout: Optional[float] = None) -> Dict[str, Any]:
        data = self._enforce_defaults(params)

        if data.get("stream"):
            return {"stream": self.chat_stream(data, timeout=timeout)}

        t0 = time.time()
        resp = self._make_request("POST", OllamaEndpoints.CHAT, data,
                                  timeout=timeout)
        elapsed = time.time() - t0
        content = resp.get("message", {}).get("content", "")
        resp["metrics"] = {
            "inference_time": elapsed,
            "tokens_per_second": len(content.split()) / elapsed if elapsed else 0
        }
        return resp

    def chat_stream(self,
                    params: Union[ChatParameters, Dict[str, Any]],
                    timeout: Optional[float] = None
                    ) -> Generator[Dict[str, Any], None, None]:
        data = self._enforce_defaults(params)
        data["stream"] = True

        resp = self._make_request("POST", OllamaEndpoints.CHAT, data,
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

    # Embeddings --------------------------------------------------------
    def get_embeddings(self,
                       params: Union[EmbeddingParameters, Dict[str, Any]],
                       timeout: Optional[float] = None) -> Dict[str, Any]:
        data = self._enforce_defaults(params)
        return self._make_request("POST", OllamaEndpoints.EMBEDDINGS,
                                  data, timeout=timeout)

    # Misc --------------------------------------------------------------
    def show_model_info(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Show metadata for the default model."""
        data = {"name": self._default_model}
        return self._make_request("POST", OllamaEndpoints.SHOW,
                                  data, timeout=timeout)

    def get_model_metrics(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Quick performance probe for the default model.
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