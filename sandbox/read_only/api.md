# API Interface Documentation  

## 1. Overview  
This package contains three utility modules that together form a tiny "LLM + Math-QA" pipeline:

| Module | Purpose | Key Class |
| ------ | ------- | --------- |
| `ollama_client.py` | Immutable wrapper around an Ollama server for text generation. | `OllamaClient` |
| `rstar_data_loader.py` | Minimal loader for math-problem datasets stored as JSON. | `rStarDataLoader` |
| `answer_evaluation.py` | Tools to extract and score answers produced by an LLM. | `AnswerEvaluator` |

Both `OllamaClient` and `rStarDataLoader` are designed to be **configuration-driven**. All configuration parameters that influence external resources (model name, server URL, dataset path) are managed through a centralized `AgentConfig` class singleton and cannot be directly specified when instantiating these classes. **IMPORTANT: Client code using these classes cannot override the model name, dataset path, or other core configuration values once they are set in the `AgentConfig` class.**

## Import Instructions

To import these modules from the `/read_only` directory in your experiment code:

```python
import sys; sys.path.append('/read_only')

# Now you can import the modules directly
from dataloader.rstar_data_loader import rStarDataLoader
from models.ollama_client import OllamaClient, GenerationParameters
from evaluator.answer_evaluation import AnswerEvaluator
```

---

## 2. Configuration System

The system uses a centralized configuration pattern where critical settings are managed through the `AgentConfig` class singleton:

```python
from core.agent.agent_config import AgentConfig

# Load configuration (should be done early in the application lifecycle)
# This only needs to be called once in the main application entry point
config_dict = AgentConfig.load_config("path/to/config.yaml")

# Later, access configuration properties using class methods
model_name = AgentConfig.reasoner_model()
dataset_path = AgentConfig.dataset_path()

# You can also lock the configuration to prevent further modifications
AgentConfig.lock()
```

**Important**: 
- `AgentConfig` is implemented as a singleton with class methods, not instance methods.
- Clients using these modules cannot directly set the model name or dataset path at instantiation time. 
- These values must be configured through the `AgentConfig` system before creating instances.
- Once the configuration is loaded and locked, it cannot be modified by client code.

---

## 3. Module Reference  

### 3.1 `ollama_client.py`  

#### 3.1.1 Dataclass `GenerationParameters`

```python
@dataclass
class GenerationParameters:
    model: str            # Will be ignored & replaced by the client.
    prompt: str
    system: str | None = None
    template: str | None = None
    context: list[int] | None = None
    stream: bool = False
    raw: bool = False
    format: str | None = None
    # dozens of generation-options …
    temperature: float | None = None
    top_k: int | None = None
    # … see source for the full list
```

`to_dict()` converts the instance to a payload compatible with Ollama's
`/api/generate` endpoint while **omitting** the `model` field (the client injects its enforced default later).

---

#### 3.1.2 Class `OllamaClient`

```python
client = OllamaClient()
```

Constructor behavior:
* Gets the model name from `AgentConfig.reasoner_model()`
* Uses `OLLAMA_BASE_URL` environment variable (default: `"http://host.docker.internal:11434"`) for the base URL
* Immediately performs `GET <base_url>/api/tags` to ensure the server is alive
* After construction the attributes `_base_url` and `_default_model` become **read-only**; any attempt to overwrite them raises `AttributeError`

##### Properties
| Property | Type | Description |
| -------- | ---- | ----------- |
| `base_url` | `str` | Read-only full base URL. |
| `default_model` | `str` | Read-only model name from configuration. |

##### Public Methods
| Method | Return | Notes |
| ------ | ------ | ----- |
| `generate(params, timeout=None)` | `dict` | Synchronous generation. If `params.stream=True` it returns `{"stream": <generator>}` for symmetry. |
| `generate_stream(params, timeout=None)` | `Generator[dict, None, None]` | Low-level streaming API. Yields JSON chunks exactly as Ollama emits them, with extra `metrics` fields injected. |
| `get_model_metrics(timeout=None)` | `dict` | Lightweight "hello world" ping that measures startup and token speed for the default model. |

##### Timing & Speed Metrics
Both `generate()` and `generate_stream()` add:
```json
"metrics": {
  "inference_time": 1.234,      # seconds until completion / current chunk
  "tokens_per_second": 8.76     # running throughput estimate
}
```

##### Example
```python
from ollama_client import OllamaClient, GenerationParameters

# Note: AgentConfig should be configured before creating the client
client = OllamaClient()              # Uses model from AgentConfig.reasoner_model()
params = GenerationParameters(
    model="ignored",                 # Value is ignored and replaced with AgentConfig.reasoner_model()
    prompt="Solve 2+2",
    temperature=0
)

resp = client.generate(params)
print(resp["response"])              # 4
```

---

### 3.2 `rstar_data_loader.py`

#### Class `rStarDataLoader`

```python
loader = rStarDataLoader()
problems: list[dict] = loader.load()
```

* Gets the dataset path from `AgentConfig.dataset_path()`
* Reads the file at the configured dataset path (expects a JSON list of objects)
* Converts each item to the canonical schema:

```json
{
  "id":        "<string>",
  "question":  "<problem text>",
  "answer":    "<ground-truth answer>",
  "solution":  "",
  "url":       ""
}
```

* Logs up to three sample items for quick inspection
* Returns an empty list on any exception

##### Properties
| Property | Type | Description |
| -------- | ---- | ----------- |
| `dataset_path` | `str` | Read-only path to the dataset from configuration. |

##### Example
```python
from rstar_data_loader import rStarDataLoader
from core.agent.agent_config import AgentConfig

# Configure dataset path in the centralized config
# This should be done before creating the loader
AgentConfig.load_config("path/to/config.yaml")

loader = rStarDataLoader()
dataset = loader.load()
first = dataset[0]
print(first["question"], " => ", first["answer"])
```

---

### 3.3 `answer_evaluation.py`

#### Class `AnswerEvaluator`

```python
evaluator = AnswerEvaluator()
extracted, correct = evaluator.evaluate(generation, ground_truth)
```

*Note: Unlike the other modules, `AnswerEvaluator` does not depend on `AgentConfig`.*

##### Core API
| Method | Purpose |
| ------ | ------- |
| `evaluate(generation, ground_truth)` | Returns `(extracted_answer, bool_is_correct)`. |
| `evaluate_batch(generations, ground_truths)` | Vectorised wrapper over `evaluate`. |

##### Internals
`evaluate_correctness()` applies **four independent checks**:

1. Exact string equality  
2. `math_verify` symbolic comparison (if the optional dependency is importable)  
3. "MathD" answer-normalisation equality  
4. SymPy algebraic equivalence  

The result dictionary contains boolean flags for each method and an aggregate
`any_method_correct`.

##### LaTeX & Text Handling
The evaluator contains an extensive suite of normalisation helpers that:

* Unbox `\boxed{…}` answers
* Canonicalise fractions, square-roots, mixed numbers, units, etc.
* Convert LaTeX to plain text before passing to SymPy
* Detect and handle tuples / intervals  

These utilities are all private (`_`-prefixed) but can be reused if desired.

##### Quick Demo
```python
from answer_evaluation import AnswerEvaluator

gen  = "We get \\boxed{\\frac{1}{2}} as the probability."
gold = "\\frac{1}{2}"

ev = AnswerEvaluator()
ans, ok = ev.evaluate(gen, gold)
print(ans)          # -> \frac{1}{2}
print(ok)           # -> True
```

---

## 4. Putting It All Together

```python
import sys; sys.path.append('/read_only')

from models.ollama_client import OllamaClient, GenerationParameters
from dataloader.rstar_data_loader import rStarDataLoader
from evaluator.answer_evaluation import AnswerEvaluator
from core.agent.agent_config import AgentConfig

# 0. Ensure configuration is properly set up
# This should be done once at the start of your program
config_dict = AgentConfig.load_config("/path/to/config.yaml")

# 1. Load problems
problems = rStarDataLoader().load()

# 2. Create client & evaluator
client = OllamaClient()
evaluator = AnswerEvaluator()

# 3. Iterate
for p in problems[:5]:
    params = GenerationParameters(model="ignored", prompt=p["question"], stream=False)
    gen_resp = client.generate(params)
    model_answer = gen_resp["response"]

    extracted, correct = evaluator.evaluate(model_answer, p["answer"])
    print(f"{p['id']}: {correct=}  extracted={extracted}")
```

---

## 5. Error Handling & Logging

* All modules log through the package-level loggers  
  (`ai_research_agent.ollama_client`, `ai_research_agent.rstar_data_loader`, …).  
  Configure `logging` in your entry-script as needed.
* Network errors raise `ConnectionError` from `OllamaClient._check_connection`.  
* Dataset issues are swallowed and reported as log errors, returning `[]`.  
* `AnswerEvaluator` catches parser/verification exceptions and records them
  in the returned dictionary under `"<method>_error"` keys.