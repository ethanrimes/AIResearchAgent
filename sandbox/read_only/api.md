# API Interface Documentation  

> Version: 0.1  
> Status: Draft  
> Language: English  

---

## 1. Overview  
This package contains three utility modules that together form a tiny “LLM + Math-QA” pipeline:

| Module | Purpose | Key Class |
| ------ | ------- | --------- |
| `ollama_client.py` | Immutable wrapper around an Ollama server for text generation. | `OllamaClient` |
| `rstar_data_loader.py` | Minimal loader for math-problem datasets stored as JSON. | `rStarDataLoader` |
| `answer_evaluation.py` | Tools to extract and score answers produced by an LLM. | `AnswerEvaluator` |

All configuration that can influence external resources (model, server URL, dataset path) **must** be supplied through environment variables; the user cannot change these at runtime through the code interface.

---

## 2. Environment Variables  

| Variable | Consumed by | Required | Default | Description |
| -------- | ----------- | -------- | ------- | ----------- |
| `OLLAMA_DEFAULT_MODEL` | `OllamaClient` | **Yes** | – | Name of the model the Ollama server should load. |
| `OLLAMA_BASE_URL` | `OllamaClient` | No | `http://host.docker.internal:11434` | Root URL of the running Ollama server. |
| `DATASET_PATH` | `rStarDataLoader` | **Yes** | – | Absolute or relative path to the dataset JSON file to load. |

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

`to_dict()` converts the instance to a payload compatible with Ollama’s
`/api/generate` endpoint while **omitting** the `model` field (the client injects its enforced default later).

---

#### 3.1.2 Class `OllamaClient`

```python
client = OllamaClient(base_url: str | None = None)
```

Constructor behaviour  
* `base_url` – optional; overrides `OLLAMA_BASE_URL`.  
* Immediately performs `GET <base_url>/api/tags` to ensure the server is alive.  
* After construction the attributes `_base_url` and `_default_model` become **read-only**; any attempt to overwrite them raises `AttributeError`.

##### Properties
| Property | Type | Description |
| -------- | ---- | ----------- |
| `base_url` | `str` | Read-only full base URL. |
| `default_model` | `str` | Read-only enforced model name. |

##### Public Methods
| Method | Return | Notes |
| ------ | ------ | ----- |
| `generate(params, timeout=None)` | `dict` | Synchronous generation. If `params.stream=True` it returns `{"stream": <generator>}` for symmetry. |
| `generate_stream(params, timeout=None)` | `Generator[dict, None, None]` | Low-level streaming API. Yields JSON chunks exactly as Ollama emits them, with extra `metrics` fields injected. |
| `get_model_metrics(timeout=None)` | `dict` | Lightweight “hello world” ping that measures startup and token speed for the default model. |

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

client = OllamaClient()              # Uses env vars
params = GenerationParameters(
    model="ignored",                 # any placeholder
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

* Reads the file at `DATASET_PATH` (expects a JSON list of objects).
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

* Logs up to three sample items for quick inspection.
* Returns an empty list on any exception.

##### Example
```python
import os, logging
os.environ["DATASET_PATH"] = "math_dataset.json"

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

##### Core API
| Method | Purpose |
| ------ | ------- |
| `evaluate(generation, ground_truth)` | Returns `(extracted_answer, bool_is_correct)`. |
| `evaluate_batch(generations, ground_truths)` | Vectorised wrapper over `evaluate`. |

##### Internals
`evaluate_correctness()` applies **four independent checks**:

1. Exact string equality  
2. `math_verify` symbolic comparison (if the optional dependency is importable)  
3. “MathD” answer-normalisation equality  
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
from ollama_client import OllamaClient, GenerationParameters
from rstar_data_loader import rStarDataLoader
from answer_evaluation import AnswerEvaluator

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

