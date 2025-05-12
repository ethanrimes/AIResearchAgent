# API Interface Documentation

## 1. Overview
This package contains three utility modules that together form a tiny "LLM + Math-QA" pipeline:

| Module | Purpose | Key Class |
| ------ | ------- | --------- |
| `ollama_client.py` | Immutable wrapper around an Ollama server for text generation with batching support. | `OllamaClient` |
| `rstar_data_loader.py` | Minimal loader for math-problem datasets stored as JSON. | `rStarDataLoader` |
| `answer_evaluation.py` | Tools to extract and score answers produced by an LLM. | `AnswerEvaluator` |

Both `OllamaClient` and `rStarDataLoader` are designed to be **configuration-driven**. Core configuration parameters like model name, server URL, and dataset path are managed internally and cannot be directly modified when instantiating these classes.

## 2. Import Instructions

To import these modules from the `/read_only` directory in your experiment code:

```python
# Now you can import the modules directly
from dataloader.rstar_data_loader import rStarDataLoader
from models.ollama_client import OllamaClient, GenerationParameters
from evaluator.answer_evaluation import AnswerEvaluator
```

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
    raw: bool = False
    format: str | None = None
    # generation options
    num_predict: int | None = None     # Will be capped by client's max_num_predict
    temperature: float | None = None
    top_k: int | None = None
    # ... see source for the full list
```

`to_dict()` converts the instance to a payload compatible with Ollama's
`/api/generate` endpoint while **omitting** the `model` field (the client injects its enforced default later).

---

#### 3.1.2 Class `OllamaClient`

```python
client = OllamaClient()
```

Constructor behavior:
* Gets the model name from internal configuration
* Gets the batch size and max_num_predict from internal configuration
* Uses `OLLAMA_BASE_URL` environment variable (default: `"http://host.docker.internal:11434"`) for the base URL
* Immediately performs `GET <base_url>/api/tags` to ensure the server is alive
* After construction the attributes `_base_url`, `_default_model`, `_batch_size`, and `_max_num_predict` become **read-only**; any attempt to overwrite them raises `AttributeError`

##### Properties
| Property | Type | Description |
| -------- | ---- | ----------- |
| `base_url` | `str` | Read-only full base URL. |
| `default_model` | `str` | Read-only model name from configuration. |
| `batch_size` | `int` | Read-only batch size from configuration (default: 4). |
| `max_num_predict` | `int` | Read-only max token count from configuration (default: 2048). |

##### Public Methods
| Method | Return | Notes |
| ------ | ------ | ----- |
| `generate(params, timeout=None)` | `dict` or `list[dict]` | Synchronous generation. Handles both single params and lists of params (batching). |
| `generate_async(params, timeout=None)` | `list[dict]` | Asynchronous generation with batching support. Returns a list of response dictionaries. |
| `get_model_metrics(timeout=None)` | `dict` | Lightweight "hello world" ping that measures startup and token speed for the default model. |

##### Batching Behavior
- When `params` is a list, the client automatically processes them in batches of size `batch_size`
- Each batch is processed concurrently using async methods
- The client enforces `max_num_predict` tokens for all requests
- If a single parameter is provided, a single response dict is returned
- If a list of parameters is provided, a list of response dicts is returned

##### Timing & Speed Metrics
All responses include:
```json
"metrics": {
  "inference_time": 1.234,      // seconds until completion
  "tokens_per_second": 8.76,    // throughput estimate
  "total_tokens": 42            // count of tokens in the response
}
```

##### Example: Single Request
```python
from models.ollama_client import OllamaClient, GenerationParameters

client = OllamaClient()              # Uses model from internal configuration
params = GenerationParameters(
    model="ignored",                 # Value is ignored and replaced with configured model
    prompt="Solve 2+2",
    temperature=0
)

resp = client.generate(params)
print(resp["response"])              # 4
```

##### Example: Batch Processing
```python
from models.ollama_client import OllamaClient, GenerationParameters

client = OllamaClient()              # Uses model, batch_size, and max_num_predict from internal configuration

# Create a list of parameter objects
params_list = [
    GenerationParameters(model="ignored", prompt="Solve 2+2"),
    GenerationParameters(model="ignored", prompt="Calculate the area of a circle with radius 5"),
    GenerationParameters(model="ignored", prompt="Factor x^2 - 4")
]

# Process all parameters in batches of size client.batch_size
responses = client.generate(params_list)

# Print all responses
for i, resp in enumerate(responses):
    print(f"Problem {i+1}: {resp['response']}")
    print(f"Time: {resp['metrics']['inference_time']:.2f}s, "
          f"Speed: {resp['metrics']['tokens_per_second']:.2f} tokens/s")
```

##### Example: Asynchronous Batch Processing
```python
import asyncio
from models.ollama_client import OllamaClient, GenerationParameters

async def process_math_problems():
    client = OllamaClient()
    
    params_list = [
        GenerationParameters(model="ignored", prompt="Solve 2+2"),
        GenerationParameters(model="ignored", prompt="Calculate the area of a circle with radius 5")
    ]
    
    # Process all parameters asynchronously
    responses = await client.generate_async(params_list)
    
    for i, resp in enumerate(responses):
        print(f"Problem {i+1}: {resp['response']}")

# Run the async function
asyncio.run(process_math_problems())
```

---

### 3.2 `rstar_data_loader.py`

#### Class `rStarDataLoader`

```python
loader = rStarDataLoader()
problems: list[dict] = loader.load()
```

* Gets the dataset path from internal configuration
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
from dataloader.rstar_data_loader import rStarDataLoader

# Configuration should be set up before creating the loader
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

*Note: Unlike the other modules, `AnswerEvaluator` does not depend on internal configuration.*

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
from evaluator.answer_evaluation import AnswerEvaluator

gen  = "We get \\boxed{\\frac{1}{2}} as the probability."
gold = "\\frac{1}{2}"

ev = AnswerEvaluator()
ans, ok = ev.evaluate(gen, gold)
print(ans)          # -> \frac{1}{2}
print(ok)           # -> True
```

---

## 4. Putting It All Together

### Example: Processing a Dataset with Batching

```python
from models.ollama_client import OllamaClient, GenerationParameters
from dataloader.rstar_data_loader import rStarDataLoader
from evaluator.answer_evaluation import AnswerEvaluator

# 1. Load problems
problems = rStarDataLoader().load()

# 2. Create client & evaluator
client = OllamaClient()
evaluator = AnswerEvaluator()

# 3. Prepare batch parameters
batch_params = []
for p in problems[:20]:  # Process first 20 problems
    params = GenerationParameters(
        model="ignored",  # Will be replaced by client
        prompt=p["question"],
        temperature=0
    )
    batch_params.append(params)

# 4. Generate all answers in batches (client handles batching internally)
responses = client.generate(batch_params)

# 5. Evaluate all responses
results = []
for i, (problem, response) in enumerate(zip(problems[:20], responses)):
    extracted, correct = evaluator.evaluate(response["response"], problem["answer"])
    
    results.append({
        "id": problem["id"],
        "question": problem["question"],
        "ground_truth": problem["answer"],
        "model_answer": response["response"],
        "extracted_answer": extracted,
        "correct": correct,
        "inference_time": response["metrics"]["inference_time"],
        "tokens_per_second": response["metrics"]["tokens_per_second"]
    })
    
    print(f"{problem['id']}: Correct={correct}, Time={response['metrics']['inference_time']:.2f}s")

# 6. Calculate summary metrics
correct_count = sum(1 for r in results if r["correct"])
accuracy = correct_count / len(results)
avg_inference_time = sum(r["inference_time"] for r in results) / len(results)

print(f"Accuracy: {accuracy:.2%}")
print(f"Average inference time: {avg_inference_time:.2f}s")
```

### Example: Asynchronous Processing

```python
import asyncio

from models.ollama_client import OllamaClient, GenerationParameters
from dataloader.rstar_data_loader import rStarDataLoader
from evaluator.answer_evaluation import AnswerEvaluator

async def process_dataset():
    # 1. Load problems
    problems = rStarDataLoader().load()[:20]  # First 20 problems
    
    # 2. Create client & evaluator
    client = OllamaClient()
    evaluator = AnswerEvaluator()
    
    # 3. Prepare parameters
    batch_params = [
        GenerationParameters(model="ignored", prompt=p["question"], temperature=0)
        for p in problems
    ]
    
    # 4. Generate all answers asynchronously (client handles batching)
    responses = await client.generate_async(batch_params)
    
    # 5. Evaluate and report
    for i, (problem, response) in enumerate(zip(problems, responses)):
        extracted, correct = evaluator.evaluate(response["response"], problem["answer"])
        print(f"{i+1}. {problem['id']}: {'✓' if correct else '✗'} {extracted}")

# Run the async function
if __name__ == "__main__":
    asyncio.run(process_dataset())
```

---

## 5. Error Handling & Logging

* All modules log through the package-level loggers  
  (`ai_research_agent.ollama_client`, `ai_research_agent.rstar_data_loader`, …).  
  Configure `logging` in your entry-script as needed.
* Network errors raise `ConnectionError` from `OllamaClient._check_connection`.  
* For batched requests, individual failures are captured in the response object with an `"error"` field.
* Dataset issues are swallowed and reported as log errors, returning `[]`.  
* `AnswerEvaluator` catches parser/verification exceptions and records them
  in the returned dictionary under `"<method>_error"` keys.