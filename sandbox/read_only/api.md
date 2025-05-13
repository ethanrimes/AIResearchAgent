# API Interface Documentation

## 1. Overview
This package contains three utility modules that together form a tiny "LLM + Math-QA" pipeline:

| Module | Purpose | Key Class |
| ------ | ------- | --------- |
| `ollama_client.py` | Immutable wrapper around an Ollama server for text generation. | `OllamaClient` |
| `rstar_data_loader.py` | Minimal loader for math-problem datasets stored as JSON. | `rStarDataLoader` |
| `answer_evaluation.py` | Tools to extract and score answers produced by an LLM. | `AnswerEvaluator` |

Both `OllamaClient` and `rStarDataLoader` are designed to be **configuration-driven**. Core configuration parameters like model name, server URL, and dataset path are managed internally and cannot be directly modified when instantiating these classes.

## 2. Import Instructions

To import these modules from the `/read_only` directory in your experiment code:

```python
import sys; sys.path.append('/read_only')  # Force path in the module itself
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
    stream: bool = False  # This will be forced to False
    raw: bool = False
    format: str | None = None
    # generation options
    num_predict: int | None = 32768
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    # ... see source for the full list
```

`to_options_dict()` converts the instance to a payload compatible with Ollama's API endpoints while **omitting** the `model` field (the client injects its enforced default later).

---

#### 3.1.2 Class `OllamaClient`

```python
client = OllamaClient()
```

Constructor behavior:
* Gets the model name from internal configuration via `AgentConfig.reasoner_model()`
* Uses `OLLAMA_BASE_URL` environment variable (default: `"http://host.docker.internal:11434"`) for the base URL
* Immediately performs a connectivity check to ensure the server is alive
* After construction the attributes `_base_url` and `_default_model` become **read-only**; any attempt to overwrite them raises `AttributeError`

##### Properties
| Property | Type | Description |
| -------- | ---- | ----------- |
| `base_url` | `str` | Read-only full base URL. |
| `default_model` | `str` | Read-only model name from configuration. |

##### Public Methods
| Method | Return | Notes |
| ------ | ------ | ----- |
| `generate(prompts, params=None)` | `str` or `list[str]` | Text generation with one or more prompts. Returns string or list of strings. |
| `chat(messages, params=None)` | `str` or `list[str]` | Chat completion with one or more conversations. Returns string or list of strings. |
| `get_model_metrics()` | `dict` | Lightweight "hello world" ping that measures startup and token speed for the default model. |

##### Processing Behavior
- Both `generate` and `chat` methods support individual inputs or lists of inputs
- For single inputs, they return a single string output
- For multiple inputs (list), they return a list of string outputs
- Inputs are processed sequentially, one at a time
- Each operation is logged with detailed timing and token information
- The client enforces the use of its default model for all requests
- Streaming is always disabled (stream=False)

##### Example: Single Text Generation
```python
import sys; sys.path.append('/read_only')
from models.ollama_client import OllamaClient, GenerationParameters

client = OllamaClient()              # Uses model from internal configuration
params = GenerationParameters(
    model="ignored",                 # Value is ignored and replaced with configured model
    prompt="Solve 2+2",
    temperature=0
)

# Generate text with a single prompt
response = client.generate("Solve 2+2", params)
print(response)                      # The generated text as a string
```

##### Example: Multiple Text Generation
```python
import sys; sys.path.append('/read_only')
from models.ollama_client import OllamaClient, GenerationParameters

client = OllamaClient()              # Uses model from internal configuration

# Create parameters object
params = GenerationParameters(
    model="ignored",                 # Value is ignored and replaced with configured model
    temperature=0.0,
    num_predict=100
)

# Create a list of prompts
prompts = [
    "Solve 2+2",
    "Calculate the area of a circle with radius 5",
    "Factor x^2 - 4"
]

# Process all prompts sequentially
responses = client.generate(prompts, params)

# Print all responses
for i, response in enumerate(responses):
    print(f"Problem {i+1}: {response[:100]}...")
```

##### Example: Single Chat Completion
```python
import sys; sys.path.append('/read_only')
from models.ollama_client import OllamaClient, GenerationParameters

client = OllamaClient()              # Uses model from internal configuration
params = GenerationParameters(
    model="ignored",                 # Value is ignored and replaced with configured model
    prompt="",                       # Not used for chat
    temperature=0
)

# Create a chat conversation
conversation = [
    {"role": "system", "content": "You are a helpful math assistant."},
    {"role": "user", "content": "What is the formula for the area of a circle?"}
]

# Generate chat completion
response = client.chat(conversation, params)
print(response)                      # The generated chat response as a string
```

##### Example: Multiple Chat Completions
```python
import sys; sys.path.append('/read_only')
from models.ollama_client import OllamaClient, GenerationParameters

client = OllamaClient()              # Uses model from internal configuration

# Create parameters object
params = GenerationParameters(
    model="ignored",                 # Value is ignored and replaced with configured model
    temperature=0.0,
    num_predict=100
)

# Create a list of conversations
conversations = [
    [{"role": "user", "content": "What is the capital of France?"}],
    [{"role": "user", "content": "What is the chemical symbol for gold?"}]
]

# Process all conversations sequentially
responses = client.chat(conversations, params)

# Print all responses
for i, response in enumerate(responses):
    print(f"Conversation {i+1}: {response[:100]}...")
```

##### Example: Chat with Multi-turn Conversation
```python
import sys; sys.path.append('/read_only')
from models.ollama_client import OllamaClient, GenerationParameters

client = OllamaClient()              # Uses model from internal configuration
params = GenerationParameters(
    model="ignored",                 # Value is ignored and replaced with configured model
    prompt="",                       # Not used for chat
    temperature=0
)

# Create a multi-turn conversation
conversation = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "The answer is 4."},
    {"role": "user", "content": "Now multiply that by 3."}
]

# Generate chat completion
response = client.chat(conversation, params)
print(response)                      # The generated chat response as a string
```

##### Example: Model Performance Metrics
```python
import sys; sys.path.append('/read_only')
from models.ollama_client import OllamaClient

client = OllamaClient()              # Uses model from internal configuration

# Get model performance metrics
metrics = client.get_model_metrics()
print(f"Model: {metrics['model']}")
print(f"Startup time: {metrics['startup_time']:.2f}s")
print(f"First token time: {metrics['first_token_time']:.2f}s")
print(f"Tokens per second: {metrics['tokens_per_second']:.2f}")
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
import sys; sys.path.append('/read_only')
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
import sys; sys.path.append('/read_only')
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

### Example: Processing a Dataset with Generate

```python
import sys; sys.path.append('/read_only')
from models.ollama_client import OllamaClient, GenerationParameters
from dataloader.rstar_data_loader import rStarDataLoader
from evaluator.answer_evaluation import AnswerEvaluator
import time

# 1. Load problems
problems = rStarDataLoader().load()

# 2. Create client & evaluator
client = OllamaClient()
evaluator = AnswerEvaluator()

# 3. Prepare problem prompts
prompts = [p["question"] for p in problems[:20]]  # Process first 20 problems

# 4. Create generation parameters
params = GenerationParameters(
    model="ignored",  # Will be replaced by client
    prompt="",        # Will be replaced by each prompt
    temperature=0.0,
    num_predict=500   # Limit output length
)

# 5. Generate all answers (processed sequentially)
start_time = time.time()
responses = client.generate(prompts, params)
total_time = time.time() - start_time

# 6. Evaluate all responses
results = []
for i, (problem, response) in enumerate(zip(problems[:20], responses)):
    extracted, correct = evaluator.evaluate(response, problem["answer"])
    
    results.append({
        "id": problem["id"],
        "question": problem["question"],
        "ground_truth": problem["answer"],
        "model_answer": response,
        "extracted_answer": extracted,
        "correct": correct
    })
    
    print(f"{problem['id']}: Correct={correct}")

# 7. Calculate summary metrics
correct_count = sum(1 for r in results if r["correct"])
accuracy = correct_count / len(results)

print(f"Accuracy: {accuracy:.2%}")
print(f"Total processing time: {total_time:.2f}s")
```

---

## 5. Error Handling & Logging

* All modules log through the package-level loggers  
  (`ai_research_agent.ollama_client`, `ai_research_agent.rstar_data_loader`, …).  
  Configure `logging` in your entry-script as needed.
* Network errors raise `ConnectionError` from `OllamaClient._check_connection`.  
* `OllamaClient` logs detailed information about each response including timing and token counts
* Dataset issues are swallowed and reported as log errors, returning `[]`.  
* `AnswerEvaluator` catches parser/verification exceptions and records them
  in the returned dictionary under `"<method>_error"` keys.