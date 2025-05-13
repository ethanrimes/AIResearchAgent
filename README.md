# AI Research Agent

This repository contains a comprehensive system for AI research automation, utilizing a sequence of specialized agents to design, implement, and evaluate experiments aimed at improving large language model (LLM) reasoning capabilities. The system automates the entire research workflow from hypothesis generation to results evaluation.

## Overview

The AI Research Agent orchestrates a multi-stage workflow where each stage is handled by a specialized agent:

1. **Hypothesis Agent**: Generates novel, testable hypotheses for improving LLM reasoning on mathematical problems through test-time techniques
2. **Planner Agent**: Creates a detailed experimental design and implementation plan with file structure, code skeletons, and build sequence
3. **Experiment Agent**: Implements code and runs experiments based on the plan, with iterative refinement based on feedback
4. **Critique Agent**: Reviews code quality, validates implementation against requirements, and tracks build progress
5. **Evaluator Agent**: Analyzes experimental results and derives insights about which strategies were most effective and why

Each agent works with a specific set of tools and constraints, forming a comprehensive research pipeline that can run autonomously.

## Tech Stack

The system is built on a modern AI and data science stack:

- **LLM Integration**:
  - **OpenAI**: Powers the agent's reasoning (default: `o4-mini`)
  - **Ollama**: Provides locally-hosted model inference for experiments (default: `qwen3:0.6b`)

- **Core Frameworks**:
  - **LangChain**: Orchestrates the agent workflow and tool interactions
  - **Pydantic**: Handles data validation and schema definitions
  - **FastAPI**: Powers the API layer for service integration
  - **PyTorch**: Supports model operations (when needed)

- **Development Tools**:
  - **Pyright**: Performs type checking for code quality
  - **Docker**: Provides containerization and deployment
  - **pytest**: Supports testing infrastructure

- **Math and Data Processing**:
  - **NumPy/Pandas**: Data manipulation and processing
  - **SymPy**: Symbolic mathematics for answer verification
  - **math-verify**: Mathematical answer validation
  - **latex2sympy2_extended**: LaTeX parsing for mathematical expressions

- **Infrastructure**:
  - **Docker Compose**: Manages service orchestration
  - **YAML**: Configuration management
  - **Logging**: Comprehensive logging infrastructure

## Project Structure

The project is organized into several key directories:

- `core/`: Contains the main agent logic, tools, and utilities
  - `agent/`: Agent implementation, configuration, and orchestration
    - `agent.py`: Main orchestration class that manages the workflow
    - `agent_config.py`: Centralized configuration manager
    - `chains/`: Specialized chain implementations for each agent type
    - `tools/`: Implementations of tools for file operations, bash execution, etc.
    - `schemas/`: Pydantic schemas for tool inputs/outputs and agent outputs
  - `config/`: Configuration files and settings
    - `default_config.yaml`: Primary configuration file with model selections and constraints
  - `directives/`: Agent prompts and instructions for each stage
  - `scripts/`: Entry points and utility scripts
  - `utils/`: Utility functions for experiment validation, logging, etc.
- `data/`: Contains datasets used for experiments (e.g., mathematical problems)
- `infra/`: Docker and infrastructure setup
  - `docker/`: Dockerfile, entrypoint script, and test files
  - `docker-compose.yml`: Service configuration for running the container
- `sandbox/`: Contains read-only dependencies and writable runtime directories
  - `read_only/`: Immutable API implementations that experiments must use
    - `dataloader/`: Dataset loading utilities
    - `models/`: Model client for Ollama interaction
    - `evaluator/`: Answer extraction and evaluation tools
  - `writable/`: Runtime directories for experiment execution

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Access to an Ollama server (default: `http://host.docker.internal:11434`)
- OpenAI API key (for the agent's decision-making)
- Sufficient disk space for Docker images and experiment outputs

### Environment Setup

1. Create a `.env` file in the root directory with the following variables:
   ```
   OPENAI_API_KEY=your_openai_key
   OLLAMA_BASE_URL=http://host.docker.internal:11434
   EXPERIMENT_TIMEOUT=6000  # Optional: Override the default timeout
   ```

2. Ensure your Ollama server is running with the required models (default: qwen3:0.6b)

3. Check that the dataset specified in `default_config.yaml` exists in the data directory

### Running an Experiment

Execute the start_experiment.sh script to begin a new experiment run:

```bash
./core/scripts/start_experiment.sh
```

This script will:
1. Create a timestamped run directory with all necessary subdirectories
2. Launch a Docker container with the appropriate volume mounts
3. Initialize the AI Research Agent
4. Execute each stage of the research workflow sequentially

To rebuild the Docker image before running (useful after code changes):

```bash
./core/scripts/start_experiment.sh build
```

## Personalization Options

The system offers several ways to customize its behavior:

### Configuration Customization

Edit `core/config/default_config.yaml` to modify:

- **Models**:
  ```yaml
  agent_model: o4-mini    # Change to your preferred OpenAI model
  reasoner_model: qwen3:0.6b    # Change to any model available in your Ollama server
  ```

- **Dataset**:
  ```yaml
  dataset_path: data/rstar/aime2024_test.json    # Choose a different dataset
  max_questions: 10    # Limit the number of questions processed (-1 for all)
  ```

- **Runtime Behavior**:
  ```yaml
  memory_window_size: 5    # Adjust context memory size (0 for unlimited)
  max_coding_iters: 10    # Limit maximum coding iterations
  max_run_duration: 3600    # Adjust maximum runtime in seconds
  bash_timeout: 1800    # Set timeout for bash command execution
  ```

- **Ollama Parameters**:
  ```yaml
  batch_size: 2    # Adjust concurrent processing batch size
  max_num_predict: 1024    # Set maximum tokens to generate per request
  ```

### Directive Customization

Modify agent behavior by editing the YAML files in `core/directives/`:

- Adjust the `role`, `task`, and `instructions` sections to change agent behavior
- Modify the `constraints` section to limit or expand agent capabilities
- Create new directive sets by copying and modifying the existing ones

### Hypothesis Types

Customize the research focus by directing the Hypothesis Agent toward specific techniques:

- **Chain-of-thought variations**: Examining different prompt structures
- **Verification strategies**: Testing self-critique methods
- **Step-by-step reasoning**: Breaking down complex problems
- **Multiple solution paths**: Exploring alternative approaches
- **Structured output formats**: Testing different answer formats

### Dataset Selection

Choose different mathematical datasets to focus on specific problem types:

- AIME: Advanced mathematical competition problems
- GSM8K: Grade school math word problems
- MATH: Diverse mathematical problems across disciplines
- AMC: American Mathematics Competition problems
- Custom datasets: Add your own JSON datasets following the schema

## Configuration Details

The system is configured via `core/config/default_config.yaml`, which includes:

```yaml
## System Configuration
agent_model: o4-mini    # Model used by the agent for reasoning and planning
reasoner_model: qwen3:0.6b    # Model used for solving math problems
directives_path: core/directives/test_time_compute_directives    # Path to agent directives

## Dataset Configuration
dataset_path: data/rstar/aime2024_test.json    # Path to the dataset file
max_questions: -1    # Maximum number of questions to load (-1 for all questions)

## Runtime Constraints
memory_window_size: 0    # Size of memory window for context (0 for unlimited)
max_coding_iters: 20    # Maximum number of coding iterations
max_run_duration: 6000    # Maximum runtime in seconds
bash_timeout: 6000    # Timeout for bash commands in seconds

## Ollama Generation Configuration
batch_size: 4    # Number of prompts to process concurrently in each batch
max_num_predict: 2048    # Maximum number of tokens to generate per request
```

## Experiment Workflow in Detail

1. **Initialization**:
   - The script creates a timestamped run directory with subdirectories for each stage
   - The Docker container is launched with appropriate volume mounts
   - The entrypoint script performs connectivity and permission checks

2. **Hypothesis Generation**:
   - The Hypothesis Agent generates testable hypotheses for improving LLM reasoning
   - Outputs structured JSON to `/hypothesis/hypotheses.json`

3. **Experiment Planning**:
   - The Planner Agent reads the hypotheses
   - Creates a detailed implementation plan with file structure, code skeletons, and build sequence
   - Outputs structured JSON to `/planning/experiment_plan.json`

4. **Experiment Implementation**:
   - The Experiment Agent implements the code according to the plan
   - Creates all necessary files in `/experiment/`
   - Creates the entry script (`run_experiment.sh`) for running the experiment
   - Iteratively refines the code based on validation and critique feedback
   - Runs the experiment with the Ollama model and dataset
   - Writes results to `/results/results_summary.json`

5. **Code Critique**:
   - The Critique Agent runs between iterations of the Experiment Agent
   - Reviews code quality, validates implementation, and tracks build progress
   - Provides structured feedback for improvement

6. **Results Evaluation**:
   - The Evaluator Agent analyzes the experiment results
   - Compares hypothesis performance against the baseline
   - Identifies patterns, insights, and future directions
   - Outputs structured findings to `/findings/final_findings.json`

## Key Components in Detail

### Tool Schemas (`tool_schemas.py`)
Defines structured interfaces for:
- File operations: Reading, writing, updating, and deleting files
- Bash execution: Running shell commands with timeout
- Python code execution: Running code snippets for testing

### Experiment Utilities (`experiment_utils.py`)
Provides utilities for:
- Code retrieval with file annotations
- Experiment output validation
- Memory management for agent iterations
- Type checking with Pyright
- Experiment prompt formatting
- Validation result logging

### API Documentation (`api.md`)
Defines interfaces for:
- `OllamaClient`: Wrapper for Ollama server with text generation capabilities
- `rStarDataLoader`: Loader for math-problem datasets stored as JSON
- `AnswerEvaluator`: Tools to extract and score answers produced by an LLM

### Output Schemas (`output_schemas.py`)
Defines structured output formats for each agent:
- `HypothesisOutput`: Hypothesis definitions
- `PlannerOutput`: Experiment plan and implementation details
- `CritiqueOutput`: Code review results and build progress
- `EvaluatorOutput`: Findings and insights from experiment results

## Output Structure

The system generates several outputs in the run directory:

- `/experiment/`: Contains all implemented code
  - `run_experiment.sh`: Entry script for running the experiment
  - Source code files implementing baseline and hypotheses
- `/findings/`: Contains the final analysis and insights
  - `final_findings.json`: Structured evaluation of hypothesis effectiveness
- `/results/`: Contains experimental results
  - `results_summary.json`: Structured results with metrics for each hypothesis
  - `run_summary.json`: Overall execution summary
- `/logs/`: Contains detailed execution logs
  - `experiment_logs_*.log`: Timestamped experiment execution logs
  - `container.log`: Container output log
- `/planning/`: Contains the implementation plan
  - `experiment_plan.json`: Detailed plan with file structure and build sequence
- `/hypothesis/`: Contains the generated hypotheses
  - `hypotheses.json`: Structured hypothesis definitions

## Implementation Notes

- The system uses a flattened directory structure within the Docker container for simplified path handling
- Type checking is performed using Pyright to ensure code quality
- The environment has restricted permissions for security (read-only rootfs, limited volume mounts)
- The agents communicate through structured JSON outputs using Pydantic models
- Performance metrics include accuracy, execution time, and reasoning traces
- The experiment is designed to test test-time compute techniques without requiring model retraining
- The Docker container runs with minimal privileges and isolation

## Docker Configuration

The `docker-compose.yml` file configures:
- Volume mounts for code, data, and experiment directories
- Environment variables for API access and configuration
- Network configuration for accessing the Ollama server
- Security constraints (read-only filesystem, no new privileges)
- Resource limitations for stable execution

## Troubleshooting

- **Connection Issues**: Check the logs directory for connectivity problems with Ollama or OpenAI
  - Verify that the Ollama server is running and accessible
  - Check that the OpenAI API key is valid and has sufficient quota

- **Execution Problems**: Review container logs for detailed error messages
  - Ensure volume mounts are correct in docker-compose.yml
  - Check that the dataset path in the configuration exists

- **Timeout Issues**: Adjust timeouts in the configuration
  - Increase `max_run_duration` for longer experiments
  - Increase `bash_timeout` for complex experiment execution

- **Memory Issues**: Check for resource constraints
  - The Docker container may need more memory for large datasets
  - Adjust batch size in the configuration for balance

## Examples

The system can be used to test various LLM reasoning improvement strategies, such as:
- Stepwise verification of intermediate reasoning steps
- Structured reasoning formats to improve clarity
- Self-critique and correction mechanisms
- Budget forcing and stop tokens
- Least-to-most decomposition of complex problems

Each experiment produces a detailed analysis of which strategies performed best and why, contributing to the broader understanding of LLM reasoning capabilities.

This project demonstrates a fully autonomous research workflow from hypothesis generation to evaluation, with minimal human intervention required, showcasing the potential of AI systems to accelerate scientific discovery.