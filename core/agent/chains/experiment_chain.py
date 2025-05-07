"""
Experiment Chain module for the AI Research Agent.

This chain turns the high-level plan into an executable project, makes the
agent populate a bash launcher script, then executes that script itself.

Workflow enforced here
----------------------
1. The agent is instructed to create *all* source files plus a bash script
   called `run_experiment.sh` inside `{ACTIVE_RUN_PATH}/experiment`.
2. After the agent stops tool calls and emits its plain-text summary,
   this chain immediately executes that bash script via `RunBashTool`.
3. The bash script must write `results_summary.json` to
   `{ACTIVE_RUN_PATH}/results` following the evaluator-directive schema.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Any

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import ChatMessageHistory
from langchain.schema import SystemMessage

# ----------------------------------------------------------------------
# Custom tools
# ----------------------------------------------------------------------
from core.agent.tools.read_file import ReadFileTool
from core.agent.tools.write_file import WriteFileTool
from core.agent.tools.run_bash import RunBashTool
from core.agent.tools.run_code import RunCodeTool
from core.utils.path_utils import get_active_run_path
from core.utils.experiment_utils import get_current_experiment_code, validate_experiment_output, update_memory_with_steps

# ----------------------------------------------------------------------
class ExperimentChain:
    """
    Implements, runs, and verifies the experiments.
    """

    ENTRY_SCRIPT = "run_experiment.sh"

    # ------------------------------------------------------------------
    def __init__(
        self,
        llm,
        workspace: Dict[str, Path],
        directive: str,
        max_iterations: int = 10,
        critique_chain = None,
    ):
        self.llm = llm
        self.workspace = workspace
        self.directive = directive
        self.max_iterations = max_iterations
        self.active_run_path = str(workspace["writable"])
        self.critique_chain = critique_chain  # Store the critique chain

        # Short-hand directories
        self.exp_dir = workspace["experiment"]
        self.results_dir = workspace["results"]
        self.entry_script_path = self.exp_dir / self.ENTRY_SCRIPT
        self.scratchpad_file = workspace["scratchpad"] / "experiment_notes.txt"

        self.logger = logging.getLogger("ai_research_agent.experiment_chain")

        # Ensure experiment directory exists (entry script will be created by agent)
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # Initialize memory object for conversation history
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="input"  # Explicitly tell memory which key to use
        )

        # Add a counter to track total steps across iterations
        self.total_steps = 0
        self.steps_in_memory = 0

        # Initialise tool set + agent
        self.tools = self._init_tools()
        self.agent = self._init_agent()

        self.logger.info(
            "ExperimentChain initialised (max_iterations=%s, critique_chain=%s)", 
            self.max_iterations, 
            "available" if self.critique_chain else "not available"
        )

    # ------------------------------------------------------------------
    # Tooling
    # ------------------------------------------------------------------
    def _init_tools(self) -> List[BaseTool]:
        """Create tool instances and keep a reference to RunBash for later."""
        read_tool = ReadFileTool(
            allowed_dirs=[
                str(self.workspace["read_only"]),
                str(self.workspace["hypothesis"]),
                str(self.workspace["planning"]),
                str(self.workspace["experiment"]),
            ]
        )

        write_tool = WriteFileTool(
            allowed_dirs=[str(self.exp_dir)]
        )

        self.run_bash_tool = RunBashTool(
            working_dir=str(self.exp_dir), output_dir=str(self.results_dir)
        )

        run_code_tool = RunCodeTool()

        return [
            read_tool,
            write_tool,
            self.run_bash_tool,  # keep reference
            run_code_tool,
        ]

    # ------------------------------------------------------------------
    # Agent
    # ------------------------------------------------------------------
    def _init_agent(self) -> AgentExecutor:
        """Structured-chat agent that does the implementation work."""
        # Create a system message that includes the ACTIVE_RUN_PATH
        system_message = (
            f"You are an AI research assistant implementing experiments based on a plan. "
            # f"The ACTIVE_RUN_PATH for this session is: {self.active_run_path}"
        )
        
        # Updated prompt template with system message containing ACTIVE_RUN_PATH
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("ai", "{agent_scratchpad}")
        ])

        agent = create_openai_tools_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt,
        )

        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,  # Use memory at the executor level
            verbose=True,
            return_intermediate_steps=True,
            handle_parsing_errors=True,
            # allow think+act pairs
            max_iterations=self.max_iterations * 2,
        )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def run(self, plan_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the experiment chain.

        Returns:
            dict  {
            "summary": <plain-text returned by agent>,
            "bash_output": <stdout/stderr from script run>,
            "results": <dict parsed from results_summary.json>
            }
        """
        self.logger.info("=== ExperimentChain starting ===")

        # Paths the agent must read
        plan_file = self.workspace["planning"] / "experiment_plan.json"
        results_json = self.results_dir / "results_summary.json"
        hypothesis_file = self.workspace["hypothesis"] / "hypotheses.json"
        api_doc_file = Path("/app/sandbox/read_only/api.md")

        # Track the state of validation
        validation_passed = False
        agent_output = None
        bash_output = ""
        critique_results = None
        
        # Initialize current iteration counter
        current_iteration = 0

        # ------------------------------------------------------------------
        # Single consolidated while loop for invocation and validation
        # ------------------------------------------------------------------
        while current_iteration < self.max_iterations and not validation_passed:
            self.logger.info(f"Starting iteration {current_iteration + 1}/{self.max_iterations}")
            
            # ------------------------------------------------------------------
            # 1) Prepare structured input for current iteration
            # ------------------------------------------------------------------
            if current_iteration == 0:
                # For initial input, include the full directive and context
                
                # Get the file tree
                read_only_files = [
                    "/app/sandbox/read_only/dataloader/rstar_data_loader.py",
                    "/app/sandbox/read_only/evaluator/answer_evaluation.py",
                    "/app/sandbox/read_only/example/code/baseline.py",
                    "/app/sandbox/read_only/example/code/evaluation.py",
                    "/app/sandbox/read_only/example/code/experiment_runner.py", 
                    "/app/sandbox/read_only/example/code/structured_approach.py",
                    "/app/sandbox/read_only/models/ollama_client.py",
                    "/app/sandbox/read_only/api.md",
                    "/app/sandbox/writable/run_dir/hypothesis/hypotheses.json",
                    "/app/sandbox/writable/run_dir/planning/experiment_plan.json"
                ]
                
                writable_dirs = [
                    f"/app/sandbox/writable/{self.active_run_path}/experiment/**",
                    # f"/app/sandbox/writable/{self.active_run_path}/results/**"
                ]
                
                # Read experiment plan
                experiment_plan = json.loads(plan_file.read_text()) if plan_file.exists() else {}
                
                # Read hypotheses
                hypotheses = json.loads(hypothesis_file.read_text()) if hypothesis_file.exists() else {}
                
                # Read API documentation
                api_docs = api_doc_file.read_text() if api_doc_file.exists() else ""
                
                # Get current experiment code (initially empty)
                experiment_code = get_current_experiment_code(self.exp_dir)
                
                input_content = (
                    f"# 1. Directive\n{self.directive}\n\n"
                    f"- Write your experiment entrypoint to the bash file:\n"
                    f"  `{self.entry_script_path}`\n"
                    "- Running the script must create "
                    f"`{self.results_dir}/results_summary.json`. "
                    f"`{self.results_dir} cannot be directly written to through the write tool. It is for code output only."
                    "conforming to the evaluator-directive schema."
                    f"# 2. File Tree\n## Read-only files\n{chr(10).join(read_only_files)}\n\n"
                    f"## Writable directories\n{chr(10).join(writable_dirs)}\n\n"
                    f"# 3. Experiment Plan\n```json\n{json.dumps(experiment_plan, indent=2)}\n```\n\n"
                    f"# 4. Hypotheses\n```json\n{json.dumps(hypotheses, indent=2)}\n```\n\n"
                    f"# 5. API Documentation\n{api_docs}\n\n"
                    f"# 6. Current Experiment Code\n{experiment_code}\n\n"
                    f"# 7. Code Analysis\nNo code analysis available for initial implementation.\n\n"
                    f"# 8. Custom Checks\n- Entry script exists: No\n"
                    f"- Python files in experiment directory: No\n"
                    f"- results_summary.json exists: No\n\n"
                    f"Implement the experiment, write the entrypoint bash script, "
                    f"execute & iterate until the JSON results file is produced.\n"
                    f"Finally output your plain-text summary."
                )
            else:
                # For subsequent iterations, focus on errors and current state
                
                # Get current experiment code
                experiment_code = get_current_experiment_code(self.exp_dir)
                
                # Perform validation checks
                validation_results = validate_experiment_output(self.exp_dir, self.results_dir, self.ENTRY_SCRIPT)
                
                # Try to execute the bash script if it exists
                if validation_results["entry_script_exists"]:
                    try:
                        bash_output = self.run_bash_tool._run(f"bash {self.entry_script_path}")
                        self.logger.info(f"Bash script execution output: {bash_output}")
                    except Exception as e:
                        self.logger.error(f"Error executing bash script: {e}")
                        validation_results["errors"].append(f"Error executing bash script: {str(e)}")
                
                # Run the critique chain to evaluate the current implementation
                if hasattr(self, "critique_chain") and self.critique_chain:
                    try:
                        self.logger.info("Running critique chain for code review")
                        critique_results = self.critique_chain.run()
                        self.logger.info("Critique chain completed successfully")
                    except Exception as e:
                        self.logger.error(f"Error running critique chain: {e}")
                        critique_results = {"error": f"Failed to run critique: {str(e)}"}
                
                # Format critique results for inclusion in prompt
                critique_section = ""
                if critique_results:
                    critique_section = f"# 7. Code Review Results\n```json\n{json.dumps(critique_results, indent=2)}\n```\n\n"
                
                # Format error message
                error_section = ""
                if validation_results["errors"]:
                    error_section = "The following errors were found in your experiment:\n- " + "\n- ".join(validation_results["errors"])
                    
                    if bash_output:
                        error_section += f"\n\nOutput from bash script execution:\n{bash_output}"
                else:
                    error_section = "All validation checks passed."
                
                input_content = (
                    f"# Iteration {current_iteration + 1} Feedback\n\n"
                    f"{error_section}\n\n"
                    f"# Current Experiment Code\n{experiment_code}\n\n"
                    f"{critique_section}"
                    f"# Custom Checks\n"
                    f"- Entry script exists: {validation_results['entry_script_exists']}\n"
                    f"- Python files in experiment directory: {validation_results['python_files_exist']}\n"
                    f"- results_summary.json exists: {validation_results['results_file_exists']}\n\n"
                    f"Please correct any issues and continue implementing the experiment."
                )
            
            # Prepare input for agent
            current_input = {
                "input": input_content
            }
            
            # ------------------------------------------------------------------
            # 2) Invoke agent and process result
            # ------------------------------------------------------------------
            # Invoke the agent with current input
            agent_output = self.agent.invoke(current_input)
            
            # Update memory with new steps
            if 'intermediate_steps' in agent_output:
                self.steps_in_memory = update_memory_with_steps(
                    self.memory, self.steps_in_memory, agent_output
                )
            
            # ------------------------------------------------------------------
            # 3) Run the critique chain after each iteration if available
            # ------------------------------------------------------------------
            if self.critique_chain:
                try:
                    self.logger.info("Running critique chain for code review")
                    critique_results = self.critique_chain.run()
                    self.logger.info("Critique chain executed successfully")
                except Exception as e:
                    self.logger.error(f"Error running critique chain: {e}")
                    critique_results = {"error": f"Failed to run critique: {str(e)}"}
            else:
                self.logger.info("No critique chain available, skipping code review")
            
            # ------------------------------------------------------------------
            # 4) Validate results
            # ------------------------------------------------------------------
            # Check if all validation conditions are met
            validation_results = validate_experiment_output(self.exp_dir, self.results_dir, self.ENTRY_SCRIPT)
            
            if all([
                validation_results["entry_script_exists"],
                validation_results["python_files_exist"],
                validation_results["results_file_exists"]
            ]):
                self.logger.info("Experiment validation passed, all checks successful")
                validation_passed = True
            else:
                self.logger.info("Validation failed, continuing to next iteration")
            
            # Increment iteration counter
            current_iteration += 1

        # ------------------------------------------------------------------
        # 5) Finalize results
        # ------------------------------------------------------------------
        # Check if we exited due to max iterations
        if not validation_passed:
            self.logger.warning("Maximum iterations reached without successful validation")
        
        # Final validation check for the entry script
        if not self.entry_script_path.exists():
            msg = f"Expected entry script not found: {self.entry_script_path}"
            self.logger.error(msg)
            raise FileNotFoundError(msg)

        # Execute the script one final time if validation passed
        if validation_passed:
            self.logger.info("Running final generated bash script via RunBashTool")
            bash_output = self.run_bash_tool._run(f"bash {self.entry_script_path}")

        # Load results JSON (should exist now after validation)
        if not results_json.exists():
            msg = (
                "Bash script completed but results_summary.json not found at "
                f"{results_json}"
            )
            self.logger.error(msg)
            raise FileNotFoundError(msg)

        try:
            results_dict: Dict[str, Any] = json.loads(results_json.read_text())
        except Exception as e:
            self.logger.error("Failed to parse results JSON: %s", e)
            raise

        self.logger.info("=== ExperimentChain finished successfully ===")
        
        # Include critique results in the final output
        final_output = {
            "summary": agent_output.get("output", ""),
            "bash_output": bash_output,
            "results": results_dict,
        }
        
        if critique_results:
            final_output["critique"] = critique_results
            
        return final_output