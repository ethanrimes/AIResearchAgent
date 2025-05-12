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
from core.utils.experiment_utils import run_pyright_type_check, format_experiment_prompt, log_validation_results

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.schema import SystemMessage
from langchain.memory import ConversationBufferWindowMemory

# ----------------------------------------------------------------------
# Custom tools
# ----------------------------------------------------------------------
from core.agent.tools.read_file import ReadFileTool
from core.agent.tools.write_file import WriteFileTool
from core.agent.tools.update_file import UpdateFileTool
from core.agent.tools.delete_file import DeleteFileTool
from core.agent.tools.list_files import ListFilesTool
from core.agent.tools.run_bash import RunBashTool
from core.agent.tools.run_code import RunCodeTool
from core.utils.path_utils import get_active_run_path
from core.utils.experiment_utils import get_current_experiment_code, validate_experiment_output

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
        memory_window_size: int = 0,
        bash_timeout: int = 600,
    ):
        self.llm = llm
        self.workspace = workspace
        self.directive = directive
        self.max_iterations = max_iterations
        self.active_run_path = str(workspace["writable"])
        self.critique_chain = critique_chain  # Store the critique chain
        self.memory_window_size = memory_window_size  # Store the memory window size
        self.bash_timeout = bash_timeout  # Store the bash timeout

        # Initialize memory with the configured window size
        # If memory_window_size is 0, this will still work but won't retain any history
        self.memory = ConversationBufferWindowMemory(
            k=self.memory_window_size,
            return_messages=True,
            memory_key="chat_history",
            input_key="input",
            output_key="output"
        )

        # Short-hand directories using root paths
        self.exp_dir = Path("/experiment")
        self.results_dir = Path("/results")
        self.entry_script_path = self.exp_dir / self.ENTRY_SCRIPT
        self.scratchpad_file = Path("/scratchpad/experiment_notes.txt")

        self.logger = logging.getLogger("ai_research_agent.experiment_chain")

        # Ensure experiment directory exists (entry script will be created by agent)
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # Initialise tool set + agent
        self.tools = self._init_tools()
        self.agent = self._init_agent()

        self.logger.info(
            "ExperimentChain initialised (max_iterations=%s, critique_chain=%s, memory_window_size=%s, bash_timeout=%s)", 
            self.max_iterations, 
            "available" if self.critique_chain else "not available",
            self.memory_window_size,
            self.bash_timeout
        )

    # ------------------------------------------------------------------
    # Tooling
    # ------------------------------------------------------------------
    def _init_tools(self) -> List[BaseTool]:
        """Create tool instances and keep a reference to RunBash for later."""
        read_tool = ReadFileTool(
            allowed_dirs=[
                "/read_only",
                "/hypothesis",
                "/planning",
                "/experiment",
            ]
        )

        # Write, Update, and Delete tools only have access to experiment directory
        write_tool = WriteFileTool(
            allowed_dirs=["/experiment"]
        )
        
        update_tool = UpdateFileTool(
            allowed_dirs=["/experiment"]
        )
        
        delete_tool = DeleteFileTool(
            allowed_dirs=["/experiment"]
        )
        
        # List Files has access to read-only and experiment directories
        list_files_tool = ListFilesTool(
            allowed_dirs=[
                "/read_only",
                "/hypothesis",
                "/planning",
                "/experiment",
            ]
        )

        self.run_bash_tool = RunBashTool(
            working_dir="/experiment", 
            output_dir="/results",
            timeout=self.bash_timeout
        )

        run_code_tool = RunCodeTool()

        return [
            read_tool,
            write_tool,
            update_tool,
            delete_tool,
            list_files_tool,
            self.run_bash_tool,
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
        
        # Updated prompt template with system message and chat history
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
            verbose=True,
            memory=self.memory,  # Add the memory to the agent executor
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
        api_doc_file = Path("/read_only/api.md")

        # Track the state of validation
        validation_passed = False
        agent_output = None
        bash_output = ""
        critique_results = None
        previous_output = None
        error_information = None
        
        # Initialize current iteration counter
        current_iteration = 0

        # ------------------------------------------------------------------
        # Single consolidated while loop for invocation and validation
        # ------------------------------------------------------------------
        while current_iteration < self.max_iterations and not validation_passed:
            self.logger.info(f"Starting iteration {current_iteration + 1}/{self.max_iterations}")
            
            # ------------------------------------------------------------------
            # 1) Prepare structured input for current iteration using the utility function
            # ------------------------------------------------------------------
            # Run type checking
            type_check_results = run_pyright_type_check(self.exp_dir)
            self.logger.info(f"VALIDATION: Type checking results summary: {type_check_results['error_count']} errors found")
            
            # Run critique chain if available
            if self.critique_chain and current_iteration > 0:
                try:
                    self.logger.info("Running critique chain for code review")
                    critique_results = self.critique_chain.run()
                    self.logger.info("Critique chain completed successfully")
                except Exception as e:
                    self.logger.error(f"Error running critique chain: {e}")
                    critique_results = {"error": f"Failed to run critique: {str(e)}"}
            
            # Get validation results for iterations > 0
            validation_results = None
            if current_iteration > 0:
                validation_results = validate_experiment_output(
                    self.exp_dir, 
                    self.results_dir, 
                    self.ENTRY_SCRIPT,
                    run_bash_tool=self.run_bash_tool
                )
                # Log validation results with clear prefacing
                log_validation_results(validation_results, current_iteration-1)
                
                # Store bash output for reporting
                bash_output = validation_results.get("bash_output", "")
            
            # Format the prompt using the utility function
            input_content = format_experiment_prompt(
                iteration=current_iteration,
                directive=self.directive,
                experiment_dir=self.exp_dir,
                entry_script=self.ENTRY_SCRIPT,
                plan_data=plan_data,
                previous_output=previous_output,
                error_information=error_information,
                type_check_results=type_check_results,
                critique_results=critique_results,
                validation_results=validation_results
            )
            
            # Prepare input for agent
            current_input = {
                "input": input_content
            }
            
            # ------------------------------------------------------------------
            # 2) Invoke agent with try/catch and process result
            # ------------------------------------------------------------------
            error_information = None
            try:
                # Invoke the agent with current input
                agent_output = self.agent.invoke(current_input)
                # Save the output for next iteration
                previous_output = agent_output.get("output", "")
            except Exception as e:
                self.logger.error(f"Error during agent invocation: {e}")
                error_information = f"Error during execution: {str(e)}"
                previous_output = f"Failed to complete iteration {current_iteration} due to error: {str(e)}"
            
            # ------------------------------------------------------------------
            # 3) Run validation to check if the implementation is complete
            # ------------------------------------------------------------------
            validation_results = validate_experiment_output(
                self.exp_dir, 
                self.results_dir, 
                self.ENTRY_SCRIPT,
                run_bash_tool=self.run_bash_tool
            )
            
            # Log validation results with clear prefacing
            log_validation_results(validation_results, current_iteration)
            
            # Update bash_output from validation results
            bash_output = validation_results.get("bash_output", "")
            
            # All three conditions must be true for validation to pass
            if all([
                validation_results["entry_script_exists"],
                validation_results["python_files_exist"],
                validation_results["results_file_exists"]
            ]):
                self.logger.info("VALIDATION: Final validation PASSED - all checks successful")
                validation_passed = True
            else:
                self.logger.warning("VALIDATION: Final validation FAILED - continuing to next iteration")
            
            # Increment iteration counter
            current_iteration += 1

        # ------------------------------------------------------------------
        # 4) Finalize results
        # ------------------------------------------------------------------
        # Check if we exited due to max iterations
        if not validation_passed:
            self.logger.warning("Maximum iterations reached without successful validation")
        
        # Final validation check for the entry script
        if not self.entry_script_path.exists():
            msg = f"Expected entry script not found: {self.entry_script_path}"
            self.logger.error(msg)
            raise FileNotFoundError(msg)

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
            self.logger.info("VALIDATION: Successfully parsed results_summary.json")
            self.logger.info("VALIDATION: Results summary contains %d keys: %s", 
                           len(results_dict), ", ".join(results_dict.keys()))
        except Exception as e:
            self.logger.error("VALIDATION: Failed to parse results JSON: %s", e)
            raise

        self.logger.info("=== ExperimentChain finished successfully ===")
        
        # Include critique results in the final output
        final_output = {
            "summary": agent_output.get("output", "") if agent_output else "",
            "bash_output": bash_output,
            "results": results_dict,
        }
        
        if critique_results:
            final_output["critique"] = critique_results
            
        return final_output