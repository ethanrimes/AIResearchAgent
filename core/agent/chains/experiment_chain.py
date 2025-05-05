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
    ):
        self.llm = llm
        self.workspace = workspace
        self.directive = directive
        self.max_iterations = max_iterations
        self.active_run_path = str(workspace["writable"])

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
            "ExperimentChain initialised (max_iterations=%s)", self.max_iterations
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
            f"The ACTIVE_RUN_PATH for this session is: {self.active_run_path}"
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

        # ------------------------------------------------------------------
        # 1) Let the agent do its work
        # ------------------------------------------------------------------
        system_requirements = (
            "## System-enforced requirements\n"
            f"- You have **{self.max_iterations} revision cycles**.\n"
            f"- Write your experiment entrypoint to the bash file:\n"
            f"  `{self.entry_script_path}`\n"
            f"- All code must live inside `{self.active_run_path}/experiment`.\n"
            "- Running the script must create "
            f"`{self.results_dir}/results_summary.json` "
            "conforming to the evaluator-directive schema."
        )
        
        initial_input = (
            f"{self.directive}\n\n"
            f"{system_requirements}\n\n"
            f"Read the experiment plan from: {plan_file}\n"
            f"You have {self.max_iterations} revision cycles.\n"
            "Implement the experiment, write the entrypoint bash script, "
            "execute & iterate until the JSON results file is produced.\n"
            "Finally output your plain-text summary."
        )
        
        # Only pass the input to invoke, no ACTIVE_RUN_PATH
        agent_input = {
            "input": initial_input
        }

        agent_output = self.agent.invoke(agent_input)
        self.logger.info(f"Agent has completed initial attempt")
        
        # Track steps from initial output
        if 'intermediate_steps' in agent_output:
            self.total_steps += len(agent_output['intermediate_steps'])
            # Add new steps to memory
            self.update_memory_with_steps(agent_output)
        
        # Validate the output and continue in a loop until validation passes or max iterations reached
        final_output = self.validate_experiment_output_and_continue(agent_output)
        
        # Log final steps
        self.logger.info(f"Agent total steps: {self.total_steps} across all iterations")

        # ------------------------------------------------------------------
        # 2) Execute the final bash script one more time to get the output
        # ------------------------------------------------------------------
        if not self.entry_script_path.exists():
            msg = f"Expected entry script not found: {self.entry_script_path}"
            self.logger.error(msg)
            raise FileNotFoundError(msg)

        self.logger.info("Running final generated bash script via RunBashTool")
        bash_output: str = self.run_bash_tool._run(f"bash {self.entry_script_path}")

        # ------------------------------------------------------------------
        # 3) Load results JSON (should exist now after validation)
        # ------------------------------------------------------------------
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
        return {
            "summary": final_output,
            "bash_output": bash_output,
            "results": results_dict,
        }
    
    def update_memory_with_steps(self, agent_output):
        """
        Updates the agent's memory with new intermediate steps.
        Converts tool actions and observations into SystemMessages.
        """
        if 'intermediate_steps' not in agent_output:
            return
            
        # Get all steps
        steps = agent_output["intermediate_steps"]
        
        # Determine which ones are new (not yet in memory)
        new_steps = steps[self.steps_in_memory:]
        
        # Add new steps to memory
        for action, observation in new_steps:
            msg = SystemMessage(
                content=f"Tool '{action.tool}' called with input: {action.tool_input}\n"
                        f"Tool output: {observation}"
            )
            self.memory.chat_memory.add_message(msg)
        
        # Update our memory counter
        self.steps_in_memory = len(steps)
    
    def validate_experiment_output_and_continue(self, agent_output):
        """
        Validates the experiment output and provides feedback to continue if errors are found.
        Runs in a loop until validation passes or max_iterations is reached.
        Uses Langchain memory to maintain conversation history.
        """
        import os
        from pathlib import Path
        
        # Define logger
        logger = logging.getLogger("ai_research_agent.experiment_chain")
        
        # Set up results json path
        results_json_path = self.results_dir / "results_summary.json"
        
        bash_output = ""
        
        # Loop until validation passes or max iterations reached
        while True:
            # Determine current iteration count based on total steps
            # Each think+act pair counts as 2 steps
            current_iterations = self.total_steps // 2
            
            logger.info(f"Validating experiment output - current iterations: {current_iterations}/{self.max_iterations}")
            
            # Check if we've already reached max iterations
            if current_iterations >= self.max_iterations:
                logger.warning("Maximum iterations reached, stopping validation loop")
                break
            
            # Check for the error conditions and collect them
            errors = []
            
            # 1. First try to execute the bash script if it exists
            entry_script_path = self.exp_dir / self.ENTRY_SCRIPT
            if entry_script_path.exists() and os.path.getsize(entry_script_path) > 0:
                logger.info("Bash script exists, attempting execution...")
                try:
                    bash_output = self.run_bash_tool._run(f"bash {entry_script_path}")
                    logger.info(f"Bash script execution output: {bash_output}")
                except Exception as e:
                    logger.error(f"Error executing bash script: {e}")
                    errors.append(f"Error executing bash script: {str(e)}")
            else:
                errors.append("No code found in run_experiment.sh or the file doesn't exist")
            
            # 2. Check if there are other Python files in the experiment directory
            python_files = list(self.exp_dir.glob("*.py"))
            if not python_files:
                errors.append("No Python files found in the experiment directory")
            
            # 3. Check if results_summary.json exists (this would be created by the bash script)
            if not results_json_path.exists():
                errors.append("results_summary.json not found in the results directory")
            
            # If all checks pass, break out of the loop
            if not errors:
                logger.info("Experiment validation passed, all checks successful")
                break
            
            # If errors found, provide feedback and continue
            error_message = "The following errors were found in your experiment:\n- " + "\n- ".join(errors)
            error_message += "\n\nPlease correct these issues and continue implementing the experiment."
            
            if bash_output:
                error_message += f"\n\nOutput from bash script execution:\n{bash_output}"
            
            # Only pass the input to invoke, no ACTIVE_RUN_PATH
            continue_input = {
                "input": error_message
            }
            
            logger.info(f"Errors found, continuing with feedback using Langchain memory")
            
            # Continue execution with the error feedback
            updated_output = self.agent.invoke(continue_input)
            
            # Update total steps count with new steps
            if 'intermediate_steps' in updated_output:
                # Calculate how many new steps were added
                new_steps = len(updated_output['intermediate_steps'])
                self.total_steps += new_steps
                
                # Update memory with new steps
                self.update_memory_with_steps(updated_output)
                
                logger.info(f"Added {new_steps} new steps, total steps: {self.total_steps}")
            
            # Update the agent_output for the next iteration
            agent_output = updated_output
        
        return agent_output