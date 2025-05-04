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

from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor, create_openai_tools_agent

# ----------------------------------------------------------------------
# Custom tools
# ----------------------------------------------------------------------
from core.agent.tools.read_file import ReadFileTool
from core.agent.tools.write_file import WriteFileTool
from core.agent.tools.read_scratchpad import ReadScratchpadTool
from core.agent.tools.write_to_scratchpad import WriteScratchpadTool
from core.agent.tools.run_bash import RunBashTool
from core.agent.tools.run_code import RunCodeTool

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

        # Short-hand directories
        self.exp_dir = workspace["experiment"]
        self.results_dir = workspace["results"]
        self.entry_script_path = self.exp_dir / self.ENTRY_SCRIPT
        self.scratchpad_file = workspace["scratchpad"] / "experiment_notes.txt"

        self.logger = logging.getLogger("ai_research_agent.experiment_chain")

        # Ensure experiment directory exists (entry script will be created by agent)
        self.exp_dir.mkdir(parents=True, exist_ok=True)

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
            ]
        )

        write_tool = WriteFileTool(
            allowed_dirs=[str(self.exp_dir)]
        )

        read_scratchpad_tool = ReadScratchpadTool(
            scratchpad_file=str(self.scratchpad_file)
        )
        write_scratchpad_tool = WriteScratchpadTool(
            scratchpad_file=str(self.scratchpad_file)
        )

        self.run_bash_tool = RunBashTool(
            working_dir=str(self.exp_dir), output_dir=str(self.results_dir)
        )

        run_code_tool = RunCodeTool()

        return [
            read_tool,
            write_tool,
            read_scratchpad_tool,
            write_scratchpad_tool,
            self.run_bash_tool,  # keep reference
            run_code_tool,
        ]

    # ------------------------------------------------------------------
    # Agent
    # ------------------------------------------------------------------
    def _init_agent(self) -> AgentExecutor:
        """Structured-chat agent that does the implementation work."""
        prompt = PromptTemplate.from_template(
            template=(
                f"{self.directive}\n\n"
                "## System-enforced requirements\n"
                f"- You have **{self.max_iterations} revision cycles**.\n"
                f"- Write your experiment entrypoint to the bash file:\n"
                f"  `{self.entry_script_path}`\n"
                "- All code must live inside `{ACTIVE_RUN_PATH}/experiment`.\n"
                "- Running the script must create "
                f"`{self.results_dir}/results_summary.json` "
                "conforming to the evaluator-directive schema.\n"
                "Tool names you can call: {tool_names}\n\n"
                "Full tool descriptions:\n{tools}\n\n"
                "{input}\n\n"
                "{agent_scratchpad}"
            )
        )

        agent = create_openai_tools_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt,
        )

        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
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
        agent_input = {
            "input": (
                f"Read the experiment plan from: {plan_file}\n"
                f"You have {self.max_iterations} revision cycles.\n"
                "Implement the experiment, write the entrypoint bash script, "
                "execute & iterate until the JSON results file is produced.\n"
                "Finally output your plain-text summary."
            ),
            "ACTIVE_RUN_PATH": str(self.workspace["writable"])
        }

        summary_text: str = self.agent.invoke(agent_input)

        # ------------------------------------------------------------------
        # 2) Execute the generated bash script
        # ------------------------------------------------------------------
        if not self.entry_script_path.exists():
            msg = f"Expected entry script not found: {self.entry_script_path}"
            self.logger.error(msg)
            raise FileNotFoundError(msg)

        self.logger.info("Running generated bash script via RunBashTool")
        bash_output: str = self.run_bash_tool._run(f"bash {self.entry_script_path}")

        # ------------------------------------------------------------------
        # 3) Validate & load results JSON
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
            "summary": summary_text,
            "bash_output": bash_output,
            "results": results_dict,
        }