"""
Pydantic output schemas for the agent chains.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# HypothesisChain
# ---------------------------------------------------------------------------

class Hypothesis(BaseModel):
    id: str = Field(..., description="Unique identifier, e.g. H1")
    name: str
    description: str
    rationale: str
    testability: str


class HypothesisOutput(BaseModel):
    hypotheses: List[Hypothesis]


# ---------------------------------------------------------------------------
# PlannerChain
# ---------------------------------------------------------------------------

class FileInfo(BaseModel):
    path: str = Field(..., description="Relative path to the file")
    purpose: str = Field(..., description="Brief description of the file's purpose")


class CodeSkeleton(BaseModel):
    file_path: str = Field(..., description="Path to the file")
    skeleton: str = Field(..., description="Code skeleton with TODOs")


class BuildStep(BaseModel):
    step: int = Field(..., description="Step number in the sequence")
    description: str = Field(..., description="Description of what to implement in this step")
    files: List[str] = Field(..., description="Files to work on in this step")


class ImplementationLocation(BaseModel):
    file: Optional[str] = Field(None, description="File containing the implementation")
    description: str = Field(..., description="How this component is implemented")


class HypothesisImplementation(BaseModel):
    id: str = Field(..., description="Hypothesis ID (e.g., H1)")
    files: List[str] = Field(..., description="Files implementing this hypothesis")
    description: str = Field(..., description="How this hypothesis is implemented")


class ImplementationLocations(BaseModel):
    baseline: ImplementationLocation
    entry_point: ImplementationLocation = Field(..., alias="entry_point")
    hypotheses: List[HypothesisImplementation]


class FileTree(BaseModel):
    description: str
    files: List[FileInfo]


class ImplementationPlan(BaseModel):
    file_tree: FileTree
    code_skeletons: List[CodeSkeleton]
    build_sequence: List[BuildStep]
    implementation_locations: ImplementationLocations = Field(..., alias="implementation_locations")


class Baseline(BaseModel):
    description: str
    implementation: str


class HypothesisTestPlan(BaseModel):
    hypothesis_id: str = Field(..., alias="hypothesis_id")
    implementation: str
    expected_outcome: str


class EvaluationSpec(BaseModel):
    metrics: List[str]
    methodology: str


class ExperimentPlan(BaseModel):
    title: str
    description: str
    baseline: Baseline
    hypotheses_testing: List[HypothesisTestPlan]
    evaluation: EvaluationSpec
    analysis_plan: str


class PlannerOutput(BaseModel):
    experiment_plan: ExperimentPlan
    implementation_plan: ImplementationPlan


# ---------------------------------------------------------------------------
# EvaluatorChain
# ---------------------------------------------------------------------------

class HypothesisResult(BaseModel):
    id: str
    name: str
    effectiveness: str
    performance_change: str
    analysis: str
    conditions: str
    limitations: str


class ReasoningPatterns(BaseModel):
    successful_patterns: List[str]
    unsuccessful_patterns: List[str]


class Findings(BaseModel):
    summary: str
    hypotheses_results: List[HypothesisResult]
    key_insights: List[str]
    reasoning_patterns: ReasoningPatterns
    future_directions: List[str]


class EvaluatorOutput(BaseModel):
    findings: Findings


# ---------------------------------------------------------------------------
# CritiqueChain
# ---------------------------------------------------------------------------

class RequirementDetail(BaseModel):
    is_met: bool = Field(..., description="Whether the requirement is met")
    comment: str = Field("", description="Explanation why the requirement is not met, if applicable")


class HypothesisImplementationDetail(BaseModel):
    hypothesis_id: str = Field(..., description="The ID of the hypothesis (e.g., H1)")
    is_implemented: bool = Field(..., description="Whether this hypothesis is fully implemented")
    comment: str = Field("", description="Explanation why the hypothesis is not fully implemented, if applicable")


class HypothesesImplementation(BaseModel):
    all_implemented: bool = Field(..., description="Whether all hypotheses are fully implemented")
    implementation_details: List[HypothesisImplementationDetail] = Field(
        default_factory=list, description="Details for each hypothesis implementation"
    )


class RequirementDetail(BaseModel):
    is_met: bool = Field(..., description="Whether the requirement is met")
    comment: str = Field("", description="Explanation why the requirement is not met, if applicable")

class HypothesisImplementationDetail(BaseModel):
    hypothesis_id: str = Field(..., description="The ID of the hypothesis (e.g., H1)")
    is_implemented: bool = Field(..., description="Whether this hypothesis is fully implemented")
    comment: str = Field("", description="Explanation why the hypothesis is not fully implemented, if applicable")

class HypothesesImplementation(BaseModel):
    all_implemented: bool = Field(..., description="Whether all hypotheses are fully implemented")
    implementation_details: List[HypothesisImplementationDetail] = Field(
        default_factory=list, description="Details for each hypothesis implementation"
    )

class RequirementsMet(BaseModel):
    hypotheses_implementation: HypothesesImplementation = Field(
        ..., description="Whether all hypotheses are fully implemented with details for each"
    )
    baseline_calculation: RequirementDetail = Field(
        ..., description="Whether a baseline is correctly calculated"
    )
    correct_entrypoint: RequirementDetail = Field(
        ..., description="Whether run_experiment.sh is the correct entrypoint"
    )
    final_results_metrics: RequirementDetail = Field(
        ..., description="Whether metrics are calculated as specified in results_summary.json"
    )
    final_results_authenticity: RequirementDetail = Field(
        ..., description="Whether results are truly the output of the experiment, not hardcoded"
    )
    dataloader_api_usage: RequirementDetail = Field(
        ..., description="Whether the dataloader API is used correctly"
    )
    ollama_client_api_usage: RequirementDetail = Field(
        ..., description="Whether the ollama LLM client API is used correctly"
    )
    extractor_evaluator_api_usage: RequirementDetail = Field(
        ..., description="Whether the answer extractor and evaluator API is used correctly"
    )
    sufficient_answer_length: RequirementDetail = Field(
        ..., description="Whether generations have enough tokens to generate complete answers"
    )
    no_cheating: RequirementDetail = Field(
        ..., description="Whether the algorithm avoids exposing ground truth unfairly"
    )


class FileStatusInfo(BaseModel):
    path: str = Field(..., description="Path to the file relative to experiment directory")
    status: str = Field(..., description="Status: unfinished, finished, or error")
    issues: List[str] = Field(default_factory=list, description="List of issues with this file if any")


class FileStatus(BaseModel):
    files: List[FileStatusInfo] = Field(..., description="Status of individual files")


class BuildStepProgress(BaseModel):
    step: int = Field(..., description="Step number from the build sequence")
    description: str = Field(..., description="Description of the build step from plan")
    files: List[str] = Field(..., description="Files associated with this build step")
    status: str = Field(..., description="Status: not_started, in_progress, or completed")
    completed_elements: List[str] = Field(default_factory=list, description="Description of elements that have been implemented")
    remaining_elements: List[str] = Field(default_factory=list, description="Description of elements still to be implemented")


class CritiqueOutput(BaseModel):
    requirements_met: RequirementsMet
    file_status: FileStatus
    build_sequence_progress: List[BuildStepProgress] = Field(default_factory=list, description="Progress on each step in the build sequence")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions for unmet requirements")