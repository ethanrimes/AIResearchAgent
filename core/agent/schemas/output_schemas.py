## 2. Updated output_schemas.py
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

class HypothesesImplementation(BaseModel):
    is_complete: bool = Field(..., description="Whether all hypotheses are fully implemented")
    missing_elements: List[str] = Field(default_factory=list, description="List of missing hypothesis elements if any")


class RequirementsMet(BaseModel):
    hypotheses_implementation: HypothesesImplementation
    correct_entrypoint: bool = Field(..., description="Whether run_experiment.sh is the correct entrypoint")
    baseline_calculation: bool = Field(..., description="Whether a baseline is correctly calculated")
    metrics_calculation: bool = Field(..., description="Whether metrics are calculated as specified")
    dataloader_api_usage: bool = Field(..., description="Whether the dataloader API is used correctly")
    ollama_client_api_usage: bool = Field(..., description="Whether the ollama LLM client API is used correctly")
    extractor_evaluator_api_usage: bool = Field(..., description="Whether the answer extractor and evaluator API is used correctly")


class FileStatusInfo(BaseModel):
    path: str = Field(..., description="Path to the file relative to experiment directory")
    status: str = Field(..., description="Status: unfinished, finished, or error")
    issues: List[str] = Field(default_factory=list, description="List of issues with this file if any")


class FileStatusSummary(BaseModel):
    total_files: int = Field(..., description="Total number of files in the plan")
    finished_files: int = Field(..., description="Number of files that are complete")
    unfinished_files: int = Field(..., description="Number of files not yet completed")
    error_files: int = Field(..., description="Number of files with errors")
    completion_percentage: float = Field(..., description="Percentage of files that are complete")


class FileStatus(BaseModel):
    files: List[FileStatusInfo] = Field(..., description="Status of individual files")
    summary: FileStatusSummary = Field(..., description="Summary of file completion status")


class TypeCheckResults(BaseModel):
    has_errors: bool = Field(..., description="Whether type checking found any errors")
    error_count: int = Field(..., description="Number of type errors found")
    errors: List[str] = Field(default_factory=list, description="List of type errors if any")


class CritiqueOutput(BaseModel):
    requirements_met: RequirementsMet
    file_status: FileStatus
    type_check_results: TypeCheckResults
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions for unmet requirements")