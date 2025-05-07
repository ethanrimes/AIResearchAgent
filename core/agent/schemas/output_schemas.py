"""
Pydantic output schemas for the agent chains.
"""

from __future__ import annotations
from typing import List, Dict
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

class CritiqueOutput(BaseModel):
    requirements_met: RequirementsMet
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions for unmet requirements")