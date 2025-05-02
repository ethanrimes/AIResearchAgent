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