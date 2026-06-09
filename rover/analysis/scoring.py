"""Five-dimension builder scoring engine."""
from __future__ import annotations

import math
from typing import Any

from rover.models import BehavioralFeatures, DimensionScores


def score_dimensions(features: BehavioralFeatures) -> DimensionScores:
    """Compute 5-dimension scores (0-100) from behavioral features."""
    scores = DimensionScores()

    # Steering: course corrections, tool diversity, adaptability
    scores.steering = _score_steering(features)

    # Execution: tool use efficiency, velocity, focus
    scores.execution = _score_execution(features)

    # Engineering: code quality thinking, iteration discipline, structure
    scores.engineering = _score_engineering(features)

    # Product Thinking: user focus, architectural language, trade-offs
    scores.product_thinking = _score_product_thinking(features)

    # Planning: pre-coding research depth
    scores.planning = _score_planning(features)

    return scores


def _score_steering(features: BehavioralFeatures) -> float:
    """Score steering dimension (adaptability, course correction)."""
    base = 50.0
    # Error recovery shows responsiveness
    base += min(features.error_recovery_count * 3, 20)
    # Tool diversity shows flexibility
    base += min(features.tool_diversity * 2, 15)
    # Autonomy shows self-direction
    base += features.autonomy_score * 10
    return _clamp(base)


def _score_execution(features: BehavioralFeatures) -> float:
    """Score execution dimension (speed, efficiency, tool mastery)."""
    base = 50.0
    # Code velocity bonus
    if features.code_velocity > 0:
        base += min(features.code_velocity / 20, 20)
    # Short sessions = fast iteration
    if features.avg_session_duration > 0:
        if features.avg_session_duration < 60:
            base += 10
        elif features.avg_session_duration > 180:
            base -= 10
    # Tool usage frequency
    total_tools = sum(features.tool_histogram.values())
    if total_tools > 0:
        bash_pct = features.tool_histogram.get("Bash", 0) / total_tools
        base += (bash_pct - 0.3) * 20  # slight bias toward bash usage
    return _clamp(base)


def _score_engineering(features: BehavioralFeatures) -> float:
    """Score engineering dimension (quality, structure, testing)."""
    base = 50.0
    # Iteration depth (edits per file) — moderate is good, too high is churn
    if features.iteration_depth > 0:
        if features.iteration_depth < 2:
            base += 10  # decisive
        elif features.iteration_depth > 5:
            base -= 10  # churn
        else:
            base += 5
    # Planning ratio — moderate research is good
    if features.planning_ratio > 0 and features.planning_ratio != float("inf"):
        if 0.5 < features.planning_ratio < 2.0:
            base += 10
        elif features.planning_ratio > 3.0:
            base += 5  # over-planning slightly penalized
    # Test mentions in prompts
    if features.prompt_style == "action":
        base += 5
    return _clamp(base)


def _score_product_thinking(features: BehavioralFeatures) -> float:
    """Score product thinking dimension (user focus, architecture)."""
    base = 50.0
    # AskUserQuestion ratio — moderate is good
    if features.total_prompts > 0:
        ask_ratio = features.ask_user_count / features.total_prompts
        if 0.05 < ask_ratio < 0.2:
            base += 10
        elif ask_ratio > 0.3:
            base -= 5
    # Prompt style
    if features.prompt_style == "architectural":
        base += 15
    elif features.prompt_style == "collaborative":
        base += 5
    # Long prompts suggest detailed thinking
    if features.avg_prompt_length > 150:
        base += 5
    return _clamp(base)


def _score_planning(features: BehavioralFeatures) -> float:
    """Score planning dimension (research depth, strategic thinking)."""
    base = 50.0
    # Planning ratio
    if features.planning_ratio > 0 and features.planning_ratio != float("inf"):
        if features.planning_ratio > 1.5:
            base += 20
        elif features.planning_ratio > 0.8:
            base += 10
        elif features.planning_ratio < 0.3:
            base -= 10
    else:
        base += 15  # inf planning ratio = all research, no coding
    # Long prompts = more context/planning
    if features.avg_prompt_length > 100:
        base += 5
    if features.avg_prompt_length > 200:
        base += 5
    # Tool diversity in research
    research_tools = features.tool_histogram.get("Glob", 0) + features.tool_histogram.get("Grep", 0) + features.tool_histogram.get("Read", 0)
    if research_tools > 10:
        base += 5
    return _clamp(base)


def _clamp(value: float) -> float:
    """Clamp score to 0-100 range."""
    if math.isnan(value):
        return 50.0
    return max(0.0, min(100.0, value))
