"""Builder archetype classification with hybrid rule-based + LLM scoring.

Paxel likely uses LLM-based archetype classification that consumes the full
BuilderProfile, not just rule-based thresholds. This module supports both:
- Fast rule-based classification (default, no LLM needed)
- LLM-enhanced classification (richer, more nuanced)
"""
from __future__ import annotations

import json
import re
from typing import Any

from rover.config import ARCHETYPES
from rover.llm.client import LLMClient
from rover.models import ArchetypeResult, BehavioralFeatures, BuilderProfile, DimensionScores


def classify_archetype(
    features: BehavioralFeatures,
    scores: DimensionScores,
    profile: BuilderProfile | None = None,
    use_llm: bool = True,
) -> ArchetypeResult:
    """Classify builder archetype using hybrid rule-based + LLM approach.

    Args:
        features: Behavioral features extracted from sessions
        scores: Dimension scores
        profile: Full builder profile (for LLM mode)
        use_llm: Whether to use LLM enhancement if available

    Returns:
        ArchetypeResult with archetype, confidence, description, traits, and growth edges
    """
    # Always compute rule-based result first
    rule_result = _rule_based_classify(features, scores)

    # If LLM available and requested, enhance with LLM judge
    if use_llm and profile is not None:
        client = LLMClient()
        if client.is_available():
            try:
                llm_result = _llm_classify(profile, rule_result)
                if llm_result:
                    # Blend rule-based and LLM results
                    return _blend_results(rule_result, llm_result)
            except Exception:
                pass

    return rule_result


def _rule_based_classify(features: BehavioralFeatures, scores: DimensionScores) -> ArchetypeResult:
    """Original rule-based classification from behavioral features."""
    total_tools = sum(features.tool_histogram.values()) if features.tool_histogram else 0
    read_count = features.tool_histogram.get("Read", 0)
    read_ratio = read_count / total_tools if total_tools > 0 else 0.0
    ask_ratio = features.ask_user_count / features.total_prompts if features.total_prompts > 0 else 0.0

    archetype_scores: dict[str, float] = {}

    # Architect
    arch_score = 0.0
    rules = ARCHETYPES["Architect"].rules
    if features.planning_ratio > rules.get("planning_ratio_min", 0):
        arch_score += 1.0
    if features.avg_session_duration > rules.get("avg_session_duration_min", 0):
        arch_score += 1.0
    if features.code_velocity < rules.get("code_velocity_max", float("inf")):
        arch_score += 1.0
    archetype_scores["Architect"] = arch_score

    # Sprinter
    spr_score = 0.0
    rules = ARCHETYPES["Sprinter"].rules
    if features.planning_ratio < rules.get("planning_ratio_max", float("inf")):
        spr_score += 1.0
    if features.code_velocity > rules.get("code_velocity_min", 0):
        spr_score += 1.0
    if features.avg_prompt_length < rules.get("avg_prompt_length_max", float("inf")):
        spr_score += 1.0
    archetype_scores["Sprinter"] = spr_score

    # Debugger
    dbg_score = 0.0
    rules = ARCHETYPES["Debugger"].rules
    if features.error_recovery_count >= rules.get("error_recovery_count_min", 0):
        dbg_score += 1.0
    if read_ratio >= rules.get("read_ratio_min", 0):
        dbg_score += 1.0
    if features.iteration_depth >= rules.get("iteration_depth_min", 0):
        dbg_score += 1.0
    archetype_scores["Debugger"] = dbg_score

    # Collaborator
    col_score = 0.0
    rules = ARCHETYPES["Collaborator"].rules
    if ask_ratio >= rules.get("ask_user_ratio_min", 0):
        col_score += 1.0
    if features.autonomy_score <= rules.get("autonomy_score_max", float("inf")):
        col_score += 1.0
    archetype_scores["Collaborator"] = col_score

    # Autonomous Agent
    aut_score = 0.0
    rules = ARCHETYPES["Autonomous Agent"].rules
    if features.autonomy_score >= rules.get("autonomy_score_min", 0):
        aut_score += 1.0
    if features.tool_diversity >= rules.get("tool_diversity_min", 0):
        aut_score += 1.0
    if features.avg_session_duration >= rules.get("avg_session_duration_min", 0):
        aut_score += 1.0
    archetype_scores["Autonomous Agent"] = aut_score

    # Pick the highest score
    best = max(archetype_scores, key=lambda k: (archetype_scores[k], k))
    confidence = archetype_scores[best] / 3.0

    info = ARCHETYPES[best]
    return ArchetypeResult(
        archetype=best,
        confidence=confidence,
        description=info.description,
        standout_traits=info.standout_traits,
        growth_edges=info.growth_edges,
    )


LLM_ARCHETYPE_PROMPT = """You are an engineering leadership coach. Based on the following builder profile,
classify this person's builder archetype.

DIMENSION SCORES (0-100):
- Steering: {steering}
- Execution: {execution}
- Engineering: {engineering}
- Product Thinking: {product_thinking}
- Planning: {planning}

BEHAVIORAL FEATURES:
- Total sessions: {total_sessions}
- Avg session duration: {avg_duration:.0f} minutes
- Planning ratio (research/coding): {planning_ratio:.2f}
- Tool diversity: {tool_diversity}
- Autonomy score: {autonomy:.2f}
- Error recovery count: {error_recovery}
- Avg prompt length: {avg_prompt_length:.0f} chars
- Prompt style: {prompt_style}

WORK STREAMS:
{work_streams}

ARCHETYPE OPTIONS:
1. **Architect**: Plans first, low code churn, long sessions, high planning ratio
2. **Sprinter**: High velocity, short sessions, action-oriented, ships fast
3. **Debugger**: Reactive to errors, methodical, Read-heavy, frequent recovery
4. **Collaborator**: Asks questions, seeks alignment, lower autonomy
5. **Autonomous Agent**: Long self-directed sessions, minimal intervention, high tool diversity

Return JSON:
{{
  "archetype": "Architect|Sprinter|Debugger|Collaborator|Autonomous Agent",
  "confidence": 0.0-1.0,
  "reasoning": "2-3 sentence explanation",
  "standout_traits": ["trait1", "trait2"],
  "growth_edges": ["area1", "area2"]
}}
"""


def _llm_classify(profile: BuilderProfile, rule_result: ArchetypeResult) -> ArchetypeResult | None:
    """Use LLM to classify archetype from full profile."""
    client = LLMClient()
    if not client.is_available():
        return None

    features = profile.features
    scores = profile.scores

    # Build work stream summary
    stream_texts = []
    for ws in profile.work_streams[:5]:
        stream_texts.append(f"- {ws.description}")
    work_streams_text = "\n".join(stream_texts) if stream_texts else "No work streams analyzed."

    prompt = LLM_ARCHETYPE_PROMPT.format(
        steering=scores.steering,
        execution=scores.execution,
        engineering=scores.engineering,
        product_thinking=scores.product_thinking,
        planning=scores.planning,
        total_sessions=features.total_sessions,
        avg_duration=features.avg_session_duration,
        planning_ratio=features.planning_ratio if features.planning_ratio != float("inf") else 999,
        tool_diversity=features.tool_diversity,
        autonomy=features.autonomy_score,
        error_recovery=features.error_recovery_count,
        avg_prompt_length=features.avg_prompt_length,
        prompt_style=features.prompt_style,
        work_streams=work_streams_text,
    )

    system = (
        "You are an expert engineering leadership coach. "
        "Classify builder archetypes objectively. Respond in valid JSON only."
    )

    try:
        raw = client.generate(prompt, system=system, json_mode=True, temperature=0.3)
        result = _extract_json(raw)
        if result:
            return ArchetypeResult(
                archetype=result.get("archetype", rule_result.archetype),
                confidence=float(result.get("confidence", 0.5)),
                description=rule_result.description,
                standout_traits=result.get("standout_traits", rule_result.standout_traits),
                growth_edges=result.get("growth_edges", rule_result.growth_edges),
            )
    except Exception:
        pass

    return None


def _blend_results(rule: ArchetypeResult, llm: ArchetypeResult) -> ArchetypeResult:
    """Blend rule-based and LLM results, preferring LLM when confident."""
    # If LLM is very confident (>0.7), use LLM archetype
    if llm.confidence > 0.7 and llm.archetype != rule.archetype:
        return ArchetypeResult(
            archetype=llm.archetype,
            confidence=llm.confidence,
            description=llm.description or rule.description,
            standout_traits=llm.standout_traits or rule.standout_traits,
            growth_edges=llm.growth_edges or rule.growth_edges,
        )

    # If same archetype, boost confidence
    if llm.archetype == rule.archetype:
        blended_conf = min(1.0, (rule.confidence + llm.confidence) / 2 + 0.1)
        return ArchetypeResult(
            archetype=rule.archetype,
            confidence=blended_conf,
            description=rule.description,
            standout_traits=list(set(rule.standout_traits + llm.standout_traits))[:4],
            growth_edges=list(set(rule.growth_edges + llm.growth_edges))[:3],
        )

    # Otherwise, use rule-based with LLM traits if available
    return ArchetypeResult(
        archetype=rule.archetype,
        confidence=rule.confidence,
        description=rule.description,
        standout_traits=rule.standout_traits,
        growth_edges=llm.growth_edges or rule.growth_edges,
    )


@staticmethod
def _extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None
