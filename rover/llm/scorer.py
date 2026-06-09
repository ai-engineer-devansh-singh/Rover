"""LLM-powered dimension scoring — Stage 2 of Paxel's pipeline.

Paxel Step 15: "Scoring episodes across 5 axes [cloud: gpt-5.5]: 88 scored, 13 skipped (no evidence)"

Consumes Stage 1 narratives + Stage 2 decisions + steering metrics + git outcomes.
Produces 5-dimension scores (0-100) with reasoning.
Sessions with "no evidence" (0 nonces, no decisions, no git activity) are skipped.
"""
from __future__ import annotations

import json
import re
from typing import Any

from rover.llm.client import LLMClient
from rover.models import DimensionScores, Session


class LLMScorer:
    """Score sessions across 5 dimensions using an LLM with full context.

    Dimensions (user-facing, inspired by Paxel's evaluator axes):
    - steering: How well did the user guide the AI? (autonomy proxy)
    - execution: How efficiently did they execute? (velocity proxy)
    - engineering: How strong was engineering quality? (quality proxy)
    - product_thinking: How much product/architectural thinking? (complexity proxy)
    - planning: How much planning before coding? (iteration proxy)
    """

    SYSTEM_PROMPT = (
        "You are an expert engineering leadership coach evaluating AI-assisted coding sessions. "
        "Score objectively using the evidence provided. Explain your reasoning briefly. "
        "Respond in valid JSON only."
    )

    SCORING_PROMPT = """Score this AI coding session across 5 dimensions (0-100).

Use the full context below — narrative, decisions, steering events, and git outcomes —
to make an evidence-based assessment.

=== SESSION CONTEXT ===
Project: {project}
Duration: {duration_min:.0f} minutes
Prompts: {prompt_count}
Tool uses: {tool_count}

=== STAGE 1 NARRATIVE ===
{narrative}

=== DECISIONS ({decision_count}) ===
{decisions}

=== STEERING EVENTS ({steering_count}) ===
{steering}

=== GIT OUTCOMES ===
{git_outcomes}

=== DIMENSION DEFINITIONS ===
1. **steering** (0-100): Quality of user guidance.
   - High (80-100): Minimal babysitting, clear intent, AI understood well
   - Medium (40-70): Some corrections needed, but smooth overall
   - Low (0-30): Constant redirection, frustration, AI misunderstood repeatedly

2. **execution** (0-100): Speed and efficiency of delivery.
   - High (80-100): Fast iteration, tool mastery, commands succeeded
   - Medium (40-70): Moderate pace, some retries
   - Low (0-30): Slow, many errors, inefficient tool use

3. **engineering** (0-100): Code quality and structural thinking.
   - High (80-100): Tests considered, clean structure, refactoring
   - Medium (40-70): Functional but minimal testing/structure
   - Low (0-30): Hacky, no tests, high churn

4. **product_thinking** (0-100): User focus and architectural awareness.
   - High (80-100): Trade-offs considered, user impact discussed, clean architecture
   - Medium (40-70): Some awareness, mostly implementation-focused
   - Low (0-30): Purely tactical, no broader context

5. **planning** (0-100): Research and strategic preparation.
   - High (80-100): Explored codebase first, researched options, planned structure
   - Medium (40-70): Some exploration, mixed planning/coding
   - Low (0-30): Dived straight into coding with minimal context

=== SCORING RULES ===
- Base scores on EVIDENCE from the context above
- If there is very little evidence (no decisions, no steering, no git), score around 50 (unknown)
- Provide brief reasoning for each dimension
- Return ONLY valid JSON

Return exactly this JSON structure:
{{
  "steering": 0,
  "execution": 0,
  "engineering": 0,
  "product_thinking": 0,
  "planning": 0,
  "reasoning": {{
    "steering": "brief explanation",
    "execution": "brief explanation",
    "engineering": "brief explanation",
    "product_thinking": "brief explanation",
    "planning": "brief explanation"
  }},
  "evidence_level": "high|medium|low|none"
}}
"""

    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or LLMClient()
        self._cache: dict[str, DimensionScores] = {}

    def score_session(
        self,
        session: Session,
        steering_metrics: dict[str, Any] | None = None,
        git_outcomes: dict[str, Any] | None = None,
    ) -> DimensionScores:
        """Score a single session with full Stage 1 + Stage 2 context."""
        cache_key = f"{session.session_id}:{session.agent_type.value}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if not self.client.is_available():
            return DimensionScores()

        # Build rich context prompt
        prompt = self._build_prompt(session, steering_metrics, git_outcomes)

        try:
            raw = self.client.generate(
                prompt, system=self.SYSTEM_PROMPT, json_mode=True, temperature=0.3
            )
            result = self._extract_json(raw)
            if result:
                scores = self._parse_scores(result)
                self._cache[cache_key] = scores
                return scores
        except Exception:
            pass

        return DimensionScores()

    def _build_prompt(
        self,
        session: Session,
        steering_metrics: dict[str, Any] | None,
        git_outcomes: dict[str, Any] | None,
    ) -> str:
        """Build the rich context prompt for scoring."""
        # Narrative from Stage 1
        narrative = session.summary.get("narrative", "No narrative available.") if session.summary else "No narrative available."
        goal = session.summary.get("goal", "") if session.summary else ""
        if goal:
            narrative += f"\nGoal: {goal}"

        # Decisions from Stage 2
        decisions_text = "No decisions extracted."
        if session.decisions:
            lines = []
            for i, d in enumerate(session.decisions[:5]):
                alt = ", ".join(d.get("alternatives", [])[:3]) or "none considered"
                lines.append(f"{i+1}. [{d.get('type', 'unknown')}] {d.get('decision', '')} (alternatives: {alt})")
            decisions_text = "\n".join(lines)

        # Steering metrics
        steering_text = "No steering events detected."
        if steering_metrics:
            events = steering_metrics.get("total_events", 0)
            severity = steering_metrics.get("severity_score", 0)
            autonomy = steering_metrics.get("autonomy_indicator", 100)
            steering_text = (
                f"Total steering events: {events}\n"
                f"Severity score: {severity}/100 (higher = more friction)\n"
                f"Autonomy indicator: {autonomy}/100"
            )

        # Git outcomes
        git_text = "No git data linked."
        if git_outcomes:
            commits = git_outcomes.get("commits_after_session", 0)
            insertions = git_outcomes.get("insertions", 0)
            deletions = git_outcomes.get("deletions", 0)
            reverted = git_outcomes.get("reverted", False)
            git_text = (
                f"Commits after session: {commits}\n"
                f"Lines changed: +{insertions}/-{deletions}\n"
                f"Reverted: {'yes' if reverted else 'no'}"
            )

        return self.SCORING_PROMPT.format(
            project=session.project_name or session.cwd or "unknown",
            duration_min=session.duration_minutes,
            prompt_count=session.prompt_count,
            tool_count=session.tool_use_count,
            narrative=narrative,
            decision_count=len(session.decisions),
            decisions=decisions_text,
            steering_count=steering_metrics.get("total_events", 0) if steering_metrics else 0,
            steering=steering_text,
            git_outcomes=git_text,
        )

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        """Extract JSON object from LLM response."""
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

    def _parse_scores(self, result: dict[str, Any]) -> DimensionScores:
        """Parse dimension scores from LLM output."""
        def _get(key: str) -> float:
            val = result.get(key, 50)
            try:
                return max(0.0, min(100.0, float(val)))
            except (ValueError, TypeError):
                return 50.0

        return DimensionScores(
            steering=_get("steering"),
            execution=_get("execution"),
            engineering=_get("engineering"),
            product_thinking=_get("product_thinking"),
            planning=_get("planning"),
        )

    def should_skip_session(
        self,
        session: Session,
        steering_metrics: dict[str, Any] | None = None,
        git_outcomes: dict[str, Any] | None = None,
    ) -> bool:
        """Determine if session should be skipped due to insufficient evidence.

        Paxel skips 13 of 101 work streams for "no evidence".
        """
        evidence_score = 0

        # Narrative evidence
        if session.summary and session.summary.get("narrative"):
            evidence_score += 1

        # Decision evidence
        if session.decisions:
            evidence_score += len(session.decisions)

        # Steering evidence
        if steering_metrics and steering_metrics.get("total_events", 0) > 0:
            evidence_score += 1

        # Git evidence
        if git_outcomes and git_outcomes.get("commits_after_session", 0) > 0:
            evidence_score += 1

        # Session activity
        if session.prompt_count >= 3:
            evidence_score += 1

        # Skip if almost no evidence
        return evidence_score < 2

    def aggregate(self, sessions: list[Session]) -> DimensionScores:
        """Score all sessions and aggregate with length-based weights."""
        if not self.client.is_available():
            return DimensionScores()

        if not sessions:
            return DimensionScores()

        total = DimensionScores()
        weights = 0.0

        for session in sessions:
            s = self.score_session(session)
            # Weight by session length (longer sessions = more reliable evidence)
            weight = max(1.0, len(session.messages) / 10.0)
            total.steering += s.steering * weight
            total.execution += s.execution * weight
            total.engineering += s.engineering * weight
            total.product_thinking += s.product_thinking * weight
            total.planning += s.planning * weight
            weights += weight

        if weights > 0:
            return DimensionScores(
                steering=total.steering / weights,
                execution=total.execution / weights,
                engineering=total.engineering / weights,
                product_thinking=total.product_thinking / weights,
                planning=total.planning / weights,
            )
        return DimensionScores()
