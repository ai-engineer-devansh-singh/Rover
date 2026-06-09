"""LLM-powered steering trace extraction.

Paxel Step 10: "Extracting steering traces: 71 sessions traced"

A steering event is any moment where the user corrected, rejected, or redirected
the AI agent. High steering frequency = low autonomy, high collaboration friction.
"""
from __future__ import annotations

import json
import re
from typing import Any

from rover.llm.client import LLMClient
from rover.models import Message, Session
from rover.parsers.common import extract_text_content


# Pattern-based steering signals — fast heuristic pre-filter
STEERING_PATTERNS = [
    (r"\b(no[,.]?\s+(do|use|try|make|add|write|change|put|set))", "rejection"),
    (r"\b(don't|dont|do not)\b", "rejection"),
    (r"\b(instead\s+(of|try|use|do))", "correction"),
    (r"\b(wait[,.]?\s+(stop|hold|not|that's|that is))", "correction"),
    (r"\b(that('s| is)?\s+(wrong|incorrect|not right|not what))", "correction"),
    (r"\b(let me clarify|to clarify|what i meant|i meant)", "clarification"),
    (r"\b(repeat(ing)?|again|once more|say that again)\b", "repetition"),
    (r"\b(back(ing)?\s+(up|out)|undo|revert)\b", "boundary"),
    (r"\b(too\s+(much|many|complex|complicated|slow|fast))", "boundary"),
    (r"\b(just|simply|only)\s+(do|use|need|want)", "boundary"),
]


class SteeringExtractor:
    """Extract steering events from session transcripts.

    Uses a hybrid approach:
    1. Fast regex pattern detection (heuristic)
    2. LLM confirmation + classification (accurate)
    """

    SYSTEM_PROMPT = (
        "You are an expert in human-AI collaboration dynamics. "
        "Analyze user prompts for steering events — moments where the user "
        "corrected, rejected, redirected, or clarified the AI's approach. "
        "Respond in valid JSON only."
    )

    STEERING_PROMPT = """Analyze the following user prompts from an AI coding session.
Identify ALL steering events — moments where the user guided, corrected, rejected, or redirected the AI.

STEERING EVENT DEFINITIONS:
- **rejection**: User explicitly says "no", "don't do X", rejects an AI suggestion
- **correction**: User redirects the AI to a different approach ("instead...", "wait, do Y")
- **clarification**: User clarifies intent because AI misunderstood ("what I meant was...")
- **repetition**: User repeats a request because AI failed first time
- **boundary**: User sets limits ("too complex", "simpler", "just do X", "don't over-engineer")

For each steering event, provide:
- type: one of rejection|correction|clarification|repetition|boundary
- severity: minor (1 word fix), moderate (direction change), major (fundamental disagreement)
- topic: the file, feature, or concept being steered
- description: 1-sentence summary of what happened
- evidence: the exact user prompt text (max 120 chars) that shows the steering

USER PROMPTS:
{prompts}

Return a JSON array of steering events. If none found, return [].
"""

    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or LLMClient()

    def extract(self, session: Session) -> list[dict[str, Any]]:
        """Extract steering events from a session."""
        # Phase 1: Fast heuristic detection
        heuristic_events = self._heuristic_extract(session)

        if not self.client.is_available():
            return heuristic_events

        # Phase 2: LLM confirmation and enrichment (only if heuristic found signals)
        # This saves LLM calls for clean sessions with no steering
        if len(heuristic_events) == 0:
            # Still do a lightweight LLM pass for subtle steering (e.g., implicit corrections)
            pass  # continue to LLM

        prompts = []
        for msg in session.messages:
            if msg.role == "user":
                text = extract_text_content(msg.content)
                if text:
                    prompts.append(text[:600])

        if not prompts:
            return heuristic_events

        prompt_text = self.STEERING_PROMPT.format(
            prompts="\n---\n".join([f"{i+1}. {p}" for i, p in enumerate(prompts[:20])])
        )

        try:
            raw = self.client.generate(
                prompt_text, system=self.SYSTEM_PROMPT, json_mode=True, temperature=0.2
            )
            json_match = re.search(r"\[.*\]", raw, re.DOTALL)
            if json_match:
                events = json.loads(json_match.group())
                # Merge with heuristic events, deduplicating by evidence text
                seen_evidence = {e.get("evidence", "") for e in events}
                for he in heuristic_events:
                    if he.get("evidence", "") not in seen_evidence:
                        events.append(he)
                return events[:20]  # Cap at 20 events
        except Exception:
            pass

        return heuristic_events

    def _heuristic_extract(self, session: Session) -> list[dict[str, Any]]:
        """Fast regex-based steering detection."""
        events = []
        for msg in session.messages:
            if msg.role != "user":
                continue
            text = extract_text_content(msg.content)
            if not text:
                continue
            lower = text.lower()
            for pattern, stype in STEERING_PATTERNS:
                if re.search(pattern, lower):
                    # Determine severity by context length and intensity words
                    severity = self._heuristic_severity(text)
                    events.append({
                        "type": stype,
                        "severity": severity,
                        "topic": self._guess_topic(text, session),
                        "description": f"User {stype}: {text[:100]}...",
                        "evidence": text[:120],
                        "source": "heuristic",
                    })
                    break  # One event per message max for heuristic
        return events

    @staticmethod
    def _heuristic_severity(text: str) -> str:
        """Guess severity from intensity markers."""
        lower = text.lower()
        major_markers = ["completely", "totally", "entirely", "wrong", "not at all", "start over", "undo"]
        if any(m in lower for m in major_markers):
            return "major"
        minor_markers = ["just", "simply", "only", "slightly", "minor"]
        if any(m in lower for m in minor_markers):
            return "minor"
        return "moderate"

    @staticmethod
    def _guess_topic(text: str, session: Session) -> str:
        """Try to identify the topic (file/feature) being steered."""
        # Look for file paths in the text
        file_matches = re.findall(r"[\w\-./]+\.(py|ts|js|tsx|jsx|go|rs|java|kt|swift|rb|php)", text)
        if file_matches:
            return file_matches[0]
        # Look for feature keywords
        feature_words = re.findall(r"\b(auth|login|api|test|component|page|route|model|database|ui|css|style|deploy|config)\b", text.lower())
        if feature_words:
            return feature_words[0]
        return "general"


class SteeringMetrics:
    """Aggregate steering events into session-level metrics."""

    @staticmethod
    def compute(events: list[dict[str, Any]]) -> dict[str, Any]:
        """Compute aggregate metrics from steering events."""
        if not events:
            return {
                "total_events": 0,
                "severity_score": 0.0,  # 0 = no steering, 100 = constant steering
                "rejection_rate": 0.0,
                "correction_rate": 0.0,
                "dominant_type": "none",
                "autonomy_indicator": 100.0,  # inverse of steering
            }

        total = len(events)
        type_counts: dict[str, int] = {}
        severity_weights = {"minor": 1, "moderate": 3, "major": 6}
        severity_sum = 0

        for ev in events:
            t = ev.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
            severity_sum += severity_weights.get(ev.get("severity", "moderate"), 3)

        # Severity score: weighted sum normalized to 0-100
        # 1 minor = 1 point, 1 moderate = 3 points, 1 major = 6 points
        # Scale: <5 points = low steering, 5-15 = moderate, >15 = high
        severity_score = min(100.0, (severity_sum / max(total, 1)) * 15)

        return {
            "total_events": total,
            "severity_score": round(severity_score, 1),
            "rejection_rate": round(type_counts.get("rejection", 0) / total, 2),
            "correction_rate": round(type_counts.get("correction", 0) / total, 2),
            "clarification_rate": round(type_counts.get("clarification", 0) / total, 2),
            "dominant_type": max(type_counts, key=type_counts.get),
            "autonomy_indicator": round(max(0.0, 100.0 - severity_score), 1),
        }
