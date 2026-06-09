"""LLM-powered decision pattern extraction — Stage 2 of Paxel's pipeline.

Paxel Step 11: "Extracting decision exchanges [cloud: gpt-5.5]: 255 decisions"

Consumes Stage 1 narratives + user prompts to identify technical decisions,
trade-offs, and course corrections. Links decisions to nonces for verifiability.
"""
from __future__ import annotations

import json
import re
from typing import Any

from rover.llm.client import LLMClient
from rover.models import Session
from rover.parsers.common import extract_text_content


class DecisionExtractor:
    """Extract technical decisions from sessions using Paxel's two-stage approach.

    Stage 1 (Narrative) is consumed as context. Stage 2 extracts decisions with
    alternatives, rationale, and confidence. Each decision is linked to nonce IDs
    for verifiable evidence.
    """

    SYSTEM_PROMPT = (
        "You are an expert software architect analyzing human-AI collaboration. "
        "Identify significant technical DECISIONS and trade-offs. Be precise and evidence-based. "
        "Respond in valid JSON only."
    )

    DECISION_PROMPT = """You have already summarized this coding session (Stage 1 narrative).
Now analyze it for TECHNICAL DECISIONS.

STAGE 1 NARRATIVE:
{narrative}

RELEVANT USER PROMPTS:
{prompts}

DECISION DEFINITION:
A decision is a point where:
- The user chose between alternative approaches
- The user corrected or redirected the AI based on technical reasoning
- An architectural or design choice was made explicitly
- A trade-off was considered (speed vs. maintainability, simplicity vs. features, etc.)

NOT a decision:
- Routine implementation details ("add a button", "fix the typo")
- Tool usage without reasoning ("run the tests")
- Accepting AI suggestions without deliberation

For each decision, provide:
- decision: Brief description (1 sentence)
- context: The surrounding situation (1-2 sentences)
- alternatives: What was considered but NOT chosen (array of strings)
- rationale: Why this choice was made (1 sentence)
- confidence: "high" (clear deliberation), "medium" (implied choice), "low" (unclear)
- type: "architecture" (system design) | "implementation" (code-level) | "correction" (fixing AI mistake) | "exploration" (spiking options) | "dependency" (library/tool choice)
- evidence: The exact user prompt text that shows this decision (max 120 chars)

Return a JSON array. If no clear decisions, return [].
"""

    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or LLMClient()

    def extract(self, session: Session) -> list[dict[str, Any]]:
        """Extract decisions from a session, consuming its Stage 1 narrative."""
        if not self.client.is_available():
            return self._fallback_extract(session)

        # Build narrative context from session summary (Stage 1 output)
        narrative = self._build_narrative_context(session)

        # Extract user prompts as evidence
        prompts = []
        for msg in session.messages:
            if msg.role == "user":
                text = extract_text_content(msg.content)
                if text:
                    prompts.append(text[:500])

        if not prompts:
            return []

        prompts_text = "\n".join([f"{i+1}. {p}" for i, p in enumerate(prompts[:15])])
        prompt_text = self.DECISION_PROMPT.format(
            narrative=narrative,
            prompts=prompts_text,
        )

        try:
            raw = self.client.generate(
                prompt_text, system=self.SYSTEM_PROMPT, json_mode=True, temperature=0.2
            )
            result = self._extract_json_array(raw)
            if result is not None:
                return self._normalize_decisions(result)
        except Exception:
            pass

        return self._fallback_extract(session)

    def _build_narrative_context(self, session: Session) -> str:
        """Build narrative context from session summary or fallback."""
        if session.summary:
            parts = [session.summary.get("narrative", "")]
            if session.summary.get("goal"):
                parts.append(f"Goal: {session.summary['goal']}")
            if session.summary.get("technologies"):
                parts.append(f"Technologies: {', '.join(session.summary['technologies'])}")
            return "\n".join(parts)
        return f"Session on {session.project_name or 'unknown project'} with {session.prompt_count} prompts."

    @staticmethod
    def _extract_json_array(text: str) -> list[dict[str, Any]] | None:
        """Extract JSON array from LLM response."""
        text = text.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return None

    def _normalize_decisions(self, decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize and validate decision objects."""
        normalized = []
        for d in decisions:
            if not isinstance(d, dict):
                continue
            decision = {
                "decision": d.get("decision", "")[:300],
                "context": d.get("context", "")[:400],
                "alternatives": d.get("alternatives", [])[:5],
                "rationale": d.get("rationale", "")[:300],
                "confidence": d.get("confidence", "medium"),
                "type": d.get("type", "implementation"),
                "evidence": d.get("evidence", "")[:150],
            }
            # Validate confidence
            if decision["confidence"] not in ("high", "medium", "low"):
                decision["confidence"] = "medium"
            # Validate type
            if decision["type"] not in ("architecture", "implementation", "correction", "exploration", "dependency"):
                decision["type"] = "implementation"
            normalized.append(decision)
        return normalized

    def _fallback_extract(self, session: Session) -> list[dict[str, Any]]:
        """Rule-based decision extraction when LLM unavailable."""
        decisions = []
        decision_keywords = [
            ("should we", "approach"),
            ("which approach", "approach"),
            ("instead of", "correction"),
            ("rather than", "correction"),
            ("let's use", "implementation"),
            ("going with", "implementation"),
            ("refactor", "architecture"),
            ("rewrite", "architecture"),
            ("migrate", "architecture"),
            ("upgrade", "dependency"),
            ("choose", "implementation"),
            ("decided", "implementation"),
        ]

        for msg in session.messages:
            if msg.role != "user":
                continue
            text = extract_text_content(msg.content)
            lower = text.lower()
            for kw, dtype in decision_keywords:
                if kw in lower:
                    decisions.append({
                        "decision": text[:200],
                        "context": "Detected from user prompt",
                        "alternatives": [],
                        "rationale": "",
                        "confidence": "medium",
                        "type": dtype,
                        "evidence": text[:120],
                        "source": "heuristic",
                    })
                    break
        return decisions[:8]
