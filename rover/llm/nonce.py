"""Nonce / Evidence extraction system.

Paxel reports: "109 nonces from 21 sessions, 88 from 88/88 episodes"

A nonce is a verifiable evidence marker — a concrete, quotable snippet from the
transcript that anchors an LLM-generated claim (summary, decision, score).

This enables:
- Explainability: Every claim traces back to source evidence
- Adversarial verification: Second LLM can verify claims against nonces
- Trust: Prevents hallucinated summaries from contaminating scores
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from rover.llm.client import LLMClient
from rover.models import Message, Session
from rover.parsers.common import extract_text_content


class NonceExtractor:
    """Extract verifiable evidence markers (nonces) from sessions.

    A nonce is a short, verifiable snippet from the transcript that supports
    a specific claim about what happened in the session.
    """

    SYSTEM_PROMPT = (
        "You are a forensic evidence extractor. Your job is to find concrete, "
        "verifiable snippets from transcripts that support factual claims. "
        "Respond in valid JSON only."
    )

    NONCE_PROMPT = """Analyze this AI coding session transcript and extract EVIDENCE MARKERS (nonces).

A NONCE is a short, verifiable snippet from the transcript that proves something happened.
Types of nonces we care about:

1. **decision_nonce** — Evidence that a technical decision was made
   Example: "let's use JWT instead of session cookies" proves an auth architecture decision

2. **tool_nonce** — Evidence of specific tool usage patterns
   Example: Agent reading 5 files before editing proves exploration-first behavior

3. **steering_nonce** — Evidence the user corrected the AI
   Example: "no, don't use a global variable here" proves a steering event

4. **outcome_nonce** — Evidence of what was achieved
   Example: "the tests are passing now" proves successful execution

5. **error_nonce** — Evidence of errors and recovery
   Example: "ImportError: No module named 'x'" followed by a fix proves error recovery

RULES:
- Each nonce must be QUOTABLE — copy the exact text from the transcript
- Nonces should be 20-150 characters long
- Only extract nonces for CLEAR, VERIFIABLE events
- Do NOT paraphrase — use exact transcript text
- If no clear nonce exists for a category, omit it

TRANSCRIPT:
{transcript}

Return JSON with structure:
{{
  "nonces": [
    {{
      "id": "nonce_1",
      "type": "decision|tool|steering|outcome|error",
      "text": "exact transcript snippet",
      "context": "what this proves (1 sentence)",
      "confidence": "high|medium|low"
    }}
  ],
  "nonce_count": 0,
  "coverage": "high|medium|low"  // how well nonces cover session events
}}
"""

    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or LLMClient()

    def extract(self, session: Session) -> dict[str, Any]:
        """Extract nonces from a session. Returns a dict with nonces array."""
        # Fast path: build a compressed transcript
        lines = self._compress_transcript(session)
        if not lines:
            return {"nonces": [], "nonce_count": 0, "coverage": "none"}

        if not self.client.is_available():
            return self._fallback_extract(session, lines)

        transcript_text = "\n".join(lines[:50])  # Cap at 50 lines
        prompt = self.NONCE_PROMPT.format(transcript=transcript_text)

        try:
            raw = self.client.generate(
                prompt, system=self.SYSTEM_PROMPT, json_mode=True, temperature=0.2
            )
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                nonces = result.get("nonces", [])
                # Validate nonces: ensure text actually appears in transcript
                validated = self._validate_nonces(nonces, lines)
                return {
                    "nonces": validated,
                    "nonce_count": len(validated),
                    "coverage": result.get("coverage", "medium"),
                }
        except Exception:
            pass

        return self._fallback_extract(session, lines)

    def _compress_transcript(self, session: Session) -> list[str]:
        """Build a privacy-safe compressed transcript for nonce extraction."""
        lines: list[str] = []
        for msg in session.messages:
            if msg.role == "user":
                text = extract_text_content(msg.content)
                if text:
                    lines.append(f"USER: {text[:400]}")
            elif msg.role == "assistant":
                if msg.tool_use:
                    name = msg.tool_use.name
                    inp = msg.tool_use.input
                    # Redact file contents, keep paths
                    if isinstance(inp, dict):
                        redacted = {k: v for k, v in inp.items() if k in ("command", "file_path", "path", "pattern")}
                        lines.append(f"AI_TOOL: {name}({json.dumps(redacted, default=str)[:200]})")
                    else:
                        lines.append(f"AI_TOOL: {name}")
                elif msg.tool_result:
                    status = "ERROR" if msg.tool_result.is_error else "OK"
                    lines.append(f"AI_RESULT: {status}")
                else:
                    text = extract_text_content(msg.content)
                    if text:
                        lines.append(f"AI: {text[:300]}")
        return lines

    def _validate_nonces(self, nonces: list[dict[str, Any]], transcript_lines: list[str]) -> list[dict[str, Any]]:
        """Check that nonce text actually appears in the transcript."""
        validated = []
        transcript_text = "\n".join(transcript_lines).lower()
        for nonce in nonces:
            text = nonce.get("text", "").lower()
            if text and len(text) >= 10:
                # Fuzzy match: nonce text should be substantially present
                if text in transcript_text or self._fuzzy_in(text, transcript_text):
                    nonce["verified"] = True
                    validated.append(nonce)
                else:
                    nonce["verified"] = False
                    # Keep it but mark as unverified (might be paraphrased)
                    validated.append(nonce)
        return validated

    @staticmethod
    def _fuzzy_in(needle: str, haystack: str) -> bool:
        """Check if most words from needle appear in haystack."""
        needle_words = set(needle.split())
        haystack_words = set(haystack.split())
        if not needle_words:
            return False
        overlap = len(needle_words & haystack_words)
        return overlap / len(needle_words) >= 0.7

    def _fallback_extract(self, session: Session, lines: list[str]) -> dict[str, Any]:
        """Rule-based nonce extraction when LLM unavailable."""
        nonces: list[dict[str, Any]] = []
        transcript_text = "\n".join(lines).lower()

        # Decision signals
        decision_patterns = [
            r"\b(let's\s+use|we should\s+use|choose|decided\s+to|going\s+with)\b",
            r"\b(instead\s+of|rather\s+than|prefer)\b",
        ]
        for pattern in decision_patterns:
            for match in re.finditer(pattern, transcript_text):
                start = max(0, match.start() - 30)
                end = min(len(transcript_text), match.end() + 80)
                snippet = transcript_text[start:end]
                nonces.append({
                    "id": f"nonce_{hashlib.sha256(snippet.encode()).hexdigest()[:6]}",
                    "type": "decision",
                    "text": snippet,
                    "context": "User expressed a technical choice",
                    "confidence": "medium",
                    "verified": True,
                    "source": "heuristic",
                })

        # Steering signals
        steering_patterns = [
            r"\b(no[,.]?\s+(do|use|try)|don't|wait[,.]?)\b",
            r"\b(that('s| is)?\s+(wrong|incorrect))\b",
        ]
        for pattern in steering_patterns:
            for match in re.finditer(pattern, transcript_text):
                start = max(0, match.start() - 20)
                end = min(len(transcript_text), match.end() + 60)
                snippet = transcript_text[start:end]
                nonces.append({
                    "id": f"nonce_{hashlib.sha256(snippet.encode()).hexdigest()[:6]}",
                    "type": "steering",
                    "text": snippet,
                    "context": "User corrected or redirected the AI",
                    "confidence": "medium",
                    "verified": True,
                    "source": "heuristic",
                })

        # Tool pattern signals
        tool_reads = transcript_text.count("ai_tool: read")
        tool_edits = transcript_text.count("ai_tool: edit")
        if tool_reads > 3 and tool_edits > 0:
            nonces.append({
                "id": f"nonce_explore_{tool_reads}",
                "type": "tool",
                "text": f"AI performed {tool_reads} reads before {tool_edits} edits",
                "context": "Exploration-first coding pattern",
                "confidence": "high",
                "verified": True,
                "source": "heuristic",
            })

        return {
            "nonces": nonces[:15],
            "nonce_count": len(nonces[:15]),
            "coverage": "low" if len(nonces) < 3 else ("medium" if len(nonces) < 8 else "high"),
        }


class NonceVerifier:
    """Adversarially verify LLM-generated claims against nonces."""

    SYSTEM_PROMPT = (
        "You are a skeptical fact-checker. Given a claim and supporting evidence snippets, "
        "determine if the claim is strongly supported, weakly supported, or contradicted. "
        "Respond in valid JSON only."
    )

    VERIFICATION_PROMPT = """Claim: "{claim}"

Supporting Evidence (nonces):
{nonces}

Evaluate whether the evidence strongly supports the claim, weakly supports it, or contradicts it.
Consider:
- Does the evidence directly support the claim?
- Is there any evidence that contradicts the claim?
- Is the claim an overgeneralization?

Return JSON:
{{
  "verdict": "supported|weakly_supported|contradicted|insufficient_evidence",
  "confidence": "high|medium|low",
  "reasoning": "1-2 sentence explanation"
}}
"""

    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or LLMClient()

    def verify(self, claim: str, nonces: list[dict[str, Any]]) -> dict[str, Any]:
        """Verify a claim against nonces."""
        if not nonces:
            return {
                "verdict": "insufficient_evidence",
                "confidence": "high",
                "reasoning": "No evidence markers provided.",
            }

        if not self.client.is_available():
            return self._fallback_verify(claim, nonces)

        nonce_text = "\n".join([
            f"- [{n.get('type', 'unknown')}] {n.get('text', '')}"
            for n in nonces[:10]
        ])

        prompt = self.VERIFICATION_PROMPT.format(claim=claim, nonces=nonce_text)
        try:
            raw = self.client.generate(
                prompt, system=self.SYSTEM_PROMPT, json_mode=True, temperature=0.2
            )
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception:
            pass

        return self._fallback_verify(claim, nonces)

    def _fallback_verify(self, claim: str, nonces: list[dict[str, Any]]) -> dict[str, Any]:
        """Rule-based verification when LLM unavailable."""
        claim_lower = claim.lower()
        support_count = 0
        for n in nonces:
            text = n.get("text", "").lower()
            # Simple overlap check
            claim_words = set(claim_lower.split())
            text_words = set(text.split())
            overlap = len(claim_words & text_words)
            if overlap >= 2:
                support_count += 1

        if support_count >= 2:
            return {
                "verdict": "supported",
                "confidence": "medium",
                "reasoning": f"{support_count} evidence snippets overlap with claim keywords.",
            }
        elif support_count >= 1:
            return {
                "verdict": "weakly_supported",
                "confidence": "medium",
                "reasoning": "Limited evidence overlap with claim.",
            }
        return {
            "verdict": "insufficient_evidence",
            "confidence": "high",
            "reasoning": "No strong evidence overlap found.",
        }
