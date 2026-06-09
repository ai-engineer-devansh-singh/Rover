"""LLM-powered session summarization — Stage 1 of Paxel's two-stage pipeline.

Paxel Step 7: "Summarizing each session [cloud: gpt-5.5]: 182/182 summaries"

Stage 1 (NarrativeAnalyzer) produces a rich, structured narrative summary.
This narrative is consumed by Stage 2 (EpisodeAnalyzer) for decisions, steering, and scoring.
The cache hit rate is 88% because narratives are deterministic per transcript.
"""
from __future__ import annotations

import json
import re
from typing import Any

from rover.llm.client import LLMClient
from rover.models import Session
from rover.parsers.common import extract_text_content


class SessionSummarizer:
    """Generate per-session narratives using an LLM. Implements Paxel Stage 1."""

    SYSTEM_PROMPT = (
        "You are a senior software engineering analyst reviewing AI coding session transcripts. "
        "Your summaries are factual, specific, and never include source code. "
        "Focus on what was built, which technologies were used, and what was achieved. "
        "Respond in valid JSON only."
    )

    NARRATIVE_PROMPT = """Analyze the following AI coding session transcript and produce a structured narrative summary.

TRANSCRIPT FORMAT:
- USER: User prompts (what they asked for)
- AI_TOOL: Tool calls made by the AI (Read, Edit, Bash, etc.)
- AI_RESULT: Results from tool executions (OK or ERROR)
- AI: Direct assistant responses

TRANSCRIPT:
{transcript}

INSTRUCTIONS:
- Be specific about technologies, frameworks, and patterns used
- Note if the session was exploratory, implementation-focused, or debugging
- Do NOT include source code in your response
- If the session appears incomplete or was abandoned, note this

Return exactly this JSON structure:
{{
  "narrative": "2-3 sentence factual summary of the session",
  "goal": "The primary objective or feature being implemented",
  "approach": "The technical approach taken (e.g., 'TDD with pytest', 'exploratory refactoring')",
  "tools_used": ["Read", "Edit", "Bash", "Glob", ...],
  "technologies": ["Next.js", "Prisma", "OAuth2", ...],
  "complexity": "low|medium|high",
  "outcome": "success|partial|failure|exploratory|abandoned",
  "key_files": ["src/auth.ts", "tests/auth.test.ts"],
  "duration_estimate_minutes": 45,
  "mood": "focused|struggling|exploratory|frustrated|confident",
  "completion_level": "0-100 estimate of how complete the task is"
}}
"""

    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or LLMClient()
        self._cache: dict[str, dict[str, Any]] = {}

    def summarize(self, session: Session) -> dict[str, Any]:
        """Generate a Stage 1 narrative summary for a session.

        Returns a rich dict consumed by Stage 2 analyzers (decisions, steering, scorer).
        """
        cache_key = f"{session.session_id}:{session.agent_type.value}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if not self.client.is_available():
            return self._fallback_summary(session)

        # Build compressed transcript (privacy-safe)
        transcript = self._build_transcript(session)
        if not transcript:
            return self._fallback_summary(session)

        prompt = self.NARRATIVE_PROMPT.format(transcript=transcript)

        try:
            raw = self.client.generate(
                prompt, system=self.SYSTEM_PROMPT, json_mode=True, temperature=0.3
            )
            result = self._extract_json(raw)
            if result:
                summary = self._normalize_summary(result)
                self._cache[cache_key] = summary
                return summary
        except Exception:
            pass

        return self._fallback_summary(session)

    def _build_transcript(self, session: Session) -> str:
        """Build a privacy-safe compressed transcript for the LLM."""
        lines: list[str] = []
        for msg in session.messages:
            if msg.role == "user":
                text = extract_text_content(msg.content)
                if text:
                    lines.append(f"USER: {text[:500]}")
            elif msg.role == "assistant":
                if msg.tool_use:
                    name = msg.tool_use.name
                    # Only include tool name + key metadata, not full content
                    inp = msg.tool_use.input
                    if isinstance(inp, dict):
                        meta_keys = ["command", "file_path", "path", "pattern", "targetFile", "relativeWorkspacePath"]
                        meta = {k: v for k, v in inp.items() if k in meta_keys and v}
                        meta_str = json.dumps(meta, default=str)[:150] if meta else ""
                        lines.append(f"AI_TOOL: {name}({meta_str})")
                    else:
                        lines.append(f"AI_TOOL: {name}")
                elif msg.tool_result:
                    status = "ERROR" if msg.tool_result.is_error else "OK"
                    lines.append(f"AI_RESULT: {status}")
                else:
                    text = extract_text_content(msg.content)
                    if text:
                        lines.append(f"AI: {text[:300]}")
        # Cap at ~30 exchanges to keep prompt size reasonable
        return "\n".join(lines[:60])

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        """Extract JSON object from LLM response text."""
        # Try direct JSON parse first
        text = text.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass
        # Fallback: regex extract
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return None

    def _normalize_summary(self, result: dict[str, Any]) -> dict[str, Any]:
        """Normalize LLM output to standard schema."""
        return {
            "narrative": result.get("narrative", ""),
            "goal": result.get("goal", ""),
            "approach": result.get("approach", ""),
            "tools_used": result.get("tools_used", []),
            "technologies": result.get("technologies", []),
            "complexity": result.get("complexity", "medium"),
            "outcome": result.get("outcome", "unknown"),
            "key_files": result.get("key_files", []),
            "duration_estimate_minutes": result.get("duration_estimate_minutes", 0),
            "mood": result.get("mood", "focused"),
            "completion_level": result.get("completion_level", 50),
            "source": "llm",
        }

    def _fallback_summary(self, session: Session) -> dict[str, Any]:
        """Generate a rule-based summary when LLM is unavailable."""
        tools = set()
        user_prompts = []
        error_count = 0
        for msg in session.messages:
            if msg.role == "user":
                text = extract_text_content(msg.content)
                if text:
                    user_prompts.append(text)
            elif msg.role == "assistant":
                if msg.tool_use:
                    tools.add(msg.tool_use.name)
                if msg.tool_result and msg.tool_result.is_error:
                    error_count += 1

        goal = user_prompts[0][:200] if user_prompts else "Unknown task"

        # Detect outcome heuristically
        outcome = "exploratory"
        if error_count >= 3:
            outcome = "failure" if error_count >= 5 else "partial"
        elif len(user_prompts) >= 5 and len(tools) >= 3:
            outcome = "success"

        # Detect complexity
        complexity = "low"
        if len(tools) >= 5 or len(user_prompts) >= 10:
            complexity = "medium"
        if len(tools) >= 8 or len(user_prompts) >= 20:
            complexity = "high"

        return {
            "narrative": f"Session with {len(user_prompts)} prompts and {len(tools)} distinct tools. Goal: {goal}",
            "goal": goal,
            "approach": "",
            "tools_used": sorted(tools),
            "technologies": [],
            "complexity": complexity,
            "outcome": outcome,
            "key_files": session.file_paths[:10],
            "duration_estimate_minutes": int(session.duration_minutes),
            "mood": "focused",
            "completion_level": 50,
            "source": "fallback",
        }
