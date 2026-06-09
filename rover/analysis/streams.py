"""Multi-day work stream grouping for sessions with LLM-powered descriptions.

Paxel Step 9: "Grouping sessions into multi-day work streams: 101 work streams"

Paxel likely generates meaningful descriptions for each work stream using LLM,
rather than just "{N} sessions on {project}". This helps users understand
their multi-day development arcs.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from typing import Any

from rover.llm.client import LLMClient
from rover.models import Session, WorkStream


class WorkStreamAnalyzer:
    """Group sessions into work streams and generate LLM-powered descriptions."""

    SYSTEM_PROMPT = (
        "You are a product manager analyzing a developer's work patterns. "
        "Describe multi-day work streams concisely. Focus on the overarching goal and outcome. "
        "Respond in valid JSON only."
    )

    STREAM_PROMPT = """You are looking at a series of coding sessions that form a single work stream.

PROJECT: {project}
START: {start}
END: {end}
SESSIONS: {session_count}

SESSION NARRATIVES:
{narratives}

Describe this work stream in 1-2 sentences. What was the overarching goal?
What was the outcome? Classify the theme.

Return JSON:
{{
  "description": "1-2 sentence summary of the entire work stream",
  "theme": "feature|bugfix|refactor|exploration|infrastructure|docs|spike",
  "success": "success|partial|ongoing|abandoned",
  "complexity": "low|medium|high",
  "technologies": ["tech1", "tech2"]
}}
"""

    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or LLMClient()

    def describe_stream(self, stream: WorkStream) -> dict[str, Any]:
        """Generate an LLM-powered description for a work stream."""
        if not self.client.is_available() or not stream.sessions:
            return self._fallback_description(stream)

        # Collect narratives from sessions
        narratives = []
        technologies: set[str] = set()
        for i, session in enumerate(stream.sessions):
            summary = session.summary or {}
            nar = summary.get("narrative", f"Session {i+1}")
            narratives.append(f"- {nar}")
            techs = summary.get("technologies", [])
            technologies.update(techs)

        if not narratives:
            return self._fallback_description(stream)

        prompt = self.STREAM_PROMPT.format(
            project=stream.project_name,
            start=stream.start_time.strftime("%Y-%m-%d %H:%M") if stream.start_time else "unknown",
            end=stream.end_time.strftime("%Y-%m-%d %H:%M") if stream.end_time else "unknown",
            session_count=len(stream.sessions),
            narratives="\n".join(narratives[:15]),
        )

        try:
            raw = self.client.generate(
                prompt, system=self.SYSTEM_PROMPT, json_mode=True, temperature=0.3
            )
            result = self._extract_json(raw)
            if result:
                return {
                    "description": result.get("description", ""),
                    "theme": result.get("theme", "feature"),
                    "success": result.get("success", "ongoing"),
                    "complexity": result.get("complexity", "medium"),
                    "technologies": result.get("technologies", []),
                    "source": "llm",
                }
        except Exception:
            pass

        return self._fallback_description(stream)

    def _fallback_description(self, stream: WorkStream) -> dict[str, Any]:
        """Rule-based work stream description."""
        technologies: set[str] = set()
        for session in stream.sessions:
            techs = session.summary.get("technologies", []) if session.summary else []
            technologies.update(techs)

        return {
            "description": f"{len(stream.sessions)} sessions on {stream.project_name}",
            "theme": "feature",
            "success": "ongoing",
            "complexity": "medium",
            "technologies": sorted(technologies),
            "source": "fallback",
        }

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


def group_work_streams(sessions: list[Session], gap_hours: int = 48) -> list[WorkStream]:
    """Group sessions into multi-day work streams.

    A work stream is a sequence of sessions in the same project
    where the gap between consecutive sessions is < gap_hours.
    """
    if not sessions:
        return []

    # Sort by start time
    sorted_sessions = sorted(sessions, key=lambda s: s.start_time or datetime.min)

    streams: list[WorkStream] = []
    current_stream: list[Session] = []
    current_project: str = ""

    for session in sorted_sessions:
        if session.is_subagent or session.is_too_short:
            continue

        if not current_stream:
            current_stream = [session]
            current_project = session.project_name or session.cwd or "unknown"
            continue

        # Check if this session belongs to the current stream
        last_session = current_stream[-1]
        gap = timedelta(hours=gap_hours)
        same_project = (session.project_name or session.cwd) == current_project
        within_gap = (
            session.start_time is not None
            and last_session.end_time is not None
            and (session.start_time - last_session.end_time) < gap
        )

        if same_project and within_gap:
            current_stream.append(session)
        else:
            # Finish current stream
            streams.append(_create_stream(current_stream, current_project))
            current_stream = [session]
            current_project = session.project_name or session.cwd or "unknown"

    if current_stream:
        streams.append(_create_stream(current_stream, current_project))

    # Enrich streams with LLM descriptions
    analyzer = WorkStreamAnalyzer()
    for stream in streams:
        desc = analyzer.describe_stream(stream)
        stream.description = desc.get("description", stream.description)
        stream.commit_count = sum(
            len(s.file_paths) for s in stream.sessions if s.file_paths
        )

    return streams


def _create_stream(sessions: list[Session], project: str) -> WorkStream:
    """Create a WorkStream from a list of sessions."""
    start = min((s.start_time for s in sessions if s.start_time), default=None)
    end = max((s.end_time for s in sessions if s.end_time), default=None)
    return WorkStream(
        stream_id=f"stream_{hash(project + str(start))}",
        project_name=project,
        start_time=start,
        end_time=end,
        sessions=sessions,
        description=f"{len(sessions)} sessions on {project}",
    )
