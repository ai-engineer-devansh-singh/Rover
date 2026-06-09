"""Discovery of transcript files across supported AI coding agents."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from rover.config import get_transcript_paths
from rover.models import AgentType, Session, SessionCounts
from rover.parsers.claude import parse_claude_project
from rover.parsers.codex import parse_codex_sessions
from rover.parsers.cursor import parse_cursor_global_db, parse_cursor_workspace


def discover_sessions(
    project_name: str | None = None,
    since: datetime | None = None,
    claude_path: Path | str | None = None,
    cursor_path: Path | str | None = None,
    codex_path: Path | str | None = None,
) -> tuple[list[Session], SessionCounts]:
    """Discover and parse all transcript sessions matching filters.

    By default, scans ALL projects found on the computer.
    If project_name is set, only scans that specific project/repo.

    Returns (sessions_list, session_counts).
    """
    paths = get_transcript_paths(
        claude_override=claude_path,
        cursor_override=cursor_path,
        codex_override=codex_path,
    )
    sessions: list[Session] = []

    # Claude Code projects
    if "claude" in paths:
        claude_dir = paths["claude"]
        if project_name:
            # Scan specific project only
            target = claude_dir / project_name
            if target.exists():
                sessions.extend(_parse_claude_with_subagents(target, since))
        else:
            # Scan ALL projects (default behavior)
            for project_dir in claude_dir.iterdir():
                if project_dir.is_dir() and not project_dir.name.startswith("."):
                    sessions.extend(_parse_claude_with_subagents(project_dir, since))

    # Cursor (always scan all workspaces)
    if "cursor_workspace" in paths:
        for ws_dir in paths["cursor_workspace"].iterdir():
            if ws_dir.is_dir():
                sessions.extend(parse_cursor_workspace(ws_dir, since))
    if "cursor_global" in paths:
        sessions.extend(parse_cursor_global_db(paths["cursor_global"], since))

    # Codex CLI (always scan all sessions)
    if "codex" in paths:
        sessions.extend(parse_codex_sessions(paths["codex"], since))

    counts = _compute_counts(sessions)
    return sessions, counts


def _parse_claude_with_subagents(project_dir: Path, since: datetime | None) -> list[Session]:
    """Parse a Claude project, tagging subagent sessions."""
    sessions = parse_claude_project(project_dir, since)
    for s in sessions:
        # Claude subagents are in subagents/ subdir
        if "subagents" in s.session_id or "/subagents/" in str(project_dir):
            s.is_subagent = True
    return sessions


def _compute_counts(sessions: list[Session]) -> SessionCounts:
    """Compute session breakdown counts."""
    counts = SessionCounts(total=len(sessions))
    for s in sessions:
        if s.is_subagent:
            counts.subagent += 1
        else:
            counts.main += 1

        if s.is_too_short:
            counts.too_short += 1
        else:
            counts.analyzable += 1

        key = s.agent_type.value
        counts.by_agent[key] = counts.by_agent.get(key, 0) + 1
    return counts


def _detect_repo_name() -> str | None:
    """Detect current git repo name from cwd."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(result.stdout.strip()).name
    except Exception:
        return None


def session_summary(sessions: list[Session]) -> dict[str, Any]:
    """Return a quick summary of discovered sessions."""
    by_agent: dict[str, int] = {}
    for s in sessions:
        key = s.agent_type.value
        by_agent[key] = by_agent.get(key, 0) + 1
    return {
        "total": len(sessions),
        "by_agent": by_agent,
    }
