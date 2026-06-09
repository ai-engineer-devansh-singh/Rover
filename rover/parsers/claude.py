"""Parser for Claude Code transcripts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rover.models import AgentType, Message, Session, ToolUse
from rover.parsers.common import extract_text_content, iter_jsonl, normalize_tool_name, parse_iso_timestamp


def parse_claude_project(project_dir: Path, since: datetime | None = None) -> list[Session]:
    """Parse all sessions in a Claude Code project directory."""
    sessions: list[Session] = []
    jsonl_files = [p for p in project_dir.iterdir() if p.suffix == ".jsonl" and not p.name.startswith("_")]
    for jsonl_path in jsonl_files:
        if since is not None:
            mtime = jsonl_path.stat().st_mtime
            if mtime < since.timestamp():
                continue
        session = _parse_claude_jsonl(jsonl_path, project_dir.name)
        if session.messages:
            sessions.append(session)
    return sessions


def _parse_claude_jsonl(path: Path, project_name: str) -> Session:
    """Parse a single Claude Code JSONL session file."""
    session_id = path.stem
    messages: list[Message] = []
    file_paths: set[str] = set()
    start_time: datetime | None = None
    end_time: datetime | None = None
    cwd: str = ""
    git_remote: str = ""

    for record in iter_jsonl(path):
        if not isinstance(record, dict):
            continue
        ts = parse_iso_timestamp(record.get("timestamp"))
        if ts:
            if start_time is None or ts < start_time:
                start_time = ts
            if end_time is None or ts > end_time:
                end_time = ts

        msg_type = record.get("type", "")
        if msg_type == "queue-operation":
            continue

        if msg_type == "user":
            content = record.get("message", {}).get("content", "")
            text = extract_text_content(content)
            messages.append(
                Message(
                    role="user",
                    content=text,
                    timestamp=ts,
                )
            )
        elif msg_type == "assistant":
            content = record.get("message", {}).get("content", [])
            tool_use = None
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "tool_use":
                        tool_name = normalize_tool_name(item.get("name", ""))
                        tool_input = item.get("input", {})
                        tool_use = ToolUse(
                            name=tool_name,
                            input=tool_input,
                            tool_use_id=item.get("id"),
                        )
                        # Extract file paths from tool input
                        if isinstance(tool_input, dict):
                            for key in ("file_path", "path", "targetFile", "relativeWorkspacePath"):
                                fp = tool_input.get(key)
                                if fp and isinstance(fp, str):
                                    file_paths.add(fp)
            text = extract_text_content(content)
            messages.append(
                Message(
                    role="assistant",
                    content=text,
                    timestamp=ts,
                    tool_use=tool_use,
                )
            )
        elif msg_type == "tool_result":
            # Tool results often carry error signals and file references
            content = record.get("result", "")
            if isinstance(content, dict):
                for key in ("file_path", "path"):
                    fp = content.get(key)
                    if fp and isinstance(fp, str):
                        file_paths.add(fp)
            # Error detection
            is_error = record.get("is_error", False) or "error" in str(content).lower()
            messages.append(
                Message(
                    role="user",
                    content=extract_text_content(content),
                    timestamp=ts,
                    tool_result=ToolResult(
                        content=content,
                        tool_use_id=record.get("tool_use_id"),
                        is_error=is_error,
                    ),
                )
            )

        # Extract cwd from top-level fields
        if not cwd:
            _cwd = record.get("cwd", "")
            if _cwd and isinstance(_cwd, str):
                cwd = _cwd

    return Session(
        session_id=session_id,
        agent_type=AgentType.CLAUDE,
        project_name=project_name,
        cwd=cwd,
        git_remote=git_remote,
        messages=messages,
        start_time=start_time,
        end_time=end_time,
        file_paths=sorted(file_paths),
    )


from rover.models import ToolResult
