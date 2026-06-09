"""Parser for Codex CLI transcripts."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from rover.models import AgentType, Message, Session, ToolUse, ToolResult
from rover.parsers.common import extract_text_content, iter_jsonl, normalize_tool_name, parse_iso_timestamp


def parse_codex_sessions(codex_dir: Path, since: datetime | None = None) -> list[Session]:
    """Parse all Codex CLI sessions from ~/.codex/sessions/."""
    sessions: list[Session] = []
    if not codex_dir.exists():
        return sessions

    jsonl_files = list(codex_dir.rglob("*.jsonl"))
    for path in jsonl_files:
        if since is not None:
            mtime = path.stat().st_mtime
            if mtime < since.timestamp():
                continue
        session = _parse_codex_jsonl(path)
        if session.messages:
            sessions.append(session)
    return sessions


def _parse_codex_jsonl(path: Path) -> Session:
    """Parse a single Codex CLI JSONL session file."""
    session_id = path.stem
    messages: list[Message] = []
    file_paths: set[str] = set()
    start_time: datetime | None = None
    end_time: datetime | None = None
    cwd: str = ""
    git_remote: str = ""
    originator: str = ""

    for record in iter_jsonl(path):
        if not isinstance(record, dict):
            continue

        # Codex session_meta is often the first line
        if record.get("type") == "session_meta":
            payload = record.get("payload", record)
            cwd = payload.get("cwd", "")
            git_remote = payload.get("git", {}).get("repository_url", "")
            originator = payload.get("originator", "")
            continue

        # Also handle flat format where session meta is top-level
        if "cwd" in record and "repository_url" in str(record):
            cwd = record.get("cwd", "")
            git_remote = record.get("git", {}).get("repository_url", "")
            originator = record.get("originator", "")
            continue

        ts = parse_iso_timestamp(record.get("timestamp"))
        if ts:
            if start_time is None or ts < start_time:
                start_time = ts
            if end_time is None or ts > end_time:
                end_time = ts

        msg_type = record.get("type", "")
        role = record.get("role", "")
        content = record.get("content", "")

        if role == "user":
            text = extract_text_content(content)
            messages.append(
                Message(
                    role="user",
                    content=text,
                    timestamp=ts,
                )
            )
        elif role == "assistant":
            tool_use = None
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "tool_use":
                        tool_name = normalize_tool_name(item.get("name", ""))
                        tool_input = item.get("input", {})
                        if isinstance(tool_input, dict):
                            for key in ("file_path", "path", "targetFile"):
                                fp = tool_input.get(key)
                                if fp and isinstance(fp, str):
                                    file_paths.add(fp)
                        tool_use = ToolUse(
                            name=tool_name,
                            input=tool_input,
                            tool_use_id=item.get("id"),
                        )
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
            is_error = record.get("is_error", False)
            result_content = record.get("result", "")
            if isinstance(result_content, dict):
                for key in ("file_path", "path"):
                    fp = result_content.get(key)
                    if fp and isinstance(fp, str):
                        file_paths.add(fp)
            messages.append(
                Message(
                    role="user",
                    content=extract_text_content(result_content),
                    timestamp=ts,
                    tool_result=ToolResult(
                        content=result_content,
                        tool_use_id=record.get("tool_use_id"),
                        is_error=is_error,
                    ),
                )
            )

    return Session(
        session_id=session_id,
        agent_type=AgentType.CODEX,
        project_name=Path(cwd).name if cwd else "",
        cwd=cwd,
        git_remote=git_remote,
        messages=messages,
        start_time=start_time,
        end_time=end_time,
        file_paths=sorted(file_paths),
    )
