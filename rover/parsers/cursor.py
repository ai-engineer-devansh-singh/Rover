"""Parser for Cursor IDE transcripts from SQLite state.vscdb."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from rover.models import AgentType, Message, Session, ToolUse, ToolResult
from rover.parsers.common import extract_text_content, normalize_tool_name, parse_iso_timestamp


def parse_cursor_workspace(workspace_dir: Path, since: datetime | None = None) -> list[Session]:
    """Parse Cursor sessions from a workspace directory."""
    db_path = workspace_dir / "state.vscdb"
    if not db_path.exists():
        return []
    return _extract_cursor_db(db_path, since)


def parse_cursor_global_db(db_path: Path, since: datetime | None = None) -> list[Session]:
    """Parse Cursor sessions from the global state.vscdb."""
    return _extract_cursor_db(db_path, since)


def _extract_cursor_db(db_path: Path, since: datetime | None = None) -> list[Session]:
    """Extract sessions from a Cursor SQLite database."""
    sessions: list[Session] = []
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Check schema
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        if "cursorDiskKV" not in tables:
            conn.close()
            return []

        # Get composer sessions
        cursor.execute("SELECT key, value FROM cursorDiskKV WHERE key LIKE 'composerData:%'")
        composer_rows = cursor.fetchall()

        for key, value in composer_rows:
            if not value:
                continue
            try:
                composer_data = json.loads(value)
            except json.JSONDecodeError:
                continue

            composer_id = composer_data.get("composerId", "")
            if not composer_id:
                continue

            # Filter by --since using createdAt (milliseconds epoch)
            if since is not None:
                created_at_ms = composer_data.get("createdAt", 0)
                created_at_s = created_at_ms / 1000.0 if created_at_ms else 0
                if created_at_s < since.timestamp():
                    continue

            # Get workspace path
            workspace_path = ""
            ws_uri = composer_data.get("workspaceIdentifier", {}).get("uri", {})
            if ws_uri:
                workspace_path = ws_uri.get("fsPath", "")

            # Get bubble IDs
            bubble_ids = []
            for header in composer_data.get("fullConversationHeadersOnly", []):
                bid = header.get("bubbleId")
                if bid:
                    bubble_ids.append(bid)
            if not bubble_ids:
                continue

            # Fetch bubbles
            messages: list[Message] = []
            file_paths: set[str] = set()
            start_time: datetime | None = None
            end_time: datetime | None = None

            for bubble_id in bubble_ids:
                # Modern key form
                bubble_key = f"bubbleId:{composer_id}:{bubble_id}"
                cursor.execute(
                    "SELECT value FROM cursorDiskKV WHERE key = ?",
                    (bubble_key,),
                )
                row = cursor.fetchone()
                if not row:
                    # Legacy bare key
                    cursor.execute(
                        "SELECT value FROM cursorDiskKV WHERE key = ?",
                        (f"bubbleId:{bubble_id}",),
                    )
                    row = cursor.fetchone()
                if not row:
                    continue

                try:
                    bubble = json.loads(row[0])
                except json.JSONDecodeError:
                    continue

                ts_raw = bubble.get("timingInfo", {}).get("clientEndTime") or bubble.get("createdAt")
                ts = parse_iso_timestamp(ts_raw)
                if ts:
                    if start_time is None or ts < start_time:
                        start_time = ts
                    if end_time is None or ts > end_time:
                        end_time = ts

                # Parse bubble content
                text = bubble.get("text", "")
                thinking = bubble.get("thinking", "")
                tfd = bubble.get("toolFormerData", {})

                # Determine role from bubble type
                bubble_type = str(bubble.get("type", ""))
                role = "assistant" if bubble_type == "2" else "user"

                if role == "user" and not tfd:
                    if text:
                        messages.append(
                            Message(
                                role="user",
                                content=text,
                                timestamp=ts,
                            )
                        )
                else:
                    content_blocks: list[dict[str, Any]] = []
                    if thinking:
                        content_blocks.append({"type": "thinking", "thinking": thinking})
                    if text:
                        content_blocks.append({"type": "text", "text": text})

                    tool_use = None
                    if tfd:
                        tool_name = normalize_tool_name(tfd.get("name", ""))
                        params = tfd.get("params", {})
                        if isinstance(params, str):
                            try:
                                params = json.loads(params)
                            except json.JSONDecodeError:
                                params = {}
                        if not isinstance(params, dict):
                            params = {}

                        # Strip streamingContent
                        params.pop("streamingContent", None)

                        # Extract file paths
                        for key in ("targetFile", "relativeWorkspacePath", "file_path", "path"):
                            fp = params.get(key)
                            if fp and isinstance(fp, str):
                                file_paths.add(fp)

                        tool_call_id = tfd.get("toolCallId")
                        tool_use = ToolUse(
                            name=tool_name,
                            input=params,
                            tool_use_id=tool_call_id,
                        )
                        content_blocks.append(
                            {
                                "type": "tool_use",
                                "name": tool_name,
                                "input": params,
                                "id": tool_call_id,
                            }
                        )

                        # Tool result
                        result_raw = tfd.get("result", "")
                        if result_raw:
                            if isinstance(result_raw, str):
                                try:
                                    result_json = json.loads(result_raw)
                                except json.JSONDecodeError:
                                    result_json = result_raw
                            else:
                                result_json = result_raw

                            if isinstance(result_json, dict):
                                rt = result_json.get("output") or result_json.get("contents") or result_json.get("result") or str(result_raw)
                            else:
                                rt = str(result_raw)

                            is_error = "error" in str(rt).lower()
                            messages.append(
                                Message(
                                    role="user",
                                    content=rt[:4000],
                                    timestamp=ts,
                                    tool_result=ToolResult(
                                        content=rt[:4000],
                                        tool_use_id=tool_call_id,
                                        is_error=is_error,
                                    ),
                                )
                            )

                    messages.append(
                        Message(
                            role="assistant",
                            content=content_blocks,
                            timestamp=ts,
                            tool_use=tool_use,
                        )
                    )

            if messages:
                sessions.append(
                    Session(
                        session_id=composer_id,
                        agent_type=AgentType.CURSOR,
                        project_name=Path(workspace_path).name if workspace_path else "",
                        cwd=str(workspace_path) if workspace_path else "",
                        git_remote="",
                        messages=messages,
                        start_time=start_time,
                        end_time=end_time,
                        file_paths=sorted(file_paths),
                    )
                )

        conn.close()
    except Exception:
        # Graceful degradation if SQLite fails
        pass

    return sessions


from datetime import datetime
