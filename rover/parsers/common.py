"""Common parsing utilities shared across agent parsers."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from rover.models import Message, ToolUse


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield decoded JSON objects from a JSONL file."""
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def parse_iso_timestamp(ts: str | int | float | None) -> datetime | None:
    """Parse various timestamp formats to datetime."""
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        # Assume milliseconds if > 1e12
        if ts > 1e12:
            ts = ts / 1000.0
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    if isinstance(ts, str):
        ts = ts.strip()
        # Try ISO8601 variants
        for fmt in (
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
        ):
            try:
                return datetime.strptime(ts, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        # Try pure numeric (epoch seconds or ms)
        try:
            num = float(ts)
            if num > 1e12:
                num = num / 1000.0
            return datetime.fromtimestamp(num, tz=timezone.utc)
        except ValueError:
            pass
    return None


def normalize_tool_name(name: str | None) -> str:
    """Map agent-specific tool names to canonical names."""
    from rover.config import TOOL_NORMALIZATION
    if not name:
        return "Other"
    name = name.strip()
    return TOOL_NORMALIZATION.get(name, name)


def extract_text_content(content: str | list[dict[str, Any]] | None) -> str:
    """Extract plain text from message content (string or structured blocks)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    texts: list[str] = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                texts.append(block.get("text", ""))
            elif "content" in block and isinstance(block["content"], str):
                texts.append(block["content"])
    return " ".join(texts)


def parse_messages_from_blocks(blocks: list[dict[str, Any]]) -> list[Message]:
    """Parse a list of content blocks into Messages with tool uses."""
    messages: list[Message] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        msg_type = block.get("type", "")
        if msg_type == "user":
            messages.append(
                Message(
                    role="user",
                    content=block.get("message", {}).get("content", ""),
                    timestamp=parse_iso_timestamp(block.get("timestamp")),
                )
            )
        elif msg_type == "assistant":
            tool_use = None
            content = block.get("message", {}).get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "tool_use":
                        tool_use = ToolUse(
                            name=normalize_tool_name(item.get("name", "")),
                            input=item.get("input", {}),
                            tool_use_id=item.get("id"),
                        )
            messages.append(
                Message(
                    role="assistant",
                    content=content,
                    timestamp=parse_iso_timestamp(block.get("timestamp")),
                    tool_use=tool_use,
                )
            )
        elif msg_type == "thinking":
            messages.append(
                Message(
                    role="assistant",
                    content=block.get("thinking", ""),
                    timestamp=parse_iso_timestamp(block.get("timestamp")),
                )
            )
    return messages
