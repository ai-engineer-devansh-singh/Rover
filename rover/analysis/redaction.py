"""Regex-based redaction of sensitive data before display/export."""
from __future__ import annotations

import re

from rover.config import REDACTION_PATTERNS


def redact(text: str) -> str:
    """Apply redaction patterns to a string."""
    if not text:
        return text
    for pattern in REDACTION_PATTERNS:
        text = re.sub(pattern, "[REDACTED]", text)
    return text


def redact_dict(data: dict) -> dict:
    """Recursively redact strings in a dictionary."""
    result: dict = {}
    for key, value in data.items():
        if isinstance(value, str):
            result[key] = redact(value)
        elif isinstance(value, dict):
            result[key] = redact_dict(value)
        elif isinstance(value, list):
            result[key] = [redact(v) if isinstance(v, str) else v for v in value]
        else:
            result[key] = value
    return result
