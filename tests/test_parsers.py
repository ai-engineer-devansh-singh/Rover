"""Tests for transcript parsers."""
from __future__ import annotations

from pathlib import Path

import pytest

from rover.models import AgentType
from rover.parsers.claude import parse_claude_project
from rover.parsers.common import extract_text_content, normalize_tool_name, parse_iso_timestamp


FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestCommon:
    def test_parse_iso_timestamp_seconds(self):
        dt = parse_iso_timestamp("2025-06-01T10:00:00Z")
        assert dt is not None
        assert dt.year == 2025
        assert dt.hour == 10

    def test_parse_iso_timestamp_millis(self):
        dt = parse_iso_timestamp(1717236000000)
        assert dt is not None
        assert dt.year == 2024

    def test_normalize_tool_name(self):
        assert normalize_tool_name("read_file") == "Read"
        assert normalize_tool_name("edit_file_v2") == "Edit"
        assert normalize_tool_name("unknown_tool") == "unknown_tool"

    def test_extract_text_content_string(self):
        assert extract_text_content("hello") == "hello"

    def test_extract_text_content_blocks(self):
        blocks = [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]
        assert extract_text_content(blocks) == "hello world"


class TestClaudeParser:
    def test_parse_fixture(self):
        project_dir = FIXTURES_DIR
        sessions = parse_claude_project(project_dir)
        # Should find at least the claude_session.jsonl
        assert len(sessions) >= 1
        session = sessions[0]
        assert session.agent_type == AgentType.CLAUDE
        assert session.prompt_count > 0
        assert session.tool_use_count > 0
