"""Tests for analysis engine."""
from __future__ import annotations

from pathlib import Path

from rover.analysis.archetypes import classify_archetype
from rover.analysis.features import extract_features
from rover.analysis.scoring import score_dimensions
from rover.models import AgentType, Session
from rover.parsers.claude import parse_claude_project


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_analysis_pipeline():
    sessions = parse_claude_project(FIXTURES_DIR)
    assert len(sessions) >= 1

    features = extract_features(sessions)
    assert features.total_sessions >= 1
    assert features.total_prompts > 0
    assert features.tool_histogram.get("Read", 0) > 0

    scores = score_dimensions(features)
    assert 0 <= scores.steering <= 100
    assert 0 <= scores.execution <= 100
    assert 0 <= scores.engineering <= 100
    assert 0 <= scores.product_thinking <= 100
    assert 0 <= scores.planning <= 100

    archetype = classify_archetype(features, scores)
    assert archetype.archetype in ("Architect", "Sprinter", "Debugger", "Collaborator", "Autonomous Agent")
    assert 0 <= archetype.confidence <= 1


def test_empty_sessions():
    features = extract_features([])
    assert features.total_sessions == 0
    assert features.total_prompts == 0
    scores = score_dimensions(features)
    # Baseline scores even with no data (scoring engine has defaults)
    assert 0 <= scores.average <= 100
