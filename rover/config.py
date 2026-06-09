"""Configuration and constants for Rover."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ArchetypeThresholds:
    """Thresholds for archetype classification."""

    name: str
    description: str
    standout_traits: list[str]
    growth_edges: list[str]
    rules: dict[str, Any] = field(default_factory=dict)


ARCHETYPES: dict[str, ArchetypeThresholds] = {
    "Architect": ArchetypeThresholds(
        name="Architect",
        description=(
            "Plans first, codifies decisions, builds scaffolding. "
            "Uses exploration tools (Glob, Grep) before writing code. Low code churn."
        ),
        standout_traits=[
            "High planning-to-coding ratio",
            "Long, detailed prompts with architectural context",
            "Prefers 2-4 hour deep focus sessions",
            "Low code churn — gets it right the first time",
        ],
        growth_edges=[
            "Ship faster by reducing upfront planning",
            "Embrace iterative prototyping over perfect design",
        ],
        rules={
            "planning_ratio_min": 1.5,
            "avg_session_duration_min": 90,  # minutes
            "code_velocity_max": 300,
        },
    ),
    "Sprinter": ArchetypeThresholds(
        name="Sprinter",
        description=(
            "High code velocity, rapid iteration, ships quickly. "
            "Short action-oriented prompts, many short sessions."
        ),
        standout_traits=[
            "High code velocity — writes code fast",
            "Short, action-oriented prompts",
            "Many 30-60 minute sessions",
            "Rapid iteration and shipping",
        ],
        growth_edges=[
            "Add more planning before coding",
            "Write tests and documentation",
            "Review code before committing",
        ],
        rules={
            "planning_ratio_max": 0.8,
            "code_velocity_min": 400,
            "avg_prompt_length_max": 80,
        },
    ),
    "Debugger": ArchetypeThresholds(
        name="Debugger",
        description=(
            "Reactive to errors, methodical investigation, frequent test execution. "
            "High error-recovery ratio, Read-heavy workflow."
        ),
        standout_traits=[
            "Methodical investigation of errors",
            "High error-recovery ratio",
            "Frequent test execution",
            "Read-heavy — studies code before changing",
        ],
        growth_edges=[
            "Add preventive testing (TDD)",
            "Write more precise initial prompts",
        ],
        rules={
            "error_recovery_count_min": 3,
            "read_ratio_min": 0.4,
            "iteration_depth_min": 3.0,
        },
    ),
    "Collaborator": ArchetypeThresholds(
        name="Collaborator",
        description=(
            "Asks clarifying questions, seeks alignment, lower autonomous execution. "
            "Values consensus and shared understanding."
        ),
        standout_traits=[
            "High question-asking rate",
            "Alignment-focused prompts ('should we...', 'which approach...')",
            "Seeks clarification before building",
            "Lower autonomous execution",
        ],
        growth_edges=[
            "Make more decisions independently",
            "Ship prototypes before full alignment",
        ],
        rules={
            "ask_user_ratio_min": 0.15,
            "autonomy_score_max": 0.6,
        },
    ),
    "Autonomous Agent": ArchetypeThresholds(
        name="Autonomous Agent",
        description=(
            "Long self-directed sessions, minimal human intervention, high tool diversity. "
            "Gives broad goals and lets the AI run."
        ),
        standout_traits=[
            "Minimal human intervention during sessions",
            "High tool diversity",
            "Broad, self-directed prompts",
            "Reviews output rather than writing code directly",
        ],
        growth_edges=[
            "Add more steering checkpoints",
            "Verify critical decisions manually",
        ],
        rules={
            "autonomy_score_min": 0.85,
            "tool_diversity_min": 6,
            "avg_session_duration_min": 60,
        },
    ),
}


# Scoring rubric for 5 dimensions (0-100)
# Each dimension is computed from features; these weights map feature → score
SCORING_RUBRIC: dict[str, dict[str, Any]] = {
    "steering": {
        "description": "Frequency and quality of course corrections",
        "features": ["error_recovery_count", "tool_diversity", "ask_user_count"],
        "weights": {"error_recovery_count": 0.4, "tool_diversity": 0.3, "ask_user_count": 0.3},
    },
    "execution": {
        "description": "Tool use efficiency and command success rate",
        "features": ["code_velocity", "tool_histogram", "avg_session_duration"],
        "weights": {"code_velocity": 0.5, "tool_histogram": 0.3, "avg_session_duration": 0.2},
    },
    "engineering": {
        "description": "Code quality, testing, and structural thinking",
        "features": ["iteration_depth", "planning_ratio", "tool_histogram"],
        "weights": {"iteration_depth": 0.3, "planning_ratio": 0.4, "tool_histogram": 0.3},
    },
    "product_thinking": {
        "description": "User focus, architectural language, and trade-off awareness",
        "features": ["ask_user_count", "planning_ratio", "prompt_style"],
        "weights": {"ask_user_count": 0.3, "planning_ratio": 0.4, "prompt_style": 0.3},
    },
    "planning": {
        "description": "Pre-coding research depth and strategic thinking",
        "features": ["planning_ratio", "avg_prompt_length", "tool_histogram"],
        "weights": {"planning_ratio": 0.5, "avg_prompt_length": 0.2, "tool_histogram": 0.3},
    },
}

# Regex patterns for redacting sensitive data before display/export
REDACTION_PATTERNS: list[str] = [
    r'sk-[a-zA-Z0-9]{48}',  # OpenAI API keys
    r'[a-zA-Z0-9_-]*api[_-]?key[a-zA-Z0-9_-]*[=:]\s*["\']?[a-zA-Z0-9_\-\.]+["\']?',  # generic API key
    r'gh[pousr]_[A-Za-z0-9_]{36,}',  # GitHub tokens
    r'AKIA[0-9A-Z]{16}',  # AWS access keys
    r'sk_live_[a-zA-Z0-9]{24,}',  # Stripe secret keys
    r'eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*',  # JWTs
    r'[a-f0-9]{32,64}',  # hex hashes that might be secrets
    r'[A-Za-z0-9+/]{40,}={0,2}',  # base64 blobs
]

# Tool name normalization map (varies per agent)
TOOL_NORMALIZATION: dict[str, str] = {
    # Claude Code
    "read_file": "Read",
    "edit_file": "Edit",
    "write_file": "Write",
    "run_terminal_command": "Bash",
    "run_terminal_cmd": "Bash",
    "ripgrep_search": "Grep",
    "grep_search": "Grep",
    "glob_file_search": "Glob",
    "file_search": "Glob",
    "list_dir": "LS",
    "task": "Task",
    "ask_user": "AskUserQuestion",
    "ask_followup_question": "AskUserQuestion",
    "explore": "Explore",
    # Cursor
    "read_file_v2": "Read",
    "edit_file_v2": "Edit",
    "run_terminal_command_v2": "Bash",
    "run_terminal_cmd_v2": "Bash",
    "ripgrep_raw_search": "Grep",
    "grep": "Grep",
    "glob_file_search": "Glob",
    "list_dir": "LS",
    "task_v2": "Task",
    "search_replace": "Edit",
    "apply_patch": "Edit",
    "reapply": "Edit",
    # Codex
    "file_read": "Read",
    "file_edit": "Edit",
    "shell": "Bash",
    "shell_command": "Bash",
    "search": "Grep",
    "find": "Glob",
    "ls": "LS",
    "subagent": "Task",
}


def get_transcript_paths(
    claude_override: Path | str | None = None,
    cursor_override: Path | str | None = None,
    codex_override: Path | str | None = None,
) -> dict[str, Path]:
    """Return transcript directories per agent.

    Priority:
    1. CLI/env overrides (if provided)
    2. Docker-mounted paths
    3. Default system paths
    """
    paths: dict[str, Path] = {}

    # --- Override paths (from CLI or env vars) ---
    if claude_override:
        p = Path(claude_override)
        if p.exists():
            paths["claude"] = p

    if cursor_override:
        p = Path(cursor_override)
        if p.exists():
            paths["cursor_workspace"] = p

    if codex_override:
        p = Path(codex_override)
        if p.exists():
            paths["codex"] = p

    # If any overrides were provided, only use those (don't fall back)
    if paths:
        return paths

    # --- Docker-mounted paths ---
    if os.environ.get("ROVER_IN_DOCKER"):
        claude_docker = Path("/claude/projects")
        if claude_docker.exists():
            paths["claude"] = claude_docker

        codex_docker = Path("/codex/sessions")
        if codex_docker.exists():
            paths["codex"] = codex_docker

        cursor_ws_docker = Path("/cursor/workspaceStorage")
        if cursor_ws_docker.exists():
            paths["cursor_workspace"] = cursor_ws_docker

        cursor_global_docker = Path("/cursor/globalStorage/state.vscdb")
        if cursor_global_docker.exists():
            paths["cursor_global"] = cursor_global_docker

        if paths:
            return paths

    # --- Default system paths ---
    home = Path.home()

    # Claude Code
    claude_dir = home / ".claude" / "projects"
    if claude_dir.exists():
        paths["claude"] = claude_dir

    # Codex CLI
    codex_dir = home / ".codex" / "sessions"
    if codex_dir.exists():
        paths["codex"] = codex_dir

    # Cursor (OS-aware)
    if sys.platform == "darwin":
        cursor_dir = home / "Library" / "Application Support" / "Cursor" / "User" / "workspaceStorage"
        cursor_global = home / "Library" / "Application Support" / "Cursor" / "User" / "globalStorage" / "state.vscdb"
    elif sys.platform.startswith("win") or os.environ.get("MSYSTEM"):
        appdata = Path(os.environ.get("APPDATA", home / "AppData" / "Roaming"))
        cursor_dir = appdata / "Cursor" / "User" / "workspaceStorage"
        cursor_global = appdata / "Cursor" / "User" / "globalStorage" / "state.vscdb"
    else:
        cursor_dir = home / ".config" / "Cursor" / "User" / "workspaceStorage"
        cursor_global = home / ".config" / "Cursor" / "User" / "globalStorage" / "state.vscdb"

    if cursor_dir.exists():
        paths["cursor_workspace"] = cursor_dir
    if cursor_global.exists():
        paths["cursor_global"] = cursor_global

    # opencode
    opencode_dir = home / ".local" / "share" / "opencode"
    if "XDG_DATA_HOME" in os.environ:
        opencode_dir = Path(os.environ["XDG_DATA_HOME"]) / "opencode"
    if opencode_dir.exists():
        paths["opencode"] = opencode_dir

    # Gemini CLI
    gemini_dir = home / ".gemini" / "tmp"
    if gemini_dir.exists():
        paths["gemini"] = gemini_dir

    return paths
