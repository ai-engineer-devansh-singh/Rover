"""Core data models for Rover analysis."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any


class AgentType(Enum):
    CLAUDE = "claude"
    CURSOR = "cursor"
    CODEX = "codex"
    OPENCODE = "opencode"
    GEMINI = "gemini"
    UNKNOWN = "unknown"


class ToolName(Enum):
    READ = "Read"
    EDIT = "Edit"
    WRITE = "Write"
    BASH = "Bash"
    GREP = "Grep"
    GLOB = "Glob"
    TASK = "Task"
    ASK_USER = "AskUserQuestion"
    EXPLORER = "Explore"
    LS = "LS"
    OTHER = "Other"


@dataclass
class Message:
    """A single message in an AI coding session."""

    role: str  # "user" | "assistant"
    content: str | list[dict[str, Any]]
    timestamp: datetime | None = None
    tool_use: ToolUse | None = None
    tool_result: ToolResult | None = None


@dataclass
class ToolUse:
    """An AI tool invocation."""

    name: str
    input: dict[str, Any] = field(default_factory=dict)
    tool_use_id: str | None = None


@dataclass
class ToolResult:
    """Result from a tool execution."""

    content: str | list[dict[str, Any]]
    tool_use_id: str | None = None
    is_error: bool = False


@dataclass
class Session:
    """A single AI coding session (one conversation)."""

    session_id: str
    agent_type: AgentType
    project_name: str = ""
    cwd: str = ""
    git_remote: str = ""
    messages: list[Message] = field(default_factory=list)
    start_time: datetime | None = None
    end_time: datetime | None = None
    file_paths: list[str] = field(default_factory=list)  # files touched
    is_subagent: bool = False  # True if this is a Claude subagent / Codex cross-tool session
    summary: dict[str, Any] = field(default_factory=dict)  # LLM-generated summary
    decisions: list[dict[str, Any]] = field(default_factory=list)  # Extracted decisions

    @property
    def duration_minutes(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds() / 60.0
        return 0.0

    @property
    def prompt_count(self) -> int:
        return sum(1 for m in self.messages if m.role == "user")

    @property
    def assistant_count(self) -> int:
        return sum(1 for m in self.messages if m.role == "assistant")

    @property
    def tool_use_count(self) -> int:
        return sum(1 for m in self.messages if m.tool_use is not None)

    @property
    def is_too_short(self) -> bool:
        """Sessions with <3 prompts or <30s duration are too short to analyze."""
        return self.prompt_count < 3 or (self.duration_minutes > 0 and self.duration_minutes < 0.5)


@dataclass
class DimensionScores:
    """Five-dimension builder scores (0-100)."""

    steering: float = 0.0
    execution: float = 0.0
    engineering: float = 0.0
    product_thinking: float = 0.0
    planning: float = 0.0

    @property
    def average(self) -> float:
        return (self.steering + self.execution + self.engineering + self.product_thinking + self.planning) / 5

    def to_dict(self) -> dict[str, float]:
        return {
            "steering": self.steering,
            "execution": self.execution,
            "engineering": self.engineering,
            "product_thinking": self.product_thinking,
            "planning": self.planning,
        }


@dataclass
class ArchetypeResult:
    """Builder archetype classification."""

    archetype: str
    confidence: float  # 0.0 - 1.0
    description: str = ""
    standout_traits: list[str] = field(default_factory=list)
    growth_edges: list[str] = field(default_factory=list)


@dataclass
class GitMetrics:
    """Git-derived velocity and activity metrics."""

    total_commits: int = 0
    active_days: int = 0
    commits_per_day: float = 0.0
    loc_per_day: float = 0.0
    files_touched: int = 0
    insertion_count: int = 0
    deletion_count: int = 0
    top_languages: dict[str, int] = field(default_factory=dict)


@dataclass
class BehavioralFeatures:
    """Raw behavioral features extracted from sessions."""

    total_sessions: int = 0
    total_prompts: int = 0
    avg_prompt_length: float = 0.0
    avg_session_duration: float = 0.0
    tool_histogram: dict[str, int] = field(default_factory=dict)
    planning_ratio: float = 0.0
    iteration_depth: float = 0.0
    autonomy_score: float = 0.0
    peak_hours: list[int] = field(default_factory=list)
    preferred_days: list[str] = field(default_factory=list)
    code_velocity: float = 0.0
    ask_user_count: int = 0
    error_recovery_count: int = 0
    tool_diversity: int = 0
    prompt_style: str = ""
    hour_distribution: dict[str, int] = field(default_factory=dict)


@dataclass
class WorkStream:
    """A multi-day work stream of related sessions."""

    stream_id: str
    project_name: str = ""
    start_time: datetime | None = None
    end_time: datetime | None = None
    sessions: list[Session] = field(default_factory=list)
    commit_count: int = 0
    description: str = ""

    @property
    def duration_hours(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds() / 3600.0
        return 0.0

    @property
    def session_count(self) -> int:
        return len(self.sessions)


@dataclass
class SessionCounts:
    """Breakdown of discovered sessions."""

    total: int = 0
    main: int = 0
    subagent: int = 0
    analyzable: int = 0
    too_short: int = 0
    by_agent: dict[str, int] = field(default_factory=dict)


@dataclass
class BuilderProfile:
    """Complete builder profile report data."""

    scores: DimensionScores = field(default_factory=DimensionScores)
    archetype: ArchetypeResult = field(default_factory=lambda: ArchetypeResult("Unknown", 0.0))
    features: BehavioralFeatures = field(default_factory=BehavioralFeatures)
    git_metrics: GitMetrics = field(default_factory=GitMetrics)
    sessions: list[Session] = field(default_factory=list)
    work_streams: list[WorkStream] = field(default_factory=list)
    session_counts: SessionCounts = field(default_factory=SessionCounts)
    generated_at: datetime = field(default_factory=datetime.utcnow)
    llm_used: bool = False  # True if LLM analysis was performed

    @property
    def builder_token(self) -> str:
        """Generate a deterministic builder token from scores."""
        payload = f"{self.archetype.archetype}:{self.scores.average:.2f}:{len(self.sessions)}"
        h = hashlib.sha256(payload.encode()).hexdigest()[:12]
        prefix = self.archetype.archetype.lower()[:4]
        return f"pxl_{prefix}_{h}"

    @property
    def total_time_hours(self) -> float:
        return sum(s.duration_minutes for s in self.sessions) / 60.0
