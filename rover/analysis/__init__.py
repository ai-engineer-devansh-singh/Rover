"""Analysis engine for behavioral feature extraction and scoring."""
from rover.analysis.archetypes import classify_archetype
from rover.analysis.code_quality import CodeQualityAnalyzer, SessionQualityReport
from rover.analysis.features import extract_features
from rover.analysis.git_metrics import analyze_git_history
from rover.analysis.redaction import redact, redact_dict
from rover.analysis.scoring import score_dimensions
from rover.analysis.streams import group_work_streams, WorkStreamAnalyzer

__all__ = [
    "extract_features",
    "score_dimensions",
    "classify_archetype",
    "group_work_streams",
    "WorkStreamAnalyzer",
    "CodeQualityAnalyzer",
    "SessionQualityReport",
    "analyze_git_history",
    "redact",
    "redact_dict",
]
