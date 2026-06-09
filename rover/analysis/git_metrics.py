"""Git history analysis for velocity and activity metrics."""
from __future__ import annotations

import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from rover.models import GitMetrics


def analyze_git_history(repo_path: Path | None = None, max_commits: int = 1000) -> GitMetrics:
    """Analyze git history for velocity metrics."""
    metrics = GitMetrics()
    cwd = repo_path or Path.cwd()

    if not (cwd / ".git").exists():
        return metrics

    try:
        # Commit metadata
        result = subprocess.run(
            ["git", "-C", str(cwd), "log", f"-{max_commits}", "--format=%H|%an|%ae|%ad|%s", "--date=short"],
            capture_output=True,
            text=True,
            check=True,
        )
        commits = result.stdout.strip().split("\n")
        if not commits or commits == [""]:
            return metrics

        metrics.total_commits = len([c for c in commits if c.strip()])

        # Active days
        dates = Counter()
        for line in commits:
            parts = line.split("|")
            if len(parts) >= 4:
                dates[parts[3]] += 1
        metrics.active_days = len(dates)

        # Commits per day
        if metrics.active_days > 0:
            metrics.commits_per_day = metrics.total_commits / metrics.active_days

        # Numstat for LOC
        result = subprocess.run(
            ["git", "-C", str(cwd), "log", f"-{max_commits}", "--numstat", "--format=COMMIT"],
            capture_output=True,
            text=True,
            check=True,
        )
        insertions = 0
        deletions = 0
        files_touched: set[str] = set()
        lang_counter: Counter[str] = Counter()

        for line in result.stdout.split("\n"):
            if line.startswith("COMMIT"):
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                try:
                    ins = int(parts[0]) if parts[0].isdigit() else 0
                    dels = int(parts[1]) if parts[1].isdigit() else 0
                    filepath = parts[2]
                    insertions += ins
                    deletions += dels
                    files_touched.add(filepath)

                    # Detect language from extension
                    ext = Path(filepath).suffix.lower()
                    lang = _ext_to_lang(ext)
                    if lang:
                        lang_counter[lang] += ins + dels
                except ValueError:
                    continue

        metrics.insertion_count = insertions
        metrics.deletion_count = deletions
        metrics.files_touched = len(files_touched)
        metrics.top_languages = dict(lang_counter.most_common(5))

        # LOC per day
        if metrics.active_days > 0:
            metrics.loc_per_day = (insertions + deletions) / metrics.active_days

    except subprocess.CalledProcessError:
        pass
    except FileNotFoundError:
        pass

    return metrics


def _ext_to_lang(ext: str) -> str | None:
    """Map file extension to language name."""
    mapping = {
        ".py": "Python",
        ".js": "JavaScript",
        ".ts": "TypeScript",
        ".jsx": "JavaScript",
        ".tsx": "TypeScript",
        ".go": "Go",
        ".rs": "Rust",
        ".java": "Java",
        ".kt": "Kotlin",
        ".swift": "Swift",
        ".rb": "Ruby",
        ".php": "PHP",
        ".cpp": "C++",
        ".c": "C",
        ".h": "C/C++",
        ".cs": "C#",
        ".scala": "Scala",
        ".r": "R",
        ".m": "Objective-C",
        ".sh": "Shell",
        ".bash": "Shell",
        ".html": "HTML",
        ".css": "CSS",
        ".scss": "CSS",
        ".sass": "CSS",
        ".json": "JSON",
        ".yaml": "YAML",
        ".yml": "YAML",
        ".toml": "TOML",
        ".md": "Markdown",
        ".sql": "SQL",
    }
    return mapping.get(ext)
