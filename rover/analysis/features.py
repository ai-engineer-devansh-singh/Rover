"""Behavioral feature extraction from parsed sessions."""
from __future__ import annotations

import re
from collections import Counter
from datetime import datetime
from typing import Any

from rover.models import BehavioralFeatures, Message, Session


def extract_features(sessions: list[Session]) -> BehavioralFeatures:
    """Extract all behavioral features from a list of sessions."""
    if not sessions:
        return BehavioralFeatures()

    total_prompts = sum(s.prompt_count for s in sessions)
    prompt_lengths = []
    durations = []
    tool_counter: Counter[str] = Counter()
    ask_user_count = 0
    error_recovery_count = 0
    hours: Counter[int] = Counter()
    days: Counter[str] = Counter()
    file_edit_counts: dict[str, int] = {}
    all_prompt_texts: list[str] = []

    for session in sessions:
        if session.duration_minutes > 0:
            durations.append(session.duration_minutes)

        if session.start_time:
            hours[session.start_time.hour] += 1
            days[session.start_time.strftime("%A")] += 1

        for msg in session.messages:
            if msg.role == "user":
                text = _text_from_msg(msg)
                if text:
                    prompt_lengths.append(len(text))
                    all_prompt_texts.append(text)

                if msg.tool_result and msg.tool_result.is_error:
                    error_recovery_count += 1

            elif msg.role == "assistant" and msg.tool_use:
                tool_name = msg.tool_use.name
                tool_counter[tool_name] += 1

                if tool_name == "AskUserQuestion":
                    ask_user_count += 1

                # Track file edits for iteration depth
                if tool_name in ("Edit", "Write"):
                    for key in ("file_path", "path", "targetFile", "relativeWorkspacePath"):
                        fp = msg.tool_use.input.get(key, "")
                        if fp and isinstance(fp, str):
                            file_edit_counts[fp] = file_edit_counts.get(fp, 0) + 1

    avg_prompt_length = sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0.0
    avg_session_duration = sum(durations) / len(durations) if durations else 0.0

    # Planning ratio: (Glob + Grep + Read) / (Edit + Write + Bash)
    research_tools = tool_counter.get("Glob", 0) + tool_counter.get("Grep", 0) + tool_counter.get("Read", 0)
    coding_tools = tool_counter.get("Edit", 0) + tool_counter.get("Write", 0) + tool_counter.get("Bash", 0)
    planning_ratio = research_tools / coding_tools if coding_tools > 0 else float("inf") if research_tools > 0 else 0.0

    # Iteration depth: avg edits per file before commit
    iteration_depth = sum(file_edit_counts.values()) / len(file_edit_counts) if file_edit_counts else 0.0

    # Autonomy score: 1 - (AskUserQuestion / total_prompts)
    autonomy_score = 1.0 - (ask_user_count / total_prompts) if total_prompts > 0 else 1.0

    # Peak hours
    peak_hours = [h for h, c in hours.most_common(3)]

    # Preferred days
    preferred_days = [d for d, c in days.most_common(3)]

    # Tool diversity: number of distinct tools used
    tool_diversity = len([t for t in tool_counter if tool_counter[t] > 0])

    # Prompt style detection
    prompt_style = _detect_prompt_style(all_prompt_texts)

    return BehavioralFeatures(
        total_sessions=len(sessions),
        total_prompts=total_prompts,
        avg_prompt_length=avg_prompt_length,
        avg_session_duration=avg_session_duration,
        tool_histogram=dict(tool_counter),
        planning_ratio=planning_ratio,
        iteration_depth=iteration_depth,
        autonomy_score=autonomy_score,
        peak_hours=peak_hours,
        preferred_days=preferred_days,
        code_velocity=0.0,  # populated later from git metrics
        ask_user_count=ask_user_count,
        error_recovery_count=error_recovery_count,
        tool_diversity=tool_diversity,
        prompt_style=prompt_style,
        hour_distribution={str(k): v for k, v in sorted(hours.items())},
    )


def _text_from_msg(msg: Message) -> str:
    """Extract plain text from a message."""
    if isinstance(msg.content, str):
        return msg.content
    texts = []
    for block in msg.content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                texts.append(block.get("text", ""))
            elif "content" in block and isinstance(block["content"], str):
                texts.append(block["content"])
    return " ".join(texts)


def _detect_prompt_style(prompts: list[str]) -> str:
    """Categorize overall prompt style based on vocabulary."""
    if not prompts:
        return ""
    all_text = " ".join(prompts).lower()
    scores = {
        "architectural": len(re.findall(r"\b(architecture|design|pattern|structure| scaffold|refactor|model|schema)\b", all_text)),
        "action": len(re.findall(r"\b(fix|add|create|write|implement|build|run|test|deploy|ship)\b", all_text)),
        "investigative": len(re.findall(r"\b(debug|investigate|why|what|how|explain|understand|check|search)\b", all_text)),
        "collaborative": len(re.findall(r"\b(should we|which approach|what do you think|clarify|confirm|agree)\b", all_text)),
    }
    return max(scores, key=scores.get) if scores else ""
