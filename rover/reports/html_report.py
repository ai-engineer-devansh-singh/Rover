"""Rich HTML report generation with actionable insights."""
from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from rover.models import BuilderProfile


def _compute_insights(profile: BuilderProfile) -> dict:
    """Generate actionable insights from the builder profile."""
    scores = profile.scores
    features = profile.features
    archetype = profile.archetype

    # Strengths (top 2 dimensions)
    score_dict = scores.to_dict()
    strengths = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)[:2]
    strength_insights = []
    for dim, val in strengths:
        if dim == "steering" and val > 60:
            strength_insights.append("You steer AI effectively — clear intent, minimal babysitting.")
        elif dim == "execution" and val > 60:
            strength_insights.append("You execute fast — high tool mastery, quick iteration cycles.")
        elif dim == "engineering" and val > 60:
            strength_insights.append("You build with quality in mind — structure, tests, clean code.")
        elif dim == "product_thinking" and val > 60:
            strength_insights.append("You think like a PM — user focus, trade-offs, architecture.")
        elif dim == "planning" and val > 60:
            strength_insights.append("You research before coding — exploration saves rework.")

    # Weaknesses / growth areas (bottom 2 dimensions)
    weaknesses = sorted(score_dict.items(), key=lambda x: x[1])[:2]
    growth_actions = []
    for dim, val in weaknesses:
        if dim == "steering" and val < 50:
            growth_actions.append({"area": "Steering", "tip": "Write more specific prompts. Include examples and constraints upfront.", "example": "Instead of 'fix the bug', say 'fix the auth bug where tokens expire after 5 min'"})
        elif dim == "execution" and val < 50:
            growth_actions.append({"area": "Execution", "tip": "Learn keyboard shortcuts and tool aliases. Batch similar operations.", "example": "Use Glob + Grep together before editing multiple files"})
        elif dim == "engineering" and val < 50:
            growth_actions.append({"area": "Engineering", "tip": "Ask the AI to add tests before accepting code.", "example": "Add 'with tests' to every implementation request"})
        elif dim == "product_thinking" and val < 50:
            growth_actions.append({"area": "Product Thinking", "tip": "Start prompts with user impact. Who benefits and how?", "example": "'Add OAuth so users can sign in with Google' vs 'implement OAuth'"})
        elif dim == "planning" and val < 50:
            growth_actions.append({"area": "Planning", "tip": "Spend 2 minutes exploring the codebase before coding.", "example": "'Read the auth middleware first, then implement the feature'"})

    # Tool insights
    tool_hist = features.tool_histogram
    top_tools = sorted(tool_hist.items(), key=lambda x: x[1], reverse=True)[:5] if tool_hist else []
    tool_insights = []
    total_tools = sum(tool_hist.values()) if tool_hist else 0
    read_pct = (tool_hist.get("Read", 0) / total_tools * 100) if total_tools else 0
    edit_pct = (tool_hist.get("Edit", 0) / total_tools * 100) if total_tools else 0
    if read_pct > 40:
        tool_insights.append(f"You're research-heavy ({read_pct:.0f}% reads). You understand before building.")
    if edit_pct > 30:
        tool_insights.append(f"You edit aggressively ({edit_pct:.0f}% edits). High output velocity.")
    if features.tool_diversity > 6:
        tool_insights.append(f"You use {features.tool_diversity} different tools — versatile builder.")
    elif features.tool_diversity < 4:
        tool_insights.append("You rely on a small toolkit. Explore more tools for efficiency.")

    # Temporal insights
    hour_dist = features.hour_distribution
    peak = features.peak_hours[0] if features.peak_hours else None
    time_insight = ""
    if peak is not None:
        if 6 <= peak <= 11:
            time_insight = "Morning coder — freshest focus, best for complex tasks."
        elif 12 <= peak <= 17:
            time_insight = "Afternoon hacker — sustained output, good for shipping."
        elif 18 <= peak <= 23:
            time_insight = "Night owl — deep focus, fewer distractions."
        else:
            time_insight = "Late-night builder — watch for burnout patterns."

    # Prompt style insight
    style_tip = ""
    if features.prompt_style == "architectural":
        style_tip = "You think in systems. Your prompts include context and constraints. Keep it up."
    elif features.prompt_style == "action":
        style_tip = "You're action-oriented. Add 'why' to prompts for better AI alignment."
    elif features.prompt_style == "investigative":
        style_tip = "You debug methodically. Consider adding proposed solutions to speed up."
    elif features.prompt_style == "collaborative":
        style_tip = "You collaborate well. Try making more decisions independently to build confidence."
    else:
        style_tip = "Mixed prompt style. Consistent structure helps the AI respond better."

    # Session quality
    session_quality = ""
    if features.avg_session_duration > 120:
        session_quality = "Long sessions (>2h). Deep work mode — great for complex features. Watch for fatigue."
    elif features.avg_session_duration > 60:
        session_quality = "Medium sessions (1-2h). Balanced focus and stamina."
    elif features.avg_session_duration > 30:
        session_quality = "Short sessions (30-60min). Fast iteration — consider batching related tasks."
    else:
        session_quality = "Very short sessions (<30min). Quick fixes — build bigger blocks for flow state."

    # Autonomy insight
    autonomy = ""
    if features.autonomy_score > 0.8:
        autonomy = "High autonomy. You set direction and let the AI execute. Efficient collaboration."
    elif features.autonomy_score > 0.5:
        autonomy = "Moderate autonomy. You steer actively — fine for learning new patterns."
    else:
        autonomy = "High steering ratio. You micromanage the AI. Try broader prompts with checkpoints."

    # Error recovery
    error_insight = ""
    if features.error_recovery_count > 5:
        error_insight = f"{features.error_recovery_count} error recoveries — resilient debugging. Consider preventive testing."
    elif features.error_recovery_count > 0:
        error_insight = f"{features.error_recovery_count} errors recovered — solid troubleshooting."
    else:
        error_insight = "Clean runs — your prompts and setup are well-calibrated."

    # Max session duration
    max_session_duration = max(
        (s.duration_minutes for s in profile.sessions if s.duration_minutes > 0),
        default=0.0,
    )

    return {
        "strengths": strength_insights,
        "growth_actions": growth_actions,
        "tool_insights": tool_insights,
        "time_insight": time_insight,
        "style_tip": style_tip,
        "session_quality": session_quality,
        "autonomy": autonomy,
        "error_insight": error_insight,
        "top_tools": top_tools,
        "avg_score": scores.average,
        "max_session_duration": max_session_duration,
    }


def _session_timeline(profile: BuilderProfile) -> list[dict]:
    """Build a timeline of sessions for the report."""
    timeline = []
    for s in profile.sessions[:20]:  # Cap at 20 for report size
        summary = s.summary or {}
        timeline.append({
            "id": s.session_id[:20],
            "goal": summary.get("goal", "")[:80] or "Session",
            "outcome": summary.get("outcome", "unknown"),
            "duration": int(s.duration_minutes),
            "prompts": s.prompt_count,
            "tools": len(set(m.tool_use.name for m in s.messages if m.tool_use)),
            "date": s.start_time.strftime("%b %d") if s.start_time else "Unknown",
            "complexity": summary.get("complexity", "medium"),
        })
    return timeline


def _decision_log(profile: BuilderProfile) -> list[dict]:
    """Extract key decisions from sessions."""
    decisions = []
    for s in profile.sessions:
        for d in s.decisions[:3]:  # Max 3 per session
            decisions.append({
                "decision": d.get("decision", "")[:120],
                "type": d.get("type", "unknown"),
                "confidence": d.get("confidence", "medium"),
                "context": d.get("context", "")[:100],
            })
    return decisions[:15]  # Cap total


def generate_html_report(profile: BuilderProfile, output_path: Path) -> None:
    """Generate a rich, actionable HTML report from a builder profile."""
    template_dir = Path(__file__).parent / "templates"
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("report.html.j2")

    insights = _compute_insights(profile)
    timeline = _session_timeline(profile)
    decisions = _decision_log(profile)

    html = template.render(
        archetype=profile.archetype,
        scores=profile.scores,
        features=profile.features,
        git_metrics=profile.git_metrics,
        generated_at=profile.generated_at.strftime("%Y-%m-%d %H:%M UTC"),
        llm_used=profile.llm_used,
        insights=insights,
        timeline=timeline,
        decisions=decisions,
        total_time=profile.total_time_hours,
        session_counts=profile.session_counts,
    )

    output_path.write_text(html, encoding="utf-8")
