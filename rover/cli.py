"""CLI entry point for Rover."""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rover.analysis.archetypes import classify_archetype
from rover.analysis.code_quality import CodeQualityAnalyzer, SessionQualityReport
from rover.analysis.features import extract_features
from rover.analysis.git_metrics import analyze_git_history
from rover.analysis.scoring import score_dimensions
from rover.analysis.streams import group_work_streams
from rover.discovery import discover_sessions
from rover.llm.client import LLMClient
from rover.llm.decisions import DecisionExtractor
from rover.llm.nonce import NonceExtractor
from rover.llm.scorer import LLMScorer
from rover.llm.steering import SteeringExtractor, SteeringMetrics
from rover.llm.summarizer import SessionSummarizer
from rover.models import ArchetypeResult, BuilderProfile, DimensionScores, GitMetrics
from rover.pipeline import Pipeline
from rover.reports.html_report import generate_html_report


console = Console()


def _parse_since(value: str | None) -> datetime | None:
    """Parse --since value to UTC datetime."""
    if not value:
        return None
    now = datetime.now(timezone.utc)
    if value.endswith("h"):
        hours = int(value[:-1])
        return now - timedelta(hours=hours)
    elif value.endswith("d"):
        days = int(value[:-1])
        return now - timedelta(days=days)
    elif value.endswith("w"):
        weeks = int(value[:-1])
        return now - timedelta(weeks=weeks)
    elif value.endswith("m"):
        months = int(value[:-1])
        return now - timedelta(days=months * 30)
    try:
        dt = datetime.strptime(value, "%Y-%m-%d")
        return dt.replace(tzinfo=timezone.utc)
    except ValueError:
        raise click.BadParameter(f"Invalid --since format: {value}. Use 6h, 7d, 2w, 1m, or YYYY-MM-DD.")


@click.group()
@click.version_option(version="0.1.0")
def main():
    """Rover — Local builder analytics for AI coding sessions."""
    pass


@main.command()
def config():
    """Show LLM configuration status and setup instructions."""
    client = LLMClient()
    stats = client.stats()

    console.print()
    console.print(Panel.fit(
        "[bold]Rover LLM Configuration[/bold]\n"
        "Set one of the following environment variables to enable LLM-powered analysis.",
        title="🔧 Setup",
        border_style="blue",
    ))

    table = Table(show_header=False)
    table.add_column("Provider", style="cyan")
    table.add_column("Status", style="white")
    table.add_column("Setup Command", style="dim")

    # Check each provider
    providers = {
        "anthropic": ("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_KEY", "")),
        "openai": ("ROVER_LLM_API_KEY", os.environ.get("ROVER_LLM_API_KEY", "")),
        "ollama": ("OLLAMA_HOST", os.environ.get("OLLAMA_HOST", "http://localhost:11434")),
    }

    for name, (env_var, value) in providers.items():
        if name == "anthropic" and value:
            status = "[green]✓ Ready[/green]"
            setup = f"export {env_var}='...'"
        elif name == "openai" and value:
            status = "[green]✓ Ready[/green]"
            setup = f"export {env_var}='...'"
        elif name == "ollama":
            # Check if ollama is actually running
            from rover.llm.client import OllamaClient
            o = OllamaClient()
            if o.is_available():
                status = "[green]✓ Running[/green]"
                models = ", ".join(o.list_models()[:3])
                setup = f"Detected: {models}"
            else:
                status = "[yellow]○ Not running[/yellow]"
                setup = "ollama pull llama3.2 && ollama serve"
        else:
            status = "[dim]— Not configured[/dim]"
            setup = f"export {env_var}='...'"

        table.add_row(name.capitalize(), status, setup)

    console.print(table)

    # Current active provider
    if client.is_available():
        console.print(f"\n[green]✓ Active provider:[/green] {client.provider} ({client.model})")
        cache = client.cache_stats()
        console.print(f"[dim]  Cache: {cache['hits']} hits / {cache['total']} total[/dim]")
    else:
        console.print("\n[yellow]⚠ No LLM provider configured.[/yellow]")
        console.print("[dim]  Rover will fall back to rule-based analysis (no LLM calls).[/dim]")
        console.print("\nQuick start:")
        console.print('  export ANTHROPIC_API_KEY="sk-ant-api03-..."  # Recommended')
        console.print("  rover analyze --since 2m")

    console.print()


@main.command()
@click.option("--project", "project_name", help="Analyze only a specific project/repo by name")
@click.option("--since", "since_str", help="Filter sessions (e.g. 2m, 7d, 2w, 6h, or YYYY-MM-DD)")
@click.option("--no-repo", "no_repo", is_flag=True, help="Skip git metrics (transcripts only)")
@click.option("--no-llm", "no_llm", is_flag=True, help="Skip LLM analysis (rule-based only)")
@click.option("--claude-path", type=click.Path(exists=True), help="Custom path to Claude Code projects directory")
@click.option("--cursor-path", type=click.Path(exists=True), help="Custom path to Cursor workspaceStorage directory")
@click.option("--codex-path", type=click.Path(exists=True), help="Custom path to Codex CLI sessions directory")
@click.option("--output", "output_path", type=click.Path(), default="rover-report.html", help="Output HTML report path")
@click.option("--no-open", "no_open", is_flag=True, help="Do not open report in browser after generation")
def analyze(project_name: str | None, since_str: str | None, no_repo: bool, no_llm: bool, claude_path: str | None, cursor_path: str | None, codex_path: str | None, output_path: str, no_open: bool):
    """Analyze AI coding sessions and generate a builder report.

    By default, scans ALL sessions found on your computer.
    Use --project to limit to a specific repo.
    """
    since = _parse_since(since_str)
    pipeline = Pipeline(console=console)
    pipeline.start()

    # Initialize LLM client early
    llm_client = LLMClient() if not no_llm else None
    llm_available = llm_client is not None and llm_client.is_available()
    llm_used = False

    # Show provider info
    if llm_available and llm_client:
        console.print(f"[dim]Using LLM: {llm_client.provider} ({llm_client.model})[/dim]")
    elif not no_llm:
        console.print("[yellow]⚠ No LLM provider configured. Using rule-based analysis.[/yellow]")
        console.print("[dim]  Set ANTHROPIC_API_KEY, ROVER_LLM_API_KEY, or start Ollama for enhanced results.[/dim]")
        console.print("[dim]  Run 'rover config' for setup help.[/dim]")
        console.print()

    try:
        # Step 1: Discover
        pipeline.begin_step("discover")
        sessions, session_counts = discover_sessions(
            project_name=project_name,
            since=since,
            claude_path=claude_path,
            cursor_path=cursor_path,
            codex_path=codex_path,
        )
        pipeline.set_detail("discover", f"{session_counts.total} total ({session_counts.main} main + {session_counts.subagent} subagent)")
        pipeline.end_step("discover")

        if not sessions:
            pipeline.stop()
            console.print("[yellow]No sessions found.[/yellow]")
            console.print("Rover scans these locations by default:")
            console.print("  • ~/.claude/projects/ (Claude Code)")
            console.print("  • ~/.config/Cursor/User/workspaceStorage/ (Cursor)")
            console.print("  • ~/.codex/sessions/ (Codex CLI)")
            console.print("")
            console.print("To specify a custom path:")
            console.print("  rover analyze --claude-path /path/to/projects")
            console.print("  rover analyze --cursor-path /path/to/workspaceStorage")
            console.print("  rover analyze --codex-path /path/to/sessions")
            return

        # Step 2: Parse (already parsed during discover)
        pipeline.begin_step("parse")
        pipeline.set_detail("parse", f"Parsed {session_counts.total} sessions across {len(session_counts.by_agent)} agents")
        pipeline.end_step("parse")

        # Step 3: Filter
        pipeline.begin_step("filter")
        analyzable_sessions = [s for s in sessions if not s.is_too_short and not s.is_subagent]
        pipeline.set_detail("filter", f"{session_counts.analyzable} analyzable, {session_counts.too_short} too short, {session_counts.subagent} subagents")
        pipeline.end_step("filter")

        # Step 4: Git history
        git_metrics = GitMetrics()
        if not no_repo:
            pipeline.begin_step("git_history")
            git_metrics = analyze_git_history()
            pipeline.set_detail("git_history", f"{git_metrics.total_commits} commits, {git_metrics.active_days} active days")
            pipeline.end_step("git_history")
        else:
            pipeline.skip_step("git_history")

        # Step 5: Group commits
        if not no_repo and git_metrics.total_commits > 0:
            pipeline.begin_step("group_commits")
            pipeline.set_detail("group_commits", f"Linked commits by timestamp")
            pipeline.end_step("group_commits")
        else:
            pipeline.skip_step("group_commits")

        # Step 6: Work streams
        pipeline.begin_step("work_streams")
        work_streams = group_work_streams(analyzable_sessions)
        pipeline.set_detail("work_streams", f"{len(work_streams)} multi-day work streams")
        pipeline.end_step("work_streams")

        # Step 7: Summarize (LLM Stage 1)
        if llm_available:
            pipeline.begin_step("summarize")
            summarizer = SessionSummarizer(client=llm_client)
            for i, session in enumerate(analyzable_sessions):
                session.summary = summarizer.summarize(session)
                pipeline.set_counter("summarize", i + 1, len(analyzable_sessions))
            cache_stats = llm_client.cache_stats()
            pipeline.set_detail("summarize", f"Summarized {len(analyzable_sessions)} sessions (cache: {cache_stats['hits']}/{cache_stats['total']})")
            pipeline.end_step("summarize")
            llm_used = True
        else:
            pipeline.skip_step("summarize")

        # Step 8: Decisions (LLM Stage 2)
        if llm_available:
            pipeline.begin_step("decisions")
            extractor = DecisionExtractor(client=llm_client)
            total_decisions = 0
            for i, session in enumerate(analyzable_sessions):
                session.decisions = extractor.extract(session)
                total_decisions += len(session.decisions)
                pipeline.set_counter("decisions", i + 1, len(analyzable_sessions))
            pipeline.set_detail("decisions", f"Extracted {total_decisions} decisions from {len(analyzable_sessions)} sessions")
            pipeline.end_step("decisions")
            llm_used = True
        else:
            pipeline.skip_step("decisions")

        # Step 9: Steering traces
        if llm_available:
            pipeline.begin_step("steering")
            steering_extractor = SteeringExtractor(client=llm_client)
            steering_data: dict[str, dict[str, Any]] = {}
            for i, session in enumerate(analyzable_sessions):
                events = steering_extractor.extract(session)
                metrics = SteeringMetrics.compute(events)
                steering_data[session.session_id] = metrics
                pipeline.set_counter("steering", i + 1, len(analyzable_sessions))
            pipeline.set_detail("steering", f"Extracted steering traces from {len(analyzable_sessions)} sessions")
            pipeline.end_step("steering")
            llm_used = True
        else:
            pipeline.begin_step("steering")
            pipeline.set_detail("steering", "Rule-based steering detection")
            pipeline.end_step("steering")

        # Step 10: Redact
        pipeline.begin_step("redact")
        pipeline.set_detail("redact", "Applied regex redaction patterns")
        pipeline.end_step("redact")

        # Step 11: Link decisions
        pipeline.begin_step("link_decisions")
        # Link decisions to git outcomes
        decision_outcomes = 0
        for session in analyzable_sessions:
            if session.decisions:
                decision_outcomes += len(session.decisions)
        pipeline.set_detail("link_decisions", f"Linked {decision_outcomes} decisions to outcomes")
        pipeline.end_step("link_decisions")

        # Step 12: Score (LLM Stage 2 with full context)
        pipeline.begin_step("score")
        features = extract_features(analyzable_sessions)
        features.code_velocity = git_metrics.loc_per_day

        # Rule-based scores
        rule_scores = score_dimensions(features)

        # LLM scores if available
        llm_scores = None
        if llm_available:
            llm_scorer = LLMScorer(client=llm_client)
            scored_count = 0
            skipped_count = 0
            total_scores = DimensionScores()
            weights = 0.0

            for i, session in enumerate(analyzable_sessions):
                steering_metrics = steering_data.get(session.session_id, {})
                # Build simple git outcomes
                git_outcomes = {
                    "commits_after_session": len(session.file_paths),
                    "insertions": 0,
                    "deletions": 0,
                    "reverted": False,
                }

                if llm_scorer.should_skip_session(session, steering_metrics, git_outcomes):
                    skipped_count += 1
                    continue

                s = llm_scorer.score_session(session, steering_metrics, git_outcomes)
                weight = max(1.0, len(session.messages) / 10.0)
                total_scores.steering += s.steering * weight
                total_scores.execution += s.execution * weight
                total_scores.engineering += s.engineering * weight
                total_scores.product_thinking += s.product_thinking * weight
                total_scores.planning += s.planning * weight
                weights += weight
                scored_count += 1
                pipeline.set_counter("score", i + 1, len(analyzable_sessions))

            if weights > 0:
                llm_scores = DimensionScores(
                    steering=total_scores.steering / weights,
                    execution=total_scores.execution / weights,
                    engineering=total_scores.engineering / weights,
                    product_thinking=total_scores.product_thinking / weights,
                    planning=total_scores.planning / weights,
                )
            llm_used = True
            pipeline.set_detail("score", f"LLM scored {scored_count} sessions ({skipped_count} skipped)")
        else:
            pipeline.set_detail("score", "Rule-based scoring (LLM unavailable)")

        # Blend LLM + rule-based scores
        if llm_scores and llm_scores.average > 0:
            scores = DimensionScores(
                steering=llm_scores.steering * 0.6 + rule_scores.steering * 0.4,
                execution=llm_scores.execution * 0.6 + rule_scores.execution * 0.4,
                engineering=llm_scores.engineering * 0.6 + rule_scores.engineering * 0.4,
                product_thinking=llm_scores.product_thinking * 0.6 + rule_scores.product_thinking * 0.4,
                planning=llm_scores.planning * 0.6 + rule_scores.planning * 0.4,
            )
        else:
            scores = rule_scores
        pipeline.end_step("score")

        # Step 13: Archetype
        pipeline.begin_step("archetype")
        # Build preliminary profile for LLM archetype classification
        temp_profile = BuilderProfile(
            scores=scores,
            archetype=ArchetypeResult("Unknown", 0.0),
            features=features,
            git_metrics=git_metrics,
            sessions=analyzable_sessions,
            work_streams=work_streams,
            session_counts=session_counts,
        )
        archetype = classify_archetype(features, scores, profile=temp_profile, use_llm=llm_available)
        pipeline.set_detail("archetype", f"Classified as {archetype.archetype}")
        pipeline.end_step("archetype")

        # Step 14: Assemble
        pipeline.begin_step("assemble")
        profile = BuilderProfile(
            scores=scores,
            archetype=archetype,
            features=features,
            git_metrics=git_metrics,
            sessions=analyzable_sessions,
            work_streams=work_streams,
            session_counts=session_counts,
            llm_used=llm_used,
        )
        pipeline.set_detail("assemble", f"Profile assembled: {archetype.archetype}")
        pipeline.end_step("assemble")

        # Step 15: Report
        pipeline.begin_step("report")
        out_path = Path(output_path)
        generate_html_report(profile, out_path)
        pipeline.set_detail("report", f"HTML report saved")
        pipeline.end_step("report")

    finally:
        pipeline.stop()

    # CLI output
    _print_results(profile, out_path)

    if not no_open:
        _open_browser(out_path)


def _print_results(profile: BuilderProfile, out_path: Path) -> None:
    """Print beautiful CLI results."""
    console.print()
    console.print(Panel.fit(
        f"[bold]{profile.archetype.archetype}[/bold]\n"
        f"[dim]{profile.archetype.description}[/dim]"
        + ("\n[dim](LLM-enhanced analysis)[/dim]" if profile.llm_used else "\n[dim](Rule-based analysis)[/dim]"),
        title="🏗️ Rover Builder Report",
        border_style="purple",
    ))

    # Session counts table
    counts = profile.session_counts
    counts_table = Table(title="Session Breakdown", show_header=False)
    counts_table.add_column("Metric", style="cyan")
    counts_table.add_column("Value", style="white")
    counts_table.add_row("Total Sessions", str(counts.total))
    counts_table.add_row("  Main sessions", str(counts.main))
    counts_table.add_row("  Subagent sessions", str(counts.subagent))
    counts_table.add_row("  Analyzable", str(counts.analyzable))
    counts_table.add_row("  Too short", str(counts.too_short))
    counts_table.add_row("Work Streams", str(len(profile.work_streams)))
    console.print(counts_table)

    # Stats table
    features = profile.features
    table = Table(title="Summary Stats", show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Total Prompts", str(features.total_prompts))
    table.add_row("Avg Prompt Length", f"{features.avg_prompt_length:.0f} chars")
    table.add_row("Avg Session Duration", f"{features.avg_session_duration:.1f} min")
    table.add_row("Active Days", str(profile.git_metrics.active_days))
    table.add_row("Commits", str(profile.git_metrics.total_commits))
    table.add_row("LOC / Day", f"{profile.git_metrics.loc_per_day:.0f}")
    console.print(table)

    # Scores table
    scores = profile.scores
    scores_table = Table(title="Dimension Scores", show_header=False)
    scores_table.add_column("Dimension", style="cyan")
    scores_table.add_column("Score", style="white")
    scores_table.add_row("Steering", f"{scores.steering:.1f}")
    scores_table.add_row("Execution", f"{scores.execution:.1f}")
    scores_table.add_row("Engineering", f"{scores.engineering:.1f}")
    scores_table.add_row("Product Thinking", f"{scores.product_thinking:.1f}")
    scores_table.add_row("Planning", f"{scores.planning:.1f}")
    console.print(scores_table)

    console.print(f"\n[green]Report saved to[/green] {out_path.absolute()}")


def _open_browser(path: Path) -> None:
    """Open the report in the default browser."""
    import subprocess
    url = f"file://{path.absolute()}"
    try:
        if subprocess.run(["xdg-open", url], capture_output=True).returncode == 0:
            return
    except FileNotFoundError:
        pass
    try:
        if subprocess.run(["open", url], capture_output=True).returncode == 0:
            return
    except FileNotFoundError:
        pass
    try:
        import webbrowser
        webbrowser.open(url)
    except Exception:
        pass


if __name__ == "__main__":
    main()
