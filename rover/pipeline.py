"""Rich TUI pipeline tracker for Rover analysis steps."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text


@dataclass
class PipelineStep:
    name: str
    status: str = "pending"  # pending | running | done | skipped
    detail: str = ""
    counter: str = ""
    started_at: float = 0.0
    ended_at: float = 0.0

    @property
    def duration(self) -> float:
        if self.started_at > 0 and self.ended_at > 0:
            return self.ended_at - self.started_at
        if self.started_at > 0:
            return time.time() - self.started_at
        return 0.0


class Pipeline:
    """Tracks and displays the Rover analysis pipeline."""

    STEPS = [
        "discover",
        "parse",
        "filter",
        "git_history",
        "group_commits",
        "work_streams",
        "summarize",
        "decisions",
        "steering",
        "redact",
        "link_decisions",
        "score",
        "archetype",
        "assemble",
        "report",
    ]

    STEP_NAMES: dict[str, str] = {
        "discover": "Discovering projects and sessions",
        "parse": "Parsing transcripts",
        "filter": "Filtering sessions",
        "git_history": "Reading git history",
        "group_commits": "Grouping commits by session",
        "work_streams": "Grouping sessions into work streams",
        "summarize": "Summarizing each session",
        "decisions": "Extracting decision exchanges",
        "steering": "Extracting steering traces",
        "redact": "Redacting sensitive data",
        "link_decisions": "Linking decisions to outcomes",
        "score": "Scoring episodes across 5 axes",
        "archetype": "Classifying builder archetype",
        "assemble": "Assembling your profile",
        "report": "Generating report",
    }

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()
        self.steps: dict[str, PipelineStep] = {
            name: PipelineStep(name=self.STEP_NAMES[name]) for name in self.STEPS
        }
        self.current_step: str | None = None
        self.overall_progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=30, complete_style="green", finished_style="green"),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            expand=False,
        )
        self.overall_task = self.overall_progress.add_task(
            "Analyzing...", total=len(self.STEPS)
        )
        self._live: Live | None = None
        self._start_time = 0.0

    def _render(self) -> Layout:
        """Render the current pipeline state as a Rich layout."""
        layout = Layout()

        # Header
        header_text = Text("🏗️  Rover Builder Analytics", style="bold cyan")
        header = Panel(header_text, border_style="cyan")

        # Overall progress
        progress_panel = Panel(self.overall_progress, border_style="blue")

        # Current step detail
        current_detail = ""
        if self.current_step:
            step = self.steps[self.current_step]
            current_detail = f"[bold]{step.name}[/bold]"
            if step.counter:
                current_detail += f"\n{step.counter}"
            if step.detail:
                current_detail += f"\n[dim]{step.detail}[/dim]"
            elapsed = step.duration
            current_detail += f"\n[dim]Elapsed: {elapsed:.1f}s[/dim]"

        current_panel = Panel(
            current_detail or "[dim]Waiting...[/dim]",
            title="Current Step",
            border_style="yellow" if self.current_step else "dim",
        )

        # Step list
        table = Table(show_header=False, box=None, pad_edge=False)
        table.add_column("Status", width=3)
        table.add_column("Step", style="white")
        table.add_column("Time", justify="right", style="dim", width=8)

        for name in self.STEPS:
            step = self.steps[name]
            if step.status == "done":
                icon = "✅"
                style = "dim"
            elif step.status == "running":
                icon = "⏳"
                style = "bold yellow"
            elif step.status == "skipped":
                icon = "⏭️"
                style = "dim"
            else:
                icon = "⏸️"
                style = "dim"
            duration_str = f"{step.duration:.1f}s" if step.duration > 0 else ""
            table.add_row(icon, step.name, duration_str, style=style)

        steps_panel = Panel(table, title="Pipeline Steps", border_style="dim")

        # Compose layout
        layout.split_column(
            Layout(header, size=3),
            Layout(progress_panel, size=5),
            Layout(current_panel, size=6),
            Layout(steps_panel),
        )
        return layout

    def start(self) -> None:
        """Start the live display."""
        self._start_time = time.time()
        self._live = Live(self._render(), console=self.console, refresh_per_second=4, screen=False)
        self._live.start()

    def stop(self) -> None:
        """Stop the live display."""
        if self._live:
            self._live.stop()
            self._live = None

    def _update(self) -> None:
        if self._live:
            self._live.update(self._render())

    def begin_step(self, step_name: str) -> None:
        """Mark a step as running."""
        if self.current_step:
            self.end_step(self.current_step)
        self.current_step = step_name
        step = self.steps[step_name]
        step.status = "running"
        step.started_at = time.time()

        # Update overall progress
        done_count = sum(1 for s in self.steps.values() if s.status in ("done", "skipped"))
        self.overall_progress.update(self.overall_task, completed=done_count)
        self._update()

    def end_step(self, step_name: str) -> None:
        """Mark a step as done."""
        step = self.steps[step_name]
        if step.status == "running":
            step.status = "done"
            step.ended_at = time.time()
            done_count = sum(1 for s in self.steps.values() if s.status in ("done", "skipped"))
            self.overall_progress.update(self.overall_task, completed=done_count)
            self._update()

    def skip_step(self, step_name: str) -> None:
        """Mark a step as skipped."""
        step = self.steps[step_name]
        step.status = "skipped"
        done_count = sum(1 for s in self.steps.values() if s.status in ("done", "skipped"))
        self.overall_progress.update(self.overall_task, completed=done_count)
        self._update()

    def set_counter(self, step_name: str, current: int, total: int, rate: float = 0.0) -> None:
        """Set a counter for the current step (e.g., '288/288 23.5 it/s')."""
        step = self.steps[step_name]
        step.counter = f"{current}/{total}"
        if rate > 0:
            step.counter += f"  {rate:.1f} it/s"
        self._update()

    def set_detail(self, step_name: str, detail: str) -> None:
        """Set detail text for the current step."""
        self.steps[step_name].detail = detail
        self._update()
