"""Live terminal dashboard for NadirClaw routing stats."""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from nadirclaw.report import load_log_entries, load_log_entries_sqlite
from nadirclaw.savings import calculate_actual_cost, calculate_hypothetical_cost, get_model_cost


def _load_entries(log_path: Path, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load log entries, preferring SQLite when available."""
    if db_path is not None and db_path.exists():
        return load_log_entries_sqlite(db_path)
    if log_path.exists():
        return load_log_entries(log_path)
    return []

HEADER = r"""
 _   _           _ _       ____ _                
| \ | | __ _  __| (_)_ __ / ___| | __ ___      __
|  \| |/ _` |/ _` | | '__| |   | |/ _` \ \ /\ / /
| |\  | (_| | (_| | | |  | |___| | (_| |\ V  V / 
|_| \_|\__,_|\__,_|_|_|   \____|_|\__,_| \_/\_/  
"""


def _safe_int(val: Any) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


def _safe_float(val: Any) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def _build_bar(value: float, max_value: float, width: int = 30, char: str = "█") -> str:
    if max_value <= 0:
        return ""
    filled = int(value / max_value * width)
    return char * filled + "░" * (width - filled)


def run_dashboard_rich(log_path: Path, refresh: float = 2.0, db_path: Optional[Path] = None):
    """Run the dashboard using Rich library for a nice terminal UI."""
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    console = Console()
    start_time = time.time()

    def make_display() -> Layout:
        entries = _load_entries(log_path, db_path)
        total = len(entries)
        uptime = time.time() - start_time

        # Tier counts
        tiers: Dict[str, int] = {}
        for e in entries:
            tier = e.get("tier", "unknown")
            tiers[tier] = tiers.get(tier, 0) + 1

        # Models used
        models: Dict[str, int] = {}
        for e in entries:
            m = e.get("selected_model", "unknown")
            models[m] = models.get(m, 0) + 1

        # Requests per minute (last 5 min)
        now_ts = datetime.now(timezone.utc)
        recent = 0
        for e in entries:
            ts_str = e.get("timestamp")
            if ts_str:
                try:
                    ts = datetime.fromisoformat(ts_str)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    if (now_ts - ts).total_seconds() < 300:
                        recent += 1
                except (ValueError, TypeError):
                    pass
        rpm = recent / 5 if recent > 0 else 0

        # Cost calculation
        actual_cost = calculate_actual_cost(entries)
        # Find most expensive model as baseline
        baseline_model = "claude-sonnet-4-5-20250929"
        max_cost = 0
        for model in models:
            ci, co = get_model_cost(model)
            if (ci + co) / 2 > max_cost:
                max_cost = (ci + co) / 2
                baseline_model = model
        baseline_cost = calculate_hypothetical_cost(entries, baseline_model)
        savings = baseline_cost - actual_cost
        savings_pct = (savings / baseline_cost * 100) if baseline_cost > 0 else 0

        # Last 10 requests
        last_10 = entries[-10:] if len(entries) >= 10 else entries

        # Build layout
        layout = Layout()

        # Header
        header_text = Text(HEADER, style="bold cyan")
        header_text.append(f"  Dashboard  |  Uptime: {_format_duration(uptime)}", style="dim")

        # Stats panel
        stats = Table.grid(padding=(0, 2))
        stats.add_row(
            Text(f"Total Requests", style="bold"),
            Text(f"{total:,}", style="bold white"),
        )
        stats.add_row(
            Text("Req/min (5m avg)", style="bold"),
            Text(f"{rpm:.1f}", style="yellow"),
        )
        stats.add_row(
            Text("Actual Cost", style="bold"),
            Text(f"${actual_cost:.4f}", style="white"),
        )
        stats.add_row(
            Text("Without Routing", style="bold"),
            Text(f"${baseline_cost:.4f}", style="dim"),
        )
        stats.add_row(
            Text("Saved", style="bold"),
            Text(f"${savings:.4f} ({savings_pct:.1f}%)", style="bold green"),
        )

        # Tier distribution
        tier_table = Table(title="Routing Distribution", show_header=True, header_style="bold")
        tier_table.add_column("Tier", style="bold")
        tier_table.add_column("Count", justify="right")
        tier_table.add_column("Bar", min_width=30)
        tier_table.add_column("%", justify="right")

        max_tier = max(tiers.values()) if tiers else 1
        tier_colors = {"simple": "blue", "complex": "red", "reasoning": "magenta", "orchestrator": "cyan", "direct": "yellow"}
        for tier, count in sorted(tiers.items(), key=lambda x: x[1], reverse=True):
            pct = count / total * 100 if total > 0 else 0
            color = tier_colors.get(tier, "white")
            bar = _build_bar(count, max_tier)
            tier_table.add_row(
                Text(tier, style=color),
                str(count),
                Text(bar, style=color),
                f"{pct:.1f}%",
            )

        # Recent requests
        recent_table = Table(title="Last 10 Requests", show_header=True, header_style="bold")
        recent_table.add_column("Time", style="dim", max_width=8)
        recent_table.add_column("Tier", max_width=10)
        recent_table.add_column("Model", max_width=35)
        recent_table.add_column("Latency", justify="right", max_width=8)
        recent_table.add_column("Tokens", justify="right", max_width=8)

        for e in reversed(last_10):
            ts_str = e.get("timestamp", "")
            try:
                ts = datetime.fromisoformat(ts_str)
                time_str = ts.strftime("%H:%M:%S")
            except (ValueError, TypeError):
                time_str = "?"
            tier = e.get("tier", "?")
            model = e.get("selected_model", "?")
            if len(model) > 35:
                model = model[:32] + "..."
            latency = e.get("total_latency_ms")
            lat_str = f"{latency:.0f}ms" if latency else "?"
            tok = _safe_int(e.get("prompt_tokens", 0)) + _safe_int(e.get("completion_tokens", 0))

            color = tier_colors.get(tier, "white")
            recent_table.add_row(
                time_str,
                Text(tier, style=color),
                model,
                lat_str,
                f"{tok:,}" if tok else "?",
            )

        # Compose layout
        layout.split_column(
            Layout(Panel(header_text, border_style="cyan"), size=9),
            Layout(name="body"),
        )
        layout["body"].split_row(
            Layout(
                Panel(stats, title="Stats", border_style="green"),
                ratio=1,
            ),
            Layout(name="right", ratio=2),
        )
        layout["right"].split_column(
            Layout(Panel(tier_table, border_style="blue"), ratio=1),
            Layout(Panel(recent_table, border_style="yellow"), ratio=2),
        )

        return layout

    with Live(make_display(), console=console, refresh_per_second=0.5, screen=True) as live:
        try:
            while True:
                time.sleep(refresh)
                live.update(make_display())
        except KeyboardInterrupt:
            pass
    console.print("\n[dim]Dashboard closed.[/dim]")


def run_dashboard_basic(log_path: Path, refresh: float = 2.0, db_path: Optional[Path] = None):
    """Fallback dashboard without Rich, using basic terminal output."""
    start_time = time.time()

    try:
        while True:
            os.system("clear" if os.name != "nt" else "cls")
            entries = _load_entries(log_path, db_path)
            total = len(entries)
            uptime = time.time() - start_time

            # Tier counts
            tiers: Dict[str, int] = {}
            for e in entries:
                tier = e.get("tier", "unknown")
                tiers[tier] = tiers.get(tier, 0) + 1

            # Cost
            actual_cost = calculate_actual_cost(entries)
            baseline_model = "claude-sonnet-4-5-20250929"
            baseline_cost = calculate_hypothetical_cost(entries, baseline_model)
            savings = baseline_cost - actual_cost
            savings_pct = (savings / baseline_cost * 100) if baseline_cost > 0 else 0

            print(HEADER)
            print(f"  Uptime: {_format_duration(uptime)}  |  Refreshing every {refresh}s  |  Ctrl+C to exit")
            print()
            print(f"  Total Requests:   {total:,}")
            print(f"  Actual Cost:      ${actual_cost:.4f}")
            print(f"  Without Routing:  ${baseline_cost:.4f}")
            print(f"  Saved:            ${savings:.4f} ({savings_pct:.1f}%)")
            print()

            print("  Routing Distribution")
            print("  " + "-" * 50)
            for tier, count in sorted(tiers.items(), key=lambda x: x[1], reverse=True):
                pct = count / total * 100 if total > 0 else 0
                bar = "█" * int(pct / 2)
                print(f"    {tier:12s} {count:>5} ({pct:4.1f}%) {bar}")

            print()
            print("  Last 5 Requests")
            print("  " + "-" * 50)
            for e in entries[-5:]:
                tier = e.get("tier", "?")
                model = e.get("selected_model", "?")[:30]
                lat = e.get("total_latency_ms", "?")
                print(f"    {tier:10s} {model:30s} {lat}ms")

            time.sleep(refresh)
    except KeyboardInterrupt:
        print("\nDashboard closed.")


def run_dashboard(log_path: Path, refresh: float = 2.0, db_path: Optional[Path] = None):
    """Run the dashboard, using Rich if available, otherwise basic fallback."""
    has_sqlite = db_path is not None and db_path.exists()
    if not has_sqlite and not log_path.exists():
        print("No log file found.")
        print("Start NadirClaw (nadirclaw serve) and make some requests first.")
        return

    try:
        import rich  # noqa: F401
        run_dashboard_rich(log_path, refresh, db_path=db_path)
    except ImportError:
        print("(Install 'rich' for a better dashboard: pip install rich)")
        print()
        run_dashboard_basic(log_path, refresh, db_path=db_path)
