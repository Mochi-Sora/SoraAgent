"""Mason-inspired terminal UI: a calm palette, panels, and tables."""

from __future__ import annotations

import sys

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

ACCENT = "#6B9BFA"
BORDER = "#4E5254"
TEXT = "#A9B7C6"
MUTED = "#80868B"
SUCCESS = "#6AAB73"
WARNING = "#D7BA7D"
ERROR = "#F07178"

console = Console(highlight=False, soft_wrap=True)


def clear() -> None:
    if sys.stdout.isatty():
        console.clear()


def banner(name: str, icon: str, version: str, subtitle: str = "") -> None:
    """Print the app header used at the top of the wizard."""
    title = f"{icon} {name}".strip()
    body = Table.grid(expand=True)
    body.add_column(ratio=1)
    body.add_row(
        f"[bold {TEXT}]{escape(title)}[/] [{MUTED}]v{escape(version)}[/]"
    )
    if subtitle:
        body.add_row(f"[{MUTED}]{escape(subtitle)}[/]")
    console.print()
    console.print(
        Panel(
            body,
            border_style=BORDER,
            padding=(0, 2),
        )
    )
    console.print()


def section(title: str, subtitle: str = "") -> None:
    console.print()
    if subtitle:
        console.print(
            Panel(
                f"[{MUTED}]{escape(subtitle)}[/]",
                title=f"[bold {TEXT}]{escape(title)}[/]",
                border_style=BORDER,
                padding=(0, 2),
            )
        )
    else:
        console.rule(f"[bold {TEXT}]{escape(title)}[/]", style=BORDER)
    console.print()


def kv_table(
    title: str,
    rows: list[tuple[str, str]],
    *,
    footer: str = "",
) -> None:
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Field", style=ACCENT, no_wrap=True)
    table.add_column("Value", style=TEXT)
    for label, value in rows:
        table.add_row(escape(label), escape(value))
    panel = Panel(
        table,
        title=f"[bold {TEXT}]{escape(title)}[/]",
        border_style=BORDER,
        padding=(0, 1),
    )
    console.print(panel)
    if footer:
        console.print(f"[{MUTED}]{escape(footer)}[/]")


def grid_table(
    title: str,
    headers: list[str],
    rows: list[tuple[str, ...]],
) -> None:
    table = Table(
        title=f"[bold {TEXT}]{escape(title)}[/]",
        border_style=BORDER,
        header_style=f"bold {ACCENT}",
        show_lines=False,
    )
    for header in headers:
        table.add_column(header, style=TEXT)
    for row in rows:
        table.add_row(*(escape(value) for value in row))
    console.print(table)


def success(message: str) -> None:
    console.print(f"[{SUCCESS}]✓[/] [{TEXT}]{escape(message)}[/]")


def warning(message: str) -> None:
    console.print(f"[{WARNING}]![/] [{TEXT}]{escape(message)}[/]")


def error(message: str) -> None:
    console.print(f"[{ERROR}]✗[/] [{TEXT}]{escape(message)}[/]")


def info(message: str) -> None:
    console.print(f"[{MUTED}]{escape(message)}[/]")


def hint(message: str) -> None:
    console.print(f"[{ACCENT}]{escape(message)}[/]")


def blank() -> None:
    console.print()


def label(text: str) -> Text:
    return Text(text, style=TEXT)


def warning_panel(title: str, lines: list[str]) -> None:
    body = Text()
    for index, line in enumerate(lines):
        if index:
            body.append("\n")
        body.append(line, style=TEXT)
    console.print(
        Panel(
            body,
            title=f"[bold {WARNING}]{escape(title)}[/]",
            border_style=WARNING,
            padding=(1, 2),
        )
    )
