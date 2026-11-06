"""Questionary prompts with a readable non-interactive fallback.

The wizard is interactive, but tests and dumb terminals should still be able to
drive it.  Every prompt returns a value or raises :class:`WizardCancelled`.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any, Callable

from . import ui


class WizardCancelled(Exception):
    """The user pressed Ctrl-C or chose to cancel."""


def _questionary():
    try:
        import questionary
    except ModuleNotFoundError as exc:  # pragma: no cover - packaging guard
        raise RuntimeError(
            "the setup wizard needs the CLI extra: "
            "pip install 'sora-agent[cli]'"
        ) from exc
    return questionary


def _style(questionary):
    return questionary.Style(
        [
            ("qmark", f"fg:{ui.ACCENT} bold"),
            ("question", f"fg:{ui.TEXT} bold"),
            ("answer", f"fg:{ui.SUCCESS} bold"),
            ("pointer", f"fg:{ui.ACCENT} bold"),
            ("highlighted", f"fg:{ui.ACCENT} bold"),
            ("selected", f"fg:{ui.SUCCESS}"),
            ("separator", f"fg:{ui.MUTED}"),
            ("instruction", f"fg:{ui.MUTED}"),
            ("text", f"fg:{ui.TEXT}"),
            ("disabled", f"fg:{ui.MUTED}"),
        ]
    )


def _normalize(
    choices: Sequence[str] | Sequence[tuple[str, Any]],
) -> list[tuple[str, Any]]:
    normalized: list[tuple[str, Any]] = []
    for item in choices:
        if isinstance(item, tuple):
            normalized.append((str(item[0]), item[1]))
        else:
            normalized.append((str(item), item))
    return normalized


def select(
    message: str,
    choices: Sequence[str] | Sequence[tuple[str, Any]],
    default: Any = None,
    *,
    instruction: str = "(↑/↓ to move, Enter to choose)",
) -> Any:
    normalized = _normalize(choices)
    if not normalized:
        raise WizardCancelled("no choices")
    if not sys.stdin.isatty():
        return _fallback_select(message, normalized, default)

    questionary = _questionary()
    items = [
        questionary.Choice(title=title, value=value)
        for title, value in normalized
    ]
    default_item = next(
        (item for item in items if item.value == default),
        next((item for item in items if item.title == default), None),
    )
    answer = questionary.select(
        message,
        choices=items,
        default=default_item,
        style=_style(questionary),
        instruction=instruction,
        qmark="›",
    ).ask()
    if answer is None:
        raise WizardCancelled()
    return answer


def _fallback_select(
    message: str, choices: list[tuple[str, Any]], default: Any
) -> Any:
    ui.console.print(f"[bold {ui.TEXT}]{message}[/]")
    for index, (title, _) in enumerate(choices, start=1):
        ui.console.print(f"  [{ui.ACCENT}]{index}[/]  {title}")
    default_index = next(
        (
            index
            for index, (_, value) in enumerate(choices, start=1)
            if value == default
        ),
        1,
    )
    raw = input(f"Choose [{default_index}]: ").strip()
    if not raw:
        return choices[default_index - 1][1]
    try:
        return choices[int(raw) - 1][1]
    except (ValueError, IndexError):
        raise WizardCancelled("invalid choice") from None


def text(
    message: str,
    default: str = "",
    *,
    validate: Callable[[str], bool | str] | None = None,
    secret: bool = False,
    instruction: str = "",
) -> str:
    if not sys.stdin.isatty():
        return _fallback_text(message, default, validate)
    questionary = _questionary()
    if secret:
        question = questionary.password(
            message,
            style=_style(questionary),
            qmark="›",
        )
    else:
        question = questionary.text(
            message,
            default=default,
            style=_style(questionary),
            qmark="›",
            instruction=instruction or None,
            validate=validate,
        )
    answer = question.ask()
    if answer is None:
        raise WizardCancelled()
    return str(answer)


def _fallback_text(
    message: str,
    default: str,
    validate: Callable[[str], bool | str] | None,
) -> str:
    suffix = f" [{default}]" if default else ""
    raw = input(f"{message}{suffix}: ")
    value = raw if raw.strip() else default
    if validate is not None:
        result = validate(value)
        if result is not True:
            raise WizardCancelled(str(result))
    return value


def password(message: str, *, allow_blank: bool = False) -> str:
    value = text(message, "", secret=True)
    if not value and not allow_blank:
        raise WizardCancelled("a value is required")
    return value


def confirm(message: str, default: bool = True) -> bool:
    if not sys.stdin.isatty():
        suffix = "Y/n" if default else "y/N"
        raw = input(f"{message} [{suffix}]: ").strip().lower()
        if not raw:
            return default
        return raw in {"y", "yes"}
    questionary = _questionary()
    answer = questionary.confirm(
        message,
        default=default,
        style=_style(questionary),
        qmark="›",
    ).ask()
    if answer is None:
        raise WizardCancelled()
    return bool(answer)


def checkbox(
    message: str,
    choices: Sequence[str] | Sequence[tuple[str, Any]],
    *,
    default: Sequence[Any] = (),
) -> list[Any]:
    normalized = _normalize(choices)
    if not sys.stdin.isatty():
        return list(default)
    questionary = _questionary()
    items = [
        questionary.Choice(
            title=title, value=value, checked=value in default
        )
        for title, value in normalized
    ]
    answer = questionary.checkbox(
        message,
        choices=items,
        style=_style(questionary),
        qmark="›",
    ).ask()
    if answer is None:
        raise WizardCancelled()
    return list(answer)


def pause(message: str = "Press Enter to continue") -> None:
    if sys.stdin.isatty():
        try:
            input(f"{message} ")
        except (EOFError, KeyboardInterrupt):
            raise WizardCancelled() from None
