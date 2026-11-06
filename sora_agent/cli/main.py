"""The ``sora`` command line: setup, chat, gateway, config, and doctor."""

from __future__ import annotations

import json
import sys
from typing import Any

import typer

from .. import __version__
from ..channels.cli import main as gateway_main
from ..model_catalog import provider_info
from ..runtime_setup import provider_kind
from ..settings import (
    MODES,
    Settings,
    SettingsError,
    config_path,
    load_secrets,
    load_settings,
    save_settings,
    secrets_path,
    update_secrets,
)
from . import doctor as doctor_module
from . import prompts, ui
from . import update as update_module
from .chat import interactive, run_once
from .setup import run_wizard

app = typer.Typer(
    add_completion=False,
    invoke_without_command=True,
    no_args_is_help=False,
    pretty_exceptions_show_locals=False,
    help="SoraAgent: setup, terminal chat, gateway, and configuration.",
)


@app.callback()
def root(
    ctx: typer.Context,
    version: bool = typer.Option(
        False, "--version", "-V", help="Show the version and exit."
    ),
) -> None:
    if version:
        typer.echo(f"sora-agent {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        raise typer.Exit(_chat_impl())


@app.command()
def setup(
    mode: str = typer.Option(
        None,
        "--mode",
        "-m",
        help="Skip the menu: recommended or advanced.",
    ),
) -> None:
    """Create or refresh ~/.sora_agent/config.json."""
    if mode is not None and mode.lower() not in MODES:
        raise typer.BadParameter(f"choose one of {', '.join(MODES)}")
    if not run_wizard(mode):
        raise typer.Exit(0)


@app.command()
def onboard(
    mode: str = typer.Option(
        None,
        "--mode",
        "-m",
        help="Skip the menu: recommended or advanced.",
    ),
) -> None:
    """Alias for `sora setup`."""
    setup(mode=mode)


@app.command()
def gateway(
    action: str = typer.Argument(
        "",
        help="Optional: `setup` to configure first, or `start`.",
    ),
) -> None:
    """Start the chat gateway, or run gateway-focused setup."""
    if action == "setup":
        if not run_wizard():
            raise typer.Exit(0)
        return
    if action not in {"", "start", "run"}:
        raise typer.BadParameter("use `sora gateway` or `sora gateway setup`")
    try:
        load_settings()
    except SettingsError as exc:
        ui.error(str(exc))
        raise typer.Exit(2) from exc
    raise typer.Exit(gateway_main([]))


@app.command()
def chat() -> None:
    """Chat with Sora in this terminal."""
    raise typer.Exit(_chat_impl())


@app.command("run")
def run_command(
    task: str = typer.Argument(..., help="One task to run, then exit."),
) -> None:
    """Run one task without opening the interactive chat."""
    settings, secrets = _load_for_use()
    if settings is None:
        raise typer.Exit(1)
    raise typer.Exit(run_once(settings, secrets, task))


@app.command()
def doctor() -> None:
    """Check configuration, credentials, paths, and safety posture."""
    settings, secrets = _load()
    raise typer.Exit(doctor_module.run_doctor(settings, secrets))


@app.command()
def update(
    check: bool = typer.Option(
        False,
        "--check",
        "--dry-run",
        help="Check for an update without installing it.",
    ),
    version: str = typer.Option(
        "latest",
        "--version",
        "--ref",
        "-v",
        help="Release tag or branch to install (latest, main, v0.2.0).",
    ),
    extras: str | None = typer.Option(
        None,
        "--extras",
        "-e",
        help="Comma-separated extras, or 'none' for the core only.",
    ),
    repo: str | None = typer.Option(
        None,
        "--repo",
        help="GitHub repository to update from (owner/name).",
    ),
) -> None:
    """Update SoraAgent to the latest release or main branch."""
    try:
        code = update_module.run_update(
            check=check,
            version=version,
            extras=extras,
            repo=repo,
        )
    except update_module.UpdateError as exc:
        ui.error(str(exc))
        raise typer.Exit(2) from exc
    raise typer.Exit(code)


@app.command("set-key")
def set_key(
    target: str = typer.Argument(
        "provider", help="`provider` or `telegram`."
    ),
) -> None:
    """Store an API key or Telegram token in secrets.env."""
    settings, _ = _load()
    if target == "telegram":
        env_name = "SORA_TELEGRAM_TOKEN"
        label = "Telegram bot token"
    elif target == "provider":
        kind = provider_kind(settings) if settings is not None else "openai"
        info = provider_info(kind)
        env_name = info.key_env if info is not None else "OPENAI_API_KEY"
        label = f"{env_name} for {kind}"
    else:
        raise typer.BadParameter("target must be `provider` or `telegram`")
    value = prompts.password(label, allow_blank=False)
    update_secrets({env_name: value})
    ui.success(f"Saved {env_name} to {secrets_path()}")
    ui.info(
        "The environment still overrides the file when the same variable is "
        "set in your shell."
    )


config_app = typer.Typer(
    add_completion=False,
    help="Read and edit ~/.sora_agent/config.json without the wizard.",
)
app.add_typer(config_app, name="config")


@config_app.command("list")
def config_list() -> None:
    """Print the full configuration as JSON."""
    settings, _ = _load()
    if settings is None:
        raise typer.Exit(1)
    ui.console.print_json(
        json.dumps(settings.to_dict(), ensure_ascii=False, indent=2)
    )


@config_app.command("get")
def config_get(key: str = typer.Argument(..., help="Dotted key.")) -> None:
    """Read one value, for example `provider.model`."""
    settings, _ = _load()
    if settings is None:
        raise typer.Exit(1)
    try:
        value = _navigate(settings.to_dict(), key)
    except KeyError:
        ui.error(f"unknown key: {key}")
        raise typer.Exit(1) from None
    if isinstance(value, (dict, list)):
        ui.console.print_json(json.dumps(value, ensure_ascii=False))
    else:
        typer.echo(value)


@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help="Dotted key."),
    value: str = typer.Argument(..., help="New value."),
) -> None:
    """Change one value and save the configuration."""
    settings, _ = _load()
    if settings is None:
        ui.error(f"No configuration at {config_path()}; run `sora setup`.")
        raise typer.Exit(1)
    data = settings.to_dict()
    try:
        current = _navigate(data, key)
    except KeyError:
        ui.error(f"unknown key: {key}")
        raise typer.Exit(1) from None
    try:
        coerced = _coerce(value, current)
        _assign(data, key, coerced)
        updated = Settings.from_dict(data)
        save_settings(updated)
    except (SettingsError, ValueError) as exc:
        ui.error(str(exc))
        raise typer.Exit(1) from exc
    ui.success(f"{key}: {current!r} → {coerced!r}")


@config_app.command("unset")
def config_unset(key: str = typer.Argument(..., help="Dotted key.")) -> None:
    """Remove one value so the preset default applies again."""
    settings, _ = _load()
    if settings is None:
        ui.error(f"No configuration at {config_path()}; run `sora setup`.")
        raise typer.Exit(1)
    data = settings.to_dict()
    try:
        _unset(data, key)
        updated = Settings.from_dict(data)
        save_settings(updated)
    except (SettingsError, KeyError) as exc:
        ui.error(str(exc))
        raise typer.Exit(1) from exc
    ui.success(f"Removed {key}.")


def _load() -> tuple[Settings | None, dict[str, str]]:
    try:
        return load_settings(), load_secrets()
    except SettingsError as exc:
        ui.error(str(exc))
        return None, {}


def _load_for_use() -> tuple[Settings | None, dict[str, str]]:
    settings, secrets = _load()
    if settings is not None:
        return settings, secrets
    if not sys.stdin.isatty():
        ui.error("No configuration found. Run `sora setup` first.")
        return None, secrets
    ui.warning("No configuration found; starting setup.")
    if not run_wizard():
        ui.error("Setup did not finish, so there is nothing to run.")
        return None, secrets
    return load_settings(), load_secrets()


def _chat_impl() -> int:
    settings, secrets = _load_for_use()
    if settings is None:
        return 1
    return interactive(settings, secrets)


def _navigate(data: Any, key: str) -> Any:
    current = data
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(key)
        current = current[part]
    return current


def _assign(data: Any, key: str, value: Any) -> None:
    parts = key.split(".")
    current = data
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            raise KeyError(key)
        current = current[part]
    if not isinstance(current, dict) or parts[-1] not in current:
        raise KeyError(key)
    current[parts[-1]] = value


def _unset(data: Any, key: str) -> None:
    parts = key.split(".")
    current = data
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            raise KeyError(key)
        current = current[part]
    if not isinstance(current, dict) or parts[-1] not in current:
        raise KeyError(key)
    del current[parts[-1]]


def _coerce(raw: str, current: Any) -> Any:
    if isinstance(current, bool):
        folded = raw.strip().lower()
        if folded in {"1", "true", "yes", "on"}:
            return True
        if folded in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"expected a boolean, got {raw!r}")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, list):
        text = raw.strip()
        if text.startswith("["):
            parsed = json.loads(text)
            if not isinstance(parsed, list):
                raise ValueError("expected a JSON list")
            return parsed
        return [part.strip() for part in text.split(",") if part.strip()]
    if current is None:
        try:
            return json.loads(raw)
        except ValueError:
            return raw
    return raw


def main() -> None:
    app()
