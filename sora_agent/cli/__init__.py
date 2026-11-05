"""The optional ``sora`` CLI.

Importing this package stays cheap.  The Typer application is imported only by
:func:`main`, so a missing CLI extra produces a clear install hint instead of a
traceback.
"""

from __future__ import annotations

import sys

_CLI_MODULES = {"typer", "rich", "questionary", "prompt_toolkit", "click"}


def main() -> None:
    try:
        from .main import main as _main
    except ModuleNotFoundError as exc:
        name = (exc.name or "").split(".")[0]
        if name in _CLI_MODULES:
            print(
                "The `sora` command needs the CLI extra:\n"
                "  pip install 'sora-agent[cli]'",
                file=sys.stderr,
            )
            raise SystemExit(1) from exc
        raise
    _main()


__all__ = ["main"]
