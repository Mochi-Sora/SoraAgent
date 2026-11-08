#!/usr/bin/env bash
# SoraAgent one-line installer for macOS and Linux.
#
#   curl -fsSL https://raw.githubusercontent.com/Mochi-Sora/SoraAgent/main/install.sh | bash
#
# Installs SoraAgent into an isolated virtual environment and puts the
# `sora`, `sora-gateway`, and `sora-memory` commands on your PATH. Run
# `sora update` to upgrade. Requires Python 3.9 or newer.
set -euo pipefail

REPO="${SORA_REPO:-Mochi-Sora/SoraAgent}"
PACKAGE="sora-agent"
DEFAULT_EXTRAS="openai,memory,cli,files"

VERSION="${SORA_VERSION:-latest}"
EXTRAS="${SORA_EXTRAS-$DEFAULT_EXTRAS}"
SOURCE="${SORA_SOURCE:-}"
INSTALL_DIR="${SORA_INSTALL_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/sora-agent}"
BIN_DIR="${SORA_BIN_DIR:-$HOME/.local/bin}"
PYTHON_BIN="${SORA_PYTHON:-}"

UNINSTALL=0
PURGE=0
NO_PATH=0

usage() {
    cat <<EOF
SoraAgent installer

Usage:
  install.sh [options]

Options:
  -v, --version VERSION   Release tag to install, or "main" (default: latest)
  -e, --extras EXTRAS     Comma-separated extras, e.g. "openai,memory,cli,files",
                          or "none" for the dependency-free core
                          (default: $DEFAULT_EXTRAS)
      --source PATH       Install from a local checkout instead of GitHub
      --install-dir DIR   Install root (default: $INSTALL_DIR)
      --bin-dir DIR       Directory for command wrappers (default: $BIN_DIR)
      --no-path           Do not add the bin directory to your shell profile
      --uninstall         Remove SoraAgent
      --purge             With --uninstall, also delete ~/.sora_agent data
  -h, --help              Show this help

Environment variables:
  SORA_VERSION, SORA_EXTRAS, SORA_SOURCE, SORA_INSTALL_DIR, SORA_BIN_DIR,
  SORA_PYTHON, SORA_REPO

Examples:
  curl -fsSL https://raw.githubusercontent.com/Mochi-Sora/SoraAgent/main/install.sh | bash
  curl -fsSL .../install.sh | SORA_EXTRAS=none bash
  curl -fsSL .../install.sh | bash -s -- --version main
  ./install.sh --uninstall
EOF
}

info() {
    printf '  %s\n' "$*"
}

warn() {
    printf 'warning: %s\n' "$*" >&2
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

is_dangerous_dir() {
    dir=${1%/}
    [ -n "$dir" ] || return 0
    case "$dir" in
        /*) ;;
        *) return 0 ;;
    esac
    case "/$dir/" in
        */../*|*/./*) return 0 ;;
    esac
    # Refuse a drive or filesystem root and any directory directly below it.
    if [ "$(dirname "$dir")" = "/" ]; then
        return 0
    fi
    if [ -n "${HOME:-}" ]; then
        home=${HOME%/}
        [ "$dir" = "$home" ] && return 0
        case "$home/" in
            "$dir/"*) return 0 ;;
        esac
    fi
    return 1
}

strip_path_block() {
    file=$1
    [ -f "$file" ] || return 0
    grep -qF '# >>> sora-agent >>>' "$file" 2>/dev/null || return 0
    tmp_file="$(mktemp "${TMPDIR:-/tmp}/sora-agent.rc.XXXXXX")"
    awk '
        /^# >>> sora-agent >>>$/ { skip = 1; next }
        /^# <<< sora-agent <<<$/ { skip = 0; next }
        !skip { print }
    ' "$file" > "$tmp_file"
    cat "$tmp_file" > "$file"
    rm -f "$tmp_file"
}

remove_wrappers() {
    for name in sora sora-gateway sora-memory; do
        target="$BIN_DIR/$name"
        [ -f "$target" ] || continue
        if grep -qF '# managed by sora-agent installer' "$target" 2>/dev/null \
            || grep -qF "$INSTALL_DIR/venv/bin/$name" "$target" 2>/dev/null; then
            rm -f "$target"
        else
            warn "leaving $target; it was not created by this installer"
        fi
    done
}

uninstall() {
    remove_wrappers
    if [ -d "$INSTALL_DIR" ]; then
        if [ -f "$INSTALL_DIR/venv/pyvenv.cfg" ] \
            || [ -x "$INSTALL_DIR/venv/bin/sora" ]; then
            info "Removing $INSTALL_DIR"
            rm -rf "$INSTALL_DIR"
        else
            warn "refusing to remove $INSTALL_DIR: it does not look like a SoraAgent install"
        fi
    fi

    for rc in "$HOME/.zshrc" "$HOME/.bashrc" "$HOME/.bash_profile" "$HOME/.profile"; do
        strip_path_block "$rc"
    done

    if [ "$PURGE" -eq 1 ]; then
        if [ -d "$HOME/.sora_agent" ]; then
            info "Removing $HOME/.sora_agent (memory, identity, sessions, calendar)"
            rm -rf "$HOME/.sora_agent"
        fi
    else
        info "Keeping user data in $HOME/.sora_agent (use --purge to delete it)"
    fi

    printf '\nSoraAgent uninstalled.\n'
    exit 0
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        -v|--version)
            [ "$#" -ge 2 ] || die "--version requires a value"
            VERSION=$2
            shift 2
            ;;
        --version=*)
            VERSION=${1#*=}
            shift
            ;;
        -e|--extras)
            [ "$#" -ge 2 ] || die "--extras requires a value (or 'none')"
            EXTRAS=$2
            shift 2
            ;;
        --extras=*)
            EXTRAS=${1#*=}
            shift
            ;;
        --source)
            [ "$#" -ge 2 ] || die "--source requires a path"
            SOURCE=$2
            shift 2
            ;;
        --source=*)
            SOURCE=${1#*=}
            shift
            ;;
        --install-dir)
            [ "$#" -ge 2 ] || die "--install-dir requires a path"
            INSTALL_DIR=$2
            shift 2
            ;;
        --install-dir=*)
            INSTALL_DIR=${1#*=}
            shift
            ;;
        --bin-dir)
            [ "$#" -ge 2 ] || die "--bin-dir requires a path"
            BIN_DIR=$2
            shift 2
            ;;
        --bin-dir=*)
            BIN_DIR=${1#*=}
            shift
            ;;
        --no-path)
            NO_PATH=1
            shift
            ;;
        --uninstall)
            UNINSTALL=1
            shift
            ;;
        --purge)
            PURGE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1 (try --help)"
            ;;
    esac
done

if [ -z "${HOME:-}" ] || [ "$HOME" = "/" ]; then
    die "HOME is not set safely"
fi
if is_dangerous_dir "$INSTALL_DIR"; then
    die "refusing to use $INSTALL_DIR as the install root"
fi
if is_dangerous_dir "$BIN_DIR"; then
    die "refusing to use $BIN_DIR as the command directory"
fi

if [ "$UNINSTALL" -eq 1 ]; then
    uninstall
fi

if [ "$PURGE" -eq 1 ]; then
    die "--purge is only valid with --uninstall"
fi

if [ -d "$INSTALL_DIR" ] && [ -n "$(ls -A "$INSTALL_DIR" 2>/dev/null || true)" ] \
    && [ ! -f "$INSTALL_DIR/venv/pyvenv.cfg" ]; then
    die "$INSTALL_DIR is not empty and does not look like a SoraAgent install"
fi

command -v curl >/dev/null 2>&1 || command -v wget >/dev/null 2>&1 \
    || die "curl or wget is required to download SoraAgent"

find_python() {
    if [ -n "$PYTHON_BIN" ]; then
        if [ -x "$PYTHON_BIN" ]; then
            candidate=$PYTHON_BIN
        elif command -v "$PYTHON_BIN" >/dev/null 2>&1; then
            candidate="$(command -v "$PYTHON_BIN")"
        else
            die "SORA_PYTHON=$PYTHON_BIN was not found"
        fi
        if ! "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
            die "SORA_PYTHON=$candidate is not Python 3.9 or newer"
        fi
        PYTHON_BIN=$candidate
        return 0
    fi

    for candidate in python3.14 python3.13 python3.12 python3.11 python3.10 python3.9 python3 python; do
        if command -v "$candidate" >/dev/null 2>&1; then
            candidate="$(command -v "$candidate")"
            if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
                PYTHON_BIN=$candidate
                return 0
            fi
        fi
    done

    die "Python 3.9 or newer is required, but it was not found. Install Python from https://www.python.org/downloads/ and re-run this installer."
}

fetch_stdout() {
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL --max-time 20 "$1"
    else
        wget -q --timeout=20 -O - "$1"
    fi
}

download() {
    url=$1
    dest=$2
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL --retry 2 --connect-timeout 15 "$url" -o "$dest"
    else
        wget -q --timeout=20 -O "$dest" "$url"
    fi
}

latest_tag() {
    json="$(fetch_stdout "https://api.github.com/repos/$REPO/releases/latest" 2>/dev/null || true)"
    [ -n "$json" ] || return 1
    printf '%s' "$json" | "$PYTHON_BIN" -c '
import json
import sys

try:
    tag = json.load(sys.stdin).get("tag_name") or ""
except Exception:
    tag = ""
print(tag)
' 2>/dev/null
}

find_python
info "Using Python $("$PYTHON_BIN" -c 'import sys; print("{}.{}.{}".format(*sys.version_info[:3]))')"

TMP_DIR=""
GET_PIP=""
PIP_LOG="$(mktemp "${TMPDIR:-/tmp}/sora-agent.log.XXXXXX")"
cleanup() {
    if [ -n "$TMP_DIR" ] && [ -d "$TMP_DIR" ]; then
        rm -rf "$TMP_DIR"
    fi
    if [ -n "$GET_PIP" ]; then
        rm -f "$GET_PIP"
    fi
    rm -f "$PIP_LOG"
}
trap cleanup EXIT

if [ -z "$SOURCE" ]; then
    if [ "$VERSION" = "latest" ]; then
        info "Looking up the latest SoraAgent release..."
        TAG="$(latest_tag || true)"
        if [ -n "$TAG" ]; then
            REF="refs/tags/$TAG"
            LABEL=$TAG
        else
            REF="refs/heads/main"
            LABEL="main (no release found)"
        fi
    elif [ "$VERSION" = "main" ] || [ "$VERSION" = "master" ]; then
        REF="refs/heads/$VERSION"
        LABEL=$VERSION
    else
        TAG=$VERSION
        case "$TAG" in
            v*) ;;
            *) TAG="v$TAG" ;;
        esac
        REF="refs/tags/$TAG"
        LABEL=$TAG
    fi

    TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/sora-agent.XXXXXX")"
    ARCHIVE="$TMP_DIR/sora-agent.tar.gz"
    URL="https://github.com/$REPO/archive/$REF.tar.gz"

    info "Downloading SoraAgent $LABEL..."
    if ! download "$URL" "$ARCHIVE"; then
        die "could not download $URL"
    fi
    if ! tar -xzf "$ARCHIVE" -C "$TMP_DIR"; then
        die "could not extract the downloaded archive"
    fi

    SOURCE=""
    for entry in "$TMP_DIR"/*; do
        if [ -d "$entry" ]; then
            SOURCE=$entry
            break
        fi
    done
    [ -n "$SOURCE" ] || die "the downloaded archive did not contain a source directory"
fi

[ -d "$SOURCE" ] || die "source directory not found: $SOURCE"
[ -f "$SOURCE/pyproject.toml" ] || die "$SOURCE does not look like a SoraAgent checkout (pyproject.toml missing)"

VENV_DIR="$INSTALL_DIR/venv"
VENV_PY="$VENV_DIR/bin/python"

mkdir -p "$INSTALL_DIR" "$BIN_DIR"

if [ -x "$VENV_PY" ] && ! "$VENV_PY" -m pip --version >/dev/null 2>&1; then
    info "Existing virtual environment has no pip; rebuilding it..."
    rm -rf "$VENV_DIR"
fi

if [ ! -x "$VENV_PY" ]; then
    info "Creating virtual environment in $VENV_DIR..."
    if ! "$PYTHON_BIN" -m venv "$VENV_DIR" >"$PIP_LOG" 2>&1; then
        # Debian/Ubuntu ship ensurepip in a separate python3-venv package.
        # Create the environment without pip and bootstrap pip inside it, so
        # the installer does not need sudo or system packages.
        rm -rf "$VENV_DIR"
        info "ensurepip is unavailable; bootstrapping pip inside the virtual environment..."
        if ! "$PYTHON_BIN" -m venv --without-pip "$VENV_DIR" >"$PIP_LOG" 2>&1; then
            cat "$PIP_LOG" >&2
            rm -rf "$VENV_DIR"
            die "could not create a virtual environment (install python3-venv for your Python version)"
        fi
        GET_PIP="$(mktemp "${TMPDIR:-/tmp}/sora-agent-get-pip.XXXXXX.py")"
        if ! download "https://bootstrap.pypa.io/get-pip.py" "$GET_PIP"; then
            die "could not download the pip bootstrap script"
        fi
        if ! "$VENV_PY" "$GET_PIP" --quiet >"$PIP_LOG" 2>&1; then
            cat "$PIP_LOG" >&2
            die "could not bootstrap pip inside the virtual environment"
        fi
        rm -f "$GET_PIP"
        GET_PIP=""
    fi
fi

info "Installing $PACKAGE..."
if ! "$VENV_PY" -m pip install --quiet --disable-pip-version-check --upgrade pip setuptools wheel >"$PIP_LOG" 2>&1; then
    cat "$PIP_LOG" >&2
    die "failed to update pip, setuptools, and wheel"
fi

EXTRAS_SPEC=""
if [ -n "$EXTRAS" ] && [ "$EXTRAS" != "none" ]; then
    EXTRAS_SPEC="[$EXTRAS]"
fi

if ! (
    cd "$SOURCE"
    "$VENV_PY" -m pip install --quiet --disable-pip-version-check \
        --no-build-isolation --upgrade --force-reinstall --no-cache-dir ".$EXTRAS_SPEC"
) >"$PIP_LOG" 2>&1; then
    cat "$PIP_LOG" >&2
    die "pip install failed"
fi

for name in sora sora-gateway sora-memory; do
    target="$VENV_DIR/bin/$name"
    [ -x "$target" ] || die "installation completed but $name is missing"
    wrapper_tmp="$BIN_DIR/.sora-agent-$name.$$"
    quoted_target="$(printf '%q' "$target")"
    printf '#!/usr/bin/env sh\n# managed by sora-agent installer\nexec %s "$@"\n' "$quoted_target" > "$wrapper_tmp"
    chmod +x "$wrapper_tmp"
    mv -f "$wrapper_tmp" "$BIN_DIR/$name"
done

PATH_UPDATED=0
case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *)
        if [ "$NO_PATH" -eq 0 ]; then
            case "${SHELL##*/}" in
                zsh) RC_FILE="$HOME/.zshrc" ;;
                bash)
                    if [ -f "$HOME/.bashrc" ]; then
                        RC_FILE="$HOME/.bashrc"
                    else
                        RC_FILE="$HOME/.bash_profile"
                    fi
                    ;;
                *) RC_FILE="$HOME/.profile" ;;
            esac
            [ -f "$RC_FILE" ] || : > "$RC_FILE"
            if ! grep -qF '# >>> sora-agent >>>' "$RC_FILE" 2>/dev/null; then
                quoted_bin="$(printf '%q' "$BIN_DIR")"
                {
                    printf '\n# >>> sora-agent >>>\n'
                    printf 'export PATH=%s:"$PATH"\n' "$quoted_bin"
                    printf '# <<< sora-agent <<<\n'
                } >> "$RC_FILE"
                PATH_UPDATED=1
            fi
        fi
        ;;
esac

"$VENV_DIR/bin/sora" --version >/dev/null 2>&1 \
    || warn "sora did not start correctly"
"$VENV_DIR/bin/sora-gateway" --help >/dev/null 2>&1 \
    || warn "sora-gateway did not start correctly"
"$VENV_DIR/bin/sora-memory" --help >/dev/null 2>&1 \
    || warn "sora-memory did not start correctly"

printf '\nSoraAgent installed.\n\n'
printf '  sora setup\n'
printf '  sora --help\n'
printf '  sora-gateway --help\n'
printf '  sora-memory doctor\n\n'
printf 'Install root: %s\n' "$INSTALL_DIR"
printf 'Commands:     %s\n' "$BIN_DIR"

case ":$PATH:" in
    *":$BIN_DIR:"*)
        ;;
    *)
        if [ "$PATH_UPDATED" -eq 1 ]; then
            printf '\n%s was added to %s. Restart your shell (or run: source %s)\n' "$BIN_DIR" "$RC_FILE" "$RC_FILE"
        else
            printf '\nAdd this to your shell profile to use the commands:\n'
            printf '  export PATH="%s:$PATH"\n' "$BIN_DIR"
        fi
        ;;
esac

case ",$EXTRAS," in
    *",cli,"*)
        printf '\nTo upgrade later, run: sora update\n'
        ;;
    *)
        printf '\nTo upgrade later, re-run this installer (the CLI extra is not installed).\n'
        ;;
esac
printf 'To remove SoraAgent: install.sh --uninstall\n'
