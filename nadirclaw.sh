#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
INSTALL_DIR="${NADIRCLAW_HOME:-$SCRIPT_DIR/.nadirclaw}"
export NADIRCLAW_HOME="$INSTALL_DIR"

"$INSTALL_DIR/venv/bin/nadirclaw" "$@"
