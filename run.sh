#!/usr/bin/env bash
set -euo pipefail

# Resolve project root dynamically so imports work on Windows/Linux/macOS.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${OS:-}" == "Windows_NT" || "${MSYSTEM:-}" == MINGW* || "${MSYSTEM:-}" == MSYS* ]]; then
  SEP=";"
else
  SEP=":"
fi
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+$SEP$PYTHONPATH}"

cd "$SCRIPT_DIR"
streamlit run app/main.py
