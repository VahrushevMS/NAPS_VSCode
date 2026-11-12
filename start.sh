#!/usr/bin/env bash
set -e
cd -- "$(dirname -- "$0")"
export PYTHONUTF8=1
exec python3 bootstrap.py "$@"
