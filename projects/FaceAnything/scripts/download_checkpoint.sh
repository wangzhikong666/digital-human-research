#!/usr/bin/env bash
# Resume the exact README checkpoint download and verify it before use.
set -euo pipefail
exec python3 "$(dirname "${BASH_SOURCE[0]}")/download_checkpoint.py"
