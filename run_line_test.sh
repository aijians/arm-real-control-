#!/usr/bin/env bash
set -euo pipefail
LINE_TEST_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LINE_TEST_PYTHON="$(/usr/bin/python3 - "$LINE_TEST_ROOT" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
path = Path(json.loads((root/'config/robot.json').read_text())['paths']['python'])
print(path if path.is_absolute() else root/path)
PY
)"
cd -- "$LINE_TEST_ROOT"
export PYTHONDONTWRITEBYTECODE=1
exec "$LINE_TEST_PYTHON" -u "$LINE_TEST_ROOT/line_motion_test.py" "$@"
