#!/usr/bin/env bash
set -euo pipefail
# 默认同时管理大臂/小臂；--axis upper 或 --axis lower 保留单轴模式。
PID_TUNER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PID_TUNER_PYTHON="$(/usr/bin/python3 - "$PID_TUNER_ROOT" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
path = Path(json.loads((root/'config/robot.json').read_text())['paths']['python'])
print(path if path.is_absolute() else root/path)
PY
)"
cd -- "$PID_TUNER_ROOT"
export PYTHONDONTWRITEBYTECODE=1
exec "$PID_TUNER_PYTHON" -u -m pid_tuner "$@"
