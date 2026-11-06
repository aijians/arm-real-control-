#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 0 ]]; then
  printf '启动入口不接收任务参数，请上传TXT到配置中的固定路径。\n' >&2
  exit 2
fi
ARM_REAL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ARM_REAL_PYTHON="$(/usr/bin/python3 - "$ARM_REAL_ROOT" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
path = Path(json.loads((root/'config/robot.json').read_text())['paths']['python'])
print(path if path.is_absolute() else root/path)
PY
)"
cd -- "$ARM_REAL_ROOT"
export PYTHONDONTWRITEBYTECODE=1
exec "$ARM_REAL_PYTHON" -u "$ARM_REAL_ROOT/main.py"
