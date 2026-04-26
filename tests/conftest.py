"""pytest 全局：把 src 加入 sys.path，避免每个 test 文件都要写。"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = str(PROJECT_ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
