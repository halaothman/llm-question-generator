"""تشغيل tools/export_human_eval_nonparametric.py لإنتاج CSV التحليل اللاپarametric."""
from __future__ import annotations

import runpy
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "tools" / "export_human_eval_nonparametric.py"

if __name__ == "__main__":
    runpy.run_path(str(_SCRIPT), run_name="__main__")
