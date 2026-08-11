"""إضافة جذر المشروع إلى sys.path — يُستورد من app.py أو بعد bootstrap في pages/."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
