"""Section 5.2 AFT models, diagnostics, and tables."""
from pathlib import Path
import sys

SECTION_ROOT = Path(__file__).resolve().parent
if str(SECTION_ROOT) not in sys.path:
    sys.path.insert(0, str(SECTION_ROOT))
