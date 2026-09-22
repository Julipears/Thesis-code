"""Section 4 VECM estimators, runners, and plot/table generation."""
from pathlib import Path
import sys

SECTION_ROOT = Path(__file__).resolve().parent
if str(SECTION_ROOT) not in sys.path:
    sys.path.insert(0, str(SECTION_ROOT))
