"""Section 5.2 AFT models, diagnostics, and tables."""
from pathlib import Path
import sys

from final_code import RESULTS

SECTION_ROOT = Path(__file__).resolve().parent
RAW_METRICS = RESULTS / "open_interest" / "raw_metrics"
_legacy_metrics = Path("open_interest_figures/native/raw_metrics")
if not RAW_METRICS.is_dir() and _legacy_metrics.is_dir():
    RAW_METRICS = _legacy_metrics

if str(SECTION_ROOT) not in sys.path:
    sys.path.insert(0, str(SECTION_ROOT))
