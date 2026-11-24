"""Curated source package for the paper pipeline."""
from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
PAPER_DATA = PROJECT_ROOT / "paper_data"
RESULTS = Path("results")
FULL_RUN = RESULTS / "full"

# Reuse saved checkpoints when opening an existing workspace.
_legacy_run = Path("sa_results/km_v8_final_01")
if not (PROJECT_ROOT / FULL_RUN).is_dir() and (PROJECT_ROOT / _legacy_run).is_dir():
    FULL_RUN = _legacy_run

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
