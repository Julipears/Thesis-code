import json
from pathlib import Path

nb_path = Path(r"c:/Users/Julia/OneDrive - University of Toronto/Desktop/School files/4th year/Thesis/Thesis-code/pull_missing_liquidity_all_markets_all_months.ipynb")
data = json.loads(nb_path.read_text(encoding="utf-8"))
for i, cell in enumerate(data.get("cells", [])):
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell.get("source", []))
    if "def prepare_aft_data" in src:
        print(f"Found in cell {i}:\n")
        print(src)
        break
else:
    print("prepare_aft_data not found in notebook")
