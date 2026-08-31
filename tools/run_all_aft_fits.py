import json
from pathlib import Path
import tempfile
import subprocess
import sys

NB_PATH = Path(r"c:/Users/Julia/OneDrive - University of Toronto/Desktop/School files/4th year/Thesis/Thesis-code/pull_missing_liquidity_all_markets_all_months.ipynb")

def extract_code_cells(nb_path):
    data = json.loads(nb_path.read_text(encoding='utf-8'))
    sources = []
    for cell in data.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
        src = ''.join(cell.get('source', []))
        # skip execution/demo cells that call the runner functions
        if any(k in src for k in ['process_market(', 'fit_market_aft_models(', 'display(', 'pd.read_parquet(rf"{MARKETS']):
            continue
        sources.append(src)
    return '\n\n'.join(sources)

def make_runner(script_path):
    code = extract_code_cells(NB_PATH)
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write('# Auto-extracted from notebook\n')
        f.write(code)
        f.write('\n\n')
        f.write('if __name__ == "__main__":\n')
        f.write('    # Fit AFT models for all markets defined in MARKETS\n')
        f.write('    for market_name, spec in MARKETS.items():\n')
        f.write('        print("\\nRunning fits for:", market_name)\n')
        f.write('        fit_market_aft_models(spec["run_dir"], overwrite=OVERWRITE_AFT_RESULTS, min_complete_observations=MIN_COMPLETE_OBSERVATIONS)\n')

def run():
    with tempfile.TemporaryDirectory() as td:
        script = Path(td) / 'run_aft_extracted.py'
        make_runner(script)
        print('Executing runner script:', script)
        # Run with same python executable
        p = subprocess.run([sys.executable, str(script)], cwd=NB_PATH.parent)
        return p.returncode

if __name__ == '__main__':
    rc = run()
    if rc != 0:
        print('Runner exited with code', rc)
        sys.exit(rc)
    print('All done')
