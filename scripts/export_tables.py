#!/usr/bin/env python3
"""Export the frozen final training/FNO tables and regenerate the HYDRUS timing report."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/tables')
    out = parser.parse_args().output_dir.resolve()
    if out == ROOT / 'data' or ROOT / 'data' in out.parents:
        parser.error('Output must be outside the frozen data directory.')
    out.mkdir(parents=True, exist_ok=True)
    frozen = ROOT / 'data/frozen_results'
    for relative in ['training_manifest_detailed.csv', 'training_manifest_summary.csv',
                     'server_audit_20260723/E3/final_results.csv',
                     'server_audit_20260723/E3/search_ranking.csv',
                     'E11/bench_hydrus_multiprocess.csv']:
        source = frozen / relative
        shutil.copy2(source, out / source.name)
    subprocess.run([sys.executable, str(ROOT / 'analysis/E11/analyze_e11.py'),
                    str(frozen / 'E11/bench_hydrus_multiprocess.json'),
                    '--out', str(out / 'hydrus_concurrency_report.md')], check=True, cwd=ROOT)
    print(f'Exported frozen tables and regenerated timing report to {out}')


if __name__ == '__main__':
    main()
