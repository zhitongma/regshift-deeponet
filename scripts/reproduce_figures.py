#!/usr/bin/env python3
"""Reproduce the 22 final manuscript figure files from small bundled inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def reproduce(output: Path, skip_diagrams: bool = False) -> list[Path]:
    output = output.resolve()
    if output == ROOT or ROOT / 'data' == output or ROOT / 'data' in output.parents:
        raise ValueError('Choose an output directory outside the source and frozen data directories.')
    manifest = json.loads((ROOT / 'plotting/figure_manifest.json').read_text())
    if skip_diagrams:
        manifest = [row for row in manifest if not row['script'].endswith('.tex')]
    elif shutil.which('pdflatex') is None:
        raise RuntimeError('pdflatex with TikZ/standalone is required for Fig. 1d and 2. '
                           'Use --skip-diagrams to regenerate the other 20 figures.')
    with tempfile.TemporaryDirectory(prefix='regshift_figures_') as tmp:
        stage = Path(tmp)
        env = dict(os.environ, MPLBACKEND='Agg', MPLCONFIGDIR=str(stage / 'mpl_cache'))
        for script in dict.fromkeys(row['script'] for row in manifest):
            if script.endswith('.tex'):
                shutil.copy2(ROOT / 'plotting' / script, stage / Path(script).name)
                command = ['pdflatex', '-no-shell-escape', '-interaction=nonstopmode',
                           '-halt-on-error', Path(script).name]
                result = subprocess.run(command, cwd=stage, env=env, capture_output=True, text=True)
                if result.returncode:
                    raise RuntimeError(f'{script} failed:\n{result.stdout[-6000:]}\n{result.stderr}')
            else:
                command = [sys.executable, str(ROOT / 'plotting' / script), '--output-dir', str(stage)]
                if script == 'generate_corrected_field_figures.py':
                    command += ['--data-dir', str(ROOT / 'data/plot_inputs/corrected_field_evidence')]
                subprocess.run(command, cwd=ROOT, env=env, check=True)
        required = [stage / (row['stem'] + '.pdf') for row in manifest]
        missing = [str(p.name) for p in required if not p.is_file() or p.stat().st_size == 0]
        if missing:
            raise RuntimeError(f'Figure generation incomplete: {missing}')
        output.mkdir(parents=True, exist_ok=True)
        written = []
        for row in manifest:
            for ext in ['.pdf', '.png']:
                source = stage / (row['stem'] + ext)
                if source.exists():
                    target = output / (row['figure'] + ext)
                    shutil.copy2(source, target)
                    written.append(target)
        for name in ['corrected_regime_gain_summary.csv', 'corrected_figure_manifest.json']:
            if (stage / name).is_file():
                shutil.copy2(stage / name, output / name)
        report = {'pdf_count': len(required), 'skipped_diagrams': skip_diagrams,
                  'figures': manifest,
                  'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in written}}
        (output / 'generation_manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'PASS: {len(required)} figure PDFs written to {output}')
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/figures')
    parser.add_argument('--skip-diagrams', action='store_true', help='Skip the two TikZ figures.')
    args = parser.parse_args()
    reproduce(args.output_dir, args.skip_diagrams)


if __name__ == '__main__':
    main()
