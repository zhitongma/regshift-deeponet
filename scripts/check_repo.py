#!/usr/bin/env python3
"""Check source syntax, bundled-input hashes and optional model/figure smoke tests."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', action='store_true', help='Run model/configuration regression checks (requires PyTorch).')
    parser.add_argument('--figures', action='store_true', help='Reproduce all 22 figure PDFs in a temporary directory.')
    parser.add_argument('--skip-diagrams', action='store_true', help='With --figures, skip the two TikZ diagrams.')
    args = parser.parse_args()
    sources = [p for folder in ['src', 'pipeline', 'analysis', 'plotting', 'scripts', 'tests']
               for p in (ROOT / folder).rglob('*.py')]
    for source in sources:
        ast.parse(source.read_text(encoding='utf-8'), filename=str(source))
    manifest = json.loads((ROOT / 'data/SHA256SUMS.json').read_text())
    for relative, expected in manifest.items():
        path = ROOT / 'data' / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Frozen input missing or changed: {relative}')
    actual = {p.relative_to(ROOT / 'data').as_posix() for p in (ROOT / 'data').rglob('*')
              if p.is_file() and p.name not in {'README.md', 'SHA256SUMS.json'}}
    if actual != set(manifest):
        raise RuntimeError(f'Frozen input manifest coverage changed: {actual ^ set(manifest)}')
    figures = json.loads((ROOT / 'plotting/figure_manifest.json').read_text())
    if len(figures) != 22 or len({row['figure'] for row in figures}) != 22:
        raise RuntimeError('Expected 22 uniquely numbered figure files.')
    for row in figures:
        if not (ROOT / 'plotting' / row['script']).is_file():
            raise FileNotFoundError(row['script'])
    print(f'PASS: {len(sources)} Python sources; {len(manifest)} frozen-input checksums; 22 figure mappings.', flush=True)
    if args.models:
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], cwd=ROOT, check=True)
    if args.figures:
        with tempfile.TemporaryDirectory(prefix='regshift_verify_') as tmp:
            command = [sys.executable, str(ROOT / 'scripts/reproduce_figures.py'), '--output-dir', tmp]
            if args.skip_diagrams:
                command.append('--skip-diagrams')
            subprocess.run(command, cwd=ROOT, check=True)
        # Recheck inputs after drawing to detect any unintended writes.
        for relative, expected in manifest.items():
            if hashlib.sha256((ROOT / 'data' / relative).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f'Figure generation modified frozen input: {relative}')


if __name__ == '__main__':
    main()
