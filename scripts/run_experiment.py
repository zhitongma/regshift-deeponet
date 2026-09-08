#!/usr/bin/env python3
"""Portable launcher for a single configured training/evaluation run."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def make_commands(args, config: Path, run_dir: Path):
    stages = args.stages.split(',')
    allowed = ['parameters', 'simulate', 'process', 'train', 'evaluate']
    if any(stage not in allowed for stage in stages) or len(set(stages)) != len(stages):
        raise ValueError(f'--stages must be an ordered subset of {allowed}')
    if stages != sorted(stages, key=allowed.index):
        raise ValueError('Stages must follow parameters,simulate,process,train,evaluate order.')
    common = ['--config', str(config), '--run-dir', str(run_dir)]
    commands = []

    def add(script, extra=()):
        commands.append([sys.executable, str(ROOT / 'pipeline' / script), *common, *extra])

    if 'parameters' in stages:
        add('01_generate_params.py')
    if 'simulate' in stages:
        add('02_run_hydrus.py', ['--exe', args.hydrus_exe])
    if 'process' in stages:
        add('03_process_data.py')
    if 'train' in stages:
        extra = ['--seed', str(args.seed)]
        if args.device:
            extra += ['--device', args.device]
        if args.epochs:
            extra += ['--epochs', str(args.epochs)]
        if args.model in {'regshift', 'shift', 'deeponet'}:
            add('04_train_m1.py', extra)
            add('05_train_m2.py', extra)
        else:
            script = {'fnn': '04b_train_fnn.py', 'fno': '04c_train_fno.py',
                      'grid_fnn': '04e_train_grid_fnn.py'}[args.model]
            add(script, extra)
    if 'evaluate' in stages:
        key = 'm1,m2_data' if args.model in {'regshift', 'shift', 'deeponet'} else args.model
        extra = ['--models', key]
        if args.device:
            extra += ['--device', args.device]
        add('06_evaluate.py', extra)
    return commands


def attach_data(source: Path, run_dir: Path, cfg):
    source = source.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f'External data directory does not exist: {source}')
    for sub, key in [('processed', 'processed_data'), ('parameters', 'parameters'), ('raw', 'raw_data')]:
        item = source / sub
        target = run_dir / cfg['paths'][key]
        if item.is_dir():
            if target.exists() or target.is_symlink():
                if target.resolve() != item.resolve():
                    raise FileExistsError(f'Refusing to replace existing data: {target}')
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(item, target_is_directory=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--model', required=True, choices=['regshift', 'shift', 'deeponet', 'fnn', 'fno', 'grid_fnn'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--data-dir', type=Path, help='External directory containing processed/, parameters/, raw/.')
    parser.add_argument('--stages', default='train,evaluate')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--epochs', type=int, help='Override each training stage for smoke runs only.')
    parser.add_argument('--hydrus-exe', default='hydrus')
    parser.add_argument('--dry-run', action='store_true', help='Validate config and print commands without writing files.')
    args = parser.parse_args()
    from src.utils.experiment import load_config
    cfg, config = load_config(ROOT, str(args.config))
    config = config.resolve()
    run_dir = args.run_dir or ROOT / 'outputs/runs' / f'{config.stem}_s{args.seed}'
    run_dir = run_dir.resolve()
    if run_dir == ROOT or run_dir == ROOT / 'data' or ROOT / 'data' in run_dir.parents:
        parser.error('Use a run directory outside the repository source and bundled data directories.')
    model = cfg['model']
    bounds = model.get('shift_deeponet', {})
    arch = model.get('arch', 'deeponet')
    if args.model in {'regshift', 'shift'} and arch != 'shift_deeponet':
        parser.error(f'{args.model} needs model.arch: shift_deeponet.')
    if args.model == 'deeponet' and arch != 'deeponet':
        parser.error('deeponet needs model.arch: deeponet.')
    if args.model == 'regshift' and not all(bounds.get(k, 0) > 0 for k in ['transform_bound_scale', 'transform_bound_shift']):
        parser.error('RegShift requires positive bounds under model.shift_deeponet.')
    if args.model == 'shift' and any(bounds.get(k, 0) != 0 for k in ['transform_bound_scale', 'transform_bound_shift']):
        parser.error('Unbounded Shift requires zero coordinate bounds.')
    if args.model == 'fno' and 'fno' not in cfg:
        parser.error('FNO requires a top-level fno configuration block.')
    if args.epochs is not None and args.epochs <= 0:
        parser.error('--epochs must be positive.')
    commands = make_commands(args, config, run_dir)
    if args.data_dir and any(s in args.stages.split(',') for s in ['parameters', 'simulate', 'process']):
        parser.error('--data-dir mounts existing data; use it with train/evaluate only.')
    for command in commands:
        print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    if args.data_dir:
        attach_data(args.data_dir, run_dir, cfg)
    if any(s in args.stages.split(',') for s in ['train', 'evaluate']) and 'process' not in args.stages.split(','):
        processed = run_dir / cfg['paths']['processed_data']
        missing = [name for name in ['train.npz', 'val.npz', 'test.npz', 'scaler.npz'] if not (processed / name).is_file()]
        if missing:
            raise FileNotFoundError(f'Missing processed dataset under {processed}: {missing}. '
                                    'Supply the external dataset with --data-dir.')
    env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get('PYTHONPATH', ''))
    for command in commands:
        subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    main()
