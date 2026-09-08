"""Check portable command construction and protection of existing datasets."""
import argparse
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('run_experiment', ROOT / 'scripts/run_experiment.py')
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class LauncherTests(unittest.TestCase):
    def test_deeponet_family_runs_both_stages_before_evaluation(self):
        args = argparse.Namespace(stages='train,evaluate', seed=123, device='cpu', epochs=None,
                                  model='regshift', hydrus_exe='hydrus')
        commands = RUNNER.make_commands(args, Path('/tmp/config with spaces.yaml'), Path('/tmp/run with spaces'))
        self.assertEqual([Path(c[1]).name for c in commands],
                         ['04_train_m1.py', '05_train_m2.py', '06_evaluate.py'])
        self.assertTrue(all(c[c.index('--config') + 1] == '/tmp/config with spaces.yaml' for c in commands))
        self.assertEqual(commands[2][commands[2].index('--models') + 1], 'm1,m2_data')
        self.assertEqual(commands[2][commands[2].index('--device') + 1], 'cpu')

    def test_out_of_order_stages_are_rejected(self):
        args = argparse.Namespace(stages='train,process')
        with self.assertRaises(ValueError):
            RUNNER.make_commands(args, Path('config.yaml'), Path('run'))

    def test_external_data_is_linked_without_copy_or_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source, run = base / 'external data', base / 'run'
            (source / 'processed').mkdir(parents=True)
            content = source / 'processed/train.npz'
            content.write_bytes(b'test fixture')
            cfg = {'paths': {'processed_data': 'data/processed', 'parameters': 'data/parameters', 'raw_data': 'data/raw'}}
            RUNNER.attach_data(source, run, cfg)
            self.assertTrue((run / 'data/processed').is_symlink())
            self.assertEqual((run / 'data/processed/train.npz').read_bytes(), b'test fixture')
            RUNNER.attach_data(source, run, cfg)
            other = base / 'other data'
            (other / 'processed').mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                RUNNER.attach_data(other, run, cfg)
            self.assertEqual(content.read_bytes(), b'test fixture')


if __name__ == '__main__':
    unittest.main()
