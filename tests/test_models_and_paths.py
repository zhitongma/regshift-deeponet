"""Regression checks for the final coordinate bounds, FNO patch and portable paths."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch
import yaml

from src.models.fno import FNO2d, build_fno_model
from src.utils.experiment import load_config, resolve_artifact_paths

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


class ModelAndPathTests(unittest.TestCase):
    def test_all_final_configs_parse_and_use_nested_bounds(self):
        paths = sorted((ROOT / 'configs').rglob('*.yaml'))
        self.assertEqual(len(paths), 43)
        for path in paths:
            cfg, _ = load_config(ROOT, str(path))
            self.assertIsInstance(cfg, dict)
            self.assertNotIn('shift_deeponet', cfg)

    def test_final_bounded_model_and_training_dispatch(self):
        cfg, _ = load_config(ROOT, 'configs/revision/E0_bounded/e0_regshift_highbudget_iid.yaml')
        for filename in ['04_train_m1.py', '05_train_m2.py', '06_evaluate.py']:
            spec = importlib.util.spec_from_file_location(filename[:-3], ROOT / 'pipeline' / filename)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            model = module.build_model_dispatch(cfg['model'])
            self.assertEqual(model.transform_bound_scale, 0.3)
            self.assertEqual(model.transform_bound_shift, 0.2)
            with torch.no_grad():
                branch = torch.randn(2, cfg['model']['branch_input_dim'])
                trunk = torch.rand(7, 2)
                out = model(branch, trunk)
            self.assertEqual(tuple(out.shape), (2, 7, 2))
            self.assertTrue(torch.isfinite(out).all())

    def test_misnested_historical_config_fails_instead_of_training_unbounded(self):
        cfg, _ = load_config(ROOT)
        cfg['shift_deeponet'] = cfg['model'].pop('shift_deeponet')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bad.yaml'
            path.write_text(yaml.safe_dump(cfg))
            with self.assertRaisesRegex(ValueError, 'model.shift_deeponet'):
                load_config(ROOT, str(path))

    def test_fno_padding_changes_forward_without_changing_checkpoint_schema(self):
        torch.manual_seed(42)
        unpadded = FNO2d(n_params=5, n_z=9, n_t=7, width=4, modes_z=3, modes_t=3, n_layers=2)
        padded = FNO2d(n_params=5, n_z=9, n_t=7, width=4, modes_z=3, modes_t=3,
                       n_layers=2, padding_z=2, padding_t=2)
        padded.load_state_dict(unpadded.state_dict(), strict=True)
        branch = torch.randn(2, 5)
        trunk = torch.zeros(63, 2)
        baseline, output = unpadded(branch, trunk), padded(branch, trunk)
        self.assertEqual(tuple(output.shape), (2, 63, 2))
        self.assertTrue(torch.isfinite(output).all())
        self.assertFalse(torch.allclose(baseline, output))
        output.square().mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in padded.parameters() if p.grad is not None))

    def test_final_fno_config_activates_eight_cell_padding(self):
        cfg, _ = load_config(ROOT, 'configs/revision/E3_fno_final/fno_best_iid.yaml')
        model = build_fno_model({**cfg['model'], 'fno': cfg['fno'], 'physics': cfg['physics']})
        self.assertEqual((model.padding_z, model.padding_t), (8, 8))
        with torch.no_grad():
            output = model(torch.randn(1, cfg['model']['branch_input_dim']), torch.zeros(101 * 49, 2))
        self.assertEqual(tuple(output.shape), (1, 101 * 49, 2))

    def test_default_outputs_do_not_write_into_bundled_data(self):
        cfg, _ = load_config(ROOT)
        paths = resolve_artifact_paths(ROOT, cfg)
        self.assertEqual(paths.root, ROOT / 'outputs/default_run')
        explicit = resolve_artifact_paths(ROOT, cfg, 'outputs/runs/test')
        self.assertEqual(explicit.processed_data, ROOT / 'outputs/runs/test/data/processed')


if __name__ == '__main__':
    unittest.main()
