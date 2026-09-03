"""Tests for reusable training, checkpoint, and data-loader infrastructure."""

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset

from omini.physical.datamodule import PhysicalDataModule
from omini.physical.ema import ExponentialMovingAverage
from omini.physical.training import build_optimizer, logistic_normal_timesteps
from omini.physical.vae import FluxVAE


class TrainingInfrastructureTests(unittest.TestCase):
    def test_logistic_normal_times_are_inside_unit_interval(self):
        times = logistic_normal_timesteps(1024, device=torch.device("cpu"))
        self.assertEqual(times.shape, (1024,))
        self.assertTrue(torch.all(times > 0))
        self.assertTrue(torch.all(times < 1))

    def test_ema_state_round_trip(self):
        parameter = nn.Parameter(torch.tensor([1.0]))
        ema = ExponentialMovingAverage([parameter], decay=0.9, use_warmup=False)
        parameter.data.fill_(3.0)
        ema.step([parameter])
        state = ema.state_dict()

        restored = ExponentialMovingAverage([parameter], decay=0.5)
        restored.load_state_dict(state)
        parameter.data.zero_()
        restored.copy_to([parameter])

        self.assertAlmostEqual(parameter.item(), 1.2)
        self.assertEqual(restored.optimization_step, 1)

    def test_optimizer_factory(self):
        parameter = nn.Parameter(torch.ones(1))
        optimizer = build_optimizer(
            [parameter],
            {"type": "AdamW", "params": {"lr": 1e-4, "weight_decay": 0.01}},
        )
        self.assertIsInstance(optimizer, torch.optim.AdamW)

    def test_flux_vae_is_registered_torch_module(self):
        self.assertTrue(issubclass(FluxVAE, nn.Module))

    def test_datamodule_can_disable_validation(self):
        config = {
            "name": "flir",
            "root": "/unused",
            "train_annotations": "/unused/train.json",
            "validation_annotations": "/unused/validation.json",
            "enable_validation": False,
        }
        datamodule = PhysicalDataModule(config, train_batch_size=1, num_workers=0)
        self.assertIsNone(datamodule.val_dataloader())


if __name__ == "__main__":
    unittest.main()
