"""Smoke tests for the standalone physical reconstruction backbone."""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from omini.physical import (
    DATASET_SIZES,
    DMT,
    FLIRDataset,
    HFRMReconstructor,
    IDMT,
    KAISTDataset,
    M3FDDataset,
)
from omini.physical.training import PhysicalTrainingModel


class PhysicalBackboneTests(unittest.TestCase):
    def test_dmt_idmt_is_exact_inverse_and_differentiable(self):
        image = torch.randn(2, 3, 32, 48, requires_grad=True)

        coefficients = DMT()(image)
        reconstructed = IDMT()(coefficients)

        self.assertEqual(coefficients.shape, (2, 12, 16, 24))
        torch.testing.assert_close(reconstructed, image)

        reconstructed.square().mean().backward()
        self.assertIsNotNone(image.grad)

    def test_hfrm_reconstructor_shapes(self):
        model = HFRMReconstructor(base_dim=8)
        infrared_ll = torch.randn(2, 3, 32, 32)
        visible_high = torch.randn(2, 9, 32, 32)

        reconstructed, predicted_high = model(infrared_ll, visible_high)

        self.assertEqual(predicted_high.shape, (2, 9, 32, 32))
        self.assertEqual(reconstructed.shape, (2, 3, 64, 64))

        reconstructed.abs().mean().backward()
        self.assertIsNotNone(model.hfrm.decoder.out_conv.weight.grad)

    def test_hfrm_training_loss_without_dit_or_vae(self):
        model = PhysicalTrainingModel(vae_path="unused", include_dit=False, hfrm_base_dim=8)
        batch = {
            "visible": torch.rand(2, 3, 64, 64),
            "infrared": torch.rand(2, 3, 64, 64),
        }

        loss, logs = model.hfrm_loss(batch)

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(logs), {
            "hfrm_loss",
            "hfrm_high_loss",
            "hfrm_pixel_loss",
            "hfrm_ms_ssim_loss",
        })
        loss.backward()
        self.assertIsNotNone(model.hfrm.decoder.out_conv.weight.grad)

    def test_dataset_presets_and_paired_resize(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            visible_path = root / "visible.png"
            infrared_path = root / "infrared.png"
            Image.new("RGB", (80, 64), (255, 0, 0)).save(visible_path)
            Image.new("L", (80, 64), 127).save(infrared_path)

            annotations_path = root / "pairs.json"
            annotations_path.write_text(
                json.dumps(
                    [
                        {
                            "filename": "example",
                            "vision_path": visible_path.name,
                            "infrared_path": infrared_path.name,
                            "description_vision_only": "ignored for text-free training",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            dataset_types = (
                (FLIRDataset, "flir"),
                (KAISTDataset, "kaist"),
                (M3FDDataset, "m3fd"),
            )
            for dataset_type, dataset_name in dataset_types:
                dataset = dataset_type(
                    annotations_path,
                    root,
                    horizontal_flip_prob=0.0,
                )
                sample = dataset[0]
                width, height = DATASET_SIZES[dataset_name]

                self.assertEqual(sample["dataset_name"], dataset_name)
                self.assertEqual(sample["sample_id"], "example")
                self.assertEqual(sample["visible"].shape, (3, height, width))
                self.assertEqual(sample["infrared"].shape, (3, height, width))
                self.assertEqual(sample["visible"].dtype, torch.float32)
                self.assertTrue(torch.all((sample["visible"] >= 0) & (sample["visible"] <= 1)))


if __name__ == "__main__":
    unittest.main()
