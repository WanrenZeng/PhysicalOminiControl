"""Tests for the compact text-free FLUX-derived DiT."""

import unittest

import torch

from omini.physical import (
    PhysicalFluxDiT,
    PhysicalFluxDiTConfig,
    pack_flux_latents,
    prepare_flux_ids,
    unpack_flux_latents,
)


class PhysicalFluxDiTTests(unittest.TestCase):
    def test_pack_unpack_is_exact_inverse(self):
        latents = torch.randn(2, 16, 8, 10)

        tokens = pack_flux_latents(latents)
        reconstructed = unpack_flux_latents(tokens, latent_height=8, latent_width=10)

        self.assertEqual(tokens.shape, (2, 20, 64))
        torch.testing.assert_close(reconstructed, latents)

    def test_default_configuration_is_about_300m_parameters(self):
        with torch.device("meta"):
            model = PhysicalFluxDiT()

        parameter_count = model.parameter_count
        self.assertGreater(parameter_count, 290_000_000)
        self.assertLess(parameter_count, 300_000_000)

    def test_forward_shape_and_gradients(self):
        config = PhysicalFluxDiTConfig(
            hidden_size=64,
            num_attention_heads=4,
            attention_head_dim=16,
            num_layers=2,
            axes_dims_rope=(4, 6, 6),
        )
        model = PhysicalFluxDiT(config)
        noisy_ir = torch.randn(2, 3, 64, requires_grad=True)
        visible_ll = torch.randn(2, 3, 64)
        timesteps = torch.tensor([0.2, 0.8])
        image_ids = prepare_flux_ids(2, 6)

        velocity = model(
            noisy_ir,
            visible_ll,
            timesteps,
            image_ids=image_ids,
        )

        self.assertEqual(velocity.shape, noisy_ir.shape)
        velocity.square().mean().backward()
        self.assertIsNotNone(model.proj_out.weight.grad)
        self.assertIsNotNone(noisy_ir.grad)

    def test_bad_token_count_is_rejected(self):
        config = PhysicalFluxDiTConfig(
            hidden_size=64,
            num_attention_heads=4,
            attention_head_dim=16,
            num_layers=1,
            axes_dims_rope=(4, 6, 6),
        )
        model = PhysicalFluxDiT(config)
        tokens = torch.randn(1, 3, 64)

        with self.assertRaises(ValueError):
            model(
                tokens,
                tokens,
                torch.tensor([0.5]),
                image_ids=prepare_flux_ids(4, 4),
            )


if __name__ == "__main__":
    unittest.main()
