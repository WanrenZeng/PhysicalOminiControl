"""Tests for DDP-safe validation infrastructure without network metric weights."""

import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset

from omini.physical.validation import (
    DistributedEvaluationSampler,
    ValidationMetrics,
    build_validation_dataloader,
)


class IndexDataset(Dataset[int]):
    def __init__(self, size: int):
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> int:
        return index


def _distributed_reduction_worker(
    rank: int, world_size: int, init_file: str, result_directory: str
) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        metrics = ValidationMetrics(
            Path(result_directory) / "metrics",
            "cpu",
            save_images=False,
            compute_fid=False,
        )
        metrics.begin_epoch(1)
        if rank == 0:
            metrics._sums = torch.tensor([10.0, 0.5, 0.1], dtype=torch.float64)
            metrics._count = torch.tensor([1.0], dtype=torch.float64)
        else:
            metrics._sums = torch.tensor([30.0, 1.5, 0.5], dtype=torch.float64)
            metrics._count = torch.tensor([3.0], dtype=torch.float64)

        result = metrics.compute()
        torch.save(result, Path(result_directory) / f"rank_{rank}.pt")
    finally:
        dist.destroy_process_group()


class ValidationInfrastructureTests(unittest.TestCase):
    def test_distributed_sampler_shards_without_padding_or_duplicates(self):
        dataset = IndexDataset(10)
        shards = [
            list(DistributedEvaluationSampler(dataset, num_replicas=3, rank=rank))
            for rank in range(3)
        ]

        self.assertEqual(shards, [[0, 3, 6, 9], [1, 4, 7], [2, 5, 8]])
        self.assertEqual(sorted(index for shard in shards for index in shard), list(range(10)))

    def test_validation_loader_is_deterministic(self):
        loader = build_validation_dataloader(
            IndexDataset(5), batch_size=2, num_workers=0, pin_memory=False
        )
        batches = [batch.tolist() for batch in loader]
        self.assertEqual(batches, [[0, 1], [2, 3], [4]])

    def test_scalar_reduction_contract_without_external_metric_packages(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            metrics = ValidationMetrics(
                temporary_directory,
                "cpu",
                save_images=False,
                compute_fid=False,
            )
            metrics.begin_epoch(1)
            metrics._sums = torch.tensor([20.0, 1.5, 0.4], dtype=torch.float64)
            metrics._count = torch.tensor([2.0], dtype=torch.float64)

            result = metrics.compute()

            self.assertAlmostEqual(result["psnr"], 10.0)
            self.assertAlmostEqual(result["ssim"], 0.75)
            self.assertAlmostEqual(result["lpips"], 0.2)
            self.assertTrue(torch.isnan(torch.tensor(result["fid_clean"])))
            self.assertEqual(metrics.epoch_dir, Path(temporary_directory) / "epoch_1")

    def test_ddp_scalar_reduction_matches_global_sample_average(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            init_file = Path(temporary_directory) / "distributed_init"
            mp.spawn(
                _distributed_reduction_worker,
                args=(2, str(init_file), temporary_directory),
                nprocs=2,
                join=True,
            )

            expected = {"psnr": 10.0, "ssim": 0.5, "lpips": 0.15}
            for rank in range(2):
                result = torch.load(
                    Path(temporary_directory) / f"rank_{rank}.pt",
                    weights_only=True,
                )
                for metric_name, value in expected.items():
                    self.assertAlmostEqual(result[metric_name], value)
                self.assertTrue(torch.isnan(torch.tensor(result["fid_clean"])))


if __name__ == "__main__":
    unittest.main()
