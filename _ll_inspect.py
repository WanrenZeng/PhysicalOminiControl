"""Inspect the DiT's LL output on TRAIN samples using the latest checkpoint.

Loads runs/.../checkpoints/last.ckpt (joint stage), applies the DiT EMA (matches
validation), runs sample_infrared on a few FLIR train samples, and saves a
5-panel figure per sample:
    visible | GT infrared | GT LL | DiT-predicted LL | HFRM reconstruction
Also reports PSNR/SSIM of the predicted LL vs the GT LL (DiT quality) and of the
final reconstruction vs GT infrared (end-to-end quality), to tell whether the
bottleneck is the DiT's LL or the HFRM.
"""
from __future__ import annotations

import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from omini.physical import train as T
from omini.physical.data import build_dataset
from omini.physical.ema import ExponentialMovingAverage

DEV = torch.device("cuda")
_SSIM = None
# pick the run dir that actually holds a checkpoint (DDP creates an empty
# timestamped dir per rank; only rank0 writes config.yaml + checkpoints)
CKPT = sorted(glob.glob("runs/physical/flir_joint/*/checkpoints/last.ckpt"))[-1]
RUN = CKPT[: -len("checkpoints/last.ckpt")]
OUT = RUN + "ll_inspect"
N_SAMPLES = 6
STEPS = 28


def to01(x):  # wavelet/LL coeff domain [-1,1] -> [0,1]
    return ((x + 1) / 2).clamp(0, 1)


def psnr(a, b):  # a,b in [0,1]
    mse = (a - b).square().mean().item()
    return float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)


def ssim(a, b):
    # class-based SSIM (matches validation.py, works on torchmetrics 0.7.0)
    global _SSIM
    if _SSIM is None:
        from torchmetrics.image.ssim import SSIM
        _SSIM = SSIM(data_range=1.0).to(DEV).eval()
    return float(_SSIM(a.unsqueeze(0), b.unsqueeze(0)))


def panel(tensors, titles, path):
    imgs = []
    for t in tensors:
        arr = (t.detach().float().cpu().clamp(0, 1).mul(255).round().to(torch.uint8)
               .permute(1, 2, 0).numpy())
        imgs.append(Image.fromarray(arr, "RGB"))
    h = min(im.height for im in imgs)
    imgs = [im.resize((int(im.width * h / im.height), h)) if im.height != h else im for im in imgs]
    W = sum(im.width for im in imgs)
    canvas = Image.new("RGB", (W, h), "black")
    x = 0
    draw = ImageDraw.Draw(canvas)
    for im, title in zip(imgs, titles):
        canvas.paste(im, (x, 0))
        draw.rectangle([x, 0, x + 9 * len(title) + 6, 16], fill="black")
        draw.text((x + 3, 2), title, fill="white")
        x += im.width
    canvas.save(path)


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    cfg = T._load_config(RUN + "config.yaml")
    model = T._build_model(cfg).to(DEV)
    model.set_vae_device(DEV)

    ck = torch.load(CKPT, map_location="cpu", mmap=True, weights_only=False)
    sd = {k[len("model."):]: v for k, v in ck["state_dict"].items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"ckpt global_step={ck.get('global_step')} epoch={ck.get('epoch')}")
    print(f"load_state_dict -> missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing sample:", missing[:5])
    if unexpected:
        print("  unexpected sample:", unexpected[:5])

    # apply DiT EMA (validation uses EMA)
    if "dit_ema" in ck and model.dit is not None:
        ema = ExponentialMovingAverage(model.dit.parameters())
        ema.load_state_dict(ck["dit_ema"])
        ema.copy_to(model.dit.parameters())
        print("applied DiT EMA (optimization_step=%d)" % ck["dit_ema"]["optimization_step"])

    model.eval()

    ds = build_dataset(cfg["dataset"]["name"], cfg["dataset"]["train_annotations"],
                       cfg["dataset"]["root"], horizontal_flip_prob=0.0)
    idxs = np.linspace(0, len(ds) - 1, N_SAMPLES).astype(int).tolist()
    print(f"train samples: {len(ds)} | inspecting idx={idxs}")

    ll_p, ll_s, rec_p = [], [], []
    for i in idxs:
        s = ds[int(i)]
        visible = s["visible"].unsqueeze(0).to(DEV)
        infrared = s["infrared"].unsqueeze(0).to(DEV)
        batch = {"visible": visible, "infrared": infrared}
        comps = model._wavelet_components(batch)
        gt_ll = comps["infrared_ll"]                       # [-1,1], (1,3,H/2,W/2)
        recon, pred_ll, _ = model.sample_infrared(visible, num_inference_steps=STEPS)

        gt_ll01 = to01(gt_ll)[0]
        pred_ll01 = to01(pred_ll)[0]
        recon01 = recon[0].clamp(0, 1)
        gt_ir01 = infrared[0]

        ll_psnr = psnr(pred_ll01, gt_ll01)
        ll_ssim = ssim(pred_ll01, gt_ll01)
        rec_psnr = psnr(recon01, gt_ir01)
        ll_p.append(ll_psnr); ll_s.append(ll_ssim); rec_p.append(rec_psnr)

        up = lambda x: F.interpolate(x.unsqueeze(0), scale_factor=2, mode="bilinear", align_corners=False)[0]
        panel(
            [visible[0], gt_ir01, up(gt_ll01), up(pred_ll01), recon01],
            ["visible", "GT_IR", "GT_LL", "DiT_LL", "recon"],
            f"{OUT}/train{int(i):05d}_{s['sample_id']}.png",
        )
        print(f"  idx={int(i):5d} LL_PSNR={ll_psnr:6.2f} LL_SSIM={ll_ssim:.3f} "
              f"recon_PSNR={rec_psnr:6.2f} | predLL[min,max]=[{pred_ll.min():.2f},{pred_ll.max():.2f}]")

    print(f"AVG  LL_PSNR={np.mean(ll_p):.2f}  LL_SSIM={np.mean(ll_s):.3f}  recon_PSNR={np.mean(rec_p):.2f}")
    print("saved panels to:", OUT)


if __name__ == "__main__":
    main()
