import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import random
import torch
import numpy as np
from PIL import Image

from huggingface_hub import hf_hub_download
from diffusers.pipelines import FluxPipeline
from datasets import load_dataset

from torchvision import transforms as T
to_tensor = T.ToTensor()
to_pil = T.ToPILImage()

from src.flux.condition import Condition
from src.train.data import ResizedInfraredDataset
from src.flux.generate_infrared import generate
from src.custom.transformer import (
    ConditionFluxTransformer2DModel,
    OCConditionFluxTransformer2DModel,
)
from metric.core import metrics as Metrics
import lpips
from src.train_hfrm.hfrm_ll import HFRMNet
from src.custom.wavelet import iwt_init_channel, dwt_init_channel
from torch.nn import functional as F


def compare_dict_keys(dict1, dict2, dict1_name="字典1", dict2_name="字典2"):
    """
    简单比较两个字典的键差异，输出多了哪些键、少了哪些键
    """
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())

    only_in_dict1 = keys1 - keys2
    only_in_dict2 = keys2 - keys1

    print(f"===== {dict1_name} 与 {dict2_name} 键差异比较 =====")
    if only_in_dict1:
        print(f"\n仅在{dict1_name}中存在的键 ({len(only_in_dict1)}):")
        for key in sorted(only_in_dict1):
            print(f"  {key}")
    else:
        print(f"\n{dict1_name} 没有独有的键")

    if only_in_dict2:
        print(f"\n仅在{dict2_name}中存在的键 ({len(only_in_dict2)}):")
        for key in sorted(only_in_dict2):
            print(f"  {key}")
    else:
        print(f"\n{dict2_name} 没有独有的键")

    print(f"\n共有键数量: {len(keys1 & keys2)}")
    print(f"{dict1_name} 总键数: {len(keys1)}")
    print(f"{dict2_name} 总键数: {len(keys2)}")


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img.convert("RGB"), box=(i % cols * w, i // cols * h))
    return grid


def data_transform(X):
    return 2.0 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


def ll_preprocess(X):
    return (X + 2.0) / 4.0


def ll_postprocess(X):
    return 4.0 * X - 2.0


@torch.inference_mode()
def inference_on_concept101_flux(
    base_model_path,
    lora_path,
    hfrm_path,
    output_dir,
    data_root=None,
    resolution=512,
    config_path=None,
    save_attention=False,
    fp8=False,
    num_inference_steps=28,
):
    lpips_model = lpips.LPIPS(net="alex")

    # HFRM
    hfrm = HFRMNet(in_channels_hf=9, in_channels_ll=3, base_dim=96).to("cuda").to(dtype=torch.bfloat16)
    hfrm_ckpt = torch.load(os.path.join(hfrm_path, "hfrm.pth"))
    hfrm.load_state_dict(hfrm_ckpt)
    hfrm.eval()


    # Load dataset (只取 validation)
    dataset = load_dataset('json', 
            data_files={
                'validation': '/root/zengpeiyi/work_base/dataset/KAIST_clean/test_set.json'
            }
    )["validation"]

    random.seed(42)
    num_samples = 200
    all_indices = list(range(len(dataset)))
    random.shuffle(all_indices)
    dataset = dataset.select(all_indices[:num_samples])

    dataset = ResizedInfraredDataset(
        dataset,
        dataset_root=data_root,
        condition_size=(640, 512),
        target_size=(640, 512),
        padding=None,
        condition_type="kaist",
        drop_text_prob=0,
        drop_image_prob=0,
        return_pil_image=True,
        enable_instruct_description=True
    )

    psnr_log = []
    ssim_log = []
    lpips_log = []
    guidance_scales = [1.0]
    output_root = output_dir

    for guidance_scale in guidance_scales:
        output_dir = os.path.join(output_root, f"cfg_{guidance_scale}")
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "a_metrics.txt"), "w+") as f:
            f.write("PSNR\tSSIM\tLPIP\n")

        for idx, batch in enumerate(dataset):
            image = batch["pil_image"]
            real_image, condition_0 = image
            condition_tensor = batch["condition_0"]
            real_image_tensor = batch["image"]

            filename = os.path.basename(batch["infrared_path"])
            filename, ext = os.path.splitext(filename)

            # data transform -> dwt -> split LL / high
            condition_tensor = data_transform(condition_tensor.unsqueeze(0))
            condition_img_dwt = dwt_init_channel(condition_tensor)
            condition_img_LL, condition_img_high = (
                condition_img_dwt[:, :3, ...],
                condition_img_dwt[:, 3:, ...],
            )


            real_image_tensor = data_transform(real_image_tensor.unsqueeze(0))
            real_image_dwt = dwt_init_channel(real_image_tensor)
            real_image_LL, real_image_high = (
                real_image_dwt[:, :3, ...],
                real_image_dwt[:, 3:, ...],
            )

            real_image_LL = real_image_LL.to(
                next(hfrm.parameters()).device, next(hfrm.parameters()).dtype
            )
            condition_img_high = condition_img_high.to(
                next(hfrm.parameters()).device, next(hfrm.parameters()).dtype
            )

            infrared_img_high_predict = hfrm(real_image_LL, condition_img_high)
            infrared_img_predict = iwt_init_channel(torch.cat((real_image_LL, infrared_img_high_predict), dim=1))
            infrared_img_predict = inverse_data_transform(infrared_img_predict).to(torch.float32).to("cpu")
            infrared_img_predict = to_pil(infrared_img_predict[0])
            

            imgs = []
            imgs.append(condition_0)
            imgs.append(infrared_img_predict)
            imgs.append(real_image)

            grid_imgs = image_grid(imgs, 1, 3)
            grid_imgs.save(os.path.join(output_dir, f"{filename}_grid{ext}"))

            # 预测分量（使用你现有的封装流程：to_pil -> np.array）
            pred = np.array(infrared_img_predict).astype(np.uint8)

            # GT 分量
            real = np.array(real_image).astype(np.uint8)

            # PSNR
            psnr = Metrics.calculate_psnr(pred, real)
            psnr_log.append(psnr)

            # SSIM
            ssim = Metrics.calculate_ssim(pred, real)
            ssim_log.append(ssim)

            # LPIPS
            lpips_metric = Metrics.calculate_lpips(pred, real, lpips_model)
            lpips_log.append(lpips_metric)

            print(f"PSNR: {psnr:.2f}, SSIM: {ssim:.4f}, LPIPS: {lpips_metric:.4f}")
            with open(os.path.join(output_dir, "a_metrics.txt"), "a+") as f:
                f.write(f"{filename}{ext}\t{psnr:.2f}\t{ssim:.4f}\t{lpips_metric:.4f}\n")

        # epoch / guidance average
        psnr_avg = torch.stack([torch.tensor(x) for x in psnr_log]).mean()
        ssim_avg = torch.stack([torch.tensor(x) for x in ssim_log]).mean()
        lpips_avg = torch.stack([torch.tensor(x) for x in lpips_log]).mean()
        print(
            f"Average PSNR: {psnr_avg:.2f}, Average SSIM: {ssim_avg:.4f}, Average LPIPS: {lpips_avg:.4f}"
        )
        with open(os.path.join(output_dir, "a_metrics.txt"), "a+") as f:
            f.write(
                f"PSNR: {psnr_avg:.2f}, SSIM: {ssim_avg:.4f}, LPIPS: {lpips_avg:.4f}\n"
            )


if __name__ == "__main__":
    data_root = "/root/zengpeiyi/work_base/dataset/KAIST_clean"
    resolution = 512

    # steps = [2000, 4000, 6000, 8000, 10000, 12000, 14000]
    steps = range(1200, 10000, 1200)
    base_path = "black-forest-labs/FLUX.1-dev"
    config_path = "/root/zengpeiyi/work_base/OminiControl-main/runs_hfrm/wavelet_hfrm_channel_ll_l1msssim_kaist_noise/20260530-154646/config.yaml"
    num_inference_steps_list = [28]

    for step in steps:
        print(f"evaluate {step} step ckpt...")
        hfrm_path = f"/root/zengpeiyi/work_base/OminiControl-main/runs_hfrm/wavelet_hfrm_channel_ll_l1msssim_kaist_noise/20260530-154646/ckpt/{step}"
        for num_inference_steps in num_inference_steps_list:
            output_dir = f"/root/zengpeiyi/work_base/OminiControl-main/runs_hfrm/wavelet_hfrm_channel_ll_l1msssim_kaist_noise/20260530-154646/evaluation/{step}_{num_inference_steps}"
            inference_on_concept101_flux(
                base_path,
                None, 
                hfrm_path,
                output_dir,
                data_root,
                resolution,
                config_path,
                save_attention=False,
                fp8=False,
                num_inference_steps=num_inference_steps,
            )
