"""Evaluate all 16 ablation models on the test set. Runs in parallel across GPUs."""

import sys, os, json, time
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

DATA_ROOT = PROJECT_ROOT / 'data' / 'cambridge_butterfly'
SPECIES = '(malleti x plesseni) x malleti'
ABLATION_DIR = PROJECT_ROOT / 'outputs' / 'ablation'


def find_image(image_id, url):
    ext = Path(url).suffix
    p = DATA_ROOT / 'images' / f'{image_id}{ext}'
    return p if p.exists() else None


def load_image(path, size=1024):
    path = str(path)
    if path.lower().endswith(('.cr2',)):
        import rawpy
        with rawpy.imread(path) as raw:
            img = raw.postprocess()
    else:
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (size, size))


def load_mask(path, size=1024):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.float32)


def compute_iou(pred, gt):
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum() - inter
    return inter / (union + 1e-8)


def build_predictor(ckpt_path, device):
    from hydra.core.global_hydra import GlobalHydra
    from hydra import initialize_config_dir, compose
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    GlobalHydra.instance().clear()
    cfg_dir = str(PROJECT_ROOT / 'src' / 'sst' / 'sam2_configs')
    with initialize_config_dir(config_dir=cfg_dir, version_base='1.2'):
        cfg = compose(config_name='sam2_hiera_l', overrides=[
            '++model._target_=sst.segment_anything_2.sam2.sam2_video_predictor.SAM2VideoPredictor',
        ])
        OmegaConf.resolve(cfg)
        model = instantiate(cfg.model, _recursive_=True)

    sd = torch.load(ckpt_path, map_location='cpu')
    # Handle both formats: {'model': state_dict} and plain state_dict
    if 'model' in sd:
        sd = sd['model']
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model


def eval_one_model(args):
    """Evaluate a single model. Runs in its own process."""
    name, ckpt_path, gpu_id = args
    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(device)

    try:
        predictor = build_predictor(ckpt_path, device)
    except Exception as e:
        return name, {'error': str(e)}

    # Load test + train data
    with open(DATA_ROOT / 'train_test_separate' / SPECIES / 'test_data.json') as f:
        test_entries = json.load(f)
    with open(DATA_ROOT / 'train_test_separate' / SPECIES / 'train_data.json') as f:
        train_entries = json.load(f)

    # Find reference image
    ref_entry = None
    for e in train_entries:
        if find_image(e[0], e[1]) and (DATA_ROOT / e[2]).exists():
            ref_entry = e
            break

    ref_img = load_image(find_image(ref_entry[0], ref_entry[1]))
    ref_mask = load_mask(DATA_ROOT / ref_entry[2])

    # Run one-shot propagation on each test image
    ious = []
    for entry in test_entries:
        img_path = find_image(entry[0], entry[1])
        mask_path = DATA_ROOT / entry[2]
        if img_path is None or not mask_path.exists():
            continue

        query_img = load_image(img_path)
        gt_mask = load_mask(mask_path)

        # 2-frame SST: [ref, query]
        with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
            state = predictor.init_state(
                None, image_inputs=[ref_img.copy(), query_img.copy()],
                async_loading_frames=False, offload_video_to_cpu=True,
                offload_state_to_cpu=True, verbose=False)
            predictor.reset_state(state)
            predictor.add_new_mask(inference_state=state, frame_idx=0, obj_id=0,
                                    mask=cv2.resize(ref_mask.astype(np.uint8), (1024, 1024)))
            for fi, _, ml in predictor.propagate_in_video(state, verbose=False):
                if fi == 1:
                    pred = (ml[0] > 0).cpu().numpy().squeeze().astype(np.float32)
                    if pred.shape != gt_mask.shape:
                        pred = cv2.resize(pred.astype(np.uint8),
                                          (gt_mask.shape[1], gt_mask.shape[0]),
                                          interpolation=cv2.INTER_NEAREST).astype(np.float32)
                    ious.append(compute_iou(pred, gt_mask))

    result = {
        'n_test': len(ious),
        'iou_mean': float(np.mean(ious)),
        'iou_std': float(np.std(ious)),
        'iou_min': float(np.min(ious)),
        'iou_max': float(np.max(ious)),
    }
    return name, result


def main():
    # Collect all ablation checkpoints
    jobs = []
    for d in sorted(ABLATION_DIR.iterdir()):
        if not d.is_dir():
            continue
        ckpt = d / 'best_model.pt'
        if ckpt.exists():
            jobs.append((d.name, str(ckpt)))

    # Also add pretrained baseline
    baseline_ckpt = str(PROJECT_ROOT / 'checkpoints' / 'sam2_hiera_large.pt')
    jobs.append(('pretrained_baseline', baseline_ckpt))

    print(f'Evaluating {len(jobs)} models across 8 GPUs')
    print('=' * 70)

    # Assign GPUs round-robin
    gpu_jobs = [(name, ckpt, i % 8) for i, (name, ckpt) in enumerate(jobs)]

    # Run in parallel (one process per model, 8 GPUs)
    t0 = time.time()
    results = {}

    # Use spawn to avoid CUDA fork issues
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(eval_one_model, job): job[0] for job in gpu_jobs}
        for future in futures:
            name, result = future.result()
            results[name] = result
            if 'error' in result:
                print(f'  {name:20s}  ERROR: {result["error"]}')
            else:
                print(f'  {name:20s}  IoU={result["iou_mean"]:.4f} ± {result["iou_std"]:.4f}  '
                      f'({result["n_test"]} samples)')

    elapsed = time.time() - t0

    # Load ablation configs for the summary table
    print(f'\n{"="*70}')
    print(f'RESULTS (evaluated in {elapsed:.0f}s)')
    print(f'{"="*70}')
    print(f'{"Name":20s} {"IoU":>8s} {"±":>6s} {"Config"}')
    print('-' * 70)

    for name in sorted(results.keys(), key=lambda n: -results[n].get('iou_mean', 0)):
        r = results[name]
        if 'error' in r:
            print(f'{name:20s}  ERROR')
            continue
        # Load config
        rj = ABLATION_DIR / name / 'results.json'
        if rj.exists():
            with open(rj) as f:
                cfg = json.load(f)
            parts = []
            if cfg.get('lr', 1e-5) != 1e-5: parts.append(f'lr={cfg["lr"]}')
            if cfg.get('bce_w', 1.0) != 1.0: parts.append(f'bce={cfg["bce_w"]}')
            if cfg.get('dice_w', 1.0) != 1.0: parts.append(f'dice={cfg["dice_w"]}')
            if cfg.get('lora_rank', 0) > 0: parts.append(f'lora_r={cfg["lora_rank"]}')
            if cfg.get('reset_memory', False): parts.append('mem_reset')
            config_str = ', '.join(parts) if parts else 'baseline'
        else:
            config_str = 'pretrained (no fine-tune)'

        print(f'{name:20s} {r["iou_mean"]:8.4f} {r["iou_std"]:6.4f}  {config_str}')

    # Save all results
    out_path = ABLATION_DIR / 'eval_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved to {out_path}')


if __name__ == '__main__':
    main()
