"""
Curriculum OC-CCL: precompute cycle reconstruction quality, then train on top-n%.

Step 1: For each training sample, run the OC-CCL cycle (no training) and measure
        how well the closing mask matches the GT mask (reconstruction IoU).
Step 2: Train OC-CCL using only samples above a quality threshold.
Step 3: Evaluate on the test set.
"""

import sys, os, json, time, argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

DEVICE = 'cuda'
SPECIES = '(malleti x plesseni) x malleti'
DATA_ROOT = PROJECT_ROOT / 'data' / 'cambridge_butterfly'


# ── Reuse model building and tracker from oc_ccl ────────────────────────────

def build_model(checkpoint, device='cuda'):
    from hydra.core.global_hydra import GlobalHydra
    from hydra import initialize_config_dir, compose
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    config_dir = str(PROJECT_ROOT / 'src' / 'sst' / 'sam2_configs')
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base='1.2'):
        cfg = compose(config_name='sam2_hiera_l')
        OmegaConf.resolve(cfg)
        model = instantiate(cfg.model, _recursive_=True)
    if checkpoint and os.path.exists(checkpoint):
        sd = torch.load(checkpoint, map_location='cpu')['model']
        model.load_state_dict(sd, strict=False)
    return model.to(device)


class Tracker:
    def __init__(self, model):
        self.model = model

    def encode(self, img):
        return self.model.forward_image(img)

    def prep(self, bo):
        return self.model._prepare_backbone_features(bo)

    def track(self, bo, mask, init, od=None):
        _, vf, vp, fs = self.prep(bo)
        if od is None:
            od = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
        if init:
            out = self.model.track_step(0, True, vf, vp, fs, None, mask, od, 2, run_mem_encoder=True)
            od["cond_frame_outputs"][0] = out
        else:
            out = self.model.track_step(1, False, vf, vp, fs, None, None, od, 2, run_mem_encoder=True)
        return out, od


def oc_ccl_cycle(tracker, x0, xu, y0):
    """Run one OC-CCL cycle, return closing mask logits."""
    bo_ref = tracker.encode(x0)
    _, od = tracker.track(bo_ref, y0, init=True)
    bo_q = tracker.encode(xu)
    q_out, _ = tracker.track(bo_q, None, init=False, od=od)
    pred_q = q_out["pred_masks_high_res"]

    rod = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
    soft = (pred_q > 0).float()
    tracker.track(bo_q, soft, init=True, od=rod)
    close_out, _ = tracker.track(bo_ref, None, init=False, od=rod)
    return close_out["pred_masks_high_res"]


def compute_iou(pred_logits, gt):
    pred = (pred_logits > 0).float()
    if gt.shape[-2:] != pred.shape[-2:]:
        gt = F.interpolate(gt, pred.shape[-2:], mode='nearest')
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum() - inter
    return (inter / (union + 1e-8)).item()


# ── Dataset ──────────────────────────────────────────────────────────────────

class FilteredDataset(Dataset):
    """ButterflyOCCCL dataset filtered to specific indices."""
    def __init__(self, base_entries, image_dir, data_root, indices=None, image_size=1024):
        self.entries = [base_entries[i] for i in indices] if indices else base_entries
        self.image_dir = image_dir
        self.data_root = data_root
        self.image_size = image_size

    def __len__(self):
        return len(self.entries)

    def _load_img(self, path):
        p = str(path)
        if p.lower().endswith('.cr2'):
            import rawpy
            with rawpy.imread(p) as raw:
                img = raw.postprocess()
        else:
            img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

    def _load_mask(self, path):
        m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        m = cv2.resize(m, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        return torch.from_numpy((m > 0).astype(np.float32)).unsqueeze(0)

    def __getitem__(self, idx):
        img_id, url, mask_rel, img_path = self.entries[idx]
        x0 = self._load_img(img_path)
        y0 = self._load_mask(self.data_root / mask_rel)
        # Pick random different entry as query
        other = idx
        while other == idx and len(self.entries) > 1:
            other = np.random.randint(len(self.entries))
        xu = self._load_img(self.entries[other][3])
        return x0, y0, xu


# ── Step 1: Precompute reconstruction IoU ────────────────────────────────────

def precompute_recon_iou(checkpoint, device):
    print("Step 1: Precomputing reconstruction IoU for all training samples...")
    model = build_model(checkpoint, device)
    model.eval()
    tracker = Tracker(model)

    # Load all entries
    with open(DATA_ROOT / 'train_test_separate' / SPECIES / 'train_data.json') as f:
        raw_entries = json.load(f)

    entries = []
    for e in raw_entries:
        ext = Path(e[1]).suffix
        ip = DATA_ROOT / 'images' / f'{e[0]}{ext}'
        mp = DATA_ROOT / e[2]
        if ip.exists() and mp.exists():
            entries.append((e[0], e[1], e[2], str(ip)))

    print(f"  Valid entries: {len(entries)}")

    recon_ious = []
    t0 = time.time()

    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        for i, (img_id, url, mask_rel, img_path) in enumerate(entries):
            # Load reference
            if img_path.lower().endswith('.cr2'):
                import rawpy
                with rawpy.imread(img_path) as raw:
                    img = raw.postprocess()
            else:
                img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (1024, 1024))
            x0 = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

            mask = cv2.imread(str(DATA_ROOT / mask_rel), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
            y0 = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

            # Pick a random query (next entry)
            q_idx = (i + 1) % len(entries)
            q_path = entries[q_idx][3]
            if q_path.lower().endswith('.cr2'):
                import rawpy
                with rawpy.imread(q_path) as raw:
                    q_img = raw.postprocess()
            else:
                q_img = cv2.cvtColor(cv2.imread(q_path), cv2.COLOR_BGR2RGB)
            q_img = cv2.resize(q_img, (1024, 1024))
            xu = torch.from_numpy(q_img).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

            # Run cycle
            closing = oc_ccl_cycle(tracker, x0, xu, y0)
            iou = compute_iou(closing, y0)
            recon_ious.append(iou)

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(entries)}] avg_iou={np.mean(recon_ious):.4f} "
                      f"({time.time()-t0:.0f}s)")

    print(f"  Done. Avg reconstruction IoU: {np.mean(recon_ious):.4f} "
          f"± {np.std(recon_ious):.4f}")
    print(f"  Distribution: min={np.min(recon_ious):.3f} "
          f"25%={np.percentile(recon_ious,25):.3f} "
          f"50%={np.percentile(recon_ious,50):.3f} "
          f"75%={np.percentile(recon_ious,75):.3f} "
          f"max={np.max(recon_ious):.3f}")

    del model, tracker
    torch.cuda.empty_cache()
    return entries, recon_ious


# ── Step 2: Train OC-CCL with filtered data ─────────────────────────────────

def train_filtered(entries, indices, name, checkpoint, device, epochs=10, lr=1e-6):
    print(f"\n  Training [{name}] with {len(indices)} samples on {device}...")
    model = build_model(checkpoint, device)
    model.train()
    for p in model.image_encoder.parameters():
        p.requires_grad = False

    tracker = Tracker(model)
    dataset = FilteredDataset(entries, DATA_ROOT / 'images', DATA_ROOT, indices)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    total_steps = len(loader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps, eta_min=lr*0.01)

    out_dir = PROJECT_ROOT / 'outputs' / 'curriculum' / name
    out_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float('inf')
    history = []
    t0 = time.time()

    for epoch in range(epochs):
        epoch_loss = 0
        n = 0
        for x0, y0, xu in loader:
            x0, y0, xu = x0.to(device), y0.to(device), xu.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                closing = oc_ccl_cycle(tracker, x0, xu, y0)
                if y0.shape[-2:] != closing.shape[-2:]:
                    y0_r = F.interpolate(y0, closing.shape[-2:], mode='bilinear', align_corners=False)
                else:
                    y0_r = y0
                loss = F.binary_cross_entropy_with_logits(closing, y0_r)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            n += 1

        avg = epoch_loss / max(n, 1)
        history.append(avg)
        if avg < best_loss:
            best_loss = avg
            torch.save({'model': model.state_dict(), 'epoch': epoch+1, 'loss': avg},
                       str(out_dir / 'best_model.pt'))

    elapsed = time.time() - t0
    print(f"  [{name}] Done in {elapsed:.0f}s. Best loss={best_loss:.4f}")

    json.dump({'name': name, 'n_samples': len(indices), 'best_loss': best_loss,
               'history': history, 'time': elapsed},
              open(str(out_dir / 'results.json'), 'w'), indent=2)

    del model, tracker
    torch.cuda.empty_cache()
    return str(out_dir / 'best_model.pt')


# ── Step 3: Evaluate ─────────────────────────────────────────────────────────

def evaluate_model(ckpt_path, device):
    """One-shot eval on test set using SAM2 video predictor."""
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
        predictor = instantiate(cfg.model, _recursive_=True)

    sd = torch.load(ckpt_path, map_location='cpu')
    if 'model' in sd:
        sd = sd['model']
    predictor.load_state_dict(sd, strict=False)
    predictor.to(device).eval()

    # Load test + train data
    with open(DATA_ROOT / 'train_test_separate' / SPECIES / 'test_data.json') as f:
        test_entries = json.load(f)
    with open(DATA_ROOT / 'train_test_separate' / SPECIES / 'train_data.json') as f:
        train_entries = json.load(f)

    # Find reference
    ref = None
    for e in train_entries:
        ext = Path(e[1]).suffix
        ip = DATA_ROOT / 'images' / f'{e[0]}{ext}'
        mp = DATA_ROOT / e[2]
        if ip.exists() and mp.exists():
            ref = e
            break

    def load_img(path):
        p = str(path)
        if p.lower().endswith('.cr2'):
            import rawpy
            with rawpy.imread(p) as raw:
                img = raw.postprocess()
        else:
            img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        return cv2.resize(img, (1024, 1024))

    def load_msk(path):
        m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        m = cv2.resize(m, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        return (m > 0).astype(np.uint8)

    ref_ext = Path(ref[1]).suffix
    ref_img = load_img(DATA_ROOT / 'images' / f'{ref[0]}{ref_ext}')
    ref_mask = load_msk(DATA_ROOT / ref[2])

    ious = []
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        for entry in test_entries:
            ext = Path(entry[1]).suffix
            ip = DATA_ROOT / 'images' / f'{entry[0]}{ext}'
            mp = DATA_ROOT / entry[2]
            if not ip.exists() or not mp.exists():
                continue
            q_img = load_img(ip)
            gt = load_msk(mp).astype(np.float32)

            state = predictor.init_state(None, image_inputs=[ref_img.copy(), q_img.copy()],
                                          async_loading_frames=False, offload_video_to_cpu=True,
                                          offload_state_to_cpu=True, verbose=False)
            predictor.reset_state(state)
            predictor.add_new_mask(inference_state=state, frame_idx=0, obj_id=0, mask=ref_mask)
            for fi, _, ml in predictor.propagate_in_video(state, verbose=False):
                if fi == 1:
                    pred = (ml[0] > 0).cpu().numpy().squeeze().astype(np.float32)
                    if pred.shape != gt.shape:
                        pred = cv2.resize(pred.astype(np.uint8), (gt.shape[1], gt.shape[0]),
                                          interpolation=cv2.INTER_NEAREST).astype(np.float32)
                    inter = (pred * gt).sum()
                    union = pred.sum() + gt.sum() - inter
                    ious.append(inter / (union + 1e-8))

    del predictor
    torch.cuda.empty_cache()
    return np.mean(ious), np.std(ious), len(ious)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-6)
    args = parser.parse_args()

    device = f'cuda:{args.gpu}'
    torch.cuda.set_device(device)
    ckpt = str(PROJECT_ROOT / 'checkpoints' / 'sam2_hiera_large.pt')

    print('=' * 65)
    print('Curriculum OC-CCL: Train on top-n% reconstruction quality')
    print('=' * 65)

    # Step 1: Precompute
    entries, recon_ious = precompute_recon_iou(ckpt, device)
    recon_ious = np.array(recon_ious)

    # Sort by reconstruction quality (descending)
    sorted_idx = np.argsort(-recon_ious)

    # Step 2: Train with different thresholds
    percentiles = [25, 50, 75, 100]
    ckpts = {}

    for pct in percentiles:
        n = max(2, int(len(entries) * pct / 100))
        top_indices = sorted_idx[:n].tolist()
        avg_recon = recon_ious[top_indices].mean()
        name = f'top{pct}pct'
        print(f'\n--- {name}: {n} samples, avg recon IoU={avg_recon:.4f} ---')
        ckpts[name] = train_filtered(entries, top_indices, name, ckpt, device,
                                      epochs=args.epochs, lr=args.lr)

    # Step 3: Evaluate all
    print('\n' + '=' * 65)
    print('Step 3: Evaluating all models on test set')
    print('=' * 65)

    # Also evaluate pretrained baseline
    results = {}
    for name, cp in [('pretrained', ckpt)] + list(ckpts.items()):
        iou_mean, iou_std, n_test = evaluate_model(cp, device)
        results[name] = (iou_mean, iou_std, n_test)
        print(f'  {name:15s}  IoU={iou_mean:.4f} ± {iou_std:.4f}  ({n_test} samples)')

    # Summary
    print('\n' + '=' * 65)
    print('SUMMARY')
    print('=' * 65)
    print(f'{"Model":15s} {"N_train":>8s} {"ReconIoU":>10s} {"TestIoU":>10s} {"±":>8s}')
    print('-' * 55)

    # Pretrained
    m, s, _ = results['pretrained']
    print(f'{"pretrained":15s} {"—":>8s} {"—":>10s} {m:10.4f} {s:8.4f}')

    for pct in percentiles:
        name = f'top{pct}pct'
        n = max(2, int(len(entries) * pct / 100))
        top_idx = sorted_idx[:n]
        avg_recon = recon_ious[top_idx].mean()
        m, s, _ = results[name]
        print(f'{name:15s} {n:8d} {avg_recon:10.4f} {m:10.4f} {s:8.4f}')

    # Save
    out = PROJECT_ROOT / 'outputs' / 'curriculum' / 'summary.json'
    json.dump({k: {'iou_mean': v[0], 'iou_std': v[1]} for k, v in results.items()},
              open(str(out), 'w'), indent=2)
    print(f'\nSaved to {out}')


if __name__ == '__main__':
    main()
