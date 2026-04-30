"""
OC-CCL ablation training with configurable hyperparameters.

Supports: learning rate, loss weights, LoRA, memory reset variants.
"""

import argparse
import math
import os
import sys
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))


# ── LoRA ─────────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """Low-rank adapter wrapping a frozen linear layer."""
    def __init__(self, orig: nn.Linear, rank=4, alpha=1.0):
        super().__init__()
        self.orig = orig
        self.orig.weight.requires_grad = False
        if self.orig.bias is not None:
            self.orig.bias.requires_grad = False
        self.lora_A = nn.Parameter(torch.randn(rank, orig.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(orig.out_features, rank))
        self.scale = alpha / rank

    def forward(self, x):
        return self.orig(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scale


def apply_lora(model, rank=4, alpha=1.0, target_modules=None):
    """Apply LoRA to attention projection layers in memory_attention and mask_decoder."""
    if target_modules is None:
        target_modules = ['q_proj', 'v_proj', 'k_proj', 'out_proj']
    count = 0
    # Collect replacements first to avoid modifying during iteration
    replacements = []
    for name, module in model.named_modules():
        for attr_name in target_modules:
            if hasattr(module, attr_name):
                orig = getattr(module, attr_name)
                if isinstance(orig, nn.Linear):
                    replacements.append((module, attr_name, orig))
    for module, attr_name, orig in replacements:
        lora = LoRALinear(orig, rank=rank, alpha=alpha)
        # Move LoRA params to same device as original
        lora = lora.to(orig.weight.device)
        setattr(module, attr_name, lora)
        count += 1
    return count


# ── Model building ───────────────────────────────────────────────────────────

def build_model(checkpoint, device='cuda'):
    from hydra.core.global_hydra import GlobalHydra
    from hydra import initialize_config_dir, compose
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    config_dir = os.path.join(PROJECT_ROOT, 'src', 'sst', 'sam2_configs')
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base='1.2'):
        cfg = compose(config_name='sam2_hiera_l')
        OmegaConf.resolve(cfg)
        model = instantiate(cfg.model, _recursive_=True)

    if checkpoint and os.path.exists(checkpoint):
        sd = torch.load(checkpoint, map_location='cpu')['model']
        model.load_state_dict(sd, strict=False)

    model = model.to(device)
    return model


# ── Tracker (same as oc_ccl.py) ─────────────────────────────────────────────

class DifferentiableSAM2Tracker:
    def __init__(self, model):
        self.model = model

    def encode_image(self, img):
        return self.model.forward_image(img)

    def prepare_features(self, backbone_out):
        return self.model._prepare_backbone_features(backbone_out)

    def track_with_mask(self, backbone_out, mask, is_init_frame=True, output_dict=None):
        _, vision_feats, vision_pos_embeds, feat_sizes = self.prepare_features(backbone_out)
        if output_dict is None:
            output_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}

        if is_init_frame:
            current_out = self.model.track_step(
                frame_idx=0, is_init_cond_frame=True,
                current_vision_feats=vision_feats,
                current_vision_pos_embeds=vision_pos_embeds,
                feat_sizes=feat_sizes, point_inputs=None, mask_inputs=mask,
                output_dict=output_dict, num_frames=2, run_mem_encoder=True)
            output_dict["cond_frame_outputs"][0] = current_out
        else:
            current_out = self.model.track_step(
                frame_idx=1, is_init_cond_frame=False,
                current_vision_feats=vision_feats,
                current_vision_pos_embeds=vision_pos_embeds,
                feat_sizes=feat_sizes, point_inputs=None, mask_inputs=None,
                output_dict=output_dict, num_frames=2, run_mem_encoder=True)
        return current_out, output_dict


# ── Loss functions ───────────────────────────────────────────────────────────

def bce_loss(pred, target):
    return F.binary_cross_entropy_with_logits(pred, target)

def dice_loss(pred, target, smooth=1.0):
    pred_s = torch.sigmoid(pred)
    pf, tf = pred_s.view(pred_s.size(0), -1), target.view(target.size(0), -1)
    inter = (pf * tf).sum(dim=1)
    union = pf.sum(dim=1) + tf.sum(dim=1)
    return (1.0 - (2.0 * inter + smooth) / (union + smooth)).mean()


# ── OC-CCL step variants ────────────────────────────────────────────────────

def oc_ccl_step(tracker, x0, xu, y0, bce_w=1.0, dice_w=1.0, reset_memory=False):
    """Full OC-CCL cycle with configurable loss weights and memory reset option."""
    # Forward: ref → query
    backbone_ref = tracker.encode_image(x0)
    ref_out, output_dict = tracker.track_with_mask(backbone_ref, y0, is_init_frame=True)

    backbone_query = tracker.encode_image(xu)
    query_out, _ = tracker.track_with_mask(backbone_query, mask=None, is_init_frame=False,
                                            output_dict=output_dict)
    pred_query_mask = query_out["pred_masks_high_res"]

    # Reverse: query → ref
    if reset_memory:
        # Fresh memory for reverse pass (no carry-over from forward)
        reverse_output_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
    else:
        # Reuse backbone features but fresh conditioning
        reverse_output_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}

    soft_query_mask = (pred_query_mask > 0).float()
    query_cond_out, reverse_output_dict = tracker.track_with_mask(
        backbone_query, soft_query_mask, is_init_frame=True, output_dict=reverse_output_dict)

    ref_closing_out, _ = tracker.track_with_mask(
        backbone_ref, mask=None, is_init_frame=False, output_dict=reverse_output_dict)
    pred_closing = ref_closing_out["pred_masks_high_res"]

    # Loss
    if y0.shape[-2:] != pred_closing.shape[-2:]:
        y0_r = F.interpolate(y0, size=pred_closing.shape[-2:], mode='bilinear', align_corners=False)
    else:
        y0_r = y0

    loss = 0.0
    if bce_w > 0:
        loss = loss + bce_w * bce_loss(pred_closing, y0_r)
    if dice_w > 0:
        loss = loss + dice_w * dice_loss(pred_closing, y0_r)

    return loss, pred_closing


# ── Training loop ────────────────────────────────────────────────────────────

def train(args):
    from sst.butterfly_dataset import ButterflyOCCCLDataset

    device = f'cuda:{args.gpu}'
    torch.cuda.set_device(device)

    model = build_model(args.checkpoint, device=device)
    model.train()

    # Freeze image encoder
    for p in model.image_encoder.parameters():
        p.requires_grad = False

    # Apply LoRA if requested
    if args.lora_rank > 0:
        n_lora = apply_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
        print(f'[{args.name}] Applied LoRA to {n_lora} layers (rank={args.lora_rank})')

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'[{args.name}] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)')

    tracker = DifferentiableSAM2Tracker(model)

    dataset = ButterflyOCCCLDataset(
        species=['(malleti x plesseni) x malleti'], split='train', image_size=1024)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    total_steps = len(dataloader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.01)

    os.makedirs(args.output_dir, exist_ok=True)
    best_loss = float('inf')
    history = []

    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n = 0
        for batch_idx, (x0, y0, xu) in enumerate(dataloader):
            x0, y0, xu = x0.to(device), y0.to(device), xu.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss, _ = oc_ccl_step(tracker, x0, xu, y0,
                                       bce_w=args.bce_w, dice_w=args.dice_w,
                                       reset_memory=args.reset_memory)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            n += 1

        avg = epoch_loss / max(n, 1)
        history.append(avg)
        elapsed = time.time() - t0
        print(f'[{args.name}] Epoch {epoch+1}/{args.epochs}  loss={avg:.4f}  ({elapsed:.0f}s)')

        if avg < best_loss:
            best_loss = avg
            torch.save({'epoch': epoch+1, 'model': model.state_dict(), 'loss': avg},
                       os.path.join(args.output_dir, 'best_model.pt'))

    # Save results
    results = {
        'name': args.name, 'best_loss': best_loss, 'history': history,
        'lr': args.lr, 'bce_w': args.bce_w, 'dice_w': args.dice_w,
        'lora_rank': args.lora_rank, 'reset_memory': args.reset_memory,
        'trainable_params': trainable, 'total_time': time.time() - t0,
    }
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f'[{args.name}] Done. Best loss={best_loss:.4f} in {time.time()-t0:.0f}s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--checkpoint', type=str, default='checkpoints/sam2_hiera_large.pt')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--bce_w', type=float, default=1.0)
    parser.add_argument('--dice_w', type=float, default=1.0)
    parser.add_argument('--lora_rank', type=int, default=0, help='0=no LoRA')
    parser.add_argument('--lora_alpha', type=float, default=1.0)
    parser.add_argument('--reset_memory', action='store_true')
    parser.add_argument('--output_dir', type=str, default='outputs/ablation')
    args = parser.parse_args()
    train(args)
