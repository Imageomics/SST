"""
OC-CCL: Open-Close Cycle Consistency Loss for SAM2.

Trains SAM2 by running a cycle:
  reference→query (predict mask on query using reference mask)
  query→reference (predict closing mask on reference using predicted query mask)
Supervised against the original GT mask with BCE+Dice loss.

The DifferentiableSAM2Tracker bypasses SAM2's @torch.inference_mode() decorators
by calling internal methods (forward_image, track_step, _prepare_backbone_features)
directly, enabling gradient flow through the tracking pipeline.
"""

import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))


def build_model(checkpoint, device="cuda"):
    """
    Build SAM2 model using hydra.initialize_config_dir to avoid pip sam2 conflicts.

    Uses the local src/sst/sam2_configs/ directory instead of the pip-installed
    sam2 package's config, preventing Hydra double-initialization errors.
    """
    from hydra.core.global_hydra import GlobalHydra
    from hydra import initialize_config_dir, compose
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    config_dir = os.path.join(PROJECT_ROOT, "src", "sst", "sam2_configs")

    # Clear any existing Hydra state to avoid conflicts with pip sam2
    GlobalHydra.instance().clear()

    with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
        cfg = compose(config_name="sam2_hiera_l")
        OmegaConf.resolve(cfg)
        model = instantiate(cfg.model, _recursive_=True)

    # Load checkpoint
    if checkpoint and os.path.exists(checkpoint):
        sd = torch.load(checkpoint, map_location="cpu")["model"]
        missing, unexpected = model.load_state_dict(sd)
        if missing:
            print(f"Warning: missing keys: {missing[:5]}...")
        if unexpected:
            print(f"Warning: unexpected keys: {unexpected[:5]}...")
        print(f"Loaded checkpoint from {checkpoint}")

    model = model.to(device)
    return model


class DifferentiableSAM2Tracker:
    """
    Wraps a SAM2Base model for differentiable mask-conditioned tracking.

    SAM2VideoPredictor uses @torch.inference_mode() which blocks gradients.
    This class calls the underlying methods directly:
      - forward_image(): encode an image through the backbone
      - _prepare_backbone_features(): flatten visual features
      - track_step(): run memory-conditioned segmentation

    This enables gradient flow for training while reusing SAM2's architecture.
    """

    def __init__(self, model):
        self.model = model

    def encode_image(self, img):
        """
        Encode a single image through SAM2's image encoder.

        Args:
            img: (B, 3, H, W) tensor, float32 in [0, 1]

        Returns:
            backbone_out: dict with 'backbone_fpn' and 'vision_pos_enc'
        """
        # SAM2 expects images in a specific format; the image encoder handles normalization
        backbone_out = self.model.forward_image(img)
        return backbone_out

    def prepare_features(self, backbone_out):
        """Flatten backbone features for track_step."""
        return self.model._prepare_backbone_features(backbone_out)

    def track_with_mask(self, backbone_out, mask, is_init_frame=True, output_dict=None):
        """
        Run SAM2's track_step with a mask input.

        Args:
            backbone_out: from encode_image()
            mask: (B, 1, H, W) mask tensor
            is_init_frame: whether this is a conditioning frame
            output_dict: memory dict for non-init frames

        Returns:
            current_out dict with 'pred_masks_high_res' (B, 1, 1024, 1024)
        """
        _, vision_feats, vision_pos_embeds, feat_sizes = self.prepare_features(backbone_out)

        if output_dict is None:
            output_dict = {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            }

        if is_init_frame:
            # For init frame, pass mask as input (will use _use_mask_as_output path)
            current_out = self.model.track_step(
                frame_idx=0,
                is_init_cond_frame=True,
                current_vision_feats=vision_feats,
                current_vision_pos_embeds=vision_pos_embeds,
                feat_sizes=feat_sizes,
                point_inputs=None,
                mask_inputs=mask,
                output_dict=output_dict,
                num_frames=2,
                run_mem_encoder=True,
            )
            # Store as conditioning frame
            output_dict["cond_frame_outputs"][0] = current_out
        else:
            # For query frame, use memory from init frame
            current_out = self.model.track_step(
                frame_idx=1,
                is_init_cond_frame=False,
                current_vision_feats=vision_feats,
                current_vision_pos_embeds=vision_pos_embeds,
                feat_sizes=feat_sizes,
                point_inputs=None,
                mask_inputs=None,
                output_dict=output_dict,
                num_frames=2,
                run_mem_encoder=True,
            )

        return current_out, output_dict


def bce_loss(pred_logits, target):
    """Binary cross-entropy loss on mask logits."""
    return F.binary_cross_entropy_with_logits(pred_logits, target)


def dice_loss(pred_logits, target, smooth=1.0):
    """Dice loss on mask logits."""
    pred = torch.sigmoid(pred_logits)
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return (1.0 - dice).mean()


def oc_ccl_step(tracker, x0, xu, y0):
    """
    Full OC-CCL forward pass (one cycle).

    1. Encode reference image x0, condition on GT mask y0 → memory
    2. Encode query image xu, track using memory → predicted query mask
    3. Condition on predicted query mask → new memory
    4. Track back to reference → predicted closing mask
    5. Supervise closing mask against y0

    Args:
        tracker: DifferentiableSAM2Tracker
        x0: (B, 3, H, W) reference image
        xu: (B, 3, H, W) query image
        y0: (B, 1, H, W) ground-truth mask for x0

    Returns:
        loss: scalar tensor
        pred_closing_mask: (B, 1, H, H) closing mask logits
    """
    # Step 1: Encode reference and condition with GT mask
    backbone_ref = tracker.encode_image(x0)
    ref_out, output_dict = tracker.track_with_mask(
        backbone_ref, y0, is_init_frame=True
    )

    # Step 2: Encode query and predict mask using reference memory
    backbone_query = tracker.encode_image(xu)
    query_out, output_dict_fwd = tracker.track_with_mask(
        backbone_query, mask=None, is_init_frame=False, output_dict=output_dict
    )
    pred_query_mask = query_out["pred_masks_high_res"]  # (B, 1, 1024, 1024)

    # Step 3: Build reverse cycle — condition on predicted query mask
    reverse_output_dict = {
        "cond_frame_outputs": {},
        "non_cond_frame_outputs": {},
    }
    # Use the predicted query mask as conditioning for the reverse pass
    # Convert logits to soft mask for conditioning
    soft_query_mask = (pred_query_mask > 0).float()
    query_cond_out, reverse_output_dict = tracker.track_with_mask(
        backbone_query, soft_query_mask, is_init_frame=True,
        output_dict=reverse_output_dict
    )

    # Step 4: Track back to reference using query memory
    ref_closing_out, _ = tracker.track_with_mask(
        backbone_ref, mask=None, is_init_frame=False,
        output_dict=reverse_output_dict
    )
    pred_closing_mask = ref_closing_out["pred_masks_high_res"]  # (B, 1, 1024, 1024)

    # Step 5: Compute loss — supervise closing mask against GT
    # Resize y0 to match prediction resolution if needed
    if y0.shape[-2:] != pred_closing_mask.shape[-2:]:
        y0_resized = F.interpolate(
            y0, size=pred_closing_mask.shape[-2:],
            mode="bilinear", align_corners=False
        )
    else:
        y0_resized = y0

    loss = bce_loss(pred_closing_mask, y0_resized) + dice_loss(pred_closing_mask, y0_resized)

    return loss, pred_closing_mask


def freeze_image_encoder(model):
    """Freeze the image encoder parameters."""
    for param in model.image_encoder.parameters():
        param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} parameters "
          f"({100*trainable/total:.1f}%)")


def train(args):
    """Main training loop."""
    from sst.butterfly_dataset import ButterflyOCCCLDataset

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    # Build model
    print("Building model...")
    model = build_model(args.checkpoint, device=device)
    model.train()

    # Freeze image encoder
    freeze_image_encoder(model)

    tracker = DifferentiableSAM2Tracker(model)

    # Dataset
    species = args.species if args.species else None
    dataset = ButterflyOCCCLDataset(species=species, split="train", image_size=1024)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )

    # Optimizer and scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    total_steps = len(dataloader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr * 0.01
    )

    # Training loop
    os.makedirs(args.output_dir, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (x0, y0, xu) in enumerate(dataloader):
            x0 = x0.to(device)
            y0 = y0.to(device)
            xu = xu.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                loss, _ = oc_ccl_step(tracker, x0, xu, y0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1

            if batch_idx % args.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch+1}/{args.epochs} "
                      f"[{batch_idx}/{len(dataloader)}] "
                      f"loss={loss.item():.4f} lr={lr:.2e}")

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1}/{args.epochs} — avg_loss={avg_loss:.4f}")

        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save({
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": avg_loss,
            }, ckpt_path)
            print(f"  Saved best model to {ckpt_path}")

        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch+1}.pt")
            torch.save({
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": avg_loss,
            }, ckpt_path)

    print("Training complete.")


def main():
    parser = argparse.ArgumentParser(description="OC-CCL: Open-Close Cycle Consistency Loss for SAM2")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/sam2_hiera_large.pt",
                        help="Path to SAM2 checkpoint")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="outputs/oc_ccl")
    parser.add_argument("--species", type=str, nargs="+", default=None,
                        help="Species to train on (default: all 5 major species)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
