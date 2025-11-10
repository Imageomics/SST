import collections
import collections.abc
import dataclasses
import logging
import pathlib

import beartype
import duckdb
import numpy as np
import polars as pl
import torch
import transformers
import tyro
from PIL import Image

import lib

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("inference")


@beartype.beartype
@dataclasses.dataclass(frozen=True, slots=True)
class FrameMeta:
    pk: str
    """Primary key."""
    group: tuple[collections.abc.Hashable, ...]
    """Group for this frame."""
    img_fpath: pathlib.Path
    original_size: tuple[int, int]


@beartype.beartype
@dataclasses.dataclass(frozen=True, slots=True)
class RefMask:
    pk: str
    mask_fpath: pathlib.Path
    obj_ids: tuple[int, ...]  # stable global ids across the video


@beartype.beartype
def _pad(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


@beartype.beartype
def get_target_size(
    sizes_wh: list[tuple[int, int]], *, multiple: int = 16
) -> tuple[int, int]:
    # Uniform target size as ceil(max W/H) to `multiple` (SAM2 usually likes multiples of 16/32)
    w = max(w for (w, h) in sizes_wh)
    h = max(h for (w, h) in sizes_wh)
    return _pad(w, multiple), _pad(h, multiple)


@beartype.beartype
def get_frames(spec: lib.Spec) -> list[FrameMeta]:
    master_df = pl.read_csv(spec.master_csv, infer_schema_length=None)
    conn = duckdb.connect()
    conn.register("master_df", master_df)
    filtered_df = conn.execute(spec.filter_query).pl()

    if spec.primary_key not in filtered_df.columns:
        logger.error(
            "Primary key column '%s' not found in filtered dataset", spec.primary_key
        )
        return []

    dup = filtered_df.group_by(spec.primary_key).agg(pl.len()).filter(pl.col("len") > 1)
    if len(dup) > 0:
        logger.error(
            "Primary key '%s' is not unique! Found %d duplicates.",
            spec.primary_key,
            len(dup),
        )
        return []

    if spec.img_path not in filtered_df.columns:
        q = f"SELECT *, {spec.img_path} AS img_path FROM filtered_df"
        filtered_df = conn.execute(q).pl()
    else:
        filtered_df = filtered_df.with_columns(pl.col(spec.img_path).alias("img_path"))

    frames: list[FrameMeta] = []
    for row in filtered_df.iter_rows(named=True):
        img_fpath = pathlib.Path(row["img_path"])
        pk = str(row[spec.primary_key])
        group = tuple(row[col] for col in spec.group_by)
        try:
            with Image.open(img_fpath) as im:
                size = im.size
        except FileNotFoundError:
            logger.warning("File '%s' does not exist.", img_fpath)
            continue
        frames.append(FrameMeta(pk, group, img_fpath, size))
    return frames


@beartype.beartype
def get_ref_masks(spec: lib.Spec, pks: set[str]) -> list[RefMask]:
    ref_masks: list[RefMask] = []
    for pk in pks:
        mask_fpath = spec.ref_masks / f"{pk}.png"
        if not mask_fpath.exists():
            continue
        with Image.open(mask_fpath) as m:
            hw = np.array(m, dtype=np.uint8)
        obj_ids = tuple(int(x) for x in np.unique(hw) if x > 0)
        if not obj_ids:
            continue
        ref_masks.append(RefMask(pk, mask_fpath, obj_ids))
    return ref_masks


@beartype.beartype
class FrameDataset:
    def __init__(self, frames: list[FrameMeta], tgt_wh: tuple[int, int]):
        self.frames = frames
        self.tgt_wh = tgt_wh

    def __getitem__(self, i: int) -> tuple[Image.Image, FrameMeta]:
        with Image.open(self.frames[i].img_fpath) as img:
            img = img.convert("RGB").resize(self.tgt_wh, Image.Resampling.LANCZOS)

        return img, self.frames[i]

    def __len__(self) -> int:
        return len(self.frames)


@beartype.beartype
def get_dataloader(
    spec: lib.Spec, frames: list[FrameMeta], group: tuple, *, batch_size: int
) -> torch.utils.data.DataLoader:
    tgt_w, tgt_h = get_target_size([f.original_size for f in frames], multiple=16)

    pks = set(f.pk for f in frames)
    ref_masks = get_ref_masks(spec, pks)
    ref_pks = set(mask.pk for mask in ref_masks)

    ref_frame_lookup = {frame.pk: frame for frame in frames if frame.pk in ref_pks}
    pred_frames = [frame for frame in frames if frame.pk not in ref_pks]

    obj_ids = set()
    for ref_mask in ref_masks:
        obj_ids.update(ref_mask.obj_ids)
    obj_ids = sorted(obj_ids)

    ref_dataset, input_masks = [], []
    for mask in sorted(ref_masks, key=lambda mask: mask.obj_ids, reverse=True):
        ref_frame = ref_frame_lookup[mask.pk]
        with Image.open(ref_frame.img_fpath) as img:
            ref_dataset.append((img.resize((tgt_w, tgt_h)), ref_frame))

        with Image.open(mask.mask_fpath) as img:
            mask_hw = np.array(
                img.resize((tgt_w, tgt_h), Image.Resampling.NEAREST), dtype=np.uint8
            )

        mask_hw_list = [torch.zeros((tgt_h, tgt_w), dtype=torch.float32)] * len(obj_ids)
        for i, obj_id in enumerate(obj_ids):
            m = (mask_hw == obj_id).astype(np.float32)
            mask_hw_list[i] = torch.from_numpy(m)

        input_masks.append(mask_hw_list)

    def _collate_fn(batch):
        mask_i = np.linspace(0, batch_size - 1, len(input_masks), dtype=int).tolist()

        with_ref = []
        start = 0
        for i, end in enumerate(mask_i):
            with_ref.extend(batch[start:end])
            with_ref.append(ref_dataset[i])
            start = end

        assert len(with_ref) <= batch_size
        imgs, frames = zip(*with_ref)

        return {
            "imgs": imgs,
            "frames": frames,
            "mask_i": mask_i,
            "masks": input_masks,
            "obj_ids": list(obj_ids),
        }

    return torch.utils.data.DataLoader(
        FrameDataset(pred_frames, (tgt_w, tgt_h)),
        num_workers=8,
        collate_fn=_collate_fn,
        batch_size=batch_size - len(ref_masks),
    )


@beartype.beartype
def main(spec: pathlib.Path, /):
    spec = lib.Spec.model_validate_json(spec.read_text())
    spec.pred_masks.mkdir(parents=True, exist_ok=True)

    frames = get_frames(spec)
    if not frames:
        return

    frame_groups = collections.defaultdict(list)
    for frame in frames:
        frame_groups[frame.group].append(frame)

    # Check all groups have at least 2 reference masks before starting any inference
    groups_with_insufficient_refs = []
    for group, group_frames in frame_groups.items():
        ref_masks = get_ref_masks(spec, set(f.pk for f in group_frames))
        if len(ref_masks) < 2:
            groups_with_insufficient_refs.append((group, len(ref_masks)))

    if groups_with_insufficient_refs:
        for group, ref_count in groups_with_insufficient_refs:
            logger.error("Group %s has %d reference mask(s).", group, ref_count)
        return

    model = transformers.Sam2VideoModel.from_pretrained(spec.sam2).to(
        spec.device, dtype=torch.bfloat16
    )
    processor = transformers.Sam2VideoProcessor.from_pretrained(spec.sam2)

    for group, frames in frame_groups.items():
        logger.info("Processing group %s with %d frames", group, len(frames))
        dataloader = get_dataloader(spec, frames, group, batch_size=32)
        logger.info("Got dataloader with %d batches.", len(dataloader))

        for batch in lib.progress(dataloader, every=1):
            assert len(batch["mask_i"]) == len(batch["masks"])
            session = processor.init_video_session(
                video=batch["imgs"],
                inference_device=spec.device,
                dtype=torch.bfloat16,
            )
            orig_size = [session.video_height, session.video_width]

            for frame_idx, input_masks in zip(batch["mask_i"], batch["masks"]):
                processor.add_inputs_to_inference_session(
                    inference_session=session,
                    frame_idx=frame_idx,
                    obj_ids=batch["obj_ids"],
                    input_masks=input_masks,
                )

            for output in model.propagate_in_video_iterator(session, start_frame_idx=0):
                # The output of post_process_masks is a list with one element.
                masks_o1hw = processor.post_process_masks(
                    [output.pred_masks], original_sizes=[orig_size], binarize=True
                )[0]

                n_obj, singleton, h, w = masks_o1hw.shape
                assert singleton == 1, str(masks_o1hw.shape)

                # Create single channel mask with object IDs
                # Shape: (n_objects, height, width) -> (height, width)
                combined_mask = torch.zeros(
                    (h, w), dtype=torch.uint8, device=masks_o1hw.device
                )
                for obj_idx, mask in enumerate(masks_o1hw[:, 0, :, :]):
                    # Use obj_idx + 1 to reserve 0 for background
                    combined_mask[mask > 0] = obj_idx + 1

                # Convert to PIL image
                mask_img = Image.fromarray(combined_mask.cpu().numpy())

                # Resize mask back to original frame dimensions if needed
                frame = batch["frames"][output.frame_idx]
                mask_img = mask_img.resize(
                    frame.original_size, Image.Resampling.NEAREST
                )

                # Use primary key for filename
                mask_fname = f"{frame.pk}.png"
                mask_img.save(spec.pred_masks / mask_fname)


if __name__ == "__main__":
    tyro.cli(main)
