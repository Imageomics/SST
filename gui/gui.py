"""
Homemade GUI for picking out reference masks for use with [Static Segmentation by Tracking](SST, https://arxiv.org/abs/2501.06749). After picking reference masks for one or more images, you can use these masks with the inference.py script in this repo to actually get masks for many images.
"""

import pathlib

import beartype
import duckdb
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import streamlit as st
import torch
import transformers
from jaxtyping import Bool, Float, jaxtyped
from PIL import Image

import lib

device = "cuda" if torch.cuda.is_available() else "cpu"


@beartype.beartype
class Frame:
    def __init__(self, path: pathlib.Path, pk: str):
        self.path = path
        self.pk = pk
        self._img = None

    def __hash__(self) -> int:
        return hash(str(self.path) + self.pk)

    @property
    def img(self):
        if self._img is None:
            with Image.open(self.path) as img:  # Load the image data and close the file
                self._img = img.copy()
        return self._img


@st.cache_resource
def load_pipeline(model: str):
    return transformers.pipeline("mask-generation", model=model, device=device)


@st.cache_data(hash_funcs={Frame: hash})
def get_all_masks(model, frame: Frame):
    pipeline = load_pipeline(model)
    return pipeline(frame.img, points_per_batch=64)


@st.cache_data(hash_funcs={Frame: hash})
@jaxtyped(typechecker=beartype.beartype)
def show_frame(
    frame: Frame,
    *,
    scale: float = 0.2,
    masks: Bool[np.ndarray, "n_masks height width"] | None = None,
    points: Float[np.ndarray, "n_points 2"] | None = None,
    boxes: Float[np.ndarray, "n_boxes 4"] | None = None,
    mask_labels: list[int] | None = None,  # n_masks
):
    """
    Display frame with masks, using consistent colors for each label ID.

    Args:
        mask_labels: List of integer labels for each mask. Each unique label will always get the same color.
    """
    img = frame.img
    orig_width, orig_height = img.size
    new_width = int(orig_width * scale)
    new_height = int(orig_height * scale)
    img_scaled = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    fig, ax = plt.subplots(figsize=(10, 10 * new_height / new_width))
    ax.imshow(img_scaled)
    ax.axis("off")

    if masks is not None:
        for i, mask in enumerate(masks):
            mask_scaled = Image.fromarray(mask.astype(np.uint8) * 255)
            mask_scaled = mask_scaled.resize(
                (new_width, new_height), Image.Resampling.NEAREST
            )
            mask_scaled = np.array(mask_scaled) > 127

            # Use mask label for consistent color selection, fallback to index if no labels
            if mask_labels is not None and i < len(mask_labels):
                color_index = mask_labels[i] % 10
            else:
                color_index = i % 10

            color = plt.cm.tab10(color_index)
            overlay = np.zeros((new_height, new_width, 4))
            overlay[:, :, :3] = color[:3]
            overlay[:, :, 3] = mask_scaled * 0.7
            ax.imshow(overlay)

            # Add mask labels at the center of each mask
            if mask_labels is not None and i < len(mask_labels):
                # Find the center of mass of the mask
                mask_indices = np.where(mask_scaled)
                if len(mask_indices[0]) > 0:
                    center_y = np.mean(mask_indices[0])
                    center_x = np.mean(mask_indices[1])
                    ax.text(
                        center_x,
                        center_y,
                        str(mask_labels[i]),
                        color=color,
                        fontsize=12,
                        fontweight="bold",
                        ha="center",
                        va="center",
                        bbox=dict(
                            boxstyle="round,pad=0.3", facecolor="black", alpha=0.7
                        ),
                    )

    if points is not None:
        points_scaled = points * scale
        ax.scatter(
            points_scaled[:, 0],
            points_scaled[:, 1],
            c="lime",
            s=100,
            marker="x",
            linewidths=2,
        )

    if boxes is not None:
        boxes_scaled = boxes * scale
        for box in boxes_scaled:
            width = box[2] - box[0]
            height = box[3] - box[1]
            rect = patches.Rectangle(
                (box[0], box[1]),
                width,
                height,
                linewidth=2,
                edgecolor="cyan",
                facecolor="none",
            )
            ax.add_patch(rect)

    plt.tight_layout()
    return fig


@jaxtyped(typechecker=beartype.beartype)
def save_reference_mask(
    frame: Frame,
    masks: Bool[np.ndarray, "n_masks height width"],
    mask_labels: list[int],
    root: pathlib.Path,
) -> pathlib.Path | None:
    """Save selected masks as a single-channel PNG with object IDs.

    Args:
        frame: Frame object containing the primary key
        masks: Binary masks array
        mask_labels: Object ID for each mask
        save_dir: Directory to save the mask file
    """
    if len(masks) == 0:
        return

    # Get image dimensions from first mask
    height, width = masks[0].shape

    # Create single-channel mask with object IDs
    combined_mask = np.zeros((height, width), dtype=np.uint8)

    for mask, label in zip(masks, mask_labels):
        # Use the label value for pixels in this mask
        combined_mask[mask > 0] = label

    # Save as PNG
    (root / "ref_masks").mkdir(parents=True, exist_ok=True)
    mask_img = Image.fromarray(combined_mask)
    save_fpath = root / "ref_masks" / f"{frame.pk}.png"
    mask_img.save(save_fpath)

    return save_fpath


def main():
    st.set_page_config(page_title="SST GUI", page_icon="🔬")

    master_csv = st.file_uploader("Choose a file")
    if master_csv is None:
        return
    master_df = pl.read_csv(master_csv.getvalue(), infer_schema_length=None)
    st.write(master_df.head())

    conn = duckdb.connect()
    # Register with DuckDB for SQL queries
    conn.register("master_df", master_df)

    filter_query = st.text_area("SQL query to filter rows.", "SELECT * FROM master_df")
    group_by = st.text_input(
        "Column(s) to group images by (comma-separated):",
        placeholder="experiment_id, subject_id",
    )
    img_path = st.text_input(
        "Column name or SQL expression for image paths:",
        placeholder="img_path or CONCAT(base_dir, '/', filename)",
    )
    primary_key = st.text_input(
        "Column name for primary key (must be unique per image):",
        placeholder="frame_id or image_id",
    )

    if not (filter_query and group_by and img_path and primary_key):
        return

    filtered_df = conn.execute(filter_query).pl()

    # Validate primary key uniqueness
    if primary_key not in filtered_df.columns:
        st.error(f"Primary key column '{primary_key}' not found in filtered dataset")
        return

    # Check for duplicates
    duplicate_count = (
        filtered_df.group_by(primary_key).agg(pl.len()).filter(pl.col("len") > 1)
    )
    if len(duplicate_count) > 0:
        st.error(
            f"Primary key '{primary_key}' is not unique! Found {len(duplicate_count)} duplicate values."
        )
        st.write("Duplicate primary keys:")
        st.dataframe(duplicate_count.head(10))
        return

    if img_path not in filtered_df.columns:
        path_query = f"SELECT *, {img_path} as img_path FROM filtered_df"
        filtered_df = conn.execute(path_query).pl()
    else:
        filtered_df = filtered_df.with_columns(pl.col(img_path).alias("img_path"))

    group_cols = [col.strip() for col in group_by.split(",")]
    # Group by columns and aggregate image paths and primary keys
    grouped = (
        filtered_df.group_by(group_cols)
        .agg([pl.col("img_path"), pl.col(primary_key)])
        .sort(by=group_cols)
    )
    # Convert to dictionary format
    img_groups = {}
    pk_groups = {}
    for row in grouped.iter_rows(named=True):
        key = tuple(row[col] for col in group_cols)
        img_groups[key] = row["img_path"]
        pk_groups[key] = row[primary_key]

    st.write(f"{len(img_groups)} image groups")

    if not img_groups:
        st.write("No image groups available. Configure data processing first.")
        return

    group_keys = list(img_groups.keys())
    # TODO: define a format_func that includes keys + values.
    group = st.selectbox("Select image group:", group_keys)

    frames = [
        Frame(pathlib.Path(path), pk=str(pk))
        for path, pk in zip(img_groups[group], pk_groups[group])
    ]

    # Configuration section
    n_ref_frames = st.number_input(
        "Number of reference frames:",
        min_value=1,
        max_value=len(frames),
        value=min(5, len(frames)),
    )

    root = st.text_input("Output directory:", placeholder="/path/to/outputs")

    if not root:
        st.warning("Please specify an output directory before proceeding.")
        return

    root = pathlib.Path(root)

    # Ask for SAM2 model
    sam2_model = st.selectbox(
        "Select SAM2 model:",
        [
            "facebook/sam2.1-hiera-tiny",
            "facebook/sam2.1-hiera-small",
            "facebook/sam2.1-hiera-base-plus",
            "facebook/sam2.1-hiera-large",
        ],
        index=0,
    )

    # Save a lib.Spec object as a spec.json file
    spec = lib.Spec(
        root=root,
        filter_query=filter_query,
        group_by=tuple(group_cols),
        img_path=img_path,
        primary_key=primary_key,
        sam2=sam2_model,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    spec_path = pathlib.Path(__file__).parent / "spec.json"
    spec_path.write_text(spec.model_dump_json(indent=2))
    st.info(f"💾 Spec saved to {spec_path}")

    # Save the master CSV to the root directory
    master_csv_path = root / "master.csv"
    master_csv_path.write_bytes(master_csv.getvalue())
    st.info(f"💾 Master CSV saved to {master_csv_path}")

    # Sample frames without replacement
    rng = np.random.default_rng(seed=0)
    frame_indices = rng.choice(len(frames), size=n_ref_frames, replace=False)
    frame_indices = sorted(frame_indices)

    selected_frames = [frames[i] for i in frame_indices]

    # Generate masks for all selected frames
    pipeline_outputs = []
    for frame in selected_frames:
        output = get_all_masks(sam2_model, frame)
        pipeline_outputs.append(output)

    st.success(f"✅ Generated masks for {n_ref_frames} frames")

    # Process each frame
    for frame_idx, (frame, pipeline_output) in enumerate(
        zip(selected_frames, pipeline_outputs)
    ):
        masks = torch.stack(pipeline_output["masks"]).numpy()

        with st.expander(
            f"Frame {frame_indices[frame_idx] + 1}/{len(frames)} - {frame.pk}",
            expanded=True,
        ):
            # Show frame info
            st.write(f"**Primary key:** {frame.pk}")
            st.write(f"**Number of masks detected:** {len(masks)}")

            mask_i = st.pills(
                "Select masks to include:",
                list(range(1, len(masks) + 1)),
                selection_mode="multi",
                default=list(range(1, len(masks) + 1)),
                key=f"mask_pills_{frame.pk}",
            )

            # Adjust mask_i to 0-indexed
            mask_indices = [i - 1 for i in mask_i]

            # Object ID assignment
            st.write("**Assign mask IDs:**")
            cols = st.columns(min(len(mask_i), 5))
            mask_labels = []

            for idx, (mask_idx, display_idx) in enumerate(zip(mask_indices, mask_i)):
                with cols[idx % 5]:
                    label = st.number_input(
                        f"Mask {display_idx}",
                        value=mask_idx + 1,
                        min_value=1,
                        max_value=255,
                        key=f"label_{frame.pk}_{mask_idx}",
                    )
                    mask_labels.append(label)

            # Display the frame with selected masks
            fig = show_frame(
                frame, scale=0.05, masks=masks[mask_indices], mask_labels=mask_labels
            )
            st.pyplot(fig)

            # Save button for this frame
            if st.button("💾 Save Reference Mask", key=f"save_{frame.pk}"):
                save_path = save_reference_mask(
                    frame, masks[mask_indices], mask_labels, root
                )
                if save_path:
                    st.success(f"✅ Saved mask to {save_path}")


main()
