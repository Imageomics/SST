import collections.abc
import logging
import pathlib
import time

import beartype
import pydantic


class Spec(pydantic.BaseModel):
    """
    Everything required to do inference. The GUI generates a JSON representation of this object.
    Masks are stored as single-channel (L-mode) .png files with $PRIMARY_KEY.png as the name, where each pixel has the object ID as the value (at most 255 objects).
    """

    root: pathlib.Path
    """Directory with master.csv, and ref_masks/ and pred_masks/ directories."""
    filter_query: str
    """SQL query to filter master_csv."""
    group_by: tuple[str, ...]
    img_path: str
    primary_key: str
    sam2: str
    """SAM2 model name/path."""
    device: str
    """Device to run inference on (cuda/cpu)."""

    @property
    def master_csv(self) -> pathlib.Path:
        return self.root / "master.csv"

    @property
    def ref_masks(self) -> pathlib.Path:
        return self.root / "ref_masks"

    @property
    def pred_masks(self) -> pathlib.Path:
        return self.root / "pred_masks"


@beartype.beartype
class progress:
    def __init__(self, it, *, every: int = 10, desc: str = "progress", total: int = 0):
        """
        Wraps an iterable with a logger like tqdm but doesn't use any control codes to manipulate a progress bar, which doesn't work well when your output is redirected to a file. Instead, simple logging statements are used, but it includes quality-of-life features like iteration speed and predicted time to finish.

        Args:
            it: Iterable to wrap.
            every: How many iterations between logging progress.
            desc: What to name the logger.
            total: If non-zero, how long the iterable is.
        """
        self.it = it
        self.every = every
        self.logger = logging.getLogger(desc)
        self.total = total

    def __iter__(self):
        start = time.time()

        try:
            total = len(self)
        except TypeError:
            total = None

        for i, obj in enumerate(self.it):
            yield obj

            if (i + 1) % self.every == 0:
                now = time.time()
                duration_s = now - start
                per_min = (i + 1) / (duration_s / 60)

                if total is not None:
                    pred_min = (total - (i + 1)) / per_min
                    self.logger.info(
                        "%d/%d (%.1f%%) | %.1f it/m (expected finish in %.1fm)",
                        i + 1,
                        total,
                        (i + 1) / total * 100,
                        per_min,
                        pred_min,
                    )
                else:
                    self.logger.info("%d/? | %.1f it/m", i + 1, per_min)

    def __len__(self) -> int:
        if self.total > 0:
            return self.total

        # Will throw exception.
        return len(self.it)


@beartype.beartype
class batched_idx:
    """
    Iterate over (start, end) indices for total_size examples, where end - start is at most batch_size.

    Args:
        total_size: total number of examples
        batch_size: maximum distance between the generated indices.

    Returns:
        A generator of (int, int) tuples that can slice up a list or a tensor.
    """

    def __init__(self, total_size: int, batch_size: int):
        """
        Args:
            total_size: total number of examples
            batch_size: maximum distance between the generated indices
        """
        self.total_size = total_size
        self.batch_size = batch_size

    def __iter__(self) -> collections.abc.Iterator[tuple[int, int]]:
        """Yield (start, end) index pairs for batching."""
        for start in range(0, self.total_size, self.batch_size):
            stop = min(start + self.batch_size, self.total_size)
            yield start, stop

    def __len__(self) -> int:
        """Return the number of batches."""
        return (self.total_size + self.batch_size - 1) // self.batch_size
