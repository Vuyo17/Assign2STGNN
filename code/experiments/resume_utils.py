"""Shared helper for every `resume_*.py` script: find the checkpoint to
resume from.

`ModelCheckpoint(save_top_k=1, ...)` only guarantees at most one "best" file
per *single* trainer.fit() call -- it does not know about files left behind
by an *earlier*, separate run (e.g. a prior resume that was itself resumed,
or a process that was restarted). This project's own history hit exactly
that case: `results/agcrn/checkpoints/` ended up holding both
`best-epoch=9-val_mae=2.7526.ckpt` (the original run's best) and
`best-epoch=12-val_mae=2.7325.ckpt` (a later resume's new, better best) side
by side. A naive `sorted(glob(...))[-1]` picks the lexicographically-last
filename, which sorts "epoch=9" after "epoch=12" (since the character '9' >
'1') and so silently resumes from the *worse*, older checkpoint. Parsing the
val_mae out of every candidate filename and picking the minimum is
unambiguous and immune to that ordering trap, regardless of how many stale
files happen to be sitting in the directory.
"""
from __future__ import annotations

import glob
import re

_VAL_MAE_RE = re.compile(r"val_mae=([\d.]+)\.ckpt$")
_EPOCH_RE = re.compile(r"epoch=(\d+)")


def find_best_checkpoint(experiment_name: str) -> str:
    """Returns the path, among all `results/<experiment_name>/checkpoints/
    best-*.ckpt` files, whose filename-encoded val_mae is lowest (best)."""
    ckpts = glob.glob(f"results/{experiment_name}/checkpoints/best-*.ckpt")
    if not ckpts:
        raise FileNotFoundError(
            f"No existing checkpoint found to resume from in "
            f"results/{experiment_name}/checkpoints/"
        )
    scored = []
    for path in ckpts:
        m = _VAL_MAE_RE.search(path)
        if m:
            scored.append((float(m.group(1)), path))
    if not scored:
        # Fallback: filenames didn't match the expected pattern at all --
        # can't score them, so just take the most recently written file.
        import os
        return max(ckpts, key=os.path.getmtime)
    scored.sort(key=lambda t: t[0])
    return scored[0][1]


def epoch_of(checkpoint_path: str) -> int | None:
    m = _EPOCH_RE.search(checkpoint_path)
    return int(m.group(1)) if m else None
