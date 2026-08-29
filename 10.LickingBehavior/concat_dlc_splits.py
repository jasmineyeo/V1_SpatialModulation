"""
concat_dlc_splits.py
====================

Concatenate the per-split DeepLabCut `.h5` pose files of ONE face-camera
recording back into a single continuous pose table.

Each recording's video is cut into 10 000-frame chunks before DLC
(`..._split1...h5`, `..._split2...h5`, ...). This stitches them back
together in split order so the pose trace is 1:1 with the original video
frames.

Reading DLC `.h5` without PyTables
---------------------------------
In the `JSY_SpMod` env, importing `h5py` before PyTables breaks PyTables'
DLL load (`utilsextension` ImportError), and this whole pipeline needs
`h5py`. So this module reads the DLC file with `h5py` directly. A DLC
`.h5` is a PyTables `frame_table` whose payload lives at
`df_with_missing/table` as a plain HDF5 compound dataset
(`index` : int64, `values_block_0` : float32[K]) — no PyTables needed to
read it. Column identities come from the group's pickled `non_index_axes`
attribute.

Output (next to the split files)
--------------------------------
`{recording}_dlc_concat.h5`   — plain h5py layout (see `save_concat_h5`)
`{recording}_dlc_concat.json` — human-readable provenance + QC warnings

CLI
---
    python concat_dlc_splits.py F:\dlc\test
    python concat_dlc_splits.py F:\dlc\test --recording 260618_JSY083_B1
    python concat_dlc_splits.py F:\dlc\test --no-verify-videos

JSY / V1_SpatialModulation · 10.LickingBehavior
"""

from __future__ import annotations

import os
import re
import glob
import json
import pickle
import argparse
import datetime as _dt

import numpy as np
import h5py

try:
    import cv2  # only used for the optional video frame-count cross-check
    _HAVE_CV2 = True
except Exception:  # pragma: no cover
    _HAVE_CV2 = False


# --------------------------------------------------------------------------
# DLC .h5 reading (h5py only, no PyTables)
# --------------------------------------------------------------------------

_SPLIT_RE = re.compile(r"_split(\d+)DLC", re.IGNORECASE)


def _parse_dlc_columns(group: h5py.Group):
    """Return (scorer, bodyparts, coord_names) from a DLC frame_table group.

    coord_names are flattened like DLC's `open_dlc_h5`: 'led_x', 'led_y',
    'led_likelihood', 'tongue_x', ...
    """
    raw = group.attrs["non_index_axes"]
    if isinstance(raw, str):
        raw = raw.encode("latin1")
    parsed = pickle.loads(raw)
    # parsed == [[axis_int, [(scorer, bodypart, coord), ...]]]
    tuples = parsed[0][1]
    scorers = sorted({t[0] for t in tuples})
    scorer = scorers[0] if len(scorers) == 1 else "|".join(scorers)
    bodyparts = list(dict.fromkeys(t[1] for t in tuples))
    coord_names = [f"{t[1]}_{t[2]}" for t in tuples]
    return scorer, bodyparts, coord_names


def read_dlc_h5(path: str) -> dict:
    """Read one DLC `.h5` with h5py.

    Returns
    -------
    dict with keys:
        data        : (N, K) float32   — pose values in `coord_names` order
        coord_names : list[str]         — e.g. ['led_x','led_y','led_likelihood', ...]
        bodyparts   : list[str]         — e.g. ['led','tongue','chin']
        scorer      : str
        n           : int               — number of frames
    """
    with h5py.File(path, "r") as f:
        if "df_with_missing" not in f:
            raise ValueError(
                f"{os.path.basename(path)}: no 'df_with_missing' group "
                f"(root keys: {list(f.keys())}) — not a standard DLC .h5"
            )
        g = f["df_with_missing"]
        scorer, bodyparts, coord_names = _parse_dlc_columns(g)

        t = g["table"]
        names = t.dtype.names
        if "values_block_0" not in names:
            raise ValueError(
                f"{os.path.basename(path)}: unexpected table fields {names}"
            )
        idx = t["index"][:]
        data = np.asarray(t["values_block_0"][:], dtype=np.float32)

    if data.ndim != 2 or data.shape[1] != len(coord_names):
        raise ValueError(
            f"{os.path.basename(path)}: value block {data.shape} does not "
            f"match {len(coord_names)} columns {coord_names}"
        )
    if not np.array_equal(idx, np.arange(len(idx))):
        raise ValueError(
            f"{os.path.basename(path)}: row index is not a clean 0..N-1 range"
        )
    return {
        "data": data,
        "coord_names": coord_names,
        "bodyparts": bodyparts,
        "scorer": scorer,
        "n": int(data.shape[0]),
    }


# --------------------------------------------------------------------------
# Finding a recording's split files
# --------------------------------------------------------------------------

def find_recording_splits(folder: str, recording: str | None = None):
    """Locate the split DLC `.h5` files for one recording in `folder`.

    A "recording" is the filename stem up to `_split` (e.g.
    `260618_JSY083_B1`). If `recording` is None and the folder holds
    exactly one, it is used; otherwise you must name it.

    Returns
    -------
    recording : str
    splits    : list[(split_number:int, h5_path:str)]  sorted by number
    """
    all_h5 = [
        p for p in glob.glob(os.path.join(folder, "*.h5"))
        if _SPLIT_RE.search(os.path.basename(p))
    ]
    if not all_h5:
        raise FileNotFoundError(f"No '*_split<n>DLC*.h5' files in {folder}")

    by_rec: dict[str, list[tuple[int, str]]] = {}
    for p in all_h5:
        b = os.path.basename(p)
        m = _SPLIT_RE.search(b)
        stem = b[: m.start()]                     # everything before "_split"
        by_rec.setdefault(stem, []).append((int(m.group(1)), p))

    if recording is None:
        if len(by_rec) == 1:
            recording = next(iter(by_rec))
        else:
            raise ValueError(
                f"{len(by_rec)} recordings found in {folder}: "
                f"{sorted(by_rec)} — pass --recording to pick one"
            )
    if recording not in by_rec:
        raise ValueError(
            f"recording '{recording}' not found; have: {sorted(by_rec)}"
        )

    splits = sorted(by_rec[recording], key=lambda t: t[0])
    return recording, splits


def video_frame_count(path: str):
    """Frame count of a video via OpenCV, or None if unavailable."""
    if not (_HAVE_CV2 and os.path.isfile(path)):
        return None
    cap = cv2.VideoCapture(path)
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    return n if n > 0 else None


# --------------------------------------------------------------------------
# Concatenation
# --------------------------------------------------------------------------

def concat_recording_dlc(folder: str, recording: str | None = None,
                         verify_videos: bool = True):
    """Concatenate all split DLC `.h5` of one recording.

    Returns
    -------
    arrays : dict
        pose         : (Ntot, K) float32
        frame        : (Ntot,) int64    — global 0..Ntot-1
        split_number : (Ntot,) int64    — which split each frame came from
        within_split : (Ntot,) int64    — frame index inside that split
    meta : dict
        recording, coord_names, bodyparts, total_frames, splits[...], warnings[...]
    """
    recording, splits = find_recording_splits(folder, recording)
    warnings: list[str] = []

    nums = [n for n, _ in splits]
    expected = list(range(nums[0], nums[0] + len(nums)))
    if nums != expected:
        warnings.append(
            f"split numbering not contiguous: found {nums}, expected {expected}"
        )
    if nums and nums[0] != 1:
        warnings.append(f"first split is {nums[0]}, not 1")

    pose_parts, frame_parts, splitno_parts, within_parts = [], [], [], []
    split_meta = []
    ref_coords = ref_bodyparts = None
    running = 0

    for num, h5_path in splits:
        d = read_dlc_h5(h5_path)

        if ref_coords is None:
            ref_coords, ref_bodyparts = d["coord_names"], d["bodyparts"]
        elif d["coord_names"] != ref_coords:
            raise ValueError(
                f"split {num} columns {d['coord_names']} != split "
                f"{splits[0][0]} columns {ref_coords}"
            )

        vframes = video_frame_count(
            os.path.join(folder, f"{recording}_split{num}.mp4")
        ) if verify_videos else None
        vmatch = None if vframes is None else (vframes == d["n"])
        if vmatch is False:
            warnings.append(
                f"split {num}: DLC rows ({d['n']}) != video frames ({vframes})"
            )

        n = d["n"]
        pose_parts.append(d["data"])
        frame_parts.append(np.arange(running, running + n, dtype=np.int64))
        splitno_parts.append(np.full(n, num, dtype=np.int64))
        within_parts.append(np.arange(n, dtype=np.int64))
        running += n

        split_meta.append({
            "split_number": num,
            "file": os.path.basename(h5_path),
            "n_frames": n,
            "scorer": d["scorer"],
            "video_frames": vframes,
            "video_match": vmatch,
        })

    scorers = {s["scorer"] for s in split_meta}
    if len(scorers) > 1:
        warnings.append(
            "splits were scored by different DLC models: "
            + "; ".join(f"split{s['split_number']}={s['scorer']}"
                        for s in split_meta)
        )

    arrays = {
        "pose": np.concatenate(pose_parts, axis=0),
        "frame": np.concatenate(frame_parts),
        "split_number": np.concatenate(splitno_parts),
        "within_split": np.concatenate(within_parts),
    }
    meta = {
        "recording": recording,
        "coord_names": ref_coords,
        "bodyparts": ref_bodyparts,
        "total_frames": int(running),
        "n_splits": len(splits),
        "splits": split_meta,
        "warnings": warnings,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "source_folder": os.path.abspath(folder),
    }
    return arrays, meta


# --------------------------------------------------------------------------
# Save / load the concatenated file
# --------------------------------------------------------------------------

def save_concat_h5(path: str, arrays: dict, meta: dict) -> None:
    """Write the concatenated pose trace as a plain h5py file.

    Layout
    ------
    /pose          (Ntot, K) float32
    /frame         (Ntot,)   int64
    /split_number  (Ntot,)   int64
    /within_split  (Ntot,)   int64
    root attrs: coord_names, bodyparts, recording, total_frames,
                created, provenance (JSON string of `meta`)
    """
    sdt = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as f:
        f.create_dataset("pose", data=arrays["pose"], compression="gzip")
        f.create_dataset("frame", data=arrays["frame"])
        f.create_dataset("split_number", data=arrays["split_number"])
        f.create_dataset("within_split", data=arrays["within_split"])
        f.attrs["coord_names"] = np.array(meta["coord_names"], dtype=sdt)
        f.attrs["bodyparts"] = np.array(meta["bodyparts"], dtype=sdt)
        f.attrs["recording"] = meta["recording"]
        f.attrs["total_frames"] = meta["total_frames"]
        f.attrs["created"] = meta["created"]
        f.attrs["provenance"] = json.dumps(meta)


def load_concat_h5(path: str):
    """Inverse of `save_concat_h5` → (arrays dict, meta dict)."""
    with h5py.File(path, "r") as f:
        arrays = {
            "pose": f["pose"][:],
            "frame": f["frame"][:],
            "split_number": f["split_number"][:],
            "within_split": f["within_split"][:],
        }
        meta = json.loads(f.attrs["provenance"])
    return arrays, meta


def concat_as_dataframe(arrays: dict, meta: dict):
    """Return a pandas DataFrame: flattened pose columns (`led_x`, ...) plus
    `frame`, `split_number`, `within_split`. Mirrors DLC `open_dlc_h5`
    column naming so `dlc_utils.split_xyl` works on it directly.
    """
    import pandas as pd
    df = pd.DataFrame(arrays["pose"], columns=meta["coord_names"])
    df["frame"] = arrays["frame"]
    df["split_number"] = arrays["split_number"]
    df["within_split"] = arrays["within_split"]
    return df


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _run(folder: str, recording: str | None, verify_videos: bool) -> None:
    arrays, meta = concat_recording_dlc(folder, recording, verify_videos)

    out_h5 = os.path.join(folder, f"{meta['recording']}_dlc_concat.h5")
    out_json = os.path.join(folder, f"{meta['recording']}_dlc_concat.json")
    save_concat_h5(out_h5, arrays, meta)
    with open(out_json, "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"recording        : {meta['recording']}")
    print(f"splits           : {meta['n_splits']}  "
          f"({', '.join(str(s['split_number']) for s in meta['splits'])})")
    for s in meta["splits"]:
        vm = {True: "ok", False: "MISMATCH", None: "n/a"}[s["video_match"]]
        print(f"  split {s['split_number']:>2}: {s['n_frames']:>6d} frames  "
              f"video={vm}  {s['file']}")
    print(f"total frames     : {meta['total_frames']}")
    print(f"bodyparts        : {meta['bodyparts']}")
    if meta["warnings"]:
        print("WARNINGS:")
        for w in meta["warnings"]:
            print(f"  ! {w}")
    else:
        print("warnings         : none")
    print(f"wrote            : {out_h5}")
    print(f"wrote            : {out_json}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="folder holding the split DLC .h5 files")
    ap.add_argument("--recording", default=None,
                    help="recording stem (e.g. 260618_JSY083_B1); "
                         "required only if the folder has more than one")
    ap.add_argument("--no-verify-videos", dest="verify_videos",
                    action="store_false",
                    help="skip the h5-rows vs video-frames cross-check")
    args = ap.parse_args()
    _run(args.folder, args.recording, args.verify_videos)


if __name__ == "__main__":
    main()
