"""
lick_sync.py
============

Camera-frame <-> VR-time alignment from the DLC `led` bodypart.

The `led` bodypart flashes on two VR events, separable by duration:

    reward  `r`  : 1-3 frame flash   (the solenoid opening, ~0.25 s AFTER
                   the logged `r` -> ~= actual water delivery)
    new trial `n`: ~16 frame (~0.5 s) flash   (the teleport, ~0.25 s
                   BEFORE the logged `n`)

`sync_led_to_vr` fits a shared slope (fps drift) with a separate intercept
per flash type, so `b_generic = (b_n + b_r)/2` is the true frame<->VR-elapsed
mapping and `b_n` / `b_r` recover the physical-event frames from the log
times. Residual on the first real session: **~15 ms** (< half a frame).

Functions
---------
extract_led_epochs(pose, coord_names, ...)   DLC `led` p>cutoff -> flash epochs
sync_led_to_vr(led_df, n_times, r_times, ...) -> model dict
frame_to_vr(frame, model, kind)              camera frame -> VR elapsed s
vr_to_frame(t, model, kind)                  VR elapsed s -> camera frame
load_sync_model(lickproc_h5)                 read a saved model back

`kind` in {"n", "r", "generic"} picks the intercept:
    "n"       -> teleport frame / VR `n` time
    "r"       -> valve-open frame / VR `r` time
    "generic" -> any other frame (position lookup, lick times, ...)

JSY / V1_SpatialModulation - 10.LickingBehavior
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FPS = 30.0

# LED defaults for this rig (DLC `led` real-LED cluster; a 2nd cluster on the
# VR monitor at ~(95,116) is noise and is rejected by the box).
LED_XY = (369, 538)
LED_BOX = 25
LED_PCUTOFF = 0.6
MERGE_GAP = 3
BRIEF_MAX = 7      # epoch <= this -> reward flash
LONG_MIN = 9       # epoch >= this -> new-trial flash


# --------------------------------------------------------------------------
# LED epochs
# --------------------------------------------------------------------------

def extract_led_epochs(pose, coord_names, led_xy=LED_XY, box=LED_BOX,
                       pcutoff=LED_PCUTOFF, merge_gap=MERGE_GAP,
                       brief_max=BRIEF_MAX, long_min=LONG_MIN):
    """DLC `led` bodypart (p > cutoff, inside the LED box) -> flash epochs.

    Returns
    -------
    df : DataFrame  [onset_frame, end_frame, dur_frames, kind]  kind in {r,n,?}
    raw : dict      lx, ly, ll (full traces) + on (bool mask)
    """
    lx = pose[:, coord_names.index("led_x")]
    ly = pose[:, coord_names.index("led_y")]
    ll = pose[:, coord_names.index("led_likelihood")]

    on = (ll > pcutoff) & (np.abs(lx - led_xy[0]) < box) & (np.abs(ly - led_xy[1]) < box)
    d = np.diff(np.r_[0, on.astype(int), 0])
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]

    ep = []
    for s, e in zip(starts, ends):
        if ep and s - ep[-1][1] <= merge_gap:
            ep[-1][1] = e
        else:
            ep.append([s, e])
    ep = np.array(ep, dtype=int).reshape(-1, 2)
    dur = ep[:, 1] - ep[:, 0]
    kind = np.where(dur <= brief_max, "r", np.where(dur >= long_min, "n", "?"))

    df = pd.DataFrame({"onset_frame": ep[:, 0], "end_frame": ep[:, 1],
                       "dur_frames": dur, "kind": kind})
    return df, dict(lx=lx, ly=ly, ll=ll, on=on)


# --------------------------------------------------------------------------
# Fit
# --------------------------------------------------------------------------

def sync_led_to_vr(led_df, n_times, r_times, fps=FPS,
                   coarse=(-20.0, 2.0), tol_n=0.35, tol_r=0.55):
    """Fit frame -> VR-time from the LED flashes.

    Shared slope, per-flash-type intercept. Returns a model dict with
    a, b_n, b_r, b_generic, fps_eff, camera_lead_s, residuals, anchors, o0.
    """
    n_on = led_df.loc[led_df.kind == "n", "onset_frame"].to_numpy() / fps
    r_on = led_df.loc[led_df.kind == "r", "onset_frame"].to_numpy() / fps

    grid = np.arange(coarse[0], coarse[1], 0.005)
    score = [np.sum(np.min(np.abs((n_on + o)[:, None] - n_times[None, :]), axis=1) < 0.20)
             for o in grid]
    o0 = grid[int(np.argmax(score))]

    def match(cam, vt, tol):
        out = []
        for c in cam:
            j = int(np.argmin(np.abs(vt - (c + o0))))
            if abs(vt[j] - (c + o0)) < tol:
                out.append((c, vt[j]))
        return np.array(out) if out else np.empty((0, 2))

    pn, pr = match(n_on, n_times, tol_n), match(r_on, r_times, tol_r)

    X = np.vstack([np.c_[pn[:, 0], np.ones(len(pn)), np.zeros(len(pn))],
                   np.c_[pr[:, 0], np.zeros(len(pr)), np.ones(len(pr))]])
    Y = np.concatenate([pn[:, 1], pr[:, 1]])
    (a, b_n, b_r), *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ [a, b_n, b_r]

    return dict(a=float(a), b_n=float(b_n), b_r=float(b_r),
                b_generic=float((b_n + b_r) / 2),
                fps_eff=float(fps * a), camera_lead_s=float(-((b_n + b_r) / 2) / a),
                resid=resid, resid_kinds=np.array(["n"] * len(pn) + ["r"] * len(pr)),
                resid_sd_ms=float(resid.std() * 1000),
                resid_max_ms=float(np.abs(resid).max() * 1000),
                anchors_n=pn, anchors_r=pr, o0=float(o0))


# --------------------------------------------------------------------------
# Clock conversion
# --------------------------------------------------------------------------

def _intercept(model, kind):
    return {"n": model["b_n"], "r": model["b_r"], "generic": model["b_generic"]}[kind]


def frame_to_vr(frame, model, kind="generic", fps=FPS):
    """Camera frame(s) -> VR elapsed time (s)."""
    return model["a"] * (np.asarray(frame, float) / fps) + _intercept(model, kind)


def vr_to_frame(t, model, kind="generic", fps=FPS):
    """VR elapsed time(s) (s) -> camera frame."""
    return (np.asarray(t, float) - _intercept(model, kind)) / model["a"] * fps


def load_sync_model(lickproc_h5):
    """Read the `sync` group of a `*_lickproc.h5` back into a model dict."""
    import h5py
    with h5py.File(lickproc_h5, "r") as f:
        m = {k: float(v) for k, v in f["sync"].attrs.items()}
        for k in ("anchors_n", "anchors_r"):
            if k in f["sync"]:
                m[k] = f["sync"][k][:]
    return m
