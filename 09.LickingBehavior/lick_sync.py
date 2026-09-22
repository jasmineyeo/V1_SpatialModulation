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

JSY / V1_SpatialModulation - 09.LickingBehavior
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FPS = 30.0

# LED defaults for this rig (DLC `led` real-LED cluster; a 2nd cluster on the
# VR monitor in the top-left is noise and is rejected).  The physical LED sits
# at a slightly different pixel every recording (the face camera was re-seated
# between sessions), so `LED_XY` is only a fallback -- `extract_led_epochs`
# auto-locates the cluster per session (`led_xy=None`, the default).
LED_XY = (369, 538)            # fallback QC label only -- not used to gate detection
LED_PCUTOFF = 0.6
LED_NOISE_BOX = ((0, 200), (0, 260))   # top-left monitor-noise region ((x0,x1),(y0,y1))
LED_MAX_SPREAD_PX = 30          # a candidate flash is kept only if its own detections
                                 # (during just that blink) don't wander past this
MERGE_GAP = 3
BRIEF_MAX = 7      # epoch <= this -> reward flash
LONG_MIN = 9       # epoch >= this -> new-trial flash


# --------------------------------------------------------------------------
# LED epochs
# --------------------------------------------------------------------------

def extract_led_epochs(pose, coord_names, pcutoff=LED_PCUTOFF, merge_gap=MERGE_GAP,
                       brief_max=BRIEF_MAX, long_min=LONG_MIN,
                       noise_box=LED_NOISE_BOX, max_spread_px=LED_MAX_SPREAD_PX):
    """DLC `led` bodypart -> flash epochs, located by BLINK PATTERN, not by a
    pre-located pixel.

    No fixed LED location is assumed anywhere. Every contiguous run of
    `led_likelihood > pcutoff` (outside the known VR-monitor noise region) is
    a flash CANDIDATE; it's kept only if the detections within that specific
    run stay spatially tight (<= max_spread_px in x AND y) -- a real LED
    blink doesn't wander, so this rejects spurious runs without knowing where
    the LED is, and it adapts automatically if the LED (or camera) is
    physically nudged at any point in the session -- mid-recording, not just
    at a DLC-split boundary -- since each blink is judged on its own.

    Returns
    -------
    df : DataFrame  [onset_frame, end_frame, dur_frames, kind]  kind in {r,n,?}
    raw : dict      lx, ly, ll (full traces) + on (bool mask, accepted flash
                    frames only) + led_xy (median location of accepted
                    flashes, for QC/reporting) + n_hi (accepted flash frames)
    """
    lx = pose[:, coord_names.index("led_x")]
    ly = pose[:, coord_names.index("led_y")]
    ll = pose[:, coord_names.index("led_likelihood")]

    (nx0, nx1), (ny0, ny1) = noise_box
    not_noise = ~((lx >= nx0) & (lx < nx1) & (ly >= ny0) & (ly < ny1))
    cand = (ll > pcutoff) & not_noise

    d = np.diff(np.r_[0, cand.astype(int), 0])
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    runs = []
    for s, e in zip(starts, ends):
        if runs and s - runs[-1][1] <= merge_gap:
            runs[-1][1] = e
        else:
            runs.append([s, e])

    ep = []
    on = np.zeros(len(ll), dtype=bool)
    for s, e in runs:
        m = cand[s:e]
        if not m.any():
            continue
        xs, ys = lx[s:e][m], ly[s:e][m]
        if (xs.max() - xs.min()) <= max_spread_px and (ys.max() - ys.min()) <= max_spread_px:
            ep.append([s, e])
            idx = np.arange(s, e)[m]
            on[idx] = True     # only the genuinely-on frames within this epoch

    ep = np.array(ep, dtype=int).reshape(-1, 2)
    dur = ep[:, 1] - ep[:, 0]
    kind = np.where(dur <= brief_max, "r", np.where(dur >= long_min, "n", "?"))

    df = pd.DataFrame({"onset_frame": ep[:, 0], "end_frame": ep[:, 1],
                       "dur_frames": dur, "kind": kind})
    n_hi = int(on.sum())
    led_xy = (float(np.median(lx[on])), float(np.median(ly[on]))) if n_hi else LED_XY
    return df, dict(lx=lx, ly=ly, ll=ll, on=on, led_xy=led_xy, n_hi=n_hi)


# --------------------------------------------------------------------------
# Fit
# --------------------------------------------------------------------------

def _coarse_offsets(cam, vt, lo=-600.0, hi=60.0, keep=3):
    """Best few camera->VR offsets (s). The face camera often started minutes
    before the VR program, so the search has to be wide; the trial-duration
    jitter breaks the near-periodicity so the true offset wins clearly."""
    if len(cam) == 0:
        return [0.0]
    g1 = np.arange(lo, hi, 0.1)
    s1 = np.array([np.sum(np.min(np.abs((cam + o)[:, None] - vt[None, :]), axis=1) < 0.25)
                   for o in g1])
    out = []
    for gi in np.argsort(s1)[::-1][:keep]:
        o1 = g1[gi]
        g2 = np.arange(o1 - 0.15, o1 + 0.15, 0.004)
        s2 = [np.sum(np.min(np.abs((cam + o)[:, None] - vt[None, :]), axis=1) < 0.12)
              for o in g2]
        out.append(float(g2[int(np.argmax(s2))]))
    return out


def _trim_knots(P, win=11, thresh=0.30):
    """Drop piecewise-anchor points that a leave-one-out local line can't
    predict (a lone mis-detected flash), while keeping smoothly-drifted
    stretches (each point still predicts its neighbours)."""
    n = len(P)
    if n < win + 2:
        return np.ones(n, bool)
    x, y = P[:, 0], P[:, 1]
    keep = np.ones(n, bool)
    h = win // 2
    for i in range(n):
        lo, hi = max(0, i - h), min(n, i + h + 1)
        idx = [j for j in range(lo, hi) if j != i]
        c = np.polyfit(x[idx], y[idx], 1)
        if abs(y[i] - np.polyval(c, x[i])) > thresh:
            keep[i] = False
    return keep


def _robust_line(xy, floor_s=0.04, k=3.5, n_iter=6):
    """Iteratively hard-trimmed 1-D line fit  y = a*x + b.

    Trims points with |residual| > max(40 ms, k * MAD) and refits. Returns
    (a, b, resid_all, keep_mask)."""
    x, y = xy[:, 0], xy[:, 1]
    keep = np.ones(len(x), bool)
    a, b = 1.0, 0.0
    for _ in range(n_iter):
        if keep.sum() < 3:
            break
        a, b = np.polyfit(x[keep], y[keep], 1)
        r = y - (a * x + b)
        mad = np.median(np.abs(r[keep] - np.median(r[keep]))) * 1.4826
        new = np.abs(r) < max(floor_s, k * mad)
        if np.array_equal(new, keep):
            break
        keep = new
    a, b = np.polyfit(x[keep], y[keep], 1)
    return float(a), float(b), y - (a * x + b), keep


def _match(cam, vt, tol, o0):
    out = [(c, vt[j]) for c in cam
           for j in [int(np.argmin(np.abs(vt - (c + o0))))]
           if abs(vt[j] - (c + o0)) < tol]
    return np.array(out) if out else np.empty((0, 2))


def _find_time_segments(cam, vt, tol, coarse, min_anchors, max_segments=4):
    """Greedily find one or more internally-consistent camera<->VR time
    segments, instead of assuming a single offset explains the whole
    recording. A camera stall / VR-log hiccup can put a discrete jump (not a
    smooth wobble) partway through a session -- anchors on the far side of
    that jump never match the single global offset at all, so they'd
    otherwise be silently lost rather than just trimmed.

    Each round: coarse-search an offset among whatever camera flashes are
    still unexplained, robust-fit it, and if it explains enough of them,
    keep it as a segment and remove its matches before the next round.
    Returns a list of dicts (o0, pairs=kept (cam,vt) pairs), largest first.
    """
    remaining = np.arange(len(cam))
    segments = []
    for _ in range(max_segments):
        if len(remaining) < min_anchors:
            break
        sub = cam[remaining]
        best = None
        for o0 in _coarse_offsets(sub, vt, coarse[0], coarse[1]):
            pairs, src = [], []
            for k, c in enumerate(sub):
                j = int(np.argmin(np.abs(vt - (c + o0))))
                if abs(vt[j] - (c + o0)) < tol:
                    pairs.append((c, vt[j]))
                    src.append(remaining[k])
            if len(pairs) < min_anchors:
                continue
            pairs = np.array(pairs)
            a, b, resid, keep = _robust_line(pairs)
            score = (int(keep.sum()), -float(resid[keep].std()))
            if best is None or score > best[0]:
                best = (score, o0, pairs, keep, np.array(src))
        if best is None or best[0][0] < min_anchors:
            break
        _, o0, pairs, keep, src = best
        segments.append(dict(o0=float(o0), pairs=pairs[keep]))
        remaining = np.setdiff1d(remaining, src[keep])
    segments.sort(key=lambda s: -len(s["pairs"]))
    return segments


def sync_led_to_vr(led_df, n_times, r_times, fps=FPS,
                   coarse=(-600.0, 60.0), tol_n=0.25, tol_r=0.90,
                   min_anchors=8, max_segments=4):
    """Fit frame -> VR-time from the LED flashes.

    The slope and the new-trial intercept `b_n` come from the PRIMARY (the
    largest) segment of long (~0.5 s) `n`-flashes -- their onset is well
    defined. The reward intercept `b_r` is a pure offset (median) of the
    brief `r`-flashes matched against that same primary segment, so a noisy
    `r`-flash onset never corrupts the clock rate.

    Some recordings have a discrete jump in the camera<->VR time
    relationship (a camera stall or VR-log hiccup), not just a smooth wobble
    -- flashes on the far side of that jump don't just get trimmed by the
    robust fit, they never match the single global offset at all. So the
    n- and r-flashes are each first split into one or more internally
    consistent time segments (`_find_time_segments`); every segment's
    matched pairs feed the piecewise anchor set (`anchors_n`/`anchors_r`),
    while only the primary segment sets the linear model (`a`, `b_n`,
    `resid_sd_ms`, etc.) used for QC and as the fallback outside the
    anchored span. An iterative robust trim within each segment, plus a
    final leave-one-out pass across the merged anchors, drops lone
    mis-detections while keeping smoothly-drifted stretches.

    Returns a model dict: a, b_n, b_r, b_generic, fps_eff, camera_lead_s,
    resid (kept), resid_sd_ms / resid_max_ms, anchors_n / anchors_r (kept),
    n_anchors, n_anchors_dropped, o0.

    Raises RuntimeError if too few `n`-flashes match the VR log (LED not
    tracked this session) -- the caller should then fall back to the VR log.
    """
    n_on = led_df.loc[led_df.kind == "n", "onset_frame"].to_numpy() / fps
    r_on = led_df.loc[led_df.kind == "r", "onset_frame"].to_numpy() / fps
    n_times = np.asarray(n_times, float)
    r_times = np.asarray(r_times, float)

    # --- n-flashes: one or more consistent time segments ---
    n_segs = _find_time_segments(n_on, n_times, tol_n, coarse, min_anchors, max_segments)
    if not n_segs:
        raise RuntimeError(
            f"LED sync failed: {len(n_on)} n-flash matched no consistent segment of "
            f"{len(n_times)} VR trials (LED not tracked / blocked this session)")

    primary = n_segs[0]
    o0 = primary["o0"]
    a, b_n, resid_n, keep_n = _robust_line(primary["pairs"])
    rn = resid_n[keep_n]

    if len(n_segs) > 1:
        extra = ", ".join(f"{len(s['pairs'])}@{s['o0']:+.2f}s" for s in n_segs[1:])
        print(f"  [sync] {len(n_segs)} time segments found (primary {len(primary['pairs'])}"
              f"@{o0:+.2f}s; also {extra}) -- discrete camera/VR time jump, bridging with "
              f"piecewise anchors from every segment")

    # --- r-flashes: same multi-segment search (a genuine clock jump moves
    # r-flashes the same way it moves n-flashes) ---
    r_segs = _find_time_segments(r_on, r_times, tol_r, coarse, min_anchors, max_segments)
    if r_segs:
        pr = r_segs[0]["pairs"]
        rr = pr[:, 1] - a * pr[:, 0]
        med = np.median(rr)
        mad = np.median(np.abs(rr - med)) * 1.4826
        keep_r = np.abs(rr - med) < max(0.12, 3.5 * mad)
        if keep_r.sum() < 3:
            keep_r = np.ones(len(pr), bool)
        b_r = float(np.median(rr[keep_r]))
    else:
        pr, keep_r = np.empty((0, 2)), np.zeros(0, bool)
        b_r = b_n - 0.5           # fall back to the nominal r/n flash-lag gap
    resid_r = (pr[:, 1] - (a * pr[:, 0] + b_r)) if len(pr) else np.empty(0)

    rk = np.concatenate([rn, resid_r[keep_r]])
    resid_t = np.concatenate([primary["pairs"][keep_n, 1], pr[keep_r, 1]]) if len(rk) else np.empty(0)

    # --- piecewise anchors = ALL matched flashes from EVERY segment (bridges
    # any discrete jump), minus lone mis-detections. n- and r- knots must be
    # sorted by frame before the leave-one-out neighbour check. ---
    all_n_pairs = np.vstack([s["pairs"] for s in n_segs])
    all_n_pairs = all_n_pairs[np.argsort(all_n_pairs[:, 0])]
    all_r_pairs = np.vstack([s["pairs"] for s in r_segs]) if r_segs else np.empty((0, 2))
    if len(all_r_pairs):
        all_r_pairs = all_r_pairs[np.argsort(all_r_pairs[:, 0])]

    kn_pw = _trim_knots(all_n_pairs)
    kr_pw = _trim_knots(all_r_pairs) if len(all_r_pairs) else np.zeros(0, bool)

    return dict(a=float(a), b_n=float(b_n), b_r=float(b_r),
                b_generic=float((b_n + b_r) / 2),
                fps_eff=float(fps * a), camera_lead_s=float(-((b_n + b_r) / 2) / a),
                resid=rk, resid_t=resid_t,
                resid_kinds=np.array(["n"] * int(keep_n.sum()) + ["r"] * int(keep_r.sum())),
                resid_sd_ms=float(rn.std() * 1000) if len(rn) else np.nan,
                resid_max_ms=float(np.abs(rn).max() * 1000) if len(rn) else np.nan,
                r_flash_spread_ms=float(resid_r[keep_r].std() * 1000) if keep_r.any() else np.nan,
                anchors_n=all_n_pairs[kn_pw], anchors_r=all_r_pairs[kr_pw],
                n_anchors=int(kn_pw.sum() + kr_pw.sum()),
                n_anchors_dropped=int((~kn_pw).sum() + (~kr_pw).sum()),
                n_linefit_anchors=int(keep_n.sum()),
                n_segments=len(n_segs),
                o0=float(o0), piecewise=True)


# --------------------------------------------------------------------------
# Clock conversion
#
# The linear model  vr = a*(frame/fps) + b_{n,r,generic}  is the base.
# When `model["piecewise"]` is set, the frame<->VR map is instead a
# straight-line interpolation THROUGH the detected LED anchors (each trial
# sits exactly on its own flash), with the linear model as the fallback
# outside the anchored span / in LED-blocked stretches. This removes the
# slow camera-clock wobble some sessions have. `resid_sd_ms` still reports
# the LINEAR residual (an honest "how non-linear was the clock" number).
# --------------------------------------------------------------------------

def _lag(model, kind):
    return {"n": model["b_n"] - model["b_generic"],
            "r": model["b_r"] - model["b_generic"], "generic": 0.0}[kind]


def _pw_points(model, fps=FPS):
    """Sorted (cam_frame, true_vr_s) anchor points for piecewise interp, or None.

    true_vr = logged event time minus that flash type's lag vs b_generic, so
    n- and r-anchors land on one common clock."""
    pts = []
    an, ar = model.get("anchors_n"), model.get("anchors_r")
    if an is not None and len(an):
        pts.append(np.c_[an[:, 0] * fps, an[:, 1] - (model["b_n"] - model["b_generic"])])
    if ar is not None and len(ar):
        pts.append(np.c_[ar[:, 0] * fps, ar[:, 1] - (model["b_r"] - model["b_generic"])])
    if not pts:
        return None
    P = np.vstack(pts)
    P = P[np.argsort(P[:, 0])]
    keep = np.r_[True, np.diff(P[:, 0]) > 0.5]        # dedupe near-identical frames
    P = P[keep]
    if len(P) < 2:
        return None
    P[:, 1] = np.maximum.accumulate(P[:, 1])          # kill sub-frame non-monotonic blips
    return P[:, 0], P[:, 1]


def frame_to_vr(frame, model, kind="generic", fps=FPS):
    """Camera frame(s) -> VR elapsed time (s)."""
    frame = np.asarray(frame, float)
    off = _lag(model, kind)
    lin = model["a"] * (frame / fps) + model["b_generic"] + off
    if model.get("piecewise"):
        pw = _pw_points(model, fps)
        if pw is not None:
            fr, vr = pw
            interp = np.interp(frame, fr, vr, left=np.nan, right=np.nan) + off
            return np.where(np.isfinite(interp), interp, lin)
    return lin


def vr_to_frame(t, model, kind="generic", fps=FPS):
    """VR elapsed time(s) (s) -> camera frame."""
    t = np.asarray(t, float)
    off = _lag(model, kind)
    lin = (t - off - model["b_generic"]) / model["a"] * fps
    if model.get("piecewise"):
        pw = _pw_points(model, fps)
        if pw is not None:
            fr, vr = pw
            interp = np.interp(t - off, vr, fr, left=np.nan, right=np.nan)
            return np.where(np.isfinite(interp), interp, lin)
    return lin


def load_sync_model(lickproc_h5):
    """Read the `sync` group of a `*_lickproc.h5` back into a model dict."""
    import h5py
    with h5py.File(lickproc_h5, "r") as f:
        m = {k: float(v) for k, v in f["sync"].attrs.items()}
        for k in ("anchors_n", "anchors_r"):
            if k in f["sync"]:
                m[k] = f["sync"][k][:]
    m["piecewise"] = ("anchors_n" in m or "anchors_r" in m)
    return m
