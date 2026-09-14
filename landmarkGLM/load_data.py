# -*- coding: utf-8 -*-
"""
landmarkGLM/load_data.py

This code is from Jasmine
"""

import os
import numpy as np
import pandas as pd

TAB = chr(9)


def load_suite2p(suite2p_dir, cfg):

    F = np.load(os.path.join(suite2p_dir, "F.npy"))
    Fneu = np.load(os.path.join(suite2p_dir, "Fneu.npy"))
    iscell = np.load(os.path.join(suite2p_dir, "iscell.npy"))
    spks = np.load(os.path.join(suite2p_dir, "spks.npy"))

    good = iscell[:, 0] == 1
    goodspks = spks[good, :]

    Fc = F[good] - cfg.neuropil_coef * Fneu[good]
    F0 = np.percentile(Fc, cfg.baseline_pct, axis=1)

    dff_cells = F0 > cfg.min_F0
    dff = (Fc[dff_cells] - F0[dff_cells, None]) / F0[dff_cells, None]
    goodspks = goodspks[dff_cells]

    cell_ids = np.where(good)[0][dff_cells]

    return dff, goodspks, cell_ids


def load_vrlog(path):

    with open(path, "r") as f:
        lines = f.readlines()

    hdr = 0
    for i, ln in enumerate(lines[:20]):
        if ln.split(TAB)[0].strip() == "CurrentTime":
            hdr = i
            break
    data_lines = lines[hdr + 1:]
    cols = ["CurrentTime", "ElapsedTime(seconds)", "EventType",
            "location_au", "location_cm", "trial_num", "reward_location"]
    rows = []
    for line in data_lines:
        line = line.rstrip("\n").rstrip("\r")
        if not line.strip():
            continue
        fields = line.split(TAB)
        fields = fields + [""] * (len(cols) - len(fields))
        rows.append(fields[:len(cols)])

    df = pd.DataFrame(rows, columns=cols)
    for c in ["ElapsedTime(seconds)", "location_au", "location_cm",
              "trial_num", "reward_location"]:
        df[c] = pd.to_numeric(df[c].replace("", np.nan), errors="coerce")
    df["EventType"] = df["EventType"].astype(str).str.strip()

    return df


def fit_clock_offset(vr_df, Y, cell_ids, cfg, smi_path=None, suite2p_dir=None,
                     offsets=None, reference=None):

    if offsets is None:
        offsets = np.arange(-3.0, 1.01, 0.25)

    if reference is None:
        if smi_path is None or not os.path.exists(smi_path):
            return getattr(cfg, "t_offset_s", 0.0), None
        import h5py
        with h5py.File(smi_path, "r") as f:
            n_tot = int(f.attrs.get("n_cells_total", 0)) or f["cell_info/med_coords"].shape[0]
            ref = np.full(n_tot, np.nan)
            rel = np.zeros(n_tot, dtype=bool)
            for lay in f["layer_smi"]:
                g = f["layer_smi"][lay]
                ci = g["cell_indices"][:]
                ref[ci] = g["preferred_positions"][:]
                rel[ci] = g["was_reliable_valid"][:]
            lo_cm = float(f["parameters"].attrs.get("exclude_start_cm", 0))
            bc = f["cell_info/bin_centers"][:]
            hi_cm = float(bc[-1]) - float(f["parameters"].attrs.get("exclude_end_cm", 0))
        iscell = np.load(os.path.join(suite2p_dir, "iscell.npy"))
        good = np.where(iscell[:, 0] == 1)[0]
        rank = np.searchsorted(good, cell_ids)
        reference, keep = ref[rank], rel[rank]
    else:
        keep = np.isfinite(reference)
        lo_cm, hi_cm = 0.0, cfg.corridor_cm

    pos_rows = vr_df["EventType"].values == "p"
    vt = vr_df["ElapsedTime(seconds)"].values[pos_rows]
    vp = vr_df["location_cm"].values[pos_rows]
    ok = np.isfinite(vt) & np.isfinite(vp)
    vt, vp = vt[ok], vp[ok]
    order = np.argsort(vt)
    vt, vp = vt[order], vp[order]
    lap_times = vr_df["ElapsedTime(seconds)"].values[vr_df["EventType"].values == "n"]
    lap_times = np.sort(lap_times[np.isfinite(lap_times)])

    Yt = np.asarray(Y).T
    n_frames = Yt.shape[0]
    nb = int(cfg.corridor_cm)
    edges = np.linspace(0, cfg.corridor_cm, nb + 1)
    xc = (np.arange(nb) + 0.5) * (cfg.corridor_cm / nb)

    win = (xc >= lo_cm) & (xc <= hi_cm)
    use = keep & np.isfinite(reference)

    best, rows = None, []
    for off in offsets:
        t_2p = off + np.arange(n_frames) / cfg.fps
        pos = np.interp(t_2p, vt, vp)
        gap_i = np.where(np.diff(vt) > cfg.iti_gap_s)[0]
        iti = np.zeros(n_frames, dtype=bool)
        for i in gap_i:
            mm = (t_2p > vt[i]) & (t_2p < vt[i + 1])
            iti |= mm
            pos[mm] = vp[i]

        d = np.diff(pos, prepend=pos[0])
        d[d < -50] = np.nan
        d = pd.Series(d).ffill().bfill().values
        speed = np.abs(d) * cfg.fps
        lap = np.searchsorted(lap_times, t_2p, side="right") - 1
        m = ((speed >= cfg.speed_thresh) & (lap >= 0) & (~iti)
             & (pos >= 0) & (pos <= cfg.corridor_cm))

        if m.sum() < 2000:
            continue

        mu, sd = Yt[m].mean(axis=0), Yt[m].std(axis=0)
        Yz = (Yt - mu) / np.where(sd == 0, 1.0, sd)
        b = np.clip(np.digitize(pos[m], edges) - 1, 0, nb - 1)
        C = np.full((nb, Yt.shape[1]), np.nan)
        ym = Yz[m]
        for k in range(nb):
            s = b == k
            if s.sum():
                C[k] = ym[s].mean(axis=0)

        C = pd.DataFrame(C).interpolate(limit_direction="both").values
        C = C - C.mean(axis=0)
        pk = xc[win][np.nanargmax(C[win], axis=0)]
        err = np.abs(pk - reference)
        med = float(np.nanmedian(err[use]))
        frac = float(np.nanmean(err[use] < 15))
        rows.append((off, med, frac))

        if best is None or med < best[1]:
            best = (off, med, frac)

    if abs(best[0] - getattr(cfg, "t_offset_s", 0.0)) > 0.13:
        print("  WARNING: Config.t_offset_s is {:+.2f} s but the fit prefers {:+.2f} s".format(
            getattr(cfg, "t_offset_s", 0.0), best[0]))

    return best[0], rows