r"""
batch_lickproc.py
=================

Run notebooks 1 -> 2 -> 3 over every recording of the licking dataset, so that
`4.SessionComparison.ipynb` has everything it needs.

For each recording folder (`.../videos/<animal>/<rec>/`) it writes, in place:
    <rec>_dlc_concat.h5
    <rec>_lickproc.h5
    <rec>_lickmetrics.csv  /  <rec>_lickmetrics_trajectory.csv  /  <rec>_lickmetrics.h5
    lick_figures/                     every figure from all 3 notebooks, named
                                       <animalid>_<sessionid>_<figurename>.png
                                       (e.g. jsy083_b1_lick_rate_vs_position.png)
    lick_figures/_executed_{1,2,3}.ipynb   the executed notebooks (kept even on failure)

and at the dataset root (F:\dlc\lick-detector_v0-JSY-2026-06-16\videos):
    all_sessions_lickmetrics.csv      one row per recording   ->  notebook 4
    all_sessions_trajectory.csv       stacked rolling within-session rows
    batch_lickproc_log.txt

Run it with the same interpreter the notebooks use (the `python3` Jupyter kernel):
    C:\Users\jasmineyeo\AppData\Local\anaconda3\envs\preg-mini2p\python.exe batch_lickproc.py

Usage
-----
    python batch_lickproc.py                       # all 22 recordings
    python batch_lickproc.py --list                # just print what would run
    python batch_lickproc.py --only 260618_JSY083_B1
    python batch_lickproc.py --animals JSY083
    python batch_lickproc.py --no-verify-videos    # skip the cv2 frame-count check
    python batch_lickproc.py --skip-existing       # don't re-run recordings already done

JSY / V1_SpatialModulation - 09.LickingBehavior
"""

from __future__ import annotations

import os
import re
import glob
import time
import base64
import argparse
import traceback
import datetime as _dt

import numpy as np
import pandas as pd
import h5py
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

NB_DIR       = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = r"F:\dlc\lick-detector_v0-JSY-2026-06-16\videos"
ANIMALS      = ["JSY083", "JSY084"]
NOTEBOOKS    = ["1.Preprocess.ipynb", "2.LickDetection.ipynb", "3.LickMetrics.ipynb"]
KERNEL       = "python3"
CELL_TIMEOUT = 1800

_SESSION_ORDER = {**{f"B{i}": i for i in range(1, 9)},
                  "Saline": 9, "DCZ100": 10, "DCZ200": 11}


# --------------------------------------------------------------------------

def discover_recordings(animals, only=None):
    recs = []
    for animal in animals:
        for d in sorted(glob.glob(os.path.join(DATASET_ROOT, animal, "*"))):
            if not os.path.isdir(d):
                continue
            if not glob.glob(os.path.join(d, "*_split*DLC*.h5")):
                continue
            name = os.path.basename(d)
            if only and only not in name:
                continue
            recs.append((animal, name, d))
    return recs


def parse_session(rec_name):
    parts = rec_name.split("_")
    label = parts[-1] if len(parts) >= 3 else rec_name
    m = re.match(r"B(\d+)$", label)
    return dict(date=parts[0] if parts else "",
                session_label=label,
                session_type="baseline" if m else label,
                baseline_day=int(m.group(1)) if m else np.nan,
                session_order=_SESSION_ORDER.get(label, 99))


def run_notebook(src_ipynb, dst_ipynb, rec_folder, verify_videos):
    os.environ["LICK_RECORDING_FOLDER"] = rec_folder
    os.environ["LICK_VERIFY_VIDEOS"] = "1" if verify_videos else "0"
    nb = nbformat.read(src_ipynb, as_version=4)
    client = NotebookClient(nb, timeout=CELL_TIMEOUT, kernel_name=KERNEL,
                            resources={"metadata": {"path": NB_DIR}})
    try:
        client.execute()
    finally:
        nbformat.write(nb, dst_ipynb)      # keep the executed copy even if a cell raised
    return nb


def harvest_figures(nb, out_dir, animal_id, session_id, seen):
    """Save every figure as `<animal_id>_<session_id>_<figure_name>.png`.

    `seen` is a {filename: count} dict shared across all 3 notebooks for this
    recording, so a repeated section name (e.g. two plots under one heading)
    gets a `_2`, `_3`, ... suffix instead of overwriting the first one."""
    slug, k = "fig", 0
    for cell in nb.cells:
        if cell.cell_type == "markdown" and cell.source.strip():
            head = cell.source.strip().splitlines()[0].lstrip("#").strip()
            head = head.split("(")[0].replace("\u2014", " ")
            head = re.sub(r"^\d+[.)]\s*", "", head)           # drop a leading "3. "
            slug = re.sub(r"[^a-z0-9]+", "_", head.lower()).strip("_")[:44] or "fig"
        for o in cell.get("outputs", []):
            png = o.get("data", {}).get("image/png")
            if png:
                k += 1
                name = f"{animal_id}_{session_id}_{slug}"
                seen[name] = seen.get(name, 0) + 1
                fname = name if seen[name] == 1 else f"{name}_{seen[name]}"
                with open(os.path.join(out_dir, f"{fname}.png"), "wb") as fh:
                    fh.write(base64.b64decode(png))
    return k


def collect_summary(rec_folder, rec_name, animal):
    row = dict(recording=rec_name, animal=animal, **parse_session(rec_name))
    csv = os.path.join(rec_folder, f"{rec_name}_lickmetrics.csv")
    if os.path.isfile(csv):
        m = pd.read_csv(csv).iloc[0].to_dict()
        m.pop("recording", None)
        row.update(m)
    lp = os.path.join(rec_folder, f"{rec_name}_lickproc.h5")
    if os.path.isfile(lp):
        try:
            with h5py.File(lp, "r") as f:
                for k in ("n_trials", "n_led_ok_trials", "n_sync_wobble_trials",
                          "led_flagged_trials", "led_blocked_runs", "reward_zone_au",
                          "clip_start_frame"):
                    if k in f.attrs:
                        row[k] = f.attrs[k]
                if "sync" in f:
                    row["sync_resid_sd_ms"] = float(f["sync"].attrs.get("resid_sd_ms", np.nan))
                    row["camera_lead_s"] = float(f["sync"].attrs.get("camera_lead_s", np.nan))
        except Exception as e:
            row["lickproc_read_error"] = str(e)
    return row


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=None, help="substring filter on recording name")
    ap.add_argument("--animals", nargs="+", default=ANIMALS)
    ap.add_argument("--list", action="store_true", help="print recordings and exit")
    ap.add_argument("--no-verify-videos", dest="verify", action="store_false")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    recs = discover_recordings(args.animals, args.only)
    print(f"{len(recs)} recording(s):")
    for a, name, _ in recs:
        s = parse_session(name)
        print(f"   {a}  {name:28s}  -> {s['session_label']} (order {s['session_order']})")
    if args.list:
        return
    print()

    log_lines, summary_rows, traj_frames, failures = [], [], [], []
    t0 = time.time()

    for n, (animal, rec_name, rec_folder) in enumerate(recs, 1):
        tag = f"[{n}/{len(recs)}] {animal} {rec_name}"
        done_csv = os.path.join(rec_folder, f"{rec_name}_lickmetrics.csv")
        if args.skip_existing and os.path.isfile(done_csv):
            print(f"{tag}  -- skip (already done)")
            summary_rows.append(collect_summary(rec_folder, rec_name, animal))
            tj = os.path.join(rec_folder, f"{rec_name}_lickmetrics_trajectory.csv")
            if os.path.isfile(tj):
                traj_frames.append(pd.read_csv(tj).assign(
                    animal=animal, session_label=parse_session(rec_name)["session_label"]))
            continue

        figdir = os.path.join(rec_folder, "lick_figures")
        os.makedirs(figdir, exist_ok=True)
        for old in glob.glob(os.path.join(figdir, "*.png")):   # clear stale figures
            os.remove(old)
        print(f"{tag}  ...", flush=True)
        ts = time.time()
        animal_id = animal.lower()
        session_id = parse_session(rec_name)["session_label"].lower()
        seen = {}
        try:
            for i, nbf_name in enumerate(NOTEBOOKS, 1):
                nb = run_notebook(os.path.join(NB_DIR, nbf_name),
                                  os.path.join(figdir, f"_executed_{i}.ipynb"),
                                  rec_folder, args.verify)
                nfig = harvest_figures(nb, figdir, animal_id, session_id, seen)
                print(f"      nb{i} {nbf_name:22s} {nfig:2d} figures", flush=True)

            row = collect_summary(rec_folder, rec_name, animal)
            summary_rows.append(row)
            tj = os.path.join(rec_folder, f"{rec_name}_lickmetrics_trajectory.csv")
            if os.path.isfile(tj):
                traj_frames.append(pd.read_csv(tj).assign(
                    animal=animal, session_label=parse_session(rec_name)["session_label"]))

            msg = (f"{animal} {rec_name}: OK ({time.time()-ts:.0f}s)  "
                   f"trials={row.get('n_trials','?')} led_ok={row.get('n_led_ok_trials','?')} "
                   f"resid={row.get('sync_resid_sd_ms', float('nan')):.0f}ms "
                   f"flagged={row.get('led_flagged_trials','?')} "
                   f"blocked={row.get('led_blocked_runs','none')}")
            print("     ", msg)
            log_lines.append(msg)
        except CellExecutionError as e:
            failures.append(rec_name)
            print(f"      {rec_name}: FAILED in a notebook cell")
            log_lines.append(f"{animal} {rec_name}: FAILED (CellExecutionError)\n{e}")
        except Exception as e:
            failures.append(rec_name)
            print(f"      {rec_name}: FAILED -- {e}")
            log_lines.append(f"{animal} {rec_name}: FAILED\n{traceback.format_exc()}")

    # ---- master outputs for notebook 4 ----
    if summary_rows:
        sdf = pd.DataFrame(summary_rows).sort_values(
            ["animal", "session_order"]).reset_index(drop=True)
        p = os.path.join(DATASET_ROOT, "all_sessions_lickmetrics.csv")
        sdf.to_csv(p, index=False)
        print("\nwrote", p, f"({len(sdf)} rows x {sdf.shape[1]} cols)")
    if traj_frames:
        p = os.path.join(DATASET_ROOT, "all_sessions_trajectory.csv")
        pd.concat(traj_frames, ignore_index=True).to_csv(p, index=False)
        print("wrote", p)

    with open(os.path.join(DATASET_ROOT, "batch_lickproc_log.txt"), "w") as fh:
        fh.write(f"batch_lickproc  {_dt.datetime.now().isoformat(timespec='seconds')}\n"
                 f"{len(recs)} recordings, {len(failures)} failed, "
                 f"{(time.time()-t0)/60:.0f} min\n\n" + "\n".join(log_lines))

    print(f"\n{'='*70}\nDONE  {len(recs)-len(failures)}/{len(recs)} ok   "
          f"{(time.time()-t0)/60:.0f} min")
    if failures:
        print("FAILED:", failures)


if __name__ == "__main__":
    main()
