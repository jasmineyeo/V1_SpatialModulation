# 09. Licking Behavior — DCZ / DREADD (JSY083, JSY084)

> Folder was `10.LickingBehavior/`; renumbered to `09.` (double-digit scheme,
> after `08.DREADD_Analysis`).

Detect and quantify **licking behavior** from a face camera (DeepLabCut
tracked) across learning and under chemogenetic silencing (DCZ), using the
VR log for position and trial/reward events.

**Behavior-only cohort — no imaging.** Separate from `9.DREADD_Analysis/`
(the JSY090/JSY093 2p-imaging project). Animals here: **JSY083, JSY084**.
Inputs: the DLC `.h5` output + the VR log (+ optionally the TM log).

---

## Experimental design (per animal)

| Phase | Sessions | Purpose |
|---|---|---|
| Baseline | consecutive days (`B1`, `B2`, …) | animal learns the VR / reward structure |
| Saline | 1 session | vehicle control — should match trained baseline |
| DCZ 100 | 1 session (dose 100) | chemogenetic silencing, low dose |
| DCZ 200 | 1 session (dose 200) | chemogenetic silencing, high dose |

**One session per condition, no replicates.** Condition comes from the
session id / folder name — confirm exact naming across the full dataset.

## Hypotheses

1. **Baseline (learning):** licking gets **spatially narrower around the
   reward zone** across days; anticipatory licking sharpens.
2. **Saline:** lick frequency + location unchanged vs. trained baseline.
3. **DCZ 100 / 200:** frequency and/or location **perturbed**, possibly
   **dose-dependent** (200 > 100).

## Core readouts

Reward location is **fixed** — the animal runs a ~14→394 au corridor and
reward fires at **~392 au (the far end)** every trial, then teleports back
to ~14 (the `sequence` row is a leftover from a previous experiment,
ignored). Spatial licking analysis is in **absolute VR position**.

**Position logging pauses for ~1.5–3 s during the reward→new-trial ITI** —
there is no position data while the animal drinks, so align consummatory
licks by **time** (reward frame), not position. `VRLog.position_at(t,
method="previous")` (step, no interp across the gap) is the right call
near events.

- **Lick-rate-vs-VR-position curve** per session → peak offset from the
  reward zone, spread (SD / FWHM), fraction of licks inside the RZ.
- **PSTH aligned to the reward** (`r_anchor_frame`) — anticipatory ramp
  vs. consummatory burst, in the ITI where there's no position.
- **Lick rate** — licks/s overall, in-RZ, pre-RZ anticipatory window; bouts.
- **Trial-level timing** — first-lick position / latency to reward.
- Session summaries → across-day baseline trajectory, then saline /
  DCZ100 / DCZ200 vs. the trained-baseline reference.

---

## Data layout (confirmed on `F:\dlc\test`, session `260618_JSY083_B1`)

```
session  ──  ONE VRlog_*.txt   +   ONE TMlog_*.txt   (cover the whole session)
  └── recording  ──  face-cam video, CFR 30 fps, 1280×720
        └── split1..split6   ( 5 × 10 000  +  1 × 4 719  =  54 719 frames )
              each split →  _splitN...h5           ← the pose data  (USE THIS)
                            _splitN....mp4          ← the split video (QC only)
                            _splitN..._full.pickle  ← raw detections  (ignore)
                            _splitN..._meta.pickle  ← DLC run meta     (ignore)
                            _splitN..._pXX_labeled.mp4 ← labeled QC video (ignore)
```

The camera ran **continuously** for the whole session. Splitting is a
DLC-chunking step only. `split6` here was scored by a **different DLC
model** (`shuffle1_snapshot_best-60` vs `shuffle3_snapshot_best-290`) —
`concat_dlc_splits.py` flags this in its warnings.

### DLC pose — bodyparts `led`, `tongue`, `chin`

(Not "LED / chin / jaw" — the trained model has **`tongue`**, which is the
primary lick signal; `chin` is secondary.)

| bodypart | use |
|---|---|
| `led` | camera↔VR sync (flashes on `n` and `r`) |
| `tongue` | lick detection (visible only during a lick) |
| `chin` | lick detection, secondary / support |

**Use the DLC `led` bodypart at likelihood > 0.6** as the LED on/off
signal (per Jasmine — the video is only for visual confirmation).

**The face camera was re-seated between recording days**, so the LED (and
the tongue) sit at a **different pixel every session** — LED x ∈ [296, 402],
y ∈ [492, 589] across the 22 recordings; notebook 2 auto-locates the tongue
the same way it once did the LED. **The LED itself is no longer gated by any
pixel location at all** — `extract_led_epochs` (in `lick_sync.py`) classifies
every candidate flash purely by its **blink pattern**: any contiguous
`led` p>0.6 run (VR-monitor-noise region excluded) is kept if the detections
*within that one blink* stay spatially tight (≤30 px). A per-session or
per-split fixed center was tried first and both failed on `260620_JSY083_B3`,
where the LED sits at two different pixels (≈50 px apart) within the same
session — the blink-pattern method needs no location at all, so it handles a
shift happening anywhere, mid-split included, and runs at the same cost.

- **LED sync** — see the sync section below.
- **Lick detection** uses `tongue` presence (likelihood + position
  excursion) plus `chin` motion, tuned on real licks during QC.

### Camera timing — from the DLC LED flashes

The videos are **constant 30.000 fps** with synthetic timestamps (FMP4) —
the container has no wall-clock, so the video alone can't place a frame on
the VR clock. The **DLC `led` flashes** do it: the camera ran continuously
at a rock-solid 30 fps, so `vr_time ≈ a·(frame/30) + b`, and the flashes
pin `a` and `b` (see sync section — residual **15 ms** on the sample).

### VR log — `VRlog_*.txt`

- Line 1: `Starting new session <M/D/YYYY h:mm:ss AM>`
- Line 2: `Log format is \t CurrentTime \t ElapsedTime(seconds) \t EventType \t trial# \t RewardLocation`
- Line 3: event-type legend
- Data from line 4. `CurrentTime` = `HH.MM.SS.ffffff`.
- **`p` rows have 4 fields**: `CurrentTime, ElapsedTime, 'p', Location(au)`.
- Other events have 5: `..., 'n'|'r'|…, trial#, RewardLocation` (an index 1–6).

**Event types:** `s` = image-acq start · `p` = position · `z` = RZ hit ·
`r` = reward · `t` = teleportation · `n` = new trial · `f` = timeout ·
`e` = session ended · `sequence` = reward-location order (ignored).

Gotchas seen in the sample:
- This session has **only** `s`, `sequence`, `n` (×98), `r` (×97), `p` —
  **no `z`/`t`/`f`/`e`**. So RZ-entry time isn't logged directly; derive
  the reward-zone VR position from animal position at `r` events.
- The log **ends with a long run of identical frozen `p` rows** — trim
  the trailing constant-position/constant-time block.
- The filename animal id (`VRlog_JSY038_...`) is a **stale template** —
  take the animal id from the DLC / folder name, not the VR log name.

### TM log — `TMlog_*.txt`

`Starting new session ...` / `Log format is \t current time \t distance \t speed`
/ `Max speed limit set to: N` / then `HH.MM.SS.ffffff \t distance \t speed`
rows (wall-clock only, no elapsed-seconds column; duplicate timestamps
occur). Gives true running speed — useful for lick-vs-speed controls.

---

## Camera ↔ VR-log sync (DLC `led` bodypart, p > 0.6)

The `led` bodypart flashes on two VR events, separable by duration:

| VR event | `led` p>0.6 run | measured on the sample |
|---|---|---|
| `n` (new trial) | **~16 frames (~0.53 s)** | 90/90 detected |
| `r` (reward)    | **1–3 frames** | 89/89 detected |

Method: `led` p > 0.6 (VR-monitor noise excluded) → contiguous runs (merge
gaps ≤ 3) → keep only runs whose OWN detections stay within 30 px of each
other (a real blink doesn't wander; no pre-located pixel needed) → classify
**brief (≤ 7 f) = `r`**, **long (≥ 9 f) = `n`**. Then the **slope + `b_n`** come from the long
`n`-flashes only (well-defined onset) via an iterative robust line fit
(trims anchors from blocked / clock-wobble stretches), and **`b_r`** is a
pure median offset of the brief `r`-flashes against that fixed slope.
`b_n − b_r ≈ 0.50 s` on every session (the physical r/n flash-lag gap — a
built-in sanity check). The camera often started **minutes** before the VR
program (camera-lead 4 s … 345 s across the 22), so the coarse offset
search is wide (−600 … +60 s).

**Piecewise sync.** `frame_to_vr` / `vr_to_frame` interpolate a straight line
**through the detected LED flashes** (n- and r-flashes on one common clock),
with the linear model as the fallback outside the anchored span / in blocked
stretches. This removes a slow camera-clock wobble ~5 sessions have (the webcam's
real capture rate isn't perfectly 30.000 fps over 45 min — DLC-row vs video-frame
counts match on every split, so it's capture jitter, not a concat error). Each
trial with its own flash then sits exactly on its anchors. `resid_sd_ms` still
reports the **linear** residual (the honest "how non-linear was the clock"
number); `n_sync_wobble_trials` counts trials whose flash is > 100 ms off the
linear anchor.

**Frame clipping.** The camera often starts minutes before the VR program
(camera-lead 4 s … 345 s across the 22). Notebook 1 drops every pose frame before
`n1 − 2 s`; `pose/frame` keeps the original frame numbers and `clip_start_frame`
records the offset, so you can still map back to the video.

Sample `260618_JSY083_B1`: residual **15 ms**, fps_eff 30.000, camera 8.7 s
early, 98/98 n + 97/97 r flashes.

### What the LED is actually marking

The LED marks **hardware** events; the VR log records the **software**
decisions, with a fixed pipeline latency between them (verified — same
with onset / center / weighted-centroid):

| interval | value |
|---|---|
| VR log: `r_time` → next `n_time` | 1.506 s (sd 4 ms) |
| LED: r-flash → n-flash | **~1.0 s** (0.5 s shorter) |

- **reward flash** (~67 ms) ≈ the **solenoid opening**, ~0.25 s *after* the
  logged `r` (command latency) → actual water delivery.
- **trial-start flash** (~0.5 s) ≈ the **teleport**, ~0.25 s *before* the
  logged `n` (the VR writes the log line at the end of its loop).

`b_generic = (b_n + b_r)/2` is the true frame↔VR-elapsed mapping; `b_r` /
`b_n` recover the physical-event frames from the log times.

### Anchors — use the log times through the sync

`vr_to_frame(t, model, kind)` with `kind ∈ {"n","r","generic"}`:

| anchor | = | is |
|---|---|---|
| `n_anchor_frame` | `vr_to_frame(n_time_vr, "n")` | teleport frame |
| `r_anchor_frame` | `vr_to_frame(r_time_vr, "r")` | valve-open frame |

100% coverage, latency-corrected, works for trial 1. With the piecewise sync
a trial that has its own flash gets an LED-exact anchor; a trial with no
flash gets the interpolation between its neighbours.

**`led_ok`** = this trial has its own LED flash(es). `led_ok = False` = no
flash (blocked / occluded / `n1`) → the anchor is interpolated (still good —
"if the rest of the sync is fine, trust the VR trial-start / reward"). These
trials are listed in `led_flagged_trials` / `led_blocked_runs` for review,
never dropped. `{n,r}_flash_resid_ms` = flash − **linear** anchor (marks
clock-wobble stretches); `sync_wobble` flags the > 100 ms ones.

---

## Scripts

### `concat_dlc_splits.py`  ✅ done, tested on `F:\dlc\test`

Concatenate one recording's `split1..splitN` DLC `.h5` → one continuous
pose trace.

```
python concat_dlc_splits.py F:\dlc\test
python concat_dlc_splits.py F:\dlc\test --recording 260618_JSY083_B1
```

Writes `{recording}_dlc_concat.h5` (plain h5py: `/pose` (N,9) f32,
`/frame`, `/split_number`, `/within_split`, provenance in attrs) and
`{recording}_dlc_concat.json` (per-split frame counts, video cross-check,
scorer-mismatch + numbering warnings). Load with
`concat_dlc_splits.load_concat_h5` / `concat_as_dataframe`.

Reads DLC `.h5` with **h5py directly, not `pandas.read_hdf`** — see below.

### `batch_lickproc.py`  ✅ done — all 22 recordings

Runs notebooks 1 → 2 → 3 over every recording under
`F:\dlc\lick-detector_v0-JSY-2026-06-16\videos\{JSY083,JSY084}\*\`.

```
<preg-mini2p python> batch_lickproc.py            # all 22
<preg-mini2p python> batch_lickproc.py --list     # dry run
<preg-mini2p python> batch_lickproc.py --only 260618_JSY083_B1
```

Run it with the **preg-mini2p** conda env's python (that's the `python3`
Jupyter kernel; it has `nbclient` + the full stack). Per recording folder:
all the per-session output files + `lick_figures/` (every PNG from the three
notebooks, named by section, plus `_executed_{1,2,3}.ipynb`). At the dataset
root: `all_sessions_lickmetrics.csv` (one row per recording), 
`all_sessions_trajectory.csv` (stacked rolling within-session), and
`batch_lickproc_log.txt`. The concat `.h5` is cached, so re-runs are ~15 s /
session (~6 min total).

---

## `dlc_utils/` — vendored DeepLabCut utilities

Copied **verbatim** (byte-for-byte, sha256-verified) from
`freely-moving-2P-preg/fm2p/utils/` (author: DMM), commit `4d4fcb4`:
`files.py` (`open_dlc_h5`, `read_h5`/`write_h5`, yaml), `helper.py`
(`split_xyl`, `apply_liklihood_thresh`, `nan_filt`), `filter.py`
(`convfilt`, `nanmedfilt`), `time.py`, `paths.py`, `correlation.py`
(`nanxcorr`). Only `__init__.py` is new (registers the package as `fm2p`
in `sys.modules` so the two files with a bare `import fm2p` resolve back
here).

**`open_dlc_h5` is NOT used by this pipeline.** It calls
`pandas.read_hdf`, which needs PyTables — and in the `JSY_SpMod` env,
importing `h5py` before PyTables breaks PyTables' DLL load
(`utilsextension` ImportError). Since the pipeline needs `h5py`
throughout, `concat_dlc_splits.read_dlc_h5` reads the DLC file with `h5py`
directly (the payload at `df_with_missing/table` is a plain compound
dataset; column names come from the pickled `non_index_axes` attr). The
verbatim `dlc_utils` are kept for `split_xyl` / `apply_liklihood_thresh` /
`nanmedfilt` / `nanxcorr`, which operate on plain arrays / DataFrames.

---

## Status

- [x] `dlc_utils/` vendored + import-tested (`JSY_SpMod`)
- [x] PyTables installed — then **abandoned** (h5py DLL conflict); DLC
      `.h5` is read with h5py directly instead
- [x] Real session inspected (`F:\dlc\test` / `260618_JSY083_B1`):
      bodyparts `led`/`tongue`/`chin`, 6 splits = 54 719 frames, CFR 30 fps,
      VR events `n`×98 / `r`×97 only, fixed reward index 5
- [x] `concat_dlc_splits.py` — split concatenation, tested
- [x] `lick_io.py` — `read_vrlog` / `read_tmlog`, tested
- [x] **`1.Preprocess.ipynb`** — concat → logs → LED extraction → LED↔VR sync
      (with diagnostic plots) → per-trial table → `{rec}_lickproc.h5`.
      Runs clean on the sample: 98 n + 97 r flashes, **15.6 ms** residual,
      0 trials flagged.
- [x] `lick_sync.py` — `extract_led_epochs` / `sync_led_to_vr` / `frame_to_vr` /
      `vr_to_frame` / `load_sync_model` (consolidated from notebook 1)
- [x] **`2.LickDetection.ipynb`** — licks = `tongue_likelihood` peaks **inside a
      chin bout**. Sample: 1500 licks, **7.5 Hz** ILI, → VR clock + position.
- [x] `lick_metrics.py` — `au_to_cm_factor` / `classify_licks` (4-cat) /
      `lick_rate_vs_position` / `speed_accel_vs_position` / `reward_psth` /
      `first_approach_lick` / `lick_scatter_data` / `session_metrics` / `rolling_metrics`
- [x] **`3.LickMetrics.ipynb`** — 4-category classify + composition · occupancy /
      lick-rate-vs-**cm**-position / speed+accel · per-session lick scatter (raw +
      binned) · reward PSTH · early-vs-late · speed control · **rolling
      within-session trajectory** → `{rec}_lickmetrics.{csv,h5}` + `_trajectory.csv`
- [x] **`batch_lickproc.py`** — driver, runs notebooks 1–3 over all 22 recordings;
      saves every figure per folder + master CSVs at the dataset root. All 22 ran.
- [x] **`4.SessionComparison.ipynb`** — baseline learning + saline / DCZ100 /
      DCZ200, within- **and** across-session, both animals; `session_comparison_summary.csv`

## Notebooks / pipeline

| notebook | in | does | out |
|---|---|---|---|
| **`1.Preprocess.ipynb`** | split `.h5` + VR/TM logs | concat · logs · LED flashes (DLC `led`, p>0.6) · **camera↔VR sync** (diagnostics) · per-trial table | `{rec}_dlc_concat.h5`, `{rec}_lickproc.h5` |
| **`2.LickDetection.ipynb`** | `lickproc.h5` | **licks** = `tongue_likelihood` peaks (`find_peaks` h≥0.3, dist≥2 f) **inside a chin bout** · **bouts** = smoothed `chin_likelihood` > 0.4 · licks → VR clock + position | `licks/` + `bouts/` → `lickproc.h5` |
| **`3.LickMetrics.ipynb`** | `lickproc.h5` | AU→cm · **4-category classify** + composition · occupancy-normalised **lick-rate-vs-position curve** + speed/accel · **per-session lick scatter** (raw + binned) · reward PSTH · first-lick (secondary) · early-vs-late · speed control · **rolling within-session trajectory** | `{rec}_lickmetrics.csv` + `_trajectory.csv` + `.h5` |
| `batch_lickproc.py` | dataset root | loop 1–3 over all 22 recordings | all per-session files |
| `4.SessionComparison.ipynb` | all sessions/animal | baseline learning curve · saline vs baseline · **DCZ 100/200 vs baseline & saline** · within- + across-session, both animals | comparison tables + figures |

`extract_led_epochs` / `sync_led_to_vr` / `frame_to_vr` / `vr_to_frame` /
`load_sync_model` → **`lick_sync.py`** (imported by 1 & 2).
`au_to_cm_factor` / `classify_licks` / `lick_rate_vs_position` /
`speed_accel_vs_position` / `reward_psth` / `first_approach_lick` /
`lick_scatter_data` / `session_metrics` / `rolling_metrics` → **`lick_metrics.py`**
(imported by 3).

### Trial timeline

`reach corridor end → black screen (~1.5 s, no VR position) → water/valve (LED
r-flash) → drink (~1.5 s) → teleport (LED n-flash)`. Position gap ~3 s. A
drinking bout straddles the valve and spills past the next teleport.

### AU → cm

`cm = au · 130 / (min(max_au, 393) − min_au)` per session (matches
`helper/BehavioralDataFiltering.py`). Sample: cm/au ≈ 0.343, reward zone ≈ 135 cm.

### 4 lick categories

A *drinking bout* = a chin bout spanning a trial's valve; the whole bout (incl.
frames past the next teleport) belongs to that reward.

| category | rule | learning → |
|---|---|---|
| `approaching_nonreward` | not drinking bout · has position · < RZ−30 cm | ↓ (exploratory) |
| `approaching_reward` | not drinking bout · has position · RZ−30…RZ−3 cm | ↑ (anticipation while running) |
| `consummatory_pre_reward` | in drinking bout · before valve | ↑ (black-screen temporal anticipation) |
| `consummatory_post_reward` | in drinking bout · at/after valve | drinking (absorbs old "carryover") |

### Learning-trajectory metrics (per session + rolling within-session)

`anticipatory_index` = (rz_rate − neutral_rate)/(rz_rate + neutral_rate) ∈ [−1,1]
— the **stable** primary metric (`anticipatory_ratio` also kept, floored).
`lick_median_cm` / `lick_com_cm` → toward RZ. `lick_spread_cm` ↓.
`frac_approach_licks_in_rz` ↑. `consummatory_pre_reward_per_trial` ↑.
`first_consummatory_lick_latency_s` ↓ (faster to lick after black-screen onset).
`anticipatory_onset_cm` is included but **flaky** (saturates) — don't lead with it.

`{rec}_lickproc.h5`:
- `pose/{data (N,9) f32, frame, split_number}` + `coord_names` attr
- `sync/{a, b_n, b_r, b_generic, fps_eff, camera_lead_s, resid_sd_ms,
  resid_max_ms, r_flash_spread_ms, o0, n_anchors, n_anchors_dropped,
  led_center_x, led_center_y, anchors_n, anchors_r}`
- `trials/{trial, n_time_vr, r_time_vr, n_anchor_frame, r_anchor_frame,
  n_flash_frame, r_flash_frame, n_flash_resid_ms, r_flash_resid_ms,
  reward_pos_au, led_ok, sync_wobble}` (frames are clip-relative)
- `licks/{frame, vr_time, position_au, trial, bout_id}` + attrs
  `tongue_p, chin_p, lick_hz, n_licks, n_bouts, tongue_center_x, tongue_center_y`
- `bouts/{start_frame, end_frame, n_licks, tongue_coverage}`
- root attrs `recording`, `reward_zone_au`, `clip_start_frame`, `n_trials`,
  `n_led_ok_trials`, `n_sync_wobble_trials`, `led_flagged_trials`,
  `led_blocked_runs`, `vrlog_path`, `fps`

Use `frame_to_vr(frame, model, kind)` / `vr_to_frame(t, model, kind)` with
`kind ∈ {"n","r","generic"}` to move between the camera and VR clocks.
