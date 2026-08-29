"""
lick_io.py
==========

Readers for the behavioral logs that accompany a face-camera recording:

    read_vrlog(path)   -> VRLog    (events, position, reward zone, n/r times)
    read_tmlog(path)   -> TMLog    (running speed / distance, wall clock)

Formats confirmed on `F:\dlc\test` (session `260618_JSY083_B1`).

VR log
------
    line 0 : "Starting new session <M/D/YYYY h:mm:ss AM>"
    line 1 : "Log format is \t CurrentTime \t ElapsedTime(seconds) \t EventType \t trial# \t RewardLocation"
    line 2 : "Event types: s=... p=... z=... r=... t=... n=... f=... e=..."
    line 3+: tab-separated rows
        'p' rows        : CurrentTime, ElapsedTime, 'p', Location(au)           (4 fields)
        other events    : CurrentTime, ElapsedTime, EventType, trial#, RewardLoc (5 fields)
        'sequence' row  : ..., 'sequence', 0, '[5,6,2,1,3,4]'   (leftover, ignored)

    Only `n` (new trial) and `r` (reward) events are used by this cohort.
    `CurrentTime` is `HH.MM.SS.ffffff` (seconds since midnight).
    The log often ends with a long frozen block of identical `p` rows
    (VR stopped but kept logging) — trimmed on load.

TM log
------
    line 0 : "Starting new session <M/D/YYYY h:mm:ss AM>"
    line 1 : "Log format is \t current time \t distance \t speed"
    line 2 : "Max speed limit set to: <N>"
    line 3+: CurrentTime, distance, speed   (tab-separated; duplicate
             timestamps occur; `speed` is in raw VR/encoder units)

Both logs' line-0 wall clock is the common reference: TM `current time`
minus that start time gives elapsed seconds comparable to VR `ElapsedTime`.

JSY / V1_SpatialModulation · 10.LickingBehavior
"""

from __future__ import annotations

import re
import datetime as _dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


_SESSION_RE = re.compile(r"Starting new session\s+(.+?)\s*$")
_CT_RE = re.compile(r"^\s*(\d{1,2})\.(\d{2})\.(\d{2})\.(\d+)\s*$")


def _parse_session_start(line: str) -> _dt.datetime | None:
    m = _SESSION_RE.search(line)
    if not m:
        return None
    for fmt in ("%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %H:%M:%S",
                "%m/%d/%Y %I:%M:%S%p"):
        try:
            return _dt.datetime.strptime(m.group(1).strip(), fmt)
        except ValueError:
            continue
    return None


def _currenttime_to_seconds(s: str) -> float:
    """`HH.MM.SS.ffffff` -> seconds since midnight (float). NaN if unparseable."""
    m = _CT_RE.match(str(s))
    if not m:
        return np.nan
    h, mnt, sec, frac = m.groups()
    return int(h) * 3600 + int(mnt) * 60 + int(sec) + float("0." + frac)


# ==========================================================================
# VR log
# ==========================================================================

@dataclass
class VRLog:
    path: str
    session_start: _dt.datetime | None
    events: pd.DataFrame          # non-position events: elapsed_s, event, trial, reward_loc, current_time_s
    position: pd.DataFrame        # p rows: elapsed_s, location_au, current_time_s
    n_trimmed: int                # trailing frozen p rows removed
    header: list[str] = field(default_factory=list)

    # --- convenience ---
    @property
    def n_times(self) -> np.ndarray:
        return self.events.loc[self.events.event == "n", "elapsed_s"].to_numpy()

    @property
    def r_times(self) -> np.ndarray:
        return self.events.loc[self.events.event == "r", "elapsed_s"].to_numpy()

    @property
    def n_trials(self) -> pd.DataFrame:
        return self.events[self.events.event == "n"].reset_index(drop=True)

    def event_times(self, code: str) -> np.ndarray:
        return self.events.loc[self.events.event == code, "elapsed_s"].to_numpy()

    def position_at(self, times, method: str = "linear",
                    max_gap_s: float | None = None) -> np.ndarray:
        """VR position (au) at the given elapsed times.

        method="linear"   : linear interpolation between p-rows (default).
        method="previous" : last logged position at or before each time
                            (step / forward-fill) — use this near events,
                            because position logging PAUSES during the
                            reward->new-trial ITI (~1.5 s) and linear interp
                            would blend across that gap.
        max_gap_s : with method="linear", NaN out points whose bracketing
                    p-rows are more than this far apart.
        """
        t = np.asarray(times, dtype=float)
        pe = self.position.elapsed_s.to_numpy()
        pl = self.position.location_au.to_numpy()
        if method == "previous":
            idx = np.searchsorted(pe, t, side="right") - 1
            out = np.where(idx >= 0, pl[np.clip(idx, 0, len(pl) - 1)], np.nan)
            return out
        out = np.interp(t, pe, pl)
        if max_gap_s is not None:
            hi = np.searchsorted(pe, t, side="right")
            lo = hi - 1
            ok = (lo >= 0) & (hi < len(pe))
            gap = np.where(ok, pe[np.clip(hi, 0, len(pe) - 1)] - pe[np.clip(lo, 0, len(pe) - 1)], np.inf)
            out = np.where(gap <= max_gap_s, out, np.nan)
        return out

    @property
    def reward_zone_au(self) -> float:
        """Reward-zone VR position, estimated as the median animal position
        at reward (`r`) events (this cohort does not log `z`)."""
        rt = self.r_times
        if rt.size == 0:
            return np.nan
        return float(np.nanmedian(self.position_at(rt, method="previous")))

    def summary(self) -> str:
        ev = self.events.event.value_counts().to_dict()
        return (f"VRLog {self.path}\n"
                f"  session_start : {self.session_start}\n"
                f"  events        : {ev}\n"
                f"  position rows : {len(self.position)} "
                f"({self.n_trimmed} frozen tail rows trimmed)\n"
                f"  elapsed span  : {self.position.elapsed_s.iloc[0]:.2f} .. "
                f"{self.position.elapsed_s.iloc[-1]:.2f} s\n"
                f"  position span : {self.position.location_au.min():.1f} .. "
                f"{self.position.location_au.max():.1f} au\n"
                f"  reward zone   : {self.reward_zone_au:.1f} au "
                f"(median pos @ {len(self.r_times)} r events)")


def read_vrlog(path: str, dedupe_position: bool = True) -> VRLog:
    """Parse a VRlog_*.txt. See module docstring for the format."""
    with open(path, "r", errors="replace") as fh:
        raw = fh.read().splitlines()

    header = raw[:3]
    session_start = _parse_session_start(header[0]) if header else None

    ev_ct, ev_el, ev_code, ev_trial, ev_rloc = [], [], [], [], []
    p_ct, p_el, p_loc = [], [], []

    for ln in raw[3:]:
        if not ln:
            continue
        f = ln.split("\t")
        if len(f) < 3:
            continue
        code = f[2]
        try:
            elapsed = float(f[1])
        except ValueError:
            continue
        if code == "p":
            if len(f) < 4:
                continue
            try:
                loc = float(f[3])
            except ValueError:
                continue
            p_ct.append(f[0]); p_el.append(elapsed); p_loc.append(loc)
        else:
            ev_ct.append(f[0]); ev_el.append(elapsed); ev_code.append(code)
            ev_trial.append(f[3] if len(f) > 3 else "")
            ev_rloc.append(f[4] if len(f) > 4 else "")

    position = pd.DataFrame({
        "current_time_s": [_currenttime_to_seconds(s) for s in p_ct],
        "elapsed_s": np.asarray(p_el, dtype=float),
        "location_au": np.asarray(p_loc, dtype=float),
    })

    # Trim the trailing frozen block: consecutive final rows with identical
    # (elapsed_s, location_au).
    n_trimmed = 0
    if len(position) > 1:
        le = position.elapsed_s.to_numpy()
        ll = position.location_au.to_numpy()
        i = len(position) - 1
        while i > 0 and le[i] == le[i - 1] and ll[i] == ll[i - 1]:
            i -= 1
        n_trimmed = len(position) - (i + 1)
        if n_trimmed:
            position = position.iloc[: i + 1].reset_index(drop=True)

    if dedupe_position and len(position):
        # drop later p rows that repeat an earlier elapsed_s (VRLog_Export rule)
        keep = ~position.elapsed_s.duplicated(keep="first")
        position = position[keep].reset_index(drop=True)

    events = pd.DataFrame({
        "current_time_s": [_currenttime_to_seconds(s) for s in ev_ct],
        "elapsed_s": np.asarray(ev_el, dtype=float),
        "event": ev_code,
        "trial": ev_trial,
        "reward_loc": ev_rloc,
    })

    return VRLog(path=path, session_start=session_start, events=events,
                 position=position, n_trimmed=n_trimmed, header=header)


# ==========================================================================
# TM log
# ==========================================================================

@dataclass
class TMLog:
    path: str
    session_start: _dt.datetime | None
    data: pd.DataFrame            # current_time_s, elapsed_s, distance, speed
    max_speed_limit: float | None
    header: list[str] = field(default_factory=list)

    def speed_at(self, times) -> np.ndarray:
        """Linear-interpolated running speed at the given elapsed times."""
        t = np.asarray(times, dtype=float)
        d = self.data.drop_duplicates("elapsed_s")
        return np.interp(t, d.elapsed_s.to_numpy(), d.speed.to_numpy())

    def summary(self) -> str:
        return (f"TMLog {self.path}\n"
                f"  session_start : {self.session_start}\n"
                f"  rows          : {len(self.data)}\n"
                f"  elapsed span  : {self.data.elapsed_s.iloc[0]:.2f} .. "
                f"{self.data.elapsed_s.iloc[-1]:.2f} s\n"
                f"  speed         : min {self.data.speed.min():.1f}  "
                f"med {self.data.speed.median():.1f}  max {self.data.speed.max():.1f} "
                f"(raw units; limit {self.max_speed_limit})")


def read_tmlog(path: str) -> TMLog:
    """Parse a TMlog_*.txt. See module docstring for the format."""
    with open(path, "r", errors="replace") as fh:
        raw = fh.read().splitlines()

    header = raw[:3]
    session_start = _parse_session_start(header[0]) if header else None
    max_speed = None
    for h in header:
        m = re.search(r"Max speed limit set to:\s*([\d.]+)", h)
        if m:
            max_speed = float(m.group(1))

    ct, dist, spd = [], [], []
    for ln in raw[3:]:
        if not ln:
            continue
        f = ln.split("\t")
        if len(f) < 3:
            continue
        try:
            d, s = float(f[1]), float(f[2])
        except ValueError:
            continue
        ct.append(f[0]); dist.append(d); spd.append(s)

    cts = np.array([_currenttime_to_seconds(s) for s in ct], dtype=float)
    if session_start is not None:
        start_s = (session_start.hour * 3600 + session_start.minute * 60
                   + session_start.second + session_start.microsecond / 1e6)
        elapsed = cts - start_s
    else:
        elapsed = cts - cts[0] if len(cts) else cts

    data = pd.DataFrame({
        "current_time_s": cts,
        "elapsed_s": elapsed,
        "distance": np.asarray(dist, dtype=float),
        "speed": np.asarray(spd, dtype=float),
    })
    return TMLog(path=path, session_start=session_start, data=data,
                 max_speed_limit=max_speed, header=header)


# ==========================================================================
# CLI - quick look
# ==========================================================================

if __name__ == "__main__":
    import argparse
    import glob
    import os

    ap = argparse.ArgumentParser(description="Summarize the VR / TM logs in a folder.")
    ap.add_argument("folder")
    args = ap.parse_args()

    vr = glob.glob(os.path.join(args.folder, "VRlog*.txt"))
    tm = glob.glob(os.path.join(args.folder, "TMlog*.txt"))
    for p in vr:
        print(read_vrlog(p).summary(), "\n")
    for p in tm:
        print(read_tmlog(p).summary(), "\n")
