# -*- coding: utf-8 -*-
"""
landmarkGLM/evaluate_labels.py

Score the pipeline's automated labels against the hand labels, and retune the
decision floors that produce them.

DMM, Sep 2026
"""

import argparse
import os

import numpy as np
import pandas as pd

UNLABELED = -99

HAND_TO_NAME = {-1: "four equal peaks", 0: "four equal peaks",
                1: "L1 preference", 2: "L2 preference", 3: "L3 preference",
                4: "L4 preference", 5: "onset only", 6: "reward only"}

CATS = ["four equal peaks", "L1 preference", "L2 preference", "L3 preference",
        "L4 preference", "onset only", "reward only"]

SHORT = {"four equal peaks": "equal", "L1 preference": "L1",
         "L2 preference": "L2", "L3 preference": "L3", "L4 preference": "L4",
         "onset only": "onset", "reward only": "reward",
         "unlabeled": "unlabeled"}


def load_pair(csv, labels, cellids=None):

    D = pd.read_csv(csv)
    hand = np.load(labels, allow_pickle=True)
    if cellids is None:
        cellids = os.path.splitext(labels)[0] + "_cellids.npy"

    if os.path.exists(cellids):
        ids = np.load(cellids, allow_pickle=True)

        _pos = {int(v): k for k, v in enumerate(np.asarray(ids))}
        _want = [int(v) for v in np.asarray(D.suite2p_cell_id)]
        _miss = [v for v in _want if v not in _pos]
        if _miss:
            raise ValueError(
                "{} has {} cell(s) absent from {} (first few: {}). Relabel or "
                "refit; do not score these against each other."
                .format(csv, len(_miss), cellids, _miss[:8]))
        hand = hand[[_pos[v] for v in _want]]
        if len(_want) != len(ids):
            print("  {} rows in {}, scored against the matching {} of {} hand "
                  "labels.".format(len(_want), csv, len(_want), len(ids)))
    else:
        print("WARNING: no cell-id sidecar at {} -- assuming row order "
              "matches.".format(cellids))
        if len(D) != len(hand):
            raise ValueError("{} rows in {}, {} hand labels.".format(
                len(D), csv, len(hand)))

    return D, hand


def score(pred, hand, title):

    m = hand != UNLABELED
    h = np.array([HAND_TO_NAME[int(v)] for v in hand[m]], dtype=object)
    p = np.asarray(pred, dtype=object)[m]

    print("\n{}".format(title))
    print("  n hand-labeled           {}".format(int(m.sum())))
    print("  of those, model abstains  {}".format(int((p == "unlabeled").sum())))

    acc_all = float(np.mean(p == h))
    named = p != "unlabeled"
    acc_named = float(np.mean(p[named] == h[named])) if named.any() else np.nan

    base = float(np.mean(h == "four equal peaks"))
    print("  accuracy, abstention wrong  {:.3f}".format(acc_all))
    print("  accuracy, model-named only  {:.3f}   (n={})".format(
        acc_named, int(named.sum())))
    print("  constant 'four equal peaks' {:.3f}   <- the number to beat".format(base))

    print("\n  CONFUSION  rows = model, cols = hand")
    print("    {:9s}".format("") + "".join(
        "{:>8s}".format(SHORT[c]) for c in CATS) + "{:>8s}".format("n"))
    for c in CATS + ["unlabeled"]:
        row = [int(np.sum((p == c) & (h == k))) for k in CATS]
        if not sum(row):
            continue
        print("    {:9s}".format(SHORT[c]) + "".join(
            "{:8d}".format(v) for v in row) + "{:8d}".format(sum(row)))

    print("\n  PER CLASS")
    print("    {:9s} {:>5s} {:>8s} {:>10s}".format(
        "hand", "n", "recall", "precision"))
    for c in CATS:
        y = h == c
        if not y.any():
            continue
        pr = p == c
        print("    {:9s} {:5d} {:8.2f} {:10.2f}".format(
            SHORT[c], int(y.sum()), float(np.mean(p[y] == c)),
            float(np.mean(h[pr] == c)) if pr.any() else np.nan))

    lm = np.isin(h, CATS[:5]) & np.isin(p, CATS[:5])
    hp = h[lm] != "four equal peaks"
    pp = p[lm] != "four equal peaks"
    ex = float(np.mean(hp == pp)) if lm.any() else np.nan
    both = lm.copy()
    both[lm] = hp & pp
    idn = float(np.mean(p[both] == h[both])) if both.any() else np.nan
    print("\n  existence (preference vs all-equal)  {:.3f}  (n={})".format(
        ex, int(lm.sum())))
    print("  identity, where both say preference  {:.3f}  (n={})".format(
        idn, int(both.sum())))

    return dict(n=int(m.sum()), acc=acc_all, acc_named=acc_named,
                baseline=base, existence=ex, identity=idn)


def sweep_margin(D, hand, stat="margin", seed=0, n_folds=5):

    m = hand != UNLABELED
    h = np.array([HAND_TO_NAME[int(v)] for v in hand[m]], dtype=object)
    v = np.nan_to_num(np.asarray(D[stat], float)[m], nan=0.0)

    ag = np.asarray(D.argmax_gain, int)[m]
    ident = np.array([CATS[j] if 1 <= j <= 4 else "four equal peaks"
                      for j in ag], dtype=object)
    end = np.zeros(len(h), dtype=object)
    end[:] = ""
    if "is_reward" in D.columns:
        end[np.asarray(D.is_reward, bool)[m]] = "reward only"
        end[np.asarray(D.is_onset, bool)[m]] = "onset only"

    def _pred(thr, sel=None):
        p = np.where(v > thr, ident, "four equal peaks").astype(object)
        p = np.where(end != "", end, p)
        return p if sel is None else p[sel]

    grid = np.percentile(v, np.arange(0, 100, 2))

    print("\nMARGIN SWEEP on '{}' (identity fixed at argmax_gain)".format(stat))
    print("  {:>10s} {:>10s} {:>10s}".format("threshold", "acc", "fires on"))
    best = (-1.0, None)
    for t in grid:
        a = float(np.mean(_pred(t) == h))
        if a > best[0]:
            best = (a, t)
    for t in np.unique(np.round(np.percentile(v, [10, 25, 40, 50, 60, 75, 90]), 4)):
        print("  {:10.3f} {:10.3f} {:10.0%}".format(
            t, float(np.mean(_pred(t) == h)), float(np.mean(v > t))))
    print("  best in-sample: acc {:.3f} at {:.3f}".format(best[0], best[1]))

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(h))
    accs, thrs = [], []
    for f in np.array_split(idx, n_folds):
        tr = np.setdiff1d(idx, f)
        bt = max(((float(np.mean(_pred(t, tr) == h[tr])), t) for t in grid))[1]
        thrs.append(bt)
        accs.append(float(np.mean(_pred(bt, f) == h[f])))
    print("  {}-fold CV:     acc {:.3f} +/- {:.3f}, thresholds {}".format(
        n_folds, float(np.mean(accs)), float(np.std(accs)),
        np.round(thrs, 3).tolist()))


def main():

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--csv", default="v09_per_cell_full.csv")
    ap.add_argument("--labels", default="hand_labels.npy")
    ap.add_argument("--cellids", default=None)
    ap.add_argument("--baseline-csv", default=None,
                    help="a second table to score alongside, e.g. the run "
                         "before a change")
    ap.add_argument("--sweep", action="store_true",
                    help="retune the margin floor against these labels")
    args = ap.parse_args()

    D, hand = load_pair(args.csv, args.labels, args.cellids)

    col = "label" if "label" in D.columns else "template"
    score(np.asarray(D[col], dtype=object), hand,
          "{}  [{}]".format(args.csv, col))
    if col != "template":
        score(np.asarray(D.template, dtype=object), hand,
              "{}  [template, shape categories only]".format(args.csv))
    score(np.asarray(D.template_shape_search, dtype=object), hand,
          "{}  [template_shape_search, no arbitration]".format(args.csv))

    if args.baseline_csv:
        B, hb = load_pair(args.baseline_csv, args.labels, args.cellids)
        bcol = "label" if "label" in B.columns else "template"
        score(np.asarray(B[bcol], dtype=object), hb,
              "{}  [{}]".format(args.baseline_csv, bcol))

    if args.sweep:
        sweep_margin(D, hand)


if __name__ == "__main__":

    main()
