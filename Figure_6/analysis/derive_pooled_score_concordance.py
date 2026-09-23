#!/usr/bin/env python3
"""Derive Figure 6c from ALL matched cell–feature pairs, without model refitting.

Run in the existing bgnmf environment. Input matrices are already separately
compositional for the 218-task and 138-pathway blocks; do not normalize again.
The public plotting notebook needs only the compact outputs, not the matrices.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import pearsonr, spearmanr, linregress

N_CELLS = 69390
N_BINS = 160
SUMMARY_FILE = "meia_vs_scknife_pooled_correlations.csv"
DENSITY_FILE = "meia_vs_scknife_pooled_density.csv.gz"
PROVENANCE_FILE = "meia_vs_scknife_pooled_provenance.json"
BLOCKS = (
    ("Task", "scCellFie::", 218, 0.8980780951601036, 0.9169521421174492),
    ("Pathway", "HumanGEM::", 138, 0.893407454253489, 0.9684032386517093),
)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fit_full_scores(x, y):
    """Descriptive OLS using all unbinned scores; independently verify coefficients."""
    x_mean, y_mean = float(np.mean(x)), float(np.mean(y))
    dx, dy = x - x_mean, y - y_mean
    denominator = float(np.dot(dx, dx))
    if denominator <= 0:
        raise ValueError("OLS requires nonconstant MEIA scores")
    slope = float(np.dot(dx, dy) / denominator)
    intercept = float(y_mean - slope * x_mean)
    del dx, dy
    reference = linregress(x, y)
    if not np.allclose([slope, intercept], [reference.slope, reference.intercept], rtol=1e-10, atol=1e-12):
        raise ValueError("Independent OLS coefficient check failed")
    return {"ols_slope": slope, "ols_intercept": intercept}


def derive(meia_path, direct_path, output_dir):
    meia = pd.read_parquet(meia_path)
    direct = pd.read_parquet(direct_path)
    for frame in (meia, direct):
        if frame.shape != (N_CELLS, 356):
            raise ValueError(f"Unexpected matrix shape: {frame.shape}")
        if not frame.index.is_unique or not frame.columns.is_unique:
            raise ValueError("Cell/feature identifiers must be unique")
    if set(meia.index) != set(direct.index) or set(meia.columns) != set(direct.columns):
        raise ValueError("Matrices must contain exactly the same cells and features")
    direct = direct.loc[meia.index, meia.columns]
    summaries, histograms, block_checks = [], [], {}
    for resolution, prefix, n_features, expected_r, expected_rho in BLOCKS:
        columns = [c for c in meia.columns if c.startswith(prefix)]
        if len(columns) != n_features:
            raise ValueError(f"{resolution}: expected {n_features} features")
        x = meia[columns].to_numpy(dtype=np.float64)
        y = direct[columns].to_numpy(dtype=np.float64)
        errors = []
        for values in (x, y):
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError("Scores must be finite and nonnegative")
            error = float(np.max(np.abs(values.sum(axis=1) - 1)))
            if error > 2e-6:
                raise ValueError(f"{resolution} is not separately compositional: {error}")
            errors.append(error)
        xv, yv = x.ravel(), y.ravel()
        r = float(pearsonr(xv, yv)[0])
        rho = float(spearmanr(xv, yv)[0])
        if not np.allclose([r, rho], [expected_r, expected_rho], atol=1e-10, rtol=0):
            raise ValueError(f"{resolution} differs from approved coefficients: {r}, {rho}")
        # Each linear axis spans its full observed range, with no trimming.
        x_max, y_max = float(xv.max()), float(yv.max())
        x_edges = np.linspace(0.0, x_max, N_BINS + 1)
        y_edges = np.linspace(0.0, y_max, N_BINS + 1)
        counts, _, _ = np.histogram2d(xv, yv, bins=(x_edges, y_edges))
        counts = counts.astype(np.int64)
        n_pairs = N_CELLS * n_features
        if int(counts.sum()) != n_pairs:
            raise ValueError("Density bins lost observations")
        ix, iy = np.indices(counts.shape)
        histograms.append(pd.DataFrame({
            "resolution": resolution, "x_bin": ix.ravel(), "y_bin": iy.ravel(),
            "count": counts.ravel(),
        }))
        summaries.append({
            "resolution": resolution,
            "comparison": f"MEIA-{resolution.lower()} vs scKNIFE-{resolution.lower()}",
            "n_features": n_features, "n_cells": N_CELLS, "n_pairs": n_pairs,
            "pearson": r, "spearman": rho, "bins_per_axis": N_BINS,
            "axis_min": 0.0, "x_axis_max": x_max, "y_axis_max": y_max,
            "meia_min": float(xv.min()), "meia_max": float(xv.max()),
            "scknife_min": float(yv.min()), "scknife_max": float(yv.max()),
            **fit_full_scores(xv, yv),
        })
        block_checks[resolution] = {
            "feature_order": columns, "max_abs_block_sum_error_meia": errors[0],
            "max_abs_block_sum_error_direct": errors[1],
            "finite_nonnegative": True, "all_pairs_binned": True,
        }
        print(f"{resolution}: {n_pairs:,} pairs; Pearson={r:.16f}; Spearman={rho:.16f}", flush=True)
        del x, y, xv, yv
        gc.collect()
    summary = pd.DataFrame(summaries)
    density = pd.concat(histograms, ignore_index=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / SUMMARY_FILE, index=False, float_format="%.17g")
    density.to_csv(output_dir / DENSITY_FILE, index=False,
                   compression={"method": "gzip", "mtime": 0})
    provenance = {
        "analysis": "Pooled matched cell–feature score concordance; two separate blocks",
        "x": "MEIA activity score", "y": "Direct scKNIFE activity score",
        "normalization": "Existing separate task/pathway per-cell compositional normalization; no additional transformation",
        "alignment": "Unique exact cell/feature sets; direct matrix reordered to MEIA identifiers",
        "cell_order_sha256": hashlib.sha256("\n".join(map(str, meia.index)).encode()).hexdigest(),
        "n_matched_cells": N_CELLS,
        "pearson": "scipy.stats.pearsonr on flattened float64 cell-by-feature values",
        "spearman": "scipy.stats.spearmanr on the same full values; average ranks for ties",
        "linear_fit": "OLS with intercept, scKNIFE on MEIA, all unbinned pairs per block; independently verified with scipy.stats.linregress; descriptive only, no P value or confidence band",
        "density": {"bins_per_axis": N_BINS, "axis_scale": "linear", "sampling": "none",
                    "trimming": "none", "binning": "numpy.histogram2d; each axis spans its full observed range"},
        "interpretation": "Descriptive pooled association includes between-feature differences; not the typical feature-wise correlation, independent validation, or measured flux. No cell-level inferential P values.",
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__},
        "sources": [{"role": role, "file": Path(path).name, "bytes": Path(path).stat().st_size,
                     "sha256": sha256(path)} for role, path in (("MEIA", meia_path), ("direct scKNIFE", direct_path))],
        "checks": block_checks,
        "outputs": [{"path": name, "sha256": sha256(output_dir / name)}
                    for name in (SUMMARY_FILE, DENSITY_FILE)],
    }
    (output_dir / PROVENANCE_FILE).write_text(json.dumps(provenance, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meia", required=True, type=Path)
    parser.add_argument("--direct", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    derive(args.meia, args.direct, args.output_dir)
