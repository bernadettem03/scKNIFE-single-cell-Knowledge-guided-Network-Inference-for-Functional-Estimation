"""Shared, plotting-only Figure 6c renderer (matplotlib fallback; no cnsplots)."""
from __future__ import annotations

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import MaxNLocator, ScalarFormatter

# Single source of truth for the two score-density plots.
SUMMARY_FILE = "meia_vs_scknife_pooled_correlations.csv"
DENSITY_FILE = "meia_vs_scknife_pooled_density.csv.gz"
POOLED_INPUTS = [SUMMARY_FILE, DENSITY_FILE]
EXPECTED = {"Task": (218, 15127020, .8980780951601036, .9169521421174492),
            "Pathway": (138, 9575820, .893407454253489, .9684032386517093)}
FONT = {"title": 9, "subtitle": 7, "feature": 8, "stat": 7,
        "axis": 8, "tick": 7, "colorbar": 6.5}
COMPACT_FONT = {"title": 6.8, "subtitle": 5.5, "feature": 6, "stat": 5.2,
                "axis": 6, "tick": 5.2, "colorbar": 5}
COLOR_STOPS = ["#E5F5F9", "#56B4E9", "#0072B2", "#002F4B"]
COLOR_PERCENTILE = 99.5  # Color saturation only; never axis/observation trimming.
GRID = dict(left=.16, right=.98, bottom=.31, top=.65, wspace=.48)
COLORBAR_RECT = [.33, .093, .40, .025]
FIT_COLOR = "#303030"
FIT_LINEWIDTH = 1.0
FIT_COMPACT_LINEWIDTH = .7
FIT_LINESTYLE = ":"


def validate_pooled(data):
    summary = data[SUMMARY_FILE].set_index("resolution")
    density = data[DENSITY_FILE]
    if not summary.index.is_unique or set(summary.index) != set(EXPECTED):
        raise ValueError("Figure 6c requires separate Task and Pathway summaries")
    if set(density.resolution) != set(EXPECTED):
        raise ValueError("Figure 6c density layers changed")
    grids = {}
    for resolution, (features, pairs, r, rho) in EXPECTED.items():
        row = summary.loc[resolution]
        if (int(row.n_cells), int(row.n_features), int(row.n_pairs)) != (69390, features, pairs):
            raise ValueError(f"{resolution} dimensions changed")
        if not np.allclose([row.pearson, row.spearman], [r, rho], atol=1e-10, rtol=0):
            raise ValueError(f"{resolution} pooled correlations changed")
        if not np.isfinite([row.ols_slope, row.ols_intercept]).all():
            raise ValueError(f"{resolution} requires finite frozen OLS coefficients")
        frame = density.loc[density.resolution.eq(resolution)]
        bins = int(row.bins_per_axis)
        if len(frame) != bins**2 or frame.duplicated(["x_bin", "y_bin"]).any():
            raise ValueError("Incomplete or duplicated density bins")
        for col in ("x_bin", "y_bin"):
            if set(frame[col]) != set(range(bins)):
                raise ValueError("Density bin indices changed")
        counts = frame.pivot(index="y_bin", columns="x_bin", values="count").to_numpy()
        if (not np.isfinite(counts).all() or (counts < 0).any()
                or not np.equal(counts, counts.astype(np.int64)).all()
                or int(counts.sum()) != pairs):
            raise ValueError("Density counts must include every matched score pair")
        if row.axis_min != 0 or row.x_axis_max < row.meia_max or row.y_axis_max < row.scknife_max:
            raise ValueError("Axis limits must retain the full score range")
        grids[resolution] = counts
    return summary, grids


def draw_pooled_concordance(container, data, label="C", compact=False):
    summary, grids = validate_pooled(data)
    font = COMPACT_FONT if compact else FONT
    positive = np.concatenate([x[x > 0] for x in grids.values()])
    vmax = max(10.0, float(10 ** np.ceil(np.log10(np.percentile(positive, COLOR_PERCENTILE)))))
    norm = LogNorm(vmin=1, vmax=vmax, clip=True)
    cmap = LinearSegmentedColormap.from_list("pooled_activity_density", COLOR_STOPS)
    cmap.set_bad("white")
    grid = container.add_gridspec(1, 2, **GRID)
    axes = []
    for index, resolution in enumerate(("Task", "Pathway")):
        row = summary.loc[resolution]
        ax = container.add_subplot(grid[0, index])
        axes.append(ax)
        x_edges = np.linspace(row.axis_min, row.x_axis_max, int(row.bins_per_axis) + 1)
        y_edges = np.linspace(row.axis_min, row.y_axis_max, int(row.bins_per_axis) + 1)
        mesh = ax.pcolormesh(x_edges, y_edges, np.ma.masked_equal(grids[resolution], 0),
                             cmap=cmap, norm=norm, shading="flat", rasterized=True,
                             edgecolors="none", linewidth=0)
        mesh.set_gid("pooled_full_data_density")
        fit_x = np.array([row.meia_min, row.meia_max])
        fit_y = row.ols_intercept + row.ols_slope * fit_x
        line, = ax.plot(fit_x, fit_y, color=FIT_COLOR, linestyle=FIT_LINESTYLE,
                        linewidth=FIT_COMPACT_LINEWIDTH if compact else FIT_LINEWIDTH,
                        zorder=3, rasterized=False, clip_on=True)
        line.set_gid("pooled_ols_fit")
        ax.set(xlim=(row.axis_min, row.x_axis_max), ylim=(row.axis_min, row.y_axis_max))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)
        ax.tick_params(labelsize=font["tick"], width=.6, length=2, pad=1)
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(MaxNLocator(nbins=2, min_n_ticks=2))
            formatter = ScalarFormatter(useOffset=False)
            formatter.set_scientific(False)
            axis.set_major_formatter(formatter)
        box = ax.get_position()
        cx = (box.x0 + box.x1) / 2
        if compact:
            cx = min(cx, .785)
        feature_label = f"{resolution}s\nn={int(row.n_features)}" if compact else f"{resolution}s (n={int(row.n_features)})"
        container.text(cx, .855, feature_label,
                       ha="center", va="center", fontsize=font["feature"], fontweight="bold")
        container.text(cx, .755, f"Pearson r = {row.pearson:.4f}\nSpearman ρ = {row.spearman:.4f}",
                       ha="center", va="center", fontsize=font["stat"], linespacing=1.4)
    heading = f"{label}  Pooled score concordance" if label else "Pooled score concordance"
    container.text(.015, .985, heading, ha="left", va="top", fontsize=font["title"], fontweight="bold")
    container.text(.57, .92, "69,390 matched cells • all score pairs", ha="center", va="center",
                   fontsize=font["subtitle"], color="#555555")
    container.text(.025, .48, "scKNIFE activity score", rotation=90, ha="center", va="center", fontsize=font["axis"])
    container.text(.57, .21, "MEIA activity score", ha="center", va="center", fontsize=font["axis"])
    cax = container.add_axes(COLORBAR_RECT)
    colorbar = container.colorbar(mesh, cax=cax, orientation="horizontal",
                                  extend="max" if positive.max() > vmax else "neither")
    ticks = [10**p for p in range(0, int(np.log10(vmax)) + 1, 2)]
    colorbar.set_ticks(ticks)
    colorbar.ax.tick_params(labelsize=font["colorbar"], length=2, pad=1)
    colorbar.ax.minorticks_off()
    colorbar.outline.set_linewidth(.4)
    container.text(.57, .005, "Cell–feature pairs per bin (log colour scale)",
                   ha="center", va="bottom", fontsize=font["colorbar"])
    return {"axes": axes, "color_vmin": 1, "color_vmax": vmax,
            "density_counts": {key: int(value.sum()) for key, value in grids.items()}}
