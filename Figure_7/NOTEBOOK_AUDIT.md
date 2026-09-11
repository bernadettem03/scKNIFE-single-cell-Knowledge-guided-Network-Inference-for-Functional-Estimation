# Figure 7 notebook audit

## Public plotting contract

`Figure_7.ipynb` is plotting-only. It loads frozen signaling, concordance, VEGF, and H-factor tables, renders panels A–J through `plotting.py`, and exports standalone and compiled PNG/PDF outputs.

## Analysis boundary

The root workflow performs no fitting, clustering, statistical inference, multiple-testing correction, feature selection, or derived-data writes. Upstream analytical provenance is retained under `analysis/`.

## Supplementary and exclusions

No empty Figure S7 directory is created because no authoritative supplementary Figure S7 exists. Repetitive historical export cells, caches, logs, staging copies, and absolute paths are excluded.

## Acceptance boundary

Panels A–J must render inline from frozen inputs, preserve the approved labels and values, export as PNG/PDF through the shared loop, and pass repository checksum and plotting-only validation.
