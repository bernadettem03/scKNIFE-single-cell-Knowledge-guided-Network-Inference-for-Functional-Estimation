# Migration report — 2026-09-10

## Scope and safety

- Read-only source: `PRIVATE_ARCHIVE_ROOT/source_snapshot/Figures/submission` (not modified).
- Destination repository: `PRIVATE_ARCHIVE_ROOT/destination_repository/manuscript`.
- Legacy destination snapshot moved, not deleted, to `PRIVATE_ARCHIVE_ROOT/private_archive_20260910/legacy_snapshot/submission`.
- Private unpublished data bundle staged at `PRIVATE_ARCHIVE_ROOT/private_archive_20260910/zenodo_bundle`.
- No Zenodo record was created or published. The initial migration was prepared locally before the clean GitHub history replacement.

## Copied and transformed

- Canonical Figure 2–7 inputs and approved PDF/PNG outputs were copied into figure-centric directories.
- Figure 3 version 2 and Figure 6 blank-corrected were treated as authoritative.
- Main and supplementary workflows were separated into Figure S2–S6 directories. No empty Figure S7 directory was created.
- Root notebooks generate every component panel live from finalized tables or coordinates, display panels inline, and compile the final figure last. Detailed analytical workflows remain under `analysis/`.
- Repository-relative paths, checksum validation, environment metadata, and private-archive restoration tooling were added.

## Private archive

- `Figure_4/data/fig4_plot_data.h5` was moved to the private unpublished bundle and is recoverable with `download_data.py`.
- Editable SVGs, TIFFs, staging material, historical versions, logs, caches, builder/fixer utilities, and redundant manifests remain only in the recoverable legacy archive and are excluded from the clean tree.

## Validation

- Eleven public root notebooks are present (Figures 2–7 and S2–S6).
- Each was executed independently from its figure directory.
- Root notebooks and plotting modules were scanned for inference, fitting, clustering, stale source names, and absolute storage paths. Executed inline outputs are intentionally retained.
- All eleven reviewer notebooks were executed successfully with live inline panel output and compiled figure output.
- `validate_repository.py` verifies checksums, portability, plotting-only boundaries, notebook cleanliness, and GitHub's 100 MB file ceiling.

## Deliberately pending

- `LICENSE` and `CITATION.cff` are not fabricated; author/license/citation approval is required before public release.
- Repository is prepared for review only; commit, push, DOI minting, and public release require explicit authorization.

## Approved-render repair (2026-09-11)

- Replaced the incorrect simplified Figure 5 renderer with the checksum-locked approved A-C-E-G-H composite and canonical standalone panels.
- Replaced Supplementary Figure 5 with its approved B-D-E-F-G composite; retained A and C as standalone audit panels.
- Restored the authoritative Figure 3 S3B Bassez metabolic-task/GO heatmap page.
- Re-executed all three plotting-only notebooks with their figures embedded inline.
- Removed the four approximate S3 heatmap matrices and their noncanonical generator.
- No files in the protected May submission or private archive were modified.

## Public repository rewrite preparation (2026-09-11)

- Public specimen labels were normalized to P1–P7 in text, notebooks, CSV/GZip, Parquet, and HDF5 data.
- Original specimen-ID mappings remain only in the recoverable private legacy archive.
- Local absolute paths were removed from notebook outputs and HDF5 provenance metadata.
- All 11 plotting notebooks were re-executed successfully with inline figures.
- The unpublished Figure 4 bundle was pseudonymized separately and remains private; no Zenodo record was published.

## Standalone main-panel exports (2026-09-11)

- Added 300-DPI PNG and vector PDF exports for every authoritative main panel: Figure 2 A–H, Figure 3 A–J, Figure 4 B–F, Figure 5 A/C/E/G/H, Figure 6 A–F, and Figure 7 A–J.
- Added per-figure JSON manifests recording panel order, relative paths, byte sizes, and SHA-256 checksums.
- Retained the Figure 3 A–D compound export for layout provenance while also providing A, B, C, and D separately.
- Removed obsolete Figure 5 B/D/F standalone exports because those panels are not part of the authoritative main Figure 5 composite.
- Extended repository validation to require all expected standalone outputs and verify their formats and manifest checksums.
- The protected May submission source and unpublished private archive were not modified.
