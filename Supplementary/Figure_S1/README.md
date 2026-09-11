# Supplementary Figure 1

Supplementary Figure 1 contains cross-dataset W concordance and pooled W-space structure. The notebook consumes frozen, checksummed inputs and performs no embedding or statistical inference.

## Caption

**Supplementary Figure 1 | Cross-dataset W concordance and pooled W-space structure.** **A,** Cross-dataset linear CKA with within-dataset factor diagnostics. **B,** t-SNE node kinds across independently normalized W blocks. **C,** t-SNE pooled clusters across independently normalized W blocks.

- `Figure_S1.ipynb`: plotting-only entrypoint.
- `plotting.py`: validation and A4 rendering.
- `data/pooled_w/`: frozen concordance and t-SNE inputs.
- `outputs/`: compiled PDF and 300-DPI PNG preview.
