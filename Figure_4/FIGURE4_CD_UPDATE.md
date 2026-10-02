# Figure 4 C/D-only update

Subsequent E update: panel E now uses the exact panel C UMAP coordinates after an ordered cell-ID checksum check. CNV scores, normal/tumor status, percentile colour limits and the violin plot are unchanged. C/D/E now share the final embedding where applicable; B and supplementary results remain unchanged. No CNV inference was rerun.

C now uses final cell-type/state labels and exact saved H-based coordinates for 69,391 cells (15 clusters; 41,625 epithelial cells). Approved red–orange and non-grey lineage colours are preserved.

D is an all-cell heatmap, not a dot plot. The ten highest-scoring positive Wilcoxon genes per cluster satisfy adjusted P < 0.05 and log2FC > 0.5. Gene-wise z-scores use population SD and are clipped to −3 to 3, matching the existing heatmap convention. Cluster order is 0–14; duplicate genes remain in each selecting cluster's ten-row block. Metadata strips retain patient, site, metastatic status, library, cell type and cluster. The accompanying fig4_D_top10_markers.csv gives all 150 selections and statistics.

The plot-only figure4_cd.py reads the portable fig4_cd_final.h5 bundle and validates its checksum against fig4_cd_manifest.json. Both copies contain identical C/D bundles. Original fig4_plot_data.h5 is unchanged and remains the source for B/E. This staged main figure intentionally still contains their earlier results; supplementary panels are also unchanged.

No clustering, DE, CNV, Hallmark or UMAP analysis was rerun. Expression was subset by exact cell IDs and selected genes solely to assemble the heatmap. No Git push was performed. Existing matplotlib rendering is retained under the TNBC figures skill; no environment/package changes.

Panel F was removed from active notebooks and main exports; its underlying data remain retained. B–E positions are unchanged.

Panel D now includes column-cluster IDs 0–14, an enlarged marker-source cluster key, and a full cluster-to-final-cell-type/state legend below the unchanged heatmap. No DE, cell ordering, expression values or annotation assignments changed.

New F: six approved broad-identity marker-expression UMAPs in a 2 × 3 grid; the old F heatmap remains removed. Frozen log1p_norm expression, cell-ID aligned to exact C coordinates, gene-specific positive-cell P98 scales. B–E unchanged.
