# Figure 6 analysis provenance

This directory holds computations that produce the frozen tables consumed by the plotting notebook. It is not imported by the root plotting workflow.

## Current main Figure 6c: pooled score comparison

Use `derive_pooled_score_concordance.py --meia INPUT.parquet --direct INPUT.parquet --output-dir ../data` in the existing bgnmf environment. The inputs are the separately compositional 356-feature MEIA matrix and its direct-scKNIFE counterpart. This script aligns exact identifiers, checks finite nonnegative values and per-cell block sums, and computes both coefficients over all matched observations. It does not use the older contrast-correlation derivation. Existing entity-wise and within-cell profile summaries remain separate supporting analyses.
