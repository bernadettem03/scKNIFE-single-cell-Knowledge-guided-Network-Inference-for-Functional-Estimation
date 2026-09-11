# scKNIFE manuscript reproducibility repository

This repository contains code and plot-ready data for reproducing Figures 2–7 of the scKNIFE manuscript. Workflows are organized by figure so they can be reviewed and run independently.

## Layout

- `Figure_2/`–`Figure_7/`: main figures, analysis provenance, inputs, and outputs.
- `Supplementary/Figure_S1/`–`Figure_S6/`: supplementary figure workflows.
- `data_manifest.tsv`: checksummed public-file inventory.
- `validate_repository.py`: portability and submission checks.

Root figure notebooks are plotting-only. Statistical inference, clustering, feature selection, and derived-table construction belong under each figure's `analysis/` directory.

## Environment

```bash
conda env create -f environment.yml
conda activate tnbc-figures-py39
python validate_repository.py
```

Large editable artwork and data unsuitable for Git are retained in a private, unpublished archive. No public Zenodo DOI is available. A code/data license and final manuscript citation must be approved before public release.


## Private unpublished data bundle

The repository does not publish or create a Zenodo record. Assets marked `private_unpublished` in `data_manifest.tsv` are inventoried in `private_archive_manifest.tsv`. After receiving the private archive from the authors, restore and validate it with:

```bash
python download_data.py --archive-dir /path/to/extracted/private_bundle
python validate_repository.py
```

Do not commit credentials, private links, or access tokens.

## Current migration status

All main and supplementary notebooks generate their component panels live from validated plot-ready inputs, display those panels inline, and compile the final figure last. Executed outputs are retained so GitHub reviewers can inspect the figures without first running the notebooks. Full analytical derivations remain under each figure’s `analysis/` directory.
