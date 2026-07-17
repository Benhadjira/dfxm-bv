# DFXM-BV

This repository contains code and notebooks to reproduce the Burgers-vector
classification results for data-driven and crystallographic-constrained neural
networks, plus experimental DFXM/EDF preprocessing and visualization workflows.

## Repository Layout

```text
bv_reproducibility_repo/
  src/bv_repro/                 Reusable Python package
    models.py                   ResNet/data-driven and crystallographic models
    data.py                     NPY/EDF parsing and data-loading helpers
    preprocessing.py            ROI, resizing, normalization, weak-beam helpers
    training.py                 Losses, metrics, training loops, benchmark runners
    visualization.py            Plotting helpers and interactive viewers
  notebooks/01_training/        Model training, benchmark, uncertainty plots
  notebooks/02_experimental/    Experimental EDF/ROI workflows and prediction
  notebooks/03_visualization/   3D dislocation and scale-bar visualizations
  data/                         Data instructions and local symlinks
  outputs/                      Generated artifacts, ignored by Git
  scripts/                      Utility scripts
```

## Installation

From the repository root:

```bash
python -m pip install -e .
python -m pip install -r requirements.txt
```

At ESRF, use the Python environment that already provides `fabio`, PyTorch, and
the scientific stack if available.

## Data

Large datasets are not included in this repository. The notebooks expect paths
such as:

```text
/data/projects/engage_id03/int_fcc_002/
/data/projects/engage_id03/int_fcc_111/
/data/projects/engage_id03/dfxm-go-ml/ex_data/
```

You can either edit the paths in the notebooks or create local symlinks inside
`data/`.

## Main Workflows

### Test imports

- `notebooks/00_tests/module_import_smoke_test.ipynb`
  verifies that the package modules import and that dummy model/preprocessing
  calls run correctly.

### Synthetic/model benchmarks

- `notebooks/01_training/crystallographic_resnet_models.ipynb`
  trains the data-driven and crystallographic-constrained models.
- `notebooks/01_training/crystallographic_resnet_visualize.ipynb`
  visualizes training curves, uncertainty, and benchmark metrics.
- `notebooks/01_training/cc_reflection_training_cases.ipynb`
  benchmarks only the crystallographic-constrained model on FCC 002, FCC 111,
  and combined reflection datasets.

### Experimental workflows

- `notebooks/02_experimental/experimental_rocking_curve_with_insets.ipynb`
  plots experimental EDF rocking curves with image insets sorted by EDF `diffry`.
- `notebooks/02_experimental/fixed_roi_scalebar_and_integration.ipynb`
  supports center-click ROI extraction and scale-bar plots.
- `notebooks/02_experimental/predict_burgers_experimental_111.ipynb`
  predicts Burgers vectors on extracted experimental images.

## Reproducibility Notes

- Generated model checkpoints and plots are written to output/artifact folders
  and are ignored by Git.
- For publication figures, rerun the notebooks after confirming dataset paths,
  ROI coordinates, pixel size, and rocking-angle motor names.
- The EDF rocking-curve notebook uses `diffry` as the rocking motor.

