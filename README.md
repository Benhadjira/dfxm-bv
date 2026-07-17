# DFXM-BV

This repository contains code and notebooks to reproduce the Burgers-vector
classification results for data-driven and crystallographic-constrained neural
networks, plus experimental DFXM preprocessing visualization workflows.

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
## Data

Large datasets are not included in this repository. The notebooks expect paths
such as:

```text
/data/projects/engage_id03/int_fcc_002/
/data/projects/engage_id03/int_fcc_111/
/data/projects/engage_id03/dfxm-go-ml/ex_data/
```

]
### Test imports

- `notebooks/00_tests/module_import_smoke_test.ipynb`
  verifies that the package modules import and that dummy model/preprocessing
  calls run correctly.



## Reproducibility Notes

- Generated model checkpoints and plots are written to output/artifact folders
  and are ignored by Git.
- For publication figures, rerun the notebooks after confirming dataset paths,
  ROI coordinates, pixel size, and rocking-angle motor names.
- The EDF rocking-curve notebook uses `diffry` as the rocking motor.

## Funding Acknowledgement:

This project has been partly funded by the European Union’s Horizon 2020 Research and Innovation Programme under the Marie Sklodowska-Curie COFUND scheme with grant agreement No. 101034267 and the European Research Council (ERC) under the European Union’s Horizon 2020 Research and Innovation Programme (grant agreement No. 10116911) and the European Research Council
(ERC) Advanced Grant No. 885022, and by the European Spallation Source (ESS) Lighthouse on Hard Materials in
3D, SOLID (Danish Agency of Science and Higher Education, grant No. 8144-00002B). 


