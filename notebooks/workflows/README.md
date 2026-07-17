# Main Workflow Notebooks

These are the clean entry-point notebooks for reproducing the main results.

1. `01_compare_data_driven_vs_crystallographic.ipynb`
   compares the data-driven and crystallographic-constrained models using saved benchmark artifacts.

2. `02_uncertainty_entropy_mutual_information.ipynb`
   plots predictive entropy and mutual information for uncertainty analysis.

3. `03_experimental_visualization_and_cropping.ipynb`
   visualizes experimental rocking-curve data, selects/crops ROIs, rescales to model input size, and saves extracted images.

4. `04_test_models_on_experimental_data.ipynb`
   evaluates trained models on extracted experimental images and plots probabilities/confusion matrices.

Recommended order:

```text
train models -> compare models -> uncertainty -> crop experimental data -> test models on experimental data
```

