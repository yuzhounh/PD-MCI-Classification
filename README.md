# PD-MCI-Classification: Machine Learning to Predict Mild Cognitive Impairment in Parkinson's Disease

Python analysis scripts for subject-level stratified classification of mild cognitive impairment in Parkinson’s disease using PPMI data.

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-D4AF37?style=flat-square)](LICENSE)

[![Paper](https://img.shields.io/badge/Paper-Frontiers%20in%20Aging%20Neuroscience-blue)](https://doi.org/10.3389/fnagi.2025.1687925)

> **📄 This repository contains the code for the following paper:**
>
> Wang, J., Chen, Y., Xie, X., Wang, P., Hu, H., Han, H., Wang, L., & Zhang, L. (2025). Diagnostic classification of mild cognitive impairment in Parkinson's disease using subject-level stratified machine-learning analysis. *Frontiers in Aging Neuroscience*, 17, 1687925.

This project uses machine learning models to predict the presence of Mild Cognitive Impairment (MCI) in patients with Parkinson's Disease (PD), based on the Parkinson's Progression Markers Initiative (PPMI) dataset.

The repository includes a complete pipeline for data preprocessing, statistical analysis, feature selection, model training, hyperparameter optimization, and results visualization.

## Key Features

-   **Data Processing**: A comprehensive workflow for filtering, cleaning, feature engineering, and label generation from the raw PPMI dataset.
-   **Feature Selection**: Utilizes LASSO Logistic Regression to identify the most relevant clinical markers associated with PD-MCI.
-   **Model Comparison**: Trains and evaluates four mainstream machine learning models: Logistic Regression (LR), Support Vector Machine (SVM), Random Forest (RF), and XGBoost.
-   **Automated Tuning**: Employs Optuna for Bayesian hyperparameter optimization to achieve the best performance for each model.
-   **Interpretability**: Analyzes models using SHAP (SHapley Additive exPlanations) and permutation importance to explain the contribution of each feature to the prediction outcomes.
-   **Rich Visualization**: Generates a suite of plots, including correlation heatmaps, ROC/PR curves, feature importance comparisons, and SHAP summary plots for clear interpretation and presentation of results.

## Dataset

This project relies on data from the [PPMI database](https://www.ppmi-info.org/access-data-specimens/download-data). The current checkout includes the following root-level files; verify that they are the intended data release before running the analysis:

1.  `PPMI_Curated_Data_Cut_Public_20250321.xlsx`: The original PPMI data file containing the clinical data.
2.  `PPMI_feature_mapping.csv`: A custom feature mapping file used to convert feature names into more readable abbreviations for plotting. This file should contain two columns: `Feature Name` and `Abbreviation`.

## How to Use

Install the pinned dependencies in an isolated Python 3.12 environment before running the scripts from the repository root:

```bash
python -m pip install -r requirements.txt
```

Please execute the Python scripts in the following order to reproduce the entire analysis pipeline.

1.  **`1_extract_data.py`**
    -   Reads data from the source Excel file.
    -   Filters for the Parkinson's Disease (PD) cohort.
    -   Extracts a predefined set of clinical features.
    -   Calculates disease duration (`duration`).
    -   Removes samples with missing values.
    -   Generates the MCI label (0: Normal Cognition, 1: Mild Cognitive Impairment) based on MoCA (Montreal Cognitive Assessment) scores.

2.  **`2_statistical_analysis.py`**
    -   Performs a statistical comparison of clinical features between the Normal Cognition (PD-NC) and MCI (PD-MCI) groups.
    -   Automatically selects the appropriate statistical test (e.g., t-test, Mann-Whitney U test, Chi-square test) based on variable type.
    -   Applies FDR (False Discovery Rate) correction to p-values.
    -   Generates a descriptive statistical analysis table.

3.  **`3_correlation_heatmap.py`**
    -   Calculates the Pearson correlation coefficients between all features.
    -   Generates and saves a feature correlation heatmap.

4.  **`4_lasso.py`**
    -   Creates a stratified train/test split over unique patient IDs using `train_test_split`, then maps those IDs back to records; `StratifiedGroupKFold` is used for subject-level cross-validation within training.
    -   Standardizes the data.
    -   Performs 10-fold cross-validation on the training set using LASSO Logistic Regression, optimizing the regularization parameter `lambda` based on AUC-PR (Area Under the Precision-Recall Curve).
    -   Selects features with non-zero coefficients at the optimal `lambda`.
    -   Saves the selected feature weights and generates diagnostic plots.
    -   Creates new, feature-selected training and testing sets.

5.  **Model Training and Evaluation**
    -   The following scripts are standalone and each trains and evaluates one machine learning model. Each script performs hyperparameter optimization, final model training, performance evaluation, and feature importance analysis.

    -   **`5_LR.py`**: Logistic Regression
    -   **`5_SVM.py`**: Support Vector Machine
    -   **`5_RF.py`**: Random Forest
    -   **`5_XGBoost.py`**: XGBoost

6.  **Results Aggregation and Visualization**
    -   These scripts aggregate and compare the results from all models.

    -   **`6_plot_ROC_PR.py`**
        -   Aggregates prediction results from all models on the test set.
        -   Plots the ROC (Receiver Operating Characteristic) and PR (Precision-Recall) curves for all four models on a single figure.

    -   **`6_plot_feature_importance.py`**
        -   Summarizes and compares feature importance scores from different models and methods (e.g., model coefficients, impurity importance, SHAP, permutation importance).
        -   Generates a 4x3 grid plot for a comprehensive overview.

    -   **`6_plot_SHAP_summary.py`**
        -   Merges the SHAP summary plots (in SVG format) generated by the four model scripts into a single 2x2 grid.
        -   Saves the combined figure for easy cross-model comparison of global feature effects.

## Supplementary Experiments

The supplementary experiments aim to validate the models' robustness and generalizability from multiple perspectives:

-   **Supplementary Experiment I**: Implements a more stringent site-level split validation strategy to simulate real-world model performance when deployed across different clinical centers.
-   **Supplementary Experiment II**: Evaluates the impact of the feature selection process itself by training models on all 12 original features without prior selection.
-   **Supplementary Experiment III**: Conducts an ablation study, assessing the performance of a more parsimonious model that uses only the top 5 predictive features.
-   **Supplementary Experiment IV**: Provides a systematic comparison of multiple feature selection methods—including Filter, Wrapper, and Embedded approaches—to confirm the stability and reliability of the predictors identified in the main experiment.

See `Supplementary_Material.pdf` for details.

## Repository Structure

- `1_extract_data.py` through `4_lasso.py`: preparation, statistics, and selection.
- `5_*.py` and `6_plot_*.py`: model experiments and visualizations.
- [utils.py](utils.py) and [requirements.txt](requirements.txt): shared functions and environment.
- `supplementary_experiment_1/` through `supplementary_experiment_4/`: additional analyses.
- [Supplementary_Material.pdf](Supplementary_Material.pdf): supplementary documentation.

## Citation

If you find this project useful for your research, please consider citing our paper:

```bibtex
@article{wang2025diagnostic,
  title={Diagnostic classification of mild cognitive impairment in Parkinson's disease using subject-level stratified machine-learning analysis},
  author={Wang, Jing and Chen, Yanfang and Xie, Xiao and Wang, Pengwei and Hu, Hang and Han, Hongfang and Wang, Lihan and Zhang, Li},
  journal={Frontiers in Aging Neuroscience},
  volume={17},
  pages={1687925},
  year={2025},
  publisher={Frontiers Media SA},
  doi={10.3389/fnagi.2025.1687925}
}
```

## License

See the existing [GPL-3.0 license](LICENSE).

## Contact

Jing Wang (wangjing@xynu.edu.cn)
