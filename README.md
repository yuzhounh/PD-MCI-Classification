# PD-MCI-Classification: Machine Learning to Predict Mild Cognitive Impairment in Parkinson's Disease

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

This project relies on data from the [PPMI database](https://www.ppmi-info.org/access-data-specimens/download-data). To run this code, you must obtain the following files and place them in the project's root directory:

1.  `PPMI_Curated_Data_Cut_Public_20250321.xlsx`: The original PPMI data file containing the clinical data.
2.  `PPMI_feature_mapping.csv`: A custom feature mapping file used to convert feature names into more readable abbreviations for plotting. This file should contain two columns: `Feature Name` and `Abbreviation`.


## How to Use

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
    -   Splits the data into training and testing sets using `StratifiedGroupKFold` to ensure that all records from a single patient belong to only one set.
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

## Contact

Jing Wang (wangjing@xynu.edu.cn)