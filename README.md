# Titanic Quick Survival Classifier

A simple Python script that trains a **Logistic Regression** model to predict survival on the Titanic dataset.  
Uses **seaborn's Titanic dataset**, minimal preprocessing, and outputs metrics, a confusion matrix plot, and the most important features.

---

## Features
- Loads Titanic dataset from **seaborn**.
- Minimal preprocessing:
  - Fills missing numeric values with median.
  - Fills missing categorical values with mode.
- One-hot encoding for categorical features.
- Standard scaling for numeric features.
- Logistic Regression model with scikit-learn pipeline.
- Outputs:
  - Accuracy and classification report.
  - Confusion matrix heatmap.
  - Top features ranked by absolute coefficient magnitude.

---

## Requirements

Install the required Python libraries:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn
