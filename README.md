# Diabetes Prediction App

A Streamlit web app that predicts diabetes risk using a tuned Random Forest classifier trained on the [Kaggle diabetes dataset](https://www.kaggle.com/datasets/tusharpotdar/diabetes-data).

## Features

- Enter 15 clinical biomarkers and 4 history/lifestyle factors on a single page
- Live prediction with confidence score and probability gauge
- Biomarker check highlighting values outside typical reference ranges
- Radar chart comparing patient values to normal limits
- Feature importance chart showing what drives the model

## Model

The model is trained in [RF_model.ipynb](https://colab.research.google.com/drive/10poAXmlHj-fP1cAlBLJkey2IFbWQ4ID2?usp=sharing):

- `RandomForestClassifier` inside a scikit-learn `Pipeline` with a `ColumnTransformer` (numeric passthrough + ordinal-encoded categoricals)
- Hyperparameters tuned with `RandomizedSearchCV` and 5-fold stratified CV
- CV accuracy: **90.3%**, test accuracy: **91.5%**
- Exported to `diabetes_rf_model.joblib`

### Input features

| Group | Features |
| --- | --- |
| Glycemic | HbA1c, Fasting Glucose, Postprandial Glucose, Random Glucose, OGTT, Fructosamine |
| Metabolic | Insulin, HOMA-IR, C-Peptide, Triglycerides, HDL Cholesterol |
| Body & Vitals | BMI, Waist Circumference, Systolic BP, Diastolic BP |
| History & Lifestyle | Family History of Diabetes, Hypertension, Physical Activity, Obesity |

## Getting started

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501.

## Project structure

```
app.py                     # Streamlit app
diabetes_rf_model.joblib   # Exported model
requirements.txt           # Dependencies
```

## Note

The model was exported with scikit-learn 1.6.1. `app.py` includes a small compatibility shim so it loads on newer versions; for best results, re-run the notebook to re-export the model with your installed scikit-learn version.

