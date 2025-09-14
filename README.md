# Marketing Mix Modeling (MMM) with Mediation Assumption

## Project Overview
This project models **weekly revenue** as a function of media spends, direct marketing levers, and control variables.  
We apply a **two-stage approach** to account for the mediation effect of Google spend between social channels and revenue.

Dataset: 2 years of weekly data  
Columns:  
- Media spends: `facebook_spend`, `tiktok_spend`, `instagram_spend`, `snapchat_spend`, `google_spend`  
- Direct levers: `emails_send`, `sms_send`  
- Controls: `average_price`, `promotions`, `social_followers`  
- Target: `revenue`

---

## Workflow
1. **Data Preparation**
   - Weekly seasonality via Fourier terms
   - Zero-spend indicators
   - Lags & rolling averages for media
   - Log transformations for skewed variables

2. **Modeling Approach**
   - Stage 1: Model Google spend as a function of social spends
   - Stage 2: Model revenue using predicted Google + other features
   - Models: Ridge Regression, RidgeCV, XGBoost

3. **Diagnostics**
   - Rolling/blocked CV to respect time order
   - Residual analysis & autocorrelation checks
   - Sensitivity analysis for price & promotions

4. **Insights**
   - Identify key drivers of revenue
   - Estimate mediated effects of social → Google → Revenue
   - Provide actionable recommendations for growth teams

---

## Repository Structure
```
MMM_MODELING/
├── data/
│   └── raw/
│       └── weekly_data.csv         # Input dataset
├── notebooks/
│   └── MMM_modeling_notebook.py    # Reproducible notebook (can run in Jupyter/VSCode)
├── outputs/
│   ├── ridge_revenue_model.joblib  # Saved Ridge model (Stage 2)
│   ├── ridge_google_model.joblib   # Saved Ridge model (Stage 1)
│   └── plots/                      # Diagnostic plots
├── README.md                       # Project overview & instructions
└── MMM_Assessment_Report.docx      # Short report (2–3 pages)
```

---

## How to Run
1. Clone/download the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Place your dataset at:
   ```
   MMM_MODELING/data/raw/weekly_data.csv
   ```

3. Run the notebook:
   ```bash
   jupyter notebook notebooks/MMM_modeling_notebook.py
   ```

4. Follow cells step by step → outputs will include diagnostics, predictions, and sensitivity analysis.

---

## Requirements
- Python 3.9+
- Libraries: pandas, numpy, scikit-learn, statsmodels, matplotlib, seaborn, xgboost, shap

---

## Deliverables
- 📓 Reproducible notebook (`MMM_modeling_notebook.py`)
- 📝 Short report (`MMM_Assessment_Report.docx`)
- 📊 Diagnostic plots & saved models (`outputs/` folder)
- 📘 README.md (this file)

---

## Notes
- Replace column names if your dataset schema differs.  
- Fill in actual RMSE values, coefficients, and sensitivity test results in the report.  
- Ensure time-based validation (no look-ahead bias) when extending the model.

---

**Author:** Meenakshi M Kumar  
