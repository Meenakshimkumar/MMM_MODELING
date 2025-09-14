# MMM Marketing Mix Modeling with Mediation (Google as mediator)
# Enhanced with output saving

# %%
# 1) SETUP
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import statsmodels.api as sm
import joblib
from IPython.display import display

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Ensure output folders exist
os.makedirs("MMM_MODELING/outputs/plots", exist_ok=True)

# %%
# 2) LOAD DATA
DATA_PATH = r"C:/Users/mahes/OneDrive/Desktop/PLACEMENTS/LIFESIGHT/ASSESSMENT_2/MMM_MODELING/data/raw/weekly_data.csv"

if not os.path.exists(DATA_PATH):
    print(f"Please place your dataset at {DATA_PATH} and re-run this cell.")
else:
    df = pd.read_csv(DATA_PATH)
    print('Loaded data:', df.shape)
    display(df.head())

# %%
# 3) BASIC PREP
df['date'] = pd.date_range(start="2020-01-01", periods=len(df), freq="W")

revenue_col = 'revenue'
google_col = 'google_spend'
social_cols = ['facebook_spend', 'tiktok_spend', 'instagram_spend', 'snapchat_spend']
media_cols = [google_col] + social_cols
extras = ['emails_send', 'sms_send', 'average_price', 'social_followers', 'promotions']

# %%
# 4) FEATURE ENGINEERING
data = df.copy()
data = data.sort_values('date').reset_index(drop=True)
data['t'] = np.arange(len(data))
data['sin52'] = np.sin(2*np.pi*data['t']/52)
data['cos52'] = np.cos(2*np.pi*data['t']/52)

# Zero spend indicators
for c in media_cols:
    data[f'{c}_iszero'] = (data[c]==0).astype(int)

# Lags & rolling averages
LAGS = [1,2,3]
for c in media_cols:
    for l in LAGS:
        data[f'{c}_lag{l}'] = data[c].shift(l)
    data[f'{c}_rmean4'] = data[c].rolling(4, min_periods=1).mean().shift(1)

# Log transforms
data['log_revenue'] = np.log1p(data[revenue_col])
for c in media_cols:
    data[f'log_{c}'] = np.log1p(data[c])

data.fillna(0, inplace=True)

# %%
# 5) TRAIN/TEST SPLIT
n = len(data)
train_end = int(n*0.8)
train = data.iloc[:train_end].copy()
test = data.iloc[train_end:].copy()

print('Train weeks:', len(train), 'Test weeks:', len(test))

# %%
# 6) STAGE 1: MEDIATION MODEL
stage1_feats = social_cols + ['t','sin52','cos52']
X1 = train[stage1_feats]
y1 = train[google_col]

ridge1 = Ridge(random_state=RANDOM_STATE)
ridge1.fit(X1, y1)

# Save Stage 1 model
joblib.dump(ridge1, "MMM_MODELING/outputs/ridge_google_model.joblib")

data['pred_google'] = ridge1.predict(data[stage1_feats])
print("Stage 1 complete — Google spend modeled as mediator")

# %%
# 7) STAGE 2: REVENUE MODEL
features = []
features.append('pred_google')
features += [f'log_{c}' for c in social_cols]
features += extras
features += ['t','sin52','cos52']

X = data[features]
y = data['log_revenue']

X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]
X_test = X.iloc[train_end:]
y_test = y.iloc[train_end:]

alphas = np.logspace(-3,3,13)
ridge_cv = RidgeCV(alphas=alphas, store_cv_results=True)
ridge_cv.fit(X_train, y_train)

# Save Stage 2 model
joblib.dump(ridge_cv, "MMM_MODELING/outputs/ridge_revenue_model.joblib")

y_pred_test = ridge_cv.predict(X_test)
rmse_log = np.sqrt(mean_squared_error(y_test, y_pred_test))
rmse_rev = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(y_pred_test)))

print("Ridge alpha:", ridge_cv.alpha_)
print("Test RMSE (log):", rmse_log)
print("Test RMSE (revenue):", rmse_rev)

# Save predictions
results = pd.DataFrame({
    "date": data['date'].iloc[train_end:],
    "actual_revenue": np.expm1(y_test),
    "predicted_revenue": np.expm1(y_pred_test)
})
results.to_csv("MMM_MODELING/outputs/test_predictions.csv", index=False)

# %%
# 8) XGBOOST MODEL
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {'objective':'reg:squarederror','seed':RANDOM_STATE,'verbosity':0}
model_xgb = xgb.train(params, dtrain, num_boost_round=500,
                      evals=[(dtest,'eval')],
                      early_stopping_rounds=20,
                      verbose_eval=False)

ypred_xgb = model_xgb.predict(dtest)
print("XGBoost Test RMSE (log):", np.sqrt(mean_squared_error(y_test, ypred_xgb)))

# Save XGBoost predictions
xgb_results = pd.DataFrame({
    "date": data['date'].iloc[train_end:],
    "actual_revenue": np.expm1(y_test),
    "predicted_revenue_xgb": np.expm1(ypred_xgb)
})
xgb_results.to_csv("MMM_MODELING/outputs/test_predictions_xgb.csv", index=False)

# %%
# 9) RESIDUAL DIAGNOSTICS
residuals = y_test - y_pred_test
plt.figure(figsize=(10,3))
plt.plot(data['date'].iloc[train_end:], residuals)
plt.axhline(0, color='k', linestyle='--')
plt.title("Residuals (log revenue)")
plt.savefig("MMM_MODELING/outputs/plots/residuals.png", dpi=300, bbox_inches="tight")
plt.show()

sm.graphics.tsa.plot_acf(residuals, lags=12)
plt.savefig("MMM_MODELING/outputs/plots/residuals_acf.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
# 10) SENSITIVITY ANALYSIS: PRICE
base = X_test.copy()
sensitivity_price = []
for change in [-0.1, -0.05, 0, 0.05, 0.1]:
    tmp = base.copy()
    if 'average_price' in tmp.columns:
        tmp['average_price'] = tmp['average_price'] * (1+change)
        pred = ridge_cv.predict(tmp)
        mean_rev = np.mean(np.expm1(pred))
        sensitivity_price.append({"change": change, "mean_revenue": mean_rev})

pd.DataFrame(sensitivity_price).to_csv("MMM_MODELING/outputs/sensitivity_price.csv", index=False)

# %%
# 11) SENSITIVITY ANALYSIS: PROMOTIONS
if 'promotions' in X_test.columns:
    tmp = base.copy()
    tmp['promotions'] = 0
    pred_off = ridge_cv.predict(tmp)
    tmp['promotions'] = 1
    pred_on = ridge_cv.predict(tmp)
    promo_effect = np.mean(np.expm1(pred_on)) - np.mean(np.expm1(pred_off))
    with open("MMM_MODELING/outputs/sensitivity_promotions.txt", "w") as f:
        f.write(f"Average revenue lift due to promotions: {promo_effect:.2f}\n")

# %%
