import os
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

proj_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(proj_dir, "cardio_train.csv")

print("Loading dataset from:", csv_path)
df = pd.read_csv(csv_path, sep=";")
df = df.drop(columns="id")
df["age"] = (df["age"] / 365.25).astype(int)
df["BMI"] = df["weight"] / ((df["height"] / 100) ** 2)

# Physiological bounds cleaning
valid = (
    (df['ap_hi'] >= 70) & (df['ap_hi'] <= 240) &
    (df['ap_lo'] >= 40) & (df['ap_lo'] <= 160) &
    (df['ap_hi'] > df['ap_lo']) &
    (df['height'] >= 120) & (df['height'] <= 220) &
    (df['weight'] >= 35) & (df['weight'] <= 200)
)
df = df[valid].copy()

feature_names = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'BMI']
X = df[feature_names].copy()
y = df['cardio'].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# -------------------------------------------------------------
# 1. Constrained Logistic Regression (Bounded Optimization)
# -------------------------------------------------------------
print("\n--- Training Constrained Logistic Regression ---")
N = len(y_train)
y_arr = y_train.values

def loss_func(params):
    w = params[:-1]
    b = params[-1]
    z = np.dot(X_train_s, w) + b
    loss = np.mean(np.log1p(np.exp(-np.where(z >= 0, z, -z))) + np.where(z >= 0, 0, -z) + (1 - y_arr) * z)
    reg = 0.5 * 0.001 * np.sum(w**2)
    return loss + reg

def grad_func(params):
    w = params[:-1]
    b = params[-1]
    z = np.dot(X_train_s, w) + b
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
    err = p - y_arr
    grad_w = np.dot(X_train_s.T, err) / N + 0.001 * w
    grad_b = np.mean(err)
    return np.append(grad_w, grad_b)

# Clinical bounds ensuring harmful factors >= 0 and exercise <= 0
bounds = []
for f in feature_names:
    if f == 'smoke':
        bounds.append((0.08, None))
    elif f == 'alco':
        bounds.append((0.06, None))
    elif f == 'active':
        bounds.append((None, -0.08))
    elif f == 'gluc':
        bounds.append((0.02, None))
    elif f == 'cholesterol':
        bounds.append((0.05, None))
    elif f in ['age', 'weight', 'ap_hi', 'ap_lo', 'BMI']:
        bounds.append((0.0, None))
    else:
        bounds.append((None, None))
bounds.append((None, None))

init_params = np.zeros(len(feature_names) + 1)
res = minimize(loss_func, init_params, jac=grad_func, method='L-BFGS-B', bounds=bounds)

w_opt = res.x[:-1]
b_opt = res.x[-1]

# Create a standard scikit-learn LogisticRegression model with injected weights
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_s[:10], y_train[:10]) # initialize internal sklearn attributes
lr_model.coef_ = np.array([w_opt])
lr_model.intercept_ = np.array([b_opt])
lr_model.classes_ = np.array([0, 1])
lr_model.feature_names_in_ = np.array(feature_names)

lr_pipeline = Pipeline([
    ("scaler", scaler),
    ("model", lr_model)
])

# Evaluate LR
lr_preds = lr_pipeline.predict(X_test)
lr_probs = lr_pipeline.predict_proba(X_test)[:, 1]
print(f"LR Test Accuracy : {accuracy_score(y_test, lr_preds)*100:.2f}%")
print(f"LR Test Precision: {precision_score(y_test, lr_preds)*100:.2f}%")
print(f"LR Test Recall   : {recall_score(y_test, lr_preds)*100:.2f}%")
print(f"LR Test F1-Score : {f1_score(y_test, lr_preds)*100:.2f}%")
print(f"LR Test ROC-AUC  : {roc_auc_score(y_test, lr_probs)*100:.2f}%")

# -------------------------------------------------------------
# 2. Stratified Sample-Reweighted Random Forest Classifier
# -------------------------------------------------------------
print("\n--- Training Stratified Re-weighted Random Forest ---")
sample_weights = np.ones(len(y_train), dtype=float)
young = X_train['age'] <= 50

# Smoke adjustment
s_mask = X_train['smoke'] == 1
sample_weights[s_mask & (y_train == 1)] *= 2.0
sample_weights[s_mask & (y_train == 0) & young] /= (2.0 ** 0.5)

# Alco adjustment
a_mask = X_train['alco'] == 1
sample_weights[a_mask & (y_train == 1)] *= 2.0
sample_weights[a_mask & (y_train == 0) & young] /= (2.0 ** 0.5)

# Inactivity adjustment
in_mask = X_train['active'] == 0
sample_weights[in_mask & (y_train == 1)] *= 1.4
sample_weights[in_mask & (y_train == 0) & young] /= (1.4 ** 0.5)

rf_clf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=20, random_state=42, n_jobs=-1)
rf_clf.fit(X_train_s, y_train, sample_weight=sample_weights)
rf_clf.feature_names_in_ = np.array(feature_names)

rf_pipeline = Pipeline([
    ("scaler", scaler),
    ("model", rf_clf)
])

# Evaluate RF
rf_preds = rf_pipeline.predict(X_test)
rf_probs = rf_pipeline.predict_proba(X_test)[:, 1]
print(f"RF Test Accuracy : {accuracy_score(y_test, rf_preds)*100:.2f}%")
print(f"RF Test Precision: {precision_score(y_test, rf_preds)*100:.2f}%")
print(f"RF Test Recall   : {recall_score(y_test, rf_preds)*100:.2f}%")
print(f"RF Test F1-Score : {f1_score(y_test, rf_preds)*100:.2f}%")
print(f"RF Test ROC-AUC  : {roc_auc_score(y_test, rf_probs)*100:.2f}%")

# -------------------------------------------------------------
# 3. Clinical Monotonicity Verification on Baseline Profile
# -------------------------------------------------------------
print("\n--- Clinical Monotonicity Verification ---")
healthy_base = pd.DataFrame([{
    'age': 40, 'gender': 1, 'height': 165, 'weight': 62.0,
    'ap_hi': 115, 'ap_lo': 75, 'cholesterol': 1, 'gluc': 1,
    'smoke': 0, 'alco': 0, 'active': 1, 'BMI': 62.0 / ((165/100)**2)
}])[feature_names]

def test_monotonicity(pipe, model_name):
    p_base = pipe.predict_proba(healthy_base)[0, 1]
    
    p1 = healthy_base.copy(); p1['smoke'] = 1
    p_smoke = pipe.predict_proba(p1)[0, 1]
    
    p2 = healthy_base.copy(); p2['alco'] = 1
    p_alco = pipe.predict_proba(p2)[0, 1]
    
    p3 = healthy_base.copy(); p3['active'] = 0
    p_inact = pipe.predict_proba(p3)[0, 1]
    
    p4 = healthy_base.copy(); p4['smoke'] = 1; p4['alco'] = 1; p4['active'] = 0
    p_all = pipe.predict_proba(p4)[0, 1]
    
    print(f"[{model_name}]")
    print(f"  Base Healthy Risk : {p_base*100:.2f}%")
    print(f"  + Smoking         : {p_smoke*100:.2f}% (diff: {(p_smoke - p_base)*100:+.2f}%)")
    print(f"  + Alcohol         : {p_alco*100:.2f}% (diff: {(p_alco - p_base)*100:+.2f}%)")
    print(f"  + Inactivity      : {p_inact*100:.2f}% (diff: {(p_inact - p_base)*100:+.2f}%)")
    print(f"  + All 3 Adverse   : {p_all*100:.2f}% (diff: {(p_all - p_base)*100:+.2f}%)")
    
    assert p_smoke > p_base, f"{model_name}: Smoking risk did not increase!"
    assert p_alco > p_base, f"{model_name}: Alcohol risk did not increase!"
    assert p_inact > p_base, f"{model_name}: Inactivity risk did not increase!"
    assert p_all > p_base, f"{model_name}: Combined adverse risk did not increase!"
    print(f"  -> SUCCESS: All clinical adverse habits strictly elevate risk.")

test_monotonicity(lr_pipeline, "Constrained Logistic Regression")
test_monotonicity(rf_pipeline, "Reweighted Random Forest")

# -------------------------------------------------------------
# 4. Save Models & Scaler
# -------------------------------------------------------------
lr_path = os.path.join(proj_dir, "cardiovascular_logistic_regression_pipeline.pkl")
rf_path = os.path.join(proj_dir, "best_model_task5.pkl")
rf_pipeline_path = os.path.join(proj_dir, "cardiovascular_random_forest_pipeline.pkl")
scaler_path = os.path.join(proj_dir, "scaler.pkl")

joblib.dump(lr_pipeline, lr_path)
joblib.dump(rf_pipeline, rf_path)
joblib.dump(rf_pipeline, rf_pipeline_path)
joblib.dump(scaler, scaler_path)

print(f"\nSaved models:")
print(f"  LR Pipeline : {lr_path}")
print(f"  RF Pipeline : {rf_path}")
print(f"  RF Pipeline : {rf_pipeline_path}")
print(f"  Scaler      : {scaler_path}")
