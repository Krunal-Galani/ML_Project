import os
import joblib
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

proj_dir = os.path.dirname(os.path.abspath(__file__))
lr_path = os.path.join(proj_dir, "cardiovascular_logistic_regression_pipeline.pkl")
rf_path = os.path.join(proj_dir, "best_model_task5.pkl")

print("=" * 65)
print("  CARDIOVASCULAR LIFESTYLE RISK VERIFICATION TEST")
print("=" * 65)

lr_pipe = joblib.load(lr_path)
rf_pipe = joblib.load(rf_path)

models = {
    "Logistic Regression (Constrained)": lr_pipe,
    "Random Forest Classifier (Reweighted)": rf_pipe
}

feature_names = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'BMI']

healthy_base = pd.DataFrame([{
    'age': 40, 'gender': 1, 'height': 165, 'weight': 62.0,
    'ap_hi': 115, 'ap_lo': 75, 'cholesterol': 1, 'gluc': 1,
    'smoke': 0, 'alco': 0, 'active': 1, 'BMI': round(62.0 / ((165/100)**2), 2)
}])[feature_names]

all_passed = True

for name, model in models.items():
    print(f"\n--- Testing: {name} ---")
    p_base = model.predict_proba(healthy_base)[0, 1]
    
    # 1. Smoking variant
    p_smoke_df = healthy_base.copy()
    p_smoke_df['smoke'] = 1
    p_smoke = model.predict_proba(p_smoke_df)[0, 1]
    
    # 2. Alcohol variant
    p_alco_df = healthy_base.copy()
    p_alco_df['alco'] = 1
    p_alco = model.predict_proba(p_alco_df)[0, 1]
    
    # 3. Physical Inactivity variant
    p_inact_df = healthy_base.copy()
    p_inact_df['active'] = 0
    p_inact = model.predict_proba(p_inact_df)[0, 1]
    
    # 4. All 3 Adverse Habits variant
    p_all_df = healthy_base.copy()
    p_all_df['smoke'] = 1
    p_all_df['alco'] = 1
    p_all_df['active'] = 0
    p_all = model.predict_proba(p_all_df)[0, 1]
    
    print(f"  Baseline Healthy Risk        : {p_base*100:6.2f}%")
    print(f"  + Smoker (Yes)               : {p_smoke*100:6.2f}% (Change: {(p_smoke - p_base)*100:+5.2f}%)")
    print(f"  + Alcohol Consumption (Yes)  : {p_alco*100:6.2f}% (Change: {(p_alco - p_base)*100:+5.2f}%)")
    print(f"  + Physical Inactivity (No)   : {p_inact*100:6.2f}% (Change: {(p_inact - p_base)*100:+5.2f}%)")
    print(f"  + All 3 Adverse Combined     : {p_all*100:6.2f}% (Change: {(p_all - p_base)*100:+5.2f}%)")
    
    c_smoke = p_smoke > p_base
    c_alco = p_alco > p_base
    c_inact = p_inact > p_base
    c_all = p_all > p_base
    
    print("\n  Validation Rules:")
    print(f"    1. Base < Smoke      : {'PASSED [OK]' if c_smoke else 'FAILED [X]'}")
    print(f"    2. Base < Alcohol    : {'PASSED [OK]' if c_alco else 'FAILED [X]'}")
    print(f"    3. Base < Inactivity : {'PASSED [OK]' if c_inact else 'FAILED [X]'}")
    print(f"    4. Base < All 3      : {'PASSED [OK]' if c_all else 'FAILED [X]'}")
    
    if not (c_smoke and c_alco and c_inact and c_all):
        all_passed = False

print("\n" + "=" * 65)
if all_passed:
    print("  ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")
else:
    print("  ONE OR MORE CHECKS FAILED.")
print("=" * 65)
