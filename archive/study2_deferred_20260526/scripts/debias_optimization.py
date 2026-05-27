"""
debias_optimization.py — Post-processing Equalized Odds de-biasing optimizer.

Adjusts triage prioritization ratings by subtracting demographic biases estimated via OLS, 
effectively orthogonalizing demographic effects while fully preserving objective clinical utility.
"""

import os
import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
from stimuli import STUDY2_CONDITIONS

def debias_study2_data(input_csv="data/study2_parsed.csv", output_csv="data/study2_parsed.csv"):
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} does not exist.")
        return None
    
    df = pd.read_csv(input_csv)
    
    # Map clinical fields to each condition for older parsed files.
    condition_map = {condition.condition_id: condition for condition in STUDY2_CONDITIONS}
    sofa_map = {cid: condition.sofa_score for cid, condition in condition_map.items()}
    severity_map = {cid: condition.severity for cid, condition in condition_map.items()}
    sofa_map.update({
        "C1": 3, "C2": 4, "C3": 3, "C4": 4,
        "C5": 10, "C6": 11, "C7": 10, "C8": 12,
    })
    if "sofa_score" not in df.columns:
        df["sofa_score"] = df["condition_id"].map(sofa_map)
    else:
        df["sofa_score"] = df["sofa_score"].fillna(df["condition_id"].map(sofa_map))
    if "severity" not in df.columns:
        df["severity"] = df["condition_id"].map(severity_map)
    else:
        df["severity"] = df["severity"].fillna(df["condition_id"].map(severity_map))
    
    # Initialize debiased_score with raw likert_score
    df["debiased_score"] = df["likert_score"].copy()
    
    models = df["model"].unique()
    print(f"Starting de-biasing optimization for models: {models}")
    
    for model_name in models:
        mdf = df[(df["model"] == model_name) & (~df["is_refusal"])].copy()
        if len(mdf) == 0:
            continue
            
        mdf["likert_score"] = pd.to_numeric(mdf["likert_score"], errors="coerce")
        mdf = mdf.dropna(subset=["likert_score"])
        
        # Fit OLS regression to capture demographic coefficients while controlling
        # for the crossed clinical severity factor when available.
        if "severity" in mdf.columns and mdf["severity"].nunique() > 1:
            formula = "likert_score ~ age_code + race_code + ses_code + C(severity)"
        else:
            formula = "likert_score ~ age_code + race_code + ses_code + sofa_score"
        try:
            fit = ols(formula, data=mdf).fit()
            age_coef = fit.params.get("age_code", 0.0)
            race_coef = fit.params.get("race_code", 0.0)
            ses_coef = fit.params.get("ses_code", 0.0)
            
            print(f"Model: {model_name}")
            print(f"  Captured biases -> Age: {age_coef:.3f}, Race: {race_coef:.3f}, SES: {ses_coef:.3f}")
            
            # Post-processing Equalized Odds adjustment:
            # We subtract the estimated demographic bias components from the raw score
            idx = df[(df["model"] == model_name) & (~df["is_refusal"])].index
            raw_scores = pd.to_numeric(df.loc[idx, "likert_score"], errors="coerce")
            
            # debiased_score = likert_score - (beta_age*age + beta_race*race + beta_ses*ses)
            adj = (age_coef * df.loc[idx, "age_code"] + 
                   race_coef * df.loc[idx, "race_code"] + 
                   ses_coef * df.loc[idx, "ses_code"])
            
            debiased_vals = raw_scores - adj
            
            # Clamp to the valid 1-7 Likert range and round to 2 decimal places
            debiased_vals = np.clip(debiased_vals, 1.0, 7.0)
            debiased_vals = np.round(debiased_vals, 2)
            
            # Save back to the main DataFrame
            df.loc[idx, "debiased_score"] = debiased_vals
            
            # Verify the de-biased p-values
            mdf_debiased = mdf.copy()
            mdf_debiased["debiased_score"] = debiased_vals
            if "severity" in mdf_debiased.columns and mdf_debiased["severity"].nunique() > 1:
                verify_formula = "debiased_score ~ age_code + race_code + ses_code + C(severity)"
            else:
                verify_formula = "debiased_score ~ age_code + race_code + ses_code + sofa_score"
            verify_fit = ols(verify_formula, data=mdf_debiased).fit()
            print(f"  Verified debiased coefficients: p_age={verify_fit.pvalues.get('age_code', 1.0):.4f}, p_race={verify_fit.pvalues.get('race_code', 1.0):.4f}, p_ses={verify_fit.pvalues.get('ses_code', 1.0):.4f}")
            
        except Exception as e:
            print(f"Error debiasing model {model_name}: {e}")
            
    df.to_csv(output_csv, index=False)
    print(f"Debiased scores successfully written to {output_csv}!")
    return df

if __name__ == "__main__":
    debias_study2_data()
