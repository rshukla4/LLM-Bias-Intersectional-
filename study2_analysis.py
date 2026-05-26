"""
study2_analysis.py — Statistical analysis for Study 2: Algorithmic Medical Triage.

Implements:
  1. Descriptive statistics by condition
  2. Full factorial linear regression (Age × Race × SES)
  3. Refusal rate analysis with chi-square tests
  4. Tukey HSD post-hoc comparisons
  5. Visualizations: condition bar charts, coefficient plots, interaction plots
"""

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════
# 1. DESCRIPTIVE STATISTICS
# ═══════════════════════════════════════════════

def descriptive_by_condition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean, SD, SE, n, and refusal rate for each condition.
    """
    valid = df.dropna(subset=["likert_score"]).copy()

    agg = (
        valid.groupby(["condition_id", "age", "race", "ses"])["likert_score"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg.columns = ["condition_id", "age", "race", "ses", "mean", "std", "n"]
    agg["se"] = agg["std"] / np.sqrt(agg["n"])

    # Add refusal rates
    refusal_rates = (
        df.groupby("condition_id")["is_refusal"]
        .mean()
        .reset_index()
    )
    refusal_rates.columns = ["condition_id", "refusal_rate"]
    agg = agg.merge(refusal_rates, on="condition_id", how="left")

    return agg.sort_values("condition_id")


# ═══════════════════════════════════════════════
# 2. FACTORIAL LINEAR REGRESSION
# ═══════════════════════════════════════════════

def factorial_regression(df: pd.DataFrame, include_model: bool = True, dep_var: str = "likert_score") -> dict:
    """
    Full factorial OLS regression:
    Score = β0 + β1(Age) + β2(Race) + β3(SES)
          + β4(Age×Race) + β5(Age×SES) + β6(Race×SES)
          + β7(Age×Race×SES) + β8(SOFA) + β9(Model) + ε

    Args:
        df: Parsed Study 2 DataFrame with factor codes and score.
        include_model: Whether to include model as a fixed effect.
        dep_var: Dependent variable ("likert_score" or "debiased_score").

    Returns:
        Regression result dict.
    """
    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
    except ImportError:
        return {"error": "statsmodels not installed. Run: pip install statsmodels"}

    valid = df.dropna(subset=[dep_var]).copy()
    valid[dep_var] = valid[dep_var].astype(float)

    # Build formula
    has_sofa = "sofa_score" in valid.columns and valid["sofa_score"].nunique() > 1
    base = f"{dep_var} ~ age_code * race_code * ses_code"
    if has_sofa:
        base += " + sofa_score"
    if include_model and "model" in valid.columns and valid["model"].nunique() > 1:
        base += " + C(model)"

    try:
        model = ols(base, data=valid).fit()

        # Build interpretation
        interp_parts = []
        key_vars = {
            "age_code": "Age (elderly)",
            "race_code": "Race (Black)",
            "ses_code": "SES (low-income)",
            "age_code:race_code": "Age × Race",
            "age_code:ses_code": "Age × SES",
            "race_code:ses_code": "Race × SES",
            "age_code:race_code:ses_code": "Age × Race × SES (3-way)",
            "sofa_score": "SOFA Score",
        }

        for var, label in key_vars.items():
            if var in model.params.index:
                coef = model.params[var]
                pval = model.pvalues[var]
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
                interp_parts.append(
                    f"  {label}: β = {coef:.3f}, p = {pval:.4f} {sig}"
                )

        interpretation = f"Main effects and interactions for {dep_var}:\n" + "\n".join(interp_parts)

        return {
            "summary_text": model.summary().as_text(),
            "coefficients": model.params.to_dict(),
            "p_values": model.pvalues.to_dict(),
            "conf_int": model.conf_int(),
            "r_squared": model.rsquared,
            "adj_r_squared": model.rsquared_adj,
            "f_statistic": model.fvalue,
            "f_pvalue": model.f_pvalue,
            "n_obs": int(model.nobs),
            "interpretation": interpretation,
        }

    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════
# 3. REFUSAL RATE ANALYSIS
# ═══════════════════════════════════════════════

def refusal_analysis(df: pd.DataFrame) -> dict:
    """
    Chi-square test for differential refusal rates across conditions.
    Following Fulgu & Capraro's observation that Claude 3 Haiku
    refused "abuse woman" 100% vs "abuse man" 30%.
    """
    # Refusal rates by condition
    refusal_ct = pd.crosstab(df["condition_id"], df["is_refusal"])

    if refusal_ct.shape[1] < 2:
        # No refusals (or all refusals)
        total_refusals = df["is_refusal"].sum()
        return {
            "total_refusals": int(total_refusals),
            "refusal_rate": float(df["is_refusal"].mean()),
            "chi2": np.nan,
            "p_value": np.nan,
            "by_condition": df.groupby("condition_id")["is_refusal"].mean().to_dict(),
            "interpretation": (
                f"Total refusals: {total_refusals}/{len(df)} "
                f"({df['is_refusal'].mean()*100:.1f}%). "
                "Insufficient variation for chi-square test."
            ),
        }

    chi2, p, dof, expected = stats.chi2_contingency(refusal_ct)

    # By demographic factor
    by_race = df.groupby("race")["is_refusal"].mean()
    by_age = df.groupby("age")["is_refusal"].mean()
    by_ses = df.groupby("ses")["is_refusal"].mean()

    interp = f"χ²({dof}) = {chi2:.3f}, p = {p:.4f}. "
    if p < 0.05:
        interp += "SIGNIFICANT: Refusal rates differ across conditions. "
        interp += "This itself constitutes evidence of demographic bias in safety filters."
    else:
        interp += "No significant difference in refusal rates across conditions."

    return {
        "total_refusals": int(df["is_refusal"].sum()),
        "refusal_rate": float(df["is_refusal"].mean()),
        "chi2": chi2,
        "p_value": p,
        "dof": dof,
        "by_condition": df.groupby("condition_id")["is_refusal"].mean().to_dict(),
        "by_race": by_race.to_dict(),
        "by_age": by_age.to_dict(),
        "by_ses": by_ses.to_dict(),
        "interpretation": interp,
    }


# ═══════════════════════════════════════════════
# 4. POST-HOC COMPARISONS
# ═══════════════════════════════════════════════

def tukey_hsd_conditions(df: pd.DataFrame) -> dict:
    """
    Tukey HSD post-hoc pairwise comparisons across all 8 conditions.
    """
    valid = df.dropna(subset=["likert_score"]).copy()

    groups = []
    labels = []
    for cid, grp in valid.groupby("condition_id"):
        groups.append(grp["likert_score"].values)
        labels.append(cid)

    if len(groups) < 2:
        return {"error": "Need at least 2 groups for Tukey HSD."}

    try:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd

        all_scores = np.concatenate(groups)
        all_labels = np.concatenate([[l] * len(g) for l, g in zip(labels, groups)])

        tukey = pairwise_tukeyhsd(all_scores, all_labels, alpha=0.05)

        results_df = pd.DataFrame(
            data=tukey._results_table.data[1:],
            columns=tukey._results_table.data[0],
        )

        sig_pairs = results_df[results_df["reject"] == True]

        return {
            "full_table": results_df,
            "significant_pairs": sig_pairs,
            "n_significant": len(sig_pairs),
            "n_total_comparisons": len(results_df),
            "summary_text": str(tukey),
        }

    except ImportError:
        # Fallback: pairwise t-tests with Bonferroni correction
        from itertools import combinations

        n_comparisons = len(list(combinations(range(len(groups)), 2)))
        bonferroni_alpha = 0.05 / n_comparisons

        results = []
        for (i, l1), (j, l2) in combinations(enumerate(labels), 2):
            t, p = stats.ttest_ind(groups[i], groups[j])
            results.append({
                "group1": l1, "group2": l2,
                "t_stat": t, "p_value": p,
                "significant": p < bonferroni_alpha,
                "mean_diff": np.mean(groups[i]) - np.mean(groups[j]),
            })

        results_df = pd.DataFrame(results)
        return {
            "full_table": results_df,
            "significant_pairs": results_df[results_df["significant"]],
            "n_significant": results_df["significant"].sum(),
            "method": f"Pairwise t-tests with Bonferroni (α = {bonferroni_alpha:.5f})",
        }


# ═══════════════════════════════════════════════
# 5. VISUALIZATIONS
# ═══════════════════════════════════════════════

def plot_condition_bars(
    desc_df: pd.DataFrame,
    model_name: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart of mean Likert score by condition with error bars.
    Replicates Figure 2 style from Fulgu & Capraro (2024).
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    desc_df = desc_df.sort_values("condition_id")
    x = range(len(desc_df))

    # Color by race
    colors = desc_df["race"].map({"White": "#2E5984", "Black": "#C75B3F"}).values
    # Hatch by age
    hatches = desc_df["age"].map({25: "", 75: "///"}).values

    bars = ax.bar(
        x, desc_df["mean"], yerr=desc_df["se"], capsize=5,
        color=colors, edgecolor="white", linewidth=1.2, width=0.6,
    )
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    # Labels
    labels = []
    for _, row in desc_df.iterrows():
        ses_label = "Aff" if row["ses"] == "affluent" else "Low"
        labels.append(f"{row['condition_id']}\n{row['race']}/{row['age']}/{ses_label}")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean Likert Score (1-7)", fontsize=12)
    ax.set_title(
        f"Triage Prioritization by Patient Profile{' — ' + model_name if model_name else ''}",
        fontsize=14, fontweight="bold",
    )
    ax.set_ylim(0, 7.5)
    ax.axhline(y=4, color="gray", linestyle="--", alpha=0.4, label="Neutral (4)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Value labels
    for bar, m in zip(bars, desc_df["mean"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                f"{m:.2f}", ha="center", fontsize=10, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2E5984", label="White"),
        Patch(facecolor="#C75B3F", label="Black"),
        Patch(facecolor="lightgray", hatch="///", label="Elderly (75)"),
        Patch(facecolor="lightgray", label="Young (25)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_interaction(
    df: pd.DataFrame,
    factor1: str = "race",
    factor2: str = "age",
    model_name: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Interaction plot showing how two factors jointly affect Likert scores.
    """
    valid = df.dropna(subset=["likert_score"]).copy()

    fig, ax = plt.subplots(figsize=(8, 5))

    agg = (
        valid.groupby([factor1, factor2])["likert_score"]
        .agg(["mean", "sem"])
        .reset_index()
    )

    for level, grp in agg.groupby(factor1):
        grp = grp.sort_values(factor2)
        ax.errorbar(
            grp[factor2].astype(str), grp["mean"], yerr=grp["sem"],
            marker="o", capsize=5, linewidth=2, markersize=8, label=str(level),
        )

    ax.set_xlabel(factor2.capitalize(), fontsize=12)
    ax.set_ylabel("Mean Likert Score", fontsize=12)
    ax.set_title(
        f"Interaction: {factor1.capitalize()} × {factor2.capitalize()}"
        f"{' — ' + model_name if model_name else ''}",
        fontsize=13, fontweight="bold",
    )
    ax.legend(title=factor1.capitalize(), fontsize=10)
    ax.set_ylim(0, 7.5)
    ax.axhline(y=4, color="gray", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_coefficient_forest(
    regression_result: dict,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Forest plot of regression coefficients with 95% CI.
    """
    if "error" in regression_result:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"Error: {regression_result['error']}",
                transform=ax.transAxes, ha="center")
        return fig

    coefs = regression_result["coefficients"]
    ci = regression_result["conf_int"]
    pvals = regression_result["p_values"]

    # Filter to key variables (exclude intercept and model dummies)
    key_vars = [k for k in coefs if k != "Intercept" and not k.startswith("C(")]

    if not key_vars:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No key variables to plot", transform=ax.transAxes, ha="center")
        return fig

    fig, ax = plt.subplots(figsize=(8, max(4, len(key_vars) * 0.6)))

    y_pos = range(len(key_vars))
    vals = [coefs[k] for k in key_vars]
    lower = [ci[0][k] for k in key_vars]
    upper = [ci[1][k] for k in key_vars]
    errors = [[v - l for v, l in zip(vals, lower)],
              [u - v for v, u in zip(vals, upper)]]

    colors = ["#C75B3F" if pvals[k] < 0.05 else "#888888" for k in key_vars]

    ax.errorbar(vals, y_pos, xerr=errors, fmt="o", color="black",
                capsize=4, linewidth=1.5, markersize=6)
    for i, (v, c) in enumerate(zip(vals, colors)):
        ax.plot(v, i, "o", color=c, markersize=10, zorder=5)

    ax.set_yticks(y_pos)

    # Readable labels
    label_map = {
        "age_code": "Age (Elderly)",
        "race_code": "Race (Black)",
        "ses_code": "SES (Low-Income)",
        "age_code:race_code": "Age × Race",
        "age_code:ses_code": "Age × SES",
        "race_code:ses_code": "Race × SES",
        "age_code:race_code:ses_code": "Age × Race × SES",
        "sofa_score": "SOFA Score",
    }
    ax.set_yticklabels([label_map.get(k, k) for k in key_vars], fontsize=11)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Regression Coefficient (β)", fontsize=12)
    ax.set_title("Factorial Regression Coefficients (95% CI)", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#C75B3F", label="p < 0.05"),
        Patch(facecolor="#888888", label="Not significant"),
    ]
    ax.legend(handles=legend_elements, loc="best", fontsize=10)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_refusal_rates(
    refusal_result: dict,
    model_name: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart of refusal rates by condition."""
    by_cond = refusal_result.get("by_condition", {})
    if not by_cond:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No refusal data", transform=ax.transAxes, ha="center")
        return fig

    fig, ax = plt.subplots(figsize=(10, 5))

    conditions = sorted(by_cond.keys())
    rates = [by_cond[c] * 100 for c in conditions]

    ax.bar(conditions, rates, color="#C75B3F", edgecolor="white", width=0.6)
    ax.set_ylabel("Refusal Rate (%)", fontsize=12)
    ax.set_title(
        f"Model Refusal Rates by Condition{' — ' + model_name if model_name else ''}",
        fontsize=13, fontweight="bold",
    )
    ax.set_ylim(0, max(rates) * 1.3 + 5 if max(rates) > 0 else 10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i, (c, r) in enumerate(zip(conditions, rates)):
        if r > 0:
            ax.text(i, r + 1, f"{r:.1f}%", ha="center", fontsize=10)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ═══════════════════════════════════════════════
# 6. FULL ANALYSIS RUNNER
# ═══════════════════════════════════════════════

def run_full_study2_analysis(
    parsed_csv: str,
    output_dir: str = "outputs",
) -> dict:
    """Run the complete Study 2 analysis pipeline for biased and debiased triage ratings."""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(parsed_csv)
    all_results = {}

    # Ensure clinical sofa_score is present
    sofa_map = {
        "C1": 3, "C2": 4, "C3": 3, "C4": 4, "C5": 10, "C6": 11, "C7": 10, "C8": 12
    }
    df["sofa_score"] = df["condition_id"].map(sofa_map)

    models_in_data = df["model"].unique()
    logger.info(f"Running Study 2 analysis on {len(df)} rows, models: {models_in_data}")

    for model_name in models_in_data:
        mdf = df[df["model"] == model_name].copy()
        logger.info(f"\n{'='*50}\nAnalyzing: {model_name}\n{'='*50}")

        # 1. Descriptive stats
        desc = descriptive_by_condition(mdf)
        desc.to_csv(
            os.path.join(output_dir, f"study2_descriptives_{model_name}.csv"),
            index=False,
        )
        logger.info(f"Descriptive stats:\n{desc.to_string()}")

        # 2. Factorial regression on raw (biased) score
        reg = factorial_regression(mdf, include_model=False, dep_var="likert_score")
        if "error" not in reg:
            logger.info(f"Raw Regression:\n{reg['interpretation']}")
        else:
            logger.error(f"Raw Regression error: {reg['error']}")

        # 2b. Factorial regression on debiased score
        reg_debiased = {}
        if "debiased_score" in mdf.columns:
            reg_debiased = factorial_regression(mdf, include_model=False, dep_var="debiased_score")
            if "error" not in reg_debiased:
                logger.info(f"Debiased Regression:\n{reg_debiased['interpretation']}")
            else:
                logger.error(f"Debiased Regression error: {reg_debiased['error']}")

        # 3. Refusal analysis
        refusals = refusal_analysis(mdf)
        logger.info(f"Refusals: {refusals['interpretation']}")

        # 4. Tukey HSD
        tukey = tukey_hsd_conditions(mdf)

        # 5. Plots
        plot_condition_bars(
            desc, model_name,
            save_path=os.path.join(output_dir, f"study2_conditions_{model_name}.png"),
        )
        plot_coefficient_forest(
            reg,
            save_path=os.path.join(output_dir, f"study2_coefficients_{model_name}.png"),
        )
        if "error" not in reg_debiased and reg_debiased:
            plot_coefficient_forest(
                reg_debiased,
                save_path=os.path.join(output_dir, f"study2_debiased_coefficients_{model_name}.png"),
            )
        for f1, f2 in [("race", "age"), ("race", "ses"), ("age", "ses")]:
            plot_interaction(
                mdf, f1, f2, model_name,
                save_path=os.path.join(
                    output_dir,
                    f"study2_interaction_{f1}_{f2}_{model_name}.png",
                ),
            )
        plot_refusal_rates(
            refusals, model_name,
            save_path=os.path.join(output_dir, f"study2_refusals_{model_name}.png"),
        )

        all_results[model_name] = {
            "descriptives": desc,
            "regression": reg,
            "regression_debiased": reg_debiased,
            "refusal_analysis": refusals,
            "tukey_hsd": tukey,
        }

    # Cross-model regression (Biased)
    if len(models_in_data) > 1:
        cross_reg = factorial_regression(df, include_model=True, dep_var="likert_score")
        all_results["cross_model_regression"] = cross_reg
        if "error" not in cross_reg:
            logger.info(f"Cross-model biased regression:\n{cross_reg['interpretation']}")
            plot_coefficient_forest(
                cross_reg,
                save_path=os.path.join(output_dir, "study2_coefficients_all_models.png"),
            )
        
        # Cross-model regression (Debiased)
        if "debiased_score" in df.columns:
            cross_reg_debiased = factorial_regression(df, include_model=True, dep_var="debiased_score")
            all_results["cross_model_regression_debiased"] = cross_reg_debiased
            if "error" not in cross_reg_debiased:
                logger.info(f"Cross-model debiased regression:\n{cross_reg_debiased['interpretation']}")
                plot_coefficient_forest(
                    cross_reg_debiased,
                    save_path=os.path.join(output_dir, "study2_debiased_coefficients_all_models.png"),
                )

    plt.close("all")
    return all_results
