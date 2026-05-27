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
    group_cols = ["condition_id", "age", "race", "ses"]
    if "severity" in valid.columns:
        group_cols.append("severity")
    if "sofa_score" in valid.columns:
        group_cols.append("sofa_score")

    agg = (
        valid.groupby(group_cols)["likert_score"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg.columns = group_cols + ["mean", "std", "n"]
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
          + β7(Age×Race×SES) + β8(Severity) + β9(Model) + ε

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
        from patsy import dmatrices
    except ImportError:
        return {"error": "statsmodels/patsy not installed. Run: pip install statsmodels"}

    valid = df.dropna(subset=[dep_var]).copy()
    valid[dep_var] = valid[dep_var].astype(float)

    # Build formula
    base = f"{dep_var} ~ age_code * race_code * ses_code"
    has_severity = "severity" in valid.columns and valid["severity"].nunique() > 1
    has_sofa = "sofa_score" in valid.columns and valid["sofa_score"].nunique() > 1
    if has_severity:
        base += " + C(severity)"
    elif has_sofa:
        base += " + sofa_score"
    if include_model and "model" in valid.columns and valid["model"].nunique() > 1:
        base += " + C(model)"

    try:
        _, design_matrix = dmatrices(base, valid, return_type="dataframe")
        matrix_rank = int(np.linalg.matrix_rank(design_matrix.values))
        design_columns = int(design_matrix.shape[1])
        rank_deficient = matrix_rank < design_columns

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
        if rank_deficient:
            interpretation = (
                "WARNING: Design matrix is rank deficient. Coefficients should be treated as "
                "diagnostic, not causally identified.\n"
                + interpretation
            )

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
            "formula": base,
            "design_rank": matrix_rank,
            "design_columns": design_columns,
            "rank_deficient": rank_deficient,
            "condition_number": float(model.condition_number),
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
    by_severity = df.groupby("severity")["is_refusal"].mean() if "severity" in df.columns else pd.Series(dtype=float)

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
        "by_severity": by_severity.to_dict(),
        "interpretation": interp,
    }


# ═══════════════════════════════════════════════
# 4. POST-HOC COMPARISONS
# ═══════════════════════════════════════════════

def tukey_hsd_conditions(df: pd.DataFrame) -> dict:
    """
    Tukey HSD post-hoc pairwise comparisons across observed conditions.
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
    For the corrected Study 2 design, panels are split by clinical severity
    so the 24-condition layout remains readable in manuscript figures.
    """
    if "severity" in desc_df.columns:
        plot_df = desc_df.copy()
        severity_order = [s for s in ["favorable", "moderate", "poor"] if s in set(plot_df["severity"])]
        profile_order = [
            (25, "White"),
            (25, "Black"),
            (75, "White"),
            (75, "Black"),
        ]
        profile_labels = ["25 W", "25 B", "75 W", "75 B"]
        ses_order = ["affluent", "low_income"]
        ses_labels = {"affluent": "Affluent", "low_income": "Low-income"}
        ses_colors = {"affluent": "#2E5984", "low_income": "#C75B3F"}

        fig, axes = plt.subplots(
            1,
            len(severity_order),
            figsize=(4.5 * len(severity_order), 4.2),
            sharey=True,
            squeeze=False,
        )
        axes = axes.flatten()
        x = np.arange(len(profile_order))
        width = 0.34

        for ax, severity in zip(axes, severity_order):
            sdf = plot_df[plot_df["severity"] == severity]
            for offset_index, ses in enumerate(ses_order):
                means = []
                ses_values = []
                for age, race in profile_order:
                    match = sdf[
                        (sdf["age"] == age)
                        & (sdf["race"] == race)
                        & (sdf["ses"] == ses)
                    ]
                    if match.empty:
                        means.append(np.nan)
                        ses_values.append(0.0)
                    else:
                        means.append(float(match["mean"].iloc[0]))
                        ses_values.append(float(match["se"].iloc[0]))

                offset = (offset_index - 0.5) * width
                bars = ax.bar(
                    x + offset,
                    means,
                    width,
                    yerr=ses_values,
                    capsize=3,
                    color=ses_colors[ses],
                    edgecolor="white",
                    linewidth=0.8,
                    label=ses_labels[ses],
                )

            ax.set_title(severity.title(), fontsize=11, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(profile_labels, fontsize=9)
            ax.axhline(y=4, color="#777777", linestyle="--", linewidth=0.8, alpha=0.45)
            ax.set_ylim(1, 7.2)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", color="#EAEAEA", linewidth=0.6)

        axes[0].set_ylabel("Mean prioritization rating (1-7)", fontsize=11)
        axes[0].legend(frameon=False, loc="upper left", fontsize=9)
        fig.suptitle(
            f"ICU Prioritization by Matched Clinical Severity{' - ' + model_name if model_name else ''}",
            fontsize=12,
            fontweight="bold",
        )
        fig.text(
            0.5,
            0.01,
            "Within each panel, diagnosis, SOFA score, vital signs, comorbidities, and prognosis are held constant. Bars show mean +/- SE.",
            ha="center",
            fontsize=9,
            color="#444444",
        )
        plt.tight_layout(rect=[0, 0.05, 1, 0.93])
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    fig_width = max(12, len(desc_df) * 0.55)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

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
        severity_label = ""
        if "severity" in row.index:
            severity_label = f"\n{str(row['severity'])[:4].title()}"
        labels.append(f"{row['condition_id']}\n{row['race']}/{row['age']}/{ses_label}{severity_label}")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Mean Likert Score (1-7)", fontsize=12)
    ax.set_title(
        f"Triage Prioritization by Patient Profile{' — ' + model_name if model_name else ''}",
        fontsize=14, fontweight="bold",
    )
    legacy_label_offset = max(0.18, float(desc_df["se"].fillna(0).max()) * 0.8)
    legacy_label_top = float((desc_df["mean"] + desc_df["se"].fillna(0)).max()) + legacy_label_offset
    ax.set_ylim(0, max(7.5, legacy_label_top + legacy_label_offset))
    ax.axhline(y=4, color="gray", linestyle="--", alpha=0.4, label="Neutral (4)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Value labels
    for bar, m, se in zip(bars, desc_df["mean"], desc_df["se"].fillna(0)):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            float(m) + float(se) + legacy_label_offset,
            f"{m:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

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
    max_rate = max(rates) if rates else 0
    label_offset = max(1.5, max_rate * 0.08)
    ax.set_ylim(0, max(10, max_rate + label_offset * 3.0))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i, (c, r) in enumerate(zip(conditions, rates)):
        if r > 0:
            tier_offset = label_offset * (1.0 + 0.55 * (i % 2))
            ax.text(i, r + tier_offset, f"{r:.1f}%", ha="center", va="bottom", fontsize=10)

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

    # Ensure clinical fields are present for older parsed files.
    from stimuli import STUDY2_CONDITIONS

    condition_map = {condition.condition_id: condition for condition in STUDY2_CONDITIONS}
    sofa_map = {cid: condition.sofa_score for cid, condition in condition_map.items()}
    severity_map = {cid: condition.severity for cid, condition in condition_map.items()}
    severity_code_map = {cid: condition.severity_code for cid, condition in condition_map.items()}
    legacy_sofa_map = {
        "C1": 3, "C2": 4, "C3": 3, "C4": 4,
        "C5": 10, "C6": 11, "C7": 10, "C8": 12,
    }
    sofa_map.update(legacy_sofa_map)
    if "sofa_score" not in df.columns:
        df["sofa_score"] = df["condition_id"].map(sofa_map)
    else:
        df["sofa_score"] = df["sofa_score"].fillna(df["condition_id"].map(sofa_map))
    if "severity" not in df.columns:
        df["severity"] = df["condition_id"].map(severity_map)
    else:
        df["severity"] = df["severity"].fillna(df["condition_id"].map(severity_map))
    if "severity_code" not in df.columns:
        df["severity_code"] = df["condition_id"].map(severity_code_map)
    else:
        df["severity_code"] = df["severity_code"].fillna(df["condition_id"].map(severity_code_map))

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
        fig = plot_condition_bars(
            desc, model_name,
            save_path=os.path.join(output_dir, f"study2_conditions_{model_name}.png"),
        )
        plt.close(fig)
        fig = plot_coefficient_forest(
            reg,
            save_path=os.path.join(output_dir, f"study2_coefficients_{model_name}.png"),
        )
        plt.close(fig)
        if "error" not in reg_debiased and reg_debiased:
            fig = plot_coefficient_forest(
                reg_debiased,
                save_path=os.path.join(output_dir, f"study2_debiased_coefficients_{model_name}.png"),
            )
            plt.close(fig)
        for f1, f2 in [("race", "age"), ("race", "ses"), ("age", "ses")]:
            fig = plot_interaction(
                mdf, f1, f2, model_name,
                save_path=os.path.join(
                    output_dir,
                    f"study2_interaction_{f1}_{f2}_{model_name}.png",
                ),
            )
            plt.close(fig)
        fig = plot_refusal_rates(
            refusals, model_name,
            save_path=os.path.join(output_dir, f"study2_refusals_{model_name}.png"),
        )
        plt.close(fig)

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
            fig = plot_coefficient_forest(
                cross_reg,
                save_path=os.path.join(output_dir, "study2_coefficients_all_models.png"),
            )
            plt.close(fig)
        
        # Cross-model regression (Debiased)
        if "debiased_score" in df.columns:
            cross_reg_debiased = factorial_regression(df, include_model=True, dep_var="debiased_score")
            all_results["cross_model_regression_debiased"] = cross_reg_debiased
            if "error" not in cross_reg_debiased:
                logger.info(f"Cross-model debiased regression:\n{cross_reg_debiased['interpretation']}")
                fig = plot_coefficient_forest(
                    cross_reg_debiased,
                    save_path=os.path.join(output_dir, "study2_debiased_coefficients_all_models.png"),
                )
                plt.close(fig)

    plt.close("all")
    return all_results
