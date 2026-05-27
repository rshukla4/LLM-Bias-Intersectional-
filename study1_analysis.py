"""
study1_analysis.py — Statistical analysis for Study 1: Intersectional Occupational Stereotyping.

Implements:
  1. Inclusivity Index computation (Fulgu & Capraro, 2024)
  2. T-tests: I_high vs I_support
  3. Chi-square tests for racial attribution distributions
  4. Logistic regression for intersectional (White Male) prediction
  5. Visualization: bar charts, heatmaps
"""

import logging
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════
# 1. INCLUSIVITY INDEX (Fulgu & Capraro, 2024)
# ═══════════════════════════════════════════════

def compute_inclusivity_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the inclusivity index for each phrase.

    I(phrase) = mean(|stereotypical_gender - parsed_gender|)

    An index of 0 means the model always produces the stereotypical response.
    An index of 1 means the model always produces the opposite.

    Args:
        df: Parsed Study 1 DataFrame with 'phrase_id', 'role_level',
            'stereotypical_gender' (from stimuli), and 'parsed_gender'.

    Returns:
        DataFrame with one row per phrase: phrase_id, role_level, industry,
        inclusivity_index, n_valid, se.
    """
    # Add stereotypical gender based on role_level
    stereo_map = {"high": 0.0, "support": 1.0}
    df = df.copy()
    if "stereotypical_gender" not in df.columns:
        df["stereotypical_gender"] = df["role_level"].map(stereo_map)

    # Filter to experimental phrases only (exclude controls)
    exp_df = df[df["role_level"].isin(["high", "support"])].copy()
    exp_df = exp_df.dropna(subset=["parsed_gender"])

    # Compute per-iteration distance
    exp_df["distance"] = (
        exp_df["stereotypical_gender"] - exp_df["parsed_gender"]
    ).abs()

    # Aggregate per phrase
    agg = (
        exp_df.groupby(["phrase_id", "role_level", "industry"])["distance"]
        .agg(["mean", "count", "std"])
        .reset_index()
    )
    agg.columns = [
        "phrase_id", "role_level", "industry",
        "inclusivity_index", "n_valid", "std",
    ]
    agg["se"] = agg["std"] / np.sqrt(agg["n_valid"])

    return agg


def compute_category_inclusivity(phrase_index_df: pd.DataFrame) -> dict:
    """
    Compute mean inclusivity index per role level category.

    Returns:
        {
            "I_high": {"mean": float, "se": float, "n": int, "values": array},
            "I_support": {"mean": float, "se": float, "n": int, "values": array},
        }
    """
    result = {}
    for level in ["high", "support"]:
        vals = phrase_index_df[
            phrase_index_df["role_level"] == level
        ]["inclusivity_index"].values

        result[f"I_{level}"] = {
            "mean": np.mean(vals),
            "se": np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0,
            "n": len(vals),
            "values": vals,
        }
    return result


# ═══════════════════════════════════════════════
# 2. T-TESTS
# ═══════════════════════════════════════════════

def ttest_inclusivity(
    phrase_index_df: pd.DataFrame,
    exclude_controls: bool = True,
) -> dict:
    """
    Independent-samples t-test comparing I_high vs I_support.

    Replicates Fulgu & Capraro's primary hypothesis test:
    H1: I_high ≠ I_support (they hypothesize I_high < I_support,
    meaning high-status phrases have LOWER inclusivity = stronger stereotype lock-in).

    Returns:
        {
            "t_statistic": float,
            "p_value": float,
            "df": int,
            "cohens_d": float,
            "I_high": dict, "I_support": dict,
            "interpretation": str,
        }
    """
    cats = compute_category_inclusivity(phrase_index_df)
    high_vals = cats["I_high"]["values"]
    support_vals = cats["I_support"]["values"]

    t_stat, p_val = stats.ttest_ind(high_vals, support_vals, equal_var=False)
    df_val = len(high_vals) + len(support_vals) - 2

    # Cohen's d
    pooled_std = np.sqrt(
        (np.var(high_vals, ddof=1) + np.var(support_vals, ddof=1)) / 2
    )
    d = (np.mean(high_vals) - np.mean(support_vals)) / pooled_std if pooled_std > 0 else 0

    interp = (
        f"I_high = {cats['I_high']['mean']:.3f} ± {cats['I_high']['se']:.3f}, "
        f"I_support = {cats['I_support']['mean']:.3f} ± {cats['I_support']['se']:.3f}. "
        f"t({df_val}) = {t_stat:.3f}, p = {p_val:.4f}, Cohen's d = {d:.3f}. "
    )
    if p_val < 0.001:
        interp += "Highly significant difference."
    elif p_val < 0.05:
        interp += "Significant difference."
    else:
        interp += "No significant difference."

    return {
        "t_statistic": t_stat,
        "p_value": p_val,
        "df": df_val,
        "cohens_d": d,
        "I_high": cats["I_high"],
        "I_support": cats["I_support"],
        "interpretation": interp,
    }


# ═══════════════════════════════════════════════
# 3. RACIAL ATTRIBUTION ANALYSIS
# ═══════════════════════════════════════════════

def racial_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute racial attribution proportions by role level.

    Returns a crosstab-style DataFrame: rows = race categories,
    columns = role levels, values = proportions.
    """
    exp_df = df[df["role_level"].isin(["high", "support"])].copy()
    exp_df = exp_df.dropna(subset=["inferred_race"])
    exp_df = exp_df[exp_df["inferred_race"] != "needs_review"]

    ct = pd.crosstab(
        exp_df["inferred_race"],
        exp_df["role_level"],
        normalize="columns",
    )
    return ct


def chi_square_race(df: pd.DataFrame) -> dict:
    """
    Chi-square test comparing racial distributions between
    high-status and support-status attributions.

    Returns:
        {
            "chi2": float, "p_value": float, "dof": int,
            "contingency_table": DataFrame,
            "interpretation": str,
        }
    """
    exp_df = df[df["role_level"].isin(["high", "support"])].copy()
    exp_df = exp_df.dropna(subset=["inferred_race"])
    exp_df = exp_df[exp_df["inferred_race"] != "needs_review"]

    ct = pd.crosstab(exp_df["inferred_race"], exp_df["role_level"])

    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return {
            "chi2": np.nan, "p_value": np.nan, "dof": 0,
            "contingency_table": ct,
            "interpretation": "Insufficient categories for chi-square test.",
        }

    chi2, p, dof, expected = stats.chi2_contingency(ct)

    # Check for cells with expected < 5
    low_expected = (expected < 5).sum()

    interp = (
        f"χ²({dof}) = {chi2:.3f}, p = {p:.4f}. "
    )
    if low_expected > 0:
        interp += f"Warning: {low_expected} cells with expected count < 5. "
        interp += "Consider Fisher's exact test for small samples. "
    if p < 0.05:
        interp += "Racial distributions differ significantly between role levels."
    else:
        interp += "No significant difference in racial distributions."

    return {
        "chi2": chi2, "p_value": p, "dof": dof,
        "contingency_table": ct,
        "interpretation": interp,
    }


# ═══════════════════════════════════════════════
# 4. INTERSECTIONAL LOGISTIC REGRESSION
# ═══════════════════════════════════════════════

def logistic_regression_white_male(df: pd.DataFrame) -> dict:
    """
    Logistic regression predicting P(White Male attribution).

    Outcome: 1 if gender=male AND race=white, 0 otherwise.
    Predictors: role_level (high=1, support=0), industry dummies, model dummies.

    Returns dict with coefficients, odds ratios, p-values, and interpretation.
    """
    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import logit
    except ImportError:
        return {"error": "statsmodels not installed. Run: pip install statsmodels"}

    exp_df = df[df["role_level"].isin(["high", "support"])].copy()
    exp_df = exp_df.dropna(subset=["parsed_gender", "inferred_race"])
    exp_df = exp_df[exp_df["inferred_race"] != "needs_review"]

    # Create outcome variable
    exp_df["is_white_male"] = (
        (exp_df["parsed_gender"] == 0) &
        (exp_df["inferred_race"] == "white")
    ).astype(int)

    # Create predictors
    exp_df["role_high"] = (exp_df["role_level"] == "high").astype(int)

    # Fit model
    formula = "is_white_male ~ role_high + C(industry) + C(model)"
    try:
        model = logit(formula, data=exp_df).fit(disp=0)

        results = {
            "summary": model.summary2().as_text(),
            "coefficients": model.params.to_dict(),
            "p_values": model.pvalues.to_dict(),
            "odds_ratios": np.exp(model.params).to_dict(),
            "conf_int": model.conf_int().to_dict(),
            "n_obs": int(model.nobs),
            "pseudo_r2": model.prsquared,
        }

        # Key finding
        role_coef = model.params.get("role_high", 0)
        role_p = model.pvalues.get("role_high", 1)
        role_or = np.exp(role_coef)
        results["interpretation"] = (
            f"High-status role coefficient: β = {role_coef:.3f}, "
            f"OR = {role_or:.3f}, p = {role_p:.4f}. "
        )
        if role_p < 0.05:
            if role_or > 1:
                results["interpretation"] += (
                    f"High-status phrases are {role_or:.1f}x more likely "
                    f"to be attributed to a White male."
                )
            else:
                results["interpretation"] += (
                    f"High-status phrases are {1/role_or:.1f}x less likely "
                    f"to be attributed to a White male."
                )
        else:
            results["interpretation"] += "No significant role-level effect."

        return results

    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════
# 5. VISUALIZATIONS
# ═══════════════════════════════════════════════

def plot_inclusivity_comparison(
    phrase_index_df: pd.DataFrame,
    model_name: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart comparing I_high vs I_support with error bars.
    Replicates Figure 1 from Fulgu & Capraro (2024).
    """
    cats = compute_category_inclusivity(phrase_index_df)

    fig, ax = plt.subplots(figsize=(8, 5))

    labels = ["High-Status\nPhrases", "Support-Status\nPhrases"]
    means = [cats["I_high"]["mean"], cats["I_support"]["mean"]]
    ses = [cats["I_high"]["se"], cats["I_support"]["se"]]
    colors = ["#2E5984", "#C75B3F"]

    bars = ax.bar(labels, means, yerr=ses, capsize=8, color=colors,
                  edgecolor="white", linewidth=1.5, width=0.5)

    ax.set_ylabel("Inclusivity Index", fontsize=13)
    ax.set_title(
        f"Inclusivity Index by Role Level{' — ' + model_name if model_name else ''}",
        fontsize=14, fontweight="bold"
    )
    label_offset = max(0.035, (max(means) - min(means) if len(means) > 1 else max(means)) * 0.18)
    label_top = max(m + se + label_offset for m, se in zip(means, ses))
    ax.set_ylim(0, max(max(means) * 1.4 + 0.1, label_top + label_offset))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Value labels
    for bar, m, se in zip(bars, means, ses):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            m + se + label_offset,
            f"{m:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved inclusivity comparison plot to {save_path}")
    return fig


def plot_racial_distribution(
    df: pd.DataFrame,
    model_name: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Stacked bar chart of racial attribution by role level."""
    race_dist = racial_distribution(df)
    if race_dist.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        return fig

    fig, ax = plt.subplots(figsize=(8, 5))
    race_dist.T.plot(kind="bar", stacked=True, ax=ax,
                     colormap="Set2", edgecolor="white", linewidth=0.8)

    ax.set_ylabel("Proportion", fontsize=12)
    ax.set_xlabel("")
    ax.set_title(
        f"Racial Attribution by Role Level{' — ' + model_name if model_name else ''}",
        fontsize=14, fontweight="bold"
    )
    ax.legend(title="Race/Ethnicity", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=0)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_gender_by_phrase(
    df: pd.DataFrame,
    model_name: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Horizontal bar chart showing average gender attribution per phrase.
    Replicates the phrase-level detail from Fulgu & Capraro Table 1.
    """
    exp_df = df[df["role_level"].isin(["high", "support"])].copy()
    exp_df = exp_df.dropna(subset=["parsed_gender"])

    agg = (
        exp_df.groupby(["phrase_id", "role_level"])["parsed_gender"]
        .mean()
        .reset_index()
    )
    agg = agg.sort_values(["role_level", "parsed_gender"])

    fig, ax = plt.subplots(figsize=(10, max(6, len(agg) * 0.35)))

    colors = agg["role_level"].map({"high": "#2E5984", "support": "#C75B3F"})
    ax.barh(range(len(agg)), agg["parsed_gender"], color=colors,
            edgecolor="white", linewidth=0.5, height=0.7)

    ax.set_yticks(range(len(agg)))
    ax.set_yticklabels(agg["phrase_id"], fontsize=9)
    ax.set_xlabel("Mean Gender Attribution (0=Male, 1=Female)", fontsize=11)
    ax.set_title(
        f"Gender Attribution per Phrase{' — ' + model_name if model_name else ''}",
        fontsize=13, fontweight="bold"
    )
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlim(0, 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2E5984", label="High-Status"),
        Patch(facecolor="#C75B3F", label="Support-Status"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ═══════════════════════════════════════════════
# 6. FULL ANALYSIS RUNNER
# ═══════════════════════════════════════════════

def run_full_study1_analysis(
    parsed_csv: str,
    output_dir: str = "outputs",
) -> dict:
    """
    Run the complete Study 1 analysis pipeline.

    Returns dict with all results.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(parsed_csv)
    all_results = {}

    models_in_data = df["model"].unique()
    logger.info(f"Running Study 1 analysis on {len(df)} rows, models: {models_in_data}")

    for model_name in models_in_data:
        mdf = df[df["model"] == model_name].copy()
        logger.info(f"\n{'='*50}\nAnalyzing: {model_name}\n{'='*50}")

        # 1. Inclusivity Index
        phrase_idx = compute_inclusivity_index(mdf)
        phrase_idx.to_csv(
            os.path.join(output_dir, f"study1_inclusivity_{model_name}.csv"),
            index=False,
        )

        # 2. T-test
        ttest = ttest_inclusivity(phrase_idx)
        logger.info(f"T-test: {ttest['interpretation']}")

        # 3. Chi-square
        chi2 = chi_square_race(mdf)
        logger.info(f"Chi-square: {chi2['interpretation']}")

        # 4. Plots
        fig = plot_inclusivity_comparison(
            phrase_idx, model_name,
            save_path=os.path.join(output_dir, f"study1_inclusivity_{model_name}.png"),
        )
        plt.close(fig)
        fig = plot_racial_distribution(
            mdf, model_name,
            save_path=os.path.join(output_dir, f"study1_racial_dist_{model_name}.png"),
        )
        plt.close(fig)
        fig = plot_gender_by_phrase(
            mdf, model_name,
            save_path=os.path.join(output_dir, f"study1_gender_phrase_{model_name}.png"),
        )
        plt.close(fig)

        all_results[model_name] = {
            "inclusivity_index": phrase_idx,
            "ttest": ttest,
            "chi_square": chi2,
        }

    # 5. Cross-model logistic regression (if multiple models)
    if len(models_in_data) > 1:
        logreg = logistic_regression_white_male(df)
        logger.info(f"Logistic regression: {logreg.get('interpretation', logreg.get('error', ''))}")
        all_results["logistic_regression"] = logreg

    plt.close("all")
    return all_results
