"""
Study 1 publication charts.

Study 2 unified charts are deferred and archived under
archive/study2_deferred_20260526/scripts/generate_unified_charts_two_study.py.
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme(style="ticks", context="paper", font="serif")
plt.rcParams.update({
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.family": "serif",
    "axes.labelsize": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "figure.titlesize": 14,
    "figure.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


MODEL_ORDER = [
    "GPT-5.5",
    "Claude Opus 4.7",
    "Claude Sonnet 4.6",
    "Gemini 3.5 Flash",
    "Gemini 3.1 Pro",
    "Gemini 3 Flash",
    "DeepSeek v4 Pro",
    "DeepSeek v4 Flash",
    "Gemma 4 31B Free",
]

MODEL_TICK_LABELS = {
    "GPT-5.5": "GPT-5.5",
    "Claude Opus 4.7": "Opus\n4.7",
    "Claude Sonnet 4.6": "Sonnet\n4.6",
    "Gemini 3.5 Flash": "Gemini\n3.5F",
    "Gemini 3.1 Pro": "Gemini\n3.1P",
    "Gemini 3 Flash": "Gemini\n3F",
    "DeepSeek v4 Pro": "DeepSeek\nPro",
    "DeepSeek v4 Flash": "DeepSeek\nFlash",
    "Gemma 4 31B Free": "Gemma\n4-31B",
}

MODEL_MAP = {
    "openai-gpt-5.5": "GPT-5.5",
    "anthropic-claude-opus-4.7": "Claude Opus 4.7",
    "anthropic-claude-sonnet-4.6": "Claude Sonnet 4.6",
    "google-gemini-3.5-flash": "Gemini 3.5 Flash",
    "google-gemini-3.1-pro": "Gemini 3.1 Pro",
    "google-gemini-3-flash": "Gemini 3 Flash",
    "deepseek-v4-pro": "DeepSeek v4 Pro",
    "deepseek-v4-flash": "DeepSeek v4 Flash",
    "google-gemma-4-free": "Gemma 4 31B Free",
}


def model_tick_labels(model_order: list[str]) -> list[str]:
    return [MODEL_TICK_LABELS.get(model, model) for model in model_order]


def observed_model_order(df: pd.DataFrame) -> list[str]:
    observed = set(df["model"].dropna())
    ordered = [model for model in MODEL_ORDER if model in observed]
    if not ordered:
        raise SystemExit("No recognized Study 1 models found in parsed data.")
    return ordered


def generate_study1_unified_inclusivity(df1: pd.DataFrame, output_dir: str, model_order: list[str]):
    print("Generating Study 1 unified inclusivity index plot...")
    stereo_map = {"high": 0.0, "support": 1.0}
    df = df1[df1["role_level"].isin(["high", "support"])].copy()
    df = df.dropna(subset=["parsed_gender"])
    df["stereotypical_gender"] = df["role_level"].map(stereo_map)
    df["distance"] = (df["stereotypical_gender"] - df["parsed_gender"]).abs()

    phrase_agg = df.groupby(["model", "phrase_id", "role_level"])["distance"].mean().reset_index()
    stats = []
    for model in model_order:
        for level in ["high", "support"]:
            vals = phrase_agg[(phrase_agg["model"] == model) & (phrase_agg["role_level"] == level)]["distance"].values
            if len(vals) > 0:
                stats.append({
                    "Model": model,
                    "Role Level": "High-Status (Leadership)" if level == "high" else "Support-Status (Junior)",
                    "Inclusivity Index": float(np.mean(vals)),
                    "SE": float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0,
                })

    plot_df = pd.DataFrame(stats)
    fig, ax = plt.subplots(figsize=(max(10.5, len(model_order) * 1.15), 5.0))
    x = np.arange(len(model_order))
    width = 0.35

    high_data = (
        plot_df[plot_df["Role Level"] == "High-Status (Leadership)"]
        .set_index("Model")
        .reindex(model_order)
        .reset_index()
    )
    support_data = (
        plot_df[plot_df["Role Level"] == "Support-Status (Junior)"]
        .set_index("Model")
        .reindex(model_order)
        .reset_index()
    )

    ax.bar(
        x - width / 2,
        high_data["Inclusivity Index"],
        width,
        yerr=high_data["SE"],
        capsize=0,
        label="High-Status Roles",
        color="#4d4d4d",
        alpha=0.85,
    )
    ax.bar(
        x + width / 2,
        support_data["Inclusivity Index"],
        width,
        yerr=support_data["SE"],
        capsize=0,
        label="Support-Status Roles",
        color="#0072B2",
        alpha=0.85,
    )

    ax.set_ylabel("Mean Inclusivity Index")
    ax.set_title("Study 1 Gender Inclusivity Index")
    ax.set_xticks(x)
    ax.set_xticklabels(model_tick_labels(model_order), rotation=0, ha="center", fontsize=8.5)
    ax.set_ylim(0, 0.5)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.7, label="Perfect Gender Parity")
    ax.legend(loc="upper left")
    sns.despine(trim=True, offset=5)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(os.path.join(output_dir, "study1_unified_inclusivity.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_study1_unified_white_male(df1: pd.DataFrame, output_dir: str, model_order: list[str]):
    print("Generating Study 1 unified White male attribution plot...")
    df = df1[df1["role_level"].isin(["high", "support"])].copy()
    df = df.dropna(subset=["parsed_gender", "inferred_race"])
    df["is_white_male"] = ((df["parsed_gender"] == 0) & (df["inferred_race"] == "white")).astype(float)

    stats = []
    for model in model_order:
        for level in ["high", "support"]:
            subset = df[(df["model"] == model) & (df["role_level"] == level)]
            if len(subset) > 0:
                p = float(subset["is_white_male"].mean())
                stats.append({
                    "Model": model,
                    "Role Level": "High-Status (Leadership)" if level == "high" else "Support-Status (Junior)",
                    "Percentage": p * 100,
                    "SE": np.sqrt(p * (1 - p) / len(subset)) * 100 if len(subset) > 1 else 0.0,
                })

    plot_df = pd.DataFrame(stats)
    fig, ax = plt.subplots(figsize=(max(10.5, len(model_order) * 1.15), 5.0))
    x = np.arange(len(model_order))
    width = 0.35

    high_data = (
        plot_df[plot_df["Role Level"] == "High-Status (Leadership)"]
        .set_index("Model")
        .reindex(model_order)
        .reset_index()
    )
    support_data = (
        plot_df[plot_df["Role Level"] == "Support-Status (Junior)"]
        .set_index("Model")
        .reindex(model_order)
        .reset_index()
    )

    ax.bar(
        x - width / 2,
        high_data["Percentage"],
        width,
        yerr=high_data["SE"],
        capsize=0,
        label="High-Status",
        color="#4d4d4d",
        alpha=0.85,
    )
    ax.bar(
        x + width / 2,
        support_data["Percentage"],
        width,
        yerr=support_data["SE"],
        capsize=0,
        label="Support-Status",
        color="#D55E00",
        alpha=0.85,
    )

    ax.set_ylabel("White Male Attribution (%)")
    ax.set_title("Study 1 White Male Attribution Across Role Levels")
    ax.set_xticks(x)
    ax.set_xticklabels(model_tick_labels(model_order), rotation=0, ha="center", fontsize=8.5)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right")
    sns.despine(trim=True, offset=5)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(os.path.join(output_dir, "study1_unified_white_male.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate Study 1 comparison charts.")
    parser.add_argument("--study1-csv", default=os.path.join("data", "study1_parsed.csv"))
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df1 = pd.read_csv(args.study1_csv)
    df1["model"] = df1["model"].map(MODEL_MAP).fillna(df1["model"])
    model_order = observed_model_order(df1)

    generate_study1_unified_inclusivity(df1, args.output_dir, model_order)
    generate_study1_unified_white_male(df1, args.output_dir, model_order)
    print("Study 1 unified visualizations generated successfully.")


if __name__ == "__main__":
    main()
