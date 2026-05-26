"""
generate_unified_charts.py - Recreates every single analytical plot and generates
unified comparative side-by-side charts representing all six frontier models
at journal publication quality (300 DPI).
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication quality styling
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

# Standardized model names in order of presentation. The main function narrows
# this list to the models present in the current parsed dataset.
MODEL_ORDER = [
    "GPT-5.5", 
    "Claude Opus 4.7", 
    "Claude Sonnet 4.6", 
    "Gemini 3.5 Flash", 
    "Gemini 3.1 Pro", 
    "Gemma 4 31B Free",
    "Gemini 3 Flash",
    "DeepSeek v4 Pro",
    "DeepSeek v4 Flash"
]

def generate_study1_unified_inclusivity(df1: pd.DataFrame, output_dir: str):
    """
    Computes and plots the Inclusivity Index for all models side-by-side in a single grouped bar chart.
    """
    print("Generating Study 1 Unified Inclusivity Index plot...")
    # Add stereotypical gender mapping
    stereo_map = {"high": 0.0, "support": 1.0}
    df = df1[df1["role_level"].isin(["high", "support"])].copy()
    df = df.dropna(subset=["parsed_gender"])
    df["stereotypical_gender"] = df["role_level"].map(stereo_map)
    df["distance"] = (df["stereotypical_gender"] - df["parsed_gender"]).abs()

    # Calculate per-phrase index
    phrase_agg = df.groupby(["model", "phrase_id", "role_level"])["distance"].mean().reset_index()
    
    # Calculate category means and standard errors per model
    stats_list = []
    for model in MODEL_ORDER:
        for level in ["high", "support"]:
            vals = phrase_agg[(phrase_agg["model"] == model) & (phrase_agg["role_level"] == level)]["distance"].values
            if len(vals) > 0:
                mean_val = np.mean(vals)
                se_val = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
                stats_list.append({
                    "Model": model,
                    "Role Level": "High-Status (Leadership)" if level == "high" else "Support-Status (Junior)",
                    "Inclusivity Index": mean_val,
                    "SE": se_val
                })
    
    plot_df = pd.DataFrame(stats_list)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(MODEL_ORDER))
    width = 0.35

    # Group colors: Grayscale / purposeful highlight
    color_high = "#4d4d4d"
    color_support = "#0072B2"

    high_data = plot_df[plot_df["Role Level"] == "High-Status (Leadership)"]
    support_data = plot_df[plot_df["Role Level"] == "Support-Status (Junior)"]

    # Robust reindexing to MODEL_ORDER to prevent dimension mismatches and ensure perfect alignment
    high_data = high_data.set_index("Model").reindex(MODEL_ORDER).reset_index()
    support_data = support_data.set_index("Model").reindex(MODEL_ORDER).reset_index()

    rects1 = ax.bar(x - width/2, high_data["Inclusivity Index"], width, 
                    yerr=high_data["SE"], capsize=0, label="High-Status Roles ($I_{high}$)", 
                    color=color_high, alpha=0.85)
    rects2 = ax.bar(x + width/2, support_data["Inclusivity Index"], width, 
                    yerr=support_data["SE"], capsize=0, label="Support-Status Roles ($I_{support}$)", 
                    color=color_support, alpha=0.85)

    ax.set_ylabel("Mean Inclusivity Index ($I$)", fontsize=11, fontweight="bold")
    ax.set_title("Gender Inclusivity Index ($I$) Comparison Side-by-Side", fontsize=13, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER, rotation=15)
    ax.set_ylim(0, 0.5)  # Perfect parity is 0.5
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.7, label="Perfect Gender Parity ($I=0.5$)")
    ax.legend(loc="upper left")
    
    # Clean spines
    sns.despine(trim=True, offset=5)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "study1_unified_inclusivity.png"), dpi=300)
    plt.close(fig)

def generate_study1_unified_white_male(df1: pd.DataFrame, output_dir: str):
    """
    Plots the percentage of White Male name attributions across High-Status vs. Support-Status for all models.
    """
    print("Generating Study 1 Unified White Male Attribution plot...")
    df = df1[df1["role_level"].isin(["high", "support"])].copy()
    df = df.dropna(subset=["parsed_gender", "inferred_race"])
    
    # Coded as White Male if gender == 0 (Male) and race == 'white'
    df["is_white_male"] = ((df["parsed_gender"] == 0) & (df["inferred_race"] == "white")).astype(float)

    stats_list = []
    for model in MODEL_ORDER:
        for level in ["high", "support"]:
            subset = df[(df["model"] == model) & (df["role_level"] == level)]
            if len(subset) > 0:
                pct = subset["is_white_male"].mean() * 100
                n = len(subset)
                # SE of proportion: sqrt(p*(1-p)/n)
                p = subset["is_white_male"].mean()
                se_pct = np.sqrt(p * (1 - p) / n) * 100 if n > 1 else 0.0
                stats_list.append({
                    "Model": model,
                    "Role Level": "High-Status (Leadership)" if level == "high" else "Support-Status (Junior)",
                    "Percentage": pct,
                    "SE": se_pct
                })
    
    plot_df = pd.DataFrame(stats_list)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(MODEL_ORDER))
    width = 0.35

    color_high = "#4d4d4d"   # Grayscale
    color_support = "#D55E00"  # Orange highlight

    high_data = plot_df[plot_df["Role Level"] == "High-Status (Leadership)"]
    support_data = plot_df[plot_df["Role Level"] == "Support-Status (Junior)"]

    # Robust reindexing to MODEL_ORDER to prevent dimension mismatches and ensure perfect alignment
    high_data = high_data.set_index("Model").reindex(MODEL_ORDER).reset_index()
    support_data = support_data.set_index("Model").reindex(MODEL_ORDER).reset_index()

    rects1 = ax.bar(x - width/2, high_data["Percentage"], width, 
                    yerr=high_data["SE"], capsize=0, label="High-Status (Leadership)", 
                    color=color_high, alpha=0.85)
    rects2 = ax.bar(x + width/2, support_data["Percentage"], width, 
                    yerr=support_data["SE"], capsize=0, label="Support-Status (Junior)", 
                    color=color_support, alpha=0.85)

    ax.set_ylabel("White Male Attribution (%)", fontsize=11, fontweight="bold")
    ax.set_title("Unified White Male Attribution Skew Across Models", fontsize=13, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER, rotation=15)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right")
    
    sns.despine(trim=True, offset=5)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "study1_unified_white_male.png"), dpi=300)
    plt.close(fig)

def generate_study2_unified_priorities(df2: pd.DataFrame, output_dir: str):
    """
    Grouped bar chart comparing the mean clinical triage prioritization rating for all eight conditions across all models.
    """
    print("Generating Study 2 Unified Triage Priorities plot...")
    df = df2[~df2["is_refusal"]].copy()
    df["likert_score"] = pd.to_numeric(df["likert_score"], errors="coerce")
    df = df.dropna(subset=["likert_score"])

    # Define Condition names with string keys matching dataset condition_id values
    cond_labels = {
        "C1": "C1 (Y, W, A)",
        "C2": "C2 (Y, W, L)",
        "C3": "C3 (Y, B, A)",
        "C4": "C4 (Y, B, L)",
        "C5": "C5 (E, W, A)",
        "C6": "C6 (E, W, L)",
        "C7": "C7 (E, B, A)",
        "C8": "C8 (E, B, L)"
    }
    # (Y=Young, E=Elderly; W=White, B=Black; A=Affluent, L=Low-Income)

    df["Condition"] = df["condition_id"].map(cond_labels)
    cond_order = [cond_labels[f"C{i}"] for i in range(1, 9)]

    # Calculate means and SE per condition per model
    stats_list = []
    for model in MODEL_ORDER:
        for cond_id in range(1, 9):
            cond_str = f"C{cond_id}"
            subset = df[(df["model"] == model) & (df["condition_id"] == cond_str)]
            if len(subset) > 0:
                mean_val = subset["likert_score"].mean()
                se_val = subset["likert_score"].std() / np.sqrt(len(subset)) if len(subset) > 1 else 0.0
                stats_list.append({
                    "Model": model,
                    "Condition": cond_labels[cond_str],
                    "Likert Score": mean_val,
                    "SE": se_val
                })
    
    plot_df = pd.DataFrame(stats_list)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # We will plot with Model on X-axis, and Conditions grouped
    x = np.arange(len(MODEL_ORDER))
    width = 0.09  # Spacing out 8 bars

    # Color palette: elegant transition from Teal (C1) to Orange-Red (C8)
    colors = [
        "#2A9D8F",  # C1 - Teal
        "#5E9CA0",  # C2
        "#8CB2B4",  # C3
        "#B8C9C8",  # C4
        "#E9C46A",  # C5 - Muted Yellow
        "#F4A261",  # C6
        "#E76F51",  # C7 - Coral
        "#D94E34"   # C8 - Deep Terracotta Red
    ]

    for idx, cond in enumerate(cond_order):
        cond_data = plot_df[plot_df["Condition"] == cond]
        # Realign to match model order
        cond_data = cond_data.set_index("Model").reindex(MODEL_ORDER).reset_index()
        
        offset = (idx - 3.5) * width
        ax.bar(x + offset, cond_data["Likert Score"], width, 
               yerr=cond_data["SE"], capsize=0, label=cond,
               color=colors[idx], alpha=0.85)

    ax.set_ylabel("Mean Triage Prioritization Score (1-7)", fontsize=11, fontweight="bold")
    ax.set_title("Unified Scarce Clinical Resource Prioritization Across All Conditions (C1-C8)", fontsize=13, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER, rotation=15)
    ax.set_ylim(1, 7.2)  # Full Likert range from 1 to 7 with breathing room, preventing clipping
    
    # Put legend outside or on top
    ax.legend(title="Patient Demographic Profiles (Age, Race, SES)", loc="upper right", ncol=2)
    
    sns.despine(trim=True, offset=5)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "study2_unified_priorities.png"), dpi=300)
    plt.close(fig)

def main():
    global MODEL_ORDER
    parser = argparse.ArgumentParser(description="Generate unified comparison charts.")
    parser.add_argument("--study1-csv", default=os.path.join("data", "study1_parsed.csv"))
    parser.add_argument("--study2-csv", default=os.path.join("data", "study2_parsed.csv"))
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)


    print("Reading parsed data files...")
    df1 = pd.read_csv(args.study1_csv)
    df2 = pd.read_csv(args.study2_csv)

    # Map raw model identifiers in the data to pretty capitalized names used in MODEL_ORDER
    model_map = {
        "openai-gpt-5.5": "GPT-5.5",
        "anthropic-claude-opus-4.7": "Claude Opus 4.7",
        "anthropic-claude-sonnet-4.6": "Claude Sonnet 4.6",
        "google-gemini-3.5-flash": "Gemini 3.5 Flash",
        "google-gemini-3.1-pro": "Gemini 3.1 Pro",
        "google-gemma-4-free": "Gemma 4 31B Free",
        "google-gemini-3-flash": "Gemini 3 Flash",
        "deepseek-v4-pro": "DeepSeek v4 Pro",
        "deepseek-v4-flash": "DeepSeek v4 Flash",
        "gpt-5.5": "GPT-5.5",
        "claude-opus-4.7": "Claude Opus 4.7",
        "claude-sonnet-4.6": "Claude Sonnet 4.6",
        "gemini-3-flash": "Gemini 3 Flash",
        "gemini-3.1-pro": "Gemini 3.1 Pro",
        "deepseek-v4": "DeepSeek v4 Pro",
        "deepseek-v4-flash": "DeepSeek v4 Flash"
    }
    df1["model"] = df1["model"].map(model_map).fillna(df1["model"])
    df2["model"] = df2["model"].map(model_map).fillna(df2["model"])
    observed = set(df1["model"].dropna()).intersection(set(df2["model"].dropna()))
    MODEL_ORDER = [model for model in MODEL_ORDER if model in observed]
    if not MODEL_ORDER:
        raise SystemExit("No overlapping models found in parsed Study 1 and Study 2 files.")

    # Generate unified comparison plots
    generate_study1_unified_inclusivity(df1, output_dir)
    generate_study1_unified_white_male(df1, output_dir)
    generate_study2_unified_priorities(df2, output_dir)
    
    print("Unified comparative visualizations generated successfully!")

if __name__ == "__main__":
    main()
