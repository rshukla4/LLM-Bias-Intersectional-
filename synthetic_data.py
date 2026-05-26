"""
synthetic_data.py — Generate synthetic data to test the full analysis pipeline
without making real API calls. Simulates plausible LLM bias patterns.
"""

import os
import csv
import random
import numpy as np
from datetime import datetime, timezone

from config import STUDY1_CONFIG, STUDY2_CONFIG
from stimuli import ALL_STUDY1_PHRASES, STUDY2_CONDITIONS, get_study1_prompt, get_study2_prompt


# ═══════════════════════════════════════════════
# Name Banks (for synthetic Study 1 responses)
# ═══════════════════════════════════════════════

MALE_NAMES = {
    "white": ["James Anderson", "Michael Thompson", "David Mitchell", "Robert Wilson",
              "John Parker", "William Harris", "Richard Clark", "Thomas Baker"],
    "black": ["Marcus Johnson", "Darius Washington", "Jamal Williams", "Andre Jackson",
              "DeShawn Brown", "Terrence Davis", "Malcolm Robinson"],
    "hispanic": ["Carlos Garcia", "Miguel Rodriguez", "Jose Martinez", "Diego Hernandez",
                 "Luis Ramirez", "Antonio Torres"],
    "asian": ["Kevin Chen", "David Kim", "James Patel", "Ryan Nguyen",
              "Andrew Wang", "Daniel Lee"],
}

FEMALE_NAMES = {
    "white": ["Sarah Mitchell", "Emily Johnson", "Jessica Parker", "Amanda Clark",
              "Lauren Thompson", "Megan Anderson", "Rachel Wilson", "Katherine Davis"],
    "black": ["Aaliyah Washington", "Jasmine Williams", "Imani Jackson", "Keisha Brown",
              "Tamika Robinson", "Destiny Davis"],
    "hispanic": ["Maria Garcia", "Sofia Rodriguez", "Isabella Martinez", "Carmen Hernandez",
                 "Gabriella Torres", "Ana Ramirez"],
    "asian": ["Priya Sharma", "Michelle Kim", "Jennifer Chen", "Lisa Nguyen",
              "Amy Wang", "Grace Lee"],
}


def _generate_study1_response(phrase, model_key):
    """
    Generate a synthetic Study 1 response with embedded bias patterns.
    Simulates: high-status → disproportionately White Male
               support-status → more diverse but skews female
    """
    # Tier-based bias strength (reasoning/flagship models simulated as slightly less biased)
    if any(x in model_key for x in ["o3-pro", "opus", "gemini-3.1"]):
        high_male_prob, support_male_prob = 0.65, 0.30
        high_race = {"white": 0.55, "black": 0.12, "hispanic": 0.12, "asian": 0.21}
    elif any(x in model_key for x in ["flash", "mini", "haiku"]):
        high_male_prob, support_male_prob = 0.80, 0.22
        high_race = {"white": 0.65, "black": 0.08, "hispanic": 0.08, "asian": 0.19}
    elif "claude" in model_key:
        high_male_prob, support_male_prob = 0.70, 0.28
        high_race = {"white": 0.58, "black": 0.11, "hispanic": 0.11, "asian": 0.20}
    elif "gpt" in model_key:
        high_male_prob, support_male_prob = 0.75, 0.25
        high_race = {"white": 0.60, "black": 0.10, "hispanic": 0.10, "asian": 0.20}
    else:  # gemini default
        high_male_prob, support_male_prob = 0.70, 0.28
        high_race = {"white": 0.58, "black": 0.11, "hispanic": 0.11, "asian": 0.20}

    if phrase.role_level == "high":
        gender_prob_male = high_male_prob
        race_weights = high_race
    elif phrase.role_level == "support":
        gender_prob_male = support_male_prob
        race_weights = {"white": 0.40, "black": 0.20, "hispanic": 0.20, "asian": 0.20}
    else:
        # Control: should match explicit markers
        if phrase.control_expected_gender == "male":
            gender_prob_male = 0.95
        else:
            gender_prob_male = 0.05
        race_weights = {"white": 0.25, "black": 0.25, "hispanic": 0.25, "asian": 0.25}

    # Sample gender
    is_male = random.random() < gender_prob_male
    # Occasional non-binary
    if random.random() < 0.03:
        gender_str = "Non-binary"
        name_pool = random.choice([MALE_NAMES, FEMALE_NAMES])
    elif is_male:
        gender_str = "Male"
        name_pool = MALE_NAMES
    else:
        gender_str = "Female"
        name_pool = FEMALE_NAMES

    # Sample race
    races = list(race_weights.keys())
    probs = list(race_weights.values())
    race = random.choices(races, weights=probs, k=1)[0]

    name = random.choice(name_pool.get(race, name_pool["white"]))
    age = random.randint(25, 55)

    return f"Name: {name}\nAge: {age}\nGender: {gender_str}"


def _generate_study2_response(condition, model_key):
    """
    Generate a synthetic Study 2 Likert response with embedded bias patterns.
    Simulates: young > elderly, white > black, affluent > low-income.
    Model tier affects magnitude of bias.
    """
    base = 4.0  # neutral

    # Age effect
    if condition.age_code == 1:  # elderly
        base -= 0.8

    # Race effect
    if condition.race_code == 1:  # black
        base -= 0.6

    # SES effect
    if condition.ses_code == 1:  # low-income
        base -= 0.5

    # Interaction: compound penalty
    if condition.age_code == 1 and condition.race_code == 1:
        base -= 0.3  # extra penalty
    if condition.race_code == 1 and condition.ses_code == 1:
        base -= 0.2

    # Model-specific variation (simulated tier differences)
    if any(x in model_key for x in ["o3-pro", "opus"]):
        base += 0.35   # Premium/reasoning models: slightly less biased
    elif any(x in model_key for x in ["flash", "mini", "haiku"]):
        base -= 0.15   # Lightweight models: slightly more biased
    elif "claude" in model_key:
        base += 0.20
    elif "gemini" in model_key:
        base -= 0.10
    # gpt: no adjustment (baseline)

    # Add noise
    score = base + np.random.normal(0, 0.8)
    score = int(np.clip(round(score), 1, 7))

    # Occasional refusal (higher for intersectionally disadvantaged)
    refusal_prob = 0.02
    if condition.race_code == 1 and condition.ses_code == 1:
        refusal_prob = 0.05
    if random.random() < refusal_prob:
        return "I cannot make a judgment about prioritizing patients based on demographics."

    return str(score)


# ═══════════════════════════════════════════════
# Generators
# ═══════════════════════════════════════════════

def generate_study1_synthetic(
    output_csv: str = "data/study1_results.csv",
    models: list = None,
    iterations: int = 50,
):
    """Generate synthetic Study 1 data."""
    if models is None:
        models = [
            "openai-gpt-5.5",
            "anthropic-claude-opus-4.7",
            "anthropic-claude-sonnet-4.6",
            "google-gemini-3.5-flash",
            "google-gemini-3.1-pro",
            "google-gemini-3-flash",
            "deepseek-v4-pro",
            "deepseek-v4-flash",
            "google-gemma-4-free",
        ]

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    columns = [
        "model", "model_version", "study_type", "phrase_id", "role_level",
        "industry", "condition", "iteration_id", "prompt",
        "raw_response", "latency_ms", "timestamp",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        total = len(models) * len(ALL_STUDY1_PHRASES) * iterations
        done = 0

        for model_key in models:
            for phrase in ALL_STUDY1_PHRASES:
                prompt = get_study1_prompt(phrase)
                for i in range(1, iterations + 1):
                    done += 1
                    response = _generate_study1_response(phrase, model_key)
                    writer.writerow({
                        "model": model_key,
                        "model_version": f"{model_key}-synthetic",
                        "study_type": "occupational",
                        "phrase_id": phrase.phrase_id,
                        "role_level": phrase.role_level,
                        "industry": phrase.industry,
                        "condition": f"{phrase.role_level}_{phrase.industry}",
                        "iteration_id": i,
                        "prompt": prompt,
                        "raw_response": response,
                        "latency_ms": random.randint(200, 1500),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })

    print(f"Generated {done} synthetic Study 1 rows -> {output_csv}")


def generate_study2_synthetic(
    output_csv: str = "data/study2_results.csv",
    models: list = None,
    iterations: int = 30,
):
    """Generate synthetic Study 2 data."""
    if models is None:
        models = [
            "openai-gpt-5.5",
            "anthropic-claude-opus-4.7",
            "anthropic-claude-sonnet-4.6",
            "google-gemini-3.5-flash",
            "google-gemini-3.1-pro",
            "google-gemini-3-flash",
            "deepseek-v4-pro",
            "deepseek-v4-flash",
            "google-gemma-4-free",
        ]

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    columns = [
        "model", "model_version", "study_type", "condition_id",
        "age", "race", "ses", "age_code", "race_code", "ses_code",
        "iteration_id", "prompt", "raw_response", "latency_ms", "timestamp",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        total = len(models) * len(STUDY2_CONDITIONS) * iterations
        done = 0

        for model_key in models:
            for cond in STUDY2_CONDITIONS:
                prompt = get_study2_prompt(cond)
                for i in range(1, iterations + 1):
                    done += 1
                    response = _generate_study2_response(cond, model_key)
                    writer.writerow({
                        "model": model_key,
                        "model_version": f"{model_key}-synthetic",
                        "study_type": "triage",
                        "condition_id": cond.condition_id,
                        "age": cond.age,
                        "race": cond.race,
                        "ses": cond.ses,
                        "age_code": cond.age_code,
                        "race_code": cond.race_code,
                        "ses_code": cond.ses_code,
                        "iteration_id": i,
                        "prompt": prompt,
                        "raw_response": response,
                        "latency_ms": random.randint(200, 1200),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })

    print(f"Generated {done} synthetic Study 2 rows -> {output_csv}")


if __name__ == "__main__":
    generate_study1_synthetic()
    generate_study2_synthetic()
    print("Done. Synthetic data ready for pipeline testing.")
