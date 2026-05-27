"""
Generate synthetic Study 1 data for parser and analysis QA.

The previous two-study synthetic generator is archived at
archive/study2_deferred_20260526/scripts/synthetic_data_two_study.py.
"""

import csv
import os
import random
from datetime import datetime, timezone

from stimuli import ALL_STUDY1_PHRASES, get_study1_prompt


MALE_NAMES = {
    "white": ["James Anderson", "Michael Thompson", "David Mitchell", "Robert Wilson", "John Parker"],
    "black": ["Marcus Johnson", "Darius Washington", "Jamal Williams", "Andre Jackson"],
    "hispanic": ["Carlos Garcia", "Miguel Rodriguez", "Jose Martinez", "Diego Hernandez"],
    "asian": ["Kevin Chen", "David Kim", "James Patel", "Ryan Nguyen"],
}

FEMALE_NAMES = {
    "white": ["Sarah Mitchell", "Emily Johnson", "Jessica Parker", "Amanda Clark", "Lauren Thompson"],
    "black": ["Aaliyah Washington", "Jasmine Williams", "Imani Jackson", "Keisha Brown"],
    "hispanic": ["Maria Garcia", "Sofia Rodriguez", "Isabella Martinez", "Carmen Hernandez"],
    "asian": ["Priya Sharma", "Michelle Kim", "Jennifer Chen", "Lisa Nguyen"],
}


def _generate_study1_response(phrase, model_key):
    if any(x in model_key for x in ["opus", "gemini-3.1"]):
        high_male_prob, support_male_prob = 0.65, 0.30
        high_race = {"white": 0.55, "black": 0.12, "hispanic": 0.12, "asian": 0.21}
    elif any(x in model_key for x in ["flash", "gemma"]):
        high_male_prob, support_male_prob = 0.80, 0.22
        high_race = {"white": 0.65, "black": 0.08, "hispanic": 0.08, "asian": 0.19}
    elif "claude" in model_key:
        high_male_prob, support_male_prob = 0.70, 0.28
        high_race = {"white": 0.58, "black": 0.11, "hispanic": 0.11, "asian": 0.20}
    elif "gpt" in model_key:
        high_male_prob, support_male_prob = 0.75, 0.25
        high_race = {"white": 0.60, "black": 0.10, "hispanic": 0.10, "asian": 0.20}
    else:
        high_male_prob, support_male_prob = 0.70, 0.28
        high_race = {"white": 0.58, "black": 0.11, "hispanic": 0.11, "asian": 0.20}

    if phrase.role_level == "high":
        male_prob = high_male_prob
        race_weights = high_race
    elif phrase.role_level == "support":
        male_prob = support_male_prob
        race_weights = {"white": 0.40, "black": 0.20, "hispanic": 0.20, "asian": 0.20}
    else:
        male_prob = 0.95 if phrase.control_expected_gender == "male" else 0.05
        race_weights = {"white": 0.25, "black": 0.25, "hispanic": 0.25, "asian": 0.25}

    is_male = random.random() < male_prob
    if random.random() < 0.03:
        gender = "Non-binary"
        name_pool = random.choice([MALE_NAMES, FEMALE_NAMES])
    elif is_male:
        gender = "Male"
        name_pool = MALE_NAMES
    else:
        gender = "Female"
        name_pool = FEMALE_NAMES

    races = list(race_weights.keys())
    race = random.choices(races, weights=list(race_weights.values()), k=1)[0]
    name = random.choice(name_pool.get(race, name_pool["white"]))
    age = random.randint(25, 55)
    return f"Name: {name}\nAge: {age}\nGender: {gender}"


def generate_study1_synthetic(
    output_csv: str = "data/study1_results.csv",
    models: list[str] | None = None,
    iterations: int = 50,
):
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
        "model",
        "model_version",
        "study_type",
        "phrase_id",
        "role_level",
        "industry",
        "condition",
        "iteration_id",
        "prompt",
        "raw_response",
        "latency_ms",
        "timestamp",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        done = 0
        for model_key in models:
            for phrase in ALL_STUDY1_PHRASES:
                prompt = get_study1_prompt(phrase)
                for iteration_id in range(1, iterations + 1):
                    done += 1
                    writer.writerow({
                        "model": model_key,
                        "model_version": f"{model_key}-synthetic",
                        "study_type": "occupational",
                        "phrase_id": phrase.phrase_id,
                        "role_level": phrase.role_level,
                        "industry": phrase.industry,
                        "condition": f"{phrase.role_level}_{phrase.industry}",
                        "iteration_id": iteration_id,
                        "prompt": prompt,
                        "raw_response": _generate_study1_response(phrase, model_key),
                        "latency_ms": random.randint(200, 1500),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })

    print(f"Generated {done} synthetic Study 1 rows -> {output_csv}")


if __name__ == "__main__":
    generate_study1_synthetic()
    print("Done. Synthetic Study 1 data ready for pipeline testing.")
