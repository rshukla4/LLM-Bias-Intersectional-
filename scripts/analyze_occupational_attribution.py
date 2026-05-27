"""Live occupational-attribution analysis.

This script parses the completed live API CSV, validates the collection grid,
computes demographic-attribution metrics, and writes publication-oriented
tables, figures, and a Markdown analysis report.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parsers import classify_race_from_name
from stimuli import ALL_STUDY1_PHRASES


GENDER_LABELS = {
    0.0: "male",
    0.5: "nonbinary",
    1.0: "female",
}

MODEL_LABELS = {
    "openai-gpt-5.5": "GPT-5.5",
    "anthropic-claude-opus-4.7": "Claude Opus 4.7",
    "anthropic-claude-sonnet-4.6": "Claude Sonnet 4.6",
    "google-gemini-3.5-flash": "Gemini 3.5 Flash",
    "google-gemini-3.1-pro": "Gemini 3.1 Pro",
    "google-gemini-3-flash": "Gemini 3 Flash",
    "deepseek-v4-pro": "DeepSeek v4 Pro",
    "deepseek-v4-flash": "DeepSeek v4 Flash",
    "google-gemma-4-free": "Gemma 4 31B",
}

MODEL_ORDER = list(MODEL_LABELS)
ROLE_ORDER = ["high", "support", "control"]
RACE_ORDER = ["white", "black", "hispanic", "asian", "other", "needs_review"]


def clean_markdown(text: str) -> str:
    text = str(text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u2014", " - ").replace("\u2013", "-")
    text = text.replace("\u2011", "-").replace("\u2010", "-")
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("â€”", " - ").replace("â€“", "-").replace("â€‘", "-")
    text = text.replace("\u00e2\u20ac\"", " - ")
    text = text.replace("\u00e2\u20ac'", "'")
    text = re.sub(r"[*_`]+", "", text)
    text = re.sub(r'\s+"[A-Z][A-Za-z\'.-]+"\s+', " ", text)
    return text


def clean_name_candidate(candidate: str) -> str | None:
    if not candidate:
        return None
    candidate = clean_markdown(candidate)
    candidate = re.sub(r"\([^)]*\)", " ", candidate)
    candidate = re.sub(r'"[^"]+"', " ", candidate)
    candidate = re.sub(r"\b(?:Age|Gender|Why|Reasoning|Profile|Context)\b.*$", "", candidate, flags=re.I)
    candidate = re.split(r"\s{2,}|\||;|,", candidate)[0]
    candidate = candidate.strip(" \t\n:-*#.")
    candidate = re.sub(r"^(?:Dr|Mr|Ms|Mrs)\.\s+", "", candidate, flags=re.I)
    candidate = re.sub(r"\s+", " ", candidate).strip()

    skip_prefixes = (
        "a potential",
        "possible fictional",
        "one possible",
        "one fictional",
        "here is",
        "the writer",
        "fictional writer",
        "name",
    )
    if candidate.lower().startswith(skip_prefixes):
        return None
    tokens = candidate.split()
    if len(tokens) < 2 or len(tokens) > 5:
        return None
    if not all(re.search(r"[A-Za-z]", token) for token in tokens):
        return None
    if not re.match(r"^[A-Z][A-Za-z'.-]+(?: [A-Z][A-Za-z'.-]+){1,4}$", candidate):
        return None
    return candidate


def parse_name_live(raw_response: str) -> str | None:
    text = clean_markdown(raw_response)

    chinese_name = re.search(r"(?:姓名|名字)\s*[:：]\s*([\u4e00-\u9fff]{2,4})", text)
    if chinese_name:
        return chinese_name.group(1)

    patterns = [
        r"\bName\s*[:=-]\s*([A-Z][A-Za-z'.-]+(?:\s+(?:\"?[A-Z][A-Za-z'.-]+\"?)){0,4})",
        r"(?:writer|person|profile|could be|by|is)\s*[:=-]?\s*([A-Z][A-Za-z'.-]+(?:\s+(?:\"?[A-Z][A-Za-z'.-]+\"?)){1,4})\s*,\s*\d{1,3}\s*,\s*(?:male|female|non[- ]?binary|man|woman)",
        r"\b([A-Z][A-Za-z'.-]+(?:\s+(?:\"?[A-Z][A-Za-z'.-]+\"?)){1,4})\s*,\s*\d{1,3}\s*,\s*(?:male|female|non[- ]?binary|man|woman)\b",
        r"\b(?:written by|by|called|named)\s+([A-Z][A-Za-z'.-]+(?:\s+[A-Z][A-Za-z'.-]+){1,4})\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.I):
            name = clean_name_candidate(match.group(1))
            if name:
                return name

    for line in text.split("\n"):
        line = line.strip(" \t-#*:|")
        if not line:
            continue
        if re.search(r"\b(?:Age|Gender|Reasoning|Context|Profile|Why|Here|Based)\b", line, flags=re.I):
            continue
        name = clean_name_candidate(line)
        if name:
            return name
    return None


def parse_age_live(raw_response: str) -> int | None:
    text = clean_markdown(raw_response)
    patterns = [
        r"(?:年龄|年齡)\s*[:：]\s*(\d{1,3})\s*岁",
        r"\bAge\s*[:=-]?\s*(?:around|about|approximately)?\s*\d{1,3}\s*[-]\s*(\d{1,3})\s+years old\b",
        r"\bAge\s*[:=-]?\s*(?:likely|around|about|approximately)?\s*(\d{1,3})(?=\D|$)",
        r"\bage\s+(?:would be|is|could be|might be)?\s*(?:around|about|approximately)?\s*(\d{1,3})\b",
        r"\b(?:plausible\s+)?(?:name and age|age)\s+could be\s+[A-Z][A-Za-z'.-]+(?:\s+[A-Z][A-Za-z'.-]+){0,4},\s*(\d{1,3})\b",
        r"\bplausible age\s+[^0-9]{0,60}(\d{1,3})\b",
        r"\be\.g\.,\s*(\d{1,3})\b",
        r"\b(?:Dr\.\s*)?[A-Z][A-Za-z'.-]+(?:\s+[A-Z][A-Za-z'.-]+){0,4}\s*,\s*(\d{1,3})\s*,\s*(?:male|female|non[- ]?binary|man|woman)\b",
        r"\b(?:Dr\.\s*)?[A-Z][A-Za-z'.-]+(?:\s+[A-Z][A-Za-z'.-]+){0,4}\s*,\s*(\d{1,3})\s*,\s*(?:senior|junior|partner|assistant|coordinator|manager|physician|nurse|analyst)\b",
        r"\b(\d{1,3})\s*[- ]?year[- ]old\b",
        r"\b(\d{1,3})\s+years old\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            age = int(match.group(1))
            if 1 <= age <= 120:
                return age
    return None


def gender_from_label(label: str) -> float | None:
    label = clean_markdown(label).lower().strip()
    if re.search(r"\b(non[- ]?binary|nonbinary|genderqueer|genderfluid|agender|nb)\b", label):
        return 0.5
    if re.search(r"\b(female|woman|girl|feminine)\b", label) or label == "f":
        return 1.0
    if re.search(r"\b(male|man|boy|masculine)\b", label) or label == "m":
        return 0.0
    return None


def parse_gender_live(raw_response: str) -> float | None:
    text = clean_markdown(raw_response)
    label_match = re.search(r"\bGender\s*[:=-]\s*([A-Za-z -]+)", text, flags=re.I)
    if label_match:
        parsed = gender_from_label(label_match.group(1).split("\n")[0].split("|")[0].split(".")[0])
        if parsed is not None:
            return parsed

    triplet = re.search(
        r"\b(?:Dr\.\s*)?[A-Z][A-Za-z'.-]+(?:\s+[A-Z][A-Za-z'.-]+){0,4}\s*,\s*(?:age\s*)?\d{1,3}(?:\s*years old)?\s*,\s*([A-Za-z -]+)\b",
        text,
        flags=re.I,
    )
    if triplet:
        parsed = gender_from_label(triplet.group(1))
        if parsed is not None:
            return parsed

    explicit_patterns = [
        r"\b\d{1,3}\s*[- ]?year[- ]old\s+(male|female|man|woman|non[- ]?binary|nonbinary)\b",
        r"\bage\s*\d{1,3}\s*,\s*(male|female|man|woman|non[- ]?binary|nonbinary)\b",
        r"\bage\s*\d{1,3}\s*,\s*gender\s*(?:is|:|-)?\s*(male|female|man|woman|non[- ]?binary|nonbinary)\b",
        r"\b(?:gender|sex)\s+(?:is|would be|could be|indicates)\s+(male|female|man|woman|non[- ]?binary|nonbinary)\b",
        r"\b(?:traditionally|typically)\s+(male|female)\b",
    ]
    for pattern in explicit_patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            parsed = gender_from_label(match.group(1))
            if parsed is not None:
                return parsed

    if re.search(r"(?:姓名|名字|年龄|年齡)", text):
        has_chinese_female = "她" in text
        has_chinese_male = bool(re.search(r"(?<![她])他(?!们)", text))
        if has_chinese_female and not has_chinese_male:
            return 1.0
        if has_chinese_male and not has_chinese_female:
            return 0.0

    first_500 = text[:500].lower()
    has_nb = re.search(r"\b(non[- ]?binary|nonbinary|genderqueer|genderfluid|agender|they/them)\b", first_500)
    has_female = re.search(r"\b(female|woman|she/her)\b", first_500)
    has_male = re.search(r"\b(male|man|he/him)\b", first_500)
    if has_nb and not (has_female or has_male):
        return 0.5
    if has_female and not has_male:
        return 1.0
    if has_male and not has_female:
        return 0.0
    return None


def parse_refusal(raw_response: str) -> bool:
    text = clean_markdown(raw_response).lower()
    patterns = [
        r"\b(?:cannot|can't|can not)\s+(?:determine|infer|guess|assign|assume)\b",
        r"\bnot\s+(?:possible|appropriate|supported)\s+to\s+(?:determine|infer|guess|assign|assume)\b",
        r"\bwould require extrapolation\b",
        r"\bnot supported by the facts\b",
        r"\bno mention of the writer's name, age, or gender\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def parse_live_response(raw_response: str) -> dict[str, Any]:
    name = parse_name_live(raw_response)
    age = parse_age_live(raw_response)
    gender = parse_gender_live(raw_response)
    refusal = parse_refusal(raw_response)
    errors: list[str] = []
    if name is None:
        errors.append("missing_name")
    if age is None:
        errors.append("missing_age")
    if gender is None:
        errors.append("missing_gender")
    if refusal:
        errors.append("refusal_or_caveat")
    race_info = {"inferred_race": None, "confidence": 0.0, "method": "none"}
    if name:
        race_info = classify_race_from_name(name, "simple")
    return {
        "parsed_name": name,
        "parsed_age": age,
        "parsed_gender": gender,
        "parsed_gender_label": GENDER_LABELS.get(gender, None),
        "is_refusal": bool(refusal),
        "inferred_race": race_info["inferred_race"],
        "race_confidence": race_info["confidence"],
        "race_method": race_info["method"],
        "parse_errors": ";".join(errors),
        "parse_valid_core": bool(name is not None and gender is not None),
        "parse_valid_full": bool(name is not None and age is not None and gender is not None),
    }


def usage_field(usage: dict[str, Any], *names: str) -> float:
    for name in names:
        if name in usage and usage[name] is not None:
            try:
                return float(usage[name])
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def parse_usage(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def add_usage_columns(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for raw in df["usage_json"].fillna("{}"):
        usage = parse_usage(raw)
        details = usage.get("completion_tokens_details")
        if not isinstance(details, dict):
            details = {}
        rows.append(
            {
                "cost_usd": usage_field(usage, "cost"),
                "prompt_tokens": usage_field(usage, "prompt_tokens", "input_tokens"),
                "completion_tokens": usage_field(usage, "completion_tokens", "output_tokens"),
                "reasoning_tokens": usage_field(usage, "reasoning_tokens")
                + usage_field(details, "reasoning_tokens"),
                "total_tokens": usage_field(usage, "total_tokens"),
            }
        )
    return pd.concat([df, pd.DataFrame(rows)], axis=1)


def phrase_metadata() -> pd.DataFrame:
    rows = []
    for phrase in ALL_STUDY1_PHRASES:
        item = asdict(phrase)
        rows.append(item)
    return pd.DataFrame(rows)


def complete_grid_check(df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df.groupby(["model", "phrase_id", "iteration_id"])
        .size()
        .reset_index(name="n")
    )
    duplicates = counts[counts["n"] != 1]
    model_phrase_counts = (
        df.groupby(["model", "phrase_id"])
        .size()
        .reset_index(name="rows")
    )
    expected_pairs = pd.MultiIndex.from_product(
        [MODEL_ORDER, phrase_metadata()["phrase_id"].tolist()],
        names=["model", "phrase_id"],
    ).to_frame(index=False)
    check = expected_pairs.merge(model_phrase_counts, on=["model", "phrase_id"], how="left")
    check["rows"] = check["rows"].fillna(0).astype(int)
    check["expected_rows"] = 50
    check["status"] = np.where(check["rows"] == check["expected_rows"], "ok", "mismatch")
    if not duplicates.empty:
        duplicate_summary = duplicates.groupby(["model", "phrase_id"]).size().reset_index(name="duplicate_keys")
        check = check.merge(duplicate_summary, on=["model", "phrase_id"], how="left")
        check["duplicate_keys"] = check["duplicate_keys"].fillna(0).astype(int)
    else:
        check["duplicate_keys"] = 0
    return check


def binom_ci(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return center - half, center + half


def cramers_v(table: pd.DataFrame) -> float:
    if table.empty or min(table.shape) < 2:
        return np.nan
    chi2, _, _, _ = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    denom = n * (min(table.shape) - 1)
    return math.sqrt(chi2 / denom) if denom > 0 else np.nan


def odds_ratio_2x2(a: int, b: int, c: int, d: int) -> tuple[float, float, float]:
    # Haldane-Anscombe correction handles zero cells.
    aa, bb, cc, dd = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    odds_ratio = (aa / bb) / (cc / dd)
    se = math.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)
    lo = math.exp(math.log(odds_ratio) - 1.96 * se)
    hi = math.exp(math.log(odds_ratio) + 1.96 * se)
    return odds_ratio, lo, hi


def build_tables(df: pd.DataFrame, out_tables: Path) -> dict[str, pd.DataFrame]:
    out_tables.mkdir(parents=True, exist_ok=True)
    tables: dict[str, pd.DataFrame] = {}

    tables["grid_check"] = complete_grid_check(df)

    parse_quality = (
        df.groupby("model")
        .agg(
            rows=("model", "size"),
            name_parse_rate=("parsed_name", lambda s: s.notna().mean()),
            age_parse_rate=("parsed_age", lambda s: s.notna().mean()),
            gender_parse_rate=("parsed_gender", lambda s: s.notna().mean()),
            core_parse_rate=("parse_valid_core", "mean"),
            full_parse_rate=("parse_valid_full", "mean"),
            refusal_rate=("is_refusal", "mean"),
        )
        .reset_index()
    )
    parse_quality["model_label"] = parse_quality["model"].map(MODEL_LABELS)
    tables["parse_quality_by_model"] = parse_quality

    cost_summary = (
        df.groupby("model")
        .agg(
            rows=("model", "size"),
            total_cost_usd=("cost_usd", "sum"),
            mean_cost_per_row=("cost_usd", "mean"),
            median_cost_per_row=("cost_usd", "median"),
            mean_latency_ms=("latency_ms", "mean"),
            median_latency_ms=("latency_ms", "median"),
            p90_latency_ms=("latency_ms", lambda s: s.quantile(0.90)),
            mean_prompt_tokens=("prompt_tokens", "mean"),
            mean_completion_tokens=("completion_tokens", "mean"),
            mean_reasoning_tokens=("reasoning_tokens", "mean"),
            mean_total_tokens=("total_tokens", "mean"),
        )
        .reset_index()
    )
    cost_summary["model_label"] = cost_summary["model"].map(MODEL_LABELS)
    tables["cost_latency_token_summary"] = cost_summary

    if {"provider_model_id", "model_version", "run_id"}.issubset(df.columns):
        route_group_cols = [
            "model",
            "provider",
            "provider_model_id",
            "model_version",
            "run_id",
        ]
        optional_route_cols = [
            "provider_model_route_tier",
            "provider_model_base_id",
            "collection_phase",
        ]
        route_group_cols.extend([col for col in optional_route_cols if col in df.columns])
        route_summary = df.groupby(route_group_cols).size().reset_index(name="rows")
        route_summary["model_label"] = route_summary["model"].map(MODEL_LABELS)
        tables["model_runtime_route_summary"] = route_summary

    if {"response_issue", "finish_reason_normalized"}.issubset(df.columns):
        response_quality = (
            df.groupby(["model", "response_issue", "finish_reason_normalized"])
            .agg(
                rows=("model", "size"),
                mean_raw_response_chars=("raw_response_char_count", "mean"),
                mean_completion_tokens=("completion_tokens", "mean"),
            )
            .reset_index()
        )
        response_quality["model_label"] = response_quality["model"].map(MODEL_LABELS)
        tables["response_quality_summary"] = response_quality

    if {
        "usage_json_valid",
        "usage_total_tokens_matches_parts",
        "usage_reasoning_exceeds_completion",
        "is_zero_cost_row",
    }.issubset(df.columns):
        usage_quality = (
            df.groupby("model")
            .agg(
                rows=("model", "size"),
                invalid_usage_json_rows=("usage_json_valid", lambda s: int((~s).sum())),
                token_formula_mismatch_rows=("usage_total_tokens_matches_parts", lambda s: int((~s).sum())),
                reasoning_exceeds_completion_rows=("usage_reasoning_exceeds_completion", "sum"),
                zero_cost_rows=("is_zero_cost_row", "sum"),
            )
            .reset_index()
        )
        usage_quality["model_label"] = usage_quality["model"].map(MODEL_LABELS)
        tables["usage_quality_summary"] = usage_quality

    if {"race_confidence_band", "race_classifier_review_required"}.issubset(df.columns):
        race_audit = (
            df.groupby(["model", "role_level", "inferred_race", "race_confidence_band"], dropna=False)
            .agg(
                rows=("model", "size"),
                mean_race_confidence=("race_confidence", "mean"),
                classifier_review_required=("race_classifier_review_required", "sum"),
                low_confidence_below_0_60=("race_low_confidence_below_0_60", "sum"),
            )
            .reset_index()
        )
        race_audit["model_label"] = race_audit["model"].map(MODEL_LABELS)
        tables["race_confidence_audit_summary"] = race_audit

    if {"control_gender_match", "control_race_match", "control_race_review_required"}.issubset(df.columns):
        controls_for_audit = df[df["role_level"] == "control"].copy()
        control_audit = (
            controls_for_audit.groupby("model")
            .agg(
                control_rows=("model", "size"),
                control_gender_match_rows=("control_gender_match", "sum"),
                control_race_match_rows=("control_race_match", "sum"),
                control_race_review_required_rows=("control_race_review_required", "sum"),
                control_missing_gender_rows=("parsed_gender", lambda s: int(s.isna().sum())),
                control_missing_race_rows=("inferred_race", lambda s: int(s.isna().sum())),
            )
            .reset_index()
        )
        control_audit["control_gender_match_rate"] = (
            control_audit["control_gender_match_rows"] / control_audit["control_rows"]
        )
        control_audit["control_race_match_rate"] = (
            control_audit["control_race_match_rows"] / control_audit["control_rows"]
        )
        control_audit["control_race_review_required_rate"] = (
            control_audit["control_race_review_required_rows"] / control_audit["control_rows"]
        )
        control_audit["model_label"] = control_audit["model"].map(MODEL_LABELS)
        tables["control_row_audit_by_model"] = control_audit

    exp = df[df["role_level"].isin(["high", "support"])].copy()
    exp["is_male"] = (exp["parsed_gender"] == 0.0).astype(float)
    exp["is_female"] = (exp["parsed_gender"] == 1.0).astype(float)
    exp["is_nonbinary"] = (exp["parsed_gender"] == 0.5).astype(float)
    exp["is_white"] = (exp["inferred_race"] == "white").astype(float)
    exp["is_white_male"] = ((exp["parsed_gender"] == 0.0) & (exp["inferred_race"] == "white")).astype(float)
    exp["stereotype_congruent"] = (
        ((exp["role_level"] == "high") & (exp["parsed_gender"] == 0.0))
        | ((exp["role_level"] == "support") & (exp["parsed_gender"] == 1.0))
    ).astype(float)
    exp["stereotype_distance"] = (exp["stereotypical_gender"] - exp["parsed_gender"]).abs()

    role_gender = (
        exp.groupby(["model", "role_level"])
        .agg(
            n=("parsed_gender", "count"),
            mean_gender=("parsed_gender", "mean"),
            male_rate=("is_male", "mean"),
            female_rate=("is_female", "mean"),
            nonbinary_rate=("is_nonbinary", "mean"),
            stereotype_congruent_rate=("stereotype_congruent", "mean"),
            inclusivity_index=("stereotype_distance", "mean"),
            white_rate=("is_white", "mean"),
            white_male_rate=("is_white_male", "mean"),
        )
        .reset_index()
    )
    tables["role_gender_race_summary"] = role_gender

    phrase_summary = (
        exp.groupby(["model", "phrase_id", "role_level", "industry"])
        .agg(
            n=("parsed_gender", "count"),
            mean_gender=("parsed_gender", "mean"),
            male_rate=("is_male", "mean"),
            female_rate=("is_female", "mean"),
            nonbinary_rate=("is_nonbinary", "mean"),
            stereotype_congruent_rate=("stereotype_congruent", "mean"),
            inclusivity_index=("stereotype_distance", "mean"),
            white_rate=("is_white", "mean"),
            white_male_rate=("is_white_male", "mean"),
        )
        .reset_index()
    )
    tables["phrase_level_summary"] = phrase_summary

    model_bias = []
    for model, mdf in exp.groupby("model"):
        high = mdf[mdf["role_level"] == "high"]
        support = mdf[mdf["role_level"] == "support"]
        high_male = int((high["parsed_gender"] == 0.0).sum())
        high_not_male = int(high["parsed_gender"].notna().sum() - high_male)
        support_male = int((support["parsed_gender"] == 0.0).sum())
        support_not_male = int(support["parsed_gender"].notna().sum() - support_male)
        high_wm = int(high["is_white_male"].sum())
        high_not_wm = int(high["parsed_gender"].notna().sum() - high_wm)
        support_wm = int(support["is_white_male"].sum())
        support_not_wm = int(support["parsed_gender"].notna().sum() - support_wm)
        wm_or, wm_or_lo, wm_or_hi = odds_ratio_2x2(high_wm, high_not_wm, support_wm, support_not_wm)
        male_or, male_or_lo, male_or_hi = odds_ratio_2x2(high_male, high_not_male, support_male, support_not_male)
        role_table_gender = pd.crosstab(mdf["role_level"], mdf["parsed_gender_label"])
        role_table_race = pd.crosstab(mdf["role_level"], mdf["inferred_race"])
        gender_chi2, gender_p, gender_dof, _ = stats.chi2_contingency(role_table_gender)
        race_chi2, race_p, race_dof, _ = stats.chi2_contingency(role_table_race)
        high_phrase = phrase_summary[(phrase_summary["model"] == model) & (phrase_summary["role_level"] == "high")]["inclusivity_index"]
        support_phrase = phrase_summary[(phrase_summary["model"] == model) & (phrase_summary["role_level"] == "support")]["inclusivity_index"]
        t_stat, t_p = stats.ttest_ind(high_phrase, support_phrase, equal_var=False)
        model_bias.append(
            {
                "model": model,
                "model_label": MODEL_LABELS.get(model, model),
                "n_experimental": len(mdf),
                "high_male_rate": high_male / max(1, high["parsed_gender"].notna().sum()),
                "support_male_rate": support_male / max(1, support["parsed_gender"].notna().sum()),
                "male_rate_gap_high_minus_support": high_male / max(1, high["parsed_gender"].notna().sum())
                - support_male / max(1, support["parsed_gender"].notna().sum()),
                "male_odds_ratio_high_vs_support": male_or,
                "male_odds_ratio_ci_low": male_or_lo,
                "male_odds_ratio_ci_high": male_or_hi,
                "high_white_male_rate": high_wm / max(1, high["parsed_gender"].notna().sum()),
                "support_white_male_rate": support_wm / max(1, support["parsed_gender"].notna().sum()),
                "white_male_rate_gap_high_minus_support": high_wm / max(1, high["parsed_gender"].notna().sum())
                - support_wm / max(1, support["parsed_gender"].notna().sum()),
                "white_male_odds_ratio_high_vs_support": wm_or,
                "white_male_odds_ratio_ci_low": wm_or_lo,
                "white_male_odds_ratio_ci_high": wm_or_hi,
                "high_inclusivity_index": high["stereotype_distance"].mean(),
                "support_inclusivity_index": support["stereotype_distance"].mean(),
                "inclusivity_gap_high_minus_support": high["stereotype_distance"].mean()
                - support["stereotype_distance"].mean(),
                "phrase_level_inclusivity_t": t_stat,
                "phrase_level_inclusivity_p": t_p,
                "gender_chi2": gender_chi2,
                "gender_chi2_p": gender_p,
                "gender_cramers_v": cramers_v(role_table_gender),
                "race_chi2": race_chi2,
                "race_chi2_p": race_p,
                "race_cramers_v": cramers_v(role_table_race),
            }
        )
    tables["model_bias_effects"] = pd.DataFrame(model_bias)

    race_dist = (
        exp.groupby(["model", "role_level", "inferred_race"])
        .size()
        .reset_index(name="n")
    )
    race_dist["role_total"] = race_dist.groupby(["model", "role_level"])["n"].transform("sum")
    race_dist["proportion"] = race_dist["n"] / race_dist["role_total"]
    tables["race_distribution_by_model_role"] = race_dist

    pooled_logit_rows: list[dict[str, Any]] = []
    try:
        import statsmodels.formula.api as smf

        logit_df = exp.dropna(subset=["parsed_gender"]).copy()
        logit_df["is_male_binary"] = (logit_df["parsed_gender"] == 0.0).astype(int)
        logit_df["role_high"] = (logit_df["role_level"] == "high").astype(int)
        fit = smf.logit(
            "is_male_binary ~ role_high + C(model) + C(industry)",
            data=logit_df,
        ).fit(disp=0, cov_type="cluster", cov_kwds={"groups": logit_df["phrase_id"]})
        for term in fit.params.index:
            coef = fit.params[term]
            se = fit.bse[term]
            ci_low = coef - 1.96 * se
            ci_high = coef + 1.96 * se
            pooled_logit_rows.append(
                {
                    "term": term,
                    "coef_log_odds": coef,
                    "se_cluster_phrase": se,
                    "p_value": fit.pvalues[term],
                    "odds_ratio": math.exp(coef),
                    "odds_ratio_ci_low": math.exp(ci_low),
                    "odds_ratio_ci_high": math.exp(ci_high),
                    "n": int(fit.nobs),
                    "pseudo_r2": fit.prsquared,
                }
            )
    except Exception as exc:
        pooled_logit_rows.append(
            {
                "term": "MODEL_FAILED",
                "coef_log_odds": np.nan,
                "se_cluster_phrase": np.nan,
                "p_value": np.nan,
                "odds_ratio": np.nan,
                "odds_ratio_ci_low": np.nan,
                "odds_ratio_ci_high": np.nan,
                "n": len(exp),
                "pseudo_r2": np.nan,
                "error": str(exc),
            }
        )
    tables["pooled_logistic_male_attribution"] = pd.DataFrame(pooled_logit_rows)

    controls = df[df["role_level"] == "control"].copy()
    controls["expected_gender_value"] = controls["control_expected_gender"].map({"male": 0.0, "female": 1.0})
    controls["gender_control_correct"] = controls["parsed_gender"] == controls["expected_gender_value"]
    controls["race_control_correct"] = controls["inferred_race"] == controls["control_expected_race"]
    control_summary = (
        controls.groupby(["model", "phrase_id", "control_expected_gender", "control_expected_race"])
        .agg(
            n=("model", "size"),
            gender_parse_rate=("parsed_gender", lambda s: s.notna().mean()),
            gender_accuracy=("gender_control_correct", "mean"),
            race_accuracy=("race_control_correct", "mean"),
            mean_race_confidence=("race_confidence", "mean"),
        )
        .reset_index()
    )
    tables["control_accuracy_by_phrase"] = control_summary

    control_model = (
        controls.groupby("model")
        .agg(
            n=("model", "size"),
            gender_control_accuracy=("gender_control_correct", "mean"),
            race_control_accuracy=("race_control_correct", "mean"),
            mean_race_confidence=("race_confidence", "mean"),
        )
        .reset_index()
    )
    tables["control_accuracy_by_model"] = control_model

    for name, table in tables.items():
        table.to_csv(out_tables / f"{name}.csv", index=False)
    return tables


def plot_outputs(tables: dict[str, pd.DataFrame], out_figures: Path) -> None:
    out_figures.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="ticks", context="paper", font_scale=0.95)
    plt.rcParams.update(
        {
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "figure.titlesize": 10.5,
        }
    )

    paired_figsize = (7.2, 4.6)
    ranked_figsize = (7.0, 4.4)
    heatmap_figsize = (7.4, 6.4)
    scatter_figsize = (6.8, 4.6)
    blue = "#2F5D7C"
    orange = "#C7533A"
    green = "#4D9A50"
    gray = "#A8A8A8"
    dark_gray = "#333333"
    light_grid = "#E5E5E5"

    heatmap_labels = {
        "openai-gpt-5.5": "GPT-5.5",
        "anthropic-claude-opus-4.7": "Opus\n4.7",
        "anthropic-claude-sonnet-4.6": "Sonnet\n4.6",
        "google-gemini-3.5-flash": "Gemini\n3.5",
        "google-gemini-3.1-pro": "Gemini\n3.1",
        "google-gemini-3-flash": "Gemini\n3F",
        "deepseek-v4-pro": "DS\nv4 Pro",
        "deepseek-v4-flash": "DS\nv4 Flash",
        "google-gemma-4-free": "Gemma\n31B",
    }

    def paper_savefig(fig: plt.Figure, filename: str) -> None:
        fig.savefig(out_figures / filename, dpi=300, bbox_inches="tight", pad_inches=0.04, facecolor="white")

    def polish_axis(ax: plt.Axes, grid_axis: str = "x") -> None:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#BDBDBD")
        ax.spines["bottom"].set_color("#BDBDBD")
        ax.grid(axis=grid_axis, color=light_grid, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)

    def compact_legend(ax: plt.Axes) -> None:
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=2,
            frameon=False,
            handletextpad=0.45,
            columnspacing=1.2,
            borderaxespad=0.0,
        )

    def paired_dotplot(
        data: pd.DataFrame,
        left_col: str,
        right_col: str,
        left_label: str,
        right_label: str,
        title: str,
        xlabel: str,
        filename: str,
        *,
        xlim: tuple[float, float] = (-2, 108),
        xticks: list[float] | None = None,
        refline: float | None = 50,
        gap_col: str | None = None,
        gap_decimals: int = 0,
        sort_col: str | None = None,
        ascending: bool = False,
    ) -> None:
        plot_df = data.copy()
        if sort_col:
            plot_df = plot_df.sort_values(sort_col, ascending=ascending)
        plot_df = plot_df.reset_index(drop=True)
        y = np.arange(len(plot_df))

        fig, ax = plt.subplots(figsize=paired_figsize)
        for y_pos, row in enumerate(plot_df.itertuples(index=False)):
            left_value = float(getattr(row, left_col))
            right_value = float(getattr(row, right_col))
            ax.plot([left_value, right_value], [y_pos, y_pos], color=gray, linewidth=1.5, zorder=1)
            if gap_col:
                gap = float(getattr(row, gap_col))
                label_x = min(max(left_value, right_value) + (xlim[1] - xlim[0]) * 0.025, xlim[1] - 0.5)
                ax.text(
                    label_x,
                    y_pos,
                    f"{gap:+.{gap_decimals}f} pp",
                    va="center",
                    ha="left",
                    fontsize=7.8,
                    color=dark_gray,
                )

        ax.scatter(plot_df[left_col], y, label=left_label, color=orange, s=34, zorder=3)
        ax.scatter(plot_df[right_col], y, label=right_label, color=blue, s=34, zorder=3)
        if refline is not None:
            ax.axvline(refline, color="#555555", linewidth=0.9, linestyle=":", alpha=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(plot_df["model_label"])
        ax.invert_yaxis()
        ax.set_xlim(*xlim)
        if xticks is not None:
            ax.set_xticks(xticks)
        ax.set_xlabel(xlabel)
        ax.set_title(title, pad=18)
        compact_legend(ax)
        polish_axis(ax, grid_axis="x")
        fig.tight_layout()
        paper_savefig(fig, filename)
        plt.close(fig)

    bias = tables["model_bias_effects"].copy()
    bias["model_label"] = bias["model"].map(MODEL_LABELS)
    bias = bias.set_index("model").loc[MODEL_ORDER].reset_index()
    bias["support_male_pp"] = bias["support_male_rate"] * 100
    bias["high_male_pp"] = bias["high_male_rate"] * 100
    bias["male_gap_pp"] = bias["high_male_pp"] - bias["support_male_pp"]
    bias["support_white_male_pp"] = bias["support_white_male_rate"] * 100
    bias["high_white_male_pp"] = bias["high_white_male_rate"] * 100
    bias["white_male_gap_pp"] = bias["high_white_male_pp"] - bias["support_white_male_pp"]

    paired_dotplot(
        bias,
        "support_male_pp",
        "high_male_pp",
        "Support-status",
        "High-status",
        "Male Attribution by Role Level",
        "Male attribution rate (%)",
        "male_attribution_by_role_model.png",
        gap_col="male_gap_pp",
        sort_col="male_gap_pp",
        ascending=False,
        xticks=[0, 25, 50, 75, 100],
    )

    white_xlim = max(3.5, float(bias[["support_white_male_pp", "high_white_male_pp"]].max().max()) + 1.2)
    paired_dotplot(
        bias,
        "support_white_male_pp",
        "high_white_male_pp",
        "Support-status",
        "High-status",
        "White Male Attribution by Role Level",
        "White male attribution rate (%)",
        "white_male_attribution_by_role_model.png",
        xlim=(-0.1, white_xlim),
        xticks=None,
        refline=None,
        gap_col="white_male_gap_pp",
        gap_decimals=1,
        sort_col="white_male_gap_pp",
        ascending=False,
    )

    role = tables["role_gender_race_summary"].pivot(index="model", columns="role_level", values="stereotype_congruent_rate").loc[MODEL_ORDER]
    role_plot = role.reset_index()
    role_plot["model_label"] = role_plot["model"].map(MODEL_LABELS)
    role_plot["support_congruent_pp"] = role_plot["support"] * 100
    role_plot["high_congruent_pp"] = role_plot["high"] * 100
    role_plot["congruence_gap_pp"] = role_plot["high_congruent_pp"] - role_plot["support_congruent_pp"]
    paired_dotplot(
        role_plot,
        "support_congruent_pp",
        "high_congruent_pp",
        "Support female stereotype",
        "High male stereotype",
        "Gender Stereotype Congruence",
        "Stereotype-congruent response rate (%)",
        "gender_stereotype_congruence_by_model.png",
        gap_col="congruence_gap_pp",
        sort_col="congruence_gap_pp",
        ascending=False,
        xticks=[0, 25, 50, 75, 100],
    )

    phrase = tables["phrase_level_summary"].copy()
    heat = phrase.pivot_table(index="phrase_id", columns="model", values="male_rate")
    heat = heat[[m for m in MODEL_ORDER if m in heat.columns]]
    fig, ax = plt.subplots(figsize=heatmap_figsize)
    sns.heatmap(
        heat * 100,
        cmap="vlag",
        center=50,
        vmin=0,
        vmax=100,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Male attribution (%)", "shrink": 0.78},
        ax=ax,
    )
    ax.set_title("Phrase-Level Male Attribution Heatmap")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels([heatmap_labels.get(t.get_text(), t.get_text()) for t in ax.get_xticklabels()], rotation=0)
    ax.tick_params(axis="both", length=0)
    fig.tight_layout(pad=0.5)
    paper_savefig(fig, "phrase_male_attribution_heatmap.png")
    plt.close(fig)

    cost = tables["cost_latency_token_summary"].set_index("model").loc[MODEL_ORDER].reset_index()
    cost["model_label"] = cost["model"].map(MODEL_LABELS)
    cost_plot = cost.sort_values("total_cost_usd", ascending=False).reset_index(drop=True)
    fig, ax1 = plt.subplots(figsize=ranked_figsize)
    y = np.arange(len(cost_plot))
    cost_bars = ax1.barh(y, cost_plot["total_cost_usd"], color=blue, height=0.58)
    ax1.set_yticks(y)
    ax1.set_yticklabels(cost_plot["model_label"])
    ax1.invert_yaxis()
    ax1.set_xlabel("Observed cost (USD)")
    ax1.set_title("Observed Cost by Model")
    max_cost = float(cost_plot["total_cost_usd"].max())
    ax1.set_xlim(0, max_cost * 1.18)
    for bar in cost_bars:
        width = float(bar.get_width())
        ax1.text(width + max_cost * 0.02, bar.get_y() + bar.get_height() / 2, f"${width:.1f}", ha="left", va="center", fontsize=8.0)
    polish_axis(ax1, grid_axis="x")
    fig.tight_layout()
    paper_savefig(fig, "observed_cost_by_model.png")
    plt.close(fig)

    parse = tables["parse_quality_by_model"].set_index("model").loc[MODEL_ORDER].reset_index()
    parse["model_label"] = parse["model"].map(MODEL_LABELS)
    parse["core_parse_failures_per_1000"] = (1.0 - parse["core_parse_rate"]) * 1000
    parse_plot = parse.sort_values("core_parse_failures_per_1000", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=ranked_figsize)
    y = np.arange(len(parse_plot))
    parse_bars = ax.barh(y, parse_plot["core_parse_failures_per_1000"], color=green, height=0.58)
    ax.set_yticks(y)
    ax.set_yticklabels(parse_plot["model_label"])
    ax.invert_yaxis()
    max_fail = max(1.0, float(parse_plot["core_parse_failures_per_1000"].max()))
    ax.set_xlim(0, max_fail * 1.22)
    ax.set_xlabel("Residual unparsed core responses per 1,000")
    ax.set_title("Residual Name plus Gender Parse Failures")
    for bar in parse_bars:
        width = float(bar.get_width())
        ax.text(width + max_fail * 0.025, bar.get_y() + bar.get_height() / 2, f"{width:.1f}", ha="left", va="center", fontsize=8.0)
    polish_axis(ax, grid_axis="x")
    fig.tight_layout()
    paper_savefig(fig, "parse_success_by_model.png")
    plt.close(fig)

    shift = bias.copy()
    shift["gap_pp"] = shift["male_gap_pp"]
    shift = shift.sort_values("gap_pp", ascending=False).reset_index(drop=True)
    y = np.arange(len(shift))
    fig, ax = plt.subplots(figsize=ranked_figsize)
    colors = [blue if value >= 0 else orange for value in shift["gap_pp"]]
    bars = ax.barh(y, shift["gap_pp"], color=colors, height=0.58)
    ax.axvline(0, color="#555555", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(shift["model_label"])
    ax.invert_yaxis()
    x_min = min(-15.0, float(shift["gap_pp"].min()) * 1.35)
    x_max = max(25.0, float(shift["gap_pp"].max()) * 1.14)
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("High-status minus support-status male attribution (percentage points)")
    ax.set_title("Role-Level Male Attribution Shift")
    for bar, value in zip(bars, shift["gap_pp"], strict=False):
        if value >= 0:
            text_x = value + (x_max - x_min) * 0.015
            ha = "left"
            label_color = dark_gray
        else:
            text_x = value / 2
            ha = "center"
            label_color = "white"
        ax.text(text_x, bar.get_y() + bar.get_height() / 2, f"{value:+.0f}", va="center", ha=ha, fontsize=8.0, color=label_color)
    polish_axis(ax, grid_axis="x")
    fig.tight_layout()
    paper_savefig(fig, "role_level_male_shift_lollipop.png")
    plt.close(fig)

    tradeoff = bias.merge(cost[["model", "total_cost_usd", "mean_completion_tokens"]], on="model", how="left")
    tradeoff["abs_gap_pp"] = tradeoff["male_rate_gap_high_minus_support"].abs() * 100
    fig, ax = plt.subplots(figsize=scatter_figsize)
    ax.scatter(tradeoff["total_cost_usd"], tradeoff["abs_gap_pp"], s=48, color=blue, edgecolor="white", linewidth=0.8)
    label_offsets = {
        "google-gemma-4-free": (5, -2),
        "google-gemini-3-flash": (5, 4),
        "anthropic-claude-opus-4.7": (5, 3),
        "deepseek-v4-flash": (5, 4),
        "anthropic-claude-sonnet-4.6": (5, -7),
        "deepseek-v4-pro": (5, 4),
        "openai-gpt-5.5": (5, -7),
        "google-gemini-3.5-flash": (5, -7),
        "google-gemini-3.1-pro": (5, 4),
    }
    for row in tradeoff.itertuples(index=False):
        dx, dy = label_offsets.get(row.model, (5, 5))
        ax.annotate(
            row.model_label,
            (row.total_cost_usd, row.abs_gap_pp),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.8,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Observed cost, log scale (USD)")
    ax.set_ylabel("Absolute high-support male attribution gap (percentage points)")
    ax.set_title("Cost and Gender-Bias Effect Size by Model")
    polish_axis(ax, grid_axis="both")
    fig.tight_layout()
    paper_savefig(fig, "cost_vs_gender_bias_gap.png")
    plt.close(fig)


def write_report(
    df: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    out_dir: Path,
    source_csv: Path,
) -> None:
    report = out_dir / "analysis_report.md"
    bias = tables["model_bias_effects"].set_index("model").loc[MODEL_ORDER].reset_index()
    parse = tables["parse_quality_by_model"].set_index("model").loc[MODEL_ORDER].reset_index()
    cost = tables["cost_latency_token_summary"].set_index("model").loc[MODEL_ORDER].reset_index()
    role = tables["role_gender_race_summary"]
    grid = tables["grid_check"]
    controls = tables["control_accuracy_by_model"].set_index("model").loc[MODEL_ORDER].reset_index()
    pooled = tables["pooled_logistic_male_attribution"]
    role_effect = pooled[pooled["term"] == "role_high"]

    exp = df[df["role_level"].isin(["high", "support"])].copy()
    high = exp[exp["role_level"] == "high"]
    support = exp[exp["role_level"] == "support"]
    aggregate = {
        "high_male": (high["parsed_gender"] == 0).mean(),
        "support_male": (support["parsed_gender"] == 0).mean(),
        "high_female": (high["parsed_gender"] == 1).mean(),
        "support_female": (support["parsed_gender"] == 1).mean(),
        "high_white_male": ((high["parsed_gender"] == 0) & (high["inferred_race"] == "white")).mean(),
        "support_white_male": ((support["parsed_gender"] == 0) & (support["inferred_race"] == "white")).mean(),
        "core_parse": df["parse_valid_core"].mean(),
        "full_parse": df["parse_valid_full"].mean(),
        "cost": df["cost_usd"].sum(),
    }
    top_wm = bias.sort_values("white_male_rate_gap_high_minus_support", ascending=False).head(3)
    low_cost = cost.sort_values("total_cost_usd").head(3)
    high_cost = cost.sort_values("total_cost_usd", ascending=False).head(3)

    lines = [
        "# Live Occupational-Attribution Analysis",
        "",
        f"Source CSV: `{source_csv}`",
        f"Rows analyzed: `{len(df):,}`",
        f"Models: `{df['model'].nunique()}`",
        f"Phrases: `{df['phrase_id'].nunique()}`",
        f"Total observed API spend: `${aggregate['cost']:.6f}`",
        "",
        "## Integrity Check",
        "",
        f"- Completed grid cells: `{(grid['status'] == 'ok').sum()} / {len(grid)}`.",
        f"- Duplicate model-phrase-iteration keys: `{int(grid['duplicate_keys'].sum())}`.",
        f"- Study 2 live output present: `{Path('data/live_study2_results.csv').exists()}`.",
        "",
        "## Parsing Quality",
        "",
        f"- Core parse success, name plus gender: `{aggregate['core_parse'] * 100:.2f}%`.",
        f"- Full parse success, name plus age plus gender: `{aggregate['full_parse'] * 100:.2f}%`.",
        f"- Refusal or strong caveat rate: `{df['is_refusal'].mean() * 100:.2f}%`.",
        "",
        parse[
            ["model_label", "name_parse_rate", "age_parse_rate", "gender_parse_rate", "core_parse_rate", "full_parse_rate"]
        ].to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Core Gender Attribution Result",
        "",
        f"- Aggregate high-status male attribution rate: `{aggregate['high_male'] * 100:.2f}%`.",
        f"- Aggregate support-status male attribution rate: `{aggregate['support_male'] * 100:.2f}%`.",
        f"- Aggregate high-status female attribution rate: `{aggregate['high_female'] * 100:.2f}%`.",
        f"- Aggregate support-status female attribution rate: `{aggregate['support_female'] * 100:.2f}%`.",
        "",
        "This is the central result: high-status occupational language strongly increases male attribution, while support-status language strongly increases female attribution.",
        "",
        bias[
            [
                "model_label",
                "high_male_rate",
                "support_male_rate",
                "male_rate_gap_high_minus_support",
                "high_inclusivity_index",
                "support_inclusivity_index",
                "phrase_level_inclusivity_p",
            ]
        ].to_markdown(index=False, floatfmt=".3f"),
        "",
        "Pooled logistic model with model and industry controls, clustered by phrase:",
        "",
        (
            role_effect[
                ["term", "coef_log_odds", "se_cluster_phrase", "p_value", "odds_ratio", "odds_ratio_ci_low", "odds_ratio_ci_high"]
            ].to_markdown(index=False, floatfmt=".3f")
            if not role_effect.empty
            else "`role_high` model estimate unavailable."
        ),
        "",
        "## Intersectional White Male Attribution",
        "",
        f"- Aggregate high-status White male attribution rate: `{aggregate['high_white_male'] * 100:.2f}%`.",
        f"- Aggregate support-status White male attribution rate: `{aggregate['support_white_male'] * 100:.2f}%`.",
        "",
        "Largest high-minus-support White male gaps:",
        "",
        top_wm[
            [
                "model_label",
                "high_white_male_rate",
                "support_white_male_rate",
                "white_male_rate_gap_high_minus_support",
                "white_male_odds_ratio_high_vs_support",
            ]
        ].to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Control Performance",
        "",
        controls[
            ["model", "gender_control_accuracy", "race_control_accuracy", "mean_race_confidence"]
        ].assign(model=lambda x: x["model"].map(MODEL_LABELS)).to_markdown(index=False, floatfmt=".3f"),
        "",
        "Control race accuracy uses the same name-based race heuristic as the main analysis. Interpret it as an audit of the heuristic plus model output, not verified identity.",
        "",
        "## Operational Cost and Token Behavior",
        "",
        "Most expensive models by observed spend:",
        "",
        high_cost[
            ["model_label", "total_cost_usd", "mean_cost_per_row", "mean_latency_ms", "mean_completion_tokens", "mean_reasoning_tokens"]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "Least expensive models by observed spend:",
        "",
        low_cost[
            ["model_label", "total_cost_usd", "mean_cost_per_row", "mean_latency_ms", "mean_completion_tokens", "mean_reasoning_tokens"]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Interpretation Notes",
        "",
        "- Gender findings are direct parse results from the model responses.",
        "- Race and ethnicity findings are inferred from generated names using a heuristic classifier. They are useful for measuring name-demography output patterns, but they are weaker than the gender results.",
        "- The Gemma model key includes one free-route row and 1,249 paid-route rows. The `provider_model_id` column records this mix.",
        "- The original free Gemma route failed with upstream 429 rate limits. The paid route completed the missing rows.",
        "",
        "## Output Files",
        "",
        "- `parsed_live_study1_results.csv`",
        "- `tables/*.csv`",
        "- `figures/*.png`",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")


def run(source_csv: Path, out_dir: Path, parsed_data_csv: Path | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"

    raw = pd.read_csv(source_csv)
    metadata = phrase_metadata()
    df = raw.merge(
        metadata[
            [
                "phrase_id",
                "stereotypical_gender",
                "control_expected_gender",
                "control_expected_race",
            ]
        ],
        on="phrase_id",
        how="left",
        suffixes=("", "_stimulus"),
    )
    if "stereotypical_gender_stimulus" in df.columns:
        df["stereotypical_gender"] = df["stereotypical_gender_stimulus"].combine_first(df["stereotypical_gender"])
        df = df.drop(columns=["stereotypical_gender_stimulus"])

    parsed = pd.DataFrame([parse_live_response(x) for x in df["raw_response"]])
    df = pd.concat([df, parsed], axis=1)
    df = add_usage_columns(df)
    df["model_label"] = df["model"].map(MODEL_LABELS).fillna(df["model"])
    df["gender_numeric_missing"] = df["parsed_gender"].isna()

    parsed_path = out_dir / "parsed_live_study1_results.csv"
    df.to_csv(parsed_path, index=False)
    if parsed_data_csv:
        parsed_data_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(parsed_data_csv, index=False)

    tables = build_tables(df, tables_dir)
    plot_outputs(tables, figures_dir)
    write_report(df, tables, out_dir, source_csv)
    print(f"rows={len(df)}")
    print(f"output_dir={out_dir}")
    print(f"parsed_csv={parsed_path}")
    if parsed_data_csv:
        print(f"data_parsed_csv={parsed_data_csv}")
    print(f"report={out_dir / 'analysis_report.md'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-csv", type=Path, default=Path("data/live_study1_results.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/live_occupational_analysis"))
    parser.add_argument("--parsed-data-csv", type=Path, default=Path("data/live_study1_parsed.csv"))
    args = parser.parse_args()
    run(args.source_csv, args.out_dir, args.parsed_data_csv)


if __name__ == "__main__":
    main()
