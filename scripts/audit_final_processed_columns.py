"""Audit the final processed occupational-attribution analysis file.

This script does not change the processed dataset. It produces a reproducible
column-level audit package that verifies parsed and derived columns against the
mechanical rules used by the final reprocessing pipeline.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.reprocess_final_analysis import (  # noqa: E402
    collection_phase,
    model_route_tier,
    race_confidence_band,
    response_issue,
)


KEY_COLUMNS = ["model", "phrase_id", "iteration_id", "response_id"]
EXPECTED_ROWS = 11250


COLUMN_RULES: dict[str, tuple[str, str]] = {
    "run_id": ("raw_collection", "Raw run identifier recorded during collection."),
    "source_type": ("raw_collection", "Raw collection source type."),
    "provider": ("raw_collection", "Resolved provider name recorded during collection."),
    "model": ("raw_collection", "Experiment model configuration key."),
    "provider_model_id": ("raw_collection", "Actual provider model id used for the call."),
    "model_version": ("raw_collection", "Provider model version string recorded at call time."),
    "response_id": ("raw_collection", "Provider response id."),
    "finish_reason": ("raw_collection", "Provider finish reason; missing values are preserved."),
    "study_type": ("raw_collection", "Study identifier from collection."),
    "phrase_id": ("raw_collection", "Stimulus phrase id."),
    "role_level": ("raw_collection", "Stimulus role level."),
    "industry": ("raw_collection", "Stimulus industry."),
    "condition": ("raw_collection", "Stimulus condition."),
    "iteration_id": ("raw_collection", "Within-phrase replicate id."),
    "prompt": ("raw_collection", "Exact prompt sent to the model."),
    "raw_response": ("raw_collection", "Exact text response returned by the model."),
    "latency_ms": ("raw_collection", "Observed request latency in milliseconds."),
    "usage_json": ("raw_collection", "Raw provider usage object as JSON text."),
    "timestamp": ("raw_collection", "Collection timestamp."),
    "stereotypical_gender": ("stimulus_merge", "Stimulus expected stereotype direction merged by phrase_id."),
    "control_expected_gender": ("stimulus_merge", "Expected gender for control rows; blank for non-control rows."),
    "control_expected_race": ("stimulus_merge", "Expected race for control rows; blank for non-control rows."),
    "parsed_name": ("parsed_demographic", "Name parsed from explicit response text."),
    "parsed_age": ("parsed_demographic", "Age parsed from explicit response text."),
    "parsed_gender": ("parsed_demographic", "Numeric gender code parsed from explicit response text."),
    "parsed_gender_label": ("parsed_demographic", "Text label derived from parsed_gender."),
    "is_refusal": ("parsed_demographic", "Refusal or caveat flag detected from response text."),
    "inferred_race": ("parsed_race", "Name-based race classifier output from parsed_name."),
    "race_confidence": ("parsed_race", "Classifier confidence for inferred_race."),
    "race_method": ("parsed_race", "Classifier method label."),
    "parse_errors": ("parsed_quality", "Semicolon-delimited parse issue list."),
    "parse_valid_core": ("parsed_quality", "parsed_name and parsed_gender are present."),
    "parse_valid_full": ("parsed_quality", "parsed_name, parsed_age, and parsed_gender are present."),
    "cost_usd": ("usage_derived", "Numeric cost parsed from usage_json."),
    "prompt_tokens": ("usage_derived", "Prompt token count parsed from usage_json."),
    "completion_tokens": ("usage_derived", "Completion token count parsed from usage_json."),
    "reasoning_tokens": ("usage_derived", "Reasoning token count parsed from usage_json details."),
    "total_tokens": ("usage_derived", "Total token count parsed from usage_json."),
    "model_label": ("audit_derived", "Human-readable model label."),
    "gender_numeric_missing": ("audit_derived", "Whether parsed_gender is missing."),
    "model_config_key": ("audit_derived", "Copy of model, explicit experiment key."),
    "served_model_id": ("audit_derived", "Copy of provider_model_id, explicit served model."),
    "served_model_version": ("audit_derived", "Copy of model_version."),
    "provider_model_base_id": ("audit_derived", "provider_model_id with :free route suffix removed."),
    "provider_model_route_tier": ("audit_derived", "free versus paid_or_standard route tier."),
    "collection_phase": ("audit_derived", "primary launch versus Gemma paid resume phase."),
    "finish_reason_normalized": ("audit_derived", "finish_reason with missing values normalized."),
    "has_provider_finish_reason": ("audit_derived", "Whether finish_reason is present."),
    "raw_response_char_count": ("audit_derived", "Character count of raw_response."),
    "raw_response_word_count": ("audit_derived", "Whitespace token count of raw_response."),
    "parse_error_count": ("audit_derived", "Count of parse_errors entries."),
    "response_issue": ("audit_derived", "Row-level response quality class."),
    "usage_json_valid": ("audit_derived", "Whether usage_json is structurally JSON-like."),
    "usage_total_tokens_matches_parts": ("audit_derived", "Whether total_tokens equals prompt plus completion."),
    "usage_reasoning_exceeds_completion": ("audit_derived", "Provider accounting flag for reasoning > completion."),
    "is_zero_cost_row": ("audit_derived", "Whether cost_usd equals zero."),
    "race_classifier_review_required": ("audit_derived", "Whether race is missing or needs_review."),
    "race_low_confidence_below_0_60": ("audit_derived", "Whether race_confidence is below 0.60."),
    "race_confidence_band": ("audit_derived", "Binned race-confidence audit label."),
    "control_gender_match": ("audit_derived", "Control-only parsed gender match flag."),
    "control_race_match": ("audit_derived", "Control-only inferred race match flag."),
    "control_race_review_required": ("audit_derived", "Control-only race review flag."),
    "name_corrected_from_workbook": ("correction_provenance", "Whether row was updated through correction reparsing."),
    "original_parsed_name_before_workbook": ("correction_provenance", "Pre-correction parsed_name snapshot."),
    "original_parse_errors_before_workbook": ("correction_provenance", "Pre-correction parse_errors snapshot."),
}


def bool_series_equal(left: pd.Series, right: pd.Series) -> pd.Series:
    return left.fillna("<NA>").astype(str) == right.fillna("<NA>").astype(str)


def parse_error_tokens(value: Any) -> set[str]:
    if pd.isna(value) or value == "":
        return set()
    return {part for part in str(value).split(";") if part}


def normalized_text(value: Any) -> str:
    text = "" if pd.isna(value) else str(value)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def write_examples(out_dir: Path, name: str, df: pd.DataFrame) -> str | None:
    if df.empty:
        return None
    path = out_dir / f"{name}.csv"
    df.to_csv(path, index=False)
    return str(path)


def audit_processed(df: pd.DataFrame, out_dir: Path, correction_audit: Path | None) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    issues: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []

    def add_check(name: str, passed: bool, observed: Any, expected: Any, severity: str = "P1", examples: pd.DataFrame | None = None) -> None:
        examples_path = write_examples(out_dir, name, examples) if examples is not None and not examples.empty else None
        checks.append(
            {
                "check": name,
                "passed": bool(passed),
                "observed": observed,
                "expected": expected,
                "severity": severity,
                "examples_path": examples_path,
            }
        )
        if not passed:
            issues.append(
                {
                    "severity": severity,
                    "check": name,
                    "observed": observed,
                    "expected": expected,
                    "examples_path": examples_path,
                }
            )

    add_check("row_count", len(df) == EXPECTED_ROWS, len(df), EXPECTED_ROWS, "P0")
    missing_columns = sorted(set(COLUMN_RULES) - set(df.columns))
    unexpected_columns = sorted(set(df.columns) - set(COLUMN_RULES))
    add_check("required_columns_present", not missing_columns, missing_columns, [], "P0")
    add_check("unexpected_columns_absent", not unexpected_columns, unexpected_columns, [], "P2")
    duplicate_keys = df[df.duplicated(KEY_COLUMNS, keep=False)]
    add_check("duplicate_key_rows", duplicate_keys.empty, int(len(duplicate_keys)), 0, "P0", duplicate_keys)

    model_counts = df.groupby("model").size()
    bad_model_counts = model_counts[model_counts != 1250]
    add_check("model_counts_1250_each", bad_model_counts.empty, bad_model_counts.to_dict(), "all models 1250", "P0")

    # Core parse consistency.
    expected_gender_label = df["parsed_gender"].map({0.0: "male", 0.5: "nonbinary", 1.0: "female"})
    label_bad = df[~bool_series_equal(df["parsed_gender_label"], expected_gender_label)]
    add_check("gender_label_matches_numeric", label_bad.empty, int(len(label_bad)), 0, "P0", label_bad)

    bad_gender_values = df[df["parsed_gender"].notna() & ~df["parsed_gender"].isin([0.0, 0.5, 1.0])]
    add_check("parsed_gender_allowed_values", bad_gender_values.empty, int(len(bad_gender_values)), 0, "P0", bad_gender_values)

    bad_age = df[df["parsed_age"].notna() & ~df["parsed_age"].between(1, 120)]
    add_check("parsed_age_in_range_1_120", bad_age.empty, int(len(bad_age)), 0, "P0", bad_age)

    expected_core = df["parsed_name"].notna() & df["parsed_gender"].notna()
    core_bad = df[df["parse_valid_core"] != expected_core]
    add_check("parse_valid_core_rule", core_bad.empty, int(len(core_bad)), 0, "P0", core_bad)

    expected_full = df["parsed_name"].notna() & df["parsed_age"].notna() & df["parsed_gender"].notna()
    full_bad = df[df["parse_valid_full"] != expected_full]
    add_check("parse_valid_full_rule", full_bad.empty, int(len(full_bad)), 0, "P0", full_bad)

    token_sets = df["parse_errors"].apply(parse_error_tokens)
    parse_error_bad_mask = (
        token_sets.apply(lambda s: ("missing_name" in s)) != df["parsed_name"].isna()
    ) | (
        token_sets.apply(lambda s: ("missing_age" in s)) != df["parsed_age"].isna()
    ) | (
        token_sets.apply(lambda s: ("missing_gender" in s)) != df["parsed_gender"].isna()
    ) | (
        token_sets.apply(lambda s: ("refusal_or_caveat" in s)) != df["is_refusal"]
    )
    parse_error_bad = df[parse_error_bad_mask]
    add_check("parse_errors_match_fields", parse_error_bad.empty, int(len(parse_error_bad)), 0, "P0", parse_error_bad)

    gender_missing_bad = df[df["gender_numeric_missing"] != df["parsed_gender"].isna()]
    add_check("gender_numeric_missing_rule", gender_missing_bad.empty, int(len(gender_missing_bad)), 0, "P0", gender_missing_bad)

    # Race and control consistency.
    race_review_expected = df["inferred_race"].isna() | df["inferred_race"].eq("needs_review")
    race_review_bad = df[df["race_classifier_review_required"] != race_review_expected]
    add_check("race_review_required_rule", race_review_bad.empty, int(len(race_review_bad)), 0, "P0", race_review_bad)

    race_low_bad = df[df["race_low_confidence_below_0_60"] != (df["race_confidence"] < 0.60)]
    add_check("race_low_confidence_rule", race_low_bad.empty, int(len(race_low_bad)), 0, "P0", race_low_bad)

    expected_bands = df.apply(race_confidence_band, axis=1)
    race_band_bad = df[df["race_confidence_band"] != expected_bands]
    add_check("race_confidence_band_rule", race_band_bad.empty, int(len(race_band_bad)), 0, "P0", race_band_bad)

    non_control = df[df["role_level"] != "control"]
    control = df[df["role_level"] == "control"].copy()
    non_control_expected_bad = non_control[
        non_control["control_expected_gender"].notna() | non_control["control_expected_race"].notna()
    ]
    add_check("non_control_expected_fields_blank", non_control_expected_bad.empty, int(len(non_control_expected_bad)), 0, "P0", non_control_expected_bad)
    control_expected_bad = control[control["control_expected_gender"].isna() | control["control_expected_race"].isna()]
    add_check("control_expected_fields_present", control_expected_bad.empty, int(len(control_expected_bad)), 0, "P0", control_expected_bad)

    expected_control_gender = control["control_expected_gender"].map({"male": 0.0, "female": 1.0})
    control_gender_expected = control["parsed_gender"] == expected_control_gender
    control_gender_bad = control[control["control_gender_match"].astype("boolean") != control_gender_expected.astype("boolean")]
    add_check("control_gender_match_rule", control_gender_bad.empty, int(len(control_gender_bad)), 0, "P0", control_gender_bad)

    control_race_expected = control["inferred_race"] == control["control_expected_race"]
    control_race_bad = control[control["control_race_match"].astype("boolean") != control_race_expected.astype("boolean")]
    add_check("control_race_match_rule", control_race_bad.empty, int(len(control_race_bad)), 0, "P0", control_race_bad)

    non_control_match_bad = non_control[
        non_control["control_gender_match"].notna()
        | non_control["control_race_match"].notna()
        | non_control["control_race_review_required"].notna()
    ]
    add_check("non_control_match_fields_blank", non_control_match_bad.empty, int(len(non_control_match_bad)), 0, "P0", non_control_match_bad)

    # Usage, response, and route consistency.
    route_expected = pd.Series(
        [model_route_tier(a, b) for a, b in zip(df["provider_model_id"], df["model_version"])],
        index=df.index,
    )
    route_bad = df[df["provider_model_route_tier"] != route_expected]
    add_check("provider_model_route_tier_rule", route_bad.empty, int(len(route_bad)), 0, "P0", route_bad)

    phase_expected = df["run_id"].map(collection_phase)
    phase_bad = df[df["collection_phase"] != phase_expected]
    add_check("collection_phase_rule", phase_bad.empty, int(len(phase_bad)), 0, "P0", phase_bad)

    finish_expected = df["finish_reason"].fillna("missing_provider_finish_reason")
    finish_bad = df[df["finish_reason_normalized"] != finish_expected]
    add_check("finish_reason_normalized_rule", finish_bad.empty, int(len(finish_bad)), 0, "P0", finish_bad)

    has_finish_bad = df[df["has_provider_finish_reason"] != df["finish_reason"].notna()]
    add_check("has_provider_finish_reason_rule", has_finish_bad.empty, int(len(has_finish_bad)), 0, "P0", has_finish_bad)

    response_issue_expected = df.apply(response_issue, axis=1)
    response_issue_bad = df[df["response_issue"] != response_issue_expected]
    add_check("response_issue_rule", response_issue_bad.empty, int(len(response_issue_bad)), 0, "P0", response_issue_bad)

    char_bad = df[df["raw_response_char_count"] != df["raw_response"].fillna("").astype(str).str.len()]
    add_check("raw_response_char_count_rule", char_bad.empty, int(len(char_bad)), 0, "P0", char_bad)

    word_expected = df["raw_response"].fillna("").astype(str).str.split().str.len()
    word_bad = df[df["raw_response_word_count"] != word_expected]
    add_check("raw_response_word_count_rule", word_bad.empty, int(len(word_bad)), 0, "P0", word_bad)

    error_count_expected = df["parse_errors"].fillna("").apply(
        lambda value: 0 if not value else len([part for part in str(value).split(";") if part])
    )
    error_count_bad = df[df["parse_error_count"] != error_count_expected]
    add_check("parse_error_count_rule", error_count_bad.empty, int(len(error_count_bad)), 0, "P0", error_count_bad)

    token_formula_bad = df[df["usage_total_tokens_matches_parts"] != ((df["total_tokens"] - df["prompt_tokens"] - df["completion_tokens"]).abs() < 1e-9)]
    add_check("usage_total_tokens_matches_parts_rule", token_formula_bad.empty, int(len(token_formula_bad)), 0, "P0", token_formula_bad)

    reasoning_bad = df[df["usage_reasoning_exceeds_completion"] != (df["reasoning_tokens"] > df["completion_tokens"])]
    add_check("usage_reasoning_exceeds_completion_rule", reasoning_bad.empty, int(len(reasoning_bad)), 0, "P0", reasoning_bad)

    zero_cost_bad = df[df["is_zero_cost_row"] != (df["cost_usd"] == 0)]
    add_check("is_zero_cost_row_rule", zero_cost_bad.empty, int(len(zero_cost_bad)), 0, "P0", zero_cost_bad)

    usage_negative = df[
        (df["cost_usd"] < 0)
        | (df["prompt_tokens"] < 0)
        | (df["completion_tokens"] < 0)
        | (df["reasoning_tokens"] < 0)
        | (df["total_tokens"] < 0)
        | (df["latency_ms"] < 0)
    ]
    add_check("usage_and_latency_nonnegative", usage_negative.empty, int(len(usage_negative)), 0, "P0", usage_negative)

    # Correction source consistency.
    if correction_audit and correction_audit.exists():
        audit_df = pd.read_csv(correction_audit)
        audit_keys = set(audit_df[KEY_COLUMNS].astype(str).agg("||".join, axis=1))
        final_keys = df[KEY_COLUMNS].astype(str).agg("||".join, axis=1)
        in_audit = final_keys.isin(audit_keys)
        add_check("correction_audit_row_count", len(audit_df) == 246, len(audit_df), 246, "P1")
        add_check("correction_audit_keys_subset_final", in_audit.sum() == len(audit_df), int(in_audit.sum()), len(audit_df), "P0")
        correction_flag_bad = df[df["name_corrected_from_workbook"] != in_audit]
        add_check("correction_flag_matches_audit_keys", correction_flag_bad.empty, int(len(correction_flag_bad)), 0, "P1", correction_flag_bad)

    # Review-focused extracts. These are expected to be nonempty.
    review_extracts = {
        "review_missing_parsed_demographics": df[df[["parsed_name", "parsed_age", "parsed_gender_label"]].isna().any(axis=1)],
        "review_noncomplete_response_issues": df[df["response_issue"] != "complete"],
        "review_control_mismatch_rows": control[(control["control_gender_match"] == False) | (control["control_race_match"] == False)],
        "review_low_confidence_race_rows": df[df["race_low_confidence_below_0_60"]],
        "review_single_token_names": df[df["parsed_name"].fillna("").str.split().str.len().eq(1)],
    }
    review_paths = {name: write_examples(out_dir, name, value) for name, value in review_extracts.items()}

    inventory = []
    for column in df.columns:
        category, rule = COLUMN_RULES.get(column, ("unmapped", "Column is not mapped in COLUMN_RULES."))
        inventory.append(
            {
                "column": column,
                "category": category,
                "dtype": str(df[column].dtype),
                "missing": int(df[column].isna().sum()),
                "unique_nonmissing": int(df[column].nunique(dropna=True)),
                "rule": rule,
            }
        )
    inventory_df = pd.DataFrame(inventory)
    inventory_df.to_csv(out_dir / "column_inventory.csv", index=False)
    pd.DataFrame(checks).to_csv(out_dir / "audit_checks.csv", index=False)
    pd.DataFrame(issues).to_csv(out_dir / "audit_issues.csv", index=False)

    summary = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "checks": int(len(checks)),
        "failed_checks": int(len(issues)),
        "p0_failures": int(sum(1 for issue in issues if issue["severity"] == "P0")),
        "p1_failures": int(sum(1 for issue in issues if issue["severity"] == "P1")),
        "p2_failures": int(sum(1 for issue in issues if issue["severity"] == "P2")),
        "missing_name": int(df["parsed_name"].isna().sum()),
        "missing_age": int(df["parsed_age"].isna().sum()),
        "missing_gender": int(df["parsed_gender_label"].isna().sum()),
        "core_parse_rate": float(df["parse_valid_core"].mean()),
        "full_parse_rate": float(df["parse_valid_full"].mean()),
        "race_review_required_rows": int(df["race_classifier_review_required"].sum()),
        "response_issue_counts": {k: int(v) for k, v in df.groupby("response_issue").size().to_dict().items()},
        "review_extracts": review_paths,
    }
    (out_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the final processed occupational-attribution CSV.")
    parser.add_argument(
        "--processed-csv",
        type=Path,
        default=Path("outputs/live_occupational_analysis_20260527/final_processed_live_study1_results.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/live_occupational_analysis_20260527/column_audit"),
    )
    parser.add_argument(
        "--correction-audit",
        type=Path,
        default=Path("outputs/live_occupational_analysis_20260527/tables/name_correction_audit.csv"),
    )
    args = parser.parse_args()

    df = pd.read_csv(args.processed_csv)
    summary = audit_processed(df, args.out_dir, args.correction_audit)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
