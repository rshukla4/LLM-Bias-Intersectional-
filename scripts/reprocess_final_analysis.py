"""One-command final reprocessing pipeline for the live occupational study.

This is the canonical final analysis entry point. It performs the complete
workflow in one place:

1. Parse the raw completed live API CSV.
2. Read and apply the manual correction workbook for rows that still lacked
   names after automated parsing.
3. Write one authoritative final processed CSV into the analysis output folder.
4. Recompute all metrics, tables, figures, and the Markdown report from the
   corrected processed data.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_occupational_attribution import (  # noqa: E402
    MODEL_LABELS,
    add_usage_columns,
    build_tables,
    parse_live_response,
    phrase_metadata,
    plot_outputs,
    write_report,
)
from scripts.merge_name_corrections import (  # noqa: E402
    bool_value,
    corrected_parse,
    read_xlsx_first_sheet,
)


KEY_COLUMNS = ["model", "phrase_id", "iteration_id", "response_id"]


def parse_raw_results(source_csv: Path) -> pd.DataFrame:
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
        df["stereotypical_gender"] = df["stereotypical_gender_stimulus"].combine_first(
            df["stereotypical_gender"]
        )
        df = df.drop(columns=["stereotypical_gender_stimulus"])

    parsed = pd.DataFrame([parse_live_response(value) for value in df["raw_response"]])
    df = pd.concat([df, parsed], axis=1)
    df = add_usage_columns(df)
    df["model_label"] = df["model"].map(MODEL_LABELS).fillna(df["model"])
    df["gender_numeric_missing"] = df["parsed_gender"].isna()
    df = add_audit_columns(df)
    return df


def model_route_tier(provider_model_id: Any, model_version: Any) -> str:
    provider_model_id = "" if pd.isna(provider_model_id) else str(provider_model_id)
    model_version = "" if pd.isna(model_version) else str(model_version)
    if ":free" in provider_model_id or ":free" in model_version:
        return "free"
    return "paid_or_standard"


def collection_phase(run_id: Any) -> str:
    run_id = "" if pd.isna(run_id) else str(run_id)
    if "gemma_paid_resume" in run_id:
        return "gemma_paid_resume"
    return "primary_live_launch"


def response_issue(row: pd.Series) -> str:
    raw_response = "" if pd.isna(row.get("raw_response")) else str(row.get("raw_response"))
    parse_errors = "" if pd.isna(row.get("parse_errors")) else str(row.get("parse_errors"))
    finish_reason = row.get("finish_reason")
    if "refusal_or_caveat" in parse_errors:
        return "explicit_refusal_or_caveat"
    if pd.isna(finish_reason):
        if len(raw_response.strip()) <= 5:
            return "missing_finish_reason_symbol_only"
        return "missing_finish_reason_truncated_or_incomplete"
    if not bool(row.get("parse_valid_core")):
        return "missing_core_demographics"
    if not bool(row.get("parse_valid_full")):
        return "missing_age_or_gender"
    return "complete"


def add_audit_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["model_config_key"] = df["model"]
    df["served_model_id"] = df["provider_model_id"]
    df["served_model_version"] = df["model_version"]
    df["provider_model_base_id"] = df["provider_model_id"].astype(str).str.replace(r":free$", "", regex=True)
    df["provider_model_route_tier"] = [
        model_route_tier(provider_model_id, model_version)
        for provider_model_id, model_version in zip(df["provider_model_id"], df["model_version"])
    ]
    df["collection_phase"] = df["run_id"].map(collection_phase)
    df["finish_reason_normalized"] = df["finish_reason"].fillna("missing_provider_finish_reason")
    df["has_provider_finish_reason"] = df["finish_reason"].notna()
    df["raw_response_char_count"] = df["raw_response"].fillna("").astype(str).str.len()
    df["raw_response_word_count"] = df["raw_response"].fillna("").astype(str).str.split().str.len()
    df["parse_error_count"] = df["parse_errors"].fillna("").apply(
        lambda value: 0 if not value else len([part for part in str(value).split(";") if part])
    )
    df["response_issue"] = df.apply(response_issue, axis=1)
    df["usage_json_valid"] = df["usage_json"].fillna("").apply(lambda value: isinstance(value, str) and value.strip().startswith("{"))
    df["usage_total_tokens_matches_parts"] = (
        (df["total_tokens"] - df["prompt_tokens"] - df["completion_tokens"]).abs() < 1e-9
    )
    df["usage_reasoning_exceeds_completion"] = df["reasoning_tokens"] > df["completion_tokens"]
    df["is_zero_cost_row"] = df["cost_usd"] == 0
    df["race_classifier_review_required"] = df["inferred_race"].isna() | df["inferred_race"].eq("needs_review")
    df["race_low_confidence_below_0_60"] = df["race_confidence"] < 0.60
    df["race_confidence_band"] = df.apply(race_confidence_band, axis=1)
    expected_gender_value = df["control_expected_gender"].map({"male": 0.0, "female": 1.0})
    is_control = df["role_level"].eq("control")
    df["control_gender_match"] = pd.NA
    df.loc[is_control, "control_gender_match"] = (
        df.loc[is_control, "parsed_gender"] == expected_gender_value.loc[is_control]
    )
    df["control_race_match"] = pd.NA
    df.loc[is_control, "control_race_match"] = (
        df.loc[is_control, "inferred_race"] == df.loc[is_control, "control_expected_race"]
    )
    df["control_race_review_required"] = pd.NA
    df.loc[is_control, "control_race_review_required"] = df.loc[
        is_control, "race_classifier_review_required"
    ]
    return df


def race_confidence_band(row: pd.Series) -> str:
    inferred_race = row.get("inferred_race")
    confidence = row.get("race_confidence")
    if pd.isna(inferred_race):
        return "missing_name"
    if inferred_race == "needs_review":
        return "review_gate_below_0_45"
    if pd.isna(confidence):
        return "missing_confidence"
    confidence = float(confidence)
    if confidence < 0.60:
        return "classified_low_0_45_to_0_60"
    if confidence < 0.80:
        return "classified_medium_0_60_to_0_80"
    return "classified_high_0_80_plus"


def correction_key_frame(df: pd.DataFrame) -> pd.Series:
    missing = [col for col in KEY_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required key columns: {missing}")
    return df[KEY_COLUMNS].astype(str).agg("||".join, axis=1)


def load_correction_rows(correction_xlsx: Path, audit_csv: Path) -> tuple[pd.DataFrame, str]:
    if correction_xlsx.exists():
        corrections = read_xlsx_first_sheet(correction_xlsx)
        correction_source = str(correction_xlsx)
    elif audit_csv.exists():
        corrections = pd.read_csv(audit_csv)
        correction_source = str(audit_csv)
    else:
        raise FileNotFoundError(
            f"No correction source found. Checked {correction_xlsx} and {audit_csv}."
        )
    if corrections.empty:
        raise ValueError(f"Correction source is empty: {correction_source}")

    missing = [col for col in KEY_COLUMNS + ["raw_response"] if col not in corrections.columns]
    if missing:
        raise ValueError(f"Correction source missing required columns: {missing}")

    stale_correction_cols = [col for col in corrections.columns if col.startswith("corrected_")]
    if stale_correction_cols:
        corrections = corrections.drop(columns=stale_correction_cols)
    return corrections, correction_source


def apply_corrections(
    parsed: pd.DataFrame,
    correction_xlsx: Path,
    audit_csv: Path,
    remaining_csv: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    corrections, correction_source = load_correction_rows(correction_xlsx, audit_csv)

    processed = parsed.copy()
    processed["_merge_key"] = correction_key_frame(processed)
    corrections["_merge_key"] = correction_key_frame(corrections)

    if corrections["_merge_key"].duplicated().any():
        duplicated = corrections[corrections["_merge_key"].duplicated(keep=False)][KEY_COLUMNS]
        raise ValueError(f"Duplicate correction keys found:\n{duplicated.to_string(index=False)}")

    missing_keys = sorted(set(corrections["_merge_key"]) - set(processed["_merge_key"]))
    if missing_keys:
        raise ValueError(f"{len(missing_keys)} correction rows do not match the parsed data.")

    raw_by_key = processed.set_index("_merge_key")["raw_response"]
    corrections["raw_response"] = corrections["_merge_key"].map(raw_by_key)
    corrected_fields = pd.DataFrame([corrected_parse(row) for _, row in corrections.iterrows()])
    corrections = pd.concat(
        [corrections.reset_index(drop=True), corrected_fields.add_prefix("corrected_")],
        axis=1,
    )

    processed["name_corrected_from_workbook"] = False
    processed["original_parsed_name_before_workbook"] = processed["parsed_name"]
    processed["original_parse_errors_before_workbook"] = processed["parse_errors"]

    field_map = {
        "parsed_name": "corrected_parsed_name",
        "parsed_age": "corrected_parsed_age",
        "parsed_gender": "corrected_parsed_gender",
        "parsed_gender_label": "corrected_parsed_gender_label",
        "is_refusal": "corrected_is_refusal",
        "inferred_race": "corrected_inferred_race",
        "race_confidence": "corrected_race_confidence",
        "race_method": "corrected_race_method",
        "parse_errors": "corrected_parse_errors",
        "parse_valid_core": "corrected_parse_valid_core",
        "parse_valid_full": "corrected_parse_valid_full",
    }

    corrections_by_key = corrections.set_index("_merge_key")
    for idx, row in processed[processed["_merge_key"].isin(corrections_by_key.index)].iterrows():
        correction = corrections_by_key.loc[row["_merge_key"]]
        changed = False
        for target, source in field_map.items():
            value = correction[source]
            if target in {"parse_valid_core", "parse_valid_full", "is_refusal"}:
                value = bool_value(value)
            if pd.isna(value):
                value = None
            if target == "parsed_age" and value not in (None, ""):
                value = int(float(value))
            if target == "parsed_gender" and value not in (None, ""):
                value = float(value)
            if target == "race_confidence" and value not in (None, ""):
                value = float(value)
            if str(processed.at[idx, target]) != str(value):
                changed = True
            processed.at[idx, target] = value
        processed.at[idx, "name_corrected_from_workbook"] = changed

    processed = processed.drop(columns=["_merge_key"])

    audit_cols = KEY_COLUMNS + [
        "raw_response",
        "parsed_name",
        "parse_errors",
        "corrected_parsed_name",
        "corrected_parsed_age",
        "corrected_parsed_gender_label",
        "corrected_inferred_race",
        "corrected_race_confidence",
        "corrected_parse_errors",
        "corrected_parse_valid_core",
        "corrected_parse_valid_full",
    ]
    audit_csv.parent.mkdir(parents=True, exist_ok=True)
    corrections[audit_cols].to_csv(audit_csv, index=False)

    remaining = corrections[corrections["corrected_parsed_name"].isna()][
        KEY_COLUMNS + ["raw_response", "corrected_parse_errors"]
    ]
    remaining.to_csv(remaining_csv, index=False)

    summary = {
        "correction_source": correction_source,
        "correction_rows": int(len(corrections)),
        "correction_rows_with_name_after_reparse": int(corrections["corrected_parsed_name"].notna().sum()),
        "correction_rows_still_missing_name": int(corrections["corrected_parsed_name"].isna().sum()),
        "rows_flagged_corrected_from_workbook": int(processed["name_corrected_from_workbook"].sum()),
    }
    return processed, summary


def validate_processed(processed: pd.DataFrame, expected_rows: int) -> dict[str, Any]:
    model_counts = processed.groupby("model").size().to_dict()
    duplicate_keys = int(processed.duplicated(KEY_COLUMNS).sum())
    validation = {
        "processed_rows": int(len(processed)),
        "expected_rows": int(expected_rows),
        "row_count_ok": bool(len(processed) == expected_rows),
        "duplicate_model_phrase_iteration_response_keys": duplicate_keys,
        "model_counts": {key: int(value) for key, value in model_counts.items()},
        "processed_missing_name": int(processed["parsed_name"].isna().sum()),
        "processed_missing_age": int(processed["parsed_age"].isna().sum()),
        "processed_missing_gender": int(processed["parsed_gender"].isna().sum()),
        "processed_core_parse_rate": float(processed["parse_valid_core"].mean()),
        "processed_full_parse_rate": float(processed["parse_valid_full"].mean()),
        "total_cost_usd": float(processed["cost_usd"].sum()),
        "missing_finish_reason": int(processed["finish_reason"].isna().sum()),
        "zero_cost_rows": int(processed["is_zero_cost_row"].sum()),
        "usage_json_invalid_rows": int((~processed["usage_json_valid"]).sum()),
        "usage_token_formula_mismatch_rows": int((~processed["usage_total_tokens_matches_parts"]).sum()),
        "usage_reasoning_exceeds_completion_rows": int(processed["usage_reasoning_exceeds_completion"].sum()),
        "provider_model_route_tier_counts": {
            key: int(value) for key, value in processed.groupby("provider_model_route_tier").size().to_dict().items()
        },
        "response_issue_counts": {
            key: int(value) for key, value in processed.groupby("response_issue").size().to_dict().items()
        },
    }
    if not validation["row_count_ok"]:
        raise ValueError(f"Processed row count is {len(processed)}, expected {expected_rows}.")
    return validation


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = args.out_dir / "tables"
    figures_dir = args.out_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    parsed = parse_raw_results(args.source_csv)
    pre_correction_csv = None
    if args.write_intermediates:
        pre_correction_csv = args.out_dir / "parsed_live_study1_results_pre_corrections.csv"
        parsed.to_csv(pre_correction_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    audit_csv = tables_dir / "name_correction_audit.csv"
    processed, correction_summary = apply_corrections(
        parsed=parsed,
        correction_xlsx=args.correction_xlsx,
        audit_csv=audit_csv,
        remaining_csv=tables_dir / "name_correction_remaining_unparsed.csv",
    )
    processed = add_audit_columns(processed)

    processed_csv = args.out_dir / "final_processed_live_study1_results.csv"
    processed.to_csv(processed_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    if args.mirror_data_csv:
        args.mirror_data_csv.parent.mkdir(parents=True, exist_ok=True)
        processed.to_csv(args.mirror_data_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    validation = validate_processed(processed, args.expected_rows)
    tables = build_tables(processed, tables_dir)
    plot_outputs(tables, figures_dir)
    write_report(processed, tables, args.out_dir, args.source_csv)

    manifest = {
        "source_csv": str(args.source_csv),
        "output_dir": str(args.out_dir),
        "pre_correction_csv": str(pre_correction_csv) if pre_correction_csv else None,
        "final_processed_csv": str(processed_csv),
        "mirror_data_csv": str(args.mirror_data_csv) if args.mirror_data_csv else None,
        "tables_dir": str(tables_dir),
        "figures_dir": str(figures_dir),
        "correction_summary": correction_summary,
        "validation": validation,
    }
    write_json(args.out_dir / "final_reprocess_manifest.json", manifest)
    write_json(tables_dir / "name_correction_summary.json", {**correction_summary, **validation})

    print(f"source_csv={args.source_csv}")
    print(f"output_dir={args.out_dir}")
    if pre_correction_csv:
        print(f"pre_correction_csv={pre_correction_csv}")
    print(f"final_processed_csv={processed_csv}")
    if args.mirror_data_csv:
        print(f"mirror_data_csv={args.mirror_data_csv}")
    print(f"rows={validation['processed_rows']}/{validation['expected_rows']}")
    print(f"core_parse_rate={validation['processed_core_parse_rate']:.6f}")
    print(f"full_parse_rate={validation['processed_full_parse_rate']:.6f}")
    print(f"missing_name={validation['processed_missing_name']}")
    print(f"correction_rows={correction_summary['correction_rows']}")
    print(f"correction_rows_still_missing_name={correction_summary['correction_rows_still_missing_name']}")
    print(f"tables={len(list(tables_dir.glob('*.csv')))}")
    print(f"figures={len(list(figures_dir.glob('*.png')))}")
    print(f"report={args.out_dir / 'analysis_report.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Final occupational-attribution reprocessing pipeline.")
    parser.add_argument("--source-csv", type=Path, default=Path("data/live_study1_results.csv"))
    parser.add_argument(
        "--correction-xlsx",
        type=Path,
        default=Path("outputs/live_occupational_analysis_20260527/not parsed yet.xlsx"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/live_occupational_analysis_20260527"))
    parser.add_argument("--mirror-data-csv", type=Path, default=None)
    parser.add_argument("--write-intermediates", action="store_true")
    parser.add_argument("--expected-rows", type=int, default=11250)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
