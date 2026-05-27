"""Merge targeted name/age parsing corrections into a processed CSV.

The correction workbook is an .xlsx file exported from the parsed data rows
where parsed_name was still missing. This script reads the workbook without
requiring openpyxl, reparses every row in it, and merges corrected fields into
a new processed CSV while leaving raw and parsed source files untouched.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parsers import classify_race_from_name  # noqa: E402
from scripts.analyze_occupational_attribution import (  # noqa: E402
    GENDER_LABELS,
    parse_gender_live,
    parse_refusal,
)


NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def column_index(cell_ref: str) -> int:
    match = re.match(r"([A-Z]+)", cell_ref)
    if not match:
        raise ValueError(f"Invalid Excel cell reference: {cell_ref}")
    value = 0
    for char in match.group(1):
        value = value * 26 + ord(char) - 64
    return value - 1


def read_xlsx_first_sheet(path: Path) -> pd.DataFrame:
    with ZipFile(path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in shared_root.findall("a:si", NS):
                shared_strings.append("".join(text.text or "" for text in item.findall(".//a:t", NS)))

        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        rows: list[dict[int, str]] = []
        max_col = 0
        for row in sheet.findall("a:sheetData/a:row", NS):
            values: dict[int, str] = {}
            for cell in row.findall("a:c", NS):
                idx = column_index(cell.attrib["r"])
                max_col = max(max_col, idx + 1)
                cell_type = cell.attrib.get("t")
                value_node = cell.find("a:v", NS)
                if cell_type == "s" and value_node is not None:
                    value = shared_strings[int(value_node.text or "0")]
                elif cell_type == "inlineStr":
                    value = "".join(text.text or "" for text in cell.findall(".//a:t", NS))
                elif value_node is not None:
                    value = value_node.text or ""
                else:
                    value = ""
                values[idx] = value
            rows.append(values)

    if not rows:
        return pd.DataFrame()
    header = [rows[0].get(i, "") for i in range(max_col)]
    records = []
    for values in rows[1:]:
        records.append({header[i]: values.get(i, "") for i in range(max_col)})
    return pd.DataFrame(records)


def normalize_text(text: Any) -> str:
    text = "" if text is None else str(text)
    replacements = {
        "\r\n": "\n",
        "\r": "\n",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2010": "-",
        "\u2011": "-",
        "\u2013": "-",
        "\u2014": "-",
        "â€”": "-",
        "â€“": "-",
        "â€‘": "-",
        "â€œ": '"',
        "â€�": '"',
        "â€": '"',
        "â€\x9d": '"',
        "â€˜": "'",
        "â€™": "'",
        "â€”": "-",
        "â€“": "-",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.replace("\u00e2\u20ac\"", "-")
    text = text.replace("\u00e2\u20ac'", "'")
    text = re.sub(r"[*_`]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_candidate(candidate: str) -> str | None:
    candidate = normalize_text(candidate)
    candidate = re.sub(r"^(?:Character|Name)\s*[:=-]\s*", "", candidate, flags=re.I)
    candidate = re.sub(r'"[^"]+"', " ", candidate)
    candidate = re.sub(r"'[^']+'", " ", candidate)
    candidate = re.sub(r"\(\s*or\s+([A-Z][A-Za-z'.-]+)\s*\)", " ", candidate)
    candidate = re.sub(r"\([^)]*\)", " ", candidate)
    candidate = re.sub(r"\b(?:Age|Gender|Reasoning|Rationale|Context|Profile|Why)\b.*$", "", candidate, flags=re.I)
    candidate = re.split(r"\s{2,}|\||;|,|\bage\b|\bgender\b", candidate, flags=re.I)[0]
    candidate = candidate.strip(" \t\n:-*#.")
    candidate = re.sub(r"^(?:Dr|Mr|Ms|Mrs)\.\s+", "", candidate, flags=re.I)
    candidate = re.sub(
        r"^(?:Surgical Nurse|Registered Nurse|Nurse|Paralegal|Legal Assistant|Administrative Coordinator|Executive Assistant|Male Writer|Female Writer)\s+",
        "",
        candidate,
        flags=re.I,
    )
    candidate = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip()

    prefix_pattern = (
        r"^(?:"
        r"a|an|the|one|possible|potential|plausible|fictional|writer|profile|"
        r"person|candidate|could|be|is|would|here|are|there|named|called"
        r")\b\s*"
    )
    changed = True
    while changed:
        new_candidate = re.sub(prefix_pattern, "", candidate, flags=re.I).strip(" :-")
        changed = new_candidate != candidate
        candidate = new_candidate

    tokens = candidate.split()
    if len(tokens) > 5:
        return None
    if len(tokens) == 1:
        single_ok = re.match(r"^[A-Z][A-Za-z'.-]{2,}$", candidate) and candidate.lower() not in {
            "based",
            "here",
            "certainly",
            "profile",
            "background",
            "context",
            "reasoning",
        }
        return candidate if single_ok else None
    if not re.match(r"^[A-Z][A-Za-z'.-]+(?: [A-Z][A-Za-z'.-]+){1,4}$", candidate):
        return None
    lower = candidate.lower()
    bad_phrases = {
        "chief executive",
        "senior partner",
        "potential writer",
        "fictional writer",
        "likely writer",
        "profile reasoning",
    }
    if lower in bad_phrases:
        return None
    return candidate


def parse_corrected_name(raw_response: str) -> str | None:
    raw_text = "" if raw_response is None else str(raw_response)
    text = normalize_text(raw_response)

    chinese_name = re.search(r"(?:姓名|名字)\s*[:：]\s*([\u4e00-\u9fff]{2,4})", text)
    if chinese_name:
        return chinese_name.group(1)

    # Prefer explicit bolded profile fields before loose comma patterns. This
    # prevents full names such as "Victoria Cross" from being reduced to the
    # surname by a later single-token fallback.
    for span in re.findall(r"\*\*([^*]+)\*\*", raw_text):
        name = clean_candidate(span)
        if name:
            return name

    patterns = [
        r"\bName\s*[:=-]\s*([A-Z][A-Za-z'.-]+(?:\s+\"?[A-Z][A-Za-z'.-]+\"?){1,4})",
        r"\bName\s*[:=-]\s*([A-Z][A-Za-z'.-]{2,})\b",
        r"\b(?:Potential writer|Possible writer|One fictional possibility|One possible fictional writer|A potential writer could be|A plausible fictional writer could be|A possible fictional writer could be)\s*[:=-]?\s*([A-Z][A-Za-z'.-]+(?:\s+\"?[A-Z][A-Za-z'.-]+\"?){1,4})",
        r"\b(?:writer|person|profile|candidate|line|phrase)\s+(?:is|as|could be|would be|might be|likely is)\s+(?:Dr\.\s+)?([A-Z][A-Za-z'.-]+(?:\s+\"?[A-Z][A-Za-z'.-]+\"?){0,4})",
        r"\b(?:I imagine|I envision|I'd imagine|I'd picture|I can imagine|I could picture|I picture|to me, this sounds like)\s+(?:a writer\s+like\s+|someone\s+like\s+|the writer\s+)?(?:as|is)?\s*(?:Dr\.\s+)?([A-Z][A-Za-z'.-]+(?:\s+\"?[A-Z][A-Za-z'.-]+\"?){0,4})",
        r"\b(?:be|as|is)\s+(?:Dr\.\s+)?([A-Z][A-Za-z'.-]+(?:\s+\"?[A-Z][A-Za-z'.-]+\"?){1,4})\s*,\s+a\s+\d{1,3}[- ]year[- ]old",
        r"\b([A-Z][A-Za-z'.-]+(?:\s+\"?[A-Z][A-Za-z'.-]+\"?){1,4})\s*,\s*(?:age\s*)?\d{1,3}(?:\s*years old)?\s*,\s*(?:male|female|non[- ]?binary|man|woman)\b",
        r"\b([A-Z][A-Za-z'.-]+(?:\s+\"?[A-Z][A-Za-z'.-]+\"?){1,4})\s*,\s*(?:age\s*)?\d{1,3}(?:\s*years old)\b",
        r"\b([A-Z][A-Za-z'.-]+)\s*,\s*(?:age\s*)?\d{1,3}\s*,\s*(?:male|female|non[- ]?binary|man|woman)\b",
        r"\b(?:by|called|named)\s+([A-Z][A-Za-z'.-]+(?:\s+[A-Z][A-Za-z'.-]+){1,4})\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.I):
            name = clean_candidate(match.group(1))
            if name:
                return name

    # Fallback for responses where the name is alone on a first profile line.
    for part in re.split(r"\n| {2,}", normalize_text(raw_response)):
        name = clean_candidate(part)
        if name:
            return name
    return None


def parse_corrected_age(raw_response: str) -> int | None:
    text = normalize_text(raw_response)
    patterns = [
        r"(?:年龄|年齡)\s*[:：]\s*(\d{1,3})\s*岁",
        r"\bAge\s*[:=-]?\s*(?:around|about|approximately)?\s*\d{1,3}\s*[-]\s*(\d{1,3})\s*years old\b",
        r"\bAge\s*[:=-]?\s*(?:likely|around|about|approximately)?\s*(\d{1,3})(?=\D|$)",
        r"\bage\s+(\d{1,3})\b",
        r"\bage\s+(?:would be|is|could be|might be)\s*(?:around|about|approximately)?\s*(\d{1,3})\b",
        r"\b(?:plausible\s+)?(?:name and age|age)\s+could be\s+[A-Z][A-Za-z'.-]+(?:\s+[A-Z][A-Za-z'.-]+){0,4},\s*(\d{1,3})\b",
        r"\bplausible age\s+[^0-9]{0,60}(\d{1,3})\b",
        r"\be\.g\.,\s*(\d{1,3})\b",
        r"\baround\s+(\d{1,3})\s*years old\b",
        r"\bearly-to-mid 20s, for instance\s+(\d{1,3})\s*years old\b",
        r"\b(\d{1,3})\s*years old\b",
        r"\b(\d{1,3})\s*[- ]?year[- ]old\b",
        r"\b[A-Z][A-Za-z'.-]+(?:\s+\"?[A-Z][A-Za-z'.-]+\"?){0,4}\s*,\s*(\d{1,3})\s*,\s*(?:male|female|non[- ]?binary|man|woman)\b",
        r"\b[A-Z][A-Za-z'.-]+(?:\s+\"?[A-Z][A-Za-z'.-]+\"?){0,4}\s*,\s*(\d{1,3})\s*,\s*(?:senior|junior|partner|assistant|coordinator|manager|physician|nurse|analyst)\b",
        r"\b[A-Z][A-Za-z'.-]+(?:\s+\"?[A-Z][A-Za-z'.-]+\"?){1,4}\s*,\s*(\d{1,3})\s*,",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            age = int(match.group(1))
            if 1 <= age <= 120:
                return age
    return None


def corrected_parse(row: pd.Series) -> dict[str, Any]:
    raw_response = row.get("raw_response", "")
    name = parse_corrected_name(raw_response)
    age = parse_corrected_age(raw_response)
    gender = parse_gender_live(raw_response)
    refusal = parse_refusal(raw_response)

    race_info = {"inferred_race": None, "confidence": 0.0, "method": "none"}
    if name:
        race_info = classify_race_from_name(name, "simple")

    errors = []
    if not name:
        errors.append("missing_name")
    if age is None:
        errors.append("missing_age")
    if gender is None:
        errors.append("missing_gender")
    if refusal:
        errors.append("refusal_or_caveat")

    return {
        "parsed_name": name,
        "parsed_age": age,
        "parsed_gender": gender,
        "parsed_gender_label": GENDER_LABELS.get(gender),
        "is_refusal": bool(refusal),
        "inferred_race": race_info["inferred_race"],
        "race_confidence": race_info["confidence"],
        "race_method": race_info["method"],
        "parse_errors": ";".join(errors),
        "parse_valid_core": bool(name is not None and gender is not None),
        "parse_valid_full": bool(name is not None and age is not None and gender is not None),
        "correction_source": "not parsed yet.xlsx",
    }


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parsed-csv", type=Path, default=Path("data/live_study1_parsed.csv"))
    parser.add_argument(
        "--correction-xlsx",
        type=Path,
        default=Path("outputs/live_occupational_analysis_20260527/not parsed yet.xlsx"),
    )
    parser.add_argument("--output-csv", type=Path, default=Path("data/live_study1_processed.csv"))
    parser.add_argument(
        "--audit-csv",
        type=Path,
        default=Path("outputs/live_occupational_analysis_20260527/tables/name_correction_audit.csv"),
    )
    args = parser.parse_args()

    parsed = pd.read_csv(args.parsed_csv)
    corrections_raw = read_xlsx_first_sheet(args.correction_xlsx)
    required = ["model", "phrase_id", "iteration_id", "response_id", "raw_response"]
    missing = [col for col in required if col not in corrections_raw.columns]
    if missing:
        raise SystemExit(f"Correction workbook missing required columns: {missing}")

    corrections = corrections_raw.copy()
    parsed_rows = []
    for _, row in corrections.iterrows():
        parsed_rows.append(corrected_parse(row))
    corrected_fields = pd.DataFrame(parsed_rows)
    corrections = pd.concat([corrections.reset_index(drop=True), corrected_fields.add_prefix("corrected_")], axis=1)

    key_cols = ["model", "phrase_id", "iteration_id", "response_id"]
    parsed["_merge_key"] = parsed[key_cols].astype(str).agg("||".join, axis=1)
    corrections["_merge_key"] = corrections[key_cols].astype(str).agg("||".join, axis=1)

    if corrections["_merge_key"].duplicated().any():
        dupes = corrections[corrections["_merge_key"].duplicated(keep=False)][key_cols]
        raise SystemExit(f"Duplicate correction keys found:\n{dupes.to_string(index=False)}")

    missing_keys = sorted(set(corrections["_merge_key"]) - set(parsed["_merge_key"]))
    if missing_keys:
        raise SystemExit(f"{len(missing_keys)} correction rows do not match parsed CSV keys.")

    processed = parsed.copy()
    processed["name_corrected_from_workbook"] = False
    processed["original_parsed_name_before_workbook"] = processed.get("parsed_name")
    processed["original_parse_errors_before_workbook"] = processed.get("parse_errors")

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
    corr_by_key = corrections.set_index("_merge_key")
    for idx, row in processed[processed["_merge_key"].isin(corr_by_key.index)].iterrows():
        corr = corr_by_key.loc[row["_merge_key"]]
        changed = False
        for target, source in field_map.items():
            value = corr[source]
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
            old_value = processed.at[idx, target]
            if str(old_value) != str(value):
                changed = True
            processed.at[idx, target] = value
        processed.at[idx, "name_corrected_from_workbook"] = changed

    processed = processed.drop(columns=["_merge_key"])
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(args.output_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    audit_cols = key_cols + [
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
    args.audit_csv.parent.mkdir(parents=True, exist_ok=True)
    corrections[audit_cols].to_csv(args.audit_csv, index=False)

    remaining_missing = int(processed["parsed_name"].isna().sum())
    correction_missing = int(corrections["corrected_parsed_name"].isna().sum())
    print(f"parsed_rows={len(parsed)}")
    print(f"correction_rows={len(corrections)}")
    print(f"correction_rows_still_missing_name={correction_missing}")
    print(f"processed_rows={len(processed)}")
    print(f"processed_remaining_missing_name={remaining_missing}")
    print(f"output_csv={args.output_csv}")
    print(f"audit_csv={args.audit_csv}")


if __name__ == "__main__":
    main()
