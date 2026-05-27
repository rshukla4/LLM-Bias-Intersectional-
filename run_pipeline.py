"""
Study 1-only live collection and analysis runner.

Current project scope:
1. Collect live Study 1 occupational-attribution responses.
2. Parse Study 1 name, age, gender, and inferred race fields.
3. Run Study 1 statistical analysis.
4. Generate Study 1 publication charts.

Study 2 is deferred and archived under archive/study2_deferred_20260526/.
"""

import argparse
import subprocess
import sys

from config import STUDY1_CONFIG


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


DEFERRED_STUDY2_MESSAGE = (
    "Study 2 is deferred for this launch. Restore archive/study2_deferred_20260526 "
    "before running Study 2 collection or analysis."
)


def run_cmd(cmd):
    print(f"\nExecuting: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.stdout:
        print(result.stdout.strip())
    if result.returncode != 0:
        print(f"Error (Code {result.returncode}):")
        if result.stderr:
            print(result.stderr.strip())
        raise SystemExit(result.returncode)
    if result.stderr:
        print(result.stderr.strip())


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Study 1 LLM bias collection, parsing, and analysis pipeline.")
    parser.add_argument("--study", choices=["1", "2", "both"], default="1", help="Defaults to Study 1. Study 2 is archived.")
    parser.add_argument("--models", nargs="+", default=None, help="Model keys to collect. Defaults to all configured models.")
    parser.add_argument("--iterations", type=int, default=None, help="Override iteration count.")
    parser.add_argument("--run-id", default=None, help="Stable run identifier for collection manifests.")
    parser.add_argument("--dry-run", action="store_true", help="Preview Study 1 collection calls without API requests.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite live Study 1 output CSV before collection.")
    parser.add_argument("--skip-collection", action="store_true", help="Parse and analyze existing live Study 1 CSV only.")
    parser.add_argument("--synthetic", action="store_true", help="Run a Study 1 synthetic-data QA workflow.")
    return parser.parse_args()


def enforce_study1_scope(study):
    if study != "1":
        raise SystemExit(DEFERRED_STUDY2_MESSAGE)


def run_collection(args):
    if args.skip_collection:
        print("\nSTEP 1: Skipping collection and using the existing live Study 1 CSV.")
        return

    cmd = [sys.executable, "collectors.py", "--study", "1"]
    if args.models:
        cmd.extend(["--models", *args.models])
    if args.iterations is not None:
        cmd.extend(["--iterations", str(args.iterations)])
    if args.run_id:
        cmd.extend(["--run-id", args.run_id])
    if args.dry_run:
        cmd.append("--dry-run")
    if args.overwrite:
        cmd.append("--overwrite")

    print("\nSTEP 1: Collecting live Study 1 API responses.")
    run_cmd(cmd)


def run_synthetic_workflow():
    print("\nSTEP 1: Generating synthetic Study 1 test data.")
    run_cmd([
        sys.executable,
        "-c",
        "from synthetic_data import generate_study1_synthetic; generate_study1_synthetic(output_csv='data/study1_results.csv')",
    ])
    run_parse_and_analysis("data/study1_results.csv", "data/study1_parsed.csv", "outputs")


def run_parse_and_analysis(study1_raw, study1_parsed, output_dir):
    print("\nSTEP 2: Parsing Study 1 responses.")
    parse_cmd = f"from parsers import parse_study1_csv; parse_study1_csv({study1_raw!r}, {study1_parsed!r})"
    run_cmd([sys.executable, "-c", parse_cmd])

    print("\nSTEP 3: Running Study 1 analysis.")
    analysis_cmd = (
        "from study1_analysis import run_full_study1_analysis; "
        f"run_full_study1_analysis({study1_parsed!r}, {output_dir!r})"
    )
    run_cmd([sys.executable, "-c", analysis_cmd])

    print("\nSTEP 4: Generating Study 1 unified charts.")
    run_cmd([
        sys.executable,
        "generate_unified_charts.py",
        "--study1-csv",
        study1_parsed,
        "--output-dir",
        output_dir,
    ])


def main():
    args = parse_args()
    enforce_study1_scope(args.study)

    print("=" * 50)
    if args.synthetic:
        run_synthetic_workflow()
        print("\nStudy 1 synthetic QA pipeline complete.")
        print("=" * 50)
        return

    run_collection(args)
    if args.dry_run:
        print("\nDry run complete. No parsing or analysis was performed.")
        print("=" * 50)
        return

    run_parse_and_analysis(
        STUDY1_CONFIG.output_csv,
        STUDY1_CONFIG.parsed_csv,
        "outputs",
    )
    print("\nStudy 1 live research pipeline complete.")
    print("=" * 50)


if __name__ == "__main__":
    main()
