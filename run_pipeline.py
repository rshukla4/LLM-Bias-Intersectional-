"""
End-to-end live collection and analysis runner.

Default behavior:
1. Collect live API responses into data/live_study*_results.csv.
2. Parse live responses into data/live_study*_parsed.csv.
3. Run the existing analysis scripts against live parsed data.

Use --synthetic to run the old local test-data workflow instead.
"""

import argparse
import subprocess
import sys

from config import STUDY1_CONFIG, STUDY2_CONFIG


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


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
    parser = argparse.ArgumentParser(description="Run live LLM bias collection, parsing, and analysis.")
    parser.add_argument("--study", choices=["1", "2", "both"], default="both")
    parser.add_argument("--models", nargs="+", default=None, help="Model keys to collect. Defaults to all configured models.")
    parser.add_argument("--iterations", type=int, default=None, help="Override iteration count.")
    parser.add_argument("--run-id", default=None, help="Stable run identifier for collection manifests.")
    parser.add_argument("--dry-run", action="store_true", help="Preview collection calls without API requests.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite live output CSVs before collection.")
    parser.add_argument("--skip-collection", action="store_true", help="Parse and analyze existing live CSVs only.")
    parser.add_argument("--synthetic", action="store_true", help="Run the old synthetic-data test workflow.")
    return parser.parse_args()


def run_collection(args):
    if args.skip_collection:
        print("\nSTEP 1: Skipping collection and using existing live CSVs.")
        return
    cmd = [sys.executable, "collectors.py", "--study", args.study]
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
    print("\nSTEP 1: Collecting live API responses.")
    run_cmd(cmd)


def run_synthetic_workflow(study="both"):
    print("\nSTEP 1: Generating synthetic test data.")
    run_cmd([sys.executable, "synthetic_data.py"])
    study1_raw = "data/study1_results.csv"
    study1_parsed = "data/study1_parsed.csv"
    study2_raw = "data/study2_results.csv"
    study2_parsed = "data/study2_parsed.csv"
    run_parse_and_analysis(study, study1_raw, study1_parsed, study2_raw, study2_parsed, "outputs")


def run_parse_and_analysis(study, study1_raw, study1_parsed, study2_raw, study2_parsed, output_dir):
    print("\nSTEP 2: Parsing responses.")
    if study == "1":
        parse_cmd = f"from parsers import parse_study1_csv; parse_study1_csv({study1_raw!r}, {study1_parsed!r})"
    elif study == "2":
        parse_cmd = f"from parsers import parse_study2_csv; parse_study2_csv({study2_raw!r}, {study2_parsed!r})"
    else:
        parse_cmd = (
            "from parsers import parse_study1_csv, parse_study2_csv; "
            f"parse_study1_csv({study1_raw!r}, {study1_parsed!r}); "
            f"parse_study2_csv({study2_raw!r}, {study2_parsed!r})"
        )
    run_cmd([sys.executable, "-c", parse_cmd])

    if study in ("1", "both"):
        print("\nSTEP 3: Running Study 1 analysis.")
        analysis1_cmd = (
            "from study1_analysis import run_full_study1_analysis; "
            f"run_full_study1_analysis({study1_parsed!r}, {output_dir!r})"
        )
        run_cmd([sys.executable, "-c", analysis1_cmd])

    if study in ("2", "both"):
        print("\nSTEP 4: Running Study 2 analysis.")
        analysis2_cmd = (
            "from study2_analysis import run_full_study2_analysis; "
            f"run_full_study2_analysis({study2_parsed!r}, {output_dir!r})"
        )
        run_cmd([sys.executable, "-c", analysis2_cmd])

    if study == "both":
        print("\nSTEP 5: Generating unified charts.")
        run_cmd([
            sys.executable,
            "generate_unified_charts.py",
            "--study1-csv",
            study1_parsed,
            "--study2-csv",
            study2_parsed,
            "--output-dir",
            output_dir,
        ])


def main():
    args = parse_args()
    print("=" * 50)
    if args.synthetic:
        run_synthetic_workflow(args.study)
        return

    run_collection(args)
    if args.dry_run:
        print("\nDry run complete. No parsing or analysis was performed.")
        return

    run_parse_and_analysis(
        args.study,
        STUDY1_CONFIG.output_csv,
        STUDY1_CONFIG.parsed_csv,
        STUDY2_CONFIG.output_csv,
        STUDY2_CONFIG.parsed_csv,
        "outputs",
    )
    print("\nLive research pipeline complete.")
    print("=" * 50)


if __name__ == "__main__":
    main()
