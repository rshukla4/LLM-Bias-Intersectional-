# Study 2 Deferred Archive

Study 2 clinical triage work is deferred as of 2026-05-26.

This archive preserves the prior two-study materials without keeping them in the active Study 1 launch surface.

## Contents

| Path | Contents |
|---|---|
| `scripts/study2_analysis.py` | Deferred Study 2 triage analysis |
| `scripts/debias_optimization.py` | Deferred Study 2 debiasing/post-processing |
| `scripts/generate_unified_charts_two_study.py` | Previous mixed Study 1 and Study 2 unified chart script |
| `scratch/chart_qc/` | Study 2 chart QA data and outputs |
| `scratch/chart_qc_clean/` | Clean Study 2 chart QA data and outputs |
| `scratch/openrouter_smoke/` | Study 2 OpenRouter smoke-test CSVs |
| `docs/` | Prior two-study manuscript, proposal, notebook, process notes, and critique |
| `data/` | Reserved for Study 2 data files if restored or archived later |
| `outputs/` | Reserved for Study 2 output files if restored or archived later |

## Active Scope

The active repository root is now Study 1 only. Do not restore Study 2 into the active pipeline until the current Study 1 live collection is complete or intentionally stopped.

## Shared Runtime Files Left In Place

The following files may still contain Study 2 definitions because they are shared or were intentionally left untouched during the active Study 1 live run:

- `collectors.py`
- `stimuli.py`
- `config.py`
- `parsers.py`

These files should be refactored only after active collection finishes or after the run is intentionally stopped.
