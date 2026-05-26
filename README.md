# LLM Bias Assessment Research

This repository contains a live API collection pipeline plus a separate synthetic test-data workflow.

## Important Data Rule

Live collection writes to:

- `data/live_study1_results.csv`
- `data/live_study1_parsed.csv`
- `data/live_study2_results.csv`
- `data/live_study2_parsed.csv`
- `data/run_manifests/`

The older `data/study*_results.csv` files are synthetic test outputs. Do not use them as empirical model evidence.

## Setup

```powershell
pip install -r requirements.txt

$env:OPENAI_API_KEY = "..."
$env:ANTHROPIC_API_KEY = "..."
$env:GOOGLE_API_KEY = "..."
$env:DEEPSEEK_API_KEY = "..."
```

You can run only the models whose keys you have:

```powershell
python collectors.py --list-models
python verify_collectors.py --models openai-gpt-4.1 google-gemini-3-flash
```

## Full Live Collection

The collector is resumable by default. If the process stops, run the same command again and completed rows will be skipped.

```powershell
python collectors.py --study both
```

Run selected models:

```powershell
python collectors.py --study both --models openai-gpt-4.1 google-gemini-3.5-flash deepseek-v4-flash
```

Run a small pilot:

```powershell
python collectors.py --study both --models google-gemini-3.5-flash --iterations 3
```

Run the full live collection, parsing, analysis, and unified charts:

```powershell
python run_pipeline.py --study both
```

Preview planned calls without making API requests:

```powershell
python run_pipeline.py --study both --dry-run
```

## Model Configuration

Defaults are defined in `config.py` and can be overridden with environment variables:

- `OPENAI_MODEL_ID`, default `gpt-4.1`
- `ANTHROPIC_OPUS_MODEL_ID`, default `claude-opus-4-1-20250805`
- `ANTHROPIC_SONNET_MODEL_ID`, default `claude-sonnet-4-20250514`
- `GOOGLE_GEMINI_3_FLASH_MODEL_ID`, default `gemini-3-flash-preview`
- `GOOGLE_GEMINI_3_5_FLASH_MODEL_ID`, default `gemini-3.5-flash`
- `DEEPSEEK_PRO_MODEL_ID`, default `deepseek-v4-pro`
- `DEEPSEEK_FLASH_MODEL_ID`, default `deepseek-v4-flash`

Rate limits and retry behavior can also be tuned:

- `LLM_MAX_RETRIES`, default `12`
- `LLM_RETRY_BACKOFF_BASE`, default `2.0`
- `LLM_RETRY_MAX_WAIT_SECONDS`, default `300`
- provider-specific `*_RATE_LIMIT_DELAY` variables in `config.py`

Gemini 3.5 Flash is listed in Google Cloud Gemini Enterprise Agent Platform documentation. If your API key is scoped to a different Google Gemini surface, override `GOOGLE_GENERATE_CONTENT_URL` or `GOOGLE_GEMINI_3_5_FLASH_MODEL_ID` to the endpoint and model string available to your account.

## Synthetic Test Workflow

Use this only to test parsing and plotting without API costs:

```powershell
python run_pipeline.py --synthetic
```

Synthetic data remains useful for pipeline QA. It is not live model evidence.
