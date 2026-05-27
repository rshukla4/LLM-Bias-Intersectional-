"""
Live API collection engine for LLM bias assessment.

This module performs stateless, official-provider API calls and writes each
successful response incrementally. It does not use synthetic fallback responses.
Transient errors are retried with exponential backoff. Authentication and
permission errors fail closed so invalid keys cannot produce contaminated data.
"""

import argparse
import csv
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import requests

from config import (
    MANIFEST_DIR,
    MAX_RETRIES,
    MODELS,
    RATE_LIMIT_DELAY,
    RETRY_BACKOFF_BASE,
    RETRY_MAX_WAIT_SECONDS,
    STUDY1_CONFIG,
    STUDY2_CONFIG,
    ModelConfig,
)
from stimuli import (
    ALL_STUDY1_PHRASES,
    STUDY2_CONDITIONS,
    PatientCondition,
    Phrase,
    get_study1_prompt,
    get_study2_prompt,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class CollectionError(RuntimeError):
    """Raised when a model call cannot produce a valid live API response."""

    def __init__(self, message: str, *, retryable: bool = False, status_code: Optional[int] = None):
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
FATAL_STATUS_CODES = {400, 401, 403, 404}


STUDY1_CSV_COLUMNS = [
    "run_id",
    "source_type",
    "provider",
    "model",
    "provider_model_id",
    "model_version",
    "response_id",
    "finish_reason",
    "study_type",
    "phrase_id",
    "role_level",
    "industry",
    "condition",
    "iteration_id",
    "prompt",
    "raw_response",
    "latency_ms",
    "usage_json",
    "timestamp",
]

STUDY2_CSV_COLUMNS = [
    "run_id",
    "source_type",
    "provider",
    "model",
    "provider_model_id",
    "model_version",
    "response_id",
    "finish_reason",
    "study_type",
    "condition_id",
    "age",
    "race",
    "ses",
    "severity",
    "sofa_score",
    "comorbidities",
    "diagnosis",
    "prognosis",
    "age_code",
    "race_code",
    "ses_code",
    "severity_code",
    "iteration_id",
    "prompt",
    "raw_response",
    "latency_ms",
    "usage_json",
    "timestamp",
]


def _json_dumps(value: Any) -> str:
    return json.dumps(value or {}, ensure_ascii=True, sort_keys=True)


def _mask_key(value: str) -> str:
    if not value:
        return "<missing>"
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


def _init_csv(filepath: str, columns: Sequence[str], overwrite: bool = False) -> None:
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    if overwrite or not os.path.exists(filepath):
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=list(columns)).writeheader()


def _append_csv(filepath: str, columns: Sequence[str], row: Dict[str, Any]) -> None:
    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        writer.writerow(row)
        f.flush()


def _read_completed_keys(filepath: str, key_columns: Sequence[str]) -> Set[Tuple[str, ...]]:
    if not os.path.exists(filepath):
        return set()
    completed: Set[Tuple[str, ...]] = set()
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [col for col in key_columns if col not in (reader.fieldnames or [])]
        if missing:
            logger.warning("Resume disabled for %s; missing columns: %s", filepath, ", ".join(missing))
            return set()
        for row in reader:
            if row.get("source_type") == "live_api" and row.get("raw_response", "").strip():
                completed.add(tuple(str(row.get(col, "")) for col in key_columns))
    return completed


def _raise_for_http_error(response: requests.Response, provider: str) -> None:
    if response.status_code < 400:
        return
    body = response.text[:1000]
    msg = f"{provider} HTTP {response.status_code}: {body}"
    if response.status_code in RETRYABLE_STATUS_CODES:
        raise CollectionError(msg, retryable=True, status_code=response.status_code)
    if response.status_code in FATAL_STATUS_CODES:
        raise CollectionError(msg, retryable=False, status_code=response.status_code)
    raise CollectionError(msg, retryable=response.status_code >= 500, status_code=response.status_code)


def _retry_after_seconds(response: Optional[requests.Response]) -> Optional[float]:
    if response is None:
        return None
    raw = response.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


def _rate_limit_pause(model_key: str) -> None:
    delay = RATE_LIMIT_DELAY.get(model_key, 0.0)
    if delay > 0:
        time.sleep(delay)


def validate_api_keys(model_keys: Sequence[str]) -> None:
    missing = []
    for model_key in model_keys:
        config = MODELS[model_key]
        if not config.api_key:
            missing.append(f"{model_key} requires {config.api_key_env}")
    if missing:
        raise SystemExit(
            "Missing API keys. Set these environment variables before live collection:\n"
            + "\n".join(f"  - {item}" for item in missing)
        )


def call_openai_compatible(config: ModelConfig, prompt: str) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": config.model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": config.max_tokens,
    }
    if config.temperature is not None:
        payload["temperature"] = config.temperature
    if config.top_p is not None:
        payload["top_p"] = config.top_p

    response = requests.post(config.base_url, headers=headers, json=payload, timeout=config.timeout)
    _raise_for_http_error(response, config.provider)
    data = response.json()
    choice = data.get("choices", [{}])[0]
    message = choice.get("message", {})
    text = (message.get("content") or "").strip()
    if not text:
        raise CollectionError(f"{config.provider} returned an empty response.", retryable=True)
    return {
        "text": text,
        "model_version": data.get("model", config.model_id),
        "response_id": data.get("id", ""),
        "finish_reason": choice.get("finish_reason", ""),
        "usage": data.get("usage", {}),
        "provider_metadata": {"system_fingerprint": data.get("system_fingerprint")},
    }


def call_anthropic(config: ModelConfig, prompt: str) -> Dict[str, Any]:
    headers = {
        "x-api-key": config.api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": config.model_id,
        "max_tokens": config.max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if config.temperature is not None:
        payload["temperature"] = config.temperature
    if config.top_p is not None:
        payload["top_p"] = config.top_p

    response = requests.post(config.base_url, headers=headers, json=payload, timeout=config.timeout)
    _raise_for_http_error(response, config.provider)
    data = response.json()
    text = "".join(
        block.get("text", "")
        for block in data.get("content", [])
        if block.get("type") == "text"
    ).strip()
    if not text:
        raise CollectionError("Anthropic returned an empty response.", retryable=True)
    return {
        "text": text,
        "model_version": data.get("model", config.model_id),
        "response_id": data.get("id", ""),
        "finish_reason": data.get("stop_reason", ""),
        "usage": data.get("usage", {}),
        "provider_metadata": {"stop_sequence": data.get("stop_sequence")},
    }


def call_google(config: ModelConfig, prompt: str) -> Dict[str, Any]:
    url = f"{config.base_url.rstrip('/')}/{config.model_id}:generateContent"
    headers = {
        "x-goog-api-key": config.api_key,
        "Content-Type": "application/json",
    }
    generation_config: Dict[str, Any] = {"maxOutputTokens": config.max_tokens}
    if config.temperature is not None:
        generation_config["temperature"] = config.temperature
    if config.top_p is not None:
        generation_config["topP"] = config.top_p
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": generation_config,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=config.timeout)
    _raise_for_http_error(response, config.provider)
    data = response.json()
    candidates = data.get("candidates", [])
    first = candidates[0] if candidates else {}
    parts = first.get("content", {}).get("parts", [])
    text = "".join(part.get("text", "") for part in parts).strip()
    if not text:
        raise CollectionError("Google returned an empty response.", retryable=True)
    return {
        "text": text,
        "model_version": data.get("modelVersion", config.model_id),
        "response_id": data.get("responseId", ""),
        "finish_reason": first.get("finishReason", ""),
        "usage": data.get("usageMetadata", {}),
        "provider_metadata": {
            "prompt_feedback": data.get("promptFeedback"),
            "safety_ratings": first.get("safetyRatings"),
        },
    }


PROVIDER_CALLERS = {
    "openai": call_openai_compatible,
    "deepseek": call_openai_compatible,
    "anthropic": call_anthropic,
    "google": call_google,
}


def _fallback_config(config: ModelConfig, model_id: str) -> ModelConfig:
    return ModelConfig(
        name=config.name,
        provider=config.provider,
        model_id=model_id,
        api_key_env=config.api_key_env,
        base_url=config.base_url,
        max_tokens=config.max_tokens,
        timeout=config.timeout,
        temperature=config.temperature,
        top_p=config.top_p,
        source_type=config.source_type,
        fallback_model_id=None,
    )


def call_model(model_key: str, prompt: str) -> Dict[str, Any]:
    config = MODELS[model_key]
    configs = [config]
    if config.fallback_model_id:
        configs.append(_fallback_config(config, config.fallback_model_id))

    last_error: Optional[BaseException] = None
    for config_index, active_config in enumerate(configs):
        caller = PROVIDER_CALLERS[active_config.provider]
        if config_index > 0:
            logger.warning(
                "[%s] switching to fallback model_id=%s after primary model_id=%s failed.",
                model_key,
                active_config.model_id,
                config.model_id,
            )
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                started = time.time()
                result = caller(active_config, prompt)
                result["latency_ms"] = round((time.time() - started) * 1000, 1)
                result["source_type"] = active_config.source_type
                if "openai" in model_key:
                    result["provider"] = "openai"
                elif "anthropic" in model_key:
                    result["provider"] = "anthropic"
                elif "google" in model_key or "gemma" in model_key:
                    result["provider"] = "google"
                elif "deepseek" in model_key:
                    result["provider"] = "deepseek"
                else:
                    result["provider"] = active_config.provider
                result["provider_model_id"] = active_config.model_id
                _rate_limit_pause(model_key)
                return result
            except requests.Timeout as exc:
                last_error = exc
                retryable = True
                status_code = None
                message = f"{active_config.provider} request timed out: {exc}"
            except requests.RequestException as exc:
                last_error = exc
                retryable = True
                status_code = None
                message = f"{active_config.provider} request failed: {exc}"
            except CollectionError as exc:
                last_error = exc
                retryable = exc.retryable
                status_code = exc.status_code
                message = str(exc)

            if not retryable:
                if config_index + 1 < len(configs):
                    logger.warning("[%s] primary model failed permanently: %s", model_key, message)
                    break
                raise CollectionError(message, retryable=False, status_code=status_code) from last_error
            if attempt >= MAX_RETRIES:
                if config_index + 1 < len(configs):
                    logger.warning(
                        "[%s] primary model failed after %s attempts: %s",
                        model_key,
                        MAX_RETRIES,
                        message,
                    )
                    break
                raise CollectionError(
                    f"{model_key} failed after {MAX_RETRIES} attempts: {message}",
                    retryable=False,
                    status_code=status_code,
                ) from last_error

            wait = min(RETRY_MAX_WAIT_SECONDS, RETRY_BACKOFF_BASE ** attempt)
            logger.warning(
                "[%s] attempt %s/%s failed for model_id=%s; retrying in %.1fs: %s",
                model_key,
                attempt,
                MAX_RETRIES,
                active_config.model_id,
                wait,
                message,
            )
            time.sleep(wait)

    raise CollectionError(f"{model_key} failed unexpectedly.", retryable=False)


def _write_manifest(
    run_id: str,
    study: str,
    model_keys: Sequence[str],
    output_csv: str,
    total_planned: int,
    completed: int,
    status: str,
) -> None:
    manifest = {
        "run_id": run_id,
        "study": study,
        "status": status,
        "output_csv": output_csv,
        "total_planned": total_planned,
        "completed_this_process": completed,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "models": {
            key: {
                "name": MODELS[key].name,
                "provider": MODELS[key].provider,
                "provider_model_id": MODELS[key].model_id,
                "api_key_env": MODELS[key].api_key_env,
                "api_key_present": bool(MODELS[key].api_key),
                "temperature": MODELS[key].temperature,
                "top_p": MODELS[key].top_p,
                "max_tokens": MODELS[key].max_tokens,
                "timeout": MODELS[key].timeout,
            }
            for key in model_keys
        },
    }
    path = os.path.join(MANIFEST_DIR, f"{run_id}_{study}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=True)


def _select_models(model_keys: Optional[Sequence[str]]) -> List[str]:
    selected = list(model_keys) if model_keys else list(MODELS.keys())
    invalid = [key for key in selected if key not in MODELS]
    if invalid:
        raise SystemExit(
            "Unknown model key(s): "
            + ", ".join(invalid)
            + "\nAvailable models: "
            + ", ".join(MODELS.keys())
        )
    return selected


def _study1_tasks(
    model_keys: Sequence[str],
    phrase_ids: Optional[Sequence[str]],
    iterations: int,
) -> Iterable[Tuple[str, Phrase, int, str]]:
    phrases = ALL_STUDY1_PHRASES
    if phrase_ids:
        wanted = set(phrase_ids)
        phrases = [phrase for phrase in phrases if phrase.phrase_id in wanted]
    for model_key in model_keys:
        for phrase in phrases:
            prompt = get_study1_prompt(phrase)
            for iteration_id in range(1, iterations + 1):
                yield model_key, phrase, iteration_id, prompt


def _study2_tasks(
    model_keys: Sequence[str],
    condition_ids: Optional[Sequence[str]],
    iterations: int,
) -> Iterable[Tuple[str, PatientCondition, int, str]]:
    conditions = STUDY2_CONDITIONS
    if condition_ids:
        wanted = set(condition_ids)
        conditions = [condition for condition in conditions if condition.condition_id in wanted]
    for model_key in model_keys:
        for condition in conditions:
            prompt = get_study2_prompt(condition)
            for iteration_id in range(1, iterations + 1):
                yield model_key, condition, iteration_id, prompt


def collect_study1(
    model_keys: Optional[Sequence[str]] = None,
    phrase_ids: Optional[Sequence[str]] = None,
    iterations: Optional[int] = None,
    output_csv: Optional[str] = None,
    dry_run: bool = False,
    resume: bool = True,
    overwrite: bool = False,
    run_id: Optional[str] = None,
) -> str:
    models = _select_models(model_keys)
    if not dry_run:
        validate_api_keys(models)
    iters = iterations or STUDY1_CONFIG.iterations
    csv_path = output_csv or STUDY1_CONFIG.output_csv
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]

    tasks = list(_study1_tasks(models, phrase_ids, iters))
    total = len(tasks)
    written = 0
    if not dry_run:
        _init_csv(csv_path, STUDY1_CSV_COLUMNS, overwrite=overwrite)
        completed_keys = (
            _read_completed_keys(csv_path, ["model", "phrase_id", "iteration_id"])
            if resume and not overwrite
            else set()
        )
        _write_manifest(run_id, "study1", models, csv_path, total, written, "started")
    else:
        completed_keys = set()

    for index, (model_key, phrase, iteration_id, prompt) in enumerate(tasks, start=1):
        resume_key = (model_key, phrase.phrase_id, str(iteration_id))
        if resume_key in completed_keys:
            logger.info("[%s/%s] SKIP existing %s | %s | iter %s", index, total, model_key, phrase.phrase_id, iteration_id)
            continue
        tag = f"[{index}/{total}] {model_key} | {phrase.phrase_id} | iter {iteration_id}"
        if dry_run:
            logger.info("%s [DRY RUN] %s", tag, prompt[:120])
            continue

        logger.info("%s", tag)
        result = call_model(model_key, prompt)
        row = {
            "run_id": run_id,
            "source_type": result["source_type"],
            "provider": result["provider"],
            "model": model_key,
            "provider_model_id": result["provider_model_id"],
            "model_version": result["model_version"],
            "response_id": result.get("response_id", ""),
            "finish_reason": result.get("finish_reason", ""),
            "study_type": "occupational",
            "phrase_id": phrase.phrase_id,
            "role_level": phrase.role_level,
            "industry": phrase.industry,
            "condition": f"{phrase.role_level}_{phrase.industry}",
            "iteration_id": iteration_id,
            "prompt": prompt,
            "raw_response": result["text"],
            "latency_ms": result["latency_ms"],
            "usage_json": _json_dumps(result.get("usage")),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        _append_csv(csv_path, STUDY1_CSV_COLUMNS, row)
        written += 1

    if not dry_run:
        _write_manifest(run_id, "study1", models, csv_path, total, written, "complete")
    logger.info("Study 1 collection complete. New rows: %s. Data saved to %s", written, csv_path)
    return csv_path


def collect_study2(
    model_keys: Optional[Sequence[str]] = None,
    condition_ids: Optional[Sequence[str]] = None,
    iterations: Optional[int] = None,
    output_csv: Optional[str] = None,
    dry_run: bool = False,
    resume: bool = True,
    overwrite: bool = False,
    run_id: Optional[str] = None,
) -> str:
    models = _select_models(model_keys)
    if not dry_run:
        validate_api_keys(models)
    iters = iterations or STUDY2_CONFIG.iterations
    csv_path = output_csv or STUDY2_CONFIG.output_csv
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]

    tasks = list(_study2_tasks(models, condition_ids, iters))
    total = len(tasks)
    written = 0
    if not dry_run:
        _init_csv(csv_path, STUDY2_CSV_COLUMNS, overwrite=overwrite)
        completed_keys = (
            _read_completed_keys(csv_path, ["model", "condition_id", "iteration_id"])
            if resume and not overwrite
            else set()
        )
        _write_manifest(run_id, "study2", models, csv_path, total, written, "started")
    else:
        completed_keys = set()

    for index, (model_key, condition, iteration_id, prompt) in enumerate(tasks, start=1):
        resume_key = (model_key, condition.condition_id, str(iteration_id))
        if resume_key in completed_keys:
            logger.info("[%s/%s] SKIP existing %s | %s | iter %s", index, total, model_key, condition.condition_id, iteration_id)
            continue
        tag = f"[{index}/{total}] {model_key} | {condition.condition_id} | iter {iteration_id}"
        if dry_run:
            logger.info("%s [DRY RUN] %s", tag, prompt[:120])
            continue

        logger.info("%s", tag)
        result = call_model(model_key, prompt)
        row = {
            "run_id": run_id,
            "source_type": result["source_type"],
            "provider": result["provider"],
            "model": model_key,
            "provider_model_id": result["provider_model_id"],
            "model_version": result["model_version"],
            "response_id": result.get("response_id", ""),
            "finish_reason": result.get("finish_reason", ""),
            "study_type": "triage",
            "condition_id": condition.condition_id,
            "age": condition.age,
            "race": condition.race,
            "ses": condition.ses,
            "severity": condition.severity,
            "sofa_score": condition.sofa_score,
            "comorbidities": condition.comorbidities,
            "diagnosis": condition.diagnosis,
            "prognosis": condition.prognosis,
            "age_code": condition.age_code,
            "race_code": condition.race_code,
            "ses_code": condition.ses_code,
            "severity_code": condition.severity_code,
            "iteration_id": iteration_id,
            "prompt": prompt,
            "raw_response": result["text"],
            "latency_ms": result["latency_ms"],
            "usage_json": _json_dumps(result.get("usage")),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        _append_csv(csv_path, STUDY2_CSV_COLUMNS, row)
        written += 1

    if not dry_run:
        _write_manifest(run_id, "study2", models, csv_path, total, written, "complete")
    logger.info("Study 2 collection complete. New rows: %s. Data saved to %s", written, csv_path)
    return csv_path


def print_model_status(model_keys: Optional[Sequence[str]] = None) -> None:
    models = _select_models(model_keys)
    for key in models:
        cfg = MODELS[key]
        print(
            f"{key}: provider={cfg.provider}, model_id={cfg.model_id}, "
            f"key_env={cfg.api_key_env}, key={_mask_key(cfg.api_key)}, "
            f"temperature={cfg.temperature}, top_p={cfg.top_p}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Live LLM Bias Data Collection")
    parser.add_argument("--study", choices=["1", "2", "both"], default="both")
    parser.add_argument("--models", nargs="+", default=None, help="Model keys to test. Defaults to all configured models.")
    parser.add_argument("--iterations", type=int, default=None, help="Override iteration count for selected study or studies.")
    parser.add_argument("--output-csv", default=None, help="Override output path. Only valid for single-study runs.")
    parser.add_argument("--run-id", default=None, help="Stable run identifier for provenance manifests.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned calls without making API requests.")
    parser.add_argument("--no-resume", action="store_true", help="Do not skip rows already present in the output CSV.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output CSV before collection.")
    parser.add_argument("--list-models", action="store_true", help="Print configured models and key status, then exit.")
    args = parser.parse_args()

    if args.list_models:
        print_model_status(args.models)
        return

    if args.output_csv and args.study == "both":
        raise SystemExit("--output-csv can only be used with --study 1 or --study 2.")

    resume = not args.no_resume
    if args.study in ("1", "both"):
        collect_study1(
            model_keys=args.models,
            iterations=args.iterations,
            output_csv=args.output_csv if args.study == "1" else None,
            dry_run=args.dry_run,
            resume=resume,
            overwrite=args.overwrite,
            run_id=args.run_id,
        )
    if args.study in ("2", "both"):
        collect_study2(
            model_keys=args.models,
            iterations=args.iterations,
            output_csv=args.output_csv if args.study == "2" else None,
            dry_run=args.dry_run,
            resume=resume,
            overwrite=args.overwrite,
            run_id=args.run_id,
        )


if __name__ == "__main__":
    main()
