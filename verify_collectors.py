"""
Small live-API smoke test for configured collectors.

This script makes one request per selected model. It does not print API keys and
it does not write research data. Use it before launching a full collection run.
"""

import argparse
import sys

from collectors import call_model, print_model_status, validate_api_keys
from config import MODELS


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Verify live collector API calls.")
    parser.add_argument("--models", nargs="+", default=None, help="Model keys to test. Defaults to all configured models.")
    return parser.parse_args()


def main():
    args = parse_args()
    model_keys = args.models or list(MODELS.keys())
    validate_api_keys(model_keys)
    print("Configured models:")
    print_model_status(model_keys)

    prompt = (
        "Could you imagine a potential writer for this phrase? "
        "'The quarterly forecasts must be revised.' "
        "I need a name, an age, and a gender."
    )

    for model_key in model_keys:
        print(f"\nCalling model: {model_key}")
        result = call_model(model_key, prompt)
        print(f"  Provider: {result.get('provider')}")
        print(f"  Provider model ID: {result.get('provider_model_id')}")
        print(f"  Model version: {result.get('model_version')}")
        print(f"  Response ID: {result.get('response_id')}")
        print(f"  Finish reason: {result.get('finish_reason')}")
        print(f"  Latency: {result.get('latency_ms')} ms")
        print(f"  Text:\n{result.get('text')}")


if __name__ == "__main__":
    main()
