#!/usr/bin/env python3
"""Small local smoke test for backend model metadata/contract endpoints."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request


BASE_URL = "http://localhost:8000/api"


def get_json(path: str) -> dict:
    req = urllib.request.Request(f"{BASE_URL}{path}")
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body)


def assert_true(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)


def main() -> int:
    try:
        metadata = get_json("/model/metadata")
        contracts = get_json("/contracts")

        print("/model/metadata:")
        print(json.dumps(metadata, indent=2))
        print("\n/api/contracts:")
        print(json.dumps(contracts, indent=2))

        assert_true("available" in metadata, "metadata.available missing")
        assert_true("tensor_contract" in contracts, "contracts.tensor_contract missing")

        tc = contracts["tensor_contract"]
        assert_true(tc.get("input_shape") == [1, 64, 12], "Unexpected input shape")
        assert_true(tc.get("output_name") == "value", "Unexpected output name")
        assert_true(isinstance(tc.get("finite_output"), bool), "finite_output must be boolean")

        print("\nContract smoke check passed.")
        return 0
    except urllib.error.URLError as e:
        print(f"Connection error: {e}")
        return 2
    except Exception as e:
        print(f"Contract smoke failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
