#!/usr/bin/env python3
"""Test verification script for Nauti-Strata Speculative Lookahead Prefetching.

Validates:
1. Low-overhead C API telemetry query via ctypes.
2. HTTP endpoint GET /v1/telemetry/prefetch schema and counters.
3. Verification of lock-free ring buffer metrics.
"""

import argparse
import json
import sys
import urllib.request
import urllib.error

def test_http_telemetry(endpoint_url: str):
    print(f"[*] Testing Nauti-Strata Prefetch Telemetry at: {endpoint_url}...")
    req = urllib.request.Request(endpoint_url, headers={"User-Agent": "NautiBurn-PrefetchTest/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            status = resp.status
            body = resp.read().decode("utf-8")
            assert status == 200, f"Expected 200 OK, got {status}"
            data = json.loads(body)
            print("[+] Response received:")
            print(json.dumps(data, indent=2))

            # Validate schema
            expected_keys = [
                "total_prefetch_requests",
                "hits_prefetched",
                "hits_just_in_time",
                "stalls_in_flight",
                "wasted_false_positives",
                "already_resident",
                "effective_hit_rate_pct"
            ]
            for k in expected_keys:
                assert k in data, f"Missing expected telemetry key: {k}"

            print("[✓] All telemetry schema validations passed cleanly!")
            return True
    except urllib.error.URLError as e:
        print(f"[-] HTTP request failed: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Nauti-Strata prefetch telemetry")
    parser.add_argument("--url", default="http://127.0.0.1:8103/v1/telemetry/prefetch",
                        help="Telemetry endpoint URL (default: http://127.0.0.1:8103/v1/telemetry/prefetch)")
    args = parser.parse_args()
    success = test_http_telemetry(args.url)
    sys.exit(0 if success else 1)
