#!/usr/bin/env bash
# Send sample requests to a running service and print the decisions.
# Usage: scripts/demo.sh [base_url]   (default http://localhost:8000)
set -euo pipefail
url="${1:-http://localhost:8000}"

post() {  # name, idempotency key, JSON body
  echo "== $1"
  curl -sS -X POST "$url/v1/decisions" -H 'Content-Type: application/json' \
    -H "Idempotency-Key: $2" -d "$3" -w '\nHTTP %{http_code}\n'
}

echo "== health"; curl -sS "$url/health"; echo
low='{"transaction_id":1,"timestamp_seconds":90000,"amount":"12.50","fields":{"ProductCD":"W","D1":300,"DeviceType":"desktop","card4":"visa","card6":"debit","P_emaildomain":"gmail.com","C1":2}}'
high='{"transaction_id":2,"timestamp_seconds":90000,"amount":"950.00","fields":{"ProductCD":"C","D1":1,"DeviceType":"mobile","card4":"visa","card6":"credit","P_emaildomain":"gmail.com","C1":2}}'
key="demo-$(date +%s)"
post "low risk" "$key-low" "$low"
post "high risk" "$key-high" "$high"
post "same key, same body (replayed)" "$key-low" "$low"
post "same key, different body (409)" "$key-low" "$high"
post "invalid amount (422)" "$key-bad" '{"transaction_id":3,"timestamp_seconds":1,"amount":"12.5.0"}'
