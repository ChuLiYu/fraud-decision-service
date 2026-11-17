#!/usr/bin/env bash
# Request latency with and without the shadow consumer (scope S-7, plan A3).
#
# Requires the compose stack to be up. Sends the same number of sequential
# decisions twice, first with the shadow consumer stopped and then running,
# and compares the server-side latency_ms recorded in the decision log (time
# spent scoring, excluding HTTP and database overhead on the client side).
# Waits for the shadow consumer to clear its backlog before the running run and
# checks it scored events while that run was sending. Fails if p50 or p99
# differ by more than the tolerance.
# Usage: scripts/shadow-latency.sh [N] [tolerance_ms]   (defaults 300, 10)
set -euo pipefail
n="${1:-300}"
tolerance="${2:-10}"
url="http://localhost:8000"
dc() { docker compose "$@"; }
sql() { dc exec -T postgres psql -U fraud -d fraud -tAc "$1"; }
fail() { echo "FAIL: $*" >&2; exit 1; }
wait_for() {  # description, command that succeeds when done
  for _ in $(seq 1 90); do
    if eval "$2"; then return 0; fi
    sleep 2
  done
  fail "timed out waiting for $1"
}
backlog() { sql "SELECT (SELECT COUNT(*) FROM outbox) - (SELECT COUNT(*) FROM shadow_scores)"; }

send() {  # run label, count
  local run="$1" count="$2" i code
  for i in $(seq 1 "$count"); do
    code=$(curl -sS -o /dev/null -w '%{http_code}' -X POST "$url/v1/decisions" \
      -H 'Content-Type: application/json' -H "Idempotency-Key: $run-$i" \
      -d "{\"transaction_id\": $((200000 + i)), \"timestamp_seconds\": 90000,
           \"amount\": \"$(( (i * 37) % 900 + 5 )).00\",
           \"fields\": {\"ProductCD\": \"W\", \"DeviceType\": \"mobile\", \"D1\": $i}}")
    [ "$code" = 200 ] || fail "request $i of $run returned HTTP $code"
  done
}

percentiles() {  # run label -> "p50 p99"
  sql "SELECT round(percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms)::numeric, 2),
              round(percentile_cont(0.99) WITHIN GROUP (ORDER BY latency_ms)::numeric, 2)
       FROM decisions WHERE idempotency_key LIKE '$1-%'" | tr '|' ' '
}

wait_for "decision service health" 'curl -sf "$url/health" >/dev/null'
id="lat-$(uuidgen | tr '[:upper:]' '[:lower:]')"
send "$id-warmup" 20

dc stop shadow >/dev/null 2>&1
send "$id-off" "$n"
read -r off50 off99 <<<"$(percentiles "$id-off")"

dc start shadow >/dev/null 2>&1
# Measure with the consumer working, not still starting: wait until it has
# scored the whole backlog, then count how many on-run events it scored while
# the run was still sending requests.
wait_for "shadow consumer to clear the backlog" '[ "$(backlog)" -eq 0 ]'
scored_before=$(sql "SELECT COUNT(*) FROM shadow_scores")
send "$id-on" "$n"
scored_during=$(( $(sql "SELECT COUNT(*) FROM shadow_scores") - scored_before ))
read -r on50 on99 <<<"$(percentiles "$id-on")"
[ "$scored_during" -gt 0 ] || fail "shadow scored no events during the shadow-running run"
wait_for "shadow consumer to score the run" '[ "$(backlog)" -eq 0 ]'

echo "requests per run: $n (sequential), tolerance: $tolerance ms"
echo "shadow stopped: p50 ${off50} ms, p99 ${off99} ms"
echo "shadow running: p50 ${on50} ms, p99 ${on99} ms"
echo "shadow scored $scored_during of $n events while the run was sending, backlog after the run: 0"
python3 - "$off50" "$off99" "$on50" "$on99" "$tolerance" <<'PY'
import sys
off50, off99, on50, on99, tol = map(float, sys.argv[1:])
d50, d99 = on50 - off50, on99 - off99
print(f"difference: p50 {d50:+.2f} ms, p99 {d99:+.2f} ms")
if d50 > tol or d99 > tol:
    sys.exit(f"FAIL: shadow scoring added more than {tol} ms")
print("PASS")
PY
