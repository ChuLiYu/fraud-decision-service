#!/usr/bin/env bash
# Request latency with and without the shadow consumer (scope S-7, plan A3).
#
# Requires the compose stack to be up. Runs PAIRS pairs of blocks; each pair is
# one block of BLOCK sequential decisions with the shadow consumer stopped and
# one with it running, in alternating order (off/on, then on/off, ...) so that
# drift in host load does not favour one side. Latency is the server-side
# latency_ms stored with each decision (scoring time, excluding HTTP and
# database overhead on the client side).
#
# For every "on" block the script first waits until the shadow consumer has
# scored the whole backlog, then requires it to score events while the block
# is still sending, and waits for the backlog to clear again afterwards.
#
# Judgement: per pair, p50 and p99 of the "on" block minus the "off" block; the
# result is the median of those differences over all pairs. Fails if either
# median exceeds TOLERANCE ms. With TOLERANCE "none" the differences are printed
# but not judged (used in CI, where shared runners make latency too noisy to
# gate on; the shadow checks still apply).
# Usage: scripts/shadow-latency.sh [PAIRS] [BLOCK] [TOLERANCE|none]
#        (defaults 5, 200, 10)
set -euo pipefail
pairs="${1:-5}"
block="${2:-200}"
tolerance="${3:-10}"
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
scored() { sql "SELECT COUNT(*) FROM shadow_scores"; }

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

run_off() {  # label
  dc stop shadow >/dev/null 2>&1
  send "$1" "$block"
}

during=0
run_on() {  # label; sets $during to the events scored while the block was sending
  local before
  dc start shadow >/dev/null 2>&1
  wait_for "shadow consumer to clear the backlog" '[ "$(backlog)" -eq 0 ]'
  before=$(scored)
  send "$1" "$block"
  during=$(( $(scored) - before ))
  [ "$during" -gt 0 ] || fail "shadow scored no events while $1 was sending"
  wait_for "shadow consumer to score $1" '[ "$(backlog)" -eq 0 ]'
}

wait_for "decision service health" 'curl -sf "$url/health" >/dev/null'
id="lat-$(uuidgen | tr '[:upper:]' '[:lower:]')"
send "$id-warmup" 50

echo "pairs: $pairs, block: $block sequential requests, tolerance: $tolerance ms"
results="$(mktemp)"
trap 'rm -f "$results"' EXIT
for p in $(seq 1 "$pairs"); do
  if [ $((p % 2)) -eq 1 ]; then
    run_off "$id-$p-off"; run_on "$id-$p-on"; order="off,on"
  else
    run_on "$id-$p-on"; run_off "$id-$p-off"; order="on,off"
  fi
  read -r off50 off99 <<<"$(percentiles "$id-$p-off")"
  read -r on50 on99 <<<"$(percentiles "$id-$p-on")"
  echo "$off50 $off99 $on50 $on99" >> "$results"
  echo "pair $p ($order): off p50 $off50 p99 $off99 | on p50 $on50 p99 $on99 ms" \
    "| shadow scored $during of $block during the on block"
done
dc start shadow >/dev/null 2>&1

python3 - "$results" "$tolerance" <<'PY'
import statistics
import sys

rows = [[float(x) for x in line.split()] for line in open(sys.argv[1])]
d50 = [on50 - off50 for off50, _, on50, _ in rows]
d99 = [on99 - off99 for _, off99, _, on99 in rows]
m50, m99 = statistics.median(d50), statistics.median(d99)
print("p50 differences: " + ", ".join(f"{d:+.2f}" for d in d50))
print("p99 differences: " + ", ".join(f"{d:+.2f}" for d in d99))
print(f"median difference: p50 {m50:+.2f} ms, p99 {m99:+.2f} ms")
if sys.argv[2] == "none":
    print("PASS (shadow checks only; latency not judged)")
    sys.exit()
tol = float(sys.argv[2])
if m50 > tol or m99 > tol:
    sys.exit(f"FAIL: median shadow overhead exceeds {tol} ms")
print("PASS")
PY
