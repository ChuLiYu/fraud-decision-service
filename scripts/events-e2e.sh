#!/usr/bin/env bash
# Fault-injection check for decision events (ADR-006, scope phase 3).
#
# Requires the compose stack to be up (docker compose up --build -d --wait).
#  1. Stop Kafka Connect, so committed decisions cannot be published yet.
#  2. Commit N decisions, then SIGKILL the decision service.
#  3. Start Connect again: every outbox row must reach the topic (no loss) and
#     every event on the topic must have an outbox row (no phantom).
#  4. Consumer: every event counted exactly once.
#  5. Rewind the consumer group to the start and restart the consumer: all
#     events are redelivered, counts must not change (duplicates tolerated).
# Usage: scripts/events-e2e.sh [N]   (default 50)
set -euo pipefail
n="${1:-50}"
url="http://localhost:8000"
topic="fraud.decision-events"
group="decision-outcome-counts"
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
topic_event_ids() {  # event ids of every message on the topic (duplicates kept)
  dc exec -T kafka /opt/kafka/bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 \
    --topic "$topic" --from-beginning --timeout-ms 10000 2>/dev/null \
    | grep -o '"event_id": *"[^"]*"' | sed 's/.*"\([^"]*\)"$/\1/' || true
}

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

before=$(sql "SELECT COUNT(*) FROM outbox")
echo "outbox rows before: $before"

echo "== 1. stop Kafka Connect (publication paused)"
dc stop connect >/dev/null 2>&1

echo "== 2. commit $n decisions, then SIGKILL the decision service"
run="e2e-$(date +%s)"
for i in $(seq 1 "$n"); do
  amount=$(( (i * 37) % 900 + 5 )).00
  code=$(curl -sS -o /dev/null -w '%{http_code}' -X POST "$url/v1/decisions" \
    -H 'Content-Type: application/json' -H "Idempotency-Key: $run-$i" \
    -d "{\"transaction_id\": $((100000 + i)), \"timestamp_seconds\": 90000, \"amount\": \"$amount\",
         \"fields\": {\"ProductCD\": \"W\", \"DeviceType\": \"mobile\", \"D1\": $i}}")
  [ "$code" = 200 ] || fail "decision $i returned HTTP $code"
done
dc kill -s SIGKILL decision >/dev/null 2>&1
committed=$(sql "SELECT COUNT(*) FROM outbox")
[ "$committed" -eq $((before + n)) ] || fail "outbox has $committed rows, expected $((before + n))"
echo "committed outbox rows: $committed (service killed before any of the new events was published)"

echo "== 3. start Connect; wait for the consumer to count every outbox row"
dc start connect >/dev/null 2>&1
wait_for "consumer to catch up" '[ "$(sql "SELECT COUNT(*) FROM consumed_events")" -eq "$committed" ]'
sql "SELECT id::text FROM outbox" | sort > "$tmp/outbox"
topic_event_ids > "$tmp/topic-all"
sort -u "$tmp/topic-all" > "$tmp/topic"
on_topic=$(wc -l < "$tmp/topic-all" | tr -d ' ')
distinct=$(wc -l < "$tmp/topic" | tr -d ' ')
lost=$(comm -23 "$tmp/outbox" "$tmp/topic" | wc -l | tr -d ' ')
phantom=$(comm -13 "$tmp/outbox" "$tmp/topic" | wc -l | tr -d ' ')
echo "topic messages: $on_topic, distinct event ids: $distinct, lost: $lost, phantom: $phantom"
[ "$lost" -eq 0 ] || fail "$lost outbox events never reached the topic"
[ "$phantom" -eq 0 ] || fail "$phantom topic events have no outbox row"

echo "== 4. consumer counted each event once"
counted=$(sql "SELECT COALESCE(SUM(n), 0) FROM outcome_counts")
[ "$counted" -eq "$committed" ] || fail "outcome counts sum to $counted, expected $committed"
echo "outcome counts sum: $counted"

echo "== 5. rewind the consumer group and restart: redelivered events must not be recounted"
dc stop consumer >/dev/null 2>&1
dc exec -T kafka /opt/kafka/bin/kafka-consumer-groups.sh --bootstrap-server localhost:9092 \
  --group "$group" --topic "$topic" --reset-offsets --to-earliest --execute >/dev/null
since=$(date -u +%Y-%m-%dT%H:%M:%SZ)
dc start consumer >/dev/null 2>&1
wait_for "redelivery of $on_topic messages" \
  '[ "$(dc logs --since "$since" consumer 2>/dev/null | grep -c "duplicate skipped")" -ge "$on_topic" ]'
recounted=$(sql "SELECT COALESCE(SUM(n), 0) FROM outcome_counts")
[ "$recounted" -eq "$committed" ] || fail "counts changed after redelivery: $recounted"
echo "after redelivering $on_topic messages, outcome counts sum: $recounted"

dc start decision >/dev/null 2>&1
echo "PASS: $n decisions committed with the publisher down and the service killed;" \
  "0 lost, 0 phantom, counted once, redelivery recounted 0"
