#!/usr/bin/env bash
# Create the events topic, then create or update the outbox connector once
# Kafka Connect answers. Safe to re-run (--if-not-exists, idempotent PUT).
set -euo pipefail
url="${CONNECT_URL:-http://connect:8083}"
/kafka/bin/kafka-topics.sh --bootstrap-server "${BOOTSTRAP_SERVERS:-kafka:9092}" --create \
  --if-not-exists --topic fraud.decision-events --partitions 3 --replication-factor 1
for _ in $(seq 1 60); do
  curl -sf "$url/connectors" >/dev/null && break
  sleep 2
done
curl -sf -X PUT -H 'Content-Type: application/json' \
  --data @/config/decision-outbox.json "$url/connectors/decision-outbox/config"
echo
for _ in $(seq 1 30); do
  status=$(curl -sf "$url/connectors/decision-outbox/status" || true)
  if echo "$status" | grep -q '"tasks":\[{"id":0,"state":"RUNNING"'; then
    echo "connector running"; exit 0
  fi
  sleep 2
done
echo "connector not running: $status" >&2
exit 1
