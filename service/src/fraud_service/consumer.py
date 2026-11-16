"""Kafka consumer: project ``decision.made`` events into per-model outcome counts.

Delivery is at least once (Debezium and Kafka may redeliver; the consumer may
crash after processing but before committing its offset). Each event's
``event_id`` and its count update are written in one database transaction,
and the Kafka offset is committed only afterwards, so a redelivered event is
recognised and skipped: counts change exactly once per event (ADR-006).

Run: ``python -m fraud_service.consumer``. Settings from environment
variables: ``DATABASE_URL`` and ``KAFKA_BOOTSTRAP_SERVERS`` (required),
``EVENTS_TOPIC`` (default ``fraud.decision-events``), ``CONSUMER_GROUP``
(default ``decision-outcome-counts``).
"""

from __future__ import annotations

import json
import logging
import os
import signal
from dataclasses import dataclass

from confluent_kafka import Consumer, KafkaError, KafkaException, Message
from psycopg import Connection

from fraud_service.store import EVENT_SCHEMA_VERSION, EVENT_TYPE, create_schema

log = logging.getLogger("fraud_service.consumer")

DEFAULT_TOPIC = "fraud.decision-events"
DEFAULT_GROUP = "decision-outcome-counts"

SCHEMA = """
CREATE TABLE IF NOT EXISTS consumed_events (
    event_id    UUID PRIMARY KEY,
    decision_id UUID NOT NULL,
    topic       TEXT NOT NULL,
    partition   INTEGER NOT NULL,
    "offset"    BIGINT NOT NULL,
    consumed_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS outcome_counts (
    model_version TEXT NOT NULL,
    outcome       TEXT NOT NULL,
    n             BIGINT NOT NULL,
    PRIMARY KEY (model_version, outcome)
);
"""


class InvalidEvent(ValueError):
    """The message is not a ``decision.made`` v1 event."""


@dataclass(frozen=True)
class Record:
    """The parts of a Kafka message the projection needs."""

    topic: str
    partition: int
    offset: int
    value: bytes | None


def parse(value: bytes | None) -> dict:
    """Decode one event payload.

    Raises:
        InvalidEvent: If the payload is not JSON or not a supported event.
    """
    if value is None:
        raise InvalidEvent("empty message value (tombstone)")
    try:
        event = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidEvent(f"not JSON: {exc}") from exc
    if not isinstance(event, dict):
        raise InvalidEvent("payload is not an object")
    if event.get("event_type") != EVENT_TYPE or event.get("schema_version") != EVENT_SCHEMA_VERSION:
        raise InvalidEvent(
            f"unsupported event {event.get('event_type')!r} v{event.get('schema_version')!r}"
        )
    missing = {"event_id", "decision_id", "model_version", "outcome"} - event.keys()
    if missing:
        raise InvalidEvent(f"missing fields: {sorted(missing)}")
    return event


def apply(conn: Connection, record: Record) -> bool:
    """Apply one event in one transaction; ``False`` if it was already applied.

    Raises:
        InvalidEvent: If the payload cannot be processed (nothing is written).
    """
    event = parse(record.value)
    with conn.transaction():
        first = conn.execute(
            'INSERT INTO consumed_events (event_id, decision_id, topic, partition, "offset") '
            "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (event_id) DO NOTHING RETURNING event_id",
            (
                event["event_id"],
                event["decision_id"],
                record.topic,
                record.partition,
                record.offset,
            ),
        ).fetchone()
        if first is None:
            return False
        conn.execute(
            "INSERT INTO outcome_counts (model_version, outcome, n) VALUES (%s, %s, 1) "
            "ON CONFLICT (model_version, outcome) DO UPDATE SET n = outcome_counts.n + 1",
            (event["model_version"], event["outcome"]),
        )
    return True


def _record(message: Message) -> Record:
    return Record(message.topic(), message.partition(), message.offset(), message.value())


def run(database_url: str, bootstrap: str, topic: str, group: str) -> None:
    """Consume until SIGTERM or SIGINT; offsets are committed after each database commit.

    An invalid event stops the consumer (it is not skipped silently): fix the
    producer or the contract, then restart.
    """
    stopping = False

    def stop(*_: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    conn = Connection.connect(database_url, autocommit=True)
    create_schema(conn, SCHEMA)
    consumer = Consumer(
        {
            "bootstrap.servers": bootstrap,
            "group.id": group,
            "enable.auto.commit": False,
            "auto.offset.reset": "earliest",
            "topic.metadata.refresh.interval.ms": 5000,  # notice a topic created later
        }
    )
    consumer.subscribe([topic])
    log.info("consuming %s as group %s", topic, group)
    try:
        while not stopping:
            message = consumer.poll(1.0)
            if message is None:
                continue
            if message.error():
                if message.error().code() == KafkaError.UNKNOWN_TOPIC_OR_PART:
                    log.info("topic %s does not exist yet; waiting", topic)
                    continue
                raise KafkaException(message.error())
            record = _record(message)
            applied = apply(conn, record)
            consumer.commit(message=message, asynchronous=False)
            log.info(
                "%s %s@%d:%d",
                "applied" if applied else "duplicate skipped",
                record.topic,
                record.partition,
                record.offset,
            )
    finally:
        consumer.close()
        conn.close()


def main() -> None:
    """Entry point for ``python -m fraud_service.consumer``."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    run(
        os.environ["DATABASE_URL"],
        os.environ["KAFKA_BOOTSTRAP_SERVERS"],
        os.environ.get("EVENTS_TOPIC", DEFAULT_TOPIC),
        os.environ.get("CONSUMER_GROUP", DEFAULT_GROUP),
    )


if __name__ == "__main__":
    main()
