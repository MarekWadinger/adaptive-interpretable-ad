"""MQTT, Redis and file-based consumer for anomaly detection results."""

import datetime as dt
import json
import logging
from argparse import Namespace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import paho.mqtt.client as mqtt
import redis
from human_security import HumanRSA
from paho.mqtt.properties import Properties
from paho.mqtt.reasoncodes import ReasonCode

from safeband.encryption import (
    init_rsa_security,
    resolve_key_path,
    verify_and_decrypt_data,
)
from safeband.parse import get_params
from safeband.streamz_tools import (
    _as_bytes,
    _as_str,
    _require_topics,
    _stream_entry_payload,
    _xread_batches,
)
from safeband.typing_extras import FileClient, MQTTClient, RedisClient

if TYPE_CHECKING:
    from redis.typing import KeyT, StreamIdT

logger = logging.getLogger(__name__)


def _log_received(
    topic: str,
    payload: bytes | bytearray,
    receiver: HumanRSA | None,
    t: dt.datetime,
) -> None:
    """Decrypt (when a key is given) and log one received message.

    Shared by every transport's receive path so the decrypt/log policy
    cannot drift between MQTT and Redis.

    Args:
        topic: The topic/channel/stream the message arrived on.
        payload: The raw message body.
        receiver: RSA key used to verify and decrypt ``payload``; ``None``
            treats the payload as plaintext.
        t: The receive time used for logging.

    """
    if receiver is not None:
        decoded = verify_and_decrypt_data(
            json.loads(payload.decode()),
            receiver,
        )
        item = json.dumps(decoded)
        field_count = len(decoded)
    else:
        item = payload.decode()
        field_count = None
    # Log only metadata at INFO; the full decrypted payload may carry
    # sensitive values, so it is emitted at DEBUG instead.
    logger.info(
        "Received message at %s on %s (%s fields)",
        t,
        topic,
        field_count if field_count is not None else "n/a",
    )
    logger.debug("Message payload at %s: %s", t, item)


# MQTT callback functions (paho callback API v2)
def on_connect(
    self: mqtt.Client,
    userdata: Namespace,
    _flags: mqtt.ConnectFlags,
    reason_code: ReasonCode,
    _properties: Properties | None,
) -> None:
    """Subscribe to configured topics after a successful broker connection.

    Args:
        self: MQTT client instance invoking the callback.
        userdata: User-specific data passed to the callback; ``topic``
            holds the list of topics to subscribe to.
        _flags: Connect flags from the broker (unused).
        reason_code: The connection result.
        _properties: MQTT v5 properties (unused).

    """
    logger.info("Connected with result code %s", reason_code)
    self.subscribe([(topic, 0) for topic in userdata.topic])


def on_message(
    _self: mqtt.Client,
    userdata: Namespace | None,
    msg: mqtt.MQTTMessage,
) -> None:
    """Decrypt and log an incoming MQTT message.

    Args:
        _self: MQTT client instance (unused).
        userdata: User-specific data passed to the callback; an optional
            ``receiver`` attribute holds the decryption key.
        msg: The message received from the broker.

    """
    receiver = getattr(userdata, "receiver", None)
    t = dt.datetime.fromtimestamp(msg.timestamp, tz=dt.UTC).replace(
        microsecond=0,
    )
    _log_received(msg.topic, msg.payload, receiver, t)


def query_file(config: FileClient, **kwargs: HumanRSA | None) -> None:
    """Read a JSON output file and log the entry closest to now.

    Args:
        config: File client configuration with an ``output`` key pointing
            to the JSON file to read.
        **kwargs: Optional keyword arguments. Pass ``receiver`` (RSA key)
            to decrypt entries before processing; with no key (or
            ``receiver=None``) entries are treated as plaintext.

    """
    receiver = kwargs.get("receiver")
    # Load the JSON file as a list of dictionaries
    with Path(config.output).open(encoding="utf-8") as f:
        data: list[dict[str, Any]] = [json.loads(line) for line in f]

    # Convert the time strings to datetime objects
    for i, item in enumerate(data):
        # Encrypted entries are detected by their signature field rather
        # than by guessing from the ciphertext's character set.
        if receiver is not None and "signature" in item:
            data[i] = cast(
                "dict[str, Any]",
                verify_and_decrypt_data(item, receiver),
            )
        data[i]["time"] = dt.datetime.strptime(
            str(data[i]["time"]),
            "%Y-%m-%d %H:%M:%S",
        ).replace(tzinfo=dt.UTC)

    # Sort the data by time in descending order
    data.sort(key=lambda x: x["time"], reverse=True)

    # Find the closest past item
    closest_item = None
    for item in data:
        if item["time"] <= dt.datetime.now(dt.UTC).replace(microsecond=0):
            closest_item = item
            break

    # Log only metadata at INFO; the full (possibly decrypted) entry may
    # carry sensitive values, so it is emitted at DEBUG instead.
    if closest_item is not None:
        logger.info(
            "Closest entry at %s (%s fields)",
            closest_item["time"],
            len(closest_item),
        )
    else:
        logger.info("No entry at or before now.")
    logger.debug("Closest entry payload: %s", closest_item)


def query_mqtt(
    config: MQTTClient,
    topics: list[str],
    receiver: HumanRSA | None = None,
) -> mqtt.Client:
    """Create an MQTT client instance and connect to the configured broker.

    Args:
        config: MQTT client configuration with ``host`` and ``port`` keys.
        topics: Topics to subscribe to once connected.
        receiver: RSA key used to decrypt received messages, or ``None``
            for plaintext.

    Returns:
        mqtt.Client: Connected MQTT client instance.

    """
    # The callbacks read the topic list and key back from userdata.
    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        userdata=Namespace(topic=topics, receiver=receiver),
    )

    # Assign callback functions
    client.on_connect = on_connect
    client.on_message = on_message

    # Connect to the MQTT broker
    client.connect(config.host, config.port, 60)
    return client


def _now() -> dt.datetime:
    """Return the current UTC time truncated to whole seconds."""
    return dt.datetime.now(dt.UTC).replace(microsecond=0)


def query_redis(
    config: RedisClient,
    topics: list[str],
    receiver: HumanRSA | None = None,
    *,
    client: redis.Redis | None = None,
    max_messages: int | None = None,
    block_ms: int = 1000,
) -> None:
    """Subscribe to Redis channels or tail Redis Streams and log results.

    Mirrors :func:`query_mqtt` for the ``[redis]`` transport. With
    ``mode="pubsub"`` the configured topics are Pub/Sub channels; with
    ``mode="stream"`` they are Redis Streams tailed from the current tip
    with blocking ``XREAD`` calls. Runs until interrupted unless
    ``max_messages`` bounds it.

    Args:
        config: Redis client configuration with ``url`` and ``mode``.
        topics: Channels or stream keys to consume.
        receiver: RSA key used to decrypt received messages, or ``None``
            for plaintext.
        client: Pre-built client (used by tests); ``None`` connects to
            ``config.url``.
        max_messages: Stop after this many messages; ``None`` runs
            forever.
        block_ms: ``XREAD`` block timeout in stream mode.

    Raises:
        ValueError: If ``topics`` is empty.

    """
    _require_topics(topics)
    own_client = client is None
    if client is None:
        client = redis.Redis.from_url(config.url)
    seen = 0

    def _done() -> bool:
        return max_messages is not None and seen >= max_messages

    try:
        if config.mode == "pubsub":
            pubsub = client.pubsub(ignore_subscribe_messages=True)
            try:
                pubsub.subscribe(*topics)
                logger.info("Subscribed to channels %s", topics)
                for message in pubsub.listen():
                    if message.get("type") != "message":
                        continue
                    _log_received(
                        _as_str(message["channel"]),
                        _as_bytes(message["data"]),
                        receiver,
                        _now(),
                    )
                    seen += 1
                    if _done():
                        break
            finally:
                pubsub.close()
        else:
            last_ids: dict[KeyT, StreamIdT] = dict.fromkeys(topics, "$")
            logger.info("Tailing streams %s", topics)
            while not _done():
                for stream, entries in _xread_batches(
                    client,
                    last_ids,
                    block_ms,
                ):
                    name = _as_str(stream)
                    for entry_id, fields in entries:
                        last_ids[name] = entry_id
                        _log_received(
                            name,
                            _stream_entry_payload(fields),
                            receiver,
                            _now(),
                        )
                        seen += 1
                        if _done():
                            break
                    if _done():
                        break
    finally:
        if own_client:
            client.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = get_params()

    receiver: HumanRSA | None = None
    if config.setup.key_path:
        safe_key_path = resolve_key_path(config.setup.key_path)
        _, receiver = init_rsa_security(str(safe_key_path))

    client = config.client
    if isinstance(client, FileClient):
        query_file(client, receiver=receiver)
    elif isinstance(client, MQTTClient):
        mqtt_client = query_mqtt(client, config.io.in_topics, receiver)
        mqtt_client.loop_forever()
    elif isinstance(client, RedisClient):
        query_redis(client, config.io.in_topics, receiver)
