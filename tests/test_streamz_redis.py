"""Offline tests for the Redis source, sink, and message adapter.

These tests never touch a live Redis server. ``redis.Redis.from_url`` is
mocked and the background workers are either replaced or driven with a
scripted client so subscribe/publish/tail behaviour can be asserted
deterministically.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import redis
from streamz import Stream

sys.path.insert(1, str(Path(__file__).parent.parent))

import safeband.streamz_tools as st
from safeband.streamz_tools import (
    RedisMessage,
    _check_redis_mode,
    _filt,
    _func,
    _stream_entry_payload,
    from_redis,
    to_redis,
)

URL = "redis://localhost:6379/0"


def _no_start(_self: object) -> None:
    """Stand-in for from_q.start keeping streamz's IOLoop out of tests."""


# --------------------------------------------------------------------------
# Message adapter and helpers
# --------------------------------------------------------------------------
class TestRedisMessageAdapter:
    """The adapter must satisfy the MQTTMessage interface ``_func`` needs."""

    def test_adapter_exposes_topic_and_payload(self) -> None:
        """A RedisMessage exposes ``.topic`` (str) and ``.payload`` (bytes)."""
        msg = RedisMessage(topic="foo", payload=b"1.")

        assert msg.topic == "foo"
        assert msg.payload == b"1."

    def test_adapter_feeds_func_and_filt_accumulation(self) -> None:
        """Adapter messages accumulate by topic exactly like MQTTMessage."""
        topics = ["foo", "bar"]
        state: dict = {}

        state = _func(state, RedisMessage(topic="foo", payload=b"1."), topics)
        assert state == {"foo": b"1."}
        assert _filt(state, topics) is False

        state = _func(state, RedisMessage(topic="bar", payload=b"2."), topics)
        assert state == {"foo": b"1.", "bar": b"2."}
        assert _filt(state, topics) is True


class TestRedisHelpers:
    """Mode validation and stream-entry payload extraction."""

    @pytest.mark.parametrize("mode", ["pubsub", "stream"])
    def test_valid_modes_pass_through(self, mode: str) -> None:
        """Both supported modes are returned unchanged."""
        assert _check_redis_mode(mode) == mode

    def test_invalid_mode_raises(self) -> None:
        """Anything else is rejected with a ValueError naming the value."""
        with pytest.raises(ValueError, match="got 'queue'"):
            _check_redis_mode("queue")

    def test_entry_payload_prefers_data_field(self) -> None:
        """The ``data`` field written by to_redis wins over other fields."""
        assert _stream_entry_payload({b"x": b"0", b"data": b"1.5"}) == b"1.5"
        assert _stream_entry_payload({"data": b"2.5"}) == b"2.5"

    def test_entry_payload_falls_back_to_first_field(self) -> None:
        """Foreign entries without ``data`` yield their first field value."""
        assert _stream_entry_payload({b"value": b"3", b"unit": b"C"}) == b"3"
        assert _stream_entry_payload({}) == b""


# --------------------------------------------------------------------------
# Source
# --------------------------------------------------------------------------
class TestFromRedisSource:
    """The source registers on Stream and stays lazy until started."""

    def test_from_redis_registered_and_lazy(self) -> None:
        """Construction parses url/channels/mode without connecting."""
        source = Stream.from_redis(url=URL, topic=["x", "y"], mode="stream")

        assert isinstance(source, from_redis)
        assert source.url == URL
        assert source.channels == ["x", "y"]
        assert source.mode == "stream"
        # Lazy: no client, subscription or thread before start().
        assert source._client is None
        assert source._pubsub is None
        assert source._worker is None
        assert source._thread is None
        source.stop()

    def test_single_topic_is_wrapped_in_a_list(self) -> None:
        """A plain string topic becomes a one-element channel list."""
        source = Stream.from_redis(url=URL, topic="x")

        assert source.channels == ["x"]
        source.stop()

    def test_invalid_mode_rejected_at_construction(self) -> None:
        """An unsupported mode fails fast instead of at start()."""
        with pytest.raises(ValueError, match="mode must be"):
            Stream.from_redis(url=URL, topic="x", mode="queue")

    def test_on_message_enqueues_adapter_with_decoded_channel(self) -> None:
        """A Pub/Sub message dict is queued as a RedisMessage adapter."""
        source = Stream.from_redis(url=URL, topic="x")

        source._on_message(
            {"type": "message", "channel": b"x", "data": b"42."}
        )

        queued = source.q.get_nowait()
        assert isinstance(queued, RedisMessage)
        assert queued.topic == "x"
        assert queued.payload == b"42."
        source.stop()

    def test_on_entry_enqueues_stream_entry(self) -> None:
        """A stream entry's ``data`` field is queued under the stream key."""
        source = Stream.from_redis(url=URL, topic="s", mode="stream")

        source._on_entry(b"s", {b"data": b"7."})

        queued = source.q.get_nowait()
        assert queued == RedisMessage(topic="s", payload=b"7.")
        source.stop()

    def test_start_pubsub_subscribes_and_runs_worker(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Pub/Sub start subscribes every channel and starts the worker."""
        client = MagicMock()
        pubsub = client.pubsub.return_value
        worker = pubsub.run_in_thread.return_value
        from_url = MagicMock(return_value=client)
        monkeypatch.setattr(st.redis.Redis, "from_url", from_url)
        # Keep streamz's own IOLoop out of the test.
        monkeypatch.setattr(st.from_q, "start", _no_start)
        source = Stream.from_redis(url=URL, topic=["a", "b"])

        source.start()

        from_url.assert_called_once_with(URL)
        client.pubsub.assert_called_once_with(ignore_subscribe_messages=True)
        pubsub.subscribe.assert_called_once_with(
            a=source._on_message,
            b=source._on_message,
        )
        pubsub.run_in_thread.assert_called_once()
        assert source._worker is worker

        source.stop()

        worker.stop.assert_called_once()
        pubsub.close.assert_called_once()
        client.close.assert_called_once()
        assert source._client is None
        assert source._worker is None

    def test_start_stream_spawns_reader_thread(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Stream mode starts a background XREAD thread and joins on stop."""
        client = MagicMock()
        monkeypatch.setattr(
            st.redis.Redis, "from_url", MagicMock(return_value=client)
        )
        monkeypatch.setattr(st.from_q, "start", _no_start)
        source = Stream.from_redis(url=URL, topic="s", mode="stream")
        monkeypatch.setattr(source, "_run_stream_loop", lambda: None)

        source.start()
        assert source._thread is not None
        client.pubsub.assert_not_called()

        source.stop()
        assert source._thread is None
        client.close.assert_called_once()

    def test_stream_loop_tails_and_advances_last_id(self) -> None:
        """The reader queues entries and resumes from the last seen id."""
        source = Stream.from_redis(url=URL, topic="s", mode="stream")
        client = MagicMock()
        source._client = client

        def _xread(last_ids: dict, block: int) -> list:  # noqa: ARG001
            if last_ids["s"] == "$":
                return [(b"s", [(b"1-0", {b"data": b"1.5"})])]
            source._stop_event.set()
            return []

        client.xread.side_effect = _xread

        source._run_stream_loop()

        assert source.q.get_nowait() == RedisMessage(topic="s", payload=b"1.5")
        assert client.xread.call_args_list[-1].args[0] == {"s": b"1-0"}
        source.stop()

    def test_stream_loop_survives_redis_errors(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A failing XREAD is logged and retried, not fatal to the loop."""
        source = Stream.from_redis(url=URL, topic="s", mode="stream")
        client = MagicMock()
        source._client = client
        # Skip the retry back-off so the test stays fast.
        monkeypatch.setattr(source._stop_event, "wait", lambda _t: None)

        def _xread(last_ids: dict, block: int) -> list:  # noqa: ARG001
            if client.xread.call_count == 1:
                msg = "down"
                raise redis.ConnectionError(msg)
            source._stop_event.set()
            return []

        client.xread.side_effect = _xread

        source._run_stream_loop()

        assert "XREAD failed" in caplog.text
        assert client.xread.call_count == 2
        source.stop()


# --------------------------------------------------------------------------
# Sink
# --------------------------------------------------------------------------
@pytest.fixture
def redis_sink() -> to_redis:
    """A Pub/Sub sink with a mocked client so nothing connects."""
    sink = to_redis(Stream(), url=URL, topic="t")
    sink.client = MagicMock()
    return sink


@pytest.fixture
def stream_sink() -> to_redis:
    """A stream-mode sink with a mocked client."""
    sink = to_redis(Stream(), url=URL, topic="t", mode="stream", maxlen=100)
    sink.client = MagicMock()
    return sink


class TestToRedisLazyConnect:
    """The client connects lazily on the first published message."""

    def test_update_connects_once(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The first update connects; later ones reuse the client."""
        client = MagicMock()
        from_url = MagicMock(return_value=client)
        monkeypatch.setattr(st.redis.Redis, "from_url", from_url)
        sink = to_redis(Stream(), url=URL, topic="t")

        sink.update(b"first")
        sink.update(b"second")

        from_url.assert_called_once_with(URL)
        assert client.publish.call_count == 2

    def test_invalid_mode_rejected_at_construction(self) -> None:
        """An unsupported mode fails fast."""
        with pytest.raises(ValueError, match="mode must be"):
            to_redis(Stream(), url=URL, topic="t", mode="queue")


class TestToRedisFanOut:
    """Key fan-out mirrors the MQTT/NATS sinks for float and dict limits."""

    def test_bytes_pass_through_to_base_channel(
        self,
        redis_sink: to_redis,
    ) -> None:
        """Raw bytes publish unchanged to the configured channel."""
        redis_sink.update(b"hello")

        redis_sink.client.publish.assert_called_once_with("t", b"hello")

    def test_float_limits_fan_out_to_three_channels(
        self,
        redis_sink: to_redis,
    ) -> None:
        """A flat dict fans out to anomaly/_DOL_high/_DOL_low channels."""
        redis_sink.update(
            {"anomaly": 1, "level_high": 0.5, "level_low": -0.5},
        )

        calls = redis_sink.client.publish.call_args_list
        assert [c.args[0] for c in calls] == [
            "tanomaly",
            "t_DOL_high",
            "t_DOL_low",
        ]
        # Non-bytes payloads are encoded as their string form.
        assert [c.args[1] for c in calls] == [b"1", b"0.5", b"-0.5"]

    def test_dict_limits_fan_out_per_signal(
        self,
        redis_sink: to_redis,
    ) -> None:
        """Nested limits fan out per signal incl. root_cause flags."""
        redis_sink.update(
            {
                "anomaly": 1,
                "level_high": {"a": 0.5, "b": 0.6},
                "level_low": {"a": -0.5, "b": -0.4},
                "root_cause": "b",
            },
        )

        calls = redis_sink.client.publish.call_args_list
        assert [c.args[0] for c in calls] == [
            "tanomaly",
            "a_DOL_high",
            "a_DOL_low",
            "a_root_cause",
            "b_DOL_high",
            "b_DOL_low",
            "b_root_cause",
        ]
        payloads = {c.args[0]: c.args[1] for c in calls}
        assert payloads["a_root_cause"] == b"0"
        assert payloads["b_root_cause"] == b"1"

    def test_stream_mode_appends_with_xadd(
        self,
        stream_sink: to_redis,
    ) -> None:
        """Stream mode XADDs the body under ``data`` with trimming."""
        stream_sink.update(
            {"anomaly": 0, "level_high": 1.0, "level_low": 0.0},
        )

        stream_sink.client.publish.assert_not_called()
        stream_sink.client.xadd.assert_any_call(
            "tanomaly",
            {b"data": b"0"},
            maxlen=100,
            approximate=True,
        )
        assert stream_sink.client.xadd.call_count == 3

    def test_redis_error_is_logged_and_dropped(
        self,
        redis_sink: to_redis,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A failing publish logs a warning instead of raising."""
        redis_sink.client.publish.side_effect = redis.ConnectionError("down")

        redis_sink.update(b"x")

        assert "message dropped" in caplog.text


class TestToRedisDestroy:
    """destroy() closes the client."""

    def test_destroy_closes_client(self, redis_sink: to_redis) -> None:
        """destroy() closes the connection and forgets the client."""
        client = redis_sink.client

        redis_sink.destroy()

        client.close.assert_called_once()
        assert redis_sink.client is None
