"""Finalization retries checkpoint, handle and sink failures without duplicate work."""

from __future__ import annotations

import pytest

from myosuite_shadow import (
    LiveShadowModel,
    ShadowModel,
    ShadowModelConfig,
    read_session,
)


class Sink:
    def __init__(self, fail_once=False):
        self.calls = 0
        self.fail_once = fail_once

    def estimate(self, frame):
        pass

    def event(self, *args):
        pass

    def close(self):
        self.calls += 1
        if self.fail_once and self.calls == 1:
            raise OSError("transient sink close failure")


def test_checkpoint_failure_keeps_shadow_close_retryable(elbow, tmp_path, monkeypatch):
    sink = Sink()
    shadow = ShadowModel(
        elbow, ShadowModelConfig(window=5), log_dir=tmp_path, sinks=(sink,)
    )
    for index in range(5):
        shadow.push(index / 20, [1.2])
    writer = shadow.log
    checkpoint, calls = writer.write_checkpoint, []

    def transient(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise OSError("transient checkpoint failure")
        checkpoint(*args, **kwargs)

    monkeypatch.setattr(writer, "write_checkpoint", transient)
    with pytest.raises(OSError, match="checkpoint"):
        shadow.close()
    assert not shadow._closed and shadow.log is writer and sink.calls == 0
    with pytest.raises(RuntimeError, match="closed"):
        shadow.push(0.3, [1.2])
    shadow.close()
    shadow.close()
    assert len(calls) == 2 and sink.calls == 1
    assert all(stream.closed for stream in writer._files.values())
    assert read_session(tmp_path).checkpoint["closed"]


def test_partially_closed_files_do_not_rewrite_the_final_checkpoint(elbow, tmp_path):
    shadow = ShadowModel(elbow, log_dir=tmp_path)
    writer = shadow.log
    name = next(iter(writer._files))
    original = writer._files[name]

    class InterruptedClose:
        def __getattr__(self, key):
            return getattr(original, key)

        def close(self):
            original.close()
            raise OSError("interrupted file close")

    writer._files[name] = InterruptedClose()
    with pytest.raises(OSError, match="file close"):
        shadow.close()
    assert original.closed
    shadow.close()
    assert all(stream.closed for stream in writer._files.values())
    assert read_session(tmp_path).checkpoint["closed"]


def test_live_close_retries_remaining_sinks_and_writes_one_summary(elbow, tmp_path):
    first, transient, last = Sink(), Sink(fail_once=True), Sink()
    shadow = ShadowModel(
        elbow,
        ShadowModelConfig(window=5),
        log_dir=tmp_path,
        sinks=(first, transient, last),
    )
    live = LiveShadowModel(shadow)
    for index in range(5):
        live.submit(index / 20, [1.2])
    live.drain(timeout=10)
    with pytest.raises(OSError, match="sink close"):
        live.close()
    assert live.health["status"] == "closing"
    assert first.calls == transient.calls == 1 and last.calls == 0
    live.close()
    live.close()
    assert [first.calls, transient.calls, last.calls] == [1, 2, 1]
    assert live.health["status"] == "closed"
    assert [event["label"] for event in read_session(tmp_path).events].count(
        "live_summary"
    ) == 1


@pytest.mark.parametrize("failure", ["before_write", "after_write", "fanout"])
def test_summary_write_and_fanout_failures_never_duplicate_csv(
    elbow, tmp_path, monkeypatch, failure
):
    class EventSink(Sink):
        def __init__(self, fail=False):
            super().__init__()
            self.events = 0
            self.fail = fail

        def event(self, *args):
            self.events += 1
            if self.fail and self.events == 1:
                raise OSError("summary fanout failed")

    first, sink = EventSink(), EventSink(failure == "fanout")
    shadow = ShadowModel(elbow, log_dir=tmp_path, sinks=(first, sink))
    writer = shadow.log
    original, attempts = writer.write_event, []

    def interrupted(*args):
        attempts.append(1)
        if failure == "before_write" and len(attempts) == 1:
            raise OSError("summary write failed")
        original(*args)
        if failure == "after_write" and len(attempts) == 1:
            raise OSError("summary write reported failure after flush")

    monkeypatch.setattr(writer, "write_event", interrupted)
    live = LiveShadowModel(shadow)
    with pytest.raises(OSError, match="summary"):
        live.close()
    live.close()
    summary = [
        event
        for event in read_session(tmp_path).events
        if event["label"] == "live_summary"
    ]
    assert len(summary) == 1 and summary[0]["data"] == {"processed": 0, "dropped": 0}
    assert sink.calls == 1
    assert first.events == 1
