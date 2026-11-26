"""Automatic clock refresh stays on the study axis and fails visibly."""

import dataclasses
import json
import socket
import threading
import time

import pytest

from myosuite_shadow.tracking import (
    ClockMapping,
    QuestClockSynchronizer,
    synchronize_quest_clock,
)


def initial() -> ClockMapping:
    return ClockMapping(
        0,
        0,
        "android_monotonic",
        unix_anchor=1000,
        uncertainty_s=0.01,
        valid_until_source_s=100,
    )


def test_background_refresh_preserves_clock_identity_axis_and_closes():
    entered = threading.Event()
    updated = dataclasses.replace(
        initial(), source_anchor_s=1, session_anchor_s=1, unix_anchor=1001
    )

    def sync():
        entered.set()
        return updated

    client = QuestClockSynchronizer(
        "unused", initial=initial(), refresh_s=50, sync_function=sync
    )
    try:
        before = client.get_mapping()
        client.refresh()
        assert entered.wait(2)
        deadline = time.perf_counter() + 2
        while (
            client.health["last_refresh_unix"] is None
            and time.perf_counter() < deadline
        ):
            threading.Event().wait(0.001)
        assert client.get_mapping() is updated
        assert before.map(5) == updated.map(5)
    finally:
        client.close()
    assert not client.health["alive"]
    with pytest.raises(RuntimeError, match="closed"):
        client.get_mapping()


@pytest.mark.parametrize(
    "changed",
    [
        {"clock_id": "new_epoch"},
        {"session_anchor_s": 500},
        {"unix_anchor": 5000},
    ],
)
def test_refresh_identity_or_axis_errors_are_visible_and_old_clock_cannot_leak(changed):
    client = QuestClockSynchronizer(
        "unused",
        initial=initial(),
        refresh_s=50,
        sync_function=lambda: dataclasses.replace(initial(), **changed),
    )
    try:
        client.refresh()
        client._thread.join(2)
        with pytest.raises(RuntimeError, match="refresh failed"):
            client.get_mapping()
        assert client.health["error"] is not None
    finally:
        client.close()


def test_shutdown_never_claims_a_blocked_clock_worker_has_stopped():
    entered, release = threading.Event(), threading.Event()

    def blocked():
        entered.set()
        release.wait(5)
        return initial()

    client = QuestClockSynchronizer(
        "unused", initial=initial(), refresh_s=50, sync_function=blocked
    )
    try:
        client.refresh()
        assert entered.wait(2)
        with pytest.raises(TimeoutError, match="did not stop"):
            client.close(timeout=0.01)
        assert client.health["alive"]
    finally:
        release.set()
        client.close()


def test_clock_service_protocol_uses_real_udp_and_nonce_bound_replies():
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind(("127.0.0.1", 0))
    port = server.getsockname()[1]
    server.settimeout(3)
    errors = []

    def serve():
        try:
            for _ in range(6):
                data, peer = server.recvfrom(1024)
                tag, nonce = data.decode("ascii").split()
                assert tag == "shadow-clock-v1"
                server.sendto(
                    json.dumps(
                        {
                            "nonce": int(nonce),
                            "clock_id": "android_monotonic",
                            "source_time_s": time.perf_counter() + 500,
                        }
                    ).encode(),
                    peer,
                )
        except Exception as error:
            errors.append(error)
        finally:
            server.close()

    worker = threading.Thread(target=serve)
    worker.start()
    mapping = synchronize_quest_clock(
        "127.0.0.1",
        port=port,
        count=6,
        interval_s=0.22,
        validity_s=10,
        max_uncertainty_s=0.1,
    )
    worker.join(2)
    assert not errors
    source_now = time.perf_counter() + 500
    mapped, unix = mapping.map(source_now)
    assert abs(mapped - time.perf_counter()) <= mapping.uncertainty_s
    assert unix is not None


def test_clock_sync_tolerates_dropped_and_delayed_pings_and_rejects_spoofed_nonces():
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind(("127.0.0.1", 0))
    port = server.getsockname()[1]
    server.settimeout(3)

    def serve_flaky():
        delayed = None
        try:
            for idx in range(5):
                data, peer = server.recvfrom(1024)
                _, nonce_text = data.decode("ascii").split()
                nonce = int(nonce_text)
                if idx == 0:
                    continue  # drop first UDP ping
                if idx == 1:
                    delayed = (nonce, peer, time.perf_counter() + 500)
                    continue  # delay reply until next ping arrives
                if delayed is not None:
                    d_nonce, d_peer, d_t = delayed
                    server.sendto(
                        json.dumps(
                            {
                                "nonce": d_nonce,
                                "clock_id": "android_monotonic",
                                "source_time_s": d_t,
                            }
                        ).encode(),
                        d_peer,
                    )
                    delayed = None
                server.sendto(
                    json.dumps(
                        {
                            "nonce": nonce,
                            "clock_id": "android_monotonic",
                            "source_time_s": time.perf_counter() + 500,
                        }
                    ).encode(),
                    peer,
                )
        finally:
            server.close()

    worker = threading.Thread(target=serve_flaky)
    worker.start()
    mapping = synchronize_quest_clock(
        "127.0.0.1",
        port=port,
        count=5,
        interval_s=0.35,
        timeout_s=0.04,
        validity_s=10,
        max_rtt_s=0.5,
        max_uncertainty_s=1.0,
    )
    worker.join(3)
    assert mapping.clock_id == "android_monotonic"
    with pytest.raises(ValueError, match="finite positive"):
        synchronize_quest_clock("127.0.0.1", port=port, interval_s=float("nan"))
