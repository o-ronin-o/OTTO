"""
Unit tests for FrameScheduler.

Run with:  pytest tests/test_frame_scheduler.py -v
"""

from pathlib import Path

import pytest

from src.config_loader import ConfigLoader
from src.frame_scheduler import FrameScheduler, ScheduledFrame


CONFIG_DIR = Path(__file__).parent.parent / "config"


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def loader() -> ConfigLoader:
    return ConfigLoader(CONFIG_DIR)


@pytest.fixture
def sedan_messages(loader) -> dict:
    """Messages for the sedan class."""
    return loader.get_messages_for_vehicle("sedan_a")


@pytest.fixture
def scheduler(sedan_messages) -> FrameScheduler:
    """Scheduler for a 60-second scenario at 100 Hz."""
    return FrameScheduler(
        messages=sedan_messages,
        base_rate_hz=100,
        duration_s=60.0,
    )


@pytest.fixture
def dummy_payloads(sedan_messages) -> dict:
    """Fake payloads: each CAN ID gets n_steps payloads, all zeros."""
    return {
        msg["id"]: [bytes(8) for _ in range(6000)]
        for msg in sedan_messages.values()
    }


# ----------------------------------------------------------------------
# Initialization
# ----------------------------------------------------------------------

def test_scheduler_initializes(scheduler):
    assert scheduler.base_rate_hz == 100.0
    assert scheduler.duration_s == 60.0
    assert abs(scheduler.base_step_ms - 10.0) < 1e-9
    assert scheduler.n_steps == 6000


def test_invalid_cycle_alignment_raises():
    """cycle_ms not a multiple of base step should fail."""
    bad_messages = {
        "weird": {
            "id": 0x123, "cycle_ms": 15, "dlc": 8, "signals": []
        },
    }
    # 15 ms is not a multiple of 10 ms (base step for 100 Hz)
    with pytest.raises(ValueError, match="not an integer multiple"):
        FrameScheduler(messages=bad_messages, base_rate_hz=100, duration_s=10.0)


# ----------------------------------------------------------------------
# Expected frame counts
# ----------------------------------------------------------------------

def test_expected_frame_count_per_message(scheduler):
    counts = scheduler.expected_frame_count_per_message()
    # 60 s = 60000 ms
    # engine: cycle 10 ms → 6000 frames
    # brake: cycle 20 ms → 3000 frames
    # dynamics: cycle 50 ms → 1200 frames
    assert counts[0x100] == 6000
    assert counts[0x1A0] == 3000
    assert counts[0x120] == 1200


def test_expected_total_frame_count(scheduler):
    total = scheduler.expected_frame_count()
    assert total == 6000 + 3000 + 1200  # 10200


# ----------------------------------------------------------------------
# Scheduling
# ----------------------------------------------------------------------

def test_schedule_returns_expected_count(scheduler, dummy_payloads):
    frames = scheduler.schedule(dummy_payloads)
    assert len(frames) == scheduler.expected_frame_count()


def test_schedule_is_chronologically_sorted(scheduler, dummy_payloads):
    frames = scheduler.schedule(dummy_payloads)
    timestamps = [f.timestamp_ms for f in frames]
    assert timestamps == sorted(timestamps)


def test_engine_frames_at_correct_intervals(scheduler, dummy_payloads):
    frames = scheduler.schedule(dummy_payloads)
    engine_ts = [f.timestamp_ms for f in frames if f.can_id == 0x100]
    # First 5 engine timestamps should be 0, 10, 20, 30, 40
    assert engine_ts[:5] == [0, 10, 20, 30, 40]
    # Last engine timestamp: 60000 - 10 = 59990
    assert engine_ts[-1] == 59990


def test_brake_frames_at_correct_intervals(scheduler, dummy_payloads):
    frames = scheduler.schedule(dummy_payloads)
    brake_ts = [f.timestamp_ms for f in frames if f.can_id == 0x1A0]
    assert brake_ts[:5] == [0, 20, 40, 60, 80]
    assert brake_ts[-1] == 59980


def test_dynamics_frames_at_correct_intervals(scheduler, dummy_payloads):
    frames = scheduler.schedule(dummy_payloads)
    dyn_ts = [f.timestamp_ms for f in frames if f.can_id == 0x120]
    assert dyn_ts[:5] == [0, 50, 100, 150, 200]
    assert dyn_ts[-1] == 59950


def test_timestamp_zero_contains_all_three_messages(scheduler, dummy_payloads):
    frames = scheduler.schedule(dummy_payloads)
    at_zero = [f.can_id for f in frames if f.timestamp_ms == 0]
    assert set(at_zero) == {0x100, 0x1A0, 0x120}


def test_timestamp_10_contains_only_engine(scheduler, dummy_payloads):
    frames = scheduler.schedule(dummy_payloads)
    at_10 = [f.can_id for f in frames if f.timestamp_ms == 10]
    assert at_10 == [0x100]


def test_timestamp_20_contains_engine_and_brake(scheduler, dummy_payloads):
    frames = scheduler.schedule(dummy_payloads)
    at_20 = sorted(f.can_id for f in frames if f.timestamp_ms == 20)
    assert at_20 == [0x100, 0x1A0]


def test_timestamp_100_contains_all_three(scheduler, dummy_payloads):
    frames = scheduler.schedule(dummy_payloads)
    at_100 = sorted(f.can_id for f in frames if f.timestamp_ms == 100)
    assert at_100 == [0x100, 0x120, 0x1A0]

def test_timestamp_50_contains_engine_and_dynamics_only(scheduler, dummy_payloads):
    """At 50 ms, only engine (10 ms cycle) and dynamics (50 ms cycle) emit.
    Brake has a 20 ms cycle and does not align at 50."""
    frames = scheduler.schedule(dummy_payloads)
    at_50 = sorted(f.can_id for f in frames if f.timestamp_ms == 50)
    assert at_50 == [0x100, 0x120]


# ----------------------------------------------------------------------
# Tie-breaking
# ----------------------------------------------------------------------

def test_ties_broken_by_can_id(scheduler, dummy_payloads):
    """At the same millisecond, frames should appear in ascending CAN ID order."""
    frames = scheduler.schedule(dummy_payloads)
    at_zero = [f.can_id for f in frames if f.timestamp_ms == 0]
    # 0x100 < 0x120 < 0x1A0
    assert at_zero == sorted(at_zero)


# ----------------------------------------------------------------------
# Payload preservation
# ----------------------------------------------------------------------

def test_payloads_are_preserved(sedan_messages, dummy_payloads):
    """The scheduler must not modify payloads; it only interleaves them."""
    # Custom payloads with distinct values per step
    custom_payloads = {}
    for msg in sedan_messages.values():
        custom_payloads[msg["id"]] = [
            bytes([i % 256, 0, 0, 0, 0, 0, 0, 0]) for i in range(6000)
        ]

    scheduler = FrameScheduler(sedan_messages, base_rate_hz=100, duration_s=60.0)
    frames = scheduler.schedule(custom_payloads)

    # First engine frame at t=0 should have payload [0, 0, ...]
    first_engine = next(f for f in frames if f.can_id == 0x100 and f.timestamp_ms == 0)
    assert first_engine.payload == bytes(8)

    # Engine frame at t=10 should have payload [1, 0, ...]
    second_engine = next(f for f in frames if f.can_id == 0x100 and f.timestamp_ms == 10)
    assert second_engine.payload == bytes([1, 0, 0, 0, 0, 0, 0, 0])

    # Engine frame at t=100 should have payload [10, 0, ...]
    engine_at_100 = next(f for f in frames if f.can_id == 0x100 and f.timestamp_ms == 100)
    assert engine_at_100.payload == bytes([10, 0, 0, 0, 0, 0, 0, 0])


# ----------------------------------------------------------------------
# Error handling
# ----------------------------------------------------------------------

def test_missing_can_id_raises(scheduler):
    """If frames_by_id is missing an expected CAN ID, raise."""
    incomplete = {0x100: [bytes(8)] * 6000}  # missing 0x120 and 0x1A0
    with pytest.raises(KeyError, match="0x"):
        scheduler.schedule(incomplete)


# ----------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------

def test_schedule_is_deterministic(scheduler, dummy_payloads):
    frames1 = scheduler.schedule(dummy_payloads)
    frames2 = scheduler.schedule(dummy_payloads)
    assert len(frames1) == len(frames2)
    for f1, f2 in zip(frames1, frames2):
        assert f1.timestamp_ms == f2.timestamp_ms
        assert f1.can_id == f2.can_id
        assert f1.payload == f2.payload


# ----------------------------------------------------------------------
# Multiple vehicle classes
# ----------------------------------------------------------------------

def test_suv_schedule_uses_suv_can_ids(loader):
    suv_messages = loader.get_messages_for_vehicle("suv_b")
    scheduler = FrameScheduler(suv_messages, base_rate_hz=100, duration_s=60.0)

    payloads = {
        msg["id"]: [bytes(8) for _ in range(6000)]
        for msg in suv_messages.values()
    }
    frames = scheduler.schedule(payloads)
    can_ids_seen = {f.can_id for f in frames}
    assert can_ids_seen == {0x200, 0x220, 0x2B0}


# ----------------------------------------------------------------------
# Dataclass behavior
# ----------------------------------------------------------------------

def test_scheduled_frame_as_dict():
    frame = ScheduledFrame(timestamp_ms=10, can_id=0x100, payload=b"\x01\x02")
    d = frame.as_dict()
    assert d["timestamp_ms"] == 10
    assert d["can_id"] == 0x100
    assert d["payload"] == b"\x01\x02"