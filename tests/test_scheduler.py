from pathlib import Path

import pytest

from oneiros.configuration import load_config
from oneiros.core.scheduler import build_schedule
from oneiros.ops.cli import DEFAULT_CONFIG_PATH


@pytest.fixture(scope="module")
def config() -> Path:
    return DEFAULT_CONFIG_PATH


def test_schedule_matches_plan_segments(config: Path) -> None:
    cfg = load_config(config)
    schedule = build_schedule(cfg, workload_cycles=200_000)
    assert len(schedule.segments) == schedule.plan.segments
    assert sum(segment.cycles for segment in schedule.segments) == schedule.plan.total_cycles
    aggregator = schedule.summary.aggregator
    assert aggregator.segments == len(schedule.segments)
    last_end = max(segment.end_time_seconds for segment in schedule.segments)
    assert aggregator.start_time_seconds == pytest.approx(last_end)
    assert aggregator.end_time_seconds >= aggregator.start_time_seconds


def test_schedule_prover_stats_utilisation(config: Path) -> None:
    cfg = load_config(config)
    schedule = build_schedule(cfg, workload_cycles=120_000, hardware_throughput=1_500_000.0)
    assert all(segment.duration_seconds > 0 for segment in schedule.segments)
    total_duration = schedule.summary.total_duration_seconds
    assert total_duration >= 0
    busy_sum = sum(stat.busy_time_seconds for stat in schedule.summary.prover_stats)
    assert busy_sum >= 0
    assert all(0.0 <= stat.utilisation <= 1.0 for stat in schedule.summary.prover_stats)
    assert any(stat.segments > 0 for stat in schedule.summary.prover_stats)
    serialised = schedule.to_dict()
    assert "segments" in serialised
    assert serialised["summary"]["aggregator"]["segments"] == len(schedule.segments)
