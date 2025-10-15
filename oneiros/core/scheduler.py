"""Segment scheduling utilities for the OneirOS prover pipeline.

The scheduler consumes execution plans produced by
:func:`oneiros.core.planner.build_execution_plan` and breaks them into
per-prover workloads with coarse yet deterministic timing estimates.
This allows operational tooling (CLI, dashboards, orchestration) to
reason about expected prover utilisation, wall clock duration and
aggregator latency without depending on hardware integrations.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Tuple

from ..configuration import AppConfig
from .planner import ExecutionPlan, build_execution_plan


@dataclass(frozen=True)
class SegmentSchedule:
    """Assignment of a single execution segment to a prover slot."""

    index: int
    prover: int
    cycles: int
    start_cycle: int
    end_cycle: int
    start_time_seconds: float
    end_time_seconds: float

    @property
    def duration_seconds(self) -> float:
        return self.end_time_seconds - self.start_time_seconds

    def to_dict(self) -> Dict[str, object]:
        return {
            "index": self.index,
            "prover": self.prover,
            "cycles": self.cycles,
            "start_cycle": self.start_cycle,
            "end_cycle": self.end_cycle,
            "start_time_seconds": self.start_time_seconds,
            "end_time_seconds": self.end_time_seconds,
            "duration_seconds": self.duration_seconds,
        }


@dataclass(frozen=True)
class AggregatorWindow:
    """Window describing when the aggregator runs."""

    start_time_seconds: float
    end_time_seconds: float
    latency_seconds: float
    segments: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "start_time_seconds": self.start_time_seconds,
            "end_time_seconds": self.end_time_seconds,
            "latency_seconds": self.latency_seconds,
            "segments": self.segments,
        }


@dataclass(frozen=True)
class ProverStats:
    """Utilisation statistics for a single prover worker."""

    prover: int
    segments: int
    busy_time_seconds: float
    utilisation: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "prover": self.prover,
            "segments": self.segments,
            "busy_time_seconds": self.busy_time_seconds,
            "utilisation": self.utilisation,
        }


@dataclass(frozen=True)
class ScheduleSummary:
    """Summary of the derived schedule."""

    total_duration_seconds: float
    prover_stats: Tuple[ProverStats, ...]
    aggregator: AggregatorWindow

    def to_dict(self) -> Dict[str, object]:
        return {
            "total_duration_seconds": self.total_duration_seconds,
            "prover_stats": [stat.to_dict() for stat in self.prover_stats],
            "aggregator": self.aggregator.to_dict(),
        }


@dataclass(frozen=True)
class ProverSchedule:
    """Container bundling segment assignments and summary data."""

    plan: ExecutionPlan
    segments: Tuple[SegmentSchedule, ...]
    summary: ScheduleSummary

    def to_dict(self) -> Dict[str, object]:
        return {
            "plan": self.plan.to_dict(),
            "segments": [segment.to_dict() for segment in self.segments],
            "summary": self.summary.to_dict(),
        }


def build_schedule(
    config: AppConfig,
    *,
    workload_cycles: int | None = None,
    working_set_bytes: int | None = None,
    hardware_throughput: float | None = None,
) -> ProverSchedule:
    """Derive a :class:`ProverSchedule` from ``config``.

    The helper mirrors :func:`oneiros.core.planner.build_execution_plan`
    and accepts the same optional overrides.  It first produces an
    :class:`ExecutionPlan` which is then expanded into a segment-level
    schedule.
    """

    plan = build_execution_plan(
        config,
        workload_cycles=workload_cycles,
        working_set_bytes=working_set_bytes,
        hardware_throughput=hardware_throughput,
    )
    return build_schedule_from_plan(plan)


def build_schedule_from_plan(plan: ExecutionPlan) -> ProverSchedule:
    """Expand ``plan`` into a prover/aggregator schedule."""

    if plan.concurrency <= 0:
        raise ValueError("Plan concurrency must be positive")
    if plan.segments <= 0:
        raise ValueError("Plan must contain at least one segment")
    if plan.cycles_per_segment <= 0:
        raise ValueError("Plan cycles per segment must be positive")

    total_cycles = plan.total_cycles
    base_segment_cycles = plan.cycles_per_segment
    segments: List[SegmentSchedule] = []
    segment_cycles: List[int] = []
    cycles_remaining = total_cycles
    for index in range(plan.segments):
        if index == plan.segments - 1:
            cycles = cycles_remaining
        else:
            cycles = min(base_segment_cycles, cycles_remaining)
        cycles = max(1, cycles)
        segment_cycles.append(cycles)
        cycles_remaining = max(0, cycles_remaining - cycles)

    aggregated_throughput = plan.resources.throughput_cycles_per_second
    if aggregated_throughput <= 0:
        raise ValueError("Plan throughput must be positive")
    per_prover_throughput = aggregated_throughput / plan.concurrency
    if per_prover_throughput <= 0:
        raise ValueError("Per-prover throughput must be positive")

    prover_available: List[float] = [0.0 for _ in range(plan.concurrency)]
    prover_busy: List[float] = [0.0 for _ in range(plan.concurrency)]

    cycle_cursor = 0
    for index, cycles in enumerate(segment_cycles):
        prover = _select_prover(prover_available)
        start_time = prover_available[prover]
        duration = cycles / per_prover_throughput
        end_time = start_time + duration
        prover_available[prover] = end_time
        prover_busy[prover] += duration
        start_cycle = cycle_cursor
        end_cycle = cycle_cursor + cycles
        cycle_cursor = end_cycle
        segments.append(
            SegmentSchedule(
                index=index,
                prover=prover,
                cycles=cycles,
                start_cycle=start_cycle,
                end_cycle=end_cycle,
                start_time_seconds=start_time,
                end_time_seconds=end_time,
            )
        )

    last_segment_end = max(segment.end_time_seconds for segment in segments)
    aggregator_latency = plan.resources.aggregator_latency_seconds
    aggregator_start = last_segment_end
    aggregator_end = aggregator_start + aggregator_latency
    aggregator = AggregatorWindow(
        start_time_seconds=aggregator_start,
        end_time_seconds=aggregator_end,
        latency_seconds=aggregator_latency,
        segments=len(segments),
    )

    total_duration = max(aggregator_end, last_segment_end)
    stats: List[ProverStats] = []
    for prover, busy in enumerate(prover_busy):
        utilisation = 0.0 if total_duration == 0 else busy / total_duration
        assigned_segments = sum(1 for segment in segments if segment.prover == prover)
        stats.append(
            ProverStats(
                prover=prover,
                segments=assigned_segments,
                busy_time_seconds=busy,
                utilisation=utilisation,
            )
        )

    summary = ScheduleSummary(
        total_duration_seconds=total_duration,
        prover_stats=tuple(stats),
        aggregator=aggregator,
    )
    return ProverSchedule(plan=plan, segments=tuple(segments), summary=summary)


def _select_prover(available: Iterable[float]) -> int:
    """Return the index of the prover that becomes available first."""

    best_index = 0
    best_time = math.inf
    for index, ready_at in enumerate(available):
        if ready_at < best_time:
            best_time = ready_at
            best_index = index
    return best_index


__all__ = [
    "AggregatorWindow",
    "ProverSchedule",
    "ProverStats",
    "ScheduleSummary",
    "SegmentSchedule",
    "build_schedule",
    "build_schedule_from_plan",
]
