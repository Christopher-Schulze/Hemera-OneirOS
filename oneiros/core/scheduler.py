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
from typing import Dict, Iterable, List, Sequence, Tuple

try:  # Python < 3.11 compatibility for Literal type
    from typing import Literal
except ImportError:  # pragma: no cover - fallback for older interpreters
    from typing_extensions import Literal

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
    idle_time_seconds: float
    first_segment_start_seconds: float | None
    last_segment_end_seconds: float | None
    idle_windows: Tuple["IdleWindow", ...]

    def to_dict(self) -> Dict[str, object]:
        return {
            "prover": self.prover,
            "segments": self.segments,
            "busy_time_seconds": self.busy_time_seconds,
            "utilisation": self.utilisation,
            "idle_time_seconds": self.idle_time_seconds,
            "first_segment_start_seconds": self.first_segment_start_seconds,
            "last_segment_end_seconds": self.last_segment_end_seconds,
            "idle_windows": [window.to_dict() for window in self.idle_windows],
        }


@dataclass(frozen=True)
class IdleWindow:
    """Represents an idle interval for a single prover."""

    start_time_seconds: float
    end_time_seconds: float

    def __post_init__(self) -> None:
        if self.end_time_seconds < self.start_time_seconds:
            raise ValueError("IdleWindow end time must be >= start time")

    @property
    def duration_seconds(self) -> float:
        return self.end_time_seconds - self.start_time_seconds

    def to_dict(self) -> Dict[str, float]:
        return {
            "start_time_seconds": self.start_time_seconds,
            "end_time_seconds": self.end_time_seconds,
            "duration_seconds": self.duration_seconds,
        }


@dataclass(frozen=True)
class TimelineEvent:
    """Describes an event on the prover or aggregator timeline."""

    kind: Literal["segment", "idle", "aggregator"]
    resource: str
    start_time_seconds: float
    end_time_seconds: float
    metadata: Dict[str, object] | None = None

    def __post_init__(self) -> None:
        if self.end_time_seconds < self.start_time_seconds:
            raise ValueError("TimelineEvent end time must be >= start time")

    @property
    def duration_seconds(self) -> float:
        return self.end_time_seconds - self.start_time_seconds

    def to_dict(self) -> Dict[str, object]:
        payload = {
            "kind": self.kind,
            "resource": self.resource,
            "start_time_seconds": self.start_time_seconds,
            "end_time_seconds": self.end_time_seconds,
            "duration_seconds": self.duration_seconds,
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


@dataclass(frozen=True)
class ScheduleTimeline:
    """Collection of timeline events that describe the schedule."""

    events: Tuple[TimelineEvent, ...]
    total_duration_seconds: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "total_duration_seconds": self.total_duration_seconds,
            "events": [event.to_dict() for event in self.events],
        }

    def filter(
        self,
        *,
        kind: Literal["segment", "idle", "aggregator"] | None = None,
        resource: str | None = None,
    ) -> Tuple[TimelineEvent, ...]:
        """Return events matching ``kind`` and/or ``resource``."""

        result = []
        for event in self.events:
            if kind is not None and event.kind != kind:
                continue
            if resource is not None and event.resource != resource:
                continue
            result.append(event)
        return tuple(result)

    @property
    def resources(self) -> Tuple[str, ...]:
        """Return the distinct resources present in the timeline."""

        return tuple(sorted({event.resource for event in self.events}))


@dataclass(frozen=True)
class ScheduleAnalytics:
    """Derived analytics for the produced schedule."""

    peak_concurrency: int
    average_concurrency: float
    busy_time_seconds: float
    idle_time_seconds: float
    effective_throughput_cycles_per_second: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "peak_concurrency": self.peak_concurrency,
            "average_concurrency": self.average_concurrency,
            "busy_time_seconds": self.busy_time_seconds,
            "idle_time_seconds": self.idle_time_seconds,
            "effective_throughput_cycles_per_second": self.effective_throughput_cycles_per_second,
        }


@dataclass(frozen=True)
class ScheduleSummary:
    """Summary of the derived schedule."""

    total_duration_seconds: float
    prover_stats: Tuple[ProverStats, ...]
    aggregator: AggregatorWindow
    analytics: ScheduleAnalytics

    def to_dict(self) -> Dict[str, object]:
        return {
            "total_duration_seconds": self.total_duration_seconds,
            "prover_stats": [stat.to_dict() for stat in self.prover_stats],
            "aggregator": self.aggregator.to_dict(),
            "analytics": self.analytics.to_dict(),
        }


@dataclass(frozen=True)
class ProverSchedule:
    """Container bundling segment assignments and summary data."""

    plan: ExecutionPlan
    segments: Tuple[SegmentSchedule, ...]
    summary: ScheduleSummary
    timeline: ScheduleTimeline

    def to_dict(self) -> Dict[str, object]:
        return {
            "plan": self.plan.to_dict(),
            "segments": [segment.to_dict() for segment in self.segments],
            "summary": self.summary.to_dict(),
            "timeline": self.timeline.to_dict(),
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
    prover_assignments: List[List[SegmentSchedule]] = [[] for _ in range(plan.concurrency)]

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
        segment = SegmentSchedule(
            index=index,
            prover=prover,
            cycles=cycles,
            start_cycle=start_cycle,
            end_cycle=end_cycle,
            start_time_seconds=start_time,
            end_time_seconds=end_time,
        )
        segments.append(segment)
        prover_assignments[prover].append(segment)

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
    idle_windows_per_prover: List[Tuple[IdleWindow, ...]] = []
    epsilon = 1e-12
    for prover, busy in enumerate(prover_busy):
        assigned_segments = tuple(sorted(prover_assignments[prover], key=lambda item: item.start_time_seconds))
        if assigned_segments:
            first_start = assigned_segments[0].start_time_seconds
            last_end = assigned_segments[-1].end_time_seconds
        else:
            first_start = None
            last_end = None
        idle_windows = _compute_idle_windows(assigned_segments, total_duration, epsilon)
        idle_time = sum(window.duration_seconds for window in idle_windows)
        utilisation = 0.0
        if total_duration > 0:
            utilisation = busy / total_duration
        idle_windows_per_prover.append(idle_windows)
        stats.append(
            ProverStats(
                prover=prover,
                segments=len(assigned_segments),
                busy_time_seconds=busy,
                utilisation=utilisation,
                idle_time_seconds=idle_time,
                first_segment_start_seconds=first_start,
                last_segment_end_seconds=last_end,
                idle_windows=idle_windows,
            )
        )

    busy_time_total = sum(segment.duration_seconds for segment in segments)
    peak_concurrency, average_concurrency = _compute_concurrency_metrics(segments, total_duration)
    idle_time_total = sum(stat.idle_time_seconds for stat in stats)
    effective_throughput = 0.0
    if total_duration > 0:
        effective_throughput = plan.total_cycles / total_duration

    analytics = ScheduleAnalytics(
        peak_concurrency=peak_concurrency,
        average_concurrency=average_concurrency,
        busy_time_seconds=busy_time_total,
        idle_time_seconds=idle_time_total,
        effective_throughput_cycles_per_second=effective_throughput,
    )

    summary = ScheduleSummary(
        total_duration_seconds=total_duration,
        prover_stats=tuple(stats),
        aggregator=aggregator,
        analytics=analytics,
    )
    timeline = _build_timeline(segments, idle_windows_per_prover, aggregator, total_duration)
    return ProverSchedule(
        plan=plan,
        segments=tuple(segments),
        summary=summary,
        timeline=timeline,
    )


def _select_prover(available: Iterable[float]) -> int:
    """Return the index of the prover that becomes available first."""

    best_index = 0
    best_time = math.inf
    for index, ready_at in enumerate(available):
        if ready_at < best_time:
            best_time = ready_at
            best_index = index
    return best_index


def _compute_idle_windows(
    segments: Tuple[SegmentSchedule, ...],
    total_duration: float,
    epsilon: float,
) -> Tuple[IdleWindow, ...]:
    """Compute idle intervals for ``segments`` over ``total_duration``."""

    windows: List[IdleWindow] = []
    if total_duration <= 0:
        return tuple()

    if not segments:
        if total_duration > 0:
            windows.append(IdleWindow(0.0, total_duration))
        return tuple(windows)

    cursor = 0.0
    if segments[0].start_time_seconds - cursor > epsilon:
        windows.append(IdleWindow(cursor, segments[0].start_time_seconds))
    cursor = segments[0].end_time_seconds
    for segment in segments[1:]:
        gap = segment.start_time_seconds - cursor
        if gap > epsilon:
            windows.append(IdleWindow(cursor, segment.start_time_seconds))
        cursor = segment.end_time_seconds
    if total_duration - cursor > epsilon:
        windows.append(IdleWindow(cursor, total_duration))
    return tuple(windows)


def _compute_concurrency_metrics(
    segments: Sequence[SegmentSchedule],
    total_duration: float,
) -> Tuple[int, float]:
    """Return peak and average concurrency for ``segments``."""

    if not segments or total_duration <= 0:
        return 0, 0.0

    events: List[Tuple[float, int]] = []
    for segment in segments:
        events.append((segment.start_time_seconds, 1))
        events.append((segment.end_time_seconds, -1))
    events.sort(key=lambda item: (item[0], -item[1]))

    concurrency = 0
    peak = 0
    busy_area = 0.0
    last_time = events[0][0]
    for time, delta in events:
        if time > last_time:
            busy_area += concurrency * (time - last_time)
        concurrency += delta
        peak = max(peak, concurrency)
        last_time = time
    average = busy_area / total_duration if total_duration > 0 else 0.0
    return peak, average


def _build_timeline(
    segments: Sequence[SegmentSchedule],
    idle_windows: Sequence[Tuple[IdleWindow, ...]],
    aggregator: AggregatorWindow,
    total_duration: float,
) -> ScheduleTimeline:
    """Construct a :class:`ScheduleTimeline` from components."""

    events: List[TimelineEvent] = []
    for segment in segments:
        events.append(
            TimelineEvent(
                kind="segment",
                resource=f"prover-{segment.prover}",
                start_time_seconds=segment.start_time_seconds,
                end_time_seconds=segment.end_time_seconds,
                metadata={
                    "segment_index": segment.index,
                    "cycles": segment.cycles,
                },
            )
        )

    for prover, windows in enumerate(idle_windows):
        for window in windows:
            events.append(
                TimelineEvent(
                    kind="idle",
                    resource=f"prover-{prover}",
                    start_time_seconds=window.start_time_seconds,
                    end_time_seconds=window.end_time_seconds,
                    metadata={"prover": prover},
                )
            )

    events.append(
        TimelineEvent(
            kind="aggregator",
            resource="aggregator",
            start_time_seconds=aggregator.start_time_seconds,
            end_time_seconds=aggregator.end_time_seconds,
            metadata={
                "latency_seconds": aggregator.latency_seconds,
                "segments": aggregator.segments,
            },
        )
    )

    events.sort(key=lambda event: (event.start_time_seconds, event.end_time_seconds, event.resource, event.kind))
    return ScheduleTimeline(events=tuple(events), total_duration_seconds=total_duration)


__all__ = [
    "AggregatorWindow",
    "IdleWindow",
    "ProverSchedule",
    "ProverStats",
    "ScheduleSummary",
    "ScheduleTimeline",
    "ScheduleAnalytics",
    "SegmentSchedule",
    "TimelineEvent",
    "build_schedule",
    "build_schedule_from_plan",
]
