"""Execution planning utilities derived from configuration profiles."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict

from ..configuration import AppConfig, Profile, profile_metadata


@dataclass(frozen=True)
class MemoryPlan:
    """Describes the memory footprint for a workload."""

    page_size: int
    working_set_bytes: int
    pages: int
    total_bytes: int

    @classmethod
    def from_values(cls, page_size: int, working_set_bytes: int) -> "MemoryPlan":
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        if working_set_bytes <= 0:
            raise ValueError("working_set_bytes must be positive")
        pages = max(1, math.ceil(working_set_bytes / page_size))
        total_bytes = pages * page_size
        return cls(
            page_size=page_size,
            working_set_bytes=working_set_bytes,
            pages=pages,
            total_bytes=total_bytes,
        )

    def to_dict(self) -> Dict[str, int]:
        return {
            "page_size": self.page_size,
            "working_set_bytes": self.working_set_bytes,
            "pages": self.pages,
            "total_bytes": self.total_bytes,
        }


@dataclass(frozen=True)
class ExecutionPlan:
    """High level execution breakdown for a workload."""

    profile: Profile
    concurrency: int
    zero_knowledge: bool
    segments: int
    cycles_per_segment: int
    total_cycles: int
    memory: MemoryPlan
    ivc_backend: str
    hardware_backend: str
    lookup_backend: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "profile": self.profile.value,
            "concurrency": self.concurrency,
            "zero_knowledge": self.zero_knowledge,
            "segments": self.segments,
            "cycles_per_segment": self.cycles_per_segment,
            "total_cycles": self.total_cycles,
            "memory": self.memory.to_dict(),
            "ivc_backend": self.ivc_backend,
            "hardware_backend": self.hardware_backend,
            "lookup_backend": self.lookup_backend,
        }


def build_execution_plan(
    config: AppConfig,
    workload_cycles: int | None = None,
    working_set_bytes: int | None = None,
) -> ExecutionPlan:
    """Derive an execution plan from ``config`` with optional overrides."""

    metadata = profile_metadata(config.oneiros.profile)
    total_cycles = workload_cycles if workload_cycles is not None else metadata.target_cycles
    if total_cycles <= 0:
        raise ValueError("workload_cycles must be positive")

    concurrency = max(1, metadata.concurrency)
    base_cycles = max(1, total_cycles // concurrency)
    cycles_per_segment = min(config.oneiros.max_cycles_per_segment, base_cycles)
    if cycles_per_segment <= 0:
        cycles_per_segment = min(config.oneiros.max_cycles_per_segment, total_cycles)
    segments = max(1, math.ceil(total_cycles / cycles_per_segment))

    working_set = working_set_bytes if working_set_bytes is not None else metadata.working_set_bytes
    memory = MemoryPlan.from_values(config.oneiros.page_size, working_set)

    return ExecutionPlan(
        profile=config.oneiros.profile,
        concurrency=concurrency,
        zero_knowledge=config.oneiros.zero_knowledge,
        segments=segments,
        cycles_per_segment=cycles_per_segment,
        total_cycles=total_cycles,
        memory=memory,
        ivc_backend=config.oneiros.ivc_backend.value,
        hardware_backend=config.oneiros.hardware_backend.value,
        lookup_backend=config.oneiros.lookup_backend.value,
    )


__all__ = [
    "ExecutionPlan",
    "MemoryPlan",
    "build_execution_plan",
]
