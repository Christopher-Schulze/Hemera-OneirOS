"""Trace analysis helpers for deriving high level diagnostics.

The analyzer operates purely on :class:`~oneiros.core.trace.TraceBundle`
instances and therefore avoids any dependency on concrete proving backends.
It is intended for unit tests, CLI diagnostics and early integrations that
need to reason about synthetic execution traces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

from .trace import CPUTraceRow, MemoryAccessType, TraceBundle


@dataclass(frozen=True)
class RegisterEvent:
    """Represents a register write at a specific cycle."""

    cycle: int
    value: int

    def to_dict(self) -> Dict[str, int]:
        return {"cycle": self.cycle, "value": self.value}


@dataclass(frozen=True)
class RegisterActivity:
    """Describes all observed writes for a single register."""

    register: int
    initial_value: int
    writes: Tuple[RegisterEvent, ...]
    final_value: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "register": self.register,
            "initial_value": self.initial_value,
            "writes": [event.to_dict() for event in self.writes],
            "final_value": self.final_value,
        }


@dataclass(frozen=True)
class PageAccessStats:
    """Aggregated statistics for a single page."""

    page_id: int
    reads: int
    writes: int
    execs: int
    unique_offsets: Tuple[int, ...]

    def to_dict(self) -> Dict[str, object]:
        return {
            "page_id": self.page_id,
            "reads": self.reads,
            "writes": self.writes,
            "execs": self.execs,
            "unique_offsets": list(self.unique_offsets),
        }


@dataclass(frozen=True)
class MemorySummary:
    """High level memory access summary."""

    reads: int
    writes: int
    execs: int
    copy_on_write_events: int
    unique_addresses: int
    unique_pages: int
    root_transitions: int
    final_root: str | None
    per_page: Tuple[PageAccessStats, ...]

    def to_dict(self) -> Dict[str, object]:
        return {
            "reads": self.reads,
            "writes": self.writes,
            "execs": self.execs,
            "copy_on_write_events": self.copy_on_write_events,
            "unique_addresses": self.unique_addresses,
            "unique_pages": self.unique_pages,
            "root_transitions": self.root_transitions,
            "final_root": self.final_root,
            "per_page": [page.to_dict() for page in self.per_page],
        }


def analyze_register_activity(
    trace: TraceBundle,
    *,
    initial_registers: Mapping[int, int] | None = None,
    include_untouched: bool = False,
) -> Tuple[RegisterActivity, ...]:
    """Return register activity summaries for ``trace``."""

    values: Dict[int, int] = {i: 0 for i in range(32)}
    initial_values: Dict[int, int] = {i: 0 for i in range(32)}
    writes: Dict[int, list[RegisterEvent]] = {}
    first_values: Dict[int, int] = {}

    if initial_registers:
        for register, value in initial_registers.items():
            _validate_register_index(register)
            coerced = int(value)
            values[register] = coerced
            initial_values[register] = coerced

    for row in trace.cpu:
        _observe_registers(row, values, writes, first_values)

    if include_untouched:
        registers = list(range(32))
    else:
        registers = sorted(writes)

    activities: list[RegisterActivity] = []
    for register in registers:
        register_writes = tuple(writes.get(register, ()))
        initial_value = (
            first_values.get(register, initial_values.get(register, 0))
            if register in first_values or initial_registers
            else 0
        )
        final_value = values.get(register, initial_value)
        activities.append(
            RegisterActivity(
                register=register,
                initial_value=initial_value,
                writes=register_writes,
                final_value=register_writes[-1].value if register_writes else final_value,
            )
        )
    return tuple(activities)


def summarize_memory(trace: TraceBundle) -> MemorySummary:
    """Derive aggregate statistics from memory trace rows."""

    reads = writes = execs = cow = transitions = 0
    addresses: set[int] = set()
    pages: Dict[int, set[int]] = {}
    per_page_counters: Dict[int, Dict[str, int]] = {}
    last_root: str | None = None

    for row in trace.memory:
        if row.acc is MemoryAccessType.READ:
            reads += 1
        elif row.acc is MemoryAccessType.WRITE:
            writes += 1
        elif row.acc is MemoryAccessType.EXEC:
            execs += 1

        if row.cow:
            cow += 1
        if last_root is not None and row.new_root != last_root:
            transitions += 1
        last_root = row.new_root

        addresses.add(row.addr)
        pages.setdefault(row.page_id, set()).add(row.page_off)
        counters = per_page_counters.setdefault(row.page_id, {"reads": 0, "writes": 0, "execs": 0})
        if row.acc is MemoryAccessType.READ:
            counters["reads"] += 1
        elif row.acc is MemoryAccessType.WRITE:
            counters["writes"] += 1
        else:
            counters["execs"] += 1

    per_page_stats = [
        PageAccessStats(
            page_id=page_id,
            reads=counters["reads"],
            writes=counters["writes"],
            execs=counters["execs"],
            unique_offsets=tuple(sorted(offsets)),
        )
        for page_id, (counters, offsets) in sorted(
            ((pid, (per_page_counters[pid], pages[pid])) for pid in pages), key=lambda item: item[0]
        )
    ]

    return MemorySummary(
        reads=reads,
        writes=writes,
        execs=execs,
        copy_on_write_events=cow,
        unique_addresses=len(addresses),
        unique_pages=len(pages),
        root_transitions=transitions,
        final_root=last_root,
        per_page=tuple(per_page_stats),
    )


def _observe_registers(
    row: CPUTraceRow,
    values: Dict[int, int],
    writes: Dict[int, list[RegisterEvent]],
    first_values: Dict[int, int],
) -> None:
    for register, value in row.registers_delta:
        _validate_register_index(register)
        coerced = int(value)
        if register not in first_values:
            first_values[register] = values.get(register, 0)
        values[register] = coerced
        writes.setdefault(register, []).append(RegisterEvent(cycle=row.cycle, value=coerced))


def _validate_register_index(register: int) -> None:
    if not 0 <= register < 32:
        raise ValueError("register index must be between 0 and 31")


__all__ = [
    "MemorySummary",
    "PageAccessStats",
    "RegisterActivity",
    "RegisterEvent",
    "analyze_register_activity",
    "summarize_memory",
]
