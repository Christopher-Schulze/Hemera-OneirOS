"""Execution planning utilities derived from configuration profiles.

This module contains the core heuristics that translate high level
configuration profiles into concrete execution and proving plans.  The
initial repository only exposed a thin wrapper around the configuration
metadata which resulted in limited insight when building operational
pipelines or CLI tooling.  The implementation has been expanded to model
proving resources such as throughput, peak memory and compression
latency.  While the values are heuristic, they are deterministic and
provide a coarse yet actionable baseline for capacity planning and
regression testing.

The estimates here intentionally stay self-contained and do not depend on
external services.  Instead, small lookup tables encode the assumptions
from the specification (cf. ``docs/zielbild/spec.md``) so that the CLI and
higher-level tooling can offer informative breakdowns even before real
hardware benchmarks exist.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict

from ..configuration import (
    AppConfig,
    HardwareBackend,
    ISA,
    IVCBackend,
    MemoryModel,
    Profile,
    Wrapper,
    profile_metadata,
)


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
class ProvingResourceEstimate:
    """Derived proving metrics for a workload.

    The values are intentionally coarse but deterministic.  They provide a
    consistent interface for the CLI and future orchestration layers while
    keeping the core implementation lightweight.
    """

    throughput_cycles_per_second: float
    estimated_proving_time_seconds: float
    aggregator_latency_seconds: float
    proof_size_bytes: int
    peak_memory_bytes: int
    segments_per_proof: int
    estimated_energy_kwh: float
    estimated_energy_cost_usd: float
    hardware_occupancy_hours: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "throughput_cycles_per_second": self.throughput_cycles_per_second,
            "estimated_proving_time_seconds": self.estimated_proving_time_seconds,
            "aggregator_latency_seconds": self.aggregator_latency_seconds,
            "proof_size_bytes": self.proof_size_bytes,
            "peak_memory_bytes": self.peak_memory_bytes,
            "segments_per_proof": self.segments_per_proof,
            "estimated_energy_kwh": self.estimated_energy_kwh,
            "estimated_energy_cost_usd": self.estimated_energy_cost_usd,
            "hardware_occupancy_hours": self.hardware_occupancy_hours,
        }


@dataclass(frozen=True)
class TimelineBreakdown:
    """High level timings for major phases of the proving pipeline."""

    setup_seconds: float
    proving_seconds: float
    aggregation_seconds: float
    total_seconds: float
    amortized_seconds_per_segment: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "setup_seconds": self.setup_seconds,
            "proving_seconds": self.proving_seconds,
            "aggregation_seconds": self.aggregation_seconds,
            "total_seconds": self.total_seconds,
            "amortized_seconds_per_segment": self.amortized_seconds_per_segment,
        }


@dataclass(frozen=True)
class PlanDiagnostics:
    """Lightweight qualitative guidance for the generated plan."""

    bottleneck: str
    utilization_score: float
    scaling_efficiency: float
    notes: tuple[str, ...]

    def to_dict(self) -> Dict[str, object]:
        return {
            "bottleneck": self.bottleneck,
            "utilization_score": self.utilization_score,
            "scaling_efficiency": self.scaling_efficiency,
            "notes": list(self.notes),
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
    resources: ProvingResourceEstimate
    timeline: TimelineBreakdown
    diagnostics: PlanDiagnostics

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
            "resources": self.resources.to_dict(),
            "timeline": self.timeline.to_dict(),
            "diagnostics": self.diagnostics.to_dict(),
        }


def build_execution_plan(
    config: AppConfig,
    workload_cycles: int | None = None,
    working_set_bytes: int | None = None,
    hardware_throughput: float | None = None,
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

    resources = estimate_proving_resources(
        config,
        total_cycles=total_cycles,
        segments=segments,
        concurrency=concurrency,
        memory_plan=memory,
        hardware_throughput_override=hardware_throughput,
    )

    timeline = _build_timeline_breakdown(
        config,
        resources=resources,
        segments=segments,
        concurrency=concurrency,
    )
    diagnostics = _diagnose_plan(
        config,
        resources=resources,
        timeline=timeline,
        memory_plan=memory,
        concurrency=concurrency,
        total_cycles=total_cycles,
    )

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
        resources=resources,
        timeline=timeline,
        diagnostics=diagnostics,
    )


def estimate_proving_resources(
    config: AppConfig,
    *,
    total_cycles: int,
    segments: int,
    concurrency: int,
    memory_plan: MemoryPlan,
    hardware_throughput_override: float | None = None,
) -> ProvingResourceEstimate:
    """Estimate proving resources for ``config`` and plan parameters.

    ``hardware_throughput_override`` allows callers (e.g. CLI users) to
    inject benchmarks gathered from real hardware.  When omitted, the
    throughput is derived from the configuration profile and hardware
    backend heuristics.
    """

    if total_cycles <= 0:
        raise ValueError("total_cycles must be positive")
    if segments <= 0:
        raise ValueError("segments must be positive")
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")

    throughput = _derive_throughput(config, concurrency)
    if hardware_throughput_override is not None:
        if hardware_throughput_override <= 0:
            raise ValueError("hardware_throughput_override must be positive")
        throughput = hardware_throughput_override

    proving_time = total_cycles / throughput
    aggregator_latency = _estimate_aggregator_latency(config, segments, concurrency)
    proof_size = _estimate_proof_size(config, segments)
    peak_memory = _estimate_peak_memory(config, memory_plan, concurrency)
    energy_kwh, energy_cost = _estimate_energy_consumption(
        config,
        total_cycles=total_cycles,
        proving_time=proving_time,
        concurrency=concurrency,
    )
    occupancy_hours = max((proving_time * max(1, concurrency)) / 3600.0, 0.0)

    return ProvingResourceEstimate(
        throughput_cycles_per_second=throughput,
        estimated_proving_time_seconds=proving_time,
        aggregator_latency_seconds=aggregator_latency,
        proof_size_bytes=proof_size,
        peak_memory_bytes=peak_memory,
        segments_per_proof=segments,
        estimated_energy_kwh=energy_kwh,
        estimated_energy_cost_usd=energy_cost,
        hardware_occupancy_hours=occupancy_hours,
    )


def _build_timeline_breakdown(
    config: AppConfig,
    *,
    resources: ProvingResourceEstimate,
    segments: int,
    concurrency: int,
) -> TimelineBreakdown:
    setup = _estimate_setup_time(config, concurrency)
    proving = max(resources.estimated_proving_time_seconds, 0.0)
    aggregation = max(resources.aggregator_latency_seconds, 0.0)
    total = setup + proving + aggregation
    amortized = total / max(1, segments)
    return TimelineBreakdown(
        setup_seconds=setup,
        proving_seconds=proving,
        aggregation_seconds=aggregation,
        total_seconds=total,
        amortized_seconds_per_segment=amortized,
    )


def _diagnose_plan(
    config: AppConfig,
    *,
    resources: ProvingResourceEstimate,
    timeline: TimelineBreakdown,
    memory_plan: MemoryPlan,
    concurrency: int,
    total_cycles: int,
) -> PlanDiagnostics:
    durations = {
        "setup": timeline.setup_seconds,
        "proving": timeline.proving_seconds,
        "aggregation": timeline.aggregation_seconds,
    }
    bottleneck = max(durations, key=durations.get)

    theoretical_throughput = _derive_throughput(config, concurrency=max(1, concurrency))
    utilization_score = min(
        1.0,
        resources.throughput_cycles_per_second / max(theoretical_throughput, 1.0),
    )

    target_concurrency = profile_metadata(config.oneiros.profile).concurrency
    scaling_efficiency = min(
        1.0,
        max(1, concurrency) / max(1, target_concurrency),
    )

    notes: list[str] = []
    if resources.peak_memory_bytes > memory_plan.total_bytes * 1.5:
        notes.append("Peak memory significantly exceeds allocated pages; consider larger paging strategy")
    if resources.proof_size_bytes > 10_000_000:
        notes.append("Proof size exceeds 10MB; downstream transmission may require compression optimisations")
    if timeline.setup_seconds > timeline.proving_seconds * 0.5:
        notes.append("Setup dominates proving time; reuse arithmetisation artifacts where possible")
    if resources.estimated_energy_kwh > 25:
        notes.append("High energy footprint detected; evaluate hardware efficiency or reduce concurrency")
    if total_cycles > 5_000_000:
        notes.append("Total cycles exceed 5M; ensure scheduler fragmentation remains manageable")
    if not notes:
        notes.append("Plan within nominal operating parameters")

    return PlanDiagnostics(
        bottleneck=bottleneck,
        utilization_score=round(utilization_score, 3),
        scaling_efficiency=round(min(scaling_efficiency, 1.0), 3),
        notes=tuple(notes),
    )


def _derive_throughput(config: AppConfig, concurrency: int) -> float:
    base = _IVC_BASE_THROUGHPUT[config.oneiros.ivc_backend]
    hardware_factor = _HARDWARE_FACTORS[config.oneiros.hardware_backend]
    isa_factor = _ISA_COMPLEXITY[config.oneiros.isa]
    mode_factor = 0.92 if config.oneiros.isa_mode.value == "per_instruction_air" else 1.0
    zk_factor = 0.88 if config.oneiros.zero_knowledge else 1.0

    throughput = base * hardware_factor * concurrency * zk_factor * mode_factor / isa_factor
    return max(throughput, 1_000.0)


def _estimate_proof_size(config: AppConfig, segments: int) -> int:
    base = _IVC_PROOF_SIZE[config.oneiros.ivc_backend]
    zk_multiplier = 1.25 if config.oneiros.zero_knowledge else 1.0
    wrapper_multiplier = _WRAPPER_COMPRESSION_MULTIPLIER[config.injector.wrapper]
    curve_multiplier = _CURVE_SIZE_MULTIPLIER[config.injector.primary_curve.value]
    per_segment = base * zk_multiplier * wrapper_multiplier * curve_multiplier
    # aggregator overhead grows logarithmically with segments
    overhead = int(per_segment * math.log2(max(1, segments)) * 0.05)
    total = int(per_segment * segments + overhead)
    return max(total, 1_024)


def _estimate_peak_memory(
    config: AppConfig, memory_plan: MemoryPlan, concurrency: int
) -> int:
    model_multiplier = _MEMORY_MODEL_MULTIPLIER[config.oneiros.memory_model]
    base = memory_plan.total_bytes * model_multiplier
    concurrency_overhead = concurrency * _CONCURRENCY_MEMORY_OVERHEAD_BYTES
    backend_multiplier = _HARDWARE_MEMORY_MULTIPLIER[config.oneiros.hardware_backend]
    estimated = int(base * backend_multiplier + concurrency_overhead)
    return max(estimated, memory_plan.total_bytes)


def _estimate_aggregator_latency(
    config: AppConfig, segments: int, concurrency: int
) -> float:
    wrapper_base = _WRAPPER_LATENCY_SECONDS[config.injector.wrapper]
    curve_penalty = _CURVE_LATENCY_PENALTY[config.injector.primary_curve.value]
    concurrency_gain = math.log2(max(1, concurrency)) * 0.1
    latency = wrapper_base * segments * curve_penalty
    latency /= max(1.0, concurrency_gain if concurrency_gain > 0 else 1.0)
    return max(latency, 0.0)


def _estimate_energy_consumption(
    config: AppConfig,
    *,
    total_cycles: int,
    proving_time: float,
    concurrency: int,
) -> tuple[float, float]:
    energy_per_cycle = _HARDWARE_JOULES_PER_CYCLE[config.oneiros.hardware_backend]
    concurrency_penalty = 1.0 + max(0, concurrency - 1) * 0.08
    zk_penalty = 1.12 if config.oneiros.zero_knowledge else 1.0
    runtime_penalty = 1.0 if proving_time <= 0 else min(1.5, 1.0 + proving_time / 10_000.0)
    total_joules = total_cycles * energy_per_cycle * concurrency_penalty * zk_penalty * runtime_penalty
    energy_kwh = total_joules / 3_600_000.0
    cost_multiplier = _ENERGY_COST_MULTIPLIER[config.oneiros.hardware_backend]
    energy_cost = energy_kwh * _BASE_KWH_COST_USD * cost_multiplier
    return energy_kwh, energy_cost


def _estimate_setup_time(config: AppConfig, concurrency: int) -> float:
    base = _HARDWARE_SETUP_BASE_SECONDS[config.oneiros.hardware_backend]
    curve_multiplier = _CURVE_SETUP_MULTIPLIER[config.injector.primary_curve.value]
    wrapper_multiplier = _WRAPPER_SETUP_MULTIPLIER[config.injector.wrapper]
    concurrency_factor = 1.0 + math.log2(max(1, concurrency)) * 0.2
    zk_multiplier = 1.25 if config.oneiros.zero_knowledge else 1.0
    return base * curve_multiplier * wrapper_multiplier * concurrency_factor * zk_multiplier


_IVC_BASE_THROUGHPUT: Dict[IVCBackend, float] = {
    IVCBackend.SUPERNOVA: 320_000.0,
    IVCBackend.HYPERNOVA: 520_000.0,
    IVCBackend.HYPERNOVA_PROTOGALAXY: 680_000.0,
}


_HARDWARE_FACTORS: Dict[HardwareBackend, float] = {
    HardwareBackend.AUTO: 1.5,
    HardwareBackend.CPU: 1.0,
    HardwareBackend.CUDA: 7.5,
    HardwareBackend.METAL: 5.0,
    HardwareBackend.VULKAN: 4.5,
    HardwareBackend.FPGA: 9.0,
}


_ISA_COMPLEXITY: Dict[ISA, float] = {
    ISA.RV32IM: 1.0,
    ISA.RV64IM: 1.05,
    ISA.EVM: 1.35,
    ISA.WASM: 1.2,
    ISA.SVM: 1.15,
}


_IVC_PROOF_SIZE: Dict[IVCBackend, int] = {
    IVCBackend.SUPERNOVA: 28_000,
    IVCBackend.HYPERNOVA: 36_000,
    IVCBackend.HYPERNOVA_PROTOGALAXY: 42_000,
}


_WRAPPER_COMPRESSION_MULTIPLIER: Dict[Wrapper, float] = {
    Wrapper.GROTH16: 0.85,
    Wrapper.PLONKY3: 1.15,
    Wrapper.SUPERSONIC: 0.95,
}


_CURVE_SIZE_MULTIPLIER: Dict[str, float] = {
    "bls12_381": 1.0,
    "bn254": 0.9,
    "pallas": 0.8,
    "vesta": 0.8,
}


_MEMORY_MODEL_MULTIPLIER: Dict[MemoryModel, float] = {
    MemoryModel.SPARSE_MERKLE_TWIST: 1.35,
    MemoryModel.FULL_MERKLE: 1.8,
}


_HARDWARE_MEMORY_MULTIPLIER: Dict[HardwareBackend, float] = {
    HardwareBackend.AUTO: 1.1,
    HardwareBackend.CPU: 1.0,
    HardwareBackend.CUDA: 1.3,
    HardwareBackend.METAL: 1.25,
    HardwareBackend.VULKAN: 1.2,
    HardwareBackend.FPGA: 0.95,
}


_WRAPPER_LATENCY_SECONDS: Dict[Wrapper, float] = {
    Wrapper.GROTH16: 0.45,
    Wrapper.PLONKY3: 0.75,
    Wrapper.SUPERSONIC: 1.05,
}


_CURVE_LATENCY_PENALTY: Dict[str, float] = {
    "bls12_381": 1.0,
    "bn254": 0.85,
    "pallas": 0.65,
    "vesta": 0.65,
}


_HARDWARE_JOULES_PER_CYCLE: Dict[HardwareBackend, float] = {
    HardwareBackend.AUTO: 2.5e-6,
    HardwareBackend.CPU: 3.2e-6,
    HardwareBackend.CUDA: 1.8e-6,
    HardwareBackend.METAL: 2.1e-6,
    HardwareBackend.VULKAN: 2.2e-6,
    HardwareBackend.FPGA: 1.5e-6,
}


_ENERGY_COST_MULTIPLIER: Dict[HardwareBackend, float] = {
    HardwareBackend.AUTO: 1.0,
    HardwareBackend.CPU: 0.95,
    HardwareBackend.CUDA: 1.1,
    HardwareBackend.METAL: 1.05,
    HardwareBackend.VULKAN: 1.08,
    HardwareBackend.FPGA: 0.85,
}


_HARDWARE_SETUP_BASE_SECONDS: Dict[HardwareBackend, float] = {
    HardwareBackend.AUTO: 35.0,
    HardwareBackend.CPU: 28.0,
    HardwareBackend.CUDA: 55.0,
    HardwareBackend.METAL: 42.0,
    HardwareBackend.VULKAN: 47.0,
    HardwareBackend.FPGA: 65.0,
}


_CURVE_SETUP_MULTIPLIER: Dict[str, float] = {
    "bls12_381": 1.0,
    "bn254": 0.9,
    "pallas": 0.75,
    "vesta": 0.75,
}


_WRAPPER_SETUP_MULTIPLIER: Dict[Wrapper, float] = {
    Wrapper.GROTH16: 0.85,
    Wrapper.PLONKY3: 1.15,
    Wrapper.SUPERSONIC: 1.05,
}


_BASE_KWH_COST_USD = 0.18


_CONCURRENCY_MEMORY_OVERHEAD_BYTES = 32 * 1024 * 1024  # 32MiB per concurrent prover


__all__ = [
    "ExecutionPlan",
    "MemoryPlan",
    "ProvingResourceEstimate",
    "TimelineBreakdown",
    "PlanDiagnostics",
    "build_execution_plan",
    "estimate_proving_resources",
]
