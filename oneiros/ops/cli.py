"""Command line utilities for Hemera OneirOS."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable, List

import yaml

from ..configuration import (
    AppConfig,
    Profile,
    available_profiles,
    build_profile_config,
    load_config,
    parse_overrides,
)
from ..core.planner import ExecutionPlan, build_execution_plan
from ..core.scheduler import ProverSchedule, build_schedule

DEFAULT_CONFIG_PATH = Path("configs/oneiros.default.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hemera OneirOS operational tooling")
    parser.add_argument("--config", type=Path, help="Path to a OneirOS YAML configuration file")
    parser.add_argument(
        "--profile",
        choices=[profile.value for profile in Profile],
        help="Load a built-in profile instead of reading a configuration file",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override configuration values using dotted paths",
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List available built-in profiles and exit",
    )
    parser.add_argument(
        "--format",
        choices=("human", "json", "yaml"),
        default="human",
        help="Output format for the loaded configuration",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the configuration without printing it",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Emit a derived execution plan instead of the raw configuration",
    )
    parser.add_argument(
        "--schedule",
        action="store_true",
        help="Emit a prover/aggregator schedule derived from the configuration",
    )
    parser.add_argument(
        "--workload-cycles",
        type=str,
        help="Override the assumed workload cycles when generating a plan",
    )
    parser.add_argument(
        "--working-set",
        type=str,
        help="Override the assumed working set (e.g. 64MiB) when generating a plan",
    )
    parser.add_argument(
        "--hardware-throughput",
        type=str,
        help="Override the throughput estimate (e.g. 1.2Gc/s)",
    )
    return parser


def format_config(config: AppConfig, output_format: str) -> str:
    if output_format == "json":
        return json.dumps(config.to_dict(), indent=2, sort_keys=True)
    if output_format == "yaml":
        return yaml.safe_dump(config.to_dict(), sort_keys=False)
    return _format_human(config)


def format_plan(plan: ExecutionPlan, output_format: str) -> str:
    if output_format == "json":
        return json.dumps(plan.to_dict(), indent=2, sort_keys=True)
    if output_format == "yaml":
        return yaml.safe_dump(plan.to_dict(), sort_keys=False)
    return _format_plan_human(plan)


def format_schedule(schedule: ProverSchedule, output_format: str) -> str:
    if output_format == "json":
        return json.dumps(schedule.to_dict(), indent=2, sort_keys=True)
    if output_format == "yaml":
        return yaml.safe_dump(schedule.to_dict(), sort_keys=False)
    return _format_schedule_human(schedule)


def _format_human(config: AppConfig) -> str:
    lines: List[str] = []
    lines.append("[oneiros]")
    for key, value in config.oneiros.to_dict().items():
        lines.append(f"{key:>24}: {value}")
    lines.append("")
    lines.append("[proving]")
    for key, value in config.proving.to_dict().items():
        lines.append(f"{key:>24}: {value}")
    lines.append("")
    lines.append("[injector]")
    for key, value in config.injector.to_dict().items():
        lines.append(f"{key:>24}: {value}")
    return "\n".join(lines)


def _format_plan_human(plan: ExecutionPlan) -> str:
    lines: List[str] = []
    lines.append(f"Execution profile: {plan.profile.value}")
    lines.append(f"Concurrency: {plan.concurrency}")
    lines.append(f"Zero-knowledge: {str(plan.zero_knowledge).lower()}")
    lines.append("Segments:")
    lines.append(f"  count: {plan.segments}")
    lines.append(f"  cycles/segment: {plan.cycles_per_segment}")
    lines.append(f"  total cycles: {plan.total_cycles}")
    lines.append("Memory:")
    lines.append(f"  page size: {plan.memory.page_size}")
    lines.append(f"  working set (bytes): {plan.memory.working_set_bytes}")
    lines.append(f"  pages: {plan.memory.pages}")
    lines.append(f"  allocated bytes: {plan.memory.total_bytes}")
    lines.append("Backends:")
    lines.append(f"  ivc: {plan.ivc_backend}")
    lines.append(f"  lookup: {plan.lookup_backend}")
    lines.append(f"  hardware: {plan.hardware_backend}")
    lines.append("Resources:")
    lines.append(
        "  throughput (cycles/s): "
        f"{int(plan.resources.throughput_cycles_per_second)}"
    )
    lines.append(
        "  proving time (s): "
        f"{plan.resources.estimated_proving_time_seconds:.2f}"
    )
    lines.append(
        "  aggregator latency (s): "
        f"{plan.resources.aggregator_latency_seconds:.2f}"
    )
    lines.append(f"  proof size (bytes): {plan.resources.proof_size_bytes}")
    lines.append(f"  peak memory (bytes): {plan.resources.peak_memory_bytes}")
    return "\n".join(lines)


def _format_schedule_human(schedule: ProverSchedule) -> str:
    lines: List[str] = []
    lines.append(f"Execution profile: {schedule.plan.profile.value}")
    lines.append(f"Concurrency: {schedule.plan.concurrency}")
    lines.append(
        "Total duration (s): "
        f"{schedule.summary.total_duration_seconds:.2f}"
    )
    lines.append(f"Segments: {len(schedule.segments)}")
    lines.append("Per-segment assignments:")
    for segment in schedule.segments:
        lines.append(
            "  "
            f"#{segment.index:02d} prover={segment.prover} cycles={segment.cycles} "
            f"start={segment.start_time_seconds:.2f}s "
            f"end={segment.end_time_seconds:.2f}s"
        )
    lines.append("Prover utilisation:")
    for stat in schedule.summary.prover_stats:
        lines.append(
            "  "
            f"prover {stat.prover}: segments={stat.segments}, busy={stat.busy_time_seconds:.2f}s, "
            f"utilisation={stat.utilisation:.2%}"
        )
    aggregator = schedule.summary.aggregator
    lines.append("Aggregator:")
    lines.append(f"  start (s): {aggregator.start_time_seconds:.2f}")
    lines.append(f"  end (s): {aggregator.end_time_seconds:.2f}")
    lines.append(f"  latency (s): {aggregator.latency_seconds:.2f}")
    lines.append(f"  segments: {aggregator.segments}")
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.list_profiles:
        _print_profiles()
        return 0

    try:
        overrides = parse_overrides(args.overrides or [])
    except ValueError as exc:  # pragma: no cover - argparse already ensures format
        parser.error(str(exc))
        return 2

    config: AppConfig
    profile_name = args.profile
    if profile_name:
        config = build_profile_config(Profile(profile_name), overrides=overrides)
    else:
        config_path = args.config or DEFAULT_CONFIG_PATH
        try:
            config = load_config(config_path, overrides=overrides)
        except Exception as exc:  # pragma: no cover - surfaced to CLI
            parser.error(str(exc))
            return 2

    if args.validate_only:
        return 0

    if args.plan or args.schedule:
        if args.plan and args.schedule:
            parser.error("--plan and --schedule are mutually exclusive")
            return 2
        try:
            workload_cycles = _parse_int_argument(args.workload_cycles, "workload-cycles")
            working_set = _parse_size_argument(args.working_set) if args.working_set else None
            throughput_override = (
                _parse_throughput_argument(args.hardware_throughput)
                if args.hardware_throughput
                else None
            )
        except ValueError as exc:
            parser.error(str(exc))
            return 2
        try:
            if args.schedule:
                schedule = build_schedule(
                    config,
                    workload_cycles=workload_cycles,
                    working_set_bytes=working_set,
                    hardware_throughput=throughput_override,
                )
            else:
                plan = build_execution_plan(
                    config,
                    workload_cycles=workload_cycles,
                    working_set_bytes=working_set,
                    hardware_throughput=throughput_override,
                )
        except Exception as exc:  # pragma: no cover - surfaced to CLI
            parser.error(str(exc))
            return 2
        if args.schedule:
            print(format_schedule(schedule, args.format))
        else:
            print(format_plan(plan, args.format))
        return 0

    print(format_config(config, args.format))
    return 0


def _parse_int_argument(value: str | None, name: str) -> int | None:
    if value is None:
        return None
    cleaned = value.replace("_", "")
    try:
        parsed = int(cleaned, 10)
    except ValueError as exc:
        raise ValueError(f"Invalid integer for --{name}: {value}") from exc
    if parsed <= 0:
        raise ValueError(f"--{name} must be positive")
    return parsed


_SIZE_UNITS = {
    "b": 1,
    "kb": 1000,
    "kib": 1024,
    "mb": 1000**2,
    "mib": 1024**2,
    "gb": 1000**3,
    "gib": 1024**3,
}


def _parse_size_argument(value: str) -> int:
    cleaned = value.strip().lower().replace("_", "")
    if not cleaned:
        raise ValueError("--working-set must not be empty")

    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([a-z]*)", cleaned)
    if not match:
        raise ValueError(f"Invalid size format: {value}")

    number, unit = match.groups()
    multiplier = 1
    if unit:
        if unit not in _SIZE_UNITS:
            raise ValueError(f"Unknown size unit '{unit}'")
        multiplier = _SIZE_UNITS[unit]

    numeric = float(number)
    if math.isinf(numeric) or math.isnan(numeric):  # pragma: no cover - defensive
        raise ValueError(f"Invalid numeric value for size: {value}")
    bytes_value = int(numeric * multiplier)
    if bytes_value <= 0:
        raise ValueError("Working set must be positive")
    return bytes_value


_THROUGHPUT_UNITS = {
    "": 1.0,
    "c/s": 1.0,
    "cycles/s": 1.0,
    "hz": 1.0,
    "khz": 1_000.0,
    "kc/s": 1_000.0,
    "mhz": 1_000_000.0,
    "mc/s": 1_000_000.0,
    "ghz": 1_000_000_000.0,
    "gc/s": 1_000_000_000.0,
}


def _parse_throughput_argument(value: str | None) -> float | None:
    if value is None:
        return None
    cleaned = value.strip().lower().replace("_", "")
    if not cleaned:
        raise ValueError("--hardware-throughput must not be empty")

    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([a-z/]+)?", cleaned)
    if not match:
        raise ValueError(f"Invalid throughput format: {value}")

    number, unit = match.groups()
    unit = unit or ""
    if unit not in _THROUGHPUT_UNITS:
        raise ValueError(f"Unknown throughput unit '{unit}'")

    numeric = float(number)
    if math.isinf(numeric) or math.isnan(numeric):  # pragma: no cover - defensive
        raise ValueError(f"Invalid numeric value for throughput: {value}")

    throughput = numeric * _THROUGHPUT_UNITS[unit]
    if throughput <= 0:
        raise ValueError("Throughput override must be positive")
    return throughput


def _print_profiles() -> None:
    profiles = available_profiles()
    lines: List[str] = []
    for profile in Profile:
        meta = profiles[profile]
        lines.append(
            f"{profile.value:>14}: {meta.description} (target cycles={meta.target_cycles:,},"
            f" working set={meta.working_set_bytes:,} bytes, concurrency={meta.concurrency})"
        )
    print("\n".join(lines))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
