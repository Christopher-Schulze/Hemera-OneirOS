"""Tests for execution planning helpers."""
from __future__ import annotations

import pytest

from oneiros.configuration import Profile, build_profile_config
from oneiros.core.planner import MemoryPlan, build_execution_plan


def test_execution_plan_defaults() -> None:
    config = build_profile_config(Profile.CONTINUATIONS)
    plan = build_execution_plan(config)
    assert plan.profile is Profile.CONTINUATIONS
    assert plan.concurrency == 4
    assert plan.segments >= 1
    assert plan.cycles_per_segment <= config.oneiros.max_cycles_per_segment
    assert plan.memory.pages >= 1


def test_execution_plan_custom_overrides() -> None:
    config = build_profile_config(Profile.STANDARD)
    plan = build_execution_plan(
        config,
        workload_cycles=500_000,
        working_set_bytes=10 * 1024 * 1024,
    )
    assert plan.total_cycles == 500_000
    assert plan.memory.working_set_bytes == 10 * 1024 * 1024
    assert plan.memory.total_bytes >= plan.memory.working_set_bytes


@pytest.mark.parametrize("page_size, working_set", [(0, 1), (4096, 0)])
def test_memory_plan_validation(page_size: int, working_set: int) -> None:
    with pytest.raises(ValueError):
        MemoryPlan.from_values(page_size, working_set)
