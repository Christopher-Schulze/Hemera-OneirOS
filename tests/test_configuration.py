"""Tests for configuration loading and validation."""
from __future__ import annotations

from pathlib import Path

import pytest

from oneiros.configuration import (
    AppConfig,
    Curve,
    ISA,
    Profile,
    Wrapper,
    available_profiles,
    build_profile_config,
    load_config,
    parse_overrides,
    profile_metadata,
)


def fixture_path(name: str) -> Path:
    return Path(__file__).parent.parent / "configs" / name


def test_load_default_config() -> None:
    config = load_config(fixture_path("oneiros.default.yaml"))
    assert isinstance(config, AppConfig)
    assert config.oneiros.profile is Profile.CONTINUATIONS
    assert config.oneiros.isa is ISA.RV32IM
    assert config.injector.wrapper is Wrapper.GROTH16
    assert config.injector.primary_curve is Curve.BLS12_381


def test_overrides_are_applied(tmp_path: Path) -> None:
    src = fixture_path("oneiros.default.yaml")
    tmp = tmp_path / "config.yaml"
    tmp.write_text(src.read_text(), encoding="utf-8")

    overrides = parse_overrides([
        "oneiros.profile=standard",
        "oneiros.page_size=8192",
        "injector.fallback_curve=pallas",
        "proving.ldt=stir",
    ])

    config = load_config(tmp, overrides=overrides)
    assert config.oneiros.profile is Profile.STANDARD
    assert config.oneiros.page_size == 8192
    assert config.proving.ldt.value == "stir"
    assert config.injector.fallback_curve is Curve.PALLAS


def test_build_profile_config() -> None:
    config = build_profile_config(Profile.STANDARD)
    assert config.oneiros.profile is Profile.STANDARD
    metadata = profile_metadata(Profile.STANDARD)
    assert metadata.defaults["oneiros"]["page_size"] == 4096
    assert config.oneiros.page_size == 4096


def test_profile_overrides_do_not_mutate_defaults() -> None:
    metadata_before = profile_metadata(Profile.DISTRIBUTED)
    original_page_size = metadata_before.defaults["oneiros"]["page_size"]

    config = build_profile_config(
        Profile.DISTRIBUTED,
        overrides={"oneiros.page_size": 16384},
    )

    assert config.oneiros.page_size == 16384
    metadata_after = profile_metadata(Profile.DISTRIBUTED)
    assert metadata_after.defaults["oneiros"]["page_size"] == original_page_size


def test_available_profiles_summary() -> None:
    profiles = available_profiles()
    assert set(profiles) == {Profile.STANDARD, Profile.CONTINUATIONS, Profile.DISTRIBUTED}
    summary = profiles[Profile.CONTINUATIONS].summary()
    assert summary["profile"] == "continuations"
    assert "target_cycles" in summary


def test_invalid_curve_pair_raises(tmp_path: Path) -> None:
    data = {
        "oneiros": {
            "profile": "standard",
            "isa": "rv32im",
            "isa_mode": "standard",
            "zero_knowledge": True,
            "ivc_backend": "hypernova",
            "recursion_curve": "pasta",
            "memory_model": "sparse_merkle_twist",
            "page_size": 4096,
            "max_cycles_per_segment": 1024,
            "transcript": "poseidon2_v2",
            "security_bits": 128,
            "lookup_backend": "lasso",
            "hardware_backend": "cpu",
        },
        "proving": {
            "ldt": "deepfri",
            "stark_field": "m31",
            "pcs": "deepfold",
            "folding": "hypernova",
        },
        "injector": {
            "wrapper": "groth16",
            "primary_curve": "bls12_381",
            "fallback_curve": "bls12_381",
            "aggregation": "snarkpack_cp_snark",
            "blob_binding": "eip4844_kzg",
        },
    }

    path = tmp_path / "config.yaml"
    path.write_text(yaml_dump(data), encoding="utf-8")

    with pytest.raises(ValueError):
        load_config(path)


def yaml_dump(data: object) -> str:
    import yaml

    return yaml.safe_dump(data)
