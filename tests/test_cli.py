"""Tests for the OneirOS CLI interface."""
from __future__ import annotations

from pathlib import Path

import pytest

from oneiros.ops.cli import DEFAULT_CONFIG_PATH, format_config, main


def test_default_config_file_exists() -> None:
    assert DEFAULT_CONFIG_PATH.exists()


def test_format_config_human(capsys: pytest.CaptureFixture[str]) -> None:
    from oneiros.configuration import load_config

    config = load_config(DEFAULT_CONFIG_PATH)
    output = format_config(config, "human")
    assert "[oneiros]" in output
    assert "page_size" in output


@pytest.mark.parametrize("fmt", ["json", "yaml"])
def test_format_serialisation(fmt: str) -> None:
    from oneiros.configuration import load_config

    config = load_config(DEFAULT_CONFIG_PATH)
    rendered = format_config(config, fmt)
    assert "oneiros" in rendered


def test_cli_validate_only(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(DEFAULT_CONFIG_PATH.read_text(), encoding="utf-8")

    exit_code = main(["--config", str(config_path), "--validate-only"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == ""


def test_cli_with_override(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(DEFAULT_CONFIG_PATH.read_text(), encoding="utf-8")

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--set",
            "oneiros.page_size=8192",
            "--format",
            "json",
        ]
    )
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "8192" in output


def test_cli_list_profiles(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["--list-profiles"])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "continuations" in output
    assert "distributed" in output


def test_cli_plan_from_profile(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "--profile",
            "distributed",
            "--plan",
            "--workload-cycles",
            "100000",
            "--working-set",
            "32MiB",
            "--format",
            "yaml",
        ]
    )
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "total_cycles: 100000" in output
    assert "working_set_bytes" in output
    assert "resources" in output


def test_cli_plan_from_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(DEFAULT_CONFIG_PATH.read_text(), encoding="utf-8")

    exit_code = main([
        "--config",
        str(config_path),
        "--plan",
        "--format",
        "json",
    ])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "\"profile\": \"continuations\"" in output
    assert "resources" in output


def test_cli_plan_invalid_working_set() -> None:
    with pytest.raises(SystemExit):
        main(["--plan", "--working-set", "nonsense"])


def test_cli_plan_with_throughput_override(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "--profile",
            "standard",
            "--plan",
            "--hardware-throughput",
            "2.5Mc/s",
            "--format",
            "json",
        ]
    )
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "2.5e+06" in output or "2500000" in output


def test_cli_plan_invalid_throughput() -> None:
    with pytest.raises(SystemExit):
        main(["--plan", "--hardware-throughput", "badvalue"])
