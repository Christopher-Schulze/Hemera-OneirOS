# Hemera OneirOS Changelog

All notable changes to this repository are documented here. Each entry should include the date (UTC), a concise summary, impacted areas, and references to specs or runbook sections when applicable.

## 2025-10-13

### Added
- Established foundational documentation structure:
  - Added `docs/zielbild/spec.md` containing the Hemera OneirOS (zkVM) v1.1 architecture specification.
  - Introduced `docs/DOCUMENTATION.md` as the single-source documentation hub.
  - Created `runbook.md` to codify mandatory workflow rules tied to the spec.
  - Added `fileandwiremap.md` capturing repository topology and integration obligations.
  - Logged this changelog to enforce auditability requirements.

## 2025-10-14

### Added
- Scaffolded the `oneiros/` implementation skeleton with module READMEs for core, frontends, proving, SDKs, and ops.
- Introduced `configs/oneiros.default.yaml` encoding canonical defaults from the specification.
- Authored `docs/architecture/overview.md` to map specification sections to implementation modules.
- Replaced the placeholder `test` file with `tests/README.md` outlining the testing roadmap.

### Updated
- Extended `docs/DOCUMENTATION.md` with the architecture overview and status of implementation guides.
- Refreshed `fileandwiremap.md` to reflect new directories, configs, and testing structure.

### Notes
- Future updates MUST keep the specification, documentation hub, runbook, file-and-wire map, and changelog in sync per the runbook.

## 2025-10-15

### Added
- Introduced a Python package (`pyproject.toml`) exposing strongly typed configuration loaders under `oneiros.configuration`.
- Implemented an operational CLI (`oneiros/ops/cli.py`) with human/JSON/YAML renders and inline override support, wired to the `oneiros` console script.
- Added pytest suites covering configuration parsing, override handling, and CLI flows.

### Updated
- Documented the new tooling in `docs/architecture/overview.md`, `fileandwiremap.md`, `oneiros/ops/README.md`, and `tests/README.md`.

## 2025-10-16

### Added
- Registered built-in execution profiles (standard, continuations, distributed) with metadata (defaults, cycle budgets, working-set estimates, concurrency hints) under `oneiros.configuration` and exposed helpers to materialise configs from them.
- Implemented `oneiros/core/planner.py` to derive execution plans (segments, concurrency, memory footprint) from typed configs.
- Extended the CLI to list profiles, load configs from the registry, and emit execution plans with overrideable workloads, plus supporting parsers for workload sizing.
- Added pytest coverage for profile metadata helpers, execution planner logic, and the expanded CLI flows.

### Updated
- Refreshed `oneiros/ops/README.md`, `docs/architecture/overview.md`, and `fileandwiremap.md` to document the profile registry, planner, and new CLI workflows.
