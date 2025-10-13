# OneirOS Implementation Overview

This overview connects the specification in [`docs/zielbild/spec.md`](../zielbild/spec.md) to the implementation skeleton introduced under `oneiros/`.

## Module Map

| Module | Spec Sections | Responsibilities |
| --- | --- | --- |
| `oneiros/core` | §3 (Cryptographic & Arithmetization), §5 (CPU Chip) | Execution trace generation, sparse memory, per-instruction AIR chips, continuations. |
| `oneiros/frontends` | §2 (Profiles), §4 (Architecture Stack) | ISA adapters (EVM/WASM/SVM) that emit normalized traces into the core. |
| `oneiros/proving` | §3, §4 | Setup-free proving pipelines (Hemera Engine) and compression layers (Hemera Injector). |
| `oneiros/sdk` | §1 Goals (DX & Safety), §4 | Developer tooling exposing envelopes, config loaders, and verification hooks. |
| `oneiros/ops` | §2 Profiles, §4 | Operational tooling for orchestrating runs and distributed proving. Includes the Python CLI wired up via `pyproject.toml` and the execution planner. |

## Configuration Defaults & Profiles

The baseline configuration in [`configs/oneiros.default.yaml`](../../configs/oneiros.default.yaml) encodes the canonical defaults listed in §1.3 of the spec. The Python module maintains an in-memory registry of profile metadata (`standard`, `continuations`, `distributed`) that captures descriptions, default configuration dictionaries, target cycle budgets, working-set estimates, and concurrency hints. Future profiles should extend this registry or provide overrides while preserving `claim_root` parity.

## Configuration Tooling

The Python package declared in [`pyproject.toml`](../../pyproject.toml) exposes `oneiros.configuration` for loading and validating YAML documents as strongly typed dataclasses. The module also ships helpers to build configs from the built-in profile registry. [`oneiros/core/planner.py`](../../oneiros/core/planner.py) consumes these dataclasses to emit execution plans (segments, concurrency, memory footprint). The `oneiros` console script (backed by [`oneiros/ops/cli.py`](../../oneiros/ops/cli.py)) surfaces these helpers for operational use, including listing profiles and producing plan summaries in human/JSON/YAML formats.

## Next Actions

1. Establish Rust workspace layout for `core`, `frontends`, and `sdk` crates.
2. Define integration tests that traverse the full pipeline (frontend → core → proving → injector).
3. Extend the Python CLI to trigger end-to-end runs once core execution artifacts are available (leveraging the execution planner for scheduling hints).
