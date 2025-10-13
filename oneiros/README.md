# Hemera OneirOS Monorepo Skeleton

This directory contains the implementation scaffolding for Hemera OneirOS, aligning with the normative specification in [`docs/zielbild/spec.md`](../docs/zielbild/spec.md).

## Directory Layout

| Path | Purpose |
| --- | --- |
| `core/` | Execution engine, chips, memory, and scheduling logic for the OneirOS zkVM core. |
| `frontends/` | ISA adapters and transpilers (EVM, WASM, SVM) that feed unified traces into the core. |
| `proving/` | Setup-free proving stack built on Hemera Engine and compression pipelines through Hemera Injector. |
| `sdk/` | Developer-facing SDKs and tooling for Rust and TypeScript consumers. |
| `ops/` | Operational assets (CLI entrypoints, orchestrators, deployment recipes). |

Refer to the subdirectory READMEs for responsibilities, interfaces, and TODO items.
