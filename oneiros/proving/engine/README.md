# Hemera Engine Integration

This module implements the setup-free proving backends defined in the specification.

## Responsibilities

- HyperNova/SuperNova folding pipelines with optional Protogalaxy multifolding.
- Circle-STARK configuration (M31 field, DEEP-FRI LDT, DeepFold PCS bindings).
- Lookup orchestration via Lasso (decomposable default) and optional Caulk.
- Deterministic hardware orchestration across CPU/GPU/Metal/Vulkan backends.

## Upcoming Tasks

1. Finalize trace commitment format consumed from `oneiros/core`.
2. Define prover configuration profiles (`standard`, `continuations`, `distributed`).
3. Implement deterministic hardware backend adapters with reproducibility tests.
