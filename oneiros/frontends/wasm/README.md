# WASM Frontend

The WASM frontend executes deterministic guest modules while enforcing sparse-memory constraints and syscall commitments defined in the spec.

## Responsibilities

- Deterministic WASM runtime with restricted host functions.
- Memory adapter that maps WASM linear memory into OneirOS sparse pages.
- Translation of WASM instructions into the unified IR while preserving stack/memory invariants.
- Support for multi-segment continuations through deterministic checkpoints.

## Milestones

1. Select and configure a Rust-based WASM runtime suitable for no-std deterministic execution.
2. Implement host call shims for `verify_proof`, cryptographic precompiles, and logging commitments.
3. Build compatibility tests with the core memory subsystem.
