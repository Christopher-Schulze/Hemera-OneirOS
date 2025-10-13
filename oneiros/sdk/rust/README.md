# Rust SDK

The Rust SDK provides first-class bindings for executing guests, orchestrating provers, and verifying proofs within Rust applications.

## Planned Components

- Configuration loader for `oneiros` YAML profiles (standard, continuations, distributed).
- Client for spawning core execution segments and collecting trace commitments.
- Prover/orchestrator hooks to Hemera Engine and Injector layers.
- Envelope v3.2 verification APIs for embedded and server-side use.

## TODO

1. Define crate layout (`oneiros-sdk`) and publishable workspace structure.
2. Implement deterministic config parsing shared with CLI tooling.
3. Provide integration tests referencing golden vectors from the core.
