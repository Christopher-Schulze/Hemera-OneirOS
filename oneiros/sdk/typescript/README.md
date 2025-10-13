# TypeScript SDK

The TypeScript SDK targets browser and Node runtimes that need to orchestrate OneirOS proofs or verify envelopes client-side.

## Planned Components

- WASM bindings to core execution outputs and verifier primitives.
- RPC client for interacting with Hemera Injector services.
- Utilities for parsing blob-bound proofs and Envelope v3.2 payloads.
- Typed configuration layer mirroring the YAML profiles.

## TODO

1. Define package layout (`@oneiros/sdk`) with build tooling (tsup/rollup).
2. Implement WASM binding pipeline and browser-friendly verifiers.
3. Provide test fixtures that mirror Rust SDK expectations for parity.
