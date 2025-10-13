# Hemera Injector Integration

This module handles SNARK compression, oracle wrappers, and on-chain verification bindings described in the specification.

## Responsibilities

- Groth16 and Plonky3 prover pipelines with dual-curve support (BLS12-381 primary, BN254 fallback).
- SnarkPack and CP-SNARK aggregation for amortized gas savings (<50k gas target).
- Blob-first commitments for EIP-4844 with KZG binding.
- Envelope v3.2 PCD interfaces ensuring byte-identical `claim_root` with the core.

## Upcoming Tasks

1. Define ABI between Hemera Engine outputs and Injector inputs.
2. Implement blob-binding workflow and reference verifier contracts.
3. Provide SDK hooks for orchestrating compression from Rust/TypeScript clients.
