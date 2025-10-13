# EVM Frontend

The EVM frontend transpiles Ethereum bytecode into the OneirOS unified IR, applying hot-path chips for opcodes that benefit from specialized AIR implementations.

## Responsibilities

- Deterministic bytecode loader and environmental context builder (chain ID, block data commitments).
- Opcode transpiler generating RISC-V compatible micro-ops.
- Specialized chips for `KECCAK256`, `ECRECOVER`, and gas accounting that respect ZK blinding semantics.
- Golden vectors to guarantee `claim_root` parity with the core/injector pipeline.

## Milestones

1. Define transpiler intermediate representation shared with the core.
2. Implement opcode coverage map and identify custom chip requirements.
3. Wire continuations/IVC boundaries for multi-transaction segments.
