# SVM Frontend (Beta)

The SVM frontend adapts Solana-style BPF programs into RISC-V compatible traces.

## Responsibilities

- BPF loader and verifier matching Solana semantics with deterministic syscalls.
- IR lowering pipeline from BPF to OneirOS micro-ops.
- Account/slot commitment layer respecting sparse memory rules.
- Compatibility harness for Solana programs and aggregator pipelines.

## Milestones

1. Define ABI mapping between Solana accounts and OneirOS memory pages.
2. Implement BPF instruction translation and validation.
3. Prototype aggregator tests for CPI-heavy workloads.
