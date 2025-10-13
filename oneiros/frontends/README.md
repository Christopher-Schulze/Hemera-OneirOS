# OneirOS ISA Frontends

The OneirOS frontends translate high-level guest programs into the unified algebraic execution trace expected by the core.

## Submodules

| Path | Role |
| --- | --- |
| `evm/` | Transpiler and chip overrides for Ethereum bytecode workloads. |
| `wasm/` | Native WASM host with deterministic syscalls and memory adapters. |
| `svm/` | Solana-style BPF pipeline feeding RISC-V IR segments. |

Each frontend must guarantee trace parity and respect the `claim_root` invariants mandated by the spec.
