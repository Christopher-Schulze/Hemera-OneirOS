# Proving Stack

The proving stack turns OneirOS execution traces into succinct proofs using Hemera Engine and compresses them for on-chain verification via Hemera Injector.

## Submodules

| Path | Role |
| --- | --- |
| `engine/` | Setup-free proving components leveraging HyperNova, Circle-STARK, DeepFold PCS, and Lasso lookups. |
| `injector/` | Oracle-optimized SNARK wrappers (Groth16/Plonky3, CP-SNARKs) and blob binding for EIP-4844. |

The stack must maintain `claim_root` parity with the core, ensuring envelope v3.2 compatibility.
