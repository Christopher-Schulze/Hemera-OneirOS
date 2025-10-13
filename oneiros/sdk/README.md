# OneirOS SDKs

SDKs expose developer tooling for interacting with OneirOS proofs and runtimes.

## Submodules

| Path | Role |
| --- | --- |
| `rust/` | Native bindings for orchestrating guests, provers, and verifiers within Rust ecosystems. |
| `typescript/` | TypeScript/Node bindings for client-side orchestration and verifier integrations. |

Each SDK must surface envelope v3.2 primitives, deterministic configuration loaders, and golden vectors aligned with the spec defaults.
