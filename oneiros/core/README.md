# OneirOS Core

The core hosts the zkVM execution engine described in [`docs/zielbild/spec.md`](../../docs/zielbild/spec.md). It is responsible for:

- Maintaining the sparse-memory model (`Twist & Shout` pages, copy-on-write semantics).
- Implementing modular per-instruction AIR chips (ALU, BRANCH, MEMORY, MUL, CUSTOM).
- Driving continuations/IVC folding through HyperNova/Protogalaxy pipelines.
- Enforcing deterministic hardware execution paths and ZK blinding defaults.

## Planned Structure

```
core/
  chips/            # AIR implementations and trait definitions
  trace/            # Execution trace builders and normalizers
  memory/           # Sparse page commitments and IO handling
  scheduler/        # Segment orchestration, continuations, and limits
  syscalls/         # Deterministic syscall surface + precompiles
```

## Next Steps

1. Define crate boundaries (`oneiros-core` Rust crate) and shared traits for chips.
2. Scaffold golden test harness with deterministic fixtures.
3. Integrate with proving layer once trace commitments are finalized.
4. Feed execution planner heuristics (`oneiros/core/planner.py`) with real trace metrics to refine the metadata-derived resource estimates for throughput, proof sizes, compression latency, energy consumption, and timeline diagnostics.
