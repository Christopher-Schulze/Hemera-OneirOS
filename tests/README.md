# Testing Strategy

The previous placeholder `test` file has been replaced with this testing roadmap. Future work should establish a structured test harness covering:

- **Golden Vectors:** Ensure `claim_root` parity across frontends, core, proving, and injector layers.
- **Property Tests:** Validate sparse memory invariants, deterministic syscalls, and continuation boundaries.
- **Integration Pipelines:** Execute end-to-end runs for each profile (`standard`, `continuations`, `distributed`).
- **SDK Compliance:** Cross-check Rust and TypeScript SDK behaviors against shared fixtures.

## Current Coverage

- `tests/test_configuration.py` validates parsing, overrides, profile metadata helpers, and guardrails around the canonical YAML profile.
- `tests/test_cli.py` exercises the Python CLI entrypoint, ensuring validation-only runs, serialization modes, profile listings, and execution plan outputs behave as expected.
- `tests/test_planner.py` covers the execution planner, including derived concurrency/memory metrics and validation guards.

Document new test suites here and link to implementation locations as they come online.
