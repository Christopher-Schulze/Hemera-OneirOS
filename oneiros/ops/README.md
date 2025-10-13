# Operations & Tooling

Operational assets live here: CLI entrypoints, orchestration scripts, and deployment recipes for distributed proving.

## Implemented Components

- `cli.py` – Operational command line interface that validates OneirOS YAML configurations, renders them as human/JSON/YAML summaries, lists built-in execution profiles, and derives execution plans (segments, concurrency, memory footprint) with optional workload overrides. It supports inline overrides via `--set section.field=value` and backs the `oneiros` console script (see below).

Invoke the CLI directly with:

```bash
python -m oneiros.ops.cli --config configs/oneiros.default.yaml --format human
```

or via the package entrypoint (after `pip install -e .`):

```bash
oneiros --set oneiros.profile=standard --format json
```

List available execution profiles and inspect their derived resource plan:

```bash
oneiros --list-profiles
oneiros --profile distributed --plan --workload-cycles 25000000 --working-set 128MiB --format human
```

## Upcoming Work

1. Extend the CLI with execution orchestration (guest packaging, artifact outputs, distributed job submission).
2. Capture orchestration requirements for multi-prover pipelines (Hub integration).
3. Document deployment prerequisites tied to deterministic hardware backends.
