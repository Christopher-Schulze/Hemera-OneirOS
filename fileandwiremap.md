# Hemera OneirOS File & Wire Map

This map tracks repository topology, key integrations ("wires"), and documentation obligations. Update this file whenever directories, interfaces, or external touchpoints change.

## Top-Level Structure

| Path | Description | Linked Docs |
| --- | --- | --- |
| `docs/` | Centralized documentation hub for Hemera OneirOS. | [`docs/DOCUMENTATION.md`](docs/DOCUMENTATION.md) |
| `docs/zielbild/spec.md` | Authoritative architecture specification (v1.1). | [`docs/zielbild/spec.md`](docs/zielbild/spec.md) |
| `docs/architecture/overview.md` | Implementation mapping for new modules. | [`docs/architecture/overview.md`](docs/architecture/overview.md) |
| `runbook.md` | Mandatory workflow and governance rules. | [`runbook.md`](runbook.md) |
| `fileandwiremap.md` | This map; reflects repository wiring and ownership. | `—` |
| `changelog.md` | Chronological log of all significant updates. | [`changelog.md`](changelog.md) |
| `configs/` | Version-controlled configuration profiles (`oneiros.default.yaml`). | [`configs/oneiros.default.yaml`](configs/oneiros.default.yaml) |
| `pyproject.toml` | Python packaging metadata exposing the `oneiros` CLI. | [`pyproject.toml`](pyproject.toml) |
| `oneiros/` | Implementation scaffolding plus Python configuration tooling (`configuration.py`, profile registry, `core/planner.py`, `ops/cli.py`). | Subdirectory READMEs (`oneiros/README.md`, etc.) |
| `tests/` | Consolidated testing roadmap and pytest-based configuration tests. | [`tests/README.md`](tests/README.md) |

## Integration Wires

* **Docs Synchronization:** Every change touching `docs/`, `runbook.md`, or `fileandwiremap.md` must concurrently update `changelog.md` to maintain traceability.
* **Spec Dependency:** Implementation modules (runtime, frontends, proving backends) must implement behaviors outlined in `docs/zielbild/spec.md`; reference this spec when adding new components.
* **Process Automation (Future):** CI should lint for consistency between this map and actual directory contents; track progress in the changelog.
* **CLI Distribution:** The `oneiros` console script (via `pyproject.toml`) must stay aligned with `configs/` defaults, the in-memory profile registry, and spec values.
* **Execution Planning:** Planner outputs (`oneiros/core/planner.py`) must remain consistent with profile metadata (cycle budgets, working-set estimates) and surface via the CLI.

## Ownership Notes

* Until formal owners are assigned, core maintainers collectively safeguard compliance with the runbook and documentation hub.
* Add explicit owner handles in this section as teams form.

_Last revised: $(date -u '+%Y-%m-%dT%H:%M:%SZ')._
