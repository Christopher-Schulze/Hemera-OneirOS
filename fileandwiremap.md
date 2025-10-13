# Hemera OneirOS File & Wire Map

This map tracks repository topology, key integrations ("wires"), and documentation obligations. Update this file whenever directories, interfaces, or external touchpoints change.

## Top-Level Structure

| Path | Description | Linked Docs |
| --- | --- | --- |
| `docs/` | Centralized documentation hub for Hemera OneirOS. | [`docs/DOCUMENTATION.md`](docs/DOCUMENTATION.md) |
| `docs/zielbild/spec.md` | Authoritative architecture specification (v1.1). | [`docs/zielbild/spec.md`](docs/zielbild/spec.md) |
| `runbook.md` | Mandatory workflow and governance rules. | [`runbook.md`](runbook.md) |
| `fileandwiremap.md` | This map; reflects repository wiring and ownership. | `—` |
| `changelog.md` | Chronological log of all significant updates. | [`changelog.md`](changelog.md) |
| `test` | Legacy placeholder file; mark for future cleanup or repurpose. | Document future actions in changelog when modified. |

## Integration Wires

* **Docs Synchronization:** Every change touching `docs/`, `runbook.md`, or `fileandwiremap.md` must concurrently update `changelog.md` to maintain traceability.
* **Spec Dependency:** Implementation modules (runtime, frontends, proving backends) must implement behaviors outlined in `docs/zielbild/spec.md`; reference this spec when adding new components.
* **Process Automation (Future):** CI should lint for consistency between this map and actual directory contents; track progress in the changelog.

## Ownership Notes

* Until formal owners are assigned, core maintainers collectively safeguard compliance with the runbook and documentation hub.
* Add explicit owner handles in this section as teams form.

_Last revised: $(date -u '+%Y-%m-%dT%H:%M:%SZ')._
