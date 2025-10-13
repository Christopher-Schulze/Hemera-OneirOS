# Hemera OneirOS Documentation Hub

This document is the single source of truth for locating and maintaining all written knowledge in the Hemera OneirOS repository. Every documentation contribution MUST reference the normative architecture definition preserved in [`docs/zielbild/spec.md`](zielbild/spec.md) and follow the operational guardrails codified in [`../runbook.md`](../runbook.md).

## Core Artifacts

| Artifact | Purpose | Maintenance Rules |
| --- | --- | --- |
| [`docs/zielbild/spec.md`](zielbild/spec.md) | Canonical specification for Hemera OneirOS (zkVM) v1.1. | Never edit without updating the changelog and runbook checkpoints. Treat as source-of-truth for product and engineering requirements. |
| [`../runbook.md`](../runbook.md) | Workflow rules and compliance checklist. | Review before each change; update when process expectations evolve. |
| [`../fileandwiremap.md`](../fileandwiremap.md) | Repository structure, ownership, and integration map. | Keep aligned with actual file layout and integration surfaces. |
| [`../changelog.md`](../changelog.md) | Chronological log of meaningful updates. | Append an entry for every merged change affecting docs, specs, or implementation. |

## Contribution Expectations

1. **Spec-First:** Validate that proposed work complies with the requirements in the specification before editing code or documentation.
2. **Update All Coupled Docs:** Whenever a change affects architecture, workflows, or repository wiring, update the runbook, file-and-wire map, and changelog in the same commit.
3. **Traceability:** Reference relevant changelog entries when raising pull requests to ensure auditability.
4. **Review Discipline:** Pull request reviewers MUST confirm that documentation has been synchronized according to this document before approving.

## Future Enhancements

* Populate implementation-specific guides (runtime, frontends, proving backends) linked from this hub.
* Automate validation that documentation references remain up to date (pre-commit or CI).
* Extend the file-and-wire map with component owners and service-level objectives.

_Last updated: $(date -u '+%Y-%m-%dT%H:%M:%SZ')._
