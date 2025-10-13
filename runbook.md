# Hemera OneirOS Runbook

This runbook codifies the mandatory workflow for contributing to the Hemera OneirOS repository. All collaborators MUST review and comply with these rules before pushing commits or opening pull requests.

## 1. Source of Truth

1. **Specification Alignment:** The architecture and product definition in [`docs/zielbild/spec.md`](docs/zielbild/spec.md) is normative. No change may contradict or diverge from the specification without first updating the spec itself.
2. **Documentation Hub:** [`docs/DOCUMENTATION.md`](docs/DOCUMENTATION.md) is the single-source index for written materials. Ensure every change keeps this index accurate.

## 2. Mandatory Artifacts Per Change

For every meaningful change (code or documentation), update **all** of the following in the same commit or pull request:

- [`docs/zielbild/spec.md`](docs/zielbild/spec.md) **iff** architectural or requirement-level behavior changes.
- [`docs/DOCUMENTATION.md`](docs/DOCUMENTATION.md) to reflect new references or ownership.
- [`fileandwiremap.md`](fileandwiremap.md) with structure/ownership/integration updates.
- [`changelog.md`](changelog.md) with a detailed entry summarizing the change, impacts, and references.

If a file does not require modifications for a specific change, explicitly document the rationale in the pull request description.

## 3. Workflow Checklist

Before merging any branch:

1. **Spec Review:** Confirm the intended work is permitted by `docs/zielbild/spec.md`. If not, amend the spec first.
2. **Implementation/Doc Changes:** Apply updates adhering to repository style guides (see upcoming AGENTS.md when available).
3. **Synchronization Pass:** Reconcile the documentation hub, file-and-wire map, and changelog to ensure they reference the new state.
4. **Validation:** Run appropriate tests/linters (define as the codebase matures) and capture results in the pull request template.
5. **PR Summary:** Reference the latest changelog entry and explicitly state compliance with this runbook.

## 4. Governance and Compliance

- **No orphan docs:** Every newly created document must be indexed from `docs/DOCUMENTATION.md` and mapped in `fileandwiremap.md`.
- **Audit Trail:** Changelog entries must include date, author (if known), scope, and links to related issues or PRs.
- **Exceptions:** Deviations from this runbook require consensus approval from project maintainers and must be recorded in the changelog.

## 5. Continuous Improvement

- Revisit this runbook whenever process friction appears; treat updates as a first-class change with full documentation.
- Future automation (CI) should lint for outdated references and missing changelog entries.

_Last enforced: $(date -u '+%Y-%m-%dT%H:%M:%SZ')._
