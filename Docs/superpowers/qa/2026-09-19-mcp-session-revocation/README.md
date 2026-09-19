# MCP session-grant revocation — TASK-32865

Revoke now targets the grant and profile captured by its actual mounted button.
Reordered/replaced listings, profile changes and clear invalidate old controls
before removal begins. Each control submits once; a failed revoke rebuilds usable
controls for retry. Completion of an older revoke cannot replace a newer profile's
listing. The existing session lifetime, permission policy and layout are preserved.

Independent branch `codex/mcp-session-revocation-review` starts from dev
`d6e2a46384`, separate from drafts PR2727/2728/2730. No new ADR: this routine UI
ownership repair follows ADR-032 permission boundaries and ADR-150 design language.
The owner approved continuation and the PR is ready for review. Current-head CI
and accumulated review remain merge gates. [Qodo follow-up](qodo-followup/README.md)
records runner hardening and fresh native replay without changing the approved UI.

## Verification

- **26 targeted cases pass:** [10 new mounted checks](tests/green-001.txt) and
  [16 existing service/runtime-gate checks](tests/adjacent-001.txt).
  [Distinct case census](qualified-cases.json). No full-suite run.
- On original code, [seven of eight ownership cases failed](tests/red-002.txt),
  and a [separate delayed completion failed](tests/red-003.txt) by replacing the
  newer profile's listing. The initial red-001 attempt had eight profile-setup
  errors before behavior ran; the existing private-profile decorator corrected
  the harness. That attempt is not regression evidence.
- [All seven derived-artifact guards pass](preflight.txt).
  [Static analysis](static-analysis.json) adds no diagnostics; existing inspector
  and workbench totals remain 13 and 65. New test/runner have zero diagnostics and
  [pass formatting](formatting.txt). No generated CSS changes.
- [Independent review](independent-review.txt) found no concrete blocker or missing
  essential case within the bounded Revoke scope.

## Native verification

[Eight inspected captures](GALLERY.md) show keyboard Revoke at 120×40 and 170×48
in both themes. The actual app, workbench, inspector, service, permission store
and built-in runtime gate run in a disposable profile. Two real in-memory grants
are seeded; revoking calculator removes only that entry and its matrix suffix.
The date/time grant and calculator grant in another profile survive. A fresh gate
check permits calculator before revoke and requires approval afterwards; its saved
Ask policy remains unchanged. No tool is executed and no network attempt occurs.

The runner verifies focused controls have positive geometry, are not occluded and
paint their complete labels. [Lifecycle receipt](lifecycle.json) records normal
exit, process absence, released lock, ten healthy private databases, zero stored
conversations/messages, unchanged default settings, no app errors and nine matching
source hashes plus the runner hash. This qualifies the native built-in grant path;
connected external server execution remains outside this review.

[Allocation census](allocation-census.json), [fresh precommit census](precommit-census.json)
and [sole-owner check](allocation-owner-check.json) protect the task ID. The CLI's
new unstaged 32829 candidate was corrected to 32865 before implementation.
[Export manifest](export-manifest.json) records source/export hashes.

Remaining: exact-input rule removal, Re-allow and other permission actions, then
connected-runtime journeys. The larger component review remains open.
