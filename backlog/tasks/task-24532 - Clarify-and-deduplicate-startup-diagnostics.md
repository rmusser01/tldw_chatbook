---
id: TASK-24532
title: Clarify and deduplicate startup diagnostics
status: Done
assignee:
  - '@codex'
created_date: '2026-08-29'
updated_date: '2026-08-30 03:58'
labels:
  - diagnostics
  - startup
  - privacy
dependencies: []
references:
  - Docs/superpowers/specs/2026-08-29-startup-diagnostic-clarity-design.md
documentation:
  - Docs/superpowers/plans/2026-08-29-startup-diagnostic-clarity.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make startup diagnostics distinguish optional feature absence, unverified security posture, recoverable cache rejection, and genuine failures without duplicate or sensitive output.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Missing HuggingFace evaluation support is reported once as actionable informational capability copy, while both production dataset-loader paths return the specific typed missing-dependency failure for recognized remote identifiers
- [x] #2 Importing OpenTelemetry support is silent, and repeated or concurrent initialization emits one authoritative unavailable or success outcome with a stable boolean result and no global-provider replacement
- [x] #3 Prometheus initialization emits one authoritative informational-unavailable or successful outcome with a stable boolean result, while server-start failures remain warnings
- [x] #4 The alternate module startup path adds no unconditional metrics success messages and unexpected initializer diagnostics expose only bounded static text plus exception type
- [x] #5 SQLite and runtime-policy unverified-platform diagnostics remain deduplicated warnings that explicitly say verification was unavailable and the named operation continues with an unverified posture
- [x] #6 Model-catalog cache rejection remains a count-only warning that states accepted entries continue loading and discovery may restore missing data
- [x] #7 Changed diagnostics exclude representative credential, path, service-name, cache-content, and exception-message sentinels under focused tests
- [x] #8 Local dataset routing, invalid source behavior, privacy decisions, runtime-policy decisions, cache validation, recovery behavior, and installed-entry-point telemetry behavior remain unchanged
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Correct optional Evals import severity and typed feature-use routing.
2. Make OpenTelemetry initialization silent-on-import, thread-safe, idempotent, and boolean-returning.
3. Make Prometheus authoritative and remove caller overclaims.
4. Clarify existing privacy, runtime-policy, and cache warning copy without policy changes.
5. Run focused secrecy, deduplication, behavior, lint, and compilation checks.
6. Complete task evidence and self-review.

Detailed plan: Docs/superpowers/plans/2026-08-29-startup-diagnostic-clarity.md
ADR required: no
ADR path: N/A
Reason: diagnostic ownership and wording within existing boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Clarified optional evaluation support at import and feature-use boundaries (`7e0855cfd0`, `73dd490f53`); made OpenTelemetry import and initialization silent, stable, retry-safe before publication, set-once-provider aware, and singleton-ownership safe (`e93aee6886`, `f20b7fac4d`, `850ffb150e`, `95a5b93a0c`); centralized Prometheus normal outcomes and bounded alternate-entry failures to exception type (`2c5c86ceac`, `a20ef4197e`); and made SQLite, runtime-policy, and model-cache degraded warnings explicit, secret-free, and temporally truthful without changing their decisions (`a1c17c1a55`, `95fb654c10`). No installed-entry-point telemetry behavior changed.

Fresh integrated diagnostics verification passed 400 tests with 7 expected skips. Ruff check, compileall, cumulative `git diff --check`, sentinel/dynamic-interpolation review, entry-point review, and independent cumulative spec/quality review passed. Ruff format retained exactly the recorded inherited baseline in `Tests/App/test_startup_init_hygiene.py`, `tldw_chatbook/Evals/eval_runner.py`, and `tldw_chatbook/app.py`; the other 13 touched diagnostic files are formatted. The existing Requests dependency warning and macOS pytest cleanup warnings for permission-hardening temporary directories remain environmental. The full repository suite was not run under the targeted-test policy.

Modified the planned Evals, Metrics, app startup, SQLite privacy, runtime-policy, model-cache, and focused test files. Added the lifecycle-fake incident to `backlog/docs/lessons-testing-evidence.md` after independent review proved that a permissive fake masked OpenTelemetry's set-once provider and singleton instrumentor semantics. ADR required: no; the work preserves existing runtime, privacy, persistence, dependency, and entry-point boundaries.

After the final `dev` rebase, reviewed the production diagnostic inventory statement-by-statement before regeneration. The six changed owners contain only the approved severity/copy changes, duplicate-success removal, bounded exception-type diagnostics, and unchanged count-only/runtime-posture fields; no user content, secrets, filesystem paths, or URLs were introduced. Regenerated `Docs/security/production-diagnostic-inventory.json` and verified it reproduces from the source tree.
<!-- SECTION:NOTES:END -->
