# Task 15 independent review

## Spec Compliance

❌ Issues found: accepted uncertainty can lose both its transient denial and durable root pause when both restriction registrations and durable settlement fail. This conflicts with the brief’s charged/paused uncertainty contract; see Important I1. The remaining scoped requirements are satisfied by the source and evidence inspected.

Reviewed fixed BASE `100fa9d8191620645e003b2e9ec5e076b7763ad2` → HEAD `9c0aa539712a2e2659a2b355d531228408932a83`. Requirements: `task-15-brief.md` and verbatim constraints in `task-15-review-requirements.md`.

## Strengths

- Both registration sites now use the narrow type-only helper, and both physical drains precede settlement and release (`tldw_chatbook/Chat/console_chat_start.py:365`, `:445`, `:447`, `:459`, `:576`). The native repeated-cancellation cases retain captured capacity until actual thread exit (`Tests/Chat/test_console_chat_start.py:3475–3498`; `green.xml` and `green-boundaries.xml`, exact commit/provider case IDs in `final-case-union.json`).
- Nested finalization preserves outcome fallback, exact preparation/current-state/active-item fences and captured-token release through the tested publication, abandonment and run-state failures (`console_chat_start.py:577–647`; `test_post_drain_cleanup_keeps_outcome_and_exact_capacity_release`, final replacement receipt `replacement-final.xml`).
- Absence is queried only after failed ordinary settlement, physical drains, and the unaccepted/unreceipted/context-unavailable guard (`console_chat_start.py:541–576`). The private predicate uses the existing serialized FULL transaction, requires a matching present runtime owner, queries the exact attempt, and returns after successful context exit (`tldw_chatbook/DB/automatic_work.py:936–952`, `:116–137`). Real SQLite controls distinguish absent/present/foreign/stale/missing authority and BEGIN/read/commit/restoration failures; no mutation or public abort change is introduced.
- The fork change accurately documents its actual parameter, tuple result and ValueError branches (`tldw_chatbook/Chat/console_chat_fork.py:1154–1188`). Executable AST is preserved.

## Issues

### Critical

None identified.

### Important

**I1 — Preserve denial when registration fails on both attempts and durable settlement fails.** `tldw_chatbook/Chat/console_chat_start.py:365–377`, `:541–576`, `:642–647`.

If `_restrict_chat_start` throws before installing an entry at both calls and `mark_chat_start_review_required` cannot commit, both registration exceptions are swallowed, the attempt remains accepted with its generation charged, the root remains active, and final cleanup removes the active item and releases capacity. A later healthy ledger handle has no uncertainty entry or durable pause to observe. Its admission path checks the transient map and root state, not an accepted attempt’s unresolved cleanup (`tldw_chatbook/DB/automatic_work.py:78–113`, `:190–202`, `:477–495`; durable pause is written only at `:975–989`). Thus siblings can resume automatic work after an outcome requiring review. Successful physical draining does not establish settlement.

This is a source-derived failure path, not a reproduced new test result. Existing evidence does not close it: `Tests/Chat/test_console_chat_start.py:3452–3456` throws on exactly one registration call, so `[commit-first_uncertain]` and `[provider-first_uncertain]` in `green-boundaries.xml` always permit the retry to install a denial. The report’s “Limits and handoff” paragraph acknowledges the gap, but describing it as a retained boundary does not satisfy the accepted-work pause requirement or reduce its severity.

Preserve a store-scoped, same-owner denial through a previously established identity/admission fence when both registrations fail, while still draining and releasing the exact physical owners. Add one focused persistent-registration-plus-settlement-failure control asserting second-handle refusal after cleanup and preserving replacement ownership. Root owns the concrete repair and any necessary scope decision.

### Minor / inherited evidence

No new minor source defect or formatter debt identified. Historical warning concerns remain: app 681/686; ready 1033/1033; preimport 557/557, 415310/425347 LOC and library 127527/135111 LOC; CSS 607326/608090 bytes; Task13 FD 14→383, growth 369 over 200. These are inherited observations from `loading-caps-Close-carry.json`, not fresh Task15 measurements or attributed regressions. Successful Task15 receipts contain no pytest warnings/skips/xfails.

## Evidence checks and review boundary

- Independently hashed all 69 safe-manifest entries and all five final source pins: no mismatch. Reversed the fixed review diff in memory against current source and matched all five recorded BASE hashes; both original test files are exact prefixes, preserving original assertions/waits/markers/imports. No git command or source/index/HEAD mutation was performed.
- Inspected `verifier-execution_map.py.txt` and its reconstruction map. Every reconstructed execution-source hash matches its chronological argv receipt; final coordinator/ledger AST matches GREEN, and final test AST matches the recorded final AST. Later boundary/decorator changes and replacement setup corrections are explicitly distinguished.
- Parsed actual XML and exit receipts: RED 14/19 failures; corrected RED-extra 4/6 failures; GREEN 24 passes; boundary run 6 passes plus one setup failure; existing run 31 passes from 16 selectors; replacement correction failed once, then passed. The chronological passing union is exactly 61 unique cases (30 new, 31 existing), matching each declared winning receipt. Foreign-owner RED initially failed on the immutable-owner setup, then failed correctly on the absent private predicate; replacement failures are retained as setup errors, not implementation RED evidence (`failure-attribution.json`, actual XML).
- Fatal Ruff, five-file formatting and whitespace success receipts are retained; changed-hunk format map records zero inherited debt. Public abort/import/authority/custody maps and Task14/QA/archive/cap carry receipts are preserved. Historical 35 loading results remain historical; no fresh loading claim is made.
- The package cuts `_run` and `console_fork_visible_selection` mid-function, so I read their complete relevant bodies to judge cleanup and documentation. One focused outside-diff check addressed I1: restriction registration/lookup, serialized transaction exit, durable review settlement and `check_active` admission in `DB/automatic_work.py`. Cutoff: those admission/settlement boundaries only; no broader branch/provider/loading review, no new test execution and no passed-cohort replay.
- Publication, current-head Qodo, CI/PerfGuard, latest-dev ancestry and merge remain root-owned external gates; this report does not approve them.

## Assessment

**Task quality: Needs fixes.** The absence proof and physical cleanup repairs are well supported. Accepted uncertainty still needs a denial that survives simultaneous registration and durable-settlement failure before this source can be approved.
