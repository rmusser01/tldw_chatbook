# Agent orchestration PR — integration status

PR [#2631](https://github.com/rmusser01/tldw_chatbook/pull/2631) integrates the
reviewed orchestration workstream with current `dev`. The shared checkout and
its unrelated changes were preserved; all integration work uses the isolated
`codex/agent-orchestration-review` worktree.

## Scope and provenance

The original candidate was reconstructed from a frozen working tree at
`b5e32f83a5db38102e38bf0d4ab163a48859ac60`, using historical common base
`2c4c657d015b13b48a5d98712e8767ddcc01003f`. Its 67 unrelated branch commits
were excluded. The preserved scope JSON describes historical commit
`a0f9a90d4b4e2a905764223d666ff2bb52783682`; it is not a live checksum manifest
for the rebased implementation. Original selection and verification evidence
remains under `.superpowers/pr-agent-orchestration-2026-09-11/` in the original
checkout. Integration evidence is in the worktree's
`.superpowers/sdd/2026-09-11-agent-orchestration-pr-integration/` directory.

The integration was reviewed against dev
`d30d8c516cc901b4b017f5214483c56ae10ccda8`; final rebase also includes dev
`d8516accde3a39c30cf41b06709cc359d2a9034d` (Settings, endpoint and registered
Improve My Prompt follow-ups), then the nonbehavioral MCP documentation merge
`eda6e13747ddcf538702e65ff156941f8ee80826`. Current dev's lifecycle custody, worktree confinement,
causal steering, provider routing, activity receipts, and extracted Console
controllers remain authoritative. Historical extraction plans are provenance;
upstream canonical TASK-3070 records are retained.

Included behavior:

- Bounded steering, retained payloads, unread terminal steering and useful
  terminal identity reporting; deterministic owned-client teardown.
- Durable budget accounting and paged history with elapsed time and continuation
  totals. Schema versions 16–18 preserve upstream v13 indexed steps, v14 spawn
  identity, and v15 activity receipts.
- Shared physical child/tool capacity with manual reserves; durable automatic
  chains, acceptance and runtime-owner fences, fair incremental wake delivery,
  and visible budget pauses with saved results.
- Bounded child-progress inboxes, explicit supervisor collection/relay, scoped
  Console inspection and discard, and counts in Chats, workspaces and Character
  navigation. Inspection does not allocate a progress store.

ADR required: no new contract for integration. Existing
ADR-129/130/131/132/134/135/136 and upstream controller/boot/CI contracts apply.
ADR-131/135 document migration version reconciliation. TASK-32493 extends the
existing serial PR fast lane under ADR-103 without changing its dependency
boundary, event cadence, or required gate. Colliding historical task IDs were
reassigned to TASK-32483–32492, preserving unrelated dev tasks.

## Integration review and verification

Independent core review found a cancellation race that could preserve terminal
status while dropping its measured budget. The corrected first-known accounting
path has a deterministic real SQLite regression; 272 affected cases passed.
A fresh core re-review passed 111 cases and found no remaining core findings.
Other integration corrections include complete submission-custody forwarding,
removal of an accidentally duplicated provider call, regeneration error-row
visibility, final survivor painting before timer cancellation, a permanent close
fence for lazy progress-store replacement, wake claims after a stuck parent, and
exclusion of ineligible live child results without pausing eligible siblings.
Fresh whole-branch review found no remaining actionable findings after both
terminal-status corrections; independent selections passed 72, 127, 32 and 20
cases (overlapping counts). All final targeted UI lifecycle cases pass; the mounted harness now supplies the
readiness state that its skipped startup would normally establish.

Verification is targeted; overlapping counts below are not added:

- Agent feature/runtime selection: 431 passed. Upstream service/fleet/provider
  selection: 575 passed. Subsequent accounting, projection and lazy-import
  changes have focused passing reruns recorded in the implementation report.
- Chat bridge, progress, regeneration and recovery: 324 passed, followed by
  23 settlement/progress cases and 49 automatic-work cases.
- DB migration/accounting integration: 80 passed; lazy ledger and automatic
  budget/migration/runtime-owner follow-up: 55 passed. Both runtime and standalone
  upgrades preserve real upstream v15 step, spawn and receipt data across reopen.
- Private SQLite inventory: 37 passed. Private-path and MCP seams: 145 passed.
- Workspace/Character progress navigation: 134 affected cases and four strengthened
  painted tests passed.
- CI target contract: 26 passed initially; the final exact contract plus wake
  attempts passed 32 cases. Eight bounded orchestration modules join the fast lane. The local bounded PR fast lane passed 1,097
  cases; twelve process cases failed during host semaphore allocation before
  application behavior (six task-store and six existing operation-lease cases).
  Clean-runner verification now passes all **1,125 tests**, including every
  process case; the required derived-artifact job also passed on head f66a87d12f.
  Evidence: [CI run](https://github.com/rmusser01/tldw_chatbook/actions/runs/34671103740).
- Derived CSS, diagnostic inventory, private-owner inventory, task IDs, SQL table
  allowlist and index-plan census pass. The index census covers 298 declarations;
  the three automatic-work indexes are pinned against real captured queries.
  Canvas assets reproduce from their hash-pinned public inputs. The complete
  derived-artifact preflight passes after the latest-dev rebase.

Boot import, UI-ready, worker and CSS checks pass at existing limits: 641/660
imported modules, 973/973 UI-ready modules, and 756,326/768,000 CSS bytes.
The whole-registry preimport guard is already over budget on pinned dev; exact
isolated baseline and integrated census both measure 513 modules / 379,059 LOC
with identical module sets. No preimport threshold was changed.

Current dev already exceeds six Console architecture guards. The integrated
Screen measures 24,389 lines / 743 methods, versus dev's 24,390 / 743; no guard
threshold was raised. Historical smaller Screen counts and old diagnostic drift
in the original ledger do not describe this integrated revision.

## Merge verification

Qodo's four review threads are resolved and its follow-up report has zero bugs
and zero rule violations. The published correction passed all 1,125 fast-lane
tests, the required artifact gate and UI latency CI
([run 34671810651](https://github.com/rmusser01/tldw_chatbook/actions/runs/34671810651)).
The final latest-dev reconciliation must pass its own head checks before merge.
TASK-32493 is Done after clean-runner verification.

## Qodo review follow-up

All four Qodo findings on head f66a87d12f have corresponding corrections:

- Historical run databases lacking `wake_delivered_at` are discovered with the
  existing survivor predicate and handed to the normal DB owner for guarded
  upgrade and recovery. Discovery stays read-only, empty/absent DBs stay cheap,
  and chainless legacy results require manual review. The real old-schema
  regression failed before the fix; 23 targeted launch/boot cases pass afterward.
- Rewritten public wake methods document their actual argument, return and
  lifecycle contracts and carry explicit types without eager runtime imports.
- The unchanged 256-result claim bound is named `MAX_RESULTS_PER_ATTEMPT`.

CI also exposed a startup scheduling difference: archive actions could load for
an empty native-session list before `_ui_ready`, raising Linux's count to 974
against the 973 cap. Blank-session refresh now skips that read, and unclaimed
archive resume paths defer their imports. Saved-session reads and late handoff
claims retain their owner fences. The corrected head passed UI latency CI.

The subsequent dev prompt-registration change adds one ready-time import. To
retain the same 973-module cap, execution capacity now initializes on first
execution access. WorkOrigin is shared lightweight model data with a compatible
legacy reexport; bridges obtain the runtime's single locked capacity lazily.
Replacement and disposal prevent new allocation under closed ownership. The
Agent selection passed 191 cases, the bridge/lazy selection passed 299, and
51 focused ownership/admission/automatic cases passed (counts overlap). The
latest-dev ready census passes at 973/973. The final combined import, worker,
CSS, ready and lazy-capacity selection passed 23 tests. Both shutdown entrypoints
retain the existing Canvas/receipt publication lock while serializing the
capacity lifetime latch; 17 targeted lazy/receipt tests pass, including two
regressions that failed before restoring that lock boundary.

## Intentional capability boundaries

Direct peer addressing, progress-triggered wakes, and durable progress inboxes
are outside approved ADR-136 scope. They are optional future design work, not
unfinished acceptance criteria. Shared versioned session tasks support coordination
alongside process-local steering and progress channels. No full repository sweep,
external-provider call, or power-loss certification is claimed.
