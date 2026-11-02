# PR3335 projection restyle integration plan

> **For agentic workers:** Use superpowers:subagent-driven-development for this single coupled integration, followed by one independent whole-branch review. The existing backlog execution authorization remains in force.

**Goal:** Rebase the projection-restyle PR, strengthen its Textual watcher guard, and qualify its existing partial delivery without inventing a performance result.

**Architecture:** Keep the existing synchronous, batched widget restyles and projection ownership. Compare the entire mirrored Textual watcher body structurally; preserve current composer authority and draft behavior through the rebase. No new runtime abstraction or dependency.

**Tech Stack:** Python >=3.12, installed Textual8.x, existing pytest/private fixtures and Ruff.

**Spec:** TASK36571 (`backlog/tasks/task-36571 - Console-a-runs-config-projection-still-spends-about-50-ms-of-UI-loop-render-per-send-window.md`) and ADR190 (`backlog/decisions/190-console-incremental-display-projections.md`); the prepared repair requirements are in `.superpowers/preparations/pr3335/repair-plan.md`.

**ADR required:** no new ADR. **ADR path:** N/A; existing ADR190 applies. **Reason:** Test/doc repair and mechanical integration of existing display-projection boundaries; no new state ownership, authority, storage, dependency or long-lived UX contract.

## Global constraints

- Selected source3c01869579a56688cfc9a1756e61413cb46865bd, original base4d6764b132edde35c8a32b5d2c92ab189416dd6d; actual integration base42eab3c3c7770d2003688dfe2fa576b328ba2d90. Live selection is OPEN/non-draft/unarmed/zero threads. Branch codex/pr3335-resync in the existing isolated pr3298-resync/tldw_tui worktree.
- Rebase first onto that exact dev base; never merge dev into the branch or copy historical snapshots over current owners. Preserve the newly created plan through rebase and commit it with the integration.
- One source/index/test/import/generator owner. Root owns all remote mutation, exact-lease publication and merge; implementer creates no subagents or additional worktrees. No main-checkout edits, stash, draft promotion/merge, release, or action on3267/3297. Release PR3245/3246 remain outside this session.
- User explicitly authorizes hosted-CI and numeric-size waivers, focused local tests, and requesting-code-review instead of Qodo. Keep correctness/security/source/custody guards and existing numeric caps. Numeric failures are WAIVED, never PASS; source-era ratchet numbers are history, not new caps.
- Keep TASK36571 In Progress and AC2 unchecked. Its 32.6ms target is unchanged. Historical paired timings and AC3 pooling/first-send qualifications stay explicitly historical; functional tests make no new latency or performance claim. TASK36571.1/38440/38441 remain open.
- Preserve attachment blocking, raw CLI authority presentation, sending-draft revision and deferred draft behavior. Stop/Send each retain one intended variant/state restyle batch; Send's disabled watcher/focus behavior remains Textual-owned.
- Follow AGENTS and the relevant architecture, design-language/component-patterns and testing/live/backlog lessons before changes. Apply the current targeted-verification guidance; no full suite, live paid provider, new probe campaign or unrelated cleanup.

## Review focus

1. A future watcher gains a non-class side effect: the entire-body AST guard must reject it rather than silently mirroring only two substrings.
2. Send becomes disabled or Stop availability changes: batching must preserve computed styles and existing focus/disabled semantics.
3. Attachment/raw-CLI/draft ownership changed on dev: the rebase must preserve those controls and their revisions.
4. Status selectors affect descendants or inherited styles: existing loaded/built-CSS guards and full-restyle equivalence remain intact; the unmounted DEFAULT_CSS follow-up stays open.
5. A functional pass is mistaken for the performance target: task notes and PR evidence must keep AC2 and current latency qualification open.

### Task 1: Rebase, repair the watcher guard, and qualify the integrated PR

**Files:** Existing source changes in `tldw_chatbook/Widgets/Console/console_composer_bar.py`, `tldw_chatbook/UI/Workbench/workbench_widgets.py`, and `tldw_chatbook/UI/Console_Modules/agent.py`; repair `Tests/UI/test_console_projection_restyles.py` and TASK36571. Reconcile actual census/ratchet/lesson conflicts; do not use source-era counts. The preparation and full source diff identify the existing owners before editing.

**Interfaces:** `_restyle_button_once` preserves Textual variant validation and batches classes; `_sync_status_classes` remains node-scoped with its existing selector/inheritance guard. `Button.watch_variant` is mirrored only if its entire actual body is exactly remove_class(old variant) followed by add_class(new variant).

- [x] Read task/spec/ADR and prepared source review; inspect actual changed paths and relevant nested guidance. Rebase onto42eab3c3c7770d2003688dfe2fa576b328ba2d90, preserving every current owner and source feature. Explain any dropped commit rather than silently losing it.
- [x] Inspect the installed watcher through the existing private/bootstrap fixture contract. Replace the substring/count pin with AST comparison of its complete body, ignoring locations, against `self.remove_class(f"-{old_variant}")` then `self.add_class(f"-{variant}")`. If actual installed semantics differ, stop that assumption and report the concrete difference to root; do not hide an extra side effect.
- [x] Add the required ADR no/N/A/reason to TASK36571's plan and notes. Keep its existing partial status, open target and historical measurement caveats. Use installed backlog-py CLI for task metadata when applicable; no installation required.
- [x] Run projection-restyle and Workbench widget tests, then select only composer ownership cases affected by the actual merged delta from the preparation's candidate pool. Keep computed-style equivalence, disabled/focus, status selector/inheritance and current attachment/raw-CLI/draft behavior covered. Use one serial private test owner with short semantic IDs and actual readiness; no all-neighbor sweep.
- [x] Validate the canonical census and any derived artifact actually changed by the integration. Preserve the exact dev census prefix and floor, append only genuinely new coverage. Record any numeric exception with unchanged caps. Do not regenerate/retest unaffected artifacts or repeat passing cases without a source change or concrete failure.
- [x] Check changed Python lint/format against the base with inherited debt separate. Run `git diff --check BASE..FINAL` on the entire integrated branch, not merely a clean working-tree diff. Self-review, commit, and prepare concise immutable full-branch review evidence with commands, outcomes, source binding, limitations and exact base/head.
- [x] Release all owned processes and hand the clean head to root for independent spec/quality review. Root publishes only the reviewed head after a fresh live eligibility/head/base/thread check.

## Implementation evidence

Rebased all seven selected commits onto the exact integration base; no commit was dropped. Current composer authority/draft owners survive with exactly one intended Send/Stop batch. Complete watcher AST pin and ADR hygiene applied. Focused behavior:44 distinct cases passed (27 projection/Workbench,17 affected composer owners); after four assertion-only formatter wraps, saved byte snapshots have identical ASTs. No runtime or fixture source changed after these runs. Census1327 retains the exact dev1326 prefix and floor1319. Composer7373 ->7403 with cap7195 unchanged:208 over,178 inherited and30 introduced, explicitly WAIVED. New tests/Workbench lint and format pass; agent/composer inherited lint/format debt remains unchanged in kind. Source-era performance figures remain historical, AC2 unchecked and TASK36571 In Progress. Private receipts and final exact-head package live in .superpowers/sdd/2026-11-01-pr3335-backlog-integration/; publication and independent review remain root-owned.
