# Textual follow-ups: implementation verification

This records pre-publication verification of the four requested audit improvements on `dev` base `773cc9729cfbc9c13c14b56724fcb02efd3a54a3`. The implementation tree was rebased to `81c3216c8ed2fbbe322f6a54a4cd46ac1fc9b1d7`; all 22 owned product/test Git blobs replayed unchanged. Task-level reviews are approved. Whole-branch review, task closure and current-head hosted CI are separate delivery gates; this document does not claim they have already passed.

| Requested task | Result |
| --- | --- |
| [TASK-37142](<../../../backlog/tasks/task-37142 - Preserve-newer-Evals-bench-input-when-queued-Save-completes.md>) (originally TASK-37031) | Queued Evals Save publication preserves newer raw input, dirty state, focus/caret and a pending user selection while refreshing persisted Catalog/Inspector data. Ordinary unchanged Save still reloads canonical persisted values. |
| [TASK-37032](<../../../backlog/tasks/task-37032 - Keep-full-reminder-form-saves-independent-across-submissions.md>) | Independent reminder forms retain their captured payload/ID/service/owner and each admitted write settles. Refused updates do not report success. Full-form offline updates serialize and replay through the existing client/schema path. |
| [TASK-37033](<../../../backlog/tasks/task-37033 - Keep-saved-Prompt-export-writes-off-the-UI-loop.md>) | Picker/Replace callbacks return after native worker submission; filesystem writing runs off-loop with captured Markdown/destination. Late collision/recheck and render failures report sanitized errors and allow retry. |
| [TASK-37034](<../../../backlog/tasks/task-37034 - Scope-Speech-playground-footer-actions-to-the-active-view.md>) | Custom footer hints and native bindings follow the actual live Playground target, including view replacement and screen recomposition. Return restores Clear; mode/global and printable input routing remain functional. |

## Fresh rebased verification

| Gate | Actual result |
| --- | --- |
| All four new mounted modules together | 22 parent passes and 22 actual private-child passes; exact node-set equality; zero failures/errors/skips. These are 22 scenarios, not 44. |
| New export render/retry and collision-recheck/retry controls plus source/import guards | 9 passes, zero failures/errors/skips. |
| CI census and shard-count contracts | 91 passes, zero failures/errors/skips. |
| Modified Python static checks | 23 files checked; no new Ruff findings. The 14 retained findings and two format failures match live-dev baseline source, as qualified below. Whitespace check passes. |
| Textual worker contract | No synchronous worker targets or new post-await DOM lookups/wait-for-dismiss paths. Existing inventory baselines remain separate. |

Fresh unchanged ceilings: boot **680/686 modules**; preimport **558/558 modules**, **424407/425347 LOC**; fattest Library route **130691/135111 LOC**; first-send **93/94 modules**. Evals and Schedules source ratchets were lowered with their smaller implementations to **4211** and **5532** lines. No import, source, timeout or shard allowance increased.

The combined command used the native private-profile streaming runner with these modules:

```text
Tests/UI/test_evals_bench_save_publication.py
Tests/UI/test_schedules_reminder_form_saves.py
Tests/UI/test_library_prompt_export_responsiveness.py
Tests/UI/test_speech_footer_scope.py
```

The runner uses the worktree explicitly, establishes native private TEMP/config roots before pytest import, disables third-party plugin autoload, loads pytest-asyncio/pytest-timeout and retains the existing 180-second per-test limit. Real mounted widgets, picker/forms, SQLite and actual App key dispatch remain beneath controlled I/O/render gates. All releases and relevant native worker settlement occur in finally. No local full-suite run was performed.

Compact evidence: [combined actual child proof](2026-10-27-textual-followups-implementation/root-final-new01-proof.json), [guard/CI results and metrics](2026-10-27-textual-followups-implementation/final-guard-ci-proof.json), [final native source hashes](2026-10-27-textual-followups-implementation/final-native-source-hashes.json), [rebase blob identity](2026-10-27-textual-followups-implementation/rebase-dev773-proof.json). Raw private profiles/XML/logs remain local; the frozen task reports retain exact commands, exclusions and receipt paths. Hosted outcomes belong to the PR's actual current-head checks.

## CI coverage and capacity

The final census/floor is **1076**, up from current dev's **965**, with the same **17 shards** and **1200-second** limits. All dev cases remain covered. The existing 19-case Settings fallback file is replaced by all 19 exact selectors: its zero-cost case remains at the original slot and the other 18 join the tail. Every other dev selector retains its original position. Actual collect-only output equals the exact 19-case split; collection is not represented as a runtime pass.

The task work adds 93 selectors representing 95 currently qualified regression/compatibility cases. Two functions with computed set/dict parametrization are gated as whole functions, preserving all four current cases and future members without weakening static resolution. New helpers are also exercised by the existing PR Fast Lane's whole-file target.

A conservative ordering model takes the per-shard maximum from qualified runs 40540997790 and 40573300731, removes the measured whole-file block, then assigns observed case costs to available tail slots. It estimates a maximum **1122.10 seconds**. This is a capacity forecast, not current-head hosted proof. The earlier two-entry whole-file swap was superseded by the complete case split when newer timings showed it would overload a different shard.

[Capacity/order proof](2026-10-27-textual-followups-implementation/final-census-capacity-proof.json), [collected-case equality](2026-10-27-textual-followups-implementation/root-settings-split-collection-proof.json), [computed selector grouping](2026-10-27-textual-followups-implementation/computed-param-selector-proof.json).

## Review and scope

Independent Superpowers requesting-code-review gates approved each task. Task3 initially had one Important finding: a late no-clobber collision's new metadata recheck could escape the worker and terminate the App. Its real atomic-collision/PermissionError RED and ten-case GREEN established safe feedback, App survival, precious-file preservation and explicit retry; scoped rereview marked it addressed with no new findings. Qodo was not invoked.

Frozen reviews/reports are in [the evidence folder](2026-10-27-textual-followups-implementation/evidence-manifest.json). Task1's original review artifact was not retained; its approval/doc-minor fix is recorded by the controller and the final whole-branch review independently covers the resulting implementation. No missing artifact is cited as portable evidence.

Existing ADRs govern: 011 (typed widget messages), 018 (local/server scheduling), 031 (active-context key/footer truth), 033/036 (session/service ownership), 150/161 (design tokens/components). No new schema, runtime/provider boundary, durability/retry policy, dependency, global state framework, stylesheet or visual token was introduced. The small Speech view-content notification and native recompose wrapper apply the existing UI ownership contract.

## Explicit limits and inherited debt

- Two old Evals picker/initial-target tests fail on original and candidate. Three other broad-run failures pass exact original/final controls; broad legacy module clearance is not claimed. Those three uncertainty controls are enrolled for actual hosted qualification.
- Five unchanged visual guards retain native Windows slash assumptions. Three worker inventory findings and five broad MRO super-call sites are source-qualified existing debt. The MRO sites are not attributed to Windows slash normalization.
- The unchanged eight-second import timing guard failed at 15.11 seconds on candidate and 16.36 seconds on exact Task4-original product files, with no heavy imports. No timing pass or speedup is claimed; module/LOC ceilings above freshly pass. Hosted timing remains a delivery gate.
- Ruff retains seven E402 sites in reminder/service files and seven E721 sites in Speech files, independently matched by offending source lines against current dev. Existing STTS_Window and rail-navigation format failures reproduce on dev. Focused lint passes with those two existing rule classes excluded; this is not clean whole-file lint/format. [Static summary](2026-10-27-textual-followups-implementation/static-summary.json), [baseline line proof](2026-10-27-textual-followups-implementation/static-ruff-baseline-proof.json).
- Pydantic, pytest-asyncio, audioop and optional-dependency/test-factory notices remain disclosed. Output is not pristine.
- Offline queue null/partial fields are preserved, but unchanged ServerNotificationsService omits None at transport. Remote null clearing, live-server integration, same-reminder ordering, navigation-durable writes, physical rollback and retry-policy changes are unclaimed.
- Prompt responsiveness proves actual App acknowledgement before physical write completion with its existing three-second watchdog. It does not establish a half-second latency SLA; the independent half-second release-window failure and diagnostic duplicate-key observation remain documented, not promoted to passing receipts.
- Speech qualification is no-audio mounted UI. Current-view/live-axis persistence beyond existing screen-construction semantics and full app/server/audio-device operation remain unclaimed. Existing readiness observers were synchronized with real native workers; the stale 4/4 capability expectation became fixture-correct 4/5 with explicit missing-OmniVoice evidence, without changing the product model.

## Task-ID provenance and executor decisions

A fresh 365-ref/registered-worktree check found no different-filename owners of the four current IDs. Evals originally used TASK-37031 (created 18:00); an earlier unrelated Console task (16:12) owns that ID. Evals therefore becomes TASK-37142 with original creation time/scope preserved. Historical original audit refs remain as explicit renumbered provenance.

[Rulings I made, in chronological order with costs if wrong](2026-10-27-textual-followups-implementation/executor-rulings.md). Superseding rulings are explicit. No decision depends on deleting local scratch or on an unreported waiver of current-head CI.
