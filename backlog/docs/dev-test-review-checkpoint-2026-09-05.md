# Dev test review checkpoint — 2026-09-05

This is an **in-progress review**, not a green full-suite or merge-ready claim.
The user requested a saved draft PR and another rebase to absorb dev churn.
All work is isolated from the original dirty checkout.

## Checkpoint scope

- Console and Library controller decomposition, ownership/AST ratchets, and
  first-use import repairs, with targeted behavioral and import-closure evidence.
- Runtime repairs include Buddy teardown without a screen, retained Notes focus
  and editor identity during Files handoffs, and three invalid splash effects.
- Test repairs restore current provider/persistence/authority contracts, real
  styled controls, attached-widget readiness, isolated resource measurements,
  and diagnostic privacy fixtures. Security/resource limits were not relaxed.

## Evidence before the new rebase

Independent complete affected-file selections include:

| Selection | Result |
| --- | --- |
| Diagnostic inventory and privacy | 327 passed |
| Buddy and Models adoption | 171 passed |
| MCP gateway tools and prompts | 143 passed |
| Audio.cpp handoff | 122 passed |
| vLLM workflow and Console provider apply | 140 passed |
| Raw CLI processes | 51 passed, 1 Windows-only skip |
| Persona publication | 53 passed; parent-descriptor pressure probe also passed |
| Historical migration, SQLite privacy, workspace roots | 159 passed |
| Console settings | 416 passed |
| Console transcript | 165 passed |
| Console exchanges | 47 passed |
| Console state and generation actions | 83 passed |
| Splash, Cast, Watchlists pagination and rebuilds | 269 passed |
| Scheduler, TTS ownership, Watchlists busy runs | 125 passed |
| Evals and interoperability | 352 passed, 6 existing unfinished-feature skips |

These selections overlap earlier sweeps and must not be added into a unique-test
total. Three staged non-UI sweeps reached 16,203, 12,440, and 5,376 passes before
stopping on distinct failure families. Two staged UI sweeps reached 2,900 and
2,397 passes. Their original failures were retained as a diagnosis ledger, not
silently treated as passing after code changes. The remaining unexecuted cases
and post-rebase integration still need completion.

## Open at the checkpoint

- TASK-31717: final integrated Console/decomposition verification and closeout.
- TASK-31707: oversized trace boundary inputs and cold reserved-call clock setup;
  diagnosis recorded, no implementation yet.
- TASK-31708: agent gateway/gate fixture signatures and regeneration failure
  reporting; diagnosis recorded, no implementation yet.
- TASK-31769: Console journey phase synchronization has six targeted passes;
  its complete-file run was intentionally stopped after 48 passes for rebasing.
- TASK-31770: Files-to-Notes browse scroll restores 6 instead of logical offset 7.
  The interrupted Notes workspace run reached 105 passes and this one failure.
- TASK-31771: thread-start fault injection mutates shared stdlib threading and
  can cause test-runner teardown warnings; diagnosis only.
- Unallocated: Notes Save-failed contrast in the light theme, pinned sync-history
  paging geometry, and load-sensitive Qwen retry/MCP child cleanup failures.
- Re-run the architecture, diagnostic, screen-size, preimport and UI-ready
  ratchets after the new rebase; their pre-rebase results do not qualify new dev.

## Environment qualification

Subprocess tests use an isolated installed review environment so `python -I`
children resolve this checkout, not the original workspace. Writable Notes
fixtures use the per-user macOS temporary directory with correct UID/GID;
`/private/tmp` inherited `wheel` and correctly failed metadata guards. The
workspace tool executor's nested-environment tests were separately qualified
with the native project environment. Platform, configured-service, and
unfinished-feature skips are not proof of executed coverage.

No full-suite completion or merge readiness is asserted by this checkpoint.

## Rebase and draft PR

[Draft PR #2427](https://github.com/rmusser01/tldw_chatbook/pull/2427) preserves
this progress. The review was rebased onto
`da2fbdbc212d16030bb2802a91944527c5db43e7`; a second fetch confirmed that dev tip
before publishing. This incorporated 73 upstream commits since the previous
review base and replayed 109 review commits. The local backup branch
`codex/dev-test-review-before-rebase-20260905` preserves the prior checkpoint.

Conflicts retained upstream last-good Scheduling display and async reachability
checks, alongside unmounted-screen guards and first-use imports. Console timer
tracking/cancellation was retained with the settings-navigation controller.
The diagnostic inventory was rebuilt from the merged owners, preserving the
upstream additions and reviewed controller movements. Review-only Backlog ID
collisions are renumbered; upstream task identities are preserved.

Post-rebase evidence:

- All 195 changed Python files parsed; branch whitespace checks passed.
- Scheduling, Library reuse, import closures, migration and workspace roots:
  241 passed, 2 failed. Both failures exposed the same newly added Console
  suspend caller still targeting a helper moved to the settings-navigation
  controller. Integration plan: retarget that call to the existing owner and
  rerun the complete reuse and settings-return selection. ADR required: no;
  this preserves an existing owner boundary, not a new lifecycle policy.
- Architecture/preimport selection: 44 passed, 3 failed. Console is 17,541
  lines against 16,873; Library is 41,651 against 41,324; preimport adds 504
  modules against 500. The ceilings remain unchanged. Upstream growth needs
  further decomposition/import work before this draft can be merge-ready.
- The suspend caller was retargeted to the existing settings-navigation owner.
  Both complete Console/Library reuse files now pass: 8 tests in 32.25 seconds.
  Full-screen Ruff and changed-line formatting also pass.
- The broader Console reuse/settings-return selection produced 32 passes and
  3 failures. The failures still expect navigation to create a fresh Console or
  cancel an unmount worker; upstream now reuses/suspends the screen. The final
  failure's cancellation-suppressing fixture required interrupting teardown.
  These test journeys need adaptation; their existing handoff assertions have
  not been relaxed for this checkpoint.
- Diagnostic inventory verification reports no drift: 584 owners, 1,336
  TASK-492 calls, 7,615 TASK-494 calls and 11 sink files.
- Final combined rebase qualification: **247 passed** in 89.46 seconds across
  complete Scheduling, Console/Library reuse, reuse-helper, import-closure,
  migration and workspace-root files. This is a targeted selection, not the
  unfinished full review. The three architecture/preimport failures and three
  settings-return failures above remain open.
- Nineteen review-only task-ID collisions were disambiguated without changing
  upstream tasks; all 3,378 task/archive records now have unique identities.

## Resumed repairs after publishing the draft

The review remains in progress. The following complete-file results supersede
the corresponding open items above; they are overlapping selections, not a
unique-test total or full-suite claim.

- Trace settlement: **37 passed**. Tests now independently reach the UTF-8 byte
  ceiling and sanitizer codepoint ceiling; recovery uses the actual reservation
  timestamp without changing the grace period or fail-closed behavior.
- Agent/regeneration/controller: **376 passed** across six complete files.
  A real agent failure now keeps its notice visible beneath the restored original
  answer. Routing/profile fixtures match current admission seams, and branching
  persistence is verified in real SQLite. The run reported descriptor growth;
  test-owned database teardown is being repaired separately, not ignored.
- Branching after review-driven database cleanup: **6 passed**. Teardown awaits
  the controller and closes the exact fixture database's worker connections.
- Console settings: **416 passed**, with runtime warnings treated as errors.
  Thread-start fault injection no longer alters global stdlib threading, and the
  Inspector test includes the upstream Subagents section in its exact ordering.
- Native Console navigation: **349 passed**. Both cached reuse and explicit
  disposal/recreation are exercised, including exact handoff claims and real
  worker cancellation on unmount.
- Chunking Lab: **346 passed** across complete UI, core, DB, service and import
  files. First-use imports restore the unchanged preimport limit: **500 modules**
  and 377,271 source lines. No ceiling or snapshot was loosened.

Remaining work includes Console/Library size decomposition, three Notes
theme/rendering failures, test-owned descriptor cleanup, and the continued
non-UI sweep. The Notes scroll mismatch has not reproduced after rebasing;
its targeted cases pass, while the complete Notes file records 157 passes and
three distinct theme/rendering failures. The resumed sweep also exposed stale
summarization diagnostic-boundary hashes after rebasing and a Library media
selection readiness failure; both remain recorded for investigation. Final
diagnostic reconciliation must follow the reviewed controller moves.

## Further resumed verification and remaining failure ledger

The fourth non-UI continuation stopped at its failure limit with **8,255 passed,
32 failed, 123 skipped** in 1,161.67 seconds. This is a partial continuation,
not a full sweep completion; passing selections overlap and must not be summed.
The XML evidence is `/private/tmp/tldw-review-nonui-remaining-wave4-20260905.xml`.

Further complete-file verification after focused repairs:

- Agent test-owned database cleanup: **47 passed**. Native descriptor probes
  confirmed five send-test descriptors and four branching-test descriptors
  return to zero; no descriptor threshold or garbage-collection guard changed.
- Guarded attachment/exchange cascades and semantic migration guards:
  **94 passed**. Tests use the real semantic mutation coordinator and explicitly
  retain raw-SQL rejection checks.
- Native grammar and accepted skill-hook fixtures: **64 passed**, preserving
  exact execution ownership and hook ordering assertions.
- Historical/current migration checks: **122 passed**. Historical assertions run
  against their exact schema version, followed by current-schema preservation.
- Retrieval extraction and dictionary-send fixtures: **184 passed**. The native
  Console journey file also passed **349 tests**; later hook-binding adjustments
  are covered by the related 184-test selection.
- Library constructor assembly: **46 passed** in the final architecture/import
  selection and **16 passed** in Notes coverage. The broader UI selection recorded
  **276 passed, 8 failed**; all eight failures were reproduced at the pre-assembly
  baseline and remain open, not waived. The Library size ceiling was tightened.

Wave-four failures already repaired above include guarded attachment deletion,
native grammar ownership, accepted skill-hook ordering, and four migration cases.
The remaining observed families are tracked explicitly:

- Three summarization diagnostic-boundary fixtures: TASK-18801; reconcile only
  after final reviewed controller movement.
- Atomic promotion context-policy ownership: TASK-31744; real SQLite probes show
  false save conflicts for staged and inherited policies.
- Inert legacy Notes timer residue: TASK-31746; AST guard and lifecycle coverage.
- Library Skills reserved-name drift: TASK-31748; fifteen missing runtime/command
  names, with the four-source guard retained.
- Two MCP stdio cases: wire-tool inventory and legacy-client startup.
- Three fork-transition census cases: audit the new settings mutation/fence paths
  before changing the classification inventory.
- Two sync-log retention deletes, a conversation-delete property, and one outbox
  deliberate-corruption fixture now encounter semantic mutation authorization.
- Two durable-turn settlement/retry cases need ownership and recovery diagnosis.
- Seven briefing-export cases seed multiple unfinished runs for one watchlist,
  conflicting with the current uniqueness contract.
- One Library media-selection live-evidence case fails its settled-state wait.

Previously observed Notes rendering/contrast, eight Library UI baseline cases,
load-sensitive retry/process-cleanup cases, final Console size/closeout evidence,
and unexecuted remainder selections are still outstanding. This draft is not
merge-ready and no full-suite green result is asserted.

### Subsequent verified repairs

- TASK-31744: **63 passed** for promotion/settings persistence. Public-flow real
  SQLite regressions cover staged post-promotion saves, failed-save retry, and
  inherited fork policies retaining revision ownership. A further 386 behavioral
  tests passed; the three fork-census failures reproduced against the unchanged
  HEAD source. The aggregate run's descriptor-growth warning remains unqualified.
- TASK-31745: **96 passed**. Environment worker forwarding now names its existing
  group/thread/exclusive arguments explicitly; scheduling choices are unchanged.
- TASK-31746: **32 passed** for retired Notes sync and live lifecycle coverage.
  The inert timer field is removed; Library's ceiling tightened to 41,302 lines.
- TASK-31747: **12 passed**, including enforcing error-state contrast checks for
  all 72 shipped themes at both wide and narrow sizes. The minimum measured Save
  failed contrast is 5.070:1; the 4.5:1 requirement and Git error style are unchanged.
- TASK-31752: **126 passed** in the recovery/checkpoint selection, including all
  25 round-one tests. Fault doubles now cross the actual dispatch callback. One
  round-two checkpoint-transition failure reproduced with the unchanged helper;
  it is being investigated as TASK-31754. The aggregate descriptor warning is
  retained, not treated as passing resource evidence.
- TASK-31755: **76 passed**. Export and query fixtures explicitly seed completed
  briefing history; the single-active-run uniqueness constraint is unchanged.
- TASK-18801: the complete summarization privacy file is **257 passed**. Statement
  review proved ten identical logs moved from ChatScreen to retrieval; checked
  and generated inventories now agree at 584 owners, 1,336 TASK-492 calls, 7,615
  TASK-494 calls and 11 sinks. Only the two stale normalized boundary hashes were
  updated. Clean upstream-dev qualification remains pending integration.

The Console private-delegate cleanup, Skills split-reader refresh defect, and
pre-dispatch failure classification are active work. One newly observed Skills
trust journey also reproduced in the pre-assembly baseline. Notes compact paint,
the previously recorded Library baseline cases, MCP/fork-census/guarded-delete
families, and remaining unexecuted review selections are not yet resolved.

## Additional continuation checkpoint

- TASK-31754: **162 passed** across dispatch/recovery selections. A failed
  pre-dispatch callback now retains the accepted owner for retry instead of
  being classified as a completed provider failure. Aggregate descriptor warnings
  remain separately qualified.
- TASK-31748 and TASK-31751: **53 passed** in the complete Skills state/trust
  selection, plus **7** exact browse-controller and **37** Library architecture
  checks. Items refresh independently of the live Work editor. Reserved-name
  coverage retains the fixed four-source guard. Library is 41,301 lines.
- TASK-31756: **67 passed** for MCP stdio and Library tools. Manifest fixtures
  reflect the 21 current tools; the legacy connection fixture explicitly accepts
  the current server-request dispatcher contract.
- Fifth partial non-UI sweep: **2,284 passed, 38 failed, 68 skipped**, stopped at
  its failure limit. Evidence: `/private/tmp/tldw-review-nonui-remaining-wave5-20260905.xml`.
  Some Console receiver failures occurred while the migration was in progress;
  this is neither a final failure count nor a green result. Additional observed
  families include exchange capture/persistence fixtures, private SQLite ownership
  inventory, provider API-key forwarding, estimate-cache ownership, Watchlists
  off-loop evidence, promotion transaction counting, and briefing-script history.
- TASK-31750: **145 passed** in source-size, private-owner, callback,
  diagnostic-inventory and cold-import gates. Actual Console size is **16,818
  physical lines / 505 methods**, with the ratchet tightened to those values.
  AST review found only the approved 64 forwarding removals, receiver changes,
  obsolete imports, and the deliberately late-bound submission callback. Existing
  behavioral assertions are preserved. The question/composer group is **121 passed**.
- The 84-file affected Console run was interrupted before the next rebase after
  **1,749 passed and 7 failed** (986 seconds); its unexecuted remainder is still
  required. Evidence: `/private/tmp/tldw-31750-affected-pass3.xml`. Two failures
  were remaining owner-fixture migrations. Other observed cases cover collapsed
  paste confirmation, empty-panel geometry, staged source details, left-rail
  ownership and a Watchlists handoff click outside the visible region. These are
  not waived or claimed to be baseline without further reproduction.

A fresh fetch found 152 newer dev commits at `4e904f54db`; integration and
post-rebase qualification are pending this checkpoint. No resource/security
guard has been weakened and no complete-suite pass is claimed.

TASK-31753 / TASK-31757: **203 passed** across the complete rewind integration and
Chat settings files plus the corrected readiness receiver regression. Modernizing
the rewind gateway exposed a runtime defect: checkpoint-persisted live messages
have unset cached parent fields. Snapshotting through the store's existing
`durable_parent_for_message` resolver restores the real ordered durable chain.
The end-to-end journey now verifies current branch-memory storage, exact summary
span, separate prepared preamble, transcript preservation, new native IDs on
restart, and no later-memory leakage after restoring before the selection anchor.
The parent-fence fault test mutates the authoritative native tree rather than an
unused cached field. The initial native-versus-persisted-ID hypothesis was corrected
by the exact snapshot probe; no lineage authorization check was relaxed.

The final complete rewind/summary-fence/parent-persistence selection is **78
passed** (31.45 seconds). Its preexisting aggregate descriptor-growth warning
(209) remains open; this is behavioral evidence, not resource-leak closure.

## Second rebase and current handoff

Rebased 144 review commits onto dev `53194eee674865bd8b4aa6daac4b1e7d97160594`,
including 156 new upstream commits since the preceding review base. The pushed
pre-rebase checkpoint and local branch
`codex/dev-test-review-before-second-rebase-20260905` retain the prior history.
Upstream lazy Environment construction, Stop/dispatch draining, trace ownership,
Library ingest ownership, and both sets of testing lessons are preserved.

The initial post-rebase selection stopped after **180 passed / 15 failed**.
Thirteen failures shared missing first-use Notes imports or stale Library class
lookup assumptions; these are repaired. The assembly-order pin now explicitly
includes the new upstream ingest constructor, whose AST matches upstream exactly.
The final complete follow-up selection is **46 passed** (25.94 seconds), covering
Library assembly/ingest, cold import boundaries, rewind integration, dispatch
draining and delayed callbacks. Separately, the first rebased run completed all
**73 summary tests**, **65 delegate guards**, and **25 Environment wiring tests**
successfully. These selections overlap and are not a unique test total.

Remaining current integration debt includes:

- Console is **16,899 lines / 508 methods** against its unchanged **16,818 / 505**
  ratchet; the added upstream work needs a further bounded paydown. The ceiling
  was not raised. Library is **39,818 / 1,295**, and its ceiling was tightened to
  that combined measurement.
- Five unresolved UI cases and the unexecuted part of the 84-file Console census,
  plus the previously recorded non-UI/DB/private-inventory and resource families.
- Diagnostic hash/inventory qualification must be rerun against the new upstream
  Meetings classification and twelfth sink; pre-rebase hashes are historical.
- Two additional Backlog collisions (31714 and 31737) arrived in the last four
  dev commits and await separate approval to renumber the review tasks.

With explicit user approval, these 18 review-created task IDs were renumbered;
the upstream tasks and their identities were left unchanged:

| Former review ID | New review ID |
| --- | --- |
| 31551 | 31758 |
| 31552 | 31759 |
| 31586 | 31760 |
| 31587 | 31761 |
| 31588 | 31762 |
| 31589 | 31763 |
| 31636 | 31764 |
| 31637 | 31765 |
| 31650 | 31766 |
| 31651 | 31767 |
| 31701 | 31768 |
| 31710 | 31769 |
| 31711 | 31770 |
| 31712 | 31771 |
| 31713 | 31772 |
| 31738 | 31773 |
| 31739 | 31774 |
| 31740 | 31775 |

This remains a draft progress checkpoint, not a complete test-suite or merge
qualification. All current production files edited during conflict resolution
parse, undefined-name checks pass, and `git diff --check` is clean.

## Continued bounded repairs after the second rebase

- TASK-31755: the remaining briefing-script scope fixture now creates completed
  history, not competing active runs. Its original isolation assertion and the
  production uniqueness guard are unchanged. Five complete briefing/feed/DB
  files passed **101 tests** (4.39 seconds).
- TASK-31776: cache regressions exercise both real character and bundled-tokenizer
  tiers, with explicit failure on tokenizer fallback. Growing history requires
  exactly 204 computations, not a ceiling that accepted zero. A process-local
  cache-bypass mutation failed all four guards as intended (20,400 computations).
- TASK-31777: corrected the stale sanitized credential-field expectation, exposing
  a genuine missing credential-decision annotation. The runtime now carries only
  a strict boolean into bounded annotation construction; credential fields stay
  absent from stored boundary and handler projections. False/absent/nonboolean
  inputs cannot fabricate a resolved annotation.
- TASK-31753: the owning-turn summary/RAG fixture now uses real conversation and
  workspace persistence, a completed selected turn, and the existing current
  auxiliary gateway. Captured provider/model and RAG configuration assertions
  remain intact, with controller and database cleanup.

The seven complete affected Chat/summary/integration/token/credential files
passed **205 tests** (58.30 seconds). This selection still reports the previously
observed aggregate descriptor-growth warning of 209; it is not resource closure.
Whole changed-file Ruff and changed-function formatting checks pass. The briefing
file has unrelated pre-existing whole-file formatting drift. Independent scoped
review found a tokenizer-fallback gap, which was repaired and re-reviewed with no
remaining findings. All wider failure families above remain open unless explicitly
superseded here. XML evidence: `/private/tmp/tldw-current-chat-repair-final.xml`
and `/private/tmp/tldw-31755-script-scope-final.xml`.

Post-rebase diagnostic qualification is now measured: the rebuild matches the
committed inventory exactly (**589 owners, 1,336 TASK-492 calls, 30 upstream
TASK-31551 calls, 7,615 TASK-494 calls, 12 sinks**). The two complete inventory
and summarization-privacy files produced **322 passed / 5 failed** in 463.74
seconds (`/private/tmp/tldw-rebased-diagnostic-qualification.xml`). Two virtualenv
exclusion fixture dictionaries omit the new `task_31551_calls: 0` summary field.
Three TASK-18801 boundary/mutant controls reject the pre-rebase manifest hash:
actual `caa76e94acdbf3d61961e90bcfe307c21ed5c09bfc061d7c054ba497ff817684`, pinned
`ac5cd5bf7bc9d5f35d80fd71a78953ea96cd6cc60fced84b2e4f60c332bc04f1`.
TASK-18801's whole-file acceptance criterion is reopened for this current tree.
No pins were changed; the upstream owner/sink delta needs governed review before
reconciliation. These five are recorded remaining failures, not regressions
silently excluded from the successful repair selections.

## Diagnostic reconciliation and trace-maintenance owner repair

The preceding five diagnostic failures are now repaired: both complete files
pass **327 tests in 409.28 seconds**. Independent review verified all 584 prior
owner rows unchanged, exactly five upstream Meetings owners and the snapshot
storage sink additions. Checked and freshly rebuilt inventories agree. Only the
two stale boundary hashes and the missing zero summary field changed; all
negative controls remain intact. The upstream Meetings exception diagnostics
retain their classification, not a new metadata-only privacy certification.
See `diagnostic-rebase-reconciliation-2026-09-05.md` and
`/private/tmp/tldw-rebased-diagnostic-repaired.xml`. TASK-18801 remains In Progress
solely for its clean-origin/dev integration criterion.

TASK-31778 registers physical trace maintenance under its actual SQLite owner,
fixing the module-owner guard without widening target kinds or backup authority.
The five complete private-SQLite, core-owner, inventory and compaction/admission
files report **383 passed / 2 failed / 2 Windows-only skips in 57.10 seconds**
(`/private/tmp/tldw-trace-owner-final.xml`). The two failures remain explicit:

- `LegacyCollectionsRecovery._read_transaction` still opens a raw read-only
  connection outside the registered seam. Its source-mode preservation contract
  needs review before migrating it; no blanket exception was added.
- `_QuiescentSQLiteConnection.backup` delegates to `super().backup`, so the
  direct-call inventory detects it. The wrapper reserves quiescence, not a new
  destination; any inventory reconciliation must preserve that behavior and
  reject genuinely new unregistered calls.

Scoped Ruff, changed-range formatting and diff checks pass. The compactor and
architecture test file retain unrelated existing whole-file formatting drift.
Independent bounded review is clear after correcting the new inventory table
row. Dependency/source warnings remain recorded. These are selected behavioral
results, not full-suite, warning-free, resource-closure or merge qualification.
All other previously recorded failure families and the two pending task-ID
renumbering decisions remain open.

## Remaining SQLite census repairs

TASK-31779 and TASK-31780 resolve the two SQLite inventory failures recorded
above. Recovery now uses a module-owned, source-mode-preserving read-only
connection without creating or migrating the database. Real tests exposed and
repair symlink/shared-parent bypasses and a failed-setup connection leak.
Independent review also caught constructor canonicalization hiding initial
aliases; leaf and parent alias regressions failed before that correction, and
the lexical absolute path now reaches the shared no-follow boundary.

The backup census distinguishes the exact existing quiescence wrapper from a
new backup operation, retaining qualified symbol, receiver and multiplicity
checks. Five negative controls reject new or altered call sites. Real backup
tests prove exclusion during callbacks and release after success or failure;
process-local missing-reservation and missing-release mutations each failed
both variants. Production backup logic and authority are unchanged.

Final seven complete recovery/private-SQLite/inventory/core-owner/quiescence/
compaction files: **434 passed / 2 Windows-only skips in 88.89 seconds**,
with two existing dependency warnings. Evidence:
`/private/tmp/tldw-sqlite-recovery-backup-reviewed.xml`. Whole changed-file Ruff
and formatting, diff checks and independent re-review pass. These results
precede any subsequent dev rebase and do not close broader review failures.
