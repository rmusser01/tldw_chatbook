# PR 2427 rebase reconciliation — 2026-09-06

Tracked by TASK-31932. This is an in-progress integration record, not a merge-readiness claim.

## Git state

The 176 review commits were replayed onto dev `c4d45c0926580a8756cfa13c5463b1d0fc808c1a`.
Rebased checkpoint: `7bd5b9f4a38bd988db900b6dc5faa885bb56e5e2`.
The original pushed head `0135bc20190bba55fc2b48d3d5863ffe1ea449dc`
is retained on `codex/dev-test-review-backup-0135bc2019`.
The original dirty user checkout and every worktree are preserved.

Conflict resolutions retain upstream Prompts state/controller ownership, Media
path redaction/recovery callouts and speaker-rename cache, named handoff timing,
Canvas terminal transaction contributions, locked promotion publication, and
Buddy listening cleanup. The review's ownership and lifecycle repairs remain.

After committing the reconciliation as `aab86d1b87`, the 177-commit series
rebased cleanly onto the newer dev `c47e0da6002475416240252b2954fcf9761d4aac`,
producing `438f6e9122188c5a92d445715d90ce36cff1b140`. The tree delta from the
preceding checkpoint is exactly dev's six files for Persona Inspector avatar
clearing and boot-worker warning handling. Backlog and diagnostic inventory
checks pass again on this revision. The bounded PR Fast Lane plus those two
newly landed test files completed with 840 passed and eight failures in 331.25s:
all eight are the new boot-worker warning probe's premature Loguru sink lifetime.
Evidence: `/private/tmp/pr2427-latest-dev-fast-lane.xml` and its matching log.

## Review-only task renumbering

The user-approved policy preserves upstream task identities even where a review
task has the earlier creation date. The final machine census found **31**
collisions (earlier prose counts 33/34 were counting errors). Immediately before
allocation, a NUL-delimited prefix scan covered 1,029 refs and 319 worktrees:
maximum 31900, with 31901–31932 unused. The CLI subsequently created TASK-31932.

Only the following review-created records moved. Creation dates, completed work,
literal XML/log paths and historical collision reports are retained. Mixed
documents were edited by classified reference line, not by global ID replacement.
Earlier checkpoint statements that 31714/31737/31758 remained duplicated are
historical and are superseded by this mapping.

| Former review ID | Current ID | Review task |
| --- | --- | --- |
| 31714 | 31901 | Preserve Loguru capture sinks across app mount |
| 31732 | 31902 | Defer Chunking Lab action imports beyond screen preimport |
| 31737 | 31903 | Close agent swap fixture owned runtime and database resources |
| 31741 | 31904 | Give provider grammar adapter fixture its required assistant owner |
| 31742 | 31905 | Align skill acceptance hook regression with published turn ownership |
| 31743 | 31906 | Separate historical migration assertions from current schema upgrades |
| 31744 | 31907 | Reconcile atomic promotion context policy revision ownership |
| 31745 | 31908 | Forward Console Environment worker scheduling arguments explicitly |
| 31746 | 31909 | Remove inert legacy Notes auto sync timer residue |
| 31747 | 31910 | Restore readable File Notes error text across shipped themes |
| 31748 | 31911 | Restore Skills shadow name coverage for current runtime and Console commands |
| 31749 | 31912 | Move pure Console rewind and settings draft policy to their existing owners |
| 31756 | 31913 | Align unified MCP fixtures with current tool and dispatcher contracts |
| 31758 | 31914 | Watchlists failure policy test bypasses the live check coordinator |
| 31796 | 31915 | Verify current Watchlists source creation off loop |
| 31797 | 31916 | Avoid rewriting committed project context after promotion |
| 31798 | 31917 | Fence combined Console settings live publication |
| 31799 | 31918 | Reconcile reviewed fork transition route inventories |
| 31800 | 31919 | Retain fork ownership through display name persistence |
| 31801 | 31920 | Fence Console conversation binding publication |
| 31808 | 31921 | Reconcile detached and delegated fork census routes |
| 31809 | 31922 | Restore terminal exchange flush and temporary chat completion |
| 31812 | 31923 | Close fixture owned rewind database and controller resources |
| 31813 | 31924 | Separate connection setup from intentional Qwen retry read timeouts |
| 31815 | 31925 | Give real MCP child reap verification a bounded scheduling allowance |
| 31816 | 31926 | Close newly attributed Console controller and hydration fixture handles |
| 31821 | 31927 | Close remaining inventory UI fixture owned database resources |
| 31822 | 31928 | Repair Console Stop clipping after Redirect action was added |
| 31823 | 31929 | Consume character Chat handoffs on cached Console resume |
| 31824 | 31930 | Ignore late screen rebuild notifications after app stack teardown |
| 31825 | 31931 | Classify synthesized leading system rows as rendered system trace provenance |

Upstream TASK-31861's renumbering provenance refers historically to review
TASK-31825, which now resolves to TASK-31931; its own former Canvas ID remains
unchanged. The upstream document is preserved as historical evidence.

## Fresh verification

Initial six-file post-rebase selection: **356 passed, 6 failed**, 2 dependency
warnings, 104.36s. Evidence: `/private/tmp/pr2427-rebase-initial.xml` and matching
log. Failures: Console 17312/16811 line ceiling; Library slack; Media browse
478/371 line ceiling; Conversations controller slack; stale Library assembly
order; Ingest modal presenter inventory. Media behavioral cases passed.

The corrected complete six-file architecture/modal/Media selection has **360
passed, 2 failed**, 2 warnings, 98.69s. Only genuine Console 17312/16811 and Media
browse 478/371 size failures remain. Library and Conversations budgets were
tightened to measured reductions; neither failing ceiling was raised. The
assembly assertion now covers dev's Prompts construction order, and the Ingest
modal edge points to its actual controller. Evidence:
`/private/tmp/pr2427-rebase-inventory-qualified.xml` and matching log.

The complete Canvas, dispatch recovery, roleplay, dictation, and character
navigation selection has **138 passed**, 6 warnings, 135.88s. The first run's
40 missing-html5lib failures were isolated-environment setup: the declared
dependency is now installed in that temporary environment, without dependency
or original-checkout changes. The new dictation regression calls the existing
dictation owner instead of retired private screen delegates. Native attribution
still found seven character-navigation cases retaining fixture-owned SQLite
handles, so passing behavior alone did not qualify resource cleanup. Evidence:
`/private/tmp/pr2427-rebase-behavior-qualified.xml` and matching log.

The complete agent-loop, Prompts state/seam/wiring/characterization, and durable
turn acceptance files have **247 passed**, 3 warnings, 29.28s, with no
`FD_RETAINED` entries. Evidence: `/private/tmp/pr2427-rebase-owner-contracts.xml`
and matching log.

CSS bundle reproduction, profile-owned path census, 113-table allowlist and
281-index decision census pass. Backlog Guard passes across 3,538 records after
the 31 renumbers and new reconciliation task. The diagnostic manifest was
regenerated only after reviewing 49 Console and four Library statements moving
to their existing controller owners: 48 Console and all four Library statements
match exactly; the remaining watchdog warning retains its copy and level with
the owner-injected timeout argument. No diagnostic statements were added to
either screen and sink topology is unchanged. The refreshed inventory verifies
596 owners and 12 sink files.

Duplicate constructor imports have been removed; undefined/redefined-name
checks across affected runtime owners and scoped full lint checks pass. Existing
canonical Library screen re-export imports remain intact. No full-repository
sweep was requested or run.

The new character-navigation file now opts into the same exact-owner real-app
fixture adapter as adjacent reuse tests. Its three constructors use the existing
builder; no behavior assertions or shared fixture internals changed. The complete
file and shared cleanup fault controls have **20 passed**, 3 warnings, 51.53s,
with zero `FD_RETAINED` entries.
Evidence: `/private/tmp/pr2427-rebase-character-resources.xml` and matching log.

## Integration gates

Do not merge until required checks pass and Qodo has reviewed the final revision.
Absence of Qodo comments and the draft-skipped CodeRabbit check are not review
approval. A thread heartbeat watches PR 2427; pause it after confirmed normal merge.

The Console Canvas/citation ownership cleanup still awaits the user's design
approval. Media browse remains 107 lines over its unchanged ceiling; its recovery
and independent page/facet fences must be preserved in any separately reviewed
ownership reduction. Publication is a progress/review checkpoint, not permission
to merge with these failures.

## First published review follow-up

Checkpoint `71389e02b13644654fe1a131d04c1a7027b2ed6e` was published with the exact
lease on the former PR head, and PR 2427 was opened for normal review. Qodo
posted five findings on that revision. CodeRabbit's success status is a skipped
review because the base is not the default branch; it is not an approval.

The new boot-worker capture repeats the already repaired TASK-31901 lifecycle
bug: app startup removes the sink installed before mount. It now captures only
inside the mounted observation window, preserving all three worker state probes
and the positive unknown-worker control. Worker factory imports are hoisted to
module scope so existing exact-app cleanup adapters can capture their products.
The unmounted smoke app remains a real TldwCli constructor and additionally
registers its exact prompts/media handles for current-thread close. The complete
worker/smoke files plus shared fault controls pass **42 tests**, 3 warnings,
33.96s (`/private/tmp/pr2427-worker-smoke-resources.xml` and matching log).

Resource qualification is **not complete**: the same log still attributes
workspace/collections handles to 12 worker cases and prompts/media handles to
smoke initialization. Existing callbacks removed the other auxiliaries. The DB
APIs close only the caller thread's connection; app initialization opens the
smoke databases in its own thread pool, and mounted workers also acquire
thread-local connections. No global closure, GC workaround, threshold relaxation,
or new cross-thread lifecycle implementation was added. A separately reviewed
owner-lifecycle solution is still required.

Qodo triage:

- `3944734941`: modal-transfer diagnostic context — open, needs privacy-safe
  failure-path design and regression verification.
- `3944734944`: settings-durability diagnostic context — open; raw exception text,
  credentials, drafts, and user-entered provider/model labels must not be added
  merely to satisfy a logging recommendation.
- `3944734946`: submission import ordering — corrected to stdlib, third-party,
  then local groups without altering imported symbols.
- `3944734948`: app accessor documentation — added its borrowed-instance return
  contract; runtime access remains unchanged.
- `3944734949`: alleged historical disclosure callback slot — the suggested
  reversal would restore the regression. Commit `8e1d9c72b6` introduced the
  disclosure callback before the established eighth continuation slot on Aug 30;
  its parent has `call_model_with_continuation` immediately after `clock`.
  TASK-31765 restores that older contract while retaining keyword disclosure.
  Do not silently undo that repair on the basis of the current-base diff alone.

The complete first-review checkpoint selection (incremental agent persistence,
tool disclosure, settings durability/navigation, dispatch recovery, worker events,
smoke, and cleanup fault controls) has **152 passed**, 3 warnings, 47.91s.
Evidence: `/private/tmp/pr2427-first-qodo-checkpoint.xml` and matching log.
It verifies the positional continuation and keyword disclosure contracts and
the non-behavioral Qodo corrections. Its native probe reports 42 retained-path
cases: 29 incremental agent-persistence cases, 12 worker cases, and smoke
initialization. These remain resource findings, not a resource-clean
qualification. Agent-persistence ownership still needs targeted attribution;
the worker/smoke cross-thread findings above have been diagnosed.

## September 7 recovery and approved review repairs

The former temporary worktree was removed outside this task. The published
branch was recovered into `.worktrees/pr2427-review-recovery`; the original
dirty checkout was not used for source edits. The 179-commit rebase onto fetched
dev `3090013cfea4dbf6133ac43d024656e2eb3a2a56` completed locally at
`4a74c5d7e02552a5351d59df7647ea8811526bab`. Published head remains
`d926e3a98021431ca6cf9f4b27d24c9adf3ed0d3` until the checkpoint is pushed.
Dev subsequently advanced to `37bf45fb6232a1d4fb50fdba8f3c19c856ae7664`;
that second reconciliation is pending, not covered by the evidence below.

After a fresh refs/worktree task census, only the three review-created collisions
were moved. Upstream identities and original dates/history remain intact:

| Historical review ID | Current review ID | Subject |
| --- | --- | --- |
| TASK-31759 (intermediate TASK-32013) | TASK-32040 | Assistant-turn production stylesheet harness |
| TASK-31901 | TASK-32014 | Loguru capture sink lifetime |
| TASK-31902 | TASK-32015 | Deferred Chunking Lab imports |

Historical IDs and evidence paths above remain incident references, not current
task pointers. The task identity check passes all 3,589 task records.

The user approved the private Canvas/citation move into the existing message
controller and safe diagnostic context at Qodo's five settings boundaries.
Both changes passed independent spec and quality reviews. ChatScreen decreased
from 18,074 lines / 531 methods to 17,521 / 520; its unchanged limits still fail
by 710 lines / 15 methods. The four complete wiring/message/citation/compiler
files pass 159 tests. Callback identity required a correction before those passes;
the incident is recorded in `lessons-testing-evidence.md`.

Settings diagnostics record a fixed operation/phase, exception type, validated
canonical v4 UUID session/submission IDs, and an exact nonnegative integer
generation. They do not attach exception text or tracebacks. Four fault paths
plus invalid-ID controls pass all 12 cases; the prior three-complete-file settings
selection passed 53 cases before three additional invalid-UUID controls were
added. The real modal-transfer failure control also passes. Qodo replies on a
published correcting SHA are still pending.

Incremental agent-step persistence now yields/closes its owned DB and supplies
the real RunLogWriter with the test-owned directory. Its complete file passes
34 cases, with zero retained SQLite descriptors under the native F_GETPATH
observer (`/private/tmp/pr2427-agent-final-resource.jsonl`, matching XML).
Worker/smoke ownership is separately being finalized and reviewed; temporary
prototype success alone is not its completion evidence.

The final worker/smoke implementation subsequently passed independent spec and
quality review. Four complete files pass **52 tests**, with zero retained SQLite
descriptors in all 52 under the native observer:
`/private/tmp/pr2427-thread-owner-implementation.HOSRfM/final.log` and `final.jsonl`.
The opt-in executor closes exact new/replacement handles on their creating
thread, preserves borrowed connection identity, and drains actual concurrent
futures before database teardown. Smoke captures only the returned app's exact
constructor-thread Prompts/Media connections. Ten new controls cover failure,
cancellation, timeout, borrowed transactions, replacement handles, and ordering.
Scoped Ruff and whitespace checks pass. This is test-only ownership repair.

Reconciliation found two stale production provider-selection calls in inactive
conversation token warmup, plus one undefined native-test factory alias. The
calls now use the existing owner on both sides of the await. Real-store/worker
controls verify immutable inputs and payload/provider replacement fences.
Receiver-specific moved-seam guard checks and explicit fixture/controller calls
remove nine false positives without exempting wrong Console calls. Five complete
affected files pass 53 cases (`/private/tmp/pr2427-seam-green.log`).

The saved-Chatbook browser roundtrip passes after that production fix. The other
two served-browser failures were an omitted runtime profile in a fixture source
response; a diagnostic correction passed both without changing assertions. The
repository fixture now uses the supported profile constant. Complete native and
browser verification is still running; do not substitute these focused passes
for the final file results.

The complete native-chat file finished **349 passed / 2 failed**. Both remaining
fixtures constructed inactive sessions with no settings, unlike the production
new-session path; token warmup correctly refused them. Supplying existing
default settings makes both focused cases pass with unchanged workspace/private
scratch assertions. The complete two-file browser selection finished **64 passed
/ 1 failed**: all served cases pass, while a native bridge-confirmation dialog
timed out and remains under investigation. Complete-file qualification must be
rerun after the newest dev reconciliation.

All six derived preflight checks pass after reviewing the exact diagnostic
statement changes: the two moved statements are unchanged, and five settings
sites delegate to one privacy-filtered helper. The manifest now has 599 owners,
7,670 TASK-494 calls, and 12 sink files. Controller receiving pins were reconciled
to the exact sanctioned extraction replay (analysis 873, Reader 774); screen
limits and the genuinely overgrown browse-controller pin remain unchanged.

Open merge blockers include Console/Library screen size, Media browse-controller
size, seven added broad CSS rules (281 versus 274 on fetched dev), final complete
verification, the newly advanced dev tip, and published final-head review/checks.
Additional Console/Library owner moves have been proposed for user approval;
there is no normal-merge qualification yet.

Environment disclosure: verification uses the recovery worktree's ignored
isolated Python 3.12 / Textual 8.2.8 environment with its own editable install and
a shared-dependency path. During initial browser setup, an agent also installed
the already-declared `html5lib==1.1` into the original checkout's `.venv`; no other
package changed. Original source files were not modified, but that environment
was changed and must not be described as untouched.

## Latest-dev checkpoint published and transport regression qualified

The second 180-commit rebase completed at
`6a65c5b3b26a91d025a6236d5159762180900dab`, containing fetched dev
`37bf45fb6232a1d4fb50fdba8f3c19c856ae7664`. That checkpoint was published to
PR 2427 using an exact lease on the former `d926e3a980` head. The two open Qodo
diagnostic comments now have replies describing the published fixing SHA,
privacy boundary, and bounded evidence. Final-head review and normal merge
remain required; this is not a merge qualification.

Latest dev also removed the previously reviewed Save-as-Note owner fix and its
test. Restoring a stronger control reproduced a save under a deliberately
different `current_user`, instead of configured `notes_user_id`. The one-line
restoration matches the existing Library owner and received independent review.
The complete message/settings-diagnostic files pass 33 cases. The other removed
upstream note-settings feature and historical task files were not restored as a
side effect of this routine ownership repair.

The intermittent Canvas bridge failure was a real transport bug, not a browser
deadline: a request advertised 16,487 bytes, but the single `StreamReader.read`
returned its first 16,384 bytes before EOF. JSON parsing failed; a canceled
confirmation remained pending and blocked the next request. The reader now
uses `readexactly(existing_limit + 1)` with `IncompleteReadError.partial`, then
applies the unchanged size/UTF-8/JSON refusal checks. No limit, authentication,
confirmation, or cancellation policy changed (existing ADR-121).

Ten deterministic real-stream controls initially had six expected failures;
the complete gateway/control files now pass **83 tests**. Both complete Canvas
browser files pass **65 tests**, no failures or skips, including exact draft
confirmation, passive download, child roundtrip, and browser-profile isolation.
Evidence: `/private/tmp/pr2427-json-final.log` and
`/private/tmp/console-browser-gateway-repaired.log`. Independent spec and
correctness/security reviews found no issues.

New upstream appearance/workspace verification initially reported five failures
in the controller unit harness. Fixture-local canonical provider snapshots and
actual session settings retain real token preparation and its fences; missing
constructor argument documentation was added. The complete controller file
passes 115 cases, and all six appearance/workspace files pass **254 tests**:
`/private/tmp/pr2427-workspace-fixture-reconcile.A6pcbO/group-final.log`.

The latest four-file native/message/settings/token run finishes **383 passed /
4 failed**. All four failures are native rail text expectations after upstream
appearance controls changed label widths/wrapping; they are under investigation,
not counted as passing from focused reruns. Evidence:
`/private/tmp/pr2427-latest-native.log`. The structural size and broad-CSS gates
also remain open, with additional owner moves and selector changes awaiting
approval.

All six latest-dev derived preflight checks pass. Rebase reconciliation changed
only the manifest summary's owner count from 599 to the measured 600; its owner
rows and sink topology already matched the current source. The Backlog identity
census now passes 3,591 records. Evidence:
`/private/tmp/pr2427-latest-preflight-final.log`.

## Current Qodo review and remaining qualification gates

Checkpoint `2ec344752d268d5ba857a851a448c89af0ab7614` was published after the
rebase. Qodo's requested review completed against that head and added comment
3954688544: two raw checkpoint reads in the dispatch-recovery test should use
the database transaction manager. Both now do, retaining identical queries and
assertions, with each context exiting before further asynchronous work. The
complete recovery file passes **22 tests**; independent review, scoped Ruff
checks/formatting, and diff checks pass. Evidence:
`/private/tmp/pr2427-qodo-transaction.log`.

Qodo also retained the disputed LoopDeps positional finding. Rechecking the
parent of `8e1d9c72b6a8e6b361ea766eadbfeb0e2609e61b` confirms that the eighth
argument was `call_model_with_continuation` before disclosure was introduced.
Both complete agent persistence/disclosure files now pass **38 tests**,
including the real positional continuation control. The thread was resolved
with this evidence, not by reinstating the regressed field order. Evidence:
`/private/tmp/pr2427-qodo-loopdeps.log`, PR reply 3954727441.

The four native label controls pass in isolation after asserting exact row
identity, full state/tooltip text, and the owning tray's actual wrap/truncation
budget. The complete 351-test native file remains running at this checkpoint;
focused evidence is not whole-file qualification. The generic text helpers
were not broadened. Evidence: `/private/tmp/pr2427-label-focused-final.log`.

Fresh architecture census: **47 passed / 3 failed**. Console is 17,534 lines
against 16,811 (+723); Library 37,063 against 36,109 (+954); media-browse
controller 555 against 371 (+184). Evidence:
`/private/tmp/pr2427-size-final-census.log`. Additional ownership moves remain
subject to the pending design approval; no limits were raised.

Latest GitHub Perf Guard run 34191172522 has two real failures: **281 broad
selectors against 274**, and **811,541 startup CSS bytes against 804,000**.
Consolidating eight widget styles in `f99371858f` added 7,655 eager bytes;
other drift subtracts 105, producing the 7,541-byte excess over the cap. A
snapshot refresh alone cannot qualify this. Read-only analysis proposes using
the existing route-owned split mechanism for Watchlists-only rules (11,203
bytes; projected startup 800,338), keeping mixed/shared rules eager. This is
pending user approval, actual build measurement, and navigation/visual tests.
The seven narrow selector replacements also remain pending approval.

Remote dev was checked again and remains
`37bf45fb6232a1d4fb50fdba8f3c19c856ae7664`, already contained in this branch.
Final-head verification/review and normal merge remain open.

## Complete native-label qualification

The complete native-chat file now finishes **351 passed**, no failures or
skips, in 522.77 seconds. Evidence:
`/private/tmp/pr2427-native-label-complete.log`. This supersedes the running
status above and the earlier 383/4 group result for the four repaired native
assertions; it does not imply the independent architecture/CSS gates pass.
Only the four tests and their wrap/truncation imports changed. Full titles
remain checked in normalized state and tooltips, actual labels against the
row's ancestor tray budget, and existing persistence/selection/service/resume
assertions remain intact. Scoped Ruff checks, formatting of changed ranges,
diff checks, and independent review pass. Unrelated formatter changes were
removed rather than expanding this test-only patch.

The ownership-sensitive fixture detail is that the mounted Console contains
multiple context trays. A first-tray query measures a hidden tray's fallback
budget (10), not the displayed row owner's budget (13). The assertions use
`row.query_ancestor(ConsoleWorkspaceContextTray)` to avoid that mismatch.
This validates label generation, not compositor-level painted-text fit: a
read-only probe also noted an older action-control/chrome discrepancy between
a 13-cell label budget and 11-cell content region, requiring separate visual
verification with the pending appearance work.

The Qodo transaction repair was published as
`1eab524d523b093d4efa471c99e12f9f96f6dc21`; reply 3954746315 records the
complete 22-test evidence. Both that thread and the historically inapplicable
LoopDeps thread are resolved. Additional commits still require final-head
review. No merge or gate bypass has been performed.

## Approved owner paydown: Console connection probe

The user approved the existing-owner Console/Library moves and scoped CSS
paydown. Plans were recorded at `72cbe56243`. Console plan Task3 now moves
the bounded connection probe to the existing settings-navigation controller,
removing its superseded constructor argument and screen wiring. The method
is source-identical except for its relative import; the imported endpoint
probe remains lazy. Two private test receivers were retargeted without
changing their request/result assertions, and production modal opening now
checks callback identity for both the original and a post-construction
replacement. No numeric cap or UI contract changed.

Evidence: the original focused test passed; the retargeted test failed for
the absent owner method, then passed after the move. The complete endpoint
probe, UI session settings, Chat session settings, and controller wiring
files pass **750 tests** in 343.78 seconds, including the real loopback model
endpoint. The complete private-delegate architecture file passes **66 tests**.
Independent spec and correctness reviews, changed-range formatting, scoped
Ruff, and diff checks pass. Logs:
`/private/tmp/pr2427-probe-{baseline,owner-red,owner-green,owner-complete,architecture}.log`.
ChatScreen loses 22 lines and one method; this is one completed cluster, not
qualification of the remaining screen-size excess.

All six derived-artifact preflight checks pass on this local checkpoint,
including concurrently regenerated narrow-selector stylesheets. Evidence:
`/private/tmp/pr2427-css-probe-preflight.log`. CSS appearance qualification
and other owner clusters remain open. The latest remote inspection found
dev at `c37d611368b2c1ac0f137db54085de032fb12b89`; a new rebase remains
required after saving qualified work. Published head `c27723b623` still fails
only the two previously identified CSS budget cases in its Perf Guard job
34192417774, while derived-artifact checks pass. No newly posted inline
review comments appeared after 05:55 UTC in that inspection. No merge was
attempted.

## Approved CSS subject-key paydown

Seven rules now use dedicated classes on their existing Button/VerticalScroll
subjects while retaining owner ancestry and type matching. IDs, labels,
variants, nesting and callbacks are unchanged. The builder regenerated only
the two widget-default sheets. The complete fastpath file passes **5 tests**;
the parsed ancestor-scoped bare-type census is **274**, at the unchanged cap
(Button181, VerticalScroll4). Evidence:
`/private/tmp/css-paydown-final-fastpath.log`.

The complete parity file passes **27 tests**, including ten paired compact
and wide cases. Both arms load the full app stylesheet union; the baseline
arm restores only the old six generated selectors and provider callout
DEFAULT_CSS in memory. Computed styles, container/button regions,
normal/disabled/focused visual styles, compositor paint/hit ownership, and
complete painted frames match exactly. Two repeated paired runs pass all ten
cases. Evidence: `/private/tmp/css-paydown-final27.log` and
`/private/tmp/css-paydown-paired-fulltier-stable{1,2}.log`.

Review caught that the first harness omitted app-tier sheets: it incorrectly
treated NewTaskChoice/provider buttons as three rows, whereas app overrides
make them one row. The corrected full-tier paired controls now pin this
incumbent behavior. Both independent reviews and scoped static checks pass.

Complete affected non-Scheduling files pass 289 tests. The Scheduling batch
passes 331/333, exposing an old overflow viewport invalidated by TASK-31712's
intentional padding reduction and its tooltip's internal reminder noun.
Separate reconciliation is underway under TASK-31932 plan15. Widget
consolidation passes 32/33; four DEFAULT_CSS declarations already present at
`c27723b623` remain outside its allowlist (LibraryCharacterRepairDialog,
RoleplayDraftNavigationDialog, RoleplayDraftRecoveryDialog,
ConsoleAppearancePickerModal). No allowlist or cap was relaxed. Watchlists
stylesheet deferral, this consolidation debt, and screen-size gates remain
open; this checkpoint does not make the PR merge-ready.

## Approved owner paydown: message presentation

The active-session presentation/context bodies now live in the existing
ConsoleMessageController. The existing live session/store ports and two
named global-name/transcript-style callables preserve current values without
reaching through the controller's screen handle. Screen appearance refresh
and transcript rendering hooks remain screen-owned. The new regression
fails before extraction and passes afterward while replacing app, session,
and style dependencies after construction and making the controller's
framework screen handle unusable.

Exact private test seams and required constructor fixtures were retargeted.
The first complete group exposed fifteen generation-fixture failures because
its now-used active-session port had been deliberately unwired. Wiring that
port and the two new ports to their actual existing sources restores the
complete 40-test file, without swallowing errors or adding defaults.

Two independent native-transcript assertions also predated TASK-31759's
summary/transcript note actions. The plain transcript harness does not use
ChatScreen presentation; its menu, transcript and action-service production
sources are unchanged by this move. Both failures were reproduced separately.
The exact ordered-label assertion now includes all six actions, and keyboard
coverage explicitly visits all six and wraps in both directions. Captured
message, menu-dismissal and selection checks remain intact.

Final complete seven-file group: **435 passed** in 150.17 seconds. Complete
native Console file: **351 passed** in 503.58 seconds. Independent spec and
correctness reviews, changed-range formatting, scoped Ruff and diff checks
pass. Evidence: `/private/tmp/pr2427-presentation-complete-final.log`,
`/private/tmp/pr2427-presentation-native.log`,
`/private/tmp/pr2427-presentation-owner-{red,green}.log`, and
`/private/tmp/pr2427-more-menu-baseline.log`. The final screen measures
17,489 lines / 517 methods; the unchanged 16,811 / 505 limits still require
the remaining approved durability and handoff owner moves.

## Scheduling fixture and terminology reconciliation

The three complete affected Scheduling files now pass **214 tests** in
130.25 seconds. TASK-31712 intentionally removed five blank rows from each
expanded DetailGroup, so the old 235x52 History-overflow precondition no
longer held. Only that test now uses a genuinely overflowing 235x40 docked
viewport; its no-History-before-scroll and painted-History-after-scroll
assertions are unchanged, and the separate 235x52 lifecycle check remains.
The diagnostic probe measured History outside the viewport at y38 before
scrolling and painted at y18 afterward, with a 20-row scroll range.

The notification tooltip now uses TASK-23106's scheduled-task vocabulary,
preserving the explanation that inbox/toast delivery is fixed while an
automation's notification setting is editable. Its dedicated test checks
both the canonical noun and the absence of a per-task setting. Independent
spec and correctness reviews, scoped Ruff, changed-range formatting, and
diff checks pass. Evidence:
`/private/tmp/pr2427-scheduling-reconcile-final.log`. The intermediate
213/1 result was its old literal `per-reminder` assertion, subsequently
reconciled; no production notification policy changed.

## Watchlists stylesheet deferral and modal consolidation qualification

The exact Watchlists-only partition moves 12,771 source bytes, including six
older `wl`/`wc`/`overview` units whose consumers were confirmed exclusively in
Watchlists. Mixed/shared selectors remain eager. The existing app route map
loads the sheet; no screen CSS_PATH or loading framework was added. Actual
startup cost after this split was **798,906 / 804,000 bytes**. Real first entry,
repeat entry (one parse), initial-route startup, lossless partition and bundle
reproduction controls pass. Removing the route entry in an isolated in-memory
negative control fails the route guard as intended.

The complete CSS integrity/budget/ratchet-message and File Notes group passed
**203 tests** in 521.28 seconds. Its initially failing Save-error expectations
predated TASK-31910's contrast repair; only those two expectations now match
the already-shipped `$ds-text-primary` / `$surface` pair. Git error styling and
disabled opacity checks remain unchanged. Evidence:
`/private/tmp/pr2427-watchlists-css-notes.log`.

The first complete five-file Watchlists group recorded **201 passed / 7 failed**.
Four failures disappeared when the full-app fixtures loaded the route-owned
sheet before their direct screen push. Three new mount-boundary controls went
RED before this fixture correction and pass afterward, including painted and
hit-tested primary controls at 160x45, 235x52 and 80x24. The remaining three
failures reproduce identically using the exact pre-split 875936d66f eager CSS
in memory; they are not split regressions. The stale filter fixture now seeds
the real local database, waits for the genuine reload and checks the exact
canonical fresh-row ID (focused test passes). Two compact-Select focus contrast
failures remain **open**, at 1.8811:1 and 1.5348:1 against the unchanged 2.0:1
floor. They trace to the upstream theme-token change; the old token makes both
unchanged tests pass in a diagnostic arm. A separate visible-cue repair was
requested for approval; neither the global token nor the thresholds were
changed. Complete affected-file requalification remains in progress.

Four independently failing modal DEFAULT_CSS declarations were reconciled via
the existing default-tier consolidation: three effective blocks now use
BUNDLED_CSS with dedicated classes on the same subjects; Recovery's genuinely
inert Navigation alias was removed. Exact original class declarations are
retained as test fixtures. Paired compact/wide arms compare every mounted
widget's computed geometry, full compositor text-and-style output, selected
emoji/swatch, and action-button normal/disabled/focus paint and hit targets.
They include both Clear and Cancel and prove the original arm removes the
migrated generated blocks. Exact standalone modal harnesses retain the same
default tier through ConsolidatedCSSApp.

The consolidation/parity/fastpath/byte group passed **51 tests**; the final
complete four-file modal/navigation group passed **47 tests** in 69.59 seconds.
The combined measured startup cost is **803,081 / 804,000 bytes**, with a
**273 / 274** selector census. The approved snapshot writer refreshed only
the CSS snapshot after the under-cap measurement, without force or a cap
change. Evidence: `/private/tmp/pr2427-modal-green.log`,
`/private/tmp/pr2427-modal-complete-final.log`, and
`/private/tmp/pr2427-css-snapshot.log`. Independent final reviews are pending;
this section is progress evidence, not merge clearance.

Both independent CSS reviews subsequently passed. A documentation-only follow-up
corrected the shared generated split header's obsolete CSS_PATH wording to name
the app/owning-screen stylesheet boundary. No selector or declaration changed;
the final snapshot is **803,075 / 804,000 bytes** (925 bytes headroom).
The frozen-input complete CSS/route/budget group then passed **43 tests** in
22.82 seconds (`/private/tmp/pr2427-css-frozen-final.log`), and the final
parity/fastpath/byte group passed **19 tests**. The combined long Watchlists/CSS
run recorded 248 passed/6 failed: its two known focus failures plus four generator
comparisons that still held the imported pre-edit header while generated files
had the corrected comment. The fresh complete 43-test group eliminates
that stale-input discrepancy; no CSS assertion was changed. Across the complete
Watchlists/context cohort, only the two qualified focus failures remain open.

## Console durability owner checkpoint (qualification remains open)

The existing durability controller now owns the twelve approved methods,
three completion/repair helpers and ten original state initializers. The audit
matches every body after only the two named provider/self-owner substitutions,
and matches all initializer ASTs/order. Named ports remain live; inspection of
the current store does not create it. The thin dynamic repair hook and writable
Screen state compatibility remain. Screen size is **16,973 lines / 506 methods**,
still above the unchanged 16,811 / 505 limits pending the handoff move.

Independent spec and source-correctness reviews pass for checkpoint only.
Fresh owner controls: **16 passed**. The exact generation fixture now builds
the existing settings owner before its genuinely used callbacks: complete
file **40 passed**. The interim 949-test group was compiled before that fixture
correction and two callback late-binding changes; it recorded **909 passed /
40 failed**, all failures in the subsequently corrected generation file. An
attempted stdin interruption also produced an ignored KeyboardInterrupt warning
in that interim run; this is not a clean final-head aggregate pass.

Fresh runtime compatibility group: **109 passed / 4 failed**. All four failures
reproduce on the baseline: shutdown-event expectation, raw-refusal restoration,
eager Canvas-controller expectation, and old resume-timer receiver strings.
The isolated retired-screen weakref assertion also fails against the properly
assertion-rewritten baseline. An earlier non-rewritten baseline pass was not
valid evidence of a new leak; a later current plain-assert run also failed,
so rewriting alone is not a proven cause. The lifecycle/test retention issue
remains open. No production workaround or weakened assertion was introduced.

Evidence: `/private/tmp/pr2427-durability-body-audit.log`,
`/private/tmp/pr2427-durability-runtime-final.log`,
`/private/tmp/pr2427-durability-baseline-rewritten.log`,
`/private/tmp/pr2427-durability-generation-green.log`, and
`/private/tmp/pr2427-durability-complete.log`.

The diagnostic inventory review found exactly three removed Screen statements
and the same three added durability-owner statements, with identical digests
6d15da687ae8593e, f9cb8d1a5b6887c4, and 437b59c84df2e755. Logger binding and
arguments are unchanged; this is relocation, not new privacy clearance for
incumbent exception diagnostics. No sink topology or inventory policy changed.
The inventory is refreshed only for those two owner rows after this review.

Fresh pre-checkpoint verification of the two complete owner/generation files:
**56 passed** in 6.29 seconds (`/private/tmp/pr2427-durability-checkpoint.log`).

## 2026-09-08 Wave7 rebase checkpoint (not merge-ready)

Replayed the branch onto dev `0fa35d00e8189b6c5404eaca2d6b401ec252ec71`
at `cc0a5bd5370856b24936abe609393c29210fa4fe`. The prior head remains
recoverable through `codex/pr2427-before-rebase-20260908`. Upstream's
LibraryMediaState/LibraryMediaController supersede the older review-only
Analysis/Reader extractions. Preserve the upstream scroll-settle fix, all real
Screen patch seams, and the independent entry-focus cleanup. Retain the six
unrelated controller assembly and the credential-metadata privacy repair.
Independent production/body audits pass; no numeric limit was raised.

Complete targeted evidence: assembly/wiring/import group 29 passed; provider
grammar 32 passed; trace final-values/execution-context 53 passed. The larger
Library group completed with **359 passed, 18 failed** in 388.83 seconds
(`/private/tmp/pr2427-rebased-library.log`). Failures include two stale removed
Screen-state references and unresolved painted focus/layout/action-label cases.
The two complete size-ratchet files report **43 passed, 5 failed**, including
Console, Library, CharacterRepair and MediaBrowse overages and Library pin slack.
These are open gates, not waived baseline failures.

The official diagnostic writer changed only owner_files 591 to 599; all
statement rows/digests and sink topology already match. A new fetched dev head,
`603812300f18562c1906ac837a8098a0a6cbff23`, requires another reconciliation,
preserving its scoped Media fault history, resume-cache reset and Copy selection.

Fresh census covered 985 refs and 324 registered worktrees. Only the review-owned
Assistant stylesheet task collided again: intermediate TASK-32013 is now
TASK-32040, preserving upstream Media debt TASK-32013 and all earlier provenance.
The identity guard passes 3,595 task files. No final-head review or merge claimed.

### Latest fetched-dev integration

Second rebase completed at `b8190be4fbf18dcf2a47ff193ab8771aa0366a7c`,
containing dev `603812300f18562c1906ac837a8098a0a6cbff23`. The two conflicts
retain dev's precise split-Library stylesheet assertions and facet-context reset,
alongside the PR's live `_sync_view` callback. Independent AST comparison verifies
all scoped fault-history/resume-cache changes; both Copy selection production
files and adaptive-reader validation are byte-identical to dev.

Retargeted nine test-only references from removed Screen Media fields to
`screen._media_state`, keeping predicates and scroll assertions unchanged.
Complete scroller, Reader/Analysis characterization and entry-focus files:
**13 passed** in 12.78 seconds (`/private/tmp/pr2427-media-state-retarget.log`).
Remaining visual failures and size gates are still open. This is a progress
checkpoint, not merge qualification.

### Existing-owner and fixture repairs after the published checkpoint

Character Repair local deduplication reduces 518 to 499 lines under its unchanged
502 ceiling. Nine repeated status updates use a fresh-query helper; three buttons
retain exact types, labels, IDs, classes, order and parentage through an ordered
loop. Existing controller logic and assertions are unchanged. Complete repair
and CSS-parity group: **57 passed**, including new DOM and widget-replacement
controls (`/private/tmp/pr2427-character-complete.log`); Ruff/format checks pass.

The approved Session handoff move places claim/release/acknowledgement and staged
evidence construction in the existing owner, retaining the composer DOM hook and
named live ports. Source is 16,742 lines / 505 methods under 16,811 / 505.
Independent root review confirms the ordering, sanitization, callback identity
and existing ownership. Owner/production handoff group **63 passed**, native
handoff selection **46 passed**, and boundary/registration group **78 passed**.
The frozen broader group was **296 passed / 15 failed**, not a full success:
14 inherited live-work failures plus a Message-test receiver alias guard.

Diagnostic statement review found precisely five identical warning statements
moved Screen to Session: 94270dbc2da185bc, 7bf3d873d385cf18,
69ab88068ad33952, d1aa7e16248a9dbc, 2c56fe86128ca98f. No content, arguments,
logger binding or sink topology changed. The official writer updates those two
owner rows only (`/private/tmp/pr2427-handoff-diagnostic-statements.log`).

The retired-Screen weakref failure was traced to a cancelled Environment-poll
TimerHandle's saved async Context, with 8.8 seconds left on its original deadline.
Keeping the real timer but shortening only that unrelated fixture cadence to
0.05 seconds restores collection; explicit before/after timer-lifecycle assertions
were added. Production code and the writer/unmount/durable-repair assertions are
unchanged. The weakref case plus complete Message and moved-seam guard files:
**32 passed** (`/private/tmp/pr2427-cleanup-fixture-green.log`). Full settings-file
qualification remains open. See the incident in lessons-testing-evidence.md.

The live-work harness lacked the app bundle at the app-CSS tier. A RED-first
exact `host.css_path == app.css_path` control now proves the production Console
boot stack, using the existing ConsolidatedCSSApp bracketing rather than loading
unrelated routes. All 13 zero-width paint/readiness/evidence failures pass with
their original assertions; the complete live-work file is **69 passed / 1 failed**
(`/private/tmp/pr2427-task4-live-complete.log`). The remaining Watchlists
latest-active-run mock failure is independent and still under investigation.

Full settings-file qualification now passes **417/417** in 247.73 seconds
(`/private/tmp/pr2427-settings-full.log`), including the uninstrumented retired
Screen collection test with the isolated real poll timer. Independent review
confirms that a retained application-owned writer/controller reference would
still fail the unchanged weakref assertion. This closes that recorded cleanup
failure; it does not qualify the separate outstanding Library/Watchlists gates.

The remaining live-work routing failure was a fixture bypass of production's
Watchlists route-stylesheet loader. Reusing `FullAppDestinationContext` for that
single case restores the real application lifecycle; a new click-hit assertion
confirms that the compositor reaches the intended button. All original route,
target, label and status assertions remain. The complete live-work file now
passes **70/70** in 72.27 seconds
(`/private/tmp/pr2427-watch-route-complete.log`); scoped Ruff and diff checks pass.
No production code or stylesheet changed in this repair.

### Complete-inventory follow-up

The four-file Library inventory finished **968 passed / 31 failed** in 1775.21
seconds (`/private/tmp/pr2427-fixture-complete.log`). This is retained as a
non-green result. The six render failures subsequently pass in the complete
render/side-by-side/focus-cue group: **162 passed** in 198.28 seconds
(`/private/tmp/pr2427-fixture-final.log`). Metadata normalization preserves
title-slot coordinates, wide empty-Reader geometry is exactly Items 145 /
Reader 46, and the two-cell destructive gap retains complete action labels.

The shell reconciliation cohort passes **25/27** in 105.45 seconds
(`/private/tmp/pr2427-shell-cohort.log`). Exact compact Notes geometry now
includes TASK-31645's one-row Lab strip, preserving fixed controls, focus,
identity and six-row surplus growth. Import is located by stable action;
grip collapse and width assertions follow TASK-31633/31951/31952. Independent
review found no weakened contracts. Its two remaining failures were attributed
separately: a real canonical Media deep-link selection mismatch, and an ingest
fixture missing production consolidated defaults. The latter's real compositor
hit the fold hint over Clear after an unstyled 23-row navigation bar; using the
existing ConsolidatedCSSApp passes the original click/clear/focus/identity test
and a new three-row navigation assertion (RED/GREEN logs:
`/private/tmp/pr2427-clear-{red,green}.log`). Broader requalification is pending.

The old Console unmount expectations now explicitly use the existing exact-owner
Textual removal helper after real navigation. TASK-31520 deliberately keeps a
normally suspended Console attached; its warm-reuse coverage remains unchanged.
All 45 original cancellation/identity/raw-draft assertions survive, plus a
single-production-Console control. The Canvas watcher case retains gateway
absence, watcher identity, disable latching and disposal while asserting the
existing shared store/controller identity. Complete ownership and screen-reuse
files: **31 passed** in 58.87 seconds (`/private/tmp/pr2427-compat-final.log`).

Existing-owner private cleanup removes exactly 22 Screen wrappers and retargets
their direct consumers; two constructor lambdas retain late owner binding.
Module AST comparison shows no unrelated logic change. New replacement-owner
controls and complete wiring files pass **24 tests**
(`/private/tmp/pr2427-private-cleanup-green.log`). LibraryScreen is now 35,393
lines / 1,260 methods; its line pin was lowered from 35,777, not raised.
The complete Screen/module ratchets are **47 passed / 1 failed**: only Media
Browse remains over its unchanged limit, **589 / 371**. Its proposed state
separation and the two Watchlists focus-cue repairs still await approval.

Current CSS qualification: **17 passed**, 803,101 / 804,000 boot bytes and
273 / 274 broad subjects. All six local derived-artifact checks pass (599
diagnostic owners, 3,597 unique task records, 113 schema tables, 281 index pins).
Qodo reports zero active findings on published 3b40d68710 and all six inline
threads are resolved; these are checkpoint observations, not final-head merge
qualification.

The Media deep-link fix now derives the requested Items identity from the
existing local Reader only in viewer mode. Both pure builders retain page
membership fallback; list mode, external detail and an empty Reader retain
their existing anchor. No new owner, mutation or scheduling was introduced.
Nine direct controls and six cold/warm RAG variants (legacy, numeric and
canonical IDs) pass **15/15** after RED reproduction; the target is explicitly
the non-first row, with exact Items/Reader identity, title and detail-call
checks. Independent correctness review found no issues.

The combined complete-file/consumer qualification is **887 passed / 15 failed**
in 694.24 seconds (`/private/tmp/pr2427-library-qualified.log`). It includes
complete Prompts/Skills canvas, characterization and Reader files; complete
Media state, Reader state/flow, browse, wiring, deep-link and new projection
files; every one of the 56 direct ingest-harness consumers; and all six RAG
variants. All Media and ingest selections pass. Seven Prompt and eight Skills
failures remain under investigation; exactly-once refresh, stale-page, hit-test,
priority-pane and dirty-draft assertions have not been waived. This checkpoint
is not merge-ready and does not claim the full Library shell file is green.

## 2026-09-08 checkpoint rebase and canvas failure attribution

Saved the preceding work at 4407a89b71, then rebased all 198 branch commits onto
dev 5aeac5ab221958ae612dd84ff47e047b23cd3f5d. The final tree differs from 4407
only by upstream's 59-line critique document; executable sources are unchanged.
Published f755da2daf631899dc3a4f394ec064ce3dbdfc4d with an exact lease against
remote 3b40d6871021b6c8b9d304486a45003828a01961. Qodo's updated checkpoint
comment reports zero active bugs/rules and all six inline threads are resolved;
fresh CI is running. This remains a progress checkpoint, not merge approval.

The seven Prompts failures separate into one obsolete version-field assertion
and six cases of competing resume/mutation reads. A read-only observer records
delete admission, then ScreenResume's snapshot and unfocused browse while
mutation_in_flight is true, then deletion's own focused browse and snapshot.
The extra read also consumes the injected post-delete refresh failure. The
relevant Screen bodies match pre-cleanup source after receiver normalization;
PromptsController is byte-identical. Evidence:
`/private/tmp/pr2427-prompt-origin.log`. Step 25 records a narrow admission guard
and ordinary/cancelled-modal negative controls; existing exactly-once and stale
page assertions stay intact.

The eight Skills failures are fixture drift: five bundle-only CSS pins omit
the Library-owned sheet; the isolated editor similarly loads only boot CSS;
the manual Items test uses 80 columns below the existing 82-column floor; and
the dirty-exit double omits the focus surface its real entry-focus method reads.
Step 26 retains real hit-testing, exact CSS/geometry contracts, a genuinely
manual closed-to-open transition and dirty-veto behavior. No production Skills
or CSS repair is proposed for these eight failures.

Skills fixture repair: the exact eight previously failing cases pass in 3.78s.
The dirty-exit fake additionally needed the existing non-Media return-candidate
seam after the focus read; actual entry-focus code remains active. Independent
spec/quality review passes, including the genuine closed-to-open transition,
real compositor checks and unchanged CSS property assertions. Complete-file
qualification remains pending (`/private/tmp/pr2427-skills-fixture-report.md`).

The first Prompt repair passes nine focused cases, but review is not yet clear:
failed delete/undo settlement refreshed only browse and would lose the suppressed
source snapshot. Before implementing a proposed retry-only worker argument,
Textual's wrapper was checked: exclusive cancellation happens before the worker
body, so that alternative could cancel an active manual retry. It was rejected
without source changes. The recorded fix uses the existing explicit dependency
pattern to check Screen-owned failure state before worker dispatch, with real
provider-read and manual-retry controls. No automatic retry of hard failures,
new state owner, or worker-signature change is authorized.

Fresh post-rebase derived-artifact preflight passes all six checks
(`/private/tmp/pr2427-preflight-post-rebase.log`). Fresh ratchets are **46 passed /
2 failed**, not the earlier 47/1: the eight-line Media deep-link repair landed
after that previous measurement and now measures 4,638 / 4,630. Its recorded
follow-up is documentation-only reduction with executable AST identity, not a
cap increase. Media Browse still measures 589 / 371 pending its separate design.

The corrected Prompt error settlement passes **23 focused checks** in 35.47s,
including real provider reads and an uncancelled manual Worker that completes
successfully. Independent re-review marks the lost-snapshot finding addressed,
with no new production findings. Media's documentation-only reduction reaches
4,630 lines with unchanged executable AST; Prompts reaches 4,997 / 4,998 and
Screen 35,392 / 35,393 lines, still 1,260 methods.

The first 677-case complete verification run was deliberately stopped near 31%
after an exact-resource check found the two new ordinary/cancel controls did not
finalize their separate Prompt DB. It is not qualifying evidence. SIGINT did not
finish teardown; the exact owned process was terminated (exit 143), and its log
is preserved at `/private/tmp/pr2427-canvas-complete.log`. Finalize only the new
controls' exact resources before restarting the complete selection; no process
termination or test-interruption result counts as a cleanup pass.

The finalized opt-in resource fixture now supplies all 15 new controls, closes
only its DB instance's captured main/thread handles after workers drain and the
harness exits, and verifies every handle rejects a query as closed. All 15 pass;
independent cleanup review passes without changed behavioral assertions.

The restarted nine-file qualification finishes **678 passed / 2 failed** in
525.15s (`/private/tmp/pr2427-canvas-final.log`). Both complete canvas files,
Prompts characterization/wiring/private-owner checks and entry-compose tests
pass. Remaining failures are the known Media Browse size and the isolated
Library suspend fixture, whose `__new__` bypass omits `_unavailable_navigation`.
Four export cases additionally surface an unawaited Textual callback warning
from the RAG indexing-guidance path (`app.py:11567`); those require attribution,
not a warning filter. Fresh final-source preflight passes all six checks
(`/private/tmp/pr2427-preflight-final-canvas.log`). The branch is not merge-ready.

The isolated suspend fixture now exercises the actual unavailable-navigation
cleanup, with its constructor-owned state and a scoped non-modal app context.
All original seven-timer stop/clear assertions remain; new assertions require
cleared return admissions, advanced generation and a hidden return control.
Its complete four-case file passed in 15.40s before the indexing cleanup below
(`/private/tmp/pr2427-reuse-fixture.log`).

The warning observer identifies the two raw reuse-app tests as producers:
their real Media DB initialization installs global ingest callbacks and an
indexer notifier bound to an app whose loop is later closed. The later Export
fixture dispatches that same callback. An opt-in fixture now restores only each
producer's exact registrations, preserves borrowed/transferred services, and
stops only a captured newly owned worker while the pytest loop can service its
callbacks. Review rejected a global reset dispatched through `to_thread`: a
replacement may arrive before execution. The final local helper rechecks
identity and notifier ownership under the existing locks before exact-object
stop. A deterministic replacement control fails against the old reset arm and
passes with this fix. Four ownership controls and both real producers followed
by an Export consumer pass (seven cases, 18.43s, only three baseline dependency
warnings; `/private/tmp/pr2427-indexer-fixture-report.md`). No production
indexing, configuration or warning-filter changes were made.

Fresh checkpoint preflight passes all six checks
(`/private/tmp/pr2427-preflight-checkpoint.log`). Scoped Ruff passes on the
Prompts owner and affected test files. MediaController's seven and Screen's
50 baseline Ruff code/message findings are unchanged; this is not a whole-file
lint-clean claim. The fresh remote still has PR head `f755da2daf` and dev
`5aeac5ab22`; all required CI checks pass on that published checkpoint, and
Qodo reports zero active findings there. Unpublished repairs require their own
review/checkpoint; the Media Browse and Watchlists design gates remain open.

Independent review passes the combined suspend/indexer fixture changes and
confirms every original assertion remains. The first complete two-file run
finished 99 passed / three baseline dependency warnings in 83.58s, but a
post-freeze fixture edit overlapped it. It is not final qualification evidence
(`/private/tmp/pr2427-reuse-entry-final.log`). The unreviewed edit was restored;
the reviewed reuse file has git blob hash
`1b5e9dca67e9a5c0e66861f0e5d8ef7fa802e1f4`. A fresh frozen-source run is required.

That fresh complete reuse + entry-compose run passes **99 tests in 81.27s**,
process exit 0 (`/private/tmp/pr2427-reuse-entry-qualified.log`). The reuse
file's hash is unchanged before/after and matches independent review. The four
unawaited-callback warnings are absent, with no warning filters; only the three
existing requests/pydub/webrtcvad dependency warnings remain. The entire
entry-compose file is unchanged. Combined with the prior complete canvas and
Prompts qualification, this closes the attributed canvas/suspend/indexer-fixture
failures, not the Media Browse size or Watchlists focus gates. The complete
Library shell rerun and final revision's external review/CI remain required.

## September 8 approved final gates and Qodo round 3

The user approved the UI-local Media Browse state separation with ADR and the
theme-aware Watchlists focus repair after checkpoint `98a9aeeb85`. ADR-128 and
the exact whole-pure-member extraction plan are committed at `839c07ea56`.
The requested repair retains the controller's generation/worker boundary and
all existing size/CSS/focus limits. New dev `1c022378cb` adds only six Backlog
documents (TASK-32041–32046); it will be incorporated after source freeze.

Qodo's new comments 3957095884/3957095896/3957095903 are verified and repaired:
historical/current authority and retention proof reads now use each database's
transaction cursor; the UI workflow keeps one documented editable install.
All 22 affected DB assertions remain equivalent under cursor normalization.
The strengthened CI test fails with two installs before deletion and passes
with one afterward. All three complete test files pass **78 tests in 23.81s**
with only two existing dependency warnings
(`/private/tmp/pr2427-qodo-round3-complete.log`). Independent spec/correctness
review passes; scoped lint and diff-check pass. Production database behavior,
standalone SQLite bootstrap fixtures and dependency choices are unchanged.

Checkpoint `9c6a003fc6` publishes those three fixes and their inline replies.
GitHub reports all nine review threads resolved; the new revision's Fast Lane
was still running at inspection, so this is not final-head CI qualification.

The approved Watchlists repair remains scoped to the two exact Select IDs.
The conservative CSS splitter leaves their mixed-prefix rule in the generated
boot bundle. This small measured eager cost is preferable to broadening the
rule to every Select in ArticleListPane merely to force lazy classification.
The existing 804,000-byte and 274-subject caps are unchanged; snapshot refresh
and rendered-theme qualification must follow an under-cap measurement.

The Media state extraction's frozen focused qualification passes **410 tests
in 42.80s** across eleven complete files, with the three existing dependency
warnings (`/private/tmp/pr2427-media-state-final-focused.log`). The controller
pin falls from 371 to its measured 323 lines; the new state is pinned at 295.
All 19 initial values and five constructor ports are preserved. The ten moved
pure methods and two helpers are text-identical; thirteen runtime methods and
all eighteen existing consumer files are AST-identical after receiver/fake
normalization, preserving 6,768 assertions. Independent review passes the
extraction and final exact-ID Watchlists nested-painter rule. Complete mounted
Library qualification is still pending.

The separate complete packaging guard fails because Screen still eagerly
re-exports Conversation Reader. History proves the original assembly test
looked up that alias, but the rebase changed the test to canonical class
imports without deleting the now-unused re-export. The three-line import is
identical at 9c6a003fc6 and is not introduced by Media state. Task step 32
records its narrow removal before implementation, with unchanged packaging,
assembly, wiring and Screen-size checks. The pre-import payload guard itself
passes at 497/500 modules; all six derived-artifact checks also pass at this
checkpoint. Neither result overrides the failed deferred-import contract.

Removing that exact stale re-export makes all four complete packaging/assembly/
conversations-wiring/Screen-size files pass **22 tests in 3.49s** (two existing
dependency warnings; `/private/tmp/pr2427-reader-import-final.log`). Independent
review passes. Screen is now 35,389 lines / 1,260 methods; the separate import
cleanup is explicitly excluded from the pure Media receiver-equivalence audit.

Before long qualification completed, dev advanced to `9a54013f07` with the
TASK-32046 canvas-filter keyboard repair and four shell tests. Root interrupted
only its own two fresh pytest processes with SIGINT and observed both exit 2:
119 and 82 cases had passed, respectively, with no failures. Those interrupted
runs are not complete-file qualification. Reviewed fixes will be checkpointed
and rebased before restarting all affected complete mounted files on new dev.

The Watchlists fix now paints both the SelectCurrent background and its nested
Static value/arrow foreground through theme tokens, scoped to the same two IDs.
The intermediate current-only foreground failed all six label-contrast checks;
the final six control/theme cases pass. The complete overlay file passes **15
tests in 73.95s** and the four complete CSS guard files pass **70 tests in
77.14s**, with existing dependency warnings only. Independent final review
passes. The measured budget is **803,658/804,000 bytes**, and the broad-subject
census is **274/274**; the exact-ID Static descendant adds one subject. The
approved snapshot writer independently measures that under-cap total and
refreshes only the CSS snapshot without force. Its diff includes the already
present 26-byte ConsoleSelectionMenu drift plus the 557-byte Watchlists rule;
neither is a cap increase or hidden stylesheet change.

The 203-commit rebase onto `9a54013f07` completed without conflicts at
`957da1ba0e`; the saved pre-rebase checkpoint remains at
`codex/pr2427-before-keyboard-rebase-20260908`. Comparing saved and rebased trees
shows only upstream keyboard code/tests and documentation additions. Its
23-line handler addition reproduces a 35,412/35,393 Screen breach. Step 33
shortens only the existing on_key docstring by 21 lines, retaining its task and
keyboard/focus facts. The full Screen AST is identical after removing that one
docstring; Screen is 35,391 lines / 1,260 methods with unchanged ceilings.

Post-rebase complete size/module/assembly/packaging/pre-import/UI-ready checks
pass **65 tests in 14.98s** (`/private/tmp/pr2427-rebased-guards.log`). UI-ready
is 973/973 modules; pre-import is 496/500 modules and 362,737/378,740 lines.
All six derived-artifact checks pass, including all 3,603 Backlog task records
(`/private/tmp/pr2427-keyboard-rebase-preflight.log`). The full rebased shell
and fourteen-file mounted cohort are running on frozen sources; they remain
required before a merge-ready claim.

## September 8 mounted qualification and Qodo round 4

On published `91f4122884`, the complete Library shell passes **846 tests in
1,579.00s**, but its unchanged descriptor sentinel reports **12 to 475 handles
(+463, limit 200)**. This is functional coverage, not resource clearance;
native-owner attribution remains open. The fourteen-file mounted cohort finishes
**681 passed / 10 failed in 912.39s**. Logs are
`/private/tmp/pr2427-shell-rebased-final.log` and
`/private/tmp/pr2427-mounted-rebased-final.log`. No warnings are suppressed.

The four navigation failures assumed every visit constructed a fresh Screen,
contrary to accepted installed-screen reuse. Step 35 preserves the cold-path
assertions with scoped route metadata and adds default warm-path identity,
retained-input/typing and full staged-launch payload checks. All 134 unrelated
functions remain AST-identical and all 27 old mounted-journey assertions remain.
The complete navigation and three reuse files pass **162 tests in 164.74s**;
independent review passes. The Reader filter failure observed settled domain
state before its Clear button's recompose settled. The existing stable-selector
wait immediately precedes the unchanged lookup and real press. Its complete
file passes **84 tests in 88.51s**, and independent review passes.

Qodo round 4's four verified findings are repaired in tests only: transaction
cursors for proof reads, cold-restart cleanup on early failure, exhaustive
agent-swap owner cleanup with error aggregation, and real SQLite evidence
complementing the unchanged synthetic fault matrix. Four complete files pass
**115 tests in 50.55s**. Independent review then found a missing early-failure
cleanup guard in the new real-SQLite control. A forced failure reproduces it;
the corrected complete resource-fixture file passes **15 tests in 2.11s**,
including that new control. The other three files remain unchanged. All
pre-existing assertions are retained, scoped Ruff checks pass, and independent
review passes the corrective increment. These findings still require published
inline responses and final-revision review.

Four Trash focus/layout failures and the descriptor warning remain open. The
Live Trash walkthrough's startup resets its preinstalled logging sinks; fixing
capture/cleanup exposes a previously masked Items-pane toggle failure. Its
latest complete run is **4 passed / 1 failed**, not qualification. Task step 37
records the capture repair and non-vacuous privacy requirements. New dev
`e38b44acee` also requires reconciliation; its two review-created Backlog ID
collisions need the requested user exception before renumbering. No merge-ready
claim is made at this checkpoint.
