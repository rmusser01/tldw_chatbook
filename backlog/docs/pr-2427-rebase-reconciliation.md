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
| TASK-31759 (intermediate TASK-32013, TASK-32040) | TASK-32110 | Assistant-turn production stylesheet harness |
| TASK-31901 (intermediate TASK-32014, TASK-32108) | TASK-32136 | Loguru capture sink lifetime |
| TASK-31902 (intermediate TASK-32015) | TASK-32109 | Deferred Chunking Lab imports |

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

Checkpoint `8fe8c40e79` publishes the reviewed Qodo/navigation/Reader fixes.
Qodo comments 3959841194/3959841197/3959841205/3959841211 now have inline
implementation and test evidence; their four threads are resolved. Latest dev
has since advanced to `2c6a7de490` (ONNX diarization, bulk-reader pilot and the
Media zero-selection explanation); final rebase and shared guards remain open.

Native shell attribution narrows the resource leak to the test-only
_LibraryIngestCanvasHarness: two passing ingest pilots retain thirteen regular
SQLite/WAL/SHM descriptors and a progress-drain thread. Temporary exact-owner
unmount cleanup reduces their process to zero regular DB descriptors and only
MainThread alive. The harness omitted its inherited ingest shutdown and
main-thread database close. Step 42 records the permanent scoped repair and
real-SQLite error-path control; the full shell still requires a native-observed
rerun after final source freeze. See
`/private/tmp/pr2427-library-shell-fd-attribution.md`.

The three Trash initial-focus failures reproduce with correct existing
assertions. Both immediate and fast-result publication precede destination
canvas mount; the ensuing fallback recomposes lose the focus settlement.
Steps 38–39 prime the initial loading state, await recompose before the one
guarded request, and scale the fold test's message from actual mounted width.
Focused RED is **4 failed / 1 passed**; GREEN is **6 passed**, including direct
loading-before-mount/request-after-mount ordering. Independent review passes;
Screen remains exactly **35,393 lines**, with no cap increase. Complete Trash,
Reader and local Live qualification is running on frozen production sources.

The Live harness separately needed keyboard-focus settlement (native observer
proved the queued page update steals focus before Enter) and separate capture
buffers for direct Loguru versus production's stdlib forwarding. Exact mutation
event counts are required independently in both channels, with canaries and all
privacy assertions retained. These fixes are not yet declared passing; see
`/private/tmp/pr2427-live-harness-report.md` for intermediate failure evidence.

The complete Reader/Trash/Live group finishes **180 passed / 1 failed in
187.93s**: all 84 Reader and 92 Trash cases pass, as do the four pure Live
controls. Its only failure is an obsolete Live expectation that normal Media
stays stale after Restore. TASK-31275 deliberately replaced that manual-Retry
policy with an authoritative page refresh. Step 44 compares the real page-2
service result, exact count/membership, new result identity and retained
selection, then preserves every Back/scroll/focus assertion. The corrected
160x50 walkthrough now passes its full privacy/path/zero-DB-handle proof.
The complete Live file still fails at the next 120x35 posture oracle (expects
compact, runtime reports wide), so it is explicitly **not** final qualification
(`/private/tmp/pr2427-live-qualified.log`: 4 passed / 1 failed in 28.01s).

The permanent ingest-harness cleanup and its error-path control pass four
focused native-observed cases in **3.91s**. Only the expected one-time FIFO and
socket pair remain (5 to 8 descriptors); zero regular SQLite descriptors and
only MainThread remain. Independent review passes the control's corrected
setup/fallback cleanup boundary. The final diff adds 63 lines and changes no
pre-existing executable AST. The full 848-case shell remains pending. These
reviewed fixes are checkpointed before the latest-dev rebase; incomplete Live
posture and final resource/CI/review gates remain open.

### Latest-dev qualification checkpoint — 2026-09-08

The reviewed checkpoint was rebased successfully onto dev `2c6a7de490` at
local head `5020cdebb9`; remote PR head remains `8fe8c40e79` pending the
lease-protected publication. Both appended lessons were preserved and the
diagnostic inventory regenerated after inspecting the incoming exception-type
log statements. Six complete architecture/import/UI-ready guard files pass
**65 tests in 11.89s**, with all limits unchanged. Eight complete incoming
Agents/Chat/ZAI files pass **420 tests in 65.55s**. Preflight passes five of
six checks; the only failure is the two explicitly pending review-only task
identity collisions, TASK-32014/32015. No renumbering exception is inferred.

The seven-file Live/settings/Media/selection/recompose/Skills cohort finishes
**411 passed / 3 failed in 350.95s**. The complete Live walkthrough now passes
all four sizes, including 120 columns, with the authoritative post-Restore
page, independent logging channels, privacy, keyboard and zero-owned-handle
assertions intact. Two failures are attributed selection-fixture mismatches:
source-order row identity and an absent optional-query interface. Step 47
repairs only those fixtures; the complete selection file then passes **7 tests
in 7.50s**, with independent review and scoped Ruff passing. The third failure
is the inherited whole-Library recompose census, **66 against the unchanged
63 cap**; attribution remains open and the cap is not raised.

The complete 15-file non-live Audio increment initially reports **544 passed,
1 failed, 8 setup errors**. All eight errors are the sandbox's loopback-bind
restriction; rerunning the complete ONNX file with the approved local-server
permission gives **47 passed**, accounting for **552 of 553** unique cases
passing. The sole real failure is a phantom optional extra: accepted
TASK-31827 moved sherpa-onnx/numpy into core while retaining its named readiness
capability. Step 46's stronger metadata/consumer controls reproduce **4 failed
/ 4 passed** before any production edit. No models, device tests, external
services or package installations are used as qualification.

The full 848-case Library shell native-observed run remains active on frozen
runtime sources. Its late native sample shows retained Notes/export database
paths from earlier tests, unlike the current-test-only sample at 50%; those
handles require separate exact-owner attribution before declaring cleanup
complete. The successful focused ingest cleanup is not evidence that every
other harness owner closes.

A fresh remote check now finds dev `a59536eb70` (Console trace source-pin
migration, Kokoro recovery and Media flag-width correction). It is fetched,
not yet integrated at this checkpoint. Final rebase, affected verification,
final-head Qodo/checks and normal merge remain open.

The diagnostic shell run was deliberately interrupted after native evidence
proved the separate owner leak: **834 passed / 11 warnings in 1412.69s**, exit
2, not a complete-file pass. Its session sentinel independently reports
**331 additional descriptors (12 to 343)**. The late native snapshot attributes
the retained Notes paths to all eleven real-service callers and the remaining
export handles to the file-backed ChaChaNotes owner. Step 48 uses opt-in fixture
finalizers and the existing same-file quiescence barrier; failure-path controls
must exercise those actual finalizers, retain foreign-path usability, and prove
the cached worker handles close. A clean complete-file rerun remains required.

The recompose history is now exact: four accepted ADR-120 character-navigation
structural sites, offset by one existing Media fallback consolidation, account
for the net three-site breach. Step 49 leaves those navigation barriers intact
and routes three older Media viewer-substate updates through the already
sanctioned targeted sync seam, including its existing missing-viewer fallback.
The 63-site ceiling is unchanged; RED/GREEN and mounted qualification are open.

Step 46 is now independently reviewed and frozen: the complete optional-deps,
meeting-owner and ingestion-capability files pass **285 tests / 10 warnings
in 26.89s**. The focused controls transition from four expected metadata
failures to **8 passed**. All 37 capability keys remain; only ONNX becomes
core-backed, and every genuine extra retains its original install commands.
Source/Utils/Library Ruff checks pass; the Audio file has exactly its five
pre-existing findings, checked against HEAD, with no new finding. Changed
functions are formatted without unrelated whole-file reformatting. Evidence:
`/private/tmp/pr2427-core-metadata-report.md`.

Step 48's valid-fixture RED controls reproduce one retained setup handle and
two retained worker/test-failure handles. After the exact finalizers, both
controls pass; the native-observed group of both controls, all eleven Notes
callers and the file-backed export case passes **14 tests in 25.58s**. All
fourteen post-test descriptor deltas are empty despite 166 observed
CharactersRAGDB opens. The actual fixture finalizer preserves the primary
error and leaves the foreign-path connection usable; outer safety cleanup also
protects intentionally failing regression runs. Independent review, Ruff and
diff checks pass. Final complete shell qualification is still required.

Step 49 passes independent review: exactly three direct Media recomposes now
use the existing viewer sync seam. Focused controls transition from three
expected failures to three passes. The two complete Reader/render files pass
**208 tests in 240.14s**; census, wiring, assembly and size guards pass
**77 tests in 22.33s**. The whole-Library census is restored to **63/63**, without
changing its pin. Existing mounted shell highlight add/delete and analysis-save
tests remain part of the pending final complete Library run. All implementation
files are frozen for checkpoint and latest-dev integration.

### Published final-code checkpoint — 2026-09-08

Checkpoint `ed42fc8c27` is preserved by
`codex/pr2427-before-trace69-rebase-20260908`. Its rebase onto dev
`a59536eb70` completed at **`8efe3b03823c5cfdde6cb001ea1386fe829c4c76`**,
published with the exact old-head lease against `8fe8c40e79`. Source provenance
and shell/selection/optional-deps hashes remain unchanged across rebase.
The Console conflict retains both upstream current-turn derived provenance and
the reviewed contiguous-prefix rendered-system classification; independent
review confirms saved/derived owners take precedence. Both appended lessons
remain, and the reviewed TTS diagnostic changes are reflected in the rebuilt
inventory (599 owners, 1348 TASK-492 calls, 55 TASK-31551 calls, 7675 TASK-494
calls, 12 sinks).

Fresh post-rebase complete-file evidence:

- Eight architecture/import/UI-ready/CSS guard files: **73 passed, 6 warnings,
  22.33s**. The runner's later default-temp garbage-cleanup warnings are separate
  from that successful test verdict; subsequent cohorts use explicit unique
  basetemps.
- Ten incoming Chat files, three migration files, agent bridge and connection
  quiescence: **647 passed, 2 warnings, 220.26s**. Qodo's transaction reads and
  cold-restart owner cleanup remain intact.
- Six offline TTS/Settings/Speech files: **102 passed, 3 warnings, 9.95s**.
  Synthetic-buffer codecs run locally; no model downloads, device playback or
  dependency installs.
- Media state/render and complete four-size Live walkthrough: **218 passed,
  7 warnings, 227.65s**. The real walkthrough itself takes 61.25s.

The final **850-case** shell run uses the existing native per-test descriptor
snapshots without the earlier diagnostic connection monkeypatches, plus an
explicit end-of-pytest snapshot. At 84%, native inspection finds only the current
test's constructor quartet and no old Notes/export/ingest paths. Completion and
the end snapshot are still pending at this checkpoint.

Qodo confirms review through exact head `8efe3b0382` on 2026-09-08 at 17:49 UTC:
**0 open bugs, 0 rule violations, 0 skill insights**; all thirteen historical
findings are resolved or previously dismissed. GitHub's PR Fast Lane reports
**754 passed / 2 failed**: both failures are Backlog uniqueness assertions, not
additional runtime failures. Preflight likewise passes every check except IDs.
The latest dev adds a third collision, review TASK-32040 versus upstream private
trace-history ownership, alongside review TASK-32014/32015. The user has been
asked for the precise three-review-task exception; no upstream or review ID is
silently reassigned. Final merge remains blocked on that approval, corrected
identity gates and completed shell evidence.

### Final Library verification completed — 2026-09-08

The complete frozen-head Library run finishes **850 passed, 10 known
dependency/deprecation warnings in 1482.71s (24m42s), exit 0**. The shell source
hash remains `3bb2b963f8dacbdc5647911786fdb97d049ce8d5feab3763fcf76dead9c5265d`
at published code head `8efe3b03823c5cfdde6cb001ea1386fe829c4c76`.
The final native observer record is exactly `exit_code: 0, sqlite_paths: []`;
the session FD-growth sentinel emits no warning. The late native sample showed
only the current test's constructor databases, with no persistent Notes,
export, Media or ingest paths. Immediate per-test deltas are not conflated with
persistent leaks: 59 of 850 snapshots had transient additions, but the late
sample and final empty snapshot establish retirement.

Evidence: `/private/tmp/pr2427-shell-850-native.log`,
`/private/tmp/pr2427-shell-850-resource.GQt7OZ`, and the independent audit
`/private/tmp/pr2427-shell-850-final-native-audit.md`. This completes the pending
full affected-file cleanup qualification without a full-repository sweep or
any warning/descriptor-limit relaxation.

The remaining merge blocker is the explicit exception for renumbering only
review-created TASK-32014, TASK-32015 and TASK-32040 while preserving upstream
IDs and historical references. Approval has not arrived; no IDs were changed.
The follow-up is paused rather than repeatedly polling this user decision.
After approval, revalidate free IDs/references, perform the narrow renumbering,
publish, and obtain green identity/final-head checks before normal merge.

### 2026-09-08: approved three-record identity repair

The user's subsequent `yes` explicitly authorizes the three-record exception
above. A fresh fetched-ref and registered-worktree census found maximum
TASK-32107. Only review records moved: Loguru 32014 to 32108, deferred Chunking
Lab 32015 to 32109, and Assistant stylesheet harness 32040 to 32110. Upstream
Meetings and private trace-history IDs and references remain unchanged. Active
review references use the new IDs; historical mappings, creation dates,
implementation notes, and literal evidence paths remain intact.

Verification on the identity-only change: all three complete-file identity tests
pass (two existing dependency warnings); the shared guard reports 3,615 unique,
Windows-compatible records; Backlog CLI resolves all three exact new paths and
retains their Done status. All six derived-artifact preflight checks and
`git diff --check` pass. No runtime or test source changed in this repair.

The refreshed dev ref is now 7e81ed55db, beyond the previously qualified
a59536eb70 base, with Notes decomposition and TTS/trace/Media changes. Rebase and
affected-file verification remain required before final-head review and normal
merge. The earlier 850-case native result remains evidence for its recorded
source, not for this incoming runtime delta.

## 2026-09-08 wave-8 reconciliation and startup paydown

The 209-commit rebase onto dev `7e81ed55db` completed at `c646154a61`.
The published lease is still `fc4ea45cef651a320868ecc91c1de4c79456e646`;
`codex/pr2427-before-wave8-rebase-20260908` preserves that checkpoint.
The three approved ID moves are complete, not awaiting further approval.
The post-rebase guard reports 3,626 unique Windows-compatible task records.

Task 31932 steps 52–60 record the qualification repairs before implementation:

- Preserve wave-8 Notes ownership: pass Undo tree reconciliation through a named
  late-bound port, retain Files admission's `render=False`, remove the obsolete
  unreachable placement patch, and retarget retired Notes fields. NotesState has
  99 fields after the already-reviewed inert timer removal. The Notes controller
  stays within its unchanged 5,276-line ceiling; LibraryScreen tightens to
  31,901 lines / 1,229 methods.
- Correct the overlapping Media zero-result branch's stale `controller` local
  to `browse`. Retarget 17 incoming test reads to the ADR-128 browse-state owner,
  retaining exact empty-list, Reader placeholder and hidden bulk-reason checks.
- Attribute the 973/974/975 readiness variation to the existing scheduler's
  first-tick imports of `emergency_stop` and `scheduler_heartbeat`. Defer settings
  diagnostics to the same five failure branches and cache the real row-action
  controller on first use via stdlib `cached_property`. No scheduler delay,
  readiness timing change, custom proxy or raised cap. Both cold-import controls
  fail before the change; complete first-use/wiring/failure/ownership guards pass.
- Bring fixtures into parity with real lifecycle/state: seed provider settings
  for a manually created inactive chat; discover controller mock slots across
  sibling wiring builders; drive flat Notes fixtures through their existing
  editor helper; wait for exact compact File Notes paint after layout changes;
  treat hidden retained conflict UI as hidden while preserving exact autosave
  state, one-save and stored-version assertions.
- Move the metadata-only Undo diagnostic pin to its new Notes-controller owner,
  without permitting new fields or exception capture.

Independent review found no actionable production/ownership/privacy issue in
the lazy-owner, settings diagnostic, Notes and Media changes, and approved the
row-switch, retained-conflict and responsiveness fixture repairs pending GREEN
qualification. The prior incoming diagnostic/trace audit confirmed accepted
TASK-32047 behavior, not a reason to reinstate the superseded logging policy.

Completed evidence on the reconciled source:

- Incoming non-live TTS complete files: **240 passed**, 3 existing warnings,
  34.18 s; no inference, audio-device or model-download qualification claimed.
- Ten complete logging/trace files: **370 passed**, 2 existing warnings,
  119.70 s, including credential filtering and rendered/discarded trace paths.
- Complete first-use, settings failure, controller wiring/ownership, Screen size
  and startup cohort: **161 passed**, 4 warnings, 31.00 s. Census **971/973**.
- Complete Reader/render/viewer/multiselect and Library guard cohort:
  **460 passed**, 5 warnings, 326.64 s; repeated census **971/973**, boot CSS
  **803,291/804,000**. Log: `/private/tmp/pr2427-wave8-media-guards-third.log`.
- Complete adaptive Reader file: **14 passed**, 3 existing warnings, 55.06 s.
  Log: `/private/tmp/pr2427-adaptive-complete-final.log`.
- All six derived-artifact preflight checks pass: diagnostic summary is 600
  owners / 7,667 TASK-494 calls; other reviewed diagnostic row digests are
  unchanged. Scoped Ruff and `git diff --check` pass.

The first current-source native shell run stopped at **503 passed / 1 failed**
on the retained hidden conflict DOM assertion. It is failure evidence, not a
clean resource qualification: its final frame still retained the failed test's
SQLite descriptors. Log/report: `/private/tmp/pr2427-wave8-shell-native.SHQelx/`.
The corrected full **851-case** run is now observed at
`/private/tmp/pr2427-wave8-shell-native.bMHrar/`; its `native.jsonl` records exact
package provenance, HEAD, diff and shell hashes, native per-test additions and
the final exit/descriptor snapshot. The runner registers only the native
observer, not the diagnostic helper's connection-patching fixture; it performs
no GC, global close or descriptor-limit change. Final qualification, remaining
Notes/navigation and diagnostic files, publication and final-head Qodo/CI review
remain open at this checkpoint. Do not merge from the partial results above.

Subsequent completed evidence: all **349** remaining Notes characterization,
reuse, selection, modal and navigation cases pass (236.48 s; 3 warnings,
`/private/tmp/pr2427-wave8-notes-remaining.log`); all **171** diagnostic/privacy
and token-preparation cases pass (178.28 s; 8 warnings,
`/private/tmp/pr2427-diagnostic-guards-final.log`); the complete responsiveness
file plus the exact repaired autosave case pass **17** cases (5.03 s; 3 warnings,
`/private/tmp/pr2427-final-focused-controls.log`). The prior File Notes run
completed that entire file before stopping at the now-repaired next-file
characterization selector; its preserved log is
`/private/tmp/pr2427-wave8-notes-complete-third.log`.

The other 21 incoming changed UI files are being qualified in two complete-file
cohorts: **575** cases in `/private/tmp/pr2427-wave8-notes-inventory.log`, and
**470** in `/private/tmp/pr2427-wave8-shell-inventory.log`. The real-local SQLite
tree walkthrough and ProductionApp File Notes owner lifecycle run separately at
`/private/tmp/pr2427-wave8-local-tree.log` and
`/private/tmp/pr2427-wave8-file-owner.log`; independent review confirmed both are
offline, and the ProductionApp process owns its collection-time environment
isolation. No inference, device, download or remote Git operation is authorized
by these runs. While these frozen-source tests execute, the shared origin/dev
ref advanced to `565dc49921` with an eight-file TTS-only increment. Integrating
and qualifying that accepted increment, then final-head review/CI, remains
required after the active tests stop.

The incoming inventory is now fully exercised, not wholly green: the 575-case
Notes group has **568 passed / 7 failed** (543.26 s), and the 470-case shell group
has **460 passed / 10 failed** (491.29 s). Exact failure lists/tracebacks are in
the two logs above. Remaining investigation covers disabled-CSS ownership,
visible toolbar/copy/harness expectations, one conversations fake and row-factory
census, exact Trash return settlement, recompose/Tab query counts, two Notes
journey paint checks, four sync-authority metadata preconditions, and the Notes
unmount ordering guard. Existing caps and filesystem security checks stay intact.
These are open qualification findings, not waived checks or completed work.

The separate ProductionApp owner file passed **12** cases (20.94 s). Native
observation of the passing real-local Notes walkthrough found **24** retained
handles for its directly created database. Step 61 adds an exact-owner yielding
fixture and failure-unwind control; independent review required unconditional
cleanup of the manually driven control too, which is included. The final
complete file passes **2** cases (10.75 s), with final `exit_code: 0` and
`sqlite_paths: []` in
`/private/tmp/pr2427-wave8-local-tree-native-final.6hMpND/native.jsonl`.
All 37 original walkthrough assertions are preserved. The fresh 851-case shell
run is still in progress; this checkpoint is intentionally not merge-ready.

Wave 8 subsequent evidence: the complete native shell run finished **851 passed,
10 warnings in 1540.57 seconds**, with final `exit_code: 0` and
`sqlite_paths: []` at
`/private/tmp/pr2427-wave8-shell-native.bMHrar/native.jsonl`. Intermediate
factory-app auxiliary handles were released by final pytest cleanup; independent
resource review found no persistent leak. This qualifies the source before
step 65, not the later performance repairs. Complete Notes journey/folder files
now pass **147** cases in 98.57 seconds using the verified per-user temporary
root (`/private/tmp/pr2427-journey-final.log`). The honesty/multiselect/toolbar
cohort passes 55 cases with three remaining fixture/action failures
(`/private/tmp/pr2427-honesty-final.log`), under investigation.

Step 65 independently identified three runtime defects: uncached ordinary rail
lookups, detail/browse callbacks preceding the initial Media structural mount,
and Trash-return settlement armed after the replacement owner's first geometry.
The focused RED reproduces five recomposes against one and three DOM queries
against one; the Trash case passes isolated but failed the complete inventory.
The fixes reuse existing seams and preserve all limits and exact-settlement
assertions. Complete affected-file qualification and independent review remain
in progress. Fresh dev also includes the accepted Library keyboard increment
at `38f7fe63d4`; rebasing and qualifying that increment remains open. This is a
progress checkpoint, not a completed review or merge claim.

Steps 65–66 are now independently reviewed and qualified: all **182** cases in
the complete performance, focus/return-settlement, honesty, Conversations
multiselect, toolbar-adaptation and both size-ratchet files pass (119.10 s;
`/private/tmp/pr2427-step65-refined.log`). The first unconditional pre-arm
attempt failed fourteen retained-viewer settlement cases; the final fix
pre-arms only structural Trash returns and preserves post-mount arming for
retained viewers. New RED/GREEN controls pin same-kind owner identity and
post-await navigation admission. The cross-kind cap measures exactly one actual
awaited recompose plus zero queued refresh fallbacks. Existing Screen
31901-line/1229-method and MediaController 4669-line ceilings remain unchanged.
All six derived-artifact preflight checks pass
(`/private/tmp/pr2427-wave8-preflight-final.log`), scoped undefined-name checks
pass, and `git diff --check` is clean. The 17 incoming-inventory failures are
addressed; final-source lifecycle verification, the newer dev increment and
final-head PR review/checks still remain. Do not mark the owning task Done yet.

## Wave 9 — latest Library keyboard and speech increment

Checkpoint `279282b5991053aa8e1aa53756351b3682891a58` is published on PR 2427;
its final per-click file passes **11** cases, including separate actual-recompose
and refresh-fallback counters (`/private/tmp/pr2427-step65-final-spies.log`).
Backup `codex/pr2427-before-wave9-rebase-20260908` preserves it. The 210-commit
rebase onto fetched dev `38f7fe63d4f476cf593e06b09a2385d003233c53` completed at
local `208660c3fdc4f8068b00e780bb78bad08e85585c`. Two conflicts preserved the
new Tab bindings and both independently appended testing lesson sets.

Complete incoming qualification initially reports **331 passed / 5 failed**
(144.11 s, `/private/tmp/pr2427-wave9-incoming.log`). Three fake audio.cpp
child-server cases cannot bind loopback ports in the sandbox; the complete two
TTS files pass **117** cases (3.19 s) outside that restriction, without models,
devices or external inference (`/private/tmp/pr2427-wave9-tts-local.log`).
The navigation allowlist erroneously inherits Shift+F6 as universal, despite
the retained route gate. Only Tab/Shift+Tab are universal; that test-only
correction is independently diagnosed and its complete navigation file is
rerunning. Backlog IDs remain unique across **3652** files.

The remaining structural gate is real: LibraryScreen is **32156 / 31901 lines**
after the accepted upstream keyboard behavior, with seven added methods.
Preserving earlier delegate deletions does not earn that saving a second time.
Inspect current unused delegates before proposing any new extraction; retain
all seven new keyboard methods and do not raise caps. Final-source native
Library verification and final-head Qodo/CI are still required before merge.

The corrected complete navigation file passes **145** cases (105.70 s,
`/private/tmp/pr2427-wave9-navigation-final.log`). Together with the complete
incoming run and unrestricted offline TTS rerun, all incoming behavioral test
files now pass; the Screen structural gate remains open at **32156 lines and
1236 methods**, versus **31901 / 1229**. Current-reference review found only
four safely dead helpers (53 lines), insufficient by itself. Do not conflate
prior extraction savings with new savings or remove the newly accepted keyboard
behavior. A bounded follow-on controller-boundary reduction requires an explicit
plan/approval before implementation. Existing native evidence is historical
until the final source is requalified; no merge-ready claim is made here.

Step 68 resolved the apparent design blocker without a new owner: the complete
reference census identified exactly three obsolete compatibility forwarders
whose few test/evidence callers can use their existing Media/Prompts/Notes
controllers directly. Together with four genuinely unused helpers, this is
mechanical cleanup within the approved reconciliation scope, not an additional
controller design. Controller-callable pins remain; explicit Screen-absence
pins now cover the removed forwarders. Independent review approves exact
scope/callers and docstring contract retention; its stale-census-comment finding
is corrected (Media23, Prompts56, Notes27). No new approval is needed for an
owner extraction because none was introduced.

Final measurements are **31868 lines / 1229 methods** for LibraryScreen
(line ceiling tightened from31901), and **5274 / 5276 lines** for NotesController
(the pre-increment ceiling is preserved, not the incoming5283 increase).
AST normalization proves only the exact seven removals, their two newly unused
imports and documentation changed (`/private/tmp/pr2427-wave9-ast-proof.log`).
Fresh complete ownership/size guards pass **92** cases (1.69 s), startup/import/
CSS/size gates pass **76** (26.58 s), and all six preflight checks pass. Scoped
changed-test/Notes lint and undefined-name checks pass; the Screen retains
its same **47 pre-existing Ruff findings**, with no new finding (the two newly
unused imports were removed). These are not reported as a clean whole-Screen
lint run.

Two final-source runs are active at this checkpoint: the **530-case** complete
affected owner/Media-preview/Notes-reader/Prompts/keyboard cohort at
`/private/tmp/pr2427-wave9-owner-green.log`, and the **851-case** complete Library
shell at `/private/tmp/pr2427-wave9-shell-native.l37Yp9/pytest.log`. Its native
report is `native.jsonl` in the same directory; require final exit_code0 and
empty sqlite_paths. Runtime source is frozen until these finish. Final-head
Qodo review and GitHub checks remain mandatory before a normal merge.

The complete affected owner/UI cohort finished **530 passed, 3 warnings in
445.47 seconds**. Qodo completed its fresh review of
`bf1f70cf0c6f153ef9163189385982cca568744d` at 2026-09-09 02:31 UTC with **0 bugs,
0 rule violations and 0 skill insights** (canonical review comment5560973127;
exact-head completion comment5594878752). The 851-case native run is still in
progress; no final cleanup or merge claim is made. Newer dev changes must be
integrated only after the frozen-source run completes, followed by affected-file
verification and review of any resulting runtime changes.

The final frozen-source Library run finished **851 passed, 10 warnings in
1371.47 seconds**, with native final `exit_code: 0` and `sqlite_paths: []` in
`/private/tmp/pr2427-wave9-shell-native.l37Yp9/native.jsonl`. This qualifies the
published runtime at `bf1f70cf0c`; its Qodo review is also complete and all 13
review threads are resolved. Fresh fetch finds dev `80f29a9a1dcd9307662714233c65605e4c517b11`,
containing 51 changed files since the prior base, including Library loaders,
handoff, recovery copy, Media polish and their tests. Integrate and qualify
that increment next; do not reuse this source qualification for changed runtime.

### Wave ten: dev80f29 integration and bounded forwarding cleanup (2026-09-09 UTC)

Rebase completed at `2c7e99f60c` onto `80f29a9a1dcd9307662714233c65605e4c517b11`.
The prior checkpoint remains recoverable through
`codex/pr2427-before-wave10-rebase-20260908`. Conflict resolution retains the
new conversation-workspace callback in the existing assembly helper and the
Media deep-link Reader-width restoration after the reviewed surface-settlement
and supersession fence. The new diagnostic is a fixed-label workspace-link
warning with exception capture, permitted by ADR-029's 2026-09-08 application
log-readability amendment. Its statement was explicitly reviewed; the rebuilt
inventory already matches exactly (600 owners, 7668 TASK-494 calls), so no
blind regeneration or logging-policy change was needed.

Step 70 reproduced Screen growth to 32193 lines / 1238 methods. It preserves
all nine incoming methods and removes nine existing two-line forwarders:
Media row-owner adoption, presentation epoch, successful-focus eligibility,
settlement deadline, Trash post-paint focus, page-control focus, filter request;
Notes role target and work-first preferences. Their concrete Screen/test callers
now use the existing controller owners. Both shared `canvas_sync` receiver
methods remain intact: it accepts Screen OR controller, so direct controller
member access there would have broken that contract. The existing guard sets
retain controller-callable proof and gain explicit Screen-absence coverage.

Historical documentation was shortened without changing executable bodies.
Normalized AST comparison proves the exact nine removals/caller retargets are
the only runtime delta from the rebased head; all five edited controller/wiring
modules are executable-AST identical. Final measured Screen is 31853 lines /
1229 methods against unchanged 31868 / 1229 caps. Reader/ingest/media/notes/
prompts/wiring measure 908/2713/4645/5269/4978/338 lines, all within original caps.

Fresh evidence:

- Initial guard RED: 1 failed / 44 passed; extended owner RED: 7 failed /
  67 passed (five line caps and two newly extended absence controls).
- Complete six-file owner/size GREEN: **97 passed, 2 warnings, 2.04 seconds**,
  `/private/tmp/pr2427-wave10-owner-green.log`.
- Startup/import/CSS budget files: **8 passed, 5 warnings, 15.79 seconds**,
  `/private/tmp/pr2427-wave10-boot.log`.
- All six derived-artifact preflight checks passed:
  `/private/tmp/pr2427-wave10-preflight.log`.
- AST proof: `/private/tmp/pr2427-wave10-ast-proof.log`. F821 and diff-check
  pass. Ruff comparison finds no new findings; existing counts remain Screen
  47, ingest controller 6, Media controller 7. Other edited Python files are
  Ruff-clean (`/private/tmp/pr2427-wave10-lint-baseline.log`).

Complete incoming/affected UI verification is running (1427 collected cases)
in `/private/tmp/pr2427-wave10-incoming.log`. The separate native 851-case
Library run is running in `/private/tmp/pr2427-wave10-shell-native.lxPpFD/`;
require final exit code zero and empty SQLite paths, not intermediate snapshots.
The expanded controller/diagnostic cohort finished **183 passed, 8 warnings in
174.10 seconds** (`/private/tmp/pr2427-wave10-diagnostic-guards.log`). The two
UI/native runs are not yet final-pass claims. Runtime source stays frozen
while qualification runs.

A subsequent read-only remote check observed newer dev `8aa2211f2b78fcb35c9d1d3db6e15d5f1978da0a`
(PR #2524, structural-wait cancellation/deadlines; 16 changed files). Integrate
that accepted increment after the active run completes, then qualify its actual
affected files and any overlap. Fresh final-head Qodo review and normal merge
checks remain mandatory. No cap increase, repository-wide sweep, unrelated
environment change, or further user approval is pending.

### Wave-ten terminal results and fixture reconciliation (2026-09-09 UTC)

The frozen native run completed **849 passed, 2 failed, 10 warnings in
1375.09 seconds**, with final `exit_code: 1` and `sqlite_paths: []`. Both
failures were the two terminal sizes of
`test_library_conversation_disabled_reason_stale_actions_and_notice`, at its
unchanged `open_console.disabled is False` assertion. The fixture had no active
workspace/membership, now required by accepted TASK-32056. An independent
two-case run reproduced both failures in 5.26 seconds. Step 71 supplies a real
active workspace and membership for the retained `chat-001`, keeping all
stale-list, enabled-state, tooltip, Retry and focus assertions. The repaired
pair plus the complete handoff file passed **11 tests, 3 warnings in 7.95
seconds** (`/private/tmp/pr2427-wave10-conversation-green.log`); scoped Ruff
and diff-check passed. No production gate changed.

The 1427-case incoming run exited **143 (SIGTERM)** before a pytest summary,
with its log ending after the 60% marker. Its termination source is not
established; no pass count is inferred from progress dots. The subsequent
process census confirmed only the native runner was still active. This cohort
remains inconclusive and will be rerun as complete-file groups after integration.

Fresh fetch confirms next base `c4a7b1911f14181faa471eb1ea209cfdeed98226` (37
changed files since dev80f29). In addition to structural waits, it contains
Collections row/count behavior and documentation. The fixed-ref independent
review identifies ten safe existing-owner forwarders to offset ten incoming
Screen methods, plus documentation-only line paydown; shared receiver,
dynamic dispatch, identity and wiring contracts are excluded. Preserve those
boundaries and all caps during step 72. The native result above establishes
clean process exit resources, not an all-green test suite or final-source
qualification after the upcoming rebase.

### Wave-eleven rebase and bounded owner reconciliation (2026-09-09 UTC)

Rebased all 215 commits onto `c4a7b1911f14181faa471eb1ea209cfdeed98226`,
finishing at `947e6ec680e4add142b1d9ee888cd934820d6f37`. Backup
`codex/pr2427-before-wave11-rebase-20260908` preserves `0753440483`.
Conflict reconciliation retains upstream structural waits and Collections
count reads, uses canonical `receipt.focus_identity` with the single settled
focus queue inside the Notes receipt helper, and supplies the live composed
Skills import status accessor through existing controller wiring. The new
Collections diagnostic uses the existing bounded `_retry_failure_reason`.

Step 73 removes ten exact two-line Media/Notes Screen delegates and retargets
only verified real-Screen callers and affected tests to their existing owners.
Shared polymorphic `canvas_sync` receivers and dynamic/identity contracts stay
unchanged. Extended owner tests pin both controller callability and Screen
absence. Fifteen historical Screen docstrings and the Export success docstring
were shortened without changing executable bodies. SkillImport uses normal
Ruff formatting, shorter redundant documentation, and removal of an optional
tuple trailing comma; its public cancellation method retains Google-style
Returns documentation. No new ADR is required: this is routine reconciliation
within existing ownership boundaries, not a new interface or state design.

Measured final sizes are Screen **31842 lines / 1229 methods**, Export **1291
lines**, and SkillImport **760 lines**, within unchanged caps of 31868/1229,
1307, and 760. No ratchet was raised.

Fresh evidence on the frozen runtime source:

- Initial cap/assembly guards: **3 failed, 71 passed**; newly extended delegate
  absence controls: **2 failed, 25 passed**. Final seven-file owner/size/assembly
  cohort: **101 passed, 2 warnings, 2.00 seconds**
  (`/private/tmp/pr2427-wave11-guards-green.log`).
- All derived-artifact preflight checks passed
  (`/private/tmp/pr2427-wave11-preflight.log`). Complete final diagnostic and
  boot/import/CSS gates: **78 passed, 11 warnings, 191.13 seconds**, exit zero
  (`/private/tmp/pr2427-wave11-final-gates.log`).
- `/private/tmp/pr2427-wave11-ast-proof.log` confirms executable AST identity
  for Export/SkillImport and exactly the enumerated delegate removal/direct
  owner retargets for Screen and the two UI test files.
- Diff-check and scoped F821 pass. Fresh Ruff comparison has no new findings:
  Screen retains its existing 47; every other edited Python file has zero
  (`/private/tmp/pr2427-wave11-lint-baseline.log`).
- Independent bounded review found no actionable findings, conditional on
  the remaining final-source test/native and external merge gates.

The six incoming structural-wait/Collections/Export files initially passed
**46 tests, 3 warnings, 176.40 seconds** before step 73; those are not final
cleanup-source evidence. They are included in the active complete-file retries.
The previously interrupted incoming inventory is now split into sequential
cohorts A/B/C/D with separate logs/JUnit outputs at
`/private/tmp/pr2427-wave11-cohort-{a,b,c,d}.{log,xml}`. Later cohorts start only
after earlier success. Native Library qualification is also active at
`/private/tmp/pr2427-wave11-shell-native.blE2Qd/{pytest.log,native.jsonl}`.
Neither run is yet a final-pass claim; require the native terminal exit zero
and empty SQLite paths. Runtime source remains frozen during these runs.

At this checkpoint the published PR head is still `83f48123fcabcb0b00ae8291f36602dc47aee94d`;
publication, fresh final-head Qodo completion, latest-base reconciliation and
normal required checks remain open. No further approval is pending.

### Wave-eleven publication and first complete retry (2026-09-09 UTC)

The reviewed checkpoint is now published at
`6b28349df17308bf1483b99fe3b010ec711bc995`, confirmed remotely after an exact
force-with-lease against `83f48123fcabcb0b00ae8291f36602dc47aee94d`.
Manual Qodo request `5595442919` received fresh exact-head completion
`5595471395`. It reports zero bugs and one new transaction-rule finding
(`3964446586`), now under technical review; earlier resolved/dismissed entries
are historical, not new failures.

Complete retry cohort A finished **754 passed, 2 skipped, 4 warnings in
455.68 seconds**. JUnit confirms 756 cases, zero errors and zero failures.
The skips are the Windows spawn/resource-tracker boundary and the explicitly
deferred TASK-32070 Search/RAG keyboard case. Cohort B subsequently finished
**518 passed, 3 warnings in 445.64 seconds**. Cohort C and native qualification
remain active; D has not yet started. Source is still frozen.

The two complete resource-control files relevant to the new Qodo finding passed
**25 tests, 2 warnings in 2.83 seconds**, exit zero
(`/private/tmp/pr2427-wave11-qodo-borrowed.log`). The flagged raw SQLite setup
intentionally creates a caller-owned transaction which must remain open across
executor submission and drain. `ThreadDatabase` is a minimal real-SQLite double,
not the production transaction API; separate real-database lifecycle controls
already use `database.transaction()`. The production transaction manager also
explicitly documents native caller-owned transactions at managed depth zero.
Do not remove that borrowed-work sentinel merely to satisfy a blanket rule.
Independent assessment agreed that no fix is warranted. Thread reply
`3964470606` records the preserved ownership invariant and fresh 25-test
evidence; thread `PRRT_kwDOOcyyl86gfuKh` was then resolved. This was an
evidence-backed rejection of an inapplicable recommendation, not a code fix.

GitHub reports the published head DIRTY against newer dev. Reconcile the
actual new base only after the frozen qualification finishes; no normal merge
is yet authorized by the verification results.

### Wave-eleven terminal results and next integration (2026-09-09 UTC)

Native Library qualification ended **850 passed, 1 failed, 10 warnings in
1425.41 seconds**, with final `exit_code: 1` and `sqlite_paths: []`. The failure
was `test_library_media_initial_error_is_unknown_and_retry_is_unique`; its
diagnostic showed the expected type-filter focus, so another conjunct in the
readiness predicate failed. This establishes clean native resource retirement,
not a green suite. Cohort C ended **355 passed, 3 failed, 3 warnings in
441.69 seconds**: analysed-secondary floor width, Trash Back toolbar focus,
and the Skills status fake's missing real helper. The chained D did not start
after that failure; its separate complete-file run then passed **160 tests,
3 warnings in 63.18 seconds**.

The exact C failure rerun reproduced the floor and status failures while Trash
passed (**2 failed, 1 passed in 20.44 seconds**). The native Retry case passed
alone (**1 passed in 3.17 seconds**). Ten fresh-process repetitions of the
Retry/Trash pair each passed both cases (20 passes total); these isolated
passes do not explain or repair the intermittent full-run failures. The Retry
failure message now includes total, Retry count and error presence alongside
focus, without exposing error bodies or weakening the predicate. Evidence:
`/private/tmp/pr2427-wave11-{c-failures-red,native-failure-red,focus-1,focus-2,focus-3,focus-4,focus-5,focus-6,focus-7,focus-8,focus-9,focus-10}.log`.

Step 74 aligns the two reproduced fixtures with accepted upstream behavior.
The floor test pins resolved Items width 36 and actual canvas content width 32
(TASK-32060), retaining complete painted analysed text and no-ellipsis checks.
The Skills fake binds four real status/wait helpers with empty wait state,
retaining visible/hidden-row and no-recompose assertions. Both repaired cases
pass (**2 passed, 3 warnings in 5.10 seconds**), and all three edited Python
files are Ruff-clean; diff-check passes. Complete affected-file qualification
and the two intermittent failures remain open.

Fresh fetch after all frozen cohorts finished found dev
`733f7a628c064005d27bed8bbcf53aae01b5dd45`: PR #2526, 136 changed files,
including accepted independent Buddy ownership/schema 70 and scheduler startup
deferral. ADR-139 was read and governs the retained incoming behavior. The
Library runtime delta is one existing WorkspaceCreateModal persona-service
argument; broader app/Console/schema/CSS changes still require affected offline
qualification and diagnostic-delta review. Preserve current work before rebasing
the exact reviewed range onto this base; do not raise ratchets or reinterpret
upstream explicit target/approval/ownership rules.

### Wave-twelve rebase and initial qualification (2026-09-09 UTC)

Saved step 74 at `c64eefd369`, with independent review finding no actionable
issues in the narrow fixture/diagnostic changes. Backup
`codex/pr2427-before-wave12-rebase-20260909` preserves that checkpoint.
All 217 review commits were rebased onto `733f7a628c064005d27bed8bbcf53aae01b5dd45`,
finishing at `be62ee0fcb924055f14e26e6fbc3e86a526cb233`.

Reconciliation preserves both upstream lessons and reviewed fixture lessons;
the index lesson remains in its relocated section. Hydration keeps the
upstream legacy-wrapper test and the distinct canonical complete-snapshot test,
with upstream canonical durable-settings assertions and the reviewed shared
exact-owner cleanup fixture. Independent conflict review confirmed that union.
Console suspend retains the incoming visibility notification before the
existing settings-navigation owner's claim release. The Workspace migration
conflict was formatting-only. No source limits were raised: Library Screen
is 31843 lines / 1229 methods after the single incoming WorkspaceCreateModal
persona-service argument.

Incoming diagnostic additions were reviewed: bounded constant messages for
default-Persona notice failure, continuation/generation hydration fallback,
Buddy publication cleanup and app migration; removed/moved statements retain
their existing privacy boundaries. The combined Session entry is measured at
26 calls with digest `a71888e4c6bcd258f247`. The unmodified generator verifies
**601 owners, 1351 TASK-492 calls, 55 TASK-31551 calls, 7665 TASK-494 calls and
12 sink files** (`/private/tmp/pr2427-wave12-inventory.log`).

Initial verification: **218 passed, 3 warnings in 32.07 seconds** across the
three complete conflict-affected hydration/Workspace test files, complete
canvas-scoped sync, and six Library size/owner/assembly files
(`/private/tmp/pr2427-wave12-guards.log`). Scoped Ruff and diff-check pass.
All derived-artifact preflight checks pass, including 3667 unique task records,
115 declared ChaChaNotes tables and 282 indexed names / 67 plan pins
(`/private/tmp/pr2427-wave12-preflight.log`). An earlier command named a
nonexistent size-test path and exited 4 without collecting tests; the corrected
complete command above is the evidence, not that invocation.

Runtime source is frozen for active verification:

- Incoming dev's 43 changed test files are covered by the three completed
  conflict files plus four sequential offline groups in
  `/private/tmp/pr2427-wave12-incoming-{1,2,3,4}.{log,xml}`. Each group has its
  own terminal evidence; later groups run even if an earlier group fails.
- Complete Media render/Trash/return-settlement files run in
  `/private/tmp/pr2427-wave12-media.{log,xml}`.
- Native complete Library Shell runs in
  `/private/tmp/pr2427-wave12-shell-native.YbMd42/{pytest.log,native.jsonl}`;
  require final exit zero and empty SQLite paths.
- Final diagnostic/startup/import/CSS tests run in
  `/private/tmp/pr2427-wave12-final-gates.log`.

These are pending results. Wave-eleven's two intermittent full-run failures
are not declared fixed by their isolated/repeated passes. Publication, fresh
head review, latest-base check and normal protected merge remain open.

Incoming group 1 subsequently ended **305 passed, 3 failed, 2 warnings in
79.40 seconds**. All failures are the three early-exit parameters in
`Tests/App/test_scheduler_startup_deferral.py`: quit/shutdown observe a Console
tab-strip `MountError`, while setup_failure does not raise the injected
`RuntimeError`. Their cause is not yet established; an exact three-case
reproduction is running in `/private/tmp/pr2427-wave12-startup-red.log`.
Group 2 continues independently. The final diagnostic cohort has recorded a
failure but has not yet emitted its final traceback/summary. Keep all these
gates open; runtime source remains unchanged for active verification.

Further terminal wave-twelve results:

- Exact startup reproduction: **3 failed, 2 warnings in 6.83 seconds**.
- Incoming group 2: **429 passed, 2 failed, 3 warnings in 61.13 seconds**.
  Both failures are bridge live/resume marker text equality for the controller
  refusal string. Exact reproduction: **2 failed, 1 passed, 273 deselected,
  2 warnings in 1.02 seconds** (`/private/tmp/pr2427-wave12-bridge-red.log`).
  `_step_record` intentionally uses `redact_log_line`; its sensitive `user:`
  assignment match redacts this payload on persistence and records the result
  field as redacted. Resumed markers therefore omit detail. Structured success
  versus blocked outcomes remain correct. Do not weaken durable redaction to
  recover stale byte-for-byte live/resume expectations.
- Incoming group 3: **86 passed, 5 warnings in 86.29 seconds**.
- Complete Media cohort: **258 passed, 1 failed, 3 warnings in 320.14 seconds**.
  `test_local_reader_chrome_stops_before_the_sixth_row` read an empty painted
  body after loaded-id settlement; its isolated retry passed (**1 passed,
  3 warnings in 3.43 seconds**, `/private/tmp/pr2427-wave12-paint-red.log`).
  The helper currently waits for request settlement and one pilot pause,
  not a rendered first content line. Root-cause confirmation remains open.
  The previously failing Trash focus case passed in this complete cohort.
- Final diagnostic/boot cohort: **77 passed, 1 failed, 11 warnings in
  187.87 seconds**. The metadata-only test still requires two Session startup
  diagnostics explicitly removed by upstream Buddy/default-settings changes;
  the generated inventory itself already verifies the accepted removals.
  Reconcile the specific obsolete entries and retain coverage for the new
  metadata-only default-resolution path, not the entire privacy guard.

All results above belong to unchanged runtime `be62ee0fcb`. Group 4 and the
native Library run remain active. No full-source pass or merged status is
claimed. The no-splash setup wrapper also explicitly catches and logs setup
exceptions, explaining why a test expecting propagation needs lifecycle-path
review; the two mount failures still need attribution before any repair.

### Wave-twelve terminal evidence and step-75 repairs (2026-09-09 UTC)

The frozen native Library run finished **851 passed, 10 warnings in 1431.36s**,
exit 0 and `sqlite_paths: []`; evidence is
`/private/tmp/pr2427-wave12-shell-native.YbMd42/{pytest.log,native.jsonl}`.
Incoming group 4 finished **641 passed, 2 failed, 9 warnings in 618.51s**:
the headless approval and Personas unchanged-resize fixtures.

Step 75 repairs preserve existing ownership and privacy boundaries:

- Tab replacement previously attempted six mounts after real strip removal.
  The deterministic RED is `/private/tmp/pr2427-wave12-tab-red3.log`; the
  live-MountError negative control already passed. Native batch removal,
  exact `is_attached` admission and one batch mount eliminate awaited gaps
  while preserving order and churn. `is_mounted` remains true after removal
  and is deliberately not the guard.
- Startup then exposed the separate queued legacy-workspace-alias worker
  mounting into a detached tray (`/private/tmp/pr2427-wave12-startup-trace.log`).
  That synchronous owner now checks its exact tray before mounting; neither
  repair catches or suppresses live mount errors. The setup-failure fixture
  observes the existing no-splash logged suppression, with deterministic sink
  cleanup and unchanged no-readiness/no-scheduler assertions.
- The complete tab/startup/workspace-rail cohort passed **72 tests, 3 warnings
  in 72.72s** (`/private/tmp/pr2427-wave12-startup-owners.log`). The final
  sink-cleanup form was also covered by the subsequent startup/reuse cohort.
- Bridge fixtures now pin explicit redacted durable text and unchanged
  structured success/blocked outcomes. No redactor or security policy changed.
  Two obsolete upstream-deleted diagnostic pins were replaced by the current
  constant-only default-Persona notice pin. Personas' unchanged-state guard
  covers pane/workbench queries, not the accepted dynamic conversation rows.
  The combined complete files initially produced **759 passed / 1 failed**;
  the sole remaining failure was a new over-strong short-output assertion:
  complete inline output intentionally has no duplicate expansion affordance.
  After correction, the complete bridge and headless files passed **294 tests,
  3 warnings in 78.90s** (`/private/tmp/pr2427-wave12-headless-bridge-final.log`).
- Headless approval now seeds the exact installed/cache-owned startup Console,
  rather than stacking an ad-hoc screen that competes for the runtime. After
  ordinary navigation, the existing exact-owner uninstall helper supplies true
  teardown. Shutdown-event, detached seams, app-wide painted toast, poll-budget
  survival, new-screen/card identity and human-verdict assertions remain.
  Production retained-navigation policy is unchanged. The prior helper attempt
  correctly failed on the unregistered screen; no precondition was weakened.
- A controlled delayed first Raw index reproduced the same empty painted
  reader body while detail was already settled. The geometry test now waits
  boundedly for real first-line paint, retaining all chrome/title/content
  assertions. Both ordinary/delayed variants and the complete file pass:
  **124 passed, 3 warnings in 168.70s**
  (`/private/tmp/pr2427-wave12-paint-green.log`). Production rendering is unchanged.
- Qodo comments 3964725128/3964725134 on head565d request helper contracts.
  `ThreadDatabase` now documents and annotates connection ownership, injected
  post-close failure and cleanup; complete helper file **10 passed, 2 warnings**
  (`/private/tmp/pr2427-wave12-qodo-helper.log`).

Fresh derived-artifact preflight passes
(`/private/tmp/pr2427-wave12-repair-preflight.log`). Ruff passes every changed
test and the session surface. ChatScreen has exactly its three pre-existing
Ruff diagnostics (same codes/messages before and after); no new diagnostic.
Whitespace checks pass. Independent review found no issue in the two production
guards; final fixture review is being recorded separately. This is not a merge
claim: fetched dev has advanced to `2e3389e694` (Library crit8 and offline TTS/MCP
changes), so the next integration/qualification remains necessary.

Final independent fixture review found no blocking issue. Its sole P3 was a
stale headless-test docstring claiming the retired Leave Console dialog; that
description now matches ordinary navigation plus explicit exact-owner removal.

Wave-thirteen plan: checkpoint the verified step-75 repairs, then rebase onto
fetched `2e3389e694` preserving incoming Library crit8, Notes editor/refresh,
Kokoro runtime, Buddy speech cancellation and MCP teardown behavior. Reconcile
conflicts at existing owners, retain all ratchets and privacy gates, run complete
changed offline files and appropriate architecture/import/preflight gates, then
final-source native Library verification and exact-head review before normal
merge. No model downloads, real inference/audio/device tests or repo-wide sweep.
ADR required: no new ADR. ADR path: `backlog/decisions/140-official-kokoro-pytorch-runtime.md`
for the incoming accepted runtime; existing Library decisions remain authoritative.
Reason: integration of upstream decisions and routine bounded reconciliation,
not a new architecture or relaxed cap.

### Wave-thirteen integration and bounded repairs (2026-09-09 UTC)

Rebased 220 commits onto `2e3389e694`, completing at
`59a5ad3826d9f8902a2be9d3264fe29b3665f7f7`. The conflicts were three independent
lesson unions and two generated inventory summaries; no production conflict.
The prior published checkpoint is `1d910d63d624e78f84a2fe806fdb8119eff0eee9`.

- Complete incoming offline TTS/DB/visibility files: **248 passed, 1 skipped
  (real MPS), 3 warnings in 13.18s** (`/private/tmp/pr2427-wave13-runtime.log`).
  No download, inference or audio/device test was run.
- Complete incoming UI cohort: **508 passed, 4 failed, 3 warnings in 665.14s**
  (`/private/tmp/pr2427-wave13-ui.log`). All four failures use obsolete Notes
  tree selectors; the retained Notes canvas uses `library-notes-row-0` for
  the same first fixture note. All five obsolete references were aligned,
  retaining the pane, focus, status and query assertions. An independent
  three-case RED reproduced the selector failure in 94.73s.
- Step 76 removed exactly eight private forwarding methods and retargeted
  their concrete callers to existing controllers. Absence controls failed
  before deletion (three Library cases and one Chat case). The full executable
  AST comparison against `59a5ad3826` passed after accounting only for these
  eight removals/call targets and stripping docstrings. All incoming behavior
  remains; Chat is **16,762 lines / 505 methods**, Library **31,865 / 1,229**,
  within unchanged caps. Historical documentation was shortened, not runtime
  behavior or cap enforcement.
- Complete architecture owner/ratchet files: **84 passed, 2 warnings in 1.98s**
  (`/private/tmp/pr2427-wave13-owner-green.log`). Complete crit8, character
  switcher, session-tab and scheduler repair files: **99 passed, 3 warnings
  in 84.68s** (`/private/tmp/pr2427-wave13-repair-green.log`).
- Both derived-artifact preflights pass; the later log is
  `/private/tmp/pr2427-wave13-final-preflight.log`. Ruff adds no diagnostics:
  Chat retains its three baseline diagnostics, Library its 47, crit8 its two
  pre-existing unused imports. Changed wiring and character-switcher files
  pass Ruff. Whitespace checks pass. Independent review found no runtime
  blocker; its stale census comments and import-spacing nits are corrected.

The final-source native run of all 851 Library shell cases is still active at
`/var/folders/p_/x47tgtn57cv43r7yxxn40tyh0000gn/T/pr2427-wave13-shell-native.p859l8`.
It observes native descriptors without connection patches or forced collection.
Production and shell-test bytes remain frozen. Two early failures are stale
Starter focus expectations: the incoming layout moved Chunking Lab into hidden
Details, so Tab wraps to the visible rail-collapse control. Both sizes reproduce
independently (**2 failed, 849 deselected in 5.16s**,
`/private/tmp/pr2427-wave13-starter-red.log`); step 77 records the bounded fixture
repair to apply after the run. The final descriptor result is not yet known.
This checkpoint is not merge-ready; native completion, those repairs, current
head/base review and required checks remain open.

Checkpoint refresh: the architecture files pass again (**84 passed, 2 warnings
in 1.99s**), reviewed test files pass Ruff, and derived-artifact preflight passes
with 3,672 unique Backlog task records. The running native cohort has reported
additional failures beyond the two Starter cases; their terminal tracebacks
remain to be read before assigning causes. A fresh Git fetch and the live dev
branch API both still resolve `2e3389e694`; the PR payload's older `baseRefOid`
is not evidence that dev advanced, so no additional rebase is justified yet.

### Wave-thirteen terminal repairs and final qualification (2026-09-09 UTC)

The first native run completed with **830 passed, 21 failed, 10 warnings in
1407.71s**, exit 1, and final **`sqlite_paths: []`**. Its source was the wave-thirteen
integration checkpoint; the terminal log and native record remain in the directory
above. The zero-descriptor result does not turn failing tests into a passing run.

- Starter wraparound and compact Notes allocation pins were stale after TASK-32064
  moved Chunking Lab into Details. Exactly one compact row returns to flexible
  content. Updated pins retain total height, fixed controls, paint, navigation,
  authority, zero-work seams and exact six-row surplus growth.
- The three-cell Clear button clipped the Library search placeholder. Reclaimed
  redundant padding and the doubled joined border within the same rail width;
  an equally specific focus rule preserves visible focus. Mounted tests cover
  both widths, full placeholder paint, adjacency, Clear visibility and keyboard
  clear/refocus behavior. Both clipping and focus styling failed before repair.
- Wide Work status deduplication had also removed the sole compact storage
  authority. Compact Work now reuses its existing authority prefix; responsive
  transitions update the existing node in place. Three compact-wide-compact
  create/loading/editor controls failed before repair. Loading still omits an
  invented Next instruction; obsolete test expectations were aligned without
  removing status, identity, error or recovery assertions.
- A controlled delayed filter projection proved that filter records precede
  settled DOM/focus. The Back fixture now waits for the actual filtered list and
  focus. A subsequent failure still occurred: stack traces showed Back restoring
  row n-18 before an older queued callback restored the filter and zero scroll.
  Notes automatic restoration now preserves a newer mounted non-grip focus owner,
  following existing Media behavior. Deterministic controls retain missing/grip
  fallback; exact Back identity, placement, filter, sort and scroll pins remain.
  Temporary stack instrumentation is removed. Notes stays at its original
  **5,276-line cap**.
- Qodo comments 3965201343 and 3965201350 are addressed by contiguous local imports
  and split-JSON test annotations/documentation, respectively.

Current qualification: **28 passed, 828 deselected, 3 warnings in 93.58s** for the
complete repaired Library case group (`/private/tmp/pr2427-wave13-all-repair.log`);
**129 passed, 3 warnings in 42.88s** across complete Notes canvas, crit8, owner
wiring and both ratchet files (`/private/tmp/pr2427-wave13-owner-final.log`);
**16 passed, 2 warnings in 0.91s** for both Qodo-affected files
(`/private/tmp/pr2427-wave13-qodo.log`). Changed Python files pass scoped Ruff;
whitespace checks and derived-artifact preflight pass
(`/private/tmp/pr2427-wave13-checkpoint-final-preflight.log`). Independent final
review found no blocker in these repairs or their assertions.

ADR required: no. ADR path: N/A. Reason: routine lifecycle/layout regression
repairs and fixture alignment within accepted contracts, with no new boundary,
dependency, privacy change or relaxed cap. The complete **856-case** native
Library rerun on the repaired checkpoint, current-head Qodo review, live-base
freshness and required GitHub checks remain merge gates. This is not a merge claim.

### Wave-thirteen final native result and current review (2026-09-09 UTC)

The complete repaired-source native run at `acaa89025033fd8cde2a9b8dbcf24d9539f875ac`
finished **855 passed, 1 failed, 10 warnings in 1389.30s**, exit 1, final
**`sqlite_paths: []`**. Source identity and terminal evidence are in
`/var/folders/p_/x47tgtn57cv43r7yxxn40tyh0000gn/T/pr2427-wave13-final-native.UHGtuL`.
No connection patches or forced garbage collection were used; the source freeze
has ended. This is not a passing complete-file result.

The sole failure, `test_library_note_wide_deep_link_back_clears_explicit_intent`,
also fails independently (**1 failed in 3.56s**,
`/private/tmp/pr2427-wave13-deeplink-red.log`). It expected Escape to focus the
Notes filter and a later resize to replay the Notes stage. The existing action
explicitly returns to the selected Library row and clears the stage intent,
consistent with ADR-086 and the existing compact Back tests. The previous late
automatic filter restore had concealed that contract. Corrected the wide focus
and subsequent compact rail/visibility assertions, retaining exact targets and
adding explicit Notes reactivation proof. No Library production change was made.
The full deep-link/Back/late-focus/compact-stage group passes **13 tests, 843
deselected, 3 warnings in 15.03s**
(`/private/tmp/pr2427-wave13-deeplink-cohort.log`).

Qodo posted five new findings on this head: export canonical-path validation,
real requests-stack coverage, row-actions import separation and parameter docs,
and roleplay plan lease cleanup when durability admission is closed. Step 83
records bounded RED/GREEN work; verification and published replies remain open.
The required derived-artifact GitHub check passed on `acaa890250`, but fresh dev
advanced to `9faf96d9f86b98916a54d7a895e3c0411fefc912`: Buddy qualification documents,
artifacts, a test file and its task, with no production changes. That upstream
task reuses the unmerged Loguru review task's 32108 ID. Preserve the landed Buddy
ID; re-sweep and move only the review task before the next publication, following
the existing user-authorized review-only renumbering policy. No merge claim yet.

Publication identity census: all 1,027 local/remote refs and all registered
worktrees agree on maximum TASK-32135. The review-only Loguru record moves
32108 to **32136**, with its original dates, completed notes and append-only
renumbering provenance retained. The landed Buddy TASK-32108 and its literal
artifact paths remain untouched; historical report mappings remain provenance.

Current-head Qodo repairs:

- Markdown export retains existing lexical rejection, then validates the chosen
  filename against its explicitly selected parent with the central canonical
  validator. Both mkdir and file opening receive the returned path. Hidden
  destinations remain permitted as before; stable final symlinks outside the
  chosen parent are rejected. This is not race-proof filesystem confinement.
  Real-file RED: **7 failed, 3 passed**
  (`/private/tmp/pr2427-export-red.r408cv/results-corrected.xml`); complete new
  export controls plus conversation action menu: **28 passed, 3 warnings in
  39.77s** (`/private/tmp/pr2427-export-green.qp9kh7/results.xml`). The two
  row-actions import/docstring findings are also corrected.
- Rejected roleplay durability admission now abandons that exact prepared plan
  and clears inflight repair state only if it owns the same plan. The two RED
  controls found the leaked token; an intermediate one-pass/one-fail control
  separately proved matching repair-state cleanup. Both final controls pass
  while preserving and successfully persisting a second session's unrelated
  plan. Complete session-settings, durability-controller and owner files:
  **437 passed, 3 warnings in 247.67s**
  (`/private/tmp/pr2427-roleplay-lease-full.log`).
- Requests factory timeout coverage now uses an actual loopback HTTP server,
  real session/request stack, omitted and explicit timeouts, and deterministic
  session/server/thread cleanup. A controlled factory-bypass mutation fails;
  the complete trust file passes **39 tests**. HTTP proves transport/factory
  behavior, not TLS certificate validation, which remains separately tested.
  There is no existing external production caller of this factory; no new
  caller was invented just to meet the review's wording.

Complete owner/ratchet group: **134 passed, 3 warnings in 6.61s**
(`/private/tmp/pr2427-qodo-new-owners.log`). Complete derived-checker tests:
**74 passed, 2 warnings in 0.77s**
(`/private/tmp/pr2427-wave14-id-checkers.log`). Derived preflight passes
(`/private/tmp/pr2427-qodo-new-preflight.log`). Independent cross-review found
no blocker in the export, exact-plan cleanup, real transport test or corrected
deep-link assertions. Latest-dev integration and final-head qualification remain.

Final privacy review found that the central validator's default logging would
add selected/resolved path details to rejection logs. An active real Loguru
capture canary proves that regression (**10 passed, 1 failed**); enabling its
existing `redact_paths=True` option preserves bounded diagnostics and a safe
user notification without changing the accepted path set. Complete export
controls now **11 passed, 2 warnings**
(`/private/tmp/pr2427-export-privacy.T4B39H/{red,green}.xml`). The trust test's
server context also owns the socket if thread startup fails; final complete
trust file **39 passed, 2 warnings in 1.44s**
(`/private/tmp/pr2427-tls-trust-final.log`). Final roleplay controls after their
parameter-docstring addition **2 passed, 417 deselected, 3 warnings in 2.90s**
(`/private/tmp/pr2427-roleplay-lease-final.log`). Scoped Ruff and whitespace
checks pass on all changed Python files.

### Wave-fourteen publication and new upstream churn (2026-09-09 UTC)

Saved the fixes at `8ad6d238a6`, retained backup branch
`codex/pr2427-before-wave14-rebase-20260909`, and rebased all 223 commits onto
`9faf96d9f86b98916a54d7a895e3c0411fefc912` without conflicts. Published head is
**`3fe94c842e0cead1720465dfc25598abd8d42a72`**, using an exact force-with-lease
against the prior `acaa890250` head. Rebase comparison proves production files
and `Tests/UI/test_library_shell.py` are byte-identical to the saved repairs.
The complete incoming Buddy qualification file passes **16 tests, 5 warnings
in 30.03s** (`/private/tmp/pr2427-wave14-buddy.log`); fresh derived preflight
passes (`/private/tmp/pr2427-wave14-preflight.log`).

All five Qodo findings were replied to with evidence and resolved on the
published head: replies 3965798157/3965798334/3965798522/3965798731/3965798948.
New exact-head review was requested by issue comment 5598000794. Native complete
Library qualification is running with frozen production/shell-test bytes in
`/var/folders/p_/x47tgtn57cv43r7yxxn40tyh0000gn/T/pr2427-wave14-native.Tem3Nf`.

After publication, live dev advanced again to
`a36fc6133c69f77b8b14596a59918de34261f8ef`. This is substantial Canvas Mermaid,
SQLite helper and TTS ownership work (202 files), not the preceding docs-only
Buddy change. It is fetched for read-only conflict/scope analysis while the
native run finishes. No rebase or runtime mutation has been attempted against
this new base yet. Its accepted decisions, overlap and offline qualification
must be reviewed before integration. The PR remains unmerged.

Read-only integration review and main-agent ADR reading are complete; task step
87 records the constraints before any new-base implementation. Main read ADR-124,
ADR-125 and the character-TTS ADR-028 amendment in full. Canvas Screen methods
already moved to `ConsoleMessageController` must receive incoming language and
served-profile fallback changes at that controller, not be restored on Screen.
The incoming compiler test doubles add snapshot/parent-profile keywords; browser
fixtures must retain controller targets while accepting upstream readiness and
profile identity. The bounded gateway JSON read remains necessary.

SQLite upstream already includes trace-maintenance/legacy-recovery registrations,
C55/C56 inventory and five backup-census controls. Keep them once, along with
C48 retirement and helper/native-probe seams. Preserve its exact quiescent
connection factory, lazy TTS ownership/close latch and Python 3.12 floor. Local
semaphore failures are not permission to repair the host or relax the upstream
lock, helper-capacity, terminal-retention or native-close contracts. The incoming
Mermaid preflight downloads pinned inputs by default: inspect its scope and use
an existing verified input cache where possible rather than silently skipping
asset integrity. Browser and platform evidence remain distinct from offline tests.

Current-head Qodo has one new documentation-only finding, comment3965835933,
thread `PRRT_kwDOOcyyl86gjNh5`, covering both public Evals import-closure tests.
Step 86 adds complete fixture/parameter types, return annotations and Google Args,
without changing production or the running Library test. Complete packaging file
**4 passed, 2 warnings in 2.16s**, scoped Ruff and whitespace checks pass
(`/private/tmp/pr2427-wave14-evals-docs.log`). Fix publication/reply remains pending
the next checkpoint; do not resolve the thread before publishing it.

Wave-fourteen terminal native result on `3fe94c842e`: **856 passed, 10 warnings
in 1313.32s**, exit 0, final **`sqlite_paths: []`**. Both pytest and native
observer completed successfully; the directory above holds exact source identity
and terminal evidence. No connection monkeypatches or forced collection were
used. The source freeze is now lifted. This qualifies that checkpoint, not the
new Canvas/SQLite base still awaiting integration.

### Wave-fifteen rebase and current-source qualification (2026-09-09 UTC)

Saved checkpoint `54ac4102263400aa58e4adb9c29b8c9f1cb39984` is retained by
`codex/pr2427-before-wave15-rebase-20260909`. The 224-commit rebase onto
`a36fc6133c69f77b8b14596a59918de34261f8ef` completed at
`cc6661100ab26fa2cb918994525d7b59aa23ff47`. Published head remains
`3fe94c842e0cead1720465dfc25598abd8d42a72`; that exact lease is required for
publication. The PR is open and not yet merge-qualified.

Conflict resolution retained upstream SQLite registrations, C48 retirement,
helper/native-probe restrictions and all five backup controls once. The obsolete
duplicate ChaChaNotes factory dictionary entry was removed in favor of upstream
`base_db._QuiescentSQLiteConnection`. Incoming TTS shutdown coverage was retained
with its consistent owner labels and meeting-session initialization. Production
DB/TTS files have no difference from this upstream base.

Canvas import language identity and served-mode failure behavior were transferred
to the existing `ConsoleMessageController`, without restoring the removed Screen
methods. The browser child keeps the upstream exact-card readiness wait and
button while calling the controller owner. Profile/snapshot fences and compiler
keyword contracts remain. AST comparisons verified the moved bodies against
upstream with only the existing owner/recovery-hook substitutions.

Current-source completed checks:

- All derived-artifact checks pass, including exact reproduction of all six
  Mermaid outputs from eight size/hash-verified public source inputs. Cache:
  `/var/folders/p_/x47tgtn57cv43r7yxxn40tyh0000gn/T/pr2427-wave15-mermaid-cqfbrd0w/inputs`.
  Log: `/private/tmp/pr2427-wave15-preflight.log`.
- Complete startup/size/private-delegate/moved-seam/packaging gate selection:
  **166 passed, 5 warnings in 39.32s**, exit 0. Preimport measured 498/500
  modules; parsed boot CSS measured 803716/804000 bytes. No limits changed.
  Log: `/private/tmp/pr2427-wave15-gates.log`.
- Complete SQLite/trace/legacy-recovery cohort: **679 passed, 2 Windows-only
  skips, 2 warnings in 100.67s**. Log:
  `/private/tmp/pr2427-cc6661100a-db-offline.log`.

Remaining current-source runs are active: complete Canvas offline files,
TTS/helper packaging, affected real Chromium served-flow/readiness files, and
native complete Library qualification. The Library run uses the unchanged native
observer without connection patches or forced GC in
`/var/folders/p_/x47tgtn57cv43r7yxxn40tyh0000gn/T/pr2427-wave15-native.MyCbEo`.
Production and test bytes remain frozen during these runs. Initial sandbox
Canvas failures must be classified from their completed output before any repair;
an environment-restricted run is not application-failure proof. Review, exact-head
publication and normal merge remain open.

Wave-fifteen follow-up evidence:

- Real installed Chromium, complete served-flow and startup-readiness files:
  **65 passed, 3 warnings in 195.75s**, exit 0
  (`/private/tmp/pr2427-wave15-browser.log`). Providers were scripted and servers
  fixture-owned loopback; no model or browser download was needed.
- Independent integration review found no actionable issue in the moved Canvas
  behavior, current SQLite census/factory, or app TTS ownership. This does not
  substitute for unfinished test/review/merge gates.
- Initial sandbox Canvas run: **87 failed, 972 passed, 10 skipped**. The complete
  failure census identifies loopback-bind denial directly or via the gateway's
  bounded startup error. A full escalated rerun is in progress with verified
  Mermaid inputs. Four QuickJS/esbuild archives were also acquired with the
  existing SHA-512-integrity-checked downloader at
  `/var/folders/p_/x47tgtn57cv43r7yxxn40tyh0000gn/T/pr2427-wave15-quickjs-pztpkqfs`
  to enable the runtime reproduction test instead of leaving it cache-skipped.
- TTS cohort: **993 passed, 14 failed, 2 skipped in 197.67s**. All fourteen
  actual-app child cases receive the non-JSON line
  `--- _setup_logging START (from Logging_Config.py) ---` on their phase pipe.
  The isolated `--showlocals` reproduction confirms this exact line; it is not
  an empty readiness message or a demonstrated semaphore failure. Logs:
  `/private/tmp/pr2427-cc6661100a-tts-offline.log` and
  `/private/tmp/pr2427-cc6661100a-tts-phase-diagnostic.log`.
- Packaging/runtime-floor cohort: **43 passed, 1 failed**. The failing oracle
  requires `3.12` literally for the derived-assets job, which accepted upstream
  deliberately pins to `3.12.11` for reproducible Mermaid generation. The exact
  local SQLite `[live]` case passed separately without enabling provider access.

Task step 88 records the minimal test-only repairs before implementation. Only
the two isolated TTS-child/runtime-floor test files may change while the native
Library/Canvas runs continue; production, shared fixtures and those active runs'
test bytes remain frozen. Strict phase parsing, foreign-cohort assertions,
timeouts, runtime floor and immutable version pin are not relaxed.

The complete escalated Canvas cohort passed **1068 tests, 1 cache skip in
197.90s**; the complete runtime-assets follow-up with both caches passed
**39 tests, no skips in 6.47s**. All 1069 distinct selected cases were exercised
successfully. Evidence directories:
`/var/folders/p_/x47tgtn57cv43r7yxxn40tyh0000gn/T/pr2427-canvas-loopback.kG7FJu`
and `/var/folders/p_/x47tgtn57cv43r7yxxn40tyh0000gn/T/pr2427-canvas-assets.YvQ5Tw`.
The 87 sandbox failures disappeared with the required loopback permission.

Step 88's first stdout redirection still failed all fourteen child cases.
Installed Textual saves `sys.__stdout__` in `app._original_stdout` and forwards
headless prints to that saved stream. The fixture now redirects that app-owned
diagnostic stream before mounting as well. All **14 child cases passed in
64.08s**, preserving strict JSON phases and real foreign-file/lifecycle checks
(`/private/tmp/pr2427-step88-children-green.log`). The complete TTS aggregate is
being rerun on these frozen test bytes. The complete packaging/runtime-floor
group is now **44 passed, 2 warnings in 34.68s**
(`/private/tmp/pr2427-step88-packaging-green.log`). Scoped Ruff passes on both
test files, and the refined fixture change received independent review. Fresh
derived preflight also passes (`/private/tmp/pr2427-wave15-final-preflight.log`).

The final complete eleven-file TTS aggregate now passes: **1007 passed, 2
expected skips, 6 warnings in 195.88s**, exit 0
(`/private/tmp/pr2427-step88-tts-full-green.log`). The separately authorized
local SQLite `[live]` case also passed; none of the original spawned helper
cases failed with semaphore errors on this run. Both test repairs are verified
without modifying production, shared fixtures, helper authority or CI pins.
Live remote dev still equals `a36fc6133c`; the published PR lease remains
`3fe94c842e`. Native Library qualification is still active and is explicitly
not included in these completed results. This checkpoint may be published for
final-head review, but normal merge still requires its terminal cleanup result
and current remote review/check status.

Published the rebased checkpoint as
`69108f4d1dd5d498c65c1f79951a478b3fc36e45` with the exact lease against
`3fe94c842e`. Qodo documentation comment 3965835933 was replied to at
3966255799 and resolved; final-head review was requested by issue comment
5598798465. GitHub now schedules the normal PR checks (including PR Fast Lane);
the merge state is blocked while those checks run, not a rebase conflict.

The active native Library run has reported one failure around 67 percent.
Its traceback and final descriptor snapshot are still pending. The observer
records retained handles after
`test_library_shell_blank_note_autosaved_then_emptied_still_gcs_on_back`, but
that attribution alone does not establish the failure cause or final cleanup
state. Do not merge or claim native qualification until the complete result is
investigated and any required repair is verified. Production and this Library
test remain byte-identical to the run's captured `cc6661100a` source.

Wave-fifteen native terminal result: **855 passed, 1 failed, 10 warnings in
1838.76s**, exit 1, final **`sqlite_paths: []`**. Session 65815 has completed;
the source freeze is lifted. The only failure is the final GC row-count assertion
at `test_library_shell_blank_note_autosaved_then_emptied_still_gcs_on_back`:
intermediate autosave succeeded, but the emptied session-created note survived
Back. Temporary per-test handle retention did not survive terminal cleanup.
Task step 90 starts controlled save-reply/GC-order investigation without relaxing
the original assertion or ADR-055's version-checked silent-GC guards. Main also
read ADR-027's serialized persistence/destructive-admission contract in full.

Qodo on `69108f4d1d` posted six test-contract documentation/type findings,
comments 3966299125/3966299137/3966299147/3966299157/3966299166/3966299174.
Task step 89 adds their missing contracts without changing executable test
bodies. The six complete files pass **185 tests, 5 warnings in 20.63s**
(`/private/tmp/pr2427-step89-docs-tests.log`); scoped Ruff and whitespace checks
pass. AST comparison after stripping annotations/docstrings confirms executable
behavior is unchanged. The worker-matrix docstring additionally explains callback
forms, literal exclusivity/group requirements and nested direct-await rejection.
Published as `538c36b9927fa1e24b494af891379d8d6a14ef96`. All six threads are
replied to and resolved (replies 3966470050, 3966480304, 3966480459,
3966480643, 3966480796, 3966480976). Final worker-docstring verification is
22 passed, 5 warnings in 7.39s. This closes these documentation findings,
not the remaining runtime qualification or final-head merge gates.

### Step 90/91: save-reply ordering and original-fixture reconciliation

Controlled real-SQLite RED held a successful save reply after its commit:
the database was version 2 while the coordinator still held version 1 and
`saving=True`. Back attempted version-1 deletion and received `ConflictError`.
Another observed interleaving lost a newly authored draft when Back closed
the session before the save chain settled. Latest main RED: **1 failed,
2 passed in 9.73s** (`/private/tmp/pr2427-gc-before.log`); the earlier authored
loss is recorded in `pr2427-blank-gc-red.Qpl1fE/pytest4.log` under the per-user
temporary directory. These are separate observations, not universal timings.

The existing Notes controller now obtains destructive admission, waits for
coalesced saves, then rechecks exact session/note identity, canonical blank
fields, title provenance and explicit-Save exemption before deleting at the
admitted version. An intervening authored draft falls through normal flushing;
failed cleanup remains best-effort. Every admission is cancelled or finished.
ADR-027/055 ownership and silent-GC limits are preserved; no new ADR or caps.

The original shell test still failed after this repair. A separate trace proved
its unfocused programmatic body clear was overwritten by TASK-32062's accepted
snapshot projection before its queued Changed event was consumed. Back therefore
correctly saw nonempty content and never entered GC. The fixture now focuses the
body and awaits the canonical empty receipt, retaining its strict deletion
assertion. Paired diagnostic evidence is in
`/var/folders/p_/x47tgtn57cv43r7yxxn40tyh0000gn/T/pr2427-blank-gc-trace.FZxJi8`.

The strengthened regression file covers successful cleanup, authored/early
external changes with **no delete call**, and an external write at the admitted
delete boundary causing a real version conflict. Every case retains an unrelated
note's exact content/version. Targeted combined GREEN: **57 passed, 845
deselected, 3 warnings in 49.76s**
(`/private/tmp/pr2427-gc-final-targeted.log`). Five complete architecture files:
**79 passed, 2 warnings in 1.63s**. Derived preflight passes. Controller/new-test
Ruff passes; Screen's 47 pre-existing findings are byte-identical to the HEAD
baseline, not newly introduced or declared clean. New-test formatting and
whitespace checks pass. Independent rereview found no blockers after strengthening
the destructive-call assertions. An earlier three-case native probe had no
retained SQLite descriptors; this is not complete-file qualification.

**Wave sixteen complete native verification is active**, session 39893, covering
the entire Library shell, new race file and note-session coordinator file, with
source frozen. Evidence:
`/var/folders/p_/x47tgtn57cv43r7yxxn40tyh0000gn/T/pr2427-wave16-native.0PgC1F`
(`pytest.log`, `native.jsonl`). Runtime repairs remain uncommitted until this
qualification finishes. Latest live dev remains `a36fc6133c`; final publication,
exact-head review/checks and normal protected merge are still pending.

Wave-sixteen frozen source SHA-256 (including the still-untracked regression,
which is not included in `git diff`'s hash):

- Notes controller: `89ef975bc14ebb6c4a4462442b6d75ea3e138718e591986c5e53de647fbf0a4c`
- Library screen: `dd5ed00e2644a7803c38f414827ea17ba1f417c0d657b645d0b1682e1c5accd3`
- Library shell tests: `5bd626ada8be4ae292314a4ee8d5e0f894bcd5a4174dc83cf43ef49dab865d85`
- New GC race tests: `62657a8cd488fb939c73678885f6ffc7bab03e5cb1e90452556f3ed20350b18c`
- Note session tests: `6999f1ce0955e7991fa59f0dac2187c02f6cb35523dcb2f304fc659c98ab3d83`

Wave-sixteen terminal verification: **902 passed, 10 warnings in 1662.29s
(27:42)**, exit 0, final **`sqlite_paths: []`**. All five frozen source hashes
match after completion. Session 39893 is closed and the source freeze is lifted.
This qualifies the entire Library shell, all four race controls and the complete
note-session coordinator file. Fresh live dev still equals `a36fc6133c`;
published `538c36b992` has green required checks and CLEAN merge state, but the
new runtime repair still requires publication and exact-head review/checks.

Published the qualified repair as
`b84605f81d596c5b36f4b6b0266f4d8bd47d0dab`; verification evidence is issue
comment 5599703742 and its Qodo request is 5599703992. Qodo completed that exact
head with five findings, comments 3966908478/3966908484/3966908490/3966908496/
3966908502. The native result remains valid; the new review findings require
their own bounded fixes and publication before final merge.

### Steps 92/93: final-head callable contracts and provenance census

The unchanged strict Console route/callsite census file reproduced **3 failed,
7 passed in 7.04s** because stored source lines drifted during the rebase.
Seventeen integer-only updates refresh all eleven changed route markers and six
gateway-call locations, including retry, actor routes, fallback and excluded
Settings generation. No route identity, capture policy, predicate, actor-chain
requirement or assertion changed. The complete strict file then passed 10 tests;
both complete provenance files pass **53 tests, 5 warnings in 10.15s**
(`/private/tmp/pr2427-qodo-provenance-full.log`). Independent review verified
every pin and the unchanged bidirectional AST/ownership checks.

Controller builders now document their screen/dependency contracts. Both lazy
evaluation hooks use the accurate built-in `type` return annotation and concise
Google-style contracts, with no new imports. Their normalized executable ASTs
are unchanged apart from documentation and the intended annotation. Six complete
wiring/import/architecture files pass **128 tests, 3 warnings in 30.69s**
(`pr2427-qodo-wiring-docs.lPPMa0/final-pytest.log` in the per-user temp directory).

Settings inspection found the undocumented adapters were thirteen shadowed
duplicates: later effective definitions already had their public contracts.
An exact-one-definition regression failed first (1 failed in 0.79s); only the
earlier duplicate set was removed, preserving the effective functions verbatim.
The intervening top-level statements were only three unrelated function
definitions, so no caller captured the removed functions. Duplicate type-only
imports were consolidated without changing their symbol set or runtime imports.
The full effective runtime AST proof and scoped Ruff pass; thirteen existing
F811 redefinition findings disappear. Proof:
`/private/tmp/pr2427-step93-settings-ast-proof.log`. No new Settings size limit
was invented or existing cap raised.

Independent review clears these contract/duplicate changes. The complete
eight-file Settings/import/startup group finished **248 passed, 6 warnings in
224.08s**, exit 0 (session 66025 closed,
`/private/tmp/pr2427-step93-settings-green.log`). Fresh derived preflight also
passes (`/private/tmp/pr2427-step92-final-preflight.log`, session 27603 closed).
The test result includes a session descriptor warning: start 12, end 274, growth
262 above its 200 threshold. This is not yet attributed and is not a proven
SQLite leak; no cleanup success is claimed for this Settings cohort. Step 94
starts native attribution with no forced GC, connection monkeypatches or host
cleanup. The five verified review fixes may be checkpointed for exact-head
review while attribution runs, but merge remains pending resource investigation
and final review/CI. No complete-repository sweep is claimed.

### Steps 94–99: final Settings resource attribution and new head findings

Steps 92/93 were published as `e38cbb280496a675a9c7b6b1d7e95e44265abeaa`;
all five previous Qodo threads received evidence replies and are resolved.
Exact-head review request 5600013712 produced two new findings:
3967103308 (persisted Console message probe transaction) and 3967103315
(quoted media-error filenames containing spaces). Live dev remains
`a36fc6133c69f77b8b14596a59918de34261f8ef` on this turn's check.

The unchanged native SQLite observer confirmed the Settings warning: **248
passed, 6 warnings in 204.34s**, exit 0, but **95 retained SQLite handles**
(82 Chroma, 13 test-app databases). Evidence:
`/var/folders/p_/x47tgtn57cv43r7yxxn40tyh0000gn/T/pr2427-settings-native.XcJ7xe`.
The former passing test counts did not qualify cleanup.

Two separate owners caused retention. Settings index-status reads construct
one-shot Chroma clients through `collection_indexes.list_indexes`; that function
and its adoption/deletion siblings never released the acquired clients.
Capability-checked exact-client close now runs in their `finally` paths, without
global cache/system shutdown or changed operation results. Older clients without
close remain compatible. New lifecycle controls first failed **17 of 26 cases**;
three complete relevant files then passed **45 tests in 0.96s**. The real control
executes fifteen local operations while preserving a separate same-path client's
system, reference count and usability, and explicitly closes all test-created
clients even on RED. Evidence: per-user `pr2427-chroma-lifecycle.63zr9U/`.
The single added diagnostic is a fixed warning string with no exception payload,
path, secret or user content; its inventory delta was explicitly reviewed.

The two Settings fixture modules also omitted the existing exact-app resource
fixtures. Reusing them reduced the first attributable case's app SQLite handles
**13 to zero**, while leaving its separate Chroma handles unchanged before the
production fix. Tool-profile plus complete fault/foreign-owner controls pass
**26 tests**, final SQLite inventory empty. Native all-FD observation after both
fixes confirms the first Settings case has zero SQLite handles and removes three
Chroma-associated kqueues; an unlinked test-app instance-lock handle remains and
is separately assigned to step 98. No global cleanup, forced GC, threshold raise
or foreign-owner close is used. Full combined native qualification is pending.

Perf Guard run 34337486297 independently failed because its fixed one-second
sample missed the ChaChaNotes FTS worker queued behind serial recovery. A
controlled real-boot case holds the first real recovery job until two seconds
after readiness and reproduces the same failure (**1 failed, 1 passed**).
The probe now retains its minimum observation period and waits for its unchanged
required-start identities under the unchanged 300-second subprocess deadline.
All allowlists and anti-vacuity checks remain intact. Six complete boot/policy
files pass **35 tests in 45.88s**, with independent review finding no blockers.
That run also exposed two unawaited-timer warnings in synchronous unmounted
policy fixtures; step 99 will repair that missing loop precondition without
changing production scheduling. Logs: `/private/tmp/pr2427-boot-delay-{red,green}.log`.

Task remains In Progress. The new review/resource repairs still need final
complete-file/native/static qualification, publication, exact-head review and CI
before normal protected merge. Earlier native Library 902-pass evidence remains
for its unchanged Notes ownership; it does not substitute for these new gates.

Step 98 closes only the exact app product's existing instance-lock handle on
the already-owned auxiliary stack, after its database callbacks. Four real
portalocker cases first failed with the owned handle retained, then passed
normal/runtime/cancellation/auxiliary-failure teardown while foreign locks stayed
held and lock files stayed on disk. Three complete fixture/instance-lock files
pass **39 tests, 2 warnings in 5.30s**. The original native Settings case now
passes with **zero SQLite and zero instance-lock handles**. Its two additional
sockets also appear in an app-free Chroma-only control executing fifteen real
operations with all clients explicitly closed; no app or lock is needed to
produce that bounded installed-dependency footprint. No all-FD-zero claim is made.
Evidence: `/private/tmp/pr2427-step98-{lock-red,lock-green,native}.log` and
`/private/tmp/pr2427-step99-chroma-control{,-retention}.log`.

Step 99's two synchronous timer-registration assertions first failed because
the unmounted app could not schedule a timer. Each fixture now records only its
own `set_interval` call and returns a stoppable token; assertions pin the real
reconcile callback, unchanged interval, gate drain and exactly-once cancellation.
The mounted real-timer test is unchanged. The complete policy file passes
**16 tests in 4.85s** with no unawaited-coroutine warning. Final six-file boot
qualification passes **35 tests, 7 dependency/deprecation/headroom warnings in
50.53s**, retaining all original module/CSS/worker budgets. Independent reviews
clear the census and timer changes. `/private/tmp/pr2427-boot-final.log`.

Final derived preflight passes all seven checks, including the explicitly
reviewed fixed-string Chroma diagnostic inventory update; 3,684 task files,
115 tables, 282 index census rows and 67 plan pins match. Log:
`/private/tmp/pr2427-step99-final-preflight.log`.

Complete native Settings requalification is now terminal: **248 passed,
5 warnings in 217.19s**, exit 0. Final native inventory has **zero SQLite and
zero instance-lock handles**, with only standard streams plus one kqueue and
four sockets—the same pytest (one kqueue/two sockets) and app-free Chroma
(two sockets) footprint established by isolated controls. The former descriptor
growth warning is gone; no threshold changed. The six frozen source hashes
match after completion. Evidence:
`/var/folders/p_/x47tgtn57cv43r7yxxn40tyh0000gn/T/pr2427-settings-final.wNGDbh/`
(`pytest.log`, `fd_identity.jsonl`, `source.sha256`). This is a complete same-file
comparison, not merely an individual passing case or a SQLite-only scan.

Four complete provenance/Console architecture files additionally pass **103
tests, 5 warnings in 17.27s** (`/private/tmp/pr2427-step97-census.log`).
Changed-file Ruff and whitespace checks pass. Live dev remains `a36fc6133c` and
is already an ancestor of this branch; no additional rebase is needed at this
checkpoint. The two new Qodo repairs still await terminal owner-file results
and publication before their review threads can be resolved.

Step 97's final complete owner results are now green: **20 menu tests in
50.37s**, **73 media-controller tests in 4.50s**, and **3 media-state tests in
1.33s**. The persisted one-row probe uses the existing borrow-safe transaction;
real SQLite controls verify both empty/nonempty eligibility and an already-open
caller transaction, then quiesce the exact created DB in `finally`. The redactor
handles single/double-quoted POSIX/Windows final filenames with spaces, including
the opposite quote inside a path, while preserving delimiters and trailing
prose. Its conservative bare-token matcher is unchanged. Independent review
cleared production and the strengthened exact-test cleanup. Logs:
`/private/tmp/pr2427-qodo97-menu-final2.log`,
`/private/tmp/pr2427-qodo97-media-final.log`, and
`/private/tmp/pr2427-media-state-final.log`.

Fresh combined six-file resource qualification passes **84 tests, 2 dependency
warnings in 5.45s** (`/private/tmp/pr2427-resources-final.log`). All scoped repairs
are ready for publication; task completion still requires the published head's
review/checks and verified normal merge.

### September 9 — Qodo follow-up on 95dc817a46 (steps 100–102)

Published head `95dc817a46a5cf7b176c33f8552528aa80b8428c` now has green
PR Fast Lane, Perf Guard, derived artifacts and all applicable platform checks.
Live dev remains `a36fc6133c69f77b8b14596a59918de34261f8ef`, already included.
Qodo added three actionable findings; these are new review work, not continuing
failures of the previously qualified resource or boot repairs.

Step 100 makes failed index-client retirement diagnosable by its sanitized
exception class, using the existing persistent-metadata helper. Raw exception
values and tracebacks remain excluded because they can disclose private paths
or user content. Two real Loguru sink controls first failed and then passed;
three complete RAG files pass **47 tests, 2 dependency warnings in 0.88s**.
Independent review confirms no expanded logging authority and exact sink cleanup.
Evidence: `/private/tmp/pr2427-close-cause-{red,green}.log`.

Step 101 bounds the existing complete, size-limited JSON body read with a fixed
**30-second absolute deadline**, not a renewable inactivity timeout. Expiry is
the existing source-free HTTP 408 refusal, with connection reuse disabled and
that flag preserved through the sanitizing security middleware. ADR-121's new
JSON request-body deadline amendment records this transport contract before
implementation. External cancellation, EOF handling, UTF-8/JSON validation and
size limits are preserved. No global runner policy or aiohttp private API changes.

Controlled stalled/trickled streams produced **4 RED failures**; real loopback
HTTP then reproduced the missing response, and an intermediate deadline-only
fix reproduced middleware loss of connection-close policy. Five complete body,
gateway and served-security files now pass **116 tests, 3 dependency warnings
in 3.09s**. The real HTTP control asserts sanitized 408/security headers, no
canary disclosure, and a subsequent complete request using the unconsumed boot
capability. Exact clients, writer and gateway close in `finally`. Evidence:
`/var/folders/p_/x47tgtn57cv43r7yxxn40tyh0000gn/T/pr2427-canvas-body-deadline.CKNoPY/`.

Step 102 uses the existing app-thread identity boundary: the known owner queues
with `Screen.call_later`; other callers use `App.call_from_thread`, with no inline
fallback after a missing or rejected handoff. Already-committed Canvas state is
unaffected and manual opening remains available. Current settlement is already
application-loop-affine under ADR-121; this hardens a permissive fallback rather
than claiming all production settlement previously ran on background threads.
The existing diagnostic text and same-session browser suppression are preserved.
Three controlled failures reproduce the old unsafe/incorrect dispatch; four
focused controls then pass, including real worker threads and accepted dispatch
back to the owner. The complete controller file passes **25 tests**; independent
production review is clear. Logs: `/private/tmp/pr2427-qodo-canvas-{red,green,full-final}.log`.

Original Screen/delegate caps pass **71 tests** and four complete strict
provenance/architecture files pass **65 tests, 5 warnings in 14.26s**. No census
location update or raised cap was necessary. Changed-file Ruff and whitespace
checks pass. The added sanitized warning was explicitly reviewed before generated
inventory refresh; the other diagnostic change is re-indentation only, with no
changed message or interpolated fields. Logs:
`/private/tmp/pr2427-qodo-canvas-caps.log`,
`/private/tmp/pr2427-qodo102-caps.log`, and
`/private/tmp/pr2427-qodo102-diagnostics-review.log`.

Independent test review additionally flagged exact app-resource ownership in
the new worker controls. Step 103 reuses the existing two exact-owner fixtures
and makes the send test's existing builder/database attachment visible to that
module-local owner. It does not change shared fixtures or production cleanup.
Complete native requalification passes **25 tests, 3 warnings in 20.22s**, exit
0, with no retained SQLite or instance-lock handles after each test or at final
observation. Only the pytest-asyncio session selector footprint remains (one
kqueue/two sockets, besides standard streams). Frozen runtime/test hashes match;
independent review clears the ownership correction. Native evidence:
`/private/var/folders/p_/x47tgtn57cv43r7yxxn40tyh0000gn/T/pr2427-qodo-canvas-native-final2.dT1EGz/fd_identity.jsonl`.

All seven derived checks pass, including 3,684 collision-free task files,
115 tables, 282 index census rows and 67 plan pins. Final preflight log:
`/private/tmp/pr2427-qodo102-preflight.log`. Scoped Ruff and whitespace checks
remain green after the test-owner correction. Publication and the new exact
head's review/checks precede normal protected merge. No repository-wide sweep
or all-descriptor-zero claim is made.

### September 9 — Qodo follow-up on f26827b949 (steps 104–107)

All applicable CI checks pass on `f26827b949a6c5009feab4e1f5ec44fab6dd1c7b`,
including Fast Lane, Perf Guard and derived artifacts. Qodo's next review adds
seven findings. Three do not require implementation after independent tracing
and fresh verification:

- **Actor policy Pydantic replacement (3967619551):** the archive validator is
  intentionally strict and byte-preserving under ADR-074/079 and TASK-31764.
  The pure-contract guard forbids both Pydantic and API-layer imports. The API
  model's default coercion and activation's normalization are not substitutes
  for archive admission. Complete architecture, contracts and activation files
  pass **95 tests, 2 warnings in 10.66s**, including malformed policy rejection
  and activation/export preservation. No architecture guard is weakened.
- **Stale display-name lease (3967619593):** the published coordinator already
  registers exact-plan abandonment in the task's done callback. The existing
  real-SQLite stale-identity control verifies empty lease maps and restored fork
  eligibility; complete lifetime file passes **10 tests, 2 warnings in 5.40s**,
  including writer/cancellation ownership. No extra early release is added.
- **Recovery modal CSS (3967619617):** immutable `875936d66f` already defines
  Recovery as a separate ModalScreen while its CSS alias contains only
  Navigation-rooted selectors. The removed alias was genuinely inert. Fresh
  complete original-versus-current geometry, compositor paint and button-state
  parity passes **13 tests, 3 warnings in 11.68s** at compact/wide sizes. No new
  layout is introduced under the premise of restoring lost effective styles.

These threads received evidence replies 3967712932, 3967713114 and 3967713286
and were resolved. Logs:
`/var/folders/p_/x47tgtn57cv43r7yxxn40tyh0000gn/T/pr2427-actor-boundary-review.ddCXE0/pytest.log`,
`/private/tmp/pr2427-display-name-evidence.log`, and
`/var/folders/p_/x47tgtn57cv43r7yxxn40tyh0000gn/T/pr2427-recovery-modal-review.NAZf8N/pytest.log`.

Step 105 confirms the world-book notification leak with **2 RED failures**.
Attach/detach now show only operation-specific safe retry copy; existing
diagnostics, mutation calls and exact dialog release remain unchanged. New
no-mount fault controls allocate no full app or database and assert no refresh
after failure. The complete retrieval/world-book files, including the existing
real-DB attach/detach round trip, pass **26 tests, 3 warnings in 18.99s**.
Independent review is clear; diagnostic statement comparison shows no changed
logging statements. Logs: `/private/tmp/pr2427-worldbook-privacy-{red,green}.log`.

Step 106 documents the workspace helper's four actual inputs and disabled/tooltip
result, and the media state's retained/requested/in-flight paging rules.
Executable ASTs are identical after stripping docstrings. Three complete
workspace/media files pass **100 tests, 2 warnings in 5.32s**; scoped Ruff,
formatting and whitespace checks pass. `/private/tmp/pr2427-qodo106-docs.log`.

Step 107 adds only the persisted aliases of already-active path nodes to the
publication guard's native identity set. Session, conversation, nonempty and
all-revisions checks remain unchanged. The new genuine restored-store control
first fails only for the persisted origin; its preceding native-origin check
passes. Both forms then pass, and both are rejected after branch/session changes;
mixed sibling revisions and conversation mismatch remain rejected. The original
ephemeral-session control is retained unchanged. This is representation hardening,
not a claim that all saved tool publications were previously broken.

Final focused controls pass **2 tests**; settlement correctness passes **6 tests**.
Complete native controller qualification passes **26 tests, 3 warnings in
20.63s**, with zero retained SQLite/instance-lock handles and only the established
pytest-asyncio selector footprint. A subsequent test docstring/return annotation
change is nonbehavioral; both focused controls pass again on the final bytes.
Runtime/test hashes are frozen and independent review is clear. Evidence:
`/private/tmp/pr2427-qodo107-{red,final-current,settlement}.log` and
`/private/var/folders/p_/x47tgtn57cv43r7yxxn40tyh0000gn/T/pr2427-qodo107-native-final.y5U6jU/fd_identity.jsonl`.

Five complete architecture/provenance files pass **131 tests, 5 warnings in
15.90s**, all seven derived checks pass, and final scoped Ruff/format/whitespace
checks are green. No diagnostic statements changed, so no inventory rewrite was
needed. Logs: `/private/tmp/pr2427-qodo107-{caps,preflight,diagnostics}.log`.
Live dev remains `a36fc6133c`, already an ancestor. Publication and exact-head
review/checks still precede normal protected merge.

The final complete Canvas controller file additionally passes **47 tests,
2 warnings in 3.79s**, not just the earlier six settlement-focused controls.
`/private/tmp/pr2427-qodo107-canvas-complete.log`.
