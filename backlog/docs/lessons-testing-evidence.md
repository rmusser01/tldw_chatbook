# Lessons: what counts as evidence a change works

Working knowledge about testing in this repo. Not decisions (see `backlog/decisions/`)
and not point-in-time audits — these are traps that have actually cost time here, kept
so the next person does not rediscover them.

**Every entry states the incident that produced it.** A lesson without its evidence
decays into folklore, and folklore is ignored. If you add one, bring the incident.

---

## Index-plan guards must accept the names the DDL actually uses

**TASK-31242 isolated PR preparation, 2026-09-05.** Five real-SQLite,
no-statistics query-plan assertions passed for the Keyword indexes, but the
index census guard rejected every pin because its evidence extractor only
recognized `idx_` and `uq_` names. The schema used descriptive
`character_conversation_search_*` names. A synthetic positive/negative pair
reproduced the missing positive pin while preserving rejection of `not in`.
Accepting standalone identifier literals fixed the guard without changing the
DDL or labeling new indexes as pre-convention. The same qualification found
that the schema allowlist scanner omitted the new dedicated DDL module; the
live-schema parity test caught all five missing tables.

**What to do.** Run both live-schema parity and index-plan inventory checks
when moving DDL into a dedicated module. Register its source explicitly, keep
real query-plan assertions, and verify a guard's name recognition before
discarding evidence or weakening its inventory policy.

## CSS ratchet paydown must preserve inherited subjects and specificity

**PR #2419, 2026-09-05.** Re-keying three snapshot rules removed a
277-versus-274 bare-type selector breach, but the first narrowed selectors
regressed the 80-column Models and F9 paint checks. A plain button ID lost to
the launcher disabled-border rule and clipped Restore. Checkbox and
CollapsibleTitle both inherit Static; excluding them lost height/wrapping
behavior and left the focused checkbox text outside the painted viewport.
Preserving button ancestor specificity and explicitly targeting those inherited
subjects restored the checks without raising the boot budget.

**What to do.** Pair parsed-selector census checks with actual small-terminal
paint checks. When replacing a type selector, inspect its subclasses and preserve
the relevant cascade precedence, not just the obvious direct widget instances.

The same PR then inherited TASK-31450's 976/972 startup breach. A controller's
closed-rail dispatch guard did not prevent its constructor and module imports
from loading four Environment modules. First-use owner/projection construction
restored 972/972, but required explicit first-open painting for a workspace-less
panel (there is no worker result to paint it). Measure imports as well as I/O,
and pair lazy-owner guards with a no-result first-use UI test.

---

## Spend forecasts must test admitted media and durable recovery IDs

**PR #2397, 2026-09-04.** The next-send estimate scanned every transcript
attachment, so failed echoes, assistant images, missing attachment bytes, and
images omitted by a non-vision model all suppressed a valid text estimate.
The recovery tests also reused one ID for the transient row and its database
record, hiding a mismatch that counted unfinished recovered turns as Current
spend. Mounted regressions reproduced the media cases; distinct persisted and
transient IDs reproduced both accepted and quarantined recovery errors.
The first media fix then captured full send configuration even for an empty
chat, pulling RAG imports onto startup: CI and the local census measured 1,030
modules against the 972 limit while clean dev passed. Checking for admitted
user attachments before resolving media capabilities removes that eager work.

**What to do.** Reuse the provider's metadata-only admission and image-budget
projection for display decisions. Give hydrated test rows distinct transient
and persisted IDs, and verify both request-context and settled-spend ownership.

---

## Measure child lifetime RSS after final serialization, not before it

**TASK-31642, 2026-09-04.** The first Chunking Lab worker diagnostic sampled
`getrusage(RUSAGE_SELF).ru_maxrss` immediately after the engine returned. Its
repeated-format stress run still had to validate, copy, and serialize the complete
RunResult afterward, so that diagnostic omitted the expensive publication path.
On the macOS qualification host, collecting the reaped child's lifetime usage
with `os.wait4` measured 480,313,344 bytes RSS for an admitted formatter whose
conservative working-payload estimate was 32,468,700 bytes. A regression child
allocated and freed 150 MB but reported an early peak of 1 byte; only the reaping
owner's metric passed the check. CPU rlimit application succeeded on this host;
address-space rlimit application did not.

**What to do.** Include serialization and IPC in the lifetime being measured.
Where a child-reaping usage API is available, collect its observation rather than
treating an earlier self-report as the lifetime peak. State the tested OS and
distinguish payload estimates, measured RSS, and successfully applied OS limits.

---

## Coalesced state equality does not prove that no content mutation occurred

**TASK-31641, 2026-09-04.** Chunking Lab initially expired Undo restore by
comparing the next durable content with the current checkpoint. A regression
test edited a restored draft and immediately used application Undo before
autosave ran. The resulting draft and undo tuple were byte-identical to the
restored state, so the comparison incorrectly retained Undo restore despite
the intervening content mutation. A persisted content-only revision counter,
incremented by edits and Undo but preserved by view navigation, made the
same real-store regression pass.

**What to do.** When a policy depends on whether an event happened, preserve
an event counter or explicit marker through coalescing. Test an edit followed
by its inverse before the next save; testing only distinct final values cannot
prove first-mutation expiration or equivalent event-sensitive behavior.

---

## SQLite progress handlers must not query their active connection

**TASK-23113.11, 2026-09-02.** The first physical trace-compaction worker
queried `page_count` and `freelist_count` on the same connection from SQLite's
VACUUM progress handler. The small real fixture happened to pass, but a
connection double that rejected SQL while VACUUM was active reproduced the
re-entrant call deterministically. Capturing the content-free byte snapshot
before installing the handler preserved progress reporting without recursively
using a connection in the middle of a statement.

**What to do.** Treat SQLite progress, authorizer, collation, and scalar-function
callbacks as non-reentrant unless the API explicitly guarantees otherwise. Read
needed metrics before the long statement, keep callbacks bounded and in-memory,
and verify the behavior against a real file-backed SQLite database.

## SQLite quiescence must cover result consumption and database identity

**TASK-23113.11, 2026-09-02.** Review of the first physical-compaction barrier
found two holes that transaction-only tests did not expose. A direct SELECT was
no longer in a native transaction after `execute()` returned, even though its
cursor still had unread rows, so the barrier could close that live connection.
Separately, two `CharactersRAGDB` objects for the same file owned independent
registries, allowing one object to open a handle while the other compacted.
Real cursor and same-file-instance tests reproduced both races.

**What to do.** Key process-local SQLite maintenance coordination by canonical
database identity, not wrapper-object identity. Track synchronous operations
and cursor result consumption through exhaustion or explicit close; acquisition
and transaction boundaries alone are insufficient evidence that a handle is
idle.

## Periodic maintenance must deduplicate work by the state it processed

**TASK-23113.11, 2026-09-02.** The initial automatic compaction loop generated a
new opaque logical-GC request every minute after legacy normalization completed.
Because completed GC request rows are durable idempotency records, an unchanged
database would have accumulated a new result row on every poll. An accelerated
runtime test observed two collections for the same graph epoch. Caching the
processed epoch and retaining a retryable completed result reduced unchanged
epochs to one durable collection while still retrying interrupted compaction
and rechecking size/free-page thresholds against live SQLite metrics.

**What to do.** A recurring maintenance timer is only a wake-up signal. Before
creating a new durable request, compare the source's monotonic change token (or
equivalent exact state identity) with the last processed token. Reuse the exact
successful prerequisite while a downstream retry is pending, and clear it only
after success or a genuinely terminal decision. Thresholds based on mutable
storage metrics are retryable, not terminal.

---

## POSIX availability does not prove a memory limit is enforceable

**TASK-23113.10, 2026-09-01.** The custom-PII worker initially treated Python's
`resource` module as evidence that its memory cap was active on every POSIX
host. Real-process tests on macOS showed that `RLIMIT_RSS`, `RLIMIT_DATA`, and
`RLIMIT_AS` were present, but every attempt to lower their effectively-unlimited
values failed with `ValueError: current limit exceeds maximum limit`. CPU and
output-file limits were enforceable there; the address-space limit was
enforceable in the Linux qualification path. Reporting the memory cap merely
because the constants existed would have made the security evidence false.

**What to do.** Treat an OS resource bound as enforced only after the child
successfully applies it, return content-free metadata naming the limits that
actually took effect, and assert platform capabilities in a real child process.
Keep parent-enforced deadline and byte/count ceilings on every platform; qualify
memory enforcement only where the host proves it can apply the limit.

---

## Durable masking evidence must remove the detector before every read path

**TASK-23113.10 review, 2026-09-01.** Custom-PII tests proved that ordinary
message revisions and generic provider artifacts retained irreversible masks,
but a saved provider-continuation sidecar still used a revision reference. The
native viewer reran the process-local custom ruleset when reconstructing that
sidecar. After a restart or registry eviction, the trace therefore omitted a
continuation that had been successfully captured, even though the surrounding
message remained readable. The domain-specific read path escaped tests that
covered the same policy at a different storage owner.

**What to do.** For every privacy transform claimed to be durable, enumerate
each persisted source domain—not just messages versus artifacts, but their
sidecars—and read it after the detector, key, registry, or worker has been
removed. Viewer, copy, and export must consume the stored masked projection;
they may fail closed when that durable projection is absent, but must not rerun
an ephemeral detector to recreate it.

---

## Sectioned projections must sort by section before row priority

**TASK-28125, 2026-09-02.** The Ctrl+K switcher initially pinned every selected
row ahead of every other row, then inserted `OPEN AGENT TABS` and `SAVED CHATS`
headings while walking that order. A selected persisted conversation therefore
produced `saved → open → saved`, mounted the Saved heading twice with the same
widget ID, and could fail even though ordinary all-open and all-saved fixtures
passed. A cross-section selected-row regression forced the sort to make section
the primary key and selection the priority only inside each section.

**What to do.** For a projection rendered as keyed sections, assert that each
section key forms one contiguous run before mounting headings. Include a
high-priority or selected row from a later section in the fixture; row-level
priority must not split a section unless repeated section identities are an
explicit part of the rendering contract.

---

## Current-version repair hooks must gate on the schema that introduced their columns

**TASK-23113.8, 2026-08-31.** The trace-GC migration matrix reopened a genuine
v56 fixture while temporarily pinning the code's target version to v56. Schema
initialization took the "already current" branch and ran the Notes organization
sync-ID repair under a `version >= 56` guard, even though the repaired `sync_id`
columns and lookup tables do not exist until v58. Ordinary current-schema tests
never exposed the mismatch because those columns were present by then. Raising
both repair-hook gates to v58 made the genuine v53→v56 reopen pass again.

**What to do.** Gate every current-version repair or index-restoration hook on
the first schema version that contains every column and table it touches, not on
the task or migration version that first motivated the repair. Include pinned
historical-current reopen fixtures; testing only migration all the way to the
latest version cannot exercise these intermediate "already current" branches.

---

## Rebase before measuring a startup import closure

**PR #2250 follow-up, 2026-08-30.** The Agent Lessons branch passed its focused
feature tests, but after rebasing onto current `dev` the UI-ready module census
failed in CI at 979 modules against a 972-module cap (the local platform measured
977). The feature's apparently small app imports pulled in seven interaction- or
sync-only implementation modules through package initializers and shared contract
constants. Deferring the Notes organization composition until just after the
first interactive frame, moving constants and leaf helpers into already-resident
modules, and making promotion/adapter imports first-use reduced the real mounted
census to 970 locally while an exact absent-at-ready list pinned the intended
seams.

**What to do.** For a long-lived feature branch, rebase onto the actual merge
base before treating startup evidence as final. Measure the mounted app's full
import closure on every supported CI platform, not just direct imports or a
feature-only test process. When a new domain is not required for first paint,
defer its composition or import it at first use and add its implementation
modules to the absent-at-ready contract; retain a little platform headroom
instead of landing exactly at the local cap.

---

## A census taken on the next tick is not a census taken at the flag

**PR #2373 / task-31281, 2026-09-04.** The UI-ready module census failed on
three consecutive CI runs of what was, after the second, the same tree: 973,
973, then 977 modules against the 972 cap. The first two carried exactly the
one new module the PR added (fixed by lazy-mounting the widget, ADR-097
response 1). The third did not contain that module at all -- it carried five
`Library.collections_capture_*` modules that neither sibling run nor `dev`'s
own run of the merged tree had. The app sets `_ui_ready` and then keeps
running its mount path, which arms 0.1s timers that are deferred past
readiness *by design*; the census polled the flag every 5ms and copied
`sys.modules` when it woke, and on a starved runner those timers won. `dev`
sits exactly at the cap, so the "+/-1 wobble headroom" the constant's own
comment promises does not exist, and a one-run race flips the check red.

**What to do.** When a guard's contract is "resident at instant X", sample at
instant X -- here, a class-level property whose setter copies `sys.modules`
synchronously inside the `self._ui_ready = True` assignment -- never on the
next scheduler turn after observing X. And when a non-required check fails on
your PR, read its `+` list before believing it: if your module is not in it,
compare against the base branch's own run of the same tree before spending a
round on a fix.

---

## Shipped migration and ADR numbers are allocation records, not merge labels

**TASK-24613, 2026-08-30.** The Agent Lessons worktree and a newer `dev`
history had each advanced independently from the same ChaChaNotes v55 schema
and had each allocated ADR-104. A textual conflict resolution could have kept
both branches' filenames while silently making two different v55→v56 and
v56→v57 transitions compete for the same durable version, and could have left
two accepted architectural decisions with one identifier. The latest-dev merge
instead treated the already-shipped `dev` allocations as immutable: it kept the
semantic-trace v55→v57 chain and token-disclosure ADR-104, appended the Notes
organization transitions monotonically at v57→v61, and renumbered the later
promotion decision to ADR-106. Real migration fixtures, the index census, and a
repository-wide stale-reference search then verified the reconciliation.

**What to do.** Before resolving a long-lived branch merge, compare durable
schema and ADR allocation tables rather than only conflicted text. Preserve
numbers that already shipped on the integration branch, append new migrations
after its current version, renumber only the unshipped decision, update every
reference, and test from each immediately preceding real schema boundary.

## A new table needs a migration from every already-shipped current version

**TASK-24308, 2026-08-30.** The first pending-note finalizer added its
Notes-owned publication-intent table to the v58→v59 receipt SQL and to fresh
schema creation. Fresh databases and v58 migrations were green, but a real
database already reopened and stamped as v59 had no transition left to run, so
the production code could query a table that did not exist. A genuine-v57
fixture made that omission deterministic. Moving the table into a guarded
v59→v60 migration, then testing real-v59 reopen, rollback, fresh-schema parity,
SQL allowlisting, and the index census closed the gap.

**What to do.** Once a schema version can exist outside the current worktree,
treat its migration as immutable. New durable owners require a new monotonic
version and a real fixture at the immediately preceding shipped boundary.
Fresh-schema success and replay from an older version do not prove that a
current-version database can reopen safely.

## Authorization must read the owner snapshot, not a normalized projection

**TASK-24309, 2026-08-30.** Agent Lesson preflight initially reused the public
Library organization projection. That projection intentionally normalizes the
durable receipt state `pending_organization` to the user-facing value `pending`.
The mutation policy classifies the durable receipt state as an Agent Lesson, so
the lossy public value made an authorization-relevant pending lesson look like
an ordinary Note before review. Real-database preflight coverage exposed the
gap. A private Notes-owner snapshot now supplies the exact raw receipt state,
versions, requested keywords, and note/organization versions while the public
API remains normalized.

**What to do.** Do not reuse display or compatibility projections for
authorization merely because they describe the same object. Inventory every
normalization, omission, alias, and redaction between the durable owner and the
consumer. If policy depends on one of those distinctions, add a narrow private
snapshot from the owner and test the real stored value against the public
projection that erased it.

## Historical evidence and current identity selection are different decisions

**TASK-24309, 2026-08-30.** The first Agent Lessons seed-race implementation
observed an exact-root upsert from synchronized history and could use it both to
advance monotonic "seed seen" evidence and to select/adopt the remote folder
identity. A stale upsert followed by a rename or tombstone is valid historical
evidence that a conventional seed once existed, but it is not the current head
and cannot prove which identity should win or that a local candidate is safe to
retire. Stale-history regressions forced the split: validated history may mark
the profile seeded, while adoption and candidate retirement require the current
applicable head plus untouched/unused intent evidence.

**What to do.** When replay reconstructs monotonic history, separate "this ever
happened" from "this object is current and authoritative." Let stale records
advance only the former. Identity selection, deletion, adoption, suppression,
and cleanup need an explicit current-head/applicability check and their own race
tests.

## Queue causality must not depend on wall-clock ordering

**TASK-24308, 2026-08-30.** The first v58 Notes publication-intent drain ordered
pending rows by `created_at` and `intent_id`. A deterministic test inserted one
note's version 2 with an earlier clock than version 1; the successor drained
first even though both immutable intents were otherwise valid. Ordering and the
supporting partial index were changed to the note's durable lineage
`(note_id, entity_version, intent_id)`. The reversed-clock restart test then
proved v1 and v2 each drained once in causal order, while a separate test kept
ordering deterministic across notes and isolated by server profile/dataset.

**What to do.** For multiple versions of one entity, order dispatch by durable
entity identity and monotonic entity version. Use timestamps only as display or
diagnostic data unless the contract explicitly establishes a trusted monotonic
clock; UUID and wall-clock tie-breakers cannot encode causal ancestry.
---

## A turn deadline cannot preempt one oversized synchronous callback

**TASK-22512 Task 6, 2026-08-30.** The persistent-terminal actor passed its
functional parser-budget tests because the injected clock advanced only between
backend chunks. The required ten-second ANSI flood report then measured about
6,127 ms p95 lateness for a 100 ms event-loop sentinel. A single 64 KiB call into
the real screen parser took roughly 275–285 ms, so checking the eight-millisecond
deadline only after that call could neither enforce the budget nor let the
sentinel catch up. Delivering admitted output to the parser in bounded 1 KiB
slices made the time check observable between calls; the same qualification path
then passed at about 34.1 ms p95.

**What to do.** A deadline checked between synchronous callbacks bounds a turn
only when each callback is independently small enough. Pair byte and time budgets
with a bounded delivery slice, add a deterministic clock test whose cost scales
with callback size, and run the real parser/event-loop flood with its latency
threshold enabled. A test that advances the clock by a fixed amount per callback
can stay green while one callback monopolizes the loop far beyond the deadline.

---

## Pilot clicks can bind a widget that recomposition removes before hit-testing

**TASK-24403, 2026-08-29.** The clean fast-PR lane passed all 670 selected tests,
then a second identical run failed one 100x30 MCP Workbench interaction:
`Pilot.click()` returned false even though the checkbox was painted, on-screen,
and focused. A pressure reproduction captured the transition during the click:
the target changed from `Region(20, 23, 63, 1)` to the zero region, and the old
coordinate now hit the next checkbox. Textual's pilot snapshots the target
widget's region, yields, and only then hit-tests; the Workbench resync recomposed
the checkbox during that yield. The real keyboard path does not retain the stale
widget identity. Replacing the pilot click with focused `Space` preserved the
viewport, label, focus, persistence, and interaction assertions, then passed 12
isolated and six concurrent ordered-prefix pressure runs.

**What to do.** When a Textual control may be replaced by an asynchronous
recompose, do not use a retained widget object as a `Pilot.click()` target unless
mouse hit-testing itself is the contract. For keyboard-operable controls, focus
the control, prove it is visible and focused, and drive the real key binding; the
app can restore focus by stable ID across recomposition without the pilot holding
a stale object. If mouse behavior is the contract, capture before/after regions
and the widget at the old coordinate first so a compositor race is distinguished
from a product interaction failure.

---

## A restored bounded reader needs a mount-time request re-kick

**TASK-18916, 2026-08-28.** The Collections pagination verification included
the shared Library entry-lifecycle file and exposed three deterministic Skills
failures already present at the exact `origin/dev` base. A restored Skills route
had begun an exact page generation before mount, when Textual could not yet own
its worker, and then remained on "Loading page 1…" forever. Media, Prompts, and
Collections already re-requested restored scope from `on_mount`; adding the same
bounded re-kick for Skills made the three red nodes pass without changing its
service or display contracts.

**What to do.** When a route can restore controller state before its screen is
mounted, treat that pre-mount begin as state preparation, not proof a worker was
dispatched. Re-request the retained exact scope from `on_mount`, fence duplicate
or stale generations in the controller, and keep a restored-route mounted test
that waits for a real source row rather than merely asserting the canvas exists.

---

## Aggregate UI gates expose partial trees and contradictory contract lineage

**TASK-24195, 2026-08-29.** The 52-test Conversation reader file and the
823-test Library shell file both passed alone, but the joined 1,879-test Notes
matrix exposed two different defects. Under sustained recomposition load, a
retained Conversation reader remained mounted while one middle child was absent;
`sync_state()` guarded its first child lookup but crashed on the later status
lookup. Removing only that middle child produced a deterministic focused
regression. The same matrix then found two tests requiring opposite behavior for
a dirty Notes reader switch. Commit lineage showed that the flush-on-switch shell
test predated TASK-23019, whose later approved contract parks dirty reader state
without persistence; changing production to satisfy the stale test broke the
newer retained-reader tests.

**What to do.** For retained Textual parents, treat the expected child set as one
availability unit: resolve all required children inside the same lifecycle guard,
cache state first, and let the replacement compose consume it. When two tests
assert mutually exclusive outcomes, inspect task and commit lineage before
changing production; update the superseded contract and run both the old boundary
case and the newer behavior together. File-level green runs are not a substitute
for the joined matrix that creates partial-tree timing and contract collisions.

## Compare retained JSON after crossing its serialization boundary

**TASK-20010, 2026-08-23.** Final first-principles evidence verification loaded
the digest-pinned original statistics runner and directly compared its rebuilt
Python summary with the parsed retained JSON. The comparison failed even though
the values were identical: the runner returns interval pairs as tuples, while a
JSON round trip necessarily restores them as lists. Serializing and parsing the
rebuilt summary before comparison produced an exact match.

**What to do.** When verifying persisted JSON against fresh producer output,
compare canonical serialized bytes or JSON-normalize the producer output first.
Use a separate in-memory contract assertion if tuple-versus-list type identity
actually matters; otherwise direct Python-container equality can manufacture an
evidence failure at a lossless serialization boundary.

## Reproduce lock inversion with controlled events, not focused-test luck

**TASK-20013, 2026-08-23.** The affected Console aggregate hung under load even
though its standalone and focused tests passed. Captured thread stacks exposed
an ABBA cycle: one thread held the config-file lock while waiting for the
settings-rebuild lock, while another held the settings-rebuild lock and waited
to re-enter the config-file path. An event-controlled child-process regression
forced that ordering without timing sleeps; enforcing the single config-file →
settings-rebuild → settings-cache lock order removed the cycle.

**What to do.** When an aggregate-only hang suggests lock inversion, capture
all thread stacks before interrupting it, encode the observed interleaving with
events or barriers in a bounded child process, and enforce one documented
global lock order. Focused green runs alone do not exercise the conflicting
owners under representative load.

## A completed Textual product contract can still be cancelled by test-loop teardown

**TASK-20009, 2026-08-21.** The real-provider three-turn Console benchmark
intermittently exited nonzero only on Change Review-enabled samples. The first
failure wrote no terminal record because `asyncio.CancelledError` sits outside
`Exception`; after preserving `BaseException` failures with a content-free
traceback function name, the failure localized to Textual's
`Screen._message_loop_exit`. Inspection of the retained failed profile showed
all three user/assistant pairs, the confined `fs_write`, and review finalization
were already durable. Textual 8.2.8 was propagating cancellation from a child
widget message-loop during `App.run_test()` shutdown, not cancelling the
benchmark task or leaving Change Review work alive. A six-sample smoke passed,
but the first 93-sample attempt reproduced at sample 34; only the long run made
the teardown flake undeniable.

**What to do.** For long mounted Textual evidence runs, distinguish three states:
the product contract completed, the caller task has a real pending cancellation,
and an owned child loop cancelled during context-manager exit. Preserve
`CancelledError` as a durable failure first, including only a privacy-safe origin.
Suppress it only when the full terminal contract is already proven and
`asyncio.current_task().cancelling() == 0`, then continue through explicit thread,
provider, database, shadow-operation, and source-write ownership checks. Never
blanket-swallow cancellation before the product assertions, and do not trust a
short smoke as the sole oracle for a lifecycle race that appears after dozens of
clean samples.

---

## A display-only durable overlay belongs at the render boundary

**TASK-19502, 2026-08-21.** Nonblocking Change Review publication needed the
mounted Console to re-derive file-change markers after the assistant turn had
already completed. The first implementation put that projection in the shared
`_native_console_messages()` accessor. Review showed that citation discovery,
message actions, image preparation, state fingerprints, and other non-render
callers would then all receive display-only TOOL rows and repeatedly pay the
join/injection cost. The focused marker tests were green because they asked only
whether the marker appeared; they did not prove unrelated consumers stayed on
the canonical store view.

**What to do.** Keep the store accessor canonical. Apply a durable, display-only
overlay at the narrow transcript render boundary, cache only its durable query
input by a content-free revision, and project it over the fresh message list.
Test both the visible result and that the source list remains byte-for-byte
unchanged; otherwise a correct-looking marker can still create hidden semantic
and performance regressions elsewhere.

---

## Collision normalization must reserve the whole untrusted namespace

**TASK-22510 review follow-up, 2026-08-28.** A regression proved that two native
tool calls carrying the same provider ID were independently displayed and
authorized by suffixing the later ID (`x`, `x` became `x`, `x#1`). A security
review then supplied the adversarial batch `x`, `x#1`, `x`: the model-controlled
middle ID already occupied the generated suffix, so the first and third commands
again shared one approval identity. Provider-generated IDs could create the same
shape. The ordinary duplicate test was green while the authorization boundary
remained bypassable.

**What to do.** When an untrusted identifier is normalized into an authorization,
deduplication, cache, or routing key, reserve every untrusted identifier in the
whole batch before generating replacements, then reject/skip both reserved and
already-generated candidates. Cover adversarial pre-suffixed IDs, processing-order
permutations, provider-generated fallbacks, and genuinely missing IDs; testing only
two identical values does not prove namespace separation.

**Recurred, TASK-24309, 2026-08-30.** The first per-call Agent Lesson approval
check rejected duplicate lesson call IDs only among Library lesson rows. A
builtin or MCP call in the same batch could carry the same ID, so the approval
decision map could alias the forced lesson decision even though the Library
subset itself was unique. The mixed-provider collision regression failed until
call IDs were required to be unique across the complete review batch before any
provider-specific classification or stamping.

For per-call authorization, the collision namespace is the whole batch and all
providers that consume the decision map—not the subset owned by the policy
being added.

## Persist test configuration through the same durable seam production reloads

**TASK-22988 (renumbered from TASK-22507), 2026-08-26.** The joined Roleplay-resume gate exposed seven
deterministic Console native-flow failures. Baseline comparison proved that all
seven predated the feature. The tests prepared provider keys, endpoints, models,
and defaults by mutating only `app.app_config`. Mounting Console legitimately
persisted missing scoped-rail state, which invalidated the config cache; provider
readiness then reloaded the hermetic per-test config from disk and lost every
snapshot-only value. The symptoms looked unrelated: prompt summaries fell back to
the default endpoint, sends became setup-blocked, the configured model vanished,
and focus stayed on the setup modal. Persisting the fixtures through the production
config writer, force-reloading, and synchronizing the active session made the exact
seven nodes green. A regression now proves the prepared provider state survives a
real cache-invalidating save/reload.

**What to do.** If mounted production code can save or reload configuration, write
test settings into the isolated `TLDW_CONFIG_PATH` through the production save seam
before mount; do not rely on direct mutation of a boot-time config snapshot. For a
post-mount provider change, update both the durable config and the active session
settings through their normal synchronization path. Include one deliberate
save/reload in the test so a future cache invalidation cannot silently erase the
fixture.

## A same-route rail press can invalidate the control a harness is about to press

**TASK-613, 2026-08-28.** The Skills import integration helper pressed the Skills
rail row and immediately queried the already-visible Import button. On an empty
Library, Skills was already the active canvas, so the query returned the old button
while the same-row rail event was scheduling its canvas recompose. The helper then
pressed that about-to-be-detached instance; its event never reached the screen, and
every import test spent 30 seconds waiting for a row that could not open. Directly
calling the handler made the feature look healthy but weakened the intended UI-path
evidence. Letting the route press settle before resolving the action restored the
real Button path and cut the focused case from a timeout to about three seconds.

**What to do.** In a mounted Textual flow, settle a route-changing press before
querying a control owned by that route, even when the requested route is already
visible. Resolve the control after settlement and then press that live instance;
otherwise a green direct-handler substitute or a red stale-widget timeout says
nothing about the production event path.

---

## A detached mount echo can erase a newer terminal snapshot

**TASK-613 review round 1, 2026-08-28.** Moving skill-import state to an
app-owned coordinator fixed routed screen cancellation, and the barrier-controlled
routes passed. The complete task file then failed only for the fastest mocked URL
success: the accepted operation published its terminal receipt before the disabled
Input's recompose finished, and that detached Input's delayed `Input.Changed` event
rewrote the shared path and cleared the newer status/review target. The service call
had landed correctly, so assertions limited to mutation count or admission would
have missed the user-visible loss.

**What to do.** When a Textual change handler can clear terminal state, require its
event control to be the currently mounted control, in addition to operation and
route generations. Include one production-shaped completion fast enough to race
the accepted-state recompose, and assert both the authoritative snapshot and the
mounted outcome copy.
---

## A third-party lifecycle fake must preserve irreversible global state

**TASK-24532, 2026-08-30.** The first OpenTelemetry initialization regression
used a fake metrics API that accepted every `set_meter_provider()` call and a
fresh instrumentor that always allowed `instrument()`/`uninstrument()`. The test
therefore reported a successful retry after a late setup failure. Independent
review against the upstream implementation showed that the real provider is
set once and the system-metrics instrumentor is a process singleton. In
production, the first attempt had already published irreversible global state;
the apparent retry could use the wrong provider, leak the replacement, expose a
partially initialized meter, or uninstrument work owned by another subsystem.
Replacing the permissive fakes with one lifecycle-accurate harness made those
failures reproducible and drove deferred publication plus ownership-aware
cleanup.

**What to do.** When code coordinates a third-party process global, model that
library's actual state machine in tests: set-once publication, singleton
identity, silent early returns, ownership transitions, and cleanup semantics.
Use distinct candidate objects and assert which one becomes globally visible.
A fake that accepts repeated setup calls proves only the application branch,
not that retry or cleanup is valid against the real library.

---

## A workflow dispatched from a branch may still test a different ref

**TASK-25706, 2026-08-30.** A temporary native-Windows validation workflow was
dispatched with `--ref codex/post-merge-windows-validation`, and its run reported
the temporary workflow's custom job and step names. That looked like branch
evidence, but the inherited resolver explicitly checked out `dev`, exported that
SHA, and the Windows job checked out the exported value. The run therefore tested
`dev`, not the temporary branch or its validation commit; matching workflow UI
labels proved only which YAML definition ran.

**What to do.** For a branch-specific validation, pin checkout to the dispatched
commit (`github.sha`) and print `git rev-parse HEAD` in the job summary. Compare
that recorded SHA with the intended branch head before treating results as
evidence. A workflow may legitimately load its definition from one ref and test
another, so the workflow name, step names, and dispatch ref are not sufficient.
## Cancelling a Textual worker does not cancel its underlying thread

**TASK-24406, 2026-08-30.** The Personal Context review modal originally ran
its commit worker with ``exclusive=True`` and still allowed Escape, backdrop,
and Close dismissal while the service call was blocked. A production-shaped
test released the blocked call after dismissal and proved that the canonical
profile mutation still completed: Textual cancelled the worker task, but it
could not stop the thread already executing the synchronous commit. The hidden
completion could leave durable data changed with no success or recovery state
visible to the user.

**What to do.** Treat an irreversible threaded operation as a modal state, not
as a cancellable UI task. Before the canonical mutation, persist a compare-and-
swap reservation such as ``committing``; freeze selection and edit controls;
block Escape, backdrop, and Close dismissal; and expose a distinct
outcome-unknown recovery state if rollback cannot restore the reserved draft.
Verify with a production-shaped blocking test that tries every dismissal path,
then releases the real thread and inspects durable state.

## A privacy assertion must inspect every default durable owner, not only the primary database

**TASK-19908, 2026-08-22.** Trace capture tests proved that AgentRunsDB and the
projected ledger omitted hidden reasoning, credentials, and local tool content. An
independent quality probe then decoded the default filesystem run log and found the
same raw tool result persisted there, including an explicit hidden-reasoning phrase.
The projection and its database owner were safe, but the product still violated the
privacy contract because a second, default-enabled audit owner had not been included
in the test oracle.

**What to do.** For any capture/privacy change, inventory every durable owner reached
by the real service seam (database rows, sidecars, files, caches, exports) and inspect
their decoded persisted bytes. A green projection or sanitized primary table proves
only that owner. When content is intentionally withheld, also verify that recovery
handles and user/model guidance do not promise a nonexistent full copy.

**Recurred, TASK-18932.1, 2026-08-26.** Clearing thinking from a deleted Chat message
was insufficient: the trigger-authored `sync_log` upsert and encrypted Sync outbox
still retained the pre-delete reasoning. The deletion tests became meaningful only
after they inspected all three durable owners and proved that the sole surviving
records were a content-free tombstone plus its hash-only conflict proof. For private
fields, deletion coverage must include queued and historical synchronization
sidecars, including rollback behavior when their cleanup shares a transaction.

**Recurred, TASK-24193 (2026-08-28).** A Trace hardening change reused the
4,000-character summary sanitizer for the filesystem run log, silently destroying the
full safe record that `search_run_log` recovery handles promised. The same path then
missed real `read_file` content because the provider had already replaced its absolute
locator with a placeholder before the generic path detector ran. The real-seam pair—a
large safe non-file result plus an actual file result—exposed both errors. Durable
sanitization must keep each owner's fidelity contract and must classify known file
tools by tool identity, not only by content that an earlier boundary may have altered.

## Privacy cleanup must preserve SQLite's concurrency contract

**TASK-24723, 2026-08-30.** Terminal Personal Context proposals were correctly
reduced to content-free receipts in their live rows, but an exact-byte inventory
found the old encrypted payload and wrapped DEK still present in SQLite storage.
The first correction forced ``journal_mode=DELETE`` together with
``secure_delete=ON``. That made the byte inventory pass, but the existing
production-shaped export snapshot tests then timed out or raised ``database is
locked``: changing the journal mode had silently removed the WAL concurrency
contract that lets writers proceed while an export holds a stable read snapshot.

**What to do.** Keep WAL enabled when the repository's readers rely on snapshot
concurrency. Use ``secure_delete=ON`` for freed database pages, attempt a
zero-wait ``wal_checkpoint(TRUNCATE)`` after privacy-sensitive commits, and retry
cleanup after an application-owned read snapshot closes. A reader may
legitimately pin historical WAL frames until it finishes, so verify both halves
of the contract: exact old ciphertext/DEK bytes disappear once the snapshot is
released, and a writer still completes while the snapshot is open. A passing
shredding test obtained by changing journal mode is not sufficient evidence.

---

## An outer SQLite rollback cannot undo a write committed by another database

**TASK-19900.1 fix review, 2026-08-22.** Console temporary promotion wrapped
conversation, policy, messages, attachments, and sidecars in one ChaChaNotes
transaction, but `create_conversation` also linked WorkspaceDB membership from
inside that transaction. Failure injection against only the Chat database made
the bundle look atomic. A real two-file probe failed the later policy write and
found a committed workspace membership with no surviving conversation; retry
then produced a second membership identity. The two connection-local transaction
managers could not provide the cross-database atomicity the call graph implied.

**What to do.** State the database boundary whenever claiming transaction
atomicity. For a derived row in another database, validate its target before the
authoritative transaction, commit the authority first, and perform an idempotent
post-commit projection with durable-source reconciliation after failure/restart.
Test with two real temporary SQLite files and inject failures both before and
after the authority commit; one in-memory database or mocked registry cannot
prove the absence of cross-database orphans.

---

## Warmups must not consume deferred work created by measured samples

**TASK-23113.4, 2026-08-30.** The provider-trace latency gate moved SQLite
auto-checkpoint work off the pre-dispatch reservation path and onto terminal
settlement. An early benchmark ran every reservation warmup and measurement
before every settlement warmup and measurement. The settlement warmups therefore
checkpointed WAL pages created by measured reservations, so the retained
settlement samples did not include the maintenance cost the optimization had
shifted downstream. A second attempted fix set a smaller checkpoint interval on
every ChaChaNotes connection; the trace gate improved, but alternating ordinary
message-write probes showed worse non-trace p95 latency. The accepted fixture
warms both phases first, then measures reservation followed by settlement, records
WAL allocation and close cost, and keeps the checkpoint override scoped to the
critical trace transaction with exact restoration.

**What to do.** When an optimization defers maintenance to a later phase, finish
all phase warmups before any retained sample can create deferred work. Measure the
downstream owner and teardown explicitly, and verify unrelated production writes
retain their prior policy. A fast critical-path sample is not evidence if an
unmeasured warmup, later writer, or close operation silently pays its cost.

---

## Textual's geometric center is not the painted row for an even-height one-line control

**TASK-16001, 2026-08-13.** A compositor regression helper sampled
`region.center.y` to verify that collapsed rail copy was vertically centered.
The focused RED failures initially hid the helper defect. Once the rail copy was
correct, all eight visual cases raised `AttributeError`: in the installed Textual,
`Region.center` is a `(float, float)` tuple, not an object with `.y`. Replacing it
with `region.y + region.height // 2` removed that error but still sampled an empty
row for even-height controls. Measurement showed a 28-row button at `y=7` painted
its sole centered row at `y=20`, the upper middle:
`7 + (28 - 1) // 2`.

**What to do.** For a one-line Textual control, sample the integer painted middle
row with `region.y + (region.height - 1) // 2`; do not assume `Region.center` has
named coordinates or that lower-middle sampling matches Textual's alignment.
Keep a separate one-painted-row assertion so the midpoint check cannot pass on a
multi-line control.

---

## A geometry harness must mount the production hierarchy and stylesheet

**TASK-16221, 2026-08-14.** The first Watchlists Read geometry harness mounted
`ArticleListPane` directly where production mounts a detail wrapper, title, and
nested pane. With the inner table capped at 42 rows, the simplified harness
reported a contained, painted pager. A production-shaped probe showed the real
detail pane growing to 51 rows inside a 50-row ITEMS region, placing the pager
exactly outside the clipped box; the rendered frame contained none of Previous,
Page 1, or Next. Rebuilding the harness with the real wrapper/title hierarchy
made the regression fail, and the correct 40-row table cap kept the pager inside.
During final live QA, a second simplified host loaded consolidated widget/screen
CSS but omitted the app bundle; its regions measured 8/5 rows until the harness
used the exact `TldwCli.CSS_PATH` stack.

**What to do.** For layout limits, reproduce every production ancestor that
contributes rows and load the same stylesheet sources in the same order as the
application. Assert containment and compositor text, not only a child widget's
declared height. A shortened DOM or partial stylesheet can make both overflow
and clipping tests pass for a product path that still hides its controls.

**Recurred, TASK-21161 (2026-08-23).** The shared Workspace-create modal's bare
`App` harness omitted `WorkspaceCreateModal.BUNDLED_CSS`. Ten click-path tests
failed on both the task branch and untouched latest `dev`: at 80 columns the
Browse/Add controls landed at x=80/96, outside the viewport. Simply widening the
Pilot screen to 120 made the same controls move to x=120/136, proving this was
not a small-screen product defect; without the modal stylesheet, each default
`1fr` child claimed another full row width. Mounting the production bundled CSS
in the harness made all 23 modal tests pass at the default viewport.

**Recurred, TASK-16478, 2026-08-15.** A picker-comparison investigation
rendered `EnhancedFileOpen` in a bare `App` (widget DEFAULT_CSS only) and
concluded the dialog was fine; the user's live app showed no Select/Cancel
buttons at all. Under the app bundle, the bare `Select { width: 100% }` rule
beat the dialog's DEFAULT_CSS, crushed the filename input to 6 columns, and
laid the buttons out at x=161/178 inside a 152-wide dialog -- clipped. The
bare-host screenshot even contained the buttons, and a truncated text
extraction of the bundled render hid their absence. The fix's regression test
(`Tests/UI/test_enhanced_file_dialog_bundle_css.py`) registers the exact
`TldwCli.CSS_PATH` stack and asserts button containment -- it failed red
against the unfixed bundle without touching app code.

**Recurred, TASK-19913, 2026-08-23.** A latest-dev merge moved the Trace
screen and timeline from automatically registered `DEFAULT_CSS` into
consolidated `BUNDLED_CSS`. The branch's plain-`App` geometry harness then
mounted both widgets without their production defaults and reported four
layout/style failures. Migrating the harness to `ConsolidatedCSSApp` fixed the
false containment failure, but full-detail and brush-theme tests still failed:
the checked-in generated widget sheets predated this branch's expanded
`BUNDLED_CSS`. Rebuilding them restored the production interactions without a
specificity workaround.

**What to do.** After merging the consolidated-CSS system into a branch that
changed class-level CSS, update production-shaped harnesses to inherit
`Tests.UI.consolidated_css.ConsolidatedCSSApp`, rebuild with
`python -m tldw_chatbook.css.build_css`, and run CSS-build integrity tests.
Loading only the app bundle is insufficient, and testing regenerated sheets
against stale source is equally misleading.

**Recurred, TASK-20937.5, 2026-08-23.** Character-art fitting passed in a
lightweight Console harness, but the production bundle produced a different
available cell box and exposed a one-row/one-column mismatch between the
requested avatar size and the mosaic grid actually painted. Thumbnail rounding
had allowed the two rendering paths to derive slightly different aspect-ratio
results. Loading the bundled stylesheet in the mounted shape matrix made the
failure deterministic; sharing the mosaic grid calculation made graphics and
mosaic settle to the same exact cell box.

**Recurred, TASK-21595, 2026-08-25 — and this time the app bundle itself was
not enough.** A geometry A/B for `PersonaBuddyWidget` used a plain `App` with
`CSS_PATH = css/tldw_cli_modular.tcss`, i.e. the whole app bundle. The test
passed, and so did the mutation that made the widget content-sized. The
bundle contains **zero** `#persona-buddy-frame` rules: consolidated widget CSS
(TASK-15450) lives in the generated `widget_defaults_{self,scoped}.tcss`, which
the app registers via `_get_default_css` and only
`Tests.UI.consolidated_css.ConsolidatedCSSApp` reproduces. The widget was
mounting unstyled, so the probe was measuring nothing at all. **"I loaded the
app bundle" is not the same as "I loaded what production loads"** — for any
widget whose CSS was consolidated, inherit `ConsolidatedCSSApp`. The tell was
the surviving mutant, not the failing test: a green geometry assertion under a
harness that cannot see the rule looks identical to a correct one.

---

## An exact live-test gate must be the first gate that can skip the test

**TASK-15676, 2026-08-13.** The opt-in Moonshot/Z.ai paid harness required an
exact provider flag plus a nonblank provider key, and its user documentation
showed that two-part command. The first default run never reached that contract:
the repository's `slow` marker skipped the test with `Need --run-slow`. Removing
that marker exposed the same problem from `optional` and `--run-optional`. Even
with the documented environment flag and key present, the documented command
could not execute the live case because unrelated collection-time gates ran
before the test body's explicit safety check.

**What to do.** When the public contract is an exact environment/key gate, do
not also apply repository markers whose plugins skip before the test body unless
every required CLI flag is part of the documented contract. Keep descriptive
markers such as `integration`/`allow_network`, put the paid-call guard at the
top of the test, and cover its truth table plus the default skip reason. That
makes default collection safe while ensuring the opt-in command can actually
reach the code it claims to verify.

---

## A targeted async completion must not rebuild a surface that is transitioning away

**TASK-15706, 2026-08-13.** Database Notes began loading its folder tree in a
background worker. If the user switched to File Notes before that worker
finished, the completion path tried to synchronize `#library-notes-canvas`.
That canvas was legitimately absent during the source transition, so the shared
sync helper used its generic full-screen recompose fallback. The fallback
invalidated the just-pressed Files source control and intermittently left the
transition without its retained File Notes surface. Folder-tree tests all
passed; only the production-shell source-switch tests reproduced the race.

**What to do.** An async completion that owns one optional child surface should
first confirm that exact surface is still mounted. If it is absent because the
user navigated away, treat the result as cached state and skip the paint; do not
invoke a generic whole-screen fallback. Verify the fix through the real route
transition, and compare the same test against the untouched baseline before
attributing nearby focus failures to the branch.

---

## A schema-version label does not make a synthetic database historical

**TASK-15705/TASK-15707, 2026-08-12.** Raising ChaChaNotes from v35 to v36
first broke migration tests that had either stamped a tiny hand-written database
with the then-current version or pinned ``_CURRENT_SCHEMA_VERSION`` while using
the evolving bootstrap SQL. The former skipped migration with required tables
missing; the latter silently included columns that did not exist at the claimed
historical version. Focused tests for the new v36 migration passed, but the full
DB suite exposed both fixture classes: current-version fixtures failed during
startup maintenance, while an incomplete v24 fixture failed only after reaching
a much later migration.

**What to do.** A migration fixture must prove the historical preconditions that
matter to the migration under test: schema version, required tables/columns, and
the absence of fields being introduced. When later code needs a complete current
database with one malformed record, create a real current database and alter only
that record or table; do not label a partial schema as current. After every schema
bump, run the complete DB migration suite, not only the new migration module.

---

## A "slow-accept listener" does not delay TCP connect() — it delays accept()

**TASK-15473, 2026-08-11.** Writing an evidence test that the event loop stays
responsive during a non-blocking socket probe, the task's own brief suggested "a
slow-accept listener" as the portable way to simulate an unresponsive server. Timed
directly before writing the test: a real `socket.listen()`ing server that never calls
`accept()` still let a client's `socket.create_connection()` complete in ~7ms. TCP's
three-way handshake completes at the OS kernel level as soon as a connection is
queued in the listen backlog — independent of whether the application ever calls
`accept()`. A "slow-accept" listener therefore cannot be used to create connect-side
delay; it only delays whatever happens *after* the client tries to read/write, which
this probe (connect-then-immediately-close, no data exchange) never does.

What actually produced a real, mutation-verified delay: connecting to a private,
non-routed address (`10.255.255.1`) that neither answers the SYN nor sends back an
ICMP unreachable — a genuine kernel-level "black hole" — measured to hang for the
full requested timeout in this sandbox (no immediate "network unreachable"). The
resulting test caught a real regression: reverting the probe to a blocking
`socket.create_connection` call inside the coroutine dropped a 5ms-period heartbeat
task from ~44 ticks to 0 during the same ~0.25s window.

**What to do.** Before trusting "slow accept" (or similar accept-side framing) to
simulate a connect-side timeout in a test, time it directly — a bound-and-listening
socket with a deliberately delayed `accept()` will not slow down a bare `connect()`
on any common OS. For a genuine connect-timeout test, either use a real black-hole
address (accepting the environment-dependence, verify empirically first) or
mutation-test whatever mechanism you do use against the blocking equivalent it's
supposed to replace — the ~44-vs-0 heartbeat contrast is what proved this test was
not vacuous.

---

## Style probes are not render evidence — capture the frame

**TASK-15421 AC3, 2026-08-11.** The Studio exact-ID input's typed text
vanished while focused in the live TUI. The hunt fixated on border rules for
hours because every probe asked `styles.border` — which was empty, correctly,
in both the live-matching harness and run_test — so the harness appeared to
CONTRADICT the live app and the divergence got recorded as an unexplained
live-vs-run_test cascade anomaly. There was no divergence: the reset-tier
accessibility rule `*:focus { outline: solid }` paints the outline OVER the
widget's outermost rendered lines (its own comment warns of this), and on a
height-1 widget that line IS the only content line. The obscuring reproduced
in run_test all along; no probe ever looked at a rendered frame. One
`export_screenshot()` assertion (`assert "studio-model" in frame`) found in
minutes what specificity analysis could not, and now pins the fix in
`Tests/UI/test_speech_live_render_defects.py` — a file whose own docstring
already teaches a version of this lesson ("the tests asserted the things a
test naturally reaches for ... none of which is what was wrong").

**What to do.** When the defect is "the user cannot SEE something," the
oracle must be the rendered frame, not computed styles: in run_test that
means `app.export_screenshot()` (the SVG carries every glyph as text, so a
plain `in` assertion works) or the compositor strips the existing UI tests
use — NOT `App.export_text()`, which does not exist in this repo's Textual
(8.2.7; the probe that first tried it died on AttributeError) — and live it
means the tmux `capture-pane` text. `styles.border`, `styles.height`, and
`region` all report the widget's own properties and are blind to anything
painted over it — outlines, overlays, tooltips, sibling z-order.
Before declaring a live-vs-harness divergence, confirm both sides were asked
the SAME question at the same oracle level; here the "divergence" was one
side being read at the style level and the other at the pixel level.

---

## Preserve the visible set across a reorder, not its former first row

**TASK-15455, 2026-08-11.** Console transcript windowing initially preserved a
lazy window across refreshes by finding the first previously visible message id
that still existed in the new ordered list. That was correct for append-only
streaming and session-local deletes, but wrong for branch/path reorder: moving a
later visible message ahead of that chosen id put it into the newly computed
hidden prefix. The focused windowing tests were all green. The pre-existing
signature-cache reorder contract caught the missing mounted row.

**What to do.** When a windowed projection accepts a reordered full list,
preserve the minimum new index of every surviving previously visible item (plus
any explicit selection handoff), not the new index of one former boundary item.
Include an order-sensitive DOM assertion in the reachable regression set; cache
counts alone prove reuse, not that every reused row stayed visible.

---

## A fix proven at one layer can be unreachable through the product path

**TASK-15420, 2026-08-11.** TASK-2260 (2026-08-04) shipped custom-endpoint
model/voice passthrough in `OpenAITTSBackend`, pinned by mutation-verified
backend tests and a real-socket keyless server test, and its user guide was
live-verified — by varying the *voice*. The Console speak path, however, had
been rerouted through the request-admission layer (2026-07-26), whose
`resolve_legacy_route` allowlist rejected every non-official OpenAI *model id*
before the backend was even constructed. The documented flow (exact custom
model name) failed on every Console speak for weeks while all ~2,900 TTS-area
tests stayed green: the tests proved the backend layer, the live check varied
the one axis the upstream layer did not constrain, and nothing exercised the
full admitted path with a custom model. Found only by end-to-end UAT driving
the real TUI against a request-recording mock server.

Sub-trap from the same session: `TTSAudioResponse.byte_stream` is lazy — a
probe that calls `synthesize_default` and prints the response "succeeding"
proves nothing, because the HTTP request only fires when the stream is
consumed. The first counterfactual probe "passed" with zero requests at the
server; only draining the stream produced the real request.

**What to do.** A regression test for a passthrough/compatibility contract
must enter at the outermost admitted path (here: `synthesize_default` down to
the adapter), not the layer that was fixed — any layer added above the fix
inherits the chance to re-impose the constraint. When live-verifying, vary the
axis the bug is about (the model, not just the voice). And never trust a
lazy-stream API's return value as evidence of I/O — consume it and assert at
the far end (the recording server).

**TASK-13204, 2026-08-10.** A clone-shutdown regression initially asserted
`TTSAdapterRegistry._total_leases()` while the provider was past its bounded
shutdown deadline. The registry had already moved the active adapter record
into its retained closing-record collection, so `_total_leases()` returned zero
even though the exact record still held a lease. That false measurement first
made the race look fixed. Inspecting the retained record proved the real bug:
generic late-operation cleanup could release an executing clone lease before
its protected materialization boundary finished. The corrected regression
asserts the closing record's lease count and mutation-fails when the protected
execution waiter is removed.

**What to do.** When testing ownership after a bounded shutdown transitions to
retained cleanup, measure the owner in its terminal collection or assert the
actual release/close barrier. A convenience counter scoped to active/retired
records may truthfully return zero after records are transferred, while
definitive cleanup is still outstanding.

---

## A registry entry needs both inventory and behavioral-ratchet coverage

**TASK-13203, 2026-08-10.** The new `tts.profile_migration_backup` SQLite owner
was present in the central policy registry, the curated owner inventory, and focused
migration lifecycle tests. Those suites all passed. The complete private-SQLite gate
still failed because the owner was absent from the generic centralized-backup
behavior matrix, so its declared `centralized_backup_allowed` capability had no
owner-parameterized seam test.

**What to do.** When adding a policy-registry owner, search for both inventory
ratchets and capability-derived behavioral matrices. Passing the feature's own test
does not prove that every capability bit has generic seam coverage; run the registry
module's complete contract suite before closeout.

---

## Deleting a duplicate guard requires tests for every bypass mode

**TASK-859, 2026-08-02.** During specification review, deleting
`SecurityValidator.ALLOWED_SCHEMES` looked like clean policy consolidation because
`Utils.egress` also has scheme policy. Direct characterization disproved that:
`Utils.egress` returns allowed immediately when `[web_security].enabled = false`,
before evaluating its scheme policy. Without the subscription-local HTTP/HTTPS
allowlist, disabled egress would therefore admit `ftp://` subscriptions. The
regression test that disables egress and submits an FTP URL remains red without
the local boundary and green with it.

**What to do.** Before deleting an apparently duplicate guard, characterize every
disabled or bypass mode of the surviving owner and test that owner's boundary in
each mode. Consolidate only after evidence shows the remaining guard preserves
the contract there.

---

## A mixed sync/async interface needs an explicit, complete test double

**TASK-2510 final review, 2026-08-24.** The Watchlists backend controller gained
the synchronous `create_form_source_types` contract alongside its existing async
operations. Four collections-screen controller doubles were blanket
`AsyncMock` instances, so backend switching turned that synchronous call into a
coroutine and the Reader recovery path failed with `TypeError: 'coroutine' object
is not iterable`. Correcting all four doubles to provide a synchronous `Mock` for
that seam restored all 88 collections-screen tests and the 155-test focused
Task-2510 suite.

**What to do.** When an interface mixes synchronous and asynchronous methods,
model every seam explicitly with the correct call shape; do not use a blanket
`AsyncMock` as a complete interface double. Exercise the double through a real
integration path that crosses the synchronous seam, not by asserting that the
mock exists.

---

## A fake written to match your call site validates the mistake

**The trap.** You write a test double to match how you are calling the real thing. If
the call is wrong, the double is wrong in the same way, and the test passes forever.

**What happened.** Three times on one branch (task-684 series):

- `cancel_media_ingest_jobs_batch` is keyword-only; it was called positionally. The
  fake declared a positional parameter, because it was written to match the call.
- The remote-ingest poller asked for an `offset` the real client did not accept. The
  fake declared `offset`. **Pagination was dead in production** and 900+ tests were green.
- `MediaIngestJobStatus.result` was typed as a different domain's model. Every fixture
  matched the wrong model, so **every completed job was unparseable** and the queue
  would have shown jobs stuck at "queued" forever.

**What to do.** For anything crossing a seam you do not own, assert against the **real
signature** and a **verbatim captured payload**, not a hand-written double:

```python
assert _accepts_keyword(RealService.method, "offset")     # the real signature
LIVE_RESPONSE = { ... }                                    # pasted from the wire
```

A fake can agree with a wrong assumption; `inspect.signature` cannot.

**Sharpest variant (task-16847, 2026-08-16).** A double can stand in for an attribute
that does not exist at all. `a8082fe85`'s launch test set
`instance.call_from_thread = ...` and `instance.push_screen = ...` on a
`ChatScreen.__new__` instance — but `Screen` defines *neither* (both are App-only in
Textual 8), so pressing `y` in the real app raised `AttributeError` inside the thread
worker while the test stayed green, and the repo-wide guard
(`Tests/test_call_from_thread_guard.py`) sat red on dev for two days. When a unit test
must fake threading/navigation seams, patch the **collaborator** (`app`) — never spell
a new attribute onto the class under test; an instance monkeypatch is also an
existence claim, and nothing checks it.

**Widest blast radius (TASK-17065, 2026-08-17).** One module diverged from the house
pattern in *two* ways at once, and the single fake at its seam mirrored both.
`RAG_Search/reranker.py` grew its own credential path (`self._settings =
load_settings()`, then a hand-rolled `if/elif` reading
`settings["API"]["<provider>_api_key"]` — a key `load_settings()` never builds) *and*
its own dispatch convention (a positional argument list handed to `chat_api_call`
through `run_in_executor`, which forwards positionals only). The seam fake,
`Tests/RAG_Search/test_reranker_degraded_paths.py`'s `def
fake_chat_api_call(api_key, messages_payload, provider, model, temp, maxp)`, declared
the caller's own wrong order *and* planted a `_settings` table so the call got past
the credential gate. Agreeing with both defects, it left ~2,500 green tests unable to
see that reranking completed a scoring call for **zero of the 29 providers**
`chat_api_call` dispatches — for the entire life of the feature. Binding the same call
through `inspect.signature(chat_api_call).bind(...)` printed the truth in one line:
`api_endpoint='THE-API-KEY'`, `api_key='openai'`, `temp='gpt-4o-mini'`,
`system_message=0.25`, `streaming=128` — the mis-binding had also silently switched
STREAMING on, a third defect nobody had filed.

Two rules out of it:

- **A fake at a seam you share with the rest of the app must bind against the real
  signature, never re-type the call site's argument list.** Note how far this reaches:
  even the guard written specifically to catch this
  (`test_reranker_dispatch_binding_against_the_real_chat_api_call_signature`) first
  asserted a literal tuple *it typed itself*, so it guarded a copy of the caller and
  caught nothing. It only became evidence once it drove the real `_call_llm_impl` and
  observed what landed. **And know exactly what `bind` buys you**, because the fixed
  fake's first docstring over-claimed it and the final review caught that:
  `inspect.signature(...).bind()` checks arity and keyword *names* only — it is BLIND
  to order. Re-measured on this very call:
  `bind("THE-KEY", [...], "openai", "gpt-4o-mini", 0.25, 128)` is ACCEPTED (it simply
  lands the key in `api_endpoint`), while `bind(provider="x", ...)` raises
  `unexpected keyword argument`. What actually catches a mis-ordering is the landing
  ASSERTIONS on a guard that drives the real caller — plus, cheaply, refusing
  positional arguments at the fake (`assert not args`) when the seam is keyword-only
  by contract. Mutation-checked: reverting the call site to positional now fails with
  *"positional arguments landed here: ('openai',)"*, not with a bind error.
- **A feature that resolves credentials itself is a divergence to justify, not a
  default.** The fix here was a DELETION: all 29 handlers already resolve their own key
  or need none, and every other `chat_api_call` caller in the repo
  (`UI/Tools_Settings_Window.py`, `UI/Screens/evals_screen.py`,
  `Chat/console_provider_gateway.py`) already passes keywords and omits `api_key`. The
  reranker was the sole outlier, and being the sole outlier is exactly what broke it.
  Before writing a lookup at a shared seam, count the callers who do not have one.

---

## Mutation-test every guard you add

**The trap.** A test that cannot fail is worse than no test: it reports safety that
does not exist.

**What happened.** Three fixes in one session were **vacuous** and only mutation
testing caught them:

- A precedence trap patched `get_cli_setting` to raise — but the code under test wraps
  that call in `except Exception:`, which **swallowed the AssertionError**. Deleting the
  entire precedence branch still passed.
- A wait helper's timeout was checked in the loop *header*, so a pause overshooting the
  deadline exited without re-testing the condition — reporting "never mounted" for a
  widget that had just appeared. The guard for "can the helper still time out?" passed,
  because it waited on a condition that is never satisfied.
- A first-run classifier keyed on an in-memory counter that resets each run, so a real
  failure in the first batch after restart was downgraded. Three tests passed with the
  bug present; none modelled a restart.

**What to do.** After writing a guard, break the thing it guards and confirm the test
fails. If it still passes, the guard is decorative. Prefer **recording and asserting
after the fact** over raising inside code that catches broadly.

---

## Mount dispatch can be attached before it is mounted

**TASK-15459, 2026-08-13.** A deterministic `asyncio.Event` barrier released
the Library source worker while `LibraryScreen.on_mount()` was still awaiting.
The fresh snapshot reached the screen and advanced its state generation, but
the rendered generation stayed behind and the targeted-sync recorder remained
empty. Instrumentation at the snapshot boundary showed the exact lifecycle
state: `is_attached` was true while `is_mounted` was false. The reconciliation
scheduler used `is_mounted`, so it silently discarded completion during the
Mount dispatch window. Changing only that scheduler boundary to attachment
authority made the RED race pass, while the detached-completion regression
continued to return `SUPERSEDED` with zero DOM calls.

**What to do.** When work may complete during a Textual Mount handler, do not
infer that attachment and mounting flags change together. Gate message-pump
scheduling on the lifecycle property the operation actually needs (attachment
for queuing to the screen), then keep a second current/attached guard at DOM
execution time. Prove both halves with Event barriers: completion during Mount
must render, and completion after detach must do nothing.

---

## Trigger cancellation from the state the test claims to cancel

**TASK-3771, 2026-08-11.** A QwenCloud native-tool regression claimed to prove
that cancelling after an incomplete streamed function call never executes the
tool. The test actually set its cancellation flag in the mock server's
``on_request`` callback, before the response body or partial call reached the
stream parser. Replacing both partial-call fixtures with ordinary final text
still passed: the test proved only pre-response cancellation and closure.

The corrected test waits until the real ``ConsoleChatStore`` receives a visible
checkpoint that follows an incomplete tool-call delta, then requests
cancellation while the chunked response remains non-terminal. A text-only
mutation now fails at the fixture guard, and the test separately proves the
live response closes exactly once without executing or pairing the partial
call.

**What to do.** If a cancellation test names a lifecycle state (after headers,
after one chunk, after partial tool state), trigger cancellation from an
observation downstream of that exact state. Do not trigger it at request
receipt and infer that later layers ran. Mutation-replace the special prefix
with a normal successful response; the test must fail for the reason it claims
to cover.

---

## A surviving mutant usually means a SECOND writer satisfies your assertion

**The trap.** You delete the code under test, the test stays green, and the reflex is
to strengthen the assertion. The real question is *who else produces the asserted
outcome* — if a second mechanism writes the same state, the test is measuring that
mechanism, not your feature.

**What happened.** Task-3313 (2026-08-09), twice in one session:

- Deleting the "Retry this batch" **options restore** stayed green because ingest
  options deliberately persist across submits — the form still held the values the
  restore was supposed to bring back. Fix: the test now *corrupts* the options and
  metadata between submit and retry, so only the restore can produce the asserted
  values. The mutant then failed on the exact line.
- Deleting the **fresh pre-flight trigger** stayed green because the test's
  programmatic `path_input.value = …` had armed the 0.8s typing debounce, which fired
  *after* the re-stage at test speed and re-ran the analysis the mutant no longer
  requested (found by wrapping the trigger and printing call stacks: the second call
  came from `_run_debounced_library_ingest_preflight`). In production that timer fires
  as a no-op long before a human reaches the button — a pure test-speed artifact. Fix:
  the harness stops the pending debounce before the action under test.

- TASK-21591 (2026-08-25), a third instance in a different domain: deleting
  `SplashScreen.on_mount`'s own `self.focus()` left **all four** new tests green.
  The second writer was Textual itself — `App.AUTO_FOCUS = "*"` auto-focuses the
  first focusable widget on screen mount, so setting `can_focus = True` was
  sufficient in the harness and the widget's own focus call was never exercised.
  Fix: the test host sets `AUTO_FOCUS = None`, so the shipped mechanism is the only
  path and the mutant reds. **Framework defaults are second writers too** — for any
  feature built on focus, selection, or auto-anything, find the classvar that does
  it for free and turn it off in the harness.

**What to do.** When a mutant survives, instrument for the *second writer* (wrap the
seam, record call stacks) before touching the assertion. Then either perturb the state
so only the code under test can restore it, or silence the background mechanism in the
harness — and re-run the mutant to see it actually die.

---

## A widget reference captured before a structural recompose is a silent key sink

**The trap.** A Pilot test grabs `path_input = screen.query_one(...)`, performs an
action that lands a state change, then `set_focus(path_input)` and `pilot.press(...)`.
If the state change took the STRUCTURAL recompose path, every form widget was
replaced; the captured reference is a detached widget. Focusing it "works"
(`screen.focused` even reports it), but keys go nowhere — no error, no typing, no
`Submitted`.

**What happened.** Task-3314 (2026-08-09): the inline-consent pilot tests captured the
ingest path Input, then let a pre-flight result land — which changes the type-group
set and forces the canvas's context-preserving full recompose. Enter then never fired
`Input.Submitted`; three probes (handler spies, app-level recorders, a source-level
log) all showed the handler simply never ran, and typing `x` changed nothing. The fix
was one line: re-query the input *after* the forecast settles. (Related but distinct
from the "pin object identity across the in-place path" lesson — here the identity
break is *expected*, and the test must follow it.)

**What to do.** In pilot tests, re-query any widget you focus or press *after* the
last action that can recompose its region; treat "keys typed but value unchanged" as
a detached-focus symptom, not a key-routing mystery.

---

## A new widget in a shared row needs geometry assertions, not just display/text

**The trap.** You add a small widget to an existing `Horizontal` row and assert it
displays with the right text. Textual's default `Widget`/`Static` width is `1fr`, so
the new child quietly claims the entire row and pushes every sibling off the screen
edge. Display and text assertions stay green — they never look at positions.

**What happened.** TASK-2154.1 added `#console-compact-status-marker` as the first
child of the Console control-bar action row. Six new Pilot tests passed (marker
visible, correct label, rails behave), but the 80x24 UAT screenshot showed **every
control button gone**: the bare `Static` took the row's full 78 cells and the buttons
laid out at x=79+, off-screen. One line — `marker.styles.width = "auto"` — fixed it;
a regression test asserting each button's `region` stays inside the screen locks it,
and was itself mutation-checked by deleting the width line (it fails: `x=90 + 16 > 90`).

**What to do.** When you mount anything into a laid-out row/column you do not own,
assert the **neighbours' geometry** (`region.x + region.width <= screen width`), not
only your widget's `display`. If your widget is a `Static`, set `width = "auto"`
explicitly unless you actually want it to eat the row.

---

## A dynamic Button label can repaint without reflowing its width

**TASK-3795, 2026-08-08.** Speech Lab composed its primary audio.cpp action as
`Test`, then passive runtime state changed the same mounted Button to
`Start & Test Connection`. Every test asserted the new label string and passed.
The live browser UAT nevertheless rendered only `Star`: Textual repainted the
reactive label but retained the original 16-cell layout width. The first honest
geometry regression measured 16 cells against the 25 required for the new label
and failed until the update called `refresh(layout=True)`.

**What to do.** When a mounted auto-width widget changes content, verify the
rendered region after refresh, not just its value. For dynamic Button labels,
request a layout refresh and assert that `region.width` can contain the visible
label; a correct reactive value does not prove that layout was recomputed.

**Recurred, TASK-22304, 2026-08-26.** A screenshot exported in the same turn as
the Console Send label changed to `Send | $` appeared to clip the dollar suffix,
even though the production callback requested a layout refresh. After one Pilot
pause, the button's settled region matched its declared width and the full label
painted at both 80x24 and 160x40. The first tooltip probe also appeared blank
because Textual's `run_test` disables tooltips unless `tooltips=True` is passed.

**What to do.** For dynamic-label and tooltip evidence, let the Pilot reach a
settled layout before capturing the frame. Enable tooltips explicitly in the test
host, hover the production control, and assert both the control geometry and the
mounted `#textual-tooltip` render. An immediate frame and a default-disabled
framework feature can manufacture two different false UI regressions.

---

## Test embedded panes at their allocated width, not the terminal width

**TASK-13205, 2026-08-11.** The Speech Lab clone-result geometry regression
mounted the pane in a 134-column Pilot viewport and proved every action was
inside the split. Live UAT still clipped **Save as Voice Profile**: the real
screen reserves a catalog rail, leaving the pane about 100 cells. At that width
the managed provenance wraps onto an extra row and the last action began one
row below the split. Re-running the same containment assertion at the pane's
actual allocated width reproduced the failure and justified a one-row minimum
height correction.

**What to do.** For a pane embedded beside a fixed or responsive rail, test the
pane at the width its parent actually allocates, including the wrapping-heavy
state. A terminal-size test can be truthful for a standalone harness and still
miss clipping caused by the production parent layout.

---

## Passing the suites a change touches is not passing the suites it can reach

**The trap.** You run the tests near your edit. The breakage is somewhere that merely
*depends* on it.

**What happened.**

- Deleting a module left a doc-enforced inventory (`Tests/RuntimePolicy`) listing it.
  Full-tree `--collect-only` passed: that only catches import errors, not stale
  assertions *about* the codebase.
- Adding a screen attribute in the wrong method broke the media viewer on **restored
  sessions**. `Tests/Library` (859 green), the state-level tests, and a live server run
  all passed and were blind to it.

**What to do.** Ask what your change can *reach*, not what it touches. For deletions,
that includes **inventories, audits and architecture-contract tests**, which assert
about the codebase rather than importing it. `--collect-only` over the full tree is
necessary and not sufficient.

---

## A new default invalidates legacy fixtures that never declared their mode

**TASK-16323, 2026-08-20.** Console project instructions became enabled for newly
created sessions. Three older test-helper families created real Console sessions but
never declared whether they exercised project instructions. Their agent, controller,
and citation tests then failed with `binding_unavailable`; two parked waiting for a
gateway call that could never happen and hit the configured timeout. The feature's
focused suites were green because those helpers lived in adjacent behavior suites.

**What to do.** When a new per-session feature changes a constructor/default, find
real-object test helpers that omit the new state. Fixtures for unrelated legacy
behavior should explicitly select the legacy-disabled mode; feature tests should
install the intended binding. Do not weaken the production default to rescue an
ambiguous fixture, and treat a newly parked test as an upstream dispatch-gate failure
before debugging its timeout body.

---

## A re-export hides a dependency from a module-name grep

**What happened.** `test_plaintext_ingest_events.py` imported a deleted function *via*
`ingest_events`, so grepping the deleted module's name never matched it. Worse, a
per-symbol reachability scan excluded it because the filename
`test_plaintext_ingest_events.py` *contains* `ingest_events.py` as a substring — the one
file that disproved the conclusion was skipped as "internal".

**What to do.** Match paths **exactly**, never by substring. Run
`pytest --collect-only` over the whole tree before deleting anything; it is the only
check that sees through a re-export.

---

## Compare failure *sets* from identical commands, never counts

**What happened.** A 3-vs-4 failure count between two *different* invocations
(one file alone vs. that file plus another suite) read as a regression. It was not.
Later, the same file gave 6, then 4, then 0, then 1, then 0 failures on unchanged code.

**What to do.** Run the **identical command** on your branch and on a clean `origin/dev`
worktree, and diff the failure **sets**. Counts across differing commands are
meaningless. Machine load changes which tests lose a race — this repo regularly has
10+ concurrent pytest processes from parallel agents.

**Second incident, and it applies to inventories, not just failures (TASK-23028,
2026-08-27).** The timer census pinned its clock-root COUNT (`>= 30`) plus a few
named roots. In one merge window a 10 Hz clock left the census (renamed callee) and
an unrelated root arrived — 35 → 35, every assertion green, and the blindness stayed
invisible until a holistic perf review re-measured idle CPU. The census now pins the
full root **set** (`EXPECTED_CLOCK_ROOTS`, equality with directional diffs). A
census whose cardinality is the assertion cannot see an exchange.

---

## A low-rate intermittent needs a loop, not a rerun

**What happened.** Five single-run attempts to capture a flaky test's traceback all
passed. Looping the file until one failed produced it on the first loop — and the
assertion identified the cause immediately (a test waiting on *state* then asserting on
*DOM*, before the recompose that renders it).

**What to do.** For anything below ~30% failure rate, loop with `-rf` and capture. A
single run is not a diagnostic. And prefer waiting for the **widget** over waiting for
the state that implies it.

---

## Reloading an IPC module can split exact type identity across spawn

**TASK-601, 2026-08-08.** A POSIX import-boundary test called
`importlib.reload()` on the module defining `WorkerContainmentIdentity`. The
already-imported executor retained the old dataclass object, while spawned workers
imported the reloaded class. Pickling still succeeded, but the parent's deliberate
exact-type bootstrap check rejected every worker identity; one test-side reload
therefore surfaced as 19 executor startup failures.

**What to do.** Test fresh-import boundaries in a subprocess. Do not reload a module
that owns IPC or serialized contract classes inside the shared pytest process; stale
importers keep the previous class identity even though the module name is unchanged.

---

## One thread must own reaping each spawned process

**TASK-601, 2026-08-08.** The local STT reader handled pipe EOF by calling
`Process.join()` before checking whether the controller had already detached that
generation. During graceful recycle, the controller also joined the same
`multiprocessing.Process`. On POSIX, the competing `waitpid()` calls occasionally left
the controller observing an unknown exit code and reporting a live worker even though
the child had exited. Removing the reader join entirely then exposed the other half of
the contract on macOS: an unreaped crashed group leader made `killpg()` return
`EPERM`.

**What to do.** Decide process ownership under the lifecycle lock before joining. A
reader may reap an unexpectedly exited generation only while it is still current; once
the controller detaches that generation, only the controller may reap it. Cover both
branches: a deterministic stale-reader test and a repeated real-spawn recycle test.

---

## A text scan for "is this method called?" passes vacuously

**What happened.** TASK-895 needed a guard proving every `WatchlistBundleService`
method has a production caller. The first version grepped the tree for `.create(`,
`.rename(`, `.delete(` and friends. It passed — against code where the methods were
still unwired.

`.create(` matches `completions.create(` in the OCR backends. `.rename(` matches
`os.rename(`. The guard was measuring the existence of unrelated method names on
unrelated objects. It was caught only by mutation: unwiring a call and watching the
guard stay green.

Rewritten as an AST walk that resolves the receiver before counting a call.

**What to do.** A guard that asserts "X is used" must resolve *what* X is, not match its
name. Bare-name greps are fine for finding candidates and useless as evidence, because
method names are not unique across a codebase — the more generic the verb (`create`,
`delete`, `run`, `send`), the more certainly the grep is counting something else.

And whatever the guard is: **mutate the thing it claims to protect and watch it fail.**
This one looked authoritative, ran green, and proved nothing.

---

## Catalog registration does not prove a TTS package is text-ready

**TASK-13202, 2026-08-10.** The pinned audio.cpp `release-0.5.1` server
successfully registered a standalone PocketTTS GGUF, and its catalog advertised
a TTS task. That looked sufficient to classify the package as ready for the
first no-reference sample. Real synthesis disproved the assumption: PocketTTS
requested a separate voice embedding (`alba.safetensors`) that the standalone
GGUF does not contain. Registration and task metadata were both true, but the
promised user journey was still impossible.

**What to do.** Before classifying an exact local-model recipe as text-ready,
run a model-specific complete-WAV request against the pinned real server and
confirm any required voice/reference material is present. Catalog registration
proves that the server accepted the model; it does not prove that text alone is
a complete synthesis input.

---

## Full component coverage, zero feature coverage

**TASK-1210, 2026-07-27.** Watchlists never checked on a schedule — not at the
wrong interval, never at all. Every component involved was tested and green:

- `test_watchlist_projection.py` — rows become `ScheduledTask`s with a `next_run_at`
- `test_watchlist_check_handler.py` — the handler checks feeds and records results
- `test_scheduler_loop.py` — the loop dispatches due tasks to their handler
- `test_config_flags.py` — asserted the flags' defaults, and **asserted the broken
  values**, pinning the bug in place

Each component was correct. Nothing tested them *joined*, and the join was where
the feature lived: `app.py` only registered the `watchlist_job` handler when a
flag was true, and that flag shipped false, so the loop logged "no handler
registered for task type" and moved on. Silently, forever.

The config test is the sharpest part. It was not absent — it was present,
passing, and encoding the defect as the expected value. A test that asserts
current behaviour without asking whether that behaviour is *right* converts a
bug into a requirement.

**What to do.** For any feature that spans components, own one test that drives
the real path end to end — here, a real `Subscriptions_DB` row through the real
projection, the real queue and a real `SchedulerLoop`, asserting the result lands
back in the database. Component tests tell you the pieces work; only that test
tells you the feature does.

And when a test asserts a configuration default, make it state *why* that value
is right, not just what it is. `assert enabled is False` is unfalsifiable
documentation of whatever was there when it was written.

## A green suite says nothing about installs that are not yours

**The trap.** The suite runs where every optional extra is already installed. It
therefore cannot see a dependency that is *declared* optional but has become
*mandatory to boot* — the one environment it never tests is the plain install.

**What happened.** 2026-07-27: the app died on start with
`RuntimeError: Unable to resolve default chat screen`. `aiohttp` is optional —
at the time declared only in the `[websearch]`/`[all-tools]` extras (task-1262
has since given image generation its own `[image_generation]` extra), and
registered `"aiohttp": False` in `Utils/optional_deps.py` — but the
`/generate-image` console feature had quietly wired it onto the **default**
screen's import chain:

```
UI/Screens/chat_screen.py
  -> Chat/console_generate_image.py        (ImageGenerationService)
    -> Media_Creation/image_generation_service.py
      -> Media_Creation/swarmui_client.py  -> import aiohttp   (module scope)
```

Nothing was red. No test asserted that the default route resolves *without* the
extras, so the suite was structurally blind to a total boot failure.

Two multipliers made it worse:

- **The masking cost more time than the bug.** `ScreenRoute.load_screen_class()`
  catches `ImportError` and returns `None`, by design, so one broken optional
  screen cannot break navigation. For the *default* screen that turned a precise
  `ModuleNotFoundError: No module named 'aiohttp'` into a message naming neither
  the module nor the file that imported it.
- **The obvious suspect was innocent.** The only dirty file in the tree was a
  `.tcss` whose diff was a regenerated timestamp comment. Reproducing first and
  reading the traceback cost one command; guessing from `git status` would have
  cost the session.

**What to do.** When a feature adds an import to a screen module, check whether
the new chain reaches an optional dependency — the import that breaks boot is
rarely the one you wrote, it is three hops down. Guard boot-critical routes with
a test that simulates absence, and run it in a **subprocess**: `sys.modules` is
process-global, so an unrelated earlier test that imported the package gives a
false pass.

```python
class _BlockAiohttp:                       # meta-path finder, installed first
    def find_spec(self, name, path=None, target=None):
        if name == "aiohttp" or name.startswith("aiohttp."):
            raise ImportError("simulated missing aiohttp")
        return None
```

See `Tests/Utils/test_optional_import_deferral.py` (the aiohttp section) and
`Tests/UI/test_screen_navigation.py` (`screen_load_error`). And when a resolver
degrades a failure to `None` on purpose, give callers for whom it is *fatal* a
way to ask why — a graceful contract should not also be a silent one.

---

## Measure a dead-code graph from both ends

**TASK-1211, 2026-07-28.** The audit that scoped this retirement measured the island
by walking *outward* from `BriefingGenerator` — who imports it, who imports them —
and arrived at ~5,100 LOC across 11 files.

Walking the other direction, *down* from the scheduler that was about to be
deleted, found the chain kept going:

```
textual_scheduler_worker  →  sole importer of Event_Handlers/subscription_events.py
                          →  sole importer of subscription_ingest_worker.py
                          →  sole caller of Subscriptions/content_processor.py
```

The real island was 8,148 lines across 13 files, plus a fourth module left
deliberately in place. Deleting only what the outward walk found would have
orphaned two files silently — the exact state that made this island expensive to
diagnose in the first place: dead, but with importers a grep can point at.

**Why one direction is not enough.** The outward walk answers "what does this dead
thing depend on?" The downward walk answers "what depended on it and is about to
become dead?" A retirement needs both: the first bounds what you may delete, the
second bounds what your deletion *creates*.

**What to do.** Before deleting a module, list its importers *and* list what it
uniquely imports. Anything it is the sole importer of joins the removal set, and
you recurse. Then re-run the runtime import trace afterwards — if a module you
kept is still in `sys.modules` with no caller, you have made a new orphan and
should either wire it, delete it, or file it. Filing is acceptable; silence is
not.

Corroboration is worth seeking: TASK-813's notes had already reached the same
conclusion about `subscription_events` from the other direction months earlier
(`handle_add_subscription` has zero dispatchers). A prior investigation's notes
are cheaper than re-deriving the graph.

## A missing extra fakes a code regression — check the env before blaming the code

**The trap.** The mirror image of the entry above. There, everything was installed
and the suite went blind. Here, an optional extra is *absent* and a test fails with a
message describing a defect that does not exist. The failure text names production
behaviour, so it reads as a regression, and you go fix code that was never broken.

**What happened.** 2026-07-28, task-1261. `test_nltk_download_false_is_not_logged_as_success`
was failing on dev with "no WARNING/ERROR mentioning punkt was logged". That is
precisely what a deleted `logger.warning` looks like. It was filed as one — *"the
warning was lost in a refactor"* — with `git log -L` even producing a plausible
culprit commit that had genuinely rewritten that function, and an orphaned
over-indented comment left behind as the apparent fingerprint.

All of it was wrong. `nltk` is an optional extra, and it was not installed. The test
sets `NLTK_AVAILABLE = True` to simulate presence, but `_ensure_nltk()` still runs a
real `import nltk`, so it returned early and never reached the warning. Installing
`nltk` turned the test green with no code change at all. The confirming probe written
to "verify" the diagnosis had been run in the same interpreter, so it hit the same
early return and agreed — a second wrong answer from the same cause reads as
corroboration.

**What to do.** When a test asserts that a log/branch/side effect is missing, check
whether the code path can even be *reached* in your environment before concluding the
behaviour was removed. One command settles it:

```bash
python -c "import importlib.util as u; print(u.find_spec('nltk') is not None)"
```

And a test that forces an availability flag must also stub the import that flag
stands for — otherwise it silently depends on which extras you installed:

```python
monkeypatch.setattr(Chunk_Lib, "NLTK_AVAILABLE", True)   # not sufficient alone
monkeypatch.setitem(sys.modules, "nltk", fake_nltk)      # _ensure_nltk() still imports
```

Corollary: a probe re-run in the same broken environment is not independent evidence.
Vary the thing you suspect — here, install the package — and see if the symptom moves.

---

## A property test with no deadline override is load-sensitive

**TASK-1260, 2026-07-28.** `test_safe_paths_always_validate` failed once inside a
three-directory run, passed alone, passed on re-run, and passed on a clean
baseline with the identical command. It is a Hypothesis `@given` property that
creates a `TemporaryDirectory` and up to four directories per example (the
strategy yields 1-5 components; the loop walks `components[:-1]`, the last being
the file), with no `settings(...)` override and no Hypothesis profile in
`Tests/conftest.py` — so it runs under the default **200 ms per-example
deadline**. On a machine with 10+ concurrent pytest processes, filesystem work
crosses that.

**The cost is in attribution, not in the failure.** Establishing that it was not
a regression took five runs across two worktrees: the identical command on a
clean pre-change baseline, `Tests/Utils/` with and without a newly added file in
that same directory, the test alone, and a re-run to show intermittency. The
failure was indistinguishable from a real regression at the moment it appeared,
and it appeared while unrelated work was in flight.

**What to do.** When a failure appears in a run that mixes suites: before
anything else, check whether the test is a Hypothesis property with no deadline
override. If it is, the load hypothesis is cheap to confirm — run it alone, then
re-run the mixed command. Do not skip the clean-baseline run, though: "it's
probably a flake" is the same shape of reasoning as "it's probably unrelated",
and this repo has punished both.

The durable fix is a Hypothesis profile registered once in `Tests/conftest.py`,
not a per-file patch — other property files carry the same exposure.

---

## A filter with no admitted callers is an off switch

**TASK-1240, 2026-07-28.** The persistent app log wrote zero bytes. The handler
was attached, the path resolved, the directory existed, the level was INFO.

`PersistentDiagnosticFilter` admits a record only if it carries a marker set
exclusively by `log_persistent_metadata()`. That function has **zero production
call sites**. Every ordinary `logger.info(...)` is rejected, so the sink is
correctly enforcing a boundary that nothing was ever migrated to cross.

The privacy work that introduced it is sound and has its own ADR. The gap is
between decision and implementation: ADR-029 requires logs be metadata-only
*with respect to user and model content*, and the design's stated goal was to
"keep persistent diagnostics **useful** without retaining private payload
values". Admitting nothing satisfies the letter of the exclusion list and defeats
the goal.

**What to do.** When a sink produces nothing, check the **admission predicate
before the plumbing**. Handler attached, path correct and level correct are all
consistent with a filter that rejects everything. Then ask the question that
distinguishes the two failures: *how many callers does the admitted path have?*
Zero is the tell.

And when the answer implicates a deliberate security boundary, record the gap
and hand the decision to that work's owner. Loosening a privacy filter to make
your own diagnostics visible is not a fix you get to make alone.

This is the fourth instance of one shape in a single session: a closed import
cycle, a flag gating the only executor, a prompt surface with no consumer, and
now a log sink with no admitted caller. Each was built, wired, and given nothing
to carry — and each read as live to a grep.

---

## Run source-inspection tests on a supported interpreter before changing them

**TASK-15706, 2026-08-13.** A repository-wide collection under Python 3.14
failed in `test_profile_store_lock.py` because the test compared integer source
lines with `None`. The production code and test were unchanged from `dev`;
inspection showed Python 3.14 emitted 12 `dis.findlinestarts()` entries whose
line is `None`. The same repository collected all 42,613 tests under the
installed, project-supported Python 3.12 interpreter.

**What to do.** When a test derives source locations from bytecode or
introspection APIs, check the interpreter version and inspect the raw API output
before patching application code. Re-run collection under a supported project
interpreter to distinguish an interpreter-assumption failure from a product
regression, and record both results.

---

## A suite that no gate runs can rot invisibly for days

**TASK-1310, 2026-07-28.** `Tests/UI/test_settings_configuration_hub.py` carried 22
failures at dev tip — byte-identical at base and branch, so none were caused by the
branch that finally surfaced them (TASK-1234's review, the first time this hub ran in
that review cycle). The suite was last known green before #1050; nothing narrower
caught the drift for days across two deliberate, well-reasoned refactors
(`d15882398` "own provider selection by lifetime" and `1df0c4cb4` "reconcile privacy
lifecycle eval and packaging hardening") that each correctly updated every production
call site and left this one 253-test file behind.

Both refactors were textbook-good: each shipped its own new, correct test coverage
(`Tests/Provider/test_provider_model_resolution.py`; the batched-save-adapter test in
this same file) and left zero live bugs — `grep` across all of `tldw_chatbook/` for the
removed symbols (`chat_api_provider_value`, `save_setting_to_cli_config` imported into
`settings_screen`) came back empty on both counts. The damage was entirely to *this*
suite's ability to say so: 22 tests calling a removed signature/attribute is worse than
0 tests, because a red suite nobody gates reports exactly as much confidence as a suite
that does not exist, while still costing the CI minutes of everyone who happens to run
it directly.

**What to do.** A suite this large (253 tests, a whole product surface) needs a home in
routine verification, not opportunistic discovery via someone else's PR review. The
Settings/Console-area verification gate must include
`Tests/UI/test_settings_configuration_hub.py` going forward — not because it is
special, but because "carries the hub's tests" is exactly the kind of suite that rots
silently when nothing runs it: too large to eyeball, too domain-specific for a generic
CI matrix to catch by accident, and it will not fail loudly for anyone except the next
person who happens to touch that screen.

---

## A hung test under timeout_method="thread" kills the whole run — and a hang can be an optional-dep condition

**TASK-1466, 2026-07-30.** The full-suite baseline run for the test-suite audit died
at ~3% progress, twice. `test_pyaudio_recording_flow` stops its recording loop from
inside the chunk callback, but the callback is gated behind webrtcvad speech
detection and the test's synthetic buffer is silence: with `webrtcvad` installed
(it is, locally and in CI), the callback never fires and the loop never exits.
After the 300s timeout, pytest-timeout's `thread` method — the repo's configured
method, and the only one that works for threaded/async tests — cannot cancel the
test, so it dumps stacks and **terminates the entire pytest process**. One hung
test cost the whole run, every run, on every webrtcvad machine; on machines
without the extra the callback fires per chunk and the test is green, which is why
it survived review. Its sibling `test_sounddevice_recording_flow` failed on clean
dev for the same root cause (a 4-sample chunk is smaller than one 20ms VAD frame,
so the VAD loop body never executes) — masked because serial runs died at the hang
before reaching it.

**What to do.** A test that stops its own loop from inside a callback must also
bound the loop at the *source* (here: the mocked `stream.read` flips the stop flag
after N reads) so no gating change can make it unbounded. When a test's behavior
depends on an optional dependency, run it in both installed and absent
configurations before trusting green. And treat "the run died at N%" as possibly
ONE test: the timeout stack dump names it — **except when the hang is an awaited
asyncio primitive; see the next entry.**

---

## The timeout stack dump does NOT name the hung test when the hang is an awaited Event

**TASK-3316 / TASK-14912, 2026-08-10.**
`test_file_notes_collections_source_transition_blocks_mutation_through_recompose`
drove `_select_library_rail_row` as a fire-and-forget `asyncio.create_task` and
then `await`ed an `asyncio.Event` only that coroutine could set. Its stub returned
`None` — correct when written (`eb036a6a1`) — until PR #1439 retyped
`_flush_library_note_save` to return `NoteFlushOutcome`; the caller then died one
line in on `AttributeError: 'NoneType' object has no attribute 'kind'`. **Nobody
retrieves a `create_task` result, so the exception was swallowed**, the signal
became unreachable, and `await event.wait()` blocked forever.

Two things this cost, beyond the one test:

1. **The stack dump was useless.** TASK-1466's advice above does not hold here: a
   coroutine suspended at an `await` has no frames on any thread stack, so
   pytest-timeout printed only `MainThread` idle in `selectors.select` and never
   named the test. Diagnosis required inspecting the *task object*
   (`task.done()` / `task.exception()`), not the dump. Reproduced deliberately
   while writing the bound: 25s timeout, stacks dumped, process terminated,
   **zero** tests reported in the summary line.
2. **Every test after it in the file silently never ran**, so the file's pass
   count was a lie for as long as the hang existed. Repairing that one test
   revealed three further failures the hang had been hiding.

**What to do.** Never `await <event>.wait()` on a signal only background work can
set. Route it through `Tests/UI/background_signals.py`:
`wait_for_background_signal(event, task, what=...)` when the test owns the task
(it re-raises what the task swallowed), or `wait_for_signal(event, what=...)` when
the product owns the work (timeout-only, but a named failure in seconds instead of
a dead process). `Tests/UI/test_background_signal_bounds.py` enforces this by AST
over the whole directory — grep cannot, because it cannot tell an unbounded
`await ev.wait()` from one already inside `asyncio.wait_for`, nor an
`asyncio.Event` from a Textual `Worker`/retained-operation handle whose `.wait()`
re-raises and therefore cannot strand.

Two corollaries worth carrying:

* **A file that has ever contained a hang has an UNKNOWN pass count** until it is
  re-run whole. A previously recorded count for such a file is not evidence.
* **In practice the re-raise branch fires less often than you would expect.** Of
  four sites broken deliberately with the stale-stub shape, only one propagated;
  the other three product paths caught the `AttributeError` internally, logged it,
  and returned early — so the bound reported "finished without signalling" rather
  than the exception. That is still a 1-3s named failure instead of a dead run,
  but it means the helper's early-return branch is not a corner case: do not drop
  it in favour of "just re-raise".

---

## `--deselect` with a wrong nodeid is silently ignored

**2026-07-30, same audit.** A full baseline attempt was relaunched with
`--deselect "Tests/Audio/test_recording_service.py::test_pyaudio_recording_flow"` —
missing the `TestAudioRecordingIntegration::` class qualifier. pytest does not
error, does not warn, and does not report `1 deselected`; the run simply hung on
the very test the flag was meant to exclude, and the mistake was only visible ~15
minutes later when progress stalled at the same file.

**What to do.** After launching a run with `--deselect`, confirm the header line
says `N deselected` before walking away. Copy nodeids from pytest's own output
(`--collect-only -q` or a failure line), never reconstruct them by hand — class
nesting is invisible in the source-file mental model.

---

## Removing a per-test gc.collect() can unmask cross-test coupling — with a rotating victim

**TASK-1468, 2026-07-30.** task-1454 replaced the double `gc.collect()` after every
test with a periodic collect (every 25). A 10-test batch then started failing ONE
UI test — a *different* one on consecutive identical runs (first the Skills trust
panel test, then a Library git-notes test). Alone, each test passed. With
`TLDW_TEST_GC_EVERY=1` the batch passed 10/10; on pre-change dev it passed 10/10.

The mechanism: a Textual `App` is a reference cycle that refcounting never frees —
only the cycle collector does. Per-test collection had been silently guaranteeing
each app-mounting test a garbage-free predecessor; without it, the previous app's
remains (timers, context vars, screen state) linger into the next app's lifetime,
and which test breaks depends on heap state at the time. A rotating victim is the
tell that you are looking at ambient-state interference, not a broken test.

**What to do.** When narrowing global per-test cleanup, ask what CLASS of object it
was silently reclaiming, and scope the cleanup to the tests that produce that class
(here: per-test collection in app-mounting dirs, periodic elsewhere) rather than
tuning frequency — no interval above 1 protects adjacent producers. And triage any
"deterministic" batch failure by rerunning the identical batch twice before
believing determinism: the rotating victim only shows on the second run.

---

## Measure the identical-run noise floor before reading an A/B outcome diff

**TASK-1459, 2026-07-30.** The CSS parse-cache canary (full Tests/UI, cache on
vs off) showed 12 pass->fail flips and zero recoveries — directional, exactly
what cross-instance cache corruption would look like, and the spike's gate said
any diff means fall back or no-go. Before ruling, a control pair of two
IDENTICAL cache-off runs was diffed: **28 regressed + 4 recovered against each
other** — the machine's own flip rate was ~3x the "cache effect", the flagged
tests A/B'd identically in isolation with the cache on and off, and the
cache-on run sat between the two cache-off runs on failures and wall time
(which grew from 13:37 to 23:12 across the evening as concurrent sessions
loaded the machine).

**What to do.** An "outcome diff must be empty" gate is unfalsifiable on a
machine whose identical-run diff is nonzero — and on shared dev machines it
usually is. Before attributing an A/B diff to the change, run the A/A control
and use *its* magnitude as the acceptance bound; attribute individual flips
only via isolation reruns under both configurations. Directionality alone
(12-vs-0) is not attribution: later runs on a loading machine flip
asymmetrically toward failure.

---

## The A/A control also tells you WHICH metric to report, not just how big a diff must be

**TASK-22215, 2026-08-26.** Measuring input latency during the first 5 s after mount
(before/after the boot-worker stagger, simulated post-upgrade profile, n=8 per arm,
interleaved in both orders), the headline that jumped out was the median keypress:
**before 90.5-92.6 ms, after 71.9-72.8 ms in three of four runs** — a tidy -22%. The A/A
control (identical tree, two labels, same shape) killed it: its four runs measured 72.7,
89.5, 77.8, 72.0 ms. The metric is **bimodal**, and both modes appear with the code held
constant, so any single pair of runs can "show" a 20 ms median win or none at all.

What survived on the same data was the metric whose RANGES separated: worst single
keypress **625-876 ms before vs 395-467 ms after**, with the A/A control's 426-476 ms
sitting inside the after range, and mean 111-124 vs 99-109 ms. p95, loop-heartbeat excess
and time-to-ready were washes, and the warm-boot shape was a wash on everything.

**What to do.** Run the A/A control with the SAME repeat count as the arms, and read its
per-run spread, not only its aggregate: a metric whose A/A runs are multi-modal cannot
carry a claim at that n no matter how clean the A/B looks, while a metric whose A/A range
is narrow and disjoint from one arm can. Report the ones that separate, say "wash" for
the rest, and never quote a median that the control also produced.

---

## A crash mid-transfer proves nothing durable if the checkpoint is never written mid-transfer

**TASK-595 Task 10, 2026-07-31.** Asked to write a real-subprocess, SIGKILL-based test
proving "a valid sidecar survives a mid-fetch crash and a fresh provision resumes it via
a Range request," the obvious design was: pause a fixture connection mid-download, kill
the child while the socket is blocked, then assert the durable checkpoint resumed. That
design cannot pass — not because of a bug, but because of how the code under test is
*supposed* to work: `_fetch_one_file` only calls `atomic_write_json` on the sidecar AFTER
`stream_fetch` returns successfully for that call; a SIGKILL during the first-ever
transfer of a file leaves no sidecar at all (confirmed directly by an existing test,
`test_provision_cancel_mid_fetch_releases_lease_and_preserves_prior_active`, which
asserts `not sidecar_path.exists()` after an asyncio-cancelled mid-fetch). A kill timed
mid-socket-read can only ever produce an *orphan* (Task 2's GC correctly deletes it), not
a resumable checkpoint — the two are mutually exclusive outcomes of the same kill point.

A durable-but-partial checkpoint (`bytes_done < size, complete: false`) only exists when
one `stream_fetch` call already returned successfully with fewer bytes than the
descriptor declares — which requires either a pre-seeded sidecar (what the existing
in-process tests do, legitimately, to isolate the resume *logic*) or, for a test that
must produce this state via a genuine, un-seeded crash: declare a file's size larger than
what the fixture route currently serves, let that first GET complete normally (a valid,
un-truncated HTTP response, so `stream_fetch` returns without error), and freeze the
*next* phase (pre-verify, via a local `threading.Event` a progress-callback hook blocks
on) before its hash comparison can clear the sidecar entry on mismatch. The kill then
lands after a real checkpoint is durable, not during the socket read that would prevent
one existing at all.

**What to do.** Before designing a crash-recovery test for any resumable-transfer
system, read the exact write-ordering of its checkpoint (grep for where the persisted
state is actually written, not just where progress callbacks fire) and check for an
existing test that already documents the "no sidecar" case — it will save you from
building a scenario the implementation cannot produce. Then time the kill off a
deterministic signal (a callback blocking on a never-set local event) rather than a
sleep or a byte-count race, so the exact crash point is provable, not probabilistic.
Mutation-test the resulting guard afterward: a one-line break in the resume path
(`resume_from = 0` unconditionally) and in the orphan classifier (`return False`
unconditionally) each turned the corresponding assertion red, confirming neither guard
was decorative.

---

## Related

- `lessons-live-verification.md` — why the suite could not see seven of these defects
- `lessons-backlog-hygiene.md` — task IDs, CLI quirks, git plumbing traps

---

## The shared UI harness never loads the app stylesheet — geometry conclusions under it are void (2026-07-30)

**Incident.** The V2 live gate failed its composer-overflow item AFTER the defect had
been "fixed" twice, each fix RED-first, mutation-checked, 500k-trial fuzzed, and
approved through two full review rounds. The real cause was one CSS rule —
`#console-composer-expanded { height: 1 }` — cropping the grown 4-row draft to a single
painted line. No test could see it: `Tests/UI` harnesses build `ConsoleHarness`, a bare
`App[None]` that pushes `ChatScreen(app_instance)` directly. The `TldwCli` instance is
only a service container there; the App that runs owns the stylesheet, and it has none.
Every rule in `tldw_cli_modular.tcss` silently does not apply under these harnesses.
Both instruments used to verify the fixes — `widget.render_line(...)` (the widget's own
paint, blind to a parent's crop) and `widget.region` (layout placement, not clipped
paint) — were also individually unable to see cropping. A 30-second tmux run of the
real app reproduced the user's report on the first try.

**What to do.** Any assertion about on-screen geometry — heights, clipping, whether a
row is visible — must run under a harness whose `CSS_PATH` is the real bundle (see
`_CssTrueConsoleHarness` in `Tests/UI/test_console_composer_overflow.py`), or against
the real app in tmux. `render_line`/`region` alone prove what a widget WOULD paint,
never what the screen shows; the composited screen is the only authority (third
recorded instance of this lesson class). When a live report contradicts a green suite,
suspect the harness before the reporter.

**Fourth instance (2026-08-07, task-2859 item 10, padding not clipping this time).** A
`.library-rag-result-snippet { padding: 0 1; }` bundle rule (fixing a snippet sitting
flush against its card border) tested green with `snippet.region.x ==
title_row.region.x` under `DestinationHarness` (`Tests/UI/test_library_content_hub.py`)
— because `region` never reflects padding at all (only layout position/size;
`content_region = region.shrink(styles.gutter)` is the one that does), AND because
`DestinationHarness` is a bare `App` with no `CSS_PATH`, so `title_row.styles.padding`
itself came back `Spacing(0,0,0,0)` regardless of what the .tcss said. Direct proof:
`screen.app.css_path == []` and `type(screen.app).CSS_PATH is None` under this harness,
vs. the real string when `TldwCli` is imported and inspected directly outside any test.
Moving the exact same assertion to `LibraryHarness` (`Tests/UI/test_library_shell.py`,
which sets `CSS_PATH` to the real bundle — "Mount a single LibraryScreen with the real
app stylesheet" is literally its docstring) reproduced the missing-padding RED
correctly and went GREEN once the CSS rule existed. Two independent traps stacked here,
either one alone would have hidden the bug: use `content_region`, not `region`, for
padding; and know which harness in a file actually loads CSS before trusting geometry
from it — `Tests/UI/test_library_content_hub.py` uses `DestinationHarness` (no CSS) for
most of its tests, `Tests/UI/test_library_shell.py` uses `LibraryHarness` (real CSS) —
same directory, same screen under test, opposite answer to "does this rule apply".

**Fifth instance (2026-08-08, task-3200 review round 1, cascade PRIORITY not just
missing rules this time).** `MainNavigationBar.DEFAULT_CSS` ghosts a straddling nav
button by setting `color`/`background`/`opacity`/`text-opacity` all to `$background`
`!important`, intending a pixel-exact invisible button once it's also `disabled`. The
bare-`App`-no-`CSS_PATH` test (`test_nav_strip_never_renders_a_partial_destination_
label`) confirmed exact-match compositor colors and stayed green — but live tmux
showed the ghosted fragment as a faintly-but-genuinely readable `rgb(43-62,43-62,
43-62)` against `rgb(16,16,16)`, not a match. Root cause was NOT a missing bundle
rule (the earlier four instances) but a PRIORITY one: `tldw_chatbook/css/components/
_buttons.tcss`'s app-wide `Button:disabled { opacity: 50%; }`, loaded via `App.
CSS_PATH`, outranks ANY widget `DEFAULT_CSS` rule as a TIER, regardless of
`!important` on the `DEFAULT_CSS` side — confirmed by direct introspection
(`button.styles.opacity` read `0.5` under the real `TldwCli` app + `HomeHarness`
despite the widget's own `opacity: 100% !important`, but read `1.0` under the bare
test harness where no `CSS_PATH` rule existed to compete). This codebase had already
hit and fixed the identical defect once before (`Tests/UI/test_mcp_inspector.py`'s
`test_disabled_action_buttons_stay_legible_with_bundled_css`, for the MCP inspector's
action buttons) — that precedent was not consulted before initially trying
`!important` in `DEFAULT_CSS`, which was the wrong tier to fight from. The fix that
actually works: add the override to a `CSS_PATH`-bundled source file (`_navigation.
tcss` here), in the SAME tier as the rule being overridden, where ordinary
specificity resolves it without needing `!important` at all.

**Sixth instance (TASK-31551 task 13, 2026-09-04, whole-screen collapse this time, not
a single rule).** The Meetings screen's task-11 brief composed its workbench Horizontal
with `classes="ds-panel destination-workbench"` — the identical two-class combination
nine sibling screens (Artifacts, Personas, Watchlists, Workflows, MCP, ACP, Skills,
Settings, Evals) already use, each paired with an `#<id>-workbench { height: 1fr; ... }`
ID-scoped override in `css/components/_agentic_terminal.tcss` that beats `.ds-panel`'s
own `height: auto; min-height: 3`. Task 11's brief never asked for that override for
`#meetings-workbench`, and every one of its mounted `Tests/UI/test_meetings_screen.py`
pilots (`_build_test_app`, no real `CSS_PATH`) stayed green through Tasks 8-11 and
review, including tests that `pilot.click("#meetings-start")` at a hard-coded
coordinate. Live-driving the real app under tmux for task 13 showed why: with the real
bundle loaded, `.ds-panel`'s `height: auto` won outright (no priority inversion needed,
just an entirely absent override), collapsing the workbench AND both its panes to
zero visible rows — no Select, Button, Static, or RichLog painted anywhere on screen,
confirmed with `tmux capture-pane` showing two adjacent empty bordered boxes. The fix
was the same one-line pattern as the other nine screens: add `#meetings-workbench` to
the existing ID list. This is not a single missing/mis-tiered rule (instances 1-5) but
the SAME root cause at the scale of an entire screen: a harness with no `CSS_PATH`
cannot fail a geometry assertion that a required per-screen CSS override was never
written, because it never applied the class-level rule that override exists to beat in
the first place.

**What to do (all six instances).** Never trust a bare-`App`-no-`CSS_PATH` test's
color/opacity or geometry as proof of live behavior — it can miss a rule entirely
(instances 1-4, 6) or miss a PRIORITY inversion where `CSS_PATH` beats `DEFAULT_CSS`
regardless of `!important` (instance 5). Before shipping any "hide via CSS" trick
(`color == background`, opacity-to-zero, etc.), grep for prior art
(`Button:disabled`, `:disabled` opacity overrides already exist for MCP inspector)
and verify with `button.styles.opacity`/`get_visual_style()` under a REAL-bundle
harness or live tmux, not just a bare widget construction. When a new screen reuses a
shared "workbench" class combination (`ds-panel destination-workbench` or similar),
grep for the sibling screens' matching ID-scoped override in the same pass that adds
the class names — the class alone renders a screen with, at best, three rows of
content no matter how many widgets it composes.

---

## A zero-latency fake makes loop-starvation bugs invisible (2026-07-30)

**Incident.** Live dictation never emitted a final during capture — voice commands
(classified on finals) were completely dead in the field — while 300+ dictation tests
stayed green. `_processing_loop` transcribed each 0.5 s audio window synchronously on
the same thread that runs the silence-finalize check; real transcription took 4-5 s per
window (proven with `sys._current_frames()` stack dumps against a live microphone), so
the check starved indefinitely. Every test fake transcribed in ~0 ms, which makes the
serial design behave identically to a concurrent one. The probe ladder that isolated it,
in increasing depth: call the transcriber+classifier directly (chain worked) → measure
`captured_bytes` during silence (VAD worked) → shim the finalize method (never called
despite an 8.6 s silence age at a 2.0 s threshold) → per-tick thread stack dumps (loop
permanently inside the transcriber). Fix: segment-at-silence architecture; the RED test
that pins it gives the fake a CONTROLLABLE latency (2× the threshold) — with a fast fake
it cannot fail.

**What to do.** Any fake standing in for an operation whose real latency exceeds the
loop/timer cadence it shares a thread with must be able to sleep. Test both fast and
slow. And when a threaded pipeline works in tests but not live, dump the worker's stack
(`sys._current_frames()[thread.ident]`) once a second before theorizing — it answered in
one run what three cheaper probes could only narrow. Bonus rig: macOS `say` through the
speakers + the real microphone is a full live STT test harness needing no human.

---

## In a render-from-state UI, the in-place updater must own EVERY conditional (2026-08-04)

**What happened.** Four separate times in the Library ingest arc (tasks 2100, 2130,
2140, 2230), a canvas element was rendered by a `compose()`-time conditional while the
hot paths deliberately skip recompose (job ticks and text-input edits must preserve
focus, cursor position, and scroll). Each time the element was correct on first render
and wrong forever after:

- "Recent ingests" expanded into an empty unlabeled shell after a clear.
- The commit-summary line rendered for PDF selections and **never** for plain text —
  a PDF adds an options panel, which forces the structural recompose that happens to
  mount the line; a text-only pre-flight applies through the non-structural path,
  which mounts nothing. It also went stale after Clear ("0 will import · 1 will match"
  above an empty field).
- The invalid-option field marker was applied at compose time only, so it never
  toggled on the edit path it existed to serve — the field stayed marked after
  becoming valid and never got marked after becoming invalid, while the gate line
  instructed the user to "fix the highlighted options".

**Why tests kept missing it.** The harness passes when it *re-queries* after the
update, because a fresh query returns whatever was composed most recently. The failure
only appears when you assert that the widget you held is the widget still mounted.

**What to do.** Two rules, both cheap:

1. Anything the in-place updater does not explicitly own must be **always mounted and
   `display`-managed**, never conditionally composed. If a canvas-level element can
   appear and disappear, the updater sets its content *and* its visibility.
2. Pin it with **object identity**, not a re-query:

```python
before = screen.query_one("#library-ingest-start", Button)
...trigger the hot path...
assert screen.query_one("#library-ingest-start", Button) is before   # no recompose
```

A re-query test agrees with the bug; `is` does not.

---

## A new distinction must be learned by every surface that aggregates the old one (2026-08-04)

**What happened.** TASK-2231 introduced "matched" as an outcome distinct from "done"
(a dedup match is not a fresh import). The row got its own glyph and word, and the
change looked complete. Review found two surfaces still folding matched into done:
the per-batch group header, and the top-level queue tally — which produced two
contradictory summaries on the same screen ("2 done" directly above "1 done ·
1 matched"). The completion toast was a third surface, caught only because it was
grepped for by hand. TASK-2220 had the same shape one PR earlier: adding a `SKIPPED`
job state updated the row and the tally but missed `queue_show_clear_finished`, which
still tested `state in (DONE, FAILED)` — so a queue holding only skipped rows could
not be cleared at all.

**What to do.** When you add a state, an outcome, or any new bucket, grep for **every
predicate that enumerates the old set** and every surface that counts it, then list
them before writing code. In this feature that list was: the row builder, the group
header, the queue tally, the completion toast, the "show clear" gate, the "finished"
count, the ledger snapshot, and the durable-history filter — eight places, and the
first attempt updated three.

Related: a fixture that omits the interesting axis hides the bug. My own attempt-marker
test used jobs with no `detected_type`, so it passed while every *typed* row rendered
the marker in the wrong position.

---

## An error fallback that returns a valid-looking value becomes a confident lie (2026-08-04)

**What happened.** `_safe_size(path)` returned `0` on `OSError` — a reasonable "I don't
know" for summing sizes. TASK-2160 then added an empty-file classifier that read
`size == 0` as "this file is empty and will fail", and surfaced it as user-facing
copy: *"1 empty file will fail — notes.txt is 0 B."* For an unreadable or unstatable
file, that sentence is a measurement nobody took, and the file was pulled out of its
type group on the strength of it.

**What to do.** A sentinel is safe for aggregation and unsafe for classification. When
a new consumer needs to *distinguish* failure from a real value, give it a probe that
says so — `_statted_size()` returning `None` on `OSError` — and leave the summing
caller on the old fallback. Before reusing any helper whose docstring says "or `0` on
error", ask what your caller will conclude from that zero.

---

## "the field is set" is not "the resource is ready" — check what the real object does before it's ready (2026-08-04/05)

**What happened.** TASK-2360's bug report said reconnect audio was dropped because
`session.session is None` during the reconnect window. Reading the wiring showed
`session.session` is actually reassigned to the NEW provider session quite early
(`_connect_console_realtime`, before `await provider_session.connect()` even runs) —
so a fix gated only on "is `session.session` set" would have looked correct and
still leaked frames into a session with no live transport yet. The REAL drop
mechanism was one layer deeper: `OpenAIRealtimeSession._enqueue` silently discards
anything sent before `connect()` populates its outbound queue. A test double
(`FakeRealtimeSession.append_audio`) that just appends to a list, with no
before-connect gate, would have made a wiring test pass for the wrong reason —
proving frames "reached the session" when a real session would have swallowed them.

**What to do.** When a bug report names a field as the cause ("X is None during the
bad window"), verify what that field actually holds moment-to-moment, not just
whether it is set — a reassigned-but-not-yet-live reference passes an `is not None`
check while still being unsafe to use. And before trusting a wiring test built on a
fake, ask whether the fake reproduces the real object's OWN pre-ready guard, or is
simply more permissive than production in exactly the window under test.

---

## A test failure dismissed as "pre-existing noise" can be a shipped crash

**TASK-2610, 2026-08-06.** `test_production_settings_actions_cross_the_pushed_screen_boundary`
failed with `DuplicateIds: lab-speech-row-playground` for weeks. Across many tasks — in
multiple programs, by multiple sessions — it was checked once ("fails identically on the
base commit"), labeled "pre-existing, unrelated," and waved through. Every one of those
dismissals was locally correct and collectively wrong: the failure was a 100%-reproducible
user-facing crash — navigating to Lab ▸ Speech took the whole app down, making the Speech
Lab (playground, voice profiles, audiobooks, voice cloning) unreachable. It was found only
when a live-verification pass tried to drive that navigation for an unrelated feature.

**Mechanism worth knowing on its own:** Textual's `MessagePump._get_dispatch_methods`
walks the MRO and invokes EVERY class's `on_mount` for a single Mount event. A subclass
handler that calls `super().on_mount()` therefore runs the parent handler TWICE.
`STTSScreen.on_mount` did exactly that over `LabFrameScreen.on_mount` — which mounts the
rail rows — so the second run collided on the row ids. Sibling screens without their own
`on_mount` never crashed, which is why the bug looked screen-specific. If a parent
`on_mount` does real work, `super().on_mount()` in a child is a crash, not a courtesy.

**What to do.** "Fails identically on base" proves you didn't cause it — it does not prove
it's noise. Before re-dismissing a persistently failing test, spend the five minutes to
classify WHAT the failure would mean for a user if the tested path were driven live; a
`DuplicateIds`/exception-shaped failure in a screen-mount test is an app crash until shown
otherwise. Budget one live drive of the affected surface the FIRST time a failure gets the
"pre-existing" label — that is when it is cheapest, and every later dismissal inherits the
first one's diligence or its negligence.

---

## A green result is not evidence until you have confirmed it could have gone red (2026-08-06)

**What happened, twice, on the same feature (PR-T3 fix rounds A and B).**

**Instance 1.** Fix round A added three tests for the Advanced runner's confirm-arm
behavior, each simulating "press Run, then press it again." Two of the three showed a
false PASS on first write — not because the code was right, but because Textual's
`Button._on_click()` ignores a click while the widget still carries the 0.2s
`-active` press-animation class (`textual/widgets/_button.py`). A bare second
`pilot.click()` inside one pump window landed on a still-cooling-down button and was
silently dropped, so the second half of the test never ran at all — the assertion
checked state that the first press had already produced, and the bug under test (does
the SECOND press do the right thing) was invisible. Caught only by comparing against
`_press_run_again` (`Tests/UI/test_mcp_inspector.py`), a helper an earlier test in the
same file already built for exactly this trap (`await pilot.pause(0.3)` before the
second click, with a comment naming the cooldown).

**Instance 2.** Reviewing fix round A, a reviewer who had JUST been told about
Instance 1 built a mutation harness specifically to check whether those three tests
actually pin their fix — reverting the fix and confirming the tests go red. The first
run reported all-green: reverting the fix changed nothing, which would itself have
been a serious finding (the tests pin nothing). The cause was a second, unrelated
mechanism: Python served stale `__pycache__` bytecode instead of the mutated source,
so the mutation never took effect and the "tests" exercised the OLD, already-fixed
code both times. Fixed by overriding `get_code` (forcing recompilation) rather than
trusting the file on disk had been re-read.

**The shared shape.** Two unrelated mechanisms — a UI framework's click-debounce, and
Python's bytecode cache — each silently prevented the code under test from being
exercised at all, while the harness reported success. Neither is specific to this
feature; both recur anywhere a test fires two rapid interactions through real Textual
widgets, or anywhere a mutation/characterization check edits a `.py` file and reruns
pytest without clearing `__pycache__`. And Instance 2 happened to someone who was
*specifically hunting* for exactly this class of false pass, one incident report old —
knowing the trap exists did not protect against a different instance of it. The
mitigation has to be mechanical, not a mental note.

**What to do.**
1. When a test simulates two rapid interactions through a real widget (double-click,
   press-then-confirm, retry), use a helper that waits out any framework-level
   debounce/cooldown before the second interaction, and name the cooldown in the
   helper's docstring so the next person does not have to rediscover it.
2. Before trusting ANY mutation/characterization test result — your own or a
   reviewer's — clear `__pycache__` for the touched modules (or run with `python -B`,
   or override `get_code`) and confirm the specific new/changed assertions go RED
   against the reverted code, not just that "some tests failed somewhere." A run count
   or an exit code is not enough; read which tests failed and why.
3. Treat "the mutation test passed on the first try" as itself slightly suspicious —
   it is the same shape as a guard that cannot fail (see "Mutation-test every guard
   you add," above), and here the false-positive mechanism was in the test harness's
   plumbing, not the guard's logic.

---

## 900+ green tests never exercised the first edit of a seeded row

**TASK-2451, 2026-08-06.** Enriching the seeded 'Default Assistant' character card
(`character_cards` id=1) meant writing a conditional `UPDATE` to that row. A quick
manual prototype — construct a real `CharactersRAGDB` against a temp file, then run
that `UPDATE` — crashed immediately with `sqlite3.DatabaseError: database disk image
is malformed` (`SQLITE_CORRUPT_VTAB`). Nothing about the prototype's content mattered:
even `UPDATE character_cards SET description = 'x' WHERE id = 1` crashed the same way,
on a completely fresh database, through the real constructor. Root cause: row 1's
`INSERT` in `_FULL_SCHEMA_SQL_V4` ran *before* `character_cards_fts` and its
`character_cards_ai` trigger were created later in the same script, so row 1 was never
indexed into the FTS5 shadow tables — on every database this schema had ever produced.
The first `UPDATE` to that row makes `character_cards_au` ask FTS5 to remove index
entries that were never inserted, and FTS5 reports that as disk corruption. This means
`update_character_card(1, ...)` — an ordinary user editing the built-in Default
Assistant via the normal Roleplay editor — already crashed the app, on every existing
install, before this task touched anything. The full `Tests/ChaChaNotesDB/` +
`Tests/DB/` suites (900+ tests, routinely green) never caught it, because no existing
test performs a write against character id=1 as the first write after database
creation — every test that touches character cards either inserts a fresh row first or
edits a different id.

**What to do.**
1. Before trusting a migration's `UPDATE`/`INSERT` against a long-lived seeded row,
   prototype it against a database built the SAME way production builds one (the real
   constructor, not a hand-rolled minimal fixture) and actually run the write — do not
   reason about FTS5/trigger ordering from reading the schema SQL alone.
2. "900+ passing tests" is not evidence a specific write path has ever been exercised.
   Ask what the very FIRST write to a specific row would look like, and whether any
   test performs exactly that — a seeded/default row (id=1, "the default X") is
   disproportionately likely to be read constantly and written to never, in the whole
   existing suite.
3. A `content=`/`content_rowid=` FTS5 table's row must be created strictly after the
   external-content table's own `INSERT` trigger exists, or the row is invisible to the
   index forever while still being readable via plain `SELECT` (FTS5 can satisfy an
   unfiltered `SELECT` straight from the content table, so `SELECT rowid FROM fts_tbl`
   looks fine and gives no warning). `PRAGMA integrity_check` also reports "ok" in this
   state — it does not catch this class of defect. The tell is `SQLITE_CORRUPT_VTAB`
   (not generic `SQLITE_CORRUPT`) the first time a row in that state is deleted or
   updated. Fix forward with `INSERT INTO fts_tbl(fts_tbl) VALUES ('rebuild')`, which is
   safe and idempotent for exactly this "shadow tables drifted from content" situation.

---

## A single red run is not causation — run both arms

**The incident (2026-08-07, Console decomposition wave 4).** I deleted 35 lines of
provably dead code from `on_button_pressed` — a branch whose button id had been
removed by an earlier commit, confirmed dead by a whole-repo sweep and by an
independent reviewer. Immediately afterwards
`test_console_workspace_context_rail.py::test_conversation_status_row_label_and_value_are_separate_visual_runs`
failed. It failed again in isolation. It passed on the commit before the deletion.
Three signals all pointing the same way, and all of them wrong.

The controlled version: three runs with the change and three without.

- **Without** the deletion: 1 passed, 2 failed.
- **With** the deletion: 2 passed, 1 failed.

Same distribution. The test is nondeterministic on its own — it asserts on
`_composited_rows(...)[0]` and the rail's composited rows do not always arrive in
the same order (filed as task-3025, as a possible product nondeterminism rather
than only a flaky test).

Had I stopped at "it passes on the parent commit and fails on mine", I would have
reverted a correct deletion and gone looking for a mechanism that does not exist.
A subagent on the same task had earlier called the same test "a cross-file flake,
investigated and cleared" after re-running it a couple of times — which was the
right conclusion reached by a method that would equally have produced the wrong
one.

**What to do.**
1. Before attributing a failure to your change, run it **N times on both arms**
   (three is usually enough to expose a coin-flip; one is never enough). Restore
   your change with `cp` from a scratch copy or an `Edit`, **never**
   `git checkout --`, which silently discards uncommitted work — that has cost a
   whole test rewrite in this repo before.
2. "Passes on the parent commit" is one sample, not a control. So is "fails in
   isolation" — isolation removes cross-file order dependence, but says nothing
   about nondeterminism inside the test itself.
3. A test that indexes into a rendered/composited collection (`rows[0]`,
   `children[2]`) is a prime candidate: it will fail the moment ordering varies,
   and ordering varies for reasons that have nothing to do with your diff. When
   you find one, ask whether the *product's* ordering is guaranteed before you
   "fix" the test to match whatever it did today.
## A fixed `pilot.pause()` before querying a worker-mounted widget is an ordering landmine (2026-08-05)

**Incident.** TASK-2154.3 added one event-loop turn to the Console left rail's
settling path (a mid-recompose fit-pass defer in
`ConsoleWorkspaceContextTray._fit_height_to_content`). Every targeted suite stayed
green, but at FILE level
`test_console_workspace_many_conversations_keep_lower_status_reachable` failed with
`NoMatches: '#console-new-workspace-conversation'` — and passed standalone, and the
whole file passed on HEAD. Bisecting showed ANY preceding pilot test (not just the
ones the diff touched) tripped it: on a warm event loop the legacy-alias mount chain
(`call_after_refresh` → `run_worker` → `await mount()`) lands one turn later than the
test's single fixed `await pilot.pause()`. A scratch test with a polling wait proved
the button still mounts promptly — pure test-timing fragility, no production
regression — so the fix was one `_wait_for_selector(...)` line in the test, not a
production change.

**What to do.** Never query a control that mounts through an async worker
(`run_worker`/`call_after_refresh` chains, e.g. ChatScreen's out-of-band legacy
aliases) after a fixed pause; poll for it like every other async-mounted widget.
When a test fails only at file level, bisect pairs (predecessor + victim) on your
tree AND on HEAD before touching production code — the pair run on HEAD (green) vs
your tree (red) separated "my change added a turn" from "the test never mounted" in
two 5-second runs, and a generous-timeout scratch replica answered the only question
that mattered: does the widget eventually appear at all?

---

## A keyboard funnel through `Button.press()` dies silently when the button gains a real disabled state

**The trap.** A key handler that "clicks" a button via `Button.press()` inherits
Textual's guard: `press()` returns early when `self.disabled or not self.display`
(Textual 8.x), posting no `Pressed` message and raising nothing. The moment that
button gains a genuine `disabled=True` state, the keyboard path stops reaching the
handler — no error, no test failure unless a test drives the *key*, and any
side-effect the key path performed first (stash, arming flags) is left stranded.

**What happened.** TASK-2154.6 gave the Console Send button a real disabled state
(FR-04). The Enter hotkey in `ChatScreen.on_key` captured the draft into a pending
stash and then routed through `query_one("#console-send-message").press()`. With
Send disabled (blocked/empty draft) the press no-opped: the blocked-attempt
feedback (toast + transcript system row) never fired, the stash stayed pending,
and the *next* Enter was swallowed as a duplicate of the stranded one. Only a
from-source read of `Button.press()` surfaced it; every existing test pressed the
button directly, so nothing else would have caught it.

**What to do.** Before adding `disabled=True` to any button, grep for
`.press()` and `pilot.click` on its id across both production and test code —
those callers silently change behavior. A keyboard funnel that must keep working
while the button is disabled needs an explicit branch that dispatches the same
handler directly (the Console voice-send path's synthesized
`handle_...(Button.Pressed(button))` pattern), plus a test that drives the *key*
in the disabled state.

---

## A keyword `-k` suite deselects behavior-affected tests whose names lack the keywords (2026-08-05)

**Incident.** TASK-2154.7 changed the Console provider-recovery resolution
(which blocker wins: provider vs model). The task's prescribed verification was
`pytest Tests/ -k "onboarding or setup_card or setup_modal or readiness"` —
it reported 3 failures. The full run of every file that calls the changed
helpers reported **6**: `test_console_empty_transcript_choose_model_opens_settings`,
`test_console_blocked_inspector_explains_impact_and_next_action`, and
`test_console_empty_transcript_exposes_beginner_activation_actions` assert the
same card action/inspector copy but share no substring with any filter keyword,
so `-k` silently deselected them. One more
(`test_console_add_api_key_recovery_tolerates_missing_session_settings`) only
surfaced by grepping Tests/ for callers of the changed functions — it
monkeypatches the display helper to return `settings=None`, a defensive
contract the rewrite had to keep.

**What to do.** A `-k` filter matches test *names*, not behavior. Before
trusting it as a completion gate, `Grep` Tests/ for every function you
changed (`_console_provider_recovery_action`, `_build_console_setup_card_state`,
...) and run the full files that reference them — renamed or
indirectly-exercised callers are exactly where stale expectations hide.

**TASK-3500 recurrence (2026-08-24).** The planned lifecycle command used
`-k shared_rag_service`; pytest selected **0/50** tests and exited **5** because
the owning tests are named `TestSharedRagService*`. Correcting it to
`-k SharedRagService` selected and passed **10** tests (with 40 deselected)
and exited **0**. Treat a keyword-filtered green result as evidence only after
asserting both a nonzero selected/passed count and a successful exit status.

---

## Classifying user copy by loose substring invents the blocker you name first

**Incident.** TASK-2154.12. `build_console_disabled_reason` mapped the
setup-blocker sentence onto a short "Send blocked — …" reason with ordered
substring checks, `"model"` first. The real missing-API-key copy is "Add API
key in Settings > **Providers & Models** before sending." — which contains
"model" as a substring of the settings screen's name, so the Console spent
weeks telling users to "choose a model" when the actual blocker was a missing
key (and the missing-endpoint copy hit the same trap). The parametrized
mapping tests never caught it because they fed clean synthetic strings
("Provider setup needed: OpenAI missing API key") that share no wording with
the strings production actually emits; the mis-mapping only surfaced in a
live UAT walkthrough of the reason strip.

**What to do.** When tests parametrize a classifier over free text, include
the **verbatim production strings** as cases (grep the producers, paste them
in) — synthetic inputs exercise the branches you designed, not the text you
ship. And when substring-matching user-facing copy, match the most specific
phrase first and treat UI names ("Providers & Models") as false-positive
carriers for every keyword they happen to contain.

---

## Textual BINDINGS on a child are preempted by an ancestor's `on_key` that stops the event

**Incident.** TASK-2154.11 made the Console transcript's jump-to-latest pill
(`ConsoleTranscriptJumpPill`, a child of `ConsoleTranscript`) keyboard
activatable by adding `BINDINGS = [Binding("enter", ...), Binding("space", ...)]`.
The pilot test pressing `enter` on the focused pill kept failing: the action
never fired. Key events bubble from the focused widget up the DOM *before*
App-level binding dispatch (`App._on_key` -> `_check_bindings` over
`focused.ancestors_with_self`) ever runs, and `ConsoleTranscript.on_key`
stops `enter` mid-bubble — so the pill's binding table was consulted nowhere.
The widget-level `key_<name>`/`on_key` path is the only dispatch guaranteed
to reach a focused child first.

**What to do.** When making a child widget key-activatable inside a parent
that has its own `on_key` handler (transcripts, lists, message rows),
intercept the key in the child's own `on_key` (stop + prevent_default), the
idiom `ConsoleTranscriptActionButton.on_key` already uses — do not rely on
the child's `BINDINGS`, and write the pilot key-press test first: it is the
only thing that reliably exposes the preemption.


---

## A long no-match input does not prove a matched scanner is linear

**TASK-856, 2026-08-08.** The sanitizer's long-input regression used only a
string with no credential labels. It therefore exercised the scanner's cheap
no-match path while completely missing repeated suffix scans after successful
quoted-label matches. The final whole-branch review added a dense matched-input
probe and measured the old quoted path performing **94,996,790 characters** of
CR/LF search work on only **46,888 input characters**.

**What to do.** A complexity claim about a scanner needs adversarial input that
repeatedly takes the expensive matched branch. Count deterministic work—such as
characters searched or cursor visits—and assert a structural bound alongside
the exact output. Do not use wall-clock thresholds: they are noisy and can pass
a superlinear implementation on a fast machine.

---

## A button's region width proves nothing about whether its label renders

**Incident.** TASK-2154.14 (DS-01) relabeled the Console composer's `☰`
button to `Menu`, widening it 4 -> 6 cells. `button.region.width` and
`content_region.width` (6 and 4) both said the 4-cell label fit, and every
geometry assertion was green — but the painted UAT capture read `Me`.
Textual 8's `Button` reserves `line-pad: 1` (one column each side of every
rendered line) *inside* the content region, on top of padding, so the real
label budget is `region - padding - 2`. The trap compounds: `line-pad: 0`
is rejected by the TCSS parser (`_process_integer` errors on a literal `0`,
and the stylesheet loses every rule after the bad one — the generated
bundle documents an earlier collision), so the pad can only be cleared
inline (`button.styles.line_pad = 0`, which parses fine). The existing
`region.width == 14` pin on the neighboring `Composer ▾` toggle had encoded
the same +2 chrome without naming it; tightening that button to 12 only
worked *because* the pad was cleared.

**What to do.** When budgeting a Textual button label, verify with
`button.render_line(0).text` or a painted SVG/text capture — never with
region arithmetic alone. If a label needs its button's full content width,
set `styles.line_pad = 0` in Python (the CSS form does not parse) and
record the budget math in a comment, the way `_bounded_button` call sites
in `console_composer_bar.py` now do.

---

## A conflict-free rebase can still replay a test for an obsolete base contract

**Incident.** PR #1435 added `Tests/MCP/test_library_tools.py` on a branch
whose original base allowed raw `tools/call` dispatch. Current `dev` had since
added a typed security refusal requiring execution through the permission-gated
action. The rebase was entirely conflict-free because the feature commit
created the test file, so Git had no overlapping lines to flag; the focused
suite was what exposed the stale expectation. A prior task note even claimed
the test had been updated, but the committed tree still expected raw dispatch.

**What to do.** Treat a clean rebase as transport evidence, not compatibility
evidence. Re-run the feature's complete focused suite after rebasing, and verify
claimed conflict adaptations in the committed files themselves. Newly created
tests are especially likely to preserve assumptions that the new base has
intentionally invalidated without producing a textual conflict.

---

## A clear/cleanup assertion cannot pin the thing it cleans up — observe transient state DURING the window

**TASK-3170 Task 8, 2026-08-07 (Console auto-retrieve send-path injection).** The
in-flight "Retrieving…" placeholder staging call
(`self._stage_console_library_rag_launch(placeholder)` inside
`_maybe_auto_retrieve_for_send`, `chat_screen.py`) was pinned by **no test**.
Replacing the stage call with a bare `pass` left all 22 existing tests in
`Tests/UI/test_console_auto_rag_on_send.py` green. The reason: every existing
assertion about the placeholder was a **clear** assertion —
`assert screen._pending_console_launch_context is None` after a timeout, a
failure, or a zero-result outcome — and all three stay true whether the
placeholder was staged and then cleared, or never staged at all. The
assertions were only *transitively* meaningful; delete the stage call and they
go vacuous while continuing to pass. A future refactor could therefore remove
the only in-flight signal the user gets during the retrieval window and the
suite would say nothing.

**Fix:** a new test whose fake retrieval service observes
`screen._pending_console_launch_context` and `screen._has_staged_console_evidence()`
**from inside** the `search()` call — the only moment the claim ("a placeholder
is staged while retrieval runs") is actually true — then asserts the settled
launch afterwards is a *different* object with `status == "staged"`. Written
RED-first: the stage call was stubbed to `pass` before the test existed,
confirmed exactly 1 failure (the new test) against 22 unaffected; reverted via
Edit, production source byte-identical to the pre-fix commit.

**What to do.** Transient state — an in-flight marker, a spinner, a lock held
across an `await`, a placeholder later replaced or cleared — must be observed
**during** the window it exists, from inside the awaited call or an equivalent
hook, or it is not tested at all. `assert x is None` taken after the fact
passes identically whether `x` was set-then-cleared or never set: it is a
clear/cleanup assertion, not a presence assertion, and clear assertions cannot
pin the thing they clean up.

---

## A config default that also ships in the config template cannot be mutation-tested through config

**TASK-3170 Task 8, 2026-08-07, same task as above.** Mutating the read
site's fallback for `rag_auto_retrieve_on_send` — `get_cli_setting("chat_defaults",
"rag_auto_retrieve_on_send", False)` → `..., True)` — failed **zero** of the
20 tests then covering the feature. The cause: Task 7 had already added
`[chat_defaults] rag_auto_retrieve_on_send = false` to `config.py`'s DEFAULT
CONFIG TEMPLATE, so every freshly-bootstrapped test config carries the key
explicitly. The lookup therefore always resolves the template's stored value
and never falls through to the Python-level default argument — the literal
`False` in the `get_cli_setting(...)` call is dead code for every test, and
for every real user with a current config. The mutation is only reachable for
a user whose `config.toml` **predates** the key, a state no test that boots a
fresh config can ever produce.

**Fix:** `test_toggle_default_is_off_at_the_read_site` monkeypatches
`get_cli_setting` with a recording stub and asserts the literal default
argument handed to it is `False`, independent of whatever the template
supplies.

**What to do.** Before trusting a "defaults to off" (or any) test for a
config-backed value, check whether that same default also ships in the app's
default config template or any fixture/bootstrap path the test uses. If it
does, every test config already carries the key, and the code-level fallback
can drift to the wrong value with **zero** test failures — a mutation of the
fallback argument is invisible through the normal read-and-assert path
because that path is testing the template, not the code. The read site's own
literal default needs a separate, direct assertion (stub the accessor, assert
the literal argument passed to it), not an inference from observed runtime
behavior.

## A guard test must be PROVEN to discriminate — twice in one day it wasn't (2026-08-08, tasks 1359/2832)

Two review-verified tests, written by the same controller, both passed while
guarding nothing:

1. **`capsys` does not observe loguru.** The task-2832 log-privacy test
   asserted a secret query never appears in `capsys.readouterr()`. The
   reviewer emitted `logger.warning("… query=<the secret>")` DURING that
   exact test via a plugin — **1 passed**. loguru's default handler binds
   pytest's *global* stderr capture at import, so the per-test fixture sees
   nothing (and `capfd` misses it too). The house pattern is a list-appending
   sink: `sink_id = logger.add(lambda m: records.append(str(m)))` /
   `logger.remove(sink_id)` in `finally` — ~15 files already use it.
2. **A single-chunk MockTransport body makes any early-abort test vacuous.**
   The task-1359 crawl regression test proved a body was "read in full past
   the sniff window" — but `httpx.Response(200, content=bytes_blob)` delivers
   ONE `iter_bytes()` chunk, which the read loop appends before any abort
   check runs, so the whole body is captured even under the buggy predicate.
   Only multi-chunk (generator) delivery lets an abort actually cut a body.

**What to do.** For any test whose value is "this would catch the
regression": run the regression. Mutate the guarded code back to the buggy
shape (Edit-based, unique marker strings) and READ the red result before
trusting the green one. Both of these were caught only because the review
step re-ran the pre-fix code against the new test; neither red-check had been
done by the author, and both tests were the SOLE pin for their spec clause.

## A regenerated gate artifact is stale the moment something merges ahead of it

**Incident (task-3750, 2026-08-08).** `Docs/security/production-diagnostic-inventory.json`
is a checked-in artifact that a test regenerates and byte-compares. Its most recent
regeneration was commit `f990464ed` — and `f990464ed` was `origin/dev`'s tip, where the
test **failed**. The commit that regenerated the file left it stale on arrival: the
author ran `--write` on a branch, and the PRs that merged ahead of theirs moved line
numbers and added diagnostics. Green on the branch, red on dev, nobody at fault.

Two things follow.

1. **"The gate passed on my branch" is not evidence the gate passes on dev** for any
   test that compares against a regenerated whole-tree artifact. The only honest check
   is after the final rebase/merge — `test_screen_size_ratchet.py` already says exactly
   this in a comment ("a budget derived from a stale base fails the moment it merges"),
   which is how you know it is a repo-wide pattern and not one script's quirk.
2. **Design these artifacts so unrelated churn cannot invalidate them.** The inventory
   hashed each logger call's *line number* alongside its text, so any refactor that
   shifted lines failed a security gate with the call count unchanged and the sink
   topology byte-identical. Measured on dev: of 47 drifted entries, 28 were pure line
   movement. A gate that fires on no-ops trains reviewers to regenerate without reading,
   which destroys exactly the review it exists to force. Key such artifacts on content,
   and keep multiplicity (a sorted list, never a set) so deletions still register.

**What to do.** Before blessing a regenerated artifact, classify the drift instead of
running `--write` and staring at a 1,000-line diff: walk each file's history for the
revision whose digest reproduces the checked-in value, and diff content-vs-position.
That is what separated the 28 no-ops from the 19 real changes here, and it is what made
it cheap to actually read the diagnostics being newly blessed.
---

## A mounted widget with healthy data can still paint nothing — assert the painted region

**Incident.** TASK-3793, 2026-08-08. The Console rail's character avatar was
invisible (even the no-character placeholder was gone) and Roleplay thumbnails
painted black stripes — while every existing avatar test passed, because they
asserted the widget mounted, the DB bytes decoded, PIL produced an image, and
the mosaic content was non-empty. Two layout root causes, both invisible to
composition assertions: (1) the default-width avatar `Static` inside the
auto/auto `ClickableAvatarBox` (task-1661) resolved to 0x0 under Textual
8.2.8 — mounted, composed, painted nothing; a headless repro against the
owner's real DB image showed `region 0x0`. (2) The three thumb containers
reserved `max-width 24` *plus* `padding: 0 1`, so every 24-cell mosaic line
folded at 22 content columns; the continuation rows painted black (stripes)
and the folded 17-row stack exceeded `max-height 10` (bottom clipped) —
`region 22x17` where 24x10 was expected.

**What to do.** For image/avatar/rendering surfaces, pin the *painted region*
(`widget.region`, `render_line(n).text`, or an SVG/text capture) in addition
to mount state and data health — a green mount test says nothing about paint.
Two layout traps to budget for: a default-width child of an auto/auto
container collapses to 0x0 under Textual 8 (size it explicitly from the
renderable grid, as `explicit_cell_size()` now does for mosaics), and padding
inside a max-width container folds full-width lines into black continuation
rows on dark themes (content width = max-width − padding; drop the padding or
shrink the build width). The regression shape that caught both: mount the
real holder and assert region non-zero with height == mosaic rows.
## A wired kwarg is not a working option — assert the OUTPUT varies with the INPUT (task-3301, 2026-08-07)

**The incident.** Task-3301 wired the ingest form's "Chunk size" through to the
chunking service and wrote a test that a plaintext file chunked with
`{"method": "sentences", "max_size": 120}` produces more than one chunk. It
produced exactly one — 2,389 characters of it. The chunking stack's methods
size in their OWN units (`sentences` = sentence COUNT, `words` = word count),
so a form labeled "characters · 100–5000" feeding a hardcoded
`method: "sentences"` meant "120 sentences per chunk": the option was dead at
a SECOND layer even after the kwarg plumbing was fixed, and the PDF path had
shipped this exact combination for months (`max_size: 500` sentences ≈ one
chunk per document) without any test noticing — because every existing test
asserted the kwarg ARRIVED, none asserted the output CHANGED.

**What to do.** For any "wire option X through" task, the end-to-end test must
vary the option and assert the observable output varies with it (governance),
not merely that the value lands in a call's kwargs. A kwarg can land perfectly
and still be a no-op because of unit or key-name mismatches downstream
(`size` vs `max_size` was ALSO live here — `improved_chunking_process` reads
only the latter). The kwargs-arrival test and the governance test catch
disjoint bug classes; you need both.

---

## An app-importing pytest probe outside `Tests/` bypasses the suite's own config isolation

**TASK-3894 (P1 eval harness) Task 4, 2026-08-09.** Capturing real chunk-count numbers
for a new fixture corpus, a throwaway probe was written under the scratchpad directory
and run with plain `pytest`. It imported `tldw_chatbook` to call the real chunking path
— and because it lived **outside `Tests/`**, `Tests/conftest.py`'s config-isolation
fixtures (which sandbox `HOME`/`XDG_DATA_HOME` before `load_settings()` ever runs) never
applied, because pytest only collects and applies a directory's `conftest.py` for tests
under that directory. The probe's `load_settings()` therefore ran against the user's real
`~/.config/tldw_cli/config.toml`. No damage this time — the file's mtime was unchanged
afterward, confirming it was read-only — but that was luck, not design: nothing in the
probe prevented a write path from firing, and the probe read as an ordinary pytest
invocation the whole time it ran.

**What to do.** The rule is narrower than "use pytest for anything that imports the
app" — it is **a probe that imports the app must live under `Tests/`**, where the
isolation fixtures are actually collected and applied. A pytest-shaped file outside that
tree runs with none of the suite's safety and is functionally the same as a bare
`python -c` invocation against the app. If you need a throwaway probe, put it in
`Tests/` (even a temp file there) and delete it afterward — or better, promote whatever
it measured into a real, permanently-checked-in test, as this incident did
(`test_the_bare_word_will_appears_nowhere_in_the_corpus`).

---

## A hand-rolled normalizer used as a safety guard must be proven canonical, not just plausible

**TASK-3894 (P1 eval harness) Task 4, 2026-08-09.** A fixture corpus needed a guard
proving no keyword-category query's unique token accidentally overlapped a
vocabulary-mismatch/paraphrase pair's vocabulary. A hand-rolled `_stem()` (strip one
suffix, return) stood in for FTS5's real porter stemmer and looked like a deliberate,
safe over-approximation — the guard's own comment called it "stricter than a real
tokenizer." Review found the opposite was true for a whole class of words: because
`_stem` stripped exactly one suffix and stopped, suffix order decided the result, so two
spellings of the *same* word produced two different stems (`readings`→`reading` but
`reading`→`read`; `classes`→`class` but `class`→`clas`). FTS5's porter tokenizer
collapses every one of these pairs to one stem. The guard was therefore **weaker than
the mechanism it stood in for, in exactly the direction that matters**: it would score a
keyword-reachable pair as "no overlap" and let it ship silently. Two real fixtures
already carried exactly this escape — `vm-blood-pressure`'s "reading" against
`note-hypertension-followup`'s "readings"; a `pr-workout-time` "classes"/"class" pair
that had only been caught by hand, not by the guard.

**What to do.** When a hand-rolled normalizer stands in for a real one as a safety
check, "it looks stricter" is not evidence — an ordering artifact made the opposite true
here, unnoticed by the author. Fix by making the reduction a **fixed point** (re-apply
until the word stops changing) so the result is a function of the word family rather
than of which suffix rule fires first, then test it against the *real* mechanism's known
collisions (the actual inflection families a porter/whatever-you're-approximating
stemmer folds together), not only against your own corpus's current wording. This was
caught only because review independently re-derived what a real stemmer would do on
these words and diffed it against the guard's actual output — not by reading the
guard's code, which reads as reasonable on its own.

---

## HF offline enforcement must be set before `huggingface_hub.constants` EVALUATES, not merely "before import"

**TASK-3894 (P1 eval harness) Task 5, 2026-08-09.** A harness that embeds real documents
through a real model needed a hard guarantee that a run never downloads anything, even
on a cache miss. The first version set `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1` from
a pytest autouse fixture at test setup and the code claimed downloads were blocked. They
were not: an instrumented check showed `ENV HF_HUB_OFFLINE='1'` alongside
`constants.HF_HUB_OFFLINE=False` and `constants.is_offline_mode()=False` in the same
process. `huggingface_hub.constants.HF_HUB_OFFLINE` is computed **once, at import**, from
the environment as it stood at that instant; `is_offline_mode()` (which `transformers`
also imports directly) just returns that frozen global. An env var written from a
fixture at test *setup* — after collection has already imported half the world — arrives
too late to matter, and a cache miss would have silently downloaded ~87 MB into the
user's real `~/.cache/huggingface/hub`, the very directory the harness was pointed at.
The first fix attempt also used the wrong condition: "before `huggingface_hub` is
imported" is not sufficient, because hf_hub loads its submodules lazily —
`huggingface_hub` can already sit in `sys.modules` while `huggingface_hub.constants` is
still unevaluated, confirmed directly by forcing the hard case (evaluating `constants`
from a module that runs before the latch): the latch still worked, proving "before
import" was never the load-bearing condition. The fix that actually closes the hole has
two parts, and both were needed: set the env vars at **module top** of the gate module,
guarded on the harness's own opt-in env var, so they land before `constants` is
evaluated in the common case; and, for the case where something earlier in the same
session already evaluated `constants` with the var unset,
`monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)` directly on the frozen global —
the only thing that still works at that point. Mutation-tested independently: removing
either half alone reintroduces `is_offline_mode() == False` through a different path.

**What to do.** For any library that freezes an "offline"/"safe mode" flag into a
module-level constant at import time (huggingface_hub is one instance; do not assume any
other library isn't), "set the env var before you need it" is insufficient — the real
requirement is "before that constant is evaluated," which can be earlier than the import
of the top-level package if the package lazy-loads its submodules. Assert the *resolved*
state (`is_offline_mode()`), never the env var's string value: a `"1"` that arrived too
late reads as success on an env-var check while the flag it was meant to control stayed
`False`. And when closing a hole like this with a two-part fix, mutation-test each half
independently — here, either half alone was silently insufficient.

**`HF_HUB_OFFLINE` is not the only frozen constant in that module (TASK-16965,
2026-08-17).** `huggingface_hub.constants.HF_HUB_CACHE` is likewise computed
once at import, from `expanduser("~")` — so any fixture that sandboxes `HOME`
(this repo's `Tests/conftest.py` does) points every later model load at an empty
cache and makes a genuinely cached model unloadable under pytest. Same
mechanism, opposite blast radius: this lesson's is a download you did not want,
that one's is a load you did want and silently did not get. See "A metric can be
graded on fallback content" at the end of this file for what that cost.

---

## "Order-dependent" in the backlog is a hypothesis, not a diagnosis — a state flip is not proof the DOM caught up

**TASK-3022, 2026-08-07.** The backlog described two `Tests/UI/test_library_shell.py`
tests as "order-dependent notes-tail failures" (plus a third found during this task's
own sweep, `test_library_shell_notes_sync_now_calls_recording_service_with_chosen_enums`).
All three, when actually run alone, repeatedly (3/3, 3/3, and 2/3 samples respectively)
failed with `NoMatches` on a widget query, not intermittently the way real cross-test
pollution would present. Each had the identical shape: poll a plain/reactive attribute
(`_library_note_detail`, `_library_notes_view` + `_library_note_autosave_state`,
`_library_notes_sync_running`) in a `for _ in range(N): await pilot.pause(...)` loop,
then immediately do a **one-shot** `screen.query_one(...)` on a widget the same state
transition is supposed to (re)mount. Task-699 (2026-07-26) had already diagnosed and
fixed the first known instance of exactly this shape in the same file; these three were
new instances introduced by later test additions that never saw that diagnosis.

**Why it happens.** The Python attribute write and the Textual recompose that renders it
are not atomic. A handler sets `self._library_note_detail = new_value` and only later
`await`s back into the event loop for the recompose to actually mount the widget that
implies. A poll loop watching the ATTRIBUTE exits the instant it flips — one event-loop
tick before the widget it implies is guaranteed to exist. Whether a given run's timing
window is wide enough to hide this varies with machine load, which is exactly why it had
been filed as "order-dependent" rather than diagnosed: it LOOKS like flakiness (some runs
pass) without actually depending on any other test.

**What to do.**
1. Do not accept a backlog description of "order-dependent"/"flaky" at face value — a
   test that fails when run completely alone, even once, is not proof of cross-test
   pollution. Run it alone, several times, before hunting for a preceding-test trigger
   that may not exist.
2. Once a poll loop has established the STATE you care about, wait for the WIDGET too
   via `_wait_for_selector` (this file's helper — polls `screen.query`, a list, so zero
   matches is just "not yet") before reading it — never a bare `query_one` right after a
   state-only poll, since it raises the moment the DOM lags the state by even one tick.
   Cheap enough to apply proactively, not just after a failure is observed.

---

## A non-breaking space does NOT stop Rich/Textual from wrapping there (2026-08-07)

**Incident.** Task-2859 item 5's mid-unit wrap fix ("Prompts 144.0 / KB" splitting a
size number from its unit in the Library rail's narrow Details column) was first
"fixed" by replacing the space between number and unit with U+00A0 (non-breaking
space) — the textbook answer, and it read correctly in two quick manual checks (widths
20 and 29). A live tmux capture at the batch's own required 170x50 caught it still
broken: `"144.0"` on one line, `"KB"` alone starting the next, NBSP already in place.
Direct proof against `rich._wrap` (the module every plain `Static` wraps through):
`rich._wrap.words()` tokenizes with `re_word = re.compile(r"\s*\S+\s*")`, and Python's
`re` module's Unicode-aware `\s` **matches U+00A0 identically to an ordinary space** —
confirmed with `re.match(r"\s", "\xa0")` returning a match. So `"144.0\xa0KB"` is parsed
as TWO separate "words" for wrap purposes exactly like `"144.0 KB"` was; NBSP only
prevented the SPECIFIC widths tried by accident (enough room remained either way, or
"Prompts" itself already pushed the whole tail to the next line together). At the
rail's real width, exactly enough room remained for "144.0" alone but not "144.0" plus
"KB", so the split happened right where NBSP was supposed to prevent it.

**What to do.** Never assume a non-breaking space stops Rich/Textual `Static` word-wrap
— it does not, because Rich's own wrap tokenizer uses a plain Unicode-aware `\s` regex
that does not special-case U+00A0. If a number/unit (or any two-token) pair must never
split, either remove the space between them entirely (verified stable at every width
20-29 for this exact case — `_unbreakable_size_text` in `library_screen.py`), or use a
genuinely non-whitespace-classified character (e.g. U+2060 WORD JOINER, category `Cf`,
zero-width) if a visible gap must be preserved. Test the actual wrap behavior against
`rich._wrap.divide_line`/`words` (or a live capture at the real target width) — a
narrower or wider width than the one actually shipped can hide this exact bug either
way, which is why two quick manual checks at the wrong widths both looked fine.

---

## `Button.press()` called from an ancestor's own click handler silently breaks message bubbling one hop early (2026-08-07)

**Incident.** Task-2859 item 5: making a rail section header's LABEL (not just its
`▸`/`▾` toggle chip) clickable, by adding `DestinationRailSectionHeader._on_click`
that resolves the toggle `Button` and calls `.press()` on it. A live capture showed the
toggle's own CSS class flip to `-active` (proof `.press()` ran) but the section never
opened — no `Button.Pressed` handler anywhere fired, in the widget, the screen, or the
app. Reproduced deterministically in isolation with a minimal `Horizontal` header
wrapping a `Static` + `Button`: calling `child_button.press()` FROM the container's own
`_on_click` (itself invoked because the Static's Click bubbled there) breaks
propagation; calling the exact same `.press()` from the Static's own `_on_click`, or
from a plain test coroutine, works fine. Root cause, found by monkeypatching
`Message._bubble_to` to log every hop: `Message.__post_init__` stamps `self._sender =
active_message_pump.get(None)` — a CONTEXTVAR tracking whichever widget's message
dispatch is CURRENTLY executing, not the widget whose code literally calls
`post_message()`. Since `Button.Pressed(self)` is constructed inside `Button.press()`
while `active_message_pump` still reads as the HEADER (we are executing inside the
header's own dispatch of the bubbled Click), the new message's `_sender` becomes the
header. `MessagePump._on_message`'s bubble step has a special case: `if
message._sender is not None and message._sender == self._parent: message.stop()` —
"parent is sender, so we stop propagation after parent" (an optimization to avoid a
widget's own self-directed message re-bubbling past the ancestor that sent it) — and
this exact shape matches by coincidence, so the Pressed message reaches the header (one
hop) and then dies, never reaching the screen-level handler every real consumer
(Console/Home/Library rails) expects it at.

**What to do.** Calling `widget.press()` (or constructing/posting any `Message`) from
inside ANOTHER widget's own event-handler execution is not equivalent to the target
widget doing it itself — the message's `_sender` provenance silently changes, and
Textual's own "parent is sender" bubble-stop can misfire as a result. Fix: reset the
`active_message_pump` contextvar (`from textual.message_pump import
active_message_pump`) to the actual sending widget around the call —
`token = active_message_pump.set(target_widget); try: target_widget.press() finally:
active_message_pump.reset(token)` (see `DestinationRailSectionHeader._on_click` in
`tldw_chatbook/Widgets/destination_rail.py`). Proving this class of bug needs watching
the FULL bubble chain, not just checking that `.press()` "ran" — the visual `-active`
class flip is a false-positive signal; instrument `Message._bubble_to` (or count how
far a message travels) when a `Button.Pressed` handler mysteriously never fires despite
the button visibly reacting to the press.

## Constructing a widget directly in a test is not the same as driving it through real navigation -- and a "real navigation" pytest attempt can itself be a non-deterministic regression gate (2026-08-08)

**TASK-3200.** Fixing the shared `MainNavigationBar`'s mid-word tab-label clip (a
straddling destination button now gets a CSS "ghost" treatment — colors matched to the
bar's background — instead of `display: none`, since hiding a button changes the
strip's virtual size and can cascade). `Tests/UI/test_master_shell_navigation.py`
already had — and my own new tests added — coverage that constructed
`MainNavigationBar(active="settings")` directly inside a bare `TestApp`, at 80/100
columns, both an early and a late active destination, and every one of those tests
passed cleanly. Live tmux verification at 80 columns then reproduced a DIFFERENT bug
in the exact same scenario: navigating from Home to Settings via the command palette
left "Schedules" straddling and fully readable (un-ghosted) while an unrelated,
already-off-screen "Watchlists" stayed ghosted for no reason. Root cause: `on_resize`
ghost-checked directly, without first re-scrolling for the CURRENT viewport — a real
screen-to-screen navigation fires several resize events while content is still
settling, and whichever resize is the LAST one to land can ghost-check against a
scroll position computed for an EARLIER, narrower or offset layout. A widget
constructed directly in a `TestApp` sees exactly one clean mount → one clean resize;
it structurally cannot reproduce a sequence of several interleaved resizes racing a
still-settling scroll target.

**Second half of the incident, easy to miss.** Having root-caused it live, I wrote a
pytest test driving the REAL app (`Tests.UI.app_factory._build_test_app()`) through an
actual `NavigateToScreen` message, polling for the nav bar to reach a correct state.
It reliably failed against the buggy code and passed against the fix — until re-run a
few more times each way: the SAME buggy code sometimes passed within an 8s poll, and
disabling `MainNavigationBar`'s periodic 0.5s interval (which calls the same
scroll-then-ghost pair on its own schedule, independent of `on_resize`) via monkeypatch
made even the FIXED code fail a 20s poll. The honest conclusion: overlapping
`call_after_refresh` chains from several resize events can interleave such that the
last ghost-check to physically execute is not guaranteed to be from the freshest
chain — a genuine, narrow residual race that the fix reduces but does not eliminate,
and that the ALWAYS-PRESENT interval (this codebase's existing, intentional
"settle every tick" mechanism) papers over within a variable amount of real time. No
pytest timeout value reliably distinguished buggy from fixed once the interval was
back in play, so a test whose pass/fail depended on it was providing FALSE confidence,
not real regression coverage — it was deleted (along with its exclusive helpers)
rather than shipped in that state.

**What to do.**
1. A test that constructs a widget directly and pumps `pilot.pause()` is evidence the
   widget's OWN logic is internally consistent — it is NOT evidence the widget behaves
   correctly when driven by the app's real navigation/layout churn, which can fire the
   same hooks (e.g. `on_resize`) multiple times with different timing than a synthetic
   single-shot test ever produces. For a defect involving scroll/layout state that must
   "settle", drive the real navigation path at least once (post the actual message,
   wait for the actual screen class to change) BEFORE declaring victory on
   direct-construction tests alone — that is what surfaces this class of bug.
2. Before trusting a NEW test as a regression gate, re-run it several times against
   BOTH the buggy and the fixed code, not once each. A single RED + a single GREEN can
   still be a coin flip if the code involves unmocked real-time async settling
   (`call_after_refresh` chains, `set_interval` timers) — the fact that it says
   "5 failed" once and "3 passed" against the identical buggy commit later in the same
   session is itself the tell, not a fluke to shrug off.
3. If a bug is fundamentally a timing race between two async mechanisms (here: a
   settle-chain fix and an always-on periodic interval), a fast deterministic pytest
   assertion may not exist for it at all. Neither "leave the interval running,
   generous timeout" nor "disable the interval, even more generous timeout" reliably
   discriminated buggy from fixed here. Don't force a flaky or falsely-reassuring test
   into existence to satisfy a coverage checklist — ship the deterministic tests that
   DO reliably discriminate (the direct-construction geometry/rendered-text ones, in
   this case) and rely on documented, reproducible LIVE verification (tmux, before vs.
   after) for the part that genuinely can't be pinned by a fast unit test.

## A mutation test can stay green because a *second* self-healing mechanism rescued the mutated code (2026-08-09)

**Incident (task-3200 round 4 / task-3225).** `MainNavigationBar.on_resize` was
wired to a focus-aware recenter, with `test_resize_does_not_strand_the_focused_
button` as its regression guard. Reverting the wiring did not turn the test red.
The first diagnosis (round 3) was "the scenario never strands" -- true, but only
half of it. A hand-built scenario that DOES strand *still* passed against the
reverted code, because two independent backstops healed it faster than any
wall-clock assertion could look: the widget's own 0.5s settle interval, and --
the one nobody had accounted for -- a "best-effort nudge"
(`scroll_to_widget(focused)`) buried inside the ghost pass, which fired off a
*stale* region that still measured as straddling. Traced: with the fix reverted,
`scroll_x` went 86 -> 75 (wrong) -> 96 (rescued) inside 40ms.

**The rule.** When a mutation test refuses to go red, "the scenario is wrong" is
only the first hypothesis. The second is "something else fixed it for me."
Before trusting any timing-sensitive guard, enumerate every mechanism in the
system that could reach the same end state -- periodic intervals, deferred
re-checks, best-effort nudges -- and either suppress them for the duration of
the assertion (isolating the unit actually under test) or pick a scenario they
provably cannot reach. Here: suppress the interval via a test-local subclass
(patching the instance attribute does nothing -- `set_interval` captured a bound
method at mount), and choose a case that drags the button *fully off-screen*
rather than into a straddle, since the nudge only rescues straddlers. Result:
3/3 red on revert, 3/3 green on restore.

**Corollary on assertions.** "not straddling" was also too weak an invariant to
distinguish the good state from the worst one: a button dragged entirely
off-screen is not straddling either, and it is strictly worse (invisible, yet
still focused and Enter-navigable). Assert the property a user would name
("still fully visible"), not the negation of the specific bug you last fixed.

## An "invisible" CSS class that touches the box model is a layout change -- and a different CSS tier can hide that from your tests (2026-08-09)

**Incident (task-3200 round 4 / task-3225).** The nav bar makes a clipped tab
invisible with a CSS class instead of `display: none`, specifically so that
geometry never changes (hiding a tab reflows the strip, breaks `max_scroll_x`,
and cascades into new clipped tabs). The rule declared
`border: solid $background !important` -- and Textual's `Button.-style-default`
default is `border: none` plus `border-top`/`border-bottom: tall`, i.e. **zero
horizontal border cells**. So "make it invisible" silently made every ghosted
button **2 cells wider** (measured: 14 -> 16), reflowing every later button and
pushing an already-corrected, focused tab back into a clipped position one
layout pass after the correction landed. That was the whole "mysterious ~0.3s
drift-back": a settle pass's own trailing invisibility pass undoing the settle.

**Two generalisable traps.**

1. Invisibility rules must declare colors only. `border`, `padding`, `width` and
   `visibility` all move the box. If you want the Textual primitive for
   "invisible but still occupies space", note that `visibility: hidden` makes
   `Widget.region` return an EMPTY region (`outer_size` keeps its real value,
   `region.width` drops to 0) -- so any code that reads `.region` to decide
   whether to *un*-hide it can never see the widget again. Measured, and the
   reason that approach was rejected here.
2. **A widget-level `DEFAULT_CSS` bug can be invisible in the real app and live
   only in your tests.** This one never bit production: the bundle's
   `Button { border: none; }` sits in the `CSS_PATH` tier, which outranks widget
   `DEFAULT_CSS` regardless of `!important`, so the bad declaration was silently
   discarded in the running app -- and *only* applied in the bare `App()` test
   harness, which is where the entire deterministic suite for this feature runs.
   The harness was modelling a different layout regime than production. When a
   geometry finding comes out of a bare-widget test, re-measure it under a
   bundled-CSS harness (`CSS_PATH = tldw_cli_modular.tcss`, as
   `test_mcp_inspector.py`'s `InspectorAppWithBundledCSS` does) before deciding
   what it means -- in both directions: a bug the harness shows may not exist
   live, and a bug live may not show in the harness.

## A static symlink fixture does not prove a scanner is no-follow (TASK-13200, 2026-08-09)

**What happened.** The guided audio.cpp package scanner correctly skipped a
nested symlink that existed before scanning, so its original path-escape test
passed. A mutation fixture then replaced an already-queued directory with a
symlink. The next `scandir(path)` followed the new target and produced an exact
candidate outside the selected tree. A pre-open `lstat` alone still left a
smaller replacement window while the iterator was being opened.

**What to do.** Test no-follow traversal at three boundaries: a link present at
discovery, a queued directory replaced before traversal, and replacement while
the directory iterator opens. Fence the queued identity both immediately before
and immediately after opening the iterator; close without iterating if either
observation differs or becomes a symlink/reparse point. For files, combine a
no-follow open with `fstat` identity/type comparison before reading metadata.
Static fixtures prove policy for stable trees, not race safety.

PR #1463 review exposed a second portability trap: treating a missing
`O_NOFOLLOW` as flag value zero silently removed the primary file-open fence
while leaving the post-open identity check looking reassuring. Add an explicit
missing-capability mutation test. If the platform cannot guarantee no-follow,
fail closed before `open()` instead of opening first and rejecting afterward.
## A targeted subtree swap must account for route-owned siblings outside that subtree (2026-08-09)

**Incident (task-13213).** The Library optimized Notes/Media navigation by
replacing only `#library-canvas`. The Notes `Database | Files` source strip is
not a canvas child; it is a route-owned sibling composed above the shell grid.
The optimization therefore made Notes look selected while omitting the only
entry into file-backed notes, and it could leave the same strip stale after
leaving Notes. A shell-identity regression stayed green because it asserted
only the widgets the optimization deliberately preserved, never the contextual
sibling it had skipped. Once the route boundary was tested, the original code
failed at both 120x40 and 160x45.

**The rule.** Before introducing a targeted recompose, inventory every
route-owned surface, including siblings and wrappers outside the replacement
host. Encode that inventory as a structural signature and use the targeted path
only when the mounted signature matches the destination; otherwise await the
canonical full composition seam. Tests must assert both halves of the boundary:
contextual chrome appears on entry and disappears on exit. If stable child IDs
are reused, hide and detach the outgoing subtree before mounting its replacement
or Textual will reject the duplicate even when the route signature matches.

---

## Adding a resource of a GUARDED KIND obliges you to run that kind's inventory suite, not just your feature's tests

**Follow-up incident (TASK-31758 / PR #2437, 2026-09-05).** Forty-five
pixel-migu seed, resource, and installed-distribution checks passed after a
rebase, but the required generated-artifact job still failed: two new startup
diagnostics in `app.py` and `config.py` were absent from the production
diagnostic inventory. The missing local inventory check cost another complete
CI cycle. Inspect added logger statements with the checker's `--statements`
mode, regenerate the reviewed inventory, and run the derived-artifact checks
before pushing; passing feature and packaging tests does not cover that pin.

**Hybrid-fusion cluster (TASK-3996) Task 5, 2026-08-09.** The new notes/conversations
keyword sub-legs opened SQLite directly:
`sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)`. The choice was deliberate and
well-argued in the commit (read-only, never `CharactersRAGDB`, whose constructor does
schema and client-registration work on the user's main DB), it was covered by six new
tests, and it survived a full task review. It was also, the whole time, **already
failing a committed repo-wide guard**: `Tests/DB/test_private_sqlite_inventory.py`
asserts that the only production `sqlite3.connect` call sites are the private-sqlite
seam's own, with every owner enumerated in an inventory document and pinned by a ratchet
count. Nothing in the task's own test selection — the feature's tests, the RAG_Search
sweep, the eval battery — includes `Tests/DB/`, so the violation was invisible to every
run that was made. It only surfaced in a review round, from reading, not from a red
test.

This is the failure mode of targeted-test discipline (which is otherwise right: this
repo's rule is branch-relevant files plus a `--collect-only` sweep, not routine full
suites). Targeted selection is chosen from *the files you changed*. An inventory guard
lives in a directory you did not touch and asserts a property of the whole repo, so
"relevant to my change" and "relevant to the guard" are different sets, and the guard's
whole purpose is to notice the case where the author did not think it applied.

**What to do.** Before finishing a task, ask what KIND of thing it added, and whether
that kind is under a census: a raw DB connection (`Tests/DB/test_private_sqlite_inventory.py`),
a CSS class or token, a tool gate, a screen route, a config key. If yes, run that kind's
inventory suite *and add the new resource's row to its inventory*, in the same commit —
a guard you satisfy by exempting yourself is not satisfied. Name those suites in the
dispatch when the task is known up front to add such a resource, because the agent doing
the work is exactly the one who will not think to look for them. The fix here was to
route the sub-legs through `connect_private_sqlite` with a registered owner
(`rag.chachanotes_keyword_leg`, read-only URI), add the inventory row, and bump the
ratchet — which is what the guard existed to make happen, roughly a day later than it
should have.

---

## `Widget.focus()` is deferred — a same-handler capture of `app.focused` sees the old widget

**task-3311, 2026-08-09.** The Ingest Clear handler called `path_input.focus()` and
then a structural recompose helper that captures `app.focused` to restore focus
afterwards. In Textual 8, `Widget.focus()` does NOT set focus synchronously — it
queues `screen.set_focus` through `app.call_later` — so the capture still saw the
just-clicked Clear button, and the post-recompose restore targeted the NEW Clear
button, hidden for an empty path. `Screen.set_focus` silently no-ops on a
non-focusable widget, so focus stayed wherever the recompose prune dropped it: the
rail search box (typed path tail became a Library search) or nowhere (a leading
"/" ran the global focus-search binding). Live it presented as a 2-of-4
intermittent; headless, with a preflight staged, it failed deterministically on
iteration 0 of an 8-pass loop.

**What to do.** When a handler must hand focus somewhere before code later in the
SAME handler reads `app.focused`/`screen.focused` (capture-and-restore helpers,
recompose context savers), use the synchronous `Screen.set_focus(widget)`, not
`widget.focus()`. And remember `set_focus` on a non-focusable (hidden/disabled)
widget does nothing and reports nothing — a focus-restore path that can name a
display-managed widget needs the target to be focusable, or a fallback.

---

## A responsive focus handoff must cover both directions of widget replacement

**TASK-16220, 2026-08-14.** The first Console rail fix moved focus from a rail
that disappeared at a resize breakpoint to its reveal handle. That passed both
single-transition regressions. Independent review then exercised consecutive
boundaries: 117→118 focused the Context handle correctly, but 118→129 reopened
Context and hid that focused handle, leaving focus as `None`. The handoff had
modeled rail→handle replacement but not handle→rail replacement.

**What to do.** When responsive layout replaces one focusable representation
with another, test both directions and at least one consecutive transition.
Capture the logical owner before applying visibility, then synchronously focus
the visible counterpart after the update; two isolated one-way tests do not
prove keyboard continuity across adjacent bands.

---

## Bisecting dev-baseline test rot without a checkout: `git archive` trees run against the same venv

**task-3315, 2026-08-09.** `Tests/UI/test_library_shell.py` carried 56 failures on
the dev base, and the question that decided every repair was WHERE each family
broke: the ingest arc, dev's own churn, or the very PR that authored the pins.
With mutating git commands off-limits (shared checkout, other agents active),
`git archive <sha> | tar -x -C scratch/tree_<sha>` + `cd tree_<sha> && <worktree>/
.venv/bin/python -m pytest ...` reproduced the suite at any historical commit —
cwd wins over the editable install on sys.path (verify once: print
`tldw_chatbook.__file__`). Running the 60x20 geometry family at `6b4ccf475` (the
notes-adaptive PR #1439 merge that INTRODUCED those tests to dev) and at the dev
base proved the identical 14-test failure set at both: the family was born broken
at its own merge, and the media-ingest arc was exonerated in one run. The same
technique caught that the pins' authoring-branch snapshot (`42c994486`) was itself
too broken to run — "the tests passed when written" is not a safe assumption for
a PR whose battery only ever ran `-k` slices.

**What to do.** When a full-file suite is red on a base you didn't build, don't
reason from blame alone: extract the tree at the suspect merge commits with
read-only `git archive` and run the failing family there. A failure set identical
at the introducing merge and at base names the culprit (pins merged unvalidated)
and scopes the fix to re-pinning; a set that appears only later points at product
churn to bisect further. Corollary of the incident: line numbers in an earlier
failure report drift as you edit the file — re-derive the failing STATEMENT
before diagnosing (a "status query" failure here was actually the post-completion
query racing the finish-of-run recompose, three asserts later than first read).

---

## A heuristic candidate list is not a complete remediation inventory (2026-08-09)

**Incident (TASK-2118 final review).** A spelling-filtered logger sweep was
correctly documented as heuristic AC evidence, but its content-bearing subset
was later copied into a follow-up task as though it were the complete
summarization privacy inventory. Reviewing every logger call in the two owned
modules found many more prompt, response/output, credential-fragment, private
endpoint/path, and exception/error-detail diagnostics that the filter was never
designed to find.

**The rule.** Preserve the stated proof boundary when evidence crosses into a
follow-up. Build remediation inventories from the complete owning population,
grouped by stable module/function/diagnostic identity; use heuristic matches
only as candidates or cross-checks, never as the denominator.

---

## A line-independent diagnostic digest can still be indentation-sensitive (TASK-14651, 2026-08-09)

**Incident.** The persistent-diagnostic inventory described its call digests as
position-independent. Moving an unchanged multiline Library diagnostic into a
more deeply nested block still changed its digest because
`ast.get_source_segment()` retains continuation-line indentation. Later in the
same reconciliation, range-formatting three already-reviewed logger calls made
the architecture gate red again even though their AST behavior was unchanged.

**The rule.** Treat this inventory as line-number-independent, not
whitespace-independent. When reviewing a delta, compare the actual logger-call
AST/source as well as the digest so a pure indentation change is not mistaken
for a policy change. Run formatter gates before the final generated-artifact
refresh, then rerun the inventory checker after formatting. Do not refresh the
manifest first and assume later formatting is harmless.

---

## A generated-inventory rebase conflict is a new review boundary (TASK-14651, 2026-08-10)

**Incident.** Rebasing the diagnostic-privacy PR 79 commits onto current dev
conflicted in the generated manifest. Regenerating it made the architecture
gate green, but also imported 17 upstream diagnostic additions since the prior
reviewed base. Sixteen carried implicit tracebacks, exception messages, bound
session/message IDs, a media ID, or user-entered trim values. Treating the
generator as a mechanical conflict resolver would have silently blessed every
one.

**The rule.** A governed generated artifact must be re-reviewed when its source
population changes during rebase. Compare the diagnostic call population from
the last reviewed base to the new base, classify each added or changed call
under the governing ADR, extend the guard for newly observed syntax such as
dynamic `exception=` values, stdlib exception/stack capture, chained
`bind(...)` fields, and direct keyword-format values, then regenerate. Passing
the generator proves consistency, not policy compliance.

---

## An aggregate scanner label must not erase an explicitly accepted candidate identity (TASK-13201, 2026-08-09)

**What happened.** Guided audio.cpp launch tests placed Supertonic and PocketTTS
in separate temporary roots, so each rescan returned discovery state `exact`.
The real user package directory held both reviewed GGUF files. The scanner
correctly returned one `ambiguous` discovery containing two exact candidates,
and both candidates had been explicitly accepted with their recipe, root,
configuration, and weight identities. Launch revalidation nevertheless required
the aggregate discovery state to be `exact`, so it rejected the otherwise
unchanged two-model setup before creating the generated configuration.

**What to do.** Keep unresolved discovery ambiguity and accepted-candidate
identity as separate concepts. Never silently choose from an ambiguous result,
but once the user has explicitly accepted a candidate, revalidate that exact
candidate and require one matching identity; do not reapply the aggregate label
as if no choice had been made. Multi-model integration fixtures must include the
common real layout where several supported packages share one selected root,
not only the tidier one-directory-per-model arrangement.

---

## A repository scanner must decode source explicitly or a platform can silently disappear files

**Console model-picker verification (TASK-3600/TASK-14812, 2026-08-10).** The
blocking-I/O architecture suite reported eight stale baseline entries for two
Chatbooks modules even though the referenced `.glob()` and `ZipFile` calls were
still present. On Windows, `_scan_package` used `Path.read_text()` without an
encoding. The locale codec could not decode bytes in those UTF-8 source files;
the scanner caught `UnicodeDecodeError` and skipped each entire module. The
stale-baseline assertion was therefore reporting missing scan input, not clean
code. Reading the same files explicitly as UTF-8 restored the intended findings
and made all six guard tests pass.

**What to do.** Repository-wide source scanners must use an explicit source
encoding, normally `encoding="utf-8"`, and must treat decode failures as visible
evidence rather than silently interpreting them as a clean file. When an
inventory entry becomes "stale" while the named code is visibly still present,
inspect the scanner's input and exception path before deleting the baseline.

---

## Windows device names still capture filenames with extensions

**task-14811.1, 2026-08-10.** A new auxiliary-attempt migration test named its
SQLite fixture `aux.db`. The preceding 16 focused cases passed, but Windows
resolved that basename as the reserved `AUX` device (`\\.\aux`) even with the
`.db` suffix. The private-SQLite directory verifier then correctly rejected the
device path as a missing/non-directory parent, producing a long security-stack
trace that initially looked like a database privacy regression. Renaming the
fixture to `attempts.db` made the unchanged production path pass; 18 focused
tests then completed green.

**What to do.** Keep temporary and fixture basenames away from Windows reserved
devices (`CON`, `PRN`, `AUX`, `NUL`, `COM1`-`COM9`, `LPT1`-`LPT9`), including
when an extension is present. When a Windows path unexpectedly canonicalizes to
`\\.\<name>`, inspect the basename before weakening private-path validation.

---

## Credential-presence probes must never print regex captures, and live config reads still need an isolated home

**task-14811.5, 2026-08-10.** A PowerShell probe intended to print only the
names of configured providers reused the automatic `$Matches` variable across
regex operations. The later operation replaced the expected provider-name
capture, so the script printed three credential values instead. In the same
verification pass, a helper that did not instantiate the application database
still imported configuration code that ensured a chat-dictionaries directory
under the real user profile. Neither behavior was required to prove the
feature, and the exposed credentials had to be treated as compromised and
rotated.

**What to do.** A credential-presence probe should emit a fixed provider label
only after testing whether a value is non-empty; never print, interpolate, or
retain the matched credential and never depend on PowerShell's process-global
`$Matches` state for the label. For a real-provider smoke, point
`TLDW_CONFIG_PATH` at the existing config only for read access, set `HOME` to a
validated scratch directory before Python imports the application, use an
explicit in-memory or scratch database, hash the real config before and after,
and remove the verified scratch path on exit. A helper that can make billable
requests must also require an explicit confirmation flag.

---

## Captured async exceptions retain non-serializable transport locals

**TASK-14811.6, 2026-08-11.** The full parallel CI suite intermittently exceeded
a fake WebSocket server's five-second receive allowance. Its handler stored the
original exception for a later test-side re-raise. That exception retained the
handler traceback, including the live `websockets.asyncio.server.ServerConnection`
local. pytest-xdist then failed while serializing the report with an execnet
`DumpError`, hiding the ordinary timeout behind an internal runner error on both
macOS and Ubuntu. The same signature reproduced on the latest `dev` baseline.
The first regression opened its own client connection and repeated the failure
when its test frame was serialized, so the durable regression had to exercise
the content-only exception copy without creating any transport objects.

**What to do.** When an async test helper captures an exception across a task,
thread, process, or worker boundary, do not retain its traceback-bearing object.
Store a content-only diagnostic exception (type name plus message), and test that
its traceback is absent without putting a transport in the regression's own frame.
Keep positive wire waits long enough for full-suite scheduling contention while
leaving negative "nothing arrived" grace windows deliberately short. Verify the
helper with the same xdist distribution flags used by CI; an isolated serial pass
does not exercise report transport.
## A forecast-equals-receipt governance test proves nothing about the backend it does not drive (TASK-14827, 2026-08-10)

**Incident.** The 14820-14826 arc rebuilt the Library ingest forecast so the
commit line, consent line, tooling fold and Start gate all derive from one
`IngestForecast`, and pinned it with
`test_forecast_counts_equal_the_real_receipt_for_a_mixed_folder`: real
pre-flight, real submit, real DB, forecast counts asserted equal to the actual
job outcomes file by file. Strong evidence — for the LOCAL backend, the only one
it drives. In the same review round, TWO server-path divergences were found by
reading, not by the suite. (1) Local tooling gaps were subtracted from a
server-bound forecast, so five .mp3 on a machine without the audio extra read
"0 will import · 5 will fail (need tooling)" for a batch the server would have
transcribed in full. (2) An unsupported file was forecast "will skip" while
`build_server_ingest_kwargs` raised and the job landed as `✗ failed`. Both sat
inside the arc's own governed area, with 1,900 tests green.

**The second trap, which the first fix walked into.** The two backends refuse
DIFFERENT sets, so "ask the backend" is not a formality. Locally, unsupported
means `get_type_group(...) == UNSUPPORTED_GROUP`. The server additionally
refuses everything it has no media type for — raster images, deliberately left
server-unmapped — while NOT refusing a web page, because the submit path routes
pages to the clipper before the ingest-jobs mapping is ever consulted. A
predicate derived from either backend alone is wrong in both directions: reuse
the local verdict and images are promised as imports; ask
`server_media_type_for` alone and every server-mode URL import is condemned. The
fix asks the same functions the submit path asks, in the same order.

**The rule.** When a screen makes a promise about an outcome and more than one
backend can deliver that outcome, the governance test is per BACKEND, not per
screen. A second such test is cheap: keep everything real down to the narrowest
seam that cannot run in a test — here `TLDWAPIClient`, i.e. the network — so the
real request builder, the real response schemas, the real registry and the real
reconciler all participate. Bind every call to the stand-in against
`inspect.signature` of the real client method so the double cannot absorb a
drifting call site, and state in the test docstring exactly what the stub decides
(this one accepts everything handed to it, so it proves what the app SENDS and
what it refuses to send — never what a real server does with a file it received).
Anything the stub would have to invent is a fixture you must leave out and name:
this fixture holds no 0-byte file, because the app sends one and only the server
decides.

**The follow-through (TASK-14910, 2026-08-11).** That last sentence turned out to
be the finding, not the caveat. A fixture you cannot write because the outcome is
unknowable is usually pointing at a CLAIM the product should not be making: the
same forecast counted that 0-byte file as a certain failure while admitting, one
segment later, that "server tooling isn't checked from here". The fix was not a
cleverer stub — it was to make the outcome knowable, by refusing to send a 0-byte
file at all (the client already knows why, the local backend already refuses one,
and the round trip buys nothing). The fixture then grew to hold `empty.txt`, whose
fate is now decided entirely by code the test runs for real; the stub never sees
it. So: when a governance test has to leave a case out, name it AND file it — the
gap is evidence about the product, and the honest close is usually to remove the
unknowability rather than to keep the fixture short forever.

---

## An `await event.wait()` on a fire-and-forget task hangs on the task's OWN exception — and the timeout dump names nothing (2026-08-10, TASK-3316/TASK-15104)

**Incident.** `Tests/UI/test_screen_navigation.py::test_file_notes_collections_
source_transition_blocks_mutation_through_recompose` hung forever on dev, so
pytest-timeout's `thread` method killed the whole pytest process and **every
test after it in the file never ran** (the task-1466 class). The test drives the
screen coroutine as a background task and then waits for a signal only that
coroutine can set:

```python
source_switch = asyncio.create_task(screen._select_library_rail_row(...))
await sync_returned.wait()          # unbounded
```

The coroutine never got there. Its stub `_flush_library_note_save` returned
`None`, matching the seam's `-> None` signature at the time the test was written
(`eb036a6a1`, 2026-07-27). PR #1439 (`6b4ccf475`, on dev 2026-08-08) retyped the
seam to `NoteFlushOutcome` and made the caller read `note_flush.kind` — so the
awaited path died on `AttributeError: 'NoneType' object has no attribute 'kind'`
one line in. A `create_task` result nobody retrieves swallows that exception
whole, and the signal became unreachable. **This predates the media-ingest arcs:**
running the test from `git archive` copies of both sides of that merge is
decisive — `86e511781` (its first parent) *1 passed in 3.64s*, `6b4ccf475`
*hang → process killed*.

TASK-2512 rediscovered the same harness drift when its repository run reached
about 83% and stopped advancing. The exact node timed out at 300 seconds on
both the feature branch and clean `origin/dev` `8d764c03`; TASK-15104 then
changed only the stub to `NoteFlushOutcome(PERMITTED)`. The exact node passed
in 1.08 seconds and its eight-node adjacent group passed in 2.18 seconds. That
branch/clean-dev comparison plus the typed-stub mutation proved a shared test
harness defect, not an MCP runtime regression.

**The trap that cost the most time.** task-1466's advice, "the timeout stack dump
names it", does NOT hold here. The dump showed only `MainThread` idle in
`selectors.select` under `run_forever` — the test coroutine is *suspended at an
await*, so it has no frames on any thread stack. The dump is silent about which
`await` and silent about the exception. The only thing that talks is bounding the
wait and asking the task: `await asyncio.wait({waiter, task}, timeout=...)`, then
`task.result()`. That turned a 300-second process-killing silence into
`AttributeError` in 0.9s.

**The rule.** A test that awaits a condition a *background task* must produce has
two failure modes fused into one hang: the task can return early, and the task
can raise. Bound the wait at its source and settle both — if the task is done and
the signal is not set, re-raise its exception (or report the silent early return)
instead of waiting. The bound is not belt-and-braces; it is the only thing that
converts "the run died and you do not know why" into a named failure. Mutation
proof for this one: restoring the stale `return None` with the bound in place
fails in 2.2s naming `AttributeError`, where before it hung.

**Two corollaries, both paid for here.**
1. *A monkeypatched stub is a copy of a contract with no type checker behind it.*
   Nothing warns when production retypes the seam; the stub keeps the old shape
   and fails at the call site, which may be somewhere nothing is watching. When
   you change a seam's return type, grep the tests for stubs of that name — the
   same PR left `test_screen_navigation.py` with three of them.
2. *You cannot know a file's pass count while it contains a hang.* Bounding this
   one test took the file from "died at test 12" to `126 passed` — and revealed
   two more hard failures (`_library_note_dirty` became a read-only property; the
   prompt editor's guarded exit now needs a running App) that had been invisible
   for days behind the hang, plus one load-sensitive flake. "That file is green"
   was never true; it was never finishing.

---

## A test double mirroring the BASE class is blind to the SUBCLASS the app actually runs (TASK-14751, 2026-08-10)

**Incident.** TASK-14751 added a keyword-only `keyword_source_types` kwarg to
`RAGService.search` and had the Library's hybrid arm pass it down. Fourteen new
tests over real media + ChaChaNotes databases were green, the Library suites
were green, 413 targeted tests were green. Then the informational gated run
(`RAG_EVAL=1 pytest Tests/RAG_Eval/`) crashed three tests with
`TypeError: EnhancedRAGServiceV2.search() got an unexpected keyword argument
'keyword_source_types'`. `EnhancedRAGServiceV2` is the class the Library
actually resolves at runtime, and it overrides `search()` with an explicit
signature, so it does not inherit new base-class kwargs. Every double in the
unit suites (`_ProfileRagService`, `FakeRagService`, the new spy) was written
to mirror `RAGService.search` — the base — so nothing in ~2,500 unit tests
could see it. The same class's docstring already warned about this for
`metadata_allowlist`, and the warning was not enough to prevent the repeat.

**The rule.** When you add a parameter to a method, `grep` for overrides of that
method (`def <name>(` in the package) before you add the caller, and add at
least one test that drives the class the PRODUCTION resolver returns, not the
base class your doubles copy. A doubles-only suite pins the contract you wrote
down, never the object graph that runs. Corollary: an "informational, expect
no metric movement" gated run is not ceremony — this one earned its slot by
being the only thing in the repo that built the real runtime class, and it
caught the defect as a hard crash, not as a metric delta. (After the fix its
deltas were +0.000 in all three modes exactly as predicted, which is why the
crash, not the numbers, was the whole value of running it.)

---

## Retuning a numeric constant obliges you to grep its LITERAL VALUES, not just its symbol (TASK-4110, 2026-08-09/10)

**Incident.** Shipping the RAG hybrid-fusion `rrf_k` default `60 -> 5` took
three separate rounds to find every place the old value had leaked into prose,
because each round only swept one surface:

- **Task 5 round 3** grepped the SYMBOL (`rrf_k`, `DEFAULT_RRF_K`,
  `resolve_rrf_k`) and found four docstrings/comments still asserting `k=60`
  verbatim.
- **Task 6** grepped a downstream literal VALUE, `0.016` — the fused-score
  ceiling (`1/(60+1)`) that `k=60` produces arithmetically — and found a fifth
  location, a module docstring that named no `rrf_k`-family symbol at all,
  only the number the old constant happened to produce.
- The **final whole-branch review** found the seventh and eighth — two
  docstrings (`Event_Handlers/Chat_Events/chat_rag_events.py`,
  `RAG_Search/pipeline_builder_simple.py`) whose precedence-chain prose read
  "... -> active profile -> 60" — that neither earlier grep could have caught,
  because a bare literal `60` sitting inside an English arrow chain matches
  neither a symbol grep nor a `0.016`-shaped value grep.

Eight stale locations, three different grep strategies, three review rounds,
on a value everyone involved already knew had been retuned.

**What to do.** When a numeric constant is retuned, a symbol-only grep is not
a complete sweep. Enumerate every SHAPE the old value can still appear in and
grep each one separately: the symbol itself; any literal downstream
arithmetic consequence the docs may quote (a derived ratio, a ceiling, a
percentage — here, `0.016`); and the bare literal in comparison/precedence
prose ("-> 60", "an order of magnitude below X", "defaults to 60"). A
docstring can assert a stale number while never naming the constant that used
to produce it — that is exactly what lets it survive a symbol-only grep, and
exactly why an inline-literal arrow chain needs a human reading the prose, not
a tool, to catch on the first pass.

**The same class again, one level up: a stale VOCABULARY, and a sweep that
stopped one file short — twice in one review round (TASK-15700,
2026-08-13).** The keyword leg's default MATCH construction moved
`and_stopword_trim -> and_then_prefix`. The implementer swept and corrected
the affected prose; the review then found **two Importants that were both
twins of corrections already made elsewhere** — `_is_fts5_stopword`'s
docstring still said the list runs on every default search (the module
comment 3,380 lines above had already been rewritten to say the opposite),
and a test's property (b) still called the all-primary case "EVERY sub-leg
under the shipped `and_stopword_trim`" (the production docstring one
directory away had already been fixed). Re-sweeping the full RAG scope
mechanically then found **two more sites the review had not listed**. And a
third shape survived all of that into the closing task: the phrase "this
SPARSE **49-document** corpus", copied from a pre-P2ab README paragraph into
**three** newly written files (`config.py`, `rag_service.py`, a test
docstring) — a corpus that has held **172** documents since 2026-08-11, so
the number qualifying the arc's own headline cost figure was wrong in every
place the arc itself had written it.

Two additions to the rule above, both cheap:

- **Sweep for the VALUE *and* the VOCABULARY.** A retuned enum-like value
  (`and_stopword_trim`) leaves the same debris a retuned number does, plus a
  second kind: prose that describes what the old value DID ("drops function
  words", "never runs a second query") without naming it. Grep the old
  identifier, then grep the old behaviour's distinctive phrases.
- **Fix by re-sweeping the whole scope, never by patching the flagged
  lines.** Every finding in this incident was a *class* with more members
  than the reviewer listed; patching the reported line and stopping is what
  produced the twins. After the last edit, re-run the grep over the full
  scope and read every surviving hit aloud as a claim about today — the four
  that survived here were all correct historical statements, and knowing
  that is the difference between a clean sweep and an unfinished one.
## A gate with several conditions can close for the WRONG one — open the others or the test pins nothing (TASK-14911, 2026-08-11)

**Incident.** `start_enabled` on the Library ingest canvas is a conjunction:
registry present AND media DB present AND a non-blank path AND nothing-importable
false AND no option errors AND no path error. The new test staged a folder of
images in server mode and asserted `state.start_enabled is False` — the defect
being that it was `True`. The first run of that test, written BEFORE the fix,
reported the gate already closed. Not because the gate worked: the shared screen
harness (`Tests/UI/app_factory.py`) leaves `media_db = None`, so a *different*
conjunct was False the whole time. Had the test asserted only `start_enabled`, it
would have passed against the unfixed code, pinned nothing, and stayed green
forever after someone later broke the backend-aware gate.

It was caught only because the same test also asserted the specific flag the fix
introduces (`selection_has_nothing_importable`) and the gate line's own wording —
those two went red while the boolean did not.

**What to do.** For any multi-condition gate:

1. In the fixture, explicitly OPEN every condition except the one under test
   (`app.media_db = SimpleNamespace()` here), and say in a comment why — a shared
   harness's defaults are not neutral.
2. Assert the REASON, not just the closed state: the flag the fix sets, and the
   user-visible sentence naming it. A boolean shared by six causes cannot
   discriminate between them.
3. Run the test before the fix and READ which assertion fails. "It failed" is not
   enough when the boolean can fail for free.

---

## Owner state is not evidence that retained rows are mounted (TASK-14904, 2026-08-10)

**Incident.** Session Git workspace tests waited until the owner-published row tuple
had two entries, then programmatically pressed Stage or Unstage. The retained
`ListView` was still clearing and mounting that generation. A second status render
could cancel the row worker mid-clear, leaving `Pilot.pause()` waiting 30 seconds on
pruned child message pumps even though the action service and owner state were
correct. Adding one disclosure control changed timing enough to make the latent race
repeatable.

**What to do.** Treat the immutable model projection and the mounted row generation
as separate readiness boundaries. Disable row-derived mutations while rows are being
replaced, and make tests wait for model count, mounted count, and list visibility to
agree before pressing a row action. While a `ListView.clear()`/extend cycle is in
flight, poll service state with `asyncio.sleep`; `Pilot.pause()` deliberately waits on
every message pump and can turn transient child teardown into a harness deadlock.

---

## A compaction threshold is not a send-admission ceiling (TASK-14913, 2026-08-11)

**Incident.** The first Console memory release routed both an unknown automatic
budget and a threshold crossing with no replaceable units to a single
"compaction cannot run safely" blocker. The exact prepared request had not
exceeded a known provider input ceiling, but the default `Ask` policy made every
send on an unrecognized model fail before dispatch. A user-supplied bounded
budget could still reach the same false blocker when no older complete unit was
eligible for replacement. Policy, lifecycle, serialization, and modal tests all
passed because none asserted the ordinary send outcome for an unavailable
compaction decision.

**What to do.** Keep the compaction high-water threshold and provider send
admission as separate decisions. `UNKNOWN_WINDOW` and `NON_COMPACTABLE` mean
"do not compact now"; they block the message only when the immutable prepared
request also proves `known_overflow`. Test the complete decision cross-product:
unknown model with inherited automatic budget, unknown model with a bounded
custom budget, and known mandatory-material overflow. A settings-only assertion
does not prove that the next send consumes the saved policy correctly.

---

## A reviewed-safe label needs adversarial provenance evidence (TASK-3796, 2026-08-10)

**Incident.** TASK-3796's exhaustive ledger initially classified 199 diagnostics as
private and froze 324 as reviewed-safe. Final review found that
`general-2efc909241862caf` rendered `event.get("type")` from a Cohere streaming
response. The value looked like bounded status metadata in the source review, but an
unknown provider event can choose that string. A sentinel passed through the real
`summarize_with_cohere()` generator and fully consumed the response; it reproduced the
provider-controlled value in captured diagnostics. Restoring the historical
interpolation made that sentinel fail, and the corrected ledger became 200 private /
323 reviewed-safe.

**What to do.** Do not freeze a dynamic diagnostic merely because its field name
sounds operational. Prove where the value originates. For response events, config
values, and adapter metadata, drive an adversarial distinctive value through the real
production function and capture the actual logger path. A reviewed-safe classification
is evidence only after the sentinel proves the producer—not the reviewer—bounds the
value.

---

## A boundary projection must reject fields it does not understand (TASK-3796, 2026-08-10)

**Incident.** TASK-3796's first permanent manifest-boundary test rebuilt an allowlist
projection from known top-level fields and excluded the derived `task_492_calls`
summary. That made two checks look stronger than they were: a newly introduced
top-level section disappeared before hashing, and a self-consistent generator could
change both the owner counts and their derived summary while the summary-delta
assertion still agreed with itself. Review mutants adding an `unreviewed_section` and
forging `task_492_calls` exposed both gaps. The repair normalizes a deep copy of the
complete manifest, masks only the two explicitly owned count/digest fields, and
recomputes the TASK-492 summary independently from every owner row.

**What to do.** For a governed artifact, project by copying the whole schema and
masking the narrowly authorized fields; do not reconstruct an allowlist of fields to
retain. Validate derived totals from their primary rows before normalization. Then
mutate an unknown field and mutate the derived value independently: both must make the
boundary test red. Equality between two values produced by the same regeneration path
does not independently validate either one.

---
## A test written in the same PR as the change it pins can be born red — and then that PR's own omission ships (TASK-14920, 2026-08-11)

**Incident.** `7dbbc401b` (TASK-2154, FB-07) moved every Console "Save as..."
confirmation from `severity="information"` to `severity="success"` — and shipped
`test_console_save_as_savers_confirm_at_success_severity`, which asserts all four
destinations do so. It missed the Chatbook destination. Nobody noticed for four days,
because the test it shipped alongside called `console._save_console_message_as_media(...)`
— a seam decomposition wave 3 (`391b7bf69`, merged ~9 hours earlier the same day) had
already moved onto `ConsoleMessageController`. The test raised `AttributeError` on line
one of its four calls and **never once ran green**; the same PR's
`test_console_settings_save_fires_success_toast` was born red the same way against
wave 2's `_ensure_active_console_session_settings`. Repointing them at their controllers
made the first fail on exactly the assertion the PR had forgotten to satisfy — proving
the never-green test had been masking a real, shipped defect the whole time.

Verified with `git archive 7dbbc401b | tar -x` into a scratch tree: both tests fail at
the very commit that introduced them.

**What to do.** A test added in the same change as the behaviour it pins deserves the
same "confirm it could have gone red" treatment as a guard: run it, read the pass count,
and read WHICH assertion moved. And when a decomposition wave moves a seam, the delegator
shim it leaves behind for the "direct-call convention" is only as good as its coverage —
wave 3 kept `_save_console_message_as_note` and `_save_console_message_image` on
`ChatScreen` and silently dropped the other three savers. `Tests/UI/test_console_moved_seam_guard.py`
now checks that shape mechanically (AST + the live classes), because "the AttributeError
is loud" is only true for tests that anything actually runs.

---

## Production's broad `except` turns a stale test double into an INVERTED contract (TASK-14920, 2026-08-11)

**Incident.** `a6cc05d8b` ("seed dynamic character chat templates") moved the character
handoff's greeting seam from `store.append_message(...)` to
`store.seed_character_roleplay(...)`. Six tests across two suites
(`test_console_native_chat_flow.py`, `test_personas_workbench.py`) drove that handoff
through hand-rolled store doubles implementing only `create_session` and
`append_message`. The handoff wraps its seed call in `except Exception: logger.warning(...)`,
so the double's missing method surfaced as a swallowed `AttributeError` — not as an error.
The tests did not blow up; they quietly started observing "no greeting was ever appended"
and their assertions (`identity_at_append is None`, `store.messages == []`) began pinning
the ABSENCE of the behaviour they were written to prove.

That is worse than the familiar "a fake written to match your call site" trap two entries
up: there the double agrees with a wrong assumption, here the double's *silence* is
laundered by production's own error handling into a false negative that reads like data.

**What to do.** When production calls a collaborator behind a broad `except`, a stub double
for that collaborator cannot report drift. Subclass the real collaborator instead and
override only what you need to observe:

```python
class _CharacterHandoffStore(ConsoleChatStore):      # real store, persistence=None
    def append_message(self, session_id, *, role, content, persist=False, **kwargs):
        self.identity_at_append = {...}              # observe
        return super().append_message(...)           # production behaviour intact
```

The greeting text then comes from production's own template expansion, so the assertion
`== ["Hello User, I am Elara."]` is a live end-to-end claim (mutation-checked: passing
`global_default="Zed"` turns it red) instead of a re-implementation of the thing under test.

## Two tests failing on the SAME missing class can have opposite causes — and the feature commit's own test diff is the authority (TASK-15121, 2026-08-11)

**Incident.** Two tests in `Tests/UI/test_console_native_chat_flow.py` went red on dev with
what looked like one symptom: the Console send button no longer carried
`console-send-blocked`. The obvious reading was a CSS-vocabulary rename — follow it in both
tests and move on. Both readings were wrong in different directions:

- `test_console_composer_stop_is_subdued_when_idle` mid-stream with an EMPTY draft: the
  button was still genuinely `disabled`, just for a different reason
  (`console-send-inactive`, the empty-draft gate) than the one pinned.
- `test_console_duplicate_send_during_stream_does_not_break_stop_control` mid-stream with a
  draft loaded: the button was `disabled is False` and `console-send-ready`. The single
  `composer.load_draft("second send")` between the two tests is the whole difference.

Had both been "fixed" as a rename, the second would have kept asserting a control was
unavailable when it is now deliberately available — the exact class of silent claim
task-14920 lost a real bug behind for four days.

Neither the class name nor the production code said which reading was right. What settled
it was the **test diff of the commit that caused it**: `git log -S 'send_blocked = not
queue_presentation.send_enabled'` named `14cc326e4` ("feat(console): add visible prompt
queue"), and that commit's own diff to a SIBLING test file
(`Tests/UI/test_console_send_disabled_state.py`) rewrote the same assertions to the new
contract and renamed a test from `..._while_run_blocked_still_shows_feedback` to
`..._queues_draft_behind_accepted_run`. The author changed the contract deliberately
(ADR-098: "once accepted, the normal `Send` action becomes `Queue`") and updated one test
file, not two.

**What to do.** On a post-merge test failure that looks cosmetic, `git log -S` the assertion's
symbol, then read that commit's *test* diff before its production diff — an author who meant
the change usually left the new contract written out somewhere. And classify each failure
separately even when the symptom string is identical: same missing class, different inputs,
different truth. Where a pinned behaviour really was removed, say so in the test and pin what
replaced it (here: the duplicate send must still not start a second run, must land in the
bounded queue, and must not break Stop) rather than deleting the assertion.

---

## What is LISTENING on your machine can change what the test suite does (2026-08-11)

**The trap.** A suite can be environment-dependent in a direction nobody checks: not a
missing dependency, but an *extra* process. If production code probes a hardcoded
localhost port, then whether a developer happens to be running a local server decides
which branch the tests take — and the difference is invisible, because the escape's
failure mode is *success*.

**What happened.** task-15111. `Tests/UI`'s Console suites were opening real TCP
connections to `127.0.0.1:8080` and `127.0.0.1:11434` on every test that mounted
`ChatScreen` with an unconfigured provider. Mechanism: a blocking setup card starts
`_maybe_start_console_local_discovery` → `discover_local_servers`, whose candidate list
*always* leads with those two well-known defaults regardless of config, and
`probe_models_endpoint` builds a real `httpx.AsyncClient` when none is injected. A
record-only socket shim logged **386 connect attempts across 20 test files in the first
12% of `Tests/UI` alone** — on a machine that happened to have an `audiocpp` server bound
to 8080. Exactly one test in the suite had ever stubbed the `console_local_server_discovery`
seam; every other Console test fell through to the network.

Two things made it worse than "a stray GET":

- **The escape was self-concealing.** `_get_models_payload` ends in
  `except Exception: return None, "No models endpoint..."`. Blocking the socket, raising,
  timing out and answering all look identical from outside. A guard that only *raises* is
  therefore not enough — the guard has to **record** the attempt and something has to
  assert on the record, or the code under test simply eats it.
- **It could have POSTed.** `_configure_native_ready_console` points the Console at
  `http://127.0.0.1:9099` and several tests then drive a REAL send through
  `ConsoleProviderGateway`. On CI nothing listens, `_is_reachable`'s `GET /health` fails,
  and the send stops — so the suite looked read-only. Standing up a stand-in server on
  9099 and re-running one such test showed it going on to send **two POSTs to
  `/v1/chat/completions`** (streaming, then the non-streaming fallback) carrying the
  test's prompt. With a real llama.cpp on that port, `pytest` would have driven inference
  on the developer's server.

**What to do.** Default-deny sockets in the test configuration (`Tests/network_guard.py`,
installed at conftest *import* time so collection and post-test worker threads are covered
too), with an explicit `@pytest.mark.allow_network` opt-in, and fix the seams that build a
real client so the guard is a backstop rather than the mechanism. And when you want to know
what a suite would do against a live endpoint, do not reason about it: **bind a stand-in
server on the port and read what it receives.** Recording connects tells you a socket
opened; recording requests tells you the verb, the path and the body — which is the
difference between "reads something" and "writes to your server".

**Windows follow-up (TASK-15100).** The first task after the guard landed exposed a
platform boundary the original evidence missed: on Windows, Python 3.12's Proactor event
loop creates its self-pipe with the TCP fallback for `socket.socketpair()`, connecting to
an ephemeral `127.0.0.1` port. Because the guard is installed at conftest import time and
defaults to denied, `pytest-asyncio` could not even construct the event-loop fixture; every
async test failed in setup before the autouse fixture (and therefore before an
`allow_network` marker) could change the guard state. TASK-15100's focused local suites
produced twelve setup/teardown errors without executing one test. The task did not weaken
the shared guard as an unrelated drive-by change; its local-only verification process
temporarily emptied the guard's family set inside that pytest process, with every selected
test using SQLite or injected fakes.

**What to do on Windows.** A process-wide egress guard must distinguish the event loop's
loopback self-pipe from application egress *before* async fixtures are created (or use a
guarding layer that does not intercept the runtime's wakeup channel). Do not paper over
the issue by marking broad UI suites `allow_network`: that restores the exact application
escape the guard exists to detect. TASK-15458 replaced the temporary family-set workaround
with ADR-058's thread-local, dynamic `socketpair()` exemption: only the calling thread is
permitted while the captured real socketpair call is active, and `finally` restores nested
depth on success or error. Literal Windows commands
`python -m pytest Tests/test_network_guard.py -q`,
`python -m pytest Tests/Library/test_library_media_content.py -q`, and their combined form
passed without changing `_INET_FAMILIES`; focused tests also proved same-thread direct
egress stays blocked and recorded after an exception, while concurrent-thread egress stays
blocked and recorded during socketpair. Keep live/external clients stubbed.

---

## A capability decision is only as pinned as the final adapter kwargs (2026-08-11)

**The trap.** Checking a resolved provider/model/endpoint and then attaching a
provider-specific request feature does not prove that the checked endpoint is the one the
adapter will call. A lower layer may reload configuration or fall back to its own endpoint
after the capability decision has already been made.

**What happened.** task-15263 added strict JSON Schema enforcement for the visual
compaction evaluator's documented OpenAI GPT-4o routes. The initial implementation checked
`ConsoleProviderResolution.base_url`, but the prepared-request dispatcher did not forward
OpenAI's resolved base URL; `chat_with_openai` could therefore reload a configured endpoint
later. The report could have claimed `provider_json_schema` based on the official endpoint
while the final call went to a custom OpenAI-compatible proxy. Self-review caught the gap
before the PR. The evaluator-only prepared request now pins the checked endpoint into the
final adapter kwargs, and a test asserts both the immutable response format and exact
`api_base_url`. A mutation that removed the endpoint guard made the custom-proxy case fail.

**What to do.** For provider capability gates, test the final dispatched kwargs, not only
the resolver result or an intermediate request object. If a lower adapter can reload config,
make the capability-bearing request pin the checked endpoint (without changing unrelated
callers), and mutation-test the unsupported route so fallback labeling cannot silently
become an unsupported capability claim.

## A race a live replay cannot trigger is often a STATE you can construct deterministically (TASK-14903, 2026-08-10)

**Incident.** A live click killed the whole app once — `AttributeError:
'NoneType' object has no attribute 'region'` inside Textual's
`Screen._forward_event` text-selection begin, ~1s after a terminal resize.
THREE live replay attempts (same screen, same resize, same click) never
triggered it again, and the originating task shipped with the crash merely
noted. Task-14903 reproduced it 100% deterministically on the first attempt —
not by replaying the timing, but by reading the framework source to name the
intermediate state the race passes through (widget pruned from the DOM, parent
already `None`, compositor's cached map not yet reflowed) and then
constructing that state directly at the seam: `await widget.remove()` with no
subsequent pause (prune complete, reflow pending), then the MouseDown driven
through `App.on_event`, the exact call the live crash traversed. The
"irreproducible" race was a two-line setup once expressed as a state instead
of a schedule.

**What to do.** When a race defies replay, stop replaying. Read the code that
crashed until you can name the exact intermediate state the timing window
produces (here: three facts — `parent is None`, stale compositor map, event
dispatched between them), then build THAT state through the narrowest public
seams available and drive the same entry point the production path uses. A
reproduction that constructs the state is strictly better than one that races
the clock: it is deterministic, it documents the mechanism in its
preconditions (each setup line asserts one fact of the attribution), and it
pins the upstream behavior — if a dependency bump fixes the bug, the
state-construction test fails loudly and tells you the workaround can be
retired, which no timing-based replay could ever do.
## Moving a config read onto the app-config snapshot silently DEFAULTS it in every `_build_test_app` test — including passing ones (TASK-15210, 2026-08-11)

`ChatScreen._maybe_auto_retrieve_for_send` used to read the auto-RAG toggle live via
`get_cli_setting("chat_defaults", "rag_auto_retrieve_on_send")`. Task-14803 (commit
`5be9e6a04`) moved that read onto the frozen per-turn `ConsoleTurnExecutionContext`, whose
`rag_defaults` are built from `app.app_config`. Both sources agree in the shipping app.
They do not agree in `Tests/UI`.

`Tests/UI/app_factory._build_test_app` patches `tldw_chatbook.app.load_settings` to return
a synthetic `{"tldw_api": ..., "first_run": ...}` — **no `[chat_defaults]`, no `[console]`**.
Production then behaves *correctly*: `_provider_readiness_app_config` only re-sources from
`load_settings()` when the snapshot carries the `general`/`logging` markers only a real
disk load emits, and this one does not, so it hands back the synthetic dict verbatim. Net
effect: `save_setting_to_cli_config(...)` still writes the toggle, `get_cli_setting` still
reads it True, and the code under test sees False.

Measured, not inferred: instrumenting one mounted test printed
`get_cli_setting=True` / `app_config chat_defaults.rag_auto_retrieve_on_send='MISSING'` /
`resolved ctx rag_defaults={'auto_retrieve_on_send': False, ...}` in the same run. The live
app assigns `self.app_config = load_settings()` (app.py), whose result carries both the
toggle and both markers — so the shipping path was fine and only the harness was blind.

**The part that cost the most.** One test went red and was triaged. Its sibling,
`test_send_proceeds_when_auto_retrieve_fails`, stayed GREEN — because with retrieval never
firing, the exploding backend it installs is never called and the test degenerates into
"an ordinary send works". A moved read does not announce itself by failing; it can just as
easily hollow out a passing test, and nothing in a green run points at it.

**What to do.** When you move a read from a live settings accessor onto a snapshot,
grep the tests that *enable* that setting and check they enable it through the new source —
a `save_setting_to_cli_config` + `_build_test_app` pair no longer reaches the code. Give
the mounted test the app's real shape (`app.app_config = load_settings()`, exactly what
`app.py` does) rather than teaching the product to fall back. And any test whose subject is
"X still works when Y fails" should assert **that Y was actually attempted** — here,
`exploding_search.await_count == 1` — or it cannot tell "handled" from "never happened".

## Textual component CSS must be proven on the concrete widget class (TASK-1990.1, 2026-08-11)

**Incident.** TASK-1990.1 extended Textual's Markdown block classes through a
Python mixin and declared the new inline component names on that mixin. Pure
parser tests passed because the expected component spans were present, and the
TCSS bundle compiled, but a real compositor test painted narration, speech,
action, and emphasis identically. Textual's class construction had populated
the concrete block's internal component registry before the ordinary mixin
attribute could affect it; type-oriented TCSS also needed a stable class hook
because the rendered blocks were concrete subclasses.

**What to do.** When adding Textual component styles through subclasses, declare
`COMPONENT_CLASSES` on every concrete widget class and give the widget a stable
CSS class selector. Then verify both `get_component_rich_style()` and final
compositor segments. Span-level or stylesheet-compilation tests alone do not
prove that Textual registered or painted a component.

## A report's "already handled / out of scope" is an UNTESTED CLAIM (supervisor-fleet PR 3a-1, 2026-08-11)

**Incident.** PR 3a-1 Task 5 gave a background sub-agent its own wall-clock ceiling and
reported the containment story in two halves. It **mutation-tested the TIME half** and was
right. It stated the **COUNT half** — "aggregate live children are still bounded by
`[agents] max_live_subagents`" — from *reading the code next to it*, and wrote that into the
report and a docstring as settled. The review then proved by **execution** that it was false:
two consecutive `run_turn` calls each spawning two blocking children ran **4 simultaneously
against a configured cap of 2**, because `run_turn` built a brand-new `FleetCoordinator` every
call, and Console built a brand-new `AgentService` per `run_reply` and injected no coordinator
at all. Before this PR the bug was structurally impossible (children could not outlive their
turn), so the claim had been true right up until the change that broke it — which is exactly
the shape that survives a careful read.

That claim had already propagated: the plan's own seam map said `FleetCoordinator` was
"already reusable, no reset, never pruned", Task 5 relied on it, and the fix (a
per-conversation coordinator owned by the bridge) had to be a whole extra task. The retraction
is now pinned by `test_live_children_are_not_capped_across_turns` so it cannot be silently
re-assumed.

**This was the third instance in one programme**, all the same shape — a confident reading
stated as a finding:

1. A "vanishing row" window a fix was ordered for; measurement showed it was **sub-millisecond
   against a 200ms poll**, i.e. unobservable.
2. A run-log "closed writer" diagnosis, **wrong twice**: the records were being *misfiled into
   the next turn's tree* (not dropped), and `close()` was never a barrier at all — it fsyncs
   and returns without clearing `_active`, and `append()` opens its own handle per record.
3. This one.

**What to do.** When a report says a risk is *already handled*, *unreachable*, *out of scope*,
or *unchanged by this task*, treat it as a hypothesis until a test executes it. Write the test
that would go red if it were false — and prefer probing with a recording double over reading
the call path, because the two failures above were both found by a probe and missed by a read.
A dismissal is a claim about behavior, and behavior is the one thing reading cannot establish.
---

## A mechanism sentence is an ORACLE — read your prose against your own tables (TASK-15020, 2026-08-11)

**Incident, twice in one arc, and the second time the prose had already
shipped.**

1. **Paper arithmetic the code refutes by 1 ULP.** Task 6 measured the RAG
   eval's scoped category flipping 0.000 -> 1.000 and wrote the mechanism
   into `golden.toml`, `README.md` and a test comment: the FTS-only row
   "exactly ties" the semantic leg's rank 9, so the tie-break convention
   decides placement. False as the shipped code evaluates it.
   `reciprocal_rank_fusion` computes `(1.0 - alpha) * fts_rrf`, and
   `1.0 - 0.7` is `0.30000000000000004`, so the FTS-only row scores exactly
   `0.05` against the semantic row's `0.7/14 = 0.049999999999999996` — a
   **strict win by 6.94e-18**. The tie-break never runs. The paper form
   `0.3 * (1/6)` IS bit-identical to the semantic value, which is where the
   phantom tie came from. The *same arc's predecessor* (the weighting arc)
   had already learned a 1-ULP lesson; it recurred one arc later, in prose.
2. **A claim contradicted by a table in the same document.** The same
   section said the class is "FTS-only" and would read 0.000 at the old
   `rrf_k=60`. Re-running the counterfactual gave **0.286**: two of the
   seven targets sit at vector rank 12 and 20, inside the over-fetched
   pool, and reach rank 1 at `rrf_k=60`. The author's own rank vector
   `(3,4,9,9,9,9,9)` could never have come from FTS-only arithmetic — the
   contradiction was already printed above the sentence.
3. **A filed task pointing its own fix at the wrong lever.** Task 7 filed
   TASK-15400 blaming the keyword leg's silence on function words in an
   implicit-AND MATCH. Measured across all 60 golden queries: a
   stopword-trimmed AND rescues **1 of 40**; OR-of-tokens rescues **34**.
   The dominant cause is AND-strictness over CONTENT words — visible in the
   author's own token table, in the same report, unread against the
   author's own prose. A filed task is an oracle for whoever implements it,
   and as filed it would have sent them at a 1-in-40 lever.

**What to do.** Treat any sentence asserting a MECHANISM — in a docstring,
a fixture comment, a README, a test comment, or a filed task's description
— as an assertion that must be checked against the running system, at the
same standard as an `assert`. Specifically: never state a numeric mechanism
in paper arithmetic; **read the provenance the engine already records**
(here, `metadata["hybrid_fusion"]` carried `fts_rank`, `vector_rank` and
the fused scores all along). And before shipping an explanation, read it
back against the tables in your own document — in all three incidents the
refuting data was already on the page. Distinct from the stale-prose trap
(see "Retuning a numeric constant obliges you to grep its LITERAL VALUES"):
that prose went stale, this prose was wrong when written.

---

## A declared divergence no test can distinguish from its own removal is a comment, not a decision (TASK-15020, 2026-08-11)

**Incident.** Task 8 made the Library RAG window's depth follow the active
profile, and deliberately kept one difference from the Console seam that
now shares its resolution: the window clamps a >50 profile down to
`LIBRARY_RAG_TOP_K_MAX`, while Console stays uncapped. It was stated in the
report, in the code comment, and covered by a test — of the *clamped* arm
only. The reviewer mutated the SHARED seam to clamp unconditionally
(`min(value, LIBRARY_RAG_TOP_K_MAX)` in `library_rag_profile_top_k`),
erasing the divergence outright. **199 tests stayed green.** A difference
the author had chosen on purpose could be deleted by anyone, at any time,
with the whole suite agreeing. The fix was one test asserting BOTH arms
together — profile 100 gives Console 100 and the window 50 — after which
the reviewer's exact mutation reds precisely that test (1 failed / 153
passed).

**What to do.** When you deliberately make two call sites behave
differently, the pin is not "test the interesting arm" — it is **one test
that asserts the pair**, so the assertion states the DIFFERENCE rather than
one of its sides. Then mutate toward *sameness* (make both arms agree) and
confirm red; the usual mutation habit of breaking the guarded behaviour
misses this class entirely, because unifying two arms breaks neither arm's
own test. Applies to any intentional asymmetry: a clamp on one path, a
stricter timeout for one caller, a feature gated in one surface and not
another. Sibling of "Mutation-test every guard you add" and "A guard test
must be PROVEN to discriminate", with the twist that here the thing left
unpinned was a DESIGN DECISION, not a behaviour.
## Holding ONE database instance turns an intermittent schema-cache race into a permanent one (TASK-15463, 2026-08-11)

Caching `SubscriptionsDB` instead of rebuilding it per service call (a ~52-statement
`executescript` per call, ~85x the cost of a held instance) made two Watchlists UI tests
fail deterministically with `sqlite3.OperationalError: no such table: subscription_items`
— on a table that `sqlite_master`, queried microseconds later on the SAME connection,
listed. An immediate retry of the identical UPDATE succeeded.

A timestamped probe over `SubscriptionsDB.__init__` / `_get_connection` explained it:

```
0.0614  INIT app instance      (main thread)
0.4502  INIT second instance   (FTS-backfill worker thread) -- _initialize_schema
0.4511  CONN opened on the app instance, by an asyncio.to_thread worker   <-- inside that window
0.6884  second instance's _initialize_schema finishes (238 ms)
3.0311  that worker's UPDATE: "no such table: subscription_items"
```

A connection opened while another connection is rewriting the schema caches a view without
the tables being rewritten. With a database rebuilt per call, that view lived for one call
and the next call built a fresh connection — so the defect surfaced only as an
*intermittent* flake, already documented in `Tests/UI/test_watchlists_inspector.py` as
"self-healed on an immediate retry". Hold the instance and the poisoned connection lives as
long as the thread does: every write that lands on it fails.

The fix was to remove the second `_initialize_schema` (the FTS-backfill worker now shares
the app's one instance — thread-local connections are exactly what makes sharing the
*instance* safe), not to add a retry.

**What to do.** Before caching any long-lived DB handle, find every OTHER construction of
that DB class against the same file — each one re-runs schema setup, and any connection
opened during it can be born stale. And treat a documented "it self-heals on retry" flake
as a live bug with a shortened fuse: it is one held connection away from being permanent.
Probing the mechanism cost ~20 minutes (init/connection timeline + one retry inside the
failing call); guessing at "sqlite locking" would have cost far more and fixed nothing.

## A "we tried this and it broke X" comment is dated evidence, not a standing constraint (TASK-15454, 2026-08-11)

`ConsoleWorkspaceContextTray.sync_state` carried a long, careful comment (TASK-251,
July) explaining that the obvious `if state == self.state: return` guard had been
implemented, had broken click targeting on grouped browser rows, and had been
withdrawn — naming the two tests that failed. A separate test pinned the
unconditional recompose so nobody could quietly reintroduce it. Every downstream
comment in the file, in `chat_screen.py`, and in two test modules repeated the
conclusion: "an equality guard here is unsafe".

Re-guarding it started by reproducing that: apply the naive guard, run the two named
tests. **Both passed.** Widening to the whole 309-test `test_console_native_chat_flow.py`
plus `test_console_rail_sections.py` produced only the two tick-gating pins (which pin
the unconditional recompose itself) and one failure that also fails at HEAD. The
regression had been dissolved by later, unrelated work — most plausibly TASK-1900's
non-echoing search input and TASK-1191's collapse of the fit-pass from three deferred
hops to one.

Two things follow, and the second matters more than the first:

1. **Re-run the witness before designing around it.** Fifteen minutes of `git log -S`
   plus two test invocations turned "this is forbidden" into "this was forbidden in
   July, for a reason that no longer exists". Without that, the natural move is to
   design elaborately around a constraint that is not there — or, worse, to accept the
   comment and skip the work entirely.
2. **A dissolved regression is not a licence to do the naive thing.** The comment's
   *diagnosis* outlived its symptom, and it was the valuable part: state equality
   answers "does this widget REMEMBER this state", which is a different question from
   "is this widget SHOWING it". Those two came apart once and can come apart again.
   The guard that shipped therefore checks the second question directly — `compose()`
   records the row ids/keys it built; the guard compares that against the rows read
   back out of the live DOM — and both directions are mutation-tested (`return state
   == self.state` reds the safety tests; `return False` reds the skip tests).

**What to do.** Treat every "deliberately reverted / do not reintroduce" comment as an
experiment with a date on it. Re-run its named witnesses first; record the result in
the task either way. Then keep the diagnosis even when the symptom is gone.
## A synthetic test config lets tests pin states no user can reach (TASK-15270, 2026-08-11)

**The trap.** A test-app factory that hands the app a small hand-written config is not a
neutral simplification. Every default the real config file carries is *absent*, so the
code under test takes fallback branches, and assertions written against those branches
look like product contracts while pinning states the shipped template never produces.

**What happened.** `Tests/UI/app_factory._build_test_app` patched `load_settings` to a
three-key dict. `ChatScreen._provider_readiness_app_config` re-sources from
`load_settings()` only when the snapshot it was handed carries the sections a real load
always emits (`_CONSOLE_LIVE_CONFIG_MARKER_SECTIONS`: `general`, `logging`) — a
deliberate guard so an injected test config is never overwritten by the developer's real
one. The synthetic dict carried neither marker and no `[chat_defaults]`/`[console]`
section, so every mounted Console test read a `ConsoleTurnExecutionContext` frozen at
defaults. `test_send_proceeds_when_auto_retrieve_fails` was green for two months without
once calling the exploding backend it existed to exercise (task-15210).

Sourcing the factory's config from the real (per-test sandboxed) `load_settings()` turned
**31 green tests red across 6,016**, and the interesting part is *why* — almost none were
product regressions:

- **Arranged through a seam production does not read.** `console_image_view.
  _chat_images_config` prefers the raw TOML nested under `COMPREHENSIVE_CONFIG_RAW`
  whenever the snapshot has it. Four avatar tests set `app_config["chat"]` instead, which
  the shipping app would have ignored — they were pinning the fallback shape.
- **Passing on a fallback the template removes.** Three llama.cpp URL tests reached that
  branch only via `provider_config_key(...) or "llama_cpp"`; the template ships
  `[chat_defaults] provider = "OpenAI"`, so the fallback never fires for a real user.
- **Absence as arrangement.** Two "cli config fallback" tests asserted `"library" not in
  app_config` rather than arranging the absence they needed.
- **Copy for an unreachable state.** Several first-run/UAT replays assert "Choose
  provider" — the branch for *no provider selected*. A genuinely fresh install has
  `provider = "OpenAI"` and no key, so the product says "Set up provider". The tests
  described a clean run no user has.

**What to do.** Give the test app the same config source the app uses, sandboxed per
test (the root conftest already re-points `TLDW_CONFIG_PATH`/`HOME`/`XDG_*`), so what a
test persists is what the app reads. And when a test needs a state — no provider, no
`[library]` section, a feature off — **arrange it explicitly**; a state you inherited
from an empty fixture is a state you never chose, and the day the fixture gets honest you
cannot tell which of your assertions were ever real.

---

## A cancellation flag does not make check-and-commit atomic

**TASK-3401.20, 2026-08-10.** The first generated-video teardown fix checked a
screen-owned cancellation flag before publishing a managed file. Review found a
real gap between that check and the filesystem commit: unmount could win in the
middle, close the staged stream, and still leave a committed file without durable
message metadata. A second version shielded `asyncio.to_thread()`, but cancellation
of the awaiting coroutine did not stop the executor thread; releasing ownership in
the async `finally` could still close bytes the thread was using. Timing-only tests
missed both defects because they never proved which side had reached lock
acquisition or the final commit boundary.

**What to do.** Make the state transition linearizable: share one lock across the
final active check and commit, and cancel under that same lock. When blocking work
runs through `asyncio.to_thread()`, retain an explicit executor task and keep resource
ownership until that task actually finishes; coroutine cancellation alone is not
completion. Put durable commit-winning metadata finalization inside the shielded
unit, but leave stale-screen UI refresh outside it and normally cancellable. Tests
must use events or instrumented locks to force both cancel-wins and commit-wins
orders, including cancellation after commit but before metadata append; a sleep and
an assertion that “nothing happened yet” are not evidence of ordering.

## Returning from a Textual handler does not make its detached child cancellation-safe

**TASK-3402, 2026-08-11.** An H3 image edit originally awaited its whole operation
inside the real `Button.Pressed` handler. That kept the screen's MessagePump occupied,
so the visible Stop press could not run. Moving the operation into an app-owned task
fixed Stop responsiveness, but the old “outer cancellation drains success” test kept
passing without ever cancelling anything: it awaited the now-immediate handler return,
then released and awaited the detached operation normally. Directly cancelling the
actual operation task exposed the gap—its `asyncio.to_thread()` runner continued while
the owning coroutine removed the registry entry before durable settlement. App
shutdown had the same problem because Textual did not drain arbitrary tasks created
with `asyncio.create_task()`.

**What to do.** For a detached, app-owned operation, test and own cancellation at the
detached task—not at a caller that has already returned. The owned task must shield the
real runner, translate cancellation into the exact shared event, await the runner to
settlement, and only then re-raise. The application shutdown path must explicitly
cancel and drain those registered tasks before tearing down screens or persistence.
Use barriers to prove both success-wins (durable append exactly once) and
cancellation-wins (no card), and mutation-check that the test fails if shielding,
event propagation, or shutdown draining is removed.

---

## Keyboard focus does not prove a nested compact control is visible (TASK-15506, 2026-08-11)

**Incident.** TASK-15506 moved File Notes push provenance into a collapsed
`Collapsible` inside the push workflow's `VerticalScroll`. At 40x20, expanding
the disclosure and pressing Tab moved focus to the nested Endpoint details
button, so a focus-only regression passed. The button was still absent from
`Screen._compositor.visible_widgets`: Textual had scrolled only far enough to
show the disclosure's earlier content, leaving the focused action below the
fixed footer. A normal non-animated `scroll_visible()` call still stopped
short. Scrolling the exact focused descendant with `force=True` and
`immediate=True` brought it into the compositor deterministically.

**What to do.** For controls nested inside disclosures within a compact scroll
owner, assert both `has_focus` and compositor visibility. If framework focus
navigation leaves the control outside the viewport, handle descendant focus
at the narrow owning component and call `scroll_visible(animate=False,
force=True, immediate=True)` on the exact control. Do not infer reachability
from focus state or a nonzero layout region alone.

## An "indexed" query can still scan the table the index exists to avoid — and the plan assertion can miss it (TASK-15469, 2026-08-11)

TASK-15469 replaced a `metadata LIKE '%active_dictionaries%'` full scan of
`conversations` with a lookup over a trigger-maintained index table. The new query
joined the index table to `conversations`, and the test asserted
`"SCAN conversations" not in plan`. It passed. It proved nothing:

* The query aliases the table (`conversations AS conversation`), and
  `EXPLAIN QUERY PLAN` prints the **alias**, so the plan said `SCAN conversation` —
  which the assertion's literal `"SCAN conversations"` never matches. The test would
  have passed with a plan made entirely of full scans.
* And there really was one. SQLite's planner chose `conversations` as the outer loop
  of the second branch (`SCAN conversation` + a covering-index probe per row): a full
  scan of the very table the index was built to stop reading. Only the FIRST branch
  used the new index; nothing in the assertion covered the second.

It surfaced from a **timing** arm, not from the plan test: on a 10,000-conversation
DB, "used-by for a dictionary attached to nothing" measured 2.07 ms when it should
have been unmeasurable. `CROSS JOIN` (which pins the left table as the outer loop and
disables that particular join reordering) took the same arm to 0.00 ms and the whole
click's DB work from 7.8 ms to 0.54 ms. The plan test now asserts on the alias prefix,
asserts `conversations` is reached only by `SEARCH ... USING INDEX
sqlite_autoindex_conversations_1`, and asserts the plan is non-empty.

One more planner subtlety worth knowing: this project never runs `ANALYZE`, so the
planner works from default row-count estimates and reliably prefers the index. Running
`ANALYZE` on a small dev database flips it back to `SCAN` on the tiny index table —
so a plan captured on a hand-seeded 50-row fixture with `ANALYZE` is not the plan
production runs.

**What to do.** When the claim is "no full-table scan", (1) grep the plan for the
identifier the query actually uses — the alias, not the table name — and assert
positively on what SHOULD happen (`SEARCH ... USING INDEX <name>`), not only
negatively on what should not; (2) assert the plan is non-empty, or an empty result
satisfies every "not in" assertion; (3) check EVERY branch of a compound query; and
(4) keep one timing arm whose expected value is ~zero (a lookup with no hits) — a
scan cannot hide from that, and it is what caught this one.

---

## An absent catalog surface still needs its synthetic identity reserved

**TASK-13216, 2026-08-12.** The replacement Console task tools were correctly absent
from the external MCP and Hub inventories, and every literal-name absence test passed.
Review still found that an external MCP profile could use the reserved `__local__`
profile ID. Its projected Hub key then collided with the synthetic workspace provider's
`local:__local__` permission identity. The same review found a current guide describing
"session-todo tools" without any literal `todo_write` or replacement name, so the stale
name scan also reported clean while the documented inventory was wrong.

**What to do.** For a synthetic catalog namespace, reserve its normalized identity at
every ingress and projection seam: save/import, load, runtime composition, and raw
catalog conversion. Prove the derived permission key cannot be forged, while pinning
nearby valid and case-distinct IDs. For negative documentation contracts, pair literal
stale-name scans with an exact positive sentence describing the current boundary;
synonyms can preserve a stale claim without preserving any searched token.

---

## A property that holds "by construction" holds for the COMPONENT — measure it at the MERGE the requirement actually names (TASK-15400, 2026-08-12)

**Incident.** The MATCH-construction arc pre-registered a hard constraint:
whatever the keyword leg's expression becomes, the golden set's one
vector-blind fixture (`kw-plant-maintenance-record`, which only the keyword
path can find) must keep its hybrid rescue. The spec then argued that the
favourite candidate — `and_then_or`, "AND first, OR only when the AND
returns nothing" — satisfied it **by construction**: *a nonempty AND never
falls back*, so the fixture's own row can never change.

That premise is TRUE, and the sweep verified it directly: the notes
sub-leg's row for that query was still there, still stamped `and`, still
its sub-leg's rank 1. **The conclusion was false anyway.** Measured, the
rescue was GONE — the fixture dropped out of the fused top-10 entirely.

The guarantee was about a **sub-leg**; the constraint was about the **leg**.
`RAGService._keyword_search` merges its four source sub-legs with
`interleave_rankings` — a round-robin over sub-leg position. The media and
conversations sub-legs returned zero AND rows for that query, fell back to
OR, and injected ten rows each; media is first in the round-robin, so the
untouched notes row moved from leg rank 1 to leg rank **2**. Fusion consumes
*leg* rank: `0.3/6 = 0.0500` became `0.3/7 = 0.0429`, which loses to the
vector rank-11 row's `0.04375`. Nothing about the fixture's own row changed;
everything about its position did. The same displacement then decomposed a
whole category exactly — scoped recall 1.000 → 0.429 is the four
note-targeted scoped queries falling behind a media fallback row while the
three media-targeted ones keep rank 1 (3/7, the measured cell to the digit).

**Why it was caught.** Only because the constraint's probe was written at
the **output** — "is this document in the FUSED top-10" — rather than at the
component the guarantee described. A probe asserting "the notes sub-leg
still returns its AND row at rank 1" would have passed, and the arc would
have shipped a construction that silently deleted the one rescue the whole
fixture exists to detect.

**What to do.** When a design argues a property holds "by construction",
write down two scopes before believing it: **what object the guarantee is
about**, and **what object the requirement is about**. If they differ by
even one level of composition — sub-leg vs leg, row vs list, component vs
merged output, one writer vs the aggregate — the argument is about a
different thing than the requirement and proves nothing about it. Put the
acceptance probe at the level the requirement names.

Two corollaries worth carrying:

- **Any positional merge (round-robin, concatenation, fixed source order)
  makes every component's rank a function of every OTHER component's row
  COUNT.** A change that only ADDS rows in one place still re-ranks
  everything downstream. Treat "this change is additive" as a claim about
  the component, never about the merged list.
- **Necessary is not sufficient, and the margin is measurable.** Re-fusing
  the same run with the fixture restored to leg rank 1 and nothing else
  changed put it back at **slot 10 of 10** — so even fixing the merge
  rescues it with zero headroom. When you find the blocking mechanism,
  measure what fixing it actually buys before scoping the follow-up around
  it (this one became TASK-15700 with that number in its description).

---

## An infrastructure "agent stopped" report is a claim, not evidence

**PR 3a-2 Task 5, 2026-08-13.** The harness twice reported the Task 5
implementation agent stopped ("stopped by the user"; after a Claude Code
process restart it also refused to resume the agent — "won't be resumed").
The worktree was verified clean at the briefed HEAD, twice, and a fresh
agent was dispatched into `.worktrees/fleet-pr3a2` with the same brief.
Both reports described an agent that was never stopped: the pre-restart
Claude Code process had survived as an orphan — `ps` showed TWO
`claude --resume <same-session-id>` processes — and its subagent kept
editing, committing, and pushing. The fresh agent adopted a commit that
sat one beyond the briefed HEAD, then collided mid-edit: "string not
found" on a file the supposedly-stopped agent had rewritten seconds
earlier (mtime observed under a minute old). It halted itself and wrote
an incident file instead of the report
(`.superpowers/sdd/2026-08-13-supervisor-fleet-pr3a2-autowake/task-5-incident-shared-worktree.md`).

The same orphan was still working at Task 6 close-out, HOURS later: three
fully-formed backlog task files appeared untracked in the worktree between
one of the Task 6 agent's commands and the next — filed under the exact
ids that agent's own sweep had just derived — followed four minutes later
by a commit made on top of the Task 6 agent's fresh commits. Two agents
were doing the same close-out in one worktree, neither told about the
other, because a "stopped" report had been believed twice.

**Why the clean-tree check wasn't enough.** "Verified clean at HEAD X" is
a statement about one instant. An agent alternates minutes-long quiet
stretches (gate batteries, provider calls) with bursts of writes, so any
point-in-time check taken during a quiet stretch passes.

**What to do.** Treat "the agent was stopped" as a claim to verify, never
a premise. Before dispatching into a worktree a reportedly-stopped agent
occupied, verify quiescence by OBSERVATION over a real interval — stable
`git log`, stable `git --no-optional-locks status --porcelain`, no fresh
file mtimes, held for minutes, not sampled once — and check `ps` for a
second `claude --resume <session-id>` process, the smoking gun in both
sightings. An OS process outlives the harness's account of it; only the
OS can tell you it is gone.

---

## A `0.00s` pytest summary is a usage error wearing a pass's clothes

**PR 3a-2 Task 5 gate verification, 2026-08-13.** A gate run passed
pytest a nonexistent path — `Tests/Chat/test_console_mcp_approval.py`;
the file lives in `Tests/UI/` — so pytest exited 4 after collecting
nothing. The habitual `| tail` read showed only "1 warning in 0.00s",
which was nearly recorded as an empty-but-fine run. The one line that
mattered — `ERROR: file or directory not found` — was at the HEAD of the
output, above everything tail kept. The same shape recurred within hours
in the same PR: a background gate run launched with a relative
`.venv/bin/python` that does not exist in that worktree "completed with
exit code 0" — the trailing `| tail -3` laundered the interpreter's
failure into the pipeline's success — and only READING the output file
revealed `no such file or directory: .venv/bin/python`. No tests had run
in either case, and both runs wore a green-looking coat.

**What to do.** A `0.00s` (or near-instant) pytest summary means nothing
ran: treat it as a FAILED gate, never a fast pass. Read the HEAD of the
output — usage errors print before the summary line, and exit codes
piped through `tail` are the pipe's, not pytest's. A gate passes only on
a READ, nonzero passed-count that matches the expected number; "no tests
ran", a count you didn't read, and a summary too fast to be real are all
the same verdict.
## A truncated pytest diff is not the diff (task-15512)

**Incident.** A failing `assert service.calls == [...]` printed its summary line
as `assert [{'include_ci...tions'), ...}] == [{'include_ci...tions'), ...}]`,
followed by one `At index 0 diff:` line that pytest itself had cut mid-value. I
read the visible fragment as a *scope* change and wrote that into the task file
as the diagnosis. It was wrong: the actual delta was `top_k` (5 vs 15), which
sat past the truncation point. The wrong diagnosis then travelled -- into a task
another person would have picked up, pointing them at "a search silently
widening its scope", which is a much more alarming and entirely fictional bug.

**Rule.** When a collection assertion fails, do not diagnose from the summary
line. Re-run that single test and read the full comparison, or print the two
values. The `...` in pytest's output is not an ellipsis for your benefit -- it
is hiding the part you need.

## Fixing a crash is how you find out what it was hiding (task-15512)

**Incident.** Three Settings tests failed with a timeout waiting for a toast. The
cause was a stdlib-logging call written in loguru's `{}` style, which raises
`TypeError` when the record is formatted; `_pytest.logging.LogCaptureHandler.
handleError` re-raises deliberately, so the Textual save worker died mid-save.
Fixing the log call made ONE of the three pass -- and the other two then failed
on their real assertion, which was a genuine product bug (pressing Save marks
untouched fields dirty-and-empty, and one of them aborts the save).

**Rule.** A crash in a code path masks every assertion downstream of it. After
fixing one, re-run and expect NEW failures rather than green; treat "same count
of failures, different reasons" as progress. This is the third time in this
programme that repairing a run-killing defect exposed defects nobody had counted
(see the hang-class sweep and the harness-config work).

**Corollary on severity.** The same log bug behaves differently in the two
environments: production stdlib logging *swallows* the formatting error and
carries on, so nothing was broken for users -- only the warning was lost. It was
tempting, and I did briefly claim, that a failing save in tests meant a failing
save in the product. Check which layer makes a failure fatal before assigning it
user impact.
## A DOM swap moved into a worker is invisible to `pilot.pause()` (task-15461, 2026-08-11)

**The trap.** `Pilot.pause(delay)` is `await self._wait_for_screen()` then
`await asyncio.sleep(delay)`. `_wait_for_screen` drains the **message pump** — it posts a
callback to every widget on the screen and waits for them all to come back. It knows
nothing about Textual **workers**. So a UI update scheduled with `call_next` is covered
by every `pilot.pause()` in the suite; the identical update scheduled with `run_worker`
is covered only by whatever wall-clock `delay` the test happened to pass.

**What happened.** Replacing Watchlists' whole-screen `refresh(recompose=True)` on a tab
click with a region-scoped swap also moved the swap from a `call_next` callback (which is
what `refresh(recompose=True)` is, internally: `_recompose_required = True;
call_next(self._check_recompose)`) onto the screen's existing surface-refresh drain, which
ran as `run_worker(..., group="wc_surface_refresh")`. Nothing about the swap's *duration*
changed — instrumented at ~250 ms for the Artifacts pane before and after — but the
suite's shared helper opens a section with `pilot.pause(0.2)`, and 250 > 200. Eight tests
in `test_watchlists_artifacts_pane.py` began failing with `NoMatches:
#watchlists-artifacts-pane`, **passing in isolation and failing in a full-file run**,
because the margin was machine load. Two more flipped between runs. It looked exactly
like flakiness and was not: it was a deterministic ordering change, mis-read as noise
because the symptom was load-dependent.

Scheduling the drain with `call_next` fixed all ten and cost nothing else — it is also
strictly safer than the worker it replaced, whose own comment explains that it needed a
private worker group so the screen's several `run_worker(..., exclusive=True)` call sites
could not cancel it mid-swap. A `call_next` callback cannot be cancelled by a worker at
all.

**What to do.** Before moving any DOM mutation onto a worker, ask what the tests (and the
app's own idle handling) actually wait for. `run_worker` is for *work*; the mount/remove
pair that lands its result belongs on the pump. And when a batch of tests starts failing
together in a full run while passing alone, do not reach for "flaky" — check whether the
change under review moved something out of what the harness waits on.

## A region factory that reads state before its `await` loses whatever lands in the gap (task-15461, 2026-08-11)

**The trap.** Textual's own `Widget.recompose` removes its children **first** and calls
`compose()` afterwards, so it always reads widget state on the late side of the yield.
Hand-rolled in-place swaps usually do the opposite — build the replacement first, so a
factory that raises leaves the old content standing rather than an empty box — and that
inversion opens a window: state read, `await remove()`, state changes, `await mount()`.

**What happened.** `watch_active_section` dispatches the new section's loader and the
region swap in the same breath. `WatchlistsWorkbench.refresh_region_content` calls the
region factory (which reads `self._loaded_rules`) *before* its remove/mount awaits. The
loader — an `AsyncMock` in the test, a fast local query in production — completed during
the removal, wrote its rows to the screen, then looked for its pane and could not find it:
the replacement existed but was not yet mounted. Result: an Alert-rules table that stayed
empty over a `_loaded_rules` holding the row, with nothing left to correct it. The
whole-screen recompose being replaced had never had the gap, purely because of Textual's
ordering.

**What to do.** When you replace a recompose with a hand-rolled swap, re-apply the state
*after* the mount (`_reseed_active_section_pane`) rather than trusting the read that
happened before it. Reactive assignments make the re-apply free when nothing moved, so
the cost is a few lines and the failure mode it closes is silent.

---

## A pathname stat and an open-handle stat need not expose the same native identity field (TASK-2062.1, 2026-08-13)

TASK-2062.1's local-GGUF admission passed on Linux and macOS but rejected an
unchanged file on Windows. CPython 3.12's Windows pathname `stat` compatibility
surface reports creation time through `st_ctime`, while `fstat` on the already
opened descriptor retains the file's ChangeTime. Comparing the complete tuples
made an unchanged pathname and its own open handle look different. The first
two native Windows runs also exposed test-only POSIX assumptions before the
real identity mismatch became visible.

The correction compares only fields with shared pathname/descriptor semantics
when proving the name still refers to the opened file on Windows, while keeping
the descriptor-to-descriptor recheck strict, including ChangeTime. Tests mutate
device, inode, mode, size, and mtime independently, and the exact three-OS lane
runs the Windows reparse and replacement cases instead of accepting skips.

**What to do.** For TOCTOU defenses, distinguish the two questions: whether a
pathname still names the opened object, and whether the opened object changed
after inspection. Do not assume every portable `stat_result` field has identical
meaning across pathname and handle APIs. Preserve strict handle rechecks, test
each stable identity field, and require native-platform evidence for filesystem
security claims.

**TASK-16230 follow-up (2026-08-14).** Host-independent Windows doubles initially
made the one-time Notes import reader look race-safe while its real `CreateFileW`
call still included `FILE_SHARE_WRITE`, and its pathname-to-handle check accepted
`st_ino == 0`. Review showed that a same-size rewrite with restored mtime could be
admitted, while a zero inode provides no promised file identity. The correction
denies write/delete sharing on source-file handles, keeps the directory-pin share
mode separate, and fails closed unless pathname and handle expose the same nonzero
inode for the same device. Test the native share-mode arguments and unavailable-ID
case explicitly; tuple equality alone does not prove the object stayed immutable.

## A whole-screen recompose is doing four things you did not ask it for

**The trap.** Converting `refresh(recompose=True)` to a region-scoped rebuild looks
like a pure narrowing: same content, fewer widgets. It is not. The recompose was also
providing services the new path silently drops, and none of them fail loudly.

**What happened.** Task-15475 (2026-08-11/13) converted four surfaces. Every one of
these was caught by an EXISTING test, not by reading the diff:

* **Mouse-capture release.** `BaseAppScreen.refresh`/`recompose` release
  `App.mouse_captured` before and after the teardown (task-627): an `Input` has no
  `_on_hide`, so a widget torn down while capturing leaves a dangling capture and
  every mouse click app-wide is silently swallowed from then on. A region swap tears
  widgets down too and got none of that. Now extracted to
  `release_mouse_capture_for_teardown` / `sweep_stale_mouse_capture` and called by
  both converted screens.
* **Callback ordering.** Textual runs a screen's recompose BEFORE its
  `call_after_refresh` callbacks, so "select the category, then focus a field in it"
  worked by construction. A region rebuild driven from a worker (or from the region's
  own `_check_recompose`) is a DIFFERENT pump with no ordering against the screen's
  callback list: the Speech deep link ran against a pane that did not exist yet and
  dropped its focus on the floor, leaving the user on `nav-home`. Follow-ups must hang
  off the swap itself.
* **Post-layout geometry.** Anything reading `virtual_size`/`container_size` (here an
  inspector overflow indicator) must still run after a REFRESH; read inline at the end
  of the swap it sees pre-layout zeros and renders the wrong state.
* **The repaint short-circuit.** `Widget.refresh(recompose=True)` returns before
  `_set_dirty`. Drop `recompose=True` and a plain reactive assignment now resolves
  `self.app` — which raises `NoActiveAppError` in every bare-screen unit test that
  sets that reactive. `repaint=False` restores the property honestly (the screen
  renders nothing from the value; its children do).

Also worth knowing: scoped is not automatically faster. Two separately-awaited region
swaps each drove their own layout pass and measured 105 ms against the 69 ms the
whole-screen recompose appeared to cost. Both numbers were wrong to compare — the Lab
frame defers its body mount OUT of the recompose, so that 69 ms excluded the expensive
half. Wrapping the swap in `self.batch()` (what `Widget.recompose` itself uses) took
it to 88 ms, and the honest end-to-end measure — trigger to content actually on
screen — was 325 ms before, 146 ms after.

**What to do.** Before converting a recompose, list what it did besides re-render:
grep the screen's `refresh`/`recompose` overrides, its `call_after_refresh` call
sites, and any geometry reads. Port each explicitly. Measure trigger-to-content, never
"time the two coroutines", and batch the swap.

**Review round 1 added three more, all measured, none visible in the diff:**

* **A "container" you empty may not be yours.** `remove_children()` on a frame region
  is only safe if the region holds nothing but mode content. `#lab-rail` and
  `#lab-inspector` each carry a frame-composed collapse header as their FIRST child —
  which is precisely why `LabScreen._populate_regions` APPENDS with `mount_all` and
  says so. The blanket removal destroyed both collapse buttons on the first click,
  permanently (no keyboard binding, no recompose left to restore them). If the
  existing code mounts with `mount_all` rather than replacing, that is a signal:
  something else already lives there.
* **Focus does not "stay put" when the widget under it is destroyed — it MOVES, to a
  neighbour you did not choose.** Both conversions landed the user on a collapse
  affordance one Space away from destroying their own context
  (`settings-category-group-domain-defaults`, `lab-rail-collapse`). Capture the focus
  token before a teardown and restore it by id, but defer the restore and yield to
  the rebuilt subtree — a freshly mounted widget may have focused itself ON PURPOSE
  (`ResultsGrid` does, so its advertised shortcuts work), and an eager restore wins
  the FIFO race and silently kills that.
* **`exclusive=True` is the wrong supersede primitive for a teardown.** It cancels the
  in-flight worker, and the cancellation can land inside `remove_children` — leaving a
  region emptied and never refilled when the superseding swap does not rebuild that
  same region, and skipping the post-teardown capture sweep. A lock plus a revision
  check supersedes just as firmly and lets the loser return before touching a widget.
  Accumulate any per-call flags (`rail_dirty`) across superseded calls, or the
  survivor silently drops the loser's work.

And one about the evidence itself: **a test that asserts on the nearest visible text
can be satisfied by a different code path.** Neutering the sync-rows region rebuild
left all six evidence tests green, because the assertion read a summary `Static` that
another path keeps current. Assert on the widgets ONLY the mechanism under test
writes, then mutation-check by neutering that mechanism.

## An absolute event-count pin records which side of a race the author's machine won (TASK-15458, 2026-08-13)

**Incident.** Task-15458's perf pin asserted `markdown_updates ==
[id(markdown_before)]` — "opening the media item parses the document exactly
once". It was written and verified on Windows, where it passed. On macOS it
failed 3/3 with `markdown_update_count=2`, and it had been red on `dev` from
the moment it merged. The count was not flaky, and it was not a platform quirk
of the test: it was reporting a real defect that the authoring machine happened
to hide. Opening a media item issues two `refresh(recompose=True)` calls — the
"Loading media…" one at click time, and one when the detail worker resolves.
Textual's `recompose()` awaits child teardown BEFORE it calls `compose()`, so a
worker landing inside that await gets picked up by the in-flight compose, and
the worker's own recompose then parses the whole 49 KB / 2,000-line document a
SECOND time. Windows lost that race the other way (both refreshes coalesced
into one recompose), so the same production code produced 1 there and 2 here.
A/B on the open click: 922/914/935 ms and 2 parses with the arrival recompose
unconditional, 710/730/841 ms and 1 parse with an identity guard on the
already-composed detail.

**What to do.** An absolute count over a window that spans a scheduling race
pins your machine's timing, not the contract. Two habits fix it. First, scope
the count to the interaction the claim is about — the sibling test in the same
file already did this (`parse_count_before_navigation = len(markdown_updates)`,
then assert no growth across the click), and it was green on both platforms
because the delta cannot absorb an unrelated race. Second, when you do want an
absolute count, first make it deterministic in PRODUCTION (here: the guard),
then pin it — a total that is only stable on one OS is evidence about the OS.
And treat a count that differs from the notes' recorded value as a defect
report until proven otherwise: the number was right, the code was wrong.
## When a screen really is widget-bound, COUNT widgets — a wall-clock A/B can't resolve the change (task-15462, 2026-08-13)

**What happened.** Profiling the Watchlists push turned up a genuine piece of waste: the
screen's `region_layout` reactive defaults to "nothing collapsed" while the shipped
first-run default collapses the RIGHT_RAIL, so every visit composes the expanded
Inspector rail and `on_mount` immediately swaps it for the one-line collapsed header. A
prototype removing the swap was measured against dev the obvious way — run the probe
process on dev, then run it with the fix, compare medians. It reported **35% faster**.

That number was an artifact. Re-run with the two arms interleaved *inside a single app
run* and ABBA-ordered (so monotonic machine drift cancels instead of favouring whichever
arm is measured second), the same change came out at **median delta −1 ms, faster in 6 of
12 pairs**. Repeated identical configurations on this machine ranged **360–925 ms within
one run** — the noise floor swallows anything under roughly 30%.

The noise-free measurement had been available all along and agreed with the paired
result: instrumenting the swap showed it discards **13 widgets** and mounts 1. A
dose-response sweep (feed page 0/24/60/100 items → 86/170/260/344 widgets →
200/218/244/342 ms) put the screen's cost at **~0.55 ms per widget**, so 13 widgets is
5–10 ms of a ~450 ms push — 1–2%, exactly what the paired A/B failed to detect.

**What to do.** Establish whether the screen is widget-bound *first* (survey + a
dose-response sweep over something that varies the widget count). If it is, size every
candidate lever by the widgets it removes and use wall clock only to confirm a prediction
big enough to clear the noise floor. If you must A/B by wall clock, interleave the arms
within one process and alternate their order; a fixed dev-then-fix ordering across
processes measures drift as effect.

This is the mirror of the defer-past-first-paint lesson, not a contradiction of it.
There, widget count *over*-predicted, because Schedules and Console were sync/DB-bound and
their hidden mass cost nothing to skip. Watchlists is genuinely widget-bound — 13 sqlite
statements and ~10 ms of application code for a whole push, everything else Textual's
per-widget CSS apply and mount. The rule is the same in both cases: find out what the
screen is bound by before choosing what to count.

## A mounted descendant is not evidence that its recompose finished (TASK-19505, 2026-08-21)

**Incident.** The Console mount profiler initially declared the deferred Context rail
"full ready" as soon as its first section header became queryable. Textual made that
header available while the same `recompose()` was still mounting later descendants.
The probe then focused the composer and typed during the unfinished hydration, reporting
a 689 ms key-to-echo p95 and a misleading full-ready distribution. Waiting for the
hydration callback itself to return moved input strictly after the recompose boundary.
The retained raw 30-sample rerun produced the honest verdict: key latency stayed within
budget, while Enter-to-worker p95 regressed 12.39% and rejected the candidate.

**What to do.** When a performance phase ends at an async mount/recompose operation,
gate the next phase on completion of that owning operation, not on the first descendant
becoming queryable. A selector proves presence, not subtree completeness. Record the
boundary before looking at the result, and keep input probes after it so deferred work is
not silently reclassified as interaction latency.

---

## A test's stimulus can rely on the exact inefficiency your fix removes (task-15459, 2026-08-13)

**Incident.** task-15459 made `LibraryScreen._apply_local_source_snapshot` skip its
`refresh(recompose=True)` when the incoming snapshot is byte-for-byte identical to
what is already rendered — the point of the task, since a warm revisit's reconcile
fetch confirming the app-scoped cache verbatim no longer needs to repaint. Two full
background suite runs afterward reported 14 failures. `test_library_note_recompose_
and_fifty_route_cycles_return_to_baseline` was one: its stress loop called
`_apply_local_source_snapshot` five times with a `dict()`-copied but otherwise
UNCHANGED snapshot, purely to force a recompose and verify a dirty note-editor
session survives being torn down and rebuilt repeatedly. That loop's own assertion
("Generic source-snapshot completion never recomposed the Notes workbench") is
exactly the behavior the fix intentionally removed — the test's PASS depended on
the inefficiency, not on anything the task changed being wrong.

Reflexively "fixing" this by loosening the guard, or by deleting/skipping the test,
would both have been mistakes: the guard is correct (measured 2 composes → 1 for a
real warm revisit), and the test's underlying intent (repeated recomposes must not
corrupt a dirty session) is still a real requirement worth pinning — its STIMULUS
was just now inert. The fix was to vary a harmless field (the notes count) each
loop iteration, restoring a genuine data change that still forces the recompose
under the new contract, matching what a real background refresh would look like.

Of the other 13 reported failures, mutation-bisection (temporarily reverting BOTH
halves of the production diff to their pre-task behavior with `Edit`, confirming
the SAME failure still reproduces, then restoring — never `git checkout --`, which
discards uncommitted work) showed 9 were pre-existing (reproduced identically with
the diff neutralized, mostly drift from an unrelated recent merge) and 4 were
load/order flakiness that passed reliably in isolation. Zero were real regressions.

**What to do.** When an optimization correctly removes redundant work and a test
goes red, do not assume either "the test is now wrong, ignore it" or "my change
broke something" — read what the test's assertion is actually FOR. If it names the
mechanism you just changed ("never recomposed", "recompose count", "refresh was
called"), check whether that mechanism was the test's STIMULUS (how it drove the
scenario) or its OUTCOME (what it was actually verifying). A stimulus that no
longer fires needs a new stimulus that still exercises the real requirement; an
outcome assertion that no longer holds needs the assertion updated to the new
contract. Across a batch of full-suite failures, mutation-bisect each one against
your own diff before writing any of them off as "pre-existing" or accepting any as
"caused by my change" — a batch this size will usually contain both, plus plain
flakiness, and a single red run distinguishes none of them.

---

## An unchanged-skip guard is only as reliable as its least reliable compared field (task-15459, 2026-08-13)

**Incident.** task-15459's `_apply_local_source_snapshot` compared an incoming
snapshot against the currently-rendered one and skipped a recompose when they were
equal — the flagship AC test asserted this held across a reconcile fetch that
should have confirmed the cache verbatim. Review reproduced the test failing
intermittently at exactly that assertion. Root cause: the flat comparison included
`study_counts` (`study_decks`/`flashcards_due`/`quizzes`) and two rail badge
counts (Prompts, Skills) — every one fetched by a `..._or_none` helper whose own
docstring says it swallows ANY exception and degrades to `None`. Under thread-pool
contention, two fetches of the SAME unchanged data could legitimately disagree on
one of these fields (one call transiently raised, the other did not), making the
guard fire a full recompose for a coin-flip on a decorative badge — "fails safe"
(a spurious recompose, not a missed one) but non-deterministic, which is exactly
as unacceptable for an "exactly once" acceptance criterion as failing unsafe.

The first attempt at writing THIS test only asserted the guard's happy path — it
never modeled a field that changes independently of the state a user would call
"the data." A single flat `==` over a snapshot dict is only as trustworthy as its
least reliable member field.

**What to do.** Before folding several fields into one equality check that gates
an expensive operation, audit each field's OWN fetch contract, not just its type.
A field fetched by a helper that swallows exceptions and degrades to a sentinel
(`None`, `""`, an empty collection) is not equivalent in reliability to a field
whose fetch either succeeds or aborts the whole call — the former can flap between
two fetches of otherwise-identical state, the latter cannot (barring the state
genuinely changing). Split the comparison into domains — STRUCTURAL fields that
must gate the expensive operation, and DECORATIVE/best-effort fields that should
be patched through a cheaper path (an in-place widget update, a `None`-tolerant
merge) instead of ever gating it. To prove the split actually closes the gap, do
not just re-run the flaky test and hope: inject the exact transient exception
deterministically (a fake service that raises on its Nth call, not the Mth) so the
flap is reproducible on demand, and mutation-test the fix by temporarily re-
merging the domains to confirm the ORIGINAL failure message comes back verbatim.

---

## A parent `on_mount()` cannot assume nested descendants are mounted (TASK-2702, 2026-08-13)

**Incident.** Three Library Prompt-history tests repeatedly crashed while a
`PromptBlockEditor` was being replaced during rapid recomposition. Its `on_mount()`
queried `#prompt-editor-validation`, a grandchild inside the editor's status container,
and raised `NoMatches`. Instrumentation at the exception showed the editor was attached
and all three direct containers already existed, but their nested children did not. An
unconditional `call_after_refresh` removed that race, then exposed the opposite defect:
two ordinary-mount tests observed an empty footer because they legitimately inspected it
before the deferred callback ran.

**What to do.** A Textual parent's Mount event guarantees neither that every descendant
message pump has finished mounting nor that consumers will wait through an extra refresh.
Initialize synchronously when the required descendants are present; if `NoMatches` proves
the nested-mount window is still open, defer that same initialization once. The deferred
callback must no-op when its original widget has detached. Verify both paths: a rapid real
recompose must kill the synchronous-only implementation, while an immediate normal-mount
assertion must kill unconditional deferral. TASK-2702's final full Prompt-canvas run passed
279 tests only after both boundaries were pinned together.

## A full-suite sweep is a checkpointed pipeline, not a command (task-15211)

**Incident.** Three attempts to run all of `Tests/UI` in one pytest invocation
died at 25-32%, each time losing everything. The fourth attempt split the 503
modules into 16 chunks, appended each chunk's summary and failures to a results
file as it completed, and skipped already-recorded chunks on relaunch. It
survived a hung chunk, an environment process-kill, and a TCC lockout, and
finished: 10,811 passed, 117 attributed failures.

**What the monoliths actually died of.** Not slowness: a product defect. The
Lab/LLM screen's Ollama probe held two event-loop threads open, so pytest
PRINTED ITS FINAL SUMMARY and then never exited -- zero CPU, main thread
joining a non-daemon thread. A wrapper waiting on the child sees an eternal
hang after a successful-looking run. Diagnosis that worked without root:
compare `ps -o time` across an interval (zero accrual = hung, not slow), then
`sample <pid>` for native thread stacks -- two threads parked in kevent were
the loops that should have died with their screen.

**Rules.** (1) Never run a >20-minute suite as one process; checkpoint per
chunk and make relaunch skip recorded work. (2) "The log stopped growing" has
two different causes -- a hung TEST (mid-run) and a hung EXIT (summary already
printed); check for the summary line before assuming the former. (3) Keep the
sweep's worktree frozen and ship fixes from another one; the sweep's chunk
results stay comparable, and later chunks re-finding an already-fixed class is
CONFIRMATION, not new work.
## A permanent gate must read its immutable baseline from a PINNED revision, not the live file it exists to police (TASK-15103, 2026-08-11)

**Incident.** TASK-15103's complete-history denominator — the thrice-reviewed
proof that every diagnostic transition since the stored baseline was
consumed exactly once — read that stored baseline from the live
`production-diagnostic-inventory.json`. The gate's entire lifecycle ends
with regenerating that exact file, so the first LEGITIMATE regeneration
broke it: the stored-revision scan went hunting through all of dev history
for post-repair populations that exist in no dev-reachable revision, and 10
gate nodes fell over — first on a merge-conflict-markered historical blob's
SyntaxError, which had nothing to do with the actual defect. The evidence
was always available immutably: `incident.recorded_base` pins the dev
revision whose committed manifest IS the stale baseline, byte-identical on
all 19 owner rows. One read-from-`recorded_base:`-tree change fixed all 10.

**Companion lesson from the same day.** The freeze-first plan this gate
belonged to never converged against live dev: three boundary re-freezes in
one day (17→18→19 owners), each invalidated by dev advancing while the
evidence was being rebuilt, zero production repairs shipped. Inverting the
order — repair to the frozen contracts first, regenerate and prove ONCE at
the end — landed all 43 repairs plus the gate in one session, and the next
dev advance (11 rows + a sink-topology change) was correctly surfaced as a
NEW incident (task-15600) instead of another re-freeze of this one.

**Postscript (TASK-15700, 2026-08-13): both halves of that forecast held,
and the row still did not ship.** The merge fix restored `and_then_or`'s
rescue **at exactly slot 10**, and scoped recall went 0.429 -> 1.000 — so
the mechanism was correctly identified and its counterfactual correctly
measured. The row was then disqualified on a *different* constraint (no
gated cell down > 0.02), by a mechanism one level further out again: tier 2
confines fallback rows inside the keyword LEG, but tier 2 still enters
FUSION, where a fallback row carrying a vector rank becomes a MERGED row
that outscores any fts-only row. The corollary to carry: **fixing the
composition level the last defect lived at buys you exactly that level.**
Before claiming a fix unblocks a candidate, ask what the NEXT composition
step does with the rows you just re-ordered.

---

## A pre-registered rule the owner then overrides must be recorded as TWO facts, never one (TASK-15700, 2026-08-13)

**Incident.** The keyword-leg arc re-ran TASK-15400's construction sweep
under a decision rule registered **before** the run (max leg census subject
to three hard constraints; ties broken by fewest extra FTS statements, then
smallest code delta). The rule ran to completion and produced a winner:
`prefix`, on a tie-break measured at 240 vs 460 SQLite statements over the
60 golden queries. But by then the arc's own reviews had **measured** a
failure shape the tie-break predates — a construction that widens as the
PRIMARY self-displaces inside one sub-leg's bm25-ordered, LIMITED result
set, where the new tiered merge can protect nothing — and the tied
runner-up (`and_then_prefix`) is immune to it by construction while being
measurement-identical on every captured axis. The owner's standing
stability-over-quick-wins ruling cut against the tie-break, and
`and_then_prefix` shipped.

**The two tempting ways to write that down are both dishonest.** Editing
the rule after seeing its output ("fewest statements — *unless* the row is
structurally unsafe") retroactively makes the sweep unfalsifiable: a rule
amended to fit its own result never rejected anything. Recording only the
shipped value is worse in a quieter way — a later reader assumes the
measurement chose it. That second failure nearly shipped here: the backlog
record read "WINNER under the rule = `prefix`" and said nothing about what
actually shipped, so the one file neither the implementation nor its review
touched was the file that told the wrong story.

**What to do.** When a standing judgement overrides a pre-registered rule,
keep both facts and keep them adjacent, at **every** site that records the
outcome — config comment, the function's own docstring, the test names, the
task record, the PR body: (1) the rule was applied verbatim and produced X,
with the deciding number; (2) the owner ruled Y ships instead, on this
named dimension, at this measured price. Rename any pin whose name asserts
the wrong provenance — `test_the_shipped_default_is_the_sweeps_winner`
became `..._is_the_owner_ruled_construction` precisely because the old name
was a false claim that would have stayed green forever. And police the
*evidence*, not just the value: the census pin added in the same task states
in its own docstring that the census is the number **both** qualifiers score
and is therefore **not** evidence for the ruling — the ruling's evidence is
structural and its price is statements, neither of which a census can see.
A number that cannot see the decision must never be cited as its
justification.

## A harness convenience call that bypasses the production entry path verifies nothing about that path (tasks 15862/15970, 2026-08-13)

**Two incidents in one live pass, same shape.** (1) The wake-UI freshness
suite injected `FleetDrained` events by calling `on_fleet_drained` from the
test coroutine — whose context carries Textual's `active_app` under
`run_test`. Production delivers the drain from the CHILD's daemon thread,
whose `call_soon_threadsafe`-copied context has no `active_app`; a
transcript-poll timer created in that bare context dies on its first tick
(`Timer._tick` reads the ContextVar; an asyncio task inherits its CREATION
context). The suite went green against a fix that did not work — live
frames showed "arm-poll" logged and zero beats, the exact frozen-UI bug the
tests claimed to kill. (2) The user-wins-ties wiring test staged the draft
with `composer.load_draft(...)`, which writes the canonical segments
directly; a live draft typed with real keys was invisible to
`draft_text()` at probe time (pane showed the text, probe read `''`), and a
wake fired straight through the user's held draft — the deferral the test
"proved" (task-15970).

**The rule.** Before trusting a test that drives an event or input, ask
which THREAD, CONTEXT, and ENTRY POINT production uses, and drive that. A
drain must come from a plain thread; typed input must be typed
(`pilot.press`), not loaded. If the harness path and the production path
diverge at any of those three, the test is verifying the harness. The fix
pattern for (1): route the drain through
`threading.Thread(target=...)` in the test, and hop UI arming through the
message pump (`call_later`) in production — after which reverting the hop
fails three tests instead of zero.
---

## A control that holds a second variable fixed measures the PAIR, not the thing you named (TASK-15965, 2026-08-13)

**Incident.** The PRF probe needed to know how many of its 22 target cells a
rescue could have been *seen* in at all, so it ran a control: feed the
retrieval the target document itself — the best expansion any feedback set
could ever produce — and count how many targets that lifts into the top-10.
It returned **8 of 22**, and I wrote that number down as a property of the
retrieval path: "the four-seam path caps ANY query-widening technique at
8/22; 14 of 22 cells are never observable." The probe printed a matching
sentence on every run: *"a target an oracle feed cannot lift into the top-10
could not have been rescued by any real feedback set."*

The control had **two** moving parts, not one. It fixed the path — which is
what I named — and it also fixed the **term selector** used to build the
oracle expression (the pre-registered TF `tf/|D|` top-8). The review re-ran
the control changing **only the ranking key**: same path, same oracle feed,
same composition, same k. It returned **15 of 22** (rarest-8-by-corpus-DF),
and at N=1-rarest with the query side dropped, **22 of 22**. Meanwhile 22 of
22 oracle expressions matched their target at k=200 in *every* row — so
nothing was ever unreachable; the misses were displacement whose severity
scales with **expansion breadth**, which is the selector's property, not the
path's. The defensible statement was narrower than the one I shipped (the
path has no cross-seam ranking and a per-seam `top_k`, so a pass matching K+
notes buries non-note targets *however hard that bites depends on breadth*),
and the correction made the arc's null **stronger**: ≥15 observable cells,
still 0 rescued.

**Two tells were on the page before the review.** (1) The number was a bound
that *flattered the conclusion* — a smaller observable population makes a null
easier to explain away. A bound you would be glad to have is one to measure
twice. (2) The printed claim quantified over something the control never
varied: "could not have been rescued by **any** real feedback set", from a
single selector.

**What to do.** Before reading a control's output as a property of X, write
down every variable the control holds fixed and ask which of them could have
produced the number. If a fixed variable is plausibly load-bearing, **vary it
and re-run** — here that was one parameter and one re-run, and it was the
difference between a measured bound and a bound of my own making. Then make
the instrument carry the scope: the probe now prints the selector-comparison
table instead of the universal sentence, and an assertion fails the run if a
non-pre-registered selector ever reaches the verdict. Sibling of "A mechanism
sentence is an ORACLE" — but distinct in cause: that prose was refuted by data
already on the page, this prose was wrong because the refuting data had not
been collected. Also sibling of "A property that holds 'by construction' holds
for the COMPONENT": both are scope errors, one about composition, this one
about which variable the measurement was actually of.

## Complete invalidation coverage is not evidence a cache is race-free (task-15471, 2026-08-14)

The starred-conversations cache added in task-15471 had provably complete invalidation
coverage — every writer went through `set_mark`/`clear_mark` on the one app-owned service
instance, and the suites were green. The review's interleaving probe still found it serving
stale data in **103 of 300 naturally-scheduled rounds**: a cache-missing reader held its rows
across the transaction COMMIT (a GIL-releasing sqlite call) and stored them AFTER a concurrent
writer had invalidated — so the "invalidated" cache got repopulated with the pre-write
snapshot and stayed wrong until the next write. Two lessons with teeth:

- **Auditing who invalidates answers the wrong question.** The bug was populate-after-
  invalidate *ordering*, which no amount of invalidation-coverage evidence touches. The fix
  shape is a generation counter captured under the lock before the read and compared before
  the store (store only if unchanged); a lost race then costs one skipped store, never a
  stale entry.
- **Only a dedicated interleaving probe surfaced it.** Unit suites exercise reader and writer
  on one thread; the deterministic repro needed a reader paused at commit-exit while a writer
  ran, and the natural repro needed a tight cross-thread hammer. When a change introduces a
  read-cache whose writers live on other threads, a probe of this shape is part of the
  evidence bar — green functional tests alone said this cache was fine.
---

## Consent must bind every independently mutable authority input (TASK-208, 2026-08-13)

**Incident.** TASK-208's first reviewed duplicate override fingerprinted the
folder path, form options, warning text, and previewed active job IDs. The app
correctly re-expanded the folder at submission time, but the Boolean override
then bypassed every match in that new expansion. A newly added member, a newly
active job absent from preflight, or an unchanged warning sentence with a larger
affected-file count could therefore ride stale consent.

**What to do.** For two-step consent, enumerate every input that can change
between arming and use, including derived cardinalities that do not alter copy.
Carry a privacy-safe identity of the exact consented set across the authority
boundary and compare it only after authoritative recomputation. An override must
cover every current match; bounded identity material must record truncation and
fail closed. Test mutations between the two user actions through at least one
real UI-to-authority path, not only each boundary in isolation.
---

## An expected value computed THROUGH the code under test cannot fail — the reference has to come from upstream of it (TASK-16071, 2026-08-14)

**Incident.** The rank-fair four-seam merge arc pinned that the merge preserves
each seam's own ordering: whatever order a seam returned its rows in, the
merged list must contain that seam's rows in exactly that relative order. The
pin needed a reference — "the order the notes seam returned" — and it got one
the obvious way: a **single-source `search()` call** per seam, notes only, then
media only, and so on. That reads like the seam's own ranking, and for a
one-seam query it *is* the same rows in the same order.

It is also the code under test. `search()` runs the merge on its way out, so
the reference travelled through the very function the pin was written to
police. The mutation exposed it: **reverse every seam's ranking before the
interleave**, and the merged list and the reference both came back reversed —
identical to each other, as always. The suite reported **5 passed**. A test
suite reported green against an implementation that had inverted the ordering
property it existed to pin.

The fix is one line of plumbing and no cleverness: a `_seam_ranking` helper
that calls the seam methods directly (`_search_notes` / `_search_media` /
`_search_conversations` / `_search_prompts`), upstream of the merge. With the
reference sourced there, the same mutation reds immediately:

```
E  AssertionError: the notes seam's rows were reordered by the merge:
   seam order [('note','7820e1a3…'), ('note','59470e66…')],
   merged order [('note','59470e66…'), ('note','7820e1a3…')]
2 failed, 3 passed
```

**Why it was invisible.** Every version of the trap looks like reuse, which is
usually a virtue: the reference is fetched by the same public API, on the same
data, in the same test — and a single-source search really is "the same"
ranking, right up until the merge is the thing you are measuring. Nothing about
the call site says "this expression is a function of the code under test"; you
have to ask.

**What to do.** For any test whose assertion compares an output against an
expected value the test itself computes, write down where that expected value
came from and check it does not route through the function under test — a
public API that *wraps* the unit is the common way it does. Source references
from the layer below (the seam method, the raw query, a pasted fixture), and
prove it by mutation: if the mutation moves the output and the expected value
identically, the test is measuring nothing. Sibling of "A surviving mutant
usually means a SECOND writer satisfies your assertion" but distinct in cause —
there a second mechanism produced the asserted state; here there is only one
mechanism and the test asked it to grade its own work.

---

## A fix recorded only in a gitignored file is not a fix — the diff is the deliverable, the scratch is the diary (TASK-16071, 2026-08-14)

**Incident.** A review round on the same arc raised four minors. The
implementer's close-out reported all four addressed, and the working ledger and
task report — both under the worktree's gitignored `.superpowers/sdd/`
directory — described the corrections in detail and accurately. The re-review
checked the **diff** rather than the write-up and found two of the four existed
nowhere else: the collateral-swap identities with their direction (the
rank-fair rotation's cost landing on the NOTE seam) and the rigorous
`r ≥ (p+1)/3` rank-fair bound had been *written down*, not *shipped*. Both were
supposed to land in the tracked `Tests/RAG_Eval/README.md`, which a later
reader would consult; instead they lived in a file that is deleted with the
worktree and invisible to anyone who does not have it.

Nothing was wrong with the corrections themselves, which is what makes the
shape durable: writing the fix and shipping the fix feel identical while you
are doing it, and the bookkeeping that says "addressed" is written by the same
person in the same session, from the same paragraph.

**What to do.** When a review item's remedy is *prose* — a README correction, a
docstring, a task's Notes — close it by naming the tracked file and the text,
then verify with `git diff`/`git status` that the change is in the diff before
recording it as addressed. Treat any scratch or SDD directory as a diary: it is
where you think, never where a deliverable lives. Sibling of the hygiene entry
"Gitignored working files die with their worktree", which is about the same
directory but a different failure — that one loses a record you correctly wrote
there; this one never wrote it anywhere else in the first place.

---

## A repro helper that ASSERTS the bug turns your suite into the bug's guard (TASK-16300, 2026-08-14)

**Incident.** The wake-integrity arc (15970/15971) needed a Console screen that
was mounted but not displayed. It built one through the real navigation API —
push a modal over Chat, navigate to Library — and its helper `_leak_resident_chat`
closed with a *precondition* assertion:

```python
assert chat in app.screen_stack, (
    "harness precondition: the nav-under-a-pushed-screen path must "
    "leave the Chat screen resident in the stack ..."
)
```

That state was a bug: `App.switch_screen` pops only the top of the screen stack,
so navigating under a modal replaced the MODAL and left the outgoing screen
running. When the leak was fixed one day later, four of that file's six tests
went red **on that assertion line** — not on a single behavioural assertion.
The failure output read exactly like "the residency fix regressed the wake
layer". It had not: mutating the 15970 probe fix and both 15971 gates back out
still turned the same tests red once their setups were rebuilt, so the tests
were sound and only their *construction* had been harvested from the defect.

The trap is that the helper was written the RIGHT way by every other rule —
real production APIs, no hand-built screens, no `load_draft` shortcut — and
fidelity to production is precisely what welded it to production's defect.

**What to do.** Before asserting a state as a harness precondition, ask whether
that state is a *contract* or an *observation*. A contract ("the composer holds
the typed text") is worth pinning. An observation of current behaviour,
especially one you reached for because it was convenient, must not be phrased as
a requirement — build the state from the smallest API that produces it legitimately
(here: push a modal over Console; push a second Console screen), and if it is only
reachable through a defect, say so in the docstring and file the defect. Note also
what the wording cost: "the nav path MUST leave the screen resident" is how a
known bug acquires a guard, and the next reader has to decide whether the test or
the fix is wrong.
## A screen refresh is not evidence that a restored child tree is settled (TASK-13207, 2026-08-14)

TASK-13207's real Settings → Model Library → Settings run returned the reviewed
package while an unrelated Speech/TTS draft was detached. The result worker
merged correctly, but publishing the draft before terminal acknowledgement
overlapped the restored panel's queued recompose: Textual `Select` mount events
occasionally observed their overlay children between removal and mount, and the
short cleanup fence could remain attached. Immediate fake leases hid the race;
a mounted test with a deliberately slow lease exit reproduced it.

**What to do.** Treat result acknowledgement, lease exit, and restored-child
composition as separate observation boundaries. Do not publish draft state
until the exact result claim is acknowledged and cleanup authority is released.
Exercise mounted handoffs with a delayed lease exit; a screen-level idle or
refresh observation alone does not prove that a recomposing child tree settled.

---

## Authority tests must vary representation and interleave the guarded write (TASK-16309, 2026-08-14)

**Incident.** The one-time Notes import executor passed its focused execution,
receipt, retry, privacy, and crash-recovery suites, but final adversarial review
still reproduced two authority escapes. First, the approval digest NFC-normalized
title, content, keywords, and template name even though execution stored their
exact Python text. Reusing an approval with composed versus decomposed Unicode
therefore reached the target instead of conflicting. Second, membership-only
execution checked a note version before an unversioned membership write. A
deterministic update inserted between those operations let the stale membership
complete. The new RED tests respectively observed the substituted target call and
the stale attached membership despite the earlier suite being green.

**What to do.** An authority digest must encode the exact representation consumed
by the effect unless the effect itself canonicalizes to the same representation;
include canonical-equivalence substitutions for every execution-effective text
field in approval tests. An optimistic check is evidence only when the expected
version participates in the atomic mutation that grants the effect. Reproduce the
read/write interleaving deterministically, assert the stale write changes nothing,
and cover every idempotent write shape (new row, revive, and already-active row).
Green sequential and crash-recovery suites do not substitute for either probe.

## An AC's enumeration of hot call sites is not the cost profile (task-15764, 2026-08-15)

Task-15764's AC enumerated the difflib work to move off the event loop by name --
`_segment_for_diff` x2, `build_change_diff`, `added_and_removed_text`,
`classify_change_type` -- and an implementation scoped to that list would have been
green on every thread-identity test while leaving most of the stall in place. The
dominant cost was `ContentExtractor.calculate_change_percentage`, a
`difflib.SequenceMatcher.ratio` over the two full raw texts that sits three lines
above the enumerated block and is not in the enumeration. Mechanism, corrected by
the independent review (the implementer's 16.2 s / "99.8%" figure on a 160 KB Latin
page pair did NOT reproduce -- Latin text at that size hits `autojunk`'s fast path,
20-40 ms across four content shapes, and autojunk incidentally returns a
meaningless `pct` for it, a separate pre-existing oddity): character-level
`ratio()` went quadratic only when the character repertoire was large enough that
autojunk junked nothing (CJK / unicode-heavy pages) -- measured clean 4x per
doubling, extrapolating to ~1 s at 160 K chars and **~7 minutes at the 10 MB
fetch cap**. (The two regimes were not even cleanly separated: task-16839's
born-red pin found a 128 KB Latin shape -- common letters junked, digits/capitals
rare enough to survive as anchors -- that was degenerate AND quadratic at once,
pct=0.47 for a 5%-edited page after ~39 s.) The off-loop move was thus MORE
justified than the original numbers suggested, and the review's own stall probe
corroborated the shape independently (164.7 ms -> 18.4 ms max stall on the same
seam). Both regimes are historical as of task-16839: `calculate_change_percentage`
now computes an O(n) order-insensitive multiset ratio over `_segment_for_diff`
segments (the same basis as the stored diff) at every size -- measured ~6 ms at
160 KB Latin, ~430 ms worst shape at the 10 MB cap. (A first revision kept an
order-sensitive `SequenceMatcher` tier below a 4,000-segment bound with the
multiset ratio as fallback; the independent review reproduced a pure-reorder
cliff at that boundary -- 99.25% vs 0.00% across one added sentence -- and the
fix round retired the tier. See the boundary-probe lesson below.) Keep both
halves of this incident: measure the whole operation,
and expect your headline number to be re-run by a skeptic. The lesson: before
implementing a perf task scoped by a list of call sites, run one measurement that
would catch an omission -- a wall/stall probe around the whole operation, not
around the listed calls. If the numbers do not drop when the listed sites move,
the list was wrong, and the AC's own wording ("the difflib work") almost always
licenses fixing the omission in the same change -- record the addition explicitly
rather than silently widening scope.

## A version-stamp rollback fixture is a promise every future migration must keep — centralize it or it breaks serially (task-15765/task-16197, 2026-08-15)

Three "historical" ChaChaNotes fixtures were built top-down: bootstrap a fresh
DB (which lands at `_CURRENT_SCHEMA_VERSION`), hand-drop the newer artifacts,
stamp `db_schema_version` back, reopen, and let the migration chain replay.
Each fixture carried its own private drop list — and every migration that
shipped a non-idempotent artifact broke them serially: `88f5f535a` (V33→V34
unguarded `ADD COLUMN compaction_representation`) broke them and task-15730
repaired them one by one; two days later `9174975b0` (V35→V36 bare
`CREATE TABLE note_folders`, task-15705) broke them AGAIN — and, decisively,
its author fixed the ONE fixture they knew about
(`test_dictionary_attachment_index.py`) and missed the other two, producing
task-15765 and task-16197 with the identical "table note_folders already
exists" error, each then repaired in a separate task (16201, 16207). Four
repair tasks for two migrations is the signature of state duplicated where no
gate forces it to stay in sync. The fix is structural, not another patch: one
shared per-version removal registry (`Tests/ChaChaNotesDB/schema_rollback.py`)
consumed by every rollback fixture, a completeness ratchet that fails BY NAME
with instructions the moment `_CURRENT_SCHEMA_VERSION` outruns the registry,
and a rollback-replay sweep over every historical target that compares the
replayed schema's object inventory against a fresh bootstrap. The sweep paid
for itself on its first run: a defensively-copied trigger drop in the V28
entry left DBs rolled back to V20..V27 silently missing ALL conversations
sync triggers after replay — a corruption no per-test fixture would ever
notice, caught only because the sweep asserts parity with a fresh DB rather
than "the test I care about passes".

**Final shape (task-16840, 2026-08-16): the registry was itself the debt, and
the durable end state is no second copy at all.** Within a week of shipping,
the registry had grown hand-written v38/v39 entries — the ratchet was
enforcing exactly the toil the guard existed to remove. The replacement is
the knowledge-free primitive that already lived in the repo: patch
`_CURRENT_SCHEMA_VERSION` to N and bootstrap, and the production chain itself
builds a genuinely vN-shaped DB (`Tests/ChaChaNotesDB/historical_bootstrap.py`)
— real sync triggers, zero future artifacts, so the "already exists"
collision class is impossible by construction and a schema bump costs
nothing anywhere. Three generalisable findings from the replacement:
(1) **a parity oracle derived from the system under test is the identity on
that system's deterministic defects** — the old sweep caught its mutations
only where the registry happened to be a DIVERGENT second copy (the review
verified: true for the DROP COLUMN shape — entry 30's bare DROP would have
raised — but FALSE for the emptied-step shape, whose entry 36 was DROP IF
EXISTS and would have stayed green too; the old design deserves less credit
than this entry first gave it);
re-run against the single-source architecture, the review's own MUT shapes
(emptied V35→V36 step; a `DROP COLUMN messages.usage_json` seeded into
V37→V38) leave the bootstrap-replay-parity sweep 35/35 green while the
migrations' CONSUMER tests red by name (9 note-folder tests; 7 usage_json
tests) — so artifact correctness must be pinned by consumers, and the sweep's
honest job is the genuine historical upgrade matrix (resume from every vN,
stamp/dispatch wiring, stop-resume vs straight-through parity; an unwired
`migration_steps` entry reds all 35 cases with "Migration path undefined").
(2) **check the claimed-pristine baseline**: the "v4" base schema has drifted
to bake in `conversation_local_marks` (a V17 artifact), so a bootstrap at ANY
version carries it — a fixture whose migration-under-test must CREATE an
artifact the base also ships has to drop that one artifact itself
(single-migration knowledge no future bump can invalidate), or the test
silently pins the base's copy instead of the migration's. (3) the
genuine-shape fixtures came out STRONGER and cheaper: the v17 fixture now
proves V17→V18 redefines LIVE sync triggers (the registry version had to
assert them absent), and bootstrap-at-vN measured FASTER than
bootstrap-current-then-rollback (~80-130ms vs ~220-255ms + replay) — the
registry was never even a perf win.

**Coda (task-19045, 2026-08-20): the disclosed escape is closed the same way
tables were.** The MUT-INDEX shape the 16840 review left open (delete a
`CREATE INDEX` from a migration step; nothing reds — 84 of 94 named indexes
had zero Tests/ references) is now pinned by an absolute census
(`Tests/ChaChaNotesDB/test_index_census.py`): a hand-maintained literal of
all index names + UNIQUE flags + column tuples, asserted both directions
against a live DB, on both a fresh and a chain-migrated bootstrap — the
VALID_TABLES/TASK-864 pattern generalized. The mutation re-run confirmed the
division of labor: under the seeded index deletion the parity sweep stayed
green 36/36 (identity, as documented) while the census redded by name on
both variants. The trap to preserve: such a census only works as an
EXPLICIT literal — derive it from the same schema code it checks and it
becomes the identity too.

## A silently-shadowed upstream sentinel is a defect class, not a file-local bug (task-16502, 2026-08-15)

Textual 8.x removed `Select.BLANK` (the blank-selection sentinel, renamed
`Select.NULL`) — but referencing it does NOT raise: the lookup falls through the
MRO to `Widget.BLANK: ClassVar[bool] = False`, an unrelated render flag added in
the same major version. Every use of the old sentinel silently became the boolean
`False`: comparisons went permanently dead, and passing it as a Select's initial
`value=` crashed at mount with `InvalidSelectValueError: Illegal select value
False.` Task-565 (2026-07-25) established exactly this mechanism and swept it —
**scoped to settings_screen.py only**, because that was the file under review.
Three weeks later the identical construct in `console_model_popover.py` crashed
the Alt+M popover at mount for any session without a configured model, and was
reported by a user. A grep at that point found **66 remaining `Select.BLANK`
usages across 23 files**, including several sites that had independently
discovered the trap and worked around it locally with comments, and several that
deliberately exploit the `False` value as a synthetic placeholder option — so the
eventual sweep (task-16503) needs per-site classification, not find-and-replace.

**What to do.** When a fix reveals that an upstream rename/removal fails
*silently* (shadowed attribute, `getattr` default, `__getattr__` fallback) rather
than loudly, the first grep result count is the real scope of the defect. Sweep
repo-wide in the same arc, or file the sweep task immediately with the grep count
and the classification burden recorded — a Done task documenting the mechanism
does not stop the next file from shipping the same crash. Evidence here: the
mechanism was fully documented on the board for three weeks while the
user-reachable crash sat live in another file.
## A dodged flake can be the only visible symptom of a deterministic bug (task-15773, 2026-08-15)

Task-15478 hit a once-in-a-full-file-run flake in `ChapterEditorWidget`/`Select`'s
mount sequence when the chapter table populated ~999 rows in one reactive update,
and (honestly, documented as a dodge) reduced the test's chapter density until it
went 0/4. Task-15773 owned the flake and started, per the reproduce-first brief, by
stress-running the interleave -- 34 un-gated iterations across three shapes, zero
trips. What found it was a five-minute CHARACTERIZATION probe of what the code
deterministically does: `chapters = reactive([], recompose=True)` on a widget whose
`compose()` is static meant `watch_chapters` populated the current DataTable and the
scheduled recompose then threw that subtree away -- the settled table had **0 rows
after every single update**, in the minimal host and in the real STTS host alike
(`detected=13 table_rows=0` at HEAD). The "rare flake" was just the narrow-window
crash variant of a 100%-reproducible data-loss defect: the remount re-ran the
Select's Compose->Mount on every data arrival, and any teardown landing between the
fresh Select's registration and its Compose dispatch made its child-mount a silent
no-op (`_pruning`) while `Mount` still fired -- `NoMatches: No nodes match
'SelectOverlay'`. Once the mechanism was named, a gated `_on_compose` interleave
reproduced the exact exception on the first run, every run, and the fix (drop the
recompose; populate the persistent children in place) closed both the flake and the
always-empty table. Two halves to keep: (1) before stress-running a flake, spend one
probe characterizing what the code does deterministically under the flake's stimulus
-- the flake may be the tail of a bug whose body is fully reproducible; (2) a
repetition budget that finds nothing (34/34 clean here) is not evidence the race is
gone -- the gated one-run interleave was both stronger and cheaper.

## Re-verify a residual's CAUSAL hypothesis before building the fix around it (task-15778, 2026-08-15)

Task-15461's Implementation Notes recorded a residual with a cause attached:
the cold Read tab's wall-clock regressed "because the scoped path does the
CONTENT remount as its own discrete remove/mount pair rather than inside one
batched recompose -- Textual's `batch()` is the obvious next move." Task-15778
was filed around that hypothesis. A neutered-batch A/B on the same HEAD
refuted it: **zero** in-swap layout passes and zero compositor refreshes with
AND without `App.batch_update`, because the entire swap already runs inside
`_drain_surface_refresh`'s single `call_next` callback -- a paint-atomicity
that 15461's own `run_worker` -> `call_next` move had bought silently, one
task before it filed the residual blaming its absence. The batch shipped
anyway, but as an explicit contract (survives a future awaiting factory or a
drain restructure), documented as such -- not as the measured win the task
title promised. Two probe traps that nearly hid this: (1) counting layout
passes over the whole settle window attributed 3 post-swap passes (loader,
reseed) to the swap -- bracket the exact call under test, not the settle;
(2) the first probe "confirmed" the premise with numbers that were real but
belonged to a different mechanism. The residual's fix-shaped hypothesis is a
hypothesis; A/B the mechanism (here: neuter the proposed fix on the same
HEAD) before writing the Implementation Notes around it.

## "Nothing happened" cannot name WHICH guard stopped it — count the dispatch (task-15860, 2026-08-16)

Second occurrence in one arc, so it is a class rather than an accident. A
headless-wake test asserted the shipped behaviour "a wake into a busy session
never streams" by giving the loop a window and checking the provider double
recorded no payload. Mutating the guard it was written for -- bypassing
`send_refusal_copy` inside `ConsoleFleetWakeCoordinator._attempt` -- left it
**green**, because `submit_draft` refuses a busy session on its own. The read
site is double-guarded, so an absence-of-effect assertion is satisfied by
EITHER guard and can never say which one it is testing; the test claimed
coverage of the coordinator's gate while actually pinning the controller's.
(The viewless landing hit the identical shape earlier in the same arc: an
unguarded `_apply_world_info` survived because the applier was unreachable in
that rig AND wrapped in a broad `except`.) The repair is cheap and general:
count the DISPATCH, not the effect -- wrap the next seam (`controller.
submit_draft`) with a recorder and assert the list is empty, which fails the
moment the outer guard stops firing. Under the same mutation the repaired test
died with its sibling (2 failed); restored, 13 passed. Corollary for the other
direction: a mutation that leaves everything green is a finding about your
tests, not a nuisance -- both survivors in this arc were real gaps.

## A registry that self-heals on the next attempt is invisible to every test that takes another attempt (task-15860, 2026-08-16)

Mutating `_deliver` so delivered run ids never left the in-memory pending
registry killed exactly ONE test out of fourteen -- and not the exactly-once
test, which is the one whose subject it is. The reason: `_rows_for` drops any
run the durable ledger already shows delivered, so the leak is repaired by the
very next `_attempt`, and any assertion taken after a retry sees a healthy
registry. Only an observation taken at a moment when no further attempt is
coming can see it; here that moment was app exit (`ConsoleRuntime.dispose()`
mid-delivery). When a component has a self-healing path, the state it heals is
untestable through the normal flow -- so a test for it has to pin a TERMINAL
moment (quit, crash, teardown) on purpose. That is also the argument for
keeping such a test when it looks redundant next to the happy-path one.
---

---

## An unbounded wait default turns leaked test rounds into post-suite interpreter hangs

**TASK-16789, 2026-08-15.** After flipping the human-prompt timeouts to a
no-deadline default (ADR-067), `Tests/Chat/test_console_skill_script_confirm.py`
printed "1 failed, 28 passed in 7.49s" — and then the pytest process sat at 0%
CPU for 20+ minutes producing no output (the `| tail` wrapper hid everything
until exit). `sample <pid>` showed the main thread in
`wait_for_thread_shutdown`: the run was over, and the interpreter was waiting
for a non-daemon worker thread. The failing test's assert had skipped its
`resolve_pending_skill_script(...)` cleanup, leaving the confirm round armed;
with the old 120s default that leaked worker self-resolved at process exit in
≤120s (invisible), with no deadline its 1s poll loop never exits at all.

**What to do.** When a wait loop's default becomes unbounded, every fixture
that can arm a round must fail it closed on teardown — the file's
`make_controller` now sets `_shutdown_requested` after each test, which
resolves any still-armed round at its next poll. Diagnosis signature to
recognize next time: pytest's own timing says the suite finished but the
process idles at 0% CPU; macOS `sample` shows `wait_for_thread_shutdown`;
`kill -ABRT` (with `PYTHONFAULTHANDLER=1`) dumps the stuck thread stacks into
stderr.

---

## A cross-suite ordering failure can be an app KILLING ITSELF, not an object crossing the boundary (task-15860, 2026-08-16)

`Tests/UI/test_console_headless_wake_fires.py` +
`Tests/UI/test_console_store_continuity.py` run together gave **1 failed, 4
passed**; each file alone was green. Every hypothesis on the obvious list was
about something *surviving* the test boundary — an undisposed app-owned
`ConsoleRuntime`, a pending delivery, a leaked DB handle, a module singleton,
a daemon thread. All of them were wrong. Four *identical* wake rounds in one
process were green, and the two poisoners followed by a plain no-wake nav probe
were green: nothing accumulated. What actually happened was that the THIRD
app killed itself — a `console-sync` worker whose screen had been closed raised
`NoMatches`, Textual's default `exit_on_error=True` handed it to
`App._handle_exception`, and from then on every `post_message` was silently
dropped, so the next `NavigateToScreen` produced 15 seconds of total silence
and "stuck on LibraryScreen". The prior tests contributed timing pressure, not
state.

**What to do.** Before hunting for the leaked object, ask whether the victim
app is still ALIVE: dump `app.is_running` / `app._closing` / `app._closed` /
`app._exception` at the point of the symptom. A dead Textual app is
indistinguishable from a hung one from the outside — the message queue is
empty, the workers list is empty, the loop is running, and nothing logs. Two
corollaries that generalise: (1) `is_mounted` stays **True** for a screen
Textual has already closed (the removed surface reported `is_mounted=True`
with `is_running=False` and no children), so a mount check is not a liveness
check — `_closing`/`_closed` are; (2) a per-file green gate structurally
cannot see this class, because the damage needs several app lifetimes in one
process. Running the whole directory in one invocation is what surfaces it.

## A coroutine that re-arms itself from `finally` escapes the framework's teardown sweep (task-15860, 2026-08-16)

Textual cancels a node's workers in `Widget._on_unmount`
(`workers.cancel_node(self)`). `ChatScreen._sync_native_console_chat_ui`
re-armed itself with `self.run_worker(...)` inside its own `finally` — which
runs *after* that sweep — so the worker it created was never in the cancelled
set, ran a full DOM sync against a screen with no children, and killed the app.
The instrumentation that named it in one pass: wrap `DOMNode.run_worker`
filtered to the suspect group and log `traceback.format_stack()` at creation;
the creating frame was the `finally` itself. Generalises to any self-scheduling
loop (timers re-arming timers, callbacks re-posting themselves): the framework's
"cancel everything this node owns" happens once, and anything scheduled after
it is invisible to it. Guard the re-arm on the owner still being alive, not
just the body.

## A `MagicMock(spec=Cls)` answers every METHOD truthily — a new guard predicate must not be one (task-15860, 2026-08-16)

A teardown guard was added as `ChatScreen._console_screen_torn_down()`, reading
`_closing`/`_closed`. Three `Tests/UI/test_ui_responsiveness.py` tests that
drive `ChatScreen._sync_native_console_chat_ui(mock)` against a
`MagicMock(spec=ChatScreen)` went red: the spec'd mock auto-provides every
method in `dir(Cls)`, and the auto-returned `MagicMock` is TRUTHY, so the new
guard reported "this screen is torn down" for every mocked screen and the code
under test returned before doing anything. Measured three ways: 15 passed at
the pre-fix baseline, 3 failed with the method form, 15 passed with the
identical logic moved to a module-level `_console_screen_is_torn_down(screen)`.
The reason the module form is immune is the same mechanism read the other way —
`_closing`/`_closed` are set in `__init__`, so they are NOT in `dir(Cls)`, a
spec'd mock raises `AttributeError` for them, and `getattr(screen, "_closing",
False)` correctly reads a mocked (or never-mounted) screen as LIVE.

**What to do.** When adding a *predicate* that new early-returns depend on, ask
what a spec'd mock of the host class will return for it before choosing where
it lives. A module function reading raw attributes is the mock-safe shape; a
method is not. The failure is nasty because it is silent — the guard fires, the
body is skipped, and the assertion that fails is about something else entirely.
Trap-detection note: neutralising the method's BODY does not restore the tests
(the mock never calls it), so the usual "mutate the fix off and compare" check
reports "identical failure sets, not mine" — the only honest discriminator is a
real pre-fix baseline worktree.
## A parity test that passes against the pre-fix tree proves nothing (TASK-16811, 2026-08-16)

The first version of `test_focus_token_parity.py` asserted a selected
NavigationButton's resolved background equals the transcript's selected-row
colour — and passed both post-fix AND against the unfixed tree. Two masks
stacked: `run_test()` auto-focuses the first focusable widget, and the app
bundle's generic `Button:focus { background: $ds-focus-bg }` rule (app tier
beats any DEFAULT_CSS rule) painted the canonical colour over the shadowed
`.active` rule the test meant to probe. The divergence only exists on the
UNFOCUSED active state. The test became meaningful only after blurring
(`app.set_focus(None)`, plus asserting `focus` absent from the pseudo-class
set) — verified by running the corrected test in a throwaway worktree at the
pre-fix commit, where it finally failed. Rules: (1) a regression test for a
visual fix is only evidence once it has been RUN against the pre-fix tree
and observed red there; (2) any style probe on a widget mounted first in a
test App is probing the focused state whether you meant it or not.
## A bare scroll_to(max) walk is not a user gesture — it self-terminates the moment the boundary stops moving

**TASK-16851, 2026-08-16.** The head-pinned-selection fix (refuse tailward
hydration while over the high mark with a blocked prune) passed its stall pin
but "failed" its Esc-recovery pin: after Esc unblocked the prune, an 80-round
`scroll_to(y=max_scroll_y)` walk never advanced a single chunk. Probe: reader
parked at exactly `scroll_y == max_scroll_y`, so every subsequent `scroll_to`
produced NO scroll_y change — `watch_scroll_y` never fired, nothing scheduled
hydration, and the loop measured the harness gesture, not the product. Every
REAL input path (wheel-down, PageDown, End) has its own boundary hook and
recovered immediately. The pre-existing two-sided walks had only ever worked
because hydration kept GROWING max_scroll_y under them, re-arming the watcher
each round — a walk test that relies on that is green only while the feature
under test keeps moving the goalposts for it.

**What to do.** Drive boundary-walk tests with the product's real gestures
(`action_page_down()`, wheel events, `scroll_end`) — or at minimum pair the
positioning `scroll_to` with one. Before concluding a recovery path is broken,
check whether the loop's gesture can still produce a state change at all.

Same task, implementation twin worth remembering: a decision that walks
`self.children` (the hydration refusal reusing `_compute_prunable_prefix`)
must run under the widget's reconcile lock — read mid-reconcile, the transient
child order faked a "blocked prune" and stalled a selection-free End drain
(218 messages stranded in the born-red End-race pin).

## A guard added by a later ADR can hollow out an older test without turning it red (task-15860, 2026-08-16)

`Tests/Chat/test_console_runtime_lifetime.py`'s two AC#2 approval pins —
"leaving Console denies a parked approval round" and "a round from the
previous visit is not resurrected" — build the controller with
`app is None`. That was fine when they were written. Then ADR-067 added a
no-`app` guard to `request_mcp_approvals` that denies every name on the spot,
and from that moment the rounds never reached the poll loop at all: both tests
passed on the guard's verdict, not on the cancellation signal they claim to
pin. Measured while mutation-testing a change to that exact signal: with
`_is_session_cancelled`'s visit check deleted outright — fail-open for every
session-scoped round — the whole file was still **14/14 green in 0.98s**.

**The tell was the clock.** A file whose tests are supposed to poll on a 1.0s
granularity cannot finish in less than one poll interval. After wiring a
`call_from_thread` app the same file takes 2.81s and the same deletion fails
both pins.

**What to do.** When you change a signal, mutation-test the OTHER files that
claim to pin it, not only your own — a green neighbour is not evidence.
And when a suite that exercises timed waits runs impossibly fast, that is a
finding, not good luck.

## pytest silently drops a directory argument when a file inside it is also listed (task-15860, 2026-08-16)

A gate invocation passed `Tests/Agents/` *and*
`Tests/Agents/test_agent_runs_wake_ledger.py` (the second arrived from a
separate "wake suites" list). pytest collected **283** tests instead of
**1,733** — the directory arg was collapsed against the more specific
file — and reported a perfectly healthy `282 passed, 1 skipped`, exit 0.
Nothing in the output says a thing was skipped; the only evidence is the
count, and 282 looks like a normal number.

**What to do.** Never pass a directory and a path inside it in the same
invocation. And "READ every count" means read it against what you MEANT to
run: `--collect-only -q | tail -2` on the exact argument list first, then
compare. A count you have not predicted cannot be checked.

## Textual's `run_test` disables notifications, so a toast assertion can never see a toast (task-15860, 2026-08-16)

`App.run_test()` defaults `notifications=False`, which sets
`_disable_notifications` and makes `Screen._extend_compose` skip the
`ToastRack` entirely. A test asserting on rendered toast widgets fails
forever under the default; a test asserting on `app._notifications` passes
without proving anything reached a screen. Pass `notifications=True` and
assert on the widget.

Second trap in the same assertion: `Toast` is a `Static` that never calls
`update()`, so its `renderable` is empty — a helper reading `renderable`
reports "no toast" for a toast that is on screen. Read `Toast.render()`.

## Do not commit to a file the running suite imports — `inspect`/`linecache` read source off disk lazily (task-15860, 2026-08-16)

A 59-minute single-process Console population (3,404 tests) came back with
**4 failures unique to the branch**, all in
`test_console_prompts_controller.py::test_screen_keeps_a_real_delegation_for_
every_outside_caller[...]` — a test that does
`inspect.getsource(getattr(ChatScreen, name))`. Its message showed it had read
a *different method's* body entirely.

The cause was a comment-only commit to `chat_screen.py` (net +7 lines at line
14903) made **while the run was in flight**. Each method's `co_firstlineno` is
fixed at import; `inspect.findsource` calls `linecache.checkcache`, re-reads
the now-changed file, and every method defined below the edit reports source
shifted by the delta. The tell was not the assertion text but the SPLIT: the
two parametrisations that passed are defined at lines 5330 and 6264, the four
that failed at 16684, 16790, 16794 and 17198 — a clean line-number boundary at
the edit point. Re-run on a stable tree: 37 passed.

**What to do.** While a long run is in flight, stage edits somewhere the run
does not import, or wait. This bites any assertion built on `inspect.getsource`
/ `inspect.getsourcelines` / traceback rendering — and those are exactly the
architecture-contract tests that a big single-process gate is there to run.
## A born-red run that dies on ImportError is not born-red evidence (TASK-16838, 2026-08-16)

The in-flight-guard test file imported the new `_IN_FLIGHT_URL_CHECKS`
registry at module top. Run against the pre-fix tree (worktree at
`1af8c0414`) it "failed" — but on collection, with `ImportError: cannot
import name '_IN_FLIGHT_URL_CHECKS'`. That red proves only that the test
mentions a symbol the fix adds — the same red a typo would produce — and it
says nothing about whether the bug (the 15764 double-check interleave) is
reproduced or the assertions could catch it. Rewritten with a lazy
`getattr(svc, "_IN_FLIGHT_URL_CHECKS", set())` lookup so the file COLLECTS
on both trees, the pre-fix run reddened on the behaviour itself: the manual
entrant's gated fetch fired while the scheduled fetch was still in flight
("went to the network too"), the exact double-report the review had
demonstrated. Rule: when new-code symbols would make a born-red file
unimportable at base, reference them lazily (or split the white-box asserts
out) so the base-tree run fails on the assertion that carries the evidence,
not on `import`.

---

## A per-tick view value needs its CACHE KEY and its SCOPE mutation-tested; display assertions see neither (turn-activity line, 2026-08-16)

The Console's in-flight assistant row gained a live activity line (`⚙
read_file · 4s`) refreshed on the 0.2s poll. Every display assertion was
green, and mutation testing then found two defects that no rendered-text
assertion could have seen:

1. **The cache key.** `ConsoleTranscript` has TWO renderers — markdown (the
   default for assistant rows, which carries the line in its *header*) and
   plain. `_message_row_signature` is built from the PLAIN renderer only, so
   the markdown row's elapsed advanced solely as a side effect of the plain
   renderer embedding the same string. Disabling the plain branch left the
   first paint correct and froze every later tick — the tell was not "the
   line is missing" but "the FIRST tick passed and the SECOND did not".
2. **The scope.** Stamping the value on every message instead of only the
   in-flight row changes nothing a reader can see (a row with content never
   renders it; only assistant rows can) — but it lands in every row's
   signature, so the whole transcript re-derives and re-syncs once per
   second for the entire turn. The mutant SURVIVED a suite of rendered-row
   assertions.

**What to do.** For any value the poll re-supplies each tick: (a) mutate the
signature/cache key and require a test that paints two ticks differing in
*nothing but that value*; (b) mutate the scope and assert **blast radius**,
not pixels — `row_render_signatures()` and
`message_signature_compute_counts()` make "exactly one row moved" a direct
assertion. Also worth knowing for this widget: a signature that renders one
of two renderers silently couples them, so name the field in the signature
outright rather than relying on it riding along inside rendered text.
## A guard sitting behind an earlier early-return is unreachable, so no fixture can own it (task-15860, 2026-08-16)

Third mutation-survivor in this arc, and a different shape from the two
above (which were two guards in SERIES at one read site). The launch-wake
loop skips a marked conversation that owes nothing —
`if not wake.has_pending(cid): continue` — and mutating that line to
`if False:` left the whole suite **green**. Investigating rather than
patching around it: every test that exercised an unowed mark used exactly
ONE mark, and with one unowed mark the function returns earlier, at
`if not wake.seed_from_marks(): return 0`, before the loop runs at all.
The guard was not weakly tested, it was *unreachable* for every fixture in
the file, so no assertion anywhere could have distinguished "we checked
each conversation" from "we never got that far". The repair is a fixture
change, not an assertion change: a test with TWO marks, one owed and one
not, gets past the earlier return and then asserts the unowed conversation
was never hydrated. Under the same mutation it now dies alone (1 failed,
8 passed). Rule: when a mutation survives, before touching assertions ask
whether the mutated line *executes* under any fixture you have — an
earlier `return` upstream of it is the commonest reason it does not, and
it is invisible in the diff you are mutating.

## A "constructs nothing" pin needs an observer the production code cannot lie to (task-15860, 2026-08-16)

The owner's ruling on wake-at-launch required that an install with no
background work pay one indexed read and build NOTHING, so startup stays
byte-identical. "Nothing was constructed" is exactly the claim a weak test
states and never checks, so the pin took four independent observations:
the marks service's call list, the four `ConsoleRuntime` slots being
`None`, no `deferred_launch_wake` task ever created — and **the absence of
the `agent_runs.db` FILE on disk**, because constructing the agent bridge
opens (and creates) it. The filesystem one is the observation that cannot
be satisfied by a mock, a stub or a lazily-`None` attribute. It earned its
place under mutation: removing both empty-marks guards was caught by the
call-count and by a sibling test's runtime assertion, and removing only
the outer guard was caught *solely* by the task-name observation — the
`None` slots stayed `None` because the inner function had its own guard.
Two guards in depth meant no single observation covered both mutations;
the four together did. Corollary: a no-work pin also needs a control that
runs the same probes WITH work present and watches every one flip,
otherwise a hook that never runs at all satisfies it perfectly.

## Your test's own harness can make the guard you are testing unreachable (task-15860, 2026-08-17)

The close-out gate had to prove one invariant no per-landing test owned:
deliveries are serialized **app-wide**, enforced by one line in
`ConsoleFleetWakeCoordinator._attempt` — `if self._delivering is not
None: return`. Two successive drafts of that test passed, and **survived
neutering that exact line**. Both were worthless, for two different
reasons, and both reasons are general:

1. **The first draft used one conversation.** A second completion in the
   same conversation is refused by the *per-session busy* gate several
   lines earlier, so `_delivering` was never the thing under test. A test
   of gate N must construct the state where gates 1..N-1 all pass —
   otherwise it is a test of gate 1 wearing gate N's name.
2. **The second draft used two conversations and still survived**, because
   the observation was "no second payload reached the provider". The
   provider double stalls in its readiness probe, and the stall belongs to
   the GATEWAY, not to a turn: with the guard removed, the second wake
   turn genuinely started and then parked at the same stall, streaming
   nothing. The two outcomes — "refused" and "started, then blocked
   identically" — are indistinguishable at the observation point the test
   was reading.

The fix was to count *entries into the readiness probe*, which separates
"a turn started" from "a turn produced output". The mutation then killed
the test immediately.

**The rule:** when a mutation survives, do not first suspect the
assertion's strength — ask **what the harness itself does to the code path
after the mutated line.** A shared blocking double, a fixture that stops
upstream, a stall that is global rather than per-attempt: each converts
"the guard fired" and "the guard did not fire" into the same measurement.
Pick an observation that is downstream of the mutated line but *upstream*
of whatever the harness blocks on.

## Measure the invariant, then write the assertion — the honest answer may not be the one the plan states (task-15860, 2026-08-17)

The same gate had to pin "exactly-once across a restart mid-commit". The
plan and the shipped User Guide both asserted the strong form: a restart
between a wake being accepted and the app exiting re-announces nothing.
Rather than encode that, the test was written to *measure* first — die
inside the window (the ledger stamp raises, leaving rows committed and the
ledger unstamped, which is byte-identical to a process kill there), then
relaunch and read what the conversation holds.

It holds **six** rows, not four: the same child result announced to the
supervisor twice, and paid for twice. `_deliver`'s own comment predicts
it ("a lost stamp risks one re-announce at a later claim, never a lost
result"); the user-facing doc had quietly promised more than the code
does. The live pass then reproduced it by accident — an app quit while a
wake turn sat blocked produced exactly one duplicate notice at the next
launch.

Two things followed, and both are the point. The doc was corrected to the
measured behaviour. And the test asserts the **bound** (at most one
re-announce, the row shape, no USER row on any of it, and that a third
launch adds nothing) rather than the measured number — so closing the
window later is an improvement, not a test failure. **Encoding a plan's
claim as an assertion turns an unverified sentence into a fixture that
future work must preserve.** Measure, then decide which part is the
invariant and which part is merely today's value.
---

## A fixture keyed to the code's invented config section hides a total production failure (task-17382, 2026-08-17)

`summarize_with_llama` indexed `loaded_config_data["llama_api"]` in ten places.
No such section has ever existed — the loader builds `llama_cpp_api` — so every
llama.cpp summarization raised `KeyError` before contacting a server, and the
`except` at the bottom returned an error STRING rather than raising. The
deep-search caller tested `summary.startswith("Error:")`, which no
provider-prefixed message matches, so `"Llama: Error occurred while processing
summary with Llama: 'llama_api'"` was stored AS the result's evidence content
and the synthesis was built from it. Citation verification kept passing because
it matches quotes against `original_content` first, so the reports were graded
sound while the model had never been shown its sources.

The reason this survived a security review of that very file:
`test_summarization_diagnostic_privacy.py`'s fixture stubbed the settings dict
with a `"llama_api"` key — the name the summarizer had invented. The tests fed
the code its own mistake and passed. The same fixture stubs `api_keys` and
`local_api_ip`, which is exactly why the Kobold and TabbyAPI summarizers'
identical defect (task-17383) also stayed invisible. Fixing the code then broke
those tests, which is the only reason anyone looked.

**What to do.** A fixture standing in for configuration must be keyed to what
the LOADER produces, not to what the code under test reads — those are the same
string only when the code is right, and a stub that mirrors the code's
assumption can never fail. When you fake a provider response, fake what the
SERVER sends: my own first fake returned llama.cpp's native `{"content": ...}`
shape, which is what the buggy parser read, so it passed while the live
endpoint (`/v1/chat/completions`, `choices[0].message.content`) returned "No
choices in response data" on every call. Cheapest check available: print the
real `load_settings()` keys once and compare, or assert the section exists.

---

## A metric can be graded on fallback content, and nothing in it says so (task-17370, 2026-08-17)

Every live research baseline recorded in this repo reports
`citation_accuracy 1.00` and healthy `claim_support_rate`. All of them were
measured with per-result summarization failing: first instantly (wrong config
section), then a 404, then an unparseable payload, and once those were fixed, a
timeout at exactly the shipped 30s per call on a local 27B. Each failure fell
back to raw source text, which is the CORRECT degradation — and completely
invisible in the metrics, because a report built from source text still
resolves its markers and still verifies its quotes.

The tell was uniformity: six summarizations completing in exactly `30.0s` is a
timeout, not a latency distribution.

**What to do.** When a pipeline has a degradation path, a metric that only
grades the OUTPUT cannot tell you which path produced it — so record the path
alongside the number (which stage ran, which fell back), and treat suspiciously
round, uniform timings as a budget being hit rather than work being done. Also:
absence of an error log is not evidence of success when the code logs successes
at INFO through stdlib `logging`, whose default level hides them; the runs above
showed zero "Summarization successful" lines whether they worked or not.

**Second instance, and the sharper rule when the number is a DELTA (TASK-16965,
2026-08-17).** Same shape, opposite tell — and no tell at all. TASK-16965 had to
answer "does cross-encoder reranking help retrieval here?" by running the gated
eval set twice, once reranked and once not, and reading the difference.
`CrossEncoderReranker` honours the TASK-3502 contract: a model that fails to
load DEGRADES (returns the caller's ordering untouched) rather than raising. And
`Tests/conftest.py` sandboxes `HOME`, while
`huggingface_hub.constants.HF_HUB_CACHE` is computed from `expanduser("~")` **at
import** — so under pytest `CrossEncoder(...)` raises `OSError` ("couldn't
connect ... and couldn't find them in the cached files") on a machine where the
model IS cached. Measured directly, before the probe was written. Compose those
two facts: every window comes back in its original order, every metric is graded
on un-reranked output, and the before/after table reads a flawless **0.000 delta
on all 105 cells** — a NULL result, publishable-looking, pre-registered as an
acceptable outcome, and entirely fabricated. Unlike task-17370's uniform `30.0s`
timings there is no tell whatsoever: a real null and a never-ran null are the
same table. The run therefore repoints the constant
(`monkeypatch.setattr(constants, "HF_HUB_CACHE", real_cache)` — hf_hub 1.x reads
it at call time off the module attribute) and **asserts the work happened**:
`rows_scored > 0` and `rows_failed == 0`, per pass. It scored 3,621 rows, 0
failed, and moved 1,950 — which is what makes the verdict it did produce
(HARMED, bimodal) mean anything at all.

**What to do.** Recording the path is enough when a bad path makes the number
look good; it is NOT enough when the measurement is an A/B and the subject
degrades to the identity, because then the failure mode is the null hypothesis
itself and no reader can tell the two apart. So: **a measurement whose subject
degrades silently must assert, inside the run, that it did work** — a positive
count of units processed and a zero count of failures — or its null is
unfalsifiable and must not be published. Write those assertions BEFORE you look
at the numbers; a 0.000 delta is the one result that never prompts anyone to go
looking for a bug. Corollary worth its own grep: the frozen-at-import
huggingface_hub constants bite in more than one place — `HF_HUB_OFFLINE` (see
"HF offline enforcement must be set before `huggingface_hub.constants`
EVALUATES" above, where the blast radius is an unwanted download) and
`HF_HUB_CACHE` (here, where the blast radius is a load you wanted and silently
did not get, under any fixture that moves `HOME`).


## When you find one inert declared surface, enumerate its whole namespace

**TASK-16174 / TASK-17600, 2026-08-16..18.** Three separate arcs each found
one config surface that was declared, switchable, sometimes documented — and
implemented by nothing:

1. `include_parent_docs` / `parent_size_threshold` /
   `parent_inclusion_strategy`: shipped, set to `true` by three profiles,
   **read by nothing** (TASK-16174 retired them).
2. `result_reranking`: a middleware declared with `enabled = true`, listed by
   the `high_accuracy` pipeline, handled by a bare `pass`.
3. `reranking_strategy`: a config key with **zero readers**, which
   TASK-16965's own design doc simultaneously told users was the lever for
   selecting a reranking strategy.

Each was found by accident, while doing something else. Nobody looked for the
CLASS until the third one — and when TASK-17600 finally enumerated the
namespace instead of the single filed name, `result_reranking` turned out to
be **one of eight**: eleven middleware names were declared by pipelines and
four implemented, with seven falling off an `if/elif` and no-opping silently.
Two entire pipelines (`technical_docs`, `research_papers`) consisted of
nothing but unimplemented middleware, and three names referenced no
definition block at all.

**What to do.** The first inert surface you find is a sample, not the
population. Before closing, enumerate its whole namespace **in both
directions** — declared-but-unimplemented AND implemented-but-undeclared —
and write the enumeration as a test rather than a one-off grep, because the
grep answers today and the test answers forever. Give that guard a
self-check (`test_the_guard_can_see_the_names_it_is_guarding`): a namespace
guard whose parser silently stops matching becomes a green test that
guarantees nothing, which is the same failure it was written to prevent.

**A corollary this cost us directly:** a doc can *create* the surface. The
`reranking_strategy` claim was written by the arc that measured the feature,
in the same commit series that carefully documented everything else
truthfully — so include documentation in the sweep, and check that the lever
a doc names is one the code actually reads.

## A migration test that pins the current version number breaks on every later migration

**The trap.** A migration test asserts what a *fresh* database looks like — and pins
the schema version (or a table-set delta) as an exact literal. The assertion is true
the day it is written and false the day the NEXT migration lands, in a test file
nobody touches when they add their own migration.

**What happened.** Task-17169's v39→v40 bump found three such tests **already red on
dev**: the v37→v38 trajectory tests asserted a fresh DB is `== 38`, and
`test_current_schema_version_is_39` had been left behind — the v38→v39 landing had
broken them and shipped anyway (nobody runs another feature's migration tests when
adding a schema version). The v40 bump then broke the visual-identity contract the
same way twice over: its `== 39` pin, and an *exact-equality* table-set delta
(`tables_after - tables_before == VISUAL_IDENTITY_TABLES`) — an upgrade from v38 runs
*every* later migration, so v40's new table legitimately appeared in the delta.

**The rule.** In a migration test, a version literal is only correct at the seeded
*starting* point. Everything asserted about the *end state* must be version-relative:
fresh/upgraded DBs assert `== CharactersRAGDB._CURRENT_SCHEMA_VERSION`, and a
"migration X added tables T" claim is a superset check (`T <= delta`), never
equality. If you are bumping the schema, run `Tests/DB/` and `Tests/ChaChaNotesDB/`
in full — the tests your bump breaks are not in your feature's test files.

**The same trap applies to fixture writes.** TASK-18932's final rebase raised the
current schema from v51 to v52, then `test_chachanotes_full_capture_migration.py`
failed before its v50→v51 assertion: it seeded a genuine v50 database through the
current `add_conversation()` method, which now correctly writes a v52-only column.
A historical migration fixture must seed rows with SQL limited to columns that
existed at its pinned starting version. Current production CRUD is valid only after
the database has migrated to the current schema; using it to populate an older
fixture makes an unrelated future column addition break the fixture before the
migration under test can run.

## Removing a widget's border box activates the global focus outline on it (task-17651, 2026-08-17)

**What happened.** Flattening the Console composer to a one-row dense-form bar
(border box → `border-left` only) shipped green through every style-level
assertion — `styles.border_left == ("thick", …)`, others empty — and then the
painted row-map probe showed the row rendering `┌─ Composer ▾ …`: corner glyphs
overpainting the bar's first two cells. The computed border styles were
CORRECT; the glyphs came from `core/_reset.tcss`'s global `*:focus { outline:
solid … }`, which had been landing on the focused composer all along but was
absorbed invisibly by the old border box's padding rows. The transcript had the
same latent hit: with its border removed, focusing it would have drawn the
outline over its outermost CONTENT rows.

Two mechanics worth keeping:

1. **A border removal is also an outline activation.** The reset rule's own
   comment documents that it obscures widgets that draw content on their
   perimeter (the TASK-1160 DataTable case) — but the trap here is the inverse
   direction: a widget that was previously SAFE becomes obscured the moment its
   border/padding buffer is removed, with zero diff to any focus rule. When
   removing a border box, grep the resets for `:focus` and add the
   `outline: none` opt-out with a replacement cue in the same change.
2. **Style probes cannot see this class of defect at all** — outline is not
   border, and `styles.border_*` reads stay pristine while the paint is wrong.
   Only the compositor row (`render_strips()`) showed it. The composer focus
   test now pins the painted first cells (`│`/`█`, never `┌─`) alongside the
   style reads; assert the paint whenever the mechanism under test is "what
   the user sees at this cell".

**A self-inflicted corollary:** the fix (`outline: none;`) then tripped this
arc's own freshly written pin `assert "outline:" not in focus` — written
minutes earlier to mean "no focus outline". Ban the specific values
(`outline: solid`, `outline: heavy`) and PIN the opt-out explicitly; a
substring ban on the property name bans the cure along with the disease.

## A bundle-less harness does not just hide styles — it changes LAYOUT MODE (task-17660, 2026-08-18)

**What happened.** Two Settings toggle tests went red on dev and looked like a
regression from recent merges. The real chain: the Console Behavior card's
stylesheet rule is `height: auto`, but the test harness (`DestinationHarness`,
a `ConsolidatedCSSApp` with no bundle) never loads it — so the card fell back
to the container default and was CLAMPED to a fraction of the pane. When two
new sections landed in the card (Status row placement, Selection side chat),
its content grew past the clamp; the paste checkbox laid out beyond the card's
box, sibling detail rows painted over its coordinates, and `pilot.click`
missed silently — `clicked` came back `False`, nothing staged, and the Save
button truthfully reported "no changes". A bundled probe against the same
build showed the card auto-growing to its full height with the control
reachable by ordinary scrolling: **no production defect existed**.

Three mechanics worth keeping:

1. **Missing CSS can flip a container from auto-sizing to fraction-sizing.**
   That is a different failure class from "the margin/padding I assert on is
   absent": the whole layout topology changes, children overflow their
   parent's box, and hit-testing lands on unrelated widgets. A geometry or
   interaction test whose subject sits deep in a card is meaningless without
   the bundle.
2. **`pilot.click` misses are silent unless you assert the return value.**
   The click "succeeded" as far as the test flow was concerned and the
   failure surfaced two steps later as a missing toast — assert `clicked` at
   the click site so the failure names the real problem.
3. This is the third distinct incident of the bundle-less-harness trap
   (`ConsoleHarness` could not see the phantom chips margin in task-17650;
   the composer padding in the same audit; now layout-mode clamping here).
   When a mounted test drives clicks or asserts geometry, subclass the
   harness with `CSS_PATH = BUNDLED_STYLESHEET` — and treat a red mounted
   test in a bare harness as unattributed until reproduced under the bundle.

## A `0.000` from a seam the harness never wired reads exactly like a real negative (TASK-17855/18255, 2026-08-18)

TASK-17855 censused the RAG eval harness's residual zero-row queries and
reported a **production defect**: the Library's plain-mode prompts sub-leg
returned zero rows for all five `prompt` golden queries, including one whose
target contains *every* content word of the query, and `prompts_fts` indexes
the matching column. Every term present, every term indexed, zero rows back
— the conclusion looked forced.

It was wrong. The harness's fake app sets `prompt_scope_service=None`, so
`_search_prompts` returns `(False, [])` — a seam reporting itself
**unavailable**. Production wires it (`app.py:5682`). The metrics table
renders "not measured" and "measured, found nothing" as the same `0.000`,
and I read the second.

**Four written warnings sat in the tree, all unread**: the harness's own
comment directly above the line (*"Leaving it None means the harness's plain
column reports 0.000 for prompts while the shipped app's plain mode does
find them"*); the B2 plan twice (*"the Library four-seam path already
searches prompts its own way — do not touch it"*); and — most pointedly —
the eval README, which says in as many words: ***"Do not read a `prompt`
0.000 as a prompts-retrieval defect."*** I filed that exact defect.

That fourth one carries its own sub-lesson, because the README's *reason*
was wrong even though its conclusion was right: it attributed `plain`'s
0.000 to prompts having no vector index, and `plain` never consults a vector
index. A warning with a wrong rationale is weak protection — the rationale
is what a reader checks against their case, and mine visibly did not apply,
which made the warning easy to step past. **When you write a "do not read X
as Y" note, the reason has to be correct per-mode, or it invites exactly the
misreading it forbids.**

**This is the dual of the trap already recorded here.** The known family is
*the intervention silently did not take* — the monkeypatch bound at import
so both arms ran identical code; the probe asked for `body` when the field
was `content` and reported `match=0` everywhere. The fix for those is a
probe-proof line: make the probe prove it did work. **That fix cannot catch
this one**, because the probe did run, did execute the real code path, and
did read a real number. The instrument was honest; it simply could not see
the thing.

So the check is a different one, and it happens *before* the measurement is
interpreted:

- **A zero is a result only if the instrument was wired to produce a
  non-zero.** Establish that first — census the seams, dependencies, and
  fixtures the metric flows through, not just the rows it returned.
- **Distinguish "unavailable" from "empty" at the source.** `(False, [])`
  and `(True, [])` mean opposite things and collapse to the same aggregate
  one layer up. When a seam can report unavailability, the harness should
  surface that as a distinct cell value (or fail loudly), never as zero.
- **Read the comments on the construction you are measuring through.** A
  deliberately-stubbed dependency is usually documented at the stub, and
  that comment is the cheapest possible refutation of a defect claim.
- **Before filing a defect from an aggregate, reproduce it against
  production wiring.** One grep for the attribute in `app.py` would have
  ended this in under a minute.

**The same seam collapses a THIRD state into that zero**, found by a reviewer
on the correction PR: `_search_prompts` ends `except Exception: return True,
[]`, so a seam that *threw* is reported as available-and-empty. Wiring the
dependency therefore does not make the metric unambiguous — a zero still
means no-match **or** threw. When a function's return type encodes status
(`(bool, list)`), check what the exception path returns before trusting
either value; a `warning` log is not a substitute, because nothing reads it.

The general form: **any code path that converts a failure into a
well-formed empty result destroys the distinction the caller needs.** Grep
for `except` blocks that `return` an empty container whenever a measurement
built on them surprises you.
## A gate built in halves is no gate — and sound-looking test deviations can hide exactly that (TASK-18705, 2026-08-18)

The `.SKILLS/` import feature had prompt-gating (kill-switch, permanent "Never",
fingerprint re-offer) specified for BOTH its triggers. Task 2 built the gating
functions; Task 4 wired them into the startup trigger; Task 5 wired the
workspace-create trigger straight past them — every per-task review passed,
because each task correctly built its own half. The final whole-branch review
found the create trigger honored none of the gates, while a checked AC and
three documents promised it did. Compounding it: the live-verification pass
had deliberately used a FRESH fixture for the create-trigger scenario, with
locally sound reasoning ("the 'Never' from the previous scenario would
correctly suppress the offer") that ASSUMED the cross-trigger gate existed —
the deviation dodged the exact broken case. Rules: (1) when a contract spans
tasks, some review must check the CONNECTION, not the halves — that is what
the whole-branch review is for; never skip it because per-task reviews were
clean. (2) A test-plan deviation justified by assumed behavior of the thing
under test is a red flag: the assumption is the test.

## pilot.pause() is a CPU-idleness heuristic, not a queue drain — every assert-after-pause races under machine load

**TASK-16842, 2026-08-16.** `Tests/UI/test_stts_profile_library.py` carried a
five-test flake family two reviews had hit (3 failures normally, 14 in one run
at 2x machine load; one test reproduced standalone: `assert None is not None`
on `app.focused` after `_wait_until` had confirmed the button was *mounted*).
A full-file run alongside 14 CPU burners reproduced exactly the two recurring
victims on the first try — `app.focused` was None while the export-choice
modal's buttons were already queryable. Three stacked mechanisms, all
load-sensitive and none a product bug:

1. **`Widget.focus()` is deferred.** It schedules `screen.set_focus` via
   `app.call_later` — one pump callback AFTER the widget becomes queryable.
   Mounted is not focused. (Not user-visible: the callback is FIFO-ordered
   before any input that arrives after the modal is on screen.)
2. **`pilot.pause()` is not a settle.** With a delay it is a bare
   `asyncio.sleep`; with none, Textual's `wait_for_idle` compares process CPU
   time to wall clock — an external load starves the process, which then
   *reads as idle while its message queue is still full*. So a fixed
   attempt-count wait (`100 × pause(0.01)`, ~1s nominal) exhausts, and any
   single-pause-then-assert idiom (selection landing via `RowSelected`,
   availability projection, focus fallback) samples pre-settle state. Under
   load the victims rotate — which is why the family looked like five
   unrelated tests.
3. **A disabled `Button` silently swallows `press()`.** One-shot presses
   issued after a heuristic pause (e.g. Continue right after toggling the
   consent checkbox that enables it) are lost forever; no downstream wait can
   recover them, so the failure surfaces as an unrelated timeout.

**What to do.** Bound waits by monotonic wall clock, not attempt count; poll
the actual asserted condition (`app.focused.id == ...`, `_selected_profile is
not None`, `not button.disabled`, label != "Checking"), never a mounted-state
or pause proxy; settle `not disabled` before any programmatic `press()`. After
the class fix: 10/10 full-file runs green (4 under the same 14-burner load)
plus 5/5 standalone runs of the old reproducer.

## A tiered design's boundary must be probed with a shape the tiers disagree on (task-16839 fix round, 2026-08-20)

Task-16839's first revision computed its change ratio with an order-sensitive
`SequenceMatcher` alignment up to 4,000 total segments and an order-insensitive
multiset ratio past that bound. The implementer's boundary checking used
scattered-edit shapes and saw a smooth 0.0500 -> 0.049975 step, which looked
like continuity. The independent review probed the same boundary with a **pure
reorder** -- the one shape on which the two mechanisms measure opposite things
-- and got 0.9925 -> 0.0000: one added sentence per side flipped "99% changed"
to "0% changed" for a page whose content had not changed at all. The smooth
ordinary-shape probe had proven nothing, because on ordinary shapes the two
tiers compute (nearly) the same number by construction; the boundary's real
behaviour lives exactly where they disagree. Rule: when a function switches
mechanisms on a size/cost threshold, first name the semantic axis on which the
mechanisms differ (here: order-sensitivity), then build the boundary probe to
maximise disagreement on that axis -- and if such a shape exists at all, the
design has a cliff no threshold placement can fix; the durable resolution is
one semantic at every size (the fix round retired the alignment tier and made
the order-insensitive ratio the sole mechanism, with "a moved segment is not a
change" as the documented decision), not a relocated boundary.

## A probe left red on a stale trivial pin stops guarding — the real regression ships behind it (task-19044, 2026-08-20)

The installed-migration probe (`Tests/Packaging/test_installed_distribution.py`,
`INSTALLED_MIGRATION_PROBE`) proves a v35 ChaChaNotes DB migrates to the current
schema under the *installed wheel*. It had been red since `4a2d48046` on pure
test-health noise: a hand-bumped `assert current_schema_version == 39` pin plus
a sentinel pair that same commit left self-inconsistent (probe printed
`...-v35-to-v39-ok`, the outer test asserted `...-v35-to-v38-ok`, function named
`..._to_v38`). Because the gate was already red on the pin, nobody could see
what arrived behind it: the v39→v40 bump (`46945ebbe`) never added
`chachanotes_v39_to_v40_transcript_annotations.sql` to
`[tool.setuptools.package-data]` (`include-package-data = false`, explicit
list), so **shipped wheels could not migrate an existing DB past v39 at all** —
`_migrate_from_v39_to_v40` reads that SQL file from the installed package and
died `FileNotFoundError → SchemaError`. The fix round proved this with a
control: pyproject line reverted → probe red through the real chain
(`Migration from V39 to V40 failed ... No such file or directory`); restored →
green (`installed-wheel-v35-to-v38` run: 2 failed at the pin; fixed run:
7 passed incl. both wheel sources and the release-checker mutation params).

Two rules, both incident-backed here:

1. **Every hand-bumped copy of a moving constant is a miss waiting for the
   next bump — compare against the constant in the environment where the
   assertion runs.** The probe already read
   `CharactersRAGDB._CURRENT_SCHEMA_VERSION` *inside the child process against
   the installed distribution* and asserted the migrated version equals it;
   the literal pin added nothing but staleness. The same class hid in the
   sentinel string pair and the test name (now version-agnostic
   `..._to_current`). Sibling class to watch: a schema bump's packaging
   contract spans four hand lists (pyproject package-data, both
   `Packaging/check_manifest.py` sets, the test's `RUNTIME_MIGRATION_PATHS`)
   — `46945ebbe` missed all four.
2. **A known-red gate is a masked gate.** "That test is just stale on the
   version pin" was true AND the reason a production bug (any install built
   from this tree could not migrate an existing DB past v39) sat undetected
   from the v40 bump until this task. Re-greening a trivially-red probe is
   not cosmetics; until it runs green, everything it guards is unguarded.

## A "pristine" probe worktree cut from `origin/dev` is not pinned to your base (task-19043, 2026-08-20)

Attributing a red test to my-change-vs-pre-existing, the standard move is a
throwaway pristine worktree. First attempt: `git worktree add --detach probe
origin/dev`. The probe PASSED the test my tree failed -- which read as "my
change broke it" and burned a diagnostic round chasing a regression that did
not exist, complete with a module-identity probe whose results contradicted
the pytest run (the assert was statically false in BOTH trees' source, yet
"passed pristine"). The resolution: the shared checkout's `origin/dev` ref had
MOVED between my branch's creation and the probe's creation (base `25500ad87`
-> `fa0268519`, hours apart, another session's fetch), and the newer dev had
already FIXED the red by rewriting the test (`ab468a4a2`). A second probe
pinned to the exact base SHA (`git worktree add --detach probe 25500ad87`)
showed the test red on pristine base code: pre-existing, fixed upstream, not
mine. Two rules with teeth: (1) a baseline probe must be cut at the **base
SHA your branch was cut from**, never at a moving ref name -- in a checkout
other sessions fetch into, `origin/dev` at probe time is routinely not
`origin/dev` at branch time; (2) `git log --oneline -1` inside the probe is
part of the probe -- a comparison whose two arms' commits were never printed
has not established which code either arm ran.

## A validator over durable data cannot be widened the way a request gate can (TASK-19170, 2026-08-20)

TASK-18803 converted exact-id REQUEST gates (`model == "kimi-k3"`) to family
predicates; TASK-19170 did the response side, where the same conversion runs
through the strict parser for PERSISTED private checkpoints. One of the two
k3 pins there is a shape invariant -- "a complete checkpoint must END with a
final no-calls reasoning round" -- and mechanically widening it to the family
would have made every pre-19170 versioned-kimi (non-k3) complete checkpoint
already stored in ChaChaNotes/exports/chatbooks UNPARSEABLE: those were
written complete with all-calls rounds, because only the k3 pipeline appended
the final round. The rule: before widening a validation predicate, enumerate
what every OLD pipeline actually persisted under the old predicate --
acceptance-widening (admit new shapes) is backward-safe, but
requirement-widening (demand a shape of more models) invalidates history
unless every covered writer always produced it. The fix kept the must-end
invariant pinned to the literal id, accepted both complete shapes for the
rest of the family, branched replay on checkpoint SHAPE instead of model id,
and pinned the old stored shape with its own test plus a shape-guard
mutation (M9b/M10b) at each import surface -- the guard-dropped mutants are
exactly the "old data wrongly discarded" bug.

## A settle whose predicate can RAISE is still a one-shot sample — and a value flip is not its message cascade (task-19047, 2026-08-20)

Follow-up to the pilot.pause() entry above: `_wait_until`-style condition polls
only settle what their predicate can *survive observing*. Two incidents from
the same file, both reproduced under CPU-burner load before patching:

1. **Raising predicates.** `test_switching_stts_view_dismisses_owned_profile_
   modal_and_worker` polled `app.query_one("#stts-profile-table").row_count
   == 1` right after assigning `current_view` — but `STTSWindow.watch_current_
   view` swaps the body in a `speech-view-mount` worker, so the table's
   *existence* is part of the condition, and the unguarded query raised
   `NoMatches` straight out of the FIRST poll (13/13 failing instances under
   load). Same class one step later: `.children[0]` in a predicate raised
   `IndexError` inside the observable empty window between `await
   remove_children()` and `await mount(...)` (3 reproductions at 20 burners —
   which only fired AFTER the first shape was fixed; catalogue shapes by
   re-running the load loop after each repair, since the first raise masks
   everything behind it). A predicate that throws mid-transition is a one-shot
   structural sample wearing a settle's clothes: predicates must return False
   while the structure is absent (guard the query, index only non-empty).
2. **Value-flip settles under-wait their cascade.** The kokoro-blend audiobook
   test settled on `provider_select.value == "openai"` — but that flips inside
   the timer callback, while the narrator-options rewrite rides the queued
   `Select.Changed` message. Under load the queued rewrite landed AFTER the
   test's own `_update_voice_options("kokoro")` and silently restored the
   openai list; the keyboard walk then honestly selected `shimmer`, the LAST
   OPENAI option — a failure that *looks* like broken key handling and is
   actually a wiped precondition. Trap on top: the cascade's output (openai
   options) was byte-identical to the compose-time options, so the watched
   widget itself could never witness the cascade landing; the settle had to
   target a DIFFERENT observable of the same dispatch step (a sibling
   `@on(Select.Changed)` handler's attribute write on a test-faked widget).
   When a reactive assignment's effects arrive by message, settle on the
   cascade's own output — and if that output is indistinguishable from the
   initial state, find any other observable the same dispatch step produces.

## Deleting a diagnostic-bearing call obliges the inventory hand-edit — and two reviewers missed it in one wave (tasks 19042/19043, 2026-08-20)

Companion to "Adding a resource of a GUARDED KIND obliges you to run that
kind's inventory suite" above — the DELETION direction, which proved harder to
see. The persistent diagnostic inventory
(`Docs/security/production-diagnostic-inventory.json`, gated by
`Tests/Architecture/test_persistent_diagnostic_inventory.py`) keys rows on
each file's diagnostic CONTENT, so removing a `logger.*` call changes that
file's row and the playbook requires a hand-edit in the same PR. In the
third-wave burn-down this was missed twice by implementers and twice by
reviewers: task-19042 initially skipped it, and its reviewer asserted the
inventory JSON "had zero consumers" — refuted by the controller's
rebuild-diff, which showed the architecture gate consuming it; then
task-19043's deletion (stts_events 30→29) shipped with BOTH implementer and
reviewer missing the step, leaving the gate red on dev (folded into
task-19191's per-row regeneration). Two rules with teeth: (1) any PR that
deletes or moves diagnostic-bearing code must run
`scripts/check_persistent_diagnostic_inventory.py` and hand-review its row
diff before merging — a deletion feels like it needs no review precisely
because nothing new was added; (2) a reviewer claim that a guarded artifact is
"unconsumed" is an untested claim until checked against the gate that
consumes it — grep for the artifact's path in `Tests/` before agreeing.

## A repair task filed against a VENDORED file turns the vendoring gate red — route the edit through the sync script's patch mechanism (task-19321, 2026-08-20)

**Incident.** task-19321 (filed from 19191's review) instructed a call-site
repair of three diagnostic leaks in `tldw_chatbook/Chunking/engine/chunker.py`.
The repair itself was routine — but the whole `Chunking/engine/` tree is
VENDORED from tldw_server at a pinned SHA, and
`Tests/Chunking/test_sync_script.py` diffs every vendored file against the pin
on every run: the moment the (correct, reviewed) edit landed, the suite failed
with `FATAL: local modification to vendored file chunker.py`. Neither the task
filer nor the implementer's first pass knew the contract existed; it only
surfaced because the WHOLE `Tests/Chunking/` directory was run, not just the
files near the change. dev's copy was verified byte-identical to the pin, so
this was a genuine new red, not pre-existing drift.

**Resolution that held.** Not a subclass (duplicates a 200-line generator and
leaves the leaky original importable) and not reverting: the sync script
already carried chatbook-side patches for ported TESTS (`TEST_PATCHES`), so
the same mechanism was extended to engine files (`ENGINE_PATCHES` in
`Helper_Scripts/sync_chunking_engine.py`, recorded in `VENDOR_MANIFEST.toml`
`[patches]` and a spec §5.2 amendment). Canonical vendored state =
upstream-at-pin + rewrite + patches; the modification check compares against
the PATCHED state; upstream drift under a patch anchor fails loudly. The
patch output must be verified byte-identical to the working-tree file against
a pinned clone, or the gate stays red for an invisible whitespace reason.

**Rules.** (1) Before editing anything under a directory with a
`VENDOR_MANIFEST.toml` (or any manifest/sync pairing), read the sync
contract first — the file being editable in the working tree says nothing
about whether a gate re-derives it. (2) A task description that names exact
lines to change is not evidence the file is directly editable. (3) When a
gate red appears after your edit, byte-compare the base file against the
gate's own source of truth before assuming pre-existing drift — here dev
matched the pin exactly, so the red was honestly mine.

## A raw equality check on two IDs that alias the same state misfires the instant that state is the common case (TASK-18310, 2026-08-20)

**What happened.** Implementing a Console resume-time reconcile that compares
a session's `workspace_id` against the workspace registry's active workspace
id, the first pass used `session.workspace_id == active.workspace_id` raw.
That looked obviously correct and passed the task's SPECIFIED regression gate
(`Tests/Workspaces/`, three named `Tests/UI/` files, a full-suite
`--collect-only`) cleanly. It was still wrong: this codebase already encodes,
elsewhere in the very same file (task-15120,
`_set_active_workspace_for_console_session`), that a session's default/unset
`workspace_id` (the `CONSOLE_GLOBAL_WORKSPACE_ID` sentinel, `"global"`) and
the registry's built-in Default workspace row (`DEFAULT_WORKSPACE_ID`,
`"workspace-default"`) are THE SAME state spelled two ways, not two
different workspaces that happen to share a session. The raw comparison
read every ordinary global/unset-workspace mounted session as "diverged"
the instant the registry's active workspace was its ordinary resting Default
row — which is the COMMON case, not an edge case — and tore the session down
to rebuild a fresh one on every single resume. Two tests well outside the
specified gate (`Tests/UI/test_console_session_settings.py`, picked up by an
extra author-initiated sweep of every `on_screen_resume`-touching UI test
file) caught it going from GREEN to RED; the specified gate never happened to
mount a session with an unset `workspace_id` against a Default-active
registry, only sessions with explicit non-default ids.

**What to do.** Before comparing two identifiers that come from different
layers of the same domain concept (a workspace id read off a session vs. one
read off a registry; a scope key vs. a storage key; anything with a
"no explicit value" sentinel), grep the surrounding module for an existing
normalization convention — this codebase had ALREADY solved this exact
equivalence once, and the second solver (this task) initially didn't reuse
it. When a task's specified gate is narrower than the surface the change
actually touches (here: "the three named workspace test files" vs. "every
call site of `on_screen_resume`"), run the broader sweep anyway before
declaring done; a gate scoped to the files the task author thought of is not
the same as a gate scoped to the files the change can reach.

## `tail -1` on a multi-line verification hides the failure (TASK-19480, 2026-08-21)

`check_bundle_sync.py` prints ONE line per generated sheet (five of them) and
exits non-zero if any is stale. Three CSS PRs in a row verified it with
`... | tail -1`, which shows only the last sheet's line — so a red guard read
as green every time. The desync (`widget_defaults_self.tcss`, one 4-space
line) was finally caught not locally but by reading GitHub's CI log, where all
five lines were visible and the error sat in the MIDDLE of them.

Rule: a verification that emits one line per checked item must be read in
full, or grepped for its failure token (`grep -E "error|out of sync"` /
check the exit code) — never `tail -1`. The habit of tailing to keep output
small is exactly what makes a per-item check unreadable. Note the exit code
alone was also insufficient here: the script printed `::error::` and still
exited 0 under the shell pipeline used.

## A contract whose enforcer list lives only in a docstring cannot notice a second implementation (TASK-19551, 2026-08-21)

`Utils/sensitive_paths.py` is the denylist that keeps agent file tools out of
`~/.ssh`, `~/.aws`, this app's `config.toml`, `mcp_permissions.json` and its
databases. Its module docstring named its enforcers precisely — the five tools
in `Tools/file_operation_tools.py` — and every one of them really did call
`is_sensitive_path`. Tests covered them thoroughly (`Tests/Tools/
test_file_tool_sandbox.py`). All of that was true and none of it helped: the
`fs_*` family (`Tools/local_tool_impls.py`, ADR-032) was added LATER as a
second, independent file-tool family, confined paths to `[console]
workspace_root`, and never joined the contract. `grep is_sensitive_path` over
`Tools/`+`Agents/` returned nine hits, all in the file that was already
correct, and zero in the three modules that were not. With the shipped default
root (the app's cwd at startup), launching from `$HOME` made `fs_read` return
`~/.ssh/id_rsa` and let `fs_write`/`fs_patch` rewrite `mcp_permissions.json` —
a one-step disarm of the permission gate that authorized the call.

The tell was structural, not behavioural: the docstring listed enforcers by
NAME, so it could only ever describe the implementations that existed when it
was written. Nothing in the test suite asked "do all families agree?", so the
second family's silence was indistinguishable from its absence.

**What to do.** When a security primitive is enforced by callers rather than
by the primitive itself, the coverage that matters is a CROSS-IMPLEMENTATION
agreement test: take a list of inputs the primitive is supposed to refuse and
drive every family through it in one test, so a new family is either wired in
or visibly red. Pair it with a structural tripwire (an AST check that every
path-taking entry point resolves through the one choke point) — a source-text
`grep`-style check is not enough, and in this task's first draft a literal
`"mkdir" not in source` assertion failed on the word `mkdir` appearing in a
DOCSTRING. Also worth knowing when writing the agreement test: two families
can refuse the same path for different REASONS (here the older family rejects
a dotted base directory at confinement, before the denylist runs), so assert
the denylist-sourced message only where the denylist is genuinely what fires,
rather than papering the difference over with a bare "is refused".

## `executescript` COMMITS — and a "rolled back cleanly" check can lie (TASK-19553, 2026-08-21)

**What happened.** The ChaChaNotes migration chain had 25 steps calling
`conn.executescript(...)`. `sqlite3.Connection.executescript` commits whatever
transaction is open and then autocommits each statement individually, so those
steps were neither atomic nor re-enterable — but nothing in `Tests/` was red,
because the whole test suite only ever runs migrations that SUCCEED. The
defect only exists on the failure path, and no test drove one.

Reproducing it took a fixture that manufactured the interrupted state
directly: bootstrap a genuine v11 DB, apply-and-commit the first 3 of
`_MIGRATE_V11_TO_V12_SQL`'s 4 `ALTER`s with a raw connection, then open the DB
the way the app does. The probe printed what it read: version stamp still 11,
three columns committed on disk, `SchemaError` on open — and on the SECOND
open a DIFFERENT column name in the error, which is the tell that the database
is permanently unreachable rather than transiently failing.

**Two things that generalise.**

1. *For a failure-path defect, the fixture IS the test.* "Run the suite and
   see" cannot find a bug that only exists when a migration dies half-way; you
   have to build the half-dead state by hand. `_pre_apply_statements` in
   `Tests/ChaChaNotesDB/test_migration_atomicity.py` is the reusable shape:
   split the real script with the production splitter, apply the first N,
   commit, reopen.

2. *A mechanical rewrite of shipped DDL needs an oracle, and the oracle needs
   its own mutation test.* Porting 25 steps from `executescript` to
   per-statement `cursor.execute` could silently change the schema. The oracle
   was a snapshot of ALL 39 bootstrap versions plus the v4→current chain replay
   plus a fresh build — capturing verbatim `sqlite_master.sql` text and
   `PRAGMA table_info` INCLUDING `cid` (column order) — taken BEFORE the edit
   and diffed after: 11,177 objects, 22,815 column entries, 0 divergences.
   That number means nothing on its own, so the oracle was mutation-tested
   twice: deleting one `CREATE INDEX` from a step produced 40 divergences, and
   swapping two `ADD COLUMN` statements (order only) produced 64. An oracle
   that has not been shown to go red is not evidence that anything is
   unchanged.

**One decision worth recording.** Making every step atomic means the chain is
now ONE transaction, so a failure at step N rewinds to the RUN's entry version,
not to step N-1. An existing test
(`test_citation_failure_after_dev_migrations_leaves_clean_v26`) had pinned the
old partial-commit behaviour as if it were a guarantee. It was an artifact of
the defect. When a test asserts the shape of a bug, rewrite it to assert the
property that actually matters — here, that the rewound database still opens
and migrates on the next attempt — and say in the docstring why the
expectation moved.

## A green test validates the assertion, not the story you told about it (TASK-19553, 2026-08-21)

**What happened.** Porting the ChaChaNotes v4 base-schema apply off
`executescript`, I wrote a permanent code comment explaining WHY it mattered:
the script's 42 `CREATE TRIGGER` statements have no `IF NOT EXISTS`, so an
interrupted apply "died on 'trigger already exists'" on the next launch. I
shipped a test alongside it that passed. The comment was **false**. The script
also ships 42 matching `DROP TRIGGER IF EXISTS` — zero creates without a
preceding drop — plus `IF NOT EXISTS` on every table/index and
`INSERT OR IGNORE` on both inserts. Sweeping all 120 interruption points of
the script on the pre-fix code, the retry succeeded **120 out of 120 times**.
The failure mode I described could not occur.

The review caught it; the mechanism that let it through is worth naming. My
test asserted *leftovers are zero after a failed apply* — which the pre-fix
code genuinely fails — so it went red at the leftovers check and returned
green after the fix, **without ever reaching the retry the comment was about**.
The test proved a real (and worthwhile) property: 111 committed
`sqlite_master` rows before versus 0 after. It never touched the claim.

**What to do.** When you write down a failure mode as the justification for a
change, run *that* failure mode, not a neighbouring one — and prefer a sweep
over a single anecdote (all 120 interruption points here, which is what turned
a plausible story into a measured 120/120). If the property your test actually
asserts is narrower than the story in the comment, either widen the test or
narrow the prose to what you measured. Two smells that should trigger this
check: a comment whose claim your test would still pass without, and any
sentence about a mechanism (`no IF NOT EXISTS`, `no guard`, `always commits`)
that you inferred from reading rather than from running. In a P0 data-integrity
file, a false incident in a comment is worse than no comment — future
maintainers will trust it, and this repo's own standard is "state the incident,
not just the rule", which only works if the incident is real.
## An interrupted pytest run leaves stale later-suite nodes in `lastfailed` (TASK-19520, 2026-08-21)

**Incident.** A repository-wide run was interrupted when its verification
scope was narrowed after 2h26m. Pytest reported 272 current failures/errors,
but `.pytest_cache/v/cache/lastfailed` contained 301 keys: the 272 nodes seen
by the interrupted run plus 29 failures from an earlier completed Skills
gate that the broad run had not reached. Treating every cache key as current
would have double-counted stale evidence and filed misleading tasks.

**Rule.** For an interrupted run, preserve both `lastfailed` and the collected
`nodeids`, record the last node the session actually reached, and partition
cache keys by collection position. Keys after the cutoff are prior-session
evidence unless independently reproduced. Keep the run's final pytest counts
as the accounting authority; the cache is a node-name recovery aid, not a
self-contained result report.
## "No dangerous flag in our argv" is not a guarantee — the repository supplies argv too (task-16801, 2026-08-21)

Change review's git modes shell out to `git` in the user's own repository. The
arc's one absolute rule was "never force-push", and it was enforced by argv
assertions: tests captured the real argv and asserted no element started with
`--force`. Those assertions were mutation-proved twice and were genuinely
load-bearing. They were also, three separate times, looking in the wrong place.

Four vectors reached destructive git behaviour without any dangerous flag ever
appearing in code we wrote:

1. **A remote named `--force`.** `git remote add -- --force <url>` succeeds, so
   `("push", remote_name)` becomes `git push --force`. Verified: `+ 33b7b99...
   4a3c4a7 main -> main (forced update)`, destroying another clone's commit,
   which the UI reported as "Pushed main to --force". `--mirror` deletes refs.
2. **A branch named `--mirror`,** via a `.git/HEAD` reading `ref:
   refs/heads/--mirror`, landing in `push -u <remote> <branch>`. This one was
   cleared by an earlier audit on the grounds that `check-ref-format` guards
   branch names -- but that validator only runs on branch CREATION, and
   `git check-ref-format refs/heads/--mirror` **exits 0** anyway.
3. **Push config, with no option-shaped name anywhere.** `remote.origin.push =
   +refs/heads/*:refs/heads/*` makes an ordinary `git push origin` a forced
   update; `remote.origin.mirror = true` also deletes refs;
   `push.default = matching` published an unrelated branch's private commit
   while the modal named a different branch. Non-dash remote, non-dash branch:
   both hardening layers passed cleanly.
4. **Pathspec magic, not options.** `--` stops option parsing but not pathspec
   magic, so a file literally named `:!nothing` -- which `git status` lists and
   the UI checkbox carries verbatim -- turned a one-file selection into a
   four-file commit. Same shape reached the diff pane: `diff.external` rendered
   `TOTALLY FABRICATED DIFF OUTPUT`, and a textconv driver rendered NOTHING for
   a genuinely changed file while `--numstat` still reported `1 1 a.txt`.

The rules with teeth:

- **Audit by tracing each argv slot to its SOURCE, never by asking "does a
  validator exist somewhere?"** That question produced a confident clean bill of
  health for vector 2. Write the slot table: for every value reaching argv, name
  what protects it -- a `--` separator, a specific validator, or a module
  literal -- and treat "a validator exists elsewhere in the file" as unprotected
  until you have traced the call.
- **An absent-bad-flag assertion cannot see any of these.** Prefer
  only-known-good-values-reach-argv (refuse a leading `-`; pass an explicit
  fully-qualified refspec; set `GIT_LITERAL_PATHSPECS=1`; pass
  `--no-ext-diff --no-textconv --no-color`) over asserting the bad thing is
  missing.
- **Prove the fix RED on the destruction, not on the exception.** Assert the
  other clone's commit, the release branch and the tag all survive. A test that
  only asserts "raises GitWorkspaceError" passes against a version that raises
  *after* damage. Every control in this arc was run first to confirm the harness
  could actually detect the damage -- a check that cannot fail is worthless.
- **Refuse, do not sanitize.** Stripping a leading dash from a remote name
  pushes to a *different* remote; there is no `--` escape for those positionals.
- **Watch for interactions when hardening env.** `GIT_LITERAL_PATHSPECS` is
  incompatible with ambient `GIT_GLOB_PATHSPECS`/`GIT_ICASE_PATHSPECS` (git
  aborts every pathspec-parsing call). Because this runner deliberately
  preserves the user's ambient environment, the security fix alone would have
  broken the feature for those users.
- **Probe with a driver that actually exercises the path.** The first test for
  that incompatibility used `rev-parse`, which parses no pathspec, so it passed
  against the broken code. Likewise a naive textconv driver (`echo IDENTICAL`)
  does not blank the diff pane -- git appends the file path -- so it reads as a
  cosmetic bug; blanking needs a driver whose output is genuinely constant. A
  reviewer using the naive form would have mis-ranked the finding.

Threat model, recorded because it is what makes the above worth the effort: no
`.git` exclusion exists in `Tools/workspace_file_roots.py` or
`Tools/file_operation_tools.py`, so an agent can write `.git/config` in the very
root these features operate on (TASK-19700).


## A setup step whose failure is only a log line builds the wrong fixture (TASK-19554, 2026-08-21)

**What happened.** `Tests/Notes/test_sync_engine.py::test_conflict_detection`
has passed since it was written, and it asserts the right things: a
`both_changed` conflict is detected with the right `db_content` and
`disk_content`. Reading it while writing the born-red pins for task-19554, I
found it never builds the scenario it claims to. It calls
`NotesInteropService.update_note_sync_metadata` twice with
`expected_version=1`. The FIRST call bumps the row to version 2, so the second
one matches no rows — and that method does not raise on a version miss, it
logs `"No rows updated ... version mismatch or deleted"` and returns `False`.
The return value was not asserted, so `last_synced_disk_file_hash` was silently
never stored.

With that column NULL, the engine's `db_changed = hash != last_synced` and
`disk_changed = hash != last_synced` are BOTH trivially true against `None`.
The test detects a "conflict" for every synced note in existence, including
ones where neither side moved. It could not tell a real conflict from a note
that had never been synced at all — which is precisely the distinction the
conflict path turns on.

**What to do.** When a fixture is built through an API that reports failure by
RETURN VALUE (`bool`) or by log line rather than by raising, assert the return
of every setup call — `assert service.update_x(...)`, not `service.update_x(...)`.
Optimistic-locking helpers in this repo are the recurring shape: they take an
`expected_version`, they return `False` on a miss, and every successful call
invalidates the version you were about to reuse. Re-read the version between
calls instead of reusing a literal. The tell that something is wrong is a
fixture that passes against a *stronger* claim than it set up: here, deleting
the entire baseline step would not have changed the test's result, which is the
definition of a setup step that is not doing anything.


## A guarantee is only proved for the sink you asserted against — and a test's stand-in is not the shipped one (TASK-19555, 2026-08-21)

**What happened.** ADR-029 says persistent application logs are metadata-only
with respect to user content. `Tests/test_remaining_diagnostic_sentinel_matrix.py`
looked like thorough proof: for seven domain owners it injected a private
sentinel, attached a **filtered** `PrivateRotatingFileHandler` and an
**unfiltered** `_CollectingHandler`, and asserted the sentinel stayed out of
the file. Green for months.

The app installs an unfiltered collector too — `TldwCli._setup_buffered_
logging`'s `PersistentLogHandler`, root logger, level `NOTSET`, no filter,
feeding an unbounded `deque` that the Logs screen's "Copy all" joined onto the
system clipboard, under an empty state telling the user to reproduce the
problem and share their logs. The suite never asserted against it. Its
`_CollectingHandler` was a test-local stand-in, present to prove the *other*
half of the design (that payloads stay available to the UI), and its existence
made the file assertion read as coverage of "the collector" in general. It was
not. Writing a first assertion against the real handler produced the sentinel,
an `sk-` key and a full traceback carrying a note title, all on the clipboard
path, immediately.

**What to do.** When a test proves a security property at a sink, count the
sinks. `grep` for every `addHandler`/`add` on the same logger and name in the
test which ones are covered — the filter that enforces the guarantee was
attached at exactly two call sites here, both the same handler, and that was
findable in one search. And when a test constructs a second handler as scenery,
ask whether the *production* object of that kind is asserted anywhere; a
stand-in beside the thing under test is the most convincing way to look covered
while covering nothing. The tell is an assertion list where the dangerous
surface appears only as a positive (`assert sentinel in collector.messages`)
and never as a negative.

**And then it happened again, to the person writing this entry, in the same
task.** The fix redacted at `PersistentLogHandler.emit`, which fills two
stores (`_log_buffer`, `_log_records`) and *then* hands the line to whichever
on-screen surface is mounted. I pinned both stores, wrote the paragraph above
about counting sinks, and shipped. Review mutated
`logs_window.append_record(…, msg)` to `…, formatted` — feeding the
UNREDACTED line to the mounted widget and therefore into `LogsWindow._records`,
which is exactly what "Copy visible logs" puts on the clipboard — and the
suite returned **111 passed, 0 failed**. A *store* is not a *feed*. When one
function writes the same value to several places, the count that matters is
the number of **assignments and calls that carry it outward**, not the number
of collections it lands in; walk the function line by line and pin each one.
Cheapest reliable check: mutate each outward hand-off in turn and require a
red for every one — three lines of test closed this, but only after a mutation
found it.

## A security test that never emitted its payload, twice (TASK-19555, 2026-08-22)

**What happened.** Fixing a truncation bug that could leave a partial API key
in the Logs view, I wrote the obvious regression test: log a line with the key
positioned across the 2,000-character cap, assert no fragment survives. It
passed against the **broken** implementation — twice, for two different
reasons.

1. The padding length was hand-computed from the cap alone. The handler's
   formatter prepends `asctime - name - LEVEL - `, roughly 68 characters of
   unpredictable width, so the key landed comfortably past the cut and was
   discarded whole. Nothing was ever astride anything.
2. The rewrite used `logger.info("%s %s", padding, secret)`. **Loguru formats
   with `str.format`, not `%`.** With no `{}` in the template, loguru logged
   the template verbatim and silently dropped every argument. The test emitted
   the literal string `%s %s` and asserted, truthfully, that it contained no
   credential.

Both were found by mutating the fix and expecting red, not by reading the
test. The shipped version measures the prefix width off a probe record,
sweeps every straddle position rather than guessing one, and carries an
anti-vacuity control that asserts the sentinel is a shape the redactor
actually recognises.

**What to do.** For a "secret X must not appear in Y" test, the assertion is
satisfied by an empty Y, so it proves nothing until you separately prove X was
there. Add a control in the same test — log the payload plainly and assert it
*was* redacted — and where the position of X matters, sweep the range and
derive the offsets from a measured value instead of arithmetic on a constant.
Then mutate the fix: a negative-space assertion that stays green under the
mutation is not a test. And in this codebase specifically: **`loguru` uses
brace formatting**; a `%s` template is a silent no-op that drops your payload,
while stdlib `logging` calls on the same page take `%s` correctly.

## A denylist over USER-NAMED values, and the mirror test that made it look guarded (TASK-19733, 2026-08-22)

`Utils/egress.py`'s cross-origin redirect rule was
`_STRIP_HEADERS = ("authorization", "cookie", "proxy-authorization",
"x-goog-api-key")` — drop those four on a hop that leaves the origin, forward
everything else. The obvious repair for the filed defect was "append
`x-api-key`". It would have been wrong, and the test suite would have gone
green on it.

The reason is one line in `Subscriptions/monitoring_engine.py`:

```python
key_header = auth_config.get("header", "X-API-Key")
headers[key_header] = auth_config.get("key", "")
```

The header NAME is user config. `X-API-Key` is only the default. So the
denylist was being asked to enumerate a set the user gets to extend at
runtime — unfixable by extending the list, whatever names you add. Written as
a born-red test with a header the user picked (`X-Feed-Token`), the base
failure showed the key arriving at the redirect target:

```
assert 'x-feed-token' not in Headers({'host': 'evil.example', ...,
    'x-feed-token': 'sentinel-not-a-real-key-19733'})
```

**Generalises to:** whenever a security filter matches on a value the user
supplies (header name, param name, filename, env var, tool name), a denylist
is not a weaker allowlist — it is not a control at all. Invert it. And write
the born-red test with a name nobody would have thought to denylist: had the
test used `X-API-Key`, the one-literal fix would have passed it and the real
hole would have shipped.

**Second half, same task.** `Model_Artifacts/fetch.py` kept a hand-mirrored
copy of that tuple, and `Tests/Model_Artifacts/test_stream_fetch.py` pinned
`set(fetch._STRIP_HEADERS) == set(egress._STRIP_HEADERS)`. That guard reads
like the drift is handled. It is not: it detects divergence only *after*
someone edits one side, and only if they run that suite — and it actively
rewards re-synchronising the copy, which preserves the defect shape. Fixed by
deleting the mirror and importing the one object; the guard became an
identity assertion plus `assert not hasattr(fetch, "_STRIP_HEADERS")`, so a
re-introduced mirror fails. **A test that pins a duplicated constant is
evidence the duplication should not exist, not evidence that it is safe.**

**Third half, found by the independent review of the same task.** The fix
above filtered the *built request object* — `request.headers` after
`client.build_request(...)` — and its docstring then claimed cross-origin hops
carry nothing but the allowlist "whether the header came from the `headers`
argument or from the client object's own defaults". A probe with
`httpx.Client(auth=("alice", <sentinel>))` put `Authorization: Basic …` on the
wire to the second origin anyway. `httpx` applies a client-level `auth` inside
`send()`, *after* `build_request` returns — so it is structurally invisible to
anything that inspects the request object. (The docstring did carry a residual
note, but scoped to an auth *callable*; a plain tuple leaked the same way.)

**Generalises to:** "I filtered the request" is not the same as "I filtered
what goes on the wire". Before trusting a strip, ask what the client library
adds *between* the object you hold and the socket — auth flows, cookie jars,
proxy headers, retry/redirect middleware. Prove it by constructing the
credential through each injection route the API offers (per-call arg, client
default header, cookie jar, `auth=`), not just the one the code under review
happens to use. Three of those four routes were already closed here; the
fourth was the one no test had ever expressed. The fix was
`send(..., auth=None)` — explicit `None`, because *omitting* the argument
leaves httpx's `USE_CLIENT_DEFAULT` sentinel in play and changes nothing.

## The pathspec-magic vector reappears wherever a FILENAME reaches argv — and the blanket fix for it breaks exclusions (task-19632, 2026-08-21)

The lesson above (task-16801) recorded pathspec magic as vector 4 in Change
Review's git modes and recommended `GIT_LITERAL_PATHSPECS=1` as the blanket
hardening. The agent-facing `git_*` tools are a *different* module with a
*different* entry point, and the same vector was live there, reached from the
other direction: not a repository filename the UI carried into a commit, but
the model's own `path` ARGUMENT.

`git_diff(path=":(exclude)notes.txt")` — a legal POSIX filename, a real file in
the repo, so `resolve_workspace_path` resolves it, confines it, and finds it
un-denylisted, exactly as designed. git then read it as MAGIC and inverted the
diff's scope, returning the rest of the repository including `~/.ssh/id_rsa`'s
content. The tool's *own* denylist refusal for `path=".ssh/id_rsa"` still
worked; the model simply asked a different question. Measured before the fix,
with an isolated `$HOME`.

Two things generalise:

- **Trace the argv slot per module, not per repository.** The 16801 audit was
  correct and complete for the module it audited. "This class was fixed" is not
  a property of a codebase; a second module building the same argv from a
  different source needs its own slot table. Here the source was the model, and
  the choke point every reviewer trusts (`resolve_workspace_path`) is a PATH
  validator — it has no opinion about pathspec syntax, and correctly so.
- **`GIT_LITERAL_PATHSPECS=1` and `:(exclude)` are mutually exclusive, and the
  conflict is SILENT.** With it set, `:(exclude,literal)<path>` is taken as a
  literal filename, matches nothing, and `git diff`/`git status` return **empty
  output with exit 0** (verified, git 2.39). Applying the previous lesson's
  blanket recommendation to this module would therefore have broken the feature
  *and* every credential exclusion at once, with no error to notice. The
  per-pathspec form (`:(literal)` on values that SCOPE, `:(exclude,literal)` /
  `:(exclude,glob)` on values that DENY) is the compatible equivalent. There is
  now a test asserting no `*_PATHSPECS` variable appears in that runner's
  environment, because the comment alone would not survive a future hardening
  pass.

One test-craft note from the same session: the born-red run for the injection
case **passed at base**, which looked like the vector was already closed. It was
not — the assertions happened to align with the injected behaviour, because the
credential file was not dirty in that scenario, so the inverted scope had
nothing to leak. Reading the base-run PASS list rather than only the FAILED list
is what caught it. A born-red test that passes at base is a defect in the test
until you have explained why.

---

## A guard that cannot report is not a gate — check where it runs, not just that it runs

**TASK-19572, 2026-08-21.** The repo carried three real derived-artifact
guards and a nightly full-suite cron, and none of them had produced a verdict
in weeks. Two separate mechanisms, both invisible from the workflow file:

1. **A cron on a non-default branch never fires.** `.github/workflows/test.yml`
   on `dev` declares `schedule: cron '30 8 * * *'` with a `nightly-deep` job.
   `gh run list --workflow=test.yml --event=schedule` returns **`[]`** for the
   entire retained history. GitHub schedules cron only from the *default*
   branch's copy of the workflow, and this repo's default branch is `main` —
   last updated 2026-07-11, **10,933 commits behind `dev`**, with no
   `schedule:` block in its `test.yml` at all. The nightly the repo believed it
   had has never run once.

2. **Completion is governed by queue time, not job time.** `css-bundle-guard`
   is one stdlib script with no install. Its `pull_request` runs over the last
   100: **61 cancelled, 19 success**, and the successes are bimodal — ten
   finished in 16–398 s, nine took **1.9–5.6 hours**. The job did not get
   slower; it waited for a runner. The `Tests` workflow fans out ~20 jobs
   (2 core legs + 12 UI shards + 3 lease legs + …) on every push and PR, at
   23–50 merges/day, and starves everything else — including itself.

**What to do.** Before claiming a guard covers something, ask where its verdict
lands. Check `gh run list --workflow=<f> --event=<e>` for the event you think
fires it, and read the *default branch's* copy of the file when the trigger is
`schedule`. When you measure a workflow, measure `createdAt → updatedAt` (wall,
including queue), not the job's own duration — cheap jobs on a saturated pool
are not fast verdicts. And if a check is meant to be **required** under branch
protection, it must not be path-filtered: a skipped run reports nothing, so
GitHub parks the PR on "Expected — waiting for status to be reported" forever.
That is why `derived-artifacts.yml` runs unconditionally while
`css-bundle-guard.yml` beside it does not.

---

## A content digest over raw source text fires on re-indentation — and `git diff` is the wrong tool to investigate it

**TASK-19572 pre-merge review, 2026-08-22.** The production diagnostic
inventory keys each logger call on a SHA of its own source segment, precisely so
that *moving* a call is not a review event (task-3750). But
`ast.get_source_segment` keeps the continuation lines' absolute indentation, so
a call that merely shifts nesting level — because someone wrapped the
surrounding block in a new `if` or `try` — produces a different digest with
identical text.

The incident: `Chat/console_fleet_wake.py` showed as `11/9df8f371… ->
11/44b53292…`, and the checker's report labelled it *"reworded / re-levelled /
new args"*. All three were wrong. Dev had landed a 328-line change (248
insertions) that re-indented exactly two `logger` calls; **not one diagnostic
statement had changed**. Whitespace-normalizing both sides proved it in one
line.

**Two things generalise.**

1. **A digest taken over raw source text is not a content digest.** If you build
   one, normalize what you do not want to review — or make the reporting layer
   pair off layout-only changes explicitly. Otherwise the artifact's own stated
   contract ("movement is not a review event") is false, and every false alarm
   teaches the reviewer to regenerate without reading, which is the one failure
   the artifact exists to prevent.

2. **Never investigate an AST-keyed artifact with a line diff.** The report's
   original trailer said `git diff $base -- <path>`; following it buried two
   re-indented calls inside an unrelated refactor. Recover the statements by
   running the *checker's own scanner* over both revisions' git blobs and
   diffing the resulting `(method, digest)` multisets — the same keys the pin
   moves. That turns a 328-line read into
   `moved/re-indented only: 2   removed: 0   added: 0`. The checker now ships
   this as `--statements <path> --since <rev>`; use it, and check multiset
   counts, not set membership, or a deleted duplicate of an identical call is
   invisible.

Corollary on scope: when a rebase resolves the pin conflict by taking the other
side wholesale, **verify the taken pin reproduces from its own tree before
concluding anything** — and separately review the delta between it and the pin
you last reviewed. Those are two different questions, and the first being green
does not answer the second.

## A construction-symbol swap silently disconnects every test that mocks it by name, and those tests are not where you'd look (TASK-19830, 2026-08-22)

**What happened.** Converting 49 `requests.Session()` construction sites
across six `LLM_Calls/` modules to a shared `create_default_session()`
factory looked purely mechanical — same object shape, same methods, an
explicit `timeout=` still wins. `Tests/LLM_Calls/ -q` went from clean to
**55 failed**. The cause was never the production code: dozens of tests across
the repo intercepted the OLD symbol by name —
`monkeypatch.setattr(module.requests, "Session", fake_session)`, a
`SimpleNamespace(Session=lambda: session)` swap of the whole `requests`
module reference, or a `unittest.mock.patch("...module.requests.Session")`
string target. Once the production code called `create_default_session()` (a
function imported into the module's own namespace) instead of
`requests.Session()`, none of those patches touched anything: the real code
path went unmocked, so real code ran a live/blocked network call, or an
`AttributeError`, or (worst) a fake that silently no longer applied — a
`transport_must_not_run` negative-path assertion that "passed" only because
the mock was no longer wired to anything, not because the code was actually
verified not to reach the network.

**Where the tests actually were.** `Tests/LLM_Calls/` alone had 8 affected
files. The largest single file was `Tests/Chat/test_chat_functions.py` — 26
occurrences, zero of which are under `Tests/LLM_Calls/`. In total: **12 test
files, 91 individual mock call sites**, across `Tests/Chat/` and
`Tests/LLM_Calls/` both, found only by grepping every import alias each
converted module is known under (`cloud_adapters`, `local_adapters`,
`llm_calls`, `lib`, `sgl`, `legacy_adapters`, `llm_api_calls_module`, ...)
against the pattern `<alias>.requests` across the **entire** `Tests/` tree —
not by running the package's own test directory and calling it done.

**What to do.** Before swapping a construction symbol (a class, a factory
function, anything a test might intercept by patching its name), grep the
**whole test tree** — not just the package under change's own test
directory — for every import alias of the module being touched, combined
with the OLD symbol's attribute name (`<alias>\.requests` here). Do this
*before* running the gate, not after chasing the first red result; the
failure signatures vary (a live network attempt, an `AttributeError`, a
values-mismatch three calls deep) and none of them says "your mock stopped
being wired up." A negative-path test (`transport_must_not_run`,
"assert this was never called") is the most dangerous case: it can go on
reporting green forever while checking nothing, because a disconnected mock
and a correctly-guarded code path are indistinguishable from the outside.

---

## A retained child can still lose typing when its parent recomposes it (TASK-19003, 2026-08-20)

The reviewed Notes import canvas learned to patch its destination `Input` in
place, and its bare-widget typing test passed. The production Library pilot
still turned a burst of `i`, `n`, `b`, `o`, `x` into only `x`: the retained
parent `LibraryNotesCanvas` recomposed the entire import child after the first
`Input.Changed`, so the remaining key events targeted a detached widget.

For live inputs, a child-level identity assertion is not enough. Drive a burst
through the production wrapper and handler, then assert both the complete value
and the exact `Input` identity. Every retained ancestor that synchronizes the
field must preserve the same-mode child; one recomposing ancestor defeats all
in-place work below it.

**Recurred, TASK-22034, 2026-08-26.** The Skills adaptive reader passed its
307-test destination gate and a 153-test cross-reader gate, including mode,
trust, import, delete, and geometry behavior. Final diff review still found
legacy whole-screen recomposes after import browsing, Back/discard, delete,
trust reset, and first-time trust setup. The outcomes were correct, but each
path could replace the supposedly permanent Items list. Adding exact list/work
identity assertions to import, trust setup, and delete exposed the missing
proof and the callbacks were changed to destination-scoped synchronization.

For retained-reader migrations, do not infer permanence from successful
behavior or from identity across mode buttons alone. Inventory every legacy
`refresh(recompose=True)` reachable in the destination and prove exact owner
identity across at least one ordinary exit, one asynchronous task settlement,
and one destructive/recovery settlement.

**Recurred, TASK-22866 (2026-08-28).** A reactive local/server backend flag on
Watchlists Sources used `recompose=True` so two local-only buttons could change
labels and disabled state. The replacement `DataTable` lost the user's focus and
cursor; worse, its synthetic initial row highlight republished the first stale local
source immediately after the screen had cleared local selection for server mode.
The create form's Watchlist destination had the same authority problem when patched
only visually. The fix updates the mounted controls in place and strips local
destination data at submit while retaining the local draft for restoration. For a
mode/capability switch, assert object identity, focus, cursor, selected entity, draft
values, disabled copy, and the submitted payload in both directions. Rendering the
right labels is not enough when replacement widgets emit selection events.

## `exclusive=True` does not cancel work already handed to `to_thread` (TASK-19003, 2026-08-20)

The first import handler scheduled an exclusive Textual worker. A repeated
activation could cancel that worker's await while the executor already running
in `asyncio.to_thread` continued mutating Notes. The controller then cleared its
cancel-event reference and remained in `IMPORTING`, so the UI had lost the task
that still owned mutation authority.

Tests for off-thread mutation must gate the executor after admission, activate
the command twice, and prove exactly one executor call reaches settlement.
Claim authority synchronously before scheduling, reject duplicates instead of
using cancellation as admission control, and shield/join admitted thread work
when an outer UI task is cancelled. Cancelling the awaitable is not evidence the
thread stopped.

**Recurred, TASK-24402 (2026-08-29).** My Profile reused one exclusive Textual
worker group for authority changes and secure-removal actions. Review forced the
cancelled worker's underlying thread to finish late: an old authority selection
could overwrite a newer restrictive selection, and delayed key deletion could
run after Start Fresh provisioned a new generation. The fix belongs below the
UI worker boundary: runtime-policy writes carry the snapshot's expected version,
and the service serializes the complete remove/finish/start lifecycle. Event-gated
tests force the reversed completion order; ordinary sequential UI tests cannot
prove this class of safety.

## `CREATE IF NOT EXISTS` can adopt foreign schema objects (TASK-19004, 2026-08-21)

The first lasting-sync migration validated its required tables after running
`CREATE TABLE IF NOT EXISTS`. A nonempty v0 database was therefore silently
adopted, and a pinned v1 database could carry an unrelated trigger into v2.
Both cases passed shape checks while preserving behavior the private owner had
never authorized.

Before migrating or reopening an owner-exclusive SQLite database, census all
user tables, indexes, and triggers. Reject unexpected objects before any DDL;
allow only explicitly repairable omissions and SQLite-owned internals. Tests
must assert the rejected database is unchanged. Post-migration shape checks
alone cannot prove provenance, because idempotent DDL preserves whatever was
already using the requested names.

## Native rename flags and post-commit errors need real platform probes (TASK-19005, 2026-08-21)

The first guarded Notes move used rename flag `1` as “no replace” on every
POSIX platform. That is correct for Linux `renameat2`, but on Darwin flag `1`
means `RENAME_SECLUDE`; an actual macOS probe replaced the destination and
removed the source. Unit tests around a mocked rename seam had made the code
look safe without testing the host primitive's semantics.

The same review found errors after atomic exchange/rename reported as ordinary
refusals. Callers could retry even though new bytes were already installed or a
source had already moved, and cleanup paths could delete the displaced bytes.

For native mutation primitives, test the real supported platform constants and
collision behavior, not only wrapper calls. Mark the exact linearization point
in code: failures before it are bounded refusals; every unverified outcome after
it is a distinct partial state, with displaced authority preserved until the
commit is durably verified. A successful syscall is not proof that the whole
method may still report an ordinary failure.

## A lock pathname is not the locked inode, and release has a commit point (TASK-19006, 2026-08-21)

The first lasting-sync coordinator correctly held an OS lock, but it trusted the
fixed lock pathname afterward. Replacing that regular file or its private
directory let a second process lock a new inode while the first admission still
reported write authority. A separate interleaving let `close_admission()` return
while another thread had removed the handle from shared state but had not yet
called OS unlock.

For path-addressed advisory locks, persist and revalidate the opened handle,
path, parent directory, protected resource, modes, owner, and link identities;
pathname equality alone does not preserve authority. Model release explicitly
as running, committed, or failed. Clearing a shared handle is not release
completion, and every concurrent close/release caller must wait for the same OS
unlock/close outcome before reporting that ownership ended.

## Canonical spelling is not filesystem identity on case-insensitive roots (TASK-19008, 2026-08-21)

The legacy Notes migrator resolved candidate and private paths before comparing
them. On the actual APFS volume, a case-variant spelling of the application data
directory resolved successfully and referred to the same inode, but the
sensitive-path string comparison returned no conflict. The private directory was
then accepted as a paused sync candidate.

When path ownership is a security boundary, test real aliases on the supported
filesystem and compare existing objects with filesystem identity (`samefile` or
verified device/inode), including ancestor relationships. Resolution removes
`..` and symlink spellings; it does not guarantee case, mount, or lexical aliases
have one string representation. Identity-comparison failures must also reject,
not fall back to a spelling-based allow decision.

## Reopening the database is not restart evidence if the request survives (TASK-19007, 2026-08-21)

The first durable sync executor tests reopened the private SQLite store after
each injected journal-stage failure, but then resumed with the original
in-memory execution request. Those tests passed while fresh reconstruction
still trusted corrupted recovery bytes, accepted a same-content file on a new
inode, and dropped a persisted direction override. A real process restart would
have rebuilt all three authorities from private recovery and current Notes/file
observations, so the reused request hid exactly the unsafe boundary under test.

For resumable work, a restart test must discard the controller, executor,
request, snapshots, and service fakes that carry reviewed authority. Construct
a fresh store and executor, enumerate incomplete operations, reconstruct only
from durable intent plus fresh authority observations, and then exercise every
advertised Resume, Restore, and Disconnect path. Reopening SQLite alone proves
storage durability; it does not prove process-durable reconstruction.

## A green lifecycle matrix can still miss the transitions between its states (TASK-19009, 2026-08-21)

The first gated lasting-sync runtime passed its inert, active, recovery, watcher,
and shutdown tests independently. Adversarial overlap probes still found that a
hint arriving during reconciliation was lost, a dead watcher continued admitting
automatic work, shutdown could race startup and reopen admission, and persisted
Failed/Partial status could be ignored or suppress the operation ID of an
incomplete journal. Each state looked correct in isolation; the broken behavior
lived in the handoff between two correct-looking states.

For an application-owned durable runtime, test transitions as overlapping event
pairs: hint-during-reconcile, shutdown-during-start, watcher-death-then-hint,
status-plus-incomplete-journal on reopen, and explicit-check-while-recovery-is
unresolved. Assert both authorities after each interleaving: the in-memory
admission/next action and the durable journal/status. A state matrix is necessary,
but it is not concurrency or restart evidence until the edges are executed.

**Second incident — TASK-31232 Canvas settings, 2026-09-04.** Tests constructed
`ConsoleRuntime` inside an already-running pytest event loop, so its constructor
successfully started a policy watcher. The real CLI constructs `TldwCli`
synchronously and only then calls `run()`: the same watcher helper silently
returned because no loop existed, leaving external disables unobserved before
the first preview. A regression that constructs the actual app before
`asyncio.run`, enters Textual's `run_test`, changes policy without explicitly
reading the gate, and waits for watcher retirement failed on that code. Starting
the existing watcher from `on_mount` made it pass and retained one owner plus
idempotent disposal. Match production construction order when claiming startup
coverage; an async test fixture can accidentally supply the missing lifecycle.

---

## A package AST sweep can descend into an ignored nested virtualenv (TASK-19906, 2026-08-22)

While verifying the Remote Models redesign, the package-wide class-CSS
consolidation test failed inside Textual's own `Widget` and `Toast` classes,
even though the changed stylesheet parsed and reproduced exactly. The test's
`Path.rglob("*.py")` walked through an old, gitignored
`tldw_chatbook/.venv/lib/python3.13/site-packages/` directory nested beneath
the package root; serializing that foreign Textual version's default rules
then crashed on a `None` link background. A targeted diagnostic that printed
the `(module, class)` pair for each parse failure proved both failures came
from the ignored virtualenv, not application code.

Rule: source-tree AST/file sweeps must prune environment and tool directories
(`.venv`, virtualenvs, caches) explicitly; `.gitignore` does not affect
`Path.rglob`. When such a gate fails in dependency code, print the exact
discovered path before changing production CSS or blaming version drift.

## A truthful fresh-profile factory is not a returning-user destination harness (TASK-19579/TASK-19642.1, 2026-08-22)

**Incident.** The shared Library app factory correctly admitted a fresh profile,
while destination integration tests assumed the complete returning-user rail.
The mismatch first recurred in TASK-19579 and then made the Skills import flow
look broken because the expected Skills row was absent. Choose the factory whose
profile posture matches the destination contract; do not redefine a truthful
fresh-profile factory merely to satisfy returning-user tests.

## A red guard protects nothing, and `raising=False` is how a stale monkeypatch hides (TASK-19569, 2026-08-22)

**What happened.** Three guards had been red on `dev` for weeks. None of them
was red because the thing it guards was broken:

- `Tests/Agents/test_tool_catalog_concurrency.py` asserted `2 == 1` on every
  run since it installed its `_ensure_catalog_cache` counter and *then* called
  `registry.list_catalog()` itself, counting its own setup call. Production was
  correct the whole time. Hoisting one line made it green — and a mutation
  restoring the historical two-snapshot `name -> id -> provider` shape in
  `_owner_record_for_name` made it red again with the same `assert 2 == 1`. For
  the whole period, a real TOCTOU guard could not detect its own regression.
- Five `Tests/MCP/` watchlists tests patched
  `local_server_tools.RuntimeSourceStateStore`, a name TASK-18609 had replaced
  with an injected `load_default_runtime_source_state`. Four errored at the
  monkeypatch line. **The other two passed `raising=False`** — so `monkeypatch`
  cheerfully created a brand-new attribute nobody reads, the test fell through
  to the real loader, and one of them *passed for an accidental reason* (the
  real loader happens to return `"local"`, which is what the fake wanted).
- Six `Tests/DB/test_core_sqlite_owner_privacy.py` failures were the only
  honest reds in the set: a genuine product defect (a `from None` severing the
  cause chain the privacy contract walks).

**Two traps worth naming.** First, `monkeypatch.setattr(..., raising=False)` on
a *seam* is never a convenience — it converts "this seam was renamed" from a
loud error into a silent no-op, and the resulting green tells you nothing.
Reserve it for genuinely-optional attributes; on an injection seam, let it
raise.

Second, mutation-testing a scrubbing guard has to hit the layer the guard
actually watches. `test_real_watchlists_provider_scrubs_unexpected_failures`
survived *three* separate leak mutations before biting, because the scrub is
layered (service `_raise_unexpected` -> provider `ToolResult.error` -> gateway
fixed-message mapping) and any one layer alone re-scrubs. The mutation that
finally exposed a real hole was the smallest one: adding `detail=%s` to
`WatchlistsToolService._raise_unexpected`'s `_LOGGER.error` leaked the sentinel
into the captured log **and the test still passed** — it asserted against
capsys and a loguru sink, and neither sees stdlib `logging` records. Its
sibling in `test_local_server_tools.py` had asserted `sentinel not in
caplog.text` all along. If a "no secrets leak" test names some output channels,
check that it names the one the code under test actually writes to.

**What to do.** When you inherit a red test, first establish *why* it is red:
a defect in the product, a defect in the test, or a seam that moved. Only the
first is a baseline you may carry. Then never leave a repaired guard at green —
mutate the behaviour it protects and watch it red, at the layer that owns that
behaviour, before you call it repaired.

## An assertion whose expected value equals the platform default proves nothing — and neither does one the platform quietly satisfies for you (TASK-19562, 2026-08-22)

**What happened.** Two of this task's ACs asked for facts about SQLite, and
the obvious tests for both were inert.

*One.* `SubscriptionsDB` set no `busy_timeout`, so it inherited one. AC:
"set a timeout." The natural pin —
`assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000` — passes
**with the pragma deleted**, because 5000 ms is exactly what
`sqlite3.connect(timeout=5.0)` already gives you. Deleting the production
line and re-running was what exposed it. The pin that can actually fail
monkeypatches the connector to pass `timeout=0` and asserts the connection
still reports 5000: red at `0 == 5000` without the pragma.

*Two.* AC: "the `-wal` is checkpointed on close." The natural pin — write
300 rows, `close()`, assert the `-wal` is 0 bytes — also passes with the
checkpoint removed, because **SQLite checkpoints and deletes the `-wal`
itself when the LAST connection to a database closes**. The test was
measuring sqlite's own cleanup. The fix was to hold a second connection open
across the assertion (which is also the leaked worker connection the task was
about) and additionally assert the file still *exists*: red at "close() left
content in the -wal" once something else keeps it alive.

The same platform behaviour refuted a whole premise: a child process that
wrote a 4.1 MB `-wal` and exited normally left only the `.db` behind, with a
new `atexit` settle hook enabled and suppressed — identical. The "the `-wal`
is left behind at exit" concern is false for a clean exit; what is real is
the standing WAL a long-running process carries, and the `os._exit(0)` signal
path, which no `atexit` hook can reach.

**What to do.** For any test whose expected value is a platform, library or
language *default*, delete the production line and re-run before believing
the green. If it still passes, the assertion is describing the platform, not
your change — either find an input where the two diverge (force the default
to something else, keep a second handle open) or write in the test that it
cannot fail on its own and name the sibling that can.

## Instrument the argument shape production uses, not the one that is easiest to call (TASK-19562, 2026-08-22)

**What happened.** A previous session had to decide whether
`SubscriptionsDB.record_check_result` really nested `transaction()`. It
instrumented the context manager across "a real call", observed **depth 1,
one entry**, and recorded the hazard as REFUTED at that call site — in the
code, in the tests' docstring, and in the task file.

Re-instrumented per argument shape:

    record_check_result WITH stats    -> 2 entries, depths [1, 2]
    record_check_result WITHOUT stats -> 1 entry,  depths [1]

The nesting runs through `_update_subscription_stats` ->
`update_subscription_stats`, reached only when `stats` is truthy — and
`execute_run`, the only production caller, *always* passes stats. The
earlier probe had called it the easy way, with `stats` omitted, and measured
the one branch production never takes. The hazard was **live**, not latent:
the daily-statistics write was durably committing the enclosing
subscription-health UPDATE.

**What to do.** When a probe decides whether a hazard is live, call the
function the way the production call site calls it — copy the arguments from
that call site, do not construct minimal ones. If the function branches on an
argument's presence, probe **both** branches and record which is which; a
single "instrumented a real call" line in a task file cannot be checked later
by anyone, and this one was wrong in the direction that closes an
investigation.

## A sequential start/join thread loop hides per-thread leaks -- and "descriptors return to baseline" is the wrong assertion (PR #1964 review, 2026-08-22)

**What happened.** Qodo found that `SubscriptionsDB._connections` held a
strong reference to every thread's sqlite connection and only ever removed
the *calling* thread's entry, so a worker that exited without calling
`close()` pinned its connection -- descriptor and WAL lock -- for the life of
the process. The first probe written to confirm it spawned 100 short-lived
threads **one at a time** (`start(); join()`), and reported **no growth at
all**: the registry stayed at 2 entries. The reason is that the OS recycles
thread idents, so `self._connections[ident] = connection` overwrote the
previous entry every iteration. The defect was real; the measurement designed
to see it could not.

Re-run with 20 threads held **concurrently** on a barrier, the same code gave
registry 21 entries and 43 descriptors, permanently, and `gc.collect()` could
not reclaim any of it (the sqlite3 statement cache is an `lru_cache` wrapping
the connection, so every used connection sits in a reference cycle and is
never freed by refcounting alone).

Two further facts cost time and are worth writing down:

* **`sqlite3.Connection` cannot be weak-referenced.** The obvious fix -- and
  the one the review suggested -- is a `weakref.WeakValueDictionary`. CPython
  3.12.11 raises `TypeError: cannot create weak reference to
  'sqlite3.Connection' object`. Check weak-referenceability before designing
  around it; C types often lack `tp_weaklistoffset`.
* **Descriptors do not return to their pre-thread baseline even when every
  connection is genuinely closed.** SQLite's unix VFS keeps closed
  descriptors for an inode in a reuse pool while any connection to that file
  remains open, releasing them together when the last one closes. Asserting
  "fds back to baseline" would have failed against a correct fix. The
  assertion that actually distinguishes fixed from broken is **"a second
  round of threads costs no more descriptors than the first"**: fixed gave
  13, 13, 13, 13, 13 over five rounds and 0 after the final close; broken
  gave 23, 43, 63, 83.

**What to do.** To measure anything keyed on thread identity, run the threads
concurrently -- a sequential loop tests ident recycling, not your registry.
And before asserting on file descriptors, establish what the *correct*
steady state looks like by measuring a known-good control (here: the same
code path with no registry at all), rather than assuming it is the baseline.

---

## The interpreter is an input to your test, and this repo spans four of them (PR #1960, 2026-08-22)

**The trap.** A test can be correct, deterministic, and still pass or fail purely on
which Python ran it. This repo makes that routine rather than exotic: the declared
floor is **3.11** (`requires-python = ">=3.11"`), the maintainer's `.venv` is **3.12**,
worktree venvs get **3.14**, and a contributor may have **3.13**. Four of six reds
found on dev in one sweep were this, by three unrelated mechanisms — each invisible on
whichever interpreter its author happened to use.

**What happened.**

- **`ast.dump()` stopped rendering empty fields.** Python 3.13 added `show_empty`,
  defaulting to `False`, so `Call(func=..., args=[...], keywords=[])` became
  `Call(func=..., args=[...])`. The summarization diagnostic ledger FREEZES dumped
  shapes, so every reviewed row holding a no-keyword call stopped matching. Measured by
  digesting all 229 shapes in `Local_Summarization_Lib.py` on four real interpreters:
  3.11 and 3.12 gave `92a85a3a989ffd13`; 3.13 and 3.14 gave `cd27a1b6c1fdf001`. The
  split is exactly where `show_empty` landed.
- **`dis.findlinestarts()` began yielding `None` lines.** On
  `ProfileStoreLease.acquire`, 3.11 yields 0 positionless entries and 3.14 yields 12.
  Comparing those against an int boundary raised `TypeError` while building a
  *module-level* constant — a COLLECTION error, so all 98 tests in the file stopped
  running rather than failing.
- **A nested same-quote f-string is a 3.11 `SyntaxError`.** PEP 701 legalised quote
  reuse in 3.12. `TTS/backends/kokoro.py` therefore could not be imported *at all* on
  the project's own floor, while every local test passed on 3.14. Note that
  `ast.parse(feature_version=(3, 11))` does **not** catch this — it does not downgrade
  the tokenizer. Only a real 3.11 interpreter does.

**A recorded diagnosis is not a fix.** The `findlinestarts` case was already written up
here nine days earlier ("Run source-inspection tests on a supported interpreter before
changing them", TASK-15706, 2026-08-13). That entry was accurate and correctly told the
reader how to tell interpreter drift from a product regression — and the 98 tests stayed
uncollectable the whole time, because diagnosing is not the same as repairing. When an
entry lands here describing something still broken, it needs a task, not just a
paragraph.

**What to do.**

1. **Verify a version-sensitive fix across the whole supported range**, not on the one
   you are sitting at, and not only at the two ends. All four matter here: the `ast.dump`
   split falls **between 3.12 and 3.13**, so an ends-only check (3.11 and 3.14) would
   have seen two disagreeing answers without locating the boundary, while the f-string
   break is visible **only at the floor**. `uv python find 3.11` / `3.12` / `3.13`
   resolve the three you are not running; re-running the same digest under each is
   seconds of work.
2. **An artifact that freezes interpreter output must pin the rendering, not chase it.**
   `stable_dump()` in `Tests/ast_shape.py` forces the pre-3.13 rendering (`show_empty=True`
   where supported, nothing below 3.13 where empty fields always rendered), so all four
   interpreters reproduce the committed digest. Regenerating the ledger against the new
   rendering was the tempting alternative and the wrong one: it invalidates a large set
   of individually reviewed privacy rows at once, and breaks the 3.11 floor instead of
   fixing anything.
3. **Treat an Optional in an introspection API as load-bearing.** `findlinestarts`
   documents its line as optional; the failure was assuming otherwise.
4. **A module-level computation turns drift into a collection error**, which reports as
   an error rather than a failure and can be skipped past with
   `--continue-on-collection-errors`. Grep for `ERROR` as well as `FAILED` when
   surveying a suite, or an entire file's worth of tests will read as "not failing".

---

## A `wait_for` around work that ignores cancellation is not a bound (task-19561)

While replacing shutdown's flat `asyncio.sleep(0.1)` with a real bounded wait,
the first draft was `await asyncio.wait_for(asyncio.gather(*tasks), timeout)`.
It read like a timeout. The test written for exactly that case —
"shutdown does not hang on a task that ignores cancellation" — **wedged the
whole pytest run for five minutes** until the harness killed it.

`wait_for` (and `asyncio.timeout`, which it is built on in 3.11+) implements
its deadline by *cancelling* what it is awaiting and then awaiting that
cancellation. Work that swallows `CancelledError` therefore hangs the very call
whose timeout was supposed to bound it. Use `asyncio.wait(tasks, timeout=...)`
when the point is to stop waiting: it returns `(done, pending)` and cancels
nothing. Then drain `.exception()` off the finished ones so a cancelled-at-
shutdown task does not resurface as "exception was never retrieved".

The generalisable half: **a timeout is only a bound if the thing it is wrapped
around can be interrupted.** Write the uncooperative-work test — it is the only
one that distinguishes the two.

## A process-lifetime mechanism must be gated on owning the process (task-19561)

The same task added an exit watchdog: a daemon thread that `os._exit`s if
shutdown has not finished within a grace period, armed from `App.on_unmount`.
Every unit test passed. What that misses is that `Textual`'s `run_test()`
mounts and unmounts a **real** `TldwCli` inside the pytest process, which then
runs thousands more tests — so every such test was arming a timer to kill the
test runner ~20 seconds later. It surfaced only because a test was written that
mounts the real app and asserts *no watchdog thread exists afterwards*.

Two rules fell out. **Gate anything that ends the process on an explicit claim
made by an entry point** (`claim_process_exit()` here), never on "the app is
shutting down" — under test those are different facts. And **when a mechanism
replaces itself with a tighter deadline, stand the superseded one down**: the
first version left the old thread asleep on its original, longer deadline,
which is a live timer nothing can cancel any more. That bug was found by the
same real-app-mount test, not by any of the seven unit tests around it.

## A timeout on shutdown is enforced against healthy work too (task-19561)

Independent review of the same watchdog asked one question its own tests did
not: *can this fire while the app is still doing something legitimate?* The
answer was yes, and only a side-by-side live probe showed it. A clean quit
(`app.exit()`, no signal) with one ordinary `run_worker(..., thread=True)`
holding an open `BEGIN IMMEDIATE`:

| | merge base | with the watchdog |
|---|---|---|
| 30 s worker, clean quit | died 28.8 s after quit, **rc 0**, both statements committed | died **20.1 s** after quit, **rc 1**, `[]` — transaction abandoned |

The grace period is not only a stuck-process bound. Textual thread workers run
on the loop's default executor and cannot be interrupted, so teardown really
does wait for them — and a *healthy* one that outlives the deadline is
`os._exit`ed exactly like a wedged one. The unit tests could not see this
because they arm the watchdog against nothing; the quiet-exit measurement
(0.6 s) could not see it either, because the whole point is that the
pathological case is the one with live work in it.

**Generalisable:** when you bound a shutdown, the acceptance evidence must
include a run with legitimate long work still in flight, not just a quiet exit
and a wedged thread. Write down which side of the trade you chose, in the
config comment the user will read — not only in the module docstring.

Second, smaller one from the same review: **`thread.is_alive()` is False for a
constructed-but-unstarted thread**, so a guard of the shape "is a watchdog
already running?" written as `self._thread is not None and
self._thread.is_alive()` has a hole between publishing the thread and starting
it. An `RLock` does not close it — signal handlers re-enter on the *same*
thread and sail straight through. Key the guard off the state you actually
care about (here: an unexpired deadline), not off thread liveness.

**Resolved:** the default went 20 s -> 120 s. The reasoning is worth keeping
because it generalises to any "how long should the timeout be" argument: the
requirement that motivated the bound (*"interpreter exit is not blocked for
seconds"*) was already satisfied by the **quiet**-exit measurement, 0.6 s,
which the constant does not affect at all — a healthy exit never reaches the
deadline. So tightening it bought nothing on the AC and cost a 30-second
ingest its write. Once you notice that a knob is inert on the metric you
picked it for, the only live consideration left is the asymmetry of the two
failure modes: a slow quit is an annoyance, an abandoned transaction is data
loss. Deliberately declined at the same time: extending the deadline whenever
a straggler is reported, which turns a bound into a suggestion.

## "It runs at startup, so nothing of ours exists yet" is an ordering claim, and ordering claims decay (task-19561)

`Subscriptions/startup_reconcile.py` shipped with a paragraph headed *"Why it
is safe to sweep unscoped"*, whose argument was: the sweep runs once, during
app startup, before any claim can have been taken in this process, so every
in-progress row it sees belongs to a dead process.

Every clause was true about the *design* and false about the *code*.
`on_mount` starts the scheduler worker; the sweep is created later, as a
deferred startup task after post-mount setup; `SchedulerLoop.run()` ticks
immediately after loading its queue. So a due watchlist check launched a real
`running` row seconds before the sweep looked at it, and the sweep marked that
live row `failed`. Single process, every launch. Nothing anywhere enforced the
ordering the docstring assumed — it was not even written down as a requirement,
only as an observation about what the code happened to do at the time.

The tests could not see it, because every one of them built the "stranded"
rows itself and then called the sweep. A hand-built row cannot tell you
*when*, relative to the rest of startup, a real row comes into existence. It
took driving the real `SchedulerLoop` + real handler + real service against a
throwaway DB, with only the HTTP fetch blocked so the check was genuinely
in flight, to make the defect appear at all.

**Generalisable, two parts.**

*Evidence:* when a claim is about ordering between two subsystems ("this runs
before that can have started"), the only evidence that tests it is a probe
that actually starts both. Seeding the row yourself tests the SQL, not the
claim.

*Design:* prefer a boundary to a sequence. Fixing this by moving the sweep
earlier in `on_mount` and pinning the order with a test would have been
correct only until the next innocent edit to `on_mount` — a file that changes
constantly. Capturing `MAX(id)` per table when the database is opened, in
`__init__` where no event loop exists yet, makes "this process's rows are out
of reach" true by construction, whatever order anything after it runs in. Two
details make it hold: the tables are `AUTOINCREMENT`, so ids are never reused
after a delete (a plain `INTEGER PRIMARY KEY` would reuse the highest freed
rowid and break the scoping silently — so that guarantee gets its own test);
and the boundary is a **required** argument, so the scoped call cannot decay
back into the unscoped one by omission. An absent boundary means *sweep
nothing*, because leaving a row wedged is recoverable on the next launch and
failing a live one is not.

*Corollary on mutation-testing your own regression test:* removing the `AND
id <= ?` from the SQL did **not** turn the headline scheduler-race test red —
its boundary was `None` (empty table), so an early return short-circuited
ahead of the mutated statement. The test was only proven load-bearing by
mutating the *whole* contract to HEAD semantics. If a mutation leaves a test
green, find out which guard absorbed it before concluding the test is weak —
or that the mutation was equivalent.

## Latching "done" before doing it disables the mechanism permanently (task-19561)

`install_termination_handlers()` called `claim_process_exit()` and set its
`_handlers_installed` flag *before* attempting `signal.signal`, and it
swallows installation errors by design (failing to install a nicety must not
stop the app starting). One failure — not on the main thread, or any other
`signal.signal` error — therefore produced both bad outcomes at once: the
latch made every later call a no-op, so handlers were never installed at all,
**and** the watchdog was armed and able to hard-exit the process anyway,
because the claim had already gone through.

**Generalisable:** an idempotence latch and a capability claim must both be
consequences of success, never preconditions of the attempt. Set them after
the thing works. And when you write the "degrade gracefully" branch, say out
loud what the resulting end state is: here the correct one for a legitimately
embedded app is *no handlers, no claim, no watchdog* — fully inert — which is
only reachable if the failure path leaves the state untouched and retryable.

## "Zero external callers" ages: re-derive the caller set, do not inherit the claim (TASK-19564, 2026-08-22)

**What happened.** TASK-19564 filed ChaChaNotes `sync_log` as a write-only
shadow copy and recommended *retiring the content columns* on the strength of
"both of its readers have **zero external callers** — nothing consumes it."
That sentence was true when the pattern was named. By the time the task was
implemented, `ChaChaNotes_DB.py` had **six** `sync_log` readers, not two, and
three of the four newer ones had live non-test callers:
`read_committed_chat_sync_intent` (`ConsoleChatStore
.ensure_provider_continuation_durable`, which **raises** on a `None` read),
`read_committed_chat_delete_intent`, and
`list_current_committed_chat_sync_intents`
(`._reconcile_restored_chat_sync_intents`, on every conversation restore).
Each compares the stored payload to the live `messages` row **field by field**;
the payload IS the commit proof. Retiring `content` as recommended would have
made every comparison fail — silently disabling Sync v2 and turning every
provider-continuation checkpoint into a hard error.

**Why the stale claim was believable.** Grepping the two READER NAMES the
filing quoted reproduces the filing's answer exactly: `get_sync_log_entries`
and `get_latest_sync_log_change_id` really do have only test callers in this
database (the production hits are `Prompt_Management/Prompts_Interop.py`
calling the *Prompts* DB's same-named methods). The claim fails only if you
grep the TABLE, `sync_log`, and read every hit — the newer readers embed it in
a `JOIN sync_log AS intent` inside a method whose name says nothing about the
log.

**What to do.** When a filing says a thing is dead, re-derive the caller set
from the artifact itself (the table, the column, the file) rather than from the
symbol names the filing chose, and check the *newest* code first — a corpse
claim decays from the direction of recent work. Then confirm the negative
direction too: what *would* break if you removed it, checked by reading the
would-be-orphaned call sites, not by running a suite. Here the suite would not
have caught it either: the readers `return None` on failure, and the callers
degrade to `{"status": "skipped"}`.

**Second-order find, same session.** Writing the direct-index witness for the
sibling task surfaced an unrelated live defect the same way: `messages_au` and
`keyword_collections_au` issued an FTS5 `'delete'` unconditionally, and
`add_keyword_collection` on a soft-deleted name raised `database disk image is
malformed` through the public API. Nothing had ever asserted against the FTS
index directly, so it had gone unnoticed. Also worth knowing before you reach
for it: FTS5 `'rebuild'` re-derives from the base table with **no** `deleted`
filter, so using it to repair an external-content index re-indexes every
tombstoned row. Use `'delete-all'` plus an explicit filtered reinsert.

**Third find, same session — `_` is a wildcard in SQL `LIKE`.** The new
retention triggers were first named `<entity>_sync_log_prune`, which silently
joined the `<entity>_sync_%` namespace that three tests assert the exact
membership of as a design invariant ("these four triggers, and only these,
write the sync log"). Only ONE of the three went red, because the other two run
against pre-migration historical databases where the new triggers do not exist
yet — so two-thirds of the collision was invisible until a later schema bump
would have surfaced it in an unrelated PR. When adding a schema object, grep
the test tree for `LIKE '<prefix>%'` patterns your new name could match, and
remember the underscore matches any single character.

**Fourth find, independent review of the same branch — a retention rule must
enumerate its WRITERS from the live schema, not from the entities the filing
named.** The filing's ACs listed "conversation, message, note or character", and
the shipped rule covered those plus keywords and keyword_collections: six
entities, all correct. Enumerating `sqlite_master` for triggers containing
`INSERT INTO sync_log` finds **nine** — `chat_dictionaries`, `world_books` and
`world_book_entries` also write the log, none was covered, and probing them on
the finished branch reproduced the original defect verbatim (a hard-deleted
world-book entry's full `keys` + `content` orphaned in `sync_log` forever;
4/4 old bodies retained across 4 edits). The sibling half of the SAME commit had
already enumerated `chat_dictionaries` and `world_books` from the schema for its
FTS census and found them — so the information was in the branch, it just did
not cross from one half to the other. Rule: when a change claims to bound a
shared table, derive the covered set from the table's own writers and assert
`covered == writers` in a census test, exactly as the FTS half did; otherwise
the AC's example list silently becomes the scope.

**Follow-on, same branch, after a third-party reviewer converged on the same
find.** Qodo's review of PR #1974 reported the identical three omissions. Two
things were learned closing it, both worth carrying:

*A documented gap is not a shipped gap.* The branch had already written the
residue up honestly, with reasons it was hard. That is better than silence, but
the docstring still said "Delete every `sync_log` row no reader can reach"
while three entities' plaintext survived deletion — an untrue contract in a
*privacy* fix. When an independent reviewer names the same gap, that is the
signal to price the fix again rather than re-defend the deferral.

*"Order-independent" is a claim that needs a control, not an argument.* The
rule that shipped had to survive SQLite's undefined firing order for same-kind
triggers. Re-running every scenario under six permutations of the emitters'
creation order and getting identical results proves nothing on its own — the
permutation might not reach the firing order at all. The evidence is the
**control**: with the retention triggers dropped, the same soft delete emits
`update@cid3, delete@cid4` in one permutation and `delete@cid3, update@cid4` in
the other. Only then does "identical with retention" mean something. The shipped
test asserts both halves — differ without, agree with — so it cannot pass
vacuously. Generalises: any experiment of the form "X does not depend on Y"
needs a run showing Y actually varied.

One more concrete trap from the same work: `CURRENT_TIMESTAMP` is constant
within a single `sqlite3_step()`, so a timestamp trigger's nested UPDATE
usually writes the *same* value the outer statement did and its emitter's
`OLD.x IS NOT NEW.x` guard stays false. The hazard only appears when the outer
statement supplies a different timestamp — which means probing the natural path
alone would have concluded, wrongly, that there was no same-version content
row. Construct the hostile input; the friendly one hid the bug.

---

## A Pilot-driven latency probe can measure the harness, not the app

**What happened (2026-08-22, holistic perf review of dev `35d4bf3a1`).** A live message
census attributed **~940 posted `Callback` messages per keypress** in the configured Console —
looking exactly like an app-side message storm, and superficially contradicting the static
lane's verdict that the keystroke path was clean. Attribution by callback qualname (patching
`MessagePump.call_later`/`call_next`) showed 18,640 of 18,648 were
`Pilot._wait_for_screen.<locals>.decrement_counter`: Textual's `pilot.pause()` posts **one
callback per mounted widget per pause**, so every `press()+pause()` latency median in the
probe (82–226 ms) was harness-inflated, and the "storm" was the probe itself. The app-side
per-key work was ~7 widget refreshes — the static read had been right all along.

**Rules.**
- Never quote an absolute latency measured through `pilot.press()+pause()`; use that shape
  only for A/B comparisons where the pause overhead is identical on both arms.
- Before believing any message/callback storm, attribute the POSTERS by qualname — counting
  message types alone cannot distinguish app traffic from harness traffic.
- When a live probe contradicts a careful static read, suspect the probe's harness before the
  static read; resolve by attribution, not by majority.

Full disclosure section: `Docs/Design/2026-08-22-holistic-perf-review.md`
("Measurement-artifact disclosure").

## Which hand-maintained literal an author updates is decided by DISCOVERY PATH, not by importance (TASK-20971, 2026-08-22)

**What happened.** `VALID_TABLES['chachanotes']` in `DB/sql_validation.py` is a
hand-maintained allowlist; `validate_table_name()` rejects anything not in it,
so a forgotten table makes every generic CRUD helper raise for that table.
TASK-864 filed it when 9 of ~47 tables were listed. TASK-19568 repaired it
after it went stale again — merged `aaec11812`, 2026-08-22 **00:16** -0700.
TASK-19057 added two Actor Pack tables in a v44→v45 migration and broke it
again — merged `2fe6ca20f`, **14:51** the same day. **Fourteen and a half
hours** between "repaired" and "red again."

**The instructive part is what that same author *did* update.** They correctly
added `idx_actor_pack_persona_intents_state` to
`Tests/ChaChaNotesDB/test_index_census.py` — an equally hand-maintained
literal, guarding the same migration, with the same "nothing connects the
migration to the literal" weakness. And they correctly bumped three
schema-version pins under `Tests/DB/`. Measured on that branch
(`git diff b593f853d 09b768239`), both were reachable by something the author
actually did:

* the index census sits in `Tests/ChaChaNotesDB/`, the directory where they had
  just written `test_actor_pack_migration.py` — running that directory turned
  it red; and
* the version pins contain the schema version number, which a grep for the
  constant finds.

`VALID_TABLES` is reachable by neither. It names no schema version, and nothing
else in `Tests/DB/test_sql_validation.py` mentions the feature. **Its guard had
been surviving by geography, and stopped the first time a migration landed from
a directory that did not happen to sit next to it.** Nothing about the
allowlist's importance, its comment block, or the correctness of its pin
mattered — the pin was right and it did report; it reported after the merge.

**What to do.** When you add a hand-maintained literal as a guard, ask what
*discovery path* connects the change to the literal, and assume the author will
have only two: (a) they run the directory their new test lives in, and (b) they
grep for a token their change forces them to touch. If the literal is in
neither, the guard depends on memory and will decay on a schedule set by how
often work lands from elsewhere. Give it a path: put the check where the author
already runs (here, `scripts/preflight.sh`, which is stdlib-only and ~0.1 s for
this check), and make its failure message print the exact lines to paste rather
than only the names of what drifted.

**Do not "fix" it by generating the literal.** TASK-19045's rule stands: a
census that re-derives its expectation from the artifact it guards is the
identity function on the defect class it exists to catch. The way out is a
*different* artifact. Here the schema's own `CREATE TABLE` text —
`DB/migrations/chachanotes_*.sql` plus the SQL string literals in
`ChaChaNotes_DB.py` — is independent of `VALID_TABLES` and was already
authoritative. Two details made that scan trustworthy: parse the `.py` sources
through `ast` and read only string constants (a raw-text scan reports three
phantom tables — `IF`, `column`, `as` — from prose in `#` comments that says
"CREATE TABLE", and a guard that reports phantoms gets muted), and prove the
new oracle against the old one rather than asserting it alone: the static scan
and a live fully-migrated `CharactersRAGDB(":memory:")` agree exactly, 69
substantive tables, symmetric difference empty.

## A packaging test that reads the packaging config is the identity function; assert against the built artifact (TASK-19860, 2026-08-22)

**Incident.** `tldw_chatbook/DB/migrations/` held 32 `.sql` files. Four separate
hand-maintained lists said which of them ship: `pyproject.toml`'s
`package-data` (13), `MANIFEST.in` (11), `Packaging/check_manifest.py` (13),
and `Tests/Packaging/test_installed_distribution.py`'s
`RUNTIME_MIGRATION_PATHS` (13 -- one of its fifteen constants was even defined
twice, with the same value). A wheel built from that config carried 13 files.
`pip install` + first launch died with
`SchemaError: Migration from V40 to V41 failed ... No such file or directory:
chachanotes_v40_to_v41_persona_visual.sql`. **The application did not start
after a normal install, and had not been able to for two schema bumps.**

**Two things made it survive a packaging suite that already had ~90 tests.**

1. *The tests agreed with the config instead of with reality.* `check_manifest`
   and the test both required exactly the 13 names the config shipped, so a
   green run meant "the wheel contains what we listed", never "the wheel
   contains what exists". Adding a migration and forgetting the lists produced
   no red anywhere.
2. *The runtime symptom under-reported by 19 files.* `_initialize_schema` walks
   v4 -> current and aborts at the FIRST missing script, so one `SchemaError`
   named one file. Fixing only that file would have moved the wall to v45, and
   the next report would have looked like a new bug. Aborting-at-first is a
   reporting property, and here it hid 95% of the defect.

**The rule.** Build the wheel and the sdist and read the members out of the
`ZipFile`/`TarFile`. If the assertion can be satisfied by editing
`pyproject.toml`, it is testing the config, not the artifact -- and it goes on
passing the day someone changes build backends. Report *every* missing file in
one assertion message; a per-file `parametrize` that stops at the first still
tells you one name at a time.

**This does not contradict TASK-19045's "do not generate the literal".** That
rule forbids re-deriving an expectation from *the artifact it guards*. The
derivation has to come from somewhere independent: here the expectation is the
`.sql` files in the source tree, and -- separately -- the `.sql` names the
artifact's own `ChaChaNotes_DB.py` opens, parsed out of the shipped module.
Neither is the packaging config, so neither can be satisfied by editing it.
Both fail closed: an empty derivation is a red check ("no migration reads
detected; the detector has drifted"), never a vacuous pass.

**Mutation-check it against the artifact, not the test.** Enumerating 31 of 32
files in `pyproject.toml` and rebuilding made four independent tests name
`chachanotes_v45_to_v46_sync_log_retention.sql` and made the installed-wheel
probe die with the real `SchemaError` -- which is what proves the test is
wired to the build and not to a fixture.

**Audit the neighbours, and do it against the artifact too.** Grouping every
non-`.py` file under `tldw_chatbook/` by directory and extension and diffing
each group against the wheel and sdist took one script and cleared 60 groups:
one real defect (migrations), one deliberate partial
(`embedding_configs_examples.toml`, forbidden in the wheel on purpose), and
three deliberately explicit single-file lists (the pinned TTS manifest and the
vendored `LICENSE` notices). Recording *why* each of those stays enumerated is
the part that keeps the next person from "helpfully" globbing them.

**Addendum from the independent review: "absent from both artifacts" is not
evidence of "excluded by design".** The group audit above sorts every group
into shipped / absent / partial, and the *absent* bucket was read as
intentional. Two of its members were not. `Evals/eval_datasets/*.json` is read
at runtime — `Evals/eval_templates/research.py` resolves an absolute path into
it and passes it to the runner as `dataset_name`, which the runner probes with
`Path(...).exists()`; absent from the artifact, the bundled research template
loses its dataset and **nothing raises**, which is exactly why it outlived the
migrations, whose absence at least produced a `SchemaError`. And
`LLM_Calls/LICENSE` / `tldw_api/LICENSE` — Apache-2.0 texts for two subtrees
this project deliberately re-licenses — shipped in neither artifact while the
modules they cover shipped in both. Note the shape of that second one: the
reason recorded for keeping licences enumerated ("a fixed legal obligation per
package, not a growing directory") is *true*, and it is precisely what makes an
incomplete list a breach rather than an oversight. A stated reason justifies
the mechanism; it says nothing about whether the list is complete, and both
have to be checked separately.

So the audit needs a positive probe, not only a grouping: for every non-`.py`
file under the package, grep the packaged Python for its basename, and treat
any hit that is missing from the wheel as guilty until explained. That sweep is
~20 lines, ran in a second over 190 assets and 1,813 modules, and it is what
surfaced both. Filter the noise by hand — generic names (`README.md`,
`LICENSE`, `pyproject.toml`, `.DS_Store`) match string literals all over the
tree — 15 raw hits, 3 real.
## A "prefer the new accessor" getattr order silently bypasses injected test doubles on MagicMock apps (TASK-21103, 2026-08-23)

Converting the eager `persona_buddy_controller` to a lazy property needed an
explicit-construction seam (`ensure_persona_buddy_controller()`) for the one
consumer allowed to build it — the Personas Workbench Buddy action handler.
The first wiring resolved it "new accessor first": `ensure =
getattr(self.app, "ensure_persona_buddy_controller", None); controller =
ensure() if callable(ensure) else getattr(self.app,
"persona_buddy_controller", None)`. Every targeted Buddy test stayed green
except `test_restart_restores_selection_open_collapsed_and_geometry`, which
failed with an apparently unrelated `FileNotFoundError` on a config.toml the
test expected the action to have written. The cause: the test's app double is
a **MagicMock**, so `getattr(mock, "ensure_persona_buddy_controller", None)`
auto-creates a callable attribute — the handler happily "constructed" a fresh
MagicMock controller and the REAL injected `PersonaBuddyController` (the one
whose preference writer persists to disk) was never touched. Nothing raised;
the action "succeeded" against a phantom.

Two rules from this:

- When adding an optional accessor consulted via `getattr` in code that
  MagicMock-backed tests drive, resolve the EXISTING seam first and fall back
  to the new accessor only when it yields None. The passive-first order is
  also the semantically correct one here — an already-built (or injected)
  controller must always win over re-construction.
- The failure surfaced two files away from the change (a missing config file,
  not a wrong controller), which is exactly why the whole feature's test
  files get re-run after a consumer-resolution change, not just the file that
  motivated it.

Same task, smaller trap: a source-pin test asserting init ordering via
`initializer.index("ConsoleRuntime(self)")` matched the substring inside MY
OWN explanatory comment ("Slots must exist before `ConsoleRuntime(self)`
below") and pinned the comment, not the construction. A source-index pin's
needle must be an expression form that cannot appear in prose (here
`"= ConsoleRuntime(self)"`), or writing a helpful comment breaks the pin —
or worse, keeps it green while pinning nothing.
---

## A consolidated widget's first DYNAMIC mount can lose its own CSS to a stale tie-breaker

**TASK-21115, 2026-08-23.** Converting 25 post-consolidation `DEFAULT_CSS` blocks
to `BUNDLED_CSS` left every compose-time harness green while 19 UI tests went red —
every one of them mounting a converted widget AFTER app boot. Textual's
`Stylesheet.add_source` keeps the lowest tie-breaker ever offered for an existing
source but does not arm `_require_parse` when lowering it (textual 8.2.8). A
class's own `DEFAULT_CSS` used to mask that: it WAS a new source at first mount,
arming the reparse itself. A consolidated class adds no source, so its dynamic
first mount resolved against a stale parse in which a bare `Vertical`'s
`width: 1fr; height: 1fr` defaults still carried tie-breaker 0 — exactly tying the
sheet's `ConsoleSelectionMenu { width: auto; ... }` rule and beating it on source
order. Measured: the menu mounted 80x40 instead of 24x6. Compose-time mounts never
show it because registration and the first parse share the mount batch.

**What to do.** Any change that stops a widget class registering its own stylesheet
source must be verified with a DYNAMIC first mount (post-boot `app.mount(...)` /
`push_screen`), not only compose-time mounts — the destination tour guard never
exercises that path. The durable fix here is `css/tie_aware_stylesheet.py`
(`TieAwareStylesheet`, used by both `TldwCli` and the `ConsolidatedCSSApp` harness),
pinned born-red-vs-plain-`Stylesheet` in `Tests/UI/test_consolidated_css_harness.py`.
## An unpaced reader-latency probe hides whole-write stalls at p95 (TASK-21124, 2026-08-23)

**TASK-21124, 2026-08-23.** The fix removed the global config file lock from
cache-hit reads so a concurrent config write (fsyncs + TOML parses under the
lock) could no longer stall event-loop-side `get_cli_setting` calls. The
obvious probe — a reader thread timing 2,000 reads while a writer loops, then
comparing p50/p95 — showed *identical* percentiles before and after the fix
(p50 ~5.5 µs both sides), twice: first because the reads finished before the
writer thread ever reached its first lock acquisition (a fixed read count
races thread startup), and then, after gating the read loop on writer
progress, because of distribution shape — an unpaced reader oversamples the
uncontended gaps, so five whole-write stalls became five huge samples in
57,000, landing beyond p99.9 and invisible at p95/p99. Yet those few samples
ARE the defect: one 18 ms block on the event loop is the jank being fixed.
Printing max and a `>1 ms` stall count made the change legible instantly
(base: max 18.2 ms; fixed: max 3.7 ms, fsync phase no longer under a
reader-visible lock). Two rules: (1) for a stall-class defect, gate the test
on a *mechanism* counter — here, a lock-acquisition count asserted to be
exactly zero on the warm path, which was also the honest red-first test (100
acquisitions per 100 reads before the fix) — and keep wall-clock numbers
informational; (2) when you do report reader latency against a bursty
contender, report max and a stall count, never percentiles alone, because an
unpaced sampler weights its own idle loop, not the user's exposure.

## A "is this feature configured?" probe that reads the store CREATES the store (TASK-21112, 2026-08-23)

**TASK-21112, 2026-08-23.** Gating the notes-sync runtime's unconditional
start on "non-empty root summaries" looked like a one-liner:
`store.list_root_summaries()`. But that call goes `transaction()` →
`_get_connection()` → `sqlite3.connect(path)`, and connect **creates the
database file** (plus the WAL side files and a full schema census) — the
probe would have manufactured the exact zero-profile state DB the gate
exists to prevent, and the "no DB file after boot" regression pin would have
gone red against the *gate itself*. The shipped gate never opens SQLite: it
is `legacy_sync_directory_configured(app_config) or state_db_path.exists()`
(config-key presence + `Path.exists()`), evaluated off-thread inside
`_start_once`, with `review_setup` force-starting the runtime on first real
feature use. Two rules: (1) a lazy-open/no-side-effect gate must be decided
from evidence that is itself side-effect-free — file presence, config keys —
never by calling into the store it guards; check the read path all the way
to `connect()` before trusting a "read-only" method. (2) the regression pin
must assert on the FILESYSTEM (`not path.exists()` after boot AND after
shutdown), not on which methods were called — that is the only shape that
catches a probe, a shutdown hook, or a migrator quietly creating the file.
This recurs for every store queued in TASK-21105 (seven more feature DBs to
be made first-use-lazy).
## A "safe_" local that only reaches the error message is not protection — mutate it to prove which value the query saw (TASK-19558, 2026-08-23)

Three `ChaChaNotes_DB` search methods computed `safe_search_term = f'"{term}"'`
and then bound the RAW term. The quoted value was interpolated into the
`logger.error` f-string in the `except` block and nowhere else. It had survived
every review of those methods because a reader who sees `safe_search_term` two
lines above a query stops reading — the NAME is the assertion, and the name was
free.

The evidence that settles it is a **mutation, not a reading**. Replace the
computed value with an absurd string
(`"ZZZ_MUTATED_NEVER_MATCHES_ANYTHING_ZZZ"`) and run the real method against a
real database: at base all three returned byte-identical results
(`['Zed the Hunter'] / ['Talk about dragons'] / ['hello world']`), which is only
possible if the value never reached SQLite. Apply the same mutation to the fixed
code and all three return `[]`. Two directions, one probe each; no amount of
staring at the diff produces that.

Generalise it: **whenever a sanitizer's output is a local rather than an
expression at the call site, the sanitizer might not be wired.** The shape is
cheap to census — a local named `safe_*`/`quoted_*`/`escaped_*` whose every AST
`Name` load sits inside a `logger.*` call is, by construction, decorative. That
census (`Tests/Utils/test_fts5_quoting_adoption_census.py`) run against the base
blob rediscovers exactly those three and nothing else, and it ships as a test so
the rediscovery is repeatable rather than a claim in a PR description.

A third face turned up in review, and it is the one to remember: the FIX for
a dead store can be a dead store. Round one replaced the unbound
`safe_search_term` with a bound whole-query phrase -- correct, protective, and
quietly halving multi-word recall at eight seams (see the recall lesson below).
Binding the sanitized value is necessary, not sufficient; you still have to
show the query means what it meant before.

The same shape has a second face, met later in the same task. The review had
asked for `("reads",)` risk tags on the read-only local agent tools "so they are
floored to ask". Measured: local tools resolve through
`permission_store.resolve_effective_state`, whose floor set is
`HIGH_RISK_TAGS = {"mutates", "process"}`; `"reads"` is in
`BUILTIN_HIGH_RISK_TAGS`, which only `resolve_builtin_state` consults, and that
function never serves the `local:__local__` server key. Adding the tag would
have produced a marking that reads as protection in review and floors nothing —
the `safe_search_term` defect, re-created while fixing it. **Before adding a
marking because a sibling has it, run the resolver that consumes it and show
the verdict change.** If the verdict does not change, the honest deliverable is
the written-down mechanism plus a test that demonstrates the inertness, not the
tag.

## Sanitizing a search box can NARROW it, and nothing red will tell you (TASK-19558 review, 2026-08-23)

Quoting fixed an injection at fourteen search seams. It also quoted each seam's
whole query as ONE FTS5 phrase, and an FTS5 phrase requires the words to be
CONTIGUOUS. `dragon lore` stopped matching a record named "lore of the dragon
reversed": recall halved at eight seams, on the Console conversation search,
the Study flashcard box and the prompt picker. Every test passed. The injection
tests passed *harder* — a narrower query closes more.

The trap is that **the security assertion and the recall assertion point the
same way.** "Returns 0 rows for `x" OR col:"y`" is satisfied by a fix and by
an over-fix alike, so a suite made only of closure tests cannot distinguish
"safe" from "broken in the user's favour of nothing". This repo had already
paid for it once: `rag_service._escape_fts5_query`'s docstring records
TASK-3995 finding that whole-query phrase quoting "is strictly stronger than
AND-of-terms, not equivalent to it", verified against a real corpus document.
The fix was re-derived from scratch three years later by someone who had read
that docstring while working in the same file.

Two things to actually do:

1. **Pair every closure probe with a recall probe on the same corpus.** Seed
   two records per seam — one where the query's words are adjacent, one where
   they are split — and assert BOTH are returned. One extra fixture row turns
   an invisible regression into a red test. A before/after table across both
   halves is the evidence; a table of only closures is not.
2. **When a "safety" change touches an expression language, name the semantics
   you are picking.** Phrase, AND-of-terms and prefix are three different
   queries, all of them injection-safe. Write down which one each seam had
   BEFORE — here the rule turned out to be mechanical (*a seam that bound RAW
   had implicit AND; a seam that bound a quoted phrase had a phrase*), and
   that rule, once stated, decided all fourteen seams and stopped the fix from
   smuggling in unmeasured behaviour changes beside the measured one.

A third, cheaper lesson from the same round: **a NUL byte is not just another
character to a quoting fix.** `sqlite3` passes a bound TEXT parameter as a C
string, so SQLite truncates at the first NUL *after* you quoted — the closing
quote is on the far side of the cut, and `unterminated string` is raised no
matter how correct the escape was. Raw binds had survived it by luck. If you
are adding quoting to anything that reaches SQLite, test `"a\x00b"`.

## An FTS5 search box quietly has two contracts, and quoting one breaks the other (TASK-19558, 2026-08-23)

Fixing the quoting above broke five Library tests, and the reason generalises to
any "sanitize at the boundary" sweep. `search_conversations_by_content` had two
kinds of caller: the Console/UI seams passing PLAIN user text (which must be
quoted), and `library_local_rag_search_service._search_conversations` passing a
pre-built, plural/singular-widened FTS5 MATCH expression (which must not be).
The second only worked BECAUSE the argument was bound raw — the defect was load
bearing. Its three sibling seams (notes, media, prompts) had already been given
an explicit `fts_match_query` parameter for exactly this; the conversations seam
had never been converted, and nothing marked it as the odd one out.

So: before quoting a parameter, **enumerate its callers and split them by what
they are actually passing**, and give the expression-supplying callers a
separate, named parameter rather than overloading one argument with two
contracts. A single parameter that means "plain text OR a MATCH expression,
depending on who is calling" cannot be made safe — every fix for one caller is a
regression for the other.

Two smaller traps from the same sweep, both worth a line:

- **A length test measured on the wrong string silently retires a branch.**
  `search_media_db` widened 1-2 character queries to a prefix match
  (`len(effective_fts_query) <= 2`). Quoting adds two characters, so testing the
  quoted string's length would have made that branch dead code with no test
  failing. Measure such predicates on the RAW input and say so in the comment.
- **A test asserting that bad input raises can be pinning the bug.**
  `test_search_with_invalid_fts_syntax_raises_error` asserted that typing
  `invalid "syntax` into the prompt search box raises `DatabaseError`. That was
  never a contract; it was the symptom of the raw bind, written down as if it
  were one. When a fix turns such a test red, read what the test is asserting
  about the USER before assuming the fix is wrong.

## "No rows" is not the safe default for an unparseable query — and an AND-joined false leg poisons the whole WHERE (TASK-19558 review round 2, 2026-08-23)

The round-one fix above had a second half nobody measured. When the new quoting
could not build a MATCH expression at all — punctuation-only input, whitespace,
a NUL — `search_media_db` answered `conditions.append("0")`. The conditions are
`" AND ".join`ed, so that one leg forced the ENTIRE query to zero rows,
including the LIKE predicates sitting right beside it that could still express
what the user typed. Measured against the merge-base on a five-row corpus:
`!!!` → 0 rows (LIKE would have found "Alert!!! urgent dragon"), `-` → 0
("well-known dashes"), `***` → 0, `""` → 0. The comment above the line even
argued for it — "simply DROPPING the condition would widen the result set" —
which is true of a query whose ONLY filter is that leg and false of this one.

Three things this generalises to:

1. **A false predicate is not a no-op, it is a veto over its siblings.** Before
   writing `1=0` / `"0"` / `AND FALSE` into a conjunction, look at what else is
   in the conjunction. If any sibling can still answer the question, the leg
   must be DROPPED, not falsified. Symmetrically, dropping is only safe when a
   sibling survives — with no other text predicate, dropping returns everything.
2. **Branch on the REASON the builder returned empty, not on the fact that it
   did.** Three reasons arrived at the same line and want three answers: a
   caller-supplied expression that came out blank means "no rows" by that seam's
   own contract (and its LIKE legs are deliberately not built, so dropping
   returns the whole table); a NUL means the LIKE fallback is *wider* than what
   was asked for, because SQLite truncates the bound parameter at the NUL and
   `%dragon\x00lore%` reaches it as `%dragon` (measured: it returned the dragon
   row at the merge-base); punctuation-only means LIKE is exactly right. One
   `if` per reason, each with the measurement in the comment.
3. **Whitespace-only input is an EMPTY search, and padding is not part of the
   query.** `"   "` should mean "I typed nothing", not "find me three spaces".
   The same strip also fixed a pre-existing narrowing nobody had noticed: the
   LIKE leg is AND-ed with the FTS leg, so `"  dragon  "` matched `MATCH` and
   was then vetoed by `LIKE '%  dragon  %'` — 1 row for `dragon`, 0 for the
   padded spelling, on dev.

The evidence shape that catches this class: a before/after table over the
AWKWARD inputs (`!!!`, `   `, `""`, `-`, `***`, empty, plus a normal control),
run against the merge-base AND the branch in the same process — `git show
<merge-base>:path/to/module.py` loaded via `importlib.util.spec_from_file_
location` under a dotted name inside the real package resolves its relative
imports fine, so both versions can be seeded and queried side by side without a
second worktree. Closure probes alone never move on any of those rows.
## A lazy package facade protects nothing that consumers import directly, and deferring at the CONSUMER can move the cost instead of removing it (TASK-21200, 2026-08-23)

TASK-21103 removed PIL and `Persona_Visual` from the `import tldw_chatbook.app`
closure and shipped a guard. Eight hours later the Actor Packs branch merged and
put them straight back: `app.py` -> `Actor_Packs/__init__` ->
`Actor_Packs.activation` -> `Persona_Visual.repository` +
`Character_Chat.visual_identity` (module-level `from PIL import Image`). The
branch was authored *before* the guard existed and merged *after* it, without a
rebase, while CI was not enforcing checks — so a trunk invariant that was one
commit old was silently undone by a branch that predated it. **When you land a
new invariant, the in-flight branches are the threat, and only enforced CI
catches them.**

Two fix shapes looked right and were both wrong; the reasoning generalises to
any import-closure repair.

1. **The house lazy-facade pattern did not apply.** TASK-21103 fixed
   `Persona_Buddy` with a PEP-562 `__getattr__` on the package `__init__`, so
   reaching for it here was the obvious move. It would have changed nothing:
   `app.py` imports `Actor_Packs.activation`/`.export`/`.importer`
   **directly**, and importing a submodule executes the package init *and* the
   submodule regardless of how lazy the init is. A facade only helps when the
   heavy module is reached **through** the package's own re-exports. Check which
   one your consumers actually write before copying the pattern.
2. **Deferring at the consumer would have made the guard green while boot stayed
   slow.** The tempting one-line-region fix was moving app.py's eight
   `Actor_Packs` imports into `_wire_character_persona_services`, their only
   caller. But that method runs from `TldwCli.__init__` (app.py:6076), so every
   real boot would still have paid PIL — the guard measures *module import*, and
   the user feels *import + construction*. Fixing the three modules at the
   source removed PIL from both. **Before deferring an import into a function,
   check when that function runs; if it runs during construction anyway, you
   have satisfied the test and not the user.**

The evidence shape that settled all of it: a `sys.meta_path` finder that records,
for each module, the module whose body triggered its import (importlib executes a
module body in a frame whose `co_name` is `<module>`, so the nearest such frame
outside the finder is the true importer). Reading `-X importtime` instead is the
trap it replaces — its indentation nests by *completion* order, and misreading it
pointed the first diagnosis at the wrong module. That tracer is now installed in
`Tests/Packaging/test_persona_buddy_import_closure.py` itself, so the guards fail
with the offending chain printed rather than a list of resident modules; the
mutation test (re-add one module-level import, watch it go red) printed
`app -> Actor_Packs -> ...activation -> ...visual_identity -> PIL` verbatim.

Two checks worth copying when you defer imports: read the pre-change public
surface from `git show HEAD:<file>` (not the edited file) and assert
`getattr(pkg, name) is <direct submodule import>` for every name — a surface that
silently shrank cannot pass that; and re-probe import ordering in fresh
subprocesses (submodule-first, package-first, heavy-dep-first), because
TASK-21160 shipped a live regression when a lazy facade unmasked a cycle the
eager init had been front-loading in a safe order.

**Recurred, TASK-21666 (2026-08-24).** `config.py` reused Media Reader
normalization with a module-level import from
`tldw_chatbook.Library.library_media_reader_state`. Importing that apparently
pure submodule first executed `Library/__init__.py`, whose eager exports reached
back into `config.py` before configuration initialization finished. Focused
Settings and shell tests then failed during collection instead of reaching any
assertion. Moving the import into `_load_settings_uncached`, after config module
initialization, preserved the shared normalizer without entering the package
cycle. The incident reinforces the package rule above: a direct submodule import
still executes its package initializer, so inspect that initializer's closure
before adding a module-scope dependency in foundational code such as config.

---

## A resting compact screenshot does not prove focused control geometry (TASK-21000, 2026-08-22)

**What happened.** Persona Buddy's 10-column fallback looked correct at rest with
two three-cell icon buttons. Preserving a two-cell lower-right resize grip moved
Close left, but focusing it still expanded `×` to the seven-cell native `Close`
button. That expansion overlapped and visually replaced Fold even though resting
screenshots, hit tests, and keyboard tests were green. An earlier repair that let
the resize corner override Close was also false evidence: both operations existed,
but some cells inside a control secretly resized instead of activating the control.

**What to do.** For terminal overlays, test the complete interaction-state geometry,
not just the resting glyphs: focus every control, assert regions remain disjoint,
assert each visible control cell resolves to that control, and exercise activation
and resize through distinct cells. Add up the cell budget before promising transient
labels; if controls plus required hit regions cannot fit, define the constrained
fallback explicitly instead of creating ambiguous overlap priority.
---

## A deferral changes WHICH objects the build binds, not just WHEN it runs (TASK-21108, 2026-08-23)

**What happened.** TASK-21108 moved `build_notes_sync_runtime_owner(...)` out of
`TldwCli.__init__` into a lazy `notes_sync_runtime_owner` property so
`Notes/notes_sync_runtime` + `Notes/notes_sync_legacy` (15 modules) would leave the app
import closure. The body was moved VERBATIM — same call, same keywords, same start gate.
Every closure probe was green, the new Packaging guard was green, the Notes and
`ProductionApp/test_notes_sync_runtime_lifecycle.py` suites were green.

Two `ProductionApp/test_file_notes_session_owner_lifecycle.py` tests went red anyway.
They replace `app.file_notes_session_owner` with a probe AFTER construction and before
mount. Under the eager build, `file_notes_binding=self.file_notes_session_owner.
current_binding` had already been read from the real owner during `__init__`; under the
lazy build the same line ran at mount and read the PROBE, which has no `current_binding`
— an `AttributeError` inside `on_mount` that also took the LibraryScreen mount with it,
so the sibling test failed with the unrelated-looking "production TldwCli did not mount
LibraryScreen".

**What to do.**

1. When you defer a construction, list every `self.<collaborator>` the moved body READS
   and decide, per name, whether the new read time is the same answer. Anything that can
   be reassigned between `__init__` and first access must be captured at the OLD time
   (`self._notes_sync_file_notes_binding = self.file_notes_session_owner.current_binding`
   in `__init__`), not re-read in the builder. Deferring *when* is the intended change;
   deferring *what it binds* is a silent second change riding along.
2. Import-closure evidence cannot see this class at all. A deferral's test set must
   include the suites that MUTATE the app object between construction and mount —
   here `Tests/ProductionApp/`, not just `Tests/Packaging/` and the deferred module's own
   unit tests.
3. Related trap from the same task: an AST fence that matches call names with
   `.endswith("build_notes_sync_runtime_owner")` (`Tests/Notes/test_notes_sync_cutover.py`)
   counts a `_build_notes_sync_runtime_owner` WRAPPER as a second call and fails its own
   `len(builds) == 1`. The wrapper was renamed `_construct_...`; if you add a wrapper
   around a fenced call, check the fence's matching rule before the name.

---

## A repaint-gate harness that forces a repaint per stimulus resyncs the gate and goes blind — validate the negative control PER MEMBER

**TASK-21122, 2026-08-23.** Gating Persona Buddy's ungated 10 Hz poll on a
"paint authority" tuple needs evidence that the tuple cannot miss a real
change. The natural harness: drive N stimuli, and after each one let the gated
poll settle, fingerprint the rendered view, then force an ungated repaint
(`_painted_authority = None; refresh_from_controller()`) and fingerprint again.
Any difference is a repaint the gate skipped. Forty-five stimuli, zero misses.

Then the negative control — deliberately dropping two tuple members — **also
reported zero misses**. The harness was blind, and its clean run had been
worth nothing.

Two causes, both worth knowing:

1. **The forced repaint resyncs the gate.** Every `_check` re-baselines
   `_painted_authority`, so a member whose only isolating stimulus arrives as
   a *sequence* (mouse-down, then a move that crosses a layout threshold) is
   repaired mid-sequence by the harness itself. The dedicated pin test, which
   never forces, caught exactly that member (`preferences.geometry`) when the
   harness could not.
2. **Eager handlers mask the poll.** The widget already repaints on
   `on_resize`, `on_descendant_focus`, `on_descendant_blur` and
   `on_mouse_move`, so several members are only load-bearing in the narrow
   window those handlers do not cover.

**What to do.** Never accept a differential harness's clean run without a
negative control, and do not settle for one crippled variant — drop *each*
member in turn and record which ones the harness can detect. That per-member
table is the real result: here it showed `screen.size` (3 misses) and
`display` (1 miss) caught by the harness, `preferences.geometry` caught only
by a dedicated non-forcing test, and the rest individually redundant with
`snapshot.generation` because the controller bumps its generation on every
state, preference and lease change. Redundant is not the same as wrong — those
members are cheap insurance against a future controller that stops bumping —
but you must know which is which before you claim the gate is proven.

A corollary from the same task: when a gate keys on an identity tuple, audit
the test fixtures for that tuple before believing a red. An earlier probe leg
"failed" because the fake frames were `SimpleNamespace(renderable=..., duration_ms=...)`
with none of the identity fields set, so `getattr(frame, "paint_digest", None)`
returned `None` for every frame and two visually different frames collapsed to
one key. The old code never noticed because it repainted every tick regardless.
## Count per-event work by instrumenting the call, not by reading the handler (TASK-21119, 2026-08-23)

**What happened.** The holistic-perf finding said the Console's click-outside
dismissal cost "~4 full-screen DOM walks per press": two `self.query(...)` calls
visible in the handler, times the two events (MouseDown + Click) of one press.
Shadowing `screen.query` on a real Console pilot and clicking said otherwise.
A press on the composer cost **3** walks, because the composer stops the Click
so the handler ran ONCE — and a single invocation costs three walks, not two:
the third is `screen.query(ConsoleSelectionMenu)` inside
`transcript._remove_selection_menu()`, a callee the review never opened. A press
on the rail cost **6**. So the reviewed number was simultaneously too high for
one press shape and 50% too low for the other, and both errors came from
counting call sites by eye.

**What to do.** For any "this runs N times per event" claim, instrument the
thing being counted and drive a real event; do not multiply what you can see in
the handler body. Two specifics that made the probe honest here:

- **Shadow the method on the instance** (`screen.query = counting_query`), not
  the class: it catches the callees that reach the same object by another route
  (the transcript's own `self.screen.query(...)` landed in the same counter),
  which is exactly where the uncounted work was hiding.
- **Measure more than one press shape.** Whether the second handler invocation
  happens at all depends on who swallows the Click, so a single target
  understates or overstates the per-press cost depending on which one you pick.

**Adjacent trap from the same task.** In zsh, an unquoted `$(...)` IS
word-split but an unquoted `$VAR` is NOT. Hoisting a file list into
`FILES=$(...)` and running `pytest $FILES` handed pytest one giant argument;
the run collected nothing, printed only a warnings block, and the compound
command still exited 0. It looked like a completed A/B. Inline the command
substitution, and read the passed-count before believing any comparison.

## Mandatory migration inputs require a real historical-reopen sweep (TASK-19900.1, 2026-08-22)

Delivery 1 made the v47→v48 Console Library seed explicit and fail-closed. Its
named 203-test foundation battery passed, but the controller-required complete
`Tests/DB/ Tests/ChaChaNotesDB/` sweep then produced 100 failures and four
setup errors: historical v4–v47 fixtures reopened through bare
`CharactersRAGDB(...)` calls, so they never supplied the new sanitized seed.
The same sweep also found stale v48 table/index/version inventories. A shared
`open_current_chachanotes_from_legacy(...)` test boundary fixed the reopeners;
the final sweep had no Delivery-1-induced failures.

When a schema upgrade adds mandatory constructor or migration authority, run
the complete database-owner subtree, not only the new migration tests. Give
historical fixtures one explicit opener that supplies sanitized authority at
the legacy-to-current boundary, while leaving the production missing-input
guard strict. Refresh absolute schema-version, table, index, and trigger
inventories in the same delivery.

## Private asyncio warning suppression makes lifecycle evidence vacuous (TASK-19900.3, 2026-08-23)

Task 13's first closed-loop cleanup probe set the private
`Task._log_destroy_pending` flag to false and manually called
`task.get_coro().close()` before garbage collection. The test then reported no
pending-task or never-awaited diagnostic and the implementation report treated
that as shutdown evidence. A replacement probe using only a public event-loop
exception handler, warning capture, and weak references exposed the real
boundary: awaited controller shutdown while the owner loop was alive reached a
terminal Task with no lifecycle diagnostic, but closing the loop first made
terminal cancellation/await impossible and correctly produced `Task was
destroyed but it is pending!` when the detached Task was collected.

For asyncio lifecycle tests, assert the supported ordering directly: call and
await the owner's shutdown API before closing its loop, then collect the Task
and owner and require no public diagnostic. Test an already-closed-loop
emergency separately: require fail-closed ownership cleanup and no dispatch,
capture the expected destroyed-pending diagnostic through the loop's public
exception handler, and assert warning capture contains no unhandled
never-awaited coroutine. Never mutate private Task warning flags or manually
close its coroutine; those operations change the behavior the test is meant to
observe.

## An isolated recovery widget does not prove one mounted action owner (TASK-19900.3, 2026-08-23)

Task 15's first recovery UI tests passed against the standalone Textual widget,
but the production ChatScreen never composed it. Once the real screen mounted
the region, a queued recovery exposed a second Retry/Discard projection through
the queue shelf, and a Button event delayed across navigation re-resolved the
new `active_session_id` instead of the owner displayed when the click occurred.

For recovery or destructive controls, mount the production hierarchy at empty
and non-empty companion-state counts, count each advertised action across all
sibling surfaces, and exercise the real callback after changing navigation
state. Keep companion UI limited to its own count/pause truth, and carry the
displayed session plus durable/ephemeral owner identity through the event rather
than looking up a mutable active selection at handling time.

---

## UI lifecycle tests must stop before optional native backends (TASK-21201, 2026-08-23)

**What happened.** The focused test for the Console Hands-free switch clicked the
real control and let `HandsFreeController.enter()` continue into audio capture.
During TASK-21201 verification that path initialized the optional parakeet-mlx
backend and aborted the test process. The test was intended to prove only that the
visible switch starts and stops the Hands-free UI lifecycle; opening a microphone
and loading a native model made it machine-dependent without adding evidence for
that contract.

**What to do.** In UI lifecycle tests, patch at the first owned boundary before
hardware, network, or optional native inference begins, while preserving the state
transition the UI observes. Test those integrations separately behind their own
explicitly marked suites. A test that can unexpectedly capture audio or initialize
a model is not a focused UI test, even if it normally passes on a configured laptop.
## An early `break` proves nothing when the list you scan was materialized for you (TASK-21121, 2026-08-23)

**What happened.** `_console_changed_files_scope()` ran on the Console's 0.2 s
run tick and carried a docstring arguing its own cost was fine: it scanned the
session's messages in REVERSE and broke on the first change-review marker, and
"markers cluster near the end", so "steady-state cost is near-constant"; the
`O(messages)` worst case was conceded only for a session with no marker at all.
Every clause was true and the conclusion was still wrong, because the thing
being scanned was `store.messages_for_session()` — which `dataclasses.replace`
-copies EVERY message in the session before the loop can look at one of them.
Measured on a 400-message session, 25 ticks: **10,050 message copies and
32.1 ms of event-loop time with a marker present** — i.e. the early break saved
nothing at all, and the "worst case" and the "steady state" cost the same.

**What to do.** When you reason about the cost of a scan, first ask who built
the sequence you are scanning. An early exit only helps over a lazily produced
or already-owned sequence; over an eagerly materialized copy the O(n) is paid
before your loop starts, and no amount of breaking early can reach it. In this
repo the shape to grep for is `for x in reversed(store.messages_for_session(
...))` — several sites still have it, and each one is a full transcript copy
regardless of how quickly it finds what it wants.

**Adjacent trap from the same task: a call counter counts your fixture too.**
The first cut of the counter probe wrapped `ConsoleChatStore._snapshot`
globally and reported 26 copies per 25 ticks AFTER the fix — which would have
read as "the fix is only partial". Those 26 were the probe's own
`append_message` + 25 `append_stream_chunk` calls, each of which returns a
snapshot; the subject's true count was 0. Arm the counter around the call under
test and disarm it for the fixture's own store traffic, or the setup you wrote
to make the measurement realistic gets billed to the code you are measuring.

---

## Sample a memo's signature BEFORE the work it describes, not after (TASK-21121 review round, 2026-08-23)

**What happened.** TASK-21121's verified memo stored
`(session_id, view_list, len(view_list), answer)` and re-checked all of it
before serving a hit — the `TokenEstimateCache` "no invalidation protocol to
get wrong" shape, and the reviewer's fuzzer found 0 violations in 138,565
probes. It was still wrong, because the length was evaluated on the line that
STORED the entry, i.e. after the scan that produced the answer:

```python
for message in reversed(view):   # answer describes the list as it is HERE
    ...
self._newest_change_review_memo = (sid, view, len(view), newest)  # ...but the
#                                             ^^^^^^^^^  length as it is HERE
```

An append landing in between records a post-append length beside a pre-append
answer. The signature then keeps matching forever, so the memo serves the stale
value for as long as the list object survives. And this is reachable: the
producer (`ConsoleAgentBridge._append_change_markers`) appends from the agent
WORKER thread with no `call_from_thread` marshalling while the consumer runs on
the event loop.

**What to do.** Snapshot every component of a verification signature at the
same instant as the value it certifies — hoist it above the computation. The
asymmetry is what makes this safe to reason about: recording a signature that
is *stale-short* only costs an extra miss, while one that is *stale-long* is a
permanent wrong answer, so when in doubt sample earlier.

**Two things this cost that generalise.** First, **a fuzzer that drives the
subject single-threaded cannot see an interleaving bug** — 138,565 clean probes
plus three passing mutation arms said nothing about this, because none of them
ever mutated the list *during* the scan. Reach for a deterministic interleaving
harness (here: a `list` subclass whose `__reversed__` performs the append after
creating the iterator) rather than more volume. Second, **a memo can make an
existing self-correcting glitch permanent**, which is a regression even when
the memo is new: the pre-memo code recomputed every tick and healed on the next
one, so "base had the same race" was true and irrelevant. Ask what the failure
DURATION becomes, not just whether the failure is new.

**Trap in the regression test itself.** That test is red at base too — but for
the wrong reason (`assert racing.fired`: base reverses a snapshot *copy*, so
the fixture's `__reversed__` hook never fires). Its real red-first evidence is
mutating the fix back out. A base red is not automatically evidence that base
has the bug; read *why* it failed.
## A filing's prescribed FIX is a hypothesis; only its own behaviour matrix can accept it (TASK-21128, 2026-08-23)

**What happened.** The finding was right about the defect: `messages_au` was
declared `AFTER UPDATE ON messages` with no column list, so every auxiliary
write to a message row — usage flush, metadata flush, variant bookkeeping —
re-tokenized the whole assistant reply into `messages_fts`. Measured over one
streamed turn: **4 index rewrites, `messages_fts_data` 55 → 12,636 bytes for a
single 400-token reply.** The AC also named the fix, in backticks: `AFTER
UPDATE OF content`. It is a one-line change, it matches the diagnosis, and it
is a **data-exposure bug**. Soft delete is `UPDATE messages SET deleted = 1 …`
and never names `content`, so the narrowed trigger would not fire and the
tombstoned message would stay in the search index — the exact guarantee
task-19567 exists to hold. Measured on a scratch matrix before any code was
written (a direct `messages_fts MATCH` returned the tombstoned rowid), and
re-proved afterwards by mutation: that shape turns all three `messages` cases
in `Tests/DB/test_fts_soft_delete_index_witness.py` red.

**What to do.** When an AC prescribes a mechanism, treat the mechanism as the
author's hypothesis and the AC as the outcome; build the behaviour matrix for
the mechanism BEFORE writing the diff, and amend the AC when the matrix
disagrees. For a trigger narrowed with `UPDATE OF`, the correct column set is
mechanical and worth writing down: **every column the derived artifact STORES,
plus every column that decides whether the row BELONGS in it.** Here that is
`content` (the only column `messages_fts` indexes) plus `deleted` (membership).
Anything less is a stale-index bug in one direction or a leak in the other, and
neither is visible through the six production `messages_fts` consumers,
because all six redundantly re-filter on `deleted` (`ChaChaNotes_DB.py:9131,
10318, 12496, 13935`; `RAG_Search/simplified/rag_service.py:2371, 2402` — the
two RAG ones are easy to miss, and the review of this task caught them missing
from an earlier draft of this very paragraph). So the failure such a mistake
produces is an INDEX-LAYER leak, not a user-visible one, which is precisely
why every witness for it must query the index directly.

The permanent form of that rule is a census, not a comment: derive the required
set from the LIVE schema (`PRAGMA table_info(messages_fts)` ∪ `{deleted}`),
parse the trigger's `UPDATE OF` list out of `sqlite_master`, and assert the
containment — so widening the fts5 table without widening the trigger fails at
authoring time.

**Adjacent trap from the same task, which cost a bogus baseline.** Editing a
production module WHILE a pytest run is in flight makes structural tests lie.
`Tests/DB/test_sql_debug_logging.py::…::test_no_eager_params_fstring_remains_in_source`
came back red in the baseline run; the test uses `inspect.getsource`, which
re-reads the file from disk using the ALREADY-IMPORTED code object's line
numbers, so inserting ~85 lines earlier in `ChaChaNotes_DB.py` mid-run made it
return a shifted, wrong span. It passed immediately on a re-run against the
stable file. This repo has many `inspect.getsource` / source-text structural
tests, so the failure looks exactly like a pre-existing red in someone else's
area. Do not edit the tree while a run you intend to quote is running — and
before A/B-ing any suspicious red, re-run it once against a quiescent tree.
## A filing's prescribed fix is a hypothesis, not a spec (TASK-21128, 2026-08-23)

The finding I filed said: scope the `messages_au` FTS trigger to `AFTER UPDATE OF
content`. The implementer built the trigger matrix *before* writing the change and
found that `OF content` alone does not fire on `UPDATE messages SET deleted = 1`, so
a soft-deleted message's tokens would have stayed in the index — a retention bug
introduced by the "fix". Shipped as `OF content, deleted` instead, and the acceptance
criteria were amended before any code was written.

**The rule this produced**, now enforced by a census test: the `UPDATE OF` column set
is *every column the index stores* ∪ *every column that decides membership or
visibility*. Widening an fts5 table without widening its trigger now fails at
authoring time.

**The generalisation**: a prescription written by whoever *found* a problem is a
hypothesis about the fix, and the person implementing it is the last one positioned
to falsify it. Build the differential harness first; if it contradicts the brief,
the brief loses. Two separate reviewers reproduced this independently — it was not
subtle, it was simply never tested before being written down.

*Scope honesty, also worth keeping:* I first wrote this up as "soft-deleted messages
stay searchable". Review corrected it — all six production `messages_fts` consumers
re-filter on `m.deleted = 0`, so the retention was reachable only by a direct index
query. Still a real regression of task-19567's guarantee, but not a user-visible
search leak. State the blast radius you can prove, not the worst one you can imagine.

## A thread-assert is only as honest as the double it runs against (TASK-21125, 2026-08-23)

The acceptance criterion prescribed offloading the Writing backend at the *controller*.
Every `WritingScopeService` method is already `async def`, so a controller-level
`to_thread` would have moved **zero** work in the shipped app — while still passing a
"runs off the event loop" assertion, because the test fake it ran against was
synchronous. The offload landed at `_service_for_mode` instead.

The same review turned up the matching failure of a *concurrency* assertion: a plain
`gather` of 8 writers passed against a deliberately broken single-thread pool. It only
began discriminating once every writer parked on a barrier inside the version check —
then the mutant failed 5/5. Before trusting either kind of assertion, **run it against
a deliberately broken implementation**; one that still passes is measuring the double,
not the subject.
## A `_maybe_await(service.call(...))` seam cannot be fixed by wrapping the VALUE — and the layer you were told to fix may not be the layer that blocks (TASK-21125, 2026-08-23)

**What happened.** The finding said the Writing screen "runs all SQLite on the
event loop" and the fix was "route the controller calls through
`asyncio.to_thread`". `WritingController` really is where the calls start, and
every one of them read
`await self._maybe_await(service.method(...))`. Two things make that shape a
trap:

- **The value is already computed.** `_maybe_await` receives a *result*, not a
  callable, so the synchronous work happened before the `await`. Anything you
  wrap around `_maybe_await` offloads nothing. (Findings 21126 and 21127 name
  the same seam in two other services — the pattern is repo-wide.)
- **The controller's callee was async, so offloading THERE would have moved
  zero work.** `WritingScopeService`'s ~70 methods are all `async def`; the
  controller awaits them on the loop and the blocking SQLite call happens one
  level down, inside the scope service. A controller-level `to_thread` would
  have passed a thread-assert against a *synchronous* test fake and still left
  the shipped app opening 180 connections on the loop.

The fix that actually worked was to wrap the *backend object* at the scope
service's single `_service_for_mode` dispatch point with a proxy whose
`__getattr__` returns `asyncio.to_thread(bound_method, ...)` for non-coroutine
callables (async backends pass straight through). One edit covered ~70 call
sites, every `_maybe_await` kept working unchanged, and `scope.local_service`
kept its identity — which a packaging test asserts.

**What to do.**

- Before writing a `to_thread`, follow the call one layer further and ask *which
  frame is actually synchronous*. An `async def` wrapper around a blocking call
  hides the blocking from the caller, not from the loop.
- Prove it with a connection/statement counter that records
  `threading.current_thread().name`, driven through the REAL object graph
  (controller → scope service → real `LocalWritingService` on a tmp file), not
  through the fake the UI tests use. Before: 180 opens, all on `MainThread`.
  After: 0 opens and every statement on `asyncio_0`.
- When a seam takes a value, change it to take the callable
  (`_call(method, *args)`), or wrap the object. Do not wrap the seam.

**Adjacent finding worth knowing (now TASK-21295).** `WritingController` calls
**seven** methods — `get_project_structure`, `autosave_scene`, `search_project`,
`assign_chapter`, `move_scene`, `reorder_items`,
`restore_version_to_working_state` — that exist on **none** of
`WritingScopeService`, `LocalWritingService` or `ServerWritingService`, none of
which defines `__getattr__`. The first is on the live, **unguarded** click path
(`Writing_Window._handle_project_selected` → `load_project_structure`, neither
with a try/except), so an `AttributeError` escapes a Textual handler and the
outline — the screen's whole purpose — cannot be loaded in the shipped app. It
is invisible because `Tests/UI/test_writing_screen.py` drives everything through
`FakeWritingScopeService`, which implements all seven.

I first found six of these and left them in this footnote. That was two
mistakes. **A green mounted-app test proves the controller talks to *a* service
correctly; it does not prove that service is the one the app wires** — so when a
perf task makes you read a screen's real object graph, diff the caller's
expected API against the wired backend's actual API, in both directions. And a
user-facing dead path belongs in a task file, not a lessons footnote: the
footnote does not get triaged.

## Moving work off the loop removes the serialization the loop was silently providing (TASK-21125 review, 2026-08-23)

**What happened.** The held-connection + `to_thread` change for the Writing
service was measured, mutation-tested and green — and it introduced a
near-certain silent lost update. `_update_row` reads the row and checks
`expected_version` in one committed transaction, then UPDATEs in the next. That
check-then-write split is **pre-existing and was harmless**: every scope call ran
inline on the event loop, so two writes could not interleave. Dispatching the
backend on the default thread pool made the window real — a reviewer's A/B
through the identical async graph measured **0/60 lost updates on base, 59/60
after**. Both writers were told they succeeded, `version` advanced once, one
writer's content vanished, and nothing downstream noticed.

Two more defects came from the same "make it concurrent" step, both in
scaffolding rather than the hot path: `close()` was called synchronously from an
async `on_unmount`, freezing the loop for the whole 5 s settle timeout (a 50 ms
ticker fired **zero** times) and then starving the very operation it waited for,
which surfaced as `Task exception was never retrieved`. And a dead-thread reaper
for the connection map never fired (recycled OS thread ids meant new threads took
the reuse branch) while its liveness test — absence from `threading.enumerate()`
— could not see a `_thread.start_new_thread` worker, so when it did fire it could
close a **live** connection.

**What to do.**

- Before offloading anything, ask what the single-threaded loop was **implicitly
  guaranteeing**. Ordering is the usual answer. Any check-then-act split in the
  code you are about to parallelise is now a race, whether or not you wrote it.
- The cheap durable fix is often a **single-thread executor**, not a lock or a
  merged transaction: it restores the exact ordering the loop gave you and keeps
  the whole latency win, because the point was getting off the loop, not going
  wide. (Merging check+write into one transaction here would have needed
  `BEGIN IMMEDIATE` too — with `isolation_level=None` and a deferred BEGIN the
  reviewer measured `OperationalError: database is locked`, because SQLite's busy
  handler does not retry `BUSY_SNAPSHOT`.)
- Make the race deterministic in the test or it will not hold. A plain
  `gather` of 8 same-version writers passed 3/3 runs against a deliberately
  broken 8-worker pool; parking every writer on a `threading.Barrier` **inside
  the version check** made the mutant fail every time with `assert 8 == 1`.
- A blocking `close()` called from async teardown needs `await
  asyncio.to_thread(...)`, and a settle timeout must **leave a busy thread's
  connection open** rather than closing it anyway — closing it converts a slow
  shutdown into `ProgrammingError: Cannot operate on a closed database` inside
  live work, which is the exact defect the settle wait existed to prevent.
---

## "No index" was wrong and it would not have mattered: 88% of that read was Python (TASK-21129, 2026-08-23)

**What happened.** The holistic-perf finding for the Notes-sync executor said its
six `list_bindings` sites had "no LIMIT, no `root_id` index" and prescribed
"indexed predicates + `to_thread`". Both halves of the diagnosis were checkable
before writing a line of fix, and both came back different:

- `idx_notes_sync_bindings_root(root_id, state, binding_id)` had existed since the
  feature's **first commit**, and `EXPLAIN QUERY PLAN` on the real thirteen-column
  statement showed SQLite already using it. Adding an index would have bought
  nothing and cost write amplification on every binding insert.
- Splitting the read showed where the time actually was: on a 1,000-binding root,
  `fetchall` was **1.83 ms of a 14.75 ms read** — the other **12.9 ms (88%)** was
  `_binding_from_row` building a dataclass, a nested serialization profile and an
  enum, per row, for call sites that kept **one string** off each record.

The fix that follows from the measurement is a different fix from the one that
follows from the filing: project only the columns the caller consumes (and answer
the existence question with `LIMIT 1`), rather than index anything.

**What to do.** Before accepting "this query is slow because it is unindexed",
run two cheap probes: `EXPLAIN QUERY PLAN` on the *actual* statement (not a
paraphrase with fewer columns — the reduced form can report a covering index the
real one cannot use), and a timing split of raw `fetchall` versus the full store
method. In this repo the shape to suspect is any `for x in store.list_*(...)`
whose loop body touches one or two attributes: the row-to-dataclass hydration is
usually the bill, and it is invisible to the query planner.

**Adjacent trap, same task: a "never crosses the boundary" assertion can be
vacuous.** The mutation battery ran eleven deliberate defects; ten were caught and
one survived — a probe rewritten to ignore its `root_id` filter entirely. The test
that should have caught it asserted "root-1 does not see root-2's binding" using a
note id that existed on **neither** root, so it passed for the wrong reason. The
fixture now gives the other root values that exist only there. Any negative
cross-scope assertion needs the value to genuinely exist in the other scope, or it
proves only that nothing matches nothing.
---

## An index is not evidence until the planner picks it in the state your users' databases are actually in (TASK-21126, 2026-08-23)

**What happened.** The holistic-perf finding said the Library Search/RAG
panel's legacy-chunk census ran an unindexed full-table `GROUP BY` and
prescribed "index or maintained count". The obvious index —
`(chunk_engine_version, media_id) WHERE deleted = 0` — is a textbook
COVERING index for that exact query, and the first probe confirmed it:
112 ms → 3.4 ms at 200k chunk rows, `SCAN … USING COVERING INDEX`, both
temp B-trees gone.

The probe had run `ANALYZE` to make the corpus "realistic". **No media
database in the wild has ever run `ANALYZE`** — there is no `ANALYZE`
anywhere in `Client_Media_DB_v2.py`, so `sqlite_stat1` does not exist on a
single user's disk. Re-measured in that state, the planner ignores the new
index completely and stays on the pre-existing `idx_...deleted`: **118.8 ms
without the index, 120.2 ms with it.** Five megabytes of disk, a schema
migration, and a green "the index exists" test, for a 1% change.

What actually works is counter-intuitive: lead the index with the
**redundant** `deleted` column that the partial predicate already pins to a
constant. That makes it answer the same equality search the no-stats
heuristic already likes, while additionally covering the GROUP BY and the
`COUNT(DISTINCT)` — chosen without stats, 23.4 ms at 200k and 122.8 ms at
1M. Four shapes were measured; two of them were never chosen at all.

**What to do.**

1. **`EXPLAIN QUERY PLAN` on a corpus built the way production builds one.**
   Before believing an index, grep the owning module for `ANALYZE` /
   `PRAGMA optimize`. If neither runs, your probe must not run them either —
   and say so in the probe, because "I made the corpus realistic" is exactly
   how the `ANALYZE` got in.
2. **Assert the PLAN in a test, not the index's existence.** `SELECT … FROM
   sqlite_master WHERE type='index'` passes for a dead index. The pin that
   catches this is the `EXPLAIN QUERY PLAN` string: index name present,
   `COVERING INDEX` present, `TEMP B-TREE` absent — plus an explicit
   assertion that `sqlite_stat1` is absent, so a future fixture that adds
   `ANALYZE` cannot quietly restore the flattering plan.
3. **Timing alone would not have caught it either.** 118.8 → 120.2 ms reads
   as noise; only the plan text says *why*. Measure both.

**Adjacent trap from the same task, worth its own line: offloading a query
to a thread can make a `:memory:` database silently return the wrong
answer.** `MediaDatabase` hands out THREAD-LOCAL connections. For a file
that is fine; for `:memory:` a worker thread opens a *different, empty*
database, so the census returns `{}` and the UI shows nothing instead of
raising. Deliberately breaking the guard proved it — the memory-backed test
went red with a zero count, not an error. Before wrapping any DB call in
`to_thread`, check what that owner's connection factory does with
`:memory:`, and keep a memory-backed test in the set.
---

## The stats-free planner is not a quirk of one query — sweep the whole database, and the fix may not be an index (TASK-21593, 2026-08-25)

**What happened.** TASK-21126 left an open question: it proved the no-stats
planner mis-chooses for *one* query, and nobody had looked at the rest. The
sweep — 39 production-exact statements against a 20,000-media / 200,000-chunk
/ 278 MB corpus built with **no `ANALYZE`** — found the same pathology
everywhere and two things worth generalising.

**1. The worst finding was not an index problem at all.** The Media search's
`must_have_keywords` filter measured **12.3 seconds**. `Keywords.keyword` is
already `UNIQUE COLLATE NOCASE`, but the predicate was
`LOWER(k.keyword) IN (?)` — and wrapping a column in a function makes it
non-sargable, so SQLite could not use the unique index and walked *every live
keyword for every candidate media row*. Deleting the redundant `LOWER()` —
one word — took it to 18.7 ms, **671×**, with no new index. It is reachable
from the chat scope picker per debounced keystroke. Before designing an index
for a slow query, read the WHERE clause for a function wrapped around the
column you were about to index.

**2. Shipping "the one obvious index" would have made three surfaces
slower.** `(deleted, is_trash, last_modified DESC, id DESC)` fixes nine list
queries by 10–1000×. It also *regresses* sort-by-date, sort-by-title and the
type facet by ~38% each — the planner switches to the new index because a
two-column equality beats a one-column one, then still sorts, and loses the
rowid-order locality it had. Three more indexes (each leading with the same
equality pair, differing only in the trailing sort key) take those to
0.08–2.5 ms instead. **A single-index A/B is not enough evidence: measure
every query the new index could be chosen for, not just the one you wrote it
for.** The mirror-image trap is real too — a narrow `(deleted, is_trash)`
index repairs the residual regressions and was *rejected* because the planner
then steals it for the ordered queries (sort=date_desc 0.46 → 26.05 ms, 57×
worse).

**3. `CROSS JOIN` is the only join-order instruction SQLite obeys.** The FTS
`COUNT` half of the same search had its join order inverted — Media outside,
one FTS probe per live row, 276 ms — while the ROWS half got it right because
its `ORDER BY fts.rank` forced the issue. Rewriting `FROM media_fts fts JOIN
Media m` as a plain `JOIN` changes nothing (266 ms; the planner reorders
straight back). `CROSS JOIN` gives 29 ms. But it removes the planner's freedom
permanently, so check the variants: with a five-id allowlist Media really is
the cheap side and the pin costs 0.10 → 1.92 ms. The fix is conditional.

**4. Mutation-test the DDL, and be honest about what the plan cannot see.**
Thirteen mutants, all caught — but two of them (dropping the partial `WHERE`,
reversing `DESC` to `ASC`) were caught *only* by the DDL-text assertion, never
by a plan assertion, because SQLite scans an ASC index backwards and a partial
index plans identically to a full one. Say which properties are pinned by
plan and which by text; a reader who assumes all thirteen were plan-caught has
been misled.

**What to do.** The convention is now mechanical:
`scripts/check_index_plan_pins.py` runs in `preflight.sh` and the required CI
job, and fails until every `CREATE INDEX` under `DB/` has a row in
`scripts/index_plan_pin_census.tsv` marked `plan-pinned` (a test naming it
alongside `EXPLAIN QUERY PLAN` *and* a `sqlite_stat1`-absence assertion) or
`pre-convention`. `Tests/DB/test_media_db_schema_v9.py` is the worked example,
including a negative control that keeps proving the rejected shape is still
never chosen.

**And the `ANALYZE` question, answered with a number so it stops being
re-asked.** On the fixed corpus `ANALYZE` costs 261 ms and buys almost
nothing the indexes have not already bought — but it is not free either: it
takes `get_deletion_candidates` from 1.9 ms to 13.9 ms. Its real wins
(`fetch_keywords_for_media_batch` 24.6 → 0.08 ms) are join-order fixes that
belong to whichever task owns those queries and can pin their plans. Running
`ANALYZE` globally re-plans every statement in the database at once, and every
plan assertion in the suite was captured without it. Do not add it as a
side effect of an unrelated change.
---

## A subsystem can have several migration entry points, and the product may not use the one you were pointed at (TASK-21130, 2026-08-23)

**What happened.** The finding cited `TTS/profile_schema.py:1439,1468` —
`_run_migrations`, reached from `open_profile_store` — as the place the v3→v4
climb snapshots the whole reference-BLOB table twice. That was accurate about
the code. It was wrong about the product: `TTSProfileRepository.
_worker_initialize_store` never lets `open_profile_store` migrate a populated
store. A below-current store is routed to `_worker_publish_migrated_store` →
`migrate_profile_store_to_candidate` → `step_profile_migration_candidate`, and
only *then* reopened, already at v4 (`profile_repository.py:1636-1638`). The
user-visible upgrade ran through `_step_candidate`, which carried the
**identical** double snapshot at different line numbers. Fixing only the cited
lines would have measured as a 966 MiB → 88 MiB win on a path a real upgrade
never takes, and shipped nothing.

**What to do.** Before optimising or hardening a migration, enumerate every
caller of the migration runner and find which one the app reaches at runtime —
`grep` the runner's name, then walk *up* from each hit to a boot or lifecycle
owner. Subsystems here routinely have three: an in-place opener, a disposable
candidate/validation copy, and a publish-through-a-candidate upgrade. They are
easy to mistake for one because they share the `MIGRATIONS` table and the same
helper functions. Fix the defect in the shared helper, not at the call sites
the filing happened to name.

## Sequential A/B arms measured a regression that interleaving erased (TASK-21130, 2026-08-23)

**What happened.** The candidate-path timings were taken as three base runs
then three fixed runs: base 10.38 / 10.57 / 10.77 s, fixed 11.43 / 11.58 /
12.87 s. Tight distributions, no overlap — it read as a clean ~1 s regression
caused by the change, which would have been reported as one. Re-running the
same two arms **alternating base, fixed, base, fixed** across five pairs gave
base 10.12 / 10.92 / 11.09 / 11.26 / 12.18 and fixed 9.62 / 9.68 / 10.05 /
10.47 / 10.75 — the fixed arm consistently ~1 s *faster*. Nothing about either
build changed; the machine's load simply drifted between the two blocks, and a
block design charges all of that drift to whichever arm ran second.

**What to do.** Interleave A/B arms, always. A block of N runs per arm gives
tight-looking intervals that measure *when* you ran, not *what* you ran, and
tightness inside a block is not evidence against between-block drift. This
matters most for the wall-clock half of a perf claim: the memory numbers here
were byte-stable across every ordering (1,013,84x,xxx vs 92,24x,xxx every
single run), which is exactly why an allocation-peak claim survives sloppy
scheduling and a wall-time claim does not.
---

## A memoized global makes N deferral sites ONE unit of work — fixing all but one banks nothing (TASK-21111, 2026-08-23)

**What happened.** The perf review filed four... three, it thought: "2–3
`keyring.get_keyring()` backend discoveries run during `__init__` (server
credentials ~13 ms; skills trust ×2)". A stack-tracing probe over a real
`TldwCli()` found **four** sites, and the ~13 ms did not belong to any of the
three named ones. The first keyring touch of every boot was
`Video_Generation/config._keyring_get`, reached from
`VideoStore.enforce_retention()` — a genuine `keyring.get_password` Keychain
query, 18.2 ms. The three named sites measured **0.33, 0.41 and 0.04 ms**.

The reason is the shape, not the accident: `keyring.get_keyring()` memoizes
its discovered backend in module state, so the expensive part is paid **once,
by whoever calls first**. Per-site timings therefore say nothing about what a
site COSTS; they say who happened to run first. Deferring the three cheap
sites would have promoted the fourth to first place, moved 0 ms, and still
passed an assertion of the form "no keyring from server credentials at boot".

**What to do.**

1. When the resource behind N call sites is a **memoized global** (a keyring
   backend, a lazily-populated config cache, a first-open DB connection, an
   import-time module init), stop attributing cost per site. Measure the
   aggregate — "how many touches, how many total ms, across import + construct
   + mount" — and make the ACCEPTANCE CRITERION that aggregate: *zero*, not
   *this site is lazy*. A per-site AC is satisfiable while the user pays
   exactly as much as before.
2. Write the guard as a spy on the **shared entry point** (here
   `keyring.core.get_keyring` and `get_password`), not on the individual
   callers. That guard caught, unprompted, the two later relocation bugs below.
3. Trace before you fix. The stack of the FIRST call is the finding; the
   others are consequences of it.

**Corollary, same task, same day: a deferral inside the boot-to-interactive
span is not a removal.** Moving the skills-trust construction out of
`TldwCli.__init__` into a lazy `skills_scope_service` property looked complete
— zero keyring calls in the construction probe. The mounted-app probe (Textual
`run_test`, spy still installed) then reported **16.45 ms of `get_keyring`
during MOUNT**: `ChatScreen._ensure_console_agent_bridge` reads
`skills_scope_service` when the default Chat destination mounts. The user's
time-to-interactive was unchanged. Only pushing the laziness one level further
— `LocalSkillsService` taking a `trust_service_factory` it calls on the first
trust decision — actually removed it. **`__init__` is not the boundary; first
paint is.** Any deferral probe that stops at construction can certify a
relocation as a win (see also TASK-21200's "deferring at the CONSUMER can move
the cost instead of removing it").

**Second corollary: a catch-all fallback can hide the mutant that proves your
test is blind.** The same task replaced a Python full-table scan with a
`json_valid`-guarded `json_extract` query, keeping the old scan as a
JSON1-less fallback behind `except Exception`. Two deliberate mutants — drop
the `json_valid` guard entirely, and demote it from `CASE` to an `AND` — both
**passed** the new differential test. Reason: `json_extract` really does raise
`malformed JSON`, the catch-all swallowed it, and the fallback scan returned
the right answer. The test was asserting the fallback's correctness, not the
query's. Two fixes were needed: assert `len(db.queries) == 1` (a fallback is
observable as a second query), and narrow the `except` to the one condition it
is for (`"no such function" in str(exc)`) so any other failure propagates.
A broad fallback under a perf fix is worse than no fallback: it turns "this is
now fast" into "this is fast unless one row is corrupt, in which case it is
silently as slow as before, forever."
## A docstring is not a measurement — and `Static.update` lays out by default

**TASK-21692, 2026-08-23.** `_render_visible_draft_only` in
`Widgets/Console/console_composer_bar.py` carried a docstring saying it "must stay
cheap and must not trigger a layout recompute on every blink phase". Its body called
`self.query_one(...).update(renderable)`. Textual 8's
`Static.update(content, *, layout: bool = True)` ends in `self.refresh(layout=layout)`
— so the default did the exact thing the docstring forbade, on a 0.53 s timer, for as
long as the composer merely held focus. Instrumenting the layout path under a real-CSS
harness put a number on it: 6 driven blink ticks produced **6 `Screen._refresh_layout`
calls, 6 full `Compositor.reflow`s, 396 `Widget.arrange` calls and 6 arrangement-cache
misses** — 3–6.5 ms of layout per tick, identical across six draft shapes. With
`layout=False` all six counters go to 0. Nothing was red before or after; the whole
defect lived in a keyword default nobody had to type.

Two things this cost, worth stealing:

- **Count the operation, do not read the source.** "Does this line cause a layout?"
  was answerable only by wrapping `Screen._refresh_layout` / `Compositor.reflow` /
  `Widget.arrange` and driving the tick. The wrong-looking answer (`Widget._arrange`,
  which does not exist in Textual 8 — it is `arrange`, and it is *cached*, so calls
  and cache-misses are different numbers) would have been reasoned about wrongly from
  the source in either direction.
- **Assert against a measured idle floor, not against `0`.** The committed test runs
  an A/A arm (the same number of event-loop settles with no blink) and asserts the
  blink arm costs no more. A bare `== 0` would become a flake the day an unrelated
  timer starts firing in that harness.

And the safety half has its own trap: `layout=False` is only sound if the size cannot
change. Here it is sound for two independent reasons — the caret cell is reserved in
*both* phases (glyph or space, wrapped in the same pass), and the Static's geometry is
pinned by inline styles (`width: 1fr`, `text_wrap: nowrap`, explicit
`height`/`min_height`/`max_height` from `_apply_draft_height`). The second reason was
discovered by mutation, not by reading: breaking the reserved cell changed the painted
row count from 1 to 2 between phases while `outer_size` stayed `Size(93, 2)`. A test
asserting only `outer_size` would have called that safe. Assert the **painted row
count and per-row cell widths** too.
## Fix the leak first: the per-op connect WAS the cost, so the offload the filing also prescribed banked nothing (TASK-21127, 2026-08-24)

**What happened.** The finding named three legs — per-op leaked connections, an
engine running as a loop coroutine, and a 30 s lease write plus a 2 s poll on
the loop — and prescribed a fix for each: held connection, `to_thread` the
engine's ~40 service calls, batch the keepalive. All three legs were real. Built
before any code changed, the harness put numbers on them: **one
`connect_private_sqlite` open with its pragmas costs 0.631 ms; the SELECT it was
opened to run costs 0.002 ms.** The connect was not a share of the cost, it was
99.7% of it. Holding the connection took an engine run's loop-side database time
from 877–906 ms to 29–33 ms per 20 runs (worst contiguous loop stall 46–51 ms →
2.0–2.5 ms) — and left the two remaining prescribed fixes with **1.6 ms per run**
and **0.020 ms per 30 s** to work on. Offloading the engine would have meant a
nested event loop (or ~40 cascading `await`s through 8 synchronous methods),
cross-thread Textual dispatch and notification dispatch from a worker thread, at
two separate call sites, to move 1.6 ms. Batching the keepalive would have
entangled the lease that prevents double execution to save 0.02 ms per half
minute. Both were declined with the measurement recorded in the ACs.

**What to do.** When a filing names a per-operation *leak* and an *offload* of
the same code path, treat them as sequenced, not parallel: fix the leak,
re-measure, and only then size the offload. The offload's apparent value is
usually the leak's cost wearing its clothes. Two corollaries. Keep the leg the
numbers *do* justify even when its headline reason evaporates: here the 2 s poll
was worth 0.023 ms a tick and would not have shipped on its own, but the same
one-line proxy also covers a bundle load that costs ~15 ms of loop time at 5.5 MB
and grows with artifact size — that, not the poll, is the case for the UI-side
offload. And measure the loop with an independent ticker, not with the awaited
call's wall time: once work is on a thread the coroutine's duration *includes*
the thread's work and stops being a loop-blocking measure at all. The sharpest
number in this task came from that ticker — at 1 ms it got **zero** wakeups
across the whole base-arm window, i.e. the shipped UI research path never
yielded to the event loop even once.

## Two mutations stayed green because a SECOND line of defence covered the walk — the fix was more tests, not a weaker assertion (TASK-21127, 2026-08-24)

**What happened.** Fifteen mutations were run against the new guards; thirteen
went red immediately. Two did not: removing `_begin`'s stale-connection heal, and
removing `_transaction`'s rollback-on-failure. Both were aimed at end-to-end walk
tests ("quit mid-run still completes", "a DB error mid-run leaves the store
usable"). Investigating instead of rewriting the assertion showed why: `close()`
POPS every connection it closes, so a later operation opens a fresh one and never
meets a closed handle — the heal is a genuine second line of defence and
unreachable through the shipped `close()`. Likewise a transaction left open by a
missing rollback is cleared by `_begin`'s *other* heal arm on the next operation,
so the end state matched either way. The walks were correct; they were just not
single-mechanism guards.

**What to do.** A surviving mutant has a third hypothesis beyond "wrong scenario"
and "something else fixed it for me": **the code is deliberately
belt-and-braces, and no single-point mutation can red a whole-path assertion.**
That is a property worth having, not a defect — but it means the redundant
mechanism needs its own test at its own level. Here: a connection closed *while
still mapped* (reaching the heal directly), and `conn.in_transaction` after a
failed body (proving the rollback happens AT the failure, not on the next
`_begin`). Both mutations then went red. Keep the walk as well; deleting it to
satisfy a mutation score would trade the property for the proof of it.

## `inspect.iscoroutinefunction` is False for an async GENERATOR — a thread-offload proxy must pass those through too (TASK-21127, 2026-08-24)

**What happened.** `ResearchScopeService` got the same `_ThreadOffloadedBackend`
proxy TASK-21125 built for the writing suite: `__getattr__` returns the attribute
unchanged when it is already async, and wraps it in `asyncio.to_thread` otherwise.
Copied verbatim, the predicate (`inspect.iscoroutinefunction`) reported **False**
for `LocalResearchService.stream_run_events`, which is `async def ... yield`. The
proxy therefore "offloaded" it, handing `stream_run_events`/`observe_run_events`
a coroutine wrapping the generator object instead of the generator, and both
consumers' `inspect.isasyncgen(result)` branch silently stopped matching. The
writing service has no async generators, so its predicate had never been wrong.
Mutation-confirmed: dropping `inspect.isasyncgenfunction` from the predicate
fails the test with `TypeError: 'async_generator' object is not iterable`.

**What to do.** Any "is this already async?" predicate guarding a thread offload
must test `inspect.iscoroutinefunction(x) or inspect.isasyncgenfunction(x)`, on
the callable *and* on its `__call__`. When cloning a proxy between services — and
this burn-down does that repeatedly (21125 → 21127 → 21131) — inventory the target
service's method KINDS first (`async def`, `async def ... yield`, plain `def`,
plain generator), because the predicate is only as complete as the service it was
written against. A plain sync generator has the mirror-image problem: routing it
through a thread returns the generator without running any of it.
---

## An import-weight guard cannot see a cost that merely moved to screen mount (2026-08-24)

**TASK-21731.** `import tldw_chatbook.app` had grown from 636 to 703 of this repo's
own modules — the whole `Chunking` engine, the `RAG_Search.simplified` tree and
`Internal_Prompts`, all in front of first paint. One module-scope import caused it
(`Library/library_local_rag_search_service.py` reading `normalize_rag_search_mode`
from `simplified.active_config`, a three-element frozenset lookup), and deferring
that one import took the count straight back to 637 and turned two red guards green:
`test_app_import_weight.py`'s own-module budget and
`Tests/Packaging/test_chunking_import_closure.py`, **both of which had been failing
on dev, so every branch inherited them red and the next regression would have been
invisible**.

That fix, on its own, bought the user nothing. A time-to-interactive probe (import →
`TldwCli()` → `run_test()` → `_ui_ready`, censusing `sys.modules` at readiness)
showed **Chunking 34, simplified 18, Internal_Prompts 10 still resident when the app
became usable**: `Event_Handlers/Chat_Events/chat_rag_events.py` — imported during
the *initial Chat screen mount*, on the event loop, via
`UI/Console_Modules/retrieval.py` — ran a module-scope `try: from
...RAG_Search.simplified import ...` availability probe. Timed with an import tracer,
that single edge cost **50 ms** at mount. The eager boot import had merely been
paying it early. Deferring the probe too (resolve on first ask, PEP 562
`__getattr__` keeping the public flag readable) took residency at readiness to
**Chunking 0, simplified 0** and the total from 984 to 928.

**What to do.** `import tldw_chatbook.app` is a *budget*, not a boundary. Before
claiming a deferral removed work, census `sys.modules` at `_ui_ready`, not after the
import — and when the count does not drop, trace which module pulled it back in: the
second importer is usually the default screen's mount chain, which every user walks
anyway. This is the third recorded instance of the same shape (`__init__` is not the
boundary, first paint is — TASK-21111; deferring at the CONSUMER can move the cost —
TASK-21200); the instrument that settles it in one run is an import tracer that
records the *importing module* and the *duration* of each first import, not a
count.

**Corollary on scope.** The bisect that filed this named three files as the cause
(`MCP/server.py`, `MCP/tools.py`, `UI/MCP_Modules/mcp_inspector.py`). None of the
three was on the boot path: `MCP/server.py` is in the closure but imports no RAG,
`MCP/tools.py` (which *does* eagerly import `simplified.search_service`) is not in
the closure at all, and `mcp_inspector.py` imports no RAG. A tracer that records
`(importer, imported)` edges for the whole boot answered this in one run and disagreed
with a plausible reading of the diff — run it before believing any reported chain,
including one that comes with a bisect.
## A production gate that fails closed on a capability the test double lacks is invisible — the symptom is "nothing happened" (TASK-21590, 2026-08-24)

**What happened.** `Tests/UI/test_console_native_chat_flow.py` was 26 red on dev and had
been for a day. Every failure looked identical and told you nothing: press Send, wait,
`AssertionError: Text not found: 'generic provider response'` with a screen dump showing
`No messages yet.` and the draft still sitting in the composer. No exception, no system row,
no toast — pressing Send simply did nothing.

The cause was a fail-closed gate meeting a double that had never needed the capability it
now demands. TASK-19900.3's `56db75386` changed the durable-turn gate from

```python
if persistence is not None and persistence.db is not None and ... and not callable(durable_commit):
    return self._block(session.id, "Durable turn acceptance is unavailable; ...")
```

to

```python
if durable_turn and not callable(durable_commit):
    return ConsoleSubmitResult(False, False, "Durable turn acceptance is unavailable; ...")
```

Two independent changes in one hunk. The first made `persistence=None` — which every
`_build_test_app` app has, since the factory patches `get_chachanotes_db_lazy` to `None` —
count as a durable turn that cannot commit. The second dropped `_block()`, which was the
only thing that put the refusal on screen. So the gate went from "refuses loudly in a
configuration tests never hit" to "refuses silently in the configuration every mounted
Console test hits". The commit's own report says plainly: *"No full repository sweep was
run."*

Repairing that surfaced **two more stale doubles underneath it**, both hidden by the first
refusal firing earlier in the same function: a fake gateway whose ready resolution had no
`resolved_destination` (mandatory since `a26cdafd8`), and a hand-rolled three-method
persistence double with no `commit_durable_turn`. Three layers of the same defect class, and
you only ever see the outermost one.

**What to do.**

1. **Trace before you read the tests.** The failure text named the assertion, not the cause.
   Wrapping the real seams in order — button handler → `ConsolePromptQueueUIController.
   dispatch` → `ConsoleChatController.run_prompt_chain` — printed
   `ConsoleSubmitResult(accepted=False, visible_copy='Durable turn acceptance is unavailable;
   the provider was not called.', provider_started=False)` on the first try. `git log -S` on
   that copy then named the commit in seconds. Reading 11,587 lines of test file first would
   have found nothing, because nothing in the test was wrong.
2. **A fail-closed gate must fail *visibly*.** If a refusal returns a bare result instead of
   going through the path that writes a system row, it is indistinguishable from a dead
   button — for the suite *and* for a user in the degraded state the gate exists to catch.
   When you harden a gate, check what the user sees when it fires.
3. **Expect the double to be the stale half, and expect more than one.** After fixing the
   first, re-run and read the *new* failure text rather than assuming you are done: each
   repair moves the failure one gate further down.
4. **Prefer the harness shape production already names.** `ConsoleRuntime.ensure_agent_bridge`
   explicitly returns `None` for a `:memory:` DB ("an in-memory harness still builds
   neither"). Attaching a *file-backed* ChaChaNotes DB fixed the durable gate but also
   switched 26 tests onto the agent loop, because `[console] agent_runtime` defaults on and
   the bridge keys off a real `db_path`. `:memory:` restored exactly the lost precondition
   and nothing else. When production distinguishes a harness case, use that case.
5. **A red test may be the only honest thing in the file.** Two of the 26 stayed red after
   the harness was correct, and a live run of the real app confirmed why: Send really is
   disabled for the whole of every Console run, showing a greyed-out button labelled "Queue"
   whose tooltip says to wait. Two shipped contracts disagree (ADR-098 vs the durable-owner
   submission block, each with its own test asserting the opposite). That is an owner
   decision, not a test edit — filed as TASK-22000 and left red.
## A held-connection map keyed by thread does nothing until `check_same_thread=False` — and the test that should have caught it passed for the wrong reason (TASK-21131, 2026-08-24)

**What happened.** TASK-21131 ported `EventStateRepository` to held connections
using the shape the sibling tasks had established: `dict[thread_ident,
Connection]` "so shutdown can reach connections it does not own", plus a
depth guard so `close()` refuses a connection that is mid-operation. A
deliberate mutation — *let `close()` close a busy connection anyway* — was
expected to reproduce the TASK-21101 signature `ProgrammingError: Cannot
operate on a closed database` inside the live operation. It stayed **GREEN**.

The reason was not the test. The connections were opened through
`BaseDB._get_connection`, which passes no `check_same_thread`, so sqlite3's
default guard was on: `close()` running on the main thread against a worker
thread's connection **raised instead of closing**, and the best-effort
`except Exception: pass` swallowed it. The whole rationale for the dict — that
shutdown can reach foreign threads' connections — was **false for this store**,
and the depth guard it justified was unreachable code. The sibling template
named in the task (`ClientNotificationsDB`) is self-consistent about this: it
uses `threading.local` and only ever closes the calling thread's connection.
The two designs are coherent; the hybrid I had written was not.

A second, worse detail surfaced when the fix was tested. The new guard —
"`close()` from the main thread really does close a worker's idle connection" —
was written as:

```python
with pytest.raises(sqlite3.ProgrammingError):
    worker_conn.execute("SELECT 1")
```

That passes **whether or not the connection was closed**: a connection merely
*refused to this thread* raises the same exception class. Under the mutation
that removed `check_same_thread=False`, it passed. `match="closed database"`
was required to make it mean anything.

**What to do.**

1. `dict[thread_ident, Connection]` and `threading.local` are **the same
   thing** unless the connections are opened with `check_same_thread=False`.
   If you copy the dict shape from another store, copy the connect kwarg with
   it, or copy `threading.local` instead and say so — do not ship the dict plus
   a shutdown guard that can never run.
2. In this repo the private-SQLite seam **enforces that a module only names
   its own registered owner id**
   (`Tests/DB/test_private_sqlite_inventory.py::test_private_sqlite_seam_calls_use_literal_module_owned_ids`).
   So "call `connect_private_sqlite` yourself to pass one extra kwarg" is not a
   local edit: it forces the module's own registry entry to describe the
   targets it actually opens. Here that was a correction, not a widening — the
   entry claimed "`memory`" while the app has been handing the store a private
   file all along, opened under `db.base` (identical allowed target kinds).
   Budget for the registry entry and `backlog/docs/sqlite-private-owner-inventory.md`.
3. `pytest.raises(SomeError)` on a **connection or handle** is a vacuous
   assertion by default: closed, wrong-thread, and wrong-state all raise
   `ProgrammingError`. Always `match=` the condition you mean. The mutation is
   what exposes it — a guard that passes under the mutation that deletes its
   subject is not testing its subject.

**Corollary from the same task: a concurrency test whose threads open their
connections AFTER the barrier has no race window.** Two interleaving guards
(12 threads recording the same event) went red against `BEGIN` instead of
`BEGIN IMMEDIATE` — until the acquisition of a held connection was moved under
the same lock hold as its in-flight registration. Then they went green under
that mutation. Connection creation serialises on that lock, so a thread that
opened *after* the barrier was released into a window where every earlier
writer had already committed; each late thread simply saw the committed row
and reported "duplicate". Warming each thread's connection **before**
`barrier.wait()` restored the failure. If the first thing your barriered
workers do is acquire a shared, lock-serialised resource, the barrier is
synchronising the wrong instant.

---

## When the guarantee is "this is never invoked", assert on the CALL, not on the effect

**TASK-21113, 2026-08-24.** The screen pre-importer gained a proportional
sleep between route imports. task-21110's initial-screen warm-up shares that
same method with a one-route list, so the guarantee for it is "nothing is
added in front of the single import". The test asserted that no `time.sleep`
happened. Mutating `if index:` (pause between routes) to `if True:` (pause
before every route) **survived**: the first route's pause is computed from a
`previous_cost` of `0.0`, so it sleeps nothing — while still probing the
navigation lock and still being able to park. The observable effect the test
watched was absent under the defect; the defect was not.

**What to do.** Where the contract is "not invoked", spy the invocation
(`monkeypatch.setattr(obj, "_pause_between_preimports", calls.append)`) rather
than one of its downstream effects. A zero-valued or short-circuiting call is
invisible to an effect-based assertion and is exactly the shape a refactor
introduces. Effect-based assertions are fine for "invoked with X"; they are
not evidence for "not invoked at all".

**Same task, cheaper trap:** a poll bound built by accumulating a float
(`waited += 0.05` against a `5.0` limit) is off by one at random — a hundred
additions of `0.05` do not reliably reach `5.0`. Count polls against an
integer budget derived once from the two constants.

## A wrong identity space can make a readiness test pass without exercising readiness (TASK-21508, 2026-08-24)

**Incident.** A missing-embeddings test expected Hybrid retrieval to exclude an
FTS-only Local source. The test passed, but an inverse mutation from
`fts and vector` to `fts or vector` also passed. The desired-selection fixture
had supplied the Local membership ID, while the canonical Local `RagScope`
stores Media IDs; the source was excluded before the readiness predicate ran.

**Rule.** When a result is an intersection of identity, selection, support, and
readiness gates, mutate each gate independently. A green expected-empty test is
not evidence for the named gate until making that gate permissive turns the test
red. Fixtures crossing association and canonical-owner boundaries must state
which ID space each value occupies.
---

## A process-global record drained at every teardown charges one test's background work to a bystander (TASK-21592, 2026-08-24)

**TASK-21592, 2026-08-24.** The filing said `Tests/App` polluted `Tests/Library`:
13-15 errors with node ids that varied run to run, zero when either directory
ran alone. Reproduced on the filing's own base (`f49956038`): 10 errors together,
0 for `Tests/Library` alone. Every one of the ten was a **teardown** error on a
test that had itself **passed**, and every one read the same thing —
`socket.create_connection -> huggingface.co:443`, with `huggingface_hub` logging
`Retrying in 1s [Retry 1/5]`.

Two globals in series produced it. `huggingface_hub.constants.HF_HUB_OFFLINE`
was frozen `False` at import (nothing set it), so a Library test that built the
default embedding model against an empty cache reached the real hub. Then
`Tests/network_guard._blocked_attempts` — a module-level list drained and
asserted empty at *every* test's teardown — collected the five retries, which
run on a worker thread with 1/2/4/8/16s backoff and outlive the test that
started them. So the attempts landed on tests B, C, D. The "zero when alone"
result was not the absence of the fetch: it was too short a tail of subsequent
tests for the retries to land on.

The count and the ids therefore carried no information about the culprit. Nine
of the ten node ids named innocent tests, and bisecting on them would have led
nowhere.

**What to do.** When a suite-wide guard keeps a process-global record and
asserts it empty per test, record provenance with each entry — at minimum
`threading.current_thread().name` — and say so in the failure message when it is
not the main thread. Read the error *body* before the node ids: a batch of
failures that all carry the same address and differ only in which test they
landed on is one background actor, not N flaky tests. And check the plugin list
before trusting an ordering claim: `-p no:randomly` is a no-op in this venv
because `pytest-randomly` is not installed, so it neither proves nor disproves
anything about order.
## Three green tests pinned the requirement; the red one was the only test that opened the database like a stranger (TASK-21441, 2026-08-24)

**TASK-21441, 2026-08-24.** Schema v48 made `_migrate_from_v47_to_v48` raise
unless the constructor was handed a `ConsoleLibraryMigrationSeed`, exempting a
*fresh* database — so `CharactersRAGDB` could no longer upgrade an existing one
without the caller supplying data only the TUI knows how to build. Three tests
covered that behaviour and all three were **green**, because each asserted the
requirement rather than the property the requirement destroyed:
`test_v47_missing_or_invalid_seed_leaves_schema_unchanged[None]`,
`test_v47_upgrade_rejects_missing_or_invalid_seed_before_ddl[None]`, and
`test_fresh_database_accepts_no_console_library_migration_seed` (which passes
either way — a fresh database was the exempted case). ADR-079 even listed
"a required sanitized migration seed" as an accepted trade-off.

The test that told the truth was
`Tests/Packaging/test_installed_distribution.py::test_installed_distribution_migrates_v35_database_to_current`,
which installs the wheel into an empty tree and opens a v35 database from a
child process with nothing but a path and a client id. It is the only test in
the repo that opens the database the way a stranger would, and it was red.

**What to do.** A test that asserts *your API raises* is not evidence the
requirement is correct; it is a transcript of the decision. When a change adds
a precondition to a component, ask which test exercises that component **from
outside every convenience the change assumed** — an installed artifact, a
child process, a caller with no access to app config — and check it is not the
one that just went red. If the only red test is the one with the fewest
privileges, the defect is in the precondition, not the test. Fixing it by
teaching that test to supply the argument deletes the repo's last witness.

**Corollary for fixtures.** The same release broke `add_message` for pre-v48
schemas by naming the newest column list unconditionally, which stopped
`Tests/ChaChaNotesDB/historical_bootstrap.py` from seeding historical fixtures
with production code — the doctrine task-16840 established precisely because
hand-rolled fixture SQL had been silently wrong. When a schema addition makes
the production writer unusable against an older schema, the pressure is to go
back to hand-rolled SQL in the test. Resist it: that trades a loud failure for
a silent one. (Writing that hand-rolled INSERT is also harder than it looks —
the first attempt at one in this task died on a `NOT NULL` column the writer
fills for you.)

## Count the calls with a stack probe before wiring a shared-state fix at the call sites you know about (TASK-22201, 2026-08-25)

The perf review said the Console run tick builds
`_build_console_workspace_context_state()` **three** times (two rail-state
legs + the workspace-context push), so the first fix threaded one prebuilt
state through exactly those three call sites. The build-once gate test then
failed with `count == 6`: a stack-capturing probe showed the other three
builds arriving through call chains the review never named — the inspector
rows leg (`_selected_console_conversation_inspector_rows`) builds the
workspace state inside BOTH rail-state calls, the control bar's own inspector
build, and the agent section's payload lambda. Parameter-threading at the
named sites would have shipped "fixed" while leaving 4 of 6 builds in place —
and only the gate asserting on the COUNT, not on the three known sites,
caught it.

Two rules. **Measure the call multiplicity with a stack probe (wrap the
function, record `traceback.extract_stack`) before designing the fix** — a
finding's count is a lower bound from the paths its author traced. And when
the calls converge on one function from many sites, **share at the converged
function (here: an opt-in, asyncio-task-scoped cache consulted inside the
build itself), not by threading state through each caller** — the callers you
did not know about are exactly the ones a threading fix misses.

---

## A programmatic `.focus()` is not the user's keystroke — Library focus semantics gate on the INPUT that moved focus

**TASK-22207, 2026-08-25.** The red-first probe for "arrow-keying the Media
Items list must not rebuild the Reader body" drove the traversal with
`row.focus()` + `pilot.pause()` — and the Reader silently ignored every one
of them: no selection, no pending request, and the settled-row wait died on
"Detail call did not start". Nothing was broken. `LibraryScreen.
on_descendant_focus` deliberately treats a focus change as user intent only
when a real input event could own it: `on_resize` arms
`_library_notes_resize_settling` (True from the mount resize onward in a
Pilot harness), and only real input handlers (`on_key`, `on_mouse_down`,
wheel) clear it via `_mark_library_notes_user_interaction()`. A programmatic
`.focus()` fires `DescendantFocus` without any of those, so the screen
classifies it as resize/restore noise — exactly as designed, to stop
background recomposes from yanking focus.

**What to do.** In a mounted-screen test, drive the gesture, not the state:
`await pilot.press("down")` (which routes through `on_key`, clears the
suppression, and moves row focus via `_move_library_list_row_focus`) is the
traversal; `row.focus()` is only the framework's half of it. If a
focus-driven behavior mysteriously does not fire in a Pilot harness, check
which interaction-intent flags the screen consults before dispatching —
and prefer the input event that a user would actually produce.
---

## tracemalloc taxes the arm you are trying to indict — never take the timing and the allocation in the same run (TASK-21532, 2026-08-25)

**What happened.** Comparing a full `list_bindings` hydration against a narrow
`binding_id` projection, the first harness wrapped each arm in
`tracemalloc.start()` / `take_snapshot()` and read the wall clock inside that
window. At 1,000 bindings it reported **92.4 ms** for the read-all and 0.81 ms
for the projection — a 114x win, and a number that would have gone straight
into the task file. Re-running the identical arms with tracemalloc off gave
**15.3 ms vs 0.30 ms**. The read-all had not got faster; tracemalloc charges
per *allocation*, and the whole point of the arm under test is that it
allocates a dataclass, a nested profile and an enum per row. The profiler was
taxing exactly the thing being measured, in proportion to the thing being
measured, and inflating the reported win six-fold. (15.3 ms is also the number
TASK-21129 measured on the same read, which is how the error was caught: the
92 ms did not reconcile with the prior art.)

The same first harness got the allocation half wrong too, in the opposite
direction. It diffed `take_snapshot()` before and after the call, but the call
*returns* only the projected ids — the hydrated tuple is already freed by the
time the second snapshot is taken, so the diff measured what survived rather
than what was allocated, and reported 265,960 B where the peak was 1,058,854 B.

**What to do.** Two separate loops, always: wall time with the profiler
**off**, allocation with `tracemalloc.start()` +
`tracemalloc.reset_peak()` + `get_traced_memory()[1]` — the **peak**, not a
snapshot diff, whenever the allocation you care about is transient. And when a
new measurement of an old subject disagrees with the prior art by an order of
magnitude, suspect the harness before you believe the win.


---

## Making an early step faster can UN-swallow a request whose guard is checked at handle time (TASK-21591, 2026-08-25)

**What happened.** Fixing the splash's dead `skip_on_keypress` made a keypress
dismiss the splash immediately instead of at its 1.5 s timer. The first version
let the key keep bubbling, so app-level bindings would still work during the
splash — reasoned about carefully, documented, mutation-tested, and green
across four splash suites *and* two real-terminal arms. A wider sweep then
red-lined `test_navigation_keypress_during_splash_is_safely_ignored`
(task-1339's crash lock, in `Tests/UI/test_screen_navigation.py` — a file that
does not mention the splash): **F9 mid-splash now landed the user on
Settings.**

The mechanism is invisible at the call site. `action_shell_destination` does
not navigate; it **posts** a `NavigateToScreen`, and the "initial screen not
yet mounted" guard lives in the *handler*. Message order on the App queue is
`Key` → `Closed` → (the key's binding posts) `NavigateToScreen`. While the
splash closed on a timer, `Closed` arrived long after the navigation had been
handled and swallowed. Once the splash closed on the key, `Closed` was already
queued ahead of the navigation, so the initial screen had been pushed by the
time the guard ran and the guard correctly did nothing. Nothing about the guard
changed; **the latency it implicitly depended on did.**

**What to do.** Two things.

1. When a guard reads state that some *other* in-flight message will set, it is
   a race with an ordering you did not write down. Before speeding up any step
   in a startup or teardown sequence, enumerate what is queued behind it and
   ask which guards are relying on the old arrival order. A "post, then check
   on handle" action is the tell: the check is not where the action is.
2. A change to **input routing** — focus, `event.stop()`, key handlers,
   bindings — must be run against the **navigation** suite, not just the
   feature's own. Here the feature's four suites, its five purpose-built tests
   and two live tmux arms all passed the version that had to be withdrawn; the
   only thing that caught it was 506 unrelated navigation tests, and it cost
   5 minutes to run.

## A mutation can escape every integration gate whose geometry structurally cannot reach the mutated leg (TASK-22203, 2026-08-25)

The TASK-22203 fix scoped the workspace action-row flip so a Tree cursor
boundary crossing no longer fans out into the rail-wide allocation pipeline.
It has two halves: the direct `request_allocation_reconcile()` call was
removed, and `_ContextBoundedSection`'s demand-change escalation was taught
to skip a scoped pass. Mutation-testing the second half — deleting the
scoped skip outright — left BOTH mounted-console boundary gates green: in
every mounted geometry (the real console at 160×44, the rail harness at
60×30, even a widened 90-col rail), the workspaces tray sits at its 12-row
`max_height` cap with `overflow_y: hidden`, so the action-row flip changes
`desired_content_lines` by exactly zero and the escalation leg the mutant
deleted never runs. A debug snapshot proved it: `fixed=12` before and after
the flip in every configuration tried. The dev test that seemed to pin a
`hidden_demand + 1` delta for the same flip was actually measuring a
coincident tree-expansion delta the flip's reconcile recorded late.

The follow-on trap: the first replacement gate mutated a real child's
`styles.height` to create a demand delta — and the scoped pass STILL
escalated, because the style change resized widgets in the same frame and
their `on_resize` handlers issued plain requests that legitimately demoted
the scoped pass (the demotion rule working as designed, masking the skip
under test). The gate that finally killed the mutant stubs the measurement
seam (`section._measure_content_lines = lambda v: value`) so the demand
delta arrives with no DOM churn at all.

Two rules. **After a mutation survives, prove which branch the gate's
geometry actually executes before writing a stronger assertion** — a
surviving mutant is evidence about the gate's reach, not only its strength.
And **when the guarded leg is unreachable in production-shaped geometry,
gate the mechanism directly at a stubbed seam** and say so in the test's
docstring; an integration test that cannot reach the leg is the wrong tool
no matter how production-faithful it looks.

## Interleaved A/B pairs carry a positional bias — run the A/A control and the reversed order before believing the delta (TASK-22213, 2026-08-25)

Ten interleaved boot-to-`_ui_ready` pairs (base first, branch second, the
obvious loop shape) showed the branch faster in 8/10 with a median of about
−250 ms — a suspiciously large win for a ~30-module import diet. An A/A
control (base vs base, same loop shape) exposed the mechanism: the
SECOND-position run in a pair is systematically faster (warmer FS cache, load
settling), and the A/A spread alone was ±400 ms on that loaded machine. Five
pairs with the ORDER REVERSED (branch first) collapsed the "win" to a wash
(median −19 ms, mean +85 ms).

Interleaving controls for load DRIFT between pairs; it does not control for
position WITHIN a pair. Before reporting any interleaved delta: (1) run an
A/A control with the identical loop shape to measure the noise floor and the
positional offset; (2) run at least a few pairs in reversed order; (3) if the
effect survives neither, report the wash and lean on the deterministic axes
instead — module censuses and closure diffs don't have a noise floor (here:
−32 modules on the bare chat_screen leg, −5 at warm `_ui_ready`, both exactly
reproducible while the wall-clock delta evaporated).

---

## A function-scope import is not lazy if the function runs at module import

**TASK-22223, 2026-08-25.** `config.py`'s `_load_settings_uncached` imported
`Library.library_adaptive_reader_state` under a comment stating "Lazy import
avoids pulling the Library package through config's module initialization
path" — but `load_settings()` is called at config **module scope**, so the
"lazy" import fired on every `import tldw_chatbook.config`. The Library
package `__init__` dragged 66 feature modules (collections/tool services →
Sync_Interop → Chat → Skills_Interop → runtime_policy) into every config
import, and closed a live cycle: any module importing
`runtime_policy.bootstrap` before config (e.g.
`Character_Chat/server_character_persona_service.py`) died with
`ImportError ... partially initialized module`, so
`Tests/Character_Chat/test_character_persona_scope_service.py` could not even
be collected when run solo — while passing in full-suite order, where an
earlier module always imported config first. The comment was reviewed and
merged; the claim was never probed.

**What to do.** A deferral claim is an assertion about *when code runs*, and
the evidence is a `sys.modules` probe in a fresh interpreter, not the comment
or the indentation. Before writing (or believing) "lazy import", trace the
enclosing function to its callers and ask whether any of them execute at
import time — `load_settings()`-style module-scope initialization defeats
every function-scope import above it. Encode the claim as a
subprocess-isolated closure guard (`Tests/Packaging/test_config_import_closure.py`
is the shape: import the module bare, assert the deferred packages absent,
plus an anti-vacuity check that the replacement seam is present). And when a
test module fails collection only when run solo, suspect an import cycle whose
direction depends on who imports first — full-suite green is not evidence the
import graph is acyclic.

---

## A green suite cannot certify a SQLite isolation-mode flip — only a per-idiom write census can

**TASK-22224, 2026-08-25.** Flipping held-connection stores to
`isolation_level = None` (autocommit) so the explicit-BEGIN transaction
managers stop being degraded by leaked implicit transactions. Empirical probes
on the shipped pair (Python 3.12.11 / SQLite 3.49.1) showed the change is
invisible to any test that only exercises success paths, while three idioms
silently change meaning: a multi-statement `with conn:` body loses atomicity
(the context manager only commits/rolls back, it never BEGINs — the statements
each self-commit, and an exception mid-body keeps the earlier writes);
`conn.commit()`/`rollback()` outside an explicit BEGIN become no-ops; and
`executemany` stops being all-or-nothing on a mid-batch error. Every one of
those degradations produces a suite that stays green — the data only diverges
on a failure injected *between* statements. Two stores (`Evals_DB`, with a
deliberately NESTED `with conn:` pair that explicit BEGIN cannot express, and
`sync_state_repository`, with an eight-DELETE commit span) were left on legacy
isolation with the exception documented at the opener, because converting them
is a write-path refactor, not a connection flag. A second trap found in the
same pass: ChaChaNotes' `vacuum()` "temporarily" set `isolation_level = None`
and *restored* `""` — after the store-level flip, that restore would have
silently returned the held connection to legacy mode for the rest of its life
(`Prompts_DB`/`Client_Media_DB_v2` still carry the same toggle and must fix it
when they flip).

**What to do.** Before flipping any store to autocommit, census every
`commit()`, `rollback()`, `with conn:` block, `executemany`, and
`executescript` in the store AND its out-of-module consumers, and classify
each span as single-statement (safe), manager-owned explicit BEGIN (safe), or
multi-statement implicit (NOT safe — convert or keep legacy and document).
Treat a green run as evidence of nothing here; the evidence is the census plus
guards that assert the transaction statements SQLite actually ran
(`set_trace_callback`), pinned red-first against both failure modes (loud
"cannot start a transaction within a transaction" and ChaChaNotes' silent
borrow, which surfaces only as an EMPTY trace).
---

## A cap silently turns a proportional control back into the flat constant it replaced — and formula tests stay green through it (TASK-22214, 2026-08-25)

TASK-21113 replaced a flat inter-route sleep in the screen pre-importer with a
proportional one, `min(previous_import_cost * ratio, cap)`, cap 0.10 s, and
pinned it with 18 tests. Two months of payload growth later every route the
pacing existed for was asking for a 150-600 ms gap and getting 0.10 s: the
control had degenerated back into exactly the flat sleep it was written to
replace, at ~66% GIL duty. **All 18 tests stayed green**, because they assert
the arithmetic (`min(cost*ratio, cap)` computes correctly) and the arithmetic
was never wrong. What had changed was the *regime*: which term of the `min()`
wins in production.

The evidence that settled it was not a duty number but the requested-gap
series itself, logged from the real thread: before `[0.0, 0.1, 0.1, 0.002,
0.003, 0.1]` — clipped flat on exactly the expensive routes, the cheap ones
untouched — after `[0.0, 0.529, 0.245, 0.003, 0.113, 0.303]`. One printed
list makes "the cap has eaten the mechanism" unarguable in a way an aggregate
percentage does not.

**What to do.** When you ship a bounded proportional control (a cap, a floor,
a clamp, a max-backoff, a LIMIT), the tests that prove the formula are not
tests that the bound is still in the intended regime. Add one assertion that
pins the *relationship* between the bound and the measured quantity it bounds
— here, `cap >= 1.0 s` documented as "above the heaviest measured single-route
cost", which reds if anyone restores a sub-second cap — and, where the
quantity can grow on its own (payload, row counts, file sizes), a budget guard
on that quantity so the growth itself lands in review. Log the per-step
requested values, not just the aggregate: a clipped series is visible at a
glance, a 66%-vs-48% duty average is not.

Related trap from the same task: **a mutation that adds an already-resident
import is not a payload regression and a good guard SHOULD stay green on it.**
Importing `Chunking.Chunk_Lib` into a screen module left the census unmoved
(an earlier route already had it) and looked at first like a surviving mutant;
the real-growth mutant (a genuinely new 40k-LOC module on that route) was
caught and named the route. Before recording a survivor as a gap, check that
the mutation actually changes the quantity under test.

## A `query_one("#id")` is not a DOM walk in Textual 8.2.8 — a `query("*")` is (TASK-22228, 2026-08-26)

Three findings in the same small-residue batch were written as "an uncached
DOM query on a hot path". Measured on the mounted Console (475 widgets), two
of them were noise and the third was 33x worse than its own brief said:

* `screen.query_one("#console-native-composer")` per mouse-up: **0.3 us**
  warm, 5.2 us cold. Textual 8.2.8's `DOMNode.query_one` takes an id-selector
  fast path — `walk_breadth_search_id` plus a per-node `_query_one_cache`
  keyed on `_nodes._updates` — so a settled DOM answers from a dict.
* the left rail's two id lookups + `max_scroll_y` per scroll frame: **2.7 us**
  warm, 10.4 us cold, against a frame that repaints the rail.
* `bounded.viewport.query("*")` over a **16-node** subtree: **74.1 us**, versus
  **2.2 us** for the identical `walk_children(Widget)`. `DOMQuery.nodes` builds
  its candidate list from exactly that walk and then runs the parsed selector
  through `match()` for every node — for the universal selector, a filter that
  admits everything, that round-trip IS the whole cost.

The trap has a second half: the "fix" the brief prescribed for the first item
— route it through the screen's memoized composer accessor — measured SLOWER
than what it replaced (0.7 us vs 0.3 us), because the memo revalidates by
building `ancestors_with_self` on every hit. Implementing the prescription
would have been a pessimization defended by a green test.

**What to do.** Before converting any "uncached query" finding, time the call
as it actually runs (warm AND with `node._query_one_cache.clear()` for the
cold arm). Attribute cost to the SELECTOR ENGINE, not to the walk: `query_one`
with a literal `#id` is cheap and cached; `query(...)` with any selector — most
of all `"*"` — pays `match()` per node and caches nothing. When the filter
admits every widget, `walk_children(Widget)` is the same list in the same
order for a thirtieth of the price, and an equivalence assertion against the
`query("*")` result is what makes swapping it safe.

## A queued-then-killed CI run renders as `fail`, and a conflicting PR gets no runs at all (2026-08-26)

Landing 29 tasks across 12 PRs in one day, two CI states cost real time before they were
understood, and both look identical to "my change is broken" from the outside.

**`gh pr checks` printed `fail` for runs that were never executed.** The required
`Derived artifacts reproduce from their sources` job showed `fail` with a 1h14m duration;
`gh run view <id> --json conclusion` said **`cancelled`** with zero steps completed — the run
sat in the queue behind another session's jobs until the concurrency group killed it. Reading
the check column alone would have sent someone hunting a content bug that did not exist. The
fix is a fresh push (`git commit --amend --no-edit && git push --force-with-lease`), which
creates a new run; `gh run rerun` loses the concurrency group and often does nothing.

**A PR with a merge conflict gets NO runs created at all.** PR #2081 sat with an empty check
list for over an hour and one apparent "no runs" diagnosis; `gh pr view --json
mergeStateStatus` said `DIRTY`. GitHub builds the merge ref before scheduling, so a conflict
means nothing is ever queued. **"No runs" is a conflict signal, not an outage signal** — check
`mergeStateStatus` before re-triggering anything. On this repo the conflict is almost always
`Docs/security/production-diagnostic-inventory.json` or an append-only `lessons-*.md`, both of
which churn several times a day.

**What to do.** Before diagnosing any red or missing gate: (1) `gh pr view --json
mergeStateStatus` — `DIRTY` means rebase, not retry; (2) `gh run view <id> --json
conclusion,jobs` — `cancelled` with no completed steps means contention, so re-push; only a
genuine `failure` is yours. A merge-on-green watcher that encodes exactly this
(pass → merge, cancelled → re-push, failure → stop) landed all 12 PRs without further
intervention. Auto-merge is disabled on this repository, so the watcher is not optional.

## The batch-PR assembly recipe, and the two ways it silently loses content (2026-08-26)

Six batch PRs (the #2070 precedent) carried 21 of the 29 burn-down tasks through a saturated
gate queue. The recipe that worked, unchanged, six times: merge each verified branch into a
fresh branch off dev; resolve `production-diagnostic-inventory.json` by taking **dev's** copy
and regenerating **once** at the end (after reading the rows); resolve append-only
`lessons-*.md` by taking dev's copy and re-appending only the batch's own entries.

Two failure modes bit, both silent:

1. **Extracting a lesson entry with a `+`-diff of the branch tip re-appends foreign content.**
   If the branch had itself merged dev, the extraction pulls dev's entries too — a batch ended
   up with another task's lesson duplicated twice. The check that catches it is a diff of the
   assembled file against `origin/dev` filtered to `^\+## ` headers: it must list exactly the
   batch's own entries and nothing else.
2. **`grep` here is aliased to ugrep**, whose regex dialect rejects `^+++` and `\b`. The
   extraction pipeline returned **zero lines with exit 0**, and the append silently wrote
   nothing. Use `command grep`, and assert the extraction's line count before using it.

Also: `for f in $CONF` does not word-split in zsh (a conflicted-file loop saw one glued token
and reported everything as unhandled) — iterate with `while read`. And a rebase-conflict loop
that exits on error leaves a **detached HEAD**; `git push` then reports "Everything
up-to-date" while pushing nothing. Check `git symbolic-ref -q HEAD` before trusting any push.

---

## A logical concurrency cap does not bound a long-lived process pool's resident memory

**TASK-22508, 2026-08-26.** A first fix limited EPUB admission to one job at
a time inside the existing three-process Library parse pool. The coordinator
test passed, yet the same three-EPUB queue-to-SQLite reproduction still peaked
at about 750 MiB. Per-process sampling exposed why: the pool assigned the
sequential books to different workers, and all three retained their parser
high-water heaps, about 410 MiB combined, after the batch. Isolating the batch
in one physical worker and retiring that generation reduced the measured peak
to about 496 MiB on the final dev-based branch and left no parse worker resident.

**What to do.** For memory-heavy multiprocessing work, measure the whole
process tree and record RSS by PID after each logical task; an admission count
only proves how many functions run at once. If workers retain high-water heaps,
bound the number of physical processes that can own that resource class and
retire or recycle those owners at a verified idle boundary. A green scheduler
fake cannot prove this property because it has neither OS worker selection nor
allocator retention.

---

## Stream-chunk sanitation needs an aggregate-owner regression (TASK-22507.1, 2026-08-26)

The Console provider gateway sanitized each streamed response chunk, and the
existing binary canaries were green, but every test supplied the data URI or
base64 value in one chunk. Final whole-branch review split those values across
multiple sub-4096-byte chunks. Each fragment was individually harmless, while
the later join reconstructed the raw binary encoding before immutable capture,
SQLite persistence, and Full export.

The production correction sanitizes once more at the final accumulated owner
without re-consuming the shared serialization budget. The regression that
proved it does not stop at the sanitizer helper: it sends split fragments
through the real gateway, store, SQLite query, and Full exporter and asserts
absolute exclusion in every decoded owner. For any streaming exclusion rule,
test both fragment-level handling and an adversarial split at the final
aggregate/storage owner; per-chunk green cannot prove a property of the joined
value.
## A probe that cannot distinguish the mechanism it tests from the one it replaces proves nothing — and `CharactersRAGDB` does not accept URI filenames (TASK-22301, 2026-08-26)

A reviewer asked why the citation-boundary tests used a file-backed SQLite
database instead of `":memory:"`. The real constraint is that `CharactersRAGDB`
opens a **thread-local** connection and TASK-22205 offloads durable DB calls to a
worker thread — with `":memory:"` that worker gets its own empty database, so
writes made there are invisible to assertions on the test thread. Measured
symptom: citation traces wrote successfully and `rag_citation_traces` still read
**0 rows**.

The obvious escape hatch is a shared-cache in-memory URI —
`"file:<name>?mode=memory&cache=shared"` — which is in-memory *and* visible
across threads. I probed it: constructed the DB, read a table from a second
thread, saw it succeed, and adopted it.

**The probe was worthless.** `CharactersRAGDB` never passes `uri=True` to
`sqlite3.connect`, so that string is not a URI at all — it is taken as a
**literal filename**. The probe's cross-thread read succeeded because the
"database" was an ordinary file on disk, which is exactly the mechanism the URI
was supposed to replace. A passing probe was indistinguishable from the failure
it was meant to rule out.

Cost: ~200 files were created in the repo root with `?` and `:` in their names,
`git add -A` committed 137 of them, and both `windows-latest` CI legs failed
because those characters are illegal in Windows filenames. The CI failure named
GGUF jobs and gave no hint that a citation test was responsible.

**Two rules, both earned here:**

1. **Design a probe so its success is only possible via the new mechanism.**
   Here that meant checking `sqlite3.connect`'s call site for `uri=True` FIRST —
   one grep, before any probe — or asserting that no file appeared on disk.
   "It worked" is not evidence when the old path would also make it work.
2. **Before `git add -A`, look at what it is about to stage.** `git status
   --porcelain | head` would have shown 137 files named `file:...?mode=...`
   immediately. A test that writes to the repo root is a bug regardless of
   filename; the illegal characters merely made it visible on another OS.

Corollary worth knowing on its own: **`:memory:` is unusable for any test whose
writes happen on a worker thread**, and this repo has no shared-cache workaround
available until `CharactersRAGDB` opts into `uri=True`. A temp-directory file
plus explicit cleanup is the only option — measured at 0 directories leaked when
an autouse fixture closes the connection and removes the directory.

---

## A bounded private-prefix probe must not buy its memory bound by failing open

**TASK-18932, 2026-08-27.** The local `<think>` splitter buffered leading
whitespace while deciding whether a response began with a reasoning block. To
bound that buffer, it switched permanently to visible-answer mode after 20
characters. The parser tests explicitly blessed that transition as a memory
guard. Final feature review then tried 21 spaces followed by
`<think>secret</think>answer`: the reasoning channel was empty and the entire
tagged value entered visible assistant content, which would also feed ordinary
history and human-readable exports. Every feature matrix had passed because
none crossed the parser's old whitespace cap before an opener.

**What to do.** For a stream prefix that may contain private data, a resource
limit must fail closed or keep a bounded decision state; it must never turn an
undecided prefix into public content merely because filler crossed a cap. Here,
dropping leading response whitespace while retaining only the possible opener
prefix removed the unbounded state without weakening start anchoring. Boundary
tests must sit immediately below and above the former cap, include a much larger
input, and partition the opener across chunks. A test that proves memory stays
small is incomplete until it also proves the private/public channel assignment.

## Simulate a candidate fix's cost with the framework's own primitives before building it

**TASK-23021, 2026-08-27.** The setup-modal snow burned 4% of a core at idle;
TASK-21134 had already made the tick cheap and the cost hid one layer down, in
`Screen._on_timer_update`. The filing named a plausible fix: rewrite the
full-viewport `Static` as a `render_line` widget that dirties only changed
cells. Instead of building it, a probe arm reproduced only its *compositor
footprint* — advance the flake physics and call `refresh(*Region(x,y,1,1)...)`
for the changed cells, without touching the content. Measured interleaved on
the real screen: **2.7–3.6% vs the shipped 3.6–4.3%** — the candidate was dead
before a line of it existed. Mechanism: Textual's
`Compositor.render_partial_update` crops to the *bounding box* of the dirty
regions, and `_get_renders(crop)` re-renders every widget whose clip overlaps
that crop — scattered one-cell dirties across a full-viewport widget re-render
the world exactly like a full-viewport dirty (124 widget renders x 44 rows both
ways). A second simulated arm (a single 3x3 dirty at the same 2.5 Hz) priced
the floor of ANY animation on that screen at ~0.55% — ~30 widgets stack under
every cell of an overlay — which is what justified retiring the animation
outright rather than optimising it.

**What to do.** When a proposed perf fix changes *what gets dirtied* rather
than *what gets computed*, the framework will usually let you dirty exactly
that shape from the existing widget (`Widget.refresh(*regions)`) — measure
that arm interleaved with shipped and the floor before building the rewrite.
And when the prior fix in the same spot "worked" while the cost moved down a
layer, instrument the layer you claim to fix (`_on_timer_update` time,
renders-per-update), not the layer you touched.
## A fidelity test that compares rstripped text is blind to every geometry bug

**TASK-22500, 2026-08-26.** The virtualized reader replaced a `Static` with a
hand-rolled `render_line`, and the safety net for that swap was a fidelity test
pinning the new widget's output against the `Static` it replaced. It compared
`strip.text.rstrip()` per row. It passed throughout — while three rendering
regressions shipped underneath it, all three found later by review:

- wide glyphs (`日本語`) emitted 43-46 cell rows into a 40 cell screen, because
  `Strip(segments, len(piece))` passes a CHARACTER count where a CELL count is
  required, and the short declaration made `adjust_cell_length` PAD rather than
  truncate;
- the last row of every long document was unreachable, because the container's
  `max-height: 18` is an OUTER bound whose two border rows the child cannot
  have, so the child overflowed by exactly the border while `ScrollView` still
  computed `max_scroll_y` from the height it thought it had;
- every full-width wrapped row was cut by 2 columns, because the wrap index was
  built at the width measured before the VERTICAL scrollbar existed, and no
  `Resize` fires when a scrollbar shrinks the content region — the widget's own
  `size` never changes.

`rstrip()` erases trailing-cell differences, and comparing `.text` never looks
at `cell_length`, at the parent's box, or at whether the document's tail can be
reached at all. The test could not have failed on any of them.

**What to do.** When a change is about GEOMETRY, assert geometry:
`strip.cell_length == Segment.get_line_length(strip._segments)` and that it
equals the widget's width; mount the widget inside the REAL container rule
(borders and padding included, not a bare harness) and assert the last line is
reachable after `scroll_end`; assert the indexed width equals the painted width
once layout settles. Each of these was written after the fact and each reds on
its bug — verified by reintroducing all three. A test harness that yields the
widget straight into the App gives it the screen's whole box, which is exactly
the geometry the bug does not live in.

## A clean rebase can orphan an import, and only a full-suite A/B sees it

**TASK-22500, 2026-08-27.** The branch stopped importing `VerticalScroll` in
`library_screen.py` — legitimately, since its reader body became a `Container`
and the scroller lookups moved to `body.scroller`. While it was in review, dev
added `_capture_library_emergency_restore_receipt`, which uses `VerticalScroll`
in three places. Neither side touched the other's lines, so `git rebase`
reported no conflict and produced a tree that raises `NameError` from
`on_resize` on any Library route with emergency geometry.

**Cost: 79 failing shell tests, not one of them a reader test.** Every gate the
branch had been running stayed green — the affected-surface suite (89 passed),
`preflight.sh`, and the reader tests — because none of them mount the
notes/emergency routes that reach that line. The earlier "no new failures"
result was true, but it was measured against the OLD merge base and stopped
being evidence the moment dev moved.

**What to do.** After rebasing onto a base that has moved, a targeted suite is
not evidence. Run the full suite of the area you touched on BOTH your branch
and a pristine worktree at the new merge base, and diff the FAILURE NAME SETS —
counts alone lie, because this repo's shell suite carries ~97 pre-existing
failures and `pytest-randomly` reshuffles them per run (two runs of the same
tree gave 174 and 175). Use `-p no:randomly` on both sides or the sets are not
comparable. Sampling is not enough either: five sampled failures all reproduced
on base and pointed at "dev's problem", while the set diff exposed 79 that were
ours. There is no undefined-name linter in this repo's CI to catch this class.

## A PR that ran zero shards also reports zero failures — never baseline one PR's failure COUNT against another's (TASK-23029, 2026-08-28)

**The incident.** PR #2166 showed nine failing shards (3 Core, 6 UI). To decide
whether they were mine, I compared against PR #2160, open at the same time,
which reported **zero** failures — which reads as "dev is fine, your branch
broke it". #2160 had no `UI Tests` check runs *at all*: its `Tests` workflow
never started its shards, the same burst-cancellation problem tracked in
TASK-22250. Zero failures out of zero tests. Had I trusted it I would have
gone hunting for a defect in a diff that could not contain one.

**What actually settled it, in this order.** (1) A *structural* argument first,
because it is free: `git diff <base>...HEAD --name-only` filtered to everything
outside `Tests/Performance/`, `backlog/`, and one script came back **empty**, so
the diff cannot reach `Tests/UI`. (2) Then reproduce: the named failures were run
locally on the branch and failed there too, which — given (1) — means they fail on
the base. (3) Only then read the failure text, which named the real cause anyway
(`'ChatScreen' object has no attribute '_library_activity'` is a rename landing
without its tests, not a perf change).

**The rule.** Before using another PR's checks as a baseline, confirm it *ran the
same checks*: `gh pr view <n> --json statusCheckRollup` and group by
`status+conclusion` — `QUEUED`, `IN_PROGRESS`, and absent are all "no data", and
only `COMPLETED` rows carry a verdict. A comparison between "0 failures of 0
shards" and "9 failures of 12 shards" is not a comparison.

## An auto-lander that polls by PR number will attribute the OLD head's verdict to a head you just pushed (2026-08-28)

**The incident.** After force-pushing a rebased #2160, a background lander polling
`gh pr view 2160` printed `GREEN -> merge` **eight seconds later** — far too fast
for CI on the new commit. The verdict belonged to the pre-rebase head that had
been green before the push. GitHub's per-head required check refused the merge, so
the damage was zero and the PR simply stayed open; the lander's own log was the
only thing that lied.

**The rule.** A lander's "merged" line is a claim, not a result. Confirm with
`gh pr view <n> --json state,mergeCommit,headRefOid` that `state == MERGED`,
`mergeCommit` is non-null, **and** `headRefOid` equals the SHA you pushed. The
same check catches the related case where a lander merges a head that is no longer
the one you verified locally.

**TASK-23019 follow-up, 2026-08-28.** The post-rebase production-shaped suite
found a different clean-merge failure: importing `Skills_Interop` reached the
foundational `Utils.input_validation` module, whose eager import of a Console
title helper executed `Chat.__init__`, cycled through Library, and tried to
re-import the partially initialized validation module. The exact Skill test
failed on pristine dev too, proving this was a moved-base baseline defect rather
than reader code. A lazy module-level proxy broke the cycle while preserving the
existing monkeypatch seam. When dev advanced again, it independently contained
the same proxy; the clean rebase preserved both definitions and only the final
Ruff gate exposed the duplicate as F811. Removing the redundant branch copy kept
the upstream fix and its test seam. This incident reinforces the same gate: run
the production-shaped suite after rebasing, A/B every surprising failure on the
new base, and lint the rebased tree because clean textual merges can still create
duplicate semantic definitions.

---

## A PASS result is not evidence until focus and mounted identity have settled (TASK-23019, 2026-08-28)

The adaptive-reader closeout initially produced PASS capability results, but the retained evidence
oracle rejected Conversations and Notes because their captured `focus_owner` was null. A later
Prompts run exposed the same timing class differently: Discard cleared the dirty flag synchronously,
the scenario treated that state change as completion, and an immediate row lookup raised
`StopIteration` while asynchronous browse recomposition had not yet remounted the target. The final
detached run found a third form: Skills Save satisfied its state predicate while the captured More
actions button was the old hidden instance, so waiting on that stale object could never observe the
replacement control becoming visible.

The reliable boundary was stronger than waiting for a state flag. Before capture, the scenario now
focuses a real visible Work target and waits until the screen owns that focus. Before an
identity-specific action, it waits for and captures the matching row only when the row is mounted,
displayed, and has painted area; the successful predicate returns that exact row without a second
query. Visible controls use the same rule: reacquire the current mounted, displayed, painted button
inside the successful wait predicate, then focus and press that exact instance. Parent diagnostics
also include the bounded, sanitized live-root name, so a generic scenario module target cannot hide
which journey failed.

**What to do.** Treat PASS as the start of evidence validation, not its conclusion. Settle and
record the user-visible focus and identity owners that the oracle requires, and after any action
that can recompose, await the mounted and visible target rather than only the first synchronous
state change. Include the scenario/root identity in bounded failure details so intermittent live
failures remain attributable.

## A verifier must not invalidate the evidence it is verifying (TASK-23019, 2026-08-28)

The retained closeout verifier passed once but imported its adjacent task-local sources into a
writable evidence directory. Python created `__pycache__`; the next exact invocation then correctly
rejected that unmanifested path as `artifact_path_not_allowed`. The verification logic was sound,
but its own runtime side effect made the documented command non-repeatable.

**What to do.** Treat evidence verification as a read-only operation all the way down to language
runtime behavior. Suppress bytecode in the documented command and inside the task-local runner,
then run the exact verifier twice and assert that neither cache directories nor unmanifested files
appear. A single PASS does not prove an evidence verifier is idempotent.

---

## An import-parent tracer names the FIRST importer, so its attribution is an upper bound — re-measure after every deferral

**TASK-23112, 2026-08-28.** ADR-097's standing import-weight breach (666 own
modules against the 660 ratchet) came with a traced list of the 17 added edges,
produced by the house `sys.meta_path` import-parent recorder. Four edges were
named as the debt. Two of them, deferred, would have bought **exactly zero
modules**, and the trace could not have told you:

* `Chat.console_runtime -> Chat.thinking_blocks` was real — and irrelevant.
  `Chat/Chat_Functions.py:98` also imports `thinking_blocks` at module scope,
  and `Chat_Functions` is on the boot path. Whoever imports first gets the
  attribution; the other importer is invisible.
* `chat_persistence_service -> Chat.library_activity` (with `Chat.trajectory`
  and `Utils.log_sanitizer`) was likewise real and likewise irrelevant:
  `Agents/library_tool_provider.py` imports it too, reached through
  `app -> UI.Tools_Settings_Window -> Agents.local_tool_provider ->
  Agents.tool_catalog -> Agents.library_rag_tool_provider`.

The recorder stores one parent per module — the module whose body triggered the
*first* import — so the parent map is a tree over a graph. A subtree's size is
therefore the **maximum** a deferral can buy, never the actual number. Measured:
the `chat_persistence_service` subtree was 31 modules and deferring its single
boot-path consumer removed 18; the `console_raw_cli` subtree was 8 and deferring
its executor import removed 2 (`Tools`, `Tools.tool_executor`, `Agents`,
`Agents.run_log_format` stayed, pulled by the `Tools_Settings_Window` chain).

**What to do.** Treat a traced chain as *where to look*, never as *what it is
worth*. Re-run the count in a fresh interpreter after each individual deferral
and diff the module sets; that is the only statement about savings you can
defend. And when a deferral buys zero, say so in the notes with the second
importer named — otherwise the next person re-files the same edge. (Companion
to the existing rule that a tracer beats reading `-X importtime` or the diff:
the tracer is still the right instrument, it just answers "who imported this
first", not "who needs it".)

---

## To prove a failure is pre-existing, use a detached worktree at the base — not a file swap in your own tree

**TASK-23112, 2026-08-28.** Establishing that four red tests belonged to dev and
not to the change under review meant running them against pristine sources. The
obvious method — copy `git show HEAD:<file>` over the working copy, run, copy the
saved version back — failed twice in one session, in two different ways:

1. **The run outlived its shell.** A 10-minute-capped command was killed by the
   harness; the `trap ... EXIT` restore never ran, and the working tree was left
   holding `HEAD` content with the implementation silently gone. Nothing failed
   loudly — the next command just measured the wrong tree.
2. **The saved copy went stale.** Two more edits landed after the "mine" snapshot
   was taken; copying it back reverted them. `git diff --stat` was the only thing
   that noticed (83 insertions became 80), and only because the number had been
   read before.

`git stash` is not the alternative here — it is repo-wide across 100+ worktrees
and banned. The safe form is `git worktree add --detach <path> <base-sha>`, run
the tests there, `git worktree remove --force`. Your own tree is never touched,
the comparison is against the exact base commit rather than "HEAD content of the
two files I happened to think about", and a killed run leaves nothing to restore.
It settled the two `Tests/ProductionApp/` failures with byte-identical assertion
messages on both sides.

**What to do.** Never mutate the tree you are trying to evaluate. If you do it
anyway, `git diff --stat` before AND after the swap and compare the numbers —
that is the only cheap detector for both failure modes above.

**Extension — TASK-25715, 2026-08-31: pick the base commit, not `dev`.** The
worktree method above is right, but "the base" is the commit your work branched
from, not wherever `dev` is now. Two Console rail tests were failing on my
branch; I ran them in a pristine `origin/dev` worktree, saw them fail there too,
and wrote "pre-existing on dev, not from this branch" into a PR body. The
measurement was real and the conclusion did not follow: `dev` had by then
absorbed my own two earlier PRs, so red-on-dev could no longer separate someone
else's regression from mine. Re-run at the pre-branch commit, both tests
**passed** — meaning I had shipped a claim of innocence I had not tested.

They did turn out to be someone else's, but only a first-parent binary search
over the 60 merges in between could say so: the header-padding failure came
from #2220 changing the *Inspector* side of a band the test pins to the Context
side, and the reveal-queue failure from #2252 resolving `#console-left-rail-body`
unguarded. The same search is what actively cleared my own merges — which is the
part "it fails on dev too" can never do.

**What to do.** Once any of your work has merged to the base branch, `dev` is no
longer a neutral witness about your work. Verify at the commit you branched from,
and if that is green, bisect — a binary search over first-parent merges costs
~6 test runs and names a commit, where "pre-existing" names nobody and quietly
includes you.

## A focusable widget can be painting nothing (TASK-23100, 2026-08-28)

A UX critique drove the real Schedules create form and found that choosing "Recurring"
rendered the Frequency select, three blank rows, then Timezone. The cron input, its
syntax helper, and the live "Runs: ..." preview were not there -- but Tab still landed
on the cron input and it still accepted keystrokes, silently flipping the preset to
"Custom cron...". Typing went into a widget the user could not see, and the form's best
safety feature (the plain-English preview of what the schedule would do) was dead code
at ordinary terminal sizes.

Two mechanisms combined, and both are easy to reproduce elsewhere:

- The field container had `max-height` with no scrolling, and Textual's auto-height
  container **clamps by clipping**, so overflow is neither scrollable nor visible.
- The field groups were plain `Vertical`s. Their default `height: 1fr` measured about
  **one row** inside the scroll's virtual size while painting six, so the scroll region
  believed the content fit. Measured during the fix: virtual height 17 against a painted
  22.

A first fix computed a height budget in Python (`overhead = 10 + error_line_count`). It
worked at the sizes it was tested at and failed at ~45x24, because a wrapped error line
occupies two terminal rows while the counter counted one -- re-introducing the same
class of bug with arithmetic instead of layout. The durable fix was structural: a
docked-bottom footer plus a `1fr` scroll area, the pattern `voice_blend_dialog.py` and
`feedback_dialog.py` already use, which deletes the arithmetic and the resize hook
entirely.

**What to do.** Never accept a style-value probe as evidence that a widget is visible --
`styles.height` and a green `query_one` both pass for a zero-region widget inside a shut
`Collapsible` or a clipped container. Assert **paint** via the compositor
(`get_widget_at` / `export_screenshot` / a tmux `capture-pane`), at the narrow floor as
well as a comfortable size; 80x24 and a ~45-column case catch what 235x52 hides. When a
form can scroll, assert that focusing a field brings it into view, because focus and
visibility are independent in Textual. The same review found the mirror image in the
Settings search landing: `.focus()` is a silent no-op on a disabled widget and happily
focuses a field inside a collapsed disclosure, so a "landing" can succeed in code while
the user sees nothing move.

## A per-frame gate's signature must carry the DECISION, not the raw input

**TASK-23151, 2026-08-28.** `library_screen.py` now has three cheap-state gates
that skip per-frame work when a signature tuple is unchanged. Building the third
one, two of its slots were wrong in ways that reasoning could not see and only
`assert n == 0` could — the ratchet went 201/100 applications before the fix,
then 1/100, then 1/1, then 0/0, one wrong slot at a time.

- **Raw bucket instead of the effective decision.** The slot held
  `ordinary_emergency_required(width)` — true below 64 cells. But
  `_apply_library_emergency_geometry` acts only when that AND
  `_library_ordinary_route_active()` hold, and on the Notes route it never
  holds. So the bucket flipped on every 60-cell frame of a route where the
  geometry is inert, and the gate applied for nothing. Carrying the conjunction
  the code actually branches on fixed it.
- **A value another subsystem owns.** `rail.display` was carried "to catch an
  outside mutation". Under an adaptive reader shell, the reader's own
  `sync_layout` owns the rail and hides it purely as a function of width — so
  the slot changed on every same-side resize, holding the wide case at 100
  applications. The leg returns before its own rail toggles in that branch, i.e.
  it never reads the value there. Carrying it only while the legacy path owns it
  fixed it.

Neither was found by reading the code. Both were found by printing the index of
the differing tuple slot on each skipped-then-unskipped frame — a ten-line probe
that names the slot, versus a plausible story about why the gate "should" work.

A third shape mattered too: recording the applied signature inside the GATE left
exactly one application per burst, because any other seam's call left the record
cleared. Recording it inside the applied function itself — cleared first, then
re-taken after the legs, since they mutate flags the signature reads — arms the
gate from all ~20 seams and gets the true zero.

**What to do.** When a gate skips work, do not accept "the signature looks
right". Instrument which slot differs on the frames that still run, and require
the ratchet's own number to reach the asserted value; and for each slot ask two
questions — does the guarded code branch on this raw value or on a conjunction,
and does anything outside the guarded code write it?

## `ruff format` on a whole file here reformats code you did not touch

**TASK-23151, 2026-08-28.** After adding ~110 lines to `library_screen.py`,
`ruff format --check` flagged the file, so the fix was formatted. It rewrote
**20+ unrelated regions** across the 38k-line file (diffstat went from 125/2 to
209/67), burying a five-hunk change in churn and forcing a full restore from
`HEAD` plus a manual re-apply of every edit. The pre-check that made it look
safe — piping the `HEAD` blob through `ruff format --check --stdin-filename` —
printed nothing and was read as "clean"; it is not a reliable clean signal.

**What to do.** This repo is not `ruff format`-clean under the installed ruff,
and no CI job enforces it. Never run the formatter over a whole file to tidy
your own addition — hand-wrap the lines you added instead, and check your diff
hunk list (`git diff -U0 | grep '^@@'`) before committing: every hunk should be
one you can name.

## A guard that names the thing it looks for is blind to the thing added next to it

**TASK-23144, 2026-08-28.** TASK-21381 repaired 115 bare-`ChatScreen` test
shells that died during *setup* because `screen._console_chat_store = store`
reaches, several frames down, a read of `self._fleet`. It shipped a ratchet so
that could not recur: an AST scan of `Tests/` for any function that builds a
shell with `ChatScreen.__new__`, assigns the store, and does not call
`stub_fleet_controller`.

Seven weeks later PR #2154 added a second controller read to the SAME hook
build — `self._library_activity.build_provider`, three lines above the fleet
read. 46 tests died in setup, in exactly the shape the ratchet existed to
prevent, and **the ratchet stayed green throughout**: it was asking "is
`stub_fleet_controller` called?", a question whose answer was still yes. The
name it was matching on was never the invariant; it was the invariant's value
at the moment the guard was written.

**What to do.** When a guard exists to stop "code path X needs setup Y", derive
Y by *performing X*, not by pattern-matching the source of X. Here that is nine
lines: assign the store on a fresh bare shell, catch the `AttributeError`,
record the attribute it names, install a stand-in, repeat until the assignment
succeeds. The derived set is `('_library_activity', '_fleet')` today and will be
whatever production makes it tomorrow — no function name, no attribute spelling
and no call shape is written down anywhere. Where something genuinely must stay
hand-written (the slot -> stub-helper mapping: only a person knows which helper
builds which controller), hold it to SET-EQUALITY with the derived set in both
directions, per `Tests/Architecture/test_framework_armed_clock_inventory.py` —
so an unmapped controller and a stale mapping both fail loudly.

Two things this cost, worth knowing before writing the same probe:

- **A cached handle turns the probe into a liar.** Re-assigning the store on the
  *same* shell reported success after one missing name: `_console_runtime()`
  memoizes the runtime as `_console_runtime_ref`, and the attach that does the
  reading only re-runs when the view changed. The probe must build a FRESH shell
  each round. It reported `['_library_activity']` — a plausible, wrong answer
  that would have shipped a guard still blind to `_fleet`.
- **Verify a widened guard by shrinking it, in both halves.** Removing one
  fixture's stub reds the ratchet naming the function and the helper it lacks;
  removing one row from the mapping reds the derivation naming the controller
  and where it is built. A guard nobody has watched fail is not evidence.

## A server contract is the API schema, adapter, and materializer together (TASK-24307, 2026-08-30)

**What happened.** The first Notes-organization two-device harness used a fake
transport that accepted an envelope as long as the domain adapter accepted it.
That made the causal retry path green with an optional cursor. Rechecking the
current server exposed a split contract: the adapter's comparison path tolerated
the optional cursor, while the API envelope and materializer required a stored
server cursor and an exact complete base before the same change could be applied.
The fake had modeled one permissive layer and therefore accepted a request the
real write pipeline would reject. The client was corrected to retain pending
successors until the predecessor supplies the complete cursor/revision/hash base,
and the test transport was tightened to exercise that rule.

**What to do.** For a server-backed mutation, do not derive the client contract
from the domain adapter alone. Read and pin all three executable boundaries: the
request/API schema, the domain adapter, and the materializer that commits state.
A fake transport must enforce their intersection, especially required cursors,
optimistic-base triples, restore metadata, and terminal acknowledgement states.
Before calling the fake evidence, send at least one deliberately incomplete
envelope and prove it fails for the same reason as the real pipeline.

**TASK-24309 addendum (2026-08-30).** The real two-device enrollment gate found
a second intersection the fake transport did not model: server idempotency is
dataset-wide on `client_envelope_id`, not private to one simulated device. Device
B pulled Device A's seeded `Agent_Lessons` folder correctly, then legacy inventory
mistook that materialized remote head for pre-enrollment local state and emitted
it again. The deterministic intent ID was identical, but its device-specific base
metadata differed, so the real server rejected it as `idempotency_conflict` while
the fake stayed green. The fix makes inventory skip a local row only when its
domain, object ID, operation, and canonical payload exactly match an already
applied remote head; a changed local row still publishes normally.

When a deterministic envelope ID can be produced on more than one device, make
the test transport index idempotency at the server's real ownership scope and
vary device identity plus base metadata in at least one convergence test. Also
assert that bootstrap inventory distinguishes pulled materializations from local
legacy state; checking final rows alone will miss a false retained outbox and a
profile-level partial failure.
---

## Process-local absence is not durable orphan evidence (TASK-22863, 2026-08-27)

A single-coordinator recovery test passed after it treated a missing in-memory
task as an orphaned durable receipt. Independent review then opened two SQLite
connections and two coordinators over one real database: while the winner was
still performing a briefing or source check, the loser saw no local task and
incorrectly changed the winner's active row to failed. An earlier version of
the briefing test even codified a second provider call as the expected recovery
behavior.

The repaired contract uses the coordinator's captured startup boundary as the
only evidence that an active receipt predates the current process. Normal
duplicate submission follows an ownerless incumbent without adopting,
terminalizing, or replaying it. For durable claim ownership, test with distinct
database connections and coordinators, capture the loser's startup boundary
before the winner creates its row, block the winner after one observable side
effect, and prove the loser cannot mutate the row or repeat the effect.

---

## A boolean config-save result cannot describe a partial commit (TASK-22864, 2026-08-28)

The new briefing-schedule gate initially used the compatibility helper that
returns only `True` or `False`. Independent review forced cache publication to
fail after the real atomic file replacement. The helper returned `False`, so
Settings said nothing changed and left the live scheduler disabled even though
the TOML already contained the enabled gate. A follow-up interaction then found
the inverse problem: a live-apply failure left an enabled “Enable” action whose
next press derived from persisted state and disabled the saved gate.

For user-facing configuration, the atomic replacement is a commit boundary.
Consume the structured mutation result and distinguish before-replace failure,
file-replaced-but-not-published, and fully applied outcomes. Test each branch
against a real temporary config file, the live runtime owner, rendered recovery
copy, and the next user action. When durable and live state diverge, either base
the action on one authoritative live state or lock it behind the stated restart
recovery; never infer “unchanged” from a lossy boolean wrapper.

---

## Safe exception copy must be reconstructed at every accepting boundary (TASK-22865, 2026-08-28)

The first Watchlists failure-classification implementation had fixed presentation
copy and a broad canary suite, but independent review found two ways around it. The
classifier granted semantics to unrelated exceptions solely because their class names
matched `AuthenticationError`, `RateLimitError`, or `FetchBlockedError`; a caller
could also construct the public `WatchlistFailure` dataclass with a valid category and
forged message/action fields, which the recorder persisted to `last_error`. A separate
scheduled fallback then leaked both the original and fallback exception chains through
the scheduler's final `logger.exception` owner.

For persisted or user-facing exception handling, concrete owned types and validated
machine fields are the authority; exception class names and preassembled presentation
objects are not. Reconstruct fixed copy from the validated category at each accepting
boundary, and test spoof types, forged structured objects, original-plus-fallback
failure chains, and the final logging owner. A green classifier helper does not prove
redaction if callers or outer fallback owners can bypass it.

---

## A single-flight choice handoff needs its own synchronous claim (TASK-22867, 2026-08-28)

The first multi-skill import implementation correctly kept the original inspection
single-flight while its choice modal was open, but treated the selected import as a
fresh ordinary coroutine. A repeated-cancellation probe then showed that cancelling
the app-worker wrapper could detach that second phase, and a stale-modal probe showed
that an older callback could target a newer retained package if it read only the
current coordinator state.

The repaired coordinator admits the exact displayed candidate synchronously, advances
the operation generation, and runs the retained-package import through the same
cancellation-resistant terminal owner as initial inspection. Modal callbacks carry
the generation they displayed, and retained bytes are cleared on cancel and every
terminal path. When one logical operation pauses for human choice, test the handoff as
a second concurrency boundary: double selection, stale callbacks, repeated outer
cancellation, routed replacement, and exact retained input all need proof even when
the pre-choice phase is already single-flight.

---

## A per-switch average over a PAIR of destinations invents a finding

**What happened.** 2026-08-29 holistic perf review. A probe measured
`get_cli_setting` calls across a Library-then-Console switch pair and divided by
two, reporting "33.5 config reads per screen switch, 19.5 of them from
`library_screen._load_library_ingest_options_from_config`". That became a filed
finding titled *Library ingest options are re-read from config on unrelated
screen switches* — a cross-screen leak, which would have been a layering bug.

It was false. Re-measured per destination, Console switches read 18–21 settings
and **not one** came from `library_screen`; all 43 Library reads belong to
Library's own `on_mount`. The average had smeared one screen's mount cost evenly
across both switches, and the smear is what created the "unrelated" framing.

The waste was real — 43 reads on *every* Library visit, because the app builds a
new `LibraryScreen` per visit — but the mechanism, the owner and the fix were all
different from what was filed. Implementing against the filed premise would have
gone looking for a cross-screen call that does not exist.

**What to do.** Never average a per-event cost across a sequence containing more
than one kind of event. Attribute to the specific destination, action or keystroke
that caused it, and only then divide. The cheap version is one measurement window
per destination — it is the same probe, run three times instead of once.

Sibling of the same cycle's other misattribution: a "typing" phase that contained
a screen switch reported a 457 ms typing stall; isolating the switch out of the
measured window left a single 113 ms stall. **A measurement window that contains
two activities cannot attribute cost to either.**

---

## `run_worker(partial(...))` silently makes the worker anonymous

**What happened.** 2026-08-29. The boot worker census failed on
`('', 'console-persisted-browser-cache')` — an unlisted worker with an EMPTY
name. The allowlist already carried the correct row,
`('_refresh_console_persisted_rows_cache', 'console-persisted-browser-cache')`.

Textual derives a worker's name as
`name or getattr(work, "__name__", "") or ""` (`worker_manager.py:112`). A
`functools.partial` has no `__name__`, so wrapping an existing `run_worker` call
in a partial — to bind kwargs — renamed the worker to `""`. The guard read as
"a new unreviewed worker appeared during boot"; the truth was "an existing,
already-approved worker lost its identity". It was also anonymous in every worker
diagnostic from that moment on, which nothing else surfaced.

**What to do.** Pass `name=` explicitly whenever `run_worker` is given a
`partial`, a lambda, or any other callable without `__name__`. 56 further
`run_worker(partial(...))` sites exist in this repo with the same latent
anonymity; none are boot workers, so only this one had a guard watching.

---

## A module-global cache passes in isolation and fails in the suite

**What happened.** 2026-08-29, task-24456. Memoising Library's 43 per-visit
config reads in a module-level `_CACHE` variable passed
`Tests/UI/test_library_screen.py::test_load_ingest_options_from_config` when run
alone and failed it inside the full suite. The cache outlived the app and served
one test's stubbed `get_cli_setting` values to the next test.

The reflex fix — add a cache-clearing fixture — would have been tests bending
around an implementation. The real defect was the cache's LIFETIME: it was
process-scoped for data whose validity is app-scoped.

Re-scoping it to the running `App` object fixed the test *and* was the better
production design: a screen with no running app (an unmounted screen built
directly by a test, or any caller that has swapped the settings source) now reads
fresh, which is what such a caller must see.

**What to do.** When a cache exists to amortise work across *navigations*, scope
it to the object that spans navigations — the app — not to the module. If a
memoisation needs a test fixture to clear it, that is evidence its lifetime is
wrong, not evidence the test needs help.

---

## Two ratchets can measure the same cost, so paying one breaches the other

**What happened.** 2026-08-29, task-24458. Deferring modules off the boot import path made
`test_app_import_weight` and `test_ui_ready_module_census` go green (662→631, 981→966). After
a rebase, `test_screen_preimport_payload_budget` went RED at 505/500 modules — and it PASSES on
pristine dev, so it looked like the branch had introduced 5 modules of new cost.

It had introduced none. That guard measures `tldw_modules()` after the pre-import pass **minus a
baseline taken before it**. Every module removed from boot lowers the baseline and raises
`pass_added` by exactly one. The breach message's module-set diff proved it: the 14 "new" modules
were precisely the 14 the deferrals had moved. Total modules loaded was unchanged.

The two guards are in tension by construction: paying down the boot budgets necessarily spends
the pre-import budget, LOC for LOC. Dev had 226 LOC of pre-import headroom and the shift moved
842 LOC of accounting, so it breached with nothing having grown.

**What to do.** When a boot-import deferral lands, expect the pre-import payload guard to move
against you by the same amount, and budget for it in the same PR. Do not raise either constant
(ADR-097) and do not revert the deferral — find real payload to shed. The honest place to look is
the deferral's own edges: here, following them turned up `UI/Widgets/__init__.py` eagerly
importing `SmartContentTree` (425 LOC) and `config_search_widget` (228 LOC), so the four MCP
modes that want only the small `table_click_select` mixin were each paying for both. That is the
**`Chunking/__init__.py` eager-package-`__init__` trap (finding 21102) recurring in a second
package**, and it is worth grepping for whenever a pre-import budget is tight.

**Corollary, and the reason this entry exists rather than a one-line note:** a guard that
subtracts a baseline is measuring a *difference*, not a cost. Before treating such a breach as a
regression, check whether the baseline moved. `git worktree` at pristine dev plus one test run
settles it in two minutes, and it is the difference between "my change is slower" and "my change
is faster and the meter moved."

---

## Reactive state can settle before its replacement control is mounted

**TASK-18917, 2026-08-29.** The Notes live closeout passed repeatedly alone but
failed in the 697-test aggregate while collapsing a folder, retaining pager
focus, and reading freshly loaded repository rows. Each wait correctly observed
the branch generation/loading state, then immediately queried or pressed a
control. Under aggregate contention, the state update won the race while the
canvas recompose was still replacing that control; the test acted on the old
child or observed the pre-recompose tree. Two full aggregate reruns were lost to
different manifestations of the same ordering gap.

**What to do.** In mounted Textual evidence, a settled reactive/service state is
not proof that its replacement widget is current. When the next assertion or
action depends on a recomposed child, also wait for that identifying mounted
control, yield one compositor cycle, then re-query it immediately before use.
Do not keep and press a control captured before a branch refresh.

---

## Callback turns are not layout evidence after same-size recomposition

**TASK-18918, 2026-08-30.** Media Back retained the correct semantic row and
requested scroll `(0, 42)`, but a same-size cross-reader recompose could keep the
previous Notes presentation long enough for the replacement Media list to clamp
at `(0, 33)`. Four remedies based on adding callback turns sometimes passed and
sometimes failed in fresh processes because none proved that the current scroll
owner had received its final geometry.

The gate became deterministic only when `LibraryMediaRowScroll`, the producer
that owns the relevant geometry, emitted Resize-derived evidence for its current
owner and presentation epoch. Five fresh-process exact-return runs and the
four-size live walkthrough then passed. When correctness depends on layout after
recomposition, fixed callback counts or sleeps are not readiness signals; wait
for a public event from the current geometry owner and fence it by identity and
epoch.

---

## A required check that exempts admins or accepts a stale base is advisory

**Incident.** TASK-25705, 2026-08-30. PR #2228 merged into `dev` before its
required derived-artifact workflow started. The workflow later detected that
new persistent diagnostics had not regenerated the canonical inventory, but
the merge had already landed because branch protection used both
`enforce_admins=false` and non-strict status checks. The result was a red
architecture checker and two red dependent summarization-privacy tests on
`dev`, even though the correct required context already existed.

**What to do.** A repository-wide generated-artifact check must apply to every
merger and to the current base revision. Enforce required checks for
administrators, require the latest base, and verify the live protection API
after changing it. The workflow's eventual failure proves the checker works;
it does not prove the branch was protected at merge time.

---

## A strict gate slower than the base branch's merge cadence punishes eager rebasing

**Incident.** PR #2260, 2026-08-31. The follow-on cost of the fix above. With
`strict: true`, any movement on `dev` makes an open PR out-of-date, so my merge
loop rebased and force-pushed the moment it saw itself behind. It never landed.

Measured afterwards: `Derived artifacts reproduce from their sources` is
`needs: [pr-fast-lane]`, and that lane runs a pytest suite — **~20 minutes**
end-to-end including shared-pool queue. `dev` was merging every **~13 minutes**
that day. Each rebase restarted a 20-minute clock that a 13-minute cadence was
already beating, so the gate never reported for the commit that was still HEAD.
It had in fact **passed three times** — for `6ab0cf5e`, `8493ddbf` and
`92e325ec` — each time finishing after I had already replaced the commit it was
green for. One run went green six minutes *after* I force-pushed past it.

I spent an hour reporting this as CI queue congestion, citing an existing
backlog task about exactly that, without once opening
`.github/workflows/derived-artifacts.yml`. The workflow's own header comment
documents the same pathology from the cancellation side (TASK-21250: 45
cancelled / 14 success over 60 runs, a 23% success rate for a required check).

**What to do.** Poll and merge the *instant* the gate is green; never rebase
pre-emptively. Move HEAD only when GitHub itself reports `BEHIND`, and prefer
`gh pr update-branch` over a rebase + force-push so the commits stay reviewable.
Check `gh pr merge --auto` first — it removes your reaction time from the race
entirely — but confirm it is enabled: `enablePullRequestAutoMerge` is off for
this repository, which is why the polling loop is necessary here.

**Before blaming shared CI, read the workflow file.** `gh run view <id> --json
jobs` shows whether the job ran and what it was waiting on, and it is the
difference between "the queue is slow" and "I keep invalidating my own green
check".

---

## A mechanism assembled from verified links is still a hypothesis

**Incident.** TASK-26835, 2026-09-01. Chasing the Console's paint stalls, I
read five Textual internals in source — `call_after_refresh`, the Screen's
idle handler, `Timer._run`, `App._end_batch`, `App._on_idle` — verified each
individually, composed them into a "screen frozen until the next input event"
mechanism, and filed a HIGH task describing it as "verified link by link".
The reproduction test refuted it the same hour: a bare app recomposing under
batch paints promptly, because two relays my derivation had missed
(`Widget.refresh` → widget Idle → `_check_refresh` → `Update`/`Layout` posted
to the screen) close the loop. A second composed theory — deferred callbacks
waiting on an ambient tick — died the same way earlier that day.

What settled it was neither derivation: instrumenting the LIVE app to sample
the actual invariants (`batch_count`, timer paused/running, dirty count, which
idle branch ran) every 10ms inside stall windows. The real mechanism was
nested App-level batches held open 250-400ms by tray recompose cascades —
observable directly as `screen.Idle guarded(batch=3)` and
`STATE batch=2 dirty=71`, no assembly required.

**What to do.** Source-reading tells you what CAN happen; only a failing
reproduction or live-state sampling tells you what DOES. Write the failing
repro BEFORE filing the mechanism as fact — it is also what stops you
shipping a fix against it. When a composed mechanism spans more than two
components, prefer instrumenting the running system to derivation: print the
invariants the theory depends on and let the system disagree. And when the
repro passes unpatched, say REFUTED in the record, loudly — the plausible
version left standing costs the next person a day.

---

## Sweep the files that ASSERT on your change, not the files you edited

**Incident.** TASK-25715, 2026-08-31. Across seven batches of Console Context
rail work I ran the test files I had touched, plus the ones whose names matched
the widgets I had touched. `Tests/UI/test_workbench_visual_snapshots.py` matched
neither: it edits nothing, is named after no widget I opened, and asserts on
whole-screen SVG renders of the Console. It is therefore precisely the file a
Console UI change breaks.

It went unswept twice. The first time it was holding four failures I had caused,
undetected across all seven batches. The second time — a post-merge sweep of the
whole `Tests/UI/` directory, run only because I was closing out the DoD — it
returned **five** failures where I was tracking two. Bisecting all five showed
none of the three new ones were mine, but I could not have known that without
running the file, and the first incident proves the coin lands the other way too.

**What to do.** Build the sweep list from what *asserts on* your change, not from
what you edited. For UI work that means the snapshot/visual-render files by
default, whatever they are named. When a change is broad enough to have batches,
run the whole containing test directory once before opening the PR — it is
minutes, and it is the only step that catches the file you did not think to name.

**Corollary.** When a sweep returns more failures than you were tracking, bisect
every one of them before writing any of them off. Three of the five here were
new since the branch point and two of those three traced to a single commit
(#2220) — a pattern invisible if you check only the failure you recognise.

---

## Establish that a failure is DETERMINISTIC before bisecting it

**Incident.** TASK-25715, 2026-08-31, immediately after the entry above. Of the
three failures I bisected there, one was flaky, and the bisect did not tell me
so — it returned a specific, plausible, wrong commit (`0ef6f3fd4`, #2252), which
I then published in two PR bodies as an attribution.

A binary search assumes the predicate is a function of the commit. Run once per
step against a ~1-in-12 flake, it instead converges on wherever the coin landed.
Measured after the fact: **11 passed / 1 failed of 12 at the very commit the
search had called FIRST BAD**, and 12/12 at the tip. Even the two failures that
made me open the investigation were the flake — hit twice while the machine was
busy running other pytest processes concurrently.

Nothing about the output looked wrong. `GOOD / GOOD / BAD / GOOD` reads exactly
the same whether the predicate is real or a coin. The two genuinely deterministic
failures in the same batch re-measured 0/5, and their bisects held.

**What to do.** Before bisecting, run the failing test **N times at the tip and N
times at the presumed-good base** (N≈10 is cheap for a UI test). Bisect only if
it is 0/N at one end and N/N at the other. If both ends are mixed, you have a
flake — a different defect with a different owner, and no commit to blame. After
a bisect names a commit, confirm by re-running the test several times at that
commit and its parent; a single run at each is the same coin flip that produced
the answer.

**And be suspicious of your own machine.** Concurrent test runs raise flake rates
for timing-sensitive UI tests, so failures observed while sweeps or other bisects
are running deserve a quiet-machine re-check before they become findings.

---

## Encrypted repository deletion must be tested with stale open instances (TASK-24400, 2026-08-29)

**What happened.** The first Personal Context repository implementation passed
single-instance key-destruction, reopen, transaction, and plaintext-canary
tests. An independent review then kept a second repository instance open across
destruction. That stale instance retained the old encryption and integrity keys
in memory and could commit a fresh encrypted outbox object after the first
instance had deleted every row and removed the protected key. The same review
found a separate first-open race: key creation happened before SQLite schema
ownership was serialized, so two processes could cache different keys while
only the last protector write survived.

**What to do.** Treat key custody, durable repository state, and cached process
state as three separate participants. Serialize first key creation with a
repository-owned write transaction and recheck schema ownership after taking
the lock. For deletion, commit a durable generation/destroyed fence inside the
same transaction that purges content, and check that fence inside every later
mutation transaction. Test with two simultaneously open repository instances,
not only close/reopen: a stale writer must either commit before the purge and be
removed by it, or acquire the lock afterward and be rejected. Also inject a
protector deletion failure and prove crypto-erasure can be retried without
re-enabling writes.

---

## A valid questionnaire is not evidence its complete answer set can commit (TASK-24407, 2026-08-29)

**What happened.** The workspace interview's production pack validated every
question independently, but several questions reused broad topics such as
`goal`, `working_context`, and `convention`. Proposal conversion reused those
topics as semantic-key subjects, so answering the whole questionnaire produced
duplicate keys and the atomic commit failed. Unit tests for individual answers
and payload types stayed green; the failure appeared only when one test answered,
reviewed, selected, and committed the complete production pack.

**What to do.** Treat a questionnaire pack as one transactional input, not a
bag of valid questions. Give each intended record a stable namespaced semantic
subject, assert the generated semantic keys are unique, and run at least one
end-to-end test that answers and commits every question in the real pack. This
is especially important when a compact topic label also participates in record
identity.
---

## Active surface order can diverge from append sequence after compaction

**Incident.** TASK-23113.9, 2026-08-31. The semantic-trace replacement benchmark
passed its first context compaction and failed on the second. The planner treated
the first and last changed entries in active display order as the numeric
replacement range. After the first compaction, however, a newly appended summary
occupied an older display ordinal with a newer sequence number, so display order
was no longer sequence-number order. The next range unintentionally swept an
unchanged active node.

**What to do.** For an append-only structure with bounded replacement records,
derive a replacement's numeric bounds from every changed active entry, then
explicitly reject the range if any unchanged active entry lies inside those
bounds. Do not infer persisted sequence bounds from the endpoints of a projected
display slice. A replacement-heavy test must compact the same surface at least
twice; a single compaction cannot expose this ordering divergence.

---

## A short post-ready timer is not a first-paint boundary

**Incident.** TASK-23113.9, 2026-08-31. After rebasing the trace rollout, the
UI-ready census repeatedly found Notes organization and Sync modules that were
supposed to load after the first interactive frame. The callback was scheduled
only after `_ui_ready` became true, but its 0.1-second timer started before the
rest of the synchronous post-ready setup completed. On a slow start, the delay
expired before the census task regained the event loop, so the callback and its
imports won the race. Merely moving the timer later reduced the remaining work
but did not establish a deterministic boundary.

**What to do.** Treat a short elapsed-time delay as load shedding, not as proof
that work happens after first paint. If a startup import must be absent at the
readiness boundary, use a comfortably separated idle-maintenance window (or an
explicit paint/phase signal) and enforce the absence in the real UI-ready
census. Keep first-use owners lazy as well: app-lifetime ownership does not
require eager construction.

---

## A profiled cost is not a felt cost — re-time hot paths without the profiler

**Incident.** TASK-25888, 2026-08-31, the Console button-latency hunt. cProfile
showed `Screen._refresh_layout` at 61 ms per rail click, and I built the whole
"110-165 ms per click" latency model on it — filed in the task description and
argued to the user. A paired direct measurement (perf_counter around the same
calls, same harness, no profiler) put layout at **8.5 ms**. The profiler's own
tracing overhead had inflated the hot path ~7x, and everything downstream of
that number was proportionally wrong.

Same investigation, two more instrument traps, all three now demonstrated in
one session:

- **Worker-thread cumtime does not add to the measured path.** The profile
  showed the config save at 166 ms/press cumulative; stubbing it moved the
  press median by ~6-11 ms. Concurrent GIL time shows up in cumtime as if it
  were serial.
- **Pick the paint that matches the felt thing.** press→FIRST paint said
  30 ms (measures the ack, misses the bill). press→SETTLED said ~210 ms on
  both sides of a fix (measures the tail of a trailing reconcile cascade the
  user never feels). The metric that tracked the fix was **summed main-thread
  blocking** inside layout+paint per press.

**What to do.** Use cProfile to find WHERE time goes, never to say HOW MUCH:
before quoting any per-call figure, re-time that call with perf_counter and no
profiler attached. Report worker-thread work as concurrent, and prove its
main-thread tax by stubbing it, not by reading cumtime. And when a fix is
claimed to make something faster, measure the quantity the user feels —
blocking time, bytes written — with the same instrument on both trees.

---

---

## A deterministic artifact is indistinguishable from a real effect by repetition

**Incident (2026-08-30 holistic perf review, TASK-25811, retracted).** A probe
reported that a Console -> Library switch spends **71% of its style work on the
screen being LEFT** — 1,107 applies of 1,577. It reproduced *to the call* across
three runs (332 / 389 / 1,577 / 384 every time). Reproducibility was cited as the
reason to trust it, a root cause was written up, a task was filed, and the finding
was committed and pushed.

It was an artifact. `switch_screen` posts `ScreenResume` to the **new** screen and
`post_message` is asynchronous. The navigation helper returned as soon as
`app.screen` changed — before that message drained — so every measurement window
opened with the **previous** navigation's resume still queued and counted it
against the switch under test. The screen it named as "being left" was simply the
screen that had just become current.

Draining messages before each window:

| window | resumes seen | applies | on "outgoing" |
|---|---|---:|---:|
| not settled | ChatScreen **and** LibraryScreen | 1,274 | 913 (72%) |
| settled | LibraryScreen only | 289 | **1 (0%)** |

Node counts also rose (207 -> 265) once settled: the unsettled runs were measuring
screens **mid-construction**.

**The rule.** Repetition tests for *noise*, not for *bias*. A window that always
straddles the same two activities always mis-attributes the same way, and looks
rock solid doing it. To test a measurement's validity you must **change the
measurement**, not repeat it: settle the system, move the window boundaries, or
measure the same quantity by an independent route. If three runs agree exactly and
the number is surprising, suspect the window before believing the finding.

The 2026-08-29 review had already recorded "a measurement window containing two
activities cannot attribute cost to either". It was quoted in the same document's
method notes and then violated three sections later — knowing the trap is not the
same as checking for it.

## Measure parse and boot costs in a COLD SUBPROCESS

**Incident (same review, TASK-25812).** An in-app probe timed a fresh
`Stylesheet` parsing the 671 KB boot bundle at **0.48 ms** — 1.4 GB/s, which no
Python tokenizer achieves. Clearing the two module-level caches that could be
found (`textual.css.parse.parse_selectors`, `is_id_selector`) did not change it.
The same content in a **cold interpreter** takes **121 ms** (5.5 MB/s, a
believable rate). Instrumenting the real `Stylesheet.parse` during boot
independently gave 191 ms, confirming the cold figure.

The fast number very nearly produced the conclusion "boot CSS parse is free",
which would have closed a real 191 ms finding — ~11-14% of a ~1.7 s cold start.

**The rule.** Anything memoised anywhere in the stack — parser caches, `lru_cache`
on selector parsing, interned strings, warm `sys.modules` — makes in-process
"cold" timing meaningless, and you cannot reliably enumerate every cache. Spawn a
fresh interpreter. **Sanity-check the rate: if a measurement implies more than
~100 MB/s of Python text processing, it is a cache artefact, not a result.**

## A failure list from a suite you have never run clean attributes nothing

**Incident (same review, verifying a global CSS-application change).** Two long
sweeps were run to check for regressions — 90 minutes and 33 minutes — producing
lists of 39-40 failing UI tests. Neither answered the question, because there was
no baseline: with no clean run to compare against, a failure list says nothing
about whether *your change* caused any of it. Both were discarded.

What worked was **paired arms**: the same 82 CSS-sensitive test files (1,105
tests) run twice, ~33 minutes each, once with the change and once with the
implementation reverted, then diffing the failure sets.

```
filter ON : 39 failed, 1064 passed
filter OFF: 40 failed, 1063 passed
broken by the change: NONE
pre-existing (both arms): 39
```

Several failures had looked alarming on the first arm — workbench visual
snapshots, eight visual-parity tests, CSS contract and build-integrity guards —
and all of them failed identically without the change.

**Two corollaries.**

* The single test that passed *only* in the changed arm was **not** a fix: it
  fails 4/4 in isolation with the change installed, and had passed through
  ordering luck. Do not claim a fix you did not engineer; check the candidate in
  isolation first.
* Paying for the baseline arm is the cost of an attributable answer. A cheaper
  run that cannot attribute is not cheaper — it is worthless.

## A manually pumped component path does not prove production wiring

**TASK-22512 Task 15, 2026-09-01.** Persistent-terminal component tests passed
because the tests themselves moved bytes between the backend and manager: they
called the backend reader, offered the bytes to the output actor, took queued
input, and wrote it back to the backend. No production owner performed those
steps. The first mounted real-shell proof therefore admitted and rendered a
Terminal workspace whose queued user input could never reach the PTY and whose
PTY output could never reach the screen. Adding the app-owned bridge made one
old native test fail for the opposite reason: its manual reader raced the now
real production reader and double-offered output.

**What to do.** When a component test connects two layers by hand, inventory
who owns that connection in production and add one mounted or end-to-end test
that crosses it without test-side pumping. After the real bridge exists,
component tests must observe it or use a backend without the runtime protocol;
they must not become a second runtime.

## Do not block a UI loop on work whose completion posts back to that loop

**TASK-22512 Task 15, 2026-09-01.** The mounted Disarm proof synchronously
called `wait_for_cleanup` from Textual's event-loop thread. Backend cleanup
finished quickly in an isolated real-PTY test, but the manager's final
subscriber notification used Textual's thread-safe UI handoff. The worker
could not return until the event loop handled that notification, while the
event loop was blocked waiting for the worker: a deterministic test-created
deadlock that looked like cleanup exceeding its five-second deadline.

**What to do.** In mounted async tests and production UI handlers, keep
blocking future/process waits off the event-loop thread. Await the async owner
API when one exists or move only the blocking wait to a worker thread; then
assert the UI projection after the loop has had a chance to process the final
notification.

## A PTY `EIO` is not final EOF while an owned descendant can still write

**TASK-22512 Task 15, 2026-09-01.** The production runtime bridge continuously
read the macOS PTY master and treated `EIO` after exact shell exit as irreversible
EOF. A same-session descendant still held the slave, reopened it after the shell
disappeared, and successfully wrote a delayed cleanup marker. The eager reader had
already latched EOF, so cleanup reported a complete stream without ever handing the
marker to the screen. The older component test had hidden this because it stopped
reading before shell exit and left final draining entirely to cleanup.

**What to do.** A running PTY read may treat `EIO` or an empty read as final only
after the shell reaper has fired and a complete ownership scan observes no owned
process. Until then it is backpressure, not proof. Keep the stronger two-scan,
deadline-bounded process and stream proof in the cleanup owner, and test delayed
descendant output through the production reader rather than a manually paused one.

---

## Hidden precomposition can move a Textual first-open race instead of removing it

**TASK-26840, 2026-09-01.** The first approval-card optimization precomposed a
hidden ordinary row so the first permission prompt could reveal an existing
subtree. Its mounted identity regression passed, but the production first-open
paint test intermittently rendered the title and tool details with the action
bar clipped. The hidden row still lived below flexible, initially hidden
containers, so precomposition changed when the widgets were registered without
giving Textual stable first-open geometry. Reverting that approach and retaining
the existing first-mount path restored the prior behavior; reusing the real row
only after its first successful mount delivered the steady-state speedup without
adding a new hidden-tree layout dependency.

**What to do.** Treat mount identity and first visible geometry as separate
contracts. When optimizing a Textual subtree that begins hidden, keep a
production-hierarchy first-open paint test and do not infer layout readiness from
precomposition alone. Prefer reusing a subtree after one successful visible
mount unless its hidden ancestors already have deterministic geometry.
## SQLite cursor wrappers invalidate exact-type ownership guards

**TASK-23113.11, 2026-09-02.** Quiescence tracking correctly moved the primary
database connection onto a `sqlite3.Connection`/`sqlite3.Cursor` subclass pair,
but Character insert and update ownership checks still required
`type(cursor) is sqlite3.Cursor`. Fresh-profile Samira seeding therefore rejected
the application's own tracked cursor. The ordinary character lifecycle test
caught the failed insert directly; the warm-start module census caught its less
obvious consequence, because every later boot retried the missing seed and kept
three parser modules resident.

**What to do.** When a database boundary intentionally supplies a standard-library
subclass, ownership checks must accept `isinstance(cursor, sqlite3.Cursor)` and
prove authority with connection identity and active transaction state. Exercise
at least one real startup or domain write through the wrapped factory; isolated
wrapper tests cannot expose an exact-type guard in a distant repository method.

## A pre-replace identity check cannot prove post-replace destination truth

**TASK-27038, 2026-09-01.** The first Tool Pack publication implementation
captured the parent and target inode, revalidated both immediately before an
atomic descriptor-relative replace, and reconciled parent-fsync failures. Review
still found two deterministic authorization/truth gaps. An incumbent could be
rewritten in place without changing its inode, so an old overwrite token remained
accepted; and a parent rename performed inside the replacement boundary published
through the still-valid old directory descriptor but returned ordinary success for
a pathname that no longer named the result. The same review also found that the
capability gate checked directory-descriptor support on `os.rename` while the code
actually called `os.replace`.

**PR #2324 follow-up, 2026-09-02.** A later review exposed the remaining gap:
even inode-plus-digest revalidation occurs before `os.replace`, so a concurrent
writer can still substitute the name between the check and mutation. No amount of
pre-replace observation turns an unconditional rename into compare-and-swap.
Tool Pack V1 therefore disabled existing-file overwrite and changed absent-target
publication to descriptor-relative `link`, whose create-only result is atomic.

**What to do.** A captured identity authorizes replacement only when the mutation
primitive atomically compares that identity. Without such a primitive, fail closed
for existing targets; for absent targets, use an atomic create-only operation rather
than check-then-replace. Probe the exact callable and exact parameters that the
mutation path will invoke. After any point where publication may have occurred,
keep the authenticated parent descriptor open and reconcile the currently named
parent, target identity, and content digest before claiming ordinary success; a
named-parent mismatch is an uncertain committed state, not proof the old file
survived. Tests must inject a competing create and parent rename inside the atomic
publication call itself—checking only the phase before it leaves the decisive race
untested.
## A method inserted above a decorated method steals the decorator -- and the symptom is a silently dead handler (task-31000, 2026-09-02)

**What happened.** While wiring `TldwCli.on_key` (forward startup-splash
skips), the new method was inserted directly above
`on_splash_screen_closed` -- between the pre-existing
`@on(SplashScreen.Closed)` decorator line and the function it decorated.
Python happily decorated the new `on_key` instead. Textual marks decorated
handlers with `_textual_on` metadata and then *excludes them from
name-based dispatch*, so the "new" `on_key` had become a Closed-message
handler that ignored its event, and no keypress ever reached it. The trap
cost a long debugging detour because monkeypatching `TldwCli.on_key` at
runtime (a fresh, undecorated function) made the skip *work*, while the
identical class-defined method stayed dead -- the discrepancy was only
explained by printing `getattr(fn, "_textual_on", None)`, which showed the
stolen `SplashScreen.Closed` binding. A duplicated-decorator syntax error
would have failed loudly; this fails silently at runtime.

Rule: when inserting a method into a class with `@on(...)`/decorated
handlers, check the insertion point is not between a decorator and its
function. When a Textual event handler is inexplicably never invoked,
print its `_textual_on` attribute before blaming dispatch order or focus --
a non-empty value means the decorator attached to the wrong function.

## Textual `set_interval` skips missed ticks, so wall-clock-paced animation jumps under load (task-31000, 2026-09-02)

**What happened.** The startup splash was intermittently jumpy or skipped
from an early frame straight to its end. Measurement (headless `run_test`
probe with a controllable `time.sleep` blocker on the app's event loop)
showed the mechanism: `Timer` defaults to `skip=True`, so callbacks that
could not run while the loop was blocked are *permanently skipped* (one
callback fires after the delay, not a catch-up burst), and the splash
effects derive their progression from `time.time() - effect.start_time`.
A 1.2s block right after arming made the first visible frame land at 1.22s
of a 2.5s reveal; a block longer than the splash duration let the
auto-close timer beat every frame. The fix re-anchors the effect clock per
*rendered* frame (virtual elapsed = frames x interval), so contention
slows the animation instead of skipping it ahead.

Rule: anything animated by `set_interval` in this repo (splash effects,
console background, tamagotchi, activity-log timestamps) must advance its
state per rendered callback, never from wall-clock elapsed, or it will
jump whenever the event loop stalls. Verify pacing behavior with a
deliberate blocker on the loop, not by watching an idle machine where
everything looks smooth.

## Unrelated-looking Agents test failures can be schema-budget collapses

Incident (TASK-28238 phase 2, Task 7, 2026-09-03): adding two ~300-token runtime
tool schemas turned `test_run_log_prompt_integration` /
`test_run_log_service_wiring` red — tests that never mention worktrees. Root
cause was not the new feature's logic: those tests use an unrecognized model
string, `get_model_token_limit` falls back to a 4096-token context (2048
response reserve), the schema surface had been fitting with 47 tokens to spare,
and `validated_fallback` is fit-or-nothing — over budget by one token drops
EVERY tool, including run-log. Lesson: when adding ANY always-disclosed tool
schema, a red test in a seemingly unrelated Tests/Agents suite is evidence of
budget collapse, not flake — reproduce by calling
`build_first_request_schema_plan` directly and printing `used` vs the
reserve before dismissing or "fixing" the test. Structural fix tracked in
TASK-31212.

Follow-up incident (TASK-31232, Canvas Task 7.4, 2026-09-04): an actual served
Chatbook fixture used the unknown `canvas-live-model` and scripted an immediate
`canvas_create`. The real first request instead disclosed `find_tools` /
`load_tools`; the tool was refused before creation. Counting `stream_chat`
native `tools=` entries as zero also misdiagnosed fenced-mode disclosure,
which is rendered in the system prompt. Once the synthetic gateway honored
discovery, the real Console finalized the assistant turn and rendered the
first revision. Provider doubles must obey the effective disclosure protocol
or declare a realistic fixture context window; diagnose with exact tool-name
membership and bounded refusal codes, not raw prompt/source dumps or the
native-tools keyword alone.

The same fixture then appeared to hang on the second turn. Inspection showed
its update states fell through to final assistant text without emitting a
tool call; after correcting that, an unsupported `title` argument caused a
real `invalid_arguments` refusal. Direct/progressive fixture-contract tests
and the actual served Console create/update test then passed (3 tests). A
provider call counter proves only a model request happened, not that tool
dispatch or settlement started. Validate the synthetic response against the
actual advertised schema and observe the next boundary before assigning a
timeout to production locking or scheduling.

## Native-realm security sentinels must not break the host being measured

Incident (TASK-31232, Canvas Task 7.4, 2026-09-05): an enumerable property added
to native `Object.prototype` by the browser security harness prevented the
served terminal from connecting, before any generated Canvas script ran. The
existing TLS flow without that probe still passed. A paired run changing only
the sentinel to non-enumerable restored terminal and Canvas readiness. Define
probe properties explicitly as non-enumerable, writable, and configurable:
enumerability can perturb host iteration, while non-writable sentinels can mask
the native overwrite the probe is meant to detect. Keep the sentinel's name
identical to the attack target and diagnose pre-execution failures separately
from containment failures.

## A painted Textual editor can still be outside its hit-testable layout

Incident (TASK-31215, 2026-09-03): demand-mounted Personas editors initially
used auto-height wrapper slots around editor roots that own `height: 100%`.
Widget queries and paint-based assertions passed, but real actor-pack button
clicks were ignored because the controls were outside the wrapper's effective
hit-testable geometry. Giving the stable slot its natural container height
restored both full-height layout and pointer dispatch.

Rule: when introducing a wrapper around an existing full-height Textual view,
verify it with a real pointer-driven workflow, not only widget queries,
visibility flags, or screenshots. Painted descendants do not prove their
ancestor geometry participates in hit testing.

## A PR's green CI never ran the MERGED combination -- re-run the changed method's fake-driven tests on the merged head (task-31237, 2026-09-04)

Incident (PR #2367, 2026-09-04): the branch added `_close_library_media_find()`
and routed `_delete_library_media_item`'s query reset through it instead of
assigning the two query attributes directly. Pre-merge CI was green. Meanwhile
dev had merged task-14901's two single-delete tests, which drive
`_delete_library_media_item` on a bare `SimpleNamespace` fake -- attribute
assignment works on a SimpleNamespace, a method call does not. On the merged
head both raised `AttributeError`; PR Fast Lane does not run
`test_library_multiselect_media.py`, the update-branch merge commit went green,
the landing loop admin-merged it, and dev needed follow-up PR #2369. Same shape
on #2366 the same night: two PRs each regenerating
`Docs/security/production-diagnostic-inventory.json` merge cleanly, but the
summary count carries only one side.

Rule: after the update-branch / merge-with-dev step and BEFORE admin-merge, grep
`Tests/` for every production method whose body you changed
(`grep -rn '<method_name>' Tests/`) and run those files on the merged head --
a direct-method fake that dev added after you forked will not be in your
branch's test list. Regenerate derived inventories on the merged head, not on
the branch tip. A green check on the pre-merge head is evidence about the
branch, not about dev-plus-branch.

## Strengthening a postcondition can expose an incomplete green fixture (TASK-30013, 2026-09-03)

**What happened.** Endpoint setup compare-and-swap observations were expanded
to include credential routing, so changing an unset environment-variable
declaration could no longer evade the locked precondition. The new focused
tests passed, but the broader provider-setup gate turned an older postcondition
test red. Its simulated successful `after` snapshot omitted
`credential_source = "stored"`, even though `build_provider_setup_mutation()`
had always written that field. The old, weaker observation simply could not
notice that the fixture described a state the production mutation would never
produce.

**What to do.** When a stronger invariant breaks an established postcondition
fixture, compare the fixture with the exact projected mutation before weakening
the invariant. A newly red old test may be revealing missing state in its
oracle, not a compatibility regression. Then run a cross-file gate that covers
both the new invariant and the existing projection tests; the isolated new
tests alone cannot expose disagreements with older test models.
---

## A bundle-only render harness misses the per-screen sheets since the agentic split

**TASK-31254, 2026-09-04.** A painted-frame test for the Settings theme editor
loaded `tldw_cli_modular.tcss` on a bare harness (the `test_checkbox_height_render.py`
pattern) and every colour `Input` rendered as a clipped tall border, so the
swatch-text assertion could not pass even after the fix. The bundle no longer
carries `.settings-compact-input` / `.settings-input-row`: TASK-25812 split the
Settings-owned rules out of `components/_agentic_terminal.tcss` into
`css/screen_agentic_settings.tcss`, a per-screen sheet the app loads alongside the
bundle. An earlier deterministic sweep of the bundle had reported those classes as
"no rule", which was true of the bundle and false of the app.

**What to do.** A harness that claims production CSS must register the bundle
AND the owning screen's split sheet (`screen_agentic_console|library|settings.tcss`),
in the app's order. When a bundle grep says a Settings/Console/Library class has no
rule, grep the split sheets before concluding the state is unstyled.

---

## Textual's `Color.hsl` hue is 0-1, not degrees

**TASK-31253, 2026-09-04.** "Generate from Primary" produced a red secondary and a
cyan accent for every primary colour (live: `#9966FF` -> `#e83735` / `#65fdff`).
`textual.color.Color.hsl` returns `HSL(h, s, l)` with `h` in the 0-1 range; the
generator fed it to a helper working in degrees, so hue was always in `[0, 1)`.
The unit test that caught it asserts hue *distance* between generated colours and
the primary, which is the assertion a palette generator needs.

**What to do.** Multiply `hsl.h` by 360 before any degree-based maths, and test
generated palettes by hue distance for several primaries, not by eyeballing one.
## Theme `variables` dict entries are NOT overrides — tcss definitions shadow them (task-31264, 2026-09-04)

**What happened.** PR #2374 "fixed" light themes inheriting dark-tuned tokens
by adding `ds-status-error-readable`/`ds-text-placeholder` entries to each
theme's `Theme.variables` dict, verified with contrast arithmetic on the dict
values. The fix was inert at runtime: a `$name: value` definition in any tcss
source shadows app-supplied variables for that source (proven with a minimal
`Stylesheet(variables=...)` probe — the file's value tokens are appended after
the app's and last-token-wins), and `_variables.tcss` defines every `ds-*`
token. The frozen `$ds-focus-bg: #51677e` literal was painting slate focus
states on every light theme regardless of what the theme dict said.

**What to do.** A theme's `variables` dict only reaches CSS for tokens NO
loaded stylesheet defines; for `ds-*` tokens it is documentation, not an
override. To make a design token theme-aware, define it in tcss as a
*reference* to one of Textual's generated polarity-aware variables
(`$text-error`, `$text-muted`, `$block-cursor-blurred-background`, …), never a
hex literal. And never verify a color fix by arithmetic on configuration
values — check the resolved paint (rule-match probe or live capture); the
dict-value contrast test in PR #2374 passed while the app painted the
opposite.

## Patch the owner of a lazy dependency, not a stale consumer alias (TASK-31301, 2026-09-04)

**What happened.** Conversation Settings endpoint tests patched
`settings_screen.probe_settings_endpoint` and
`chat_screen.probe_settings_endpoint` after the production call sites had moved
those imports inside their methods for the boot-budget boundary. One patch used
`raising=False`, which silently installed an attribute the production code never
read. The test then reached a real localhost endpoint instead of its fake; only
the network guard exposed that the apparently isolated test no longer controlled
its dependency.

**What to do.** When production lazily imports a dependency inside a method,
patch the symbol on the module that owns it. Do not use `raising=False` to make a
missing consumer alias patchable. Pair the fake-driven test with one explicit,
owned-loopback integration test so both isolation and the real call path remain
observable.

---

## Formatter directive guards must follow logical owners, not physical lines

**TASK-26947, 2026-09-04.** The first formatter structural guard measured an
inline directive from the preceding physical `NEWLINE`, then from only an
`ast.stmt` ancestor. Ruff split semicolon-separated siblings and added grouping
parentheses, so the unchanged directive appeared to move even though its AST route,
comment text, and association had not changed. A later `# noqa` on a same-line
`ExceptHandler` header exposed the other hole: an `ExceptHandler` is not an
`ast.stmt`, and falling back to the enclosing `try` would make unrelated suite
tokens influence the header directive. Guard v3 fixed all three cases with
logical-owner boundaries, full-module shadow parsing to exclude only independently
proven AST-neutral parentheses, and a uniquely validated `except` clause through
its unique depth-zero colon; its regression tests cover semicolon splitting,
grouping parentheses, and header-versus-handler-body directives.

**What to do.** Treat a directive-position metric change as a new baseline
contract, not a comparison against old ordinal data: recapture the structural
baseline before reformatting. Fail closed for ambiguous exception headers, never
discard tuple commas or semantic grouping, and keep the ordinary nearest-statement
or decorator boundary for every non-header directive.

---

## A hand-written JSON fixture that omits one real field can hide a 100%-reproducible crash (TASK-31551 task 13, 2026-09-04)

**Incident.** `Audio/meeting_owner.py::recover_folder()` reads a crashed meeting's
`meeting.json`, updates a few fields, and calls
`update_meeting_json(folder, **payload)` — a function whose first parameter is
also named `folder`. Every meeting.json the app's own writer ever produces
(`MeetingSession`/`meeting_owner.start()`/`stop()`) includes a `"folder"` key, so
in production this call *always* raises
`TypeError: update_meeting_json() got multiple values for argument 'folder'`.
The recover button's worker is `@work(thread=True)` with Textual's default
`exit_on_error=True`, so this single `TypeError` did not just fail the recovery —
it took the entire app down. Both of the feature's existing `recover_folder` unit
tests (`test_scan_and_recover_unfinished_folder`,
`test_recover_folder_survives_missing_mixed_wav`) hand-write their own
`meeting.json` fixture from scratch, and both happen to omit the one key
(`"folder"`) that the real writer always includes and that triggers the crash —
so a fully reviewed, 100%-passing feature crashed on the very first live
recovery attempt (reproduced by `kill -9` on the running app mid-meeting,
relaunching, and pressing Recover). It was found only by live-driving the actual
crash → relaunch → Recover cycle in the real app under tmux, not by reading the
diff or the passing suite.

**What to do.** When a test constructs a JSON/dict fixture by hand to feed a
function that reads a file your OWN code also writes elsewhere, do not write the
fixture from what the function's logic *seems* to need — grep for every call
site that actually produces that file/payload in production and copy its real
shape (or better, round-trip through the real writer function itself) into the
fixture. A hand-typed fixture that "looks about right" is a guessed contract,
and the field most likely to be missing is exactly the one the code path
under test never gets to touch precisely because the guess omitted it. For any
`@work(thread=True)`-decorated callback that can reach unvalidated on-disk
data (recovery, import, migration), also check whether Textual's default
`exit_on_error=True` means a single unhandled exception there takes down the
whole app, not just the feature — that raises the cost of an untested edge case
from "one broken button" to "total data-loss-adjacent crash."
---

## A green count from a partial glob is not a green gate — paste the exact tail, and reviewers re-run it themselves

**Incident.** Schedules redesign PR-4, Task 3 (2026-09-04). The task report claimed
the suite gate was met and quoted a passing count ("292 passed"). The number was
real, but it came from a **partial glob** over the test files the task had touched —
not from the gate the plan specified. Two pinned tests outside that glob were red at
the same moment. Nothing in the report was fabricated; the run simply did not cover
what the claim covered, and a summarised count carries no evidence of its own scope.

It was caught only because the reviewer re-ran the gate independently instead of
reading the report's claim. Had the reviewer trusted the number, a red branch would
have gone up as green — and the two failures were in exactly the pinned tests a
reviewer would assume the implementer had run.

This is the same failure shape as `A failure list from a suite you have never run
clean attributes nothing` (above), inverted: there, an unrun suite was used to
attribute failures; here, a partially-run suite was used to deny them.

**What to do.**

- **Paste the exact tail lines from the FINAL run**, including the invocation that
  produced them. `pytest ... -q` plus its `N passed, M failed` line is evidence; a
  count retyped into prose is a claim. The invocation is the load-bearing half —
  it is what makes the scope auditable.
- **Never quote a count from a run that is not the gate.** A scoped run while
  iterating is fine and normal; it just is not the thing you report as the gate.
- **Reviewers re-run the gates.** Do not adjudicate a green claim from the report.
  The re-run costs minutes; this one caught a red branch, and it is the only step
  in the loop that is independent of the implementer's own scoping mistake.
- If the gate is expensive, that is an argument for naming it precisely in the plan,
  not for approximating it with a glob.

---

## An assertion that reads back the value the code just wrote confirms nothing — assert on PAINTED output

**Incident (round 1).** Schedules-handoff PR-6, live task 6 (2026-09-02). The Results
tab's unread badge never rendered. `pane.label = f"Results ({n})"` on a `TabPane` sets
an **inert attribute** — Textual 8.2.8 stores the title in `_title`, and `TabPane` has
no `label` reactive at all, so the assignment did nothing to the UI. The regression
test (`Tests/UI/test_schedules_results_tab.py:408`) passed the entire time, because it
asserted on `pane.label` — reading back the attribute the code had just set. A test
shaped that way passes for **any** write to any attribute name; it cannot fail while
the assignment executes. The badge had also been broken in the Conflicts tab before
PR-6 copied the pattern, so a green test had been guarding a dead feature for months.

**Incident (round 2), same programme.** The Automations table silently ate its
`[<server id>]` owner prefix: `DataTable` cells go through
`rich.text.Text.from_markup`, whose lowercase-tag regex matches `[http://…]`. Here too
`table.get_cell_at()` returns the **stored** value and passes regardless of whether the
content survives rendering. The fix migrated the assertions to
`Tests/UI/schedules_test_helpers.py::rendered_row_cells`, which routes the stored row
through the widget's own `_get_row_renderables` -> `default_cell_formatter` — the exact
path where a bracket token is eaten.

**The tell.** Ask of every assertion: *what code path must break for this to fail?* If
the answer is "the assignment statement two lines above in the production code", the
test is a self-confirmer. Both of these were written in good faith by someone who had
just watched the feature not work, and both passed on the broken build.

**What to do.**

- When the point of a test is that something **renders**, assert on the rendered
  artifact: painted cells (`rendered_row_cells`), compositor strips, `render_line`,
  `region.width > 0`. Not the stored value, not the attribute you assigned.
- Treat "framework attribute assignment" as an unverified hypothesis until a painted
  assertion confirms it. `widget.foo = x` on a non-reactive attribute is a silent
  no-op in Textual, and the read-back is indistinguishable from success.
- When a live run finds a feature dead that a green test covers, **read that test
  first**. It is more often a self-confirmer than a coverage gap, and fixing the
  feature without fixing the test leaves the next regression unguarded.

---

## Textual projection writes must suppress their own deferred change messages

**TASK-31389, 2026-09-03.** The mounted vLLM generation-fencing test initially
hung after one model edit and advanced the connection generation thousands of
times. The view's `_rendering` flag covered the synchronous `Input.value` writes,
but Textual delivered their `Input.Changed` messages after the flag had already
been cleared. Because two source controls projected the same draft field, stale
echoes alternated values and recursively triggered draft invalidation and another
projection. A timeout traceback finally caught the loop applying state while the
view was being torn down.

**What to do.** Wrap imperative `Input.value` and `TextArea.text` projection
writes in the widget's `prevent(...Changed)` context. A synchronous rendering
flag alone does not suppress a message delivered later. Prove the boundary with
a mounted test that applies state, performs one user edit, and asserts exactly one
semantic generation advance.

## Closing a thread-local database on the fixture thread does not close worker-owned handles

**TASK-31392 Task 6 Fix Round 2, 2026-09-04.** The qualified vLLM primary
reported 237 additional file descriptors. File-level bisection isolated the
growth to mounted Textual tests; `lsof`, GC inspection, and live connection
registries then showed one `_QuiescentSQLiteConnection` per app instance. The
real on-mount FTS backfill opened each handle on a worker thread, while fixture
teardown invoked `close()` only from the main pytest thread. That closed the
main thread-local handle but could not reach the worker thread's handle. Two
mounted cases retained 2 SQLite/9 regular descriptors after teardown even
after GC. Draining the database's process-local quiescence registry reduced
that to 0 SQLite/3 regular descriptors, and repeated mounts stopped growing
linearly.

**What to do.** When a database owns thread-local connections and tests run
real worker-backed startup work, teardown must use the database's all-handle
quiescence/registry boundary rather than a single-thread `close()`. Diagnose a
session FD warning by splitting test files, classifying descriptors with
`lsof`, and inspecting live owners after finalizers; GC or a higher threshold
cannot establish ownership or fix a registered worker handle.

## Lifecycle relocation tests must include production change notifications

**TASK-21123, 2026-09-04.** Moving Buddy ownership to the app initially passed
the old mount harness, which manually reconciled state and omitted the new
controller notification callback. Wiring the production callback exposed tests
that installed mount gates after enable had already mounted the view, and a
close/geometry test that sent input after the view had retired. Explicit caller
cancellation tests now isolate their caller; the merge test gates owner retirement
while admitting both edits. Independent review also reproduced late mounts after
shutdown during a geometry flush and reuse of a generation-invalidated view after
canceled retirement. Both received failing-then-passing regression tests.

**What to do.** Bind production lifecycle notifications in integration harnesses.
Gate the specific await boundary a race test intends to exercise; a persistence
writer starting does not prove the originating view is still current. Check
shutdown and generation authority again after teardown awaits, before reuse/mount.

## `Screen.CSS_PATH` loads under EVERY app — including the unstyled-tier harnesses (TASK-24459, 2026-09-04)

Splitting `features/_scheduling.tcss` off the boot bundle, the first wiring
put the generated sheet on `SchedulesWorkbench.CSS_PATH` — the same pattern
TASK-25812 used for the library/settings agentic sheets. The paired
evals/schedules arm then flipped three destination-shell geometry tests
that were green on the pristine base. The probe showed the mechanism:
Textual's `_load_screen_css` fires when ANY app pushes the screen, and the
`ConsolidatedCSSApp` harnesses deliberately load no app bundle — so a
harness-mounted workbench now rendered with ONLY the moved half of the
module, a hybrid of the styled and unstyled tiers that no user ever sees
(the automation-detail overlay covered `#schedules-follow-in-console`; the
click landed on the overlay and the mock never fired). The inverse bite
followed an hour later: harnesses that model the PRODUCTION stylesheet via
a hard-coded sheet list (`ProductionCSSDestinationHarness`, and
`test_schedules_responsive_floor`'s `CSS_PATH = BUNDLED_STYLESHEET`) lost
the moved rules and failed four MORE geometry tests — that list had
already gone stale once when 25812 landed, and went stale again the day a
new split shipped.

Rules, each half of the incident:
- A split-off feature sheet must load from an APP-owned seam
  (`TldwCli._SCREEN_OWNED_ROUTE_CSS` → `_ensure_screen_owned_css`), never
  `Screen.CSS_PATH`, unless every harness that mounts the screen has been
  audited for the hybrid. Guards now pin both directions (map
  completeness + a CSS_PATH ban on the owning screens).
- A harness that claims "the production stylesheet set" must DERIVE it
  from the build authority (`consolidated_css.APP_STYLESHEETS`), not name
  sheets by hand.
- The failure set of a paired arm settles attribution only test-by-test:
  of the eight failures on the branch arm, five were pre-existing, three
  were real — and the real three were invisible without the pristine-base
  arm run in the same mode.

Recurrence, PR2416 (2026-09-05): the rebased Chunking Lab route/ingest selection
passed146 tests but failed both progress-detail color assertions. `_QueuePanelHost`
still named only `tldw_cli_modular.tcss`, so Library-owned progress and row colors
both resolved to white. Switching this local host to the existing
`APP_STYLESHEETS` authority made the exact two compositor assertions pass in4.53s;
the combined targeted gate then passed635. No production styling changed. Extending
the shared authority does not repair local hosts that continue to bypass it.

## A `-k` name filter is not a gate for a behaviour change that flips an existing pin (media wave 5 PR E, 2026-09-05)

**The incident.** Task 3 of the wave-5 bulk-mutation PR moved the receipt's `Undo` off the stale gate — a deliberate behaviour change. The implementer's verification ran `test_library_shell.py -k "undo or receipt or delete"` and reported parity with the base. The task reviewer then found two red tests, one in `test_library_shell.py` and one in `test_library_media_side_by_side.py`, that assert `#library-media-bulk-delete-undo` is DISABLED under a stale page. Their names carry the gate ("stale", "write_gated"), not the action, so the filter never selected them; they had been red since the change and nobody had run them. A whole-file run of both files would have caught it in the same session; it took a second reviewer and a fix round instead.

**What to do.** When a change flips what an existing pin asserts (a gate, a disabled state, a focus target), the gate for that change is the WHOLE files that pin the gate — here `test_library_shell.py` and `test_library_media_side_by_side.py` — compared as failing-name sets against the base, not a `-k` subset named after the action. The 80-minute whole-file shell run is the price; run it once per PR at the review boundary, not per task. `-k` stays fine for iterating, never for the parity claim.

## Covered reusable screens can still report visible

PR2419 reuse integration (2026-09-05): after Console became reusable, Environment
collectors still dispatched while covered because Textual suspension preserves
widget `display`. A mounted regression proved four unwanted dispatches. Its first
attempt also exposed that `Screen.is_current` includes background screens. Use
`app.screen is screen` for top-screen-only I/O, including deferred dispatch gates;
exercise real cover/return and retained-owner refresh, not a visibility mock.

## Stop must drain an offloaded dispatch CAS before terminal settlement

**TASK-31585, 2026-09-05.** Real DeepSeek UAT requested Stop as soon as the
Console reported streaming, before the first token. The UI ended BLOCKED; its
retained isolated SQLite database still held an empty `dispatch_started`
assistant and checkpoint revision 2 after the matching child exited. A later
after-text Stop passed, but that did not resolve the earlier user-facing race.
The worker-thread CAS outlived cancellation before its result was published;
the terminal guard mistook the accepted in-memory checkpoint for a previous
settlement failure. Gate both before and after the real file-backed CAS, drain
its publication before settling Stop, and verify the persisted terminal owner
and checkpoint deletion through another connection after the worker finishes.
Cover both direct-provider and agent pre-worker paths, repeated Stop, and
another live session. A transcript marker read from the store is only in-memory
evidence unless the database is checked separately.

PR2428 review exposed a second instance through the real generic synchronous
gateway: `run_coroutine_threadsafe` detached the dispatch callback from the
cancelled stream. Shielding only the callback did not make the stream wait.
The regression now uses that actual gateway and waits for its worker's terminal
acknowledgement; both callback and stream drain the same assistant-owned task
before terminal settlement.

## Trace integration tests must use the production boundary factory (TASK-31714, 2026-09-05)

During mounted snapshot UAT, ordinary second sends failed even without any
snapshot operations. Existing controller tests used a hand-built append-only
trace boundary; lower-level factory tests supplied already-correct descriptors.
Both missed private persisted-ID annotations surviving in semantic history but
being removed from the wire, which caused saved descriptors to become artifacts
at the real agent boundary. A real SQLite/controller/agent/gateway test using
`ConsoleTraceBoundaryFactory` reproduced the failure before the correction.
The same UAT then exposed a separate tool-loop `trace_turn_unavailable` rejection
that the custom test boundary also bypasses.

Use the production factory for at least one multi-turn and tool-loop integration
test; fake only inference. Compare native-reader reconstruction before/after a
later turn, and keep a changed-history negative control. A completed call row or
a mocked append boundary alone does not prove historical request reconstruction.

Follow-up, TASK-31737: the real production-boundary tool test also exposed that
response settlement is queued when the next model call begins; requiring a
terminal `complete` row incorrectly blocked a valid `response_started` chain.
Independent review then reproduced an old chain being admitted after a newer
chain dispatched the identical prompt. Surface equality did not prove current
call ownership. A failing same-surface/new-run test led to an atomic ordered
call-boundary event and latest-call check. Use durable call order for supersession,
not surface identity or timestamps, and exercise deferred settlement in the real
agent path rather than forcing it synchronous in the test.


## Built-in asset cleanup must distinguish failed return from failed commit (pixel-migu, 2026-09-05)

During pixel-migu first-install review, a coordinator wrapper that committed the
Persona and SQLite graph and then raised `RuntimeError` exposed a cleanup bug:
the caller deleted the graph's newly published PNGs even though restart treated
the Persona as already installed. The focused regression also exercises
`KeyboardInterrupt` and `SystemExit`. Cleanup now checks durable graph ownership
and retains assets when that ownership cannot be determined. A separate
interleaved-service regression found that a losing caller needed to refresh its
Persona JSON cache before the winning installation became selectable. A green
rollback-only test did not cover either postcommit behavior.

## A screen's own recovery dialog suspends it too (TASK-31756, 2026-09-05)

The mounted Parakeet Retry/Keep draft tests reproduced a composer stuck on
`Dictate…` even though the dialog completed. Tracing showed that opening the
retry dialog invoked Console's suspend cleanup, discarding retained audio and
clearing the originating session before its completion handler resumed. Tests
that replaced `push_screen_wait` with a returned boolean never exercised this
suspend transition.

Keep recovery retention scoped to the exact owned dialog; actual navigation and
unmount still need unconditional cleanup. Verify both choices through a mounted
dialog, plus teardown before a late affirmative answer. The latter regression
failed with an unwanted replay when the post-dialog session fence was removed.

## Resource-limit probes must distinguish refusal from a broken engine

Incident (TASK-31232, Canvas Task 7.2, 2026-09-04): the first direct QuickJS
quota probe turned every native exception into a successful limit refusal.
Injected API/disposal failures therefore produced apparently valid heap/stack
evidence. Strict error classification and positive controls exposed that real
deep recursion instead caused an exact native engine stack trap and poisoned
that runtime; it did not return a normal guest stack-limit error. The corrected
probe reports guest out-of-memory refusal separately from engine-trap
containment, closes the trapped context, and explicitly does not claim that
the configured stack ceiling caused the trap.

Rule: demonstrate a successful positive control, identify the actual failure
mechanism, and fail qualification on unexpected host/API/disposal errors.
A generic exception or timeout is evidence of failure, not proof that the
resource boundary under test enforced its configured limit.

## A tested quota helper is not evidence that the production owner uses it

Incident (TASK-31232 final Canvas review, 2026-09-05): the standalone staging
store enforced Canvas admission quotas, but the real Console wired its own
controller/coordinator. That owner appended revisions without the helper's
admission checks. A direct production-coordinator probe staged and confirmed
11 Canvases despite the limit of 10. Repository commit checks could not protect
temporary confirmed history or bound the earlier staged allocation.

Rule: trace the actual provider-to-mutation-owner wiring before selecting a
quota test target. Assert admission at the production owner with existing and
concurrent work, not only on a similarly named helper or final durable commit.

The same review's raw Python probe imported application configuration before
establishing owned test directories. Logs reported loading ambient config and
ensuring the user's chat_dicts directory; no pre-probe snapshot could determine
whether the directory was newly created. No database/provider/browser launched,
but that is not proof of no filesystem effect. Prefer repository pytest fixtures
and establish owned config/data before application imports: an in-memory subject
does not make its module's bootstrap side-effect-free. Do not delete ambient
state afterward or suppress the deviation from the verification record.

Follow-up incident (TASK-31232 six-baseline repair, 2026-09-05): the same mistake
recurred while counting public Library descriptors with a direct Python import,
despite an explicit isolated-pytest-only instruction. The import again reached
ambient config and directory bootstrap. Static inspection would have answered the
question. Executable worker probes were stopped; the coordinator took ownership
of all subsequent pytest execution while the worker was limited to static edits.
When a procedural warning demonstrably fails, narrow the execution workflow
instead of repeating the warning and treating intent as isolation evidence.

## Parameter IDs can accidentally activate keyword-based test gates

Incident (TASK-31232 DOM-only correction, 2026-09-05): a local Chromium
regression parameter named `live` meant an existing rendered child, not an
external service. The repository's collection hook checks `"live" in
item.keywords`, so the parameter case was skipped without `--run-live`.
Renaming it `existing` exercised the intended case and exposed the expected
duplicate-create failure. The definitive RED separately parameterized both
selection outcomes too, preventing the first failure from masking the second.

Rule: inspect collected outcomes and skip reasons, not just command success or
test-function count. Use parameter IDs that do not collide with keyword-based
suite gates; do not opt into external/live suites to rescue a local case.
Independently parameterize distinct required failures when an earlier assertion
would otherwise prevent a later one from running.
