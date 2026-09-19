# Conversations Use as source — 2026-09-16

TASK-32706 qualifies the separate **Use as source** journey in dark/light themes
at 170×48 and 80×24. Eight handoffs preserve the current Console session and its
populated draft. Link-on-use, repeated use, Un-stage and Undo link pass. The review also exposed
and repaired TASK-2502: a reused Console ignored the separate live-work channel.
First and replacement live-work launches now refresh the visible evidence controls.
No CSS changed.

The staged reference currently contains the conversation identity and a generic
title label, **not transcript text**. This is the known, still-open TASK-2376
content-fidelity defect; this qualification does not close it.

ADR required: no new ADR. This verifies existing
[ADR-147](../../../../backlog/decisions/147-conversation-archive-and-exact-resume.md)
and [ADR-005](../../../../backlog/decisions/005-console-workspace-server-readiness.md).
The Impeccable audit workflow is adapted to Textual focus, compositor paint,
terminal geometry and truthful state.

## Native qualification

The [runner](native_check.py) uses real TldwCli, LinuxDriver, mounted actions,
the CHAT handoff consumer, Console store, workspace registry and SQLite. It
validates private paths before app imports, verifies the imported checkout and
requires the exclusive profile lock. Its SHA-256 is bound into [result.json](result.json).
The checkout baseline was `0960d40433`; the runner receipt does not independently
record the Git revision or dirty state.

[Fixtures](fixtures.json) contain four distinct, initially unlinked global
conversations, each with a user and assistant message. A fifth conversation is
linked only to a different workspace. The initial Console tab has a populated
draft entered by a Paste event: `Unrelated unfinished draft: keep [this] and café.`

For each theme/size cell, final private run-002 establishes:

1. Ctrl+3 opens Library; filtering selects the exact fixture. Enter opens its
   actual row. At 80×24 Items is explicitly collapsed. The reader paints the
   focused Use as source action and its link-on-use explanation.
2. Enter activates Use as source. The exact next CHAT handoff revision settles,
   Console receives the original source ID and local conversation metadata,
   and the registry contains exactly one new Default membership.
3. Console keeps the same active tab and complete session-ID list. Both its
   canonical composer text and stored draft match the original; no messages
   appear. The source title and Un-stage action paint above the draft.
4. Returning to the retained reader preserves the original link receipt. The
   unrelated foreign source blocks the aggregate workspace handoff, but using
   this eligible conversation again succeeds. The payload and membership are
   unchanged; no duplicate session or link appears.
5. Enter on Un-stage removes the pending evidence while preserving the draft.
   Return to the retained reader, then Enter on its focused Undo link removes
   the membership this journey created. The original foreign link remains.

The runner checks eight conversation-source handoffs and four Undo actions.
After Undo in each cell, it additionally sends two synthetic launches through
`app.open_console_for_live_work`: first with no resident evidence, then replacing
the first. Both exact CONSOLE_LIVE_WORK revisions settle on the same reused
Console, and their titles paint above the unchanged draft. These eight additional
launches qualify the public staging boundary, not a RAG retrieval or source
authority check. There are eight Un-stage actions in total. Returning to the retained reader matters: selecting a new reader can
replace the transient receipt. Native Undo occurs in the same active workspace;
cross-workspace receipt ownership and pre-existing links have separate targeted
test coverage. The native fixture does not claim those additional journeys.

The no-send evidence is unchanged in-memory messages, unchanged durable source
records and the reviewed staging-only production paths. There is no independent
network/provider-call counter in this native runner. No Send action is used.
Focus is assigned programmatically before actual Enter activation; complete Tab
traversal and mouse operation are outside this native qualification.

## Persistence and lifecycle

Complete [before](source-before.json) and [after](source-after.json) snapshots
match without ignored fields. After normal Ctrl+Q exit, the independent
[checker](verify_profile.py) opens SQLite read-only and compares every original
field, normalizing only equivalent datetime encodings.
[Persistence](persistence.json) confirms five original conversations, ten original
messages and exactly the original foreign membership after all four new links
are undone. All ten private databases pass integrity checks. Default-profile
config, UI-state and runtime-policy hashes are unchanged; logs contain no
ERROR/CRITICAL lines, traceback headers or faulthandler output.

[Lifecycle](lifecycle.json) records PID 36432, normal `app.run()` return, shell
exit 0 and independent process-absence verification. Only the owned terminal
was closed afterward. Draft preservation is qualified within this process,
not across restart. Both native runs passed: [history](native-history.json) records
the earlier CHAT-only qualification; run-002 extends it after the warm-path fix.

## Capture review

All sixteen final SVGs were rendered and inspected together after the production
repair. The twelve original captures were inspected before that repair; the
second batch is justified by the newly found live-work defect.
[Inspection](inspection.json) records the observations; [hashes](capture-hashes.json)
distinguish raw captures from stored SVGs with trailing source whitespace removed.

| Theme and size | Link-on-use | Staged source and draft | Retained Undo receipt | Warm replacement |
| --- | --- | --- | --- | --- |
| Dark 170×48 | [Reader](textual-dark-170-reader.svg) | [Console](textual-dark-170-staged.svg) | [Linked](textual-dark-170-linked.svg) | [Live work](textual-dark-170-warm.svg) |
| Dark 80×24 | [Reader](textual-dark-80-reader.svg) | [Console](textual-dark-80-staged.svg) | [Linked](textual-dark-80-linked.svg) | [Live work](textual-dark-80-warm.svg) |
| Light 170×48 | [Reader](textual-light-170-reader.svg) | [Console](textual-light-170-staged.svg) | [Linked](textual-light-170-linked.svg) | [Live work](textual-light-170-warm.svg) |
| Light 80×24 | [Reader](textual-light-80-reader.svg) | [Console](textual-light-80-staged.svg) | [Linked](textual-light-80-linked.svg) | [Live work](textual-light-80-warm.svg) |

The focused source action, inline link promise, source strip and Undo link are
readable in both themes/sizes. At 80×24 the draft wraps to three visible lines,
the link receipt wraps to two lines, and the reader transcript extends below
the viewport while header actions are focused. The wide inspector truncates
some secondary text; the complete staged identity remains in the source strip.
No actionable source-control paint defect was observed in these captures.

## Automated verification

The [baseline](baseline-targeted-tests.txt) returned 113 passes and 12 failures.
Initial repairs corrected the shared live-work test fixtures:

- Disable the unrelated seven-second splash in isolated test config. Full-app
  handoff checks previously timed out after six seconds, before Console mounted.
- Construct the bare session controller before assigning the store, whose
  setter now binds that controller's view hooks.
- Load real app styles for the Watchlists inspector, reveal the scrolled action,
  verify its hit target and retain the actual click/routing assertion.
- Assert the current Environment/Tasks/Fleet ordering and bounded live-work
  viewport ancestry through both card replacements. No ordering or replacement
  behavior assertion is removed.

After the fixture repairs, [124 tests passed and one failed](fixture-repaired-targeted-tests.txt).
That remaining failure was a real warm-screen regression. Strengthened tests
wait for actual Library navigation and exact handoff settlement; both first and
replacement launches [failed with the revision still pending](warm-live-work-red.txt).
The repaired consumer claims, assigns and acknowledges before refreshing through
one shared staging path. Its cancellable resume timer applies work on ordinary
warm returns. Two fault-injection cases confirm repaint failure cannot requeue
already-owned evidence. Exact saved-chat resume ordering is unchanged.

The [expanded selection](expanded-targeted-tests.txt) returned 141 passes and
8 failures in neighboring suspend and roleplay-resume tests. With only the
production screen restored to the original commit, [the neighbor baseline](neighbor-baseline-tests.txt)
reproduced the same eight named failures (13 passes). They remain unqualified;
this repair does not claim those broader lifecycle paths pass.

Imports in the touched test module are sorted. [Targeted results](targeted-tests.txt),
[verification](verification.json) and [independent review](review.json) record
**130 passing checks in 155.21s**, exit 0. The changed tests and QA helpers pass
Ruff and formatting.
The production screen retains the same 212 pre-existing Ruff diagnostics with
zero additions; changed ranges are formatted. Saved test logs only normalize
trailing whitespace. No full suite was run.

## Reproduce and remaining scope

Prepare a fresh no-secret private profile with `data/`, `data/db/`, `config/`,
catalog refresh disabled and all configured database/data paths beneath it.
The runner reuses the fail-closed [profile guard](../2026-09-16-ingest-lifecycle/native_check.py)
and refuses an existing evidence directory. Retain a pre-launch
`default-before.json` mapping the three default-profile files to SHA-256 hashes
(or null if absent). In an owned tmux terminal at 170×48, set `PYTHONPATH` to
this checkout and run:

```text
.venv/bin/python Docs/superpowers/qa/2026-09-16-conversation-source/native_check.py PROFILE SOCKET SESSION
```

Record shell exit and independently verify process absence, then run:

```text
.venv/bin/python Docs/superpowers/qa/2026-09-16-conversation-source/verify_profile.py PROFILE
```

Raw private evidence remains at `/private/tmp/tldw-32706-run-002` on this host.
Terminal capability probing is primed before app input ownership, so ordinary
unprimed startup timing is outside scope. Other unqualified native paths include
empty-draft prefill, remote/archived sources, full Tab traversal, workspace
switching during Undo, provider generation and transcript content delivery.

Next bounded repair: TASK-2376, actual media/conversation excerpts in staged
evidence. The remaining Library destinations and broader feature review follow;
this task does not mark all Conversations behavior complete.
