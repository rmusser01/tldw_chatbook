# Library conversation Archive and Restore — TASK-32703

The bounded Archive/Restore journey passes against `a4efad7056`: Cancel,
Archive, version-aware Undo, View archived and Restore only preserve the intended
saved conversation and its messages. A fresh native process finds the archived
record and restores that same identity without changing its current Console
session or draft. No production change was needed.

ADR required: no. This verifies the existing [archive contract](../../../../backlog/decisions/147-conversation-archive-and-exact-resume.md),
[adaptive reader](../../../../backlog/decisions/086-library-adaptive-reader-shell.md)
and [design language](../../../../backlog/decisions/150-design-token-system-and-design-language.md).
The Impeccable audit was adapted to Textual controls, compositor paint and terminal
geometry; this is not a complete accessibility or application review.

## Native evidence

[Exercise](exercise-result.json) and [restart](restart-result.json) are separate
native `TldwCli`/`LinuxDriver` processes using real shipped services and SQLite in
one validated private profile. The [runner](native_check.py) verifies its imported
application path and exclusive profile lock. No recovery service is mocked.

Two disposable conversations, Alpha and Beta, each contain one User and one
Assistant message. The [fixture manifest](fixtures.json) records their exact
IDs and bodies. Only Alpha is mutated; Beta remains active at version 1.

The first process exercises textual-dark and textual-light at both 170×48 and
80×24. In each matrix cell it:

1. Opens Alpha from the Active results for `Archive audit`.
2. Cancels Archive with Escape and verifies the conversation version is unchanged.
3. Confirms Archive, verifies one version increment and the matching recovery
   receipt version, then activates Undo to recover Alpha.
4. Archives Alpha again and uses View archived to find it with the same query.
5. Confirms Restore only, undoes that restoration, and restores Alpha again.

Every confirmed direct mutation checks the exact Alpha version and receipt;
every settled result checks its count and retained query. The final first-process
action leaves Alpha archived at version 26. A read-only database check after exit
records that durable state. A fresh process finds Alpha through Archived and
Restore only returns it to Active at version 27, with the same four total saved
messages. These are the final `run-002` results; an earlier successful exercise
was superseded by this run with explicit Cancel and receipt-version assertions.

Both processes compare the Console active session ID, session ID list and current
draft before and after the Library journey. The draft is empty in this fixture;
this does not establish preservation of nonempty drafts or attachments, or
persistence of a temporary Console session across processes.

[Lifecycle](lifecycle.json) records normal Ctrl+Q shutdown, `app.run()` return,
shell exit 0 and independent process-absence checks for PIDs 62342 and 62973.
Only the owned terminal session was closed afterward. [Persistence](persistence.json)
records ten healthy private databases, exact conversation/message ownership and
content hashes, unchanged default-profile configuration/state hashes, zero
ERROR/CRITICAL lines, zero traceback headers and an empty faulthandler log.
No external model request was sent.

## Painted controls

All ten SVG captures were rendered and visually inspected in one review round.
[Inspection](inspection.json) records each source/stored hash and observation.
Only whitespace-only SVG source lines were normalized for storage.

| State | Wide, dark | Compact, dark | Wide, light | Compact, light |
| --- | --- | --- | --- | --- |
| Archive confirmation | [170×48](confirm-textual-dark-170.svg) | [80×24](confirm-textual-dark-80.svg) | [170×48](confirm-textual-light-170.svg) | [80×24](confirm-textual-light-80.svg) |
| Archive receipt | [170×48](receipt-textual-dark-170.svg) | [80×24](receipt-textual-dark-80.svg) | [170×48](receipt-textual-light-170.svg) | [80×24](receipt-textual-light-80.svg) |

The dialogs show their complete recoverability copy and Cancel/Archive controls.
Receipts show Undo with visible focus, View archived and the retained query. The
wide reader keeps both Alpha message bodies readable after archiving. At 80×24,
Items is explicitly reopened for the receipt and collapsed for reader actions;
the complete transcript is not simultaneously visible in the receipt capture.
Buttons are programmatically focused, checked for compositor-painted labels,
then activated with Enter. This qualifies the mounted controls and actions,
not complete Tab traversal or automatic focus placement.

The dark 170×48 restart captures show [Alpha in Archived](archived-after-restart.svg)
and [the same Alpha restored to Active](restored-after-restart.svg), with both
message bodies readable. No actionable defect was found in these captures.

## Automated checks and limits

[Targeted tests](targeted-tests.txt): **56 passed in 21.64s**, exit 0:

```text
.venv/bin/python -m pytest Tests/UI/test_library_conversation_recovery_flow.py Tests/UI/test_library_archive_review_v5.py Tests/DB/test_conversation_archive.py Tests/Chat/test_conversation_archive_actions.py Tests/Chat/test_conversation_archive_scope_service.py -q --tb=short --show-capture=no
```

The existing tests cover stale receipt refusal, partial batch results and service
boundaries in addition to the successful native mutations. Stale-receipt and
bulk failure cases were not recreated in the native app. The mounted SQLite
recovery test stubs Resume/source routes, so its pass does not qualify exact
Resume. Export, Restore and resume, optional/remote providers and arbitrary
transcript shapes remain outside this task. No full suite was run.

[Verification](verification.json) pins the tested baseline and runner hash;
[static checks](static.json) and [independent review](review.json) record closeout
checks. The review ledger and earlier semaphore-status notes were reconciled:
TASK-32700 now has successful real Import/restart evidence. No new general lesson
was uncovered beyond the existing private-profile and native-lifecycle rules.

## Reproduce

Prepare a new disposable profile with `data/`, `data/db/` and `config/`, all
configured database/data paths beneath it, no secrets and catalog refresh
disabled. The helper reuses the fail-closed [profile guard](../2026-09-16-ingest-lifecycle/native_check.py)
before application imports. Set `PYTHONPATH` to this checkout's absolute root and
use its `.venv/bin/python` in an owned tmux session sized 170×48. Run in order as
separate processes, recording each shell exit and confirming its PID is absent
before proceeding:

```text
.venv/bin/python Docs/superpowers/qa/2026-09-16-conversation-archive/native_check.py PROFILE SOCKET SESSION exercise
.venv/bin/python Docs/superpowers/qa/2026-09-16-conversation-archive/native_check.py PROFILE SOCKET SESSION restart
```

Each phase refuses an existing output directory. The exercise seeds its own two
conversations; restart requires that same profile and fixture manifest. The
terminal capability probe is primed before stdin ownership, so ordinary unprimed
startup timing is not qualified. Preserve independent lifecycle, database, log
and default-profile checks alongside the runner's own results. Raw logs, private
databases and rendered PNGs remain in ignored audit scratch.
