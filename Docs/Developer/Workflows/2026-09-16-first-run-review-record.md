# First-run implementation and review record

Status: implementation qualified; all planned reviews reconciled. Integration
has not been performed or authorized by this record.

This records the narrow session-bound delivery governed by
[ADR-138](../../../backlog/decisions/138-portable-workflow-definitions-and-local-execution.md)
and the [approved plan](../../superpowers/plans/2026-09-16-workflows-first-run.md).
Implementation base: `cf61cb68505fc62991a0488c964a78cb7ab31cbb`.
The [UAT record](2026-09-16-first-run-uat.md) separates deterministic integration,
actual llama.cpp requests, fresh-process restart, and compositor screenshots.

## Per-task gates

| Task | Reviewed head | Evidence at that gate | Independent disposition |
| --- | --- | --- | --- |
| Bounded llama.cpp | `b9fea1dd553f053ea179e26c17c3973befc69bcc` | 460 targeted tests | Approved after cancellation-retention/DNS fix |
| Fresh permissions | `57410bbb3e714b454c758780395ae051f29a3a58` | 227 targeted tests | Spec and quality approved |
| Local file / Note bridge | `78f252fd5788f452205bfd807c1b9d95f63dd13f` | 237 targeted tests, including 70 new cases | Spec and quality approved |
| In-memory session | `23f144879088278ba94578b1903f2a0e592e8eec` | Initial 681; admission fix 75 covering tests | Approved after missing-input validation fix |
| App / UI / quit | `b3e94fa245af0770cdf710a7f82033a8d1aee805` | 61 visual/UI covering; final 181 lifecycle covering tests | Visual SHIP and scoped lifecycle review approved |
| Joined qualification | `0b7b1e1db52dad69ef3c38f2e92f3f5176dee3c6` | Final 159 covering; 165 Notes; 74 logging; 8 passing live processes | Spec and quality approved; no Critical/Important findings |

These are overlapping targeted runs, not a summed unique-test count. Existing
dependency/deprecation warnings and retained legacy static debt are disclosed in
the UAT evidence. No full repository suite was authorized or run.

## Findings resolved during implementation and qualification

- Bounded HTTP cancellation now retains physical transport settlement despite
  repeated cancellation. All DNS answers must be loopback before selecting one
  numeric endpoint; no dispatch fallback was introduced.
- Admission checks all `inputs.*` references after materializing input values,
  before destination capture or effects.
- Quit no longer permanently disposes Console while workflow preparation can
  still fail. Renewed Stay or cancelled reconfirmation preserves the same usable
  authoring owners and allows a fresh run only after successful settlement.
- Repeated quit attempts get fresh retained persistence barriers; an obsolete
  preparation cannot re-fence work admitted after an abort. Cancelled runs and
  their old approvals/review identities never resume.
- Approval panels show the action and captured destination before inspectable
  payload details. Invalid setup fields receive specific correction messages
  without losing user input. The footer no longer contradicts an active run.
- Library presentation updates now wait for the nested controls to finish
  mounting; the existing completion hook applies the latest retained state.
- Deferred editor view restoration cannot replace a newer attached field's
  focus or scroll. The regression uses keyboard invalidation and preserves
  the visible-repair-field contract without a corrective test scroll.
- The privacy test's log capture restores its explicitly selected HTTP logger
  level, so prior app logging setup cannot disable the positive control.

Tests use actual owner identities and persistence/readback boundaries where
those boundaries are claimed. A stub-only Stay test had missed permanently
closed owners; that incident is recorded in the testing lessons.

The merged runs remain recorded as 1,552 passed / 2 failed / 1 skipped, then
1,555 passed / 1 failed / 1 skipped. Each observed failure was reproduced and
fixed; final affected coverage passed 159 tests with the opt-in live test skipped.
This is not an invented all-green merged invocation. An additional Skills-import
test-double failure reproduces against unchanged baseline source; it remains
deferred outside this Workflow change. Raw SVG padding and one captured failing
test-output padding line are attributed in the evidence; source/documentation
whitespace checks pass without rewriting original captures.

## Visual evidence

A separate generic reviewer performed the Task 5 visual review because the
named Impeccable reviewer role was unavailable. The controller later inspected
all 39 initial three-size live-app captures at 160×48, 110×36, and 60×20. The workflow controls,
edited review, exact accepted Note body, quit warning, and restart state were
visible. No new Workflow layout blocker was identified. Long details at the
narrow width use scrolling; existing Library presentation was not redesigned.

After the Library nested-mount fix, a new live 110×36 walkthrough and separate
restart passed with before/after effective-path checks. The controller inspected
its review, Open Note, and restart PNGs: accepted text, the single saved Note,
and no restored run. That makes 42 inspected PNGs, not a claim that every SVG
from every failed or passing attempt received a separate visual review.

PNGs render unchanged Textual SVGs in an isolated offline browser with installed
font fallback. They are driver/compositor evidence, not native PTY or terminal
font qualification. The initially unsuitable CairoSVG rendering was not used to
claim visual approval.

## Acceptance evidence

These map TASK-32691's nine criteria to inspected implementation and recorded
verification. The final branch finding is resolved and independently reviewed below.

| Criterion | Evidence |
| --- | --- |
| 1. Saved revision and real five-step flow | Joined real-control/HTTP/Notes integration and four passing actual llama.cpp walkthroughs in the UAT record |
| 2. Navigation survives; restart does not resume | Mounted navigation/review tests and four distinct-process, same-profile restarts retaining revision/Notes with zero replay |
| 3. Captured destinations and fresh authority | Permission and session permission tests cover missing/corrupt authority, revocation, Off, exact approval identity and review rejection/expiration; local effects verify the captured destination |
| 4. Bounded keyless llama.cpp only | Bounded-client tests cover numeric loopback, deadline, bytes, no redirect/proxy/retry/fallback and retained physical cancellation; live manifests show the selected actual model |
| 5. One Note attempt and commit-wins cancellation | Real file-backed transaction barriers, duplicate/readback/content/client checks, late commit and worker cleanup tests; existing policy and no Sync dispatch |
| 6. No duplicate effects; ordered quit | Session duplicate Start/Accept tests and actual app lifecycle coverage for Stay, repeated quit, failed flush/drain, retained workers and same usable authoring owners |
| 7. No execution persistence or new SQLite infrastructure | Reviewed production diff and historical-row equality assertions; no migration/storage-owner/lock/PID/helper additions |
| 8. Targeted, UI, live and static qualification | Exact commands/results, deterministic failure fixes, performance figures and attributed static delta in the UAT packet; no full-suite or native-PTY claim |
| 9. Private payloads absent from ordinary diagnostics | Positive-control logging tests, actual Notes failure canaries, bounded-client errors and all retained ordinary live-log scans |

The tests are in `Tests/Workflows/`, `Tests/LLM_Calls/test_llamacpp_bounded.py`,
`Tests/UI/test_workflows_run.py`, `Tests/ProductionApp/test_workflows_session_lifecycle.py`,
and the focused permission/logging manifests linked from the UAT record. Counts
are overlapping evidence, not additive unique-test totals.

Controller handoff verification at `766e2ca43222703a9b07fe67cbcb9dcf3a75811f`
reran the final affected selection: **159 passed, 1 opt-in live skip, 1 existing
dependency warning in 205.32s**. The [command/output](artifacts/2026-09-16-first-run/controller-handoff-covering.txt)
records this additional run without new live endpoint requests. The three
new/clean Task 6 integration/editor paths pass Ruff and formatter checks;
source/documentation whitespace and all 34 checked local documentation links
pass. This does not supersede the historical broad-run failures or claim an
all-green repository suite.

## Rulings I made

Chronological controller decisions and their costs:

1. One implementer at a time, independently reviewed. Cost: less parallel throughput.
2. Inherit the parent model because no model override was requested. Cost: no automatic lower-tier cost saving.
3. Targeted tests only, per repository/user instructions. Cost: unrelated regressions are not exhaustively checked.
4. Prefer IPv4 after validating every localhost DNS answer is loopback. Cost: IPv6-only servers must use their explicit numeric IPv6 endpoint; no fallback.
5. Remove private Note titles from two existing error logs. Cost: less diagnostic detail; no DB behavior change.
6. Add a payload-free `LocalNoteCleanupError` for escaping cleanup failures. Cost: conservative nonacceptance even if cleanup finished before raising; existing swallowed SQLite errors unchanged.
7. Add read-only run bindings and private review-instructions projection. Cost: two small public interfaces; no second owner or durable history.
8. Abort the temporary fence when pre-close draft flushing is cancelled. Cost: explicit cancellation cleanup and regression coverage; accepted run cancellation remains irreversible.
9. Add the captured actual Notes DB path to existing protected paths. Cost: narrow session integration; no duplicate capture or raw DB probe.
10. Use a separate generic visual reviewer because the named Impeccable role was unavailable. Cost: an additional read-only review seat without automatic specialized-role injection.
11. Separate reversible persistence preparation from final resource destruction in existing workflow owners. Only successfully drained sessions may accept a new run after aborted quit; cancelled attempts never resume. Cost: small owner lifecycle interfaces, and a resource-close failure after committed exit follows existing teardown policy rather than returning to a fully usable app.
12. Fix the existing Library nested-mount readiness boundary reached by Open Note after deterministic reproduction. Cost: a small shared-Library lifecycle change requiring focused recompose coverage; no framework or storage change.
13. Guard the existing editor's deferred view restoration against a newer field selection after reproducing the exact off-screen invalid-field failure. Cost: a small authoring interaction change requiring normal restoration and invalid-field visibility coverage; no focus framework or delays.
14. Review generated evidence through manifests and exact outputs alongside a complete source/documentation diff; retain the full unfiltered diff. Cost: no line-by-line generated XML/log review, so reviewers must verify evidence claims and integrity explicitly.
15. Make Open Note's existing captured-route check nonblocking on the app loop while retaining blocking worker semantics and all ownership checks. Cost: a busy route requires user retry and a small optional guard mode; no new owner, lock, database or polling framework.

## Final gate

All six task reviews are approved. Task 6 independently verified 52 SVG hashes,
42 PNG hashes, 20 ordinary-log hashes, retained configuration hashes, live-process
counts, and static/whitespace attribution. Its inherited warning and unrelated
Skills-fixture findings remain deferred for final review. Cross-task guarantees
are covered by the earlier task gates and passed to the whole-branch review;
native-terminal and early configuration-snapshot limits remain disclosed.

The [single whole-branch review](artifacts/2026-09-16-first-run/final-branch-review.md)
at `766e2ca43222703a9b07fe67cbcb9dcf3a75811f`
found one Important/P1 defect: after an earlier saved Note, Open Note can wait
on the existing Notes-owner lock while a later Note worker holds it and waits
for the app loop's pre-write callback. The circular wait can block navigation,
Cancel and Quit. This is an implementation defect, not a plan change; the
initial branch verdict is **not ready to merge**. No other introduced findings
were supported. The review independently reconciled the earlier deferred items
as disclosed baseline debt or qualification limits.

The single final fix wave is committed as `cdc7c57909a76cafc451757fb99f9aaa2f8719d4`
and `bcc8290dd2d883893c781a83a3e5cd0f13f27eb5`. It adds a nonblocking option to
the existing guard while preserving blocking worker behavior and fresh captured
identity checks. Busy navigation returns explicit manual-retry feedback; successful
navigation clears that transient message. No new owner, lock, worker or database
infrastructure was introduced.

The [implementation report](artifacts/2026-09-16-first-run/final-review-fix/implementation-report.md)
records deterministic RED/GREEN around the real owner mutex. Two-Note tests cover
queued events, heartbeat, Quit/Stay, cancellation/physical settlement, exact contents,
stale identities and seven destination-change cases. RED intercepts the unsafe
blocking acquire safely; it does not claim a deliberately frozen process. Initial
covering checks passed 229 tests and separate ProductionApp lifecycle passed 12;
the later two-line status correction passed nine focused cases. No new static debt.

Controller [final-head verification](artifacts/2026-09-16-first-run/controller-final-covering.txt)
at `bcc8290dd2d883893c781a83a3e5cd0f13f27eb5` passed **11 tests, 1 opt-in live skip,
1 existing warning in 38.16s**: nine new UI cases plus two actual-app joined
HTTP/Notes cases. These use an owned HTTP peer, not new llama.cpp requests.
The earlier live qualification and its limits remain unchanged.

The [single scoped re-review](artifacts/2026-09-16-first-run/final-review-fix/scoped-review.md)
verdict is **P1 addressed; no new Critical, Important or Minor findings**.
All planned reviews are reconciled and all nine acceptance criteria are supported.
TASK-32691 is closed through the Backlog CLI. Source remains at the reviewed head;
subsequent closure changes are documentation/evidence only. No push, PR or merge
has been performed, and those integration actions still require the user's choice.

Cleanup: this plan's 2.5 GB of disposable profiles, temporary packages and ledger
was moved out of `.superpowers/sdd/` to the recoverable local archive
`/private/tmp/chatbook-workflow-32691-archive.hboWVC/2026-09-16-workflows-first-run`.
Committed evidence above remains in the repository. No worktree, branch, sibling
plan, TASK-32601 edit or `.uat-workflows-9NUT5t/` content was removed. The temporary
archive is not durable product storage and may be cleared by operating-system
temporary-file cleanup.
