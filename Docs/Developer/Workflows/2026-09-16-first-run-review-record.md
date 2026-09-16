# First-run implementation and review record

Status: implementation qualification in progress; not a merge approval.

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

## Final gate

All six task reviews are approved. Task 6 independently verified 52 SVG hashes,
42 PNG hashes, 20 ordinary-log hashes, retained configuration hashes, live-process
counts, and static/whitespace attribution. Its inherited warning and unrelated
Skills-fixture findings remain deferred for final review. Cross-task guarantees
are covered by the earlier task gates and passed to the whole-branch review;
native-terminal and early configuration-snapshot limits remain disclosed.

The single whole-branch review is pending. TASK-32691 remains In Progress.
No push, PR, or merge is authorized by this record.
