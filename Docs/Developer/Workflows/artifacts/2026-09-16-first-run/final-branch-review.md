# Final independent whole-branch review

Date: 2026-09-16. Base: `cf61cb68505fc62991a0488c964a78cb7ab31cbb`. Reviewed HEAD: `766e2ca43222703a9b07fe67cbcb9dcf3a75811f`. Existing workflows-authoring-dev worktree only.

## Strengths

- The implementation preserves session-only sequential ownership, immutable saved revisions, detached input/destination bindings and explicit admission/resource bounds. No new execution schema, database owner, lock/PID file, recovery scanner or persistent runner is introduced.
- The bounded llama.cpp path validates all DNS answers before pinning numeric loopback, disables proxies/redirects/retries, bounds raw response bytes and retains physical transport/cleanup work through cancellation. Fresh strict permission checks use exact effect identity.
- File reads are bounded and identity-checked without permission mutation. Notes use the captured existing owner, one create ID per attempt, exact authorized reconciliation and commit-wins cancellation.
- Reversible quit preparation retains the same usable authoring owners until final teardown. Real-owner tests address earlier Stay/reconfirmation and repeated-preparation defects.
- Qualification preserves failed attempts and historical broad failures. Permanent regressions cover the Library nested-mount and editor deferred-focus defects; documented evidence does not invent an all-green broad run.

## Issues

### Critical (Must Fix)

None identified.

### Important (Should Fix)

#### 1. [P1] Open Note can deadlock the app loop against a later Note worker

Primary location: [workflows_screen.py:290](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/workflows-authoring-dev/tldw_chatbook/UI/Screens/workflows_screen.py:290).

**Confirmed source trace:**

1. Admission allows multiple sequential `notes` steps; a Note need not terminate the workflow ([session.py:299](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/workflows-authoring-dev/tldw_chatbook/Workflows/session.py:299)).
2. A successful first Note publishes its ID. The next-step update does not clear that ID, and Open Note remains enabled during subsequent steps ([session.py:1003](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/workflows-authoring-dev/tldw_chatbook/Workflows/session.py:1003), [session.py:1133](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/workflows-authoring-dev/tldw_chatbook/Workflows/session.py:1133), [run_controls.py:400](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/workflows-authoring-dev/tldw_chatbook/UI/Workflows_Modules/run_controls.py:400)).
3. The later Note worker holds the existing nonreentrant owner lock through `_bound_destination` and invokes `before_write` inside that context ([local_steps.py:169](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/workflows-authoring-dev/tldw_chatbook/Workflows/local_steps.py:169), [local_steps.py:196](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/workflows-authoring-dev/tldw_chatbook/Workflows/local_steps.py:196), [local_steps.py:206](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/workflows-authoring-dev/tldw_chatbook/Workflows/local_steps.py:206), [Notes_Library.py:230](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/workflows-authoring-dev/tldw_chatbook/Notes/Notes_Library.py:230)).
4. The callback schedules `_ready(...)` on the app loop and synchronously waits on its future using `.result()` ([session.py:970](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/workflows-authoring-dev/tldw_chatbook/Workflows/session.py:970), [session.py:901](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/workflows-authoring-dev/tldw_chatbook/Workflows/session.py:901)).
5. If an Open Note event is serviced after the worker acquires the lock but before the loop completes that callback, the synchronous UI handler attempts to acquire the same lock. It runs directly on the loop, and its run/Note identity checks accept this valid event ([workflows_screen.py:263](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/workflows-authoring-dev/tldw_chatbook/UI/Screens/workflows_screen.py:263), [workflows_screen.py:1252](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/workflows-authoring-dev/tldw_chatbook/UI/Screens/workflows_screen.py:1252)).

The loop waits for the worker's lock while the worker waits for the loop. This is a circular wait, not simply a slow database call. Attempt timeout, Cancel and Quit cannot execute on the blocked loop.

**User impact:** An admitted multi-Note workflow can freeze the entire application when opening an already-saved Note. Forced termination can lose session-only review/results. The first committed Note is not shown to be corrupted or rolled back.

**Correction:** Remove the potentially contended blocking-lock acquisition from the app loop. Use off-loop or nonblocking route validation with fresh run/destination checks before navigation, preserving fail-closed ownership and the worker's fresh pre-write authority check. Merely disabling a button without guarding queued events is insufficient.

**Minimal controller regression:** Under the existing pre-import disposable-profile harness, use two Note steps and the real Notes owner. Complete the first; hold the second worker after acquiring the owner lock but before its inner authority callback. Queue Open Note for the first confirmed ID, release the barrier from an independent thread and require continued app heartbeat, callback settlement and Cancel/Quit responsiveness. Use an external watchdog for RED because an asyncio timeout cannot rescue a blocked loop. Preserve destination-change rejection coverage.

**Verification status:** Confirmed by the source-level wait graph and a concrete feasible interleaving; not runtime-reproduced in this review. Existing single-Note held-commit coverage begins without a prior Note ID, so its disabled Open Note assertion does not cover this case. This is an implementation defect, not a plan defect.

### Minor (Nice to Have)

No other introduced actionable findings supported. Deferred limitations/debt are assessed below rather than counted as new defects.

## Recommendations

Fix Important finding 1 and add the isolated barrier regression before merging. Preserve existing owner, authority and physical-settlement guarantees; no new execution infrastructure or broad refactor is needed.

## Assessment

**Ready to merge? No.**

The delivery largely follows the approved architecture and has substantial targeted evidence, but the reachable UI/worker circular wait breaks liveness for an admitted sequential workflow.

**Explicit spec verdict:** Not fully compliant at reviewed HEAD. The first-run flow, bounded keyless transport, captured authority, in-memory ownership and no-new-storage constraints are supported. Navigation/cancellation/quit responsiveness remains deficient in the multi-Note path. The clarified reversible quit preparation and narrow Library/editor fixes are justified, plan-aligned changes.

## Checks and evidence actually reviewed

- Completed all source/docs passes: the full brief/rubric, approved spec, all 859 plan lines, relevant ADR-138 amendment, complete design-language constitution, all 11,985 lines/all 41 paths of the companion diff in bounded reads, and progress/deferred rulings, Task 6 report/review, UAT, guide and review record. Completed passes were not restarted.
- Verified companion diff bytes match all non-artifact blocks of the retained full package. SHA-256: `918525c824842e5a16ffe23ee0f7aeb59b1bef82289839ed1930486b1fcf6749`.
- Traced unchanged context only for named ownership/lock/navigation risks. Library's existing Note route selects database Notes; no separate remote-routing finding is supported.
- Read exact outputs: [final affected selection](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/workflows-authoring-dev/Docs/Developer/Workflows/artifacts/2026-09-16-first-run/editor-final-covering.txt) records 159 passed / 1 live-only skip / 1 warning in 207.79s; [controller handoff](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/workflows-authoring-dev/Docs/Developer/Workflows/artifacts/2026-09-16-first-run/controller-handoff-covering.txt) records the same selection at reviewed HEAD, 159 passed / 1 skip / 1 warning in 205.32s, exit 0 and no live endpoint POSTs. Also read [Notes 165](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/workflows-authoring-dev/Docs/Developer/Workflows/artifacts/2026-09-16-first-run/library-focused-notes.txt), [logging 74](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/workflows-authoring-dev/Docs/Developer/Workflows/artifacts/2026-09-16-first-run/logging-order-green.txt) and [lifecycle 12](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/workflows-authoring-dev/Docs/Developer/Workflows/artifacts/2026-09-16-first-run/production-lifecycle.txt). Counts overlap and are not additive.
- Read both historical merged outputs: [1552 passed / 2 failed / 1 skipped](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/workflows-authoring-dev/Docs/Developer/Workflows/artifacts/2026-09-16-first-run/merged-regressions.txt) and [1555 passed / 1 failed / 1 skipped](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/workflows-authoring-dev/Docs/Developer/Workflows/artifacts/2026-09-16-first-run/merged-regressions-final.txt). Read permanent editor and held-Library-mount RED/GREEN outputs and their code/tests. No final all-green broad run is claimed.
- Before the final scope clarification, independently checked all 52 SVG, 42 PNG and 20 ordinary-log hashes: all matched. This was integrity checking, not line-by-line generated XML/log review or a new comprehensive visual audit. Manifest cross-checks support 10 process IDs, eight passing processes (four walks/four restarts), 18 numeric-loopback POSTs and six created Note IDs, including failed qualifications.
- Parsed four retained qualified walk/restart before/after TOMLs with stdlib only; checked hashes, unchanged authority tables, keyless localhost intent, disabled catalog refresh and recorded effective-path equality. These checks establish retained evidence consistency, not current runtime state.
- Read static and whitespace attribution. Thirty-path totals: 1232 baseline / 1230 current lint diagnostics and 101 / 101 formatter spans. The three unmatched UP035 entries concern the same pre-existing typing symbols on the reordered Notes import. Exact commit outputs record lint success and three files already formatted.
- Final read-only HEAD check still returned the requested hash. Status showed the preserved TASK-32601/UAT changes plus controller-owned review-record and handoff evidence. This report is the reviewer's sole write.

## Independent deferred-item dispositions

- **Skills-import test-double failure:** Read the exact AttributeError output and independently verified both implicated files are byte-identical from implementation base to reviewed HEAD. Not a supported Workflow regression; keep disclosed, and do not call the entire Library selection green.
- **Historical merged failures:** Focused logging, Library-mount and editor correction evidence supports their recorded resolution, without changing the historical outcomes. It does not cover finding 1.
- **Dependency warning / static debt:** Existing environment/code debt, not a new blocker established by this diff. No dependency installation, suppression or broad formatting is warranted.
- **77 generated whitespace diagnostics:** Attributed retained capture/failing-output padding, not a reason to rewrite artifacts. Source/doc whitespace and 34 resolved documentation links are controller-reported checks, not rerun here.
- **Native PTY/fonts and earliest 160-column missing before-config hash:** Genuine disclosed qualification limits, not newly established implementation defects. Later retained snapshots do not retroactively fill the early gap.
- Earlier cancellation, admission, quit-owner, import-order, presentation and readiness rulings were assessed against current source rather than inherited as approval.

## Cannot verify / remaining verification

The review is substantively complete; no further source/evidence pass is needed for this verdict.

- **Unverified:** Runtime reproduction/frequency of finding 1 and the corrective regression result. The exact isolated controller probe is specified above; no app execution is authorized here.
- **Unverified:** Native terminal input/font fidelity, an all-green final broad selection, full-repository behavior and current endpoint state.
- Hash consistency is not independent replay of every live assertion. Every full generated model-response privacy canary was not reconstructed; no fresh comprehensive visual audit is claimed.
- No tests, app/profile imports, live requests, GUI sessions or subagents were run. Retained test/live results are not fresh execution claims.
- No source/index/HEAD/config/profile changes, PR, push or merge. The requesting-code-review rubric shaped the report; systematic-debugging guided the causal trace and verification-before-completion kept claims evidence-bounded, subject to the explicit no-rerun contract.
- Two report-patch attempts failed validation (duplicate target operations, then mismatched context); neither changed a file. These were patch-format errors, not permission/tool-access blockers. No outstanding tool blocker remains after saving this report.

## Historical pass checkpoint (superseded by final findings above)

Scope: `cf61cb68505fc62991a0488c964a78cb7ab31cbb..766e2ca43222703a9b07fe67cbcb9dcf3a75811f`, existing `workflows-authoring-dev` worktree only.

## Completed read passes

- Read final-review-brief.md in full (87 lines), the complete supplied code-reviewer rubric, approved spec, and all 859 plan lines.
- Read every line of final-source-review.diff (1–11985), covering all 41 source/test/documentation paths. Bounded contiguous passes covered 1–1251, 1252–7904, and 7905–11985 without truncating the diff reads.
- Read ADR-138 through its governing session-bound amendment, the complete design-language constitution, all progress/deferred-ruling entries, complete Task 6 report/review, UAT, user guide, and review record.
- Traced admission, detached bindings, strict permissions, retained model/file/Note work, review controls, reconciliation, app quit/Stay ordering, authoring barriers, shared Library/editor changes, and their new tests.
- Initial HEAD matched the requested head; original unrelated TASK-32601 and .uat-workflows-9NUT5t paths were present and untouched.

## In-progress checks (not a final verdict)

- Investigating concrete lock inversion: workflows_screen.py:290 takes NotesInteropService._db_lock on the app loop; local_steps.py:169/196/206 holds it while session.py's before-write callback synchronously waits on the app loop. Run admission permits multiple Note steps; the earlier note_id remains displayed and Open Note enabled while a later Note runs. Need finish source trace and describe an isolated barrier regression for the controller; no probe will run here.
- Generated-artifact integrity and exact-output audit underway. Read the full UAT/report provenance; manifests/outputs must independently substantiate totals and limitations.
- Controller supplied a new evidence-only handoff run: 159 passed / 1 opt-in live skip / 1 RequestsDependencyWarning, 205.32s, no live POST. Exact output still to inspect; production remains at the reviewed head. Working review-record changes are AC mapping only and are outside the pinned review diff.

## Contract

No tests, app/profile imports, network requests, subagents, source/documentation edits, staging, checkout or HEAD changes. This scratch report is the sole write. Final report will replace this checkpoint with rubric-formatted findings and an explicit merge/spec verdict.
