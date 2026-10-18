# PR3162 reviewed merge qualification

Local qualification checkpoint for TASK35585. Hosted review, required CI and the guarded merge are recorded separately after publication; this document does not claim that the PR has merged.

## Scope and custody

The tracked-clean remote head1f9c62c1d7b5296f686e9958c32482e6273ed03a rebased cleanly as four commits onto dev5eb9a37b34a1b0af3e91ad360c0b997497ee7894, producing the immutable initial review head7fc2abc21082a0b9cdf7ddc0b94709cffce5ed8a. The original verified feature branch remains ea5c335462b33e0b8c3c19011b60feca9360576c. The follow-up patch is identified by the accompanying source hashes and commit history. Existing historical QA/capture payloads remain unchanged.

ADR required: no. These are routine bug, compatibility, test-lifecycle and ratchet repairs within ADR166/097/228; UI remains governed by ADR150/161. No schema version, dependency, approval or disclosure boundary is added.

## Repairs

- CI's task33164 collision: the older archive follow-up task keeps its identity. The younger Dreams task becomes TASK35586 with creation/add provenance and its mutable source/test comments updated. The reservation scan covered876refs and41worktrees, finding maximum35585. Original archive evidence is not renumbered.
- Settings Undo moves verbatim into Settings_Modules/workspace_archive.py; extraction execution AST was checked identical. Revision/name CAS guards, receipt ownership, cancellation shielding and UI checks remain. The Settings ratchet drops35818→35743; the CSS common-type-selector ratchet drops274→272. Archive labels use their own class, actions their existing widget type and the scope Select its existing ID. Generated CSS comes from the normal builder.
- Fresh review: newly opened conversation acquisition that expires after registration clamps its own checkpoint close to zero remaining wait. A pinned live-reader regression protects the overall deadline without touching borrowed timeouts/transactions.
- Search and exact-target restore normalize workspace archive timestamps identically; exact-second, Z, offset, invalid and missing legacy timestamps no longer create false staleness. Revisions, names, ownership and actual lifecycle changes still compare.
- Undated rows follow proven archive dates and use normalized last-modified then identity within the existing keyset tuple. Archive-date validation remains timezone-strict. The legacy UTC-naive allowance applies only to last-modified fallback, which native pages and the tool description explicitly label.
- Library/Preview navigation runs in app-owned workers and rechecks the requesting source before executing. Refresh/restore retain the existing serialized message path. Queued result mounts recheck actual source answerability before/after mount; same-generation retirement clears its registration so returning to the source can remount.
- Derived gates: completion notifications render literally; archive chronology's index plan is pinned without sqlite_stat1/ANALYZE; the reviewed diagnostic inventory changes one warning digest, with its pre-existing tool-name/exception inputs and sink unchanged. The architecture text now correctly says real archive transitions advance local versions without emitting archive-only sync payloads/events.
- Current dev Collections schema is5; the stale4 expectation is updated and its owned memory database closed. The native card module retains its actual collection-bound private config. Test scrolling waits for layout and captures only the transcript viewport; imported JSON inertness waits for real coalesced UI publication. Assertions are retained or strengthened.

## Evidence

Raw argv, exit codes, times and logs live under /private/tmp/archive-recovery-pr3162-merge-20261018. validation.json records their hashes and distinguishes diagnostic/superseded failures from closing runs. Counts overlap and must not be added as unique coverage.

- Exact initial CI regressions:4failed/1passed; the five selected checks then passed. Both ratchet limits were subsequently tightened, with the Settings slack check included.
- New deadline/ordering regressions:3RED→3GREEN. Workspace timestamp regressions:4RED/1passed→all5GREEN within the operations run. The real leave-dialog regression reached NoActiveWorker on both routes, then both routes passed through real view dispatch. Same-generation queued-mount cache regression:1RED/1passed→bothGREEN.
- Broad storage/runtime/replay qualification:369passed and one stale Collections expectation. Closing full Collections/card cohort:52passed.
- Broad mounted cohort:92passed, one compact scroll/click failure and the long five-cycle cohort deliberately excluded. Later focused runs exposed render/capture timing and coalesced sync assumptions; those failures and the deferred-only negative control are preserved. Closing native card/queued/imported cohort:10passed.
- Final interaction cohort: 14passed,29deselected in245.46seconds on the final frozen source. It retains both viewport sizes, button/agent entry, date/unknown labels, keyboard activation, original-history Preview/Open, imported text inertness and exact confirmation.
- Final focused fixes/governance cohort:49passed with five isolated old-card config setup errors; those five are closed by the52passed and10passed cohorts. The49 include operations, cold archive index plan, duplicate-ID and tightened ratchets, and token governance.
- All11 relevant derived source checkers pass after repairs. The final worker graph separately passes after narrowing workers to navigation.23 changed Python files pass syntax and have zero introduced per-file Ruff code/message multiplicities against the exact rebased head; inherited lint debt is not represented as clean.

The fresh independent review is [recorded here](2026-10-18-archive-recovery-reviewed-merge/review.md); source/evidence hashes are [in this directory](2026-10-18-archive-recovery-reviewed-merge/). No findings may remain open before publication/arming.

## Limits and merge gates

No full suite, new live/model requests, installed-build qualification or long five-cycle descriptor repeat was run. The approved localhost request cap remains30/30 consumed. Prior qualified descriptor/owned-handle plateau evidence remains historical; current targeted logs report no descriptor-growth warning. Two previously demonstrated unrelated voice controls and inherited lint debt remain outside this change.

MERGE_QUEUE was freshlyon; the PR was still draft with auto-mergeoff at the last prepublication read. Publish with an exact lease, mark ready, obtain Qodo review on the exact current head and resolve every review thread before arming. The queue owns subsequent rebases and required pull_request CI; never manually merge/update an armed branch or dispatch the required gate. Actual mergedAt/mergeCommit must be checked before reporting the user's merge request complete.
