# Fresh senior code review — PR3162

Date: 2026-10-18. Reviewer: the single fresh read-only reviewer requested for TASK35585, applying the requesting-code-review reviewer contract and Ponytail full.

## Reviewed identity and scope

- Base: `5eb9a37b34a1b0af3e91ad360c0b997497ee7894`.
- Initial immutable rebased feature head: `7fc2abc21082a0b9cdf7ddc0b94709cffce5ed8a`.
- Final reviewed working-source snapshot: `25b4892df0ec4ad861d16b8f2144b130b0f31d9a6b6380318a738a664550b066` across 99 changed production/test/script source files, including generated CSS. The source was frozen by the implementer before this report. The final commit must reproduce this snapshot before publication.
- Snapshot algorithm: take the sorted union of `git diff --name-only -z BASE` and untracked paths; keep files under `tldw_chatbook/`, `Tests/`, or `scripts/` ending in `.py`, `.tcss`, or `.tsv`; SHA256 each file (or use `<deleted>`), then SHA256 the UTF-8 concatenation of `path + NUL + digest + newline`. After commit, use `git diff --name-only BASE COMMIT` as the path source; its result must match.

Reviewed the whole archive-recovery feature in passes through discovery/privacy, stable references and pagination, native authority and fenced dispatch, exact confirmation and cross-store writes, cancellation/shutdown custody, capture/thinking ownership, mounted controls, and connected storage/activation callers. This includes the rebased LibraryCollections SQLiteStoreKernel integration and the maintenance delta; it is not merely a review of the final cleanup diff. The governing requirements are ADR166, ADR097, ADR228, the approved archive-recovery specification and the reviewed-merge plan. Historical QA/capture payloads were excluded and left immutable.

The reviewer made no application, test, index, HEAD or branch edits. Only this report and isolated synthetic test artifacts were written; no subagent, live/model request, external message or full-suite run was made.

## Strengths

- Recovery capabilities authenticate exact app-issued session/run ownership separately from the general Library provider. Reserved-name collision protection, interactive-primary eligibility, persona limits, Library policy and the existing destination/kill gates remain connected to actual dispatch. Supported fenced transport reuses the existing runtime call correlation without changing endpoint capability facts.
- Both predicates and snippet projections exclude system, reasoning and tool payloads. Results use bounded literal SQL search, stable immutable references, complete keyset continuation and explicit unknown chronology. Manual refresh never becomes model-observed authority merely by resembling a tool result.
- Conversation versions and workspace archive revisions/names protect their write transactions. Workspace-first recovery preserves actual partial outcomes and reobserves selected-chat ownership; no compensation silently rearchives or moves data. Exact one-shot confirmation remains mandatory, and Open retains the existing original-ID/history/draft handoff.
- The coordinator retains started work independently of tool waiters and views, including cancellation and shutdown. Finite connection retirement follows native identity and preserves unchanged borrowed handles/transactions. The Collections kernel MRO, thread-local cache and close seam remain compatible with that ownership helper.
- Capture fixes preserve current-user and saved-thinking provenance, retain old profile semantics, and narrowly qualify the supported new local response profile. The regressions inspect real SQLite persistence and production controller/parser/confirmation paths rather than substituting final outcomes.

## Issues

### Critical (Must Fix)

None found.

### Important (Should Fix) — all addressed

1. **Expired cold acquisition could checkpoint beyond the search deadline.** `tldw_chatbook/DB/ChaChaNotes_DB.py:3252`. If the budget expires after connection registration, the internal retirement path previously used the restored 15-second busy timeout before the outer finite-operation finalizer could clamp it. With a pinned older reader, the reviewer reproduced 1.610 seconds for a 0.3-second budget and a forced 0.35-second registration delay. The smallest repair now sets zero busy timeout only for that newly opened expired handle before its own close. The unchanged reproduction completed in 0.458 seconds after the fix; borrowed ownership is untouched. A real pinned-WAL regression now covers the path.

2. **Equivalent archive timestamps could make an unchanged target permanently stale.** `tldw_chatbook/Chat/archive_recovery/operations.py:77`. Search normalized the workspace timestamp to fixed microseconds, while mutation preflight retained its raw string. A valid exact-second timestamp, `Z` or equivalent offset therefore failed target equality despite an identical archive revision; refreshing reproduced the same refusal. The reviewer confirmed the mismatch with real stores. The read-side snapshot now uses the same strict normalization while archived and None while active. No identity, revision, name or state comparison was removed. Real search-to-read-to-restore regressions cover exact-second, Z, offset, invalid and unknown observations.

3. **Library/Preview navigation could reach a leave dialog outside a Textual worker.** `tldw_chatbook/UI/Console_Modules/archive_recovery.py:396`. The original message-handler await chain could raise NoActiveWorker when canonical navigation needed `push_screen_wait`. Only the three navigation entry points now create app-owned workers and pass the requesting session; `open_library` rechecks that source at worker execution. Refresh and restoration retain their serialized message-pump behavior. Mounted regressions reach the real action branch and an actual leave dialog.

4. **Queued result mounting needed an actual-source fence and retryable cache retirement.** `tldw_chatbook/UI/Console_Modules/archive_recovery.py:202`. View-local session state alone did not prove that the source was still current while a queued mount ran. Entry and post-await checks now use actual answerability alongside existing session/generation checks. A refused mount clears its rendered marker only in the captured current generation, so returning to the source can remount without erasing a newer generation's registration. Two mounted before/during-source-loss controls prove refusal and remount. The separate imported-card test now waits for the documented coalesced sync completion before asserting inertness; its original row/control assertions remain.

### Minor (Nice to Have) — all addressed

- `tldw_chatbook/DB/conversation_archive_queries.py:96`: undated rows previously used identity alone despite the approved last-modified fallback requirement. They now use normalized legacy last-modified followed by identity, retaining the complete three-part cursor and strict archive-date interpretation. Filtered/unfiltered pagination tests cover microsecond/offset ties and invalid timestamps; tool/UI copy distinguishes fallback order from archive dates. The UI legend appears only when the page actually uses this fallback.
- `Docs/Architecture/database-layer.md:45`: the text incorrectly denied local version advancement for archive transitions. It now accurately distinguishes local optimistic versions from absence of archive-only sync events/payload fields.
- The CSS rekey retains the intended Static coverage for result, confirmation, imported fallback and receipt text. Dedicated button/scope selectors replace the broad subjects; the census ceiling was lowered rather than widened. Settings Undo was extracted with its receipt ownership, shielding, error handling and revision/name guards preserved. The younger duplicate Dreams task was renumbered with comment/reference-only changes to its consumers.

## Verification assessed

Evidence is under `/private/tmp/archive-recovery-pr3162-merge-20261018` and the implementer's durable qualification note.

- Reviewer independently executed the isolated real-SQLite deadline RED/GREEN probe and the valid-timestamp mismatch probe after the normal test bootstrap. No production profile or provider was used.
- `review-green.log`: 3 passed for the deadline and filtered/unfiltered undated ordering regressions.
- `profile-schema-green.log`: 52 passed for the current Collections/card qualification, superseding the stale schema-4 expectation and card-module config-source setup errors.
- `mounted-qualified.log`: 14 passed, 29 deselected, including compact/wide button and agent journeys, dates/unknown times, keyboard actions and Preview/Open behavior.
- `queued-remount-green.log`: 10 passed, including the two queued-source-loss/remount cases and the unchanged imported-card inertness flow.
- The earlier broad batches had 369 storage/runtime passes plus one stale Collections expectation, and 92 UI passes plus a compact observer failure. Those failures were retained and diagnosed; they are not represented as green full-batch runs. Compact capture now waits for layout, uses supported deferred scrolling and asserts/crops the actual transcript viewport. A deferred-scroll-only attempt failed; no claim is made that it alone fixed the observer. The reviewer independently completed the full compact button journey with post-layout capture, preserving its actual click, history and draft assertions.
- Local source review confirms unchanged authority/privacy/native ownership floors, fresh/upgraded schema successor declarations, restricted backup migration integration and explicit no-statistics archive-index qualification. The implementer is recording the final interaction rerun and refreshed static outcomes before publication; this report does not substitute an earlier static snapshot for that final record.

## Recommendations and assessment

**Ready to merge? Yes, for the reviewed local code; no unresolved code-review finding remains.** The fixes are narrow and covered at the failure boundaries, and the final source preserves the approved recovery, trace and storage contracts.

Actual publication/merge remains conditional on the final local qualification record, immutable-commit source-hash verification, current-head hosted Qodo review, zero unresolved review threads and the required pull_request CI. With the repository queue enabled, the root implementer must let the queue own rebasing and merge execution; this report is not evidence that those remote gates have already completed.

No full suite, new live inference, remote-only recovery or historical QA payload audit was performed. Those are the explicit scope limits of this review, not passing claims.

## Key final source SHA256 values

- `tldw_chatbook/DB/ChaChaNotes_DB.py`: `0715631b8b51f435be29c84d1b5cd73c394b4dd0d743c808f69e6a59e6e96c4a`
- `tldw_chatbook/DB/conversation_archive_queries.py`: `9557ebb334097e2d800d560fe87c7aecfb0c7423ebe0be1e5a8e80c17794a934`
- `tldw_chatbook/Chat/archive_recovery/operations.py`: `a0c661e3a03a98184ec68ebe7dd24b6b66f8b13dba41e3c87352f8f09c7bcb89`
- `tldw_chatbook/Chat/archive_recovery/runtime.py`: `57731c8cb663275a0574842f652af6f70165be8598b5cc156ea6b88e9c1ceab7`
- `tldw_chatbook/UI/Console_Modules/archive_recovery.py`: `93a4b34c4c996f4552b924a6121411c58f35de02a2a05111e8943ecb6e44b4a1`
- `tldw_chatbook/UI/Settings_Modules/workspace_archive.py`: `7d487723f00284c52dcfb3913d087eb2e48ef655e860a815536b67a90f9c48d5`
- `tldw_chatbook/Widgets/Chat_Widgets/archive_recovery_cards.py`: `73dd384559576c65571360c1617eb95e060d919ae4e34b4075b9b9c8d7a35edd`
- `Tests/UI/test_console_archive_recovery_cards.py`: `fe3c0d520aef3f500db806621262bdb834961800471d3c39a58e4d0937e81d99`
- `Tests/UI/test_console_archive_recovery_flow.py`: `12cef161dd12eb5dab124acf55070adc3ad1db2a5c4a6eea13084f29b8fa6026`
