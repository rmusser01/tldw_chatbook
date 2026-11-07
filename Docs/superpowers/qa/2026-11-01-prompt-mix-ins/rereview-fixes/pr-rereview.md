# PR3346 scoped re-review

Base: `963fe12e87ae4d2fb415bff3814943ca448cb0e0`.
Reviewed product/test head: `1a58f0af2cbc5129f9c8f01595d67c330a7defda`.
Scope: original I1/M1 fixes and new breakage in their diff; the additional SQL/index/recovery, worker/DOM, markup and diagnostic-inventory corrections. No broad feature repeat, subagents, source/index/HEAD mutation, Qodo, or full suite.

## Original finding verdicts

- **I1 — closed.** Checkbox changes now propose an intent and immediately retain the last authoritative value. Silent restoration prevents a second mutation. Dirty refusals, async failures, successful actions followed by a newly dirty editor, and subsequent clicks reconcile values/count without replacing the editor or silently advancing its stale intent. The added mounted tests exercise the reported sequence and these relevant completion branches.
- **M1 — original failure closed, fix not yet approvable because of R1 below.** Capture is provisional; hook/provider/native-commit refusal no longer publishes usage facts. Native acceptance, successful queue admission and receipt acceptance publish used revisions without re-reading sources or re-adding removed selections. Tests include the original refusal, real SQLite abort/retry, fresh durable/temporary acceptance, unsupported current preferences, queue drain after removal, and real durable flush/reopen plus a held-save/new-selection race. However, discarding accepted placement/order observations changes later injection order.

## New findings in the fix diff

### Important R1 — Accepted sends forget placement observations and reorder later sends

**Location:** `tldw_chatbook/Chat/console_prompt_mixin_service.py:452–457` and `:483–492`; the new accepted-usage paragraph in ADR-286 also encodes this behavior.

Capture reconciles the snapshot's order using current definitions, but `publish_accepted_usage` persists only `last_used_version`. The next send therefore starts from the old placement/order even after the preceding send accepted and used a placement change. The approved ordering rule says a newly observed placement moves to the end of its new group, including observations made during send preparation. This must preserve the order of changes observed by prior accepted sends, not just changes observed by opening the Workbench.

**Native reproduction:** Start with selected A then B, both Before. Change B's Library definition to After and send `first`; the accepted payload correctly contains `body A / first / body B`. Without opening the Workbench, change A to After and send `second`. Expected current envelope: `second\n\nbody B\n\nbody A` because B was already observed in After and A now joins that group's end. Actual: `second\n\nbody A\n\nbody B`. A narrow diagnostic using the existing native `trace_harness`, real controller sends, current `select` setup and normal source records reproduced the wrong provider payload; assertion failed in 1.83 seconds. No product function was replaced.

**Impact:** Modifier order changes incorrectly across accepted sends, changing instruction precedence and contradicting the specified append-to-group behavior. Reopening the Workbench after these sends also reconciles from the stale layout rather than the accepted sequence. The new one-placement test does not expose this two-step case.

**Smallest complete fix:** Retain accepted placement/order observations through an appropriate existing owner, and base subsequent captures on them, while leaving refused preparation provisional. Preserve accepted-request source witnesses, exact queued/Retry snapshots, current user selection edits, and stale-layout intent refusal. Do not “fix” this by weakening the approved ordering requirement or making every future send depend on a Workbench visit. Add successive accepted-placement observations (including durable reopen) as a regression. Align the ADR annotation with the final behavior.

### Important R2 — The new artifact worker can overwrite a newer dirty editor

**Location:** New worker dispatch at `tldw_chatbook/Widgets/Console/console_prompts_modal.py:2218–2225`; newly reachable unfenced continuations at `:1286–1296` (Recipe) and `:1315–1320` (ordinary structured Prompt).

Moving ArtifactSelected out of the event pump is necessary for nested confirmation, but `open_artifact` does not revalidate navigation/detail ownership after `_capture_manual_apply_target()`. While that await is pending, the now-responsive UI can switch modes and accept a new edit. When the old capture completes, its worker unconditionally replaces `_editor_state`, changes state, and mounts the old artifact editor. `exclusive=True` only cancels another artifact worker; it does not cancel this worker when the user enters Mix-ins or changes source.

**Mounted actual-event reproduction:** Open an ordinary schema-v2 Prompt by posting the real `ConsolePromptsBrowse.ArtifactSelected` event, with the existing manual-resolution callback held. While pending, click the real Mix-ins button, Inspect/Edit selected Style, and type `unsaved current work` in Instructions. Release the old manual-resolution await. The stale artifact worker changes the mode from Mix-ins to Edit and removes the newer dirty mix-in editor without its save/discard guard. The narrow mounted diagnostic failed with `stale completion replaced dirty editor with edit` in 3.67 seconds. It used the production event handler, normal Local source and actual controls; only the injected async callback seam was held.

**Impact:** A delayed ordinary Prompt open can discard newer unsaved mix-in work. The newly added confirmation/Cancel tests prove pump responsiveness but not ownership after navigation during the wait.

**Smallest complete fix:** Keep worker dispatch, but fence/cancel artifact-open ownership across navigation/source changes and revalidate after manual-target capture before publishing any editor/state. Apply the same ownership rule to both ordinary Prompt and Recipe continuations, including publication of a captured manual target. Add an actual-event held-callback test that navigates away, edits the new view, and verifies the old completion leaves it untouched; keep the confirmation responsiveness tests.

## CI correction review

- Four allowlist additions name the actual installed mix-in tables; no validation bypass.
- Three index pins inspect real SQLite plans with a 200-row operational corpus, no sqlite_stat1 and no INDEXED BY. The digest query matches the production lookup; a real messages DELETE exposes both FK SET NULL index searches. No phantom index was added.
- Recovery index selection now compares SQL token structure and the intended name prefix. For the fixed canonical schema inputs this retains the same actual statements while avoiding a false index declaration in the census.
- New post-await DOM accesses use bounded query results. The Library acknowledgement returns if its literal editor has disappeared rather than raising NoMatches. No census repin hides these accesses.
- Literal Rich Text for the Mix-ins count is appropriate. Diagnostic inventory changes correspond to reviewed constant mix-in notices and removal of keyword values/exception text; existing ordinary-prompt path candidates remain legacy-unreviewed. No additional issue found in these corrections.
- ArtifactSelected worker responsiveness is valid, subject to R2's required completion fencing.

## Evidence and limitations

Read the durable fix report, changed tests, final follow-up/held-toggle argv and exit records, diagnostic review, and updated verifier. Reported completed runs were not rerun merely to recreate their counts. The two extra diagnostics above address new concrete doubts and are additional assertions injected in memory into selected existing pytest fixture contexts; they are not claims that the unchanged selected original test bodies failed. Both used no cache provider, PYTHONDONTWRITEBYTECODE, and isolated private basetemp directories. The checkout remained clean.

The verifier correctly checks actual-terminal source hashes against their captured port commit, not this fix head. Fresh-clone/current-manifest checks qualify published byte provenance; they do not turn older captures into current fix behavior. No new external-provider/cache, actual-terminal whole journey, OS-font, or global FD qualification is claimed. Existing baseline failures, M4 retirement noise and launcher limitations remain outside this fix scope.

## Assessment

**Ready to merge? With fixes.**

The original false-off and refused-send-used-revision failures are corrected, and the CI corrections mostly preserve their intended boundaries. R1 changes injection order and R2 can discard newer dirty work; both are concrete regressions introduced by the fixes and must be addressed before approval. No Critical or additional Minor findings in this scoped pass.

Skipped full-suite and external/live-provider reruns. Remaining risks are incorrect successive placement order and stale artifact completion discarding newer edits.
