# Character active display batch implementation plan

> For agentic workers: use superpowers:executing-plans inline, with one fresh whole-branch review at the end.

**Goal:** Reduce the measured first-send Character refresh work without changing freshness or resource custody.
**Architecture:** Forward the original private display-owner closure through the current run-active scope check into the existing stock finite batch. Direct actions and custom no-argument readers keep their existing dispatch.
**Tech Stack:** Python3.12, existing asyncio/Textual/SQLite, no dependency.
**Spec:** backlog/tasks/task-34406.8 - Reuse-finite-Character-display-batch-during-active-Console-sends.md
**ADR required:** no. **ADR path:** N/A; existing ADR126 Character custody and ADR230 preparation ownership apply. No new runtime, storage, authority or public interface.

## Global constraints

- Preserve the original one-second run check and invalidation; two-second display memo remains display-only.
- No new cache, pool, timer, TTL, permission reuse, deadline or raised module budget.
- All actual native callbacks and borrowed/owned connection retirement remain original.
- Root alone writes/runs. Targeted tests only; live runs use the frozen disposable admission and guarded credential path.
- Instrumented spans locate work; only unprofiled matched first-send packets support a time comparison.

## Review focus

- Source customization between initial stock qualification and optional keyword dispatch, including callbacks and awaits.
- Queued display becomes run-active while waiting on the existing presentation lock.
- Stale or closing screen/session/database/generation cannot publish or establish display freshness.
- Repeated cancellation drains the original native callback; custom and memory routes retain affinity and errors.
- Changed-owner/error/force_fresh observations preserve the existing run check and memo policies.

### Task 1: Reuse the existing guarded batch on active displays

**Files:** character_context.py; test_character_refresh_finite_batch.py; test_character_presentation_cadence.py; relevant ADR126/Console architecture; task record. No exports, fixtures or configuration changes.
**Interfaces:** refresh_if_scope_changed(force=False) keeps existing zero-argument behavior; one private optional _presentation_is_current closure is issued only by the defining stock display method. refresh(_presentation_is_current=...) and the existing batch remain the consumers.

- [x] Step1: Parameterize test_changed_character_display_batches_only_inner_original_reads over idle/active/queued-to-active. Keep original direct4 callbacks, three pairs/one recent, identical state and physical retirement before the desired display2 assertion. Expect idle PASS and active/queued FAIL4!=2 at unchanged production.
- [x] Step2: Move only the existing owner closure declaration so both original active branches can issue it. Capture its key only on the selected route. Recheck stock source after run-active/owner callbacks. Forward through the private optional scope-check argument, checking stock before each refresh dispatch. Custom methods still receive no keywords. Do not change _mark_checked semantics.
- [x] Step3: Verify the causal3 controls; parameterize actual owner/native/cancellation controls for active route and add dynamic no-argument customization controls at active callback, owner capture and final scope await. Preserve all original positive and refusal assertions.
- [x] Step4: Run the complete connected Character cadence, finite batch, entry controls, current read lifetime, actual display callers and run scope checks via the admitted targeted launcher. Compare touched lint/format against exact merged production. Update governing docs and task notes with actual evidence.
- [x] Step5: Commit, bind unprofiled matched first-send measurement to exact baseline/candidate source and identical existing settings/schema/trace/cleanup. Retain every failure and slower result; no p95/native paint/Windows claim. One fresh final whole-branch reviewer; settle every finding and limitation before publishing follow-up work.

Ruling before fixture edit: connected201P3F150.92s; all3mounted failures stop at raw_source_selection_changed before Character behavior. Exact merged production reproduces3setup errors with the same refusal, guards0/0/source maps equal. Add only existing per-node bootstrap_profile markers to those3tests so the existing admitted private source remains selected. Their complete bodies/assertions stay unchanged; no global fixture/admission bypass. Include active variants of existing entry controls and an explicit force_fresh run-cache control before final connected verification. Root interrupted unintended connected-r2 launched after inspection asserted failure rather than error; no source restore had occurred and no qualification is claimed.

Completed source qualification: exact12c803b210P177.86s,5-file lint0, independent whole-branch/frozen-chain0/0/0. Balanced8fresh-profile measurements show medians1427->1048.5ms post-save and2.708766->2.120106s Enter-toHTTP,3of4pairs faster, allfunctional/source/schema/settings/normal-quit/observed cleanup checks satisfied. Full sample/limitations report Docs/QA/2026-10-17-character-active-first-send.md. Release CI/merge remains external; task remains In Progress until that gate. Parent target remains open.
