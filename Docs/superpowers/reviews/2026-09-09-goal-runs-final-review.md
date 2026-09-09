# Final whole-branch review record

Controller-retained findings and disposition from the independent reviewer. File/line references below identify the reviewed commit, before the final fix wave.

Reviewer: /root/goal_final_review. Range77bc58dc171c3dcd2178f4433d19a6ceb59b1e7b..2c63e578e1bcfc36c994014b7f78d4195933df0e. Verdict: Needs fixes. Complete21-commit package inspected in bounded passes, including all textual production/test/doc hunks and four QA renders. No repository mutations, suite reruns or model calls.

## Strengths

Durable launch identity, exact provisioning, FULL acceptance and atomic checkpoint settlement preserve ownership. Typed verifier evidence, latest-observation precedence and fresh artifact review address false-success risks. Distinct goal/fleet policy shares accounting, capacity, manual reserves and startup audit. Permanent _closed differs from resettable per-dispatch Stop; source inspection supports runtime reuse, while saved tests establish five controlled shutdown interleavings rather than every successive-goal scenario. Tests exercise actual native/controller/tool/SQLite, subprocess/restart/UI/request boundaries.

## Findings (operative reviewer text)

1. **[P2] Freeze or invalidate every setup field across awaited validation.** Widgets/Console/console_goal_setup_modal.py:274 snapshots objective, criteria, human-review choice and tool selections before awaiting configure(), but only Start is disabled. Afterward it checks only _binding_version and locks the current widgets while retaining the earlier _submitted request. Changes to other fields leave final review displaying different intent from the request Start executes. A mounted probe reproduced locked fields showing “New visible objective” and human review enabled while _submitted retained “Repair fixture” and human review disabled. Disable all relevant fields before validation, or invalidate validation on every relevant edit. Restore consistent controls after failure and render confirmation from the exact submitted request.

2. **[P2] Include the selected verifier invocation in the model-facing launch context.** Agents/goal_iteration.py:439 includes protocol, objective, criteria and checkpoint memory, but omits selected command, arguments and checked inputs, despite exact-invocation runtime authorization. The first user request in deterministic-cli.json:34 says only “Repair fixture”/“Validation exits zero”; test_goal_cli_verification.py:554 obtains verifier, scripts/check.py and project argument from fixture variables. The live fixture manually duplicates these details in its objective, masking the setup-to-handoff gap. Provide a bounded model-facing projection of the exact authorized invocation and target/input information, preserving execution-time checks. Test actual outgoing requests with selections absent from prose and without fixture-only command knowledge.

3. **[P2] Connect additional read-only source selections to usable scoped context or tools.** Widgets/Console/console_goal_setup_modal.py:257 serializes sources, but production source_bindings references stop at model and registry/freshness validation. There is no source-content/tool consumer. Console_chat_controller.py:5521 uses only request.binding for the local provider, and the handoff omits source selections. A sibling read-only source therefore exposes no content or authorized read route; its only runtime effect may be blocking admission if missing. Connect selections to an explicit bounded read-only context/tool path, keeping primary-root write confinement and existing permissions. Add real selected-source reads and refusal of an unselected sibling. If deliberately deferred, remove/reject selection and reconcile spec/guide.

4. **[P3] Restore tool-choice enabled state after rejected validation.** Carried Task5 minor: console_goal_setup_modal.py:206 and cleanup326. A binding discovery finishing while _busy disables choices; rejected validation restores only Start. Restore choices when discovery is complete and no request is submitted. Coordinate with finding1.

5. **[P3] Keep ADR-141 inside the index table.** backlog/decisions/README.md:85 has a blank separator before the new row, terminating the Markdown table. Remove the separator so ADR-141 remains an indexed table entry.

## Evidence and limits

Saved logs inspected: Task1 165/39; Task2 146/5/45; Task3 60/84 and107+corrected11; Task4 164/78; Task5 durable main/fix gates. Counts overlap. Architecture97passed/3failed/1skip; exact baseline proofs classify TTS checksum and two missing-label assertions as prerequisites, with the overall gate still red. Deterministic CLI proves actual7→0, authorized edit, unchanged verifier, twoincrements/fivecalls; not autonomous model competence. Both live trials failed quality with no tools, malformed reports and unchanged invalid artifact. Four images qualify modals80x24/160x44, not an operationalrail walkthrough. Investigated retention ambiguity is NOT a defect:4MiB is evidence pergoal,128MiB aggregate payload.

Only new behavioral probe: /private/tmp/native-goal-final-review-form-race.py. Exact command and substantive output are in /private/tmp/native-goal-final-review-form-race-result.txt, explicitly a transcription, not a rerun log. No model/tool calls. Configuration imported outside pytest isolation, including existing-config/bootstrap-directory ensures; artifact discloses that limitation.

Recommendation: one fix wave for allfive findings and focused regressions; clarify usable launch selections in the handoff plan. Plan/spec: Needs fixes. Core ownership/accounting/evidence/recovery substantially implemented, immutable setup and resource usability incomplete. Successful live-model qualification remains unmet. Code quality/readiness: With fixes; not ready to merge at reviewedHEAD. Integration still must reconcile preserved prerequisites and baseline diagnostics.

## Root disposition

Accept allfive findings. Implement selected-source usability under existing read-only authority; do not defer/remove the promised selection. Preserve all prior runtime, budget, evidence, retention, privacy and permission contracts. Root will clarify accepted ADR-141/spec/plan before the fix wave, reopen affected Backlog tasks and record new behavioral AC before code. No successful live-model claim is allowed; existing negative traces remain historical evidence.

Root finding5 implementation clarification: a focused MarkdownIt parse also shows an earlier pre-existing separator before ADR129 leaves ADR129–138 outside the table. The minimal formatting fix may remove both separators, leaving entry content unchanged, so ADR141 is actually a table row. This is a structural check of the reported index problem, not a second broad code review.

## Fix wave submitted for scoped re-review

Implementation525c61a944 addresses allfive findings. Canonical skill owners project invocation details into model-facing context, existing file tools read selected source roots with permission and filesystem identity checks, setup freezes all fields and restores clean controls after failure, and ADR index rows render inside the table. The affected gate reports279passed/1live skip/1proven unchanged missing-run_id baseline failure; overlapping geometry and cleanup followups each pass2tests. Static checks add no legacy diagnostics. Current deterministic CLI evidence derives invocation from the actual outgoing request and retains real7→0/twoincrements/fivecalls. Historical live failures remain unchanged. Independent scoped re-review is pending; this submission does not assert approval. See the [qualification report](2026-09-09-goal-runs-qualification.md) for durable evidence and baseline proofs.
