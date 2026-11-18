# Final whole-branch review record

Final status: **all five findings addressed; scoped re-review approved** at `052fd4b95f`, after implementation `525c61a944`. Earlier findings below are retained as review history. Integration and qualification limitations remain explicit.

Controller-retained findings and disposition from the independent reviewer, with spacing and formatting clarified for the durable record. References in the initial findings identify the reviewed commit before the final fix wave; references in the final disposition identify the approved fix.

Reviewer: `/root/goal_final_review`. Initial range: `77bc58dc171c3dcd2178f4433d19a6ceb59b1e7b..2c63e578e1bcfc36c994014b7f78d4195933df0e`. Initial verdict: Needs fixes. The complete 21-commit package was inspected in bounded passes, including all textual production, test and documentation hunks and four QA renders. No repository mutations, suite reruns or model calls.

## Strengths

Durable launch identity, exact provisioning, FULL acceptance and atomic checkpoint settlement preserve ownership. Typed verifier evidence, latest-observation precedence and fresh artifact review address false-success risks. Distinct goal/fleet policy shares accounting, capacity, manual reserves and startup audit. Permanent `_closed` differs from resettable per-dispatch Stop; source inspection supports runtime reuse, while saved tests establish five controlled shutdown interleavings rather than every successive-goal scenario. Tests exercise actual native/controller/tool/SQLite, subprocess/restart/UI/request boundaries.

## Initial findings

1. **[P2] Freeze or invalidate every setup field across awaited validation.** Widgets/Console/console_goal_setup_modal.py:274 snapshots objective, criteria, human-review choice and tool selections before awaiting configure(), but only Start is disabled. Afterward it checks only _binding_version and locks the current widgets while retaining the earlier _submitted request. Changes to other fields leave final review displaying different intent from the request Start executes. A mounted probe reproduced locked fields showing “New visible objective” and human review enabled while _submitted retained “Repair fixture” and human review disabled. Disable all relevant fields before validation, or invalidate validation on every relevant edit. Restore consistent controls after failure and render confirmation from the exact submitted request.

2. **[P2] Include the selected verifier invocation in the model-facing launch context.** Agents/goal_iteration.py:439 includes protocol, objective, criteria and checkpoint memory, but omits selected command, arguments and checked inputs, despite exact-invocation runtime authorization. The first user request in deterministic-cli.json:34 says only “Repair fixture”/“Validation exits zero”; test_goal_cli_verification.py:554 obtains verifier, scripts/check.py and project argument from fixture variables. The live fixture manually duplicates these details in its objective, masking the setup-to-handoff gap. Provide a bounded model-facing projection of the exact authorized invocation and target/input information, preserving execution-time checks. Test actual outgoing requests with selections absent from prose and without fixture-only command knowledge.

3. **[P2] Connect additional read-only source selections to usable scoped context or tools.** Widgets/Console/console_goal_setup_modal.py:257 serializes sources, but production source_bindings references stop at model and registry/freshness validation. There is no source-content/tool consumer. Console_chat_controller.py:5521 uses only request.binding for the local provider, and the handoff omits source selections. A sibling read-only source therefore exposes no content or authorized read route; its only runtime effect may be blocking admission if missing. Connect selections to an explicit bounded read-only context/tool path, keeping primary-root write confinement and existing permissions. Add real selected-source reads and refusal of an unselected sibling. If deliberately deferred, remove/reject selection and reconcile spec/guide.

4. **[P3] Restore tool-choice enabled state after rejected validation.** Carried Task 5 minor: console_goal_setup_modal.py:206 and cleanup at line 326. A binding discovery finishing while `_busy` disables choices; rejected validation restores only Start. Restore choices when discovery is complete and no request is submitted. Coordinate with finding 1.

5. **[P3] Keep ADR-141 inside the index table.** backlog/decisions/README.md:85 has a blank separator before the new row, terminating the Markdown table. Remove the separator so ADR-141 remains an indexed table entry.

## Evidence and limits

Saved logs inspected: Task 1 gates of 165 and 39; Task 2 gates of 146, 5 and 45; Task 3 gates of 60, 84 and 107 plus 11 corrected checks; Task 4 gates of 164 and 78; Task 5 durable main/fix gates. Counts overlap. The architecture gate reported 97 passed, 3 failed and 1 skipped. Exact baseline proofs classify the TTS checksum and two missing-label assertions as prerequisites, with the overall gate still red. Deterministic CLI evidence proves actual exits 7→0, an authorized edit, an unchanged verifier, two increments and five calls; it does not establish autonomous model competence. Both live trials failed quality with no tools, malformed reports and an unchanged invalid artifact. Four images qualify modals at 80×24 and 160×44, not an operational rail walkthrough. The investigated retention ambiguity is not a defect: 4 MiB is evidence per goal, while 128 MiB is aggregate payload.

Only new behavioral probe: `/private/tmp/native-goal-final-review-form-race.py`. The exact command and substantive output are in `/private/tmp/native-goal-final-review-form-race-result.txt`, explicitly a transcription, not a rerun log. No model/tool calls. Configuration imported outside pytest isolation, including existing-config/bootstrap-directory ensures; the artifact discloses that limitation.

Initial recommendation: one fix wave for all five findings and focused regressions; clarify usable launch selections in the handoff plan. Initial plan/spec verdict: Needs fixes. Core ownership, accounting, evidence and recovery were substantially implemented, while immutable setup and resource usability were incomplete. Successful live-model qualification remained unmet. Initial code quality/readiness: With fixes; not ready to merge at the reviewed HEAD. Integration must still reconcile preserved prerequisites and baseline diagnostics.

## Root disposition

Root accepted all five findings and selected-source usability under existing read-only authority, preserving the promised selection and prior runtime, budget, evidence, retention, privacy and permission contracts. Root clarified accepted ADR-141, the spec and plan before the fix wave, reopened affected Backlog tasks and recorded new behavioral acceptance criteria before code. No successful live-model claim is allowed; existing negative traces remain historical evidence.

Root's finding 5 clarification: a focused MarkdownIt parse also showed an earlier pre-existing separator before ADR-129 left ADR-129–138 outside the table. Root authorized removing both separators, leaving entry content unchanged, so ADR-141 is actually a table row. This was a structural check of the reported index problem, not a second broad code review.

## Fix wave submitted for scoped re-review

Implementation `525c61a944` addressed all five findings. Canonical skill owners project invocation details into model-facing context, existing file tools read selected source roots with permission and filesystem identity checks, setup freezes all fields and restores clean controls after failure, and ADR index rows render inside the table. The affected gate reported 279 passed, 1 live skip and 1 proven unchanged baseline failure for a missing `run_id`; overlapping geometry and cleanup followups each passed 2 tests. Static checks added no legacy diagnostics. Current deterministic CLI evidence derives invocation from the actual outgoing request and retains real exits 7→0 across two increments and five calls. Historical live failures remain unchanged. This submission preceded the independent approval below. See the [qualification report](2026-09-09-goal-runs-qualification.md) for durable evidence and baseline proofs.

## Final scoped re-review: approved

Reviewer `/root/goal_final_review` inspected the complete three-commit `2c63e578e1..052fd4b95f` package, clarified contracts, report and saved red/green/static/trace evidence. No mutations, regenerated diffs, repeated tests/probes or model calls.

1. All-field setup: ADDRESSED (console_goal_setup_modal.py:219; mounted test_console_goal_setup.py:487), with synchronous locking, consistent restoration and actual persisted-request verification.
2. Selected verifier invocation: ADDRESSED (local_skills_service.py:2299, console_goal_runs.py:954, goal_iteration.py:444). Canonical trusted-owner projection reaches initial/later native requests and fails explicitly above the ceiling. The actual provider fixture derives commands from outgoing JSON.
3. Selected sources: ADDRESSED (console_chat_controller.py:5549, local_tool_provider.py:379, console_goal_runs.py:80). Existing fs_read/fs_list permission routes preserve primary-only writes, ordinary defaults and instruction isolation; registry plus filesystem identity revalidation and real native/provider refusal tests cover the reviewed boundary.
4. Rejection cleanup: ADDRESSED (console_goal_setup_modal.py:205,339; test_console_goal_setup.py:587). Both rejection and stale-binding invalidation restore enabled current choices without stale mutators.
5. ADR index: ADDRESSED (backlog/decisions/README.md:83); both necessary separators removed and entry contents unchanged.

No new Critical, Important or Minor fix breakage. No new outside observations. All five findings addressed. Specification compliance: Approved for the clarified fix scope. Code quality/readiness: Approved for this fix wave; root may finish bookkeeping and handoff.

Evidence verified: 279 passed, 1 live skip and 1 established baseline failure for a missing `run_id`; 20 strengthened setup/source passes and 2 geometry/2 cleanup followups overlap. Scoped static and differential lint pass. The affected gate remains globally red, as do the separately documented baseline diagnostics. The current trace retains exact invocations, real exits 7→0 across two increments and five calls, an unchanged verifier hash, the valid file/diff and an unchanged sentinel. Historical live-model qualification remains unsuccessful and geometry remains modal-only. Prerequisite reconciliation remains required before integration.
