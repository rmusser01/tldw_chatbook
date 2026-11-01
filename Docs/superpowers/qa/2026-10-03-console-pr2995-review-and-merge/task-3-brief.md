### Task 3: Fix Qodo input validation, Redirect layout and transcript-note findings

**Files:**
- Modify: tldw_chatbook/Chat/console_agent_bridge.py (validate_new_chat_arguments wrapper/docstring).
- Modify: tldw_chatbook/Utils/input_validation.py (shared strict Pydantic chat-creation model/validation boundary).
- Modify: tldw_chatbook/Widgets/Console/console_composer_bar.py (_actions_row_width duplicate active-run width only).
- Modify: tldw_chatbook/Chat/console_chat_controller.py (build_transcript_note speaker label only).
- Test: Tests/Agents/test_agent_chat_create_tools.py; Tests/Chat/test_console_chat_create_integration.py; Tests/Chat/test_console_note_span_actions.py; Tests/UI/test_console_composer_run_controls.py; focused shared-validator regression owner.
- Read: backlog/docs/design-language.md and lessons-testing-evidence.md.

**Interfaces:**
- Consumes: existing strict string/default/length/mode/destination chat argument contract; shared agent-model constants; origin-bearing USER messages; single Redirect width reservation.
- Produces: one shared Pydantic validator at both existing creation callers, Google Args/Returns/Raises docs, unchanged literal prompt bytes and trusted authority separation; stable active/rest layout; saved note provenance labels.

- [ ] Verify Qodo2/3/4/6 against actual source. Add RED controls for active/rest Redirect threshold with expected external geometry and agent/untrusted/human saved-note labels. Keep existing payload/default/type/limit controls.
- [ ] Delegate public creation validation to a shared strict Pydantic model using existing agent-model constants. Preserve exact defaults, title trim-after-length check, literal prompt/instructions, optional routing, unknown-authority-field discard and non-coercion. Preserve stable public ValueError categories; no raw ValidationError/payload diagnostic leakage. Add Google doc sections.
- [ ] Remove only duplicate active-run ten-cell budget. Existing Redirect reservation remains the single width source. Verify actual mounted run activation/deactivation without row/control shifts and room-fitting boundary; do not make the expected geometry depend on the buggy method.
- [ ] Saved transcript notes label agent_chat_start USER rows Agent handoff and untrusted USER rows Unverified handoff; ordinary human USER/ASSISTANT and span/provenance behavior stay unchanged.
- [ ] Run complete directly affected argument/integration/note/layout owners plus new shared-validator controls, private profiles and exact retained receipts. Diagnose actual failures against immutable BASE before expanding fixture repairs. No full suite/new markers/suppression/dependency installs.
- [ ] Run fatal Ruff, formatter ratchet for touched Python paths against BASE and source whitespace. Commit only owned source/tests; report exact SHA, RED/GREEN, warnings and self-review.
- [ ] Independent scoped spec/quality review; startup ratchets qualify shared-validation integration after runtime follow-up.

## External review and publication

- [x] Independent task review of Task1, then whole-branch review of the rebased PR. Package diffs before dispatch; reviewers do not repeat completed test runs.
- [x] Push with force-with-lease pinned to the verified old remote head; update the PR description and mark ready to enable reviews.
- [ ] Retrieve all PR issue/review comments and Qodo suggestions. Verify each actionable finding; dispatch concrete follow-up fixes with reproductions and covering tests. Reply and resolve addressed threads with commit/test evidence.
- [ ] Wait for current-head checks and Qodo completion. If future waiting is required, create a quiet thread heartbeat that continues review fixes and merge under the user's authorization.
- [ ] Fetch latest dev again; update/requalify if it advanced. Merge using normal repository gates and exact reviewed head, without admin bypass.
- [ ] Confirm merged state/SHA, close current Backlog criteria, archive this plan's evidence and remove only its disposable SDD directory after completion.
