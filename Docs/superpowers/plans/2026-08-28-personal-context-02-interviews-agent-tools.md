# Personal Context 02 — Interviews and Controlled Agent Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add bounded personal/workspace interviews, final diff review, durable agent proposals, and locally governed profile tools to the local Chatbook profile.

**Architecture:** A reusable coordinator owns a 20-turn state machine over fixed or model-backed question providers and encrypted expiring drafts. Approved interview diffs call the existing application service directly; agent learning goes through separate durable proposals or the narrowly evidenced direct-write path exposed by a run-scoped ToolProvider.

**Tech Stack:** Python 3.11+, Shared Profile Core, Textual 8.x, existing Chatbook LLM caller, ToolCatalogRegistry, SQLite, pytest, Hypothesis.

**Spec:** `Docs/superpowers/specs/2026-08-28-unified-personal-context-profile-design.md`

## ADR check

```text
ADR required: yes
ADR path: backlog/decisions/099-personal-context-profile-authority-sync-and-encryption.md
Reason: This plan defines durable proposal lifecycle, agent mutation authority,
evidence requirements, and setup/workspace interview UX.
```

## Global Constraints

- Complete Plan 01 first and pin its exact Shared Core release.
- Every provider/model-backed interview displays the provider and model before
  the first answer is entered.
- The fixed local questionnaire performs no network or model call.
- Every provider turn consumes one of the 20 question attempts, including an
  invalid compound-question response.
- The interview model receives no tools and cannot call the profile service.
- Raw Q&A remains encrypted, local, unsynchronized, unlogged, and expires after
  30 days.
- Final review is the only interview path that creates records.
- `profile_propose` never creates an active fact. `profile_update` is absent
  unless effective runtime authority is `direct_write`.
- Agent permissions remain runtime-local and are intersected with profile
  state, active scope, lifecycle, and per-record visibility.
- Agents never approve proposals, change privacy/sync controls, directly
  archive/delete, access user-only records, enumerate other workspaces, or
  purge the profile.

---

### Task 1: Implement the interview coordinator and encrypted draft lifecycle

**Files:**
- Create: `tldw_chatbook/Personal_Context/interview_coordinator.py`
- Create: `tldw_chatbook/Personal_Context/interview_provider.py`
- Create: `tldw_chatbook/Personal_Context/interview_draft_repository.py`
- Create: `tldw_chatbook/Personal_Context/interview_diff.py`
- Test: `Tests/Personal_Context/test_interview_coordinator.py`
- Test: `Tests/Personal_Context/test_interview_draft_repository.py`
- Test: `Tests/Personal_Context/test_interview_diff.py`

**Interfaces:**
- Consumes: Shared Core `InterviewPack`, `InterviewQuestion`, `InterviewTurn`,
  `InterviewProposalBatch`; Chatbook `PersonalContextService`.
- Produces:
  - `InterviewQuestionProvider.next_question(request: InterviewProviderRequest) -> InterviewQuestion`
  - `FixedQuestionProvider`
  - `ConfiguredModelQuestionProvider`
  - `ProfileInterviewCoordinator.start(kind, scope_id, mode) -> InterviewSession`
  - `answer(session_id, answer) -> InterviewProgress`
  - `finish(session_id) -> InterviewDiff`
  - `commit(session_id, selections, enable_runtime) -> InterviewCommitReceipt`
  - `discard(session_id) -> None`

- [ ] **Step 1: Write failing state-machine and privacy tests**

```python
def test_model_turns_stop_at_twenty_even_when_provider_returns_invalid_questions(coordinator, invalid_provider):
    session = coordinator.start(kind="personal", scope_id="global", mode="adaptive")
    for index in range(20):
        progress = coordinator.answer(session.session_id, f"answer {index}")
    assert progress.question_attempts == 20
    assert progress.can_ask_another is False
    assert invalid_provider.calls == 20


def test_fixed_mode_never_calls_configured_provider(coordinator, configured_provider):
    session = coordinator.start(kind="personal", scope_id="global", mode="fixed")
    coordinator.answer(session.session_id, "Call me Sam")
    assert configured_provider.calls == 0


def test_finish_builds_diff_without_writing_records(coordinator, service_spy):
    session = coordinator.start(kind="personal", scope_id="global", mode="fixed")
    coordinator.answer(session.session_id, "Prefer concise replies")
    diff = coordinator.finish(session.session_id)
    assert diff.additions
    assert service_spy.mutations == []
```

Add tests for one-question validation, skip, finish early, 30-day expiry,
save/resume, memory-only mode, provider/model pinning, user-only existing records
excluded from adaptive input, strict output validation, secret-material refusal,
re-interview keyed updates, possible private duplicates, atomic selected commit,
draft-key destruction, workspace interviews producing only workspace-scoped
goal/working-context/convention records, no workspace-to-global leakage, and no
raw answers in logs/database/WAL.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/Personal_Context/test_interview_coordinator.py Tests/Personal_Context/test_interview_draft_repository.py Tests/Personal_Context/test_interview_diff.py -v`

Expected: interview modules do not exist.

- [ ] **Step 3: Implement the provider boundary and coordinator**

```python
class InterviewQuestionProvider(Protocol):
    def next_question(self, request: InterviewProviderRequest) -> InterviewQuestion:
        """Return exactly one schema-valid question without tools."""


@dataclass(frozen=True, slots=True)
class InterviewProgress:
    session_id: str
    question: InterviewQuestion | None
    question_attempts: int
    answered_count: int
    can_ask_another: bool
    provider_label: str
    model_id: str | None


class ProfileInterviewCoordinator:
    MAX_QUESTION_ATTEMPTS = 20

    def answer(self, session_id: str, answer: str) -> InterviewProgress:
        draft = self._drafts.require_active(session_id, now=self._clock())
        self._drafts.append_answer(draft, self._validate_answer(answer))
        if draft.question_attempts >= self.MAX_QUESTION_ATTEMPTS:
            return self._finished_progress(draft)
        question = self._provider_for(draft.mode).next_question(self._request(draft))
        self._drafts.append_question_attempt(draft, question)
        return self._progress(draft, question if question.is_single_question else None)
```

The configured-model adapter calls the existing unified LLM seam with the
pinned provider/model, `tools=None`, non-streaming structured output, and the
Shared Core JSON Schema. It returns a typed provider failure; the coordinator
preserves the encrypted draft and offers retry/fixed fallback.

- [ ] **Step 4: Implement encrypted drafts and deterministic diffing**

Store draft envelopes through the Task 3 repository cipher with one draft DEK,
an explicit expiry, and no Sync outbox. Diff by structured key and exact record
ID. Never semantic-merge text. `commit()` calls one service transaction for the
selected mutations, then destroys the draft DEK.

- [ ] **Step 5: Run interview tests**

Run: `pytest Tests/Personal_Context/test_interview_coordinator.py Tests/Personal_Context/test_interview_draft_repository.py Tests/Personal_Context/test_interview_diff.py -v`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Personal_Context Tests/Personal_Context
git diff --cached --check
git commit -m "feat: add bounded profile interview coordinator"
```

---

### Task 2: Build the interview and final-review UI

**Files:**
- Create: `tldw_chatbook/UI/Screens/profile_interview_screen.py`
- Create: `tldw_chatbook/Widgets/Settings_Widgets/personal_context_review_modal.py`
- Modify: `tldw_chatbook/Widgets/Settings_Widgets/personal_context_panel.py`
- Create: `tldw_chatbook/css/components/_profile_interview.tcss`
- Rebuild: consolidated CSS via `python -m tldw_chatbook.css.build_css`
- Test: `Tests/UI/test_profile_interview_screen.py`
- Test: `Tests/UI/test_personal_context_review_modal.py`
- Test: `Tests/UI/test_settings_personal_context.py`

**Interfaces:**
- Consumes: Task 1 coordinator and Plan 01 Settings panel.
- Produces: `ProfileInterviewScreen`; `PersonalContextReviewModal`;
  `ProfileInterviewResult(status, committed_record_ids, runtime_enabled)`; a
  Settings action labelled `Run interview again` for personal or selected
  workspace scope.

- [ ] **Step 1: Write failing production-shaped UI tests**

```python
async def test_interview_discloses_provider_before_answer_input(pilot, adaptive_coordinator):
    screen = ProfileInterviewScreen(adaptive_coordinator, mode="adaptive")
    await pilot.app.push_screen(screen)
    assert "OpenAI / gpt-profile" in str(screen.query_one("#profile-interview-provider").render())
    assert screen.query_one("#profile-interview-answer").disabled is False


async def test_final_review_commits_only_checked_rows(pilot, fixed_coordinator):
    screen = ProfileInterviewScreen(fixed_coordinator, mode="fixed")
    await pilot.app.push_screen(screen)
    await screen.show_review()
    screen.query_one("#proposal-row-2").value = False
    await screen.action_commit_review()
    assert fixed_coordinator.last_commit.selected_change_ids == ("change-1",)
```

Add tests for skip, finish early, save, discard, provider failure fallback,
question count, compound-question refusal copy, record privacy controls,
possible-private-duplicate copy, Save and use with agents versus Save only,
safe Escape behavior, narrow layout, and no hidden destructive keybindings.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/UI/test_profile_interview_screen.py Tests/UI/test_personal_context_review_modal.py Tests/UI/test_settings_personal_context.py -v`

Expected: screen and review modal imports fail.

- [ ] **Step 3: Implement the UI as a coordinator client**

```python
class ProfileInterviewResult(NamedTuple):
    status: Literal["committed", "saved", "discarded", "cancelled"]
    committed_record_ids: tuple[str, ...]
    runtime_enabled: bool


class ProfileInterviewScreen(SafeModalDismissMixin, ModalScreen[ProfileInterviewResult]):
    BINDINGS = [
        Binding("escape", "request_close", "Close", show=True),
        Binding("f", "finish_early", "Finish", show=True),
    ]

    @work(thread=True, exclusive=True, group="profile-interview-next")
    def submit_answer(self, text: str) -> None:
        result = self._coordinator.answer(self._session_id, text)
        self.app.call_from_thread(self.apply_progress, result)
```

Keep service/LLM calls off the event loop. The UI displays structured diffs;
it never writes record tables or parses model JSON itself.

- [ ] **Step 4: Rebuild CSS and run UI tests**

Run:

```bash
python -m tldw_chatbook.css.build_css
pytest Tests/UI/test_profile_interview_screen.py \
  Tests/UI/test_personal_context_review_modal.py \
  Tests/UI/test_settings_personal_context.py Tests/UI/test_css_bundle_sync_guard.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/profile_interview_screen.py \
  tldw_chatbook/Widgets/Settings_Widgets/personal_context_review_modal.py \
  tldw_chatbook/Widgets/Settings_Widgets/personal_context_panel.py \
  tldw_chatbook/css Tests/UI/test_profile_interview_screen.py \
  Tests/UI/test_personal_context_review_modal.py Tests/UI/test_settings_personal_context.py
git diff --cached --check
git commit -m "feat: add profile interview review UI"
```

---

### Task 3: Chain optional interviews after setup and workspace creation

**Files:**
- Modify: `tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py`
- Modify: `tldw_chatbook/app.py`
- Modify: `tldw_chatbook/Widgets/workspace_create_modal.py`
- Modify: `tldw_chatbook/UI/Console_Modules/workspace.py`
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Create: `tldw_chatbook/Personal_Context/interview_launch.py`
- Test: `Tests/UI/test_first_run_profile_interview.py`
- Modify: `Tests/Workspaces/test_workspace_create_modal.py`
- Test: `Tests/Workspaces/test_workspace_profile_interview_handoff.py`

**Interfaces:**
- Consumes: Task 2 `ProfileInterviewScreen`.
- Produces:
  - first-run result key `offer_profile_interview: bool`
  - `WorkspaceCreateResult.offer_profile_interview: bool`
  - `launch_profile_interview_after_commit(app, request, continuation) -> None`

- [ ] **Step 1: Write failing completion-order tests**

```python
def test_first_run_marks_setup_complete_before_interview_launch(app_harness):
    app_harness.handle_wizard_result({
        "completed": True,
        "exit_route": None,
        "offer_profile_interview": True,
    })
    assert app_harness.persisted_first_run["setup_completed"] is True
    assert app_harness.pushed_screens[-1].__class__.__name__ == "ProfileInterviewScreen"


def test_workspace_exists_when_interview_is_cancelled(workspace_harness):
    result = workspace_harness.create(offer_profile_interview=True)
    workspace_harness.cancel_interview(result)
    assert workspace_harness.registry.get_workspace(result.workspace_id) is not None
```

Cover no-offer, cancelled interview, failed provider, rerun setup, explicit
first-run exit route continuation, all three WorkspaceCreateModal callers, and
exactly-once continuation after the interview result.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/UI/test_first_run_profile_interview.py Tests/Workspaces/test_workspace_create_modal.py Tests/Workspaces/test_workspace_profile_interview_handoff.py -v`

Expected: result types have no interview-offer field.

- [ ] **Step 3: Implement post-commit launch orchestration**

```python
@dataclass(frozen=True, slots=True)
class ProfileInterviewLaunchRequest:
    kind: Literal["personal", "workspace"]
    scope_id: str
    local_workspace_id: str | None


def launch_profile_interview_after_commit(app, request, continuation) -> None:
    def after_interview(_: ProfileInterviewResult) -> None:
        continuation()
    app.push_screen(ProfileInterviewScreen.for_request(request), after_interview)
```

The first-run callback persists setup completion through the existing wizard
finalize path, then launches the interview. Workspace callers receive a fully
created `workspace_id`, create/map its profile scope through the service, and
then launch. The modal itself does not mutate the registry a second time or own
caller navigation.

- [ ] **Step 4: Run ordering and existing wizard/workspace tests**

Run:

```bash
pytest Tests/UI/test_first_run_profile_interview.py \
  Tests/Workspaces/test_workspace_create_modal.py \
  Tests/Workspaces/test_workspace_profile_interview_handoff.py \
  Tests/Workspaces/test_console_workspace_create_handler.py -v
```

Expected: all tests pass; existing creation results remain compatible when the
new flag is false.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py tldw_chatbook/app.py \
  tldw_chatbook/Widgets/workspace_create_modal.py \
  tldw_chatbook/UI/Console_Modules/workspace.py \
  tldw_chatbook/UI/Screens/settings_screen.py \
  tldw_chatbook/UI/Screens/library_screen.py \
  tldw_chatbook/Personal_Context/interview_launch.py \
  Tests/UI/test_first_run_profile_interview.py \
  Tests/Workspaces/test_workspace_create_modal.py \
  Tests/Workspaces/test_workspace_profile_interview_handoff.py
git diff --cached --check
git commit -m "feat: offer profile interviews after setup"
```

---

### Task 4: Add durable proposal operations and profile ToolProvider

**Files:**
- Create: `tldw_chatbook/Personal_Context/proposal_service.py`
- Create: `tldw_chatbook/Agents/profile_tool_provider.py`
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `tldw_chatbook/Agents/tool_catalog.py`
- Test: `Tests/Personal_Context/test_proposal_service.py`
- Test: `Tests/Agents/test_profile_tool_provider.py`
- Test: `Tests/Agents/test_profile_tool_scope.py`

**Interfaces:**
- Consumes: Personal Context service/repository, ToolProvider protocol, trusted
  current user message ID/text, active canonical scope.
- Produces:
  - `ProfileProposalService.create/accept/reject/supersede/expire`
  - `ProfileToolProvider.list_catalog/load_schema/invoke`
  - `ProfileToolProvider.stamp_scope(run_id, scope: ProfileToolRunScope)`
  - tools `profile_search`, `profile_get`, `profile_propose`, `profile_update`, `profile_promote`

- [ ] **Step 1: Write failing authority and evidence tests**

```python
def test_default_propose_catalog_omits_update(provider, propose_scope):
    with provider.stamp_scope("run-1", propose_scope):
        names = {entry.name for entry in provider.list_catalog()}
    assert names == {"profile_search", "profile_get", "profile_propose", "profile_promote"}


def test_direct_update_requires_exact_current_user_span(provider, direct_scope):
    with provider.stamp_scope("run-1", direct_scope):
        result = provider.invoke("profile_update", {
            "record_id": "record-1",
            "expected_version_id": "version-1",
            "message_id": "other-message",
            "evidence_span": "I prefer concise replies",
            "value": "concise",
        })
    assert result.ok is False
    assert result.error == "review_required"


def test_proposal_is_not_visible_to_context(provider, propose_scope, context_service):
    with provider.stamp_scope("run-1", propose_scope):
        result = provider.invoke("profile_propose", {"kind": "preference", "subject": "tone", "value": "warm"})
    assert result.ok is True
    assert result.content["status"] == "proposal_created"
    assert "warm" not in context_service.build_snapshot_for_scope(propose_scope.scope_id).serialized_block
```

Add tests for read-only catalog, disabled/locked profile, user-only omission,
other-workspace refusal, private duplicate non-disclosure, five-per-turn and
25-per-session quotas, 200 unresolved ceiling, conflict freeze, promotion always
proposed, workspace-to-global promotion creating a new ID with `derived_from`,
secret refusal, a maximum 1000-character exact evidence span, stored evidence
hash/reference without raw span, and scope/catalog invalidation after authority
changes.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/Personal_Context/test_proposal_service.py Tests/Agents/test_profile_tool_provider.py Tests/Agents/test_profile_tool_scope.py -v`

Expected: proposal service and provider imports fail.

- [ ] **Step 3: Implement proposal lifecycle and receipts**

```python
class ProfileProposalService:
    def create(self, request: ProposalRequest, run_scope: ProfileToolRunScope) -> ProfileProposal:
        self._quota.require_capacity(run_scope.run_id, run_scope.session_id)
        self._policy.require_propose(run_scope.scope_id)
        proposal = request.to_pending_proposal(
            proposal_id=self._ids.new(),
            profile_id=run_scope.profile_id,
            scope_id=run_scope.scope_id,
            expires_at=self._clock() + timedelta(days=90),
        )
        return self._repository.commit_proposal(proposal)

    def accept(self, proposal_id: str, *, user_actor: UserActor) -> ProfileRecord:
        proposal = self._repository.require_pending_proposal(proposal_id)
        record = self._apply_as_user_mutation(proposal, user_actor)
        self._repository.resolve_proposal_and_shred(proposal_id, "accepted", record.version_id)
        return record
```

- [ ] **Step 4: Implement the run-scoped ToolProvider**

```python
@dataclass(frozen=True, slots=True)
class ProfileToolRunScope:
    run_id: str
    session_id: str
    profile_id: str
    scope_id: str
    authority: AgentAuthority
    current_user_message_id: str
    current_user_text: str


class ProfileToolProvider:
    SOURCE = "personal-context"

    @contextmanager
    def stamp_scope(self, run_id: str, scope: ProfileToolRunScope):
        token = self._scope.set((run_id, scope))
        try:
            yield
        finally:
            self._scope.reset(token)
```

Build schemas from Shared Core tool contracts. Verify evidence before mutation
with exact trusted message ID, trusted user authorship supplied by the Console
controller, and exact substring containment within the current user message.
Persist only SHA-256 evidence hash and message reference; never persist or log
the raw evidence span. Return bounded structured statuses, never exception text.

Wire `profile_provider` into the fresh per-run registry construction and combine
its scope with the existing run scopes. Do not register tools when the profile
is disabled, locked, purging, unmapped, or policy-ineligible.

- [ ] **Step 5: Run provider, catalog, and concurrency tests**

Run:

```bash
pytest Tests/Personal_Context/test_proposal_service.py \
  Tests/Agents/test_profile_tool_provider.py Tests/Agents/test_profile_tool_scope.py \
  Tests/Agents/test_tool_catalog.py Tests/Agents/test_tool_catalog_concurrency.py -v
```

Expected: all tests pass and existing catalogs are unchanged with no profile provider.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Personal_Context/proposal_service.py \
  tldw_chatbook/Agents/profile_tool_provider.py \
  tldw_chatbook/Chat/console_agent_bridge.py \
  tldw_chatbook/Chat/console_chat_controller.py \
  tldw_chatbook/Agents/tool_catalog.py \
  Tests/Personal_Context/test_proposal_service.py \
  Tests/Agents/test_profile_tool_provider.py Tests/Agents/test_profile_tool_scope.py
git diff --cached --check
git commit -m "feat: add governed personal context tools"
```

---

### Task 5: Complete proposal review, privacy evidence, and user documentation

**Files:**
- Modify: `tldw_chatbook/Widgets/Settings_Widgets/personal_context_panel.py`
- Modify: `tldw_chatbook/Widgets/Settings_Widgets/personal_context_review_modal.py`
- Test: `Tests/UI/test_personal_context_proposal_review.py`
- Test: `Tests/Personal_Context/test_profile_durable_owner_inventory.py`
- Modify: `Docs/User_Guide/console/chat-basics.md`
- Create: `Docs/User_Guide/settings/personal-context-profile.md`

**Interfaces:**
- Consumes: Task 4 proposal service.
- Produces: user review actions and documented privacy/authority behavior.

- [ ] **Step 1: Write failing review and durable-owner tests**

```python
async def test_agent_proposal_review_shows_source_and_cannot_change_hidden_record(pilot, proposal_fixture):
    modal = PersonalContextReviewModal.for_proposal(proposal_fixture)
    await pilot.app.push_screen(modal)
    assert "Agent proposal" in str(modal.query_one("#review-source").render())
    assert "possible private duplicate" in str(modal.query_one("#review-warning").render())
    assert proposal_fixture.private_duplicate_value not in pilot.app.export_screenshot()


def test_rejected_proposal_canary_absent_from_every_default_durable_owner(profile_harness):
    canary = "REJECTED-PROPOSAL-CANARY-41a927"
    proposal_id = profile_harness.propose(canary)
    profile_harness.reject(proposal_id)
    for owner_bytes in profile_harness.decoded_default_durable_owners():
        assert canary.encode() not in owner_bytes
```

Inventory the profile DB/WAL, outbox, logs, diagnostics, caches, exports,
crash-report input, and any run log reached by the real tool path.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/UI/test_personal_context_proposal_review.py Tests/Personal_Context/test_profile_durable_owner_inventory.py -v`

Expected: proposal review actions/inventory harness are missing.

- [ ] **Step 3: Implement review actions and content shredding**

Accept, edit-and-accept, reject, and expire through `ProfileProposalService`.
Acceptance creates the canonical record first and content-shreds the proposal in
the same profile-database transaction. Rejection/supersession/expiry destroys
the proposal DEK and retains only a content-free receipt.

- [ ] **Step 4: Run all Plan 02 targeted tests**

Run:

```bash
pytest Tests/Personal_Context/test_interview_coordinator.py \
  Tests/Personal_Context/test_interview_draft_repository.py \
  Tests/Personal_Context/test_interview_diff.py \
  Tests/Personal_Context/test_proposal_service.py \
  Tests/Personal_Context/test_profile_durable_owner_inventory.py \
  Tests/Agents/test_profile_tool_provider.py Tests/Agents/test_profile_tool_scope.py \
  Tests/UI/test_profile_interview_screen.py \
  Tests/UI/test_personal_context_review_modal.py \
  Tests/UI/test_personal_context_proposal_review.py \
  Tests/UI/test_first_run_profile_interview.py \
  Tests/Workspaces/test_workspace_profile_interview_handoff.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Perform scratch-profile user journeys**

Verify fixed first-run interview, adaptive-provider disclosure with a configured
non-paid test provider, workspace interview, agent proposal, user acceptance,
direct explicit correction, disabled profile, and private duplicate warning.
Fingerprint the real profile before/after.

- [ ] **Step 6: Document and commit**

Document question limits, provider disclosure, raw-answer deletion, proposal
review, direct-write evidence, agent permissions, private records, and re-running
the interview through `Run interview again` in Settings.

```bash
git add tldw_chatbook/Widgets/Settings_Widgets/personal_context_panel.py \
  tldw_chatbook/Widgets/Settings_Widgets/personal_context_review_modal.py \
  Tests/UI/test_personal_context_proposal_review.py \
  Tests/Personal_Context/test_profile_durable_owner_inventory.py \
  Docs/User_Guide/console/chat-basics.md \
  Docs/User_Guide/settings/personal-context-profile.md
git diff --cached --check
git commit -m "feat: complete personal context review workflow"
```

## Plan 02 completion gate

- Fixed and adaptive interviews are optional, bounded, resumable only when
  protected storage exists, and raw drafts are destroyed after use/expiry.
- First-run and workspace creation complete before an interview begins.
- Interview diffs cannot affect agents before user approval.
- Agent proposals are durable, non-injectable, reviewable, quota-bound, and
  crypto-shredded after resolution.
- Direct writes require current trusted explicit user evidence.
- Settings can re-run interviews and review proposals independently of Sync conflicts.
