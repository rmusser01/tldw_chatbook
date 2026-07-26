# Destination Handoff Completion Implementation Plan (TASK-646)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete revisioned ownership for Study, Artifact, and ACP handoffs; repair exact destination recovery; remove the dead Notes slot and every legacy raw pending field; then run the integrated installed-wheel and full-suite gates.

**Architecture:** Extend the TASK-645 single-slot owner with three domain channels plus independent Study scope/section channels. Study claims after mount so its explicit inputs override restored view state without leaking a claim from an unmounted screen. Artifacts claims on the app thread, performs exact service lookup in its existing worker, and uses an app-thread generation/lifecycle guard to release on restart, cancellation, or unmount before any stale callback can apply or settle; ACP compares one canonical helper-generated target with the current runtime session and never invents session history.

**Tech Stack:** Python 3.11+, typed dataclasses, `copy.deepcopy`, Textual lifecycle/workers, async service lookup, pytest/pytest-asyncio, existing installed-distribution integration gate.

**Backlog:** [TASK-646](../../../backlog/tasks/task-646%20-%20Complete-destination-handoff-ownership-and-ACP-target-recovery.md)

**Specification:** [Application Session State Ownership Design](../specs/2026-07-26-application-session-state-ownership-design.md)

**Depends on:** TASK-645

**ADR required:** yes

**ADR path:** `backlog/decisions/026-application-session-state-ownership.md`

**Reason:** ADR-026 already defines the remaining cross-screen ownership, exact-target, current-session recovery, settlement, and final architecture boundary.

---

## Execution Environment

This worktree has no `.venv`, and `/usr/bin/python3` is Python 3.9. Before
running any command in this plan:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/activate
python -c "import pathlib, tldw_chatbook; print(pathlib.Path(tldw_chatbook.__file__).resolve())"
```

The printed path must be inside
`.../.worktrees/privacy-lifecycle-eval-wheel-hardening/tldw_chatbook`, not the
main checkout or site-packages. The verified environment is Python 3.12.11,
pytest 8.4.2, and Ruff 0.15.22.

## File Structure

- Modify `tldw_chatbook/UI/Navigation/pending_handoff_store.py`: add Study scope/section, Artifact target, and ACP target normalizers and channels.
- Modify `tldw_chatbook/app.py`: migrate Study, Artifact, and ACP producers and remove the dead Notes slot.
- Modify `tldw_chatbook/UI/Screens/study_scope_models.py`: expose the existing
  valid Study section identifiers from a dependency-light module shared by the
  store and screen.
- Modify `tldw_chatbook/UI/Screens/study_screen.py`: claim and settle scope and section independently after restore/mount.
- Modify `tldw_chatbook/UI/Screens/artifacts_screen.py`: separate ordinary latest rendering from exact requested-target lookup and settle claims on the app thread.
- Modify `tldw_chatbook/UI/Screens/acp_screen.py`: consume current-session targets with visible exact-match or stale/unsupported recovery.
- Modify `tldw_chatbook/ACP_Interop/runtime_session.py`: add one canonical ACP session record-ID helper used by producer and consumer.
- Modify `Tests/UI/test_pending_handoff_store.py`: remaining channel normalization, deep-copy, and replacement tests.
- Modify `Tests/UI/test_study_screen.py`, `Tests/UI/test_study_dashboard.py`, `Tests/UI/test_study_quizzes_screen.py`, `Tests/UI/test_study_flashcards_screen.py`, and Phase 3 product-maturity Study tests: migrate fixtures and preserve mounted precedence.
- Modify `Tests/UI/test_console_live_work_handoffs.py`: exact Artifact lookup/races, producer routing, and ACP target staging.
- Modify `Tests/UI/test_destination_shells.py`: mounted Artifact and ACP recovery behavior.
- Modify `Tests/test_application_state_ownership.py`: final prohibition of every legacy pending field and owner bypass.
- Run, but do not weaken, `Tests/Packaging/test_installed_distribution.py`: integrated installed-wheel gate from TASK-545/ADR-025.

## Task 1: Extend Typed Channels and Migrate Producers

**Files:**

- Modify: `tldw_chatbook/UI/Navigation/pending_handoff_store.py`
- Modify: `tldw_chatbook/app.py`
- Modify: `tldw_chatbook/UI/Screens/study_scope_models.py`
- Modify: `tldw_chatbook/ACP_Interop/runtime_session.py`
- Modify: `Tests/UI/test_pending_handoff_store.py`
- Modify: `Tests/UI/test_console_live_work_handoffs.py`

- [ ] **Step 1: Write failing remaining-channel tests**

Add `HandoffChannel` members:

```python
STUDY_SCOPE = "study_scope"
STUDY_INITIAL_SECTION = "study_initial_section"
ARTIFACT_CHATBOOK_TARGET = "artifact_chatbook_target"
ACP_SESSION_TARGET = "acp_session_target"
```

Test that:

- Study scope accepts only `StudyScopeContext` and deep-copies nested `StudySourceItem.locator` mappings at stage and claim;
- Study section accepts only identifiers from the existing `StudyScreen._VALID_INITIAL_SECTIONS` contract without importing the screen class (define the shared allowed set next to the scope models or inject it as a module constant);
- Artifact accepts exactly `local:chatbook:<non-empty-id>`;
- ACP accepts exactly `local:acp_session:<non-empty-id>`;
- malformed prefixes, empty suffixes, and non-strings reject without replacing the prior valid slot;
- each channel has an independent revision and claim;
- staging a replacement during an older claimed target preserves only the latest replacement.
- clearing an optional Study channel during an older claim prevents release
  from resurrecting that older value.

- [ ] **Step 2: Add one canonical ACP ID helper**

In `ACP_Interop/runtime_session.py`:

```python
ACP_SESSION_RECORD_PREFIX = "local:acp_session:"


def acp_session_record_id(session_id: Any) -> str | None:
    normalized = str(session_id or "").strip()
    if not normalized:
        return None
    return f"{ACP_SESSION_RECORD_PREFIX}{normalized}"
```

Use this helper in `ACPRuntimeSessionState.to_console_live_work_launch()` instead of reconstructing the string inline. Add unit coverage for empty, whitespace, and normalized IDs.

- [ ] **Step 3: Run tests to verify missing channels/helpers**

Run:

```bash
pytest Tests/UI/test_pending_handoff_store.py Tests/UI/test_console_live_work_handoffs.py -q -k "study or artifact or acp or record_id"
```

Expected: FAIL until the channels and helper exist.

- [ ] **Step 4: Implement remaining normalizers**

For Study:

```python
if channel is HandoffChannel.STUDY_SCOPE:
    if not isinstance(value, StudyScopeContext):
        raise ValueError("invalid Study scope")
    return copy.deepcopy(value)
```

Move the valid section identifiers to a dependency-light constant in `study_scope_models.py` and import it into both `StudyScreen` and the handoff store. Target normalization must require exact prefix and non-empty stripped suffix while returning the complete canonical record ID; do not coerce arbitrary objects.

- [ ] **Step 5: Migrate app producers**

`open_study_screen()` stages each non-`None` scope/section value and calls
`clear_pending()` for each corresponding `None` argument before navigating.
This preserves the current behavior in which every call replaces or clears
both optional raw slots. `open_console_live_work_primary_action()` stages exact
Artifact/ACP targets before navigation and returns `False` with bounded
recovery if staging rejects. Delete any remaining initialization or assignment
for:

- `pending_study_scope_context`
- `pending_study_initial_section`
- `pending_artifacts_chatbook_target_id`
- `pending_acp_session_target_id`
- `pending_notes_workspace_context`

Do not add a Notes replacement channel: the slot is dead.

- [ ] **Step 6: Run protocol and producer tests**

Run:

```bash
pytest Tests/UI/test_pending_handoff_store.py Tests/UI/test_console_live_work_handoffs.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit channel and producer completion**

```bash
git add tldw_chatbook/UI/Navigation/pending_handoff_store.py tldw_chatbook/UI/Screens/study_scope_models.py tldw_chatbook/ACP_Interop/runtime_session.py tldw_chatbook/app.py Tests/UI/test_pending_handoff_store.py Tests/UI/test_console_live_work_handoffs.py
git commit -m "refactor(handoffs): own remaining destination channels (task-646)"
```

## Task 2: Settle Study Scope and Section Independently

**Files:**

- Modify: `tldw_chatbook/UI/Screens/study_screen.py`
- Modify: `Tests/UI/test_study_screen.py`
- Modify: `Tests/UI/test_study_dashboard.py`
- Modify: `Tests/UI/test_study_quizzes_screen.py`
- Modify: `Tests/UI/test_study_flashcards_screen.py`
- Modify: `Tests/UI/test_product_maturity_phase3_knowledge_entry.py`
- Modify: `Tests/UI/test_product_maturity_phase3_library_study_context.py`
- Modify: `Tests/UI/test_product_maturity_phase3_source_study_generation.py`

- [ ] **Step 1: Write failing Study settlement and precedence tests**

Prove:

- scope and section can be staged/claimed/settled separately;
- restore runs first, then staged scope and section win;
- successful scope application acknowledges only the scope claim;
- invalid section is rejected at staging and does not disturb scope;
- scope apply cancellation releases and re-raises;
- transient setup/apply failure releases;
- a newer scope staged while `_apply_scope_context_and_refresh()` awaits survives the older acknowledge/release;
- mutating the producer's or first consumer's nested locator cannot alter a released/retried claim;
- mounted product-maturity flows still land on the requested section and scope.

- [ ] **Step 2: Run focused Study tests**

Run:

```bash
pytest Tests/UI/test_study_screen.py Tests/UI/test_product_maturity_phase3_knowledge_entry.py Tests/UI/test_product_maturity_phase3_library_study_context.py Tests/UI/test_product_maturity_phase3_source_study_generation.py -q -k "pending or handoff or scope or initial_section or precedence"
```

Expected: FAIL while Study reads and clears raw fields.

- [ ] **Step 3: Claim only after navigation restore and mount**

Remove pending reads from `StudyScreen.__init__`; initialize from default/restored state only. In `on_mount()` and `on_screen_resume()`, claim each channel on the app thread. Apply a scope claim through `_apply_scope_context_and_refresh()`, acknowledge after successful application, release on cancellation/transient exception, and continue using current/restored scope when no claim exists.

Apply the section claim after scope and restored state, immediately before `_apply_section_layout()`. A valid applied section acknowledges. Do not make the two claims share settlement.

- [ ] **Step 4: Preserve deterministic failure policy**

Use an explicit helper per channel so `CancelledError` is re-raised:

```python
try:
    await self._apply_scope_context_and_refresh(...)
except asyncio.CancelledError:
    store.release(scope_claim)
    raise
except Exception as exc:
    store.release(scope_claim)
    logger.warning(
        "Study scope handoff failed (exception_category={})",
        type(exc).__name__,
    )
else:
    store.acknowledge(scope_claim)
```

Never log scope objects, locator mappings, workspace IDs, titles, or summaries.

- [ ] **Step 5: Run all affected Study suites**

Run:

```bash
pytest Tests/UI/test_study_screen.py Tests/UI/test_study_dashboard.py Tests/UI/test_study_quizzes_screen.py Tests/UI/test_study_flashcards_screen.py Tests/UI/test_product_maturity_phase3_knowledge_entry.py Tests/UI/test_product_maturity_phase3_library_study_context.py Tests/UI/test_product_maturity_phase3_source_study_generation.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Study migration**

```bash
git add tldw_chatbook/UI/Screens/study_screen.py Tests/UI/test_study_screen.py Tests/UI/test_study_dashboard.py Tests/UI/test_study_quizzes_screen.py Tests/UI/test_study_flashcards_screen.py Tests/UI/test_product_maturity_phase3_knowledge_entry.py Tests/UI/test_product_maturity_phase3_library_study_context.py Tests/UI/test_product_maturity_phase3_source_study_generation.py
git commit -m "refactor(study): settle owned scope and section handoffs (task-646)"
```

## Task 3: Resolve Artifact Handoffs by Exact ID

**Files:**

- Modify: `tldw_chatbook/UI/Screens/artifacts_screen.py`
- Modify: `Tests/UI/test_console_live_work_handoffs.py`
- Modify: `Tests/UI/test_destination_shells.py`

- [ ] **Step 1: Write failing exact lookup and race tests**

Provide fakes with both `list_chatbooks()` and `get_chatbook()` call recording. Cover:

```python
async def test_requested_artifact_uses_exact_lookup_not_first_page_latest():
    service = FakeChatbookService(
        listed=[_record(99, "Latest")],
        exact={"77": _record(77, "Requested")},
    )
    # Stage local:chatbook:77, mount Artifacts, wait for context.
    assert service.get_calls == ["77"]
    assert service.list_calls == []
    assert "Requested" in visible_text
    assert "Latest" not in visible_text
```

Also prove:

- `KeyError` shows explicit missing-target recovery and acknowledges;
- absent/unready service releases;
- non-`KeyError` lookup failure releases and logs no target/exception text;
- a malformed or wrong-ID record returned by `get_chatbook()` is never
  selected as success and releases as a service-contract failure;
- success acknowledges only after exact selection is installed;
- a replacement staged during awaited lookup remains pending;
- ordinary no-handoff mount still lists and selects the latest record;
- a requested target can never settle through the 25-record list or latest fallback.
- unmounting before lookup completion releases the active claim immediately,
  prevents the old screen from applying the result, and lets a later screen
  claim the target;
- explicitly cancelling the current worker releases its claim through the
  screen's matching terminal worker-state handler;
- restarting the exclusive refresh releases its previous active claim before
  claiming again, so exclusive cancellation cannot strand an in-flight slot;
- a late callback from an older generation cannot acknowledge, release, or
  apply over a claim held by the current generation.

Because the lookup runs inside a `thread=True` worker and its awaitable is
driven by `asyncio.run()` in that worker, use `threading.Event` barriers for
the cross-thread replacement race. Do not share an `asyncio.Event` between the
application loop and the worker loop, and do not use timing sleeps.

- [ ] **Step 2: Run Artifact tests and verify fallback failure**

Run:

```bash
pytest Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_destination_shells.py -q -k "artifact or chatbook"
```

Expected: new tests FAIL because the current consumer eagerly clears and falls back to the latest first-page record.

- [ ] **Step 3: Separate ordinary latest loading from requested exact loading**

Track `_active_chatbook_claim`, `_chatbook_refresh_worker`, a monotonic
`_chatbook_refresh_generation`, and `_chatbook_unmounted`. On mount, mark the
screen live before starting a refresh. On unmount, mark it unavailable and
release/clear the exact active claim on the app thread before returning; never
rely on `is_mounted`, which does not prove that this screen is still the
active destination.

Route mount, resume, and manual/exclusive restarts through one app-thread
starter. Before a new generation begins, it releases and clears any prior
active claim, increments the generation, claims
`ARTIFACT_CHATBOOK_TARGET`, and passes only the claim's normalized target
string plus the generation into the existing worker. This covers a thread
worker cancelled before it can post a callback and prevents an old in-flight
claim from blocking the next lifecycle consumer. Parse the non-empty suffix
and call:

```python
result = service.get_chatbook(chatbook_id)
if inspect.isawaitable(result):
    result = asyncio.run(result)
```

Use `list_chatbooks(limit=25, ...)` only when there is no claimed target.
Return a small outcome category (`success`, `missing`, `transient`) plus launch
data, generation, and the exact claim reference to the app-thread callback; do
not return/log exception objects or target values.
Before returning `success`, reconstruct the record's canonical Chatbook target
with the existing `_chatbook_target_id()` helper and require exact equality
with the claim value. Treat a malformed or mismatched record as `transient`;
it is neither an exact success nor definitive evidence that the target is
missing.

- [ ] **Step 4: Settle in the app-thread callback**

Before touching UI or settling, require all of:

- the screen is not marked unmounted;
- callback generation equals `_chatbook_refresh_generation`;
- callback claim is the same object as `_active_chatbook_claim`.

If any check fails, do not apply UI. Attempting to release the stale claim is
safe but may return `False` because unmount/restart already settled it. For the
current generation:

- `success`: install requested launch/context, then acknowledge;
- `missing`: install explicit missing-target recovery, notify bounded copy, then acknowledge;
- `transient`: install service-unavailable recovery, release;
- unexpected callback failure: release and emit metadata-only diagnostics.

Clear `_active_chatbook_claim` in `finally` only when it is still the
callback's exact claim. The callback settles the exact claim object held by
this refresh and cannot clear a newer target. The worker must post a contained
`transient` outcome for handled failures; screen teardown or a restarted
generation provides the app-thread release path when cancellation prevents a
callback. Remove `_requested_chatbook_target_id` and all raw-field
consumption.

Handle `Worker.StateChanged` for only the exact
`_chatbook_refresh_worker`. On `CANCELLED`, `ERROR`, or a terminal state that
somehow leaves its exact claim active, release and clear that claim on the app
thread. Ignore older workers after a restart: the starter already released
their claim, and their identity no longer matches. This terminal-state path is
the cancellation safety net when the worker never reaches its callback.

- [ ] **Step 5: Run Artifact mounted flows**

Run:

```bash
pytest Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_destination_shells.py -q -k "artifact or chatbook"
```

Expected: PASS.

- [ ] **Step 6: Commit exact Artifact recovery**

```bash
git add tldw_chatbook/UI/Screens/artifacts_screen.py Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_destination_shells.py
git commit -m "fix(artifacts): resolve owned Chatbook targets exactly (task-646)"
```

## Task 4: Complete ACP Current-Session Target Recovery

**Files:**

- Modify: `tldw_chatbook/UI/Screens/acp_screen.py`
- Modify: `tldw_chatbook/ACP_Interop/runtime_session.py`
- Modify: `Tests/UI/test_destination_shells.py`
- Modify: `Tests/UI/test_console_live_work_handoffs.py`

- [ ] **Step 1: Write failing ACP consumer tests**

Mounted tests must prove:

- an exact canonical target matches the helper-generated current record ID;
- the existing `#acp-session-list-row` remains the selected current row;
- `#acp-detail-pane.scroll_visible(animate=False)` is invoked after mount;
- success shows bounded informational recovery and acknowledges;
- no runtime session, different current session, malformed defensive fake claim, or missing detail pane shows explicit "only the current ACP runtime session is available" recovery and acknowledges;
- ACP never changes `acp_runtime_session_state`, searches history, or navigates elsewhere;
- a newer target staged during settlement remains pending.

Producer tests must prove malformed ACP/Artifact action targets are rejected with visible bounded recovery before navigation.

- [ ] **Step 2: Run ACP tests and verify the missing consumer**

Run:

```bash
pytest Tests/UI/test_destination_shells.py Tests/UI/test_console_live_work_handoffs.py -q -k "acp and (target or session or primary_action)"
```

Expected: FAIL because `pending_acp_session_target_id` is staged but never consumed.

- [ ] **Step 3: Consume after ACP mount**

Add `on_mount()`:

```python
def on_mount(self) -> None:
    super().on_mount()
    self.call_after_refresh(self._consume_pending_session_target)
```

The consumer claims `ACP_SESSION_TARGET`, reconstructs current ID with `acp_session_record_id(state.session_id)`, and compares complete strings. On exact match, query the current row and detail pane, preserve the row's selected presentation, call `detail.scroll_visible(animate=False)`, notify `"Opened the current ACP session details."`, and acknowledge.

If the session is absent/mismatched or detail focus is unavailable, notify bounded stale/unsupported copy and acknowledge. A malformed value is unreachable through the real store but the consumer should still fail terminally when exercised with a defensive fake. No failure path logs the target or runtime state.

- [ ] **Step 4: Run ACP mounted flows**

Run:

```bash
pytest Tests/UI/test_destination_shells.py Tests/UI/test_console_live_work_handoffs.py -q -k "acp"
```

Expected: PASS.

- [ ] **Step 5: Commit ACP recovery**

```bash
git add tldw_chatbook/UI/Screens/acp_screen.py tldw_chatbook/ACP_Interop/runtime_session.py Tests/UI/test_destination_shells.py Tests/UI/test_console_live_work_handoffs.py
git commit -m "fix(acp): consume current-session target handoffs (task-646)"
```

## Task 5: Close the Ownership Boundary and Add Privacy Sentinels

**Files:**

- Modify: `Tests/UI/test_pending_handoff_store.py`
- Modify: `Tests/UI/test_study_screen.py`
- Modify: `Tests/UI/test_console_live_work_handoffs.py`
- Modify: `Tests/UI/test_destination_shells.py`
- Modify: `Tests/test_application_state_ownership.py`

- [ ] **Step 1: Add remaining privacy sentinels**

Stage unique secrets in a Study locator, Artifact target suffix, and ACP session suffix. Force consumer/service/focus failures, capture all logs, and assert the sentinel is absent. Stable diagnostics may include channel, revision, outcome, and exception category only.

- [ ] **Step 2: Finalize the AST owner guard**

Reject all production attributes:

```text
pending_chat_handoff
pending_console_launch
pending_console_prompt_insert
pending_study_scope_context
pending_study_initial_section
pending_notes_workspace_context
pending_artifacts_chatbook_target_id
pending_acp_session_target_id
_screen_states
```

Also reject direct `PendingHandoffStore._slots` access, persistence/serialization from the handoff module, and application projection writes outside the TASK-643 boundary. Parse assignments, annotated assignments, augmented assignments, deletes, and `getattr`/`setattr`; do not rely on substring grep.

- [ ] **Step 3: Run the final focused guard**

```bash
pytest Tests/UI/test_pending_handoff_store.py Tests/UI/test_study_screen.py Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_destination_shells.py Tests/test_application_state_ownership.py -q
```

Expected: PASS.

- [ ] **Step 4: Commit final guards**

```bash
git add Tests/UI/test_pending_handoff_store.py Tests/UI/test_study_screen.py Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_destination_shells.py Tests/test_application_state_ownership.py
git commit -m "test(state): close destination handoff ownership boundary (task-646)"
```

## Task 6: Run Integrated Release Gates

**Files:**

- No production changes expected; fix only verified regressions within TASK-643–646 acceptance criteria.

- [ ] **Step 1: Run all focused application-state suites**

```bash
pytest Tests/RuntimePolicy Tests/UI/test_screen_state_store.py Tests/UI/test_screen_navigation.py Tests/UI/test_pending_handoff_store.py Tests/UI/test_chat_first_handoffs.py Tests/UI/test_console_command_composer.py Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_study_screen.py Tests/UI/test_study_dashboard.py Tests/UI/test_study_quizzes_screen.py Tests/UI/test_study_flashcards_screen.py Tests/UI/test_destination_shells.py Tests/UI/test_ux_audit_smoke.py Tests/test_application_state_ownership.py -q
```

Expected: PASS.

- [ ] **Step 2: Run product-maturity UI sentinels**

```bash
pytest Tests/UI/test_product_maturity_phase1_harness.py -q
pytest Tests/UI/test_product_maturity_phase1_core_loop.py Tests/UI/test_product_maturity_phase3_knowledge_entry.py Tests/UI/test_product_maturity_phase3_library_study_context.py Tests/UI/test_product_maturity_phase3_source_study_generation.py -q
```

Expected: PASS with existing visible Chat/Console and Study flows preserved.

- [ ] **Step 3: Run the installed-wheel gate**

```bash
pytest Tests/Packaging/test_installed_distribution.py -q
```

Expected: PASS. This must build fresh distributions, install the wheel outside the checkout with `--no-deps`, resolve `tldw_chatbook` from the installed target, exercise packaged loaders/entry points, and preserve installed-target hashes. Do not replace it with a source-checkout import smoke.

- [ ] **Step 4: Run static verification**

Planning-time verification found 46 existing full-tree Ruff diagnostics and
five files outside the Ruff formatter baseline. A raw
`ruff check tldw_chatbook Tests` or full-tree format gate would therefore fail
before this tranche starts. Re-run the scoped gates from TASK-643–645, then
run the remaining destination scope below. The two targeted F841 ignores cover
only the verified pre-tranche diagnostics in `config.py` and
`settings_screen.py`; do not absorb unrelated lint or mass-format cleanup.

```bash
python -m compileall -q tldw_chatbook
python -m ruff check tldw_chatbook/runtime_policy tldw_chatbook/state tldw_chatbook/app.py tldw_chatbook/UI/Navigation tldw_chatbook/UI/Screens/media_ingest_screen.py tldw_chatbook/UI/Screens/study_screen.py tldw_chatbook/UI/Screens/study_scope_models.py tldw_chatbook/UI/Screens/home_screen.py tldw_chatbook/UI/Screens/workflows_screen.py tldw_chatbook/UI/Screens/schedules_screen.py tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/UI/Screens/artifacts_screen.py tldw_chatbook/UI/Screens/acp_screen.py tldw_chatbook/Chat/console_live_work.py tldw_chatbook/ACP_Interop/runtime_session.py Tests/RuntimePolicy Tests/UI/test_screen_state_store.py Tests/UI/test_screen_navigation.py Tests/UI/test_pending_handoff_store.py Tests/UI/test_chat_first_handoffs.py Tests/UI/test_console_command_composer.py Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_study_screen.py Tests/UI/test_study_dashboard.py Tests/UI/test_study_quizzes_screen.py Tests/UI/test_study_flashcards_screen.py Tests/UI/test_destination_shells.py Tests/UI/test_ux_audit_smoke.py Tests/UI/test_product_maturity_phase1_core_loop.py Tests/UI/test_product_maturity_phase1_harness.py Tests/UI/test_product_maturity_phase3_knowledge_entry.py Tests/UI/test_product_maturity_phase3_library_study_context.py Tests/UI/test_product_maturity_phase3_source_study_generation.py Tests/test_application_state_ownership.py
python -m ruff check --ignore F841 tldw_chatbook/config.py tldw_chatbook/UI/Screens/settings_screen.py
python -m ruff format --check tldw_chatbook/runtime_policy/source_state.py tldw_chatbook/runtime_policy/bootstrap.py tldw_chatbook/runtime_policy/server_capabilities.py tldw_chatbook/state/app_state.py tldw_chatbook/state/__init__.py tldw_chatbook/UI/Navigation tldw_chatbook/UI/Screens/media_ingest_screen.py tldw_chatbook/UI/Screens/study_screen.py tldw_chatbook/UI/Screens/study_scope_models.py tldw_chatbook/UI/Screens/home_screen.py tldw_chatbook/UI/Screens/workflows_screen.py tldw_chatbook/UI/Screens/schedules_screen.py tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py tldw_chatbook/UI/Screens/artifacts_screen.py tldw_chatbook/UI/Screens/acp_screen.py tldw_chatbook/Chat/console_live_work.py tldw_chatbook/ACP_Interop/runtime_session.py Tests/RuntimePolicy Tests/UI/test_screen_state_store.py Tests/UI/test_screen_navigation.py Tests/UI/test_pending_handoff_store.py Tests/UI/test_chat_first_handoffs.py Tests/UI/test_console_command_composer.py Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_study_screen.py Tests/UI/test_study_dashboard.py Tests/UI/test_study_quizzes_screen.py Tests/UI/test_study_flashcards_screen.py Tests/UI/test_destination_shells.py Tests/UI/test_ux_audit_smoke.py Tests/UI/test_product_maturity_phase1_core_loop.py Tests/UI/test_product_maturity_phase1_harness.py Tests/UI/test_product_maturity_phase3_knowledge_entry.py Tests/UI/test_product_maturity_phase3_library_study_context.py Tests/UI/test_product_maturity_phase3_source_study_generation.py Tests/test_application_state_ownership.py
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 5: Run the full suite**

```bash
pytest -q
```

Expected: PASS. Record exact pass/skip/warning counts and duration. Do not mark TASK-646 Done on a partial or stale run.

## Task 7: Reconcile the Integrated Tranche

**Files:**

- Modify: `Docs/superpowers/specs/2026-07-26-application-session-state-ownership-design.md`
- Modify: `backlog/tasks/task-646 - Complete-destination-handoff-ownership-and-ACP-target-recovery.md`
- Modify: `backlog/tasks/task-643 - Make-runtime-policy-the-sole-application-runtime-source-authority.md`
- Modify: `backlog/tasks/task-644 - Move-cross-visit-screen-snapshots-behind-an-in-memory-owner.md`
- Modify: `backlog/tasks/task-645 - Move-Chat-and-Console-handoffs-behind-revisioned-single-slot-ownership.md`

- [ ] **Step 1: Re-audit all four tasks against fresh sentinel evidence**

Run `backlog task 643 --plain` through `backlog task 646 --plain`. Confirm all
four tasks are still In Progress with unchecked acceptance criteria, and
verify every criterion against the fresh focused, installed-wheel,
product-maturity, static, and full-suite results. If an earlier invariant
regressed, fix it under that task's acceptance criteria before reconciliation;
never paper over it only in TASK-646 notes.

- [ ] **Step 2: Self-review TASK-646 acceptance criteria**

Confirm exact evidence for independent Study settlement, nested locator detachment, exact Artifact lookup and replacement race, current-only ACP recovery and detail exposure, removal of every raw field/dead Notes slot, AST ownership enforcement, privacy redaction, installed-wheel behavior, product-maturity sentinels, and the full suite.

- [ ] **Step 3: Complete Backlog hygiene**

Use the Backlog CLI to:

- check every acceptance criterion in TASK-643 through TASK-646;
- add concise task-specific Implementation Notes to each task with approach,
  tradeoffs, modified files, ADR-026, and the exact focused plus shared gate
  evidence;
- set TASK-643 through TASK-646 Done only after all four task files and the
  evidence have been re-read.

Update the design status to Implemented only when all four tasks are Done and every integrated gate above is green.

- [ ] **Step 4: Commit final implementation closeout**

```bash
git add Docs/superpowers/specs/2026-07-26-application-session-state-ownership-design.md 'backlog/tasks/task-643 - Make-runtime-policy-the-sole-application-runtime-source-authority.md' 'backlog/tasks/task-644 - Move-cross-visit-screen-snapshots-behind-an-in-memory-owner.md' 'backlog/tasks/task-645 - Move-Chat-and-Console-handoffs-behind-revisioned-single-slot-ownership.md' 'backlog/tasks/task-646 - Complete-destination-handoff-ownership-and-ACP-target-recovery.md'
git commit -m "docs(state): close application ownership tranche (task-646)"
```
