# Provider Selection Ownership Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove `TldwCli.chat_api_provider_value` and give persisted defaults, active Console selections, and away-from-Console commands explicit non-overlapping owners.

**Architecture:** Settings persists and refreshes `chat_defaults`; each native Console session owns its `ConsoleSessionSettings`; command-palette selection crosses destinations through one typed `PendingHandoffStore` channel with existing single-slot revision semantics. Provider/model resolution becomes a pure explicit-input function.

**Tech Stack:** Python 3.11+, frozen dataclasses, `dataclasses.replace`, Textual command providers/screens, pytest/pytest-asyncio, AST privacy guards.

**Backlog:** [TASK-648](../../../backlog/tasks/task-648%20-%20Move-provider-selection-to-Settings-Console-sessions-and-a-typed-handoff.md)

**Specification:** [TldwCli Reactive State Decomposition Design](../specs/2026-07-26-tldwcli-reactive-state-decomposition-design.md)

**Depends on:** TASK-647

**ADR required:** yes

**ADR path:** `backlog/decisions/006-provider-aware-generation-settings.md`; `backlog/decisions/026-application-session-state-ownership.md`

**Reason:** The accepted ADRs already define Settings defaults, per-session Console authority, and memory-only single-slot handoffs.

---

## Execution and Test Boundary

Use the worktree environment check from TASK-647. Mounted tests go only in
`Tests/ProductionApp/test_provider_selection_ownership.py` and use the normal
`TldwCli`, registered `SettingsScreen`/`ChatScreen`, and actual
`ConsoleChatStore`. Protocol and resolver tests call the real store/models and
pure functions directly. Do not construct a namespace or mock application for
provider resolution.

## File Structure

- Modify `tldw_chatbook/UI/Navigation/pending_handoff_store.py`: add the typed provider-selection channel and normalization.
- Modify `tldw_chatbook/UI/Navigation/__init__.py`: export the typed provider-selection intent.
- Modify `tldw_chatbook/UI/Screens/provider_model_resolution.py`: replace implicit app reads with explicit configuration/default/session inputs.
- Modify `tldw_chatbook/UI/Screens/chat_screen.py`: apply and consume provider selections against the exact active session.
- Modify `tldw_chatbook/UI/Screens/settings_screen.py`: persist/reload defaults without writing runtime provider/model fields.
- Modify `tldw_chatbook/Widgets/model_search_picker.py`: pass explicit provider/model mappings and the catalog scope service to the resolver.
- Modify `tldw_chatbook/app.py`: update command-palette behavior and remove the root descriptor, initializer, and watcher.
- Move `Tests/UI/test_pending_handoff_store.py` to `Tests/State/test_pending_handoff_store.py`: keep the app-independent protocol suite outside the UI surrogate-harness tree.
- Replace `Tests/UI/test_provider_model_resolution.py` with `Tests/Provider/test_provider_model_resolution.py`: pass explicit mappings and a narrow catalog-service collaborator, never a fake app.
- Delete `Tests/Widgets/test_model_search_picker.py`: replace the simplified host apps with direct resolver coverage and a real Console popover path in the production-app suite.
- Create `Tests/ProductionApp/test_provider_selection_ownership.py`.
- Modify `Tests/test_application_state_ownership.py`.

## Task 1: Start TASK-648 and Specify the Typed Channel

- [ ] Move the task In Progress and add its task-local plan:

```bash
backlog task edit 648 -s "In Progress"
backlog task edit 648 --plan $'ADR required: yes\nADR path: backlog/decisions/006-provider-aware-generation-settings.md; backlog/decisions/026-application-session-state-ownership.md\nReason: Existing ADRs define durable defaults, active session authority, and memory-only handoff semantics.\n\n1. Add the typed provider intent channel.\n2. Make provider/model resolution explicit.\n3. Apply selections to exact Console sessions.\n4. Remove the root cache and Settings mirror writes.\n5. Verify privacy and production behavior.'
```

- [ ] Define a frozen `ConsoleProviderIntent` carrying only a normalized provider identifier. Add `HandoffChannel.CONSOLE_PROVIDER` using the existing per-channel monotonic revision, last-write-wins, claim, acknowledge, and release protocol.
- [ ] Direct tests must cover invalid/blank provider rejection, replacement while claimed, stale claimant settlement failure, release/retry, and a `repr`/log sentinel proving no credentials, endpoints, prompts, response bodies, or catalog payloads enter the intent.
- [ ] Run:

```bash
pytest Tests/State/test_pending_handoff_store.py -q -k provider
```

Expected: FAIL before the channel exists, then PASS after its implementation.

## Task 2: Make Provider/Model Resolution Explicit

- [ ] Change `resolve_effective_provider_model()` and
  `resolve_provider_model_options()` to accept explicit:
  - persisted chat defaults;
  - active Console settings/control values;
  - Settings draft values.
  - saved provider/model mappings and the narrow catalog scope service used by
    model-option merging.
- [ ] Remove `_chat_default(app_instance, ...)`, `getattr(...chat_api_provider_value...)`, and app-reactive source labels. Preserve precedence: Settings draft, active Console selection, persisted default.
- [ ] Replace `ChatScreen._console_resolution_view()` and its `SimpleNamespace` with a direct immutable/default mapping passed to the resolver.
- [ ] Update `ModelSearchPicker` to read the production app's mappings and narrow catalog service explicitly at the resolver boundary; it must not restore an app-shaped resolver argument.
- [ ] Convert legitimate app-independent resolver tests to mappings/value objects. Delete or rewrite cases that depend on a fake application rather than retaining a compatibility argument.
- [ ] Run:

```bash
pytest Tests/Provider/test_provider_model_resolution.py -q
```

Expected: PASS using direct explicit inputs and no app surrogate.

## Task 3: Apply Commands to the Exact Console Session

- [ ] Add a narrow `ChatScreen` operation that:
  - validates the provider against current configured provider identities;
  - captures the active session ID before mutation;
  - chooses that provider's configured default model when valid;
  - clears an incompatible old-provider model;
  - preserves unrelated settings, including `system_prompt`;
  - writes `source="user"` through `ConsoleChatStore.replace_session_settings()`;
  - refreshes Console controls and readiness from the replaced snapshot.
- [ ] Claim the provider intent only after the Console store/session is ready. A valid selection acknowledges; an invalid/unsupported provider notifies bounded recovery and acknowledges; a transient readiness failure releases the exact claim.
- [ ] Use the same handoff path when Console is already active: stage, then ask the mounted real `ChatScreen` to consume. Away from Console, stage once and honestly notify that the next Console entry will apply it.
- [ ] Add mounted production tests for active-session application, away-from-Console fresh navigation, replacement races, invalid terminal rejection, transient release/retry, and preservation of a user session across a Settings save.
- [ ] Exercise model search through the real `ConsoleModelPopover` mounted by the production `ChatScreen`; do not retain a simplified picker host application.

## Task 4: Remove the Root Cache and Settings Mirror

- [ ] In `settings_screen.py`, delete `_sync_provider_runtime_defaults()`. After successful persistence, update/reload the actual `app_config` defaults and provider settings only; never modify an active `source="user"` session.
- [ ] In `LLMProviderProvider`, report the mounted Console session provider when Chat is active and the persisted default otherwise. Selection must use the typed channel, not a root assignment.
- [ ] In `app.py`, delete `chat_api_provider_value`, its boot assignment, and `watch_chat_api_provider_value()`.
- [ ] Remove the legacy model-select refresh path that existed only for this watcher. Do not introduce a compatibility property.
- [ ] Extend the AST guard to prohibit a `TldwCli` descriptor, assignment, attribute access, or dynamic string access named `chat_api_provider_value`.
- [ ] Run:

```bash
pytest Tests/ProductionApp/test_provider_selection_ownership.py Tests/State/test_pending_handoff_store.py Tests/Provider/test_provider_model_resolution.py Tests/test_application_state_ownership.py -q
```

Expected: PASS.

## Task 5: Verify and Close TASK-648

- [ ] Run:

```bash
python -m compileall -q tldw_chatbook/UI/Navigation/pending_handoff_store.py tldw_chatbook/UI/Navigation/__init__.py tldw_chatbook/UI/Screens/provider_model_resolution.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/UI/Screens/settings_screen.py tldw_chatbook/Widgets/model_search_picker.py tldw_chatbook/app.py
python -m ruff check tldw_chatbook/UI/Navigation/pending_handoff_store.py tldw_chatbook/UI/Navigation/__init__.py tldw_chatbook/UI/Screens/provider_model_resolution.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/Widgets/model_search_picker.py tldw_chatbook/app.py Tests/State/test_pending_handoff_store.py Tests/Provider/test_provider_model_resolution.py Tests/ProductionApp/test_provider_selection_ownership.py Tests/test_application_state_ownership.py
python -m ruff check --ignore F841 tldw_chatbook/UI/Screens/settings_screen.py
python -c 'import json, subprocess, sys; p = subprocess.run([sys.executable, "-m", "ruff", "check", "--select", "F841", "--output-format", "json", "tldw_chatbook/UI/Screens/settings_screen.py"], capture_output=True, text=True); findings = json.loads(p.stdout); assert len(findings) == 2 and all(item["code"] == "F841" and "`config_path`" in item["message"] for item in findings), findings'
python -m ruff format --check tldw_chatbook/UI/Navigation/pending_handoff_store.py tldw_chatbook/UI/Navigation/__init__.py tldw_chatbook/Widgets/model_search_picker.py Tests/State/test_pending_handoff_store.py Tests/Provider/test_provider_model_resolution.py Tests/ProductionApp/test_provider_selection_ownership.py Tests/test_application_state_ownership.py
git diff --check
```

- `app.py`, `chat_screen.py`, `provider_model_resolution.py`, and
  `settings_screen.py` are verified pre-task Ruff-format baseline exceptions;
  do not mass-format them. The two existing `settings_screen.py` F841 findings
  are isolated with the explicit targeted ignore above; the JSON assertion
  fails if that exact two-finding baseline grows or changes.

- [ ] Commit implementation:

```bash
git add tldw_chatbook/UI/Navigation/pending_handoff_store.py tldw_chatbook/UI/Navigation/__init__.py tldw_chatbook/UI/Screens/provider_model_resolution.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/UI/Screens/settings_screen.py tldw_chatbook/Widgets/model_search_picker.py tldw_chatbook/app.py Tests/State/test_pending_handoff_store.py Tests/Provider/test_provider_model_resolution.py Tests/ProductionApp/test_provider_selection_ownership.py Tests/test_application_state_ownership.py
git add -u Tests/UI/test_pending_handoff_store.py Tests/UI/test_provider_model_resolution.py Tests/Widgets/test_model_search_picker.py
git commit -m "refactor(console): own provider selection by lifetime (task-648)"
```

- [ ] Re-read the task, record exact verification evidence in Implementation Notes, check all acceptance criteria, mark Done, and commit only its task file:

```bash
backlog task 648 --plain
backlog task edit 648 --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --check-ac 5 -s Done
git add 'backlog/tasks/task-648 - Move-provider-selection-to-Settings-Console-sessions-and-a-typed-handoff.md'
git commit -m "docs(backlog): close provider ownership (task-648)"
```
