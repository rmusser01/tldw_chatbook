# Console First-Chat Handoff Ownership Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the exact eight first-chat handoff policy methods and their revision state from `ChatScreen` into `ConsoleSessionController` without changing claim, session, rollback, retry, privacy, mounted UI, or focus behavior.

**Architecture:** Extend the existing session controller; do not create another controller or leave screen delegates. The controller owns policy and durable/session coordination, while four narrow late-bound wiring callbacks expose mounted state, an opaque presentation snapshot, control selection projection, and focus restoration. Existing session-controller synchronization callbacks remain the only repaint seams.

**Tech Stack:** Python 3.11+, Textual 8, pytest/pytest-asyncio, Ruff, stdlib AST checks, governed persistent-diagnostic inventory, Backlog.md CLI.

---

## Governing References

- Task: `backlog/tasks/task-3070.9 - Extract-Console-first-chat-handoff-ownership.md`
- Approved design: `Docs/superpowers/specs/2026-08-22-task-3070-9-console-first-chat-handoff-design.md`
- Parent Wave 6 design: `Docs/superpowers/specs/2026-08-13-console-decomposition-wave6-design.md`
- Testing evidence: `backlog/docs/lessons-testing-evidence.md`
- Backlog hygiene: `backlog/docs/lessons-backlog-hygiene.md`
- Repository instructions: `AGENTS.md`

ADR required: no

ADR path: N/A

Reason: this is a mechanical implementation of the already approved Wave 6
ownership boundary. It introduces no new storage, sync, security, dependency,
service, or application-lifetime decision.

## Frozen Starting Evidence

- Approved implementation base: `0da426e1e4c2846f13671690b8f981f72e673359`.
- `chat_screen.py`: 19,995 physical lines.
- `ChatScreen`: 640 direct methods.
- First-chat family: exactly eight direct methods / 328 definition lines.
- Expected arithmetic projection: at most 19,667 lines / 632 direct methods
  before ordinary import and wiring adjustments.
- The family is already declared in `WAVE6_GROUPS["first_chat"]`; preserve its
  historical `raw_lines=328` and source revision.

If a later rebase changes the eight-method membership or behavior, stop and
amend/re-review the design. Do not silently rewrite the historical oracle.

## Latest-Dev Implementation Baseline (Task 0)

- Pre-rebase base/head: `0da426e1e4c2846f13671690b8f981f72e673359` /
  `c8c8671638e0267daad1cf4bae25591f5df32463`.
- Freshly fetched `origin/dev` and post-rebase base:
  `ede2162143331e324c44832ff6a3910e1185cf58`; rebased documentation head:
  `0d24696182483521203263093147ad34405662a1`.
- `git range-diff` paired all three reviewed documentation commits unchanged.
- The immutable approved-base and rebased-candidate AST measurements both read
  19,995 physical lines / 640 direct methods and the exact ordered eight-method
  family / 328 definition lines. Both normalized family ASTs hash to
  `3a2968883c63dc89de430ee72b40444ebd97fb9b36c1dbc8a46e19d063a715ee`.
- The exact focused baseline completed with `230 passed, 2 warnings in 148.85s
  (0:02:28)`. The warnings were the environment's Requests dependency-version
  warning and the Python 3.13 `audioop` deprecation warning.
- The non-write checker exited `1`; this is an inherited-red baseline, not a
  passing checker. A validated `/private/tmp` archive of frozen rebased base
  `ede2162143331e324c44832ff6a3910e1185cf58` reproduced byte-identical checker
  output (SHA-256
  `bdd245044db1597a76c95543d9d6bb56bee1cf6d86f4d96a8f9524f0cfe47f77`):
  7,206 to 7,211 calls, `Subscriptions_DB.py`
  `5/8d4a08a1d2b297b3ea78 -> 8/aba72ffb44d7eaba6204` (+3), and
  `scheduler/loop.py`
  `6/3a01bd3222d1bf8254f1 -> 8/c454d267a78237dcdf00` (+2), with no
  sink-topology row. The candidate metadata-only node passed with `1 passed, 1
  warning in 2.23s`; the frozen-base counterfactual also passed with `1 passed,
  1 warning in 2.39s`. The task series has no source change, no writer ran, and
  no registry or manifest changed. This proves non-regression only; Task 5 still
  requires final reconciliation and a passing checker.
- Backlog status remains `In Progress`; the concise five-step implementation
  plan is present and all acceptance criteria remain unchecked.

## File Map

**Production**

- Modify `tldw_chatbook/UI/Console_Modules/session.py`: own the eight methods,
  revision state, imports, and narrow presentation dependencies.
- Modify `tldw_chatbook/UI/Console_Modules/wiring.py`: bind the new presentation
  callbacks late and add at most two small module-level presentation helpers.
- Modify `tldw_chatbook/UI/Screens/chat_screen.py`: delete the eight methods and
  revision state; route mount/resume calls to `_session`; remove dead imports.
- Modify `tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py`: query the mounted
  Console screen's `_session` owner instead of probing a screen method.

**Tests and governance**

- Modify `Tests/UI/test_console_session_settings.py`: direct no-mount and mounted
  first-chat behavior to `_session`; patch the owning module for config races.
- Modify `Tests/UI/test_console_controller_wiring.py`: prove callback shape and
  late binding.
- Modify `Tests/Wizards/test_first_run_setup_wizard.py`: preserve producer
  behavior through the controller owner.
- Modify `Tests/Architecture/test_console_wave6_inventory.py`: completed
  ownership, dependency, no-DOM/no-sibling, multiplicity, ratchet, and synthetic
  non-vacuity oracles.
- Inspect and modify `Tests/Architecture/test_persistent_diagnostic_inventory.py`
  only if its reviewed metadata registry names the moved call owner.
- Modify `Docs/security/production-diagnostic-inventory.json` only after the
  final reviewed three-way diagnostic comparison and exactly one canonical
  writer invocation.
- Modify this plan and the Backlog task only for truthful execution evidence and
  closeout.

Do not add a compatibility method, descriptor, protocol, controller, or generic
callback abstraction.

### Task 0: Freeze baseline evidence and commit the reviewed plan

**Files:**
- Modify: `Docs/superpowers/specs/2026-08-22-task-3070-9-console-first-chat-handoff-design.md`
- Modify: `Docs/superpowers/plans/2026-08-22-task-3070-9-console-first-chat-handoff.md`
- Modify: `backlog/tasks/task-3070.9 - Extract-Console-first-chat-handoff-ownership.md`

- [x] **Step 1: Verify the isolated worktree and reviewed documentation scope**

Run:

```bash
git status --short --branch
git rev-parse HEAD origin/dev
git merge-base HEAD origin/dev
```

Expected: only the reviewed design, plan, and task files are modified/untracked;
`HEAD` descends from the approved base. Do not rebase with these documentation
edits uncommitted. Preserve `0da426e1...` as immutable design evidence.

- [x] **Step 2: Validate the approved-base screen and exact method family**

Run:

```bash
../../.venv/bin/python - <<'PY'
import ast
from pathlib import Path

path = Path("tldw_chatbook/UI/Screens/chat_screen.py")
source = path.read_text(encoding="utf-8")
tree = ast.parse(source)
screen = next(
    node for node in tree.body
    if isinstance(node, ast.ClassDef) and node.name == "ChatScreen"
)
methods = [
    node for node in screen.body
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
]
names = {
    "_first_chat_defaults_match",
    "_current_first_chat_defaults",
    "eligible_console_first_chat_session_id",
    "_release_first_chat_claim",
    "_log_first_chat_handoff_exception",
    "_resync_console_after_first_chat_rollback",
    "_resync_mounted_console_after_first_chat_rollback",
    "consume_pending_console_first_chat_intent",
}
family = [node for node in methods if node.name in names]
print(len(source.splitlines()), len(methods))
print(len(family), sum(node.end_lineno - node.lineno + 1 for node in family))
print([node.name for node in family])
PY
```

Expected at the approved base: `19995 640`, then `8 328`, then the exact ordered
family. This is immutable design evidence, not the final implementation baseline.

- [x] **Step 3: Run the rebased implementation focused baseline without changing source**

Run:

```bash
../../.venv/bin/python -B -m pytest -q \
  Tests/UI/test_console_session_settings.py \
  Tests/UI/test_console_controller_wiring.py \
  Tests/Wizards/test_first_run_setup_wizard.py::test_finalize_stages_exact_first_chat_after_successful_setup_mutation \
  Tests/Wizards/test_first_run_setup_wizard.py::test_first_chat_stage_failure_leaves_mounted_console_byte_exact \
  Tests/Wizards/test_first_run_setup_wizard.py::test_finalize_reserves_future_target_only_without_console_owner \
  Tests/Wizards/test_first_run_setup_wizard.py::test_finalize_reserves_future_target_when_mounted_console_is_ineligible \
  Tests/UI/test_first_run_wizard_live_contract.py::test_mounted_wizard_producer_to_console_consumer_preserves_user_work \
  Tests/UI/test_first_run_wizard_live_contract.py::test_mounted_wizard_stage_failure_leaves_console_and_focus_unchanged \
  Tests/UI/test_first_run_wizard_live_contract.py::test_mounted_wizard_generation_race_rolls_back_and_retries_intent \
  Tests/Architecture/test_console_wave6_inventory.py::test_wave6_inventory_matches_the_implementation_base \
  Tests/Architecture/test_console_wave6_inventory.py::test_wave6_compatibility_inventory_is_complete_and_phase_safe \
  Tests/Architecture/test_console_wave6_inventory.py::test_wave6_structural_oracles_are_non_vacuous
```

Expected: all selected tests pass. Record counts, warnings, and duration.

- [x] **Step 4: Record the rebased implementation non-write diagnostic baseline**

Run:

```bash
../../.venv/bin/python scripts/check_persistent_diagnostic_inventory.py
../../.venv/bin/python -B -m pytest -q \
  Tests/Architecture/test_persistent_diagnostic_inventory.py::test_reviewed_diagnostic_changes_are_metadata_only
```

Expected: the checker normally passes and the metadata-only node passes. An
inherited-red checker baseline may be accepted only when the exact frozen
`origin/dev` counterfactual reproduces the candidate failure byte-for-byte or
with an exactly equivalent owner/call/topology signature, the metadata-only
node passes on both trees, the task series contains no source change, and the
writer remains deferred to final reconciliation. Record the checker exit as
red and describe this only as non-regression evidence, never as a passing
checker. Do not run `--write` during baseline.

- [x] **Step 5: Verify the concise Backlog implementation plan**

Run:

```bash
backlog task edit 3070.9 --plan $'1. Freeze latest-dev baseline and focused evidence\n2. Lock controller ownership and behavior with RED tests\n3. Move exact first-chat policy into ConsoleSessionController\n4. Repair focused callers and prove mutation sensitivity\n5. Rebase, reconcile diagnostics once, run focused/static gates, and close the task'
```

Expected: task remains `In Progress`; all ACs remain unchecked.

- [x] **Step 6: Commit the reviewed plan**

Run:

```bash
git add Docs/superpowers/specs/2026-08-22-task-3070-9-console-first-chat-handoff-design.md \
  Docs/superpowers/plans/2026-08-22-task-3070-9-console-first-chat-handoff.md \
  'backlog/tasks/task-3070.9 - Extract-Console-first-chat-handoff-ownership.md'
git diff --cached --check
git commit -m "docs(console): plan first-chat ownership extraction"
```

- [x] **Step 7: Fetch, rebase, and collect the implementation baseline**

Only after Step 6 leaves a clean worktree, run:

```bash
git status --porcelain
git fetch origin dev
task30709_plan_pre_base=$(git merge-base HEAD origin/dev)
task30709_plan_pre_head=$(git rev-parse HEAD)
: "${task30709_plan_pre_base:?missing pre-rebase base}"
: "${task30709_plan_pre_head:?missing pre-rebase head}"
git rebase origin/dev
task30709_plan_post_base=$(git merge-base HEAD origin/dev)
task30709_plan_post_head=$(git rev-parse HEAD)
: "${task30709_plan_post_base:?missing post-rebase base}"
: "${task30709_plan_post_head:?missing post-rebase head}"
test "${task30709_plan_post_base}" = "$(git rev-parse origin/dev)"
git range-diff \
  "${task30709_plan_pre_base}..${task30709_plan_pre_head}" \
  "${task30709_plan_post_base}..${task30709_plan_post_head}"
```

Then repeat Steps 2–4 on the rebased candidate and record those results as the
implementation baseline. If the eight-method membership or behavior changed,
or a baseline gate is red, stop and amend/re-review the design and ratchet
before writing RED tests. The sole baseline exception is a non-write diagnostic
checker failure accepted under every Step 4 inherited-red condition; that is
non-regression evidence only and does not relax Task 5's final green diagnostic
requirements.

### Task 1: Lock the ownership and behavior contract with RED tests

**Files:**
- Modify: `Tests/UI/test_console_session_settings.py`
- Modify: `Tests/UI/test_console_controller_wiring.py`
- Modify: `Tests/Wizards/test_first_run_setup_wizard.py`
- Modify: `Tests/Architecture/test_console_wave6_inventory.py`

- [ ] **Step 1: Route existing first-chat behavior tests to the intended owner**

In `Tests/UI/test_console_session_settings.py`, add one helper and use it for the
existing first-chat tests:

```python
from unittest.mock import MagicMock

from loguru import logger as loguru_logger

import tldw_chatbook.UI.Console_Modules.session as session_module
from tldw_chatbook.UI.Console_Modules.session import ConsoleSessionController


def _first_chat_owner(console: ChatScreen) -> ConsoleSessionController:
    return console._session
```

Add the same `from unittest.mock import MagicMock` import to
`Tests/UI/test_console_controller_wiring.py` for its new late-binding test.

Replace policy calls such as:

```python
console.consume_pending_console_first_chat_intent()
console.eligible_console_first_chat_session_id()
```

with:

```python
_first_chat_owner(console).consume_pending_console_first_chat_intent()
_first_chat_owner(console).eligible_console_first_chat_session_id()
```

Move config-race monkeypatches from `chat_screen_module` to the owning
`session_module`. During the RED commit, pass `raising=False` to each owner
module patch because the moved imports do not exist until Task 2; this keeps RED
on missing controller ownership instead of fixture setup. After Task 2 the same
patches bind real module symbols. Move direct assertions on
`console._first_chat_handoff_notified_revision` to `console._session`.

Do not weaken any existing store snapshot, claim replacement, rollback,
notification, mounted projection, or focus assertion.

- [ ] **Step 2: Add explicit no-mount presentation-boundary assertions**

Extend the existing
`test_first_chat_consumer_activates_once_and_acknowledges_exact_target` after
its current `ChatScreen(app_instance)` and intent/store setup. The screen is
deliberately unmounted, so wrap the future owner callbacks without inventing a
new fixture:

```python
owner = console._session
real_apply = owner._apply_first_chat_control_selection_fn
presentation = MagicMock(side_effect=real_apply)
restore_focus = MagicMock()
owner._apply_first_chat_control_selection_fn = presentation
owner._restore_first_chat_focus_fn = restore_focus

assert owner._screen_mounted_accessor() is False
assert owner.consume_pending_console_first_chat_intent() is True
presentation.assert_called_once_with("llama_cpp", "local-a")
restore_focus.assert_not_called()
```

Use the real store and pending-handoff fixtures already present in this file.
The exact assertion is that non-DOM control mirror projection still occurs, but
no mounted focus callback runs. In
`test_first_chat_failed_acknowledgement_rolls_back_and_requeues`, wrap the same
apply callback and the real pending-handoff `release` in one event trace before
calling the consumer:

```python
events: list[tuple[object, ...]] = []
owner = console._session
real_apply = owner._apply_first_chat_control_selection_fn
real_release = app.pending_handoffs.release

def apply_and_record(provider, model) -> None:
    events.append(("project", provider, model))
    real_apply(provider, model)

def release_and_record(claim) -> bool:
    events.append(("release", claim.revision))
    return real_release(claim)

owner._apply_first_chat_control_selection_fn = apply_and_record
monkeypatch.setattr(app.pending_handoffs, "release", release_and_record)

assert owner.consume_pending_console_first_chat_intent() is False
assert [event[0] for event in events[:3]] == ["project", "project", "release"]
assert events[0][1:] == ("openai", "model-a")
assert events[1][1:] == (None, None)
```

Keep the existing store and pending-intent assertions after this trace. They
continue to prove rollback/retry outcome; the trace uniquely proves ordering.

Add this exact privacy test using Loguru's removable sink:

```python
def test_first_chat_exception_log_is_metadata_only() -> None:
    records: list[str] = []
    sink_id = loguru_logger.add(
        lambda message: records.append(str(message)),
        level="WARNING",
    )
    try:
        ConsoleSessionController._log_first_chat_handoff_exception(
            "guarded-acknowledgement",
            RuntimeError("SECRET-FIRST-CHAT-EXCEPTION"),
        )
    finally:
        loguru_logger.remove(sink_id)

    rendered = "".join(records)
    assert "guarded-acknowledgement" in rendered
    assert "RuntimeError" in rendered
    assert "SECRET-FIRST-CHAT-EXCEPTION" not in rendered
```

- [ ] **Step 3: Lock production wiring and late binding**

In `Tests/UI/test_console_controller_wiring.py`, add
`test_session_first_chat_edges_are_late_bound_and_presentation_only` using the
existing `_unmounted_console()` helper. The screen remains unmounted, so mounted
state must be `False`; late binding is proven by changing the presentation
values after controller construction:

```python
screen = _unmounted_console()
controller = screen._session
screen._console_control_provider = "late-provider"
screen._console_control_model = "late-model"
focus_token = MagicMock()
focus_token.is_mounted = True

assert controller._screen_mounted_accessor() is False
assert controller._first_chat_presentation_snapshot_fn() == (
    "late-provider",
    "late-model",
    None,
)
controller._apply_first_chat_control_selection_fn("next-provider", "next-model")
assert (screen._console_control_provider, screen._console_control_model) == (
    "next-provider",
    "next-model",
)
controller._restore_first_chat_focus_fn(focus_token)
focus_token.focus.assert_not_called()
```

Set `focus_token.is_mounted = True`; the restore helper must still decline it
because the screen is unmounted. Replace the screen fields after controller
construction so a captured bound value or eager snapshot fails. Mounted focus
late binding remains covered by the existing mounted rollback tests.

- [ ] **Step 4: Lock wizard routing to the controller owner**

In `Tests/Wizards/test_first_run_setup_wizard.py`, change the mounted Console
double to:

```python
session_owner = MagicMock()
session_owner.eligible_console_first_chat_session_id.return_value = "session-exact"
console = SimpleNamespace(_session=session_owner)
```

Assert the owner method is called once. Add a screen with no callable `_session`
owner to the reserve-new case so the producer remains fail-safe.

- [ ] **Step 5: Add completed-phase architecture oracles**

In `Tests/Architecture/test_console_wave6_inventory.py`, add:

```python
TASK_3070_9_IMPLEMENTATION_BASE = "0da426e1e4c2846f13671690b8f981f72e673359"
TASK_3070_9_BASE_LINES = 19_995
TASK_3070_9_BASE_METHODS = 640
TASK_3070_9_LINE_CEILING = 19_667
TASK_3070_9_METHOD_CEILING = 632


def test_first_chat_family_has_completed_controller_ownership() -> None:
    group = WAVE6_GROUPS["first_chat"]
    screen_methods = _methods(_SCREEN_PATH, "ChatScreen")
    owner_methods = _methods(_REPO_ROOT / group.target_path, group.target_class)
    assert group.moved.isdisjoint(screen_methods)
    assert group.moved <= owner_methods.keys()


def test_first_chat_task_ratchet_is_earned() -> None:
    source, screen = _class_node(_SCREEN_PATH, "ChatScreen")
    assert len(source.splitlines()) <= TASK_3070_9_LINE_CEILING
    assert _method_count(screen) <= TASK_3070_9_METHOD_CEILING
```

Pin `CURRENT_BASE_*` to the frozen implementation source without changing the
historical post-image constants. Count direct definitions with multiplicity,
not a unique-name dictionary. Also assert the moved methods do not call
`query`/`query_one`, access `_workspace`/other sibling owners, or define a
screen compatibility replacement.

Extend the existing synthetic non-vacuity fixture so it independently rejects:

- one moved method still on a synthetic `ChatScreen`;
- one DOM query in a moved session method;
- one sibling-controller reach-through;
- duplicate controller definitions;
- one forbidden compatibility delegate.

- [ ] **Step 6: Run the exact RED selection**

Run:

```bash
../../.venv/bin/python -B -m pytest -q \
  Tests/UI/test_console_session_settings.py \
  Tests/UI/test_console_controller_wiring.py::test_session_first_chat_edges_are_late_bound_and_presentation_only \
  Tests/Wizards/test_first_run_setup_wizard.py::test_finalize_stages_exact_first_chat_after_successful_setup_mutation \
  Tests/Wizards/test_first_run_setup_wizard.py::test_finalize_reserves_future_target_when_mounted_console_is_ineligible \
  Tests/Architecture/test_console_wave6_inventory.py::test_first_chat_family_has_completed_controller_ownership \
  Tests/Architecture/test_console_wave6_inventory.py::test_first_chat_controller_has_only_named_non_dom_dependencies \
  Tests/Architecture/test_console_wave6_inventory.py::test_first_chat_task_ratchet_is_earned \
  Tests/Architecture/test_console_wave6_inventory.py::test_first_chat_move_oracles_are_non_vacuous
```

Expected: collection succeeds; tests fail only because the methods/state and
callbacks are not yet on `ConsoleSessionController`, callers still target the
screen, and the ratchet has not been earned. Fix fixture/import errors now;
do not add production behavior in this task.

- [ ] **Step 7: Run targeted static checks and commit RED**

Run:

```bash
../../.venv/bin/ruff check \
  Tests/UI/test_console_session_settings.py \
  Tests/UI/test_console_controller_wiring.py \
  Tests/Wizards/test_first_run_setup_wizard.py \
  Tests/Architecture/test_console_wave6_inventory.py
../../.venv/bin/ruff format --check \
  Tests/UI/test_console_session_settings.py \
  Tests/UI/test_console_controller_wiring.py \
  Tests/Wizards/test_first_run_setup_wizard.py \
  Tests/Architecture/test_console_wave6_inventory.py
git diff --check
git add Tests/UI/test_console_session_settings.py \
  Tests/UI/test_console_controller_wiring.py \
  Tests/Wizards/test_first_run_setup_wizard.py \
  Tests/Architecture/test_console_wave6_inventory.py
git commit -m "test(console): lock first-chat ownership extraction"
```

### Task 2: Move first-chat ownership into ConsoleSessionController

**Files:**
- Modify: `tldw_chatbook/UI/Console_Modules/session.py`
- Modify: `tldw_chatbook/UI/Console_Modules/wiring.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py`
- Inspect/modify only if required: `Tests/Architecture/test_persistent_diagnostic_inventory.py`

- [ ] **Step 1: Add only the required session-module dependencies**

Move/import the existing types and helpers used by the eight bodies:

```python
from ..Navigation.pending_handoff_store import (
    ConsoleFirstChatIntent,
    HandoffChannel,
    PendingHandoffStore,
)
from ...config import (
    get_runtime_config_snapshot,
    run_if_runtime_config_generation_current,
)
from ...Chat.console_session_settings import (
    ConsoleSessionSettings,
    build_default_console_session_settings,
    # preserve the module's existing session-settings imports
)
```

Use the repository's actual import locations at implementation time. Reuse the
already imported `CONSOLE_GLOBAL_WORKSPACE_ID`, `ConsoleChatStore`,
`ConsoleSessionSettings`, `provider_config_key`, `logger`, and `Any`. Add
`build_default_console_session_settings` to the module's existing
`console_session_settings` import block. Remove those imports from
`chat_screen.py` only when no other screen consumer remains.

- [ ] **Step 2: Extend the constructor with four presentation seams**

Add exact keyword-only dependencies:

```python
screen_mounted_accessor: Callable[[], bool],
first_chat_presentation_snapshot: Callable[[], tuple[Any, Any, object | None]],
apply_first_chat_control_selection: Callable[[Any, Any], None],
restore_first_chat_focus: Callable[[object | None], None],
```

Store them without invocation:

```python
self._screen_mounted_accessor = screen_mounted_accessor
self._first_chat_presentation_snapshot_fn = first_chat_presentation_snapshot
self._apply_first_chat_control_selection_fn = apply_first_chat_control_selection
self._restore_first_chat_focus_fn = restore_first_chat_focus
self._first_chat_handoff_notified_revision: int | None = None
```

Do not add `Callable[..., Any]`, a screen/controller handle, or a new state
container.

Update `session.py`'s module and constructor docstrings so the ownership count
and moved-family inventory include the eight first-chat methods. Preserve the
existing documented `_screen` framework-service exception; do not claim the
whole legacy controller is screen-free.

- [ ] **Step 3: Move the exact eight method bodies**

Copy the eight reviewed bodies into `ConsoleSessionController`. Preserve branch
and side-effect order. Adapt only presentation access:

```python
prior_control_provider, prior_control_model, prior_focus_token = (
    self._first_chat_presentation_snapshot_fn()
)

self._apply_first_chat_control_selection_fn(
    target.settings.provider,
    target.settings.model,
)
if self._screen_mounted_accessor():
    self._sync_console_chat_core_state()
    self._sync_console_settings_summary()
    self._sync_console_control_bar()
```

Rollback applies the prior provider/model, performs the same three synchronous
syncs only when mounted, then schedules the existing async native-chat sync.
After that sync, call `_restore_first_chat_focus_fn(prior_focus_token)` only if
the screen is still mounted. The callback decides whether the token remains a
mounted focusable widget.

Preserve these exact boundaries:

- claim/config/default/store exceptions gain no new broad containment;
- notification/release/rollback/guarded-ack exceptions remain contained;
- rollback happens before release;
- replacement claims are never released or acknowledged;
- logging includes category and exception type, never exception text;
- `_first_chat_handoff_notified_revision` clears on successful ack and the
  existing non-current/replacement paths only.

- [ ] **Step 4: Add wiring-only presentation helpers and late-bound callbacks**

In `wiring.py`, add no more than these two helpers:

```python
def _apply_first_chat_control_selection(screen: Any, provider: Any, model: Any) -> None:
    screen._console_control_provider = provider
    screen._console_control_model = model


def _restore_first_chat_focus(screen: Any, token: object | None) -> None:
    if screen.is_mounted and getattr(token, "is_mounted", False):
        token.focus()
```

Bind:

```python
screen_mounted_accessor=lambda: screen.is_mounted,
first_chat_presentation_snapshot=lambda: (
    screen._console_control_provider,
    screen._console_control_model,
    screen.app.focused if screen.is_mounted else None,
),
apply_first_chat_control_selection=(
    lambda provider, model: _apply_first_chat_control_selection(
        screen, provider, model
    )
),
restore_first_chat_focus=lambda token: _restore_first_chat_focus(screen, token),
```

Every lambda resolves `screen` state at call time. Do not pass a bound method or
widget into the constructor.

- [ ] **Step 5: Delete screen ownership and rewire all production callers**

In `chat_screen.py`:

- delete all eight definitions;
- delete `_first_chat_handoff_notified_revision` initialization;
- change both consumption sites to
  `self._session.consume_pending_console_first_chat_intent()`;
- remove imports used only by the moved family.

In `FirstRunSetupWizard.py`, replace the direct method probe with:

```python
session_owner = getattr(screen, "_session", None)
eligible_session = getattr(
    session_owner,
    "eligible_console_first_chat_session_id",
    None,
)
```

Preserve the loop's continue/break behavior and reserve-new fallback.

- [ ] **Step 6: Inspect diagnostic ownership before touching its registry**

Run:

```bash
rg -n 'First-chat handoff operation failed|first.chat' \
  Tests/Architecture/test_persistent_diagnostic_inventory.py \
  Docs/security/production-diagnostic-inventory.json
```

If the reviewed metadata-only registry names the source owner, transfer the
unchanged label/field allowlist from `chat_screen.py` to `session.py`. Do not
write the generated manifest yet. If no reviewed registry entry exists, change
nothing here.

- [ ] **Step 7: Run GREEN and static checks**

Run the exact Task 1 selection.

Expected: all selected tests pass; `ChatScreen` owns none of the eight methods;
the controller owns each exactly once; measured screen totals satisfy the frozen
ceilings.

Then run:

```bash
../../.venv/bin/ruff check \
  tldw_chatbook/UI/Console_Modules/session.py \
  tldw_chatbook/UI/Console_Modules/wiring.py \
  tldw_chatbook/UI/Screens/chat_screen.py \
  tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py \
  Tests/Architecture/test_persistent_diagnostic_inventory.py
../../.venv/bin/ruff format --check \
  tldw_chatbook/UI/Console_Modules/session.py \
  tldw_chatbook/UI/Console_Modules/wiring.py \
  tldw_chatbook/UI/Screens/chat_screen.py \
  tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py \
  Tests/Architecture/test_persistent_diagnostic_inventory.py
git diff --check
```

Format only changed files whose formatter delta is caused by this task. Inspect
formatter diff before applying it; do not mix unrelated whole-file churn into
the behavior commit.

- [ ] **Step 8: Commit the ownership move**

Run:

```bash
git add tldw_chatbook/UI/Console_Modules/session.py \
  tldw_chatbook/UI/Console_Modules/wiring.py \
  tldw_chatbook/UI/Screens/chat_screen.py \
  tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py \
  Tests/Architecture/test_persistent_diagnostic_inventory.py
git diff --cached --check
git commit -m "refactor(console): extract first-chat handoff ownership"
```

Omit the diagnostic test path if it was not changed.

### Task 3: Repair focused callers and preserve mounted integration behavior

**Files:**
- Modify only as returned by the stale-owner scan: first-chat-focused tests
- Expected primary files: `Tests/UI/test_console_session_settings.py`,
  `Tests/Wizards/test_first_run_setup_wizard.py`
- Inspect: `Tests/UI/test_first_run_wizard_live_contract.py`

- [ ] **Step 1: Scan every moved name and state field**

Run:

```bash
rg -n '_first_chat_defaults_match|_current_first_chat_defaults|eligible_console_first_chat_session_id|_release_first_chat_claim|_log_first_chat_handoff_exception|_resync_console_after_first_chat_rollback|_resync_mounted_console_after_first_chat_rollback|consume_pending_console_first_chat_intent|_first_chat_handoff_notified_revision' \
  tldw_chatbook Tests
```

Classify every hit as controller ownership, direct `_session` production/test
call, architecture string, or stale screen call. Repair every stale executable
screen call; do not edit comments or unrelated fixtures without evidence.

- [ ] **Step 2: Run the focused behavior and mounted integration matrix**

Run:

```bash
../../.venv/bin/python -B -m pytest -q \
  Tests/UI/test_console_session_settings.py \
  Tests/UI/test_console_controller_wiring.py \
  Tests/Wizards/test_first_run_setup_wizard.py::test_finalize_stages_exact_first_chat_after_successful_setup_mutation \
  Tests/Wizards/test_first_run_setup_wizard.py::test_first_chat_stage_failure_leaves_mounted_console_byte_exact \
  Tests/Wizards/test_first_run_setup_wizard.py::test_finalize_does_not_stage_first_chat_when_setup_mutation_fails \
  Tests/Wizards/test_first_run_setup_wizard.py::test_finalize_reserves_future_target_only_without_console_owner \
  Tests/Wizards/test_first_run_setup_wizard.py::test_finalize_reserves_future_target_when_mounted_console_is_ineligible \
  Tests/UI/test_first_run_wizard_live_contract.py::test_mounted_wizard_producer_to_console_consumer_preserves_user_work \
  Tests/UI/test_first_run_wizard_live_contract.py::test_mounted_wizard_stage_failure_leaves_console_and_focus_unchanged \
  Tests/UI/test_first_run_wizard_live_contract.py::test_mounted_wizard_generation_race_rolls_back_and_retries_intent
```

Expected: all selected tests pass. Preserve real mounted `pilot`/focus behavior;
do not replace it with direct widget mutation.

- [ ] **Step 3: Run exact static checks on changed tests**

Run Ruff check, Ruff format-check, and `git diff --check` on every test returned
by the scan. If a whole-file format check is inherited-red, compare the base
blob and prove this task introduced no formatter delta before deciding whether
to make a separate mechanical formatting commit.

- [ ] **Step 4: Commit caller repairs if any remain**

Run:

```bash
git add Tests/UI/test_console_session_settings.py \
  Tests/UI/test_console_controller_wiring.py \
  Tests/Wizards/test_first_run_setup_wizard.py \
  Tests/UI/test_first_run_wizard_live_contract.py
git diff --cached --check
git commit -m "test(console): cover first-chat session controller"
```

Skip this commit if Task 1 already contains every required test-only change.

### Task 4: Prove mutation sensitivity and exact restoration

**Files:**
- Temporarily modify and exactly restore:
  `tldw_chatbook/UI/Console_Modules/session.py`
- Test: `Tests/UI/test_console_session_settings.py`
- Test: `Tests/Architecture/test_console_wave6_inventory.py`

- [ ] **Step 1: Record the clean checksum boundary**

Run:

```bash
git status --porcelain
git diff --binary -- tldw_chatbook/UI/Console_Modules/session.py | shasum -a 256
git rev-parse HEAD
```

Expected: clean status and the empty-diff SHA-256.

- [ ] **Step 2: Mutate the configuration/active-session fence**

Use `apply_patch` to remove one required `fence_matches` conjunct. Run the exact
race test that distinguishes it, for example:

```bash
../../.venv/bin/python -B -m pytest -q \
  Tests/UI/test_console_session_settings.py::test_first_chat_consumer_refuses_session_switch_and_config_generation_races
```

Expected RED: the target is mutated/acknowledged or the pending intent is lost.
Apply the exact inverse patch and rerun GREEN.

- [ ] **Step 3: Mutate rollback-before-release ordering**

Temporarily change `rollback_and_release` to call `_release_first_chat_claim`
before `rollback_mutation`. Run:

```bash
../../.venv/bin/python -B -m pytest -q \
  Tests/UI/test_console_session_settings.py::test_first_chat_failed_acknowledgement_rolls_back_and_requeues
```

Expected RED: the shared event trace begins with `release` before prior-state
projection. Restore and rerun GREEN.

- [ ] **Step 4: Mutate replacement safety**

Temporarily remove the exact-current-claim guard around release or
acknowledgement. Run:

```bash
../../.venv/bin/python -B -m pytest -q \
  Tests/UI/test_console_session_settings.py::test_first_chat_replacement_and_session_switch_during_create_roll_back_old_target \
  Tests/UI/test_console_session_settings.py::test_first_chat_ack_exception_after_replacement_preserves_replacement
```

Expected RED: the replacement claim is released or acknowledged. Restore and
rerun GREEN.

- [ ] **Step 5: Mutate guarded acknowledgement**

Temporarily treat a false guarded acknowledgement as success. Run:

```bash
../../.venv/bin/python -B -m pytest -q \
  Tests/UI/test_console_session_settings.py::test_first_chat_failed_acknowledgement_rolls_back_and_requeues
```

Expected RED: the test observes missing rollback/retry state. Restore and rerun
GREEN.

- [ ] **Step 6: Mutate privacy-safe logging**

Temporarily include `str(exc)` in the first-chat warning. Run the focused privacy
assertion added to `test_console_session_settings.py`.

```bash
../../.venv/bin/python -B -m pytest -q \
  Tests/UI/test_console_session_settings.py::test_first_chat_exception_log_is_metadata_only
```

Expected RED: secret/sentinel exception content appears. Restore and rerun GREEN.

- [ ] **Step 7: Mutate mounted focus restoration**

Temporarily remove `_restore_first_chat_focus_fn` after async resync. Run:

```bash
../../.venv/bin/python -B -m pytest -q \
  Tests/UI/test_console_session_settings.py::test_mounted_first_chat_ack_exception_during_resume_restores_ui
```

Expected RED: prior focus is not restored after repaint. Restore and rerun GREEN.

- [ ] **Step 8: Prove structural oracles independently**

Run:

```bash
../../.venv/bin/python -B -m pytest -q \
  Tests/Architecture/test_console_wave6_inventory.py::test_first_chat_move_oracles_are_non_vacuous
```

Expected: pass; the committed synthetic fixture independently exercises screen
ownership, DOM, sibling reach-through, duplicate definition, and compatibility
delegate failures.

- [ ] **Step 9: Prove exact restoration**

After every mutation and at the end, require:

```bash
git diff --binary -- tldw_chatbook/UI/Console_Modules/session.py | shasum -a 256
git diff --check
rg -n 'TASK3070_MUTATION' tldw_chatbook Tests || true
git status --porcelain
```

Expected: checksum equals the clean boundary; no token residue; worktree clean;
HEAD unchanged. Mutation probes produce no commit.

### Task 5: Rebase, reconcile diagnostics, verify, and close out

**Files:**
- Modify after reviewed reconciliation only:
  `Docs/security/production-diagnostic-inventory.json`
- Modify: `Docs/superpowers/plans/2026-08-22-task-3070-9-console-first-chat-handoff.md`
- Modify: `backlog/tasks/task-3070.9 - Extract-Console-first-chat-handoff-ownership.md`

- [ ] **Step 1: Fetch and rebase onto the latest dev**

Run:

```bash
git status --porcelain
git fetch origin dev
task30709_pre_base=$(git merge-base HEAD origin/dev)
task30709_pre_head=$(git rev-parse HEAD)
: "${task30709_pre_base:?missing pre-rebase base}"
: "${task30709_pre_head:?missing pre-rebase head}"
git rebase origin/dev
task30709_post_base=$(git merge-base HEAD origin/dev)
task30709_post_head=$(git rev-parse HEAD)
: "${task30709_post_base:?missing post-rebase base}"
: "${task30709_post_head:?missing post-rebase head}"
test "${task30709_post_base}" = "$(git rev-parse origin/dev)"
git range-diff \
  "${task30709_pre_base}..${task30709_pre_head}" \
  "${task30709_post_base}..${task30709_post_head}"
```

Expected: clean rebase, merge base equals the frozen latest `origin/dev`, and
range-diff preserves task commits. If the eight-method source family changed,
stop and amend/re-review the design and ratchet before continuing.

- [ ] **Step 2: Remeasure ownership and ratchet**

Rerun Task 0 Step 2 against the rebased tree and the frozen base blob. Assert:

- every reviewed name occurs exactly once in the rebased base screen;
- every reviewed name occurs zero times on candidate `ChatScreen`;
- every reviewed name occurs exactly once on `ConsoleSessionController`;
- the family AST/body content is preserved except the reviewed callback
  substitutions;
- final line/method totals earn the exact rebased reduction without increasing
  a ceiling.

- [ ] **Step 3: Repeat mutation proof on the final rebased candidate**

Repeat every Task 4 mutation and exact inverse against the rebased candidate,
even when range-diff reports no conflict. Record each final RED discriminator,
restored GREEN result, identical pre/post binary-diff SHA-256, empty residue
scan, clean status, and unchanged HEAD. A pre-rebase mutation run is development
evidence only; closeout uses this post-rebase run.

- [ ] **Step 4: Preview diagnostic reconciliation without writing**

Run this exact read-only preview. It writes only under a validated temporary
directory and does not invoke the canonical writer:

```bash
../../.venv/bin/python - <<'PY'
from __future__ import annotations

import difflib
import hashlib
import io
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path

from scripts import check_persistent_diagnostic_inventory as inventory

repo = Path.cwd().resolve()
checked_path = repo / "Docs/security/production-diagnostic-inventory.json"
base_revision = subprocess.run(
    ["git", "rev-parse", "origin/dev"],
    cwd=repo,
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
assert base_revision

checked_text = checked_path.read_text(encoding="utf-8")
candidate_inventory = inventory.build_inventory()
candidate_text = inventory._encoded(candidate_inventory)
archive = subprocess.run(
    ["git", "archive", base_revision, "tldw_chatbook"],
    cwd=repo,
    check=True,
    capture_output=True,
).stdout

with tempfile.TemporaryDirectory(
    prefix="task-3070-9-diagnostic-preview.",
    dir="/private/tmp",
) as temporary_root:
    root = Path(temporary_root)
    assert root.is_dir() and not root.is_symlink()
    assert root.stat().st_uid == os.getuid()
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
        bundle.extractall(root, filter="data")
    assert all(not path.is_symlink() for path in root.rglob("*"))

    inventory.REPO_ROOT = root
    inventory.PACKAGE_ROOT = root / "tldw_chatbook"
    base_inventory = inventory.build_inventory()
    base_text = inventory._encoded(base_inventory)

    artifacts = {
        "checked": checked_text,
        f"base-{base_revision[:12]}": base_text,
        "candidate": candidate_text,
    }
    for label, content in artifacts.items():
        artifact = root / f"{label}.json"
        artifact.write_text(content, encoding="utf-8")
        assert artifact.is_file() and not artifact.is_symlink()
        print(label, hashlib.sha256(content.encode()).hexdigest())

    for left_label, right_label in (
        ("checked", f"base-{base_revision[:12]}"),
        (f"base-{base_revision[:12]}", "candidate"),
    ):
        print(f"--- {left_label} -> {right_label} ---")
        print(
            "".join(
                difflib.unified_diff(
                    artifacts[left_label].splitlines(keepends=True),
                    artifacts[right_label].splitlines(keepends=True),
                    fromfile=left_label,
                    tofile=right_label,
                )
            )
        )

    owner_pair = (
        "UI/Screens/chat_screen.py",
        "UI/Console_Modules/session.py",
    )

    def combined_calls(package_root: Path) -> list[tuple[str, str]]:
        calls: list[tuple[str, str]] = []
        for relative in owner_pair:
            path = package_root / relative
            diagnostics, _ = inventory.scan_source(
                path.read_text(encoding="utf-8"),
                filename=str(path),
            )
            calls.extend(
                (entry["method"], entry["digest"]) for entry in diagnostics
            )
        return sorted(calls)

    base_calls = combined_calls(root / "tldw_chatbook")
    candidate_calls = combined_calls(repo / "tldw_chatbook")
    assert base_calls == candidate_calls
    assert (
        base_inventory["persistent_sink_topology"]
        == candidate_inventory["persistent_sink_topology"]
    )
    print("base_revision", base_revision)
    print("chat_screen+session diagnostic multiset: exact")
    print("persistent sink topology: exact")
PY
```

Expected: checked manifest -> rebased `origin/dev` output classifies inherited
drift; rebased base -> candidate contains only the reviewed first-chat logger
owner move from `chat_screen.py` to `session.py`; the combined `(method,
digest)` multiset and complete persistent-sink topology are exact.

If any assertion fails, stop. Do not run the writer.

- [ ] **Step 5: Write the canonical diagnostic inventory exactly once**

Only after Step 4 passes, run:

```bash
../../.venv/bin/python scripts/check_persistent_diagnostic_inventory.py --write
git diff -- Docs/security/production-diagnostic-inventory.json
../../.venv/bin/python scripts/check_persistent_diagnostic_inventory.py
../../.venv/bin/python -B -m pytest -q \
  Tests/Architecture/test_persistent_diagnostic_inventory.py::test_production_diagnostic_inventory_and_sink_topology_are_unchanged \
  Tests/Architecture/test_persistent_diagnostic_inventory.py::test_reviewed_diagnostic_changes_are_metadata_only
```

Expected: the diff is only the reviewed owner redistribution plus separately
classified inherited latest-dev reconciliation; checker and both tests pass.

- [ ] **Step 6: Run final affected-only behavior gates**

Run the Task 3 matrix plus:

```bash
../../.venv/bin/python -B -m pytest -q \
  Tests/Architecture/test_console_wave6_inventory.py::test_wave6_inventory_matches_the_implementation_base \
  Tests/Architecture/test_console_wave6_inventory.py::test_wave6_compatibility_inventory_is_complete_and_phase_safe \
  Tests/Architecture/test_console_wave6_inventory.py::test_first_chat_family_has_completed_controller_ownership \
  Tests/Architecture/test_console_wave6_inventory.py::test_first_chat_controller_has_only_named_non_dom_dependencies \
  Tests/Architecture/test_console_wave6_inventory.py::test_first_chat_task_ratchet_is_earned \
  Tests/Architecture/test_console_wave6_inventory.py::test_first_chat_move_oracles_are_non_vacuous
```

Expected: all selected tests pass with no skips or xfails. Do not claim a red or
skipped node passed. A baseline-identical failure is evidence of non-regression,
not a passing gate; either amend the completion contract transparently or leave
the task open.

- [ ] **Step 7: Run final changed-file static and scope gates**

Enumerate changed Python files from the rebased merge base, then run Ruff check
and Ruff format-check on every one. Compile changed production modules into a
validated temporary directory, never the source tree:

```python
import py_compile
from pathlib import Path
from tempfile import TemporaryDirectory

paths = [
    Path("tldw_chatbook/UI/Console_Modules/session.py"),
    Path("tldw_chatbook/UI/Console_Modules/wiring.py"),
    Path("tldw_chatbook/UI/Screens/chat_screen.py"),
    Path("tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py"),
]
with TemporaryDirectory(prefix="task-3070-9-compile.") as root:
    root_path = Path(root)
    for index, path in enumerate(paths):
        py_compile.compile(path, cfile=str(root_path / f"{index}.pyc"), doraise=True)
```

Also require:

```bash
git diff --check
git status --short
rg -n 'eligible_console_first_chat_session_id|consume_pending_console_first_chat_intent|_first_chat_handoff_notified_revision' tldw_chatbook Tests
```

Inspect all changed files for conflict markers, secrets, generated artifacts,
fallback shims, stale screen calls, and unrelated scope.

- [ ] **Step 8: Run repository-wide pre-PR tests and linting**

Run the repository-mandated full gates:

```bash
../../.venv/bin/python -B -m pytest
../../.venv/bin/ruff check .
../../.venv/bin/ruff format --check .
```

Expected: all tests pass with no unexpected skips/xfails, and both Ruff commands
pass. Record counts and durations. A baseline-identical failure proves
non-regression but does not satisfy this gate; keep the task open unless the
failure is fixed in authorized scope or the user explicitly approves and the
task/spec transparently records a completion-contract exception.

- [ ] **Step 9: Self-review and independent reviews**

Review cumulative `origin/dev...HEAD` for exact behavior/ordering, privacy,
late binding, callback typing, DOM/sibling reach-through, wizard fallback,
formatter churn, and truthful evidence. Request independent specification and
quality reviews. Fix only verified findings with focused RED/GREEN evidence.

- [ ] **Step 10: Close the task only after every gate is satisfied**

Update this plan with exact RED/GREEN/mutation/test/static/diagnostic/rebase
evidence. Then use Backlog CLI:

```bash
backlog task edit 3070.9 \
  --check-ac 1 --check-ac 2 --check-ac 3 \
  --notes $'Moved the exact first-chat claim, fencing, rollback, acknowledgement, retry, and privacy policy into ConsoleSessionController; ChatScreen now supplies only late-bound presentation and focus callbacks. Rewired mount/resume and first-run wizard callers, strengthened ownership/wiring/mutation coverage, and reconciled the diagnostic inventory after the final rebase. Changed production and test files are enumerated with exact RED/GREEN, mutation, focused/full-suite, Ruff/format, compile, diagnostic, ratchet, and review evidence in the linked implementation plan. ADR required: no; this implements the approved Wave 6 ownership boundary.' \
  -s Done
backlog task 3070.9 --plain
```

Expected: all three ACs checked, Implementation Notes present, status `Done`.
Do not mark Done if a task-caused failure, skip, diagnostic mismatch, or review
finding remains.

- [ ] **Step 11: Commit closeout and verify clean state**

Run:

```bash
git add Docs/security/production-diagnostic-inventory.json \
  Docs/superpowers/plans/2026-08-22-task-3070-9-console-first-chat-handoff.md \
  'backlog/tasks/task-3070.9 - Extract-Console-first-chat-handoff-ownership.md'
git diff --cached --check
git commit -m "docs(console): close first-chat ownership task"
git status --short --branch
```

Expected: clean worktree/staging and only branch tracking output.

- [ ] **Step 12: Push, open the PR, address review, and merge**

Push the rebased branch with lease if necessary, create a ready PR, wait for
Qodo, verify every top-level and inline finding against current code, apply only
technically valid minimal fixes, reply in original threads, rerun affected gates,
and merge only when the branch is current, required checks are green, all review
feedback is addressed, and GitHub permits the merge. The full suite does not
replace the specified focused behavior, architecture, mutation, diagnostic, and
changed-file static evidence; both layers are required.
