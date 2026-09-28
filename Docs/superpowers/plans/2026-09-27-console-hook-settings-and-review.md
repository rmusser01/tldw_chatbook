# Console Hook Settings and Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Dispatch subagents only after the user chooses that execution option.

**Goal:** Give Console users persistent, exact-definition hook consent, a native review modal on next Send, and a staged editor in canonical Settings.

**Architecture:** Keep the existing hook executor and config writer. Add one application-owned consent owner, a small Console controller, one shared review modal, and a focused Settings panel. Enforce admission outside the view and serialize consent checks with process creation.

**Tech Stack:** Python >=3.12, Textual >=8.0.0,<9, existing pytest/asyncio, toml, portalocker, and private-path utilities; no new dependency or database migration.

**Spec:** [Reviewed design](../specs/2026-09-27-console-hook-settings-and-review-design.md)

**Backlog:** [TASK-33163](../../../backlog/tasks/task-33163%20-%20Add-Console-hook-settings-and-consent-review.md)

**ADR required:** no

**ADR path:** [ADR-197](../../../backlog/decisions/197-console-hook-configuration-review.md)

**Reason:** Direct implementation of accepted ADR-197; preserve ADR-148 execution, ADR-033 Settings commits, and ADR-150 tokens.

## Global Constraints

- Standalone user-config hooks only: UserPromptSubmit, PreToolUse, PostToolUse, ApprovalRequested, Stop, SubagentStop. No plugin/v2/project-hook implementation.
- Automatic review opens on next actual Send before accepting/queueing its message. Existing enabled hooks receive one-time review; approvals persist across restarts.
- Unselected hooks remain pending. Cancel/Escape/Settings navigation preserves the draft and cancels continuation. Only explicit disable removes an enabled row from the requirement.
- Hook permission never grants tool permission. Keep deny-only guards, manual-only UserPromptSubmit, bounded pools/capture, timeout cleanup, and execution-error fail directions.
- Consent failure is a separate typed refusal; it cannot inherit UserPromptSubmit's generic exception fail-open.
- Config lock precedes consent lock. Neither the Textual thread nor the engine pool-admission lock waits for disk/config/consent locks. Release launch locks before waiting for output.
- Stage Settings configuration through Save/Revert. Modal disable/revoke is explicitly immediate and fences locally before persistence. Report post-replacement refresh failure truthfully.
- Preserve raw invalid entries, unknown fields, unrelated sections, and existing queue/durable custody. Do not merge hook arrays or trust `app.app_config` as the authoritative saved section.
- Exact-definition consent does not sign executable contents or kill already-running hooks. Same-user arbitrary processes and old application versions are outside this runtime boundary.
- Use `$ds-*` tokens and existing classes. Edit CSS source modules and rebuild generated outputs; never edit generated sheets directly. No new terminal-convention shortcuts.
- Run targeted tests and isolated live verification. Never execute the user's real hooks or run the full suite without explicit authorization.

## Execution setup and file boundaries

The inspected checkout is `/Users/macbook-dev/Documents/GitHub/tldw_chatbook` and contains unrelated changes. The plan's file names below are relative to the executor's checkout; commands run from that checkout.

- [ ] Read TASK-33163, the spec, ADR-197, `backlog/docs/design-language.md`, `backlog/docs/lessons-testing-evidence.md`, `backlog/docs/lessons-live-verification.md`, and `backlog/docs/lessons-console-wiring.md` before editing.
- [ ] Inspect this chat's managed worktrees with `list_artifacts`; reuse a suitable free checkout or use `create_worktree` from the committed plan baseline. Use a `codex/` branch. Do not copy or commit unrelated shared changes.
- [ ] Recheck the named symbols on the selected baseline. The plan was informed by a dirty checkout; line numbers are not authoritative. Preserve any newer caller contracts.
- [ ] Reuse the existing Python environment without installing into it. If the worktree has no `.venv`, create an ignored symlink to the checked environment, then prove imports resolve from this checkout.

```bash
# Machine-agnostic: resolve any existing tldw_chatbook checkout's environment
# (the environment's site-packages carries the pinned deps; the import check
# below proves this checkout's sources are the ones imported). Qodo round:
# the previous snippet hard-coded one developer's absolute path.
python3 - <<'PY'
from pathlib import Path
candidates = [
    p / ".venv"
    for p in [Path.cwd(), *Path.cwd().parents]
    if (p / ".venv" / "bin" / "python").exists()
]
assert candidates, "No .venv found in this checkout or its parents"
PY
.venv/bin/python -c 'from pathlib import Path; import tldw_chatbook; assert Path(tldw_chatbook.__file__).resolve().is_relative_to(Path.cwd().resolve())'
```

```bash
git status --short
rg -n 'RunHooksEngine\(|load_hooks_config|ensure_run_hooks' tldw_chatbook Tests/Agents/test_run_hooks.py
rg -n 'submit_draft\(|queue_prompt\(|_dispatch_console_draft_send' tldw_chatbook/Chat tldw_chatbook/UI
rg -n 'action_settings_save_category|action_settings_revert_category|_render_detail_pane' tldw_chatbook/UI/Screens/settings_screen.py
```

| File | Responsibility |
| --- | --- |
| `tldw_chatbook/Agents/run_hooks.py` | Existing execution/protocol; lossless inventory, stable targets, typed consent refusal, launch guard. |
| `tldw_chatbook/config.py` | Locked hooks-only snapshots and compare/replace through the existing writer. |
| `tldw_chatbook/Agents/hook_permissions.py` (new) | One concrete consent owner, private JSON storage, reconciliation, review decisions and launch authority. |
| `tldw_chatbook/Chat/console_runtime.py` | Singleton owner composition/disposal and viewless controller wiring. |
| `tldw_chatbook/Chat/console_chat_controller.py` | Shared pre-acceptance gate and preserved refusal/recovery behavior. |
| `tldw_chatbook/UI/Console_Modules/prompt_queue.py` | Await shared queue admission; retain existing draft commits and typed outcomes. |
| `tldw_chatbook/UI/Console_Modules/hooks.py` (new) | DOM-free review/Send operation ownership and current-session/draft checks. |
| `tldw_chatbook/UI/Console_Modules/wiring.py` | Named late-binding dependencies for the Console controller. |
| `tldw_chatbook/Widgets/Console/console_hooks_review_modal.py` (new) | Shared native modal, rows, actions and safe dismissal; no persistence implementation. |
| `tldw_chatbook/UI/Screens/settings_hooks.py` (new) | Focused Hooks panel and draft helpers; canonical Settings owns its lifecycle. |
| Existing Console/Settings registries and CSS source | Small action/category/search/impact integrations, followed by generated CSS rebuild. |

Keep consent storage and policy in one file/class; do not add a generic permission framework, store interface, plugin adapter, watcher, or factory.

## Task 1: Lossless hook identity and config transactions

**Files:** Modify `Agents/run_hooks.py` and `config.py`. Create `Tests/Agents/test_hook_config_inventory.py` and `Tests/test_hooks_config_snapshot.py`. Extend `Tests/Agents/test_run_hooks.py` only for changed parser behavior.

**Interfaces produced:**

- Keep the existing four-field `HookSpec` constructor unchanged.
- `HookInventoryRow(index: int, key: str, spec: HookSpec | None, enabled: bool | None, error: str | None)` retains source position for editing, never for grants.
- `HookInventory(master_enabled: bool | None, container_error: str | None, rows: tuple[HookInventoryRow, ...])` exposes `requires_authority: bool`. A verified absent/empty section or valid master false needs no authority; malformed enabled state does.
- `inspect_hooks_config(config: Mapping[str, object]) -> HookInventory` validates the same execution fields as today's parser plus optional ID/enable fields. Duplicate explicit IDs invalidate all colliding rows.
- `fingerprint_hook(spec: HookSpec) -> str` computes the versioned execution fingerprint.
- `HookConfigSnapshot(config_path: Path, section_present: bool, section: object, section_stamp: str)` contains a detached, non-logging raw section.
- `locked_hooks_config_snapshot() -> ContextManager[HookConfigSnapshot]` owns the config lock through its caller's critical section; `read_hooks_config_snapshot() -> HookConfigSnapshot` returns a detached snapshot.
- `replace_hooks_config_snapshot(expected: HookConfigSnapshot, replacement: Mapping[str, object]) -> LiteralConfigMutationResult` compares effective path and original section under the existing transaction lock. It replaces the complete hook list and returns the existing structured outcome.

- [ ] **Step 1: Add the failing identity/inventory tests.** Use the existing HookSpec and new functions; no subprocess is needed for parsing.

```python
from tldw_chatbook.Agents.run_hooks import (
    HookSpec, fingerprint_hook, inspect_hooks_config,
)

def test_disabled_invalid_row_is_visible_without_execution_authority():
    inventory = inspect_hooks_config({"hooks": {"hook": [
        {"id": "one", "enabled": False, "event": "unknown", "command": "bad"}
    ]}})
    assert len(inventory.rows) == 1
    assert inventory.rows[0].error
    assert not inventory.requires_authority

def test_fingerprint_ignores_numeric_spelling_but_preserves_arguments():
    first = HookSpec("PreToolUse", ("python3", "guard.py", "a b"), "fs_*", 5)
    equivalent = HookSpec("PreToolUse", first.command, "fs_*", 5.0)
    changed = HookSpec("PreToolUse", ("python3", "guard.py", "a", "b"), "fs_*", 5)
    assert fingerprint_hook(first) == fingerprint_hook(equivalent)
    assert fingerprint_hook(first) != fingerprint_hook(changed)
```

- [ ] **Step 2: Run the red checks.** Expected failure is a missing new import, not an environment failure.

```bash
.venv/bin/python -m pytest Tests/Agents/test_hook_config_inventory.py -q
```

- [ ] **Step 3: Add the inventory and deterministic fingerprint.** Refactor `_parse_hook` into a reusable validation result and preserve its sanitized warning wrapper. `load_hooks_config` projects only valid enabled rows. Keep invalid rows in inventory, even when master false. Explicit false suppresses execution despite an invalid command; non-boolean switches remain invalid. Keep the current case-sensitive glob matcher.

```python
import hashlib
import json

def fingerprint_hook(spec: HookSpec) -> str:
    encoded = json.dumps(
        {"version": 1, "event": spec.event, "command": list(spec.command),
         "matcher": spec.matcher, "timeout_s": float(spec.timeout_s)},
        ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
```

Use `id:<explicit-id>` keys or `legacy:<fingerprint>:<duplicate-occurrence>` keys; reserve separate invalid-row keys. Never derive authority from a display label or array index. Add parametrized tests for every execution field, duplicate IDs, malformed master/table/list, absent versus empty sections, NUL, bool/nonfinite timeout and matcher misuse.

- [ ] **Step 4: Add locked snapshot/save tests, then the config functions.** Use `TLDW_CONFIG_PATH` and `tmp_path` under the existing isolated test fixture. Compare a section stamp based on presence and its stable TOML representation; this handles NaN in invalid originals without Python NaN equality. Do not include unrelated provider changes in this stamp.

```python
import toml
from tldw_chatbook import config

def test_hooks_save_rejects_stale_section_without_replacing_file(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text(toml.dumps({"hooks": {"enabled": True, "hook": []}}))
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(path))
    original = config.read_hooks_config_snapshot()
    path.write_text(toml.dumps({"hooks": {"enabled": False, "hook": []}}))
    before = path.read_bytes()
    result = config.replace_hooks_config_snapshot(
        original, {"enabled": True, "hook": []},
    )
    assert not result.file_replaced
    assert path.read_bytes() == before
```

Reuse `apply_literal_settings_transaction_to_cli_config` and its authoritative `raw_values` builder. Compare path/section in that lock, retain unknown keys, and assign the whole `hook` list at once. Do not add a hooks revision-owned section. Explicit repair of a malformed non-table section belongs to guarded Advanced Config; the guided editor must not discard it automatically. Test unchanged-revision raw edits, unrelated-section preservation, unknown fields, reassigned config path, source/draft deep-copy separation and post-write refresh failure.

- [ ] **Step 5: Run green checks, review, and commit this unit.**

```bash
.venv/bin/python -m pytest Tests/Agents/test_hook_config_inventory.py Tests/Agents/test_run_hooks.py Tests/test_hooks_config_snapshot.py Tests/test_config_raw_snapshot.py -q
git diff --check
git add tldw_chatbook/Agents/run_hooks.py tldw_chatbook/config.py Tests/Agents/test_hook_config_inventory.py Tests/Agents/test_run_hooks.py Tests/test_hooks_config_snapshot.py
git commit -m "feat: add lossless hook identity and guarded config saves"
```

## Task 2: Persistent consent and serialized process launch

**Files:** Create `Agents/hook_permissions.py` and `Tests/Agents/test_hook_permissions.py`. Modify `Agents/run_hooks.py`, `Chat/console_runtime.py`, relevant engine tests, and sensitive-path tests. Reuse `Utils/private_paths.py` without weakening it.

**Consumes:** Task 1 inventory, fingerprints, snapshots and config transaction.

**Interfaces produced:**

- `HookTarget(config_scope: str, key: str, fingerprint: str, spec: HookSpec, approval_token: str | None)` is an immutable execution target. The token identifies the store instance and grant epoch; stale queued work cannot revive after revoke/reapprove or an observed disable.
- `HookLaunchRefused(reason: str, skip: bool = False)` is defined with the engine types. It is handled explicitly before generic execution errors; observers omit, pending blocking guards refuse.
- `HookReviewRow(entry: HookInventoryRow | None, state: str, change: str)` uses states `approved`, `pending`, `disabled`, `invalid`, `recovery` and change labels `Existing`, `New`, `Modified`. A global container/store recovery row has no entry and cannot be selected.
- `HookReviewSnapshot(config: HookConfigSnapshot, rows: tuple[HookReviewRow, ...], store_path: Path, store_revision: tuple[str, int], blocked_reason: str | None, notice: str | None = None)` exposes `ready` and `pending_count` properties. `ready` means `blocked_reason is None`; pending enabled definitions, invalid enabled state and live recovery fences must supply a bounded static reason. Use `("", 0)` for an unusable store; it is never an approval revision. Error/recovery rows needing action count as attention, not approvable definitions. A mutation's non-persisted `notice` reports the actual saved/refresh outcome, including a successful config replacement followed by failed publication.
- `default_hook_permissions_path() -> Path` resolves the canonical current user data directory plus `hook_permissions.json` at call time.
- One concrete `HookPermissions` owner exposes `snapshot() -> HookReviewSnapshot`, `approve(expected: HookReviewSnapshot, keys: Collection[str]) -> HookReviewSnapshot`, `revoke(expected: HookReviewSnapshot, key: str) -> HookReviewSnapshot`, `disable(expected: HookReviewSnapshot, key: str) -> HookReviewSnapshot`, `recover() -> HookReviewSnapshot`, and `reset_invalid_state(expected: HookReviewSnapshot) -> HookReviewSnapshot`.
- Execution methods are `targets(event: str, tool_name: str | None) -> tuple[HookTarget, ...]`, `notification_targets(event: str, tool_name: str | None) -> tuple[HookTarget, ...]`, `launch_guard(target: HookTarget, *, tool_name: str | None) -> ContextManager[None]`, and `close() -> None`. `targets` refreshes off-thread; `notification_targets` pins the latest published immutable inventory without disk I/O on the emitting UI thread. Launch always revalidates authoritative state, including current pending/malformed guards affecting that event and tool.
- `HookReviewConflict` reports a stale file/definition/store decision without raw values. `ConsoleRuntime.ensure_hook_permissions() -> HookPermissions` owns the singleton.

- [ ] **Step 1: Add restart and legacy-duplicate regressions.** Define this shared fixture in `Tests/Agents/test_hook_permissions.py`; UI/runtime tests can import it explicitly.

```python
import sys
import toml
import pytest
from tldw_chatbook import config
from tldw_chatbook.Agents.hook_permissions import HookPermissions

@pytest.fixture
def hook_file(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir(mode=0o700)
    monkeypatch.setattr(config, "get_user_data_dir", lambda: data)
    path = tmp_path / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(path))
    path.write_text(toml.dumps({"hooks": {"hook": [{
        "id": "one", "event": "PostToolUse",
        "command": [sys.executable, "-c", "pass"], "timeout_s": 5,
    }]}}))
    return path

def test_consent_survives_a_new_owner(hook_file):
    owner = HookPermissions()
    pending = owner.snapshot()
    assert not pending.ready
    owner.approve(pending, [pending.rows[0].entry.key])
    assert HookPermissions().snapshot().ready

def test_deleting_an_approved_legacy_duplicate_does_not_transfer_grant(hook_file):
    raw = toml.loads(hook_file.read_text())
    hook = raw["hooks"]["hook"][0]
    hook.pop("id")
    raw["hooks"]["hook"] = [hook.copy(), hook.copy()]
    hook_file.write_text(toml.dumps(raw))
    owner = HookPermissions()
    pending = owner.snapshot()
    owner.approve(pending, [pending.rows[0].entry.key])
    raw["hooks"]["hook"].pop(0)
    hook_file.write_text(toml.dumps(raw))
    assert not owner.snapshot().ready
```

- [ ] **Step 2: Run the red checks, then add the private store and reconciliation.**

```bash
.venv/bin/python -m pytest Tests/Agents/test_hook_permissions.py -q
```

The JSON schema has version 1, a store-instance UUID, monotonic revision, config-scope records, observed keys/fingerprints/enabled state, and grants containing fingerprints plus grant UUIDs. Use this initial state; each canonical config path maps to `{"observed": {}, "grants": {}}`. Observed records contain `fingerprint`, `enabled` and `change`; grant records contain `fingerprint` and `token`. The execution approval token combines the store ID and grant token. Validate exact types and versions before using any field; booleans are not revision integers.

```python
from uuid import uuid4

def _empty_state() -> dict[str, object]:
    return {"schema_version": 1, "store_id": str(uuid4()),
            "revision": 0, "configs": {}}
```

Persist no argv/payload/output. Under config then consent locks: re-read disk, validate schema, reconcile observed execution changes/removals/legacy multiplicity, and apply only current explicit decisions. Update grant epochs on revoke, reapproval and observed eligibility loss without losing approval merely from disable/re-enable. Preserve unchanged legacy consent during guided ID assignment only when the writer owns the exact old-to-new mapping; otherwise require review.

Use `open_private_binary`, `atomic_private_write_text` and `PrivateFileWritePrecondition` from `Utils/private_paths.py`. Copy the small stable-lock pattern in config's `_config_interprocess_lock` using `create_private_text` and `open_private_text_append_stream` for `hook_permissions.json.lock`; do not use a replaceable state-file inode as the lock. A short in-process lock serializes owner state; use a nonblocking local seal request followed by the same launch authority lock before persistence. Fresh disk decisions, not cached grants, govern every launch. Unsupported/corrupt/unreadable stores never grant. Explicit reset produces a new store UUID and zero grants. Recover clears a fence only after verified disabled/no-hook state or a successful current decision/refresh.

The existing direct-child sensitive-path rule already covers grant, lock and hidden temporary files. Add behavior tests using the actual accessor, `.hook_permissions.json.<random>.tmp`, overrides, and filesystem/context/Git exclusions. Add an explicit dynamic-file entry only if these checks expose a real gap; do not duplicate the directory rule. Verify POSIX ownership/modes and retain existing platform-specific private-path behavior.

- [ ] **Step 3: Wire engine authority and a real command positive control.** Update constructor callers to pass `target_provider`, `notification_targets`, and keyword-only `launch_guard`; production obtains all three from the same runtime owner. Existing protocol-only test engines may use an explicit `nullcontext` launch guard and inventory-derived test targets; never ship an implicit allow default.

```python
import asyncio
from tldw_chatbook.Agents.run_hooks import RunHooksEngine

@pytest.mark.asyncio
async def test_revoke_prevents_real_command_from_starting(hook_file, tmp_path):
    marker = tmp_path / "executed"
    raw = toml.loads(hook_file.read_text())
    raw["hooks"]["hook"][0]["command"] = [
        sys.executable, "-c",
        "from pathlib import Path; Path(" + repr(str(marker)) + ").write_text('ran')",
    ]
    hook_file.write_text(toml.dumps(raw))
    owner = HookPermissions()
    pending = owner.snapshot()
    owner.approve(pending, [pending.rows[0].entry.key])
    engine = RunHooksEngine(
        owner.targets, lambda: str(tmp_path),
        notification_targets=owner.notification_targets,
        launch_guard=owner.launch_guard,
    )
    try:
        await engine.fire_async("PostToolUse", session_id="one")
        assert marker.read_text() == "ran"
        marker.unlink()
        current = owner.snapshot()
        owner.revoke(current, current.rows[0].entry.key)
        await engine.fire_async("PostToolUse", session_id="one")
        assert not marker.exists()
    finally:
        engine.close()
        for pool in (engine._notify_worker, engine._notification_pool, engine._pool):
            await asyncio.to_thread(pool.shutdown, wait=True, cancel_futures=True)
```

Retain the existing constructor's cwd provider and pool bounds; change its first dependency to the target provider instead of adding a second authority source. Thread target and tool name through `_fire`, `_run_hook`, `_execute_hook` and `_capture_hook`; hold `launch_guard(target, tool_name=tool_name)` across `await loop.subprocess_exec(...)` inside the worker's event loop and release before stdin/output/completion waiting. The engine admission lock must already be released. Fresh guard checking includes malformed/current matching guard definitions; revoked/mismatched authority is a typed refusal, not a generic exception. Handle `HookLaunchRefused` explicitly in `_run_hook` before its existing generic fail-open branch and at target selection, including the case where no target is returned.

`notify()` captures immutable targets from `notification_targets` when the event is admitted, then queues their bounded descriptors with the already-frozen payload. The worker never selects new targets for an old event. Add controlled barriers around actual process creation: revoke-before-launch prevents start; launch-before-revoke owns one already-running process; revoke/reapprove cannot resurrect queued work. Use spawned-process marker checks and real `fire_async`, not a nested event-loop failure that happens to deny.

- [ ] **Step 4: Test persistence failures and singleton lifecycle, then commit.** Inject writer failure before replace and after visible replacement, stale approve after revoke, two store owners and subprocessed independent instances, config/data-root retarget, malformed enabled guard, no hooks/master false with corrupt store, and queued observer definition changes. Seal before revoke/disable writes; do not hold locks until child completion. Runtime disposal closes owner and engine; optional view attachment cannot change authority.

```bash
.venv/bin/python -m pytest Tests/Agents/test_hook_permissions.py Tests/Agents/test_run_hooks.py Tests/Chat/test_console_run_hooks_regressions.py Tests/Chat/test_run_hooks_metadata.py Tests/Utils/test_sensitive_paths.py Tests/Tools/test_local_tool_sensitive_paths.py Tests/Tools/test_git_tool_sensitive_paths.py -q
git diff --check
git add tldw_chatbook/Agents/hook_permissions.py tldw_chatbook/Agents/run_hooks.py tldw_chatbook/Chat/console_runtime.py Tests/Agents/test_hook_permissions.py Tests/Agents/test_run_hooks.py Tests/Utils/test_sensitive_paths.py Tests/Tools/test_local_tool_sensitive_paths.py Tests/Tools/test_git_tool_sensitive_paths.py
git commit -m "feat: enforce persistent hook consent at process launch"
```

## Task 3: Shared Send admission and draft custody

**Files:** Modify `Chat/console_chat_controller.py`, `Chat/console_runtime.py`, and `UI/Console_Modules/prompt_queue.py`. Create `UI/Console_Modules/hooks.py` and `Tests/Chat/test_console_hook_admission.py`. Update the existing queue callers in the test files listed in Step 2 and the hook regressions.

**Consumes:** `HookPermissions.snapshot`, `HookReviewSnapshot.blocked_reason`, the existing `ConsoleSubmitResult`, queue `DISPATCH_REFUSED` recovery, and revision-pinned `ConsoleDraftStash`.

**Interfaces produced:**

- The controller receives named `hook_permissions_accessor: Callable[[], HookPermissions] | None`, supplied by ConsoleRuntime independently of view hooks.
- `ConsoleChatController.hook_admission_reason() -> Awaitable[str | None]` offloads only the readonly snapshot operation. `queue_prompt` becomes async with its existing parameters and `PromptQueueMutationResult` result; registry/custody work stays on its existing thread after the await.
- `ConsoleHooksController.dispatch(draft: str, *, session_id: str, stash: ConsoleDraftStash | None, dispatch: Callable[[], Awaitable[ConsolePromptDispatchResult]]) -> ConsolePromptDispatchResult` owns one visible attempt, including review and the normal dispatcher. It consumes continuation once and remains busy until dispatch completes.
- `ConsoleHooksController.review_current() -> None` opens inspection without Send; `refresh() -> None` refreshes published counts off-thread; `cancel_pending() -> None` invalidates continuation synchronously.
- Named constructor dependencies: `hook_permissions_accessor: Callable[[], HookPermissions]`, `request_review: Callable[[HookReviewSnapshot, bool, Callable[[], None]], Awaitable[HookReviewResult]]`, `current_session: Callable[[], str]`, `current_stash: Callable[[], ConsoleDraftStash | None]`, `on_state: Callable[[HookReviewSnapshot], None]`, and `notify: Callable[[str, str], None]`. No DOM ownership.
- `HookReviewResult(kind: Literal["ready", "cancel", "settings"], snapshot: HookReviewSnapshot | None)` is defined in `UI/Console_Modules/hooks.py` in this task; Task 4's modal imports it. A test callback returns the real type, without a modal dependency.

- [ ] **Step 1: Add a real controller rejection/approval control.** Import the Task 2 fixture and existing controller/gateway harness.

```python
import pytest
from Tests.Agents.test_hook_permissions import hook_file
from Tests.Chat.test_console_chat_controller import (
    ConsoleChatController, ConsoleChatStore, RecordingStreamingGateway,
)
from tldw_chatbook.Agents.hook_permissions import HookPermissions

@pytest.mark.asyncio
async def test_pending_consent_refuses_before_echo_then_approved_send_succeeds(hook_file):
    owner = HookPermissions()
    store = ConsoleChatStore()
    session = store.ensure_session()
    controller = ConsoleChatController(
        store=store, provider_gateway=RecordingStreamingGateway(),
        hook_permissions_accessor=lambda: owner,
    )
    refused = await controller.submit_draft("draft", session_id=session.id)
    assert not refused.accepted
    assert not refused.should_clear_draft
    assert store.messages_for_session(session.id) == []
    pending = owner.snapshot()
    owner.approve(pending, [pending.rows[0].entry.key])
    accepted = await controller.submit_draft("draft", session_id=session.id)
    assert accepted.accepted
```

- [ ] **Step 2: Run red, then put the gate before new admission's echo/queue/durable acceptance.** Use the same helper on direct and actual queued/wake submissions. Keep the existing UserPromptSubmit firing later and manual-only; consent preflight is not another lifecycle event.

```bash
.venv/bin/python -m pytest Tests/Chat/test_console_hook_admission.py -q
```

The worker-side read below is the shared admission primitive; compose it into `_submit_draft_lifecycle` before a new attempt can echo or acquire custody. For already accepted preparations, retain their recovery owner and refuse further dispatch without reclassifying acceptance.

```python
import asyncio
from tldw_chatbook.config import read_hooks_config_snapshot
from tldw_chatbook.Agents.run_hooks import inspect_hooks_config

def _hook_admission_reason(self) -> str | None:
    try:
        if self._hook_permissions_accessor is not None:
            return self._hook_permissions_accessor().snapshot().blocked_reason
        snapshot = read_hooks_config_snapshot()
        inventory = inspect_hooks_config(
            {"hooks": snapshot.section} if snapshot.section_present else {},
        )
        if not inventory.requires_authority:
            return None
        return "Hook review required; permission owner unavailable."
    except Exception:
        return "Hooks unavailable; review or disable hooks before sending."
```

Add `hook_admission_reason` as the async wrapper below. Import `read_hooks_config_snapshot` from config and `inspect_hooks_config` from run_hooks. A directly constructed controller without an owner can only admit a verified absent/disabled hook configuration. Never default to unconditional ready. For a new refusal return `ConsoleSubmitResult(False, False, reason, session_id=owner_key, origin=origin, queue_entry_id=queue_entry_id)` through the current refusal route. Do not append automatic hook bodies to diagnostics or recovery metadata.

```python
async def hook_admission_reason(self) -> str | None:
    return await asyncio.to_thread(self._hook_admission_reason)
```

In `queue_prompt`, retain input validation first, then perform this check before capturing configuration/prefill or constructing custody. Leave the rest of its existing implementation unchanged except for the async declaration.

```python
reason = await self.hook_admission_reason()
if reason is not None:
    return PromptQueueMutationResult(
        QueueMutationStatus.INVALID,
        self.prompt_queue_registry.snapshot(session_id),
        detail=reason,
    )
```

Both queue calls in `ConsolePromptQueueUIController.dispatch` and `_stage_normal_chain` await `controller.queue_prompt`. Change the `_FakeChatController.queue_prompt` test double in `Tests/UI/test_console_prompt_queue.py` to async. Update direct callers and synchronous `_queue` helpers in `Tests/Chat/test_console_prompt_queue_coordinator.py`, `test_console_turn_execution_context.py`, `test_console_send_gate_queue_race.py`, `test_console_turn_library_authority.py`, `test_console_automatic_library_preparation.py`, `Tests/UI/test_console_prompt_queue.py`, `test_console_button_routing.py`, `test_console_turn_navigation_continuity.py`, and `Tests/integration/test_console_library_control_integration.py`. Convert the affected sync tests/helpers to async and await their callers rather than adding a synchronous disk check or nested event loop.

Visible queue admission passes through the review controller before the existing dispatcher. Runtime `_run_custodied_turn` calls the same `submit_draft`; retain `_submit_queued_turn`/`_submit_fleet_wake` custody and existing refusal/recovery settlement. Queue execution checks again via the real controller and uses `DISPATCH_REFUSED`. Recovered preparations keep their accepted owner. Leave registry/custody mutations on their existing owner/thread; only readonly consent I/O is offloaded.

- [ ] **Step 3: Implement the DOM-free operation owner and its tests.** Store an incrementing generation and the originating session/stash until dispatch completes. Ignore repeated activation while the attempt is pending. After a Ready response, refresh authority, compare current stash text/edit_serial/generation and session, consume the continuation once, and call the captured dispatcher. Cancel increments generation immediately; later completion may persist an explicit decision but never dispatch.

```python
from dataclasses import dataclass
from typing import Literal
from tldw_chatbook.Agents.hook_permissions import HookReviewSnapshot
from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleDraftStash

@dataclass(frozen=True, slots=True)
class HookReviewResult:
    kind: Literal["ready", "cancel", "settings"]
    snapshot: HookReviewSnapshot | None = None

def same_captured_draft(current: ConsoleDraftStash, expected: ConsoleDraftStash) -> bool:
    return (current.text, current.edit_serial, current.generation) == (
        expected.text, expected.edit_serial, expected.generation,
    )
```

Use this predicate after the modal and before continuation; `capture_draft_for_send()` already provides a non-destructive current stash. Do not clear, restore, or recapture the original draft for another session. Return the existing typed REFUSED outcome when cancelled/stale; successful normal/queued custody continues to use existing commit functions.

Add controlled-future tests for duplicate Send, duplicate Allow, Escape during write, A-to-B-to-A session return, same-text edit with changed serial, queued refusal/recovery, durable resumed ownership, and wake without a view. Assert accepted/queued counts and current text, not only notification strings.

- [ ] **Step 4: Run green checks, review, and commit.**

```bash
.venv/bin/python -m pytest Tests/Chat/test_console_hook_admission.py Tests/Chat/test_console_run_hooks_regressions.py Tests/Chat/test_console_prompt_queue.py Tests/Chat/test_console_prompt_queue_coordinator.py Tests/Chat/test_console_viewless_hooks.py Tests/UI/test_console_prompt_queue.py -q
.venv/bin/python -m pytest Tests/Chat/test_console_turn_execution_context.py Tests/Chat/test_console_send_gate_queue_race.py Tests/Chat/test_console_turn_library_authority.py Tests/Chat/test_console_automatic_library_preparation.py Tests/UI/test_console_button_routing.py Tests/UI/test_console_turn_navigation_continuity.py Tests/integration/test_console_library_control_integration.py -q
git diff --check
git add tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/Chat/console_runtime.py tldw_chatbook/UI/Console_Modules/prompt_queue.py tldw_chatbook/UI/Console_Modules/hooks.py Tests/Chat/test_console_hook_admission.py Tests/Chat/test_console_run_hooks_regressions.py Tests/UI/test_console_prompt_queue.py
git add Tests/Chat/test_console_prompt_queue_coordinator.py Tests/Chat/test_console_turn_execution_context.py Tests/Chat/test_console_send_gate_queue_race.py Tests/Chat/test_console_turn_library_authority.py Tests/Chat/test_console_automatic_library_preparation.py Tests/UI/test_console_button_routing.py Tests/UI/test_console_turn_navigation_continuity.py Tests/integration/test_console_library_control_integration.py
git commit -m "feat: gate Console sends on current hook consent"
```

## Task 4: Native review modal and persistent Console action

**Files:** Create `Widgets/Console/console_hooks_review_modal.py` and `Tests/UI/test_console_hooks_review.py`. Modify `UI/Console_Modules/hooks.py`, `UI/Console_Modules/wiring.py`, small delegates in `UI/Screens/chat_screen.py`, `Widgets/Console/console_control_bar.py`, `Widgets/Console/console_workbench_state.py`, `Chat/console_glyphs.py`, `Widgets/glyph_fallback.py`, and CSS source `css/components/_agentic_terminal.tcss`.

**Consumes:** Task 2 review snapshots/actions and Task 3 operation owner.

**Interfaces produced:**

- `ConsoleHooksReviewModal(SafeModalDismissMixin, ModalScreen[HookReviewResult])` imports the mixin from `Widgets/modal_dismissal.py`. Constructor keywords are `snapshot: HookReviewSnapshot`, `waiting_for_send: bool`, `approve: Callable[[HookReviewSnapshot, Collection[str]], Awaitable[HookReviewSnapshot]]`, `revoke` and `disable` each `Callable[[HookReviewSnapshot, str], Awaitable[HookReviewSnapshot]]`, `recover: Callable[[], Awaitable[HookReviewSnapshot]]`, `reset: Callable[[HookReviewSnapshot], Awaitable[HookReviewSnapshot]]`, and `on_cancel: Callable[[], None]`.
- `HookReviewResult` is the concrete result defined in Task 3's contract.
- Toolbar Workbench action ID `hooks`, widget ID `console-control-hooks`, placed immediately after `settings`. Add a shared hook glyph with ASCII `H` fallback; counts remain in the action label/tooltip, separate from the tool Approvals chip.
- Screen delegates `_open_console_hooks_review()` and `_refresh_console_hooks()` route through the controller. `_dispatch_console_draft_send` delegates review then its existing typed queue dispatcher.

- [ ] **Step 1: Add a mounted route/geometry regression, then run it red.** Extend the real ConsoleHarness pattern from `test_console_workbench_contract.py`, not an isolated bar alone.

```python
import asyncio
import pytest
from Tests.Agents.test_hook_permissions import hook_file
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_console_workbench_contract import (
    ConsoleHarness, _configure_native_ready_console,
)
from tldw_chatbook.Widgets.Console.console_hooks_review_modal import (
    ConsoleHooksReviewModal,
)

@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 40)])
async def test_hooks_action_stays_reachable_and_opens_review(size, hook_file):
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=size) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-control-hooks")
        button = console.query_one("#console-control-hooks")
        assert button.region.right <= console.region.right
        await pilot.click("#console-control-hooks")
        async with asyncio.timeout(3):
            while not isinstance(host.screen, ConsoleHooksReviewModal):
                await pilot.pause(0.01)
        await _wait_for_selector(host.screen, pilot, "#console-hooks-review")
        await pilot.press("escape")
        await pilot.pause()
        assert host.screen is console
        assert console.focused is button
```

The harness publishes a separate provider mapping; consent must still read `hook_file`, so this test deliberately does not copy hooks into `app.app_config`. Wait for modal creation by class, then its selector, because `host.screen` may still be Console immediately after clicking.

```bash
.venv/bin/python -m pytest Tests/UI/test_console_hooks_review.py -q
```

- [ ] **Step 2: Build the modal and literal command formatting.** Header/footer remain reachable; use VerticalScroll for rows and lazy expanded details. Needs review includes invalid/recovery attention with disabled checkboxes and explicit repair/retry guidance. All hooks adds current Approved/Disabled states and revoke. Start selection empty. Allow selected retains unchecked pending rows and does not resume early. Manage in Settings returns `kind="settings"` after cancelling continuation.

```python
import json
from tldw_chatbook.Agents.run_hooks import HookSpec

def command_json(spec: HookSpec) -> str:
    return json.dumps(list(spec.command), ensure_ascii=True, indent=2)
```

Render this exact string as literal text with markup disabled. Summaries may shorten sanitized labels; details must preserve all arguments. Async callbacks use worker/to-thread owner actions; disable conflicting controls while a write is pending. Display a returned snapshot's static `notice` without inventing persistence success. Check `app.screen is self` and action generation before publishing results. Safe dismissal invokes `on_cancel` before late work can finish. Reset invalid persisted state requires the existing confirmation dialog and never grants anything.

Put token-backed modal rules in its `BUNDLED_CSS`, so Settings-first launch works without Console CSS. Use existing control/scroll/focus tokens; no ad-hoc Python `styles.*` values.

- [ ] **Step 3: Wire the actual toolbar and Send route.** Add `hooks` to TOP_ACTION_IDS, widget ID map, fallback and Workbench action state. Bind through `WorkbenchActionRequested`, not a second button message route. Construct the controller only in `wiring.py` with late-binding lambdas. Register a view slot only if a runtime projection needs it, declare it in CONSOLE_VIEW_HOOK_SLOTS in the same change, and test detach behavior; consent itself must never depend on a view slot.

```python
result = await self._hooks.dispatch(
    draft, session_id=session_id, stash=stash,
    dispatch=lambda: self._prompt_queue.dispatch(
        draft, session_id=session_id, stash=stash,
    ),
)
```

This is the screen's typed dispatch fragment after command parsing and session checks. Preserve its existing diagnostics and return-value mapping. Refresh counts on mount/activation, review open, config save, Send and owner reconciliation; add no timer polling. Compact secondary toolbar labels if needed to keep the hook icon reachable at 80 columns.

- [ ] **Step 4: Add real interaction/race checks, rebuild CSS, and commit.** Exercise row disclosure, exact long/control-character argv, both tabs, keyboard activation, no-hooks/master-disabled/error states, selective approval, permission revoke and failed writes, cancellation with a composer draft, and late workers after the modal leaves the stack. Add Settings-first modal styling in Task 5's harness. Capture compositor output for wrapped/clipped details; a renderable string alone is insufficient.

```bash
.venv/bin/python tldw_chatbook/css/build_css.py
.venv/bin/python -m pytest Tests/UI/test_console_hooks_review.py Tests/UI/test_console_workbench_contract.py Tests/UI/test_console_control_bar_coalescing.py Tests/UI/test_console_controller_wiring.py Tests/UI/test_console_runtime_ownership.py Tests/UI/test_design_token_governance.py -q
git diff --check
git add tldw_chatbook/Widgets/Console/console_hooks_review_modal.py tldw_chatbook/UI/Console_Modules/hooks.py tldw_chatbook/UI/Console_Modules/wiring.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/Widgets/Console/console_control_bar.py tldw_chatbook/Widgets/Console/console_workbench_state.py tldw_chatbook/Chat/console_glyphs.py tldw_chatbook/Widgets/glyph_fallback.py tldw_chatbook/css/components/_agentic_terminal.tcss Tests/UI/test_console_hooks_review.py
git commit -m "feat: expose native Console hook review"
```

Before that commit, explicitly stage the generated CSS/build-manifest files actually changed by the rebuild and any extended tests. Do not stage unrelated outputs or weaken architecture ratchets to accommodate a large screen implementation.

## Task 5: Canonical Settings editor and complete feature verification

**Files:** Create `UI/Screens/settings_hooks.py` and `Tests/UI/test_settings_hooks.py`. Modify `UI/Screens/settings_screen.py`, `settings_config_models.py`, `settings_search_index.py`, CSS source `css/components/_agentic_terminal.tcss`, generated outputs, and `Docs/User_Guide/console/agent-runs-and-tools.md`. Add a scoped verification record under `Docs/superpowers/reviews/2026-09-27-console-hook-settings-and-review.md` at closeout.

**Consumes:** Config snapshots/writer, runtime consent owner, shared modal, `SettingsDraft`, category/ownership/impact/navigation registries.

**Interfaces produced:**

- `SettingsCategoryId.HOOKS = "hooks"` joins Expert, summary, field search, ownership, category state/scope, guided mutation membership, save/revert and impact branches.
- `HooksSettingsPanel` exposes named `load(snapshot: HookReviewSnapshot)`, `capture_values() -> Mapping[str, object]`, and `load_save_result(result: LiteralConfigMutationResult, submitted: Mapping[str, object])` methods; emitted edit/save/revert/review messages route to canonical Settings. Keep draft originals and staged values in its existing SettingsDraft owner, not a parallel durable store.
- IDs: `settings-hooks-enabled`, `settings-hooks-add`, `settings-hooks-list`, `settings-hooks-event`, `settings-hooks-command`, `settings-hooks-matcher`, `settings-hooks-timeout`, `settings-hooks-toggle`, `settings-hooks-remove`, `settings-hooks-save`, `settings-hooks-revert`, `settings-hooks-review`.
- A shared modal launched from Settings has `waiting_for_send=False`, the same singleton owner/actions, and no Send continuation.

- [ ] **Step 1: Add mounted category/save tests and run red.** Reuse DestinationHarness from the existing Settings hub tests. Route with the actual navigation context, then verify search/ownership rather than setting the category only in a bare model.

```python
import pytest
import toml
from Tests.Agents.test_hook_permissions import hook_file
from Tests.UI.test_destination_shells import (
    DestinationHarness, _active_destination_screen, _build_test_app,
    _wait_for_selector,
)
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId

@pytest.mark.asyncio
async def test_hooks_edit_is_staged_until_canonical_save(hook_file):
    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(120, 40)) as pilot:
        screen = _active_destination_screen(host)
        screen.apply_navigation_context({"category": "hooks"})
        await _wait_for_selector(screen, pilot, "#settings-hooks-enabled")
        before = hook_file.read_bytes()
        await pilot.click("#settings-hooks-enabled")
        await pilot.pause()
        assert hook_file.read_bytes() == before
        assert screen._category_has_unsaved_changes(SettingsCategoryId.HOOKS)
        await pilot.click("#settings-hooks-save")
        await host.workers.wait_for_complete()
        await pilot.pause()
        assert toml.loads(hook_file.read_text())["hooks"]["enabled"] is False
```

`DestinationHarness(app, "settings")` is the existing real route constructor. Use the worker completion above before checking disk; the test must prove a staged change and the subsequent canonical save separately.

```bash
.venv/bin/python -m pytest Tests/UI/test_settings_hooks.py -q
```

- [ ] **Step 2: Build the focused panel and integrate canonical draft/save/revert.** Show master switch, Add, Review permissions, selectable rows, event Select, JSON argv TextArea, optional matcher and timeout fields, contextual Enable/Disable, Remove, Save and Revert. New rows are disabled, get UUIDs on guided save, and never get implicit permission. Merge edits onto detached original raw tables to preserve unknown fields. Disabled invalid originals may be retained or explicitly disabled; enabled changed definitions must validate before Save. A malformed whole section displays repair guidance to Advanced Config rather than dropping its content.

```python
import json

def parse_command_argv(text: str) -> list[str]:
    value = json.loads(text)
    if (not isinstance(value, list) or not value
            or not all(isinstance(arg, str) and "\x00" not in arg for arg in value)
            or not value[0]):
        raise ValueError("Enter a nonempty JSON array of strings with an executable.")
    return value
```

Use runtime validation for event/matcher/finite numeric timeout; do not add a second schema with different limits. Stage complete `enabled`/`hook` values in SettingsDraft with deep copies. Capture a submitted draft revision before the worker; successful Save clears only submitted edits, not later typing. Revert confirms before replacing dirty values and reloads authority. On `file_replaced=True, caches_reloaded=False`, display saved-but-refresh-pending and preserve the local execution fence; Retry refresh must not reapply an old replacement.

- [ ] **Step 3: Integrate review/deep links/impact and run focused UI checks.** The impact pane states User config, all Console chats using this file, current saved permission, and execution changes needing review. Unsaved edits are never reviewed as saved definitions; explain Save/Revert before opening permission review without discarding them. Add every guided field to category search and all existing registries. Canonical global `s`/`r` actions and explicit buttons must agree; text-entry focus must keep its existing shortcut rules. The Console modal's Settings action posts `NavigateToScreen(TAB_SETTINGS, screen_context={"category": "hooks"})` after cancelling Send.

Verify Settings-first modal geometry/style, guided ID transfer, unknown fields, new hook enable/disable/remove, invalid disabled originals, matcher event change, JSON/control characters, stale raw edits, Save with newer typing, confirmed Revert, permission count refresh, and responsive category/detail/impact panes. Keep legacy Settings parallels untouched.

```bash
.venv/bin/python tldw_chatbook/css/build_css.py
.venv/bin/python -m pytest Tests/UI/test_settings_hooks.py Tests/UI/test_settings_search_index.py Tests/UI/test_settings_configuration_hub.py Tests/UI/test_settings_raw_draft.py Tests/UI/test_console_hooks_review.py Tests/UI/test_design_token_governance.py -q
```

- [ ] **Step 4: Verify the complete feature with isolated live state and update guidance.** Run the new focused suites plus any still-unrun existing checks named above. Repeat a passed suite only after a relevant change/failure. Run Ruff/format checks on changed Python files using explicit paths. Rebuild CSS and compare outputs; do not edit generated sheets by hand.

For a live terminal run, create a private temporary config with `[paths] data_dir` pointing to an already-created private temporary directory and a hook using `sys.executable` that writes a temporary marker. Set only `TLDW_CONFIG_PATH` for that process; do not repurpose HOME or the user's config/data. Run `.venv/bin/python -m tldw_chatbook.app` through the repository's terminal verification workflow, keep stderr attached, and verify toolbar/modal/Settings at 80 and 120 columns. Assert config/data resolution before sending, and prove no writes touch the user's real paths.

Record evidence for: pending Send keeps text and creates no marker; explicit approval sends once and creates the marker; restart keeps approval; editing argv/matcher/event/timeout requires review; revoke blocks future launches; selective approval leaves other rows pending; disabled/master-off allows Send; cancelled Settings navigation retains text; and queued/viewless refusals retain their recovery owner. Test background behavior through its real controller/coordinator, not a UI-only fake. No provider/network sweep is required; use the existing fake gateway for admission tests and a qualified isolated provider only for the live Send if configured safely.

Update the user guide's existing hooks section with the implemented controls, next-Send behavior, JSON argv format, persistent consent, invalid/recovery guidance, scope and explicit integrity/revocation limits. Record actual commands/outcomes and any baseline failures in the verification record. Keep TASK-33163 In Progress until every acceptance criterion, applicable checks, docs and self-review are complete.

- [ ] **Step 5: Commit the editor/integration closeout and hand off the branch.** Stage only the listed feature files, actual generated outputs, verification record and completed task notes. Use Backlog CLI for AC completion/status after evidence is recorded.

```bash
git diff --check
git add tldw_chatbook/UI/Screens/settings_hooks.py tldw_chatbook/UI/Screens/settings_screen.py tldw_chatbook/UI/Screens/settings_config_models.py tldw_chatbook/UI/Screens/settings_search_index.py tldw_chatbook/css/components/_agentic_terminal.tcss Tests/UI/test_settings_hooks.py Docs/User_Guide/console/agent-runs-and-tools.md Docs/superpowers/reviews/2026-09-27-console-hook-settings-and-review.md
git commit -m "feat: manage hook configuration in canonical Settings"
```

## Coverage and review checkpoints

| Spec requirement | Owning task/check |
| --- | --- |
| Legacy identity, execution fingerprint, duplicates, stable IDs | 1 parsing/config tests; 2 reconciliation/restart tests; 5 guided ID save. |
| Private storage, schema failure, overrides, multiple instances | 2 real store/locking/sensitive-path tests. |
| Revocation/process race, disabled states, notification ownership | 2 real process/barrier tests and grant epoch checks. |
| Direct, queued, durable, recovered and wake admission | 3 real controller/coordinator tests. |
| Next Send, cancelled/stale callbacks, exactly-once continuation | 3 controlled-future ownership tests; 4 mounted Send flows. |
| Toolbar accessibility, details, selection, current permissions | 4 mounted modal/geometry/compositor tests at 80/120 columns. |
| Settings editor, search/deep links, staged Save/Revert, impact | 5 canonical Settings route and concurrent-save tests. |
| Tokens, lazy modal CSS, generated outputs and architectural slots | 4/5 governance, Settings-first mount and runtime ownership checks. |
| User documentation and isolated live evidence | 5 closeout record, guide and Backlog AC checks. |

After each task, review its diff against these contracts before proceeding. At final review confirm no grants in TOML, no implicit allow defaults, no hook bodies in metadata/log additions, no direct config writer, and no unrelated checkout changes. Document real unresolved limitations; never mark an unrun runtime scenario as verified.
