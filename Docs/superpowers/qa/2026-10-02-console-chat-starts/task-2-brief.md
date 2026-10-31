# Binding context

# Global constraints

- “The new destination and mode arguments belong only to `new_chat`.”
- “Retain the existing 120-character title cap and 20,000-character per-field prompt/instructions caps.”
- `destination=same_workspace|casual`; `mode=draft|start`; defaults are `same_workspace` and `draft`.
- “Both modes keep the user's current chat, active workspace, composer, and focus intact.”
- Casual persistence is `scope_type=global`, `workspace_id=NULL`; the session uses `CONSOLE_GLOBAL_WORKSPACE_ID`.
- Destination defaults and explicit standing instructions determine the fresh assistant; source identity, bindings, staged inputs and grants do not transfer.
- Both grant caches use requesting session incarnation, tool, resolved destination identity and mode. Denial ceilings remain per run/tool.
- “Each accepted chat start consumes one automatic generation; draft-only creation consumes none.”
- “Start attempts and fleet wakes share the existing automatic-primary admission limit and manual capacity reserve.”
- “Capacity refusal leaves a draft; this feature adds no waiting/retry timer.”
- AgentRunsDB acceptance is the ownership cutoff. Both durable fences must succeed before dispatch or `started`.
- Human decisions or paused preparation before acceptance return `not_started`; interrupted or uncertain starts never replay automatically.
- Version-2 handoffs persist edits/clears until accepted consumption or explicit discard. Unversioned handoffs keep the legacy contract.
- Bodies stay in private conversation storage; budget tables, badges and generic diagnostics carry bounded provenance/reasons only.
- Targeted tests only. A full sweep requires user opt-in. UI changes follow `backlog/docs/design-language.md` and its existing tokens.
- Planning changes no application code. At execution, inspect the dirty checkout and attached worktrees, then use a suitable isolated checkout under the worktree skill; preserve unrelated changes.


## Task 2: Destinations, approval, durable drafts and one background start

**Backlog:** [TASK-33805](../../../../backlog/tasks/task-33805%20-%20Console-workspace-or-casual-chats-with-bounded-background-starts.md). Depends on TASK-33804.

**Independently reviewable outcome:** All four `new_chat` combinations work in the
real Console; the first task's ledger remains the only automatic budget owner.

**Files:**

- Modify the tool, creation, draft, durable-submission, view and scheduling owners in the file map.
- Create: `tldw_chatbook/Chat/console_chat_start.py`, `Tests/Chat/test_console_chat_start.py`.
- Create: `tldw_chatbook/DB/migrations/chachanotes_v73_to_v74_agent_chat_starts.sql` at the inspected version.
- Modify: `Tests/Chat/test_console_chat_create_integration.py`, `test_console_chat_create_confirm.py`, `test_chat_create_confirm_card.py`, `test_console_chat_store.py`, `test_chat_persistence_service.py`.
- Modify: `Tests/Chat/test_console_fleet_wake.py`, `test_console_prompt_queue_coordinator.py`, `Tests/UI/test_console_runtime_ownership.py`.
- Modify: `Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py`, `Tests/Chat/test_console_dispatch_recovery.py`, `Tests/DB/test_chachanotes_console_library_policy_migration.py`.
- Create: `Tests/DB/test_chachanotes_v74_agent_chat_starts_migration.py` at the inspected version.
- Update: `Docs/User_Guide/console/agent-runs-and-tools.md`, the existing Chat creation tools section.

### Interfaces consumed and produced

Consume Task 1's native attempt methods and `AutomaticWorkContext` exactly as
defined above. Reuse `ConsoleRuntime._resolve_new_console_assistant`,
`ConsoleChatStore.create_session`/`restore_persisted_session`, the generation
metadata codec, `ChatPersistenceService.commit_durable_turn`, and normal
`ConsoleChatController.submit_draft` preflight.

In the new start module, define:

```python
from dataclasses import dataclass, field
from typing import Literal

from .console_turn_context import ConsoleTurnConfigurationSnapshot

@dataclass(frozen=True, slots=True)
class AgentChatStartRequest:
    attempt_id: str
    source_run_id: str
    source_session_id: str
    source_session_incarnation: str
    conversation_id: str
    session_id: str
    session_incarnation: str
    draft_revision: int
    context_epoch: int
    opening_prompt: str = field(repr=False)
    configuration: ConsoleTurnConfigurationSnapshot = field(repr=False)

@dataclass(frozen=True, slots=True)
class AgentChatStartOutcome:
    launch_status: Literal["not_started", "started", "review_required"]
    reason: str | None = None
```

`AgentChatStartAuthorization` is opaque, constructed only by
`ConsoleChatStartCoordinator`, and bound to coordinator/token/attempt/target
incarnation. Do not expose a constructor through tools. The coordinator provides:

| Interface | Contract |
| --- | --- |
| `async start(request: AgentChatStartRequest) -> AgentChatStartOutcome` | Own target task until first refusal/confirmed acceptance; return then while the runtime continues the target task |
| `authorizes(authorization: AgentChatStartAuthorization, session_id: str) -> bool` | Exact current coordinator, attempt and target incarnation only |
| `withdraw_prepared(session_id: str, reason: str) -> bool` | Serialize manual/close/source revocation against the durable acceptance transition; accepted work is never withdrawn by source revocation |
| `is_prepared(session_id: str) -> bool` | Let ordinary manual UI/dispatcher gates distinguish prepared start from accepted running work |
| `async accept(authorization: AgentChatStartAuthorization) -> bool` | Last owner/revision/policy/capacity checks and Task 1 acceptance cutoff; execute on the same owning loop as withdrawal |
| `dispose() -> None` | Fence runtime ownership; cancellation/settlement retains uncertainty rather than replaying |

Expose this coordinator on the app-owned runtime/controller, never on the screen.
The worker-thread tool executor uses `asyncio.run_coroutine_threadsafe` on that
existing owning loop and waits for the coordinator's acceptance outcome, not its
whole generation. Do not nest an event loop or launch from a screen completion.

Extend `build_chat_create_tool_closures` with a trusted
`prepare: Callable[[dict], dict] | None = None` callback. Production `new_chat`
requires it; `fork_chat` retains its existing path. The controller's new
`prepare_agent_chat_create(payload: dict[str, Any]) -> dict[str, Any]` validates
live source authority, captures destination/defaults and returns an internal
payload carrying `_grant_scope` plus an opaque `_creation_token`. Confirm and
execute use that same token, backed by one bounded controller-owned record until
creation finishes. Model arguments never supply internal keys; they are copied
from the validated public field allowlist. The token's owner record carries the
frozen assistant/settings selection and true source run, so argument rewriting
or a skipped card cannot change what was approved. Release records on denial,
completion, source cancellation and shutdown. Add a fresh
`ConsoleChatSession.incarnation_id` UUID per object lifetime; never persist it or
derive it from reusable session/conversation IDs.

### Steps

- [ ] **1. Add strict schema and grant tests before implementation.** Parameterize the bridge's real closure test with invalid destination, mode, nonstring fields, overflow and blank start prompt; assert no confirmation and no execution. Add this concrete cache-separation test using the existing fakes:

```python
def test_remembered_draft_does_not_grant_start_or_casual():
    confirm = _FakeConfirm([{"allow": True, "remember": True}])
    executor = _FakeExecutor()

    def prepare(payload):
        return {
            **payload,
            "_grant_scope": (
                "incarnation-1", "new_chat",
                "global" if payload.get("destination") == "casual" else "workspace",
                None if payload.get("destination") == "casual" else "workspace-1",
                payload.get("mode", "draft"),
            ),
            "_creation_token": object(),
        }

    _, new_chat = build_chat_create_tool_closures(
        confirm=confirm, execute=executor, prepare=prepare,
        session_id="s1", run_id="message-1",
    )
    assert new_chat({"opening_prompt": "draft"}).ok
    assert new_chat({"opening_prompt": "later draft"}).ok
    assert not new_chat({"opening_prompt": "run", "mode": "start"}).ok
    assert not new_chat({"destination": "casual"}).ok
    assert len(executor.calls) == 2
```

Extend the controller's real threaded round tests for exact request IDs, stale
Allow after Stop, destination identity changes, live remembered grants with a
detached view, fresh session/restart, and per-tool refusal ceiling across modes.
Bind an actual run using `use_run_id`; assert the executor receives that run
identity independently of the old assistant-message field. Keep all fork tests.

- [ ] **2. Extend only NEW_CHAT_TOOL_SCHEMA and match both caches.** Add these schema entries, update the tool description for the four outcomes and explicit no-repeat recovery, and enforce the same enum/type/length checks before approval and at the mutation boundary:

```python
"destination": {
    "type": "string", "enum": ["same_workspace", "casual"],
    "default": "same_workspace",
},
"mode": {
    "type": "string", "enum": ["draft", "start"], "default": "draft",
},
```

Resolve source session scope and freeze destination/default settings before the
card. Use the tuple `(source_incarnation, tool, scope_type, workspace_id, mode)`
in both new-chat caches. Preserve the existing fork cache contract. Read trusted
`current_run_id()` at invocation, validate it against source execution ownership,
and carry source message ID separately. Recheck live source ownership even when
memoization skips the card; clear grants on close/restart. Approval view absence
refuses a required new card, but a valid remembered grant needs no mounted card.

- [ ] **3. Save/restore destination defaults through the existing owners.** Add real-SQLite cases for same-workspace/global/default sources and explicit casual destinations. Seed different source and destination personas/models; assert creation and reopened assistant identity, system prompt and durable generation snapshot agree. Call the existing runtime assistant resolver with explicit target settings; nonblank instructions use `replace(base_settings, system_prompt=instructions)` before resolution and therefore plain/custom identity. Never derive target settings from the source session. Save assistant identity, generation metadata and version-2 handoff in initial creation metadata/columns, then restore `activate=False` with that same snapshot. Keep target availability validation and workspace membership normalization at the ordinary persistence boundary. Fresh scratch/project control state comes from normal new-session creation, not legacy restored instruction state.
- [ ] **4. Give new handoff drafts one revision owner.** Store this versioned record inside private conversation metadata:

```python
{
    "version": 2,
    "created_via": "new_chat",
    "state": "pending",
    "draft_revision": 1,
    "draft": opening_prompt,
    "source_run_id": trusted_source_run_id,
}
```

Extend `set_session_draft` for pending version-2 handoffs: every edit/clear bumps
its revision, writes the current text through the persistence owner, and merges
metadata using expected conversation version plus handoff revision. Do not add
generic autosave for ordinary chats. Keep one latest pending revision and one
owned off-loop writer per handoff; coalesce intermediate edits so typing cannot
create an unbounded write queue. Track outstanding writes and serialize/drain
them before start acceptance, manual send, explicit close or shutdown. A stale write
must not overwrite newer metadata or resurrect text. Activation consumes only
unversioned legacy records. Durable consumption writes `state="consumed"`, an
empty draft and a bumped revision while retaining exact accepted attempt ID.
Add reopen tests for edit, clear, activation, stale write, successful consumption,
legacy fork/new and unknown version. Block dispatch on unconfirmed draft custody.
- [ ] **5. Extend the existing durable checkpoint schema and codecs.** Rebuild the local checkpoint table through a versioned ChaChaNotes migration, copying every old column/row and preserving indexes/constraints. Add origin `agent_chat_start` and a nullable unique `agent_chat_start_attempt_id`. Require that ID for the machine origin, forbid it for manual/queued origins, and keep their existing queue-entry rules. Add matching optional fields to `ConsoleDurableTurnAcceptance` and `ConsoleDispatchCheckpoint`, with default `None` for old callers. Extend repository SELECT/INSERT/parser/acceptance validation. Add a frozen `AgentChatStartMetadata` record in `message_metadata.py` with `attempt_id`, `source_run_id`, and `source_conversation_id`; add `agent_chat_start: AgentChatStartMetadata | None = None` to `MessageMetadata`. Validate bounded exact keys/IDs through its existing codec style, require it for this origin, and extend serialization/hydration. Set that machine metadata in the same durable turn transaction; do not rely on a later metadata patch. Invalid machine provenance must never degrade into trusted user authority. In `ChatPersistenceService.commit_durable_turn`, consume the matching version-2 draft revision in that same transaction and write the exact attempt receipt. Duplicate acceptance reads the existing complete receipt; a partial/mismatched pair refuses dispatch. Reuse the existing checkpoint state transitions and explicit Retry UI. Reopen/migration tests must prove all three origins survive and old rows retain their values.
- [ ] **6. Share existing automatic slots and add the runtime coordinator.** Add loop-owned exact-token claims to the existing fleet admission owner with interfaces `try_claim_automatic_primary(session_id: str, token: object) -> bool` and `release_automatic_primary(session_id: str, token: object) -> bool`; both wakes and starts call them. Publish one read-only `runtime_owner_id` for both coordinators, retain the existing 2-primary limit and `max_parallel_runs - 1` reserve, and include prepared starts in occupancy without double-counting their validating session. The start's exact unchanged handoff draft is its owned input; do not let the wake composer-priority probe mistake it for competing user intent. Revision changes, a real manual dispatch, queued work or other staged input still refuse the start. Release only the matching token after actual worker cleanup; losing callbacks cannot release a replacement. Reuse ordinary shutdown/recovery ownership. The new coordinator creates no timers or waiting queue. Before native preparation, refuse unsupported/disabled runtime, missing source chain, unavailable target, or capacity; after durable creation these are `not_started` outcomes, not creation errors.
- [ ] **7. Add AGENT_CHAT_START deliberately throughout submission.** Add `ConsoleSubmissionOrigin.AGENT_CHAT_START = "agent_chat_start"` and the matching message-origin constant. Add `chat_start_authorization` to `submit_draft` and its internal delegates; validate it before any transient echo. Audit each `AGENT_WAKE`, `MANUAL`, `QUEUED` conditional and exhaustive origin codec, using this required matrix:

| Branch | Chat-start behavior |
| --- | --- |
| Work scope and stream mapping | AUTOMATIC; exact target local chain and `attempt_kind="chat_start"`; never `manual_work_scope` or `create_chain` |
| Prefixes and inputs | Slash/@ literal; no command expansion, pending attachments, one-shot prefill or foreground evidence |
| Request row | Durable request-role row with machine metadata and exact-attempt receipt |
| Destination features | Normal readiness, project instructions, hooks, tool permissions, configured capture and retrieval |
| Pre-acceptance pause/decision | Refuse this attempt, preserve draft/current edits, return required action; no dialog continuation |
| User authority | No prompt-history insertion, composer clear callback or trusted profile mutation authority |
| Ownership | Runtime survives view detachment/navigation; target Stop/close works after acceptance |

Check `_ordinary_library_text`, `_admit_capture_policy`, durable-turn origin tests,
request metadata hydration, trusted-profile IDs, leave-console cancellation,
capture/citation preparation and provider adapters explicitly. Keep configured
retrieval for the target rather than blindly applying all wake shortcuts. The
new origin's literal slash/@ text is ordinary retrieval input; the existing
command-prefix exclusion must not classify it as a composer command. The
last acceptance checks and AgentRunsDB transition occur on the same owning loop
as withdrawal; then commit the conversation receipt off-loop, verify both fences,
mark automatic context accepted, resolve the source's `started` outcome, and
continue the owned target turn. Any failure after the cutoff retains charges and
requires review; no subsequent provider dispatch is authorized by a missing receipt.
- [ ] **8. Put manual intent before the existing busy gates.** In the screen's Send availability/refusal, queue-aware dispatcher and controller entry guard, distinguish `is_prepared(session_id)` from accepted work. Manual dispatch calls `withdraw_prepared(..., "manual_send")` before ordinary queue/run admission. Invalidate the exact preparation token and reservation, then use normal manual validation and a fresh manual allowance on acceptance. Edit, explicit discard and closure invalidate only still-prepared starts; source cancellation cannot withdraw accepted targets. Remove old preparation/run state only if its ownership token still matches. Add barrier tests at source-before-cutoff, manual-before-cutoff, target-after-cutoff and conversation-write failure so the loser never clears newer text or run state.
- [ ] **9. Make view completion an observer and verify the mounted UI.** Move new-chat restore/launch ownership from `ChatScreen._complete_agent_chat_create` into the runtime/controller completion path. Keep the screen hook for list refresh, notice and toast; preserve the legacy fork observer's behavior. Extend the existing card with destination, mode, assistant/model and complete instructions text using markup-disabled existing widgets/status classes. Ensure any changed observer hook belongs to `CONSOLE_VIEW_HOOK_SLOTS` and clears on view detachment. Return `ok=True` on known durable creation with conversation/destination/mode and a truthful status/reason. Do not turn restore/launch failure after creation into a generic tool error or delete the saved chat.
- [ ] **10. Add end-to-end tests and run the target matrix.** Reuse the real SQLite creation fixture for creation-only cases, and the existing `_controller` rig in `test_console_agent_swap.py` for actual agent-enabled starts. Use explicit event barriers and injected provider/write refusals. The minimum matrix is:

| Case | Required evidence |
| --- | --- |
| Four destination/mode combinations | Correct scope/membership, assistant snapshot, draft or exactly one accepted target generation; source focus/composer untouched |
| Remembered and parked approval | Both caches separated; stale decisions denied; full mounted card; exact trusted run provenance |
| Disabled/unready/paused/capacity blocked | Durable chat plus draft and `not_started`; zero provider calls; no scheduled automatic retry |
| New origin with `/help` and `@file` | Literal saved/request wire content; no parsing/expansion; no trusted profile write |
| Recursive starts with child/wake work | Same canonical counters/deadline; local conversation chains and run parents; generation 1 per accepted start |
| One wake plus start candidates | Combined 2 automatic primaries and retained manual slot; delayed worker cleanup keeps claims held |
| Draft edit/clear/manual/close races | Newer text/state survives; prepared loser releases only its own claim; accepted source-independent target remains owned |
| Crash after each DB fence | No automatic resend, truthful review state, conservative charge, exact receipt and explicit recovery controls |
| View detachment/restart | Runtime owns launch; required new approval fails without surface; valid grant runs without view; new process has fresh grants |

- [ ] **11. Complete focused checks, guide update and real Console verification.** Run the changed Chat/UI tests and foundation suites; record actual command results. In a real Console with a configured provider, create workspace/casual drafts, start each, reopen saved drafts after restart, verify focus/composer, reproduce a blocked start, and exercise target Stop. Use the existing native UI/live-verification skill at execution. If provider credentials or the live app are unavailable, record the missing evidence and keep the task In Progress. Update the existing tool guide and task notes/AC only after evidence passes. Commit only integration files with `feat: create workspace or casual chats with bounded starts`.

## Scoped verification commands

Run the specific red test before each new behavior, then its owning file. At each
task's closure, run the relevant groups below; do not expand to the full suite:

```bash
.venv/bin/python -m pytest Tests/DB/test_automatic_chat_starts.py Tests/DB/test_automatic_work_budget.py Tests/DB/test_automatic_work_deadlines.py Tests/DB/test_automatic_work_migration.py Tests/DB/test_automatic_wake_attempts.py Tests/Chat/test_automatic_work_lineage.py -q
.venv/bin/python -m pytest Tests/Chat/test_console_chat_create_integration.py Tests/Chat/test_console_chat_create_confirm.py Tests/Chat/test_chat_create_confirm_card.py Tests/Chat/test_console_chat_start.py Tests/Chat/test_console_chat_store.py Tests/Chat/test_chat_persistence_service.py Tests/Chat/test_console_prompt_queue_coordinator.py Tests/Chat/test_console_fleet_wake.py -q
.venv/bin/python -m pytest Tests/UI/test_console_runtime_ownership.py Tests/UI/test_console_fleet_wake_hidden_screen.py Tests/UI/test_console_launch_wake.py Tests/UI/test_design_token_governance.py -q
git diff --check
```

Run `Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py`,
`Tests/Chat/test_console_dispatch_recovery.py`,
`Tests/DB/test_chachanotes_console_library_policy_migration.py`, and the new
`Tests/DB/test_chachanotes_v74_agent_chat_starts_migration.py` as well.
Use `.venv/bin/python -m ruff check --select E9,F63,F7,F82` on the explicit
modified Python files. New files must pass `ruff format --check`; for large
pre-existing owners, record a formatter baseline before changes and verify the
repository ratchet after committing so no unrelated formatting sweep is introduced:

```bash
.venv/bin/python scripts/terminal_qualification/format_ratchet.py snapshot --base HEAD --output /tmp/console-chat-start-format.json --path tldw_chatbook/Chat/console_chat_controller.py --path tldw_chatbook/Chat/console_chat_store.py --path tldw_chatbook/UI/Screens/chat_screen.py
.venv/bin/python scripts/terminal_qualification/format_ratchet.py verify --baseline /tmp/console-chat-start-format.json --head HEAD
```

Include every pre-existing modified file that already has formatter debt in the
snapshot's repeated `--path` arguments. Record checks for both immutable migration
artifacts, new-file formatting and introduced diagnostics; do not waive a new failure.

## Spec coverage and handoff

| Spec section | Plan owner |
| --- | --- |
| Tool contract and four outcomes | Task 2 steps 1–3, 9–10 |
| Destination/defaults/scratch authority | Task 2 steps 2–3, 7, 10 |
| Approval and trusted lineage | Task 2 steps 1–2, 9–10 |
| Draft custody/version compatibility | Task 2 steps 4–5, 8, 10 |
| Shared allowance and automatic context | Task 1 steps 2–6; Task 2 steps 6–7 |
| Shared capacity/manual intent | Task 2 steps 6, 8, 10 |
| Two-store acceptance and recovery | Task 1 steps 3, 5–6; Task 2 steps 5, 7–8, 10 |
| Mounted and live verification | Task 2 steps 9–11; scoped commands |

The design is approved. This plan is the next reviewable artifact; application
implementation has not begun. Execute inline with checkpoints or use delegated
task execution when the user selects that option. Keep task status and unchecked
ACs honest until automated and live evidence is complete.


## Shared reading and ownership reference


Read both the spec and ADR-211, each Backlog task before its changes, and:

- `backlog/docs/lessons-testing-evidence.md`, `lessons-live-verification.md`, `lessons-console-wiring.md`, and `lessons-backlog-hygiene.md`.
- ADR-150 agent creation, ADR-134/135 automatic budgets/recovery, ADR-079 workspace assistant defaults, ADR-069 project instructions and ADR-082 scratch.
- `backlog/docs/design-language.md` before card or screen changes.

The code below defines new contracts or shows the core changes; it is plan content,
not a claim that those interfaces exist today. Existing code locations are search
anchors because the checkout contains concurrent edits.

## File ownership map

| Owner | Files and changes |
| --- | --- |
| Automatic ledger | `DB/AgentRuns_DB.py`, `DB/automatic_work.py`: immutable allowance membership, root-wide accounting, native start-attempt state transitions and recovery |
| Automatic authority | `Agents/automatic_work_budget.py`, `Agents/automatic_work_runtime.py`: typed attempt projection and explicit accepted-attempt kind |
| Tool contract | `Agents/tool_catalog.py`, `Chat/console_agent_bridge.py`: schema, strict validation, per-invocation trusted run provenance and matching grant keys |
| Fresh defaults | `Chat/console_runtime.py`, `Chat/console_chat_controller.py`, `Chat/console_chat_store.py`, `Chat/console_generation_settings_metadata.py`: reuse creation resolver and snapshot codec; save and restore the same destination selection |
| Draft persistence | `Chat/console_chat_store.py`, `Chat/chat_persistence_service.py`: revision-fenced version-2 handoff edits, clears and accepted consumption |
| Durable submission | `DB/ChaChaNotes_DB.py`, `Chat/console_dispatch_checkpoint.py`, `Chat/console_dispatch_repository.py`, `Chat/message_metadata.py`: native machine origin and exact-attempt receipt |
| Start scheduling | New `Chat/console_chat_start.py`; existing `Chat/console_fleet_wake.py`, `Chat/console_runtime.py`, `Chat/console_chat_controller.py`: one immediate attempt, shared slots/owner, runtime-owned execution |
| Manual intent and view | `Chat/console_prompt_queue_coordinator.py`, `UI/Screens/chat_screen.py`, `Widgets/Chat_Widgets/chat_create_confirm_card.py`: manual arbitration before busy guards, observable outcomes, full approval preview |
| Migrations | AgentRunsDB 18→19 and ChaChaNotes 73→74 at the inspected checkout; verify the actual versions at execution and rename both artifacts if another migration has landed |

Only one new production module is planned. Do not extract a generic scheduler,
chat-management service or provider adapter. No stylesheet change is expected;
if an existing token-backed status class is insufficient, change source CSS and
rebuild through `css/build_css.py` under the design constitution.



## Task 1 API reference (dependency)


Add `AutomaticChatStartAttempt` to `Agents/automatic_work_budget.py`:

```python
@dataclass(frozen=True)
class AutomaticChatStartAttempt:
    id: str
    source_run_id: str
    source_chain_id: str
    chain_id: str
    conversation_id: str
    session_id: str
    session_incarnation: str
    owner_id: str
    draft_revision: int
    context_epoch: int
    request_fingerprint: str
    generation_reservation_id: str
    state: str
```

On `AutomaticWorkLedger`, add these exact interfaces:

| Method | Contract |
| --- | --- |
| `allowance_root(chain_id: str) -> str` | Return the direct canonical root ID; do not change scope or renew limits |
| `prepare_chat_start(*, attempt_id: str, source_run_id: str, target_conversation_id: str, target_session_id: str, target_session_incarnation: str, owner_id: str, draft_revision: int, context_epoch: int, request_fingerprint: str, limits: AutomaticWorkLimits | None = None) -> AutomaticChatStartAttempt` | Derive source chain from the persisted run; validate current owner; atomically create the target member, reserve generation 1, and insert the exact attempt |
| `read_chat_start_attempt(attempt_id: str, *, owner_id: str) -> AutomaticChatStartAttempt` | Validate exact persisted owner and return a body-free projection |
| `accept_chat_start(attempt_id: str, *, owner_id: str, limits: AutomaticWorkLimits | None = None) -> bool` | Atomic prepared→accepted cutoff and generation commit; only the first `True` can proceed to the conversation fence |
| `abort_chat_start(attempt_id: str, *, owner_id: str) -> bool` | Release only a still-prepared, proven uncommitted reservation |
| `complete_chat_start(attempt_id: str, *, owner_id: str) -> bool` | Settle accepted attempt once; no wake claim or delivery mark |

Append `attempt_kind: Literal["wake", "chat_start"] = "wake"` to
`AutomaticWorkContext`. Use the kind to select the corresponding ledger reader
in both `mark_accepted()` and `check()`; all existing wake constructors retain
their default. Do not set its acceptance latch until the conversation receipt has
also committed. Source run lineage stays in the attempt; target primaries have no
cross-conversation `parent_run_id`.


## ADR path disambiguation

This repository has several records with the same historical numeric prefix. Read the relevant titles: `backlog/decisions/150-agent-chat-fork-and-spawn.md`, `150-design-token-system-and-design-language.md`, `134-fleet-admission-and-automatic-work-budgets.md`, `135-fleet-completion-delivery-and-crash-recovery.md`, `079-workspace-assistant-defaults.md`, `069-console-project-instruction-local-state-and-preflight.md`, and `082-console-per-chat-private-scratch-space.md`.

## Controller closure supplement: affected preview seam

Backlog AC8: Context Next Send can preview the Console request containing the new chat tools without executing creation or depending on undefined builder references. Repair the existing bridge preview builder undefined fork_chat_tool/new_chat_tool references at the affected tool-aware preview seam. Prove a real builder preview succeeds without creating a chat, and restore scoped syntax lint. This routine bug fix preserves existing runtime/authority boundaries; ADR211 remains the governing ADR.
