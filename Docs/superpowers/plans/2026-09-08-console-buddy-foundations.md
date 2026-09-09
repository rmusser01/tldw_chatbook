# Console Buddy foundations implementation plan

> For agentic workers: use subagent-driven-development to implement the independent
> tasks, followed by targeted review. The user approved implementation; keep working.

**Goal:** Establish verified navigation continuity, independent Buddy artwork, and
consistent workspace Persona defaults before adding the management/reply surfaces.

**Architecture:** Existing ConsoleRuntime/controller/store remain run authorities;
TASK-31520 retains the Console on ordinary tab navigation. Reuse private immutable
visual publication and add genuine Buddy ownership. Reuse WorkspaceAssistantDefaults
for creation-time inheritance, distinguishing omission from explicit None.

**Tech stack:** Python 3.11+, Textual 8, SQLite, existing private asset publication.

**Spec:** `Docs/superpowers/specs/2026-09-08-console-buddy-management-design.md`

ADR required: yes
ADR path: backlog/decisions/139-independent-buddy-conversation-and-workspace-bindings.md
Reason: new visual ownership and scope; explicit-None provisioning amends ADR-079 §5.
Existing lifecycle objective: backlog/decisions/094-console-turn-lifetime-and-navigation-boundary.md,
interpreted against actual reusable-screen behavior from TASK-31520.

## Global constraints

- Work only in `.worktrees/buddy-console-management`, based on c0a42150d1.
- No dummy Personas or implicit active-session reply targets. Persona means assistant
  identity; User Profile is human identity (ADR-037).
- Existing, copied, moved conversations retain assignment. No request broadening.
- Targeted tests only; isolated profile/db; use the main checkout's `.venv` with
  explicit PYTHONPATH/cwd pointing to this worktree. Verify imported source path.
- Keep source artwork notices and unknown license values intact. Do not infer licenses.
- No server changes yet. No unrelated checkout or dependency updates.
- Add new Console logic in UI/Console_Modules; do not grow ChatScreen unnecessarily.

## Task 1 — TASK-32078: navigation continuity

Files: `Tests/UI/test_console_screen_reuse.py`,
`Tests/Chat/test_console_runtime_lifetime.py`,
`Chat/console_chat_controller.py`, `Chat/console_interrupt_rounds.py`,
`Chat/console_runtime.py`, relevant Console lifecycle module hooks and User Guide.

1. Read TASK-31520 and the existing reuse/navigation, interrupt, notification and quit
   tests. Baseline reuse tests pass, but did not exercise actual accepted streams.
2. Add mounted navigation coverage using deterministic stalled gateway execution and
   real parked decisions. Navigate away, release chunks/resolve on return, assert
   no duplicate results, loss or retargeting; explicit shutdown must still cancel.
3. Fix attachment-as-visibility mistakes: a reused hidden Console is not an answerable
   approval surface. Hidden pending decisions notify once and remain available on
   return. Follow ADR-094 answerable-time clocks for configured finite timeouts.
4. Keep normal suspend separate from cancelling final teardown. Verify modal coverage,
   rapid return and background-session behavior. Update inaccurate current guide copy.
5. Run focused reuse/interrupt/approval/runtime suites; record evidence and limitations.

Mounted evidence refinement: a queued turn for the original conversation failed
Canvas selection capture after another conversation became active. Resolve the
accepted run's exact live session/conversation/branch from ConsoleRuntime separately
from the browser's active-view resolver. Keep browser operations restricted to their
existing active-view authority and reject closed or mismatched run owners (ADR-121
`backlog/decisions/121-local-versioned-canvas-artifacts-and-browser-sandbox.md`
and ADR-139 execution ownership).

## Task 2 — TASK-32079: independent artwork ownership

Files: `DB/ChaChaNotes_DB.py` and versioned migration, `Persona_Visual/repository.py`,
`publication.py`, `runtime.py` and shared validation; new `Persona_Buddy/library.py`
(or narrow equivalent), `Persona_Buddy/preferences.py`, `controller.py`, app composition,
existing builtin import and focused DB/Persona_Visual/Persona_Buddy tests.

1. Read existing visual publication/identity contracts and task acceptance criteria.
2. Add a real Buddy record/binding in the next schema migration, including fresh DB
   parity and source-key uniqueness for idempotent legacy migration. Reuse versioned
   pack/assets, but copy a Persona graph into an independently owned pack so later
   publication by the Persona cannot change or invalidate it.
3. Extend shared snapshot/read/publication/runtime authority explicitly for Buddy
   owners while preserving Persona APIs. Do not encode Buddy identity as Persona ID.
4. Expose library list/import/copy/read/select operations for the upcoming modal.
   Publish native imported artwork without Persona creation; preserve supplied notices
   and reject invalid/unbounded assets through existing checks. Export/attribution
   code from earlier approved buddy-import-design work may be adapted where necessary;
   avoid importing unrelated pending features or their entire branch.
5. Extend selection/controller to resolve a Buddy independent of Persona service.
   Migrate selected legacy appearances idempotently, updating config only after commit.
   Preserve geometry and enabled/open state; retain legacy selection on failure.
6. Seed independent built-in artwork without changing existing Persona records or user
   tombstones. Expose display name, stable Buddy ID and immutable graph to the modal.
7. Exercise migration/restart/publication failure, source deletion/edit, builtin and
   native import attribution, and existing Persona compatibility using targeted tests.

## Task 3 — TASK-32080: workspace default consistency

Files: `Workspaces/registry_service.py`, `agent_provisioning.py`, `models.py`,
`DB/Workspace_DB.py` and versioned migration if required; `Chat/console_chat_store.py`,
`console_chat_controller.py`; `UI/Console_Modules/session.py`, `workspace.py`,
`Widgets/workspace_create_modal.py`, Console workspace details and canonical Settings.

1. Reuse ADR-079 storage/resolver. Add omission-vs-explicit-None admission to workspace
   creation and ensure a deliberate clear cannot be backfilled on a later launch.
2. Resolve eligible new blank conversation identity through a shared creation seam
   with the explicit target workspace, rather than ambient selection. Existing,
   restored, forked and explicit Character/Persona/None inputs bypass inheritance.
3. Persist Persona memory mode consistently in settings and session identity.
4. Expose optional defaults in shared workspace creation/details by reusing existing
   Persona selectors and read-write confirmation; preserve canonical Settings controls.
5. Test Console/Settings/Library creation, first activation, bootstrap, new temporary
   chats, explicit None, unavailable default, persistence/reopen/fork/move behavior.

## Review and integration

Tasks 1 and 3 share controller.py; implementations coordinate nonoverlapping methods.
Task 2 owns visual DB migration; Task 3 owns WorkspaceDB migration. Root integrates
composition/UI wires, reviews cross-task state and runs combined targeted suites.
Do not stage or commit another worker's incomplete changes. Each task gets a scoped
review; the completed foundation receives one broader review before UI integration.

## Subsequent approved UI delivery

Continue with TASK-32081 management and scope controls, TASK-32082 conversation replies,
TASK-32083 workspace inbox, then TASK-32084 named speech and end-to-end validation.
The spec is authoritative; write the UI implementation plan against the completed
library/creation interfaces before editing dependent UI. These tasks are intentionally
separate from the foundation to keep execution and artwork authority reviewable.
