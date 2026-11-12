# Console follow-up investigation

Update: the skill-context item below is implemented and verified in TASK-35590.
See [CONSOLE_SKILL_RESULTS.md](CONSOLE_SKILL_RESULTS.md) for before/after numbers,
native lifetime controls, remaining token-estimation stalls and limitations.
The rest of this document preserves the original investigation.

Follow-up: the final Settings/Library tour still shows a 1,167 ms worst key
latency (3.1 ms median across twelve keys). Typing-stage main-thread samples
include local skill-context admission (10), active-workspace reads (8), and
token-estimator setup (4). These overlapping sample counts are not CPU budgets.
This makes local skill-context preparation the next bounded Console target;
see FIX_RESULTS.md for the complete measurement limits.

Read-only source investigation of the retained Windows measurements. No Console
product code was changed and no new performance timing run was performed. This
avoids competing with the Settings/Library before-and-after measurements.

The retained `plain-ready-samples.json` has 298 main-thread samples across
`chat-0`, `chat-1`, `chat-2`, and `return-console`. The stages include post-refresh
settling. Counts below overlap, are not CPU percentages, and cannot be converted
to promised latency savings. Current source was inspected to explain the paths.

## 1. Attention refresh reads durable marks during synchronous visibility changes

41 samples contain `recompute_console_attention` and `list_console_unseen_marks`.
All 41 reach them through pending-decision projection/remount on visibility change.

Source path:

- `UI/Screens/chat_screen.py:22772`: `on_screen_resume` calls
  `controller.on_console_view_visibility_changed(True)` before its attach gate.
- `Chat/console_runtime.py:5109`: `_project_pending_decision_to_attached_view`
  recomputes attention, including when there is no pending decision.
- `Chat/console_runtime.py:1619`: `recompute_console_attention` serializes its
  work, then synchronously reads the durable receipt marks.
- `Chat/conversation_local_marks_service.py:653`: `list_console_unseen_marks`
  explicitly reads uncached through `db.transaction()`, even for an empty result.

This is storage admission/native validation work on the UI thread, not an
expensive SQL query demonstrated by this empty-profile sample. Shell navigation
also calls the same runtime method, so improving only one screen caller would
leave sibling entry points blocking.

Candidate: retain the original checked durable reader, but separate its finite
worker read from the main-loop attention projection and coalesce pending refresh
requests. The runtime already has attention revision checks and last-known marks.
Preserve those checks, one-time notices, hidden-decision publication, failed-read
fallback, and exact rendered-receipt acknowledgement. Never move the whole method
blindly off-thread: its notifications and view hooks belong to presentation.

Regression controls: hold the durable reader while a UI heartbeat and Stop remain
responsive; change active view/receipt/revision before releasing it and reject old
publication; fail a read without clearing known attention; acknowledge one receipt
while a newer receipt lands and keep the newer one. Existing suites include
`Tests/Chat/test_console_terminal_attention.py`,
`Tests/Chat/test_console_rendered_receipt_acks.py`, and
`Tests/Chat/test_console_interrupt_attention.py`.

## 2. Skill discovery is async in name but still performs synchronous local work

30 samples contain `_refresh_console_skill_candidates`, `_fetch_console_skill_context`,
`SkillsScopeService.get_context`, and native local-content admission.

- `UI/Screens/chat_screen.py:22959` schedules skill refresh on every warm resume,
  with `exclusive=False`.
- `UI/Console_Modules/skill.py:62` already prepares cold trust through a checked
  worker seam, but line 103 then awaits the ordinary scope-service context call.
- `Skills_Interop/skills_scope_service.py:165` calls the backend and awaits it.
- `Skills_Interop/local_skills_service.py:1250`: `get_context` is an async function
  with no await in its body. It enumerates `_visible_records`, validates summaries,
  sorts, and builds context text synchronously.
- `Backup_Recovery/local_content_lifetime.py:105`: its async decorator enters a
  synchronous `operation(...)` before awaiting the function.

Thus a Textual coroutine worker does not make this storage/scan path run on a
worker thread. The existing cold-trust optimization does not cover the whole read.

Candidate: a finite original local context read with its full native lifetime in
one worker, retaining the current source/view fences before publishing candidates.
Coalesce same-visit requests while that reader is outstanding. Custom/server
callbacks must retain their own affinity; cancellation must retire the native
read before its owner closes. Merely changing `exclusive=False` to `True` is not
enough and may cancel a waiter while native work continues.

Regression controls: original local context read blocked at admission while the
loop runs; profile/view/source replacement before result return; trusted-to-blocked
skill edit; duplicate resume while a read is outstanding; custom service path.
Relevant existing foundation: `Tests/Chat/test_console_cold_trust_preparation.py`
and TASK-34563.35. Do not treat popup candidates as execution authority.

## 3. FULL reconciliation reads cold configuration before display deferral

32 samples contain `_sync_console_chat_core_state` and provider selection;
31 of them are under attach reconciliation. `load_settings()` is reached before
the later display-only checked projection can defer.

- `UI/Screens/chat_screen.py:17616`: FULL sync calls core state synchronization.
- `UI/Screens/chat_screen.py:9470`: core synchronization derives provider/workspace
  selection and updates the store/controller and agent runtime gate.
- `UI/Screens/chat_screen.py:7399`: readiness uses `load_settings()` for a real
  disk-loaded app snapshot. The per-derivation memo only lives for one opt-in pass.

TASK-35444 already documents this exact remaining gap; its recovery-display fix
does not cover full-sync entry. A held-config-lock regression should call the
actual FULL route, then verify next-Send configuration freshness and attach
completion after release, including replaced/hidden visits.

**Do not pass the existing display projection through this core callback.**
ADR-231 explicitly rejects that alternative: core state publication is
authoritative. The broad polling split remains Proposed; only the bounded stock
manual-Preparing amendment is accepted. Any further split needs the existing
ADR-230/231 ownership design and transition evidence. Moving the whole FULL
body off-thread would also move DOM and authoritative mutations incorrectly.

## 4. Workspace resume's common-case early exit still pays for a guarded read

23 samples contain `Workspaces/registry_service.py:get_active_workspace`.
`UI/Console_Modules/workspace.py:5724` describes an O(1) aligned-session early
return, but reads the registry first at line 5781. The getter at
`Workspaces/registry_service.py:1327` enters a guarded DB connection. Its SQL is
already bounded to one row; reducing row count cannot remove admission cost.

Candidate: prepare the current active workspace off-thread under the existing
owner, then reconcile on-loop with visit/session/currentness checks. Do not cache
the old active workspace across arbitrary returns: Settings and Library can change
it while Console is hidden. The service's mutation generation is a possible local
signal, not by itself proof of external-process or source/profile freshness.
Retain ordered explicit conversation-resume priority and Default/global identity
normalization (TASK-18310, TASK-2033).

## 5. Existing checked-display fallback still appears; qualify before changing

26 samples contain `run_console_config_sync`, reached from rail/control and cost
presentation. `config_sync.py:46` probes the Python locks nonblocking, then line
53 enters native `operation(config)`. That avoids lock waiting but does not make
native admission free. There is already a specialized warm display route in
`console_spend_projection.py:825` that avoids redundant scopes when its proof is
current; expired/unqualified proofs deliberately retain the fresh fallback.

Next diagnostic: record original callback metadata identifying why that fallback
was selected (cold/expired/source-changed/custom) and number of calls per visit,
without retaining frame locals or replacing checked functions. Only then consider
coalescing multiple legitimate refresh requests or sharing one synchronous display
derivation pass. Do not loosen proof age/source/native checks.

Use `Tests/UI/test_console_checked_display_scope.py` as the safety baseline: it
already tests warm no-extra-scope behavior, original native fallback, concurrent
saved-generation invalidation, and pre/post-body owner changes.

## Scaling issue separate from this empty-profile reproduction

`list_console_unseen_marks(limit=100)` fetches every matching row before parsing
and stopping at 100 valid receipts. `has_console_unseen_marks` also fetches all
rows before testing validity. With many durable marks, streaming/chunked reads
can bound materialization; a naive SQL LIMIT can change behavior if malformed
rows precede valid ones. This is a source-observed scaling candidate, not a
measured cause of the current empty-profile delays.

## Sequencing and boundaries

After Settings and Library, start with attention read/publication, then the local
skill scan. They have observed main-thread storage work and narrower boundaries
than general FULL reconciliation. Benchmark sequentially with original checks,
fresh and warm visits separated. Include a real Send and input responsiveness:
the repository's lessons document worker offloading increasing first-Send latency
when the freed loop starts competing native work.

No new architecture decision is made by this investigation. Implementation that
changes attention/read ownership or the FULL contract needs an ADR assessment;
existing ADR-126/230/231 rules remain in force. Existing overlap includes
TASK-35444, TASK-34563.33/.34/.35/.38, TASK-34406.16, and TASK-32804.12.
