# Task 2 implementation report

## Status and review boundary

Implementation prepared against Task 2 BASE `529a9ea9d1c14ba4a2e085c0355c0a304ac562ab`, including independently approved Task 1 `b45ce02976`. Work stayed in the managed `console-chat-starts` worktree; the original checkout supplied only the Python environment. TASK-33805 remains **In Progress**, with ACs unchecked pending independent review. No full suite, dependency change, subagent, reviewer, scheduler, or accepted ADR body change was made. ADR-211 governs the existing approved design.

Implementation commit and final check results are recorded in the closure section below. Controller-owned actual Console evidence is [live-qualification.md](live-qualification.md); it is separate from the scripted-provider tests in this report.

## Result

`new_chat` accepts `destination=same_workspace|casual` and `mode=draft|start`, retaining compatible defaults and its existing title/prompt limits. Destination scope, assistant, settings and approval authority are fixed before confirmation. Saved drafts use a revisioned v2 handoff. A runtime-owned coordinator may start exactly one target turn under the shared automatic allowance and shared wake/start capacity. Ledger acceptance is the source-cancellation cutoff; matching conversation acceptance is the second fence. Provider dispatch and the `started` outcome require both receipts. No automatic replay or queue timer was added.

### Ownership and implementation decisions

- `console_agent_bridge.py` copies the public allowlist, rejects malformed inputs before approval and uses the true current invocation run. The preparation callback is mandatory for new chats. Fork behavior retains its prior contract.
- Controller creation records are bounded to64 and identified by opaque tokens. Both caches key source session incarnation, tool, resolved destination identity and mode. Source checks require the live primary run, matching assistant/cancellation state and persisted nonterminal run. Denials remain per run/tool; every denial, finish, cancellation and shutdown releases the exact record.
- Fresh destination defaults use `ConsoleRuntime._resolve_new_console_assistant`. The controller saves assistant identity, system prompt, generation metadata and handoff in initial conversation creation, then restores without activation. Casual is global/SQL NULL, with ordinary global runtime membership. Scratch/project control and staged inputs are fresh.
- Controller ruling: canonical blank provider uses the ordinary generation codec ABSENT state, while preserving captured settings in the current runtime. Any nonblank configured snapshot remains durable and immutable. Reopen may use ordinary fallback only for the absent case; this is an explicit qualification, not a valid configured snapshot silently dropped.
- The handoff writer coalesces revisions on the owner loop and uses an operation-owned database connection with conversation-version and pending-handoff revision CAS. Edits, including empty text and human edits beyond the tool's input limit, persist until consumption/discard. Manual submit/start/close/shutdown drain custody. Consumption retires only the pending writer and does not clear later human text. The mounted Input.Changed seam now persists explicit Ctrl+U clears immediately into this writer.
- New `console_chat_start.py` is the sole new production Python module. It owns requests, outcomes, exact authorization, target tasks, withdrawal, receipt verification and cleanup. It shares fleet automatic-primary claims (two automatic plus the existing manual reserve) and Task 1's canonical ledger; no second accounting owner was added.
- Source revocation and manual withdrawal serialize against ledger acceptance on the owning loop. Final live checks and acceptance have no intervening await. Conversation receipt commits off-loop. Only after the matching checkpoint is returned does the coordinator call `mark_accepted`. Exact preparation ID and validating-state identity prevent losing cleanup from clearing a replacement manual preparation.
- Accepted targets stay owned independently of source Stop. Target Stop also covers the acceptance-to-conversation-write gap before an assistant owner exists. Close/shutdown include coordinator tasks and draft writers. Known uncertain settlement returns review_required and retains conservative charge.
- Native origin uses ordinary target provider/controller/preparation and configured retrieval while treating slash/@ text literally. It excludes foreground attachments, one-shot prefill, staged evidence, prompt history and trusted profile-message authority. Saved metadata never reconstructs live automatic authorization. Invalid machine provenance is quarantined as untrusted rather than decoded as human.
- Controller ruling: explicit human Retry, both live and reopened, uses ordinary new manual work. Saved request provenance stays machine-origin; old attempt/root membership and charges stay unchanged. Only the exact Retry seam clears automatic execution context for this new dispatch.
- The existing confirmation card shows destination/mode, destination assistant/model and complete instructions. Completion is a view observer with Draft/Started/Not started/Review required copy. Agent request rows show Agent handoff (malformed provenance remains visibly unverified).
- Actual live Stop exposed an inherited composer layout omission: Redirect consumed10 unbudgeted cells, clipping Stop. The existing width calculation now reserves its existing width only while active; no new visual token/theme was added. Native starts project Running instead of Preparing/provider setup. A mounted test checks Stop lies inside its parent and clicks it.
- AC8/controller ruling extends the directly affected Context Next Send preview seam: the builder now receives explicit creation-schema booleans from the controller. This removes inherited undefined live-tool variables without executing creation or changing provider/runtime ownership.

## Files

### Runtime, authority and storage

`Agents/tool_catalog.py`; `Chat/console_agent_bridge.py`; `Chat/console_chat_controller.py`; new `Chat/console_chat_start.py`; `Chat/console_runtime.py`; `Chat/console_fleet_wake.py`; `Chat/console_prompt_queue_coordinator.py`; `Chat/console_turn_preparation.py`; `Chat/console_chat_models.py`; `Chat/console_chat_store.py`; `Chat/chat_persistence_service.py`; `Chat/message_metadata.py`; `Chat/console_dispatch_checkpoint.py`; `Chat/console_dispatch_repository.py`; `Chat/console_roleplay_identity.py`; `DB/ChaChaNotes_DB.py`; new `DB/migrations/chachanotes_v73_to_v74_agent_chat_starts.sql` (all under `tldw_chatbook/`).

### View, tests and documentation

`UI/Console_Modules/prompt_queue.py`; `UI/Screens/chat_screen.py`; `Widgets/Chat_Widgets/chat_create_confirm_card.py`; `Widgets/Console/console_composer_bar.py` (under `tldw_chatbook/`). New `Tests/Chat/test_console_chat_start.py` and `Tests/DB/test_chachanotes_v74_agent_chat_starts_migration.py`; changes to creation integration/confirmation, fleet wake, dispatch checkpoint, library migration, runtime ownership and launch wake test owners. User guide Chat creation tools section updated in place. Task notes retain In Progress and link ADR-211. A concrete testing-evidence lesson records the live clear and clipped Stop incidents. The controller owns the plan supplement and committed live QA artifacts separately.

## Verification method

Every pytest invocation used the original checkout's Python with the **managed worktree as cwd** and an owned unique basetemp under this SDD directory. Large UI/closure runs used documented `TLDW_TEST_GC_EVERY=1`. No optional provider fake is represented as real provider evidence. New integration tests use real SQLite, actual Console controller/agent bridge and ledger owners, with scripted providers and explicit event barriers for fault injection. Mounted tests use Textual Pilot; actual PTY/local-provider qualification is controller-owned.

Command prefix below:

```sh
P=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python
S=.superpowers/sdd/2026-10-02-console-chat-destinations-and-starts
# Each invocation appends --basetemp="$S/pytest-<unique-run>" -q;
# all commands run in /Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook.
```

### Baseline and failure classification

Controller's planned existing-test baseline completed **17 failed,844 passed,9 warnings in977.45s**, before production or old-test edits (`task2-baseline.log`). Source ownership barrier was honored; baseline was not duplicated.

Existing failures: close finalization used `session_id` before assignment; runtime fixture expected older custody fields/to_thread ordering; UI fixtures used the superseded singular wake owner or omitted `chat_controller`; checkpoint fixtures tried corruption through now-protected SQL or omitted mandatory terminal receipt fields. The parent inspected tracebacks and confirmed these owners were unchanged from pre-foundation BASE. The in-scope close bug was fixed. Fixtures now use current fields/owner shapes, valid terminal receipts, and an explicit raw database connection only for deliberate corruption setup. Production semantic guards were not weakened.

Initial implementation regressions were repaired rather than hidden: fresh incarnation UUID changed dataclass semantic equality (now compare=False while authority explicitly compares incarnation); old new-chat fixtures bypassed trusted preparation; a mistakenly placed await caused SyntaxError in prompt_queue.py and the first actual boot (corrected before qualification); live Retry had a local import ordering error; worker draft persistence leaked an operation connection; refund-CAS false incorrectly implied known settlement. Logs preserve intermediate failed runs.

### RED/GREEN record

Commands use `-m pytest` and the file/selection shown. Individual iteration logs remain beside this report; GREEN in a filename alone is not a success claim.

| Command body / behavior | RED evidence | Passing evidence |
| --- | --- | --- |
| `Tests/Chat/test_console_chat_start.py` initial validation/schema/grants/v2 cases | `task2-red-1.log`:26failed | `task2-green-1.log`:26passed6.61s; bridge selection22passed5.72s |
| `Tests/DB/test_chachanotes_v74_agent_chat_starts_migration.py` | `task2-red-db.log`:6failed | Six migration cases passed in `task2-green-native4.log` and final-owner run; runtime/standalone artifact and exact receipt conflict coverage |
| New start file `-k 'preparation or defaults or approval'` and creation selections | `task2-red-create.log`:3failed | Destination selection5passed4.54s; completed owner group below |
| New start file `-k durable_acceptance`; migration duplicate test | `task2-red-consume.log`:2failed; `task2-red-duplicate.log`:1failed | `task2-green-consume.log`:31passed10.19s; exact-duplicate body mismatch repaired and included in owner groups |
| New start file `-k native_start` | `task2-red-native.log`:3failed (module absent); iterations exposed frozen library policy timing | `task2-green-native4.log`:40passed16.91s including migration, controller, both receipts and literal request |
| New start file `-k manual_send` | `task2-red-manual.log`:2failed | `task2-green-manual.log`:2passed2.87s; exact losing-preparation cleanup barrier |
| New start file `-k machine_request_presentation` | `task2-red-label.log`:2failed | `task2-green-label.log`:38passed25.91s (then-current file) |
| New start file `-k 'retry or completion_observer'` | `task2-red-retry-outcome.log`:5failed; live Retry RED1failed | `task2-green-retry2.log`:2passed2.66s; `task2-green-fences-outcome.log`:7passed7.16s |
| New start file `-k 'cutoff or before_cutoff'` | `task2-fences.log`:1failed5passed, target Stop before conversation fence | `task2-green-fences-outcome.log`:7passed; source-after-cutoff, target Stop, conversation-write failure and outcomes |
| New start file `-k 'refused_start or denied_records or coalesces'` | `task2-red-custody.log`:6failed3passed | `task2-green-custody.log`:9passed11.06s |
| New start file `-k 'thread_connection or settlement'` | `task2-red-resource.log`:2failed; `task2-red-refund-cas.log`:1failed1passed | `task2-final-refinement.log`:3passed9.73s includes false/raised settlement and immutable configured reopen |
| New start file `-k visible_pending_handoff`; runtime UI `-k clearing_agent_handoff` | direct event RED2failed; mounted Ctrl+U RED1failed11.07s | `task2-repaired-ui-clear.log`:13passed79.38s includes both event cases and mounted empty clear |
| Runtime UI `-k accepted_agent_chat_start_has_visible_stop` | First barrier drafts incorrectly made run_reply a generator and blocked UI unpacking; these timeouts are not mechanism evidence. Corrected RED `task2-red-stop-preview.log`:Stop edge169 exceeded parent159 | `task2-green-mounted-stop.log`:1passed13.73s, Stop inside parent and actual click→STOPPED; charge1 |
| New start file `-k actual_next_send_builder`; existing UI preview/presentation/cap owners | `task2-red-stop-preview.log`:missing preview args; default run-log enabled then correctly returned uncertain preview | Explicit run-log-disabled actual builder on/off plus existing owner selection: `task2-stop-preview-owners.log`,30passed14.72s; no gateway call or created run/session |
| New start file `-k human-edit-over-tool-cap` then `-k v2_edit_or_clear` | `task2-red-human-edit.log`:1failed3.87s | `task2-green-human-edit.log`:3passed7.14s including20,001-character human edit/reopen |

### Planned closure groups

1. `Tests/Chat/test_console_chat_create_integration.py Tests/Chat/test_console_chat_create_confirm.py Tests/Chat/test_chat_create_confirm_card.py Tests/Chat/test_console_chat_start.py Tests/Chat/test_console_chat_store.py Tests/Chat/test_chat_persistence_service.py Tests/Chat/test_console_prompt_queue_coordinator.py Tests/Chat/test_console_fleet_wake.py`: **700passed410.88s**, `task2-chat-final.log`. Later changes have dedicated new-owner/preview/mounted refinement evidence below.
2. Foundation files `Tests/DB/test_automatic_chat_starts.py test_automatic_work_budget.py test_automatic_work_deadlines.py test_automatic_work_migration.py test_automatic_wake_attempts.py` plus `Tests/Chat/test_automatic_work_lineage.py`, checkpoint/recovery/migration owners: initial combined **230passed14failed1warning89.23s**, `task2-ledger-recovery.log`. All foundation cases passed; failures were confined to checkpoint/recovery/schema owners described above. Rerun `Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py Tests/Chat/test_console_dispatch_recovery.py Tests/DB/test_chachanotes_console_library_policy_migration.py`: **136passed1failed1warning91.27s**, `task2-recovery-fixed2.log`; last fixture still expected the old origin CHECK literal. Corrected case passed in `task2-repaired-ui-clear.log`. Initial collection failure using a nonexistent receipt-field constructor is preserved in `task2-recovery-fixed.log` and was corrected to real serialized metadata.
3. `Tests/UI/test_console_runtime_ownership.py Tests/UI/test_console_fleet_wake_hidden_screen.py Tests/UI/test_console_launch_wake.py Tests/UI/test_design_token_governance.py`: initial **84passed8failed7warnings298.78s**, `task2-ui.log`. Five stale ownership fixtures and three launch tests were repaired. Launch tests now wait for terminal delivery instead of shutting down immediately when the fake provider enters; receipt evidence remains authoritative. All eight reran successfully within the **13passed** repaired selection. No UI failure is claimed green solely because unrelated tests passed.
4. `$P -m pytest Tests/Chat/test_console_chat_start.py Tests/DB/test_chachanotes_v74_agent_chat_starts_migration.py Tests/UI/test_console_prompt_queue.py Tests/UI/test_console_composer_reason_width.py --basetemp="$S/pytest-final-new-owners" -q`: **131passed3failed234.39s**, `task2-final-new-owners.log`. All new start/migration/composer cases passed. Three recovery-presentation tests used a `_FakeChatController` missing the newly required `_chat_start` owner; its fixture now explicitly supplies an inactive owner. `$P -m pytest Tests/UI/test_console_prompt_queue.py Tests/UI/test_console_runtime_ownership.py -k "fresh_controller_projects or restore_restages or discard_releases or accepted_agent_chat_start_has_visible_stop or clearing_agent_handoff" --basetemp="$S/pytest-final-queue-repair" -q`: **5passed104deselected28.39s**, `task2-final-queue-repair.log`, covering every failure and mounted Stop/clear. The later over-tool-cap human-edit case passed separately as recorded above.

No combined total is offered across overlapping runs. The baseline FD-growth warning was +1080 handles; subsequent combined ledger/recovery and UI runs reported +325/+305/+633, plus inherited AST invalid-escape SyntaxWarnings. These warnings are preserved, not called successful resource verification. The new handoff worker connection leak was independently reproduced and fixed with an exact connection-count test. Documented per-test GC was used for later owned UI/final runs, which emitted no FD-growth warning; this does not claim the repository-wide preexisting resource debt is fixed.

## Static and migration checks

Inherited formatter snapshots were captured before editing relevant existing owners: `task2-format.json`, `task2-format-extra.json`, and (before the live Stop fix) `task2-format-composer.json`, and the final queue-fixture `task2-format-queue-ui.json`, all against Task2 BASE. Only introduced/intersecting formatter hunks were normalized; large owners were not whole-file reformatted. Three new Python files must be fully Ruff-clean. `task2-static.log` preserves the initial identical BASE/current F821 diagnostics in the preview builder, at5059/5060. Controller extended AC8 before repair; current full-file scoped syntax lint passes with no diagnostics in `task2-static-final.log`; the earlier debt is not waived.

Required command: `$P -m ruff check --select E9,F63,F7,F82 <every modified/new Python path>`, new-file `ruff format --check`, all four `format_ratchet.py verify --baseline ...` (again with `--head HEAD` after commit), and `git diff --check`. Both Task1's immutable ledger migration artifact and Task2's standalone conversation migration artifact are exercised by the listed migration suites.

## Self-review

Reviewed source ownership/token lifetime, two distinct durable fences, exact cleanup ownership, source versus target Stop, shared slot release, draft CAS/connection lifetime, manual Retry authority, invalid provenance hydration, blank/configured generation states, and view detachment. New module adds no timer or global dispatcher. All automatic call authority still comes from the existing ledger context; no payload or restored metadata can manufacture it. Body text remains in private conversation/draft/request storage and approval card, not ledger identity or generic badge/log reasons.

The review caught the overly broad human draft persistence cap and fixed it with RED/GREEN. The controller's actual QA caught empty-event persistence and action-row clipping that store/backend-only tests had missed. The added preview check uses the actual builder but explicitly disables run logging because the existing disposable preview must decline uncertain writer binding. Existing uncertain-preview coverage still passes.

Remaining qualification: tests inject interruptions at the durable boundaries; they do not simulate disk corruption, power loss or OS hard-kill at every instruction. Actual restart evidence is clean app shutdown/reopen. Full suite was deliberately not run. Independent formal review remains required before Task33805 is marked Done.

## Real Console/provider evidence

Controller's isolated normal application entry point, real OS PTY and configured local llama_cpp endpoint supplied genuine model tool requests, full approval cards and target provider replies. It verified workspace/casual drafts and starts, separated remembered grants, preserved source workspace/chat/composer sentinel, literal `/settings @everyone` machine content, durable edited and empty drafts across restart, preserved configured generation snapshots, disabled-start truthful refusal with zero target messages/attempts, no replay after restart with the gate enabled, and visible target Stop. Final Stop saved assistant stopped with one committed generation charge. See controller `live-qualification.md` for exact captures, receipts, environment and final isolation hash.

Streaming adapter replies had no confirmed token usage: ledger review_required/usage_unknown is the honest conservative outcome, not evidence of confirmed settlement. A separate non-streaming real target completed with confirmed usage. No native macOS screenshot was available; terminal captures are explicitly PTY evidence. Fake providers are not used to claim these real results.

## Closure

- Implementation commit: `c9b3738434b9a1f95a37e0930491b47aa96a779f` — `feat: create workspace or casual chats with bounded starts`;34 files,3598 insertions,286 deletions. Only the controller-owned plan supplement remained modified immediately after commit.
- Final scoped E9/F63/F7/F82 check on every changed/new Python file: exit0, All checks passed. New three Python files: formatter check exit0, already formatted. All four uncommitted formatter ratchets and `git diff --check`: exit0 (`task2-static-final.log`).
- Required post-commit `format_ratchet.py verify --baseline <each of the four snapshots> --head HEAD`: all exit0 against the implementation commit (`task2-postcommit-ratchets.log`). No full-file inherited formatter debt was swept away.
- Final behavior evidence is the planned700-pass Chat group, passing foundation cases, explicitly repaired recovery/UI selections,131-pass added-owner group plus all3 repaired queue cases,30-pass preview/presentation selection, mounted clear/Stop5-pass follow-up and3-pass human-edit/reopen check. The report deliberately retains intermediate failed totals and avoids adding overlapping counts.
- Controller final real qualification completed target Stop and refused-start restart/no-replay. The private app/PTY exited cleanly and the real config SHA256 remained unchanged. See `live-qualification.md` and controller's subsequent committed QA record.
- Concerns/limits: independent review pending; streaming local token usage remains conservatively uncertain, while non-streaming settlement passed; inherited aggregate FD-growth/AST warnings are preserved above; no OS hard-kill/power-loss sweep or full repository suite was claimed. Commit emitted existing Git loose-object/gc warnings; no prune or repository maintenance was performed.

TASK-33805 stays In Progress with unchecked ACs until independent review and controller closure. Implementation is ready for review.


## Task 2 fix round 1 — independent review response

### Boundary and result

FIX_BASE is `c9b3738434b9a1f95a37e0930491b47aa96a779f`. The unified findings are [task-2-review.md](task-2-initial-review.md). This round addresses all five Important findings and the requested native-start → child → wake integration gap. TASK-33805 remains **In Progress** for independent scoped re-review. The controller-owned plan supplement and QA directory were preserved and are excluded from this implementation commit. ADR-211 remains the governing decision; no new architecture owner, production module, schema, scheduler, dependency, or accepted ADR edit was added.

1. **Physical completion owns capacity.** The native coordinator retains the exact shielded bridge-worker task. Stop may finish the visible turn, but coordinator cleanup drains the real worker before completing the native attempt or releasing its shared automatic claim. The original provider argument block stays intact. The strengthened mounted Stop test clicks the actual button, keeps the thread barrier held, waits for an explicit drain timeout and inspects the still-claimed slot before release. A separate real generic-provider-adapter barrier verifies the same ownership below the bridge.
2. **Project decisions precede both fences.** Native preflight checks the frozen project binding, live authority and destination consent key before ledger acceptance and the conversation receipt. A needed selection or consent produces `not_started/project_binding_required` or `project_consent_required`, with the draft pending, no user/assistant rows and no generation charge. The later project setup callbacks also refuse automatic prompting if authority changes after admission. Existing scratch-only behavior remains valid.
3. **Prepared starts permit manual Send.** The real queue presentation exposes enabled Send for a prepared start. The actual mounted click reaches existing exact withdrawal/drain before manual admission. The machine reservation is refunded and the saved user row is ordinary human work. Accepted starts still expose Running/Stop.
4. **Initial preparation has a lifecycle owner.** The shared slot and task are registered before the first ledger-preparation await. Source Stop, caller cancellation and shutdown retain the exact preparation until its thread returns, then settle it. False or raised abort keeps conservative reservation and returns `review_required/settlement_unconfirmed`. A second barrier found during this fix round showed that swallowing cancellation while writing the accepted status could dispatch after Stop; status publication now drains its exact write and propagates cancellation before provider dispatch. Final cleanup publication remains owned until release.
5. **Saved outcome is display-only.** The existing v2 handoff now carries strict bounded `launch={mode,status,reason}` facts. Draft creation records Draft; an unresolved start initially records Review required/outcome_unconfirmed, then the coordinator saves the confirmed outcome. Status writes use the existing private conversation metadata transaction and do not change draft revision/state or either acceptance authority. Hydration reads the label only. Unknown fields, malformed types and body-shaped reasons are ignored. Native/saved browser rows, switcher Active and History use existing vocabulary/classes; bounded blocked/review activity uses existing INPUT NEEDED. Controller live QA found the History widget rebuilt its subline and discarded the correctly projected label; the four-line renderer repair now retains the bounded label and has a mounted real-SQLite History test. Runtime-unavailable early creation also saves its truthful Not started outcome.
6. **Real runtime family integration.** The test uses real SQLite, native coordinator, controller, agent bridge, runtime wake submission and provider gateway, with an HTTP transport double at the external network boundary. One native target launches a child that outlives its initial turn; its settled result triggers a real machine wake. The original canonical root records generations=2, child launches=1, model calls=4 and confirmed tokens=12. Child parent runs stay in the target conversation. Root and descendant retain the original limits/deadline; raising current generation config cannot renew the root, and separate generation-exhaustion/deadline cases refuse another start with no extra provider request. This is deterministic integration evidence, not a live provider claim.

### Files and plan deviation

Changed production owners: `Chat/console_chat_start.py`, `console_chat_controller.py`, `chat_persistence_service.py`, `console_chat_store.py`, `console_chat_models.py`, `message_metadata.py`, `UI/Console_Modules/prompt_queue.py`, `workspace.py`, and `Widgets/Console/console_session_switcher_modal.py`. Updated the existing user-guide creation section. Test changes are in `Tests/Chat/test_console_chat_start.py`, `test_message_metadata.py`, `Tests/UI/test_console_runtime_ownership.py`, and `test_console_prompt_queue.py`.

**Plan deviation:** The original brief listed edits to several existing owner test files (`test_chat_create_confirm_card.py`, `test_console_chat_store.py`, `test_chat_persistence_service.py`, `test_console_prompt_queue_coordinator.py`, and `test_console_dispatch_recovery.py`). Most new scenarios were consolidated into `test_console_chat_start.py`, using those real owners and existing fixture helpers, while the existing owner tests were run as regression coverage. Those listed files were not all amended; this report does not imply otherwise. This fix round follows the same consolidation, with actual mounted controls retained in the runtime UI owner and exact old-fake contract repairs made in place.

### RED/GREEN and final targeted commands

Use the existing `P` and `S` prefixes defined above; every command used the managed worktree as cwd, `-q --tb=short`, and the unique `--basetemp=$S/<name>` listed below. Large groups used `TLDW_TEST_GC_EVERY=1`. Outputs remain verbatim in the named logs. No full sweep was run.

| Command after `$P -m pytest` / unique basetemp | Evidence |
| --- | --- |
| `Tests/Chat/test_console_chat_start.py -k 'stop_keeps_native_claim or initial_preparation or native_project_decision or refused_native_outcome'`; `pytest-fix1-red` then `pytest-fix1-green` | `task2-fix1-red.log`:9 failed/1 passed; `task2-fix1-green.log`:10 passed/73 deselected,29.32s. The first physical probe was too early and passed; strengthened held-worker RED below established the actual failure. |
| Start file `-k stop_keeps_native_claim`; `pytest-fix1-physical-red` | `task2-fix1-physical-red.log`:1 failed/82 deselected,2.45s, because the coordinator drained before the held thread. Fixed barrier passes in the final102 group. |
| `Tests/UI/test_console_runtime_ownership.py -k prepared_native_start_allows_mounted_manual_send`; `pytest-fix1-manual-red` | `task2-fix1-manual-red.log`:1 failed/71 deselected,20.75s, actual Send disabled. First integration rerun needed a current frozen UI snapshot after synchronization; mounted Send passed in `task2-fix1-manual-family2.log` and final controls below. |
| Start file `-k 'native_start_child_and_wake or stop_during_started_status'`; `pytest-fix1-family-status-red` | `task2-fix1-family-status-red.log`:3 failed. Cancellation case reached1 provider call instead of0. Earlier family iterations exposed a missing real archive service, an insufficient resolution double, and then the need to move the provider double below gateway accounting. Those are fixture failures, not repaired production behavior. |
| Start file `-k 'native_start_child_and_wake or saved_launch_status_projects or stop_during_started_status'`; `pytest-fix1-family-activity` | `task2-fix1-family-activity.log`:3 passed/1 failed,8.59s. Real gateway family/deadline and Stop-during-publication passed; activity was not yet projected. |
| Start file `-k saved_launch_status_projects`; `pytest-fix1-mounted-history-red3` | `task2-fix1-mounted-history-red3.log`:1 failed showing the actual rendered `SAVED CHAT / Chats · Saved chat · now` label. Earlier two attempts corrected a Textual query API misuse and the fixture's mismatched authority/query state; they are not mechanism RED evidence. |
| Start file plus `Tests/UI/test_console_activity_switcher.py -k 'saved_launch_status_projects or result_rows_use_left or zero_active_matches_widens'`; `pytest-fix1-mounted-history-green` | `task2-fix1-mounted-history-green.log`:3 passed/133 deselected,8.71s. |
| Start file `-k creation_without_owner_loop`; `pytest-fix1-no-loop-red` | `task2-fix1-no-loop-red.log`:1 failed because saved status remained Review required while result reported Not started; fixed in10-case fixture follow-up and final102 group. |
| Start file `-k generic_provider_adapter`; `pytest-fix1-adapter-barrier` | `task2-fix1-adapter-barrier.log`:1 passed/94 deselected,3.58s. Real generic provider thread held after Stop; claim remains until actual exit. |
| `Tests/Chat/test_console_chat_start.py Tests/DB/test_chachanotes_v74_agent_chat_starts_migration.py`; `pytest-fix1-final-starts` | **102 passed,124.11s**, `task2-fix1-final-starts.log`. Covers all new deterministic review scenarios and migration contracts after final production edits/formatting. Earlier full owner run:100 passed95.75s, `task2-fix1-start-final.log`. |
| `Tests/UI/test_console_prompt_queue.py Tests/UI/test_console_runtime_ownership.py Tests/UI/test_console_activity_switcher.py -k 'prompt_queue or native_start or prepared_native or prepared_start or active_is_immediate or history'`; `pytest-fix1-ui-final` |48 passed/8 failed/95 deselected80.79s, `task2-fix1-ui-final.log`. All8 failed because the existing fake lacked `withdraw_for_manual`; no production guard was relaxed. |
| `Tests/UI/test_console_prompt_queue.py Tests/Chat/test_console_chat_start.py -k 'dispatch_admits_exact or dispatch_stages_one or runtime_custody_succeeds or dispatch_uses_explicit or synchronous_custody_refusal or dispatch_keeps_captured or pre_acceptance_race or finished_chain_boundary or creation_without_owner_loop or saved_launch_status_projects'`; `pytest-fix1-fixtures-green` | **10 passed/122 deselected,7.17s**, `task2-fix1-fixtures-green.log`; every one of the8 UI failures reran successfully plus the early outcome and row cases. |
| `Tests/UI/test_console_runtime_ownership.py -k 'accepted_agent_chat_start_has_visible_stop or prepared_native_start_allows_mounted_manual_send'`; `pytest-fix1-mounted-controls-final` | **2 passed/70 deselected,26.86s**, `task2-fix1-mounted-controls-final.log`. Final Stop keeps the barrier held for0.5s after the actual click and checks the shared claim before release. |
| `Tests/Chat/test_console_agent_project_instructions.py Tests/Chat/test_console_fleet_wake.py Tests/Chat/test_automatic_work_lineage.py`; `pytest-fix1-related-owners` | **79 passed/2 failed,80.98s**, `task2-fix1-related-owners.log`. Both failures were reproduced at exact FIX_BASE as described below; all fleet/lineage cases passed. |
| `Tests/Chat/test_console_agent_swap.py Tests/Chat/test_message_metadata.py -k 'stop or cancel or close or metadata'`; `pytest-fix1-stop-metadata` |61 passed/1 failed/35 deselected25.04s, `task2-fix1-stop-metadata.log`. The old origin-vocabulary test constructed the new machine origin without mandatory provenance. The fixture now supplies exact provenance and also asserts its absence is refused. |
| `Tests/Chat/test_message_metadata.py`; `pytest-fix1-metadata-green` | **51 passed,5.74s**, `task2-fix1-metadata-green.log`; full metadata owner passes, including the repaired fixture. |

No sum is offered across overlapping selections. Neither a log filename containing “green” nor unrelated passing cases is treated as proof of a failed scenario.

### Exact baseline qualification for the two residual failures

The two direct AgentService failures are `test_child_chain_uses_its_own_exact_first_request_budget` and `test_primary_token_omission_is_delivery_local_when_child_admits`. Both assert `outcome.status == RUN_DONE` and receive `error`. A disposable reporting-only pytest plugin records the actual outcome: each test's monkeypatched `_count_model_messages` lambda rejects the existing keyword `reasoning_replay`. Current evidence is `task2-fix1-budget-probe.log`.

Unchanged source was not used as sufficient baseline proof. `git archive c9b3738434b9a1f95a37e0930491b47aa96a779f tldw_chatbook Tests pyproject.toml config.toml` was extracted into the isolated owned `fix1-baseline-revision/` directory, with that directory as cwd and first PYTHONPATH component. Thus application, ledger, metadata and tests all resolve from the exact fixed baseline. Command:

```sh
# cwd=$S/fix1-baseline-revision; absolute S/P as above
PYTHONPATH="$PWD:$S" "$P" -m pytest -p task2_fix1_budget_probe   Tests/Chat/test_console_agent_project_instructions.py -q --tb=short   -k 'child_chain_uses_its_own_exact_first_request_budget or primary_token_omission_is_delivery_local_when_child_admits'   --basetemp="$S/pytest-fix1-budget-baseline"
```

Result: **2 failed,38 deselected,1 warning,3.27s**, `task2-fix1-budget-baseline.log`, with the identical unexpected-keyword outcomes. The warning is the inherited invalid escape in `Tools/patch_tool_impls.py:32`. No unrelated test lambda, AgentService budget path, or production budget semantics was changed. These remain a qualified preexisting test-fixture debt for the reviewer.

### Static checks, approval and infrastructure evidence

- Before mutations, captured `task2-fix1-format.json` against FIX_BASE; added snapshots before touching the queue fake, History widget and metadata test: `task2-fix1-format-queue-ui.json`, `task2-fix1-format-switcher.json`, `task2-fix1-format-metadata-test.json`. Existing Task2 snapshots remain authoritative too. Only introduced/intersecting hunks were normalized; new feature module/test and new migration test stay fully formatted.
- `task2-fix1-static-final.log` records exact full commands and exit0 for scoped `ruff check --select E9,F63,F7,F82` on every amended Python file, `ruff format --check` on all3 new Python files, all8 formatter ratchets, and `git diff --check`. Post-commit verification is recorded below.
- A broad coordinator replacement was rejected **before execution** by automatic approval review: “This replaces the core chat-start coordinator lifecycle across preparation, cancellation, provider execution, settlement, outcome persistence, and shutdown; mistakes could leak automatic claims or mis-settle durable generation charges, and the broad rewrite is not specifically authorized.” No broad replacement was applied. The controller reaffirmed existing approval and required smaller patches. Separate exact-worker retention, initial preparation custody and outcome-publication edits were accepted. [task2-fix1-approval-rejection.md](fix-approval-review.md) preserves the event. No unresolved approval block remains.
- `task2-fix1-family-row-green.log` is an **infrastructure failure**, not GREEN evidence: the Data volume had117MiB free and SQLite reported disk-full/I/O errors. With controller authorization, only4 completed owned disposable pytest basetemps were removed (`pytest-task2-chat-final`, `task2-regression-chat2-temp`, `pytest-task2-ledger-recovery`, `pytest-task2-recovery-fixed2`). All logs/reports/baselines/live evidence were retained. Space recovered to2.1GiB before reruns; later read-only check showed34GiB. No user data, Git prune, original checkout, live profile, or running test directory was removed.

### Self-review and actual live qualification

Self-review checked task registration before prepare, shielding of exact physical work, cancellation propagation during accepted-status persistence, both receipts before authority/Started, no target dialogs for required project decisions, exact-token manual cleanup, conservative false/raised refunds, display-only hydration and shared canonical root/deadline in real native→child→wake execution. The publication cancellation test and real generic adapter barrier were added because bridge-only or ordinary success tests would miss those timing risks. All displayed status strings use closed vocabulary; prompt bodies stay out of labels and generic diagnostics.

Controller-owned actual PTY/provider follow-up is in [live-qualification.md](live-qualification.md) and `Docs/superpowers/qa/2026-10-02-console-chat-starts/live-qualification.md`. Normal boot7/8 verified new refusal metadata, browser Not started, Active INPUT NEEDED, pending draft across restart with the gate enabled,0 target messages/checkpoints/attempts and total attempts unchanged at6. The live History omission was reproduced and fixed; normal boot9 then visibly rendered **Chats · Saved chat · Not started**. No prompt was sent during that final read-only boot. The controller closed app/PTY cleanly and confirmed the real configuration SHA256 was unchanged. Earlier live Stop establishes UI behavior/charge; physical thread completion is established by the deterministic held-barrier tests in this round, not retroactively claimed from that live capture.

Remaining limits: independent scoped re-review is required; no full repository suite or OS power-loss/hard-kill sweep was run; the2 preexisting project-budget fixtures fail identically at FIX_BASE; original streaming usage uncertainty and earlier aggregate FD-growth evidence remain as qualified above. Final targeted round1 runs emitted no new resource-growth warning, which does not establish a repository-wide cleanup fix.

### Round 1 commit and post-commit closure

Commit SHA and post-commit checks are appended after the implementation commit. Task status and unchecked ACs remain unchanged.


Final self-review refinement: the no-owner-loop case saved Not started correctly but left the current restored session at its initial Review required. `task2-fix1-no-loop-display-red.log` records the exact current-session assertion failure. The same bounded status is now published on the existing app loop after the successful metadata write. `$P -m pytest Tests/Chat/test_console_chat_start.py -q --tb=short -k 'creation_without_owner_loop or refused_native_outcome or saved_launch_status_projects' --basetemp="$S/pytest-fix1-no-loop-display-green"` passed **3/92 deselected in8.20s**, covering early result, reopen and actual mounted History. This narrow final edit follows the102-pass owner run; its affected paths were rerun rather than implying the102 run occurred afterward. Scoped lint, new-file format, all8 formatter ratchets and diff check were rerun after it and passed (final `task2-fix1-static-final.log`).

Round 1 implementation commit: **71e5fe1b9c95dfd1e21908409e5868dbac0eb795** (`fix: preserve native chat start ownership and saved outcomes`), 14 files changed, 1108 insertions and 73 deletions. Commit completed successfully. Git emitted an existing automatic-gc warning about unreachable loose objects; no prune or repository cleanup was attempted.

Post-commit command: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python /private/tmp/task2-fix1-static.py HEAD` from the managed worktree. Result: exit 0, **All scoped static checks passed**. Exact commands and results are in `task2-fix1-static-postcommit.log`: scoped syntax lint, full formatting of all three new Python files, all eight existing-file formatter ratchets against committed HEAD, and `git diff --check` each returned 0. `git status --short` after the commit contains only the controller-owned plan supplement and QA folder; both were preserved and excluded from this implementation commit. No staged changes remain. Task remains In Progress pending independent scoped re-review.

## Task 2 fix round 2 — historical launch status and current activity

Fix base: `71e5fe1b9c95dfd1e21908409e5868dbac0eb795`. Authorized scope is the one Important finding in `task-2-fix1-review.md`; its exact isolated recovery probe is `task-2-fix1-recovery-probe.py`. The original five repairs and native→child→wake integration were accepted by that review. Task status remains In Progress pending this scoped re-review.

### Change and ownership

`ConsoleChatController.activity_for` now publishes a handoff launch label as live activity only while handoff custody is unresolved (`pending` or malformed/unknown-version `review_required`). Successful ordinary Manual Send consumes the handoff through the existing durable submission owner, so its historical Not started/Review required label no longer creates a false blocked signal in Active. Existing run/queue/approval signals retain their normal precedence. Session metadata, the persisted launch record, browser rows and History retain the original launch mode/status/reason.

No dispatch, admission, persistence mutation, allowance, recovery-authority or stylesheet code changed. ADR-211 remains the governing decision; this is a correction within its existing status and recovery contracts, not a new boundary. The user guide now explains that handoff attention clears on successful manual consumption while launch history remains visible.

Files owned in this round:

- `tldw_chatbook/Chat/console_chat_controller.py`: derive live handoff activity from current unresolved custody.
- `Tests/Chat/test_console_chat_start.py`: two real SQLite/controller recovery cases, using the production workspace Active/History projection.
- `Docs/User_Guide/console/agent-runs-and-tools.md`: clarify historical launch status versus current handoff attention.

Plan deviation: the regression remains consolidated in the existing native-start integration test owner. No separate UI test file or new model field is needed: the test calls the actual workspace projection with the real controller and store. The production UI consumer is unchanged because its controller activity input now represents current handoff custody. Controller-owned plan/QA edits remain untouched.

### RED / GREEN evidence

The regression catches this concrete break: after a refused native launch, ordinary human Manual Send completes and consumes the handoff, but the actual Active entry remains `blocked` / `WAITING_FOR_YOU`. Parameterized setup uses (1) the actual runtime-disabled refusal and (2) an unavailable preflight plus injected false ledger refund to produce a real `review_required` outcome. Only those failure boundaries are injected; submission, durable conversation/checkpoint operations, agent runtime, launch metadata codec, allowance ledger and Active/History projection are real. The existing scripted provider fixture is deterministic test evidence, not a live provider claim.

Both cases first assert that the unresolved pending handoff is blocked, then perform ordinary `submit_draft`, await runtime work, and require MANUAL origin, provider_started, completed run, consumed handoff, no live/queued work, and unblocked Active. They also verify the original launch facts remain in memory/SQLite/reopened metadata/History and the previous root's generation reservation remains respectively 0 or 1. Recovery never reassigns or refunds the earlier uncertain reservation.

Commands run from `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook`, using:

```sh
P=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python
S=.superpowers/sdd/2026-10-02-console-chat-destinations-and-starts
TLDW_TEST_GC_EVERY=1 "$P" -m pytest Tests/Chat/test_console_chat_start.py -q --tb=short \
  -k manual_recovery_clears_handoff_attention --basetemp="$S/pytest-fix2-recovery-red"
TLDW_TEST_GC_EVERY=1 "$P" -m pytest Tests/Chat/test_console_chat_start.py -q --tb=short \
  -k manual_recovery_clears_handoff_attention --basetemp="$S/pytest-fix2-recovery-green"
```

- RED (`task2-fix2-recovery-red.log`): **2 failed, 95 deselected, 6.04s**. Both fail at the exact post-completion assertion: `activity_state` is still `blocked`, after accepted MANUAL/provider_started/completed/consumed assertions pass. No production edit preceded this RED.
- GREEN (`task2-fix2-recovery-green.log`): **2 passed, 95 deselected, 5.48s** after the narrow projection edit.

### Static evidence and review

Before editing the old controller, captured its exact FIX_BASE formatter debt with:

```sh
"$P" scripts/terminal_qualification/format_ratchet.py snapshot \
  --base 71e5fe1b9c95dfd1e21908409e5868dbac0eb795 \
  --output "$S/task2-fix2-format.json" \
  --path tldw_chatbook/Chat/console_chat_controller.py
```

`task2-fix2-static-final.log` records exit 0 for scoped Ruff `E9,F63,F7,F82` on both amended Python owners, full `ruff format --check` on the three new Task2 Python files, all three affected controller formatter baselines (`task2-format.json`, `task2-fix1-format.json`, `task2-fix2-format.json`), and `git diff --check`. Command driver: `$P /private/tmp/task2-fix2-static.py`; it records every exact subprocess command and result. The controller's preexisting whole-file formatting debt was not mass-formatted or described as clean.

Self-review confirms the fix changes presentation only, preserves unknown-version review handling, leaves ordinary queue/run/approval attention intact, and retains launch history without turning it into execution authority. The two earlier direct AgentService fixture failures remain the separately proven baseline qualification from round 1; this round does not repair or rerun those unrelated fixtures. No full sweep was requested or run.

Final targeted group, controller live evidence, commit SHA and postcommit checks follow below.

### Controller-owned live recovery qualification

After focused GREEN, the controller ran normal isolated Console boot10 against the existing real local provider and saved `LIVE_REFUSED_FIXED` target. Ordinary manual Send completed the local-model reply. The filtered Active surface showed CURRENT without Input needed / Waiting for you, while its row retained historical Not started. Read-only receipt inspection found consumed v2 custody, a user request and complete assistant, one ordinary manual primary/root, zero target native attempts and total native attempts unchanged at 6. Thus manual recovery did not replay the old automatic launch. This is controller-supplied actual app/provider evidence; the deterministic tests above are separate.

The controller closed the owned app and PTY cleanly (client session 18667 exit 0) before releasing the final targeted pytest group. Canonical captures/receipt and configuration-hash evidence are maintained by the controller in `Docs/superpowers/qa/2026-10-02-console-chat-starts/`; implementation commits do not take ownership of those files.

### Final targeted verification

After the live app was closed, ran:

```sh
TLDW_TEST_GC_EVERY=1 "$P" -m pytest \
  Tests/Chat/test_console_chat_start.py \
  Tests/Chat/test_console_switcher_state.py \
  Tests/UI/test_console_activity_switcher.py \
  -q --tb=short --basetemp="$S/pytest-fix2-final"
```

Result: **163 passed in 127.97s**, exit 0, recorded in `task2-fix2-final.log`. This includes the entire native-start integration owner, both new manual-recovery cases, pure Active/History state handling and mounted switcher consumers. No pytest warning summary or new resource-growth warning was emitted. This targeted run does not claim a full repository sweep or resolve earlier separately qualified baseline/aggregate resource observations.

The canonical controller QA now records boot10 captures/receipt and confirms unchanged real configuration SHA256 `15c6cb224a6a51c7de5c3f716fbe9dfaef7b7ca05cf9daa7e3ebe247df9310da`. The Review required / uncertain-refund recovery variant is established by deterministic controller tests; it was not separately injected into the live app.

### Round 2 commit and postcommit closure

Commit: **967d52e1dc5ef611586417e6b7d1d08397028e33** (`fix: clear resolved handoff attention after manual recovery`). Three owned files changed, 129 insertions and 3 deletions. Git again emitted its preexisting automatic-gc / unreachable-object warning; no prune or unrelated cleanup was attempted.

Postcommit command: `$P /private/tmp/task2-fix2-static.py HEAD`. Result: exit 0, **All round 2 scoped static checks passed**. `task2-fix2-static-postcommit.log` records exact commands and exit 0 for scoped syntax lint, all three new-file format checks, all three affected controller formatter ratchets against committed HEAD, and `git diff --check`.

Final status contains only the controller-owned plan supplement and untracked QA folder, preserved outside this commit. The index is clear. No new functional concern remains from this round's targeted evidence; independent scoped re-review remains required. Earlier baseline fixture/resource/streaming qualifications remain as recorded, and Task 2 is still In Progress.


## Final whole-branch fix wave

All four final-review findings were repaired in **9cb68f456ee749ea2fc9209ba1edc73f22f15e73**: archived/removed and changed destinations refuse before mutation/admission, current runtime/bridge/owner is rechecked at acceptance, approval discloses remembered supplied-body authority and explicit instruction override, and the canonical missing-Persona notice reaches the existing preview/target notice owner. Exact approved destination also survives the asynchronous library-capture boundary.

See `final-fix-report.md` for RED/GREEN commands, full logs, self-review, real Console boot11 evidence provenance and postcommit verification. The complete amended owner group passed **140 tests** before the final one-line capture refinement; the final capture/admission selection passed **14 tests**. Postcommit scoped lint, new-file formatting, all five old-owner formatter ratchets against HEAD and diff checks passed. The broad owner group’s inherited fork fake failure is reproduced at exact FIX_BASE with a verified 6,615-blob manifest and shown to originate at overall BASE; the broad +258 FD warning remains qualified. Counts overlap and are not summed. Controller Backlog/QA changes remain uncommitted by this wave. The controller owns the sole scoped re-review and eventual task closure.
