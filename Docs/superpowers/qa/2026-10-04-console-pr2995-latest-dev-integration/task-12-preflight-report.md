# Task12 latest-dev read-only preflight

Status: coherent integration candidate. No tracked/index/HEAD/ref/object edits, fetch/rebase/commit, tests, production execution, child agents or current-working-source maps. All feature reads use immutable `5c0c57b48f8d478e620a7bd1dafc7785f4e046d9`; upstream raw files use exact GitHub API ref `7d155170dc95557736a239a1ce7427981f4d50ec`; common dev is `1b12df2757f901e5c8e7a1e8946ad98f315eb8f9`. Task9's later summary trigger and remaining two fixture spy changes are separate root-controlled work and were not mapped as frozen source here.

The API compare is **26 commits / 46 paths**, including the task rename's removed and added paths. Eleven production paths change. Read current upstream task plans for TASK33628.7/.8/.9/.11/.12,32564,33003.20,31245 and ADR067. Feature ownership follows frozen ADR219/220 and Task9 source constraints, Task10 actual report. No new ADR is required for integration: existing ADR067/092/120/219/220 govern these same contracts.

## Exact intersection and resolution

The private exact upstream patches apply to all eleven frozen production files with exit0 and **no fuzz**. Every incoming changed function/method (and the changed destination dataclass) in the private raw proposal has the exact upstream AST. Every other declaration retains the frozen AST. There is **zero changed-method intersection**; cumulative class AST changes are not method conflicts. This is source compatibility evidence, not a passing integration test/rebase claim. BSD `diff3` was unusable on several large files (`invalid print range`); its partial outputs were discarded. The final proposal was constructed by exact unified patches with `patch --batch --forward`, checked by AST and formatter in task-private storage.

| Incoming owner | Exact changed behavior / resolution |
| --- | --- |
| `ConsoleChatStore._delete_message` | Preserve one `_dispatch_branch_mutation` for all deletes; use `console_legacy_flat_roots.delete_seeds`, selected saved ID or first saved descendant as anchor; keep active-leaf persistence inside the mutation and project committed tombstones afterward. Both unsaved-note Resend and hidden flat roots must persistently disappear and Undo must restore committed tombstones. |
| `ConsoleChatStore._resolve_voice_promotion_destination`, `_publish_voice_pair`; `ResolvedVoicePromotionDestination.__post_init__`; `ChatPersistenceService.commit_completed_voice_pair`, `_reconcile_completed_voice_pair` | Carry immutable `user_root_fork=False` destination field; refuse True with a parent; derive marker using existing USER/no-parent/root/fork-projection rule; write and reconcile exact expected user metadata JSON; publish in-memory marker so full-record rewrite and temporary save retain it. Preserve every other exact retry/identity/receipt check. |
| `console_legacy_flat_roots.voice_user_root_fork`, `delete_seeds`, `_root_rows`, `hidden_rows_after`, `_was_chained`, `_never_shown` | Preserve six incoming functions and `ROOT_ROWS_PAGE_SIZE=500`. Read only parentless roots, keyset pages, presence flags. Gate requires saved native parent; hidden marked root and shown image/generation/continuation roots survive. Prompt count still counts transcript nodes. |
| `CharactersRAGDB.get_root_message_rows_page`, `_update_message_uncoordinated` | Add bounded ordered root projection without content/image bytes; retain conversation/live predicates and raw timestamp/rowid cursor. Preserve both recursive CTE unary-plus conversation terms so each descent uses parent index without statistics. No schema/migration/catalog change; native receipt schema/Task7 semantic writer bodies remain outside changed methods. |
| `_StdioJSONRPCConnection.__init__`, `close`, `_handle_incoming_payload`, `_handle_server_request`, new `_handle_server_cancellation` | Keep `_server_request_tasks` separate from outgoing `_pending_requests`; exact int/string IDs (exclude bool/float), duplicate active-ID ownership, cancel only referenced inbound task, identity-safe done cleanup and disconnect map cleanup. Existing live elicitation `finally` expires cancelled confirmations; late approval cannot revive them. |
| `build_live_elicit_fn` (nested `_timeout`, `elicit`), `UnifiedMCPControlPlaneService.approval_timeout_seconds` | Missing/unparsable config fallback120→0; deadline only for positive values; retain confirmation-only schema gate, answer/deny and finally expiry. Positive finite timeouts remain. |
| `ConsoleWorkspaceController._open_character_conversation_activation` | Await exact `_sync_native_console_transcript` before existing focus/ready proof. Cancel before ownership transfer synchronously rolls back only exact owned cold runtime and rethrows; warm/prior restoration stays caller-owned. Task9 Session recovery adapters and Task7 Session/adoption/hooks do not move. |
| Controller and config | Incoming controller is only `task201/T2`→`ADR067` documentation; constant is already0. Config is one template comment covering both consumers. No incoming resolver method modification exists. |

Task10 pure fork projection methods/functions do not overlap the three store changes. Task8 Close/source/acceptance methods do not overlap them. Task7 adopted Session/hooks and native receipt methods retain frozen ASTs in the proposal. No standing prompt, workspace bindings, grants, run budgets, launch metadata, source observation, lock ownership or AgentRuns acceptance fence is reassigned.

## Actual Task9 resolver route and required fixture adaptation

Frozen controller `ConsoleChatController._resolve_mcp_approval_timeout_seconds` at15510–15511 forwards to `self._interrupt_host._resolve_mcp_approval_timeout_seconds`. Actual implementation is `InterruptRoundHost._resolve_mcp_approval_timeout_seconds` at2811–2826. It dynamically reads controller timeout seam, controller-module `get_cli_setting`, and controller-module `_DEFAULT_MCP_APPROVAL_TIMEOUT_SECONDS=0.0` through named getters. Controller init bindings are at4433/4463/4538; no incoming production change belongs in this moved body. Preserve current live lookup and nullable seam; do not paste the upstream old controller body back.

The new upstream `Tests/MCP/test_approval_timeout_policy.py::test_console_and_service_share_approval_timeout_policy` creates `ConsoleChatController.__new__`, sets `mcp_approval_timeout_seconds=None`, then calls that wrapper. It has no `_interrupt_host`, so source inspection establishes an uninitialized fixture route. No test was run to label this RED.

Smallest fixture proposal: import canonical `Tests.Chat.console_interrupt_test_bindings.make_interrupt_host` (**signature `make_interrupt_host(seams)`**) and assign `controller._interrupt_host = make_interrupt_host(controller)` after the existing seam assignment. No additional fake fields are required for this resolver: the adapter supplies the existing nullable `read_controller_mcp_approval_timeout_seconds` getter and live `read_global_get_cli_setting` / `read_global__DEFAULT_MCP_APPROVAL_TIMEOUT_SECONDS` getters, creates native host registries/lock, and handles optional historical aliases with `hasattr`. It does not call unrelated controller services merely to construct the host.

Preserve every original statement/assertion and parameter list exactly: `[("missing",0.0),(None,0.0),("bad",0.0),(0,0.0),(-1,-1.0),(30,30.0)]`. Retain controller/service config monkeypatches and both equality assertions. The adapter's globals remain late-bound to the controller module, so the original patch reaches the moved body. No new marker/profile policy is needed for this patched configuration boundary.

## Exact private count / formatter evidence

Installed Ruff0.16.6 was invoked through stdin with the frozen `pyproject.toml` and each real source filename. Nothing was installed. No whole-file formatted output was applied to tracked source.

| Module | Frozen raw / formatted | Private integrated raw / formatted | Existing cap |
| --- | ---: | ---: | ---: |
| Store |22338 /22338|22332 / **22334**| **22344** |
| Controller |29327 /29320|29327 /29320|29367|
| Workspace |8002 /8002|8015 /8015| existing row carried |

Store has **10 actual formatted lines of slack**. Upstream raw store22493 formats22495: the same two blank-line differences appear in its incoming local-import hunks. Frozen store is fully formatted. Preserve targeted honest formatting around those incoming hunks; no cap change. Existing fork owner's Task10 bytes/280 qualification carry because it receives no upstream patch.

Whole-file formatter inheritance already exists in frozen controller, persistence, MCPclient/live wiring and config. Private formatting of them was measurement only; `format-map` records raw/formatted/AST identity for frozen/dev/proposal. Do not apply their unrelated whole-file drift. `console_legacy_flat_roots` is new upstream-owned formatting drift221→381raw→385formatted; its four formatter lines can be qualified as incoming scope. DB proposal24283 is already formatted although upstream DB has older unrelated drift. Controller/host eventual Task9 source counts need the root's clean handoff pin; this preflight does not assert its later trigger bytes.

The source changes add no new eager project module: store imports legacy-flat owner locally in the affected methods; that owner was already used by existing store paths. Legacy-flat's hydration role import remains local in `_never_shown`, and voice context is TYPE_CHECKING only. Native SQLite root paging is invoked only inside a real chained delete, with no message text/image bytes in its returned row projection. No new log sink or config/secret body was inspected or emitted. Ready1033/preimport557 caps and old receipts carry as historical; final root real loading qualification remains required.

## Smallest proposed combined selection

`task-12-preflight-selection.json` gives exact selectors and reasons. It selects all incoming behavior cases and two existing changed-delete authority/rollback controls, while avoiding duplicate full feature/fork/Console/provider/schema suites:

- Eleven new delete/voice integration nodes (25 parameter cases), three new voice-pair persistence nodes (four cases), two existing delete rollback/conversation-fence controls.
- Both new DB contract files (11 root-page + five descendant/bind cases).
- New timeout policy file (21 cases), new real-wire cancellation node (two cases), existing positive elicit timeout and confirmation-only-schema control.
- Three exact activation-owner nodes (six cases) plus installed coalesced-transcript/focus/reuse node.
- Relocated `Tests/UI/test_approval_batch_geometry.py` guard once; its previous source node is removed upstream.
- Exact changed CI Fast Lane target-set node and the two store-row ratchets. Run `python scripts/check_index_plan_pins.py` and `python scripts/check_ui_pr_gate_census.py` for the actual changed derived rows. Census floor135→136 and geometry target must be retained; queryplan rows `idx_messages_variants_by_parent` and `idx_msgs_conv_ts` stay plan-pinned.

The behavioral proposal contains80 parameter cases by source inspection, before CI/ratchet/script checks; this is not a collection/pass receipt. It deliberately carries the service bridge default/garbage tests covered by the new shared synchronous policy file and old elicit approve/deny nodes covered by its new async decision matrix. The real cancellation test itself covers duplicate/malformed/opposite-direction/disconnect paths, so no broad MCPclient suite is proposed. Carry all unaffected source/AST receipts and historical QA. Root performs the one final actual startup/import/public-navigation qualification after Task9 clean source handoff and integration.

## Artifacts and limits

Own safe artifacts are `task-12-preflight-report.md`, `task-12-preflight-overlap-map.json`, `task-12-preflight-selection.json`, `task-12-preflight-upstream-summary.json`, and `task-12-preflight-production-delta.patch`. They contain hashes, spans, exact production hunks and selector proposals; no profiles/config bodies/databases/cache/bundles. Private exact upstream/base/frozen sources and full formatted proposal stay under `/private/tmp/task12-preflight` for root inspection. Only the named safe artifacts are copied into the ignored own SDD.

Initial read-only GitHub collection inside an unapproved subprocess returned network error; the exact `gh api` and collection were retried with read-only network escalation and succeeded. No rejected approval. Upstream API misses for feature-only ADR219/220 and nonexistent guessed `test_loading_budget.py` were recorded, not invented; relevant feature ADRs were read from frozen Git. No correctness or current-tree test success is claimed. Root may select Task12 source/test/derived scope now, including the explicit one-node bare-fixture adapter; any rebase must retain actual current Task9 source rather than replacing it with this immutable preflight's old bytes.
