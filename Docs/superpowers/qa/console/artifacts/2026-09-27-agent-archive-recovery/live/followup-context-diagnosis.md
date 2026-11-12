# Task33020 AC2 — read-only next-step diagnosis

Inspected source HEAD `4af6b2eb670035bfaae455b037b8e294545532d9` in `/Users/macbook-dev/.codex/worktrees/agent-archive-recovery/tldw_chatbook`. `git status --short` was empty at the initial check. Read the assigned brief, Task33020, ADR-166, approved spec sections 3/5/8, durable live qualification, provider-response diagnosis, budget-probe report, relevant live-verification lessons, and production paths described below. This report is the only written file. No source/config/profile/cache edits, tests, inference, app launch, server interaction, subagents, or reopened reviews were performed.

## Decision

**Do not spend another full live qualification attempt on unchanged source yet.** There is a demonstrated fresh-turn context omission: the live source session retains authenticated recovery cards and their references, but a new ordinary Capture-Off follow-up does not receive their immutable rows, order, or `result_ref` values. The omitted second row in retained assistant prose makes this concrete, not a hypothetical prompt problem. A restricted disposable configuration can enforce the no-RAG-dispatch condition, but configuration cannot repair that missing selection context.

The smallest justified next step is a narrowly scoped production regression at the fresh native user-turn history assembly seam, followed by a recovery-only context correction if the regression confirms these inspected bytes. Root owns that work and any later live run. Completed Open/manual qualification and independent review waves remain closed.

ADR required: no new ADR. ADR path: `backlog/decisions/166-agent-assisted-archive-recovery.md`. Reason: this is direct implementation/qualification of spec section 5's existing immutable-card selection contract, preserving its authority and provenance boundaries.

## What is established, and what is not

- The 4096-token probe's ten calls all returned raw localhost HTTP200, usable named native calls, and `finish_reason=tool_calls`. It never loaded restore or reached a restore proposal. It establishes a discovery/working-set loop for that run, not a restore-schema or token-budget defect.
- Generic `search_library_rag` caused the documented 91,567,205-byte MiniLM download and empty chromadb initialization. An initially absent index and local-only LLM transport did not enforce the intended boundary.
- The historical 512-token final carried502 remains unattributed. Loading restore has successful controls; no deterministic schema failure is established. The new result-context finding does not explain the raw/normalized provider failure.
- No claim is made about captured successor recovery, the completed same-screen Open fix, or resource-review results. Capture-Off is the supported qualification surface for this continuation.

## Exact fresh-turn selection-context omission

Within one agent loop, search JSON is returned normally to the model by `Chat/archive_recovery/runtime.py:496–505`; `Agents/agent_runtime.py` appends native assistant/tool rows to that loop's local `messages`. The app also retains the validated projection and displays the native numbered card.

Across a new ordinary user follow-up:

1. `Chat/console_chat_controller.py:9438` builds a fresh request through `_provider_messages_for_session`.
2. `_provider_messages_for_session` at line27616 collects source session messages and delegates to `_provider_message_payloads` at line27990.
3. `_lightweight_provider_message_rows` at line27807 explicitly admits only `ConsoleMessageRole.USER` and `ASSISTANT` (the role filter begins around line27886). Tool-result markers, including their full JSON, are dropped. Failed assistant rows are dropped as well.
4. `_context_content_for` at line27371 only substitutes trusted character-greeting template content. It has no recovery result/context projection.
5. `_run_agent_reply` at line26147 makes `agent_messages = list(provider_messages)`, separates the leading system row, and passes that history to `ConsoleAgentBridge.run_reply` at line26495. `build_console_first_request_plan` at `Chat/console_agent_bridge.py:4215` composes registry/schemas and ordinary riders; there is no recovery-result rider.
6. A targeted reference search across controller, bridge, turn-context/prepared-request and recovery modules found recovery admission, policy, schema, closure and UI state, but no compensating fresh-turn export of the retained card's rows/ref/order. No broad TOOL replay is present.

`ConsoleAgentBridge._append_marker` at line9618 writes `role=TOOL` with `tool_output_full`; that preserves the display record, not a fresh request row. The fresh primary run's local model message list is rebuilt rather than resumed from the prior loop.

The runtime reference remains valid independently: `ArchiveResultStore.resolve` at `Chat/archive_recovery/result_store.py:203` checks the same source session, TTL and agent-visible provenance; `ArchiveRecoveryRuntime._close_authority` at line302 closes the old run's capability but does not clear the session result pages. A later authorized run can use an existing reference if it has that reference. Closing the session/restarting expires it.

### Retained evidence confirms the consequence

I opened the two existing scratch databases with SQLite URI `mode=ro`, without importing application code, and compared stored successful archive tool results with their persisted final assistant messages. Only scalar counts/presence were emitted; no model reasoning, tool arguments, keys, or message bodies were exported.

| Existing run | Search result refs | Final assistant echoes of refs | Assistant mentions Amber | Assistant mentions Orchid |
| --- | ---: | ---: | --- | --- |
| `9089bbd7-2ab1-488f-aec0-9d915c7c1cd1` | 2 | 0 | yes | no |
| `084bf530-621c-425c-8071-f2edf4a05ea4` | 2 | 0 | yes | no |
| `3d4da9bc-a589-4e45-a3ae-21017f086bff` | 1 | 0 | yes | no |

The first two final assistant messages are complete, 156 and 138 characters respectively. The native two-row card still showed both targets, as the durable live report records, while the only assistant prose that ordinary fresh-turn history admits omitted Orchid and every opaque result ref. The later failed restore-follow-up assistant record is empty/failed and contributes no alternate context.

Therefore “Restore the second one” on the earlier two-row card has no reliable second-row identity in the next request. The model may retain Amber's title from assistant prose and fresh-search it, but that returns a **new live result set**; its ordering does not certify selection of the earlier immutable second row. Model-authored summaries cannot be treated as a source of app-owned card authority. Prompting the model to reproduce opaque refs would be fragile and would not fix the source omission.

Confidence: high for the inspected normal Capture-Off fresh-turn path. This does not establish that every model must fail: a model could voluntarily echo all refs/rows into ordinary prose, or recover via a fresh search. Neither behavior supplies the reliable earlier-card context required by spec section5.

## Narrow existing integration seam

Use app-owned recovery state rather than replaying all tool transcript rows:

- `ArchiveRecoveryRuntime.search` at line364 publishes an already bounded projection from `ArchiveResultStore` and retains `ArchiveRecoveryPresentation(session_id, request, projection, model_visible, created_at)` in a maximum10 deque.
- `ArchiveRecoveryRuntime.presentations(session_id)` at line380 returns detached typed projections with the original30-minute TTL.
- `ArchiveResultStore._RetainedPage` also carries `model_visible`; `resolve(..., for_agent=True)` explicitly rejects manual-only results.

The narrow correction belongs where native next-send provider history/preview is constructed, before compaction/token preparation and first-request planning: `_provider_messages_for_session` / the shared request builder it feeds. Root should keep preview and live construction consistent and use an authenticated per-turn recovery projection seam, not parse `TOOL` text, display card JSON, persisted transcript, or imported content. If admissible recovery projections are attached later than that builder, estimates/preview and the actual sent payload must use the same projection.

Necessary boundaries of the regression/correction:

- Same live source session only; retain original set identity, row order and opaque refs, not regenerated ordinals.
- Only model-visible search records originally produced by the authenticated native recovery search; manual Refresh/Preview remains excluded.
- Current frozen Library Allowed/destination disclosure and native user-origin/primary/local/non-temporary eligibility must gate **sending** this retained data. Earlier Allowed state is not permission for a later Blocked turn.
- Expired/closed/restarted/cross-session refs and records cannot become live again. Retained UI presentations and result-store eviction are separate: do not disclose a presentation as selectable if its corresponding result reference was evicted earlier by later pages.
- Preserve ordinary multiple-card ambiguity; do not silently pick the newest card, renumber it, or auto-search to reinterpret an old ordinal.
- The context is untrusted evidence. It never creates a callback, confirmation, write permission, new result ref, or imported live control. Existing lookup, per-operation native confirmation and coordinator gates remain the only effect path.

The most direct regression uses two real retained native search rows with distinct titles and known ordinal order, a completed assistant summary that omits the second row/refs, then constructs a fresh ordinary Capture-Off “Restore the second one” request through the production builder. On current bytes the actual provider history has neither second-row identity nor refs. The positive pin should retain the exact original immutable projection; negative pins should cover manual-only, later Blocked, expired/evicted and another session. This is a bounded context integration check, not a new generic replay facility and not a repeat of Open review.

## Supported scratch no-RAG controls

### Minimal enforced guard: deny-only PreToolUse hook

Existing supported TOML format (merge into the scratch table; do not append a duplicate `[hooks]` table):

```toml
[hooks]
enabled = true

[[hooks.hook]]
event = "PreToolUse"
matcher = "search_library_rag"
command = ["/usr/bin/false"]
timeout_s = 2
```

This uses no new script/dependency: the command's nonzero crash/failure exit is a documented fail-closed denial. For a clearer intentional-denial message, the already-used scratch Python interpreter can run an argv-list command that prints JSON `{"decision":"deny","reason":"Generic RAG disabled for no-index archive qualification."}` and exits0; `run_hooks._interpret` recognizes explicit deny irrespective of a clean exit. Either choice is supported. `/usr/bin/false` existence was not separately verified in this diagnosis; root can check it before selecting the command. Do not execute a guard helper that imports application/RAG code.

Accepted configuration and exact call path:

1. `config.py:1931` projects raw `[hooks]` into app config; template documentation is at line3665.
2. `Agents/run_hooks.py:99` `_parse_hook` requires a valid event, nonempty argv **list of strings**, optional nonempty tool-name glob for Pre/PostToolUse, and finite positive timeout. `load_hooks_config` at line150 reads `hooks.hook` as a list of tables. Invalid entries are dropped, so parsed presence must be checked; a misspelled matcher is not an effective restriction.
3. `Chat/console_runtime.py:3566` `ensure_run_hooks` owns one engine, with a current app-config provider.
4. `ConsoleAgentBridge.run_reply` at line6436 resolves it once per turn and sets a separate `guard_tool_calls` using `wrap_review` (line6442), passed into `AgentService` at line6602.
5. `AgentService._run_one` threads the guard into LoopDeps. `Agents/agent_runtime.py:2267` applies it to **every proposed call**, before preauthorization and ordinary review. Review excludes guard-refused calls; `verdicts.update(guard_refusals)` at line2376 preserves the refusal.
6. `RunHooksEngine.wrap_review` at `Agents/run_hooks.py:820` fires PreToolUse with the tool name and prefixes denial text `hook: `, which cannot equal the `proceed` sentinel. `_interpret` at line264 denies explicit JSON deny, exit2 shorthand without an overriding decision, non0/2 failure, and timeout/engine failure via the firing layer.
7. Only `proceed` calls dispatch. For generic RAG the effect route is registry invocation → `LibraryRagToolProvider.invoke/_run_search` (`Agents/library_rag_tool_provider.py:369`) → `run_library_rag_search` → `LibraryLocalRagSearchService.search` at line263 → `_search_rag` at line326 → `_resolve_rag_runtime` at line1051 → `get_shared_rag_service` at line1092. The guard refuses **before that route begins**.

**Answer: yes, a correctly loaded exact PreToolUse deny hook is sufficient to block the demonstrated model-triggered generic RAG lazy initialization before it can download/create an index.** It preserves authenticated recovery, normal discovery/loading, and every normal authority/permission gate; it never permits anything. It still advertises generic RAG, so it may waste a turn returning a refusal or encourage another discovery loop. It does not globally prohibit RAG from another UI/startup/background/ingestion surface. Keep the scratch interaction limited to archive qualification, with no staged RAG evidence, ingestion, RAG readiness/init action or index-building path; inspect starting/ending scratch cache/index state. An offline HF setting alone is insufficient: local cached assets could still initialize an index.

Selecting a plain/BM25 RAG profile is **not** the no-init fix: `LibraryLocalRagSearchService._search_rag:379` calls `_resolve_rag_runtime()` **before** inspecting `profile_mode`, so even its later plain/keyword branch passes through construction. Disabling Assistant Library is also unsuitable because it excludes the recovery capability.

### Better discovery posture: supported persona pure-deny rule

To keep the wrong tool out of the next model's discovery and working set, an active local scratch Persona can carry this validated profile field:

```json
{
  "policy_rules": [
    {"rule_kind":"mcp_tool","rule_name":"search_library_rag","allowed":false}
  ]
}
```

`LocalPersonaProfileCreate/Update` accepts `policy_rules` (`tldw_api/character_persona_schemas.py:550/576`); `PersonaPolicyRule` at line615 permits `rule_kind` mcp_tool/skill, nonempty `rule_name`, `allowed`, `require_confirmation`, and optional positive call cap, rejecting extra fields. Existing local service `create_persona_profile`/`update_persona_profile` at `Character_Chat/local_character_persona_service.py:1086/1113` validates and persists the scratch profile. Bind it to the source chat as a real persona identity through the native Buddy Persona assignment (`Chat/console_persona_assignment.py:139`) or the existing native named-workspace assistant default. Default/global never inherits a persona automatically (`console_assistant_defaults.py:46`); merely creating a Persona or writing an invented TOML `policy_rules` setting does nothing.

`resolve_turn_persona_policy_rules` (`Chat/console_turn_context.py:76`) loads rules from the source session's durable active persona identity. `_compose_run_registry_and_allowed` (`Chat/console_agent_bridge.py:4123`) applies the narrowing filter after authenticating/registering the real recovery and Library providers. `persona_policy.evaluate_tool_policy` at line100 treats a **pure deny** rule set as a carve-out: unmatched tools remain advertised. Thus both `search_archived_conversations` and `request_conversation_restore` remain eligible without adding allow rules. `AgentService.find_tools:4999` filters by `config.allowed_tools`; loader at line5047 refuses excluded schemas; registry dispatch wrapper at line3340 refuses names outside allowed/disclosed sets. Recovery closure at line7632 separately requires allowed/disclosed names and registry capability authentication.

This Persona rule is better for avoiding the demonstrated wrong discovery route, but costs an additional native identity/profile setup and changes the source Persona system prompt. Do not add a broad `mcp_tool` allowlist just to achieve one deny; it would also alter unrelated discovery. Prefer **pure-deny Persona plus the exact deny hook as independent backstop** for the eventual no-index live qualification, with its changed Persona posture explicitly recorded. If root wants the smallest guard-only diagnostic preserving generic assistant identity, use the hook alone and label any discovery refusal/loop honestly. Neither option fixes the fresh-turn context omission.

## Exact next experiment and attempt cap

**Immediate experiment: zero inference.** Root should first construct the bounded production fresh-turn regression described above on this HEAD. If it fails as predicted, correct only the recovery context seam under ADR-166, verify its positive/negative boundaries and review only the new change. This read-only task does not execute or authorize a broad test selection.

After that context boundary is established, the smallest credible live run is:

1. Fresh scratch HOME/XDG/config/data/cache, verified source/interpreter provenance and null Keyring; no copied old DB/cache/index. Merge the deny hook and (if chosen) bind the pure-deny Persona. Verify parsed guard presence, source Persona rules, Library Allowed, Capture-Off, native tools, and empty index/cache before first send. Do not change the single local Gemma server or normal profile. Keep `max_tokens=4096`, streaming and reasoning controls as already observed; no causal token-budget claim.
2. Seed exactly two archive candidates with distinct titles and explicit nondeleted original IDs/message IDs/active leaves, ordered deterministically by known archive episode times. Use a workspace-only target as row2, or two equivalent freshly isolated fixtures per retrieval-mode run. Preserve a separate unsent Chat2 draft before sending.
3. Native ordinary search prompt, short enough not to trigger paste collapse: `Find my archived conversations.` Wait for one real validated two-row card and a settled initial turn. Verify its immutable set/ref/order. Before follow-up, verify that the actual prepared model request contains that same authenticated projection under current Allowed/destination; no user-pasted tool JSON/refs.
4. Native submit exactly `Restore the second one`. It must model-discover/load and invoke `request_conversation_restore` against row2's original ref, reach the exact native confirmation/coordinator, and allow the user to confirm the disclosed scope. A fresh search producing a different card does not prove selection of the original second row.
5. After the natural follow-up settles, set a source preservation draft. Verify confirmed storage scope and unrelated target state; explicit receipt Open must show the original persisted ID/messages/active leaf. Check both drafts and unchanged request count across Confirm/Open/draft switches. No implicit send is acceptable.
6. Only if the first retrieval-mode journey passes, repeat once in the other mode with fresh fixtures/profile. Both Direct and RAG-only keep recovery eligible independent of index through `recovery_eligibility(...):38` (it discards `direct_library_tools`). Any existing Direct provider constructor failure remains separately labeled; it does not justify another feature repair wave.

Bound the eventual run to **one journey per mode**, sequentially, with **5 logical model turns per submitted user turn**, zero model and transport retries, and an ephemeral whole-probe HTTP POST backstop of **10 per mode /20 total** that also counts budget wrap-up requests. Five leaves the documented four-call search path one spare and bounds the equally narrow discovery/load/restore/final-answer follow-up. No concurrent inference. Settings supported by `console_run_budget():544`: `agent_max_model_turns=5`, `agent_max_steps=40`, `agent_max_wall_seconds=180`, `agent_max_model_retries=0`; accepted model-turn minimum is1. Transport controls remain the observed scratch `api_retries=0`/compatibility `retries=0`. A run wall budget is not a hard timeout of an in-flight zero-chunk provider request, so keep the transport observation/backstop and native stop handling.

Stop on first provider failure, mismatched/ambiguous original selection, generic-RAG execution/init attempt that escapes denial, or repeated unproductive discovery; do not send an equivalent retry. If native stop races fast tool continuation, count the actual wire calls, with the backstop enforcing the fixed ceiling. Do not add prompt guidance with loader syntax or result refs to force success. Preserve scalar wire metadata/finish category and native UI/storage evidence, then Quit natively. No runtime loop bound should be relabeled as raw provider failure.

**Alternative if root does not make the context correction:** leave AC2 unchecked and Task33020 In Progress with this concrete missing-context diagnosis. The existing manual recovery/Open and dedicated Direct/RAG-only search evidence remain valid. A named follow-up or prompt-engineered ref echo would qualify a different path and cannot substitute for the requested earlier-card ordinal selection. No further full live retry is justified merely by switching hooks/persona/budget on unchanged source.
