# TASK-40608: remaining preparation source audit

Read-only audit of the source present after remote checkpoint `8c34df84ee`. No product changes, product imports, native launches, or tests. Inputs were the original preparation sources, `task62-provider-a2-postcommit-comparison.md`, and the safe `task62-preplan-profile-a3-markers.txt`.

## Result

No additional preparation patch is justified from these observations. Several apparent duplicates already avoid native work or enforce different authority/request boundaries. The clearest redundant structure is nested whole-domain definition validation inside a single source capture; its measured upper bound is less than half a second in the observed configuration worker, and safely changing its boundary needs more than a cache.

## Source-supported findings

| Candidate | Actual behavior | Consequence |
| --- | --- | --- |
| Reuse first-request plan | `console_context_compaction.py:3456` calls the preview against the assessed memory-projected request. `console_tool_surface_preview.py:1` explicitly defines its catalog as frozen schema estimates from prior composition. `console_agent_preparation.py:532` builds the fresh dispatch plan after provider/authority composition and request riders. | These are different inputs and guarantees. Do not reuse a preview as an executable dispatch plan. |
| Remove repeated `prepare_chat_request` | At compaction lines3469/3509/3528 the calls prepare the retained request, then memory-bearing semantic request when applicable, then original-system-only accounting when a substituted prompt exists. | Distinct payloads/accounting, not identical redundant calls. Original continuation/thinking checks must remain. |
| Group budget/router config reads | Budget resolution in `console_agent_bridge.py:569` performs eight setting reads and is reread at dispatch intersection (`:4833`). Routing `Agents/agent_routing.py:90` reads four keys. However `config.py:4332`/`:4390` serves warm config without storage admission, retaining selector/generation/external-edit checks. | Do not infer eight/four native admissions from accessor count. The later budget intersection preserves live narrowing. Adding an unconditional config operation could add native work. |
| Cache global memory policy | `console_context_policy.py:248` already memoizes parsed overrides only for the exact installed config/reader/parser, with pause and warm-cache checks; misses use one checked operation. | Already implemented. Another cache duplicates existing behavior and risks stale policy. |
| Remove second progress preparation | `console_agent_preparation.py:179` binds/checks initial owner; `:1958` later decides whether a temporary fleet inbox is needed. `console_chat_store.py:5446` reads the DB only when the exact inbox is absent or its saved conversation differs, then rechecks exact session/store/owner before publication. | The later call is mostly an identity/hint refresh for existing durable inboxes; it also has a different creation condition. Do not skip it. |
| Batch roster rows | `_console_first_request_runtime_context` (`console_agent_bridge.py:4354`) already owns one connection around `list_agent_definitions`. The latter is one SELECT plus pure decoding (`DB/AgentRuns_DB.py:2120`). | No N+1 roster DB access here. |
| Combine roster and pending notes | Pending notes (`console_agent_preparation.py:426`, `AgentRuns_DB.py:1597`) are another one-query read, with optional-read failure handling unlike mandatory roster failure. Plugin preset admission, runtime settings, and message construction separate the calls. | A scope spanning both would retain DB custody across unrelated/plugin work. Moving the read changes ordering and exception boundaries; no measured evidence currently warrants that. |
| Batch memory lineage | `_durable_context_snapshots` (`console_chat_controller.py:23039`) already prefers batched message versions; applicable branch metadata (`console_context_repository.py:1026`) reads selection, memory, and scope in one transaction. | No per-message version N+1 on the stock persistence path. Grouping these two remaining operations may be possible only in a new finite synchronous capture; do not hold an owner through compaction/provider awaits. Current markers do not isolate a useful gain. |

## Actual multiplicative source validation

`capture_console_turn_configuration_owned.current()` (`console_configuration_preparation.py:1909`) calls `_configuration_receiver_current` (`:140`), which performs `domain.require_current()`. It then obtains `_source_references` (`:254`) through `ConsoleDomainServices.configuration_sources` (`console_domain_services.py:442`). That method performs another full check, enters `_ready_domain_published_plugin` (`:625`, another full check), calls `published_plugin_service` (`:412`, another full check), and calls that published-service reader again for the callback tuple. An unused skill-trust source can introduce further full validation. These are nested currentness checks within one capture, not native filesystem admission calls.

Each `require_current` (`:248`) validates domain methods/helpers, runtime callbacks, concrete presentation definitions, plugin construction inputs, and all source definitions from eleven modules (`:148`/`:161`). The observed accepted gate2 configuration worker recorded:

- `require_current`: 223 calls, total0.4178651s.
- `configuration_sources`: 26 calls, total0.3652036s, nested within the surrounding capture/checks.
- `_definition_current`: 32,413 calls (446 recursive), total0.2365976s, nested within the domain checks.

These times overlap. They must not be added. Even deleting every observed domain check cannot save more than approximately0.418s in this instrumented worker; ordinary unprofiled cost may be lower. Gate1 diagnostic coverage was incomplete, and these rows are not an end-to-end speedup claim.

The smallest plausible future change is a single qualified source-capture body that reads its concrete fields once and validates the whole domain at its entry and publication boundaries. It would need to preserve the existing plugin construction comparison, trust/source/receiver identities, config generation, native retirement, and failure behavior. A cross-await/global cache or generic `skip_validation` flag is not equivalent. An arbitrary injected callback or another thread can change definitions between checks; a new finite boundary must be justified explicitly, not assumed from synchronous syntax.

If pursued later, causal controls should establish: one checked source snapshot with identical fields and original reader calls; immediate refusal for method/code/default/keyword/closure/wrapper changes before entry; refusal for callback-induced replacement before adoption; exact receiver/source and plugin construction drift; unchanged custom-source fallback; and no successful result when original retirement fails. Count original checks and native acquisitions separately, then compare complete accepted-turn timing. This is a bounded opportunity, not a selected implementation or a sufficient sub-five fix.

## Decision

Keep the current product and prioritize the coherent mounted Console path. The accepted backend remains above five seconds. This audit does not establish that the native preparation prefix, memory preflight, or domain-definition scan is the remaining dominant cause, and it supplies no justification for weakening fresh checks.
