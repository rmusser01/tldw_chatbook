# TASK-40608: provider-a2 postcommit comparison

Source-only review of the safe fixed-marker files `task62-provider-diagnostic-a1-markers.txt` (271 lines) and `task62-provider-diagnostic-a2-markers.txt` (263 lines). Root supplied a2's completed-run evidence: actual backend and workspace codec pass, expected history missing-receipt red; normal-after 19:47:09Z, SID1002, 2854 inputs unchanged, zero workers. The history receipt product was installed only after this measured run.

The markers do **not** support a cohort-induced fallback or repeated sensitive-input construction as the explanation. They show a broader increase in original work before model entry, partly offset by the faster container group. This comparison localizes work; two instrumented runs do not establish a performance regression or its cause.

## Non-overlapping accepted path

These rows partition the interval from the accepted (second) plan reservation to original `make_call_model.<locals>.call_model` entry. They sum to 5.078 s in a1 and 6.735 s in a2. Nested callbacks are deliberately excluded from the sum.

| Interval | a1 seconds | a2 seconds | Change |
| --- | ---: | ---: | ---: |
| Plan reservation → postcommit entry | .422 | .641 | +.219 |
| Postcommit entry → hook admission entry | .468 | .625 | +.157 |
| Hook admission | .391 | .563 | +.172 |
| Hook return → original reply entry | .656 | 1.187 | +.531 |
| Reply entry → original bridge worker entry | 1.172 | 1.594 | +.422 |
| Bridge entry → native preparation entry | .031 | .031 | .000 |
| Native preparation entry → yield | 1.000 | 1.313 | +.313 |
| Preparation yield → AgentService entry | .094 | .078 | -.016 |
| AgentService entry → agent loop entry | .781 | .672 | -.109 |
| Agent loop entry → call_model entry | .063 | .031 | -.032 |

Accepted plan starts at 22.860/30.265; call_model starts at 27.938/37.000. Its complete duration is .672/.640 s. The actual HTTP receipt timestamp belongs to root's test evidence; this table does not relabel call_model entry as the HTTP POST.

## Grouping, currentness and memo evidence

- Both runs contain exactly one `_sensitive_db_paths` entry and one `_direct_child_rule_container_dirs` entry. Every emitted DB `current` result is true and both choose `collect.selected`. DB resolution is .172 s in each run.
- Container resolution falls from .218 to .094 s. Its two RAG getter bodies take .063/.093 s in a1 and .000/.016 s in a2. The diagnostic has no explicit container-route marker, so these timings support but do not independently prove selection of that group.
- `RunLogWriter.bind` falls from .625 to .516 s; AgentService entry-to-loop falls by the same .109 s. The DB and container times are **inside** bind, not additional time.
- `_raw_inputs` (`Utils/sensitive_paths.py:764`) still memoizes only after successful collection, unchanged failure count and a second equal key. The new container group neither clears that memo nor writes its key. The DB/container functions only read the config cache/generation; the original `config_participants.operation` snapshots state and restores it on failure. It does not bump `_CONFIG_GENERATION` on normal admission/exit. The generation increment remains the original runtime-config reload at `config.py:5008`.
- A single observed build cannot prove its result was memoized: key changes and optional accessor errors are not logged by this diagnostic. There is no observed second build or direct source mutation from either cohort patch that explains this run's delay.
- The hook patch only adds the already-installed `hook_permissions` route to the existing related-path acquisition branch (`raw_participants.py:1451` in the current source after history-receipt adoption). It does not change installed-source selection, hook snapshot sharing, locks or fresh dispatch checks. Final admission intentionally remains a fresh read (`Agents/hook_permissions.py:93` and `:714`). The .563 s hook wall interval contains .5625 s of that worker's CPU, so this observation is not primarily waiting for a held config lock.
- The diagnostic installs local code monitoring and preserves original function identities; it does not itself introduce the wrapper identity changes that would select the custom-source paths.

## Largest unresolved interval, mapped to original source

The +.531 s between hook return and reply entry has four observable pieces:

| Piece | a1 seconds | a2 seconds |
| --- | ---: | ---: |
| Owner active before first suspension | .094 | .171 |
| First suspended interval | .140 | .250 |
| Owner active through prepare_chat_request and maintenance entry | .219 | .438 |
| Maintenance call | .156 | .250 |
| Owner active after maintenance, before reply | .047 | .078 |

The ordinary manual path is `resume_durable_postcommit` → `_stream_assistant_response` → `_stream_assistant_response_inner`:

- `console_chat_controller.py:24664` awaits `PromptMixinService.accept_attempt`. Its original `_run_owned_save` (`console_prompt_mixin_service.py:613`) owns the exact Chat DB save before publishing the attempt. This is the first suspension on this source path, consistent with the marker sequence. There is no direct accept_attempt marker, so attributing the entire interval to its native body would overstate the evidence.
- The next synchronous region includes original payload fingerprint construction and `_apply_conversation_memory_preflight` (`controller:24778`, implementation `console_context_compaction.py:3353`), then original trace/setup bookkeeping. It is .219/.438 s, with approximately matching owner CPU. No separate callback markers distinguish these contributors.
- The marked first `_run_maintenance_agent_call` is the original manual `ledger.create_chain` (`controller:24952`; `DB/automatic_work.py:507`), not an early AgentService/run_reply call. It has its own AgentRuns DB transaction and original settings read. The later bridge call is separate and occurs once.

Two other bounded contributors grow:

- Provider composition .547 → .812 s, including its existing MCP composition and project-index preparation awaits (`console_provider_composition.py:352`, `:389`). It occurs once. The current markers cannot attribute this increase to one inner reader.
- The native-preparation prefix **before** first-request planning grows .438 → .750 s. The planner itself remains .344 s and its following preparation tail remains .218/.219 s. That prefix includes progress-inbox preparation (`console_agent_preparation.py:179`), original skill/plugin selection, light-surface selection and the AgentRuns roster reader (`:375`; bridge helper `console_agent_bridge.py:4354` already owns one connection). This is CPU work in the original worker, not the later run-log bind.

## Decision

No new product change is justified merely to reverse either cohort: their observed sensitive-path contribution improved, and the hook patch preserves its fresh-read contract. No broad new profiler is needed to establish that conclusion. If a further timing observation is required to choose a fix, the missing boundaries are specifically the first mix-in save, memory-preflight body, and the native preparation prefix's progress/roster readers. Existing markers already isolate those finite regions; re-profiling the whole provider would repeat known evidence.

No live source changes, product imports, native launches or tests were performed for this comparison. Root retains ownership of runtime verification and any next optimization.
