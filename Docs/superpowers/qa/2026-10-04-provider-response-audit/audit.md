# PR #3017 cross-provider response audit

Date: 2026-10-04. Initial reviewed head: `16e5f0270e1ee02ed807abde940b25843a5b4c1d`. Rechecked after incorporating dev `49206beea9` (PR #3014), then `54ac2758af` (PR #3019).

## Result

The initial audit reproduced the DeepSeek parser failure in five other providers. Together was subsequently fixed independently in [PR #3014](https://github.com/rmusser01/tldw_chatbook/pull/3014). That historical recheck had **eight expected-success failures** across Groq, OpenRouter, Fireworks and Cerebras, and **eight passes** including Together and the three controls. The later TASK-34367 reconciliation below records the additional repairs; historical resync evidence remains scoped to its recorded revision.

Initial offline fixtures passed through `Chat_Functions.chat_api_call`, real provider adapters/generic handlers, HTTP boundary and strict shared parser: **10 expected-success cases failed** (five providers × complete/SSE); **six controls passed** (Moonshot, Mistral and ZAI × complete/SSE). HTTP alone was mocked. These are contract reproductions, not live calls or comprehensive provider qualifications.

| Provider | Exact probe additions | Result in both modes | Follow-up |
| --- | --- | --- | --- |
| Groq | envelope `x_groq: {id: "fixture"}`, choice `logprobs: null` | malformed response/event | TASK-34367.1 |
| OpenRouter | choice `native_finish_reason: "stop"` | malformed choice | TASK-34367.2 |
| Together | choice `logprobs: null` | now passes complete/SSE after PR #3014 | TASK-34367.3: optional annotations remain |
| Fireworks | choice `logprobs: null` | malformed choice | TASK-34367.4 |
| Cerebras | envelope `time_info: {total_time: 0.1}`, choice `logprobs: null` | malformed response/event | TASK-34367.5 |

All fixtures supply a real-shaped assistant reply, index 0, terminal stop and token usage. Complete failures are exposed as `ChatProviderError`; streamed failures originate as `HostedChatProtocolError` during iteration. Existing Console retry settlement is shared across providers, so PR #3017 repairs the retry deadlock for their known no-response failures too. It does not repair the four remaining failing response contracts. Together’s provider-scoped allowance and streamed-usage fix arrived through the newer dev base.

## Official evidence and boundaries

- [Groq API reference](https://console.groq.com/docs/api-reference) includes `x_groq` and null `logprobs` in its ordinary response example. [Official stream SDK schema](https://raw.githubusercontent.com/groq/groq-python/main/src/groq/types/chat/chat_completion_chunk.py) defines nested `x_groq.usage` and `x_groq.error`. Ignoring the entire field can lose accounting or mask terminal errors; nonstream `x_groq.usage` has different hardware-cache meaning. Provider-specific normalization requires dedicated regressions.
- [OpenRouter overview](https://openrouter.ai/docs/api_reference/overview) explicitly specifies `native_finish_reason` for both body and stream choices. Its [current endpoint reference](https://openrouter.ai/docs/api/api-reference/chat/create-a-chat-completion) also documents optional `openrouter_metadata` and `service_tier`. Audit these alongside provider/logprobs/reasoning annotations rather than accepting arbitrary fields.
- [Together reference](https://docs.together.ai/reference/chat-completions) specifies additional choice annotations including logprobs and seed, message reasoning and envelope prompt/warnings.
- [Fireworks reference](https://docs.fireworks.ai/api-reference/post-chatcompletions) specifies logprobs; streaming metrics documentation also describes terminal perf_metrics.
- [Cerebras reference](https://inference-docs.cerebras.ai/api-reference/chat-completions) includes time_info and logprobs in its ordinary example, with further service-tier/reasoning fields in the schema.
- [Moonshot reference](https://platform.kimi.ai/docs/api/chat), [Mistral reference](https://docs.mistral.ai/api/endpoint/chat), and [ZAI reference](https://docs.z.ai/api-reference/llm/chat-completion) baseline text/usage fixtures pass. Moonshot documents optional logprobs, but its sample does not show them emitted by default: optional response-shape reconciliation remains unqualified.

Generic presets already have level-specific allowance support under ADR-179. At the initial audited head, Together/Fireworks/Cerebras intentionally pinned empty provisional sets in `test_inference_cloud_presets.py`; those tests did not prove live compatibility. PR #3014 replaces Together’s provisional empty choice set with captured `logprobs` evidence and enables streamed usage; Fireworks/Cerebras choice allowances remain provisional. PR #3019 independently fixes Fireworks tool-call object extras and null streamed continuation fields; those tool-call fixes do not admit choice-level `logprobs`. Registry inspection found additional presets with empty choice allowances. No provider-wide compatibility claim is made without an official/captured envelope and actual-adapter replay. OpenAI/Anthropic/Google and local adapters use other parser paths; the specific hosted-parser defect is not demonstrated there.

## Reproduction

`adapter_probe.py.txt` preserves the exact temporary pytest module used. In an isolated checkout, copy it to `Tests/LLM_Calls/test_pr3017_provider_audit_probe.py`, run that file with Python 3.12 pytest and the repository bootstrap-profile fixture, then remove the temporary module. It is intentionally archived outside normal test collection because it exposes unresolved follow-up bugs. Command used:

```sh
python -m pytest Tests/LLM_Calls/test_pr3017_provider_audit_probe.py -q --tb=short --show-capture=no --basetemp=/private/tmp/pr3017-provider-probes
```

Recorded outcome: `10 failed, 6 passed in 0.86s`. No credentials or live paid requests were used.

## Merge review

Qodo's changed-target retry finding was reproduced in 14 cases across both wire formats, then repaired at the final durable-header transaction. Provider/model/endpoint/generation/response/reasoning now match the preceding failed call; tool schemas and literal envelope components also match. Route AGENT_FIRST → TOOL_LOOP remains valid. All 40 ownership cases pass. Import grouping finding fixed. Independent reviewer found no remaining Critical/Important issue. Six affected PR modules pass **58 tests**; no full-suite claim.

Broader targeted trace runtime/service/system-prompt run: **160 passed, two failed**. Both failing cases are `test_failed_rollback_releases_observers_and_reports_ambiguous_outcome[False/True]`: the unchanged test constructs a transaction manager and exits without entering, so `_maintenance_context` is absent and masks its intended rollback/commit error. The same two failures reproduced after temporarily restoring trace_service from origin/dev; the DB and test files are byte-identical to origin/dev. The repaired service was restored afterward. These pre-existing test-fixture failures were not silently excluded or repaired in this provider PR.

## TASK-34367.1-.5 reconciliation (2026-10-04)

The five known failures are repaired in the combined Console-fixes worktree.
The preserved audit fixture became the permanent
`Tests/LLM_Calls/test_documented_provider_response_contracts.py`: original RED
was **10 failed, 6 passed**, and the final affected-provider run is **409 passed**
across nine modules. Only `requests.Session.post` is replaced; dispatch,
provider adapters/generic handlers, transport ownership, SSE framing, strict
normalization, public answer and terminal accounting are real. This remains
an offline primary-schema qualification, not a paid/live capture receipt.

Current primary references were re-read on 2026-10-04. Provider-scoped changes:

- **Groq:** nullable/object choice logprobs and ordinary x_groq/service_tier
  metadata; complete x_groq requires its request ID and cache usage remains
  distinct from top-level accounting. SSE x_groq.usage (including timings and
  prompt/completion token details) is promoted and retained. Matching duplicated
  envelope usage is accepted; conflicting/malformed usage and nested error
  strings fail closed with redacted errors. Stream-only obfuscation padding is
  admitted as an annotation. Sources: [API reference](https://console.groq.com/docs/api-reference),
  [complete schema](https://raw.githubusercontent.com/groq/groq-python/main/src/groq/types/chat/chat_completion.py),
  [chunk schema](https://raw.githubusercontent.com/groq/groq-python/main/src/groq/types/chat/chat_completion_chunk.py),
  [usage schema](https://raw.githubusercontent.com/groq/groq-python/main/src/groq/types/completion_usage.py).
- **OpenRouter:** native_finish_reason/logprobs, provider/service_tier and
  openrouter_metadata are scoped annotations. A documented content-free final
  usage choice repeating the already accepted finish is converted to the
  neutral accounting frame. Changed finish/native finish/index/role, new
  text/tool/reasoning deltas, unknown fields, duplicate accounting and error
  frames are rejected. Native finish strings remain upstream-defined.
  Sources: [overview](https://openrouter.ai/docs/api_reference/overview),
  [endpoint](https://openrouter.ai/docs/api/api-reference/chat/create-a-chat-completion),
  [streaming](https://openrouter.ai/docs/api_reference/streaming),
  [errors](https://openrouter.ai/docs/api_reference/errors-and-debugging).
- **Together:** choice logprobs/seed/top_logprobs/text, message reasoning and
  envelope prompt/warnings use existing level-scoped allowances. Prompt is a
  documented array, correcting the older provisional comment's string guess.
  Source: [chat schema](https://docs.together.ai/reference/chat-completions).
- **Fireworks:** choice logprobs/raw_output and envelope perf_metrics/
  prompt_token_ids are scoped. The SSE fixture puts perf_metrics on the terminal
  chunk with usage; metrics do not authorize a missing finish or error success.
  Source: [chat schema and streaming metric contract](https://docs.fireworks.ai/api-reference/post-chatcompletions).
- **Cerebras:** time_info/service_tier/service_tier_used, choice logprobs/
  reasoning_logprobs and message reasoning are scoped.
  Source: [chat schema](https://inference-docs.cerebras.ai/api-reference/chat-completions).

Existing allowance filtering remains annotation-only; answer/usage/tool/finish
fields are not dropped. Existing ignored reasoning stays ignored. OpenRouter's
nonempty structured reasoning_details requires a separately governed durable
block contract and remains fail-closed (null/empty annotation accepted); it is
not silently discarded or claimed supported. Likewise opt-in nonempty
Fireworks choice token_ids remain unsupported by the existing level-value
contract. Neither vendor-hosted execution nor arbitrary annotation arrays are
newly enabled. [OpenRouter reasoning shapes](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens)
were audited to establish that boundary.

A narrow optional call-owned wire normalizer, governed by the pre-code
[ADR-179 amendment](../../../../backlog/decisions/179-generic-hosted-provider-engine-and-preset-registry.md),
handles Groq/OpenRouter semantics between bounded JSON loading and unchanged
strict parsing. Other providers supply no normalizer. Eighty both-mode
unknown/required-field negatives, provider-scope controls, baseline
Moonshot/Mistral/ZAI contracts and existing DeepSeek null-logprobs checks pass.
Provisional empty-set assertions for the three generic presets were removed
in favor of actual-handler behavioral fixtures.

Verification: Python 3.12 on Windows, private Tests bootstrap profile and
private TEMP/TMP; nine affected modules only, **409 passed, one existing
Pydantic deprecation warning**. Ruff check passes with E721 ignored for the
existing deliberate exact-type checks that reject booleans; adapter/test Ruff
format checks and scoped git diff whitespace checks pass. No full suite,
credentials, paid requests, main-checkout edits or child commits were used.
All five tasks remain In Progress pending the combined independent review/PR.

### Merged dev qualification (2026-10-04)

The provider fixes in `b4ed4f91df` were qualified after merging current
`origin/dev` (`49206beea9`) as `2a3d59baa2`. The automatic merge duplicated
Together's `choice_allowances` keyword and failed compilation. Removing only
the narrower duplicate retained the documented allowance superset and the
live-proven `stream_include_usage=True`, with `stream_usage_optional=False`.
The target's [Together capture](../../../../Tests/fixtures/cloud_live/together.json)
(captured 2026-10-04T17:56:04Z), null streamed logprobs, trailing accounting,
and bare-array model listing were preserved. No new live request was made.

The same nine provider modules plus `test_doc_derived_presets.py`,
`test_live_capture_replay.py`, and
`test_openai_compatible_model_discovery.py` passed: **644 passed, one existing
Pydantic warning in 44.03 seconds**. This includes the retained live capture
replay/listing and oversized model-template regression controls. Scoped Ruff
checks, four adapter/test formatting checks, and scoped diff whitespace
checks passed. Evidence log:
`C:/Users/GDesktop-1/.config/tldw-console-pause-34402/provider-merged-qualification.txt`.
The qualified `provider_registry.py` SHA256 is
`5C62B6DD78D3B299180B84E36BBE11A7A1C7711D8864A9B8240976A9C6137D80`;
Groq, OpenRouter, shared parser, and documented-contract test source are
unchanged from the preceding qualification.
## Historical PR #3017 integration and resync evidence

The following records describe the earlier PR #3017 heads. Their then-open
followups and permissions do not replace the later TASK-34367 reconciliation
above or the current combined PR #3023 status.

Initial recorded outcome: `10 failed, 6 passed in 0.86s`. Repeating the exact module after the dev merge produced `8 failed, 8 passed in 0.68s`; both Together cases pass with reply and usage retained, and the four other provider failures are unchanged. No credentials or live paid requests were used. TASK-34367.3 AC #1 is now satisfied by PR #3014; its optional annotation reconciliation and broader regression criteria remain open.

## Merge review

Qodo's changed-target retry finding was reproduced in 14 cases across both wire formats, then repaired at the final durable-header transaction. Provider/model/endpoint/generation/response/reasoning now match the preceding failed call; tool schemas and literal envelope components also match. Route AGENT_FIRST → TOOL_LOOP remains valid. All 40 ownership cases pass. Import grouping finding fixed. Independent reviewer found no remaining Critical/Important issue. Six affected PR modules passed **58 tests** before the newer dev merge. After incorporating PR #3014, those six modules plus captured-provider replay, inference presets and OpenAI-compatible discovery pass **248 tests**. The post-merge derived-artifact preflight also passes. This is targeted verification; no full-suite claim.

Broader targeted trace runtime/service/system-prompt run: **160 passed, two failed**. Both failing cases are `test_failed_rollback_releases_observers_and_reports_ambiguous_outcome[False/True]`: the unchanged test constructs a transaction manager and exits without entering, so `_maintenance_context` is absent and masks its intended rollback/commit error. The same two failures reproduced after temporarily restoring trace_service from origin/dev; the DB and test files are byte-identical to origin/dev. The repaired service was restored afterward. These pre-existing test-fixture failures were not silently excluded or repaired in this provider PR.

## Subsequent dev rebases

- Dev `a78a9a900b` added PR #3015 roleplay markup handling and PR #2996 merge-queue rules. A conflict-free rebase preserved all three PR patches (`git range-diff` equality). Six PR modules plus new roleplay/CI contracts: **259 passed**; derived-artifact preflight passed.
- Dev `8c4dfe59a2` added PR #3006 Console slash-command worker handoff and origin-chat/draft custody. Its chat_screen.py changes do not overlap this PR’s established-store hunk. Another conflict-free rebase preserved the reviewed patches. Six PR modules plus command-draft, generation-action, command-origin and video-send tests: **147 passed, six failed**; derived-artifact preflight passed. All six failures are in `Tests/Chat/test_console_generation_actions.py`, where generation template reads raise `RecoveryRequired("raw_source_selection_changed")` under the per-test profile redirect. Each of these exact nodes also failed with the same error after switching the isolated worktree to exact dev; the PR branch was restored in a `finally` block. No failure was excluded or fixed here:

```text
test_generate_image_handler_threads_prepared_fields_into_batch
test_generate_image_handler_no_prompt_uses_llm_composed_context_end_to_end
test_generate_image_handler_no_prompt_llm_call_raises_falls_back
test_generate_image_handler_no_prompt_llm_timeout_falls_back
test_generate_image_handler_no_prompt_llm_empty_response_falls_back
test_generate_image_handler_no_prompt_kill_switch_off_skips_llm_path
```

This is targeted offline evidence, not a full-suite or new paid-live qualification. Auto-merge was disabled before manual resync per the newly merged AGENTS.md/ADR-218 rules. Re-arming requires Qodo review on the current head and all review threads resolved.


- Dev `3146bbd8da2f30eaa97d72bd0ab860de4c40de75` added PR #3011 Buddy saved settings, Resume ownership and shutdown settlement. Rebase required one append-only conflict resolution in `backlog/docs/lessons-testing-evidence.md`; both complete lessons were retained. Production and test patches are identical after normalizing only diff positions and index hashes. Six PR modules plus Home Resume, queued-shutdown rebuild, roleplay Resume navigation and CI workflow contracts: **109 passed in 247.56s**. Full derived-artifact preflight passed. Logs: `/private/tmp/pr3017-buddy-base-targeted.log` and `/private/tmp/pr3017-buddy-base-preflight.log`. No full suite or new live-provider requests.

- Dev `74557e202ac38c6d29510d0940a062ca7cc7f38b` added PR #2995 Console chat starts, session/Resume ownership and persistence/recovery integration. The sole conflict was appended testing lessons; all current dev lessons and the complete PR lesson were retained. Reviewed production/test patches are unchanged after normalizing only diff positions and index hashes. Six PR modules plus store continuity, session controller, active-path Resume, provider gateway and Resume handoff registration: **574 passed, 39 failed in 380.68s**. All 58 cases in the six PR modules passed. The 39 failures are five store-continuity and 34 active-path Resume cases, all `RecoveryRequired("raw_source_selection_changed")`; every exact failed node and exception line reproduced on exact dev (**39 failed in 16.70s**) after temporarily switching the isolated worktree, restoring the PR branch in a `finally` block. No failure was excluded or fixed outside PR scope. Full derived-artifact preflight passed. Logs: `/private/tmp/pr3017-chat-start-base-targeted.log`, `/private/tmp/pr3017-chat-start-dev-baseline.log`, `/private/tmp/pr3017-chat-start-base-preflight.log`. No full suite or new live-provider requests.

Exact failed nodes reproduced on this dev base:

```text
Tests/UI/test_console_store_continuity.py::test_manual_stream_completes_once_across_real_navigation
Tests/UI/test_console_store_continuity.py::test_reattach_render_failure_retries_before_decisions_and_view_timers
Tests/UI/test_console_store_continuity.py::test_a_wake_that_ran_while_console_was_unmounted_is_in_the_transcript
Tests/UI/test_console_store_continuity.py::test_transcript_payload_db_and_active_leaf_all_agree
Tests/UI/test_console_store_continuity.py::test_rapid_route_switching_leaves_console_interactive
Tests/UI/test_console_resume_active_path.py::test_restart_memory_banner_matches_dispatch_effective_state[auto-generated_prefix-prefix-u2-\u2935 Earlier turns summarized for context \u2014 full history above]
Tests/UI/test_console_resume_active_path.py::test_restart_memory_banner_matches_dispatch_effective_state[manual_prefix-generated_prefix-prefix-u2-\u2935 Earlier turns summarized for context \u2014 full history above]
Tests/UI/test_console_resume_active_path.py::test_restart_memory_banner_matches_dispatch_effective_state[range-generated_range-range-u2-Context uses a summary of turns #2-#2 - full transcript remains visible.]
Tests/UI/test_console_resume_active_path.py::test_restart_memory_banner_matches_dispatch_effective_state[reset-raw-None-None-None]
Tests/UI/test_console_resume_active_path.py::test_restart_memory_banner_matches_dispatch_effective_state[corrupt-raw-None-None-None]
Tests/UI/test_console_resume_active_path.py::test_restart_memory_banner_matches_dispatch_effective_state[dangling-raw-None-None-None]
Tests/UI/test_console_resume_active_path.py::test_restart_memory_banner_matches_dispatch_effective_state[sibling-raw-None-None-None]
Tests/UI/test_console_resume_active_path.py::test_restart_legacy_prefix_banner_matches_dispatch_effective_state
Tests/UI/test_console_resume_active_path.py::test_restart_hides_off_lineage_banner_and_returning_restores_it[range-u2-generated_range]
Tests/UI/test_console_resume_active_path.py::test_restart_hides_off_lineage_banner_and_returning_restores_it[sibling-a3-alt-generated_prefix]
Tests/UI/test_console_resume_active_path.py::test_file_backed_banner_add_replace_clear_restore_is_presentation_only
Tests/UI/test_console_resume_active_path.py::test_mounted_file_backed_sibling_navigation_clears_and_restores_banner
Tests/UI/test_console_resume_active_path.py::test_console_messages_from_conversation_tree_flattens_all_branches
Tests/UI/test_console_resume_active_path.py::test_resume_reconstructs_older_branch_from_active_leaf
Tests/UI/test_console_resume_active_path.py::test_resume_loads_off_path_siblings_for_swipe
Tests/UI/test_console_resume_active_path.py::test_resume_falls_back_to_recent_leaf_and_repairs_pointer_when_missing
Tests/UI/test_console_resume_active_path.py::test_resume_chains_legacy_flat_roots_into_full_transcript
Tests/UI/test_console_resume_active_path.py::test_resume_chains_flat_prefix_then_preserves_real_continuation
Tests/UI/test_console_resume_active_path.py::test_resume_falls_back_when_pointer_dangles
Tests/UI/test_console_resume_active_path.py::test_resume_valid_leaf_wins_over_marker_and_repairs_cursor_pair
Tests/UI/test_console_resume_active_path.py::test_resume_dangling_leaf_ignores_valid_marker_and_repairs_to_newest
Tests/UI/test_console_resume_active_path.py::test_resume_invalid_marker_on_empty_tree_clears_cursor_pair
Tests/UI/test_console_resume_active_path.py::test_resume_marker_reads_current_durable_prompt_content
Tests/UI/test_console_resume_active_path.py::test_legacy_flat_before_first_then_new_root_restart_preserves_all_rows
Tests/UI/test_console_resume_active_path.py::test_resume_second_root_loads_off_path_but_is_not_shown
Tests/UI/test_console_resume_active_path.py::test_resume_clears_stale_persisted_summary_with_dangling_boundary
Tests/UI/test_console_resume_active_path.py::test_resume_leaves_valid_persisted_summary_boundary_untouched
Tests/UI/test_console_resume_active_path.py::test_resume_chains_degenerate_all_user_legacy_conversation
Tests/UI/test_console_resume_active_path.py::test_resume_restores_usage_from_usage_json
Tests/UI/test_console_resume_active_path.py::test_resume_tolerates_null_and_garbage_usage_json
Tests/UI/test_console_resume_active_path.py::test_resume_restores_metadata_from_metadata_json
Tests/UI/test_console_resume_active_path.py::test_resume_restores_an_empty_transcript_row_and_its_explanation
Tests/UI/test_console_resume_active_path.py::test_resume_tolerates_null_and_garbage_metadata_json
Tests/UI/test_console_resume_active_path.py::test_durable_resume_restores_only_guarded_roleplay_context
```

- Dev `f922b3591e0e36a7a9dada7bb846c6d6d5304a2f` added PR #3021 Library/Notes data-integrity fixes and expanded CI coverage. No production or test file overlaps this PR. The sole conflict was appended testing lessons; the entire dev documentation and complete PR lesson were retained. All twelve production/test patches remain identical after normalizing only diff positions and index hashes. Six PR modules plus derived-artifact workflow, CI queue-pressure/dispatch contracts and quit-flow architecture guards: **109 passed in 65.46s** (including all 58 PR cases). Full derived-artifact preflight passed, including the new 143-file UI gate census. Logs: `/private/tmp/pr3017-library-base-targeted.log` and `/private/tmp/pr3017-library-base-preflight.log`. No full suite or new live-provider requests. Auto-merge remains disabled pending current-head Qodo review.

- CI on Library/Notes-base head `49f1058bdd` initially failed `Tests/UI/test_library_notes_sync_delete_restore.py::test_delete_holds_the_folder_and_undo_returns_it_to_up_to_date` with `NoMatches` on `.library-notes-sync-root-status` after the helper's post-selector pause; **657 passed** in the same shard. The test and Library runtime are byte-identical to exact dev and untouched by this PR. Exact-node replays passed on PR (**1 passed in 11.92s**) and exact dev (**1 passed in 9.68s**), restoring the isolated PR branch in `finally`. The single failed-job rerun passed without code changes; all head CI, including the required derived-artifact gate, passed by 22:40Z. This supports an intermittent failure but does not prove its cause. Logs: `/private/tmp/pr3017-ui-shard2-ci.log`, `/private/tmp/pr3017-derived-ci.log`, `/private/tmp/pr3017-library-sync-pr-repro.log`, `/private/tmp/pr3017-library-sync-dev-repro.log`. Workflow: https://github.com/rmusser01/tldw_chatbook/actions/runs/37377190637/attempts/2.

- Dev `54ac2758af81730b3e8b9effdabde2cbf898275c` added PR #3019 bounded model-metadata discovery and Fireworks complete/streamed tool-call fixes. No PR file overlaps; conflict-free rebase preserves all twelve production/test patches after normalizing only diff positions and index hashes. Six PR modules plus hosted parser/streaming/allowances, actual generic handler, inference presets, captured provider replay, offline capture tooling and discovery: **445 passed in 82.64s**, including all 58 PR cases. Full derived-artifact preflight passed. The exact archived provider probes still report **8 failed, 8 passed in 1.33s**; Fireworks's separate tool-call fix does not resolve its audited complete/SSE `logprobs` rejection. No follow-up task was implemented or closed. Logs: `/private/tmp/pr3017-fireworks-base-targeted.log`, `/private/tmp/pr3017-fireworks-base-preflight.log`, `/private/tmp/pr3017-fireworks-base-provider-probes.log`. No full suite or new paid live requests. Auto-merge remains disabled pending current-head Qodo review.


## Fireworks-base CI investigation (2026-10-05)

Head `b7a87f5d0a` failed PR Fast Lane in workflow `37384475397`, attempts 1 and 2, at `Tests/UI/test_console_runtime_ownership.py::test_prepared_native_start_allows_mounted_manual_send`, line 3263. The coordinator returned `AgentChatStartOutcome(launch_status='not_started', reason='target_changed')` before the held readiness resolver entered. Each attempt passed 1410 cases with one skipped in its first phase, then 265 cases with one failed and two xfailed in the admission-sensitive phase. The required derived-artifact job failed solely because of that dependency; all artifact guards, the three UI shards and latency guardrails passed. Exactly one failed-job retry was requested; no third retry was dispatched.

An uninstrumented exact-node replay passed on this PR (1 passed in 9.57s) and reproduced the identical failure on exact dev `54ac2758af` (1 failed in 6.98s). The test and coordinator are unchanged by this PR. Subsequent temporary diagnostic probes did not identify the changed field or duplicate active start: seven exact-node instrumented probes passed, the two immediate predecessors plus the node passed (3 in 20.60s), and the targeted full runtime-ownership module on exact dev passed (80 passed, one existing xfailed, five warnings in 142.88s). Five passive failure-report-hook probes also passed; a sixth ended in a hook-only INTERNALERROR because the diagnostic assumed non-null session settings. That diagnostic error is not evidence of the CI failure or its cause. The corrected hook's probes passed. Every temporary branch switch restored the PR branch in finally; no repository code changed.

The initial sync assertion passed before the coordinator task was scheduled. A queued UI mutation or existing active start is plausible, but the precise cause remains unproven. Passing local probes do not clear GitHub's repeated failure. Further repeated replays require a new hypothesis; no custody check was weakened or unrelated baseline runtime/test code changed.

Logs: `/private/tmp/pr3017-fireworks-base-fast-lane-ci.log`, `/private/tmp/pr3017-fireworks-base-fast-lane-retry-ci.log`, `/private/tmp/pr3017-fireworks-base-derived-ci.log`, `/private/tmp/pr3017-fireworks-base-derived-retry-ci.log`, `/private/tmp/pr3017-native-start-pr-repro.log`, `/private/tmp/pr3017-native-start-dev-repro.log`, `/private/tmp/pr3017-native-start-dev-ordered-diagnostic.log`, `/private/tmp/pr3017-runtime-ownership-dev-ordered-diagnostic.log`, and the diagnostic/field/passive exact-node log series recorded during the investigation. Workflow: https://github.com/rmusser01/tldw_chatbook/actions/runs/37384475397/attempts/2.


## Console first-reply base resync (2026-10-06)

Dev `f99ad86ebd9189fd477992c8c6c3976bc2297ef1` adds PR #3025 first-run handoff, first-reply errors, first-token planning and visible receipt handling. Three PR production files overlap at file level (`console_provider_gateway.py`, `console_trace_runtime.py`, `chat_screen.py`), but the actual hunks are separate: credential-safe provider reason/copy and overflow handling, a missing-revision diagnostic, and setup-card/toast positioning. The conflict-free rebase preserves all three commits by `git range-diff` equality, including all twelve reviewed production/test patches. No new runtime or test changes were made here.

Six PR modules plus provider gateway/failure copy, trace revision diagnostics, first-reply/first-token and first-request planning, first-mount handoff, the previously failing mounted native-start node and CI contracts: **613 passed, one failed in 190.43s**. All 58 PR cases and the previously failing native-start node passed. The sole failure is `Tests/Chat/test_console_provider_failure_copy.py::test_agent_failure_row_carries_body_and_image_recovery_hint`, line 104: `result.accepted` is false with `visible_copy='Hooks unavailable; review or disable hooks...'`. Its setup also reports `raw_source_selection_changed`. The exact node reproduced the identical line and refusal on exact dev (**one failed in 0.76s**); the isolated branch and audit were restored in finally. This baseline fixture failure was neither excluded nor repaired. Full derived-artifact preflight passed, including the newly expanded **146-file UI gate census**. Logs: `/private/tmp/pr3017-first-reply-base-targeted.log`, `/private/tmp/pr3017-first-reply-dev-baseline.log`, `/private/tmp/pr3017-first-reply-base-preflight.log`.

The repository merge queue is now `on`; this PR remains unarmed pending current-head Qodo review. Rebase and fresh CI do not prove the earlier intermittent native-start failure is repaired. No old CI run was redispatched, no third retry was requested, and no paid live-provider requests or full suite were run.


## Switcher/finite-owner base resync (2026-10-06)

Before this resync, head `7bb4f5c746` completed all GitHub CI successfully: Derived Artifacts `37400423472` and Perf Guard `37400423470`, including PR Fast Lane, all three UI shards, latency guardrails, required derived-artifact gate and queue tick. The previous native-start failure was absent; its original cause remains unproven.

Dev `dce291d0fa8c5e27bc999656766d96bd6b00626c` adds PR #3024 Console switcher, rename and finite database-owner retirement changes. The only PR production file overlap is ChatScreen, where the base changes model row lookup, rename-view publication, finite citation/review reads and sidebar hydration; the PR's established-store registry-read change remains a separate hunk. The sole rebase conflict was appended testing lessons. All dev lessons and the full PR lesson were retained. Range-diff changes only that lesson's surrounding context; all twelve reviewed production/test patches are identical after ignoring index hashes and hunk coordinates.

Six PR modules plus finite-operation replacement retirement, run-log ownership, frozen-workspace read admission, Console finite reads, the complete runtime-ownership module, controller wiring/session controller and CI contracts: **326 passed, one existing xfailed, four warnings in 432.67s**. All 58 PR cases and the earlier native-start node passed. Full derived-artifact preflight passed, including the **151-file UI gate census** (floor 150). Logs: `/private/tmp/pr3017-switcher-base-targeted.log` and `/private/tmp/pr3017-switcher-base-preflight.log`. An initial command used the wrong directory for the frozen-workspace test and collected no tests; the corrected complete run above is the validation evidence. Pytest later warned while cleaning unrelated stale temporary directories; it exited zero. No runtime/test edits, full suite or new paid live probes. Auto-merge remains disabled pending fresh CI and current-head Qodo review.


## Voice/TTS base resync (2026-10-06)

Before this resync, head `56477ed370e2d6eb9bfbad11c53f91d230c2a879` completed all GitHub CI successfully as of 06:16Z: Derived Artifacts `37420137791` and Perf Guard `37420137716`, including PR Fast Lane, all three UI shards, latency guardrails, the required derived-artifact gate and queue tick. Both review threads were resolved; Qodo had reviewed only the original head. Auto-merge remained disabled.

Dev `5daa67c2d1614baee0edf90aa40fca371553edb8` adds PR #3026 Voice setup and native pocket-tts changes. None of its 38 changed files overlap this PR's production or test files. The conflict-free rebase preserves all three commits by range-diff equality, including all twelve reviewed production/test patches. Only audit/plan documentation was subsequently amended.

Six PR modules plus native pocket-tts, STTS settings reconfiguration, Speech/TTS settings model, module-size ratchet and CI queue/dispatch contracts: **299 passed, 12 failed in 104.14s**. All 58 PR cases passed. Three failures are existing STTS settings assertions (stale confirmation retained, ElevenLabs revision 2 rather than 1, config save returns false); nine are module-size ratchet failures in files untouched by this PR. Every exact failed node reproduced at the same assertion on exact dev (**12 failed in 0.51s**), with the isolated PR branch restored in finally. No failure was excluded, no budgets were raised, and no unrelated runtime/test fixes were made. Full derived-artifact preflight passed, including the **151-file UI gate census** (floor 150). Logs: `/private/tmp/pr3017-voice-base-targeted.log`, `/private/tmp/pr3017-voice-base-dev-baseline.log`, `/private/tmp/pr3017-voice-base-preflight.log`. No full suite or new paid live probes.

Exact failed nodes reproduced on this dev base:

```text
Tests/TTS/test_stts_settings_reconfiguration.py::test_cross_provider_event_persists_fields_and_removes_confirmation
Tests/TTS/test_stts_settings_reconfiguration.py::test_mixed_provider_save_retires_only_effectively_changed_provider
Tests/TTS/test_stts_settings_reconfiguration.py::test_audio_cpp_mapping_serializes_as_nested_toml_table
Tests/Architecture/test_module_size_ratchet.py::test_module_does_not_grow_past_its_budget[tldw_chatbook/Chat/console_chat_store.py]
Tests/Architecture/test_module_size_ratchet.py::test_module_does_not_grow_past_its_budget[tldw_chatbook/Chat/console_interrupt_rounds.py]
Tests/Architecture/test_module_size_ratchet.py::test_module_does_not_grow_past_its_budget[tldw_chatbook/UI/MCP_Modules/mcp_workbench.py]
Tests/Architecture/test_module_size_ratchet.py::test_module_does_not_grow_past_its_budget[tldw_chatbook/UI/Screens/llm_screen.py]
Tests/Architecture/test_module_size_ratchet.py::test_module_does_not_grow_past_its_budget[tldw_chatbook/UI/Screens/watchlists_collections_screen.py]
Tests/Architecture/test_module_size_ratchet.py::test_module_does_not_grow_past_its_budget[tldw_chatbook/Widgets/Console/console_transcript.py]
Tests/Architecture/test_module_size_ratchet.py::test_module_does_not_grow_past_its_budget[tldw_chatbook/app_lifecycle.py]
Tests/Architecture/test_module_size_ratchet.py::test_module_does_not_grow_past_its_budget[tldw_chatbook/app_service_wiring.py]
Tests/Architecture/test_module_size_ratchet.py::test_module_does_not_grow_past_its_budget[tldw_chatbook/tldw_api/client.py]
```

Queue mode remains on. Auto-merge was disabled before resync and stays disabled pending current-head Qodo review and fresh CI. Posting `/agentic_review` remains blocked by automatic approval review pending explicit human authorization; the existing approval question is not repeated.

## Portable setup base resync (2026-10-06)

Before this resync, head `99d3befd02d678754975f191006f2da5e39951e6` completed all GitHub CI successfully as of 10:18Z: Derived Artifacts `37445226642` and Perf Guard `37445226516`, including PR Fast Lane, all three UI shards, latency guardrails, the required derived-artifact gate and merge queue tick. Both review threads were resolved; Qodo had reviewed only the original head. Auto-merge remained disabled.

Dev `76d5d157aa6a628584ead573976c9adee72f5412` adds PR #3030 portable setup: shared `--config`/`--no-splash` launch flags, pre-fence profile selection and Backup/Restore setup entry and refusal messages. None of its 21 changed files overlap this PR's twelve production/test files. The conflict-free rebase preserves all three commits by range-diff equality; all twelve production/test files remain byte-identical to the prior head. Only existing audit/plan documentation was subsequently amended. Both dev testing lessons and the PR lesson are retained.

Six PR modules plus portable launch options, Backup/Restore setup entry, startup unlock, first-run Backup/Restore wizard and CI queue/dispatch contracts: **161 passed in 152.19s**, including all 58 PR cases. Full derived-artifact preflight passed, including the **152-file UI gate census** (floor 151). Logs: `/private/tmp/pr3017-portable-base-targeted.log` and `/private/tmp/pr3017-portable-base-preflight.log`. Pytest warned during post-summary cleanup of unrelated stale temporary directories and exited zero. No runtime/test edits, full suite or new paid live probes.

Queue mode remains on. Read-only GitHub receipts confirmed auto-merge was disabled before rebase and before the lease push. Automatic approval review rejected a combined `gh pr merge --disable-auto`/rebase command as a premature merge; it did not execute. CLI help verified that `--disable-auto` disables auto-merge, and GitHub still reported `autoMergeRequest: null`. The authorized local rebase then succeeded separately; no merge was attempted. Current-head Qodo review remains required before arming. Posting `/agentic_review` still awaits explicit human permission after the earlier approval rejection; the pending question was not repeated.


## Orchestration base resync (2026-10-07)

Before resync, head `a94e1bbb7c45ad9cd14fd532ece891a5785f3b02` completed all required GitHub CI successfully as of 2026-10-06 12:12Z: Derived Artifacts `37458572001` and Perf Guard `37458572364`, including Fast Lane, all three UI shards, latency, required artifact gate and merge queue tick. Both threads were resolved; auto-merge was disabled and Qodo had reviewed only the original head.

Dev `6feb84c1d2203bc3c6d0eecd2823f4e2409b4b4e` merges PR #2918 orchestration routing, durable progress/wakes, typed provider failure presentation and preserving integration repairs. Gateway and ChatScreen overlap at file level in separate hunks. The sole rebase conflict was appended testing lessons; all dev lessons and the full PR lesson remain. Range-diff changes only lesson context, and normalized comparisons confirmed all twelve PR production/test patches unchanged immediately after rebase.

The initial fourteen-module targeted selection reported **666 passed, 42 failed in 159.50s**. Forty failures were in this PR's retry ownership regression: its fixture emits ChatRateLimitError, but its old oracle expected the generic ChatProviderError wrapper. Incoming ADR-211 and project_provider_error now preserve the typed rate-limit error. Updated only the test import/exception expectation and added exact class/provider assertions; all durable call state, custody, request equality and negative admission assertions remain. No production change was made. The six PR modules then passed **58 cases in 46.89s**. The other 666 passing cases from the original selection were not replayed or summed into this result.

Two unrelated provider-copy nodes reproduced at identical assertions on exact dev (**2 failed in 0.90s**): `test_agent_failure_row_carries_body_and_image_recovery_hint` refuses with Hooks unavailable/raw_source_selection_changed; `test_stream_chat_provider_400_reports_one_consistent_status` expects Status: 400. in content-free diagnostic prose. The isolated branch was restored in finally; neither baseline fixture was changed or excluded. Full derived-artifact preflight passed, including the 152-file UI census (floor 151). Changed-test Ruff lint and format and git diff --check pass. Initial Ruff commands could not create a cache under the restricted worktree; no-cache checks succeeded.

Logs: `/private/tmp/pr3017-orchestration-base-targeted.log`, `/private/tmp/pr3017-orchestration-base-dev-baseline.log`, `/private/tmp/pr3017-orchestration-base-pr-green.log`, `/private/tmp/pr3017-orchestration-base-preflight.log`. Post-summary pytest warnings concern stale temporary directories; no foreign cleanup was performed. No full suite or paid live probes.

MERGE_QUEUE is now **off**, checked during this resync. Auto-merge remains disabled. Fresh head CI and current-head Qodo review are required before normal protected merge. The explicit /agentic_review trigger permission is still pending after automatic approval rejection; no trigger was posted and the question was not repeated.
