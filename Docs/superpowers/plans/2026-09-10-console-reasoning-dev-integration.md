# Console local reasoning integration plan

> **For agentic workers:** Use subagent-driven-development for the independent surfaces below; root owns integration and review. User has authorized rebase, review fixes and merge of PR 2575.

**Goal:** Complete TASK-32273 and merge PR 2575 with model-aware local reasoning replay on current dev.
**Architecture:** Keep ADR-090 canonical ThinkingEnvelope, typed provider events, generation ownership and trace admission. Replay settings refine optional conversation Auto; explicit Include/Exclude and Required retain their existing authority. Optional history is projected before serialization and accounting.
**Tech Stack:** Python 3.12, Textual 8, SQLite, httpx, pytest.
**Spec:** backlog/decisions/090-console-thinking-block-ownership-and-replay.md (integration amendment below).

ADR required: yes.
ADR path: backlog/decisions/090-console-thinking-block-ownership-and-replay.md.
Reason: amend the existing provider/history interface without a second storage schema or parallel display system.

## Global constraints
- Only the isolated console-reasoning-history worktree may be edited. Keep the draft backup ref.
- Preserve canonical thinking and continuation through variants, persistence, imports, exports and sync. No raw private hosted thinking exposure.
- Optional Exclude wins; Required continuation is never removed. Explicit conversation Include retains strict compatibility errors and requests all eligible local history. Auto uses local global/endpoint-model preference (auto/current/all/off).
- Review template fingerprints already captured in Tests/fixtures/reasoning_templates; unknown templates use server defaults. Native tools require declared support or explicit remembered configuration.
- No full suite; focused tests plus relevant preflight guards. No merge until Qodo review of the final change and required checks are resolved.

## Shared interfaces
`local_reasoning.py` owns immutable `ReasoningReplayPolicy(mode, source, template_family='', supports_preserve=False, verified=False, native_tools=False)`, existing `resolve_reasoning_policy`, `reasoning_override_key`, `reasoning_mode_setting`, `reasoning_replay_context`, `apply_local_reasoning_template_options`, `supports_local_reasoning`, `REASONING_HISTORY_OPTIONS`.
`ConsoleProviderResolution.reasoning_replay: ReasoningReplayPolicy | None` freezes policy for the request. `ThinkingReplayTarget.reasoning_replay` carries it into canonical history resolution. No persisted LocalReasoningReplay body.
`AgentConfig.reasoning_replay` passes policy to native-tool selection, budget estimation and child config. Internal per-call thinking references are consumed in the bridge before preparation; canonical envelopes remain the only thinking source.
Runtime guidance uses `EXCHANGE_CONTINUATION_KEY = '_tldw_exchange_continuation'`; real user rows start exchanges; fenced tool rows and project instructions do not.

## Task 1: Canonical history and template policy (history worker)
Files: Chat/local_reasoning.py, console_thinking_history.py, console_prepared_request.py, console_history_budget.py; focused history tests.
- [ ] Replace original duplicate replay records with policy-only helpers; preserve reviewed fixtures.
- [ ] Add failing tests using canonical ThinkingEnvelope sidecars: structured reasoning_content/reasoning serialization, current vs old owner groups, Off/Exclude, Include safety, same-model/provider-family eligibility, no tool-trace aggregation.
- [ ] Implement policy projection on complete semantic units, filtering matching thinking provenance with the same groups. Serialization and provenance must agree on both content and separate reasoning fields.
- [ ] Preserve runtime guidance as current exchange, and reviewed Gemma/Qwen rendered templates without losing provenance. Coordinate row-shape changes with root before implementing.
- [ ] Run focused thinking history/prepared-request tests and report evidence. Do not commit during root's rebase.

## Task 2: Per-call agent ownership (agent worker)
Files: Chat/console_agent_bridge.py, console_thinking_capture.py; Agents/agent_models.py, agent_service.py, agent_runtime.py, native_tools.py; focused agent tests.
- [ ] Add failing active native-tool regression using typed thinking events and canonical envelopes, not fabricated aggregate text.
- [ ] Carry exact successful per-call canonical thinking through assistant echoes, consumed before gateway/trace preparation. A child owns only its own call thinking and inherits policy.
- [ ] Distinguish tool-round thinking from final-answer reasoning in canonical source provenance so saved tool traces cannot be replayed as final-answer thoughts. A final call without reasoning must not inherit an earlier tool thought.
- [ ] Pin policy to AgentConfig; count the same projected fields sent, preserve native protocol independently from Off, annotate runtime guidance as exchange continuation.
- [ ] Run focused agent and capture tests and report evidence. Do not commit during root's rebase.

## Task 3: Canonical settings (settings worker)
Files: config.py, UI/Screens/settings_screen.py, Tests/UI/test_settings_console_reasoning_history.py.
- [ ] Add mounted UI/persistence tests for Automatic default, current/all/off, normalized endpoint/model overrides, clear override, explicit native-server setting.
- [ ] Port the original draft replay controls from backup ref, retain existing dev Show model thinking UI and conversation Auto/Include/Exclude. Explain that controls refine conversation Auto.
- [ ] Reuse endpoint/config authority and snapshots; no duplicate show_thinking toggle or legacy display callback.
- [ ] Run focused mounted tests and report evidence. Do not commit during root's rebase.

## Task 4: Gateway and integrated verification (root)
Files: Chat/console_provider_gateway.py, Chat/Chat_Functions.py, LLM_Calls/LLM_API_Calls_Local.py, provider/trace regressions, user guide.
- [ ] Add failing structured local reasoning capture and native-tool transport tests at real adapter boundaries.
- [ ] Resolve bounded template metadata against the frozen endpoint each send; carry immutable policy. Retain existing trace admission and exact serialized payloads.
- [ ] Emit typed local reasoning events for declared formats, support native local tool responses and template preservation options, with streaming and non-streaming parity.
- [ ] Integrate workers; run canonical persistence/display, provider/prepared/trace/agent/UI regression matrix and available live Gemma checks.
- [ ] Review final diff independently, finish rebase, push with exact force-with-lease, update PR description and mark ready for Qodo.
- [ ] Address review comments with evidence, rerun affected checks, confirm reviewed commit/checks/base, then merge as authorized.
