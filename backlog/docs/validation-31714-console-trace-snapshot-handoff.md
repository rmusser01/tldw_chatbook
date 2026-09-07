# TASK-31714: Console trace fix and snapshot handoff validation

Date: 2026-09-05. Base: `c14dadd77080be929d47e3acef41be790ee5d8d1` (`dev`).
Branch: `codex/chatbook-snapshot-trace-fix` in an isolated temporary worktree.
The dirty shared Chatbook checkout was not modified.
Existing architecture: [ADR-097](../decisions/097-console-reference-backed-semantic-trace-ledger.md).
No schema, trace ownership contract, capture policy or snapshot support gate changed.

## Diagnosis and correction

The ordinary second-send failure reproduced both after snapshot restoration and
with no runtime lifecycle operations. The production trace service rejected
`unsupported_surface_change`: a previously saved user reference arrived as an
`ACTIVE_REQUEST` artifact alongside appended history.

The native owner ID had not been lost. Saved history also carried the private
`_tldw_persisted_message_id` and `_tldw_persisted_conversation_id` annotations.
The durable trace builder retained them in semantic provider values, while
serialization removed them. The agent request factory's exact row comparison
therefore could not reuse the saved descriptor. `_build_durable_trace_request`
now excludes all three private identity annotations from both compared visible
rows and frozen provider values. Native owner lookup and exact content matching
remain authoritative; no text-only owner inference or broader delta allowance
was introduced.

## Automated evidence

- New regression uses real SQLite, persistence, controller, agent bridge, gateway
  and production `ConsoleTraceBoundaryFactory`; only network inference is faked.
  Before the fix, the second send never reached its second boundary. After the
  fix it completes, even with identical text in two differently owned user rows.
- Changed history remains explicitly blocked. Native-reader reconstruction of
  the first trace is identical before and after either second-send outcome.
- Final targeted run: **339 passed, 1 deselected, 1 warning, 37.74s** across
  controller, trace runtime, service and native-reader modules.
- The deselected tool-catalog count assertion expects 29 exclusions but current
  dev has 26. It failed in the initial whole-module run and was independently
  reproduced with the complete unchanged HEAD controller loaded before pytest.
  It is not silently counted as passing or repaired by this trace fix.
- Ruff 0.16.6 found **no introduced diagnostics**, including security rules:
  controller 176 baseline/current, test module 16 baseline/current. Repository
  lint debt remains; whole-file lint is not clean. Changed-range formatting and
  `git diff --check` pass. No full-suite sweep or dependency installation.
- Independent read-only review: no critical/important findings. Its suggested
  historical reconstruction assertion was added and both cases passed again.

Logs: `/private/tmp/chatbook-trace-{red,green,reconstruction,final-targeted}.log`,
`/private/tmp/chatbook-catalog-baseline.log` and scoped Ruff output.

## Real mounted Chatbook and native runtime

Disposable fixture: `/private/tmp/llamacpp-chatbook-validation.3vIa7q`.
Actual `TldwCli.run_test`, visible composer send action, production Admin HTTP
snapshot routes, supervisor and native llama-server. This is mounted-app UAT,
not physical-terminal input or a new browser walkthrough.

Config/XDG/data paths were synthetic; an OS fence denied real-home writes and
all outbound network except localhost ports 18584/18585. `HOME` was unchanged.
Runtime: official macOS ARM64 b10816, supplied Gemma 4 26B/4A Q4_K_M,
CPU, context 16384, one slot, explicit full-SWA cache, reasoning disabled.
Only the fixture admitted its executable hash; production allowlist stays empty.

Two ordinary sends first passed without lifecycle operations. Then:

| Observation | Measured result |
| --- | --- |
| Save / Stop / Start / Restore | Saved and restored 6038 tokens |
| Actual Chatbook follow-up | Reused 6031, processed 27 tokens |
| Identical captured request, cold native control | Reused 0, processed 6058 tokens |
| Lifecycle effects | No automatic send; messages, selected durable records and provider settings unchanged |
| Pause / Resume | No snapshot operations or automatic sends; messages and selected durable records unchanged |

The cold control is a direct request with the captured payload, not another
Chatbook message. Each successful send required a fresh native response as well
as a settled run state. The mounted app exited without an app exception.
Payloads omit `id_slot`; this establishes single-slot reuse, not concurrent
conversation-to-slot affinity. Prior stale-wait captures remain invalid.

Evidence: `chatbook-fixed-lifecycle.log`, `chatbook-result.json`,
`chatbook-wire.json`, `control-fixed.log`, and the mounted SVG screenshot.

## Remaining tool-loop blocker

The model requested exactly calculator `17 * 19`. The real approval card stayed
pending across save/restart/restore (6085 tokens), with unchanged messages,
durable state and settings and no automatic inference. The harness checked the
single tool name, builtin server and arguments before pressing Approve once.
Continuation then blocked **before another native request**:
`console_trace_runtime.py:148`, `ValueError: trace_turn_unavailable`.

The factory currently requires the final payload descriptor to identify the
saved active turn (except its explicit direct-prefill case). A tool-loop tail
is a provider-only tool artifact, so that rule does not establish its owning
turn. Fixing this needs a separate, validated tool-chain ownership path, not
loosening the guard or selecting an arbitrary earlier saved message.
Tool execution/continuation and warm tool-loop cache reuse are **not validated**.
A separate mounted calculator control with no snapshot or lifecycle operations
reproduced the identical error before its continuation dispatch. This is an
independent Chatbook tool-chain problem, not evidence of broken cache restoration.

Evidence: `chatbook-tool-diagnostic.log`, `chatbook-tool-trace-errors.json`,
`chatbook-tool-no-lifecycle.log`, `chatbook-tool-wire.json` and
`chatbook-tool-pending.svg`. Server snapshot
acceptance remains open; successful plain chat does not open production support.

## Cleanup

The owned native profile and API process were stopped. The two remaining
synthetic tool-run cache copies (668,892,856 bytes combined) were permanently
deleted; no saved copies remain. Ten completed operation receipts, synthetic
databases and diagnostic logs were retained. Neither fixture port has a listener.
Original model/executable assets and real user profiles were unchanged.
