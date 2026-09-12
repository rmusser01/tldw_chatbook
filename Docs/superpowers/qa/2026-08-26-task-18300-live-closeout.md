# TASK-18300 real-provider lifecycle closeout

Date: 2026-08-26

Provider: OpenAI (`gpt-4o-mini`)

Scope: Conversation Inspector exchange-capture lifecycle and Stop cleanup

## Isolation and disclosure

- The replay ran under disposable `HOME`, XDG config/data/cache, config, and
  SQLite paths below one scratch directory.
- The provider credential was injected into the process environment and was
  never written to the scratch config, test output, or this report.
- Captured prompt/response bodies are intentionally omitted here. Evidence is
  limited to statuses, byte counts, row counts, and usage presence.
- Before and after the replay, the real config remained byte-identical
  (`sha256: 1b23f3a533632631678644eaeabcb1c5737e2cb7b9d6dc1d6e9323f622fece12`,
  stat tuple `1787672060 53804`) and the real data-tree metadata fingerprint
  remained `391f4883d86762a95efb7dc8b2b073c6c3562fec543e9eb84212960dac55b4e4`.

## Structural results

| Scenario | Result | Evidence |
|---|---|---|
| Stop after first real delta | Pass | Assistant `stopped`, one content byte retained; native capture `stopped`, response 2 bytes; one durable row, response 2 bytes. |
| Stop current regeneration | Pass | Original complete (82 bytes); two sibling assistants; regenerated sibling `stopped`, one content byte; capture `stopped`, response 2 bytes. |
| Legacy abandoned tag on current regeneration | Correctly absent | Count 0 because current regeneration persists a sibling instead of restoring a mutated variant. Legacy behavior remains covered by focused tests. |
| Ephemeral real call | Pass | One governed in-memory capture (`complete`, response 8 bytes, usage present); no persisted conversation/message; durable exchange-row delta 0. |
| Hard delete | Pass | Durable exchange rows changed from 1 to 0 through the conversation FK cascade. |

Every captured request reported one omitted key, confirming the credential
allowlist disclosure remained active. The replay made four small real-provider
calls and stayed below the task's $0.10 budget. No exact stopped-call price is
claimed because the provider's terminal usage bucket is unavailable after an
intentional early close.

## Defect and regression

The first successful structural replay emitted
`RuntimeError: generator ignored GeneratorExit` while stopping each OpenAI
stream. Root cause: `chat_with_openai()` yielded the synthetic SSE `[DONE]`
sentinel from its `finally` block. A Console Stop closes the suspended generator,
so that yield occurred while handling `GeneratorExit` and interfered with
response/session cleanup.

The regression
`test_stopping_stream_closes_transport_without_yielding_after_generator_exit`
failed before the fix with the same runtime error. The production change keeps
response/session cleanup in `finally` and emits the normal/error sentinel only
after the block. The focused test passed, and the complete real-provider replay
then passed without an ignored-`GeneratorExit` warning.

## Related verification

The existing focused gate covers the pieces that are either deterministic UI
contracts or no longer current live paths:

- legacy restored variants retain captures marked abandoned;
- the cost-chip action, Ctrl+Shift+P binding, and command-palette command open
  the unified Inspector;
- Costs, Exchange, and Next Send tabs render in the same Inspector;
- ephemeral capture persistence, kill-switch behavior, reported/estimated
  token labels, and no-capture rows remain pinned.

The earlier real-provider session recorded the multi-call tool-loop and
kill-switch scenarios. This closeout replay adds the non-empty Stop,
regeneration-sibling, ephemeral, deletion, and transport-cleanup evidence that
was previously missing.

Final targeted gate:

- OpenAI streaming module: 6 passed;
- OpenAI Stop through the real Console gateway/provider chain: 1 passed and
  proven red against the pre-fix yield-from-`finally` behavior;
- Inspector/capture/UI/import-provenance selection: 110 passed;
- Ruff on the changed Python files: clean;
- `git diff --check`: clean;
- retired modal class definitions/runtime references: absent.
