# Console Thinking Blocks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show only actual, adapter-approved model thinking as durable collapsible Console activity, preserve compatible reasoning history when requested, and keep proprietary/private reasoning and answer-oriented surfaces honest.

**Architecture:** A versioned generation-owned thinking envelope sits beside visible assistant content and ADR-063 continuation. Provider adapters emit explicit displayable deltas or content-free proprietary evidence; one prepared history projection performs provider-specific replay and accounting; the existing Assistant-turn disclosure hierarchy renders the result under device-local visibility and conversation-owned replay policy. Four atomic child tasks deliver persistence first, then provider/history behavior, UI/settings, and finally exchange/privacy integration.

**Tech Stack:** Python 3.11+, Textual 8.x, SQLite/FTS5, Rich, dataclasses and `Literal`, pytest/pytest-asyncio, generated TCSS.

**Spec:** `Docs/superpowers/specs/2026-08-26-console-thinking-blocks-design.md`

**Task:** `backlog/tasks/task-18932 - Console-toggleable-live-reasoning-display.md`

**ADR required:** yes
**ADR path:** `backlog/decisions/090-console-thinking-block-ownership-and-replay.md`
**Reason:** ADR-090 records the accepted schema, variant ownership, sync/conflict, provider event, history replay, privacy/export, backend capability, and Assistant-turn interaction boundaries. The implementation must extend ADR-063 and ADR-066 rather than reopening them.

## Global Constraints

- Execute each child as an independent reviewable slice in dependency order. Do not start a child before its dependencies are complete.
- Before the first implementation edit, use `superpowers:using-git-worktrees`; the current checkout contains unrelated user work in `console_transcript.py` and generated CSS that must not be overwritten.
- At each child start, set only that Backlog task to `In Progress` and add its linked implementation plan. The documented five-digit Backlog CLI addressing bug permits a careful `apply_patch` edit when the CLI cannot address `TASK-18932.x`.
- Recheck `CharactersRAGDB._CURRENT_SCHEMA_VERSION` immediately before creating the migration. The rebased implementation found schema 51 and therefore uses `chachanotes_v51_to_v52_console_thinking.sql`; future rebases must still use the mechanically next version and update every named migration reference in the foundation plan.
- Run schema tests only with isolated temporary database/data directories. Never launch the app against the developer's shared profile during a schema-changing worktree.
- Preserve raw thinking/private values from `repr`, logs, warnings, exceptions, test IDs, snapshots, and approval text. Assertions may compare explicit decoded values only inside privacy-focused tests.
- Provider capability or a thinking-enabled request is never current-turn evidence. Only `ProviderThinkingDelta` or `ProviderProprietaryThinkingEvidence` creates a block.
- Preserve ADR-063 timing and privacy. Displayable thinking never enters `provider_continuation_json`; proprietary raw text never enters `thinking_blocks_json`.
- Preserve today's durability boundary: live variants own complete generation envelopes in session, while only the selected variant is projected to the durable message row.
- Generic content writes preserve supported and opaque thinking by default. Only explicit generation replacement, confirmed assistant edit, discard, or deletion clears it.
- Human-readable and derivative answer surfaces consume `message.content`; they do not concatenate thinking or proprietary application copy.
- Use targeted tests listed in the child plans. Per repository policy, ask before any full test sweep.
- Finish every child with Ruff format/check on touched Python files, `git diff --check`, a self-review against its acceptance criteria, checked ACs, Implementation Notes, and Backlog status `Done` only after all DoD evidence exists.

---

## Delivery Order

| Order | Backlog child | Focused plan | Produces |
| --- | --- | --- | --- |
| 1 | [TASK-18932.1](../../../backlog/tasks/task-18932.1%20-%20Persist-selected-generation-thinking-and-replay-policy.md) | [Persistence foundation](2026-08-26-console-thinking-blocks-foundation.md) | Envelope, schema, selected-generation ownership, sync/backend capability |
| 2 | [TASK-18932.2](../../../backlog/tasks/task-18932.2%20-%20Normalize-provider-thinking-events-and-history-replay.md) | [Provider and history](2026-08-26-console-thinking-blocks-provider-history.md) | Stream events, splitter, accumulation, exact compatible replay/accounting |
| 3 | [TASK-18932.3](../../../backlog/tasks/task-18932.3%20-%20Render-collapsible-Console-thinking-and-settings.md) | [UI and settings](2026-08-26-console-thinking-blocks-ui-settings.md) | Disclosures, lifecycle, device toggle, conversation policy, Planning rename |
| 4 | [TASK-18932.4](../../../backlog/tasks/task-18932.4%20-%20Complete-thinking-exchange-privacy-and-integration.md) | [Exchange and integration](2026-08-26-console-thinking-blocks-exchange-integration.md) | Import/export privacy, documentation, joined evidence and live QA |

## Cross-Child Interfaces

TASK-18932.1 produces these stable seams before provider or UI code consumes them:

```python
ThinkingVisibility = Literal["displayable", "proprietary"]
ThinkingStatus = Literal["complete", "stopped", "failed"]
ThinkingHistoryPolicy = Literal["auto", "include", "exclude"]
ThinkingBlock = DisplayableThinkingBlock | ProprietaryThinkingBlock

def read_thinking_blocks_json(value: object) -> ThinkingEnvelopeRead: ...
def dump_thinking_blocks_json(envelope: ThinkingEnvelope | None) -> str | None: ...

class ThinkingPersistenceCapability(Protocol):
    def thinking_round_trip_version(self) -> int | None: ...
```

`DisplayableThinkingBlock` owns bounded `text` with `repr=False`; `ProprietaryThinkingBlock` has no text member at all. The current production Console persistence implementation is `ChatPersistenceService`. Any additional production adapter found at execution time must prove exact version-1 round-trip before advertising support; missing/legacy/server-mode adapters fail the pre-provider compatibility gate for thinking-capable runs.

`ConsoleChatMessage`, `ConsoleVariant`, and `_VariantStreamBase` expose the parsed/opaque thinking state without rendering it. The selected-generation persistence API accepts the exact canonical JSON or preserves the existing opaque value; no child reaches around this API with ad hoc SQL outside migration, DB repository, or sync code.

TASK-18932.2 produces the only stream evidence and replay seams:

```python
ProviderStreamItem = (
    str
    | ProviderThinkingDelta
    | ProviderProprietaryThinkingEvidence
    | ProviderToolCalls
)

def resolve_thinking_history(
    *,
    target: ProviderResolution,
    policy: ThinkingHistoryPolicy,
    messages: Sequence[Mapping[str, object]],
    thinking_sidecar: Sequence[ProviderThinkingSidecar],
    continuation_sidecar: Sequence[ProviderContinuationSidecar],
) -> ResolvedHistoryProjection: ...
```

TASK-18932.3 consumes envelope snapshots and stream events through store/controller methods. Textual widgets never parse provider wire shapes or raw JSON. TASK-18932.4 consumes canonical export/import helpers and explicit visible-content projections; it does not add another thinking model.

## Joined Definition of Done

- [ ] All four child tasks are `Done`, with every AC checked and Implementation Notes recording targeted evidence.
- [ ] Parent ACs map to at least one passing child or joined test, including actual-turn no-evidence controls and exact proprietary copy.
- [ ] A genuine historical database migrates in an isolated location; a fresh database and the migrated database expose identical supported columns/triggers.
- [ ] Selected generation answer/thinking/continuation remain paired through completion, stop, failure, regeneration, selection, edit, delete, restart, sync, export, and import.
- [ ] Prepared-request tests prove serialized and counted history are one artifact and that proprietary/failed/unknown thinking is absent.
- [ ] UI tests prove expanded-live to one-time auto-collapsed behavior, manual choice precedence, historical collapsed restore, lazy bodies, immediate display toggle, Planning distinction, keyboard semantics, and narrow painted layout.
- [ ] Decoded privacy inspection proves human-readable/search/title/summary/log/usage/speech surfaces omit thinking and that proprietary raw content exists only where ADR-063 requires it.
- [ ] A persistent thinking-capable path with an unsupported backend fails before the injected provider call; supported and unaffected paths still dispatch.
- [ ] Importable exchange warns and round-trips supported displayable thinking/policy; invalid one-conversation input is rejected before that conversation mutates.
- [ ] User documentation and ADR/task links are current; no unsupported promise of hidden chain-of-thought appears.
- [ ] Focused tests, Ruff, CSS bundle verification, diagnostic/privacy inventories, and `git diff --check` pass. A full suite is run only if the user explicitly opts in.

## Final Parent Closeout

- [ ] Run the joined targeted command from the exchange/integration plan in one isolated worktree.
- [ ] Perform the plan vagueness scan and spec-to-test matrix review.
- [ ] Update the parent task ACs and Implementation Notes with child commit/PR references and exact evidence.
- [ ] Set TASK-18932 to `Done` only after all four children and the joined definition above are complete.
- [ ] Use `superpowers:verification-before-completion` before claiming completion, then `superpowers:finishing-a-development-branch` for merge/PR choices.
