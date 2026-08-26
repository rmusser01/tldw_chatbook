# ADR-087: Console read-only next-send estimate projection

Status: Accepted
Date: 2026-08-25
Related Task: [TASK-22304](../tasks/task-22304%20-%20Show-an-estimated-next-send-price-on-the-Console-Send-button.md)
Related Spec: [Console next-send price indicator design](../../Docs/superpowers/specs/2026-08-25-task-22304-console-next-send-price-design.md)
Preserves: ADR-011

## Decision

The Console store will expose one owner-thread, read-only active-path snapshot for pre-send presentation. It returns detached `ConsoleChatMessage` values and projects any current stream chunks onto only those detached copies. It must not mutate store-owned messages, collapse stream buffers, advance materialization counters or revisions, or invoke persistence.

The Console chat controller will expose one named, synchronous next-send history projection for UI estimates. It consumes the store's read-only snapshots and shares the same system/greeting fold and provider-history filtering implementation as native dispatch. Dispatch keeps its existing materializing history path; the estimate path cannot call `messages_for_session()` because that method may persist a newly materialized pending assistant row.

The no-DOM send-price controller in `UI/Console_Modules` consumes that named projection through `wiring.py`'s late-bound callback graph. It does not inspect store-private message collections or duplicate provider-history filtering. It counts only string content and explicit text parts. Non-text historical parts and pending binary/media attachments receive no fabricated text tokens or dollars; each is surfaced as a separate unpriced-media caveat and makes the full-request total unavailable.

The estimate remains a synchronous, local, ephemeral presentation. It does not run later asynchronous send-time transforms, persist a quote, fetch pricing, or become dispatch authority.

## Context

The next-send indicator needs current system/history/draft/staged text while the user types and while an accepted run allows a queued follow-up. The existing `ConsoleChatStore.messages_for_session()` is not a read-only getter: it materializes buffered stream chunks and can persist a pending assistant row. Calling it from a composer tooltip would let presentation reads write to the database, repeat work on the UI sync path, and blur dispatch ownership.

Reading `ConsoleChatStore._messages_by_session` or its stream buffers from the UI module would avoid that write but violate store ownership and make the estimator depend on private layout. Reimplementing `_provider_message_payloads` filtering in the price module would drift on failed/empty rows, seeded leading greetings, and assistant-generation-state policy. A small owner-provided snapshot plus controller-owned projection keeps those rules at their existing authorities.

Canonical provider history may contain multimodal content lists. The repository's generic local token estimator assigns non-text parts a coarse token allowance for context-window safety, but applying the text input rate to that allowance would present a fabricated media price. The UI therefore needs a text-only accounting projection with explicit unpriced-media disclosure.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Call `messages_for_session()` from the tooltip controller | It may materialize and persist streaming rows, so a presentation read can write. |
| Read store-private message and stream-buffer dictionaries from `send_price.py` | Breaks store ownership and couples UI code to mutable implementation details. |
| Duplicate provider-history filters in the send-price module | The estimator would drift when dispatch rules for failures, empty rows, greetings, or assistant states change. |
| Build the full asynchronous context preview on each edit | Skills, dictionaries, world info, retrieval, provider resolution, and serialization are too expensive and side-effect-prone for per-keystroke UI presentation. |
| Price generic non-text token allowances at the text rate | Fabricates a dollar amount for media whose provider-specific charging model is not represented. |
| Estimate only the visible draft | Omits the conversation context and reply reservation that dominate many requests. |

## Consequences

- `ConsoleChatStore` gains a narrow read-only snapshot contract with tests proving it does not mutate or persist while still reflecting buffered stream text in detached copies.
- `ConsoleChatController` gains a narrow synchronous projection method shared across the Chat/UI boundary; its implementation shares dispatch filtering rather than forking those rules.
- A queued follow-up can estimate against currently buffered assistant text without causing a database write. The eventual completed reply may still change before queued dispatch, so the tooltip remains explicitly estimated.
- Historical non-text parts and pending attachments force an unavailable full total while known text input/reply components remain visible.
- The screen and composer do not gain data ownership. `ChatScreen` forwards a callback, and the existing accumulated-spend cost chip remains independent.

## Links

- [Approved design](../../Docs/superpowers/specs/2026-08-25-task-22304-console-next-send-price-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-08-25-task-22304-console-next-send-price.md)
- [ADR-011: Chatbook Workbench UI System](011-chatbook-workbench-ui-system.md)
