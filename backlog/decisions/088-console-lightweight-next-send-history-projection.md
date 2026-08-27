# ADR-088: Console lightweight next-send history projection

Status: Accepted
Date: 2026-08-25
Related Task: [TASK-22304](../tasks/task-22304%20-%20Show-an-estimated-next-send-price-on-the-Console-Send-button.md)
Related Spec: [Console next-send price indicator design](../../Docs/superpowers/specs/2026-08-25-task-22304-console-next-send-price-design.md)
Supersedes: ADR-087
Preserves: ADR-011

## Decision

The Console store exposes one owner-thread, read-only active-path snapshot for pre-send presentation. It returns detached `ConsoleChatMessage` values and projects current stream chunks onto only those detached copies. It must not mutate store-owned messages, collapse stream buffers, advance materialization counters or revisions, or invoke persistence.

The Console chat controller owns one shared lightweight history projector before provider serialization. It applies the same role/status/empty-row/leading-greeting/assistant-state filters and vision-budget selection used by dispatch, but retains only source identity, role, resolved text, and references to eligible media attachments. It does not call `image_url_part`, base64-encode bytes, or build provider wire content.

Native dispatch consumes that shared projection and performs its existing provider serialization afterward. The read-only estimate path consumes the same projection and returns an immutable `ConsoleNextSendHistoryProjection` containing text rows plus an eligible historical-media count. This makes filter and media-admission changes shared while keeping serialization entirely off the composer/action-sync path.

The no-DOM send-price controller in `UI/Console_Modules` consumes the immutable projection through `wiring.py`'s late-bound callback graph. Non-text historical parts and pending binary/media attachments receive no fabricated text tokens or dollars; each is surfaced as a separate caveat and makes the full-request total unavailable.

The estimate remains synchronous, local, and ephemeral. It does not run later asynchronous send-time transforms, persist a quote, fetch pricing, or become dispatch authority.

## Context

ADR-087 correctly rejected `ConsoleChatStore.messages_for_session()` because it materializes buffered streams and can persist pending assistant rows. Its first controller design still proposed reusing `_provider_message_payloads` before stripping non-text parts. That method is a serializer: for eligible historical images it calls `image_url_part` and base64-encodes their bytes. Running it before the token cache on every draft edit or 0.2-second action sync can allocate and encode megabytes solely to discard the result.

The selection rules and serialization work therefore need separate boundaries. Dispatch and estimation must agree on which transcript rows and historical media are eligible, but only dispatch needs data URLs. A shared pre-serialization projection preserves that agreement without moving provider-wire work into presentation.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Reuse `_provider_message_payloads` and strip media afterward | It already base64-encodes eligible bytes, so stripping is too late to protect per-keystroke performance. |
| Give the estimate path an independent lightweight filter | Avoids serialization but will drift from dispatch when row or media-admission rules change. |
| Cache serialized provider history | Retains large sensitive base64 payloads, needs invalidation across streaming/model/capability changes, and still performs the first unnecessary encoding. |
| Ignore historical media entirely | Can display a numeric total that silently excludes content the provider will receive. |
| Call the asynchronous exact-context preview | Too expensive and side-effect-prone for synchronous composer presentation. |

## Consequences

- `ConsoleChatStore` gains a tested detached snapshot contract that reflects buffered stream text without writes.
- `ConsoleChatController` gains a private shared pre-serialization projector and a public immutable estimate projection. Dispatch output remains covered by its existing serializer tests.
- Estimate tests monkeypatch `image_url_part` and prove historical-media pricing never calls it; dispatch characterization proves serialization still occurs only on the provider path.
- A queued follow-up can estimate current buffered text without database writes or media encoding. The eventual reply may still change before dispatch, so the tooltip remains explicitly estimated.
- Historical media counts only attachments admitted by the same current vision/budget policy as dispatch. Omitted media contributes the same text placeholder dispatch uses and does not force a media caveat.
- `ChatScreen` only forwards the late-bound callback, and the accumulated-spend cost chip stays independent.

## Links

- [Approved design](../../Docs/superpowers/specs/2026-08-25-task-22304-console-next-send-price-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-08-25-task-22304-console-next-send-price.md)
- [Superseded ADR-087](087-console-read-only-next-send-estimate-projection.md)
- [ADR-011: Chatbook Workbench UI System](011-chatbook-workbench-ui-system.md)
