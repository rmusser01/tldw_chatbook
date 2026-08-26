# TASK-22304 — Console next-send price indicator design

**Status:** Approved for implementation
**Design direction:** Approved in conversation
**Target:** Native Console composer send/queue action
**Visitor mode:** Operate

## Context

The Console already reports accumulated token spend in its cost chip after work has happened. It does not tell a user what the message currently waiting in the composer may cost before they commit to it. The result is a blind paid action, especially when a long conversation, staged evidence, or a large reply allowance makes the next request materially larger than the visible draft suggests.

This feature adds a compact pre-send affordance to the existing action instead of creating another chip, panel, or settings surface. It must preserve the Console's one-row composer, existing Send/Queue state machine, blocked-state recovery copy, and accumulated-spend cost chip.

## Goals

- Make the presence of a next-request cost estimate visible as soon as a sendable payload exists.
- Explain the estimate on mouse hover with separate input and maximum-reply token and cost lines.
- Base the input estimate on the request the Console is preparing, not just the characters visible in the draft.
- Stay honest when pricing, token counts, reply limits, or attachment charges cannot be estimated.
- Keep draft editing responsive by avoiding repeated full-context tokenization on the Console's periodic sync path.

## Non-goals

- Do not predict the response's actual length; the reply line is an upper bound from the configured maximum response tokens.
- Do not replace, merge with, or change the accumulated-spend cost chip.
- Do not add a billing settings screen, fetch live prices, or persist estimate state.
- Do not promise cache discounts before the provider reports actual cache usage.
- Do not estimate provider-specific image, audio, video, tool, or sub-agent charges that the current pricing and pending-attachment models cannot attribute safely.
- Do not add a new key binding, modal, animation, or composer row.

## Interaction contract

The button keeps its current base action label. The price affordance is appended only when the payload is currently admissible through that action.

| State | Button label | Tooltip |
| --- | --- | --- |
| Empty draft, no pending attachment | Existing `Send` or `Queue` | Existing empty-draft guidance |
| Sendable text draft | `Send | $` | Next-request estimate details |
| Sendable queued follow-up | `Queue | $` | Next-request estimate details |
| Sendable attachment-only payload | `Send | $` or `Queue | $` | Cost-unavailable headline, text components when known, and an attachment caveat |
| Provider/setup/run/wake/recovery blocked | Existing unsuffixed base label | Existing blocker/recovery tooltip; blocker truth has priority over price details |
| Unknown model pricing | `Send | $` or `Queue | $` | Token details when known and explicit pricing-unavailable copy |

`| $` is plain terminal text, not an icon or color-only signal. The existing dynamic button-width calculation expands from the base label, so `Send | $` and `Queue | $` remain dimensionally stable while shown. The composer retains its current one-row action cluster and focus/hover styling.

Textual's existing button tooltip supplies the requested mouse-hover behavior. The compact dollar affordance remains visible to keyboard users, but this task does not invent a separate focus-triggered overlay or keyboard command.

## Tooltip copy

When text pricing, the input estimate, and a response-token cap are all known and no unpriced binary attachment is pending:

```text
Next request: up to ~$0.0874
Input: ~1,284 tokens · ~$0.0039
Reply: up to 4,096 tokens · ~$0.0836
anthropic · claude-sonnet-4-6 · rates as of 2026-08-01
```

When the catalog has no rate for the selected provider/model:

```text
Next request: cost unavailable
Input: ~1,284 tokens
Reply: up to 4,096 tokens
anthropic · custom-model · pricing not configured
```

When a binary/media attachment is pending, the full-request headline is unavailable rather than presenting a text-only figure as the total:

```text
Next request: cost unavailable
Input text: ~1,284 tokens · ~$0.0039
Reply text: up to 4,096 tokens · ~$0.0836
Attachments: 2 · media cost not estimated
anthropic · claude-sonnet-4-6 · rates as of 2026-08-01
```

The same honesty rule applies when canonical conversation history already carries multimodal content. Text parts remain countable, non-text parts are excluded from the text token estimate, the headline stays unavailable, and the tooltip adds `Media context: N items · media cost not estimated`. If pending attachments and historical media are both present, both caveat lines appear so the counts retain their distinct meaning.

If the input token estimate fails, the input line reads `Input: token estimate unavailable`. If no maximum response-token value is configured, the reply line reads `Reply: limit not configured`. Either condition makes the total unavailable. Missing provider/model identifiers are omitted from the provenance line rather than leaving empty separators.

Money uses the shared `format_cost_amount` vocabulary: at least two decimal places, up to four below one dollar, and two at or above one dollar. Token counts use comma grouping in this explanatory tooltip. No unknown value is formatted as zero.

## Estimate semantics

The estimated text input is the locally estimated token count of:

1. the active system prompt, when set;
2. the active session's current conversation messages;
3. the live non-whitespace composer draft as the next user message;
4. staged prompt-eligible evidence text from `console_prompted_evidence_text`.

Inline text-file attachments already become draft text and are therefore counted normally. Pending binary/media attachments are not assigned fabricated tokens or dollars; their count is surfaced and the full total remains unavailable.

Canonical historical rows pass through the same current vision/budget admission policy as dispatch before local token estimation. Text is retained directly; each eligible non-text media attachment is counted only for the `Media context` caveat, contributes neither tokens nor dollars, and makes the full total unavailable. Media omitted by dispatch contributes the same text placeholder dispatch uses and does not create a media caveat.

For known text pricing:

```text
input_cost = estimated_input_tokens × input_per_mtok / 1,000,000
reply_cost = max_reply_tokens × output_per_mtok / 1,000,000
upper_bound = input_cost + reply_cost
```

Input uses the standard uncached input rate. The estimate does not assume a prompt-cache hit or discount before actual provider usage exists, making the displayed amount a conservative text-cost upper bound. Local providers with the catalog's explicit zero rates are known `$0.00`, not “pricing unavailable.” Existing catalog `as_of` metadata is always shown when rates are known.

## Architecture

### Send-price module

Add one focused no-DOM module at `tldw_chatbook/UI/Console_Modules/send_price.py`, keeping all new Console-specific behavior/state inside the boundary required by the screen decomposition. The module owns:

- an immutable next-send estimate/presentation value;
- pure upper-bound text cost math;
- honest availability decisions;
- deterministic tooltip formatting;
- a thin memoized orchestration controller.

The pure builder receives token counts, maximum reply tokens, resolved `ModelPricing` or `None`, provider/model identifiers, pending attachment count, and historical non-text media-part count. It performs no I/O, global config reads, widget access, or catalog refresh, and remains directly unit-testable without constructing its controller.

Construct the controller in `wiring.py`, following the existing named late-binding dependency contract. Its dependencies are limited to accessors for:

- active session settings;
- the current Console chat store;
- the chat controller's named, read-only canonical synchronous provider-history projection;
- the pending live-work launch/staged context;
- the pricing catalog and token counter, injectable for tests.

The store exposes a detached read-only active-path snapshot that projects buffered stream chunks onto the copies without mutating store-owned messages, materialization counters, revisions, or persistence. The chat controller passes that snapshot through the same lightweight pre-serialization history projector native dispatch uses. This shared projector folds the seeded leading assistant greeting into the system row, excludes transcript-only system rows, failed/empty rows, leading assistant rows, and assistant generation states that are not legal provider history, and applies the current historical-media vision/budget admission rule. It retains source identity, role, resolved text, and eligible media references but never calls `image_url_part` or base64-encodes bytes. Dispatch serializes that projection afterward; the named estimate method instead returns immutable text rows plus an eligible historical-media count.

The send-price controller adds the live draft and prompt-eligible staged evidence to those text rows. Historical media receives no generic local-tokenizer allowance and no fabricated text price. The controller resolves model pricing, counts pending attachments, and returns the tooltip. Later asynchronous send-time transforms (skills, dictionaries, world info, compaction, provider serialization, and tools) remain outside this pre-send UI estimate; the tooltip is an estimate, not a frozen wire receipt. The controller owns a verified `TokenEstimateCache` entry whose signature contains provider, model, canonical text-only projected history rows (including the resolved system/greeting row), staged text, and draft text. A cache key collision can only cause recomputation because every hit is checked against the complete signature. `wiring.py` is the sole owner of the named late-binding callback graph; `ChatScreen` does not construct, query through, or mutate estimate internals.

### Composer presentation

`ConsoleComposerBar` gains an optional late-binding tooltip provider. Existing standalone composer tests and callers that omit it keep today's labels and generic tooltip.

During `sync_action_state`:

1. derive the existing base label and send-ready state;
2. only when the payload is send-ready, ask the provider for next-send details;
3. append ` | $` to the displayed base label when details are returned;
4. use the returned details as the hover tooltip;
5. preserve existing blocked, wake-turn, setup, and empty-draft tooltip precedence.

The cached `_send_label` remains the unsuffixed base label. This prevents repeated draft-side resyncs from producing `Send | $ | $` and lets Queue/Preparing state transitions remain authoritative.

The mounted ChatScreen only forwards a late-binding callback owned by `wiring.py` when it constructs the composer. No new estimate method or state is added to `ChatScreen`; new Console behavior remains in `UI/Console_Modules/` in line with the screen-size ratchet.

## Refresh and data flow

```text
draft/session/settings/staged-context/attachment change
    → existing composer or Console action-state sync
    → send-price controller receives current draft and reads current dependencies
    → verified token-estimate cache hit or local recompute
    → pure price builder creates tooltip presentation
    → composer updates label, width, and tooltip
```

Draft edits update through the composer's existing `_sync_current_action_state`, so the visible affordance changes on the edit rather than waiting for a periodic tick. Session, provider, model, maximum-token, staged-context, and pending-attachment changes already cause the screen's central control/action sync, which invokes the same provider again with a new signature.

No network call, database write, or price persistence occurs. Unexpected estimation exceptions are caught at the controller boundary and return an unavailable tooltip rather than interrupting typing or Send.

## Accessibility and layout

- `| $` is a readable text signal and does not rely on color.
- Existing focus outlines, button variants, blocked-reason strip, and disabled contrast remain unchanged.
- Hover/focus styles must not change dimensions; only the state-driven label transition changes the fixed button width.
- Tooltip lines are concise, ordered from decision to detail, and use explicit `estimated`, `up to`, `unavailable`, and `not estimated` language.
- The existing dynamic action-row width calculation remains the sole layout authority, including `Queue | $` and narrow-terminal operation.

## Failure handling

- Unknown pricing: keep the dollar affordance, show token lines when available, and say pricing is not configured.
- Token counter failure: keep the dollar affordance, name the unavailable input estimate, and omit a numeric total.
- Missing reply limit: keep the dollar affordance, name the unconfigured limit, and omit a numeric total.
- Pending media: keep the dollar affordance, show text components when available, name the attachment count, and omit a full-request numeric total.
- Historical media: count only its text parts, name the non-text media-part count separately, and omit a full-request numeric total.
- Controller failure: fall back to a short `Next request: cost unavailable` tooltip; never block or alter dispatch.
- Blocked Send: existing recovery copy always wins over estimate copy.

## Testing strategy

### Pure unit tests

- Known input/output pricing produces the expected upper-bound components and shared money formatting.
- Unknown pricing never fabricates `$0.00` and retains token details.
- Explicit local zero rates produce a known zero-dollar estimate.
- Missing token estimates, missing reply limits, and pending attachments make the headline unavailable with the correct reason lines.
- Provider/model provenance omits blank identifiers and includes `as_of` only for known rates.

### Controller tests

- Context starts from the dispatch path's canonical provider-history projection, then includes the live draft and prompt-eligible staged evidence exactly once.
- Failed/empty rows, transcript-only system rows, and disallowed assistant states remain excluded; a seeded leading assistant greeting is represented only through the folded system row.
- Historically attached media admitted by the shared vision/budget projection forces the headline unavailable and appears in the `Media context` caveat; the estimate path never serializes or base64-encodes it.
- Repeated identical syncs reuse the verified estimate; draft, provider, model, settings, history, or staged-text changes recompute.
- Attachment-count changes update the caveat without assigning media cost.
- A token-counter exception preserves the detailed unavailable-input line, reply line, and provenance; broader controller/catalog failures degrade to the short unavailable presentation.

### Composer tests

- Empty payload keeps `Send` and existing guidance.
- Ready draft renders `Send | $`; an accepted-run follow-up renders `Queue | $`.
- A pending attachment with a blank text draft still renders the suffix and attachment caveat.
- Unknown pricing still renders the suffix and unavailable tooltip.
- Blocked states retain unsuffixed labels and existing recovery tooltips.
- Repeated local resyncs never duplicate the suffix and preserve dynamic action-row geometry.

### Mounted Console regressions

- A priced Anthropic draft displays the suffix and detailed tooltip using the sandboxed fake gateway/config harness.
- Changing the draft, model, session, staged evidence, maximum reply tokens, and attachments refreshes the presentation.
- The accumulated-spend cost chip remains unchanged before send and follows its existing post-send behavior.
- Existing send/queue/blocked-state, context-estimate, pricing-catalog, composer layout, and screen-size-ratchet tests remain green or retain their documented baseline status.

## ADR check

**ADR required:** yes
**ADR path:** `backlog/decisions/088-console-lightweight-next-send-history-projection.md`
**Reason:** The feature adds a cross-module detached store snapshot and shared pre-serialization history projection so per-keystroke pricing can observe buffered text and admitted media without writes or base64 work.
