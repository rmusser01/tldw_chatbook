# Console Current and Next-Send Spend Design

Status: implemented in [PR #2397](https://github.com/rmusser01/tldw_chatbook/pull/2397), merged into `dev` as `c418d4516c`.

Closeout: [TASK-31591](../../../backlog/tasks/task-31591%20-%20Close-Console-context-spend-workstream-and-repair-context-control-tests.md).

## User-facing contract

The context/spend button shows request-context fullness first, followed by money with explicit timing:

- Wide: `Context 45% · Current $0.48 · On next send ~+$0.13`
- Narrow: `Ctx 45% · Now $0.48 · Next ~+$0.13`

`Current` is settled conversation spend, using reported provider usage when available and clearly marked local transcript estimates otherwise. It excludes the active draft, staged evidence, and unfinished turn owners. A user's local estimate is not counted again when the following assistant's actual usage covers that turn. Unpriced fleet tokens remain disclosed separately; they do not fabricate a dollar contribution.

`On next send` is the estimated **additional input charge** for the next request. It includes the represented history, active composer draft, and staged evidence. It uses the selected model's uncached input rate. Output cost is added after completion; cache reads may lower the input charge and cache writes may raise it. This is a baseline estimate, not a total conversation projection or maximum bill.

Context occupancy is estimated request usage relative to safe input capacity, not cumulative billed tokens. The tooltip explains request usage, safe capacity, conversation budget, compaction timing, cache status, and spend. Activating the button opens Conversation Inspector.

## States and ownership

- Current pricing and next-send pricing may be unavailable independently. A missing price or token estimate is `unavailable`, never invented zero spend.
- Known zero input pricing can produce `~+$0.00`. A clean empty draft with known pricing and context produces `On next send —`.
- Invalid drafts are unavailable until corrected.
- Pending media or historical media admitted by the provider makes the forecast unavailable, with text-token detail retained. Failed echoes, assistant attachments, missing bytes, lifecycle-excluded rows, and images omitted by capability/budget rules do not suppress an otherwise available text forecast.
- Accepted recovery owners remain request context but are excluded from Current. Quarantined/unaccepted owners are excluded from both. Durable checkpoints use persisted message IDs; local preparation uses transient IDs.
- Unavailable snapshots remain distinguishable from a truly empty conversation.
- Cache details and alerts remain in the tooltip/style without an extra unexplained amount or trailing cache glyph in the approved labels.

## Automatic compaction

Applying untouched quick settings preserves the sparse inherited compaction setting. A deliberate Off → Automatic edit persists an explicit Automatic choice, even when Automatic was the initial effective value. Existing compaction policy and failure handling remain owned by ADR-052.

## Refresh and performance

The chip reuses the shared context token estimate. Idle draft edits use one coalesced refresh owned by `ConsoleDraftSpendRefresh` in the canonical controller wiring; active sends and teardown cancel the pending refresh. Width changes reapply full/compact labels even if spend has not changed.

Media checks reuse the provider's metadata-only admission path and never serialize attachments. Full send configuration is resolved only when admitted user attachments require capability/budget filtering, keeping RAG work off text-only refreshes and startup.

The separate Send-button price tooltip retains its existing contract; this chip's forecast remains the input-only additional charge described above.

## Architecture and verification

ADR required: no. [ADR-052](../../../backlog/decisions/052-console-conversation-memory-and-compaction-policy.md) and [ADR-095](../../../backlog/decisions/095-conversation-owned-console-generation-settings.md) already define context/compaction policy and conversation-owned settings.

See the [completed implementation record](../plans/2026-09-04-console-current-and-next-send-spend.md) for tested behavior and merge evidence.
