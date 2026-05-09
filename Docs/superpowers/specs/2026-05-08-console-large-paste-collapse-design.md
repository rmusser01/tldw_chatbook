# Console Large Paste Collapse Design

## Purpose

Large pasted text should not take over the Console composer or disrupt the agentic control flow. The Console should preserve the exact pasted payload for sending, while rendering large paste chunks as compact, reversible tokens in the input surface.

This is a Console composer interaction design, not a replacement for the chat transcript or a staged attachment system.

## Scope

In scope:

- Collapse paste chunks over 50 characters when the setting is enabled.
- Collapse only explicit paste-like programmatic inserts that call the same paste insertion API.
- Keep normal typing literal, even when the total draft grows beyond 50 characters.
- Preserve the exact send payload.
- Let users intentionally unfurl a collapsed paste token with a two-click confirmation.
- Let users remove a collapsed paste chunk as one atomic Backspace operation.
- Add a user setting, enabled by default, so users can disable auto-collapse.
- Cover the behavior with automated mounted tests and actual rendered screenshot QA.

Out of scope:

- Full multiline editor replacement.
- Rich editing inside collapsed paste chunks.
- Turning pasted text into attachments, files, or Library sources.
- Applying collapse to transcript messages after sending.

## Current Context

The current Console composer is `ConsoleComposerBar`. It owns a visible draft renderer and a hidden compatibility `Input` that keeps the canonical draft string for existing send paths.

The existing auto-grow behavior already keeps large visible drafts bounded. This design extends that model by compacting only large paste or insert chunks before they consume the bounded composer height.

## User-Facing Behavior

### Paste Threshold

- If a user pastes a chunk with length greater than 50 characters, the chunk auto-collapses.
- Explicit paste-like programmatic insertions use the same threshold only when they call the paste insertion API.
- If the chunk length is 50 characters or less, it is inserted as normal text.
- Normal keyboard typing never auto-collapses solely because the draft becomes longer than 50 characters.
- Character count uses Python string length for the exact inserted text. Newlines count as characters.

### Programmatic Insert Boundary

This first implementation treats only these paths as collapse-eligible:

- Textual paste events routed from `ChatScreen.on_paste()`.
- Future source-aware composer calls that explicitly use `insert_pasted_text(text)`.

These paths are not collapse-eligible:

- `load_draft(text)` for restored drafts, test setup, or session recovery.
- `insert_text(text)` for normal typing and command-style text insertion.
- Existing chat/session state hydration.

This prevents restored or preloaded drafts from changing shape unexpectedly.

### Collapsed Token

A collapsed paste chunk renders as:

```text
Pasted Text: X Characters
```

The token appears inside the Console composer's draft area, visually separated enough to be clickable/focusable without looking like an external attachment.

### Unfurl Flow

The token has a two-step interaction:

1. First click on `Pasted Text: X Characters` changes only that token to:

```text
Unfurl?
```

2. Second click on the same `Unfurl?` token converts that chunk into normal text in the composer.

If the user clicks elsewhere, moves focus away, or continues typing outside that token, the token returns to `Pasted Text: X Characters`.

### Editing

- Backspace at the end of a collapsed paste token removes that entire paste chunk.
- Backspace inside normal text removes one character.
- Once a paste token is unfurled, it becomes ordinary text and follows normal text editing behavior.
- Clearing the composer clears all segments and any pending unfurl state.

### Send Payload

Sending joins all draft segments exactly:

- Text segments contribute their text.
- Collapsed paste segments contribute the original pasted text, not the display label.
- Unfurled paste segments contribute their text like normal text.

No sent message should contain `Pasted Text: X Characters` or `Unfurl?` unless the user literally typed those words.

## Data Model

Add an internal draft segment model owned by `ConsoleComposerBar`.

Suggested shape:

```python
@dataclass
class ConsoleDraftSegment:
    id: str
    kind: Literal["text", "paste"]
    text: str
    collapsed: bool = False
```

Composer state:

- `segments: list[ConsoleDraftSegment]`
- `pending_unfurl_segment_id: str | None`

Rules:

- Adjacent text segments should be merged when practical.
- `draft_text()` returns `"".join(segment.text for segment in segments)`.
- `load_draft(text)` resets to one normal text segment and does not auto-collapse.
- `insert_text(text)` remains literal and is used for normal typing.
- `insert_pasted_text(text)` applies the threshold and setting.
- Programmatic callers must opt into collapse behavior by calling `insert_pasted_text(text)`. Generic draft restore paths must use `load_draft(text)` or `insert_text(text)`.
- The hidden compatibility input is updated from `draft_text()` whenever segments change.

## Rendering Model

The visible composer should render from segments rather than from the raw string alone.

Recommended implementation:

- Keep the existing composer frame and bottom placement.
- Replace or extend the current single visible draft renderer with a segment-aware renderer.
- Collapsed paste segments render as compact clickable/focusable token widgets or segment spans.
- Normal text segments render as text.
- If Textual inline span hit-testing is unreliable, render collapsed paste tokens as compact inline-adjacent token widgets within the draft area. The important UX requirement is that the token appears in the composer input surface and remains clearly associated with the draft.

The bounded auto-grow behavior still applies to the visible draft area. Collapsed tokens should generally keep the composer at its minimum height unless surrounding normal text needs more space.

## Settings

Default behavior: enabled.

Preferred config section:

```toml
[console]
collapse_large_pastes = true
large_paste_collapse_threshold = 50
```

Settings UI:

- Add the control under global Settings -> App-level behavior or Console behavior.
- Label: `Collapse large pasted text`
- Help text: `Pastes over 50 characters appear as compact tokens until unfurled.`
- The setting should update the app config and persist via the existing config save helper.

The threshold should be implemented as a constant or config-backed value. The user-facing requirement is 50 characters; exposing threshold editing is not required for the first implementation.

## Event Handling

Paste handling should route through the Console composer when focus is not inside another real text editor.

Expected paths:

- Keyboard characters call `insert_text()`.
- Paste events call `insert_pasted_text()`.
- Programmatic moves into the composer collapse only when they explicitly call `insert_pasted_text()` or an explicit source-aware insertion method that delegates to it.
- Draft restore, session hydration, and setup paths must not call the paste insertion API unless they are intentionally simulating a user paste.
- Send uses `draft_text()`.
- Backspace delegates to segment-aware deletion.
- Token click events stop propagation after handling.
- Composer or screen click-away resets `pending_unfurl_segment_id`.

## Accessibility And Recognition

The collapsed token should be recognizable and reversible:

- The label includes the character count.
- The first click does not immediately expand a large blob.
- The second click is an explicit confirmation.
- The click-away reset prevents accidental expansion state from lingering.
- Keyboard users should be able to focus a token and activate the same two-step flow if practical in the implementation slice.

## Tests

Add failing tests before implementation.

Required automated coverage:

- Paste over 50 characters renders `Pasted Text: X Characters`.
- Paste of exactly 50 characters remains literal.
- Paste under 50 characters remains literal.
- Normal typing past 50 characters remains literal.
- `draft_text()` returns the full pasted payload while display is collapsed.
- Send path uses the full pasted payload.
- Backspace at the end of a collapsed token removes the whole chunk.
- First token click shows `Unfurl?`.
- Second click unfurls that token into normal text.
- Click elsewhere resets `Unfurl?` to `Pasted Text: X Characters`.
- Multiple large paste chunks remain independently addressable.
- Disabled setting inserts large pasted text literally.
- `load_draft()` and session-style draft restore keep long text literal instead of auto-collapsing.

Required visual QA:

- Capture an actual rendered screenshot of the Console with a collapsed large paste token.
- Capture an actual rendered screenshot after first click showing `Unfurl?`.
- Capture an actual rendered screenshot after unfurl showing normal text.
- Do not treat the screen as approved until the user approves the actual screenshots.

## Risks And Mitigations

Risk: inline click targeting in a single `Static` renderer may be brittle.

Mitigation: prefer segment widgets for collapsed tokens if Textual span hit-testing is not robust enough.

Risk: the hidden compatibility input and segment model could diverge.

Mitigation: make the segment list the source of truth and update the compatibility input only through one refresh method.

Risk: large unfurled text can still consume composer space.

Mitigation: preserve the bounded auto-grow behavior already in the composer.

Risk: paste collapse could hide important text from first-time users.

Mitigation: use explicit labels, character counts, reversible unfurl, and an enabled-by-default but user-disableable setting.

## Acceptance Criteria

- Large paste chunks over 50 characters collapse by default.
- Only paste events and explicit paste-like insert APIs trigger collapse.
- Restored drafts and normal programmatic draft loading remain literal.
- Normal typing does not auto-collapse.
- The user can two-click unfurl a collapsed token.
- Click-away resets an unfurl prompt.
- Backspace removes a collapsed paste chunk atomically.
- Sending preserves the exact text payload.
- The behavior can be disabled in Settings.
- Focused tests and actual screenshot QA verify the behavior.
