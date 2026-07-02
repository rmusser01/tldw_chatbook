# Console Dual-Audience UX Design

**Date:** 2026-07-02
**Status:** Approved
**Scope:** Console screen (`ChatScreen` + `Widgets/Console/*`) — layout, information architecture, first-run experience, keyboard layer, visual hierarchy.

## Goal

Make the Console screen serve two audiences with one UI:

- A **first-time user** (fresh install, no API key) reaches a first successful message without reading docs.
- A **power user** navigates sessions, switches models, and acts on messages keyboard-first, with plumbing detail out of the way.

**Strategy: progressive disclosure.** One UI that starts minimal and reveals depth with use. No Simple/Advanced mode switch. The existing skeleton — left rail, transcript canvas, right inspector rail, composer — is kept; everything is improved in place.

## Current pain (from 2026-06 UAT captures)

1. The left rail conversation list is a wall of identical `Chat 1 [workspace]` rows with no timestamps, ordering cues, or grouping.
2. Rail plumbing (Storage, Sync, File tools, Server handoff, Handoff references, sampling params) renders at equal visual weight with everyday controls.
3. First-run blocking is an ASCII-arrow banner (`Provider setup needed: OpenAI missing API key ---->`) over a giant empty canvas.
4. Console-level keyboard coverage is thin (F1 help, F6 pane cycling only).
5. Frame density: doubled vertical rules around the rail, two flat header rows, mixed glyph language (`>`, `x`, dash arrows).

## 1. Left rail information architecture

The rail becomes **four fixed, collapsible sections**, top to bottom:

```
┌ Session ──────────────────┐
│ Workspace: Default      ▾ │
│ Conversations         [⌕] │
│ ▸ API refactor plan   2m  │
│   Groq testing        1h  │
│   Chat 1              3d  │
│   [ + New ]               │
├ Context ──────────────────┤
│ No staged work   [Attach] │
├ Model ────────────────────┤
│ openai / gpt-4o         ▾ │
│ T 0.6 · 8k ctx · stream off│
├───────────────────────────┤
│ ▸ Details (storage, sync, │
│   handoff…)               │
└───────────────────────────┘
```

- **Session**: workspace selector line (opens the existing workspace switcher modal), conversation list, `+ New`.
- **Context**: staged context, behavior unchanged, re-framed as a section.
- **Model**: two-line summary (`provider / model`, then sampling/context/streaming). Enter/click opens the quick model popover (§3); a `Configure` affordance opens the full settings modal.
- **Details**: collapsed-by-default disclosure absorbing Storage, Sync, File tools, Server handoff, and the Handoff reference list. Nothing is removed — it is demoted.

**Conversation list fixes:**

- **Auto-titling.** When the first user message is persisted to a conversation that still has a default `Chat N` name, the conversation title is set to the first message truncated to ~30 characters. Explicit renames are never overwritten (default-name pattern check).
- **Recent-first ordering** with relative timestamps (`2m`, `1h`, `3d`) on each row.
- **Workspace scoping.** The list shows the current workspace's conversations by default; the existing debounced grouped search (reached from the section header) covers cross-workspace lookup.

**Persistence.** Section expand/collapse state persists via the existing `ConsoleRailPreferences`. First-run defaults: Session, Context, Model expanded; Details collapsed.

## 2. First-run experience

The top blocker banner is **removed**. Its job moves to a **setup card** in the empty transcript canvas plus the composer status line.

```
┌ Get started ──────────────────────────────────┐
│ 1. ✓ Add an API key          openai · key set │
│ 2. ● Pick a model               [ Choose… ]   │
│ 3. ○ Send your first message  (type below,    │
│      Enter to send)                           │
└───────────────────────────────────────────────┘
```

- **Live state.** Step 1 reflects credential presence (reusing the merged API-key recovery action); step 2 reflects model selection. Steps derive from the same provider-readiness check as the composer blocker (§5).
- **Keyboard-reachable.** Card buttons are focusable and included in pane cycling.
- **Composer.** Typing is always allowed. Send stays blocked until the provider is ready, with the reason inline: `Send blocked — add an API key to continue`.
- **Lifecycle:**
  - Setup incomplete + empty transcript → full card.
  - Setup complete, user has never sent a message → card collapses to one line: `Ready — type a message to begin.`
  - After the first successful send ever (persisted flag alongside rail preferences) → empty transcripts show only `No messages yet.`, forever, including new tabs and new workspaces.

## 3. Keyboard layer

Principle: **the composer is home base; every frequent action is one binding away; every binding is discoverable.**

**Overlays:**

- **Ctrl+K — Session switcher.** Fuzzy-find across conversations and workspaces, recent-first, built on the existing grouped-search service and workspace switcher modal. Enter opens in the current tab; Ctrl+Enter opens in a new tab; F2 renames in place.
- **Ctrl+M — Model popover.** Provider column ▸ model column, arrow-key driven, with a compact temperature/streaming row. Enter applies to the current session; `Full settings…` opens the existing settings modal.

**Navigation and message actions:**

- F6/Shift+F6 pane cycling retained. **Escape from any pane returns focus to the composer.**
- Transcript focused: up/down selects messages; the existing message-action row becomes keyboard-driven — `c` copy, `e` edit, `r` regenerate.
- Tab strip: Ctrl+T new tab; Alt+1…9 jump to tab N.
- Exact keys are verified against terminal/Textual conflicts at implementation time (e.g., Ctrl+W is off-limits: delete-word in inputs). Conflicted bindings get alternates; the *actions* are fixed.

**Discoverability:**

- The screen footer shows contextual key hints for the focused pane (max 4–5, via Textual's binding display).
- Every action above is also registered in the command palette (Ctrl+P). Nothing is keyboard-only-and-secret.

## 4. Visual hierarchy & density

Four rules, applied consistently and mirrored in both stylesheet files (`css/components/_agentic_terminal.tcss` and `css/tldw_cli_modular.tcss`, inline `#6f7782` frame color per existing convention):

1. **One frame per region.** Rail, transcript, composer each get a single-line border; the doubled rail rules go away. Rail sections separate with a horizontal rule + bold header, not nested boxes.
2. **One header line.** The `Console` title row merges with the `Chat/RAG/Follow | … | Approvals 0` status row. Zero-valued counters render dimmed; non-zero counters render bright.
3. **Composer reads as primary.** Border brightens on focus (existing focus/draft state classes). Send is the only primary-styled button; Stop appears only while streaming; Attach/Save are quiet secondaries. When blocked, Send dims with the reason beside it.
4. **Consistent glyphs.** `▸` active/selected, `●` in-progress, `✓` done, `▾` expandable. Transcript role labels dimmed, message text full contrast, selected message gets the accent border. Tab strip gets a blank line of separation from the transcript title.

## 5. Architecture

| Unit | Role |
|---|---|
| `Chat/console_rail_state.py` (extend) | Section expand/collapse prefs + persisted first-send flag, via `ConsoleRailPreferences` |
| `Chat/console_display_state.py` (extend) | Conversation row display: titles, relative timestamps, ordering |
| `Widgets/Console/console_setup_card.py` (new) | Setup card; renders from provider-readiness state, emits button events |
| `Widgets/Console/console_session_switcher_modal.py` (new) | Ctrl+K fuzzy switcher; generalizes the existing workspace switcher over the grouped-search service |
| `Widgets/Console/console_model_popover.py` (new) | Ctrl+M popover; delegates persistence to existing session-settings machinery |
| `UI/Screens/chat_screen.py` | Bindings, rail composition, footer hints, palette registration — orchestration only |

**Data-flow decisions:**

- **Single source of truth for readiness.** Setup card, composer blocker, and Model section all derive from the same provider-readiness check in `console_provider_support`. No parallel logic.
- **Auto-titling** lives in the Chat layer at first-user-message persistence time, gated on the default-name pattern.

**Error handling:**

- Model popover with no configured providers shows a setup hint linking back to the card flow.
- Missing conversation timestamps render blank rather than erroring.
- Unknown/corrupt persisted section states fall back to first-run defaults.

## 6. Design reference: posting

[posting](https://github.com/darrenburns/posting/tree/main/src/posting) (darrenburns) is the reference for how a well-factored Textual app answers backend design questions. Patterns to follow when implementing:

- **Context-aware command palette provider** (`posting/commands.py`): a single `textual.command.Provider` subclass whose `commands` property inspects current screen state and yields only currently-applicable commands (e.g., only the layout you're *not* in). Our §3 palette registration should be one `ConsoleCommandProvider` built this way — not ad-hoc per-widget registration.
- **Typed `Message` dataclasses** (`posting/messages.py`): widgets communicate upward via small frozen dataclass messages, never by reaching into parents. New widgets (setup card, switcher, popover) emit messages; `chat_screen.py` handles them.
- **One module per concern, widgets grouped by feature directory** (`widgets/request/*`, `widgets/response/*`): matches and validates our `Widgets/Console/*` layout — keep new widgets small and single-purpose rather than growing `chat_screen.py`.
- **Jump mode** (`posting/jumper.py`, `jump_overlay.py`): an Amp-style `Jumpable` protocol (widgets declare a `jump_key`; an overlay paints keys over the screen; one keystroke focuses the target). Not in scope for the four phases, but the pattern to adopt if pane cycling (F6) proves too slow — noted as a candidate follow-up.
- **Centralized help data** (`posting/help_data.py` + `help_screen.py`): binding descriptions live in one place feeding both footer hints and the help screen. Our footer hints (§3) should source from binding metadata the same way, not duplicated strings.

## 7. Testing

- House pattern: failing pilot/widget test first per behavior. Existing files `Tests/UI/test_console_internals_decomposition.py` and `Tests/UI/test_console_persistent_rails.py` extend; new test files cover the setup card, session switcher, and model popover.
- Both stylesheets mirrored; CSS build step run.
- Screenshot QA via the textual-serve live-capture workflow with real CSS.
- Results judged against the known ~33-failure pre-existing UI test baseline; CI checks may be cancelled by design — verify locally.

## 8. Phasing

Four independently mergeable phases:

1. **Rail IA** — sections, Details disclosure, conversation list fixes, auto-titling.
2. **First-run card** — setup card, banner removal, lifecycle flag.
3. **Keyboard layer** — switcher, popover, message actions, footer hints, palette registration.
4. **Visual polish** — frames, header merge, button priority, glyph pass.

Each phase follows spec → failing tests → implementation → screenshot QA, and each Console screen change requires explicit user approval before merge (per standing project rule).
