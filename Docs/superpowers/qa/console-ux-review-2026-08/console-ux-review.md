# Console Screen — UX/HCI Review & UAT Findings

**Date:** 2026-08-04 · **Reviewer lens:** senior UX/HCI designer · **Surface:** `ChatScreen` (TAB_CHAT, "Console") — `tldw_chatbook/UI/Screens/chat_screen.py` (22.5k lines) + `Widgets/Console/*` + `Chat/console_*`

**Personas**
- **FT-T** — first-time user, technical (comfortable in terminals, new to this app)
- **FT-NT** — first-time user, non-technical (new to TUIs and LLM tooling)
- **PU-T** — regular power user, technical (keyboard-first; agents/RAG/tools)
- **PU-NT** — regular power user, non-technical (daily user, mouse/menu-driven)

**Method**
1. Static review of the screen, widgets, and state modules (exact-copy evidence, `file:line`).
2. Intent comparison vs `Docs/User_Guide/console.md` + six console design specs (2026-05 → 2026-08).
3. Hands-on UAT: the real `TldwCli` app driven headless with Textual `Pilot` in a fully sandboxed HOME/config; 12 scenarios; SVG screenshots + visible-text dumps in `output/ux-review-console/captures/` (harness: `output/ux-review-console/uat_console.py`).

**Severity** — S1 blocks a persona's core task · S2 major friction/confusion · S3 moderate · S4 polish.
**Status** — ✅ verified in UAT capture · 📄 verified in code.

---

## Executive summary — the ten issues that matter most

1. **LY-08 (S1, all)** At 80×24 the **transcript disappears** — the left rail eats the whole workspace grid; only control bar, chips, and composer survive. At 60×18 the screen is an empty frame. (`p4-narrow-80x24`, `p4-narrow-60x18`)
2. **LY-11 (S1, PU-T/RAG users)** The **Inspector rail cannot be opened below 150 columns** — force-collapsed, and clicking the handle at 140 cols silently toggles a preference the user never sees take effect. Staged Sources, retrieval scope, run inspector, and settings summary are unreachable on common terminal widths (VS Code integrated terminals are typically 80–140 cols). (`p7b`, `sandbox/config/config.toml` shows `right_open=true` persisted with no visual change)
3. **LY-01/LY-07 (S2, all)** The **left rail is too narrow for its own content at every tested width**: `Workspace Defau…`, `New conversati…`, `RAG S…`, a search box rendered as `S`/`Sear` overlapping its `Clear` button. At 160 cols it is cramped; at 140 it is borderline unusable. (`p1`, `p2c`, `p7b`, `p4-narrow-110x32`)
4. **FR-04 (S2, FT)** The **Send button is never disabled** — blocked state is CSS+tooltip only, an empty draft gets *no tooltip at all*, and the disabled-reason Static is permanently `display:none`. First-timers get no affordance-based guidance about why nothing happens. (`console_composer_bar.py:1178-1195, 3906-3919`)
5. **AC-02 (S2, PU-T)** **Tab traversal crosses all 15 app-nav buttons** mid-tour; after ~30 Tab stops focus is still inside the left rail — transcript, chips, and Inspector are never reached. F6 exists, but the default Tab order contradicts the visual region order. (`p5-focus-tour.txt`)
6. **FR-07 (S2, FT-NT)** Console Settings modal shows a broken template with an **empty provider name**: `Provider blocked: '' is not available in Console yet.` (`f3c-enter-no-provider`)
7. **TX cluster (S2, FT-NT/PU-NT)** Untranslated jargon in primary surfaces: `Transcript / Event Stream`, `Library RAG`, `RAG: off`, `Scope: empty` (+ `∩`/`→` tooltips, raw `scope_empty` token), `Tools: not loaded`, `prefill/Arm`, `Save Chatbook`.
8. **FB-06 (S2, all)** Stream-failure feedback leaks engineering artifacts — `Assistant [failed]` with no content, system row `Provider stream failed: RuntimeError error (Connection refused: …)` — and **no toast**; the header badge still reads `Ready`; the only ambient signal is a `failed` badge on the *collapsed* Inspector handle. (`p6-send-failure`)
9. **FR-01/FR-05 (S2, FT)** Onboarding choice ambiguity: wizard offers `Cancel` vs `Skip — explore on my own` vs `Esc finish later` with no consequence explanation; the Console setup card then shows step 1 `Finish provider setup` while its only button reads `Choose model` — step and action are out of sync. (`f1`, `f2b`)
10. **DS cluster (S2, FT-NT/PU-NT)** Discoverability depends on hover: icon-only `☰`/`✕`/`▾▸`/`👍👎` controls, an inert-but-focusable `Sources`/`Tools` chip pair, a visible `Narrate Entire Conversation (not implemented yet)` menu entry, click-only jump pill, undocumented middle-click tab close.

---

## Findings (full list, by theme)

### 1. First-run & onboarding

| ID | Sev | Personas | Finding | Evidence |
|----|-----|----------|---------|----------|
| FR-01 | S2 | FT-T, FT-NT | **Three escape hatches, no consequences explained**: wizard shows `Cancel` button, `Skip — explore on my own` link, and `Esc finish later` hint. Cancel/Esc = "finish later" (re-offered via toast next launch); Skip = *never offered again* (commits `setup_completed`). The irreversibility of Skip is invisible. | ✅ `f1-first-run-wizard`; 📄 `FirstRunSetupWizard.py:3015-3019, 3161-3171` |
| FR-02 | S4 | FT-NT | Wizard step tracker shows 4 steps (`Welcome/Provider/Model/Summary`) after the user picks "Quick setup — provider & model" — a small expectation mismatch (Welcome/Summary aren't "provider & model"). | ✅ `f1-first-run-wizard` |
| FR-03 | S3 | FT-NT | **Empty transcript has no action affordance** — copy is only `No messages yet.`; the panel stores a `Choose model` action label but `compose()` never renders it. Dead-end quiet state. | 📄 `console_transcript.py:510-521`, `console_onboarding_state.py:15` |
| FR-04 | S2 | FT-T, FT-NT | **Send is never actually disabled** (`send_button.disabled = False` always); blocked = CSS class + tooltip; **empty draft → tooltip is `None`**; `#console-send-disabled-reason` Static permanently `display:none`. No persistent, perceivable reason why Send won't work. | 📄 `console_composer_bar.py:1178-1195, 3906-3919` |
| FR-05 | S2 | FT-T, FT-NT | **Setup card step/action mismatch**: active step reads `Finish provider setup`, but the card's only button reads `Choose model`. The user is told to finish the provider and offered a model picker. | ✅ `f2b-console-after-skip` |
| FR-06 | S4 | FT | Footer advertises `Enter send` while the composer is locked behind the setup card. | ✅ `f2b-console-after-skip` |
| FR-07 | S2 | FT-NT | Console Settings modal renders an **empty provider name**: `Provider blocked: '' is not available in Console yet. Choose a supported provider.` Broken interpolation when no provider is selected. | ✅ `f3c-enter-no-provider` |
| FR-08 | S3 | FT-NT | The blocking setup card + **animated snow backdrop hides the entire workbench** — the user can't preview what the Console even is before committing to setup, and the card offers no "look around first" (only the top tab bar escapes). Full-screen motion behind a blocking modal is also a vestibular-accessibility consideration. | ✅ `f2b-console-after-skip` |
| FR-09 | S4 | FT-T | Typing while the composer is locked is **silently swallowed** — letters produce zero visible feedback (Enter does open Settings via the focused card action, which is good). | ✅ `f3b-typed-no-provider` |
| FR-10 | S3 | FT-NT | Setup card step 1 label falls back to provider-jargon variants (`Connect a provider (API key or local server)`, `Save the provider endpoint`) — "provider"/"endpoint" are never explained in-card. The wizard explains these better; the card doesn't. | 📄 `console_onboarding_state.py:18-24` |

### 2. Layout & visual hierarchy

| ID | Sev | Personas | Finding | Evidence |
|----|-----|----------|---------|----------|
| LY-01 | S2 | all | **Left rail content overflows at all tested widths**: `Workspace Defau…`, `RAG S…` clipped at rail edge, conversation search row shows `Sear`/`S` colliding with `Clear`. | ✅ `p1`, `p2c`, `p7b`, `p4-narrow-110x32` |
| LY-02 | S2 | all | **Tab strip clips the `Temporary` button** (`Temporar`) — the Inspector handle crowds the strip; no space budget across tab strip + handle. | ✅ `p1`, `p2c`, `p7b` |
| LY-03 | S3 | all | **Status chip strip overflows**: at 160 the cost chip renders `~$0` jammed at the edge; at 140 chips cut at `Sources: 0 stage`; at 110 `Approvals` is gone entirely. No wrap, scroll, or "more" affordance. | ✅ `p1`, `p2c`, `p4-narrow-110x32` |
| LY-04 | S3 | FT-NT, PU-NT | **Stacked empty-state noise**: `Local stars unavailable`, `Starred ▾ No starred conversations.`, `Workspaces ▾ No workspace conversations.` — three empty groups announcing themselves before any intent. | ✅ `p1` |
| LY-05 | S3 | all | **Duplicate "Chats" concept**: rail shows `{session title} - Chats` header *and* a `Chats ▾` group in the same scroll area — two list metaphors for one thing. | ✅ `p1`, `p2c` |
| LY-06 | S4 | FT-NT | Session section shows gating copy `Add another workspace before switching.` before the user has expressed intent — reads like an error; the `Switch` button it justifies is simultaneously rendered disabled/hidden. | ✅ `p1`, `p2c` |
| LY-07 | S2 | all | Rail Session rows cram label+value with no spacing: `WorkspaceDefau…`, buttons clipped to `Switc`, `RA` — the rail's 3fr/min-24 width is below its content's intrinsic width. | ✅ `p2c`, `p7b` |
| LY-08 | S1 | all | **At 80×24 the transcript vanishes** — the workspace grid renders only the left rail, full width; no tab strip, no transcript, no Inspector handle. The min-width contract (rail 24 + main 56 + handles) silently breaks below ~100 cols with no fallback (e.g. rail auto-collapse or a single-pane mode). | ✅ `p4-narrow-80x24` |
| LY-09 | S2 | all | **At 60×18 the screen is an empty frame**: bordered blank region where the transcript should be; `Ready — type a message to begin.` never renders; control bar clips (`Run Library`). | ✅ `p4-narrow-60x18` |
| LY-10 | S3 | all | **Header disappears below 35 rows** (`-console-compact`) including the `Ready/Running/Blocked` badge — the only persistent status identity is lost exactly when space is tight; nothing replaces it. | ✅ `p4-narrow-80x24`, `p4-narrow-60x18`; 📄 `chat_screen.py:21068-21094` |
| LY-11 | S1 | PU-T, PU-NT | **Inspector unreachable below 150 cols** — force-collapsed by width rule; clicking the handle at 140 toggles the stored preference (`right_open=true` persisted) with **zero visual feedback**; the auto-open rule only fires in a narrow 118–128-col band with specific content. Staged Sources/scope/run inspector/settings summary have no alternate surface at these widths (Sources chip is inert — DS-06). | ✅ `p7b-140-after-inspector-click` + sandbox config; 📄 `console_rail_state.py:17,491-498`, `chat_screen.py:12133-12181` |
| LY-12 | S4 | all | Transcript messages bottom-anchor with a large void above on a fresh session at 48 rows — reads as "broken/empty" rather than "new chat". | ✅ `p2c-reply-complete` |
| LY-13 | S4 | PU-NT | `Chat Context` viewer opens as a **near-fullscreen modal containing one line** (`No conversation context.`) — modal size doesn't match content; empty state has no guidance. | ✅ `p3d-context-viewer` |

### 3. Terminology & jargon

| ID | Sev | Personas | Finding | Evidence |
|----|-----|----------|---------|----------|
| TX-01 | S2 | FT-NT, PU-NT | `Transcript / Event Stream` — "Event Stream" is developer vocabulary in the primary pane title (now also `… | {session title}`). | 📄 `console_session_surface.py:40` ✅ p1 |
| TX-02 | S2 | FT-NT, PU-NT | `Library RAG` control-bar button + `RAG: off` chip — "RAG" never expanded anywhere. | 📄 `console_control_bar.py:34-55` ✅ p1 |
| TX-03 | S3 | FT-NT, PU-NT | `Scope: empty` chip, math-notation tooltips (`conversation A ∩ workspace B → N`), raw cause token `scope_empty`. | 📄 `console_status_chips.py:477-487`, `rag_scope.py:34,54` |
| TX-04 | S3 | FT-NT, PU-NT | `Tools: not loaded` exposes a lazy-loading implementation detail to everyone (loads on first send). | 📄 `console_display_state.py:113` ✅ p1 |
| TX-05 | S3 | FT-NT | `/prefill` = "Arm, pin, or clear a response prefill" — "Arm"/"prefill" jargon; commands missing from the description dict show **empty descriptions**. | 📄 `console_command_suggestions.py:30-37,104` ✅ f4 |
| TX-06 | S4 | FT-NT, PU-NT | `System Prompt` chip is a bare noun when unset vs `System Prompt: set` when set — breaks the `name: value` grammar of sibling chips. | 📄 `console_display_state.py:346-347` ✅ p1 |
| TX-07 | S4 | PU-NT | `Save Chatbook`, `Sources: N staged`, `STT…`, `Inspector`, `Temporary` — product vocabulary the UI never teaches in place. | 📄 multiple ✅ p1 |

### 4. Consistency

| ID | Sev | Personas | Finding | Evidence |
|----|-----|----------|---------|----------|
| CN-01 | S3 | all | **One feature, three names**: `Mic` (button), `Dictate` (tooltip), `STT…` (busy label). | 📄 `console_composer_bar.py:334-336,1252-1268` |
| CN-02 | S3 | all | **"Temporary" worded 3 ways** (tab `Temporary`, chip `Temporary — not saved`, tooltip `Temporary — not saved locally`); `◌` means both "temporary session" and "voice working". | 📄 `console_session_surface.py:235-260`, `console_ephemeral.py:62-68`, `console_composer_bar.py:1308` |
| CN-03 | S3 | FT-NT | **`Attach` in two places, two meanings**: control bar ("Stage Library or workspace context") vs composer ☰ menu (file attachment). Same label, different action. | 📄 `console_control_bar.py`, `console_composer_menu_modal.py` |
| CN-04 | S4 | all | Attachment clear tooltip: `Remove the pending attachment.` (compose) vs `Clear the attachment.` (live sync). | 📄 `console_composer_bar.py:3977,3676-3679` |
| CN-05 | S4 | PU-T | Glyph collisions: `◌` = temporary AND in-progress; `●` = recording AND agent-running — same shapes, different meanings, adjacent regions. | 📄 `console_glyphs.py:28`, `console_composer_bar.py:285` |
| CN-06 | S4 | PU-T | App tab labeled `Roleplay` but its nav id is `nav-personas`; palette says "Switch to Console", tab says "Console", guide says "Console", wizard exit says "Start chatting" — identity drift across chrome layers. | ✅ `p5-focus-tour.txt`; 📄 app.py |

### 5. Discoverability & affordances

| ID | Sev | Personas | Finding | Evidence |
|----|-----|----------|---------|----------|
| DS-01 | S2 | FT-NT, PU-NT | **Icon-only controls throughout** (tooltip-only identity): `☰` composer menu, `✕` clear-attachment, per-tab `✕`, rail chevrons `▾▸`, glyph message actions `👍👎🗑♻`. Non-technical users must hover to learn the UI; in terminals hover tooltips are easy to miss. | 📄 multiple ✅ p1 |
| DS-02 | S3 | PU-T, PU-NT | ☰ menu tooltip under-sells: `More composer actions (image, caption, impersonate).` vs the actual 8 entries (Prompts, Attach, Save Chatbook, Generate Image…). | 📄 `console_composer_bar.py:3820-3826`, `console_composer_menu_modal.py:96-146` |
| DS-03 | S2 | all | **Visible non-functional menu entry**: `Narrate Entire Conversation` — `Per-speaker voices (not implemented yet)`. Shipping dead UI erodes trust in every other entry. | 📄 `console_composer_menu_modal.py:139` |
| DS-04 | S4 | PU-T | Middle-click closes a session tab — never surfaced anywhere. | 📄 `console_session_surface.py:88-121` |
| DS-05 | S3 | PU-T | "Jump to latest" pill is click-only (Static + on_click), not keyboard-focusable. | 📄 `console_transcript.py:539-572` |
| DS-06 | S3 | PU-T, PU-NT | `Sources` and `Tools` chips are **focusable but inert** — they take Tab focus and activation does nothing. Doubly harmful combined with LY-11 (Sources has no other surface <150 cols). | 📄 `console_status_chips.py:355-364` |
| DS-07 | S4 | PU-T | Idle `Stop` tooltip (`No active run to stop in this tab.`) unreachable — button is `display:none` when idle. | 📄 `console_composer_bar.py:1209-1217` |
| DS-08 | S4 | PU-T | Session-switcher result rows are center-aligned with no in-modal key hints (`F2 rename`, `Ctrl+Enter open in new tab` exist but are documented only in F1). | ✅ `p3b-session-switcher`; 📄 help groups |
| DS-09 | S4 | PU-T | Slash popup renders as unframed plain rows that **cover the status chip strip** while open — functional, but it visually wipes the chips exactly when a user is composing a command. | ✅ `f4-slash-popup` |

### 6. Feedback & status communication

| ID | Sev | Personas | Finding | Evidence |
|----|-----|----------|---------|----------|
| FB-01 | S3 | all | **Raw status tokens in message content**: assistant rows append literal `[streaming]`/`[stopped]`/`[failed]`; a failed row with no partial content renders as bare `Assistant [failed]`. | 📄 `console_transcript.py:119-127` ✅ p6 |
| FB-02 | S4 | all | Message-action guide contains literal ASCII art: `··· ---> Continue ···`. | 📄 `console_transcript.py:65-68` |
| FB-03 | S4 | PU-NT | Selected-message action buttons are glyph-only (`♻👍👎🗑`) — the guide line is the sole legend, mixing keys, glyphs, and ASCII. | 📄 `console_transcript.py:69-88,1784-1785` |
| FB-04 | S4 | all | Retry button for failed responses is labeled **`Try`** (tooltip says "Retry") — a weak, ambiguous verb next to `[failed]`. | 📄 `console_message_actions.py:88,222-235` |
| FB-05 | S3 | all | **No toast on stream failure** — feedback is confined to a transcript system row + run-state copy (whose mode-bar surface is hidden); the header badge stays `Ready`; the collapsed Inspector handle gains a `failed` badge. A user composing their next message sees no ambient failure signal. | ✅ `p6-send-failure`; 📄 controller 6433-6453 |
| FB-06 | S2 | all | **Exception class names leak into user copy**: `Provider stream failed: RuntimeError error (…)` — the `{ExceptionName} error` fallback is user-visible. | ✅ `p6-send-failure`; 📄 `provider_failures.py:56-107` |
| FB-07 | S3 | all | **Success is silent**: 182 `notify()` calls in the screen — 92 warning, 40 error, 29 information, **1 success**. Sends, saves, retries, and recovery completions give no positive confirmation. | 📄 chat_screen.py notify inventory |
| FB-08 | S4 | PU-T | The run-state copy surface `#console-mode-bar` (`{mode} | Run: {status}`) is mounted **hidden** as a compat static — run feedback has no persistent on-screen home between the header badge and the transcript. | 📄 `chat_screen.py:14452-14457,16823-16843` |
| FB-09 | S4 | PU-T | Background catalog merge logs a traceback (`merge_saved_and_discovered_models`) when Alt+M opens the model popover with an empty local catalog — popover still renders, but the log noise suggests an unhandled edge that could surface differently in production. | ✅ UAT log 17:01:42 |

### 7. Accessibility & terminal robustness

| ID | Sev | Personas | Finding | Evidence |
|----|-----|----------|---------|----------|
| AC-01 | S3 | all | **Glyph rendering risk**: `◌` (U+25CC) commonly missing/narrow in terminal fonts; geometric glyphs (`▦⤵▾▸◂▌`) font-dependent; emoji (`📎🖼👍👎🗑♻`) double-width and color-emoji dependent — alignment/meaning degrade on minimal terminals. | 📄 `console_glyphs.py` + widget inventory |
| AC-02 | S2 | PU-T | **Tab order crosses all 15 app-nav buttons** between the composer cluster and the Console control bar; ~30 stops without reaching transcript/chips/Inspector. F6 mitigates, but the default order contradicts the visual layout and traps keyboard users in app chrome. | ✅ `p5-focus-tour.txt` |
| AC-03 | S3 | PU-T | **Alt-chord bindings break on default macOS terminals** (Option-as-Meta off): Alt+M/Alt+W/Alt+V/Alt+1–9 type composed characters instead. The code already hit this for Alt+H (moved to Ctrl+Shift+H) — the same caveat applies to six remaining Alt bindings; the only fallback is the palette. | 📄 `chat_screen.py:1738-1744,1781-1790`; guide "Alt chords" quirk |
| AC-04 | S3 | FT-NT, PU-NT | **Motion**: full-screen animated snow behind the blocking setup card, plus 87 splash effects by default — no `reduce-motion` path was observed in the review (config offers enable/disable, not reduced-motion). | ✅ `f2b`; 📄 `console_setup_modal.py` backdrop |

### 8. App-level navigation bleeding into Console

| ID | Sev | Personas | Finding | Evidence |
|----|-----|----------|---------|----------|
| NV-01 | S3 | all | **Top tab bar overflows/truncates at ≤140 cols**: `Settings` renders as `Set`, `Lab`/`Logs`/`Settings` have no digit shortcuts, and 13 destinations compete for one row. The Console inherits this chrome on every visit. | ✅ `f2b`, `p7b`, `p4-narrow-80x24` |
| NV-02 | S4 | FT | First-run routing lands users on **Home** beneath the wizard (deliberate per TASK-1508), and after Skip they stay on Home — the Console's own onboarding (setup card) is only discovered by choosing tab `2 Console`. Reasonable, but the Home card's `Start a conversation` button (visible when ready) vs `Set up Console model` (when blocked) swap labels by state — the same slot teaches two different verbs. | ✅ `f2a-home-after-skip`, `p2a` |

### 9. Intent-vs-implementation drift

| ID | Sev | Personas | Finding | Evidence |
|----|-----|----------|---------|----------|
| DR-01 | S4 | PU-T | Model popover binding: 2026-07-02 spec says `Ctrl+M`; implementation and user guide say `Alt+M`. **Implementation is self-consistent — the spec is stale.** Update the spec to avoid future regressions. | 📄 `chat_screen.py:1738` vs spec |
| DR-02 | S4 | all | Rail IA drift: dual-audience spec fixed `Session/Context/Model/Details`; live rail is `Session/Model/Agent/Details/(Character)` with sources in the Inspector. The live IA matches the guide — the old spec is stale. | 📄 compose 14582-14895 vs spec |
| DR-03 | S3 | PU-T | The persistent-rails spec says **"No auto-open, ever"**; the implementation auto-opens the Inspector in the 118–128-col band when specific rows exist, and on pending live-work launches. Either the spec or the code should move. | 📄 `chat_screen.py:12133-12181` vs spec non-goals |

---

## What works well (keep doing)

- **Setup wizard copy** — plain language, everything skippable, "changed later in Settings", live-but-never-blocking probe, "Couldn't verify — you can save anyway." Genuinely good FT-NT writing. (`f1`)
- **Draft safety** — every blocked/refused/crashed send path restores the draft; failed assistant messages keep partial content and are retryable. (`agent-C`, `p6`)
- **Recovery pairing** — provider/setup problems almost always pair message + action (card button, Settings deep-link with provider/model/field context). (`f3c`)
- **First-run routing** — wizard over Home (not over the Console's own card) avoids the double-onboarding trap; Home's `Set up Console model` card is a clear next step. (`f1`, `f2a`)
- **Keyboard layer intent** — composer-as-home-base, F6 pane cycle, Esc-returns-to-composer, full shortcut list in F1, every action in the palette. The skeleton is right; the Tab order (AC-02) betrays it. (`p3a`, `p5`)
- **Session auto-titling** from the first message; tab-state glyphs with decoding tooltips. (`p2c`)
- **Toast placement moved top-right** after earlier UAT found bottom-right toasts swallowing the Send cluster — evidence the feedback loop works. (`chat_screen.py:1699-1706`)
- **Compact header intent** — one-line identity header with pinned status badge at wide widths is clean and matches spec. (`p1`)

## Priority recommendations

- **P0 (this quarter)**: LY-08/LY-09 responsive fallback (auto-collapse the left rail below ~100 cols; single-pane mode below ~80), LY-11 Inspector access <150 cols (open as overlay or drop the force-collapse; make the inert Sources chip open a modal), FR-04 real disabled state + persistent reason, FR-07 empty-provider template, FR-05 card step/action sync.
- **P1**: AC-02 Tab order (scope tab cycles to the active pane; F6 between panes), TX jargon pass on chips/titles, DS-03 remove the dead Narrate entry, FB-05 toast on stream failure + FB-06 sanitize exception names, LY-01/LY-07 rail width budget (min 30 or content-aware), FR-01 wizard exit labeling (`Keep setup for later` vs `Don't ask again`).
- **P2**: DS-01 text labels on critical icon-only controls, FB-07 success confirmations for send/save, LY-03 chip overflow strategy, CN naming unification (Mic/Dictate/STT; Temporary), AC-03 non-Alt alternates documented in F1, AC-04 reduced-motion option, DR-03 spec/code reconciliation.

## Process appendix — sandboxing incident (resolved)

During early UAT runs the test-app factory's config/DB patches expired after app init, and runtime code (wizard mount/skip persistence, section-default writes) **rewrote sections of the real `~/.config/tldw_cli/config.toml`** (`chat_defaults` → openai/gpt-4o, `splash_screen.duration` → 7.0, `model_catalog` flips, `first_run.setup_completed=true`). All four were restored surgically to the Aug 3 13:38 backup values (verified by diff); later hunks (`[app_tts]`, `[file_notes]`, `[logs]`, the `audio_cpp` api_key containing pasted prompt text) predate/postdate the user's own handsfree-TTS work and were left untouched. The harness now runs fully sandboxed (`TLDW_CONFIG_PATH` + `HOME`/`XDG` redirects into `output/ux-review-console/sandbox/`); all findings after f1/p1 come from the sandboxed run.

## Evidence index

- Key captures (16 files, checked in alongside this doc): `Docs/superpowers/qa/console-ux-review-2026-08/captures/`.
- Full capture set (SVG + text dumps + PNG): `output/ux-review-console/captures/` — `f1`, `f2a/f2b`, `f3a–f3c`, `f4`, `f5`, `p1`, `p2a–p2c`, `p3a–p3d`, `p4-narrow-{140x42,110x32,80x24,60x18}`, `p5-focus-tour.txt`, `p6`, `p7a/p7b`.
- Harness: `output/ux-review-console/uat_console.py` (12 scenarios, sandboxed; re-runnable; copy checked in beside this doc).
