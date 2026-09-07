# Roleplay / Character-Card / Personas — UX & HCI Expert Review

- **Tasks filed:** TASK-425 – TASK-445 (2026-07-21; originally drafted as 412–432, renumbered after ID-range collision with dev tip and two in-flight worktrees)
- **Date:** 2026-07-21
- **Build:** origin/dev tip `dc196563f` (worktree, live-driven), fresh profile (`TLDW_CONFIG_PATH` scratch, empty DB) = authentic first-run
- **Provider:** local llama-server on :9099 (gemma-4-26B gguf), configured through the app's own guided setup
- **Method:** full new-user journey driven in the real TUI (tmux + SGR clicks), cross-checked against a code map of the surface (subagent exploration of `UI/Screens/personas_screen.py`, `UI/Persona_Modules/`, `Widgets/Persona_Widgets/`, `Character_Chat/Character_Chat_Lib.py`, `UI/Screens/chat_screen.py`, `config.py`)
- **Test asset:** authored Character Card v2 (`chara_card_v2`) with greeting, alt greeting, tags, embedded avatar, and an embedded `character_book` lorebook; imported as PNG-with-`chara`-chunk. Corrupt JSON used for error-path testing.

## TL;DR

The workbench authoring layer (import fidelity, card view, editor validation, preview test loop, attach seams) is in good shape. **The journey fails at both ends**: a new user who completes the app's own onboarding cannot get a character reply (P0-1), and any user who follows the app's "Start Chat"/"Open in Console" path ends with the model asking them to paste their character sheet manually (P0-2). Between those ends, card import silently drops embedded lorebooks (P1-3) and the file picker is the highest-friction import surface I have driven in this app (P1-4).

The full journey as observed, in order: first-run "Get started" card → guided provider setup (works, two trust bugs) → Personas tab → blank center pane → select seeded character → collapsed preview → **"anthropic is not ready"** despite green setup → no UI remedy exists → (config hand-edit) → import PNG card (picker fight, then full fidelity minus lorebook) → preview chat works, in character → "Open in Console" → blank transcript, staged invisible context, prefilled meta-instruction → send → **agent harness spawns a sub-agent → "Please provide the previous conversation and the character details you'd like me to use"**.

---

## P0 — journey-breaking

### P0-1. Character chat cannot use the provider the app's own onboarding configures — and there is no UI remedy
**Observed live.** Completed the first-run "Get started" flow: Settings → Providers & Models, llama.cpp selected, endpoint/model saved, provider test green ("endpoint reachable (1 model)"), readiness green. Then in Roleplay preview, first send fails:

> `anthropic is not ready: Missing API key. Set ANTHROPIC_API_KEY or add api_key under [api_settings.anthropic].`

Cause: the preview gateway hardwires `config["character_defaults"]` (`personas_preview_controller.py:270-276`), which ships as `provider="Anthropic", model="claude-3-haiku-20240307"` (`config.py:2762-2770`). The guided setup writes only `[chat_defaults]` + `[api_settings.*]` — verified by reading the config it produced. There is **no UI to change the character provider**:
- Preview pane has no provider/model picker or even a readout.
- Settings → Domain Defaults → Personas is a read-only "ownership contract" page ("Writes allowed: No — destination ownership must be implemented before mutation").
- Only escape hatch is Expert → Advanced Config raw TOML, and no error/message anywhere names `[character_defaults]`.

Compounding: the error steers the user to configure *Anthropic* (i.e., go get an API key) rather than to their already-working provider. The workbench header shows **"Ready"** and the inspector **"Console ready"** throughout.

**Fix direction:** resolve character provider like Console does (fall back to `chat_defaults` when `character_defaults` is unset — one-line semantic change that dissolves the trap); surface provider/model in the preview pane with a Settings deep-link; make the error name the actual remedy.

### P0-2. "Start Chat" / "Open in Console" loses the character — the model asks the user to re-describe it
**Observed live.** From a working preview conversation with the imported character:
- "Open in Console" → Console with **blank transcript** ("Ready — type a message to begin"), composer prefilled `Continue this conversation in character.`, toast "Preview conversation staged in Console". The only trace of the character is "Live work: Personas preview conversation" deep in the Inspector. Status strip says "Assistant: General".
- Sending the prefilled message routed through the **agent harness**: a `spawn_subagent` tool call whose task note itself says *"If no previous context is provided, please ask the user for the conversation and character details first."* Final assistant reply: **"Please provide the previous conversation and the character details you'd like me to use, so I can continue the conversation in character!"** The staged context never reached the model.
- The conversation persists named **"Continue this con..."** (from the meta-instruction).
- Inspector "Start Chat" behaves the same (prefills `Respond as Captain Elara Voss.`), and staged **into the existing dirty tab** rather than a fresh conversation.

Code anchors match: native Console has no tab container, so the character-session path is skipped and the handoff degrades to "staged live work" (`chat_screen.py:8476-8514`, `:8490-8497`); native sessions carry no `character_id` — `character_label` is free-text only (`:5795-5853`). The greeting only ever renders in the workbench preview.

**Fix direction:** Start Chat must create a *fresh, character-identified* conversation: greeting seeded as the first assistant message, session titled/labelled with the character, card system-prompt applied to the send path, plain provider send (no agent harness) by default. Until that exists, the buttons overpromise — consider relabelling ("Stage as context…") or gating them.

---

## P1 — major friction

### P1-3. Card import silently drops embedded lorebooks (`character_book`)
**Observed live.** Imported a v2 card containing `character_book` ("Second Chance Lore", keyed entry). Everything else survived (tags, alternate greeting shown with text, `Avatar: embedded`, all prose fields). But: character → "World Books (embedded copies): No world books attached", Lore mode → "No lore books yet". No warning, no toast. For the RP audience (SillyTavern-style cards), the lorebook is behavior-critical card content; the character then plays *without* its lore and the user has no idea. (`Character_Chat_Lib.py` import path; the P2f embedded-snapshot seam `extensions['character_world_books']` exists but import never populates it.)

**Fix:** on import, convert `character_book` → world book + auto-attach to the character, with a toast naming it; or at minimum warn "card contains a lorebook, not imported".

### P1-4. File-picker friction cluster (the only import route)
All observed live while importing a card:
- **Selection cursor nearly invisible** — bold+underline with bg `#1e1e1e` on `#141414`; in practice you cannot tell which row is selected.
- **Inconsistent activation:** single-click on a *directory* navigates immediately; a *file* needs select+Enter. Clicking ".." just to focus the list teleports you up a level. I (an expert driver) needed ~10 interactions to open a file.
- **Path bar (Ctrl+L) with a full file path** only navigates to the parent listing — it does not open/select the file (both Enter and Go).
- **Ctrl+L toggles**; when the bar is closed, keystrokes silently become list type-ahead (this landed me in /Applications).
- **Esc in the Recent overlay closes the whole picker** (flat dismissal hierarchy).
- **Reopens at last global location** (whatever screen used it last), not a stable, RP-relevant default; Recent tracks files only, so it stays empty until your first success.
- **Filter mismatches:** "Character Cards" includes `.md` (a docs folder lists every README as an importable card) but excludes `.webp`, which the importer supports (`personas_screen.py:3734` vs `Character_Chat_Lib.py:2067`); `.charx` unsupported everywhere; a filtered folder can render as just "📁 .." with no "N files hidden by filter" hint.

### P1-5. Provider test reports the wrong endpoint (trust bug)
**Observed live.** With an unsaved draft endpoint `:9099`, the test correctly *used the draft* (nothing listens on the displayed URL; "1 model" matches :9099) but its evidence line printed the stale saved value `api_settings.llama_cpp.api_url=http://localhost:8080/completion | status=ready`. The proof text contradicts what was tested. Also: "Test Provider" renders like a button but is inert; the test only runs via the `t` category hotkey, and only when focus is outside an input — nothing explains this.

### P1-6. llama.cpp's prefilled endpoint is broken for the actual caller
The Settings default/hint is `http://localhost:8080/completion`, but the chat caller always appends `v1/chat/completions` unless the URL already ends with it (`LLM_API_Calls_Local.py:213-222`) → `.../completion/v1/chat/completions`. A user who keeps the suggested default gets failures after a green-looking setup. (Code-verified.)

### P1-7. Workbench forgets everything on each Console round-trip
**Observed live.** Return from Console → "Selected: none", blank center, "Console blocked: select an item", preview collapsed. The design's own core loop (author → test → stage to Console → come back) resets working context every bounce.

---

## P2 — moderate

- **P2-8. Naming split:** nav says **"Personas"**, screen header says **"Roleplay"**, and a *mode* inside is also called **"Personas"** (who you are) — three overlapping meanings. A new user hunting "character chat" has to guess that the Personas tab is the RP hub.
- **P2-9. Characters-mode first-visit is a blank wall:** giant empty center pane with no empty-state copy (Lore/Personas library rails *do* have guidance lines). Combined with "Console blocked: select an item", the first impression is "broken", not "select or create a character".
- **P2-10. Preview chat presentation:** speakers labelled literally `character:` / `you:` (never the card/persona name); RP `*action*` asterisks rendered as raw text; no streaming (20s silent "Running" for a long reply); no alternate-greeting selector (the card view *shows* the alt greeting but you can't start from it); no provider/model readout.
- **P2-11. Readiness surfaces lie by omission:** header "Ready" + inspector "Console ready" while character replies are impossible (P0-1). "Ready" reflects internal wiring, not "you can chat". Also the first-run card's clipped copy: "Send your first message  Composer unlocks after" (sentence cut off — observed on first run).
- **P2-12. Persona model is stage-only:** no default/active persona; a persona is one-shot staged text for Console; the preview always calls you "User" even if you authored a persona; personas can't be imported (New only); no character↔persona pairing.
- **P2-13. Inspector actions are mode-blind:** Start Chat / Export PNG render in Dictionaries/Lore modes where they don't apply; Duplicate exists in Lore/Dictionaries rails but not Characters.
- **P2-14. Jargon on user surfaces:** "(embedded copies)", "Authority: Local", "Source: Local | Attachments: Console", and Settings pages exposing governance language ("Read-only contract", "destination ownership must be implemented before mutation").

## P3 — minor / polish

- Conversation auto-named "Continue this con..." after the meta-instruction (P0-2 side effect).
- Footer hint noise: "ctrl+s save unavailable | esc back unavailable" as persistent text.
- "1 characters" / "2 characters" count grammar.
- Model name must be hand-typed (50-char gguf) even though "Discover models from configured endpoint" exists right below — discovery result doesn't offer itself into the field as part of the flow (not exercised).
- Transient rendering artifact: a tall empty selection frame appeared under the selected library row once after first selection (disappeared after preview expand).

## What works well (keep)

- **First-run "Get started" card** with 3-step checklist and a deep-link into Settings — right idea, right place.
- **Settings three-pane** (category / detail / scope inspector) with per-field "Saved as / Validation / Runtime owner" guidance — genuinely excellent for experienced users; the test actually exercising the *draft* endpoint is correct behavior.
- **Import fidelity** (minus lorebook): PNG `chara` chunk, v1/v2 autodetect, name-dedup with honest "already existed; selected it", auto-select + reveal after import, corrupt-file import gives an honest failure toast.
- **In-workbench preview chat** as a DB-clean test loop — replies arrived fully in character (card system prompt + fields applied).
- **Editor validation** — inline "Validation errors: name: required" on empty-name save.
- **Keyboard model** — Ctrl+1..4 mode chips, `[`/`]` cycling, F6 pane traversal, ctrl+n/ctrl+f.
- **Attach seams** (character-level and conversation-level dictionaries/world books, try-it panes) — the power-user depth exists; it's the front door that's broken.

## Top 5 recommendations, in order

1. **Dissolve the provider trap** (P0-1): fall back `character_defaults` → `chat_defaults`; add provider/model readout + Settings link to the preview pane; fix the error copy.
2. **Make Start Chat real** (P0-2): fresh conversation, greeting as first assistant message, character-identified session, card system prompt on the send path, no agent harness for character chats, never stage into an unrelated open tab.
3. **Preserve lorebooks on import** (P1-3): convert + auto-attach with a toast; warn if skipped.
4. **Picker rescue pass** (P1-4): visible cursor, uniform activation (select-then-open for both dirs and files, or double-click both), file-path Go opens the file, Esc pops one layer, per-context remembered start dir, `.webp` in and `.md` out of the card filter, "N hidden by filter" line.
5. **Round-trip continuity + honest readiness** (P1-7/P2-11): preserve workbench selection across navigation; make "Ready" mean "a character reply will work", or say what isn't ready.

## Raw session log

The full interaction sequence (captures at every step) is in the session transcript; key artifacts: guided-setup captures, the `anthropic is not ready` failure, the sub-agent "please provide the character details" reply, and the lorebook-drop verification. Test card generator inline in transcript (PNG + JSON + corrupt JSON).
