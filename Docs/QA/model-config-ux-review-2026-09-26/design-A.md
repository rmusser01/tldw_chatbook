## name

Switchboard: an incremental redesign of model selection and configuration (one job per surface, 2-key A/B toggle)

## thesis

Keep the four surfaces. Give each one job and its own name, and stop every surface re-editing the same 16 fields at 3 rows per field.

1. **Alt+M becomes a real switcher.** It is a list-first picker across all providers, ordered PREVIOUS, then FAVOURITES, RECENT, ALL READY and NEEDS SETUP. Below the list sits one row of values: Temperature, Max tokens and Streaming. Enter applies to this chat. When it opens, the previous model is already highlighted, so the A/B toggle is Alt+M then Enter.
2. **The full "Conversation settings" modal becomes "Chat settings".** It handles the long tail for this chat: core fields first, Sampling folded with one-line help, and fields the provider doesn't support hidden behind a one-line summary.
3. **Settings ▸ Providers & Models is reordered.** The order becomes Connect, Key check, Default model for new chats (a real picker, not free text), Model defaults (showing effective values), then Advanced as one-row disclosures.
4. **Status is honest everywhere.** The possible states are READY ✓ verified, READY, REACHABLE, UNREACHABLE, NO KEY, REJECTED and NOT CHECKED. "Applies to" lines name the scope.

**Density comes from tokens, not a rewrite.** Controls are 1 row high. Widths are sized to the content. There is only one frame level. Rules and focus reach ≥3:1 contrast.

**Everything rides seams that already work:**
- Apply goes through the ADR-095 transaction unchanged: `live_committer` (console_model_popover.py:1396), `rebase_console_settings_draft` (console_chat_controller.py:12946), `remember_model_draft` for A→B→A drafts, and provenance copy (console_model_popover.py:389-398).
- The Settings Default model field reuses `ModelSearchPicker` (model_search_picker.py:60), including its discovery overlay (363-396).
- Recents come from the conversation snapshot ADR-095 already persists: `list_all_active_conversations` (ChaChaNotes_DB.py:11160) + `get_conversations_metadata_by_ids` (11306) + `parse_console_generation_settings` (console_generation_settings_metadata.py:240). No schema change, no new index.
- Key verification uses the authenticated models listing, which already maps 401/403 to "rejected credentials" (openai_compatible_model_discovery.py:726-737).
- The contrast fix follows the `ensure_readable_text_hues` precedent (css/Themes/themes.py:83-124).
- Settings F6 uses the shared `focus_relative_workbench_pane` (Widgets/workbench_focus.py:20).

**Kept:**
- Staged drafts: Settings s/r, and the A→B→A remembered drafts.
- Provenance tags: EDITED / inherited / "carried from …".
- Readiness rows, now labelled rows instead of a pipe dump.
- Esc safety: SafeModalDismissMixin, plus a new unsaved-edits prompt.
- Both rails: the Console left rail and the Settings category list.

**Two review findings needed re-reading against the code:**
- **"Saved defaults don't reach the open chat" (the P0).** ADR-095 says "Existing/open conversations do not rebase" (095:26). That is a missing "applies to" label, not a bug. This plan labels it and makes the switch 2 keys. Changing the behaviour instead would reverse ADR-095, and that is the owner's decision.
- **Test Provider.** The copy already says "provider acceptance has not been tested" (settings_screen.py:15249-15250), but it opens with "configuration is complete". The fix is to lead with what was actually checked, and to add a real key check.

## surface_map

CONSOLE (the live work surface)

**S1 · Alt+M "Switch model · <chat title>".** Same `ConsoleModelPopover` class and constructor seams; `compose()` is replaced.
- **Job:** pick the model for THIS chat, plus 3 core values.
- **Opened by:** Alt+M (chat_screen.py:1939), the model chip Enter/click (console_status_chips.py:75-97), `/model [query]`, the rail "Change" button, and the palette.
- **Rows** (1 row each): marker, favourite key, ★, provider, model, context, status word, last used.
  - PREVIOUS: the most recent model that isn't the current one. Preselected, so it is the A/B target.
  - FAVOURITES: Alt+1..9.
  - RECENT: last 50 chats, from ADR-095 metadata.
  - ALL READY PROVIDERS: top 3 per provider. Typing searches every provider's catalog.
  - NEEDS SETUP: listed but not selectable, with the reason and the fix (NO KEY → Ctrl+O; UNREACHABLE at host:port).
- **Current model:** marked "● CURRENT" in text (today it is not marked at all).
- **Provider and model change together** through `rebase_console_settings_draft`. That fixes "provider switch keeps the old model", and the provider Select goes away (popover 472-496).
- **Value row:** Temperature, Max tokens and Streaming as 1-row fields.
  - Max tokens is new here; today it is a read-only "Response max" Static (530-536).
  - Streaming becomes a Select; today it is a toggle Button (513-517).
  - Tab from a list row rebases the draft to that row first, so the values shown are that model's effective values with EDITED / inherited / "carried from" tags.
- **Keys:**
  - Enter: APPLY_TO_CHAT through the unchanged `live_committer`; focus goes back to the composer.
  - Ctrl+S: SAVE_MODEL_DEFAULT. Ctrl+N: MAKE_NEW_CHAT_DEFAULT. Both keep their ADR-095 semantics; the "Unavailable: …" block copy (1070-1083) moves to the hint row.
  - Ctrl+F: toggle favourite.
  - Ctrl+O: open Chat settings with the draft (the existing `ConsoleSettingsTransfer`).
- **Deleted from this surface:**
  - The "Defaults…" subview and both 2x2 grids of 3-row buttons (253-262, 592-670).
  - The context and compaction block (524-591). It already lives in Chat settings ▸ Context and memory and the status-row context meter.
  - The duplicate "Conversation settings" title (455).

**S2 · "Chat settings · <chat title>".** This is `ConsoleSettingsModal`, renamed from "Conversation settings" (1648).
- **Job:** everything else for THIS chat. Opened by Ctrl+O from the switcher, `/settings`, the rail "Configure…" and the palette.
- **Model view order:**
  1. MODEL summary row with [Change… Alt+M].
  2. CORE: Temperature, Max tokens, Streaming, plus Thinking/Reasoning only when supported.
  3. Sampling, folded. Top P, Min P, Top K, Seed, Presence and Frequency, each with one line of help.
  4. Connection, folded. It is provider-configuration owned per ADR-095.
  5. Request estimate.
  6. Your name in this chat.
- **Unsupported fields** are hidden with one summary line, reusing the Settings precedent (settings_screen.py:17293-17304) instead of the per-row "unknown" statics (2042-2046…).
- **Esc with edits** shows a new "unsaved" mode of the existing close guard (4041-4053: today it dismisses silently unless a reset or compaction is pending).
- **Ctrl+Enter** is printed on the Apply button (the binding already exists at 1141).
- The Context and memory view is unchanged apart from the density rules.

**S3 · Display surfaces.** Kept; only the copy changes.
- Status-row provider and model chips gain a readiness word.
- Left-rail Model section (left_rail.py:2263-2410): keeps Temperature and Max tokens, adds Streaming, and "Configure" becomes "Change  Alt+M".
- Inspector run recipe and transcript headers: unchanged.

**S4 · First-run card** (console_setup_modal.py).
- The snow field (ConsoleSetupBackdrop, 113) goes; the backdrop becomes a plain `$background` dim. The review lists it as an anti-reference.
- Keys: Enter = use the detected local server; c = cloud provider; n = notes.
- The cloud path carries a return intent to Settings (chat_screen.py:20546-20563 posts NavigateToScreen with no way back today). The Settings "Save and return to Console" button reuses `#settings-provider-return` (settings_screen.py:16989-16994).

**Entry points collapse to two verbs:**
- "Change model": Alt+M, the chip, `/model`, the rail "Change" button, the palette.
- "All chat settings": Ctrl+O, `/settings`, the rail "Configure…", the palette.
- F4 always means app-wide defaults, never "this chat".

SETTINGS (prepare and configure)

**S5 · Providers & Models.** Order today (settings_screen.py:16698-17437): Snapshots → Connect → readiness → context → discovery → 14 checkboxes → endpoints → Generation (collapsed) → prose. New order:

- **A · CONNECT.**
  - Provider picker (OptionList at 16749): CONFIGURED group first, legacy aliases folded under "Other names" in `build_provider_picker_groups` (settings_provider_view_model.py:190). No fixed-6-row clip.
  - API key and env var share one row.
  - Endpoint is shown only for URL providers.
  - "Key check" row with an honest tri-state; `t` runs it.
- **B · DEFAULT MODEL FOR NEW CHATS.**
  - `ModelSearchPicker(show_provenance=True)` replaces the free-text Input (16800-16808). Discovery merges in via `set_discovered_models`.
  - An "Applies to" line names the open chat's model.
- **C · MODEL DEFAULTS · provider/model.**
  - Expanded (today it is a collapsed Collapsible at 17203-17207), in a 2-up grid.
  - Shows effective values and provenance instead of placeholders.
  - One Streaming Select: Inherit / On / Off.
- **D · ADVANCED.** One-row disclosures:
  - Sampling
  - Context window (17033-17071)
  - Saved model list: the old discovery list plus "Save selected" (17072-17119). It stops posing as the way to set a default.
  - Custom endpoints (17200)
  - Catalog refresh: 14 checkboxes (17173-17195) become a 7-row table with On/Off words.
  - Prompt-cache snapshots: moved from the top (16698).
  - Reasoning replay override: renamed from "Override current Console model" (18036).
- **Moved out:** catalog and key-policy prose (17410-17437) goes to Inspector ▸ Config keys.

**S6 · Console Behavior ▸ Global fallback defaults** (18562-18705).
- Stays the owner of global fallbacks.
- Labels come from the shared generation-field table.
- Streaming Checkbox (18588) becomes the same Inherit/On/Off Select.
- The raw "chat_defaults.streaming is canonical…" line (18704) is deleted.
- Every "Console Defaults" string (1975, 5486, 17214, 17427) becomes "Console Behavior", the real rail label (1376).

**S7 · Settings Inspector.** Fixed at 36 columns (was fr-2, about 45 at 211, i.e. 21%). Sections: APPLIES TO, NEXT NEW CHAT WILL USE, WHERE EACH VALUE COMES FROM, KEY. Config keys sit in a disclosure.

**S8 · State bar.**
- When clean: "No unsaved changes" (today "Draft — save with s" shows even when clean, 9418-9419).
- Scope: "Applies to NEW chats · open chats keep their model (Console: Alt+M)" instead of "Shared with Console" (9448-9449).

WIZARD (FirstRunSetupWizard.py, 10k lines, low ROI): only the status placeholders change. They become Static lines instead of disabled RadioButtons (3836-3910), so "Authentication failed…" no longer draws as a radio option.

MERGED OR DELETED:
- Label drift ends with one shared `GENERATION_FIELD_COPY` table (label, unit, help), following the `STORAGE_FIELD_LABELS` precedent (settings_storage_defaults.py). All four editors use it:
  - "Think budget" / "Budget" → "Thinking budget"
  - "Endpoint" / "Base URL" → "Endpoint URL"
  - "When limit nears" / "Behavior" → "When context is nearly full"
  - "Budget strategy" / "Budget mode" → "Conversation budget"
- Streaming was a Select, a Checkbox and 2 toggle Buttons. It becomes one Inherit/On/Off Select, with On/Off only where there is nothing to inherit.
- Three titles called "Conversation settings" become three distinct nouns: Switch model / Chat settings / Settings ▸ Providers & Models.

## mockups

Every box below was produced by a script that checks each row is exactly the stated width: scratchpad/mock/{mk,console,modal,settings,firstrun}.py under /private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-chatbook/73fb7a69-fb3c-49ea-81ff-f711a75d6336/scratchpad/. Primary size is 211x44. At 235x52 the switcher grows to max-height 80% (about 30 list rows), the Settings detail pane gains 24 columns, and the Inspector stays at 36.

Glyph key:
- `▌value` — an editable field: a 1-column left edge on a 1-row control. Focus turns the edge thick, fills the focus background and makes the text bold.
- `▶` — the highlighted row. It is text, not only colour.
- `▸` / `▾` — a closed / open one-row disclosure.
- CAPS words — states.

=== POWER-USER + A/B PATH · Console + Alt+M switcher, just opened · 211x44 · switcher 124x24 centred at col 44, row 8 ===
 1| Home │▌Console▐│ Library  Study  Research  Lab F2  Logs F3  Settings F4  Research F5  Meetings F7   ·  Ctrl+P commands
 2| 1 Refactor plan ●  2 Release notes  3 Temp chat   +
 3|▾ Workspaces                      │ You  12:01
 4|  tldw_chatbook                   │   Can you outline the refactor for the provider card?
 5|▾ Conversations                   │
 6|  Refactor plan  ◂ active         │ Assistant · openai · gpt-4.1  12:01
 7|  Release notes                   │   Sure — here is a plan in four steps: …
 8|▾ Model                           │        ╭─ Switch model · Refactor plan ──────────────────────────────────────────────── now: openai · gpt-4.1 · T 0.7 · max 4096 ─╮
 9|  Temperature   0.7               │        │ Find ▌█                                                    ↑↓ move · Enter apply · 61 models in 4 ready providers        │
10|  Max tokens    4096              │        │ ── PREVIOUS · Enter switches back ───────────────────────────────────────────────────────────────────────────────────────│
11|  Streaming     On                │        │▶ Alt+1 ★ anthropic     claude-sonnet-4-5                     200k   READY ✓ verified   used 2 h ago in this chat         │
12|  System        default           │        │ ── FAVOURITES ───────────────────────────────────────────────────────────────────────────────────────────────────────────│
13|  [Change  Alt+M]                 │        │  Alt+2 ★ openrouter    anthropic/claude-sonnet-4.5           200k   READY              used yesterday                    │
14|▸ Agent                           │        │  Alt+3 ★ openai        gpt-4.1                               1M     READY ✓ verified   ● CURRENT                         │
15|▸ Details                         │        │ ── RECENT · from your last 50 chats ─────────────────────────────────────────────────────────────────────────────────────│
16|                                  │        │          anthropic     claude-3-7-sonnet-latest              200k   READY ✓ verified   used 3 d ago                      │
17|                                  │        │          local_ollama  qwen3:14b                             40k    REACHABLE · local  used 5 d ago                      │
18|                                  │        │ ── ALL READY PROVIDERS · top 3 each · type to search everything ─────────────────────────────────────────────────────────│
19|                                  │        │          openai        gpt-4.1-mini                          1M     READY ✓ verified                                     │
20|                                  │        │          openai        o4-mini                               200k   READY ✓ verified                                     │
21|                                  │        │          … 21 more openai models                                                                                         │
22|                                  │        │          local_ollama  llama3.2:3b                           128k   REACHABLE · local                                    │
23|                                  │        │          … 5 more local_ollama models                                                                                    │
24|                                  │        │ ── NEEDS SETUP · listed, not selectable ─────────────────────────────────────────────────────────────────────────────────│
25|                                  │        │          mistralai     NO KEY — Ctrl+O opens Settings ▸ Providers & Models                                               │
26|                                  │        │          llama_cpp     UNREACHABLE at 127.0.0.1:8080 — start the server; rechecked each time this opens                  │
27|                                  │        │ ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────│
28|                                  │        │ Temperature ▌0.7 ▏ current      Max tokens ▌4096  ▏ current       Streaming ▌On ▾ current                                │
29|                                  │        │ Enter apply to this chat · Tab edit values · Ctrl+F ★ · Ctrl+S save as model default · Ctrl+N default for new chats      │
30|                                  │        │ Ctrl+O full chat settings · Esc cancel                                                   Applies to: this chat only      │
31|                                  │        ╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
32-41| … rows 32-41: transcript continues underneath (dimmed) …
42|▌Message Refactor plan…  (Enter send · Shift+Enter newline · / commands)
43| openai · gpt-4.1 · READY ✓ verified │ T 0.7 · max 4096 · stream On │ Context 12.4k/1M │ Approvals 0
44| Alt+M model · Ctrl+K sessions · Alt+W workspace · Alt+I inspect · Ctrl+T new · F1 help
(Today the same popover is 172x32, 56% blank, with Temperature below the fold and no max tokens.)

=== POWER-USER PATH · after typing "son", Tab, 0.9, Tab, 8192 · 124 cols, height auto (12 rows) ===
 1|╭─ Switch model · Refactor plan ────────────────────────────────────── → anthropic · claude-sonnet-4-5 · T 0.9 · max 8192 ─╮
 2|│ Find ▌son█                                                 3 matches · 4 providers ready                                 │
 3|│ ── FAVOURITES ───────────────────────────────────────────────────────────────────────────────────────────────────────────│
 4|│▶ Alt+1 ★ anthropic     claude-sonnet-4-5                     200k   READY ✓ verified   used 2 h ago                      │
 5|│  Alt+2 ★ openrouter    anthropic/claude-sonnet-4.5           200k   READY              used yesterday                    │
 6|│ ── RECENT ───────────────────────────────────────────────────────────────────────────────────────────────────────────────│
 7|│          anthropic     claude-3-7-sonnet-latest              200k   READY ✓ verified   used 3 d ago                      │
 8|│ ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────│
 9|│ Temperature ▌0.9 ▏ EDITED (model default 1.0)   Max tokens ▌8192  ▏ EDITED (model default 4096)   Streaming ▌On ▾        │
10|│ Enter apply to this chat · Shift+Tab back to list · Ctrl+S save as model default · Ctrl+N default for new chats          │
11|│ Esc → “2 edited values: Enter apply · d discard · Esc keep editing”                         Applies to: this chat only   │
12|╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

=== Chat settings (was "Conversation settings", 195x48 with 3-row controls) · 150x18 centred on 211x44 ===
 1|╭─ Chat settings · Refactor plan ────────────────────────────────────────────────────────────────────────────────────────────────── 2 unsaved edits ─╮
 2|│ ▌Model and generation▐   Context and memory                                                                                                        │
 3|│ Applies to this chat only · saved with the conversation after its first message                                                                    │
 4|│── MODEL ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────│
 5|│ Model              anthropic · claude-sonnet-4-5   READY ✓ verified 12:04 · 200k context      [Change…  Alt+M]                                     │
 6|│── CORE ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────│
 7|│ Temperature        ▌0.9     EDITED · model default 1.0           Max tokens         ▌8192     EDITED · model default 4096                          │
 8|│ Streaming          ▌On ▾    inherited · Console Behavior         Thinking           ▌Off ▾    inherited · extended thinking                        │
 9|│ Thinking budget    ▌        used only when Thinking ≠ Off                                                                                          │
10|│ ▾ Sampling (advanced) · 2 of 2 supported fields inherit · hidden for Anthropic: Min P, Seed, Presence, Frequency                                   │
11|│   Top P            ▌        keep the smallest set of likely tokens whose probabilities add up to P (0–1). Leave blank to inherit.                  │
12|│   Top K            ▌        sample only from the K most likely tokens (whole number). Leave blank to inherit.                                      │
13|│ ▸ Connection · api.anthropic.com (provider config — edit in Settings ▸ Providers & Models) · key check · 1-token test                              │
14|│ ▸ Request estimate · 12.4k of 200k tokens (6%)                                                                                                     │
15|│ ▸ Your name in this chat · User (global default)                                                                                                   │
16|│ ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── │
17|│ Esc close (asks first: 2 unsaved edits)   Ctrl+S save as model default   Ctrl+N default for new chats   ▌Apply to this chat  Ctrl+Enter▐           │
18|╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

=== Esc with a dirty draft · new "unsaved" mode of #console-settings-close-guard · 72x5 ===
 1|╭─ Close Chat settings? ───────────────────────────────────────────────╮
 2|│ You edited 2 values (Temperature 1.0 → 0.9, Max tokens 4096 → 8192). │
 3|│                                                                      │
 4|│ ▌Apply to this chat  Enter▐   Discard  d    Keep editing  Esc        │
 5|╰──────────────────────────────────────────────────────────────────────╯

=== Settings ▸ Providers & Models · 211x44 · panes 32 | 143 | 36 ===
(The category-rail group names are illustrative; the rail itself is unchanged.)
 1| Home  Console  Library  Study  Research  Lab F2  Logs F3 │▌Settings F4▐│ Research F5  Meetings F7   ·  Ctrl+P commands
 2|╭─ Categories ─────────────────╮╭─ Providers & Models ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮╭─ Inspector ──────────────────────╮
 3|│ / search settings + fields   ││State: No unsaved changes │ Applies to NEW chats · open chats keep their model (Console: Alt+M)                                              ││APPLIES TO                        │
 4|│                              ││── CONNECT ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────││ New chats        yes             │
 5|│ SETUP                        ││Provider          ▌anthropic · Anthropic          ▾  CONFIGURED: anthropic, openai, local_ollama │ 24 more in list                           ││ Open chats       no → Alt+M      │
 6|│   Overview                   ││API key           ▌••••••••••••••••••••••sk-…9f2   or env var ▌ANTHROPIC_API_KEY          Source: config (saved)                             ││ Other devices    no (local)      │
 7|│▶  Providers & Models         ││Endpoint           api.anthropic.com (provider default — no URL needed)                                                                      ││                                  │
 8|│   Console Behavior           ││Key check          ✓ ACCEPTED — /v1/models listed 41 models · 12:04 today                     [Verify  t]                                    ││NEXT NEW CHAT WILL USE            │
 9|│   Speech & TTS               ││── DEFAULT MODEL FOR NEW CHATS ──────────────────────────────────────────────────────────────────────────────────────────────────────────────││ anthropic                        │
10|│   Library & RAG              ││Model             ▌claude-sonnet-4-5                        ▾  Served now · 200k context · in catalog                                        ││ claude-sonnet-4-5                │
11|│ DATA                         ││Applies to         New chats (Ctrl+T, temporary, workspace). The open chat keeps anthropic · claude-3-7-sonnet.                              ││ T 0.7 · max 8192 · stream On     │
12|│   Storage                    ││── MODEL DEFAULTS · anthropic/claude-sonnet-4-5 ─────────────────────────────────────────────────────────────────────────────────────────────││                                  │
13|│   Workspaces                 ││Temperature       ▌0.7     SET HERE  (global 1.0)          Max tokens        ▌8192     SET HERE  (global —)                                  ││WHERE EACH VALUE COMES FROM       │
14|│   Privacy & Security         ││Streaming         ▌Inherit ▾  → On  (Console Behavior)       Thinking          ▌Off ▾   inherits · extended thinking                         ││ Temperature  this model          │
15|│ APPEARANCE                   ││Think budget      ▌        only used when Thinking ≠ Off    Reasoning/Verbosity  hidden — not sent to Anthropic                              ││ Max tokens   this model          │
16|│   Theme                      ││▸ Sampling · Top P, Top K · both inherit · hidden for Anthropic: Min P, Seed, Presence, Frequency                                            ││ Streaming    Console Behavior    │
17|│   Splash screen              ││── ADVANCED (folded; each row opens in place) ───────────────────────────────────────────────────────────────────────────────────────────────││ Others       provider default    │
18|│ ADVANCED                     ││▸ Context window · 200,000 tokens (detected) · no override                                                                                   ││                                  │
19|│   Tool profiles · Agents     ││▸ Saved model list · 12 saved · 41 discovered, 29 not saved                                                                                  ││KEY                               │
20|│   Internal prompts           ││▸ Custom endpoints · none                                                                                                                    ││ accepted 12:04 (41 models)       │
21|│   Network · Diagnostics      ││▸ Catalog refresh · APPLIES IMMEDIATELY · on at startup, every 24 h · 7 providers                                                            ││                                  │
22|│   Advanced config            ││▸ Prompt-cache snapshots · off (llama.cpp only)                                                                                              ││▸ Config keys (for editing        │
23|│                              ││▸ Reasoning replay override · local models only                                                                                              ││  config.toml by hand)            │
24|│                              ││                                                                                                                                             ││                                  │
25|│                              ││▾ Catalog refresh  (shown expanded for illustration: a table with text state, never colour-only)                                             ││                                  │
26|│                              ││  Provider      Auto-refresh    Save new models to config                                                                                    ││                                  │
27|│                              ││  OpenAI        [x] On          [ ] Off                                                                                                      ││                                  │
28|│                              ││  Anthropic     [x] On          [x] On                                                                                                       ││                                  │
29|│                              ││  OpenRouter    [ ] Off         [ ] Off                                                                                                      ││                                  │
30|│                              ││  … 4 more     Refresh every ▌24   hours (0 = every launch)                                                                                  ││                                  │
31-42| … rows 31-42 EMPTY: the whole provider card now ends at row 30 even with one Advanced group open (today it is ~115 rows) …
43|╰──────────────────────────────╯╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯╰──────────────────────────────────╯
44| s save · r revert · t verify key · / find field · F6 next pane · Esc back      │ Unsaved edits stay when you switch category
Chrome is 5 of 44 rows (11%): nav, 2 pane borders, state bar, footer. Today it is 21-25%.

=== FIRST-RUN PATH (1/2) · Console "Get started" card · plain dim backdrop, no snow field · 100x13 ===
 1|╭─ Get started ────────────────────────────────────────────────────────────────────────────────────╮
 2|│ Chatbook needs one model connection before Console can send.                                     │
 3|│                                                                                                  │
 4|│ 1  Connect a provider     NOT DONE — no provider has a key or a reachable server                 │
 5|│ 2  Choose a model         WAITING on step 1                                                      │
 6|│ 3  Send a message         WAITING                                                                │
 7|│                                                                                                  │
 8|│ Found on this machine:   Ollama at 127.0.0.1:11434 · 7 models                                    │
 9|│                                                                                                  │
10|│ ▌Use Ollama (local)  Enter▐   Connect a cloud provider  c   Write notes without a model  n       │
11|│                                                                                                  │
12|│ Cloud setup opens Settings ▸ Providers & Models and brings you back here when the key is verified│
13|╰──────────────────────────────────────────────────────────────────────────────────────────────────╯

=== FIRST-RUN PATH (2/2) · Settings Connect strip in setup mode · detail pane 143 (141 inner) ===
 1|╭─ Providers & Models · setup ────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
 2|│State: SETUP — 3 steps │ Applies to new chats; you will return to Console when done                                                          │
 3|│── CONNECT ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────  │
 4|│Provider          ▌anth█                              anthropic · Anthropic   ← Enter picks the first match                                  │
 5|│API key           ▌(paste key)                        or env var ▌                          Stored in config.toml (local)                    │
 6|│Key check          NOT CHECKED — press Enter in the key field or t; lists models, costs nothing                                              │
 7|│── DEFAULT MODEL FOR NEW CHATS ────────────────────────────────────────────────────────────────────────────────────────────────────────────  │
 8|│Model             ▌                                    unlocks after the key is ACCEPTED                                                     │
 9|│                                                                                                                                             │
10|│                                                          ▌Save and return to Console  Enter▐   (appears when all 3 are done)                │
11|╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

## keystroke_budgets

**Counting rules.**
- Every key press or paste counts as 1 action. Typed text counts per character.
- Focus returns to the composer after Apply. `_restore_focus_after_dismissal` already exists at Widgets/modal_dismissal.py:88.
- Tabbing into a value field selects its contents (Textual Input `select_on_focus`), so typing replaces the value.
- "Today" numbers come from the live walkthroughs unless marked "est.".

**PERSONA 1 · FIRST-TIMER GETS A MODEL WORKING**

(a) Cloud key already in the clipboard. Today it is ≈30+ actions (est.):
- Setup card Enter lands in Settings with no return intent (chat_screen.py:20543-20563).
- Model is Tab stop #23 and is free text (settings_screen.py:16802).
- Test Provider reports "configuration is complete" even for a fake key.
- `s`, then navigate back to Console by hand.

Proposed: **10 actions, ending in the Console composer**.
1. c — Connect a cloud provider.
2. a n t h — filter the providers.
3. Enter — choose Anthropic; focus moves to the key field.
4. Paste the key.
5. Enter — runs the key check (models listing, no charge); on ACCEPTED, focus moves to Model.
6. Enter — accept the highlighted catalog model.
7. Enter — Save and return to Console.

(b) Local server already running (Ollama or llama.cpp). Today: the detected-server action exists (chat_screen.py:4837, 15397), then the full modal opens with focus_model (20537), then the model has to be picked in a 3-row form. Proposed: **2 actions**.
1. Enter — Use Ollama. The switcher opens filtered to local_ollama.
2. Enter — pick the highlighted model.

(c) Key typo. The key check shows "REJECTED (401) — check the key" instead of "configuration is complete". Recovery: Shift+Tab, fix the key, Enter = **3 + typing**.

**PERSONA 2 · POWER USER: switch to Anthropic Sonnet, temperature 0.9, max tokens 8192, back to typing**

Today: **45 actions across 2 modals.** The popover has no max tokens, so Full settings… is needed, and there Temperature and Max tokens sit inside the collapsed "Advanced generation" section with 3-row controls.

Proposed: **14 actions, one surface.**
1. Alt+M
2. s o n — the favourite row is highlighted.
3. Tab — rebases the draft to that model and focuses Temperature.
4. 0 . 9
5. Tab
6. 8 1 9 2
7. Enter — applies and returns to the composer.

Variants:
- Model only: Alt+M, s o n, Enter = **5**.
- By command: `/model son` Enter (11 characters). "son" matches 3 models, so the switcher opens pre-filtered; Enter = **12**. A unique query applies directly and posts "Switched to … · Undo: Alt+M Enter".
- Save as that model's default in the same pass: Ctrl+S instead of the final Enter = **14**.

**A/B TOGGLE BETWEEN TWO MODELS**
- Today: **16-20 keys**, no recents or favourites, and the current model is not marked.
- Proposed: **2 keys** — Alt+M (the PREVIOUS row is preselected), Enter. Pressing it again toggles back.
- A specific favourite: Alt+M, Alt+2 = **2**. Alt+digit is an accelerator only; on macOS it needs Option-as-Meta, and the Enter path always works.

**SECONDARY BUDGETS**

Change the default model for new chats in Settings:
- Today: F4, then Tab ×23, then free text, then s = ≈27+ (est.).
- Proposed: **13** — F4, /, m o d e l, Enter (field search lands on Model; DESIGN.md:287-290), s o n, Enter, Ctrl+S. Ctrl+S is new because bare `s` is inert while an Input has focus (settings_screen.py:2751-2754).

Other paths:
- Cancel the switcher with no edits: Esc = **1**.
- Cancel with edits: Esc, d = **2**, and you are told what you would lose. Today the full modal discards silently (console_settings_modal.py:4041-4053).
- Open every chat setting from the switcher: Ctrl+O = **1**, and the draft is carried over (the existing ConsoleSettingsTransfer).

## density_rules

All rules go in app-tier sheets (css/features/*.tcss, css/components/*.tcss) and are regenerated with build_css.py. Widget DEFAULT_CSS/BUNDLED_CSS loses to app-tier rules (lessons-textual.md:978-1001). Rules target a class on the subject itself, never an ancestor-scoped bare type (lessons-textual.md:810).

**D1 · One-row controls on all three surfaces.**
- Reuse the existing `.settings-compact-input` (components/_forms.tcss:454-477): height 1, border none, a 1-column left edge `$ds-control-edge`; on focus a thick `$ds-action-focus` edge, `$ds-focus-bg` fill and bold text.
- Make `.settings-compact-select` 1 row as well; today it is 3 rows with a solid border (_settings.tcss:600-610):
  ```
  .settings-compact-select { height:1; min-height:1; border:none; border-left:solid $ds-control-edge; padding:0; }
  .settings-compact-select:focus { border:none; border-left:thick $ds-action-focus; background:$ds-focus-bg; text-style:bold; outline:none; }
  .settings-compact-select > SelectCurrent { height:1; border:none; padding:0 1; }
  ```
  Also set `.settings-select-row` and its label to height 1 (_settings.tcss:470-475, 556-560).
- Chat settings: MODAL_CONTROL_HEIGHT 3 → 1 (console_settings_modal.py:168, applied at 1066-1071), and `.console-settings-modal-label` min-height 3 → 1 (_console_panels.tcss:65-74). Inputs and Selects get the compact classes.
- In-form buttons use compact=True.
- Primary button focus: `background:$primary-lighten-1; text-style:bold reverse; outline:none`. It must get lighter on focus, never darker.
- Every border removal pairs with `outline:none` on :focus, because the global `*:focus` outline paints over 1-row content (lessons-testing-evidence.md:7222-7240).

**D2 · Content-sized widths.**
- Label column is 18 cells; it is 23-24 today (_settings.tcss:546, console_settings_modal label w-23).
- Field width by value type, using the existing utilities w-10 and w-16, plus three new ones in css/utilities/_helpers.tcss:
  - numbers: `.w-10`
  - enum Selects: `.w-16`
  - env var names: `.w-32` (new)
  - model IDs and secrets: `.w-48` (new)
  - URLs: `.w-64` (new)
- The rest of the row is one muted Static for effect and provenance: `width:1fr; color:$ds-text-muted; text-wrap:nowrap; text-overflow:ellipsis`.
- No input is wider than 64 columns. Today inputs are 111 columns wide at 211, and up to 136 in the modal.

**D3 · Two-up grid for numbers and enums** when the pane is at least 120 inner columns (the Settings detail pane is about 141 at 211; Chat settings is 148):
```
.dense-grid { layout:grid; grid-size:2; grid-columns:1fr 1fr; grid-rows:1; grid-gutter:0 2; height:auto; }
```
Each cell is label 18 + field 10 + provenance ~36.

**D4 · A closed disclosure costs 1 row.** Today the app-wide Collapsible uses min-height 3, a round border and a bottom margin of 1 — about 5-6 rows closed (components/_widgets.tcss:13-18).
```
Collapsible.dense-disclosure { border:none; margin:0; padding:0; min-height:1; background:transparent; }
Collapsible.dense-disclosure > CollapsibleTitle { height:1; padding:0 1; }
Collapsible.dense-disclosure > Contents { padding:0 0 0 2; background:transparent; }
```
The title carries a summary, for example "▸ Sampling · Top P, Top K · both inherit".

**D5 · One frame level.**
- Only pane borders and modal borders remain.
- Remove the `.settings-focus-card` border (_settings.tcss:800-806) and the `.settings-instant-apply-group` border (689-694). Instant-apply scope becomes a CAPS word in the header ("APPLIES IMMEDIATELY").
- Section headers are 1 row: a bold `$ds-text-primary` label with a `─` fill in `$ds-grid-line`.
- No row is crossed by more than 2 vertical rule glyphs (today 10).

**D6 · Contrast floors (WCAG 1.4.11 non-text, 3:1), measured per theme.**
- Change `$ds-grid-line: $surface-lighten-1` (core/_variables.tcss:11), which measures 1.0-1.05:1, to `$border-blurred`. Do the same for `$ds-control-edge` (core/_variables.tcss:43).
- Add `ensure_visible_rules(theme)` next to `ensure_readable_text_hues` (css/Themes/themes.py:83-124). It pins the generated `border-blurred` to the first blend toward the text pole that reaches ≥3.0:1 against both surface and panel.
  - Only generated names can be overridden per theme, which is why it pins `border-blurred` (themes.py:1-18).
  - Pure-ANSI themes are skipped, as the precedent already does (themes.py:103-106).
  - Gate it in Tests/UI/test_theme_contrast.py.
- A highlighted list row or focused rail row gets `$ds-focus-bg` (≥3:1 shift from its resting background), bold text and a `▶` glyph. That covers the Settings category rail (1.1:1 today) and the Select highlight (1.12:1 today).

**D7 · Text carries state; colour only reinforces it.**
- Checkboxes render "[x] On" / "[ ] Off".
- The saved-model list prefixes rows with "[saved]" / "[ ]".
- State words are in caps: READY, REACHABLE, UNREACHABLE, NO KEY, REJECTED, NOT CHECKED, EDITED, SET HERE. "inherited" stays lower case as secondary text.
- Favourites show ★ and sit under a FAVOURITES header.

**D8 · Chrome budget at 211x44.**
- Settings: 5 of 44 rows (nav, 2 pane borders, state bar, footer). The pane title lives in the border title, not in an extra row.
- Switcher: 5 of its rows are chrome (2 borders, the values rule, 2 hint rows). Its height is auto, max 80%, so about 24 rows at 44 and up to about 41 at 52.

**D9 · Inspector** at 36 columns fixed when the viewport is ≥180 (`#settings-impact-pane { width:36 }`; today `$ds-fr-2`, _settings.tcss:61-67). Config keys sit only inside a dense-disclosure.

**D10 · Provider OptionList.** Today it is a fixed height of 6 and its focus border clips the first column (_settings.tcss:458-463). New rule: `#settings-provider-picker { height:auto; max-height:12; border:none; border-left:solid $ds-control-edge; padding:0; }`. Options carry a 2-column status gutter, so focus never overwrites text.

**D11 · Never a placeholder where a value belongs.** An empty, inherited field shows the effective value and its source in the provenance column ("inherits 1.0 · Console Behavior"). Placeholders only state the unit or range.

**D12 · Chat settings modal size.** Height is auto with max 90%; width is 150 at ≥180 columns. This replaces the 85%/196 tier (_console_panels.tcss:200-203), because content-sized fields no longer need the width.

## phasing

Every PR can ship on its own and updates the matching Docs/User_Guide page (CLAUDE.md "UI changes"). New keys are swept through all four layers: binding, copy strings, tests and Docs (lessons-textual.md:1004-1024). Each PR runs `./scripts/preflight.sh`, which checks CSS bundle sync. Ordered by user-visible gain per unit of risk.

**PR-1 · Honest copy and one field vocabulary** (S; copy only; no dependencies)
- New `Chat/generation_field_copy.py` holding GENERATION_FIELD_COPY (label, unit, one-line help), following STORAGE_FIELD_LABELS. Wired into the labels of the 4 editors: popover, modal 1962-2137, Settings 17218-17409, Console Behavior 18585-18700.
- Settings copy:
  - state-bar badge when clean (settings_screen.py:9418-9419)
  - scope "Applies to NEW chats · open chats keep their model (Console: Alt+M)" (9448-9449)
  - "Console Defaults" → "Console Behavior" (1975, 5486, 17214, 17427)
  - "Override current Console model" → "Reasoning replay override" (18036)
  - delete the raw config-key line (18704)
  - verdict leads with "Key present — not verified" (15248-15254)
  - readiness result as labelled rows, not a " | " dump (15325-15332)
- Titles: "Switch model" (popover 455) and "Chat settings" (modal 1648).
- Wizard status placeholders become Static lines (FirstRunSetupWizard.py:3836-3910).
- Tests: update the string pins.
- Visible win: consistent names everywhere, and no false "complete".

**PR-2 · Density tokens and navigation fixes** (M; mostly CSS; no dependencies; can run in parallel with PR-1)
- Implements D1, D4, D5, D6, D9 and D10: the compact Select at 1 row, `.dense-disclosure`, new width utilities, frame removal, the contrast pin, the Inspector at 36, and the provider list height.
- MODAL_CONTROL_HEIGHT → 1. Deliberately rewrite Tests/UI/test_console_session_settings.py:4555-4560, which pins the defect (asserts == 3).
- Settings F6 / Shift+F6 via the shared `focus_relative_workbench_pane` (Widgets/workbench_focus.py:20; same pattern as personas_screen.py:16333-16347). Add Ctrl+S alongside `s` (settings_screen.py:2747-2764).
- Tests:
  - theme-contrast gate for grid-line ≥3:1 across all shipped themes
  - paint probes for 1-row fields with real keypresses (not `.value`, per lessons-live-verification.md:1030-1043)
- Visible win: every Settings and Chat-settings screen loses about 60% of its rows. Borders and focus become visible.

**PR-3 · Alt+M switcher** (L; depends on PR-2)
- Replace `ConsoleModelPopover.compose()` with the list-first switcher, 1-row value row and keyed default actions.
- New controller `UI/Console_Modules/model_switcher.py`:
  - row assembly: PREVIOUS / RECENT / ALL READY / NEEDS SETUP
  - recents via list_all_active_conversations → get_conversations_metadata_by_ids → parse_console_generation_settings, in a worker
  - readiness via the existing `default_readiness_resolver`
  - Wired in UI/Console_Modules/wiring.py (DESIGN.md:486-498).
- chat_screen.py touches only the constructor call (5528-5548).
- The Apply path is unchanged: ConsoleSettingsSubmission(APPLY_TO_CHAT, QUICK_POPOVER) through `live_committer`; rebases go through `rebase_console_settings_draft`.
- QUICK_MODEL_DEFAULT_FIELDS gains max_tokens (console_settings_apply.py:19). This needs an ADR-095 amendment first (see risks).
- Keep the IDs console-popover-apply, -temperature, -streaming, -save-model-default and -make-new-chat-default on the new widgets; 17 test files query popover IDs.
- Visible win: the power-user loop drops from 45 to 14 actions, and A/B drops from 16-20 keys to 2.

**PR-4 · Favourites and `/model <query>`** (M; depends on PR-3)
- `[chat_defaults] favorite_models = [{provider, model}, …]`, written through ADR-095's locked reread-and-field-patch default path.
- Keys: Ctrl+F toggles a favourite; Alt+1..9 picks one.
- `/model` argument:
  - grammar hint "" → "[model]" (console_command_grammar.py:110)
  - `_console_command_run_action` (chat_screen.py:19362-19383) delegates the parsed argument to model_switcher.py
  - a unique match applies directly and posts a transcript notice with an undo hint; otherwise the switcher opens pre-filtered
- Rail Model section: "Change  Alt+M" and a Streaming row (left_rail.py:2263-2410).
- Visible win: favourites, a `/model` that honours its argument, and the rail teaching the key.

**PR-5 · Settings ▸ Providers & Models reorder** (L; depends on PR-1 and PR-2)
- Move composition into a new region widget, `UI/Settings_Modules/providers_models_card.py` (One Home Rule, DESIGN.md:359-364), in the order Connect / Default model / Model defaults (expanded, 2-up, effective values) / Advanced.
- Model Input → ModelSearchPicker(show_provenance=True) with discovery merged in. Keep a hidden `#settings-model-value` adapter, following the modal's legacy-adapter pattern (console_settings_modal.py:1810-1841); 14 test files query that ID.
- Streaming becomes one Select in both places (17399 and 18588).
- `build_provider_picker_groups` gets a CONFIGURED group and folds aliases.
- Auto-refresh becomes a table.
- Rewrite Tests/UI/test_settings_configuration_hub.py:4386-4420: Connect stays first, but model defaults are now expanded.
- Visible win: the card goes from about 115 rows to 30. Model moves from Tab stop #23 to #3, and Generation from about #48 to #5.

**PR-6 · Honest key check and readiness** (M; depends on PR-5 for placement)
- `t` / Verify runs the authenticated models listing for catalog cloud providers (the model_catalog_settings.py:13-21 set) and the existing live probe for URL providers.
- The result shows 4 labelled rows with a timestamp and is kept per provider in process memory.
- Readiness words across Console:
  - READY ✓ verified — key checked
  - READY — configured, not checked
  - REJECTED
  - UNREACHABLE
- The switcher runs a bounded, cached local reachability probe when it opens, reusing `_test_console_connection` (chat_screen.py:3427). This fixes the false Ready for unreachable local endpoints.
- Visible win: status can be trusted.

**PR-7 · Chat settings IA** (M; depends on PR-1 and PR-2)
- Core-first order.
- Sampling folded with help text; Connection folded.
- Unsupported fields hidden behind one summary line.
- Close-guard "unsaved" mode (console_settings_modal.py:4041-4053, guard at 2615-2634).
- Apply button labelled with Ctrl+Enter; Ctrl+S and Ctrl+N default actions.
- Visible win: the modal goes from 195x48 to 150x18, and no work is lost silently.

**PR-8 · First run: connect in place** (S-M; depends on PR-5 and PR-6)
- Return intent in the provider-recovery NavigateToScreen context (chat_screen.py:20546-20563).
- "Save and return to Console" reuses `#settings-provider-return`.
- Snow backdrop → plain dim (console_setup_modal.py:113).
- After "Use Ollama", the switcher opens filtered to that provider.
- Visible win: a first-timer needs 10 actions (cloud) or 2 (local), and comes back to Console.

## risks

1. **ADR decisions that belong to the owner — do not decide them in a PR.**
   - (a) The quick field mask {temperature, streaming} (095:74; console_settings_apply.py:19) gains max_tokens.
   - (b) Compaction editing leaves the quick surface (095:103). The switcher still submits the untouched sparse override via `_compaction_override_for` (console_model_popover.py:938-944), so persisted behaviour does not change.
   - (c) The review's P0, "saved defaults don't reach the open chat", is ADR-095 by design (095:26). This plan labels it; it does not rebase open chats. Changing that would reverse the ADR.

2. **Ratchet headroom.** chat_screen.py is 25,331 lines against a budget of 25,363 (Tests/Architecture/test_screen_size_ratchet.py:85), so there are 32 lines of headroom. All switcher, recents, favourites and `/model` logic must live in UI/Console_Modules/model_switcher.py plus wiring.py. Any growth in the screen fails CI.

3. **settings_screen.py is 32,167 lines.** The reorder must be a move into a region widget, not an in-place rewrite.
   - Keep every widget ID. Tests query #settings-model-value in 14 files, popover IDs in 17 and the modal in 16.
   - Two tests pin today's defects and must be rewritten deliberately:
     - test_console_session_settings.py:4555-4560 asserts MODAL_CONTROL_HEIGHT == 3.
     - test_settings_configuration_hub.py:4386-4420 asserts Generation defaults is collapsed.

4. **One-row control traps that are already on record.**
   - Removing a border switches on the global `*:focus` outline over content (lessons-testing-evidence.md:7222-7240).
   - A field without the compact class paints nothing in a 1-row row, yet `.value`-based tests pass (lessons-live-verification.md:1030-1043).
   - Widget CSS loses to app-tier CSS (lessons-textual.md:978-1001).
   - Stylesheet splitting can reorder precedence (lessons-testing-evidence.md:12665-12676).
   - Verify with real-stylesheet harnesses (lessons-textual.md:790) and real keypresses.

5. **The contrast pin is app-wide.** Retargeting `$ds-grid-line` to `$border-blurred` and pinning it per theme also raises Textual's default blurred Input border everywhere. That is intended, but expect broad visual churn. Measure it in a running terminal across the shipped themes (DESIGN.md:163-186 method); ANSI themes are skipped.

6. **Key conflicts.**
   - Textual Input owns ctrl+a/e/d/k/u/w/x/c/v (textual/widgets/_input.py:76-140), so the switcher uses only Ctrl+F/S/N/O plus Enter/Tab/Esc.
   - Alt on macOS types composed characters (chat_screen.py:1955-1959), so Alt+1..9 is only an accelerator.
   - Confirm that Ctrl+S reaches the app under the owner's terminal. Textual raw mode should disable XON/XOFF, but check it live.
   - A Select posts Changed on mount (lessons-textual.md:711). The switcher's Streaming Select needs the same mount-echo guard as the popover's provider Select (console_model_popover.py:356-361).

7. **Recents cost and correctness.**
   - Run in a worker, capped at 50 conversations, batched metadata read. No new SQL index, so the CLAUDE.md index plan-pin obligation is not triggered.
   - Malformed or future-version metadata fails closed (console_generation_settings_metadata.py:240).
   - Temporary, unpersisted chats contribute through open sessions only.

8. **Limits of the key check.**
   - A 200 from the models listing proves the key is accepted, not that the account can generate (quota, model entitlement). The copy must say "Key accepted (models listed)", never "ready to generate".
   - Only the OpenAI-compatible module's 401/403 mapping was traced (openai_compatible_model_discovery.py:726-737). Confirm the Anthropic listing path in the catalog-refresh service before PR-6 promises it for Anthropic.
   - The paid 1-token test stays opt-in, with the existing consent copy (console_settings_modal.py:193-195).

9. **UI-loop stalls.** The local reachability probe and the recents read must stay off the UI thread and be bounded and cached. The Fedora lag program found network and keyring calls on the loop.

10. **Config writes.** Favourites and default actions write config.toml. Tests and live verification must use a scratch TLDW_CONFIG_PATH; there are two recorded incidents of agents rewriting the user's real config. Use ADR-095's locked reread-and-patch, never whole-file writes.

11. **Discoverability of new keys.** Ctrl+F/S/N/O, Alt+1..9 and `/model [query]` must appear in the switcher hint rows, F1 help, the palette and Docs/User_Guide. The hint rows are the primary teacher, so they must stay visible at 44 rows.

12. **Scope creep in the wizard.** FirstRunSetupWizard.py is 10,462 lines. This plan touches only its status placeholders; resist folding the wizard into the Connect strip here.