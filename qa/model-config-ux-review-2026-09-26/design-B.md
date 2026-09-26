## name

Switch & Tune: two model surfaces

## thesis

The model job has two verbs, switch and tune, so it gets two surfaces and nothing else.

(1) SWITCH is a command-palette-style overlay on Alt+M. It is one fuzzy list of provider/model PAIRS across every provider, so you can never pick a provider without a model. Each row carries a readiness word, and the list ends in a one-row strip with the three params people actually change (Temp, Max tokens, Stream). The previous model is pre-highlighted, Alt-Tab style, so an A/B toggle is 2 keys.

(2) TUNE is a single ModelSettingsPanel widget. It is the SAME widget in both hosts: a right-side drawer in Console (scope defaults to "This chat") and the "Models" category in Settings (scope defaults to "<model> default" / "All models").
- Every row shows its effective value, a Source word saying which layer supplied it, and inline help.
- The scope bar maps 1:1 onto the typed actions that already exist: APPLY_TO_CHAT, SAVE_MODEL_DEFAULT and MAKE_NEW_CHAT_DEFAULT (tldw_chatbook/Chat/console_settings_apply.py:40-45). It also maps onto ADR-006's precedence chain: session > model profile > chat_defaults > provider (backlog/decisions/006-provider-aware-generation-settings.md:27).
- As a result, "where does this value come from" and "where will my edit go" are always on screen.

Everything else is deleted or folded into these two: the 1,432-line popover, the 7,807-line ConsoleSettingsModal, the ~115-row Providers & Models card, Console Behavior's duplicate fallbacks, the wizard's Provider/Model steps and the starfield "Get started" card.

Most of the plumbing already exists and is reused rather than rebuilt:
- staged drafts with A->B->A memory (console_settings_apply.py:94-113)
- field provenance (console_settings_apply.py:64-70)
- the honest readiness verdict (Chat/provider_test_evidence.py:17-40, 311-367), which the current surfaces bypass
- the provider param map as the single support table (Chat/Chat_Functions.py:232-299)
- the searchable catalog with provenance groups (Widgets/model_search_picker.py:37-42)
- the Ctrl+K switcher modal pattern (UI/Screens/chat_screen.py:5248-5260)

The redesign is mostly deletion plus two thin views.

## surface_map

AFTER THE REDESIGN: 2 editing surfaces (today: 4 plus the setup card) and 8 entry points (today: 11 or more). Every entry point is advertised in the footer or the palette.

SURFACE 1: MODEL SWITCHER (new Widgets/Console/console_model_switcher.py ModalScreen, plus controller UI/Console_Modules/model_switcher.py, per DESIGN.md:512-522 ratchet)
- It is one fuzzy list of (provider, model) rows across all providers. Choosing a provider without a model is impossible. Today the popover rebases to (provider, None) on a provider change (Widgets/Console/console_model_popover.py:1086-1101).
- Columns: model, provider, context size, readiness word, tags.
  - The readiness word comes from provider_readiness_verdict (Chat/provider_test_evidence.py:311-367): Verified / Reachable / Not tested / Needs key / Rejected / Offline. It replaces the config-only "Ready".
  - Tags: current / previous / pin / default for new / unlisted. The current model is finally marked.
- Groups:
  - RECENT: process MRU plus distinct provider/model from recent conversations' console_generation_settings metadata, so no new store is needed.
  - PINNED.
  - PROVIDERS: one row per provider, and Enter filters to that provider.
  - Legacy aliases (Chat/provider_catalog.py:58-68) are hidden unless currently configured.
- Filtering is in-memory only and capped at 20 results, like ModelSearchPicker.MAX_RESULTS (Widgets/model_search_picker.py:67). Typing never triggers discovery.
- A one-row param strip shows Temp / Max tokens / Stream / Thinking (Thinking only if supported), each with its Source word. Tab moves into the strip and keeps the highlighted model as the choice.
- Enter applies to the chat (APPLY_TO_CHAT) through the existing live-commit path (UI/Screens/chat_screen.py:5528-5570) and returns focus to the composer. Esc changes nothing.
- Alt+M or Ctrl+O while the switcher is open opens Surface 2 for the highlighted model. The draft is transferred as a ConsoleSettingsTransfer (console_settings_apply.py:171-175).
- Setup state (mockups M3/M3B): the switcher auto-opens when no model is ready. Detected local servers are listed first, cloud providers show "Needs key", and dead servers show fix text.
  - The first successful pick also runs MAKE_NEW_CHAT_DEFAULT when chat_defaults has no ready model.
  - With the ADR amendment, "Needs key" expands an inline masked key row.
- Entry points:
  - Alt+M (chat_screen.py:1939)
  - the Provider/Model chips (Widgets/Console/console_status_chips.py:75-96)
  - `/model [query]`, which prefills the query. Today the argument is dropped at chat_screen.py:19366-19374.
  - the palette entry "Console: Switch model…" (UI/console_command_provider.py:42-46)

SURFACE 2: MODEL SETTINGS (new Widgets/model_settings_panel.py region widget, plus controller UI/Console_Modules/model_settings.py). One widget, two hosts.
- Console host: a right-anchored drawer, 100 columns. The scope resets to "This chat" on every open.
  - Opened by Alt+M twice or Ctrl+O from the switcher, `/settings`, the rail "MODEL · Tune ›" button, or the palette entry "Console: Model settings…".
- Settings host: the PROVIDERS_MODELS category (UI/Screens/settings_config_models.py:14) is relabelled "Models" and renders the same widget.
  - Scope choices are "<subject> default" and "All models". The subject model is picked with Surface 1.
  - A SAVED MODEL DEFAULTS list shows which models carry their own profile values; today these are invisible.
  - The impact pane is hidden for this category, and the component takes its width.
- The scope bar maps to existing writers:
  - "This chat": live Apply (ADR-095).
  - "<model> default": SAVE_MODEL_DEFAULT (api_settings.<p>.model_defaults[<m>], via Chat/console_settings_defaults.py:118-139).
  - "All models": chat_defaults sampling. This is today's Console Behavior block (UI/Screens/settings_screen.py:18562 onward).
  - The "New chats use" row: MAKE_NEW_CHAT_DEFAULT.
  - The Ctrl+S label always names the target, e.g. "apply to this chat" or "save sonnet default".
- Rows: Label, Value (the effective value, never a placeholder), Source word (chat / model / all / provider / built-in / catalog / env / live / edited), Help.
- Sections: MODEL, SAMPLING, REASONING, CONTEXT, CONNECTION.
  - A section with no layer at the chosen scope renders read-only and its header says why.
  - Unsupported fields are not rendered; one summary line names them. "Hidden for Anthropic: Min P, Seed, Presence, Frequency" is derived from the PROVIDER_PARAM_MAP anthropic entry (Chat_Functions.py:281-299).
- CONNECTION is provider-owned in both hosts and labelled as such: key or env var, endpoint, catalog refresh, custom-endpoint rename/edit/delete, "Test connection" (a free model-list probe) and "1-token test…" (the existing consent flow, Widgets/Console/console_settings_modal.py:1896-1953). Keyless providers (Chat/provider_readiness.py:89) show no key row.
- Staged drafts are kept in both hosts. Esc goes through SafeModalDismissMixin and a new dirty guard (Apply / Discard / Keep editing). Today Esc discards silently: console_settings_modal.py:4041-4054 guards only reset and compaction.

DISPLAY-ONLY (kept, no editing)
- The status chips open Surface 1.
- The rail summary is retitled "Model", with a "Tune ›" button that opens Surface 2 (Widgets/Console/console_settings_summary.py:452-468).
- The Inspect rail keeps its read-only context view.

DELETED OR MERGED
1. ConsoleModelPopover (console_model_popover.py, 1,432 lines) becomes Surface 1. Its "Defaults…" sub-view (:592-621, :652-670) becomes the scope bar.
2. ConsoleSettingsModal (console_settings_modal.py, 7,807 lines):
   - The Model view (about :1714-2200) becomes Surface 2's MODEL, SAMPLING, REASONING and CONNECTION sections.
   - The "Advanced generation" collapsible (:1954) is dissolved; supported fields are always visible.
   - The Context and memory view (:2209-2460) becomes CONTEXT.
   - "Conversation identity" (:2139) moves to the Assistant chip's character picker, because it is not a model setting.
   - "Request estimate" (:2168) becomes the CONTEXT Usage row.
3. The Settings Providers & Models card (settings_screen.py:16660-17438):
   - The search Input and the height-6 OptionList picker (:16745-16758; css/features/_settings.tcss:458-463), the hidden Select and manual row (:16766-16797), and the free-text Model Input (:16800-16808) are all replaced by Surface 1.
   - Discover / Save selected / Clear (:17091-17110) go away: discovered models become switcher rows instead of feeding a save list.
   - The "Generation defaults" collapsible (:17203-17409) becomes Surface 2.
   - The catalog prose rows (:17410-17438) are deleted.
   - Test Provider (:16926), which says "configuration is complete" without any request (:15246-15254), becomes CONNECTION Test using the honest verdict.
4. Prompt-cache snapshots (:16698-16736) move to Settings ▸ Storage.
5. Automatic refresh, 1 + 14×2 checkboxes (:17135-17195), becomes the CONNECTION "Catalog" row for the subject provider. The global startup/stale-hours pair moves to Diagnostics.
6. Custom endpoints (:17470 onward) become the CONNECTION section when the subject is custom-ep:. Creation stays in the existing ConsoleEndpointTemplateModal, reached from the switcher's "+ New endpoint…" row. It is a creation dialog, not a model surface.
7. Console Behavior:
   - The global fallback block (:18562 onward) and its config-key detail rows (:21303-21310) become the "All models" scope.
   - The context defaults "Budget strategy" and "When limit nears" (:18396-18440) become CONTEXT at "All models".
   - "Override current Console model" (:18036-18060), which actually holds reasoning replay and native tools, becomes REASONING rows at "Model default" scope, for local servers.
8. ConsoleSetupModal and its snow-field backdrop (console_setup_modal.py:1-27) become the switcher setup state plus a one-row recovery line above the composer.
9. First-run wizard:
   - ProviderStep (UI/Wizards/FirstRunSetupWizard.py:1169) is removed.
   - ModelStep (:3321; it renders an auth failure as a disabled radio at :3877-3883) is removed.
   - The wizard hands off to the switcher setup state.
10. Palette:
   - The provider-only "LLM Provider Management: Switch to X" commands (app.py:1493-1600) are deleted.
   - "Console: Session settings…" (console_command_provider.py:96-100) becomes "Console: Model settings…".
11. The provider-recovery teleport, NavigateToScreen(TAB_SETTINGS) with no way back (chat_screen.py:20533-20563), now opens Surface 1 or 2 in place.
12. Three provider pickers become one: ConsoleProviderPicker (Widgets/Console/console_provider_picker.py:48), the popover Select (console_model_popover.py:472), and the Settings OptionList.
13. Four streaming controls become one compact Select. It offers On/Off at chat scope and Inherit/On/Off at default scopes, per ADR-095. Today there are:
   - a Select (settings_screen.py:17399-17408)
   - a Checkbox (:18585-18592)
   - a Button (console_settings_modal.py:2020-2034)
   - a Button (console_model_popover.py:513-517)
14. Label drift ends because one FieldSpec table feeds both surfaces. Current pairs:
   - "Think budget" (settings_screen.py:17378, :18695) vs "Budget" (modal :2125)
   - "Budget strategy" (:18403) vs "Budget mode" (modal :2250)
   - "When limit nears" (:18427) vs "Behavior" (modal :2295)
   - "Endpoint" (:16811) vs "Base URL" (modal :1755)
   - Three surfaces titled "Conversation settings" (popover :455, modal :1648, rail :454) become "Switch model", "Model settings" and rail "Model".

## mockups

SIZES AND PLACEMENT AT 211x44 (primary)
- Switcher: ModalScreen `align: center top`. Fixed width 104 (x=53..156), starting at row 3, height auto up to 30 rows. The transcript stays visible around and below it.
- Console drawer: ModalScreen `align: right top`. Fixed width 100 (x=111..210), rows 3-35. The left rail and about 74 columns of transcript stay readable.
- Settings host: category rail 31 columns plus the component at 1fr (178 columns). There is no nested frame (the pane is the frame), and the impact pane is hidden.

AT 235x52 (secondary)
- The switcher and drawer keep their fixed widths; the extra 24 columns go to the transcript or the help column.
- The extra 8 rows go to switcher list rows (max 36) and remove all scrolling in the drawer.
- No percentage widths anywhere, so both sizes render the same surfaces.

POWER-USER KEY PATH (shown in M1, then M2)
Goal: switch to Anthropic sonnet, temperature 0.9, max tokens 8192, then keep typing.
1. Alt+M
2. type `sonn` (the Recent row "claude-sonnet-4-5" is highlighted)
3. Tab (focus moves to Temp; its value is select-all)
4. type `0.9`
5. Tab
6. type `8192`
7. Enter. The change applies to Chat 3, the switcher closes and the caret is back in the composer.

That is 15 keys.

A/B TOGGLE (M2): Alt+M, then Enter. The previous model is pre-highlighted.

FIRST-RUN KEY PATHS (M3, M3B)
- Local server running: the switcher auto-opens with the detected model highlighted. Enter is the only key. It applies to the chat and becomes the default for new chats.
- Cloud: type `anth`, Enter, paste the key, Enter. The key is verified with a free model-list call and the model is applied.

=== M1 Console 211x44 — Alt+M switcher, power-user frame after typing "sonn", Tab, "0.9", Tab, "8192"
 Home  [Console]  Library  Personas  Study  Workflows  Schedules  MCP  Lab   Settings F4                                                            Ctrl+P palette
 Console · Chat 3 · workspace tldw_chatbook                                         New Ctrl+T · Sessions Ctrl+K · Model Alt+M · Inspect Alt+I
 CONTEXT                            │                ┌─ Switch model ──────────────────────────────────────────────────────────────── applies to this chat ─┐                                            │Inspect ▸
 Workspace  tldw_chatbook           │ You  14:01     │ › sonn▏                                                                            9 of 214 models   │                                            │
 Sources    3 staged                │   Summarise the│ RECENT                                                                                               │                                            │
 MODEL                   Tune ›     │                │ › claude-sonnet-4-5                       Anthropic         200k  Verified      pin · previous       │                                            │
 Model      qwen3:32b               │ qwen3:32b · Oll│ ALL MATCHES                                                                                          │                                            │
 Provider   Ollama · Reachable      │   ADR-006 split│   claude-sonnet-4-0                       Anthropic         200k  Verified                           │                                            │
 Sampling   temp 0.7 · max 4096     │   values, adapt│   claude-3-7-sonnet-latest                Anthropic         200k  Verified                           │                                            │
 Context    21.4k / 32k             │   Settings; Con│   anthropic/claude-sonnet-4.5             OpenRouter        200k  Needs key                          │                                            │
 SESSIONS                           │   conversation-│   anthropic/claude-sonnet-4               OpenRouter        200k  Needs key                          │                                            │
 Chat 3     now                     │                │   sonnet-distill:14b                      Ollama             32k  Reachable     unlisted             │                                            │
 Chat 2     14:00                   │ You  14:03     │   … 3 more — keep typing to narrow                                                                   │                                            │
                                    │   Now draft the├──────────────────────────────────────────────────────────────────────────────────────────────────────┤                                            │
                                    │                │ Temp [0.9▏ ] edited   Max tokens [8192  ] edited   Stream [On ▾] all   Thinking [Off ▾] model        │                                            │
                                    │ qwen3:32b · Oll│ Enter apply to chat · Tab next field · Alt+M/Ctrl+O all settings · Ctrl+F pin · Esc cancel           │                                            │
                                    │   Draft: The qu└──────────────────────────────────────────────────────────────────────────────────────────────────────┘                                            │
                                    │   to this chat; Alt+M twice opens Model settings...                                                                                                                │
                                    │                                                                                                                                                                    │
            (rows 20-40: transcript continues; unchanged)                                                                                                                                                  │
                                    │                                                                                                                                                                    │
 Provider Ollama · Model qwen3:32b · Temp 0.7 · Max 4096 · Stream On · Assistant General · Context 21.4k/32k
 ▌ Now draft the migration note for the switcher.
 F1 Help  Alt+M Model  Ctrl+K Sessions  Alt+C Context  Alt+I Inspect  Ctrl+P Palette

=== M2 Switcher overlay at true width (104 cols): empty query, A/B frame. Alt+M then Enter swaps to "previous".
┌─ Switch model ──────────────────────────────────────────────────────────────── applies to this chat ─┐
│ › ▏                                                                        type to search 214 models │
│ RECENT                                                                                               │
│ › claude-sonnet-4-5                       Anthropic         200k  Verified      pin · previous       │
│   qwen3:32b                               Ollama             32k  Reachable     pin · ● current      │
│   gpt-5.1                                 OpenAI            400k  Not tested                         │
│ PINNED                                                                                               │
│   deepseek-reasoner                       DeepSeek          128k  Verified      pin                  │
│ PROVIDERS  (Enter filters to one provider)                                                           │
│   Anthropic · 9 models                                            Verified                           │
│   Ollama · 12 served now                  localhost:11434         Reachable     default for new      │
│   OpenAI · 41 models                                              Not tested                         │
│   OpenRouter · 512 models                                         Needs key                          │
│   + New endpoint…                                                                                    │
├──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Temp 0.7 chat   Max tokens 4096 model   Stream On all   Thinking —   (Tab to edit)                   │
│ Enter apply to chat · Tab next field · Alt+M/Ctrl+O all settings · Ctrl+F pin · Esc cancel           │
└──────────────────────────────────────────────────────────────────────────────────────────────────────┘

=== M3 First run: the switcher auto-opens in setup state (replaces the starfield "Get started" card and the wizard's Provider/Model steps). Local path: Enter = 1 key.
┌─ Set up a model ───────────────────────────────────────────────────────────────────────── first run ─┐
│ Nothing can answer yet. Pick a model; Chatbook sets it up here and makes it your default.            │
│ › ▏                                                                                type to search    │
│ DETECTED ON THIS MACHINE                                                                             │
│ › qwen3:8b                                Ollama             32k  Reachable     recommended          │
│   llama3.2:3b                             Ollama            128k  Reachable                          │
│ CLOUD — needs an API key                                                                             │
│   claude-sonnet-4-5                       Anthropic         200k  Needs key     recommended          │
│   gpt-5.1                                 OpenAI            400k  Needs key                          │
│   gemini-2.5-pro                          Google Gemini       1M  Needs key                          │
│ NOT RUNNING                                                                                          │
│   llama.cpp server                        localhost:8080          Offline       start it, then R     │
├──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Enter on a Ready model: applies to this chat AND sets the default for new chats.                     │
│ Enter use model · R retry detection · Esc close (Console stays blocked, send is disabled)            │
└──────────────────────────────────────────────────────────────────────────────────────────────────────┘

=== M3B First run, cloud: "anth", Enter opens the inline masked key row; paste, Enter verifies (free) and applies. 6 keys + 1 paste, no screen change. Needs the ADR-012 amendment; the fallback is in the risks.
┌─ Set up a model ───────────────────────────────────────────────────────────────────────── first run ─┐
│ › anth▏                                                                            2 of 214 models   │
│ CLOUD — needs an API key                                                                             │
│ › claude-sonnet-4-5                       Anthropic         200k  Needs key     recommended          │
│    API key  [••••••••••••••••••••••••sk-ant…Q2x▏    ]   Enter verify (free model-list call)          │
│             Saved to config.toml [api_settings.anthropic]. Safer: set ANTHROPIC_API_KEY instead.     │
│   claude-haiku-4-5                        Anthropic         200k  Needs key                          │
├──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Verifying… Anthropic model list → 200 OK · 9 models · claude-sonnet-4-5 confirmed                    │
│ Enter verify & use · Esc back to list (key not saved) · Ctrl+O all settings                          │
└──────────────────────────────────────────────────────────────────────────────────────────────────────┘

=== M4 Console 211x44 — Model settings drawer (Alt+M twice). Scope = This chat. Help cells are ellipsized at 100 columns; full help shows in the 1-row focus footer.
 Home  [Console]  Library  Personas  Study  Workflows  Schedules  MCP  Lab   Settings F4                                                            Ctrl+P palette
 Console · Chat 3 · workspace tldw_chatbook                                         New Ctrl+T · Sessions Ctrl+K · Model Alt+M · Inspect Alt+I
 CONTEXT                            │                                                                          ┌─ Model settings ──────────────────────────────────────────────────────────────────────── Chat 3 ─┐
 Workspace  tldw_chatbook           │ You  14:01                                                               │ Scope  (•) This chat   ( ) claude-sonnet-4-5 default   ( ) All models        2 edited            │
 Sources    3 staged                │   Summarise the three ADRs on provider settings and say which one owns cr│ Writes Chat 3 only. Differs from the sonnet default in 2 fields · Ctrl+R adopt defaults          │
 MODEL                   Tune ›     │                                                                          │ MODEL                                                                                            │
 Model      claude-sonnet-4-5       │ qwen3:32b · Ollama  14:01                                                │   Model             claude-sonnet-4-5  ›      chat      Enter opens the switcher (same list as A │
 Provider   Anthropic · Verified    │   ADR-006 splits ownership: Settings persists defaults, Console resolves │   Provider          Anthropic                           Verified 14:02 · key from env ANTHROPIC_ │
 Sampling   temp 0.7 · max 4096     │   values, adapters translate request shape. ADR-012 keeps durable credent│   Context window    200,000                   catalog   from the provider's model list           │
 Context    21.4k / 200k            │   Settings; Console only surfaces recovery. ADR-095 makes Apply-to-this-c│   New chats use     qwen3:32b (Ollama) ▾      all       pick 'claude-sonnet-4-5' to start new ch │
 SESSIONS                           │   conversation-owned and adds two explicit default actions...            │ SAMPLING                                                                                         │
 Chat 3     now                     │                                                                          │   Temperature       0.9 *                     edited    0–1 on Anthropic; lower = focused, highe │
 Chat 2     14:00                   │ You  14:03                                                               │   Max tokens        8192 *                    edited    reply length cap; this model allows up t │
                                    │   Now draft the migration note for the switcher.                         │   Streaming         On ▾                      all       show the reply as it is generated        │
                                    │                                                                          │   Top P             0.95                      model     nucleus sampling; usually tune this OR T │
                                    │ qwen3:32b · Ollama  14:03                                                │   Top K             — not sent                provider  only the K likeliest tokens; blank = pro │
                                    │   Draft: The quick popover is replaced by a palette-style switcher. Enter│   Hidden for Anthropic: Min P, Seed, Presence, Frequency — this provider does not accept them.   │
                                    │   to this chat; Alt+M twice opens Model settings...                      │ REASONING                                                                                        │
                                    │                                                                          │   Thinking          Off ▾                     model     extended thinking; On spends Think budge │
                                    │                                                                          │   Think budget      —                         model     tokens reserved for thinking when On (mi │
                                    │                                                                          │ CONTEXT                                                                                          │
                                    │                                                                          │   Usage             21,400 / 200,000  11%     live      last request 19,900 · no compaction need │
                                    │                                                                          │   Compaction        Ask ▾                     all       ask before summarising older turns       │
                                    │                                                                          │   Compact at        80 %                      all       of the conversation budget               │
                                    │                                                                          │   Actions           [Compact now] [Reset memory…]                                                │
                                    │                                                                          │ CONNECTION · Anthropic — shared by every chat and default using Anthropic                        │
                                    │                                                                          │   API key           env ANTHROPIC_API_KEY     env       Verified 14:02 (model list 200 OK)       │
                                    │                                                                          │   Endpoint          api.anthropic.com         built-in  change only for a proxy                  │
                                    │                                                                          │   Catalog           Auto-refresh On ▾                   9 models · refreshed 2 h ago · save new  │
                                    │                                                                          │   Test              [Test connection] [1-token test…]          connection test is free; 1-token  │
                                    │                                                                          ├──────────────────────────────────────────────────────────────────────────────────────────────────┤
                                    │                                                                          │ Temperature — Anthropic accepts 0–1. Edited, not applied. Ctrl+Z restores 0.7 (model default).   │
                                    │                                                                          ├──────────────────────────────────────────────────────────────────────────────────────────────────┤
                                    │                                                                          │ Ctrl+S apply to this chat · Esc close (asks if edited) · Alt+M switch model                      │
                                    │                                                                          └──────────────────────────────────────────────────────────────────────────────────────────────────┘
                                    │                                                                                                                                                                    │
 Provider Anthropic · Model claude-sonnet-4-5 · Temp 0.7 · Max 4096 · Stream On · Assistant General · Context 21.4k/200k
 ▌ Now draft the migration note for the switcher.
 F1 Help  Alt+M Model  Ctrl+K Sessions  Alt+C Context  Alt+I Inspect  Ctrl+P Palette

=== M5 Settings ▸ Models 211x44 — the SAME component, scope = model default. Rail 31 columns + component 178; no inner frame, no impact pane. At 211 the help column is about 110 characters, which fixes "no help text" without an inspector.
 Home  Console  Library  Personas  Study  Workflows  Schedules  MCP  Lab  [Settings F4]                                                            Ctrl+P palette
 / search settings             │ Models                                                                                                     subject: anthropic / claude-sonnet-4-5 · 2 unsaved — Ctrl+S save
 Overview                      │ ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
›Models *                      │ Scope  (•) claude-sonnet-4-5 default   ( ) All models          This chat: in Console press Alt+M twice
 Web Search                    │ Saves to config.toml. Used by new chats and by any chat you switch to this model; chats already open keep their own values (Ctrl+R adopt defaults there).
 Speech & TTS                  │ MODEL
 Appearance                    │   Model             claude-sonnet-4-5  ›                whose defaults you are editing · Enter picks another (does not change New chats use)
 Theme                         │   Provider          Anthropic                           Verified 14:02 · key from env ANTHROPIC_API_KEY
 Storage                       │   Context window    200,000                   catalog   from the provider's model list
 Workspaces                    │   New chats use     qwen3:32b (Ollama) ▾      all       pick 'claude-sonnet-4-5' to start new chats with this model
 Tool Profiles                 │ SAMPLING
 Privacy & Security            │   Temperature       0.9 *                     edited    0–1 on Anthropic; lower = focused, higher = varied
 Network                       │   Max tokens        8192 *                    edited    reply length cap; this model allows up to 64,000
 Personal Context              │   Streaming         On ▾                      all       show the reply as it is generated
 Console Behavior              │   Top P             0.95                      model     nucleus sampling; usually tune this OR Temperature
 Library & RAG                 │   Top K             — not sent                provider  only the K likeliest tokens; blank = provider default
 Artifacts                     │   Hidden for Anthropic: Min P, Seed, Presence, Frequency — this provider does not accept them.
 Personas                      │ REASONING
 Skills                        │   Thinking          Off ▾                     model     extended thinking; On spends Think budget first
 Schedules                     │   Think budget      —                         model     tokens reserved for thinking when On (min 1,024)
 Watchlists                    │ CONTEXT · no per-model layer — shown read-only; switch Scope to All models to edit
 Workflows                     │   Budget            Automatic                 all       conversation token budget follows the model window
 MCP Defaults                  │   Compaction        Ask                       all       ask before summarising older turns
 ACP Defaults                  │   Compact at        80 %                      all       of the conversation budget
 Image Generation              │ CONNECTION · Anthropic — shared by every chat and default using Anthropic
 Video Generation              │   API key           env ANTHROPIC_API_KEY     env       Verified 14:02 (model list 200 OK)
 Agents                        │   Endpoint          api.anthropic.com         built-in  change only for a proxy
 Internal Prompts              │   Catalog           Auto-refresh On ▾                   9 models · refreshed 2 h ago · save new to config Off
 Diagnostics                   │   Test              [Test connection] [1-token test…]          connection test is free; 1-token test may bill
 Advanced Config               │ SAVED MODEL DEFAULTS · 3 models have their own values (Enter on one makes it the subject)
 About                         │   claude-sonnet-4-5 temp 0.9 · max 8192       Anthropic editing now
                               │   qwen3:32b         temp 0.6 · think Off      Ollama    also: history replay Automatic, native tools On
                               │   gpt-5.1           reasoning Medium          OpenAI
                               │ ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
                               │ Temperature — Anthropic accepts 0–1. Blank = inherit All models (0.7).   config: api_settings.anthropic.model_defaults["claude-sonnet-4-5"].temperature
                               │ Ctrl+S save sonnet default · Ctrl+Z revert field · Alt+M pick subject model · Esc to categories (asks if unsaved)
                               │
 F1 Help · / search · Ctrl+S save · Ctrl+Z revert · F6 next pane · Alt+M pick model

The source frames were generated at exact widths. M1's 20 identical empty transcript rows are elided above.

## keystroke_budgets

BASELINES FROM THE REVIEW
- Alex's task: 45 actions across 2 modals.
- A/B toggle: 16-20 keys.
- Settings Model field: Tab #23. Generation defaults: about Tab #48.
- The first-timer cloud path teleports to F4 with no way back (chat_screen.py:20558-20562).

PERSONA 1: FIRST-TIMER, FRESH CONFIG. Goal: a working model, then type.
- 1a. Local server running, e.g. Ollama detected. The switcher auto-opens in setup state (M3) with the recommended detected model highlighted. Enter = 1 key. It applies to this chat and, because no ready default exists, also sets the new-chat default; the status line says so. Budget ≤ 2.
- 1b. Cloud, key already in an env var. The row reads "Key found (env) · not tested". Type "anth" (4) + Enter (1) runs the free verify probe and applies = 5 keys. Budget ≤ 6.
- 1c. Cloud, pasted key (M3B). "anth" (4) + Enter (1) opens the inline masked key row, then paste (1 action), then Enter (1) verifies and applies. 6 keys + 1 paste, 0 screen changes. Budget ≤ 8.
  - Fallback if the ADR-012 amendment is rejected: Enter jumps to Settings ▸ Models with the key field focused; paste, Ctrl+S, and the existing ConversationSettingsReturnIntent auto-returns (chat_screen.py:3576-3600). That is 7 keys + paste and 2 screen changes, but no dead end.
- 1d. Nothing available. The row text names the fix ("Ollama isn't running at localhost:11434 — start `ollama serve`"). After fixing, R = 1 key.

PERSONA 2: POWER USER
- 2a. Alex: switch to Anthropic sonnet, temp 0.9, max tokens 8192, back to typing. 15 keys, 1 surface, focus returns to the composer. Budget ≤ 16.

  | Step | Keys | Count |
  | --- | --- | --- |
  | Open switcher | Alt+M | 1 |
  | Filter | "sonn" | 4 |
  | Move to Temp | Tab | 1 |
  | Set temperature (field is select-all on focus) | "0.9" | 3 |
  | Move to Max tokens | Tab | 1 |
  | Set max tokens | "8192" | 4 |
  | Apply | Enter | 1 |

- 2b. Switch model only: Alt+M + 3-5 characters + Enter = 5-7 keys.
- 2c. A/B toggle between two favourites: Alt+M, Enter = 2 keys each way. The previous model is pre-highlighted, Alt-Tab style. Unfinished per-model edits restore through ConsoleSettingsDraftState.model_drafts (console_settings_apply.py:94-113). Budget = 2.
- 2d. Pinned model not in Recent: Alt+M + 2-4 characters + Enter ≤ 6 keys (or arrows).
- 2e. Save this chat's tuning as the model default: Alt+M, Alt+M (or Ctrl+O), Shift+Tab (to scope), → (Model default), Ctrl+S = 5 keys.
- 2f. Make the current model the new-chat default: Alt+M, Alt+M, Tab (New chats use), Enter, ↓, Enter, Ctrl+S = 7 keys.
- 2g. Settings, change a model's default temperature: F4 (1), move to Models + Enter (≤3), Model row Enter + 4 characters + Enter to pick the subject (6), Tab ×2 (2), "0.9" (3), Ctrl+S (1) ≈ 16 keys. If the subject is already the new-chat default, skip subject picking: ≈ 10 keys.

FOCUS-ORDER CEILINGS (identical in both hosts, enforced by pilot tests)
- Model = Tab stop 1, Temperature = stop 3, Max tokens = stop 4, last editable field ≤ stop 18.
- The scope bar is Shift+Tab from Model.
- Read-only rows (Provider, Context window, Usage) are not focus stops.

Every accelerator is printed in the surface's own key row, so the unadvertised Ctrl+Enter problem does not recur.

## density_rules

1. ONE FRAME PER SURFACE
- Switcher and drawer: `border: round $ds-grid-line` on the outer container only.
- Settings host: no component border, because the detail pane is the frame.
- No Collapsible anywhere in either surface. Today collapsed sections cost rows and hide values (console_settings_modal.py:1954, :2139, :2168; settings_screen.py:16698, :17203).
- Sections are 1-row bold headers (`height:1; text-style:bold; color:$text; margin:0`) with 0 blank rows between fields. At most 2 rule glyphs cross any row (the frame edges); at most 1 horizontal rule per surface above the footer.

2. ONE-ROW CONTROLS
- Every Input, Select, Button and RadioButton is constructed with `compact=True`, which Textual 8.2.8 supports (textual/widgets/_input.py:283 reactive `compact`; also Select and ToggleButton).
- Rules: `height:1; border:none; border-left: solid $ds-control-edge`, following the DESIGN.md dense-form convention (DESIGN.md:248-270).
- Focus: `border-left: thick $ds-action-focus; background: $ds-focus-bg; text-style: bold`, giving three signals.
- This replaces MODAL_CONTROL_HEIGHT = 3 (console_settings_modal.py:168) and the 2×2 grid of 3-row buttons (console_model_popover.py:253-262).

3. FIXED COLUMN GRID FOR FIELD ROWS
- `.ms-row {layout:horizontal; height:1}`
- `.ms-gutter {width:2}`: the focus marker "›" is text, not colour.
- `.ms-label {width:18}`
- `.ms-value {width:26}`: inputs are capped at the value column and are never `width:100%`. Today inputs run 111-136 columns for about 25-character values.
- `.ms-source {width:10; color:$text-muted}`
- `.ms-help {width:1fr; text-wrap:nowrap; text-overflow:ellipsis; color:$text-muted}`. text_overflow exists in textual/css/styles.py.
- The help column absorbs spare width: about 40 characters in the drawer, about 110 in Settings. Full help text plus the config key (muted, prefixed "config:") appear only in the 1-2 row focus footer. Config keys never appear as labels. Today's examples: settings_screen.py:21303-21310 and "Console Defaults".

4. SWITCHER GEOMETRY
- `ModelSwitcher {align:center top}` and `#switcher {width:104; max-width:100%; height:auto; max-height:70%; margin-top:1; padding:0 1}`.
- The query Input is compact, 1 row.
- OptionList: `height:auto; max-height:24`, one row per option. Rows use fixed columns: marker 2, model 40, provider 16, context 6, readiness 14, tags 20 (= 98 of 100).
- Group headers are disabled Options restyled via the `option-list--option-disabled` component class to ≥ 4.5:1. They must not inherit the dim that the Legible Disabled Rule warns about (DESIGN.md:163-186).
- Param strip: 1 row (Inputs width 7, Selects width 8). Key row: 1 row.
- Fixed 104 columns, not 85% of the viewport. Today the popover grows to 170 columns with 56% blank rows (console_model_popover.py:162-184).

5. DRAWER GEOMETRY
- ModalScreen `align: right top; background: $background 20%`. Textual's default is 60% (textual/screen.py:2165-2168); 20% keeps the transcript legible.
- `#drawer {width:100; max-width:60%; height:100%}`: fixed, not a percentage tier (compare _console_panels.tcss:200-203).

6. SETTINGS HOST
- For the Models category: `#settings-impact-pane {display:none}`, and the component takes the full detail width.
- The State bar badge reflects real dirty state ("2 unsaved — Ctrl+S save" / "Saved"), not the category constant it shows today (settings_screen.py:9418-9419).

7. CHROME BUDGET
- Switcher ≤ 4 chrome rows (2 border rows + separator + key row): ≤ 17% at minimum list size and 13% typical.
- Drawer ≤ 5 of about 33 rows (15%).
- Settings host ≤ 3 rows.
- Today: 21-25%.

8. CONTRAST (measured in a running terminal per DESIGN.md:163, not inferred from token names)
- Outer frame border ≥ 3:1 against its panel. Today nested frames measure 1.0-1.05:1.
- Switcher highlight row ≥ 3:1 background delta (`background:$accent 40%`, plus bold and the "›" marker). Today: 1.12:1.
- Primary button focus uses `outline: heavy $accent`; it must never darken.
- Help and source text ≥ 4.5:1.
- Rail focus row ≥ 3:1 plus a "›" marker. Today: 1.1:1.

9. STATE IS TEXT
- Every enum and toggle renders its word (On/Off/Inherit, Ask/Automatic/Off). Streaming uses ONE compact Select everywhere.
- Dirty = "*" plus Source "edited".
- Readiness uses the verdict words.
- Pinned, current, previous and default render as tag words.
- No SelectionList or colour-only checkboxes in these surfaces.

10. VALUES, NOT PLACEHOLDERS
- Each field shows its effective value. Blank means inherit, and the Source column then names the inherited layer.
- Unsupported fields are removed, not greyed. One line names them, derived from PROVIDER_PARAM_MAP (Chat_Functions.py:232-299).
- One FieldSpec table (label, help, range, param-map key) feeds the switcher strip and both hosts, so labels cannot drift.

11. PROVIDER LIST HYGIENE
- Canonical providers only.
- Aliases (provider_catalog.py:58-68) appear only if configured, tagged "legacy alias → use X".

## phasing

Phases are PR-sized and in dependency order. Every UI PR updates its Docs/User_Guide page (console.md, settings.md, First_Run_Setup.md) and runs ./scripts/preflight.sh. New Console code goes in UI/Console_Modules/ (screen-size ratchet, Tests/Architecture/test_screen_size_ratchet.py).

PHASE 0 — ADR, docs only. Owner approval gate; I do not decide it.
- Proposes "Two model surfaces":
  - Amend ADR-012 (backlog/decisions/012-provider-credential-settings-boundary.md): provider credentials may be entered in the shared ModelSettingsPanel or the switcher's setup row regardless of host, through the same Settings-owned config writer and masking.
  - Extend ADR-095: the switcher is the quick surface and ModelSettingsPanel is the full surface; add an "All models" scope that writes chat_defaults sampling through the existing default-mutation owner; state that open chats never rebase silently and that "Ctrl+R adopt defaults" is the explicit path.
- Blocks Phase 8's inline key entry only.

PHASE 1 — Field truth. Small; independent; ships value today.
- New pure module Chat/generation_fields.py:
  - a FieldSpec table (label, help, range, PROVIDER_PARAM_MAP key or keys; openai top_p = "maxp")
  - supported_generation_fields(provider, model) = the existing capability projection (Chat/console_settings_defaults.py:344-367) intersected with PROVIDER_PARAM_MAP keys
- Route settings_screen.py:12596-12603 (samplers always True) and console_settings_defaults.py:344 through it. This stops Min P / Seed / penalties from being shown or saved for Anthropic in every current surface.
- One unit test covering anthropic, openai and ollama. About 150 lines.

PHASE 2 — Honest readiness. Independent; survives into CONNECTION.
- Settings Test Provider runs the existing connection probe and renders provider_readiness_verdict facets (Credential / Endpoint / Model / Generation) as labelled rows. This replaces the pipe-dump "configuration is complete" (settings_screen.py:15246-15254).
- Console readiness chip and rail use verdict words, so there is no "Ready" for an unreachable local server.
- The Settings State badge tracks dirty state (settings_screen.py:9418).
- Keyless providers (provider_readiness.py:89) get no key field and no key write.
- About 300 lines plus tests.

PHASE 3 — Model switcher. Depends softly on 2 for readiness words.
- New ModalScreen and controller, reusing ModelSearchPicker's catalog and provenance resolution plus the Ctrl+K modal pattern.
- Recents come from the process MRU plus recent conversations' console_generation_settings.
- Param strip: Temp / Max tokens / Stream / Thinking.
- Enter goes through the existing live commit path (chat_screen.py:5528-5570) with surface QUICK_POPOVER.
- Rebinding:
  - Alt+M, the chips, and palette "Console: Switch model…" open the switcher.
  - `/model [query]` gets its argument passed (chat_screen.py:19366-19374).
  - The provider-only palette commands are deleted (app.py:1493-1600).
- The old popover stays reachable via Alt+M twice as a bridge until Phase 5.
- Pilot tests pin the 15-key Alex path and the 2-key A/B toggle, and assert that filtering never triggers discovery.

PHASE 4 — Pins. Small; ship only if recents alone fail UAT for the A/B case.
- `[console] pinned_models` list, instant-apply and labelled per ADR-033 rule 3; Ctrl+F toggle; PINNED group.
- About 120 lines.

PHASE 5 — ModelSettingsPanel, Console host. Depends on 1 and 3.
- Region widget plus controller.
- Scope RadioSet mapped to ConsoleSettingsAction.
- Source column from ConsoleSettingsFieldDraft.provenance plus the layer lookup.
- Sections MODEL / SAMPLING / REASONING / CONNECTION. CONNECTION is read-only in Console until Phase 0 lands.
- Dirty-guarded Esc; right drawer.
- Deletes ConsoleModelPopover (1,432 lines and its tests) and the ConsoleSettingsModal Model view. The modal temporarily keeps only "Context and memory".
- Updates _FOCUS_CONTROL_IDS (UI/Navigation/conversation_settings_navigation.py:17-46) in the same PR.

PHASE 6 — Settings host. Depends on 5.
- The category is relabelled "Models" and renders the same widget (scopes: model default / All models) plus SAVED MODEL DEFAULTS.
- Deletes settings_screen.py:16738-17438 (picker, manual row, free-text model, Discover/Save selected, Generation defaults, prose).
- Deletes Console Behavior's fallback block (:18562 onward, :21303-21310) and "Override current Console model" (:18036-18060); the latter moves to REASONING.
- Prompt-cache snapshots move to Storage.
- Catalog auto-refresh becomes the CONNECTION row plus Diagnostics.
- Custom-endpoint management moves into CONNECTION.
- Impact pane hidden for Models.

PHASE 7 — Context section and modal removal. Depends on 5 and 6.
- The modal's Context and memory view (console_settings_modal.py:2209-2460) and Console Behavior's context defaults (:18396-18440) become CONTEXT, with This chat / All models layers.
- ConsoleSettingsModal (7,807 lines) is deleted.
- "Your name in this chat" moves to the Assistant chip picker.
- The rail summary is retitled "Model".

PHASE 8 — First run. Depends on 3 and 5; inline key needs Phase 0.
- The switcher setup state replaces ConsoleSetupModal (console_setup_modal.py) and the wizard's ProviderStep and ModelStep (FirstRunSetupWizard.py:1169, :3321).
- Provider recovery (chat_screen.py:20533-20563) opens the switcher in place.
- First pick with no ready default also runs MAKE_NEW_CHAT_DEFAULT.
- If Phase 0 is rejected, "Needs key" uses the Settings round trip with ConversationSettingsReturnIntent.

## risks

1. ADR-012 CONFLICT (blocking for the inline key only)
- "Durable provider credentials remain owned by Settings" and "Put provider setup only in Console settings — rejected" (decisions/012). The shared component writes through Settings' own writer, but entering a key from a Console-hosted surface still needs an explicit owner ruling.
- Fallback: CONNECTION is read-only in the Console host, and the key is entered in Settings ▸ Models with auto-return via ConversationSettingsReturnIntent, which is already built (chat_screen.py:3576-3600; conversation_settings_navigation.py). Cost: +1 screen change.

2. THE "P0" IS ADR-SANCTIONED
- "Saved defaults don't reach the open chat" is designed behaviour: ADR-095 says "Existing/open conversations do not rebase" (decisions/095-conversation-owned-console-generation-settings.md:22-25). Conversations also store complete snapshots of their effective values.
- Auto-rebasing would break A→B→A drafts and durable snapshots. This design fixes the honesty gap instead: scope copy plus "Ctrl+R adopt defaults".
- The owner must confirm that this, and not silent rebase, is the intended fix.

3. "ALL MODELS" SCOPE FROM CONSOLE CAN WRITE GLOBAL CONFIG FROM THE LIVE SURFACE
Mitigations:
- The scope resets to This chat on every open.
- The Ctrl+S label names the target.
- A one-row consequence line.
- The writer is the existing masked default-mutation owner with locked reread (console_settings_defaults.py), extended for chat_defaults paths. That extension needs tests for concurrent Settings edits.

4. TEST AND ID CHURN
Hundreds of tests likely pin control ids (console-popover-*, console-settings-*, settings-model-profile-*, settings-console-default-*), as do _FOCUS_CONTROL_IDS and the palette entries. Grep for defect-pinning tests before each deletion (known repo lesson), and delete the tests with the surface in the same PR, not after.

5. HUGE CATALOGS
OpenRouter discovery can return thousands of models. Mitigations:
- Filter in memory only.
- Cap rendered options at 20-24.
- Never discover per keystroke.
- Warm catalogs off-thread when the switcher opens.

6. PROBING ON OPEN
- Local-provider reachability probes must be bounded (under 1 s), run in workers and be cached per open.
- Cloud providers are never auto-probed, for quota and privacy reasons. They show "Not tested" until Test or key entry.
- Risk: users read "Not tested" as broken. The copy must say "configured · not tested".

7. ALT KEY RELIABILITY
Alt+M may not arrive on macOS terminals without Option-as-Meta (chat_screen.py comment near :1955). Chips, `/model`, the palette and Ctrl+O inside the switcher remain as non-Alt paths. "Alt+M twice" has the Ctrl+O equivalent.

8. SCREEN-SIZE RATCHETS
chat_screen.py is 25,331 lines and settings_screen.py is 32,167. New code must land in UI/Console_Modules and Widgets. Deletions in Phases 5-7 should lower the ratchet; lower it in the same PRs that earn it.

9. TEXTUAL LIMITS
- Disabled OptionList group headers inherit a dim, so a component-class override is needed and must be measured.
- A horizontal RadioSet needs `layout:horizontal; border:none; height:1`.
- The compact Select overlay near the drawer bottom may flip upward; verify at 211x44.
- The ModalScreen background alpha must stay low enough that the transcript is legible behind the drawer. This needs live contrast measurement, not token inference.

10. RECENTS FROM CONVERSATION METADATA
Needs a bounded, indexed read of recent conversations. Verify the query plan with sqlite_stat1 absent before adding any index (CLAUDE.md gotcha 1); a process MRU alone is the fallback.

11. REMOVING WIZARD STEPS
This touches FirstRunSetupWizard's step graph and REQUIRED_STEP_MANUAL_SETTINGS_CATEGORIES (FirstRunSetupWizard.py:394-404), its summary/trust-chain logic (:3920 onward) and First_Run_Setup.md. The wizard summary must say "Model: set up in Console (Alt+M)", or a user may think setup is incomplete.

12. SCOPE RULE AMBIGUITY
- Context has no per-model layer.
- Connection has no scope at all.
- "New chats use" is a pointer, not a sampling layer.
If read-only-at-this-scope rows are unclear in UAT, fall back to hiding those sections at scopes that cannot write them, with one line naming where they live.

13. DOWNSTREAM IMPORTERS
ConsoleModelPopoverResult is already kept only for import stability (console_model_popover.py:64-73). Other importers of ConsoleSettingsModal and the popover (Widgets/Console/__init__.py) need a deprecation pass rather than a hard break.