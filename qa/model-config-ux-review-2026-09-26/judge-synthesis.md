**Director's verdict: two model-configuration redesigns at 211x44, and the combined recommendation**

Checkout: review-wt @ c4225b5d38. I only read and grepped code. I built the mockups with `scratchpad/synth/mk.py`, which asserts that every frame is exactly 211 columns by 44 rows.

## (1) Scores

| Criterion | A: Switch & Tune | B: Switchboard | Deciding evidence |
|---|---|---|---|
| First-timer success | **8** | 7 | A lists detected and offline servers in place, in the switcher's setup state. Its cloud path needs the key entered in Console, which ADR-012 rejects (012:25, :31); without that it falls back to B's round trip. B fixes the dead-end recovery (`chat_screen.py:20543-20563` posts `NavigateToScreen` with no way back) by reusing the return path Conversation settings already has (`ConversationSettingsReturnIntent`, `chat_screen.py:3583`; `#settings-provider-return`, `settings_screen.py:16989-16993`). Both reuse `_apply_detected_local_server` (`chat_screen.py:15397`). |
| Power-user speed | 9 | 9 | Alex's task takes 15 keys in A and 14 in B. Both toggle A/B in 2 keys by pre-highlighting the previous model. B's Alt+1..9 needs Option-as-Meta on macOS (`chat_screen.py:1955-1959`). |
| Density at 211x44 | **9** | 8 | A: one frame, no Collapsible, a fixed Label/Value/Source/Help grid, and fixed widths. B: 1-row controls and 1-row disclosures, but three bordered panes. Both write raw `height:1`/`width:36` literals, which are banned outright outside `core/_variables.tcss` (`Tests/UI/test_component_pattern_governance.py:266-289`). |
| Consistency | **9** | 6 | A: one FieldSpec table and one widget in both hosts. B: only a shared label table, so four editors keep separate layouts and behaviour. B's switcher mock also prints raw provider keys (`anthropic`, `local_ollama`), which reintroduces the naming drift in C7(d). |
| ADR compliance | 3 | **6** | Problems in A: (1) key entry in Console conflicts with ADR-012:25/31. (2) It deletes Discover/Save selected, but ADR-020:52 says "Manual Discover/Save/Clear flows from ADR-002 remain unchanged". (3) It uses Ctrl+S, Ctrl+R and Ctrl+Z, which ADR-031:8 bans. (4) It replaces the Settings State badge, which states the category's save model (task-1717, DESIGN.md:272-278, `settings_screen.py:9408-9419`), with a dirty counter. (5) Its new Console "All models" scope writes `chat_defaults`, but "ordinary Apply never mutates configuration" (095:20) and Settings owns global defaults (006:10). B: Ctrl+S twice (ADR-031:8) and the same badge replacement, but it flags the 095:74 quick-mask amendment honestly. |
| Implementation risk | 3 | **6** | A deletes `console_settings_modal.py` (7,807 lines), the popover (1,432 lines), the Settings card and two wizard steps. It also makes one widget serve two commit models (ADR-033's staged draft and ADR-095's live Apply). B keeps widget ids and ships in PR-sized steps, but its PR-7 edits a module whose size-ratchet ceiling equals its current length (`Tests/Architecture/test_module_size_ratchet.py:68` = `wc -l` 7,807). B's fix for unsupported samplers also has no data source, because the controller treats every sampler as supported (`console_chat_controller.py:774-802`). |
| **Total** | 41 | **42** | |

**What both designs missed:**
- The fold hint only checks for overflow, so it still shows at the bottom of the scroll (C5c, `console_settings_modal.py:3850-3866`).
- Test Provider appends the stored evidence twice (C3, `settings_screen.py:15143-15150` and `:15458-15469`).
- The first-run handoff is staged only for the "Start chatting" exit (`FirstRunSetupWizard.py:9936`).
- An untouched chat is converged to new defaults only while its provider is blocked (`session.py:3783-3786`).

**A's field-support fix is incomplete.** Its Phase 1 routes `settings_screen.py:12596` and `console_settings_defaults.py:344` through the new table. It misses the controller copy at `console_chat_controller.py:774`, which is the verified root cause. `console_settings_defaults.py:347` even says it "Mirror[s] the controller's" projection.

**Winner and base: B.** It ships in independent PRs along seams that already work and stays inside the ADRs. From A I graft four things: the field-truth data, pairs-only model choice, the Source column, and deletion discipline.

## (2) Recommended design: "Switchboard with field truth"

Three surfaces, each with its own name and one job:
- **Switch model** (Alt+M): choose the model for this chat.
- **Chat settings** (Ctrl+O, `/settings`): tune everything for this chat.
- **Settings ▸ Providers & Models** (F4): connect providers and set defaults for new chats.

Rules:
1. **Pairs only.** Wherever a model is chosen you pick a provider·model pair, so a provider can never be chosen without a model. Chat settings' `[Change Alt+M]` opens the switcher in pick-only mode. `ConsoleProviderPicker`, used only by the modal, is deleted.
   - Root fix: `resolve_effective_chat_configuration` should apply `chat_defaults.model` only when the explicit provider is the same (canonically) as `chat_defaults.provider` (`console_session_settings.py:1269-1275`).
2. **One field table.** Each field has a label, help text, range and request-parameter key.
   - Supported fields = the existing capability projection ∩ `PROVIDER_PARAM_MAP[provider]` (`Chat_Functions.py:232-300`). A provider with no map entry keeps today's behaviour, which satisfies TASK-30012 AC#3.
   - Every editor uses one row grammar: Label (18) | one-row control sized to its value type | Source word | help.
   - Source words: `edited *`, `this chat`, `model default`, `Console Behavior`, `provider`, `built-in`.
3. **One readiness vocabulary from one evidence owner.** A process-memory store keyed by `ProviderDraftIdentity` is a narrow owner, not a root AppState, so it fits ADR-033. The words are:
   - `Ready · not tested`
   - `Ready · verified HH:MM`
   - `Ready · reachable HH:MM`
   - `Not ready · <reason>`

   This keeps TASK-30011 AC#2 ("Ready" means no known blocker) and fixes AC#6 (surfaces must not overclaim one another).
4. **Keys.**
   - Console: Alt+M opens the switcher; Enter applies to this chat; Ctrl+N makes it the default for new chats; Ctrl+O opens Chat settings; Ctrl+Enter applies in Chat settings (already bound at `console_settings_modal.py:1141`); Esc asks before discarding edits.
   - Settings: `s`, `r` and `t` stay; F6 is fixed.
   - Never Ctrl+S, Ctrl+R, Ctrl+Z or Ctrl+D (ADR-031:8).
5. **Honest scope.** Every commit surface prints "Applies to …". The Settings badge stays and gains a count: "Draft — save with s · 2 unsaved".
6. **No new stores.**
   - Recents come from the existing conversation store and snapshot parser: `list_all_active_conversations` (`ChaChaNotes_DB.py:11160`), `get_conversations_metadata_by_ids` (`:11306`), `parse_console_generation_settings` (`console_generation_settings_metadata.py:240`).
   - Favourites/pins wait until user testing shows that PREVIOUS plus RECENT is not enough (A's own Phase 4 stance).

**Keystroke budgets:**
- Alex (switch to sonnet, temperature 0.9, max tokens 8192): Alt+M, s o n, Tab, 0 . 9, Tab, 8 1 9 2, Enter = **14**.
- A/B toggle: **2**.
- Change model only: **5**.
- Also save as model default: **16**.
- First run, local server: **2**.
- First run, cloud key: about **11**, returning to Console.

### (a) Fast switcher: Console 211x44, Alt+M just opened (previous model highlighted, so Enter is the A/B swap)
```
 Home  [Console]  Library  Personas  Study  Workflows  Schedules  MCP  Lab   Settings F4                                                                                                            Ctrl+P palette 
 Console · Refactor plan · workspace tldw_chatbook                                                                      New Ctrl+T · Sessions Ctrl+K · Model Alt+M · Inspect Alt+I                                 
 CONTEXT                            │ You  14:01                                                                                                                                                                   
 Workspace  tldw_chatbook           │   Summa╭─ Switch model · Refactor plan ────────────────────────────────────────── now: Ollama · qwen3:32b · T 0.7 · max 4096 ─╮                                              
 Sources    3 staged                │        │ Find ▌█                                                    type to search 214 models in 5 providers · Enter applies  │                                              
 MODEL                 Change Alt+M │ Ollama │ PREVIOUS · Alt+M, Enter swaps back                                                                                   │                                              
 Model      qwen3:32b               │   ADR-0│▶ claude-sonnet-4-5             Anthropic     200k  Ready · verified 14:02        used 2 h ago in this chat           │                                              
 Provider   Ollama                  │   ADR-0│ RECENT · your last 50 chats                                                                                          │                                              
 Status     Ready · reachable       │        │  qwen3:32b                     Ollama         32k  Ready · reachable 14:01       ● CURRENT                           │                                              
 Sampling   T 0.7 · max 4096        │ You  14│  gpt-5.1                       OpenAI        400k  Ready · not tested            used 3 d ago                        │                                              
 Streaming  On                      │   Now d│  deepseek-reasoner             DeepSeek      128k  Ready · verified 09:12        used 5 d ago                        │                                              
 Context    21.4k / 32k             │        │ READY PROVIDERS · top 3 each · typing searches every provider's catalog                                              │                                              
 SESSIONS                           │        │  claude-haiku-4-5              Anthropic     200k  Ready · verified 14:02                                            │                                              
 Refactor plan   now                │        │  claude-opus-4-1               Anthropic     200k  Ready · verified 14:02                                            │                                              
 Release notes   14:00              │        │    … 6 more Anthropic models                                                                                         │                                              
                                    │        │  llama3.2:3b                   Ollama        128k  Ready · reachable 14:01                                           │                                              
                                    │        │    … 10 more Ollama models                                                                                           │                                              
                                    │        │ NEEDS SETUP · Enter opens the fix                                                                                    │                                              
                                    │        │  (any model)                   OpenRouter          Not ready · no key            Enter: add key in Settings ↩        │                                              
                                    │        │  (any model)                   llama.cpp           Not ready · refused :9099     start it; rechecked on open         │                                              
                                    │        ├──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤                                              
                                    │        │ Values for claude-sonnet-4-5 · Temperature ▌1.0 ▏model   Max tokens ▌4096 ▏model   Streaming ▌On ▾▏Console Behavior  │                                              
                                    │        │ Enter apply to this chat · Tab edit values · Ctrl+N default for new chats · Ctrl+O chat settings · Esc cancel        │                                              
                                    │        │ [Save as model default]  saves Temperature, Max tokens, Streaming                          Applies to: this chat only│                                              
                                    │        ╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯                                              
                                    │                                                                                                                                                                              
                                    │                                                                                                                                                                              
                                    │                                                                                                                                                                              
                                    │                                                                                                                                                                              
                                    │                                                                                                                                                                              
                                    │                                                                                                                                                                              
                                    │                                                                                                                                                                              
                                    │                                                                                                                                                                              
                                    │                                                                                                                                                                              
                                    │                                                                                                                                                                              
                                    │                                                                                                                                                                              
                                    │                                                                                                                                                                              
                                    │                                                                                                                                                                              
                                    │                                                                                                                                                                              
                                    │                                                                                                                                                                              
                                    │                                                                                                                                                                              
 ▌Message Refactor plan…   (Enter send · Shift+Enter newline · / commands)                                                                                                                                         
 Provider Ollama · Model qwen3:32b · Ready · reachable 14:01 │ T 0.7 · max 4096 · stream On │ Assistant General │ Context 21.4k/32k                                                                                
 F1 Help  Alt+M Model  Ctrl+O Chat settings  Ctrl+K Sessions  Alt+C Context  Alt+I Inspect  Ctrl+P Palette                                                                                                         
```
- Typing `son` narrows the list. Tab rebases the draft to the highlighted pair through `rebase_console_settings_draft` and focuses Temperature. The value row then shows `EDITED` or the inherited source.
- Enter goes through the unchanged `live_committer` path.
- Esc with edits asks: "Enter apply · d discard · Esc keep editing".
- The box is 120 columns wide from a width token, with height auto up to 80%.

### (b) Model settings surface: "Chat settings" 150x22 over Console, Temperature focused (┃ = thick focus edge)
```
 Home  [Console]  Library  Personas  Study  Workflows  Schedules  MCP  Lab   Settings F4                                                                                                            Ctrl+P palette 
 Console · Refactor plan · workspace tldw_chatbook                                                                      New Ctrl+T · Sessions Ctrl+K · Model Alt+M · Inspect Alt+I                                 
 CONTEXT                            │ You  14:01                                                                                                                                                                   
 Workspace  tldw_chatbook           │   Summarise the three ADRs on provider settings and say which one owns credentials.                                                                                          
 Sources    3 staged                │                                                                                                                                                                              
 MODEL                 Change Alt+M │ Ollama · qwen3:32b  14:01                                                                                                                                                    
 Model      qwen3:32b               │   ADR-006 splits ownership: Settings persists defaults, Console resolves values, adapters translate request shape.                                                            
 Provider   Ollama                  │   ADR-012 keeps durable credentials in Settings; Console only surfaces recovery. ADR-095 makes Apply conversation-owned.                                                      
 Status     Ready · reachable       │                                                                                                                                                                              
 Sampling   T 0.7 · max 4096        │ You  14:03                                                                                                                                                                   
 Streaming  On                ╭─ Chat settings · Refactor plan ──────────────────────────────────────────────────────────────────────────────────── Anthropic · claude-sonnet-4-5 ─╮                               
 Context    21.4k / 32k       │ ▌Model and generation▐   Context and memory                                                               2 unsaved edits                          │                               
 SESSIONS                     │ Applies to this chat only · saved with the conversation · defaults live in Settings ▸ Providers & Models (F4)                                      │                               
 Refactor plan   now          │ MODEL                                                                                                                                              │                               
 Release notes   14:00        │   Model            claude-sonnet-4-5 · Anthropic    this chat        Ready · verified 14:02 · 200k context       [Change  Alt+M]                   │                               
                              │ CORE                                                                                                                                               │                               
                              │   Temperature      ┃0.9      ▏ edited *          0–1 on Anthropic; lower = focused, higher = varied · was 1.0 (model default)                      │                               
                              │   Max tokens       ▌8192     ▏ edited *          reply length cap; this model allows up to 64,000 · was 4096 (model default)                       │                               
                              │   Streaming        ▌On ▾     ▏ Console Behavior  show the reply as it is generated                                                                 │                               
                              │   Thinking         ▌Off ▾    ▏ model default     extended thinking; On spends the thinking budget first                                            │                               
                              │   Thinking budget  ▌         ▏ built-in          tokens reserved for thinking when On (min 1,024)                                                  │                               
                              │ ▾ Sampling · both inherit · hidden for Anthropic: Min P, Seed, Presence, Frequency (this provider does not accept them)                            │                               
                              │   Top P            ▌0.95     ▏ model default     keep the smallest set of likely tokens whose probabilities add up to P (0–1)                      │                               
                              │   Top K            ▌         ▏ provider          blank = provider default; sample only from the K most likely tokens                               │                               
                              │ ▸ Connection · api.anthropic.com · key from env ANTHROPIC_API_KEY · change it in Settings ▸ Providers & Models                                     │                               
                              │ ▸ Request estimate · 21.4k of 200k tokens (11%)                                                                                                    │                               
                              │ ▸ Your name in this chat · User (global default)                                                                                                   │                               
                              ├────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤                               
                              │ Temperature — Anthropic accepts 0–1. Blank inherits the model default (1.0). Not applied until you apply.                                          │                               
                              ├────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤                               
                              │ Esc close (asks: 2 unsaved)   [Use saved defaults]  [Save as model default]  [Default for new chats Ctrl+N]  [Apply to this chat  Ctrl+Enter]      │                               
                              ╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯                               
                                    │                                                                                                                                                                              
                                    │                                                                                                                                                                              
                                    │                                                                                                                                                                              
                                    │                                                                                                                                                                              
                                    │                                                                                                                                                                              
                                    │                                                                                                                                                                              
                                    │                                                                                                                                                                              
                                    │                                                                                                                                                                              
                                    │                                                                                                                                                                              
 ▌Message Refactor plan…   (Enter send · Shift+Enter newline · / commands)                                                                                                                                         
 Provider Ollama · Model qwen3:32b · Ready · reachable 14:01 │ T 0.7 · max 4096 · stream On │ Assistant General │ Context 21.4k/32k                                                                                
 F1 Help  Alt+M Model  Ctrl+O Chat settings  Ctrl+K Sessions  Alt+C Context  Alt+I Inspect  Ctrl+P Palette                                                                                                         
```
"Use saved defaults" is A's "adopt defaults", moved onto a button because Ctrl+R is banned. It rebases the draft to the model's saved profile, which is the ADR-095-compliant way for an open chat to pick up new defaults. The whole Model view fits without scrolling.

### (c) Settings ▸ Providers & Models 211x44 (panes 32 | 143 | 36, one frame level)
```
 Home  Console  Library  Personas  Study  Workflows  Schedules  MCP  Lab  [Settings F4]                                                                                                             Ctrl+P palette 
╭─ Categories ─────────────────╮╭─ Providers & Models ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮╭─ Inspector ──────────────────────╮
│ / search settings + fields   ││State: Draft — save with s · 2 unsaved │ Applies to new chats · open chats keep their own settings (in Console: Alt+M)                       ││APPLIES TO                        │
│                              ││ CONNECT ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────││ New chats       yes              │
│ ▶ Providers & Models *       ││Provider          ▌Anthropic            ▾  ▏ new-chat default  configured first (Anthropic, OpenAI, Ollama) · 24 more · legacy names last    ││ Open chats      no → Alt+M there │
│   Console Behavior           ││API key           ▌••••••••••••••sk-…9f2   ▏ saved in config   masked · used before the env var · Clear removes it                           ││ Model defaults  chats that switch│
│   Web Search                 ││Key env var       ▌ANTHROPIC_API_KEY       ▏ set in shell      safer: keeps the key out of config.toml                                       ││                 to this model    │
│   Speech & TTS               ││Endpoint           api.anthropic.com           built-in            editable only for URL providers and custom endpoints                      ││                                  │
│   Appearance                 ││Key check          Ready · verified 14:04 — models list returned 41 models; generation not tested                       t test key           ││NEXT NEW CHAT WILL USE            │
│   Theme                      ││ DEFAULT MODEL FOR NEW CHATS ────────────────────────────────────────────────────────────────────────────────────────────────────────────────││ Anthropic · claude-sonnet-4-5    │
│   Storage                    ││Model             ▌claude-sonnet-4-5     ▾ ▏ new-chat default  type to search 41 models (catalog 14:04 + saved) · 200k context               ││ T 0.7 · max 8192 · stream On     │
│   Workspaces                 ││Applies to         new chats (Ctrl+T, temporary, workspace). Open chat “Refactor plan” keeps Ollama · qwen3:32b.                             ││                                  │
│   Tool Profiles              ││ MODEL DEFAULTS · Anthropic · claude-sonnet-4-5 ─────────────────────────────────────────────────────────────────────────────────────────────││FOCUSED FIELD · Temperature       │
│   Privacy & Security         ││Temperature       ┃0.7                     ▏ model default *   0–1 on Anthropic; blank inherits Console Behavior (1.0)                       ││ 0–1 on Anthropic. Blank inherits │
│   Network                    ││Max tokens        ▌8192                    ▏ model default *   reply length cap; this model allows up to 64,000                              ││ Console Behavior (1.0).          │
│   Personal Context           ││Streaming         ▌Inherit ▾               ▏ Console Behavior  → On · Inherit / On / Off                                                     ││ ▸ config key                     │
│   Library & RAG              ││Thinking          ▌Off ▾                   ▏ model default     extended thinking; On spends the thinking budget first                        ││                                  │
│   Artifacts                  ││Thinking budget   ▌                        ▏ built-in          used only when Thinking is On (min 1,024)                                     ││KEY                               │
│   Personas                   ││▸ Sampling · Top P 0.95 (model default) · Top K inherits · hidden for Anthropic: Min P, Seed, Presence, Frequency                            ││ saved in config.toml, masked     │
│   Skills                     ││ ADVANCED · each row opens in place ─────────────────────────────────────────────────────────────────────────────────────────────────────────││ outranks the env var             │
│   Schedules                  ││▸ Context window · 200,000 tokens (catalog) · no override                                                                                    ││ t lists models; no charge        │
│   Watchlists                 ││▸ Saved model list · 12 saved in config · 41 discovered, 29 not saved · Discover / Save selected live here                                   ││                                  │
│   Workflows                  ││▸ Catalog refresh · APPLIES IMMEDIATELY · startup refresh On · every 24 h · 7 providers                                                      ││                                  │
│   MCP Defaults               ││▸ Custom endpoints · none · + New endpoint                                                                                                   ││                                  │
│   ACP Defaults               ││▸ Prompt-cache snapshots · llama.cpp only · Off                                                                                              ││                                  │
│   Image Generation           ││▸ Reasoning replay override · local models only                                                                                              ││                                  │
│   Video Generation           ││                                                                                                                                             ││                                  │
│   Agents                     ││                                                                                                                                             ││                                  │
│   Internal Prompts           ││                                                                                                                                             ││                                  │
│   Diagnostics                ││                                                                                                                                             ││                                  │
│   Advanced Config            ││                                                                                                                                             ││                                  │
│   About                      ││                                                                                                                                             ││                                  │
│                              ││                                                                                                                                             ││                                  │
│                              ││                                                                                                                                             ││                                  │
│                              ││                                                                                                                                             ││                                  │
│                              ││                                                                                                                                             ││                                  │
│                              ││                                                                                                                                             ││                                  │
│                              ││                                                                                                                                             ││                                  │
│                              ││                                                                                                                                             ││                                  │
│                              ││                                                                                                                                             ││                                  │
│                              ││                                                                                                                                             ││                                  │
│                              ││                                                                                                                                             ││                                  │
╰──────────────────────────────╯╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯╰──────────────────────────────────╯
 s save · r revert · t test key · / find field · F6 next pane · Esc back                                                Unsaved edits stay when you switch category                                                
```
- The card ends at row 25; today it runs about 115 rows. Model is Tab stop 5 (today 23).
- The Source column replaces B's "where each value comes from" inspector block.
- The model row keeps today's coupling: the default model is the subject of the Model defaults section. To edit another model's defaults, use Chat settings ▸ Save as model default. No new "subject" concept is added.

## (3) Surface map

**Keep and reshape:**
- **`ConsoleModelPopover` → "Switch model".** `compose()` is replaced; the class and its ids (`console-popover-apply`, `-temperature`, `-streaming`, `-save-model-default`, `-make-new-chat-default`) are kept. `/model [query]` gets its argument: grammar at `console_command_grammar.py:110`, dispatch map at `chat_screen.py:19211`, logic in `UI/Console_Modules/model_switcher.py`. `chat_screen.py` has 32 lines of headroom (25,363 − 25,331).
- **`ConsoleSettingsModal` → "Chat settings".** New order: Core, then Sampling, Connection and estimate as 1-row disclosures, then name. The Context and memory tab is unchanged apart from density.
- **Settings ▸ Providers & Models.** Order: Connect / Default model / Model defaults / Advanced. It moves into `UI/Settings_Modules/providers_models_card.py`, keeping a hidden `#settings-model-value` adapter.
- **Settings ▸ Console Behavior** keeps its global fallbacks (ADR-006, ADR-052) and adopts the shared field rows.
- **Display-only surfaces:** the rail Model section ("Change Alt+M" plus a Streaming row), status chips (display names plus readiness word) and the inspector.
- **Get started card:** a plain dim replaces the snow backdrop (`console_setup_modal.py:113`).
- **Wizard:** status placeholders become Static lines (`FirstRunSetupWizard.py:3836-3910`); nothing else changes.

**Merge:**

| Today | After |
|---|---|
| 4 streaming controls (`settings_screen.py:17399`, `:18588`; modal `:2020`; popover `:513`) | One Select: Inherit/On/Off, or On/Off at chat scope (ADR-095:79) |
| 3 field-support projections (`console_chat_controller.py:774`, `console_settings_defaults.py:344`, `settings_screen.py:12596`) | One function |
| 3 test-evidence stores (`console_settings_modal.py:1304`, `settings_screen.py:2949`, `:13316`) | One owner |
| Scattered field labels | One field table, fixing the label pairs A listed |
| `ConsoleProviderPicker` plus popover provider Select | The switcher's pair list |
| Popover "Defaults…" subview | Inline [Save as model default] plus Ctrl+N |
| Catalog-refresh checkboxes | One table with On/Off words |

**Delete:**
- Popover: the 2×2 grids of 3-row buttons (`console_model_popover.py:253-262`) and its duplicated context/compaction block.
- The provider-only palette commands (`app.py:1493-1600`), which switch provider without a model and title-case raw keys.
- Modal: the "Advanced generation" collapsible (`console_settings_modal.py:1955`) and the per-row "unknown" statics.
- Settings: the catalog prose rows (`settings_screen.py:17410-17438`, moved to an inspector disclosure) and the raw `chat_defaults.streaming is canonical…` line (`:18704`).
- The global CSS leaks: `_conversations.tcss:298-303` and `:313-316`, and `_evaluation_unified.tcss:59-63`. Scope them to their own features.

**Not taken from A:** the Console "All models" scope, deleting the wizard steps, deleting Discover/Save, and inline key entry.

## (4) Phased plan (dependency order)

Every UI phase also updates `Docs/User_Guide/{console,settings}.md` and runs `./scripts/preflight.sh`. New task ids start at 32960, or 33001 if the uncommitted 33000 counts.

**P1: Root-cause fixes, no layout change.** Closes C7(a), C8(1) data, the C3 double append, C8(3) and C1's first-run gap.
1. Guard `chat_defaults.model` by provider (`console_session_settings.py:1269-1275`). Add the missing test for a cross-provider rebase with `model=None`.
2. Add `supported_generation_fields()` = projection ∩ `PROVIDER_PARAM_MAP`. Route all three callers through it and delete the two mirrors.
3. Append test evidence only once (`settings_screen.py:15150` / `:15462`).
4. Give Settings F6 and Shift+F6 via `focus_relative_workbench_pane` (`Widgets/workbench_focus.py:20`), and drop the known-limitation line at `settings.md:877`.
5. Stage the CONSOLE_FIRST_CHAT handoff for every wizard exit, not just "Start chatting" (`FirstRunSetupWizard.py:9936`).
6. Delete the palette commands at `app.py:1493-1600`.

Absorbs **task-14812**: it fixes the regression against AC#6, then closes the task, whose status is stale.

**P2: Field table and honest copy.** Closes C1(a), C3 (labelled rows instead of the pipe dump at `:15325/15332`), C7(d) and the label drift.
- Save copy names its scope (`settings_screen.py:30215`, `:30242`).
- Chips use display names (`chat_screen.py:9826-9831`, `console_display_state.py:776`). Add display names for `local_onnx` and `local_transformers`.
- Rename "Console Defaults" to "Console Behavior" and "Override current Console model" to "Reasoning replay override".
- Keep the State badge and add the unsaved count.

Absorbs **task-194** and **task-486**.

**P3: Density tokens and CSS root fixes.** Closes C5(a–c), C8(2) and C8(4).
- Width and height tokens go only in `core/_variables.tcss`.
- `MODAL_CONTROL_HEIGHT` becomes compact. Rewrite the test that pins today's value (`test_console_session_settings.py:4555-4560`) on purpose.
- Scope the leaked Collapsible and Select rules.
- The fold hint re-checks `scroll_y`.
- Chat settings Esc gets a dirty guard (Apply / Discard / Keep editing).
- Grid-line and rail-focus contrast must reach at least 3:1 (`ensure_readable_text_hues` precedent, `themes.py:83`).
- Modal edits must be net ≤ 0 lines. Moving the DEFAULT_CSS rules at `:1066-1071` to the app tier pays for them.

Absorbs **task-25890**. Covers **task-32465** for these surfaces only.

**P4: Switch model.** Depends on P1–P3. Closes C6 and C7(b), and C7(a) in the UI.
- Pair rows with PREVIOUS, RECENT, READY and NEEDS SETUP groups.
- `● CURRENT` mark and a pre-highlighted previous model.
- Value row: Temperature, Max tokens, Streaming, and Thinking when supported.
- Recents are read in a worker, capped at 50.
- Pick-only mode for Chat settings' Change.

Absorbs **task-338**, **task-32859** (the switcher consumes one provider-selection builder) and the popover half of **task-194**.

**P5: Shared readiness evidence and key check.** Closes C2 and C3, with the label question settled by D2.
- The Console header readiness call (`chat_screen.py:9865-9869`) reads the shared evidence.
- `t` in Settings runs the authenticated models listing for cloud providers, which already maps 401/403 (`openai_compatible_model_discovery.py:726-737`).
- A bounded, cached local reachability probe runs when the switcher opens, in a worker.

Must respect the performance constraints in **task-24454** and **task-32804.3**. Coordinate with **task-32806.1**, which is in progress and covers placeholder keys; do not absorb it.

**P6: Chat settings layout.** Depends on P2 and P3. Closes the C8(1) UI (the hidden-for-provider line) and C1(b) through "Use saved defaults" (plus D1).
- `[Change Alt+M]` opens the switcher in pick mode; delete `console_provider_picker.py`.
- Lower the module ratchet in the same PR.

New task (**task-32864** excludes this modal).

**P7: Settings ▸ Providers & Models reorder.** Depends on P2, P3 and P5. Closes C4 and the C1(a) "Applies to" row.
- `ModelSearchPicker` replaces the free-text Model input (`settings_screen.py:16801-16807`), with discovery merged in.
- The Save list moves under Advanced, as ADR-002 and ADR-020 require.
- Rewrite the test that pins Generation defaults as collapsed (`test_settings_configuration_hub.py:4386-4420`).

Absorbs **task-31202** (add the `settings_screen.py` ratchet row at the measured post-extraction size) and the card slice of **task-1378**.

**P8: First run, connect in place.** Depends on P4 and P7. Closes the rest of C1 (recovery dead end).
- Recovery routes through `PendingHandoffStore` with a return path, as ADR-033 requires.
- "Use Ollama" opens the switcher filtered to that provider.
- NEEDS SETUP rows use the same route.

Absorbs **task-32572**. Then run **task-1379**, the Settings critique re-run.

## (5) Owner decisions (with recommendation)

1. **Should untouched open chats pick up a Settings save (C1b)?**
   - **Recommend:** converge a pristine chat (no messages, no edited fields) to the new defaults whatever its readiness, by extending the task-177 refresh (`session.py:3783-3786`). This stays within ADR-095:23-26's "initial pristine chat" wording. Chats that hold any work stay put and show scope copy plus "Use saved defaults".
2. **Readiness words and the cloud key check (C2, C3).**
   - **Recommend:** keep "Ready" for "no known blocker" (TASK-30011 AC#2), qualified as not tested / verified / reachable. Share evidence so a known failure shows Not ready (AC#6).
   - Let `t` run an explicit, non-generating authenticated model listing that reports "key accepted (models listed); generation not tested". Record this as an ADR-012 note, because 012:33 excludes provider-specific secret validation. Never auto-probe cloud providers.
3. **Should ADR-095:74's quick-surface default fields (`temperature`, `streaming`) include `max_tokens`?**
   - **Recommend:** yes, amend it. The switcher shows Max tokens, so "Save as model default" would otherwise drop a visible edit without saying so. Apply-to-chat needs no amendment, because Apply sends an empty default mask (`console_model_popover.py:1388-1392`).
4. **Where are credentials entered?**
   - **Recommend:** keep ADR-012 (Settings only), reached from NEEDS SETUP rows and the card with a return path. Decline A's inline key entry in the switcher: it saves one screen change but reverses an accepted ADR.