# Spec: Model configuration redesign — "Switchboard with field truth"

Status: **Approved design** 2026-09-26. This is the judge's recommended design
plus the owner's decisions D1–D4 (§3). Implementation has not started. All
evidence was measured at dev `c4225b5d38`. Phase parent tasks: TASK-33001–TASK-33008
(§8).

Target: full-screen terminals only. Design for **211x44** first and
**235x52** second. There is no layout work for smaller terminals.

Source artifacts live in
[`qa/model-config-ux-review-2026-09-26/`](../../qa/model-config-ux-review-2026-09-26/).
They hold the detail this spec leaves out.

| File | Contents |
|---|---|
| [report.md](../../qa/model-config-ux-review-2026-09-26/report.md) | The critique: scores, priority issues, personas, verified corrections |
| [verified-claims.md](../../qa/model-config-ux-review-2026-09-26/verified-claims.md) | Fact checks C1–C8, with corrected statements and root-cause file:line |
| [backlog-adr-check.md](../../qa/model-config-ux-review-2026-09-26/backlog-adr-check.md) | Open tasks to absorb or link, ADR constraints, task-id sweep |
| [judge-synthesis.md](../../qa/model-config-ux-review-2026-09-26/judge-synthesis.md) | Scored comparison, the recommended design, surface map, phase plan |
| [design-A.md](../../qa/model-config-ux-review-2026-09-26/design-A.md) | "Switchboard", the base design. The judge calls it **B** |
| [design-B.md](../../qa/model-config-ux-review-2026-09-26/design-B.md) | "Switch & Tune", the design the judge grafted from. The judge calls it **A** |
| [mockups-211x44.md](../../qa/model-config-ux-review-2026-09-26/mockups-211x44.md) | The three final frames, copied into §7 |
| [settings-reviewer-mockup.txt](../../qa/model-config-ux-review-2026-09-26/settings-reviewer-mockup.txt) | An earlier Settings sketch by a live reviewer, superseded by frame (c) |

The letters are swapped: the judge's A is design-B.md and the judge's B is
design-A.md.

## 1. Problem

Both surfaces scored **14/40 (Poor)** on the ten heuristics
([report](../../qa/model-config-ux-review-2026-09-26/report.md)). Measured at
full screen:

- **Space goes to chrome.**
  - Only 4–8% of the Chat settings modal's cells hold text, and 56% of the
    popover's rows are blank.
  - Every Input is 3 rows tall (`MODAL_CONTROL_HEIGHT=3`) and about 125
    columns wide, even for a 4-character value.
  - A collapsed section costs 7 rows because three global CSS rules leak into
    it.
  - The Settings card runs about 115 rows, which is 3.5 viewports at 211x44.
- **Structure is invisible.** Field borders measure 1.05:1, the modal frame
  1.01:1, Settings rail focus 1.1:1, the select highlight 1.12:1, and
  placeholders 2.8–3.5:1.
- **Four editors, four vocabularies.**
  - The four editors are the popover, the Chat settings modal, Settings and
    the wizard. Three of them are titled "Conversation settings".
  - Streaming appears as a Select, a Checkbox and two toggle buttons.
  - Underneath are three field-support projections and three test-evidence
    stores.
- **Model identity breaks on a provider switch.**
  - `resolve_effective_chat_configuration` ranks `chat_defaults.model` above
    the target provider's own model (`console_session_settings.py:1269-1275`).
    This is a regression against task-14812 AC#6.
  - The current model is never marked in the picker.
- **Status is unscoped or overclaims.**
  - Saving never says what the change applies to.
  - A known-failed test never reaches the Console header
    (`chat_screen.py:9865-9869`).
  - The Test result is a pipe-joined dump, and it appends its evidence twice.
  - For Anthropic, four samplers are shown and then silently dropped
    (`console_chat_controller.py:774-802`).
- **Keyboard paths are long.**
  - Alex's switch-and-tune task takes 45 actions across 2 modals, and an A/B
    toggle takes 16–20 keys.
  - In Settings, Model is Tab stop #23, and F6 is broken.
  - `/model` ignores its argument.
  - Esc in Chat settings discards edits without asking.

Several reviewer findings are **by design** and are not reopened here:
- Open chats do not rebase on a Settings save (ADR-095). D1 amends this
  narrowly.
- "Ready" is config-only for untested endpoints (TASK-30011 AC#2).
- Test does not check cloud keys (ADR-012:33). D2 amends this.
- Discover feeds Save selected (ADR-002).
- Model chips are capped at 25 characters (TASK-1671).
- Legacy aliases stay selectable (task-180, ADR-066).

See [verified-claims.md](../../qa/model-config-ux-review-2026-09-26/verified-claims.md).

## 2. Three surfaces, one job each

| Surface | Opened by | Job | Commits | Scope line |
|---|---|---|---|---|
| **Switch model**: `ConsoleModelPopover`; `compose()` is replaced, class and ids are kept | Alt+M, `/model [query]`, model chip, rail "Change Alt+M" | Choose the provider·model pair for this chat, plus three quick values | Enter applies to this chat · Ctrl+N makes it the default for new chats · [Save as model default] | "Applies to: this chat only" |
| **Chat settings**: `ConsoleSettingsModal`, renamed from "Conversation settings" | Ctrl+O, `/settings` | Tune everything for this chat | Ctrl+Enter applies to this chat · [Use saved defaults] · [Save as model default] · Ctrl+N | "Applies to this chat only · saved with the conversation" |
| **Settings ▸ Providers & Models**: moves to `UI/Settings_Modules/providers_models_card.py` | F4 | Connect providers and set the defaults for new chats | `s` save · `r` revert · `t` test key (staged draft, ADR-033) | "Applies to new chats · open chats keep their own settings (in Console: Alt+M)" |

Other surfaces:
- **Display only, no editing:** the rail Model section ("Change Alt+M" plus
  a Streaming row), the status chips (display names plus a readiness word),
  and the inspector.
- **Settings ▸ Console Behavior** keeps the global fallbacks (ADR-006,
  ADR-052) and adopts the shared field rows.
- **The Get started card** replaces its snow backdrop with a plain dim.
- **The wizard** changes only its status placeholders, which become Static
  lines.

**Keystroke budgets.** Pilot tests hold every phase to these:

| Task | Keys |
|---|---|
| Alex: switch to sonnet, temperature 0.9, max tokens 8192 (Alt+M, `s o n`, Tab, `0 . 9`, Tab, `8 1 9 2`, Enter) | 14 |
| A/B toggle | 2 |
| Change the model only | 5 |
| Change the model and save it as the model default | 16 |
| First run, local server | 2 |
| First run, cloud key, ending back in Console | about 11 |

## 3. Owner decisions (2026-09-26)

- **D1: pristine open chats converge; chats with work keep their settings.**
  - An untouched open chat (no messages, no edited fields) converges to newly
    saved defaults whatever its readiness. This extends the task-177 refresh:
    `_maybe_refresh_stale_default_console_settings` in `session.py` loses its
    readiness gates at `:3786-3791` and `:3814`.
  - A chat with any work keeps its settings. Its surfaces show scope copy and
    offer **Use saved defaults**.
  - Recorded in the ADR-095 amendment of 2026-09-26.
- **D2: Settings `t` checks a cloud key by listing models.**
  - For cloud providers, `t` runs one explicit, non-generating, authenticated
    model listing and reports "key accepted (models listed); generation not
    tested".
  - Cloud providers are never probed automatically.
  - Recorded in the ADR-012 amendment of 2026-09-26.
- **D3: `max_tokens` joins the quick-surface default mask.**
  - The mask becomes temperature, max_tokens and streaming. Save as model
    default from Switch model then keeps every value that surface shows.
  - Recorded in the ADR-095 amendment of 2026-09-26.
- **D4: credentials stay in Settings.**
  - There is no inline key entry in Console, so ADR-012 is unchanged here.
  - NEEDS SETUP rows and the Get started card open the exact Settings
    credential control, with a return path.

## 4. Rules

1. **Pairs only.**
   - Wherever a model is chosen, the user picks a provider·model pair, so a
     provider can never be chosen without a model.
   - Chat settings' `[Change Alt+M]` opens the switcher in pick-only mode.
     `ConsoleProviderPicker`, which only the modal uses, is deleted.
   - Root fix: apply `chat_defaults.model` only when the explicit provider
     canonically equals `chat_defaults.provider`
     (`console_session_settings.py:1269-1275`).
2. **One field table.**
   - Each field has one label, one help line, one range and one
     request-parameter key.
   - Supported fields = the existing capability projection ∩
     `PROVIDER_PARAM_MAP[provider]` (`Chat_Functions.py:232-300`). One
     function computes this and replaces the three projections
     (`console_chat_controller.py:774`, `console_settings_defaults.py:344`,
     `settings_screen.py:12596`).
   - A provider with no map entry keeps today's behaviour (TASK-30012 AC#3).
   - Unsupported fields are hidden behind one line that names them, e.g.
     "hidden for Anthropic: Min P, Seed, Presence, Frequency".
   - Provider names come from `provider_display_name` everywhere, chips
     included. Raw config keys never appear in UI copy.
3. **One readiness vocabulary and one evidence owner.** See §5.
4. **Keys.**
   - Console: Alt+M opens the switcher. Enter applies to this chat. Ctrl+N
     makes it the default for new chats. Ctrl+O opens Chat settings.
     Ctrl+Enter applies in Chat settings; it is already bound
     (`console_settings_modal.py:1141`) and is now printed on the button.
   - Esc with edits asks "Enter apply · d discard · Esc keep editing", on
     both Console surfaces.
   - Settings keeps `s`, `r` and `t`, and F6 / Shift+F6 start working there.
   - Never bind Ctrl+S, Ctrl+R, Ctrl+Z or Ctrl+D (ADR-031).
   - Every key is printed where it works (ADR-031 rule 4).
   - Ctrl+O is a new screen binding. Verify live that it reaches the app.
5. **Honest scope.**
   - Every commit surface prints "Applies to …".
   - The Settings State badge stays (ADR-033, task-1717) and gains a count:
     "Draft — save with s · 2 unsaved".
   - Save copy names its scope (`settings_screen.py:30215`, `:30242`).
6. **No new stores.**
   - Recents come from `list_all_active_conversations`
     (`ChaChaNotes_DB.py:11160`), `get_conversations_metadata_by_ids`
     (`:11306`) and `parse_console_generation_settings`
     (`console_generation_settings_metadata.py:240`). They are read in a
     worker, capped at 50, with no new index.
   - The evidence owner lives in process memory.
   - Favourites and pins wait until user testing shows that PREVIOUS plus
     RECENT is not enough.

## 5. Readiness vocabulary

"Ready" keeps its TASK-30011 meaning: no known blocker. It always says what
evidence backs it.

| Word | Meaning | Evidence from |
|---|---|---|
| `Ready · not tested` | No known blocker, and nothing has been checked | nothing |
| `Ready · verified HH:MM` | The provider's authenticated model listing accepted a cloud key at HH:MM. Generation has not been tested | Settings `t` (D2) |
| `Ready · reachable HH:MM` | A local or URL endpoint answered a model listing at HH:MM | Settings `t`, or the switcher's local probe |
| `Not ready · <reason>` | A config blocker or a known failure, e.g. `no key`, `key rejected`, `refused :9099`, `timed out`, `no model` | the readiness function plus evidence |

- **One owner.**
  - A process-memory evidence store keyed by `ProviderDraftIdentity`.
  - Its readers are the Console header (`chat_screen.py:9865-9869` passes no
    evidence today), the rail, the chips, the switcher, Chat settings and
    Settings.
  - It replaces the three local stores (`console_settings_modal.py:1304`,
    `settings_screen.py:2949`, `:13316`).
  - It is a narrow owner, not a root AppState, so it fits ADR-033.
  - Nothing is persisted.
- **A known failure shows `Not ready` on every surface.** This fixes
  TASK-30011 AC#6. A semantic edit to a draft invalidates that draft's
  evidence.
- **Cloud providers are never probed automatically (D2).** Local endpoints
  get one bounded, cached reachability probe, run in a worker when the
  switcher opens.
- **A public model listing proves nothing about a key.** OpenRouter's listing
  needs no key (ADR-020), so there `t` reports "models listed; key not
  checked" and the word stays `Ready · not tested`.
- **Performance.** Readiness is never recomputed per keystroke or on the idle
  credential poll (task-24454, task-32804.3).
- **Test results are labelled rows:** Config, Key, Endpoint, Model,
  Generation. They are never a " | " dump, and each fact appears once.

## 6. Density and contrast

**Size at 211x44 and at 235x52:**
- **Switch model:** 120 columns from a width token, height auto up to 80%. At
  235x52 the list grows to about 30 rows and the width stays the same.
- **Chat settings:** 150x22. The whole Model view fits at 211x44 without
  scrolling.
- **Settings ▸ Providers & Models:**
  - Panes are 32 | 143 | 36 columns at 211 columns wide.
  - The card ends at row 25 (today about 115 rows), and Model is Tab stop 5
    (today 23).
  - At 235 columns the detail pane takes the extra 24 columns and the
    Inspector stays at 36.

**Row grammar, the same in every editor:** Label (18) | one-row control sized
to its value type | Source word | help line.
- **Source words:** `edited *`, `this chat`, `model default`,
  `Console Behavior`, `provider`, `built-in`.
- **Control width by value type:**
  - numbers about 10 columns
  - enums about 16
  - environment variable names about 32
  - model ids and secrets about 48
  - URLs up to 64

  No input is wider than 64 columns.
- **Values, not placeholders.** A field shows its effective value with its
  source. It never shows a placeholder where a value belongs.

**Density:**
- **Controls are 1 row.**
  - Width and height tokens live only in `css/core/_variables.tcss`. ADR-150
    and ADR-161 ban raw literals in `.tcss`
    (`test_component_pattern_governance.py:266-289`).
  - `MODAL_CONTROL_HEIGHT` becomes compact. The test that pins it at 3
    (`test_console_session_settings.py:4555-4560`) is rewritten on purpose.
- **A closed disclosure costs 1 row.** Scope the leaked global rules to their
  own features: `_conversations.tcss:298-303` and `:313-316`, and
  `_evaluation_unified.tcss:59-63`.
- **One frame level.** Only pane and modal borders remain, and section
  headers are 1 row.
- **Small fixes.** The fold hint re-checks `scroll_y`
  (`console_settings_modal.py:3850-3866`). The popover's 2×2 grid of 3-row
  button slots is removed.

**Contrast.** Measure it per theme in a running terminal; never infer it from
token names.
- Grid lines, control edges and rail focus reach at least 3:1 against their
  background, following the `ensure_readable_text_hues` precedent
  (`themes.py:83`).
- Focus shows three signals: a thick edge, the focus background and bold
  text. A highlighted row also carries `▶`. A focused button never gets
  darker.
- State is always text: On / Off / Inherit, `● CURRENT`, and text checkboxes
  instead of colour-only ones. Colour only reinforces the word.
- Help and source text reach at least 4.5:1.

**Ratchets:**
- ADR-097 ratchets never rise.
- `console_settings_modal.py` has zero headroom (module-size ratchet 7,807).
  Its edits must net ≤ 0 lines until P6 lowers the ratchet.
- `chat_screen.py` has 32 lines of headroom, so new Console logic goes in
  `UI/Console_Modules/model_switcher.py`.

## 7. Mockups (211x44)

These are copied verbatim from
[mockups-211x44.md](../../qa/model-config-ux-review-2026-09-26/mockups-211x44.md).
A script built them and checked every frame at exactly 211x44.

Glyph key:
- `▌` is an editable field's edge; `┃` is a focused edge.
- `▶` marks the highlighted row.
- `▸` and `▾` are a closed and an open disclosure.

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

## 8. Phase map

Each UI phase updates `Docs/User_Guide/{console,settings}.md` and runs
`./scripts/preflight.sh`.

| Phase | Parent | Depends on | Closes | Absorbs or links |
|---|---|---|---|---|
| Root-cause fixes | TASK-33001 | — | C7(a); C8(1) data; C3 double append; C8(3); C1 first-run gap; C1(b) convergence (D1) | absorbs task-14812 |
| Field table and honest copy | TASK-33002 | — | C1(a); C3 pipe dump; C7(d); label drift | absorbs task-486 |
| Density tokens and CSS root fixes | TASK-33003 | — | C5(a–c); C8(2); C8(4) | absorbs task-25890; covers task-32465 for these surfaces only |
| Switch model | TASK-33004 | Root-cause fixes; Field table; Density tokens | C6; C7(b); C7(a) in the UI; D3 | absorbs task-338, task-32859, task-194 |
| Shared readiness evidence and key check | TASK-33005 | — | C2; C3 labelling (D2) | respects task-24454 and task-32804.3; coordinates with task-32806.1 (not absorbed) |
| Chat settings layout | TASK-33006 | Field table; Density tokens; Switch model | C8(1) in the UI; C1(b) "Use saved defaults" | new work (task-32864 excludes this modal) |
| Settings ▸ Providers & Models reorder | TASK-33007 | Field table; Density tokens; Shared readiness evidence | C4; C1(a) "Applies to" row | absorbs task-31202 and the card slice of task-1378 |
| First run, connect in place | TASK-33008 | Switch model; Settings reorder | C1 recovery dead end (D4) | absorbs task-32572; then run task-1379 |

**Root-cause fixes.** No layout change.
- Guard `chat_defaults.model` by provider (rule 1). Add the missing test for
  a cross-provider rebase with `model=None`.
- Add `supported_generation_fields()`, route all three callers through it,
  and delete the two mirrors.
- Append test evidence once (`settings_screen.py:15150` / `:15462`), and add
  a test for repeated runs.
- Give Settings F6 and Shift+F6 via `focus_relative_workbench_pane`
  (`Widgets/workbench_focus.py:20`). Drop the known-limitation line at
  `settings.md:877`.
- Stage the CONSOLE_FIRST_CHAT handoff for every wizard exit, not just
  "Start chatting" (`FirstRunSetupWizard.py:9936`).
- D1 convergence: remove the readiness gates from the task-177 refresh, as
  the ADR-095 amendment specifies.
- Delete the provider-only palette commands (`app.py:1493-1600`).
- Close task-14812 once its AC#6 regression is fixed.

**Field table and honest copy.**
- One field table feeds the labels and help of every editor, so each field
  has one label.
- Save copy names its scope.
- Chips use display names (`chat_screen.py:9826-9831`,
  `console_display_state.py:776`). Add display names for `local_onnx` and
  `local_transformers`.
- Renames: "Console Defaults" becomes "Console Behavior", and "Override
  current Console model" becomes "Reasoning replay override".
- Test results become labelled rows instead of the pipe dump
  (`settings_screen.py:15325/15332`), with custom credential query parameters
  redacted (task-486).
- The State badge gains the unsaved count.

**Density tokens and CSS root fixes.** Apply the §6 rules.
- The compact `MODAL_CONTROL_HEIGHT`, with its pin rewritten.
- Scope the leaked rules.
- The fold hint re-checks `scroll_y`.
- Esc in Chat settings gets a dirty guard: Apply / Discard / Keep editing
  (ADR-031, task-16211).
- Grid-line and rail-focus contrast reach at least 3:1.
- Modal edits net ≤ 0 lines. Moving the DEFAULT_CSS rules at `:1066-1071` to
  the app tier pays for them.

**Switch model.**
- Pair rows grouped PREVIOUS, RECENT, READY and NEEDS SETUP. The current
  model is marked `● CURRENT`, and the previous model is pre-highlighted, so
  the A/B toggle is Alt+M, Enter.
- The value row is Temperature, Max tokens and Streaming, which is exactly
  the quick mask. `QUICK_MODEL_DEFAULT_FIELDS` gains `max_tokens` (D3).
- Recents are read in a worker, capped at 50.
- Pick-only mode for Chat settings' Change.
- `/model [query]`: grammar at `console_command_grammar.py:110`, dispatch at
  `chat_screen.py:19211`, logic in `UI/Console_Modules/model_switcher.py`.
- Removed: the Defaults… subview, the 2×2 button grids, and the
  context/compaction block. Apply still submits the unchanged compaction
  override, so ADR-095's rule on compaction in quick Apply still holds.
- Keep the ids `console-popover-apply`, `-temperature`, `-streaming`,
  `-save-model-default` and `-make-new-chat-default`.
- The rail shows Streaming and "Change Alt+M" (task-338).
- One provider-selection builder (task-32859).
- Popover rows use display names (task-194).

**Shared readiness evidence and key check.**
- Build the §5 evidence owner. The Console header reads it.
- `t` runs the authenticated listing for cloud providers (D2). The discovery
  client already sends Anthropic's `x-api-key` and maps 401/403
  (`openai_compatible_model_discovery.py:726-737`).
- A bounded, cached local probe runs when the switcher opens, in a worker.

**Chat settings layout.**
- Core first: Model, Temperature, Max tokens, Streaming, and Thinking when
  supported.
- Sampling, Connection, Request estimate and your name become 1-row
  disclosures.
- Add the hidden-for-provider line, the Source column and
  `[Use saved defaults]` (D1).
- `[Change Alt+M]` opens the switcher in pick mode. Delete
  `console_provider_picker.py`, the "Advanced generation" collapsible
  (`:1955`) and the per-row "unknown" statics.
- Lower the module-size ratchet in the same PR.

**Settings ▸ Providers & Models reorder.**
- New order: Connect / Default model for new chats / Model defaults /
  Advanced. The card moves to `providers_models_card.py`, keeping a hidden
  `#settings-model-value` adapter.
- `ModelSearchPicker` replaces the free-text Model input
  (`settings_screen.py:16801-16807`), with discovery merged in.
- The saved model list, with Discover and Save selected, moves under
  Advanced. ADR-002 and ADR-020 keep both.
- One Streaming Select (Inherit / On / Off).
- The catalog-refresh checkboxes become one table with On/Off words, still
  labelled APPLIES IMMEDIATELY (ADR-033).
- Delete the catalog prose rows (`:17410-17438`) and the raw
  `chat_defaults.streaming is canonical…` line (`:18704`).
- Rewrite the test that pins Generation defaults as collapsed
  (`test_settings_configuration_hub.py:4386-4420`).
- Add the `settings_screen.py` ratchet row at the measured post-extraction
  size (task-31202).

**First run, connect in place.**
- Recovery routes through `PendingHandoffStore` with a return path, as
  ADR-033 requires.
- "Use Ollama" opens the switcher filtered to that provider.
- NEEDS SETUP rows use the same route (D4).
- Then run task-1379, the Settings critique re-run.

**Deliberate changes from the judge's plan:**
1. **D1's convergence moves from Chat settings layout to Root-cause fixes.**
   D1 is now decided, and the convergence is a root fix with no layout
   change. Chat settings layout keeps the "Use saved defaults" button.
2. **Chat settings layout also depends on Switch model.** Its
   `[Change Alt+M]` needs the pick-only mode that Switch model builds.
3. **Thinking is not on the switcher's value row.** The judge listed it
   "when supported", but every editable field on the quick surface must be in
   the quick mask (ADR-095 amendment), and D3 added only `max_tokens`.
   Mockup (a) already leaves Thinking out.
4. **task-194 closes with Switch model.** task-194 is about popover display
   names, and Switch model rebuilds the popover's rows. Field table fixes the
   chips.

## 9. ADRs

- **ADR-095** is amended on 2026-09-26 for D1 and D3.
- **ADR-012** is amended on 2026-09-26 for D2. D4 reaffirms it unchanged.
- **Constraints this program must hold:**
  - ADR-006: Settings owns persisted defaults and Console resolves them. An
    unsupported control shows "unavailable" or is omitted.
  - ADR-033: three honestly labelled commit models, a truthful State badge,
    no new root state, and handoffs through `PendingHandoffStore`.
  - ADR-002 and ADR-020: manual Discover/Save stays, and so does the
    catalog-refresh consent gate.
  - ADR-031: keys.
  - ADR-146: custom endpoints; `custom` and `custom_2` stay.
  - ADR-066: legacy aliases can be hidden, but not deleted.
  - ADR-150 and ADR-161: geometry comes from tokens.
  - ADR-097: ratchets.

## 10. Out of scope

- **Declined from "Switch & Tune":** inline key entry in Console (D4), a
  Console "All models" scope that writes `chat_defaults`, deleting wizard
  steps, and deleting Discover / Save selected.
- **Favourites, pins and Alt+1..9,** until testing shows a need.
- **Probes and paid tests:** automatic cloud probes, and any change to the
  paid 1-token test (its consent flow stays as it is).
- **Layouts for terminals smaller than 211x44.**
- **Schema migrations and new indexes.**

## 11. Verification

- Pilot tests pin the §2 keystroke budgets and the pairs-only rule (no path
  picks a provider without a model).
- 1-row controls get paint probes driven by real keypresses, not `.value`
  checks (lessons-live-verification).
- Live runs at 211x44 and 235x52 in tmux use a scratch `TLDW_CONFIG_PATH`
  profile, never the real config.
- Contrast is measured per shipped theme in a running terminal.
- Tests that pin today's defects are rewritten on purpose and named in the
  PR: `test_console_session_settings.py:4555-4560` and
  `test_settings_configuration_hub.py:4386-4420`.
- After the last phase, task-1379 re-runs the critique like-for-like against
  14/40 and 14/40.
