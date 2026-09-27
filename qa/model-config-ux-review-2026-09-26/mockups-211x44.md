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
