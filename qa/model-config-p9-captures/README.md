# Phase 9 (model-config riders) full-screen captures

Live captures of the surfaces Phase 9 changed, by keyboard, with a click or a mouse-wheel
scroll where a row says so. The set was taken 2026-10-18 at `79633ad177` (product code equal
to `79380c721d`); the captures the capture fix wave (`704a6ffc03..08afcd4e5c`) changed were
retaken the same day at `6212bba292` (product code equal to `08afcd4e5c`; see "Retaken at
6212bba292" below), and `06d2` is new there. Each moment was captured at 211x44, the tmux
window was resized to 235x52 and captured again, then resized back. `.txt` is `tmux
capture-pane -p`; `.ansi.txt` is the same moment with `-e` colour escapes (focus shows
there: a focused button or tab is bold and underlined on the accent fill, and a focused
Settings field has a █ at its left edge). Clock times in the captures are local (PDT).

## Procedure

- **Isolation.** Every process that imported `tldw_chatbook` (the profile writer, the
  schedules seeder and the app) ran under `env -i` with `HOME`, `XDG_CONFIG_HOME`,
  `XDG_DATA_HOME`, `XDG_CACHE_HOME`, `XDG_STATE_HOME` and `TLDW_CONFIG_PATH` in a fresh
  scratch directory per launch, the null keyring backend
  (`PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring`), `TERM=xterm-256color`,
  `COLORTERM=truecolor`, `LANG=en_US.UTF-8` and `PATH=/usr/bin:/bin`, so `OPENAI_API_KEY`,
  `GROQ_API_KEY`, `SWARMUI_TOKEN` and the rest were unset. `PYTHONPATH` was the frozen
  tree of the commit under test.
- **Sandbox.** The same processes ran under a `sandbox-exec` profile that denied reads and
  writes under `~/.config/tldw_cli`, `~/.local/share/tldw_cli` and `~/.config/tldw-live`,
  and every outbound connection except to loopback. macOS's AirPlay Receiver
  (ControlCenter) listens on `*:5000`, one of the template's local-server ports, so the
  profile denied port 5000 on loopback too. In batch A it allowed loopback only to an
  explicit port list, because a port-specific deny under an allow of every loopback port
  did not take effect there. A positive control under the profile, with stdlib Python,
  ran before each batch: a connect to 1.1.1.1:443 and to 127.0.0.1:5000 (and to the other
  spellings of port 5000 tried, [::1]:5000 and localhost:5000) failed with
  `PermissionError: [Errno 1] Operation not permitted`, `os.stat` of the three profile
  directories failed the same way, and loopback stayed open (a stub connected, or a port
  with nothing listening gave `ConnectionRefusedError`). The app tried port 5000 once, in
  `a1` at `01f`'s first pick-mode open, and was denied; its Switch model and pick-mode
  lists read the same NOT RUNNING rows as in the Phase 8 set.
- **Launch.** `python -m tldw_chatbook` (never `-m tldw_chatbook.app`, TASK-35302) in its
  own tmux server per batch, in a 211x44 window started with `-f /dev/null` (no user
  tmux.conf) and with the
  scratch `HOME` as its working directory: from the Phase 9 worktree as the working
  directory, `sys.path[0]` found `tldw_chatbook` there instead of in the frozen tree.
  Before each launch `lsof` showed nothing listening on the template's local-server ports
  (1234, 5001, 5678, 7801, 8000, 8080, 8188, 11434) or on the batch's refused ports; only
  ControlCenter on 5000.
- **Profiles.** Each one is the shipped template (`CONFIG_TOML_CONTENT`) deep-merged with
  the launch's edits and written with `toml.dumps`, under the same environment and sandbox,
  before the app started. Every profile sets `users_name = "verify_capp9_<launch>"`, the
  splash off, `[model_catalog] auto_refresh_enabled = false`, `[first_run] setup_completed
  = true` and `[console.onboarding] first_send_completed = true`, except where a launch
  says otherwise. `a1` (`01`): llama.cpp · model-a at 127.0.0.1:18612, where nothing
  listens, `providers.Llama_cpp = ["model-a", "model-c"]`, a saved `reasoning_effort =
  "minimal"` for model-a, and `first_send_completed = false`. `a2` (`02`): vLLM ·
  vllm-model-a. `a3` (`03`): llama.cpp · model-a at the 18611 stub. `b1` (`04`): OpenAI ·
  gpt-5.6-terra with a made-up key and `api_base_url` the 18621 stub, a made-up OpenRouter
  key, and a made-up legacy `[API] groq_api_key` under `credential_source = "none"`. `b2`
  (`05`): the same OpenAI key against the 18622 stub. `b3` (`06`): llama.cpp · model-a at
  the 18623 stub and a made-up SwarmUI token. `c1` (`07`): only the users name, the splash
  and the catalog keys, so the first-run wizard opens. `c2` (`08`): a schedules database
  seeded through `ScheduledTasksDB` before the launch: "Weekly digest" (weekly, due
  2026-10-14 09:00 UTC and dispatched, so completed with its next run 2026-10-21),
  "Backup" (one-time, fired 2026-10-15 12:00 UTC) and "Paused check-in" (daily,
  disabled). `c3` (`09`): llama.cpp · model-a at 127.0.0.1:18633, where nothing listens.
  `c4` (`10`): ONNX Runtime (local) · onnx-model. `c5` (`11`): provider
  `custom-ep:gone[/]` (a dangling id that is Textual markup) and a custom endpoint
  "GPU [b]box[/b]" at the 18632 stub. `c6` (`12`): `default_theme = "auto_probe"`, a user
  theme whose variables use `auto` values. No key is real, and no capture shows one.
- **Stubs.** Loopback stdlib `http.server`s answering `GET /v1/models` (and `/models`),
  each logging method, path and whether an Authorization header was sent, never its
  value. First run: 18611 (model-a, model-b, served-y; 9 requests in `a1` and 4 in `a3`,
  all without a key), 18619 (other-z; 2), 18613 (gpu-x, gpu-y, answering after 2 s; 2 in
  the kept `a3` launch), 18621 (gpt-5.6-sol, gpt-4.1-2025-04-14; 4 `GET /v1/models` with a
  key, one from `04d`'s **t** and three from `04h`'s), 18622 (401 to every request; 1 with
  a key), 18623 (model-a, model-x, model-y; 1 without a key) and 18632 (model-a; 1
  without a key). Nothing listened on 18612, 18614, 18631 or 18633. Retakes: 18621 logged
  3 requests with a key, 18622 1 with a key and 18623 1 without; `c2` has no stub.
- **Input.** `tmux send-keys`, one key per command, at least 0.2 s apart (a burst runs
  editing keys out of order); text with `send-keys -l`. Ctrl+digit cannot be sent, so
  screens were reached with F2, F4 or the palette (**Ctrl+P**, the command typed, **Down**,
  **Enter**). A click is an SGR press and release at a label's first cell, read from the
  latest 211x44 capture; a wheel scroll is SGR wheel-down events over a region.
- **Timing.** Notices time out after a few seconds; a row says where one had timed out by
  the 235x52 shot. Waits longer than the plan's are in "Plan corrections".
- **No key check or send reached a real provider.** No message was sent. The key checks
  went to the 18621 and 18622 stubs and to 127.0.0.1:18633 (refused); the wizard's Test
  went to 127.0.0.1:18631 (refused); every model listing went to loopback.

## Captures

| File prefix | What it shows |
|---|---|
| `01a-console-ready-line-not-tested-and-chrome` | Launch `a1` (`first_send_completed = false`): llama.cpp at the refused 127.0.0.1:18612. Console "Chat 1", nothing tested yet. The shot was taken 3 s after the composer drew. The empty transcript reads "Setup complete — llama.cpp · model-a. Ready · not tested — type a message to begin.", and the header badge reads "Ready · not tested". The nav bar's rows 1-2 end in `48;2;30;30;30`, and row 3 carries it. The separator row is (119,119,119) on (30,30,30), 3.72:1. The active "⌃2 Console" tab's round outline is (140,194,235) on (0,101,190), 3.06:1. The resting tabs sit on (30,30,30). The left rail's `───` section rules are (119,119,119) on (30,30,30), 3.72:1. All of this holds at both sizes. |
| `01b-rail-footer-after-chat-settings-esc` | A click on the rail's "Model ▸" header opens it ("Model ▾": Temperature 0.60, Max tokens 4096, Streaming On, Ready · not tested, System: none ▸, Change Alt+M) and focuses its toggle. Then **Ctrl+O** and **Esc**: Chat settings closes, and focus is back on the toggle (bold underline). The footer at 211x44 reads "Esc composer · F6 panes \| Ctrl+Shift+F focus \| F1 help \| Alt+M switch model \| Ctrl+O chat settings \| Alt+I inspect \| Alt+A approval \| Alt+C context rail \| Ctrl+K switch session \| F6 · Ctrl+P · Ctrl+Q". At 235x52 it reads "… \| Ctrl+K switch session \| Ctrl+T new tab \| Ctrl+P palette \| F6 · Ctrl+Q". Neither has "F6 next pane". |
| `01c-chat-settings-model-view-saved-minimal` | **Ctrl+O**: the title reads "Chat settings · Chat 1", and the tabs read "● Model and generation  Context and memory", with "Response rules" on the line under the scope line. The MODEL row reads "model-a · llama.cpp  this chat  Ready · not tested · context unknown  Change Alt+M". Reasoning effort's Select reads "Minimal ▼": the 13-cell Select shows only the first word of "Minimal (not supported here)" (the label is whole in `01d`'s open list). Its Source is "model default", and its help reads "Minimal is not supported here, so it is not sent.". Focus is on Temperature (accent bar, value on the focus fill). The Source words line up in one column, column 52 at 211x44 and 64 at 235x52. |
| `01d-chat-settings-reasoning-effort-options` | A click on that Select opens its list ("▲"): "default", "None", "Low", "Medium", "High", "X-High", then "Minimal (not supported here)" last, wrapped over two rows of the list. There is no plain "Minimal". The 235x52 resize kept the list open. **Esc** then closed it, and focus stayed on the Select. |
| `01e-chat-settings-context-view-tabs-whole` | A click on "Context and memory": the tabs read "Model and generation  ● Context and memory", each whole, and focus is on Budget strategy ("█ Automatic ▼"). The scope line reads "Applies to this chat only · defaults live in Settings ▸ Console Behavior (F4)". The view grows the dialog to the top of the screen. |
| `01f-chat-settings-model-view-after-alt-m-esc` | **Alt+M** from Budget strategy opened pick mode (not kept: model-a "Not ready · refused :18612 ● CURRENT", with llama.cpp also under NEEDS SETUP). **Esc** returns to the Model view with "Change Alt+M" focused (bold underline). The tabs read "● Model and generation  Context and memory", each whole, at both sizes. From here through `01j`, at both sizes, nothing of the Console shows around the dialog: only the nav bar and the footer, with a 66-cell box edge ("│ … │" over "└───…───┘") under the dialog. Triage at the phase base `111f8e48fc`, closing the dialog at this point, found the Get started card under it ("1. ● Reconnect the provider server", "llama.cpp: Not ready · refused …"), whose fill covers the Console as in `07e`; it is gone again from `01k`, after the apply to :18611. |
| `01g-chat-settings-focused-selected-tab` | **Shift+Tab** ×3 from Change: Response rules, then Context and memory, then Model and generation. The tab paints "● Model and generation" whole beside a whole "Context and memory". It is bold (225,230,235) on (25,68,102) with **no underline**, the same as its unfocused selected style in `01f`, and no cell on the screen is underlined. That focus is there was checked with **Tab** (the underline moves to Context and memory) and **Shift+Tab** (output byte-identical to this shot). |
| `01h-chat-settings-endpoint-listed-at-session-server` | A click opened "▶ Connection · 127.0.0.1:18612 · no key needed · change it in Settings ▸ Providers & Models" (focus on its title). Then **Tab** to the Endpoint field (no Configure credential… shown), **Ctrl+A**, **Ctrl+K**, `http://127.0.0.1:18611` typed, **Tab** ×2 to "Test connection & list models" (bold underline), and **Enter**. The shot 3 s later reads "3 models listed", "Ready · reachable 09:52", "Endpoint · Reachable", "Model · Confirmed" and "Generation · Not tested". The title reads "Chat settings · Chat 1 · 1 unsaved edit". The footer reads "Esc close (asks: 1 unsaved)  Use saved defaults  Default for new chats (Ctrl+N)  Apply to this chat (Ctrl+Enter)", under "Unavailable: llama.cpp failed its last connection test (connection refused).". A notice reads "Connection test succeeded; 3 models listed.". The body is scrolled, so the MODEL row is above the fold, and no control shows focus. The 18611 log has `GET /props?model=model-a&autoload=false` (09:52:12, during the typing) and one `GET /v1/models` (09:52:18), both `auth=no`. Test generation was not pressed. |
| `01i-pick-mode-drops-a-listing-once-the-endpoint-moves` | **Shift+Tab** ×8 to the Endpoint field (the Test button dropped focus when it disabled itself for the listing: TASK-35566), **Ctrl+A**, **Ctrl+K**, `http://127.0.0.1:18619` typed, then **Alt+M**. Pick mode reads "Find type to search 1 model in 1 provider · Enter picks". RECENT has "▶ model-a  llama.cpp  ?  Ready · reachable 09:53  ● CURRENT", and READY PROVIDERS has "other-z  llama.cpp  ?  Ready · reachable 09:53". model-b and served-y are not offered, and llama.cpp is not under NEEDS SETUP. The 18619 log has `GET /props?model=model-a&autoload=false` and `GET /v1/models` (09:53:56-57). **Esc** then closed pick mode, with Change Alt+M focused. |
| `01j-unsaved-prompt-names-the-endpoint` | A click on the Endpoint field, **Ctrl+A**, **Ctrl+K**, `http://127.0.0.1:18611` typed, then **Esc**: "1 unsaved edit to this chat: Endpoint." over "Enter apply · d discard · Esc keep editing", with the buttons "Apply to this chat", "Discard" and "Keep editing". **Enter** then applied. |
| `01k-console-ready-line-reachable` | 2 s after that **Enter**: "Setup complete — llama.cpp · model-a. Ready · reachable 09:52 — type a message to begin.", and the header badge reads "Ready · reachable 09:52". At both sizes the notices are "Thinking controls on llama.cpp need llama-server started with --jinja; per-request reasoning_budget_tokens needs llama.cpp b9982 or newer." and "This chat updated". The "…saved for this session only…" notice did not show (TASK-35560). |
| `01l-switch-model-rows-are-the-session-servers-listing` | **Alt+M**, shot 4 s later. "Switch model · now: llama.cpp · model-a" reads "Find type to search 3 models in 1 provider · Enter applies". RECENT has model-a "Ready · reachable 09:56 ● CURRENT". READY PROVIDERS has model-b and served-y, each "Ready · reachable 09:56". llama.cpp is not under "NEEDS SETUP · Enter opens the fix or explains it", and model-c is not offered. NOT RUNNING lists "Aphrodite Engine · Ollama · TabbyAPI · vLLM · Custom OpenAI-compatible · Custom OpenAI-compatible #2" and "MLX-LM (Apple silicon) (not checked)". The key line reads "Enter apply to this chat · Tab edit values · Ctrl+N default for new chats · Ctrl+O chat settings · Esc cancel", with "Applies to: this chat only". The 18611 log has one `GET /v1/models` (09:56:34). Nothing listens on 18612, so a probe there leaves no log line. |
| `01m-switch-model-ctrl-n-names-the-configured-server` | `served-y` typed, then **Ctrl+N**. Under "Find served-y" the error line reads "Unavailable for new chats (http://127.0.0.1:18612): llama.cpp failed its last connection test (connection refused)." Below it, "READY PROVIDERS · matches from every provider's catalog" lists "▶ served-y  llama.cpp  ?  Ready · reachable 09:56". After the launch, config.toml still had `chat_defaults.model = "model-a"`, the llama.cpp `api_url` was still :18612, and "18611" appeared nowhere. **Esc** closed it. |
| `01n-pick-mode-offers-the-session-listing-without-retyping` | **Ctrl+O**, then **Alt+M** (3 s): pick mode reads "3 models in 1 provider · Enter picks", listing model-a (RECENT, ● CURRENT), model-b and served-y, all "Ready · reachable 09:57". The Endpoint was not retyped. Opening Chat settings sent `GET /health`, `/props` and `/props?model=model-a&autoload=false` to 18611 (09:57:03), and no listing. Pick mode's own `GET /v1/models` followed at 09:57:05. |
| `01o-chat-settings-pick-keeps-the-session-endpoint` | `served-y` typed, then **Enter**: back in Chat settings. The title reads "Chat settings · Chat 1 · 1 unsaved edit", and the MODEL row reads "served-y · llama.cpp  edited *  Ready · reachable 09:57 · context unknown  Change Alt+M". Reasoning effort now reads "default ▼ provider". The Connection title reads "▶ Connection · 127.0.0.1:18611 · no key needed · change it in Settings ▸ Providers & Models". The Console is painted behind the dialog again. The 18611 log has `GET /props?model=served-y&autoload=false`. |
| `01p-unsaved-prompt-names-only-model-after-a-pick` | **Esc**: "1 unsaved edit to this chat: Model." over "Enter apply · d discard · Esc keep editing". **d** then discarded (the status row reads Model: model-a). |
| `02a-chat-settings-vllm-endpoint-unsaved-model-view` | Launch `a2`, Console on vLLM · vllm-model-a. **Ctrl+O**, **Tab** ×5 to "▶ Connection · localhost:8000 · no key needed · …" (bold underline), **Enter**, **Tab** to the Endpoint field (it held `http://localhost:8000/v1/chat/completions`), **Ctrl+A**, **Ctrl+K**, then `http://127.0.0.1:18614/v1` typed. The Connection title reads "127.0.0.1:18614", with "Not ready · endpoint unsaved" and "Provider setup needed: save the endpoint in Chat settings". Above the footer: "Save endpoint & use model: new chats start on this vLLM endpoint and model.". The footer reads "Esc close (asks: 1 unsaved)  Use saved defaults  Save endpoint & use model  Default for new chats (Ctrl+N)", with no Apply to this chat. The MODEL row is above the fold at both sizes. A wheel-up after the shots read it as "Not ready · endpoint unsaved · context unknown". Nothing on the footer was pressed. No stub ran on 18614 (nothing listens there). |
| `02b-chat-settings-vllm-context-view-says-where-to-save` | A click on "Context and memory": the tabs read "Model and generation  ● Context and memory", with Budget strategy focused. Above the footer: "Save the vLLM endpoint in Model and generation to continue.". The footer has only "Esc close (asks: 1 unsaved)", and Apply is hidden. Afterwards **Esc** ("1 unsaved edit to this chat: Endpoint.") and **d**. |
| `02c-lab-gguf-model-source-one-row` | **F2**: Lab ▸ Models, "Llama.cpp Configuration" ("Servers: none running"). "Model source:" and "External GGUF ▼" are on one row, with a blank row above and below at both sizes, and no scroll was needed. The Lab rail's right edge `│` is (119,119,119) on (36,47,56), 3.05:1. Nothing was started, and no request was made. |
| `03a-created-entry-notice-under-the-close-guard` | Launch `a3`, retaken with the 18613 stub's delay at 2 s (see "Plan corrections"). **Ctrl+O**, **Ctrl+A**, **Ctrl+K**, `0.5` typed in Temperature ("0.5 edited *", 1 unsaved). Then **Tab** ×6 to the Connection title, **Enter**, **Tab** ×2 to New endpoint…, and **Enter**: "New endpoint from template" opens with the llama_cpp template and focus on Display name. `GPU box` typed, **Tab** ×2 to Endpoint, **Ctrl+A**, **Ctrl+K**, `http://127.0.0.1:18613` typed (Models held model-a, Params was empty), **Tab** ×4 to Create, and **Enter** ("Listing the models GPU box serves…"). **Esc** 0.7 s later. The guard reads "1 unsaved edit to this chat: Temperature." over "Enter apply · d discard · Esc keep editing", and pick mode did not open over it. The notice "GPU box is saved. Pick its model with Change (Alt+M)." shows at 211x44, about 2 s after Create, and is still up at 235x52. The 18613 log has one `GET /v1/models` (10:05:49, `auth=no`). |
| `03b-change-offers-the-created-entrys-models` | **Esc** (Keep editing), then **Alt+M**: pick mode reads "Find type to search 6 models in 2 providers · Enter picks". RECENT has model-a (llama.cpp, ● CURRENT). READY PROVIDERS has model-b and served-y (llama.cpp, "Ready · reachable 10:06"), and GPU box's model-a, gpu-x and gpu-y ("Ready · not tested"). The 18613 log has pick mode's `GET /v1/models` (10:06:05). Afterwards **Esc** (Change focused), **Esc** and **d**. config.toml kept `[custom_endpoints.gpu-box]` with display_name "GPU box", family "llama_cpp", base_url `http://127.0.0.1:18613`, `models = ["model-a"]` and created_from "llama_cpp". |
| `04a-switch-model-at-rest-whole-provider-names` | Console on OpenAI · gpt-5.6-terra (made-up OpenAI key, endpoint the 18621 stub; made-up OpenRouter key), plain **Alt+M**, read 4 s later. Find "type to search 28 models in 2 providers · Enter applies"; RECENT "▶ gpt-5.6-terra OpenAI ? Ready · not tested ● CURRENT"; READY PROVIDERS · top 3 each: OpenAI's gpt-5.6-sol, gpt-5.6-luna, gpt-4.1-2025-04-14, "… 10 more OpenAI models", OpenRouter's openai/gpt-4o-mini, anthropic/claude-3.7-sonnet, google/gemini-2.0-flash-001, "… 11 more OpenRouter models", every row "Ready · not tested". NEEDS SETUP: eight "(any model) … Not ready · no key Enter: add key in Settings" rows, every name, note and readiness word whole, among them "ByteDance Seed (BytePlus)" and "Cloudflare Workers AI", then "… 38 more providers need setup · type a name to find one". NOT RUNNING: "Aphrodite Engine · Ollama · TabbyAPI · vLLM · Custom OpenAI-compatible · Custom OpenAI-compatible #2" and "MLX-LM (Apple silicon) (not checked)". No row is a port-5000 server: the template's only 5000 endpoint is `[api_settings.oobabooga]`, which no capture lists. Footer "Ctrl+P palette \| Ctrl+Q quit". At 211x44 the status row's cost chip reads "Context 0% · Current $0.", cut at the screen edge (TASK-35561). |
| `04b-find-openai-lists-openais-rows-first` | "openai" typed into Find. RECENT keeps "▶ gpt-5.6-terra OpenAI … ● CURRENT", and the ▶ highlight is on that row. The heading reads "READY PROVIDERS · matches from every provider's catalog". OpenAI's 13 other models come first (gpt-5.6-sol … gpt-4o-mini-2024-07-18), then OpenRouter's openai/gpt-4o-mini and openai/gpt-4.1. NEEDS SETUP lists "(any model) Azure OpenAI", "@cf/openai/gpt-oss-120b Cloudflare Workers AI" and "openai/gpt-oss-120b Hugging Face". "Custom OpenAI-compatible · Custom OpenAI-compatible #2" are under NOT RUNNING, not NEEDS SETUP. |
| `04c-find-tabby-filters-not-running` | **Backspace** ×6 (one per command), then "tabby" typed. Find "tabby"; "NOT RUNNING · local servers never set up · rechecked on open" names "TabbyAPI" only (its template endpoint is localhost:8080, which nothing listens on); below it "TYPED MODEL ID · not in any catalog" with "▶ tabby OpenAI ? Ready · not tested typed id · in no catalog" highlighted. **Esc** followed (no Enter). |
| `04d-settings-key-check-unlisted-model` | **F4**, **F6** (rail), **Down**, **Enter** (Providers & Models, █ on "> Providers & Models"), **t**, read 3 s later. Toast at the bottom right, at both sizes: "Ready · reachable 09:47 — key accepted (2 models listed) · generation not tested. gpt-5.6-terra is not listed by the server — pick a listed model." Key check "Ready · reachable 09:47" with the compact "Test (t)"; API key "Paste to replace  Clear  saved in config"; Endpoint "http://127.0.0.1:18621/v1 saved in config"; the State line is one row ("State: No changes · edits save with s \| Applies to new and unused open chats · used chats keep theirs (Console: Alt+M)"). Its round border is (254,166,43), but so is the border of 04k's information toast: on textual-dark `$accent` and `$warning` are the same colour, so a capture cannot tell an information toast from a warning one. The stub logged one `GET /v1/models auth=yes`. |
| `04e-settings-inspector-identity-and-test-rows` | **F6**, **F6**: the Inspector holds focus. It has no visible focus style; a third F6 put the █ on the rail, and **Shift+F6** brought focus back. The Key block is below the fold, so the Inspector was wheeled down ×11 at 211x44, plus ×6 more after the resize at 235x52. Key block "Provider readiness: OpenAI / gpt-5.6-terra — Ready · reachable 09:47". Test rows: "Readiness   Ready · reachable 09:47", "Model       gpt-5.6-terra · not listed by the server — pick a listed model", "Config      OpenAI is configured", "Key         saved in config · key accepted (2 models listed)", "Endpoint    http://127.0.0.1:18621/v1 · model listing reached", "Generation  not tested". The toast has timed out. |
| `04f-switch-model-verified-only-for-listed-rows` | Palette "Tab Navigation: Switch to Console" (**Ctrl+P**, typed, **Down**, **Enter**), then **Alt+M**, read 3 s later. Header badge "Ready · reachable 09:47". RECENT "▶ gpt-5.6-terra … Ready · reachable 09:47 ● CURRENT"; READY PROVIDERS: gpt-5.6-sol "Ready · verified 09:47", gpt-5.6-luna "Ready · reachable 09:47", gpt-4.1-2025-04-14 "Ready · verified 09:47"; OpenRouter's rows "Ready · not tested". The status row reads "Provider: OpenAI     Model: gpt-5.6-terra" and has no readiness word. **Esc** followed. |
| `04g-settings-overview-after-key-check` | **F4**, **/**, "overview", **Enter** (█ on "> Overview (view)"). Status: "Configuration: OpenAI / gpt-5.6-terra; Status: Ready · reachable 09:47"; "Last connection test: Readiness: Ready · reachable 09:47; Endpoint: http://127.0.0.1:18621/v1 · model listing reached" ("reached" wraps at 211x44, one line at 235x52). The "▶ Advanced / Diagnostics" and "▶ Where changes happen" frames are (119,119,119) on (30,30,30). |
| `04h-settings-tested-draft-model-rows-marked-draft` | **/**, "providers & models", **Enter**; **F6** (█ on Provider); **Tab** ×4 (█ checked on API key, Env var, Endpoint, Model); "gpt-5.6-sol" typed ("1 found · Enter picks · Esc cancels"); **Enter**, **Esc**, **t**; **F6** ×3 (rail, detail, Inspector). The first toast had timed out before the shot, so **t** was pressed twice more, and after the third the Inspector was wheeled ×11. Toast (211x44 only; timed out by the 235x52 shot): "Ready · verified 09:52 — key accepted (2 models listed) · generation not tested."; "State: Draft — save with s · 1 unsaved \| …"; Model "gpt-5.6-sol  edited *"; Inspector "Unsaved changes", Key block "Provider readiness: OpenAI / gpt-5.6-sol — Ready · verified 09:52". Test rows: "Readiness   Ready · verified 09:52", Config, Key, Endpoint, then "Model       gpt-5.6-sol (draft) · listed by the server" (after Endpoint here; in `04e` the unlisted Model row came right after Readiness), "Generation  not tested". |
| `04i-settings-save-redraws-test-rows` | **s**: "State: No changes · edits save with s \| …"; Model "gpt-5.6-sol  new-chat default"; Key check "Ready · verified 09:52" with "Provider settings saved: new chats and open chats nobody has used yet take them; chats with work keep their own settings (change them in Console with Alt+M)." under it; Inspector "No unsaved changes". The save put the Inspector back at its top, so it was wheeled ×11 at each size. Test rows: "Readiness   Ready · verified 09:52" … "Model       gpt-5.6-sol · listed by the server", with no "(draft)". The "Provider and model settings saved: …" toast showed at 2 s and had timed out by the shot. The scratch profile now has `chat_defaults` provider "openai", model "gpt-5.6-sol". |
| `04j-settings-revert-dialog` | (retaken at `6212bba292`) Launch b1 replayed from a fresh profile through 04a-04i (made-up OpenAI key, endpoint the 18621 stub; 04i's **s** saved gpt-5.6-sol). Then **F6** ×2 (rail "█  > Providers & Models", then █ on Provider), **Tab** ×3 (█ checked on API key, Env var, Endpoint), **End**, "x" (Endpoint "http://127.0.0.1:18621/v1x  edited *", Key check back to "Ready · not tested"; the Inspector's guide read "Focused setting: Endpoint"), **Esc** (no █ anywhere; the guide changed to "Focused setting: None — Tab to a setting"), **r**. Dialog "Revert Settings changes" / "Discard all unsaved changes to Providers & Models?"; "Keep editing" has focus (bold underline on (1,120,212)), beside "Discard changes" (bold, no underline, on (185,60,91)). Behind it: "State: Draft — save with s · 1 unsaved \| …", rail "> Providers & Models *", Inspector "Unsaved changes" and, under "Focused field guide", "Focused setting: None — Tab to a" / "setting" (wrapped onto two lines at 211x44, one line at 235x52; not cut). |
| `04k-settings-after-discard-focus-stays-released` | (retaken at `6212bba292`) **Tab** ("Discard changes" focused: bold underline on (185,60,91)), **Enter**, **Enter**. Toast "Settings category changes reverted." at both sizes; Settings still on screen ("Mode: Providers & Models"); "State: No changes · edits save with s \| …"; Endpoint "http://127.0.0.1:18621/v1  saved in config"; Key check "Ready · verified 14:47"; Inspector "No unsaved changes". No █ at either size. The nav bar's ANSI row is byte-identical to the check run's 04d and 04g (not kept; same md5), so "⌃1 Home" has no focus style. The Focused field guide reads "Focused setting: None — Tab to a" / "setting" at 211x44 and "Focused setting: None — Tab to a setting" at 235x52 (first run: "Focused setting: Endpoint"). |
| `04l-settings-f1-help-says-the-draft-is-kept` | (retaken at `6212bba292`) **F1**: the modal "Settings: Providers & Models" lists "- Save contract: Draft — save with s. Switching categories keeps the" / "draft; the rail marks it with *." (two lines at both sizes), then Scope, Runtime owner, Writes here, Boundary, Recovery, "Test provider (t)" and the s / r / t shortcuts, with a "Close" button. Behind it the guide reads "Focused setting: None — Tab to a setting". **Esc** followed. |
| `04m-settings-fallback-search-status` | Nothing focused, **/**, "fallback temperature". The rail's search status reads "Filter: fallback temperature \| 1 match \| Enter opens Console Behavior › Global fallback Temperature (Interface)", on five rail lines, over the one match: "Console Behavior", indented under "Interface". |
| `04n-settings-fallback-search-lands-on-the-fallback` | **Enter**. At 1.5 s the screen still showed Providers & Models and the filter; this shot is about 5 s after the Enter. "Mode: Console Behavior"; the filter is cleared ("Filter categories (/)", "No filter \| / focus"); █ on "Temperature █ 0.6  Console Behavior  Lower stays focused; higher is more varied." under "Global fallback defaults"; footer "Esc, s save category \| Esc, r revert category \| …". No card frame is drawn inside the detail pane, only the pane's own (119,119,119) border. **Esc** followed. |
| `04o-settings-groq-legacy-key-under-credential-none` | **/**, "providers & models", **Enter**, **F6** (█ on Provider), "Groq" typed ("1 found · Enter picks · Esc cancels"), **Enter**. Provider "Groq  edited *"; API key "Paste API key  missing  paste one to save it in config", although `[API] groq_api_key` holds a made-up key; Env var "GROQ_API_KEY  not set"; Endpoint "https://api.groq.com/openai/v1  built-in"; Key check "Not ready · no key"; Model "llama-3.3-70b-versatile  edited *"; "State: Draft — save with s · 2 unsaved". Discarded afterwards (**Esc**, **r**, **Tab**, **Enter**): back to OpenAI, "State: No changes". |
| `05a-settings-key-check-401` | A fresh launch (endpoint the 18622 stub, which answers 401). **F4**, **F6**, **Down**, **Enter**, **t**. Toast (both sizes): "Not ready · key rejected — OpenAI rejected the API key (unauthorized); enter a valid key, then Test again."; Key check "Not ready · key rejected". The stub logged one `GET /v1/models auth=yes`. |
| `05b-console-composer-says-the-key-was-rejected` | (retaken at `6212bba292`) Launch b2 from a fresh profile (endpoint the 18622 stub, which answers 401), 05a replayed (**F4**, **F6**, **Down**, **Enter**, **t**: "Not ready · key rejected"). Then palette "Tab Navigation: Switch to Console" (**Ctrl+P**, typed, **Down**, **Enter**). Header badge "Not ready · key rejected"; the composer's reason strip reads "Send blocked — key rejected; update the API key ›" whole, with no "…", at both sizes. Toast "Switched to Console" at both sizes; "Set up provider" in the transcript area. "add an API key" appears nowhere. |
| `05c-console-inspector-setup-row-key-rejected` | (retaken at `6212bba292`) **Alt+I**. At 1 s the Inspector had not opened yet; it was open about 3 s later. It starts with "What happens if I send now?" (Where: Default › Chat 1, Scope: Everything available, "Run: Recovery required", Sources: None staged, Approvals: None pending), then Environment, Sources and Scope. The Setup rows are in its Run section below the fold, so the Inspector was wheeled ×8 at 211x44; the scroll held through the resize, so the 235x52 shot is scrolled too. Rows: "Run recipe: OpenAI / gpt-5.6-terra / sources not staged / tools 0 / approvals 0", "Live work: No active work", "Setup: Not ready · key rejected", "Blocked impact: Send is blocked until setup is finished. - Provider setup needed: OpenAI credential was rejected", "Next action: Set up provider", "Provider: blocked - Provider setup needed: OpenAI credential was rejected". The composer strip reads "Send blocked — key rejected; update the API key ›". "Provider configuration required" appears nowhere. |
| `06a-settings-image-gen-backend-rows-top` | A fresh launch (llama.cpp · model-a on the 18623 stub, a made-up SwarmUI token). **F4**, **/**, "image gen", **Enter**; "Loading Image Gen settings…" was gone within 1 s. Default backend "SwarmUI (local)". The rows "SD.cpp (local)", "SwarmUI (local)" ("Configured ▐✓▌ On ★ Default Test"), "ComfyUI (H3 image edit)", "OpenRouter", "Novita" and "Together" are whole, each followed by at least one blank cell, with "Not configured"/"Configured" starting in one column. Each status runs straight into its switch ("Not configured▐ ▌ Off  Test"). There are three blank rows between rows. |
| `06b-settings-image-gen-backend-rows-scrolled` | Wheel ×6 over the panel: "ModelStudio", "fal.ai" and "Gemini (AI Studio)" whole, in the same status column. |
| `06c-settings-image-gen-swarm-clear-counts-one` | **/**, "swarm token", **Enter** (█ on Swarm token, "Local config key saved; paste a replacement to change it"), **Tab** ("Clear" focused, bold underline), **Enter**. The State line reads "State: Draft — save/revert below · 1 unsaved \| Defaults affect future Console image generations."; the Swarm token reads "Paste a key/token to save locally in config", with "missing" under it. Then **r** reverted it at once, with no dialog. |
| `06d-settings-video-gen-backend-rows` | (retaken at `6212bba292`) Launch b3 from a fresh profile (llama.cpp · model-a on the 18623 stub, a made-up SwarmUI token), 06a-06c replayed (06c's Clear reverted with **r**: "State: Draft — save/revert below \| …", no unsaved count). Then **/**, "video gen", **Enter**. "Mode: Video Gen"; "MiniMax H3 (cloud)      Not configured▐ ▌ Off", "ComfyUI (local server)  Configured    ▐ ▌ Off", "SD.cpp video (local)    Not configured▐ ▌ Off ★ Default": whole, a blank cell before each status, one status column. Under "Backend settings" the editor is now drawn: "▶ MiniMax H3 (cloud)" and "▶ ComfyUI (local server)" are each a whole round frame (top edge, title row, blank row, bottom edge); "▼ SD.cpp video (local)" is expanded. At 211x44 only its top edge and title show above the pane's bottom; at 235x52 its body shows too ("Binary path", "Diffusion model path", "Timeout (seconds) 7200", "Advanced keys for SD.cpp video (local) live in config.toml -> [video_generation.stable_diffusion_cpp] (not editable here).") but its bottom edge is still below the fold. That bottom edge is drawn once the pane scrolls (06d2 at 235x52). "Generation defaults" is below the fold at both sizes (in the first run it was on screen, under the editor's lone top edge). |
| `06d2-settings-video-gen-save-reachable` | (new at `6212bba292`) From 06d's screen (focus on the rail's "> Video Gen" row): **F6** (into the detail pane; the compact Default backend select shows no █), **Tab** ×16, one per command (█ shows on Binary path at the 8th, Diffusion model path 9th, Timeout 10th, "Store cap (MB)" 14th; the 15th and 16th show no █), **Shift+Tab** ×2 (█ back on "Store cap (MB)  █ 2048"), **PageDown**. The detail pane scrolled to its end. Under the "State: Draft — save/revert below \| Defaults affect future Console video generations." line, the 211x44 pane starts at "Generation defaults"; at 235x52 it starts higher, with the SD.cpp video frame's last rows ("Diffusion model path", "Timeout (seconds) 7200", the Advanced keys hint) and its bottom edge (╰──╯). (Before the **PageDown**, after the Tabs, the 211x44 screen also showed that bottom edge; it was not saved.) Then Generation defaults (Retention, "▐✓▌ Confirm cost before paid generation: on", Retention TTL (hours) 24, Store cap (MB) █ 2048), Style templates, Diagnostics ("MiniMax H3 (cloud): not configured · key: missing", "ComfyUI (local server): configured", "SD.cpp video (local): not configured", "ffmpeg: MISSING (playback/streaming degraded)", then the same line for ffplay and for yt-dlp: the `env -i` PATH is /usr/bin:/bin), then "Save            Revert" at both sizes. Save's ink is (158,158,158), the disabled style: the panel disables Save and Revert while nothing is staged, so Tab skips them and only a scroll brings them on screen. Focus stays on Store cap. Save was not pressed. After the shots, the resize back to 211x44 scrolled Save off screen again, and **Esc** released the field before 06e. |
| `06e-settings-discovered-unchecked-row-under-cursor` | **/**, "providers & models", **Enter**, **/**, "model context window tokens", **Enter**, **Tab** ("▶ Saved model list · 1 saved in config · none discovered" focused), **Enter**, **Tab** ("Discover models" focused), **Enter**. "Discovered 3 models from llama.cpp." showed 3 s later and had timed out by the shot. **Tab** ×3 put the list in focus (█ by each row) with no row highlighted; **Space** there did nothing, and one **Down** put the cursor on the first row. Header "▼ Saved model list · 1 saved in config · 3 discovered, 2 not saved". The cursor row, on (87,165,226), reads "▐X▌ not selected · model-a · saved · discovered · capabilities unknown"; its X is (119,119,119) on (119,119,119), the box fill, as on the two unchecked rows at rest below it (model-x, model-y "not saved"). The stub logged one `GET /v1/models auth=no`. |
| `06f-settings-discovered-checked-row-under-cursor` | **Space**: the same row reads "selected · model-a · …" with its X (225,230,233) on (27,63,90). |
| `06g-settings-discovered-next-unchecked-under-cursor` | **Down**: the cursor is on "not selected · model-x · not saved · …", with an empty box ((119,119,119) on (119,119,119)). Above it, model-a "selected" shows its X at rest ((225,230,233) on (27,63,90)). Header "… 3 discovered, 2 not saved · 1 selected"; State "No changes". Save selected was not pressed. |
| `07a-wizard-provider-test-refused` | First-run wizard, Provider step (step 2 of 6): **Enter** on Quick setup. The list opens with OpenAI highlighted, so the plan's **Down** ×4 went one past llama.cpp, to Arcee AI (the first Cloud row: "Couldn't discover models for Arcee AI. Set ARCEE_API_KEY…"), and **Up** ×1 came back to llama.cpp (endpoint `http://localhost:8080`, "Couldn't discover models for llama.cpp. You can continue anyway."). Then **Tab** to Endpoint, **Ctrl+A**, **Ctrl+K**, `http://127.0.0.1:18631` typed, **Tab** ×2 to Test connection, **Enter**. The endpoint reads `http://127.0.0.1:18631` and the Chat URL `http://127.0.0.1:18631/v1/chat/completions`. The Test line reads "✗ The connection was refused - nothing is listening at that address. Start the server, or check the endpoint and port.". "▶ Authentication (optional)" is closed. Focus stays on Test connection (bold underline on the accent fill). |
| `07b-wizard-summary-after-refused-test` | **Ctrl+N** to the Model step: "The llama.cpp server isn't reachable at http://127.0.0.1:18631. Start it, then Retry — or enter a model ID below.", with the model-id field already focused (no **Tab** needed). `model-a` typed, **Enter**: the "Continue anyway?" dialog ("The server couldn't be reached, so this model setup is unverified. Continue anyway?"). **Tab** to Continue anyway, **Enter** (this lands on Voice), **Ctrl+N** (Protect), **Ctrl+N** (Summary). The stepper marks Provider and Model "!". The Summary reads "✗ Provider — saved, but the server couldn't be reached when models were checked" and "✓ Default model — model-a". Focus is on the first action, "Review provider setup". The footer reads "Ctrl+B back · choose an action below to finish". |
| `07c-home-not-ready-after-the-wizards-test` | **Ctrl+N** did nothing on the Summary. **Tab** ×3 (Add your first document, Write your first note, Explore Home, checked), **Enter**. Home's header reads "Home \| Not ready · Local". The next-action card reads "Reconnect Console model", then "Console's model failed its last connection check.", then a "Reconnect Console model" button. Needs Attention reads "No approvals or failures pending.", Running "Nothing running right now.", and Details is closed (▸). The `#home-triage-grid`, rail and canvas frames paint (119,119,119) on (36,47,56), 3.05:1. |
| `07d-home-details-model-not-ready` | **Tab** ×17 (13 tab-bar stops, then Needs Attention, Running, Recent and Details), **Enter** on Details' toggle. Details opens (▾) and reads "Model: Not ready \| ACP: Blocked \| Mode: Local \| Server: Configured; local mode \| Active: 0 \| Approvals: 0" and "Agent readiness: Model not ready, ACP blocked". No "Not set up" anywhere. Focus is on the Details toggle. |
| `07e-console-card-reads-the-wizards-refused-test` | Palette "Tab Navigation: Switch to Console" (**Ctrl+P**, typed, **Down**, **Enter**). The Get started card: "1. ● Reconnect the provider server" over "llama.cpp: Not ready · refused :18631", "2. ○ Pick a model", "3. ○ Send your first message  Composer unlocks after setup". "Retry connection" is the focused action (heavy accent frame), with "Write a note in Library" under it. The card's dim fill covers the whole workbench, header badge included, so no header text shows. The footer reads "Ctrl+Shift+F focus \| Enter continue setup \| F1 help \| Ctrl+P palette \| Ctrl+Q quit". "Switched to Console" had timed out by the shot. |
| `08a-schedules-recurring-completed-keeps-its-next-run` | (retaken at `6212bba292`) Palette "Tab Navigation: Switch to Schedules" (**Ctrl+P**, typed, **Down**, **Enter**), about 7 s for the "Switched to Schedules" notice to time out, then the first run's working path: a click on the "Weekly digest" row (column 9, row 22) to put focus in the queue (the plan's **Down**/**Up** was not tried again). All chip. "Weekly digest" is the focused row: bold underline, (225,230,233) on (27,63,90). The queue row reads "○  Weekly digest    2026-10-21 09:00 (in 2d) · Weekly on Wednesd" at 211x44 and "○  Weekly digest    2026-10-21 09:00 (in 2d) · Weekly on Wednesday at 09" at 235x52: the next run comes first and "(in 2d)" shows at both sizes, and the schedule summary is now the part cut at the column's right edge, with no ellipsis. It does not read "· —". The second row reads "⏸  Paused check-in  — (disabled) · Daily at 09:00 UTC". The All view has no "Backup" row. The detail reads "Title:    Weekly digest", "Status:    Completed" and "Next Run: 2026-10-21 09:00 UTC (in 2d)", and Frequency's "At" row reads "Weekly on Wednesday at 09:00 UTC" whole. The Inspector reads "Last Run: 2026-10-14 09:00 UTC". At 235x52 the detail also shows "▶ History", "Recent runs: No runs recorded yet" and "Open incidents: No open incidents". |
| `08b-schedules-disabled-reads-disabled` | (retaken at `6212bba292`) **Down**: "Paused check-in" is the focused row (bold underline on the (27,63,90) fill); "Weekly digest" drops to (224,224,224) on (39,39,39). The row reads "⏸  Paused check-in  — (disabled) · Daily at 09:00 UTC", whole at both sizes. The detail reads "Title:    Paused check-in", "Status:    Disabled" and "Next Run: — (disabled)", with "At  Daily at 09:00 UTC". The Inspector reads "Last Run: Never run". |
| `08c-schedules-fired-one-time-reads-dash` | (retaken at `6212bba292`) **4** (Completed), wait 1 s. The only row is "✓  Backup           — · One-time at 2026-10-15 12:00 UTC", focused (bold underline on the (27,63,90) fill), whole at both sizes. The detail reads "Title:    Backup", "Status:    Completed" and "Next Run: —", with "Repeat  One-time" and "At  One-time at 2026-10-15 12:00 UTC ▾". The Inspector reads "Last Run: 2026-10-15 12:00 UTC". The chip row still shows no visible mark on the selected chip: All, Active, Paused and Completed are drawn alike (Completed in (223,223,223), the others in (224,224,224), all on (13,13,13)). |
| `09a-settings-key-check-refused` | llama.cpp profile at 127.0.0.1:18633 (nothing listening). The Console first read "Ready · not tested". **F4**, **F6**, **Down**, **Enter** opens Providers & Models. **t**: the Key check row reads "Not ready · refused :18633". A warning notice (amber edge) reads "Model listing failed (connection refused) — start the server or check the URL; generation not tested.", and it still shows at 235x52. Endpoint row: "http://127.0.0.1:18633   saved in config". |
| `09b-settings-identity-row-refused` | **F6** ×2 to the Scope Inspector. At 211x44 the Key block was below the fold ("▼ more — scroll the inspector"), so **End** then **Up** ×6 brought it into view. The Key block reads "Provider readiness: llama.cpp / model-a — Not ready · refused :18633" (wrapped after "llama.cpp /"), then "Readiness   Not ready · refused :18633" and "Endpoint    http://127.0.0.1:18633 · model listing failed (connection refused) — start the server or check the URL" (wrapped), then "Config      llama.cpp is configured", "Key         not required", "Model       model-a" and "Generation  not tested". The notice had timed out. |
| `09c-settings-overview-refused` | **/**, `overview` typed, **Enter**: Overview. Status reads "Configuration: llama.cpp / model-a; Status: Not ready · refused :18633" and "Last connection test: Readiness: Not ready · refused :18633; Endpoint: http://127.0.0.1:18633 · model listing failed (connection refused) — start the server or check the URL". The Advanced / Diagnostics and Where changes happen Collapsible frames paint (119,119,119) on (30,30,30), 3.72:1. |
| `09d-console-blocked-retry-hint` | Palette "Tab Navigation: Switch to Console". The header badge reads "Not ready · refused :18633". The composer hint reads "Send blocked — retry the connection to continue ›". Status row: "Provider: llama.cpp", "Model: model-a", "Ctx 0% · Now $0.00". "Switched to Console" still shows at both sizes. |
| `09e-console-inspector-setup-row-refused` | **Alt+I** opens the Inspector on "What happens if I send now?" ("Run: Recovery required"). The Run rows were below the fold, so 5 mouse-wheel-downs scrolled to them. The Inspector reads "Setup: Not ready · refused :18633", "Blocked impact: Send is blocked until setup is finished. - Provider setup needed: endpoint unreachable", "Next action: Retry connection" and "Provider: blocked - Provider setup needed: endpoint unreachable". |
| `10a-chat-settings-onnx-fields-say-support-not-verified` | ONNX Runtime (local) · onnx-model. The header reads "Not ready · unsupported". **Ctrl+O**: Chat settings · Chat 1. The Model row reads "onnx-model · ONNX Runtime (local) … Not ready · unsupported · context unknown". All eight rows show (Temperature, Max tokens, Streaming, Reasoning effort, Reasoning summary, Verbosity, Thinking, Thinking budget), and each help line starts "Support not verified for this model." followed by the field's own help, e.g. "Support not verified for this model. Lower stays focused; higher is more varied." and "… The provider decides · How much a reasoning model thinks first.". "▶ Sampling" is closed. "▼ Connection · localhost:8000 · no key needed …" and "Choose a supported provider to continue." are below it. |
| `10b-chat-settings-onnx-sampling-open` | A click on "Sampling" (column 23, row 27) opens it (▼, focused). Top P 0.95, Min P 0.05, Top K 50, Seed, Presence penalty and Frequency penalty all show, and each help line leads with "Support not verified for this model.", e.g. "… Use only top tokens whose chances sum to P.". None is hidden. **Esc** closed the dialog afterwards. |
| `11a-console-dangling-markup-provider-literal` | Profile provider `custom-ep:gone[/]`, a registry id with no endpoint behind it. The app draws with no MarkupError (none in its log either). The status chip reads "Provider: custom-ep:gone[/]" with its brackets, plain ink. The header reads "Not ready · unsupported" and the composer "Send blocked — finish provider setup to continue ›". The cost chip reads "Context 0% · Curren", cut at the right edge, in the 211x44 shot taken before any resize. It reads "Ctx 0% · Now 0 tok · Next unavailable" at 235x52 and "Context 0%" back at 211x44. |
| `11b-console-markup-endpoint-name-literal-after-switch` | **Alt+M**, `GPU` typed (Switch model, not kept: titled "Switch model · now: custom_ep:gone[/] · model-a", an underscore where the chip has a hyphen, TASK-35562; "▶ model-a   GPU [b]box[/b]   ?  Ready · reachable 10:00" highlighted under READY PROVIDERS, and the typed id "GPU   custom_ep:gone[/]   ?  Not ready · unsupported" under TYPED MODEL ID), **Enter** on that row. The shot is the Console after the switch. The status chip reads "Provider: GPU [b]box[/b]", literal and not bold. The header reads "Ready · reachable 10:00" and the cost chip "Ctx 0% · Now 0 tok". "This chat updated" shows at both sizes. The 18632 stub's log has exactly one request, `GET /v1/models auth=no`, from opening Switch model. |
| `11c-library-frames-at-3-to-1` | Palette "Tab Navigation: Switch to Library": Library's Get started canvas. The shell grid, the Navigation rail and the canvas frames paint (119,119,119) on (36,47,56) on textual-dark, 3.05:1. The same colours at the old (62,62,62) would be 1.28:1. |
| `11d-workflows-rails-search-focused-empty` | Palette "Tab Navigation: Switch to Workflows", wait 4 s, **F6**. The Workflow library and Step navigator rails paint (36,47,56), each with a (119,119,119) "│" edge down its right side. The focused search field fills (25,68,102) with a (87,165,226) outline, and its empty hint "Search workflows" is italic in (225,230,235). |
| `11e-workflows-search-query-selected-on-return` | `rev` typed, **Tab**, **Shift+Tab**: back in the field, the whole "rev" is reversed (SGR 7) in (225,230,235) on the (25,68,102) fill, at both sizes. |
| `11f-workflows-focused-new-workflow-button` | "New workflow" is focused: bold underline, (225,230,235) on (25,68,102), on the (36,47,56) rail. From the search field, 2 **Tab**s reach it (the list, then New workflow; Previous and Next are disabled and are skipped). This run went 2 Tabs past it, to Add step, and came back with **Shift+Tab** ×2. |
| `12a-console-draws-on-a-user-theme-with-auto-backgrounds` | `default_theme = "auto_probe"`, the user theme with `auto` values. The Console draws: its fills are the theme's own (24,24,24) surface and (32,32,32) panel (100 and 83 background runs at 211x44). The scratch log has four WARNING lines, `Theme auto_probe.toml: dropping invalid variable 'surface-lighten-2'`, then the same for `'block-cursor-blurred-background'`, `'text-primary'` and `'tldw-boundary'` (`themes:651`). There is no line for text-muted or scrollbar-color, and no "CSS parsing failed" or "Invalid value". |

## Retaken at 6212bba292

The capture fix wave changed four surfaces the first set showed. Retaken, each from a fresh
profile with its launch replayed through the moments before it (`b1` through `04i`, `b2`
through `05a`, `b3` through `06c`, `c2` from the start):

- **F1** (TASK-33641.4), `08a`, `08b`, `08c`: the Schedules queue row puts the next run
  first. `08a` reads "2026-10-21 09:00 (in 2d) · Weekly on Wednesd" at 211x44, where the
  first run read "Weekly on Wednesday at 09:00 UTC · 2026-10-2"; "(in 2d)" now shows at
  both sizes, and the summary is the part the column cuts. `08b` reads "— (disabled) ·
  Daily at 09:00 UTC" and `08c` "— · One-time at 2026-10-15 12:00 UTC". Only the queue
  rows and the "last tick" counter differ from the first run.
- **F2** (TASK-33005.7), `05b`, `05c`: the key-rejected composer reason fits. The strip
  reads "Send blocked — key rejected; update the API key ›", whole, at both sizes, where
  the first run read "Send blocked — key rejected; update the API key t…".
- **F3** (TASK-33007.11), `04j`, `04k`, `04l`: Esc resets the Settings field guide. After
  **Esc** the Inspector reads "Focused setting: None — Tab to a setting", where the first
  run's three still read "Focused setting: Endpoint".
- **F4** (TASK-33641.3), `06d` and the new `06d2`: the Video Gen editor is whole and its
  Save comes on screen. Under "Backend settings" `06d` draws the three backend frames
  (the first run had a lone top edge), and `06d2` scrolls the detail pane to its end, where
  "Save" and "Revert" show, disabled while nothing is staged.

F5 and F6 changed a task file and the User Guide only, so no capture changed with them.

A check run replayed every other moment in the retaken launches, at 211x44 only, and
compared each with the first run's file (clock times ignored). 11 moments were the same
(`04d`, `04e`, `04g`, `04h`, `04i`, `04m`, `04o`, `05a`, `06e`, `06f`, `06g`; `06e`-`06g`
byte for byte). `06b` and `06c` differed only by the Settings rail's scroll, a procedure
difference: the check run took `06a` at 211x44 only, and the resize back to 211x44 is what
scrolls the rail to its focused row. `04n` and `06a` differed only by scroll position (the
detail pane and the Inspector in `04n`, the Inspector's overflow hint in `06a`), most likely
because the check run took no resize (not proved). In `04a`, `04b`, `04c` and `04f` the "Chat 1 💬" row of the
Console rail, behind the popover, was clipped; triage reproduced it at the phase base
`111f8e48fc` and filed TASK-35568. Their status row's long cost-chip label ("Context 0% ·
Current $0.", cut at the screen edge, where the first run's `04b`, `04c` and `04f` read
"Ctx 0% · Now $0.00") is TASK-35561. None of these moments shows a surface the wave
changed, so the first-run captures of all of them are kept. Launch `c2` has no other
moment, so the Schedules retake needed no check.

After the retakes, the review of the fix wave added `0571847dce`: Esc on a Settings field
now also empties the focus-help line (`#settings-focus-help`). In `04j`-`04l` that line was
already empty, because the field left by Esc, Endpoint, has no tooltip, so those captures
still show the head. The round's other two commits change only test lanes.

## Plan corrections

Where the run differed from the capture plan's keys, waits or expectations:

- `a1`-`a3` (`01`-`03`): the sandbox's outbound deny also denied the connect to
  mDNSResponder, so the template's `localhost` URLs failed at name resolution rather than
  with a refused connection. The NOT RUNNING rows read the same either way.
- Every launch: the precheck found ControlCenter on `*:5000`. The plan said not to launch
  then; each launch went ahead with port 5000 denied in the sandbox (see "Procedure").
- `01h`: after the Test the dialog body is scrolled, so the MODEL row is above the fold;
  "Ready · reachable 09:52" shows as the Connection section's readiness line instead.
- `01h`: the 18611 log also has `GET /props?model=model-a&autoload=false` (llama.cpp's
  context-window probe), sent while the endpoint was typed, and the dialog shows
  "Unavailable: llama.cpp failed its last connection test (connection refused).", which
  the plan did not name.
- `01i`: focus had left the Test button, so reaching the Endpoint field took **Shift+Tab**
  ×8, not ×2 (TASK-35566).
- `02a`: the MODEL row is above the fold; "Not ready · endpoint unsaved" shows as the
  Connection section's readiness line, and a wheel-up after the shots read the MODEL row.
- `03a`: after **Enter** on New endpoint… focus is already on Display name, so the plan's
  **Tab** before typing was dropped (it moved to the Family select, whose list the typed
  space opened).
- `03a`: with the plan's 10 s stub delay the listing gave up at the app's 2.5 s probe
  timeout, and its notice had timed out before the shot. It was retaken on a fresh `a3`
  launch with the stub answering after 2 s, **Esc** 0.7 s after Create, and the screen
  polled for the notice.
- `03b`: pick mode also lists GPU box's own model-a (the form's Models field came
  prefilled from the llama_cpp template), beside gpu-x and gpu-y.
- `04b`: "Custom OpenAI-compatible · Custom OpenAI-compatible #2" are under NOT RUNNING, not
  NEEDS SETUP, as in `04a`.
- `04e`: after **F6** ×2 the Inspector holds focus with no visible focus style (a third
  **F6** and **Shift+F6** showed it); the Key block and Test rows needed wheel ×11 at
  211x44 and ×6 more at 235x52.
- `04f`: the status row has no readiness word; only the header badge reads "Ready ·
  reachable 09:47" (F5 reworded TASK-33005.11 to say the header badge).
- `04h`: the toast had timed out by the planned shot, and a second **t** put the Inspector
  back at its top; the kept shot is after a third **t** and wheel ×11.
- `04i`: the save put the Inspector back at its top; wheel ×11 at each size.
- `04n`: at 1.5 s the screen still showed Providers & Models; the shot is about 5 s after
  the **Enter**.
- `04n`: Console Behavior has no card frame, by design (TASK-33007's review sets `border:
  none` on `#settings-console-behavior-card`).
- `04o`: the API key row's help reads "paste one to save it in config", not "paste one, or
  set GROQ_API_KEY": under `credential_source = "none"` readiness names no env var.
- `05c`: the Inspector opens on "What happens if I send now?"; the Setup rows are in its
  Run section, below the fold at 211x44 (wheel ×8), in the first run and the retake.
- `05c` (retake): the Inspector opened about 3 s after **Alt+I**, not within the plan's
  1 s.
- `06e`: after **Tab** ×3 the list holds focus with no row highlighted (**Space** did
  nothing); one **Down** put the cursor on the first row before the shot.
- `06e` (retake): `06d2` leaves focus in Store cap, where **/** would type, so one **Esc**
  was added before `06e`'s **/**.
- `07a`: the provider list opens with OpenAI highlighted, so **Down** ×3 reaches llama.cpp;
  the run sent the plan's ×4 and came back with **Up** ×1.
- `07e`: no header shows; the Get started card's fill covers the workbench, header badge
  included. The card's step 1 reads "llama.cpp: Not ready · refused :18631".
- `08a`: after the palette switch focus is on the tab bar, and **Down**/**Up** did nothing;
  a click on the "Weekly digest" row (column 9, row 22) focused the queue, in the first
  run and the retake.
- `08a` (retake): "(in 2d)" shows at both sizes, and the schedule summary is cut at the
  column's right edge, with no ellipsis.
- `09b`: the Key block was below the fold; **End**, then **Up** ×6 in the Inspector.
- `09e`: the Run rows were below the fold; five wheel-downs over the Inspector.
- Check run (`04a`, `04b`, `04n`, `06a`): differences from the first run, not key paths;
  see "Retaken at 6212bba292".

## Known pre-existing issues seen in these captures

Each predates Phase 9 and has a follow-up task. Each was reproduced at the phase base
`111f8e48fc`, except TASK-35569, whose code no Phase 9 change touches:

- TASK-35560: Chat settings' session-only endpoint warning is dead. `01k` shows only
  "This chat updated" and the llama.cpp thinking notice, with no "…saved for this session
  only…" warning.
- TASK-35561: the cost chip's first paint can keep a label that does not fit until a
  resize: `11a` at 211x44 ("Context 0% · Curren") and `04a` at 211x44 ("Context 0% ·
  Current $0."), and the check run's `04b`, `04c` and `04f`.
- TASK-35562: Switch model shows a dangling custom endpoint id raw: on the way to `11b` it
  was titled "Switch model · now: custom_ep:gone[/] · model-a", where `11a`'s status chip
  reads "custom-ep:gone[/]".
- TASK-35563: the Schedules queue chips show no selected filter: in `08a` and `08c` All,
  Active, Paused and Completed are drawn alike.
- TASK-35564: in local mode, opening Schedules logs the server-mode refusal as an ERROR
  ("Failed to load server automations for the Queue list …", launch `c2`, `08a`-`08c`);
  nothing on screen shows it.
- TASK-35565: focus on the selected Chat settings view tab is invisible (`01g`: bold, no
  underline, the same as at rest).
- TASK-35566: Test connection & list models drops focus when it disables itself for the
  check (`01h` shows no focused control; `01i` took **Shift+Tab** ×8).
- TASK-35568: the Console rail's Chats list can clip its last row ("Chat 1 💬") after the
  Conversations tray rebuilds: the check run's `04a`, `04b`, `04c` and `04f`, and in this
  set "Today" with no row under it in `01b`, `01c`, `01e` and `02b` (both sizes), `01d`
  and `03b` (211x44), and `01k` and `02a` (235x52).
- TASK-35569: the Settings Inspector shows no focus style when F6 moves focus to it
  (`04e`: a third **F6** puts the █ back on the rail).

## Moments with no capture

The plan lists 34 Phase 9 changes with no capture (one entry groups seven tasks closed
without code): 10 test-only or CI-lane changes; 7 internal ones with no visible effect
(thread safety, typing, recovery infrastructure, token swaps equal to the literals they
replaced, regenerated stylesheets); 6 closed without a product change; 4 whose only effect
is a provider call or config.toml on disk; and 7 visible changes that no state reachable
here shows (for example a sent message, a relaunch after a send, or a width below 211
columns).
