# First-run setup shape: design

- **Status:** Draft for owner review, revision 1, 2026-10-03. **Not approved.** No wizard code changes under TASK-34100.17 until the owner records approval (§15). Implementation follows as the follow-up tasks in §10, filed only after approval.
- **Owner:** the project owner (@rmusser01).
- **Task:** [TASK-34100.17](../../../backlog/tasks/task-34100.17%20-%20Owner-approved-design-spec-for-the-setup-flow-Quick-track-tldw-server-re-run-dashboard-Say-hello.md), under the burn-down programme [TASK-34100](../../../backlog/tasks/task-34100%20-%20First-run-setup-wizard-burn-down-of-the-2026-10-02-UX-review.md).
- **Evidence:** the 2026-10-02 senior design / HCI review, [`Docs/superpowers/qa/first-run-wizard-ux-review-2026-10-02/README.md`](../qa/first-run-wizard-ux-review-2026-10-02/README.md). This spec cites it as "report §N", its structural fixes as SF1–SF10, its enhancements as E1–E12 and its register issues by id in brackets (for example [coverage-06]).
- **Prior designs this builds on:** the [first-run setup wizard design](2026-07-28-first-run-setup-wizard-design.md) (2026-07-28, "the wizard spec"), the [master shell UX design](2026-05-02-new-user-first-run-shell-ux-design.md), [`PRODUCT.md`](../../../PRODUCT.md) and [`DESIGN.md`](../../../DESIGN.md).
- **Base:** `origin/dev @ fcebe51a09` (2026-10-03). Every statement about current behaviour is cited `path:line` against that commit; §13 lists them in one place. Path shorthand: `FRSW` = `tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py`; `FRSS` = `tldw_chatbook/UI/Wizards/first_run_setup_state.py`; `SS` = `tldw_chatbook/UI/Screens/settings_screen.py`.
  - **Re-anchoring rule.** Line numbers are pinned to `fcebe51a09` and will drift: TASK-34100.1 moves every step class out of `FRSW`. Each follow-up re-anchors by symbol (`active_step_ids`, `WelcomeStep.compose_step`, `SummaryStep._render_rows`, `handle_switch_runtime_source`, `ServerSwitchModal._run_connection_test`), never by line.
- **Mockups:** every mockup line was measured by script against its stated width (80 or 120 columns). Model ids, paths and host names in mockups are illustrative.

---

## Summary for the owner

Setup today is one fixed corridor. This spec gives it a shape that matches what people came to do, and asks you to approve eleven decisions:

| # | Decision | In one line |
|---|---|---|
| D1 | Quick track | **4 steps: Welcome → Connect → Model → Ready.** Voice and Protect leave Quick. This reverses TASK-21148 AC#5 but keeps its real guarantee, now written as an invariant: once you leave Welcome, the step count never changes. |
| D2 | Full track | **11 steps, every one real:** Welcome, Connect, Model, Server, Search, Tools, Spoken replies, Dictation, Appearance, Protect keys, Ready. Untouched optional steps write nothing. "Finish with defaults" appears once Model is saved. |
| D3 | tldw server | An always-shown **Runtime** row on Ready, a **"Connect a tldw server…"** next step, an optional **Server** step on Full only, one shared probe that says *unreachable / token rejected / not a tldw server*, and one commit path shared with Settings (ADR-033). Settings' button leaves the collapsed Advanced section. No Welcome router. |
| D4 | Ready screen | A **"✓ Ready to chat — OpenAI · gpt-4.1-mini (1M context)"** verdict computed by Console's own send preflight; rows only for steps you saw; Data and Config lines; **at most three exits** (Start chatting, Add your first document, Write your first note); everything else in a "What's next" list. Fits 80x24. |
| D5 | Say hello (E1) | A one-line test message sent through Console's real path. **Local servers: runs on arrival, costs nothing billable. Cloud: never automatic** — only when you press Say hello, with the computed token count on screen. The exchange becomes Console's first turn. |
| D6 | What's next (E9) | Ready lists up to three relevant next steps (Hear replies aloud, Connect a tldw server, Add a project folder…). Console arrival becomes one dim line instead of toasts. Pressing Speak or Dictate with nothing set up opens a small setup sheet right there. |
| D7 | Re-run (E5) | A re-run opens **"Review your setup"**: current values, Enter on a row changes just that area and comes back, **Done** returns where you started. The full walkthrough is the secondary action. |
| D8 | Documents first | A third Welcome choice, **"Start with my documents or notes — set up AI later"**, finishes setup at once and opens Library Import. |
| D9 | Keychain-first keys (E8) | New keys go to the **system keychain** by default; plain text and password encryption stay as explicit choices; plain text is the fallback where no secure keychain exists (SSH, headless Linux). Existing keys are never moved without your action. "Remember on this device" is not shipped. |
| D10 | Plain-text setup (E12) | `tldw-cli setup --plain`: the same state machine and commit path as the TUI, one prompt per decision, keys from a hidden prompt or an environment variable, never from argv. |
| D11 | Non-interactive setup (E6) | **Build it, last:** "Export these settings (no keys)" plus `tldw-cli setup --from FILE --non-interactive`. It costs little once D10 exists, and it replaces today's riskiest second-machine habit: copying a config.toml that may hold plain-text keys. |

You are also asked to confirm the report's seven "Rejected, on purpose" items (D12), and to rule on the eight open questions in §16, the last section. §12 drafts one new ADR and five ADR amendments or confirmations; none is edited until you approve.

**What this spec does not do.** Sixteen sibling subtasks (TASK-34100.1–.16) fix the defects inside today's steps. This spec covers only the shape decisions they leave open (§0.4). It changes no code.

---

## 0. Problem, goals, personas

### 0.1 Problem

Setup does not adapt to what the user came to do. Five shape problems remain after the sibling subtasks:

1. **Quick carries two steps most people don't need.** The Quick track is Welcome, Provider, Model, Voice, Protect, Summary (`FRSS:982-989`). Voice preselects PocketTTS (`UI/Wizards/first_run_voice_step.py:123-127`), a separate local server at `127.0.0.1:8765` (`first_run_voice_step_state.py:36`) that most newcomers don't run [voice-speech-03]. Protect is always on the track since TASK-21148 (`FRSS:1271-1286`) and renders "No API keys saved yet — nothing to protect" for keyless, local and env-key users (`FRSW:6463-6467`) [coverage-16] [protect-summary-06]. Welcome still sells it as "Quick setup — provider, model, voice, protection" (`FRSW:6382-6386`).
2. **A tldw server is never offered.** Nothing under `tldw_chatbook/UI/Wizards/` references `tldw_api` (a repository grep on `fcebe51a09` returns no match). The only control is "Switch Source / Server" (`SS:17676-17681`), inside a collapsible titled "Advanced / Diagnostics" (`SS:17648-17651`) that starts collapsed (`SS:3462`). The Summary never says whether chatbook runs local-only [coverage-06].
3. **A documents-first user must get through Provider and Model.** Welcome offers only Quick and Full plus "Restore a backup" (`FRSW:6377-6390`) [entry-exit-handoff-28].
4. **A re-run replays first run.** Every re-run entry pushes the same wizard with `rerun=True` (`app_command_providers.py:975-988`, `app.py:4148-4171`, `SS:31197-31209`) and lands on Welcome. TASK-34100.10 makes that corridor respect current values; it remains a corridor [cross-cutting-05].
5. **"Done" means "saved".** The Summary's primary action becomes "Start chatting" when the Provider and Default model rows are configured and no probe failed (`FRSS:641-665`, `FRSW:6807-6818`). Nothing checks that a turn can be sent [cross-cutting-01]. TASK-34100.9 AC#10 adds a "Ready to chat?" row; the screen around it, the optional test message and the hand-off are still undesigned.

Two further gaps belong here because they change who can use setup at all:

6. **Keys live in config.toml or nowhere.** Provider credentials resolve from `stored`, `environment` or `none` (`Chat/provider_readiness.py:235`). The OS keyring already holds tldw server tokens (`SS:26584-26597`, `runtime_policy/server_credentials.py:452-467`) and MCP bindings (`MCP/credential_bindings.py:144-176`), but never a provider key. The only protection offered is a master password typed at every launch, the most fragile area of the review (SF6).
7. **Setup exists only as a full-screen Textual app.** Textual exposes no accessibility tree, so screen-reader users cannot use it, and there is no scripted path: `tldw-cli` parses only recovery selectors (`tldw_chatbook/cli.py:30-56`) [coverage-10].

### 0.2 Goals

- **G1 — Fewest decisions to a reply.** Quick asks only what a first chat needs, and ends on a verdict computed by Console's own preflight, with an optional real reply.
- **G2 — Every step earns its place.** Every Full step changes something the runtime reads; untouched, it writes nothing.
- **G3 — Local-vs-server status is always visible.** PRODUCT.md asks users to "understand what is local, server-backed…" (`PRODUCT.md:17`); Ready always says.
- **G4 — Re-running setup is safe and fast.** Change one thing in a few keys, from wherever you are, and go back there.
- **G5 — Keys are safe without daily friction.** The default store is the OS keychain where one exists.
- **G6 — Setup is usable without the TUI.** By screen-reader users, over SSH, and from a script, on the same state machine.
- **G7 — Keep what works.** Every report §5.5 "Preserve" item survives (§7).

### 0.3 Non-goals

- Fixing defects inside steps. They belong to TASK-34100.1–.16 (§0.4).
- New provider facts, model ranking, encryption mechanics, download coordination, the input policy or the 80x24 frame. This spec states requirements on them and names their owners.
- A setup "hub of cards" for first run. The report rejects it (report §4.3); the hub is the re-run dashboard only.
- Mobile, web or server-side setup. `tldw-serve` is out of scope.
- Changing what Settings owns. Settings stays the owner of durable configuration (ADR-012); setup is a guided path into the same owners.

### 0.4 Scope boundary with the sibling subtasks

| Sibling | Owns | This spec adds on top |
|---|---|---|
| .1 step extraction | Steps in their own modules, `wizard_worker()` helper, busy line | Requires the single-step host (D7) and plain mode (D10) to reuse those modules |
| .2 first chat works | Catalog repair, **shared readiness verdict with the capacity blocker** (AC#2), real key checks, env-aware Get started card | Ready row layout and states (D4); Say hello rules (D5) |
| .3 Moonshot | Continuation persistence | Say hello must exercise the saved-chat path that .3 fixes (D5) |
| .4 encryption lifecycle | One unlock path, refuse second enable, Settings Encryption card, state-aware Protect | Protect leaves Quick (D1); Ready "Encrypt saved keys…" option; keychain-first (D9) |
| .5 handoff | False toast, minimal plain-chat prompt, error categories (AC#4), single arrival line (AC#11) | Arrival line content (D6); Say hello's failure copy reuses .5's categories |
| .6 provider step | Catalog-driven form, "Ready on this machine", filter, key-source choice when an env key exists (AC#21) | Display name "Connect"; key storage selector (D9) |
| .7 model step | Curated picker, skip keeps the key | "Finish with defaults" after Model (D2) |
| .8 voice step | Untouched Voice writes nothing; "No voice for now" | Voice moves to Full as "Spoken replies" (D1, D2) |
| .9 honest status | Outcome record, tracker, Summary rows, **"Ready to chat?" row** (AC#10) | Ready screen composition, exits, What's next (D4, D6) |
| .10 setup session | Entry contract `open_setup_wizard(origin, resume, start_step)`, re-run prefill, "Review your setup" header (AC#1), one finish path, E10 queue | The dashboard body and single-step Change (D7); documents-first finish (D8) |
| .11 input policy | Highlight browses; Enter/Space/click selects | The dashboard list and What's next list follow it |
| .12 terminal frame | Short tier at 80x24, glyph checkboxes, contrast | Ready and dashboard row budgets assume it |
| .13 full-track steps | Search, Tools, Notes removal, Speech engine choice, Appearance comfort settings, Welcome no-oversell copy (AC#15) | Full order (D2); Welcome copy is restated here and supersedes .13 AC#15's time line (D1) |
| .14 downloads | Download coordinator | Places download-bearing steps late (D2) |
| .15 registry | Glossary, "change it later" homes, User Guide | The names "Connect", "Ready", "Review setup" enter the glossary (D1, D7) |
| .16 portable | Second-machine docs, `--config`, `--no-splash`, Restore Inspect | `setup --plain`, export and `--from` (D10, D11) |

**One backlog conflict to resolve.** TASK-34100.10's coordination note says "Narrow task-28019 to its media-first path", while TASK-34100.17 says "Narrow it to its modal-sequencing AC (#3)". Both cannot hold. This spec follows TASK-34100.17 AC#3 (D8): task-28019 is narrowed to its modal-sequencing AC#3, and its media-first ACs #1–#2 move into the documents-first follow-up (F7, §10). TASK-34100.10's note is corrected to match. Because TASK-34100.10 AC#18 (E10) delivers exactly that modal sequencing, the narrowed task-28019 closes when .10 lands. Both edits are made after approval, not now.

### 0.5 Personas

The report's three personas, plus two the shape decisions serve directly.

| Persona | Situation | What the shape must give them |
|---|---|---|
| **Sam** | First-time, has an OpenAI key, fuzzy on jargon | Four steps, a verified "Ready to chat", a reply before leaving setup if they want one, a key that is safe without a password at every launch |
| **Jo** | First-time, wants private local AI; may have nothing running | No cloud-key detours; a local test reply that runs on its own; "Runtime — this computer only" said plainly |
| **Riley** | Power user: env keys, local servers, several machines, re-runs to change one thing | A dashboard instead of a corridor; Finish with defaults; palette jumps; `--plain`, export and `--from` for the next machine |
| **Dee** (new) | Came for documents and notes; may never use an AI provider | One choice on Welcome that goes straight to Library and writes nothing else |
| **Ash** (new) | Uses a screen reader, or works over SSH without a keychain | `tldw-cli setup --plain`; a plain-text key fallback that is named honestly |

### 0.6 Rules

The programme's four rules (report §5) bind every decision: (1) done means a reply, not a write; (2) write nothing the user didn't touch, drop nothing the user typed without saying so; (3) every mark is computed from persisted, verified state through the runtime's own resolver; (4) one owner per fact. This spec adds four shape rules:

- **S1 — Stable total (restates TASK-21148's guarantee).** From the moment the user leaves Welcome until they return to it, the run's step list never changes. Only the Welcome choice sets the total, and the tracker shows that total before Next (TASK-34100.9 AC#5).
- **S2 — Options are not steps.** Anything conditional (encrypt now, hear replies aloud, connect a server) is an option on Ready or a next step, never a step that joins or leaves the track.
- **S3 — One surface per job.** First run is a corridor; a re-run is a dashboard; a single change is a single-step sheet. All three are hosts for the same step modules and the same commit path.
- **S4 — Every exit says where it goes.** No exit, link or hint names a home that cannot do the job (report §4.4; owned by TASK-34100.15's registry).

---

## 1. Decisions

Each decision states what is decided, why, what was rejected, which report §5.5 "Preserve" items it must not break, and how it stays consistent with the sibling subtasks and ADRs. Screen-level detail is in §3.

### D1 — Quick track: Welcome → Connect → Model → Ready

**Decision.**
- The Quick track is four steps: `welcome`, `provider` (titled **Connect**), `model` (**Model**), `summary` (titled **Ready**). Step ids stay as they are, because drafts are keyed by them (`FRSS:991-1002`); only display titles change, through TASK-34100.15's registry.
- **Voice leaves Quick** and moves to Full as "Spoken replies" (D2). On Quick, Ready offers it as the next step **"Hear replies aloud…"** (D6).
- **Protect leaves Quick.** Ready shows the option **"Encrypt saved keys with a password…"** only when this run stored a provider key as plain text in config.toml. With D9 shipped, that means only when the user chose plain text, or no keychain exists.
- `active_step_ids(TRACK_QUICK, …)` returns the four ids for every value of `key_entered` (today `FRSS:1271-1286` returns six).

**This supersedes TASK-21148 AC#5.** That AC reads: "Protect appears in the quick track from the start (marked skipped when keyless); the step total never changes mid-flight." It had two halves:
- *the guarantee* — the total never changes mid-flight. UAT N-6 showed "Step 2 of 5" becoming "Step 3 of 6" when a key was typed (TASK-21148 notes);
- *the mechanism* — Protect is always present, so it can never join.

This spec keeps the guarantee and retires the mechanism. The guarantee now holds by construction: Quick has no conditional step at all, because the only conditional thing (encryption) became an option on Ready (rule S2), and options are not counted. The mechanism was paid for by every keyless, local and env-key user, who walks an empty step that ticks ✓ ("No API keys saved yet — nothing to protect", `FRSW:6463-6467`).

**Why.**
- **The evidence.** Jo's local path and Riley's env-key path both meet an empty Protect step that ticks ✓ (report §1 Jo Path A; §2(a) stage 11). Sam's Protect step is where the P0 lock-out happens [protect-summary-04]: the step chosen for peace of mind is the one that bites. Voice adds a detour inside a "2-minute" track and, today, a write on Next [voice-speech-01].
- **Relevance at the point of need.** Voice matters the first time a user presses Speak, not as a setup detour (E9). D6 puts it there.
- **Protection without a step.** With D9, Sam's key goes to the keychain by default, so the main reason Protect was on Quick — "make my key safe" — is met at the key field without a password.
- **Stability without padding.** Rule S1 is stronger than TASK-21148's: it also covers the Full track and every future conditional idea.

**Rejected alternatives.**
- *Keep six steps, but make Voice and Protect truly skippable* (TASK-34100.8 and .4 alone). They become honest, not relevant: two Nexts on every Quick run, and a Quick label that still has to name "voice, protection".
- *Five steps, keeping a conditional Protect.* This reintroduces the mid-flight count change TASK-21148 fixed.
- *Fold encryption into Connect as a checkbox under the key field.* This grows the busiest form in the wizard for every user; the report's verifier preferred the Ready option ("the verifier's lower-risk alternative", report SF10).
- *Three steps (Connect folds Model in).* The provider list and the model list both need the full step height at 80x24 (TASK-34100.6 AC#15, .7 AC#4).

**Welcome copy (restated; supersedes TASK-34100.13 AC#15's time line, because Quick no longer has optional downloads).**

| Element | Copy |
|---|---|
| Title | Welcome to chatbook |
| Pitch | Chat with cloud or local AI models, keep notes, and work with your own documents — all in your terminal. With a model on this computer, your chats stay on it. |
| Question | What would you like to do first? |
| Choice 1 (default) | Quick setup — connect AI and pick a model (recommended) |
| Choice 2 | Full setup — adds server, search, tools, voice, appearance |
| Choice 3 | Start with my documents or notes — set up AI later |
| Time line | Quick takes about 2 minutes. Full takes about 10, plus any downloads you choose. You can change any of this later by running setup again. |
| Restore line | Moving from another computer?  [ Restore a backup ] |

- The three labels are 55, 58 and 50 characters, inside the 61-character no-wrap budget TASK-21148 set (TASK-21148 notes).
- "You can change any of this later by running setup again" replaces "Everything can be changed later in Settings" (`FRSW:6371-6375`). Until TASK-34100.15 fills the missing Settings homes (Dictation, Encryption, Analysis model), the Settings claim is false (report §4.4); the re-run claim is true because of D7.
- **"About 2 minutes" is a measured claim.** The live verification (§8.4) must reach Ready ✓ in two minutes or less, hands-on, following the recommendations with a pasted key and with a running local server. If it cannot, the copy changes before release; it does not ship as aspiration.

**Tests and docs that pin six steps.** All are rewritten deliberately in F5 (§10), never deleted:
- `Tests/Wizards/test_first_run_setup_state.py:509-528` (`TestActiveStepIds.test_quick_track`, `test_quick_track_with_key`);
- `Tests/Wizards/test_task_25818_tracker_integration.py:17` (`_TRACK`);
- `Tests/Wizards/test_first_run_setup_wizard.py:11645, 11662, 11679-11681, 11759, 11973, 12847-12872` (`len(…) == 6`, "Step 1 of 6", "Step 2 of 6");
- `Tests/UI/test_first_run_wizard_live_contract.py:86-92, 3170` (imports and uses `STEP_VOICE`/`STEP_PROTECT` on Quick);
- `Docs/User_Guide/First_Run_Setup.md:51-59` (the two-tracks section) and `:61-73` (step table).

**Preserve.** Restore a backup reachable from Welcome (kept, with the explanatory line); the Get started card catches Skip and Exit (Quick's exits are unchanged in kind); model-list consent asked once (unchanged; it stays on Ready).

### D2 — Full track: eleven real steps, in dependency order

**Decision.**

| # | Step (tracker label) | Default if untouched | Writes when untouched | Why this position | Content owner |
|---|---|---|---|---|---|
| 1 | Welcome | Quick | nothing | Entry | this spec |
| 2 | Connect | the detected or current provider | nothing new | Start of the only required pair; everything that talks to a model depends on it | .6 |
| 3 | Model | current or recommended model | nothing | Completes the pair. After it, "Finish with defaults" is offered | .7 |
| 4 | Server (optional) | **This computer only** | nothing | Decides where the runtime and data live, so it precedes the data features | this spec (§3.4) |
| 5 | Search (optional) | **only when you ask** | nothing | "What the assistant can see" … | .13 |
| 6 | Tools | the saved gates | nothing | … "then what it can do" (report §4.3) | .13 |
| 7 | Spoken replies (optional) | **No voice for now** (or the saved voice) | nothing | Late, because it can download (OmniVoice, 1.1 GB) | .8 |
| 8 | Dictation (optional) | **No dictation for now** (or the saved engine) | nothing | Next to Spoken replies so the pair reads as input and output, renamed so they stop reading as synonyms [cross-cutting-11]; downloads (Parakeet, 633 MiB) | .13, .14 |
| 9 | Appearance | the saved theme and splash | nothing | Nothing depends on it | .13 |
| 10 | Protect keys | **Keep as is** | nothing | After every step that can store a secret: provider key (2), server token (4), voice key (7) | .4, D9 |
| 11 | Ready | — | — | End | this spec |

- **The count stays 11** (today 11, `FRSS:969-981`). Notes leaves (TASK-34100.13 AC#7); Server joins. Voice moves from position 4 to 7; Speech becomes Dictation at 8; RAG becomes Search at 5.
- **"Finish with defaults".** On Full, once the Model step has an outcome (a saved model, or TASK-34100.7's "Skip — keep the key, no default model"), the nav row of steps 4–10 carries a secondary button **[ Finish with defaults ]**. It goes straight to Ready and leaves every unvisited step untouched (nothing written). Ready then shows those steps as one line: "Left at their defaults: Search, Tools, Spoken replies, Dictation, Appearance, Protect keys — change any of them from Review your setup." The button is not offered on Quick, where the next step is already Ready. It has **no new key chord**: Ctrl+Enter, which E5 suggested, reaches the app as plain Enter in most terminals, so a hint teaching it would teach a key that often does nothing. It is a Tab stop and a palette command ("Setup: Finish with defaults").
- **Every optional step defaults to "not now" and writes nothing when left untouched.** This spec states the requirement; the step owners implement it (TASK-34100.8 AC#1, .13 AC#1/#8/#12, .4 AC#7). The guard is the byte-identical invariant test from TASK-34100.9 AC#1, extended to Finish with defaults (§8.1).

**Why this order.** The report's rationale (report §4.3), accepted with one addition: Protect is last before Ready because it must review every secret stored earlier in the run. With D9, Protect becomes a review of where each key lives (keychain, environment, plain text, encrypted), not only an encryption offer.

**Rejected alternatives.**
- *Spoken replies stays at position 4.* It puts a download-bearing optional step in the early path. A Full user who abandons halfway should leave with the higher-value steps done.
- *Server first.* Most users have no server. The chat pair is the only required dependency and comes first.
- *Protect right after Connect.* The server token and voice keys come later, and Protect would miss them.
- *Appearance first, for reduced motion.* Reduced motion must apply before any animation plays, which means at launch or on Welcome (E7, owned by TASK-34100.13 AC#16), not as step 2.
- *A shorter Full track.* Expert speed comes from defaults and the exit ramp, not from fewer steps (report §4.3).

**Preserve.** Delta-aware writes (Tools, Appearance and Speech already write only what changed); the Parakeet install review (unchanged, moves with Dictation); scope restraint (no sampling, system prompt, hooks or MCP exposure in setup).

### D3 — tldw server: Runtime row, next step, optional Server step, one owner

**Decision.** Five parts.

1. **Ready always shows a Runtime row**, on both tracks and in the dashboard. Its value comes only from `RuntimePolicyContext`, ADR-033's sole authority for the active source (`runtime_policy/types.py:62-70`: `active_source`, `active_server_id`, `server_configured`, `server_reachability`, `last_known_server_label`):

   | State | Row |
   |---|---|
   | local, no server configured | `✓ Runtime   this computer only` |
   | server active, reachable | `✓ Runtime   this computer + tldw server lab.example.org:8000` |
   | server active, not reachable at the last check | `! Runtime   this computer + tldw server lab.example.org:8000 — not reachable now` |
   | server configured but not active | `✓ Runtime   this computer only · server lab.example.org:8000 is set up but not in use` |

   Local-only is a working, verified state, so it gets ✓, not the "skipped" dash. The `[tldw_api]` template values never count as a server: the template's `base_url = "http://127.0.0.1:8000"` (`tldw_chatbook/config.py:4064`) is not a binding, and the template token `default-secret-key-for-single-user` (`config.py:1510`, `:4066`) is screened by the existing `resolve_tldw_api_auth_token` (`config.py:1558-1577`, task-31417), which is reused, not copied.

2. **Ready offers "Connect a tldw server…"** in What's next (D6) whenever the runtime is local-only: always on Quick, and on Full when the Server step was left on "This computer only". It opens `ServerSwitchModal` (`Widgets/Settings_Widgets/server_switch_modal.py:31`) and commits its result through the shared coordinator (part 4). On success, the Runtime row refreshes in place.

3. **An optional Server step on Full only** (§3.4): "This computer only" (default, writes nothing) or "Also connect to a tldw server" (address, token, Test connection). The commit runs on **Save & continue**, never while typing.

4. **One commit path and one probe, shared with Settings.**
   - *Commit.* Today the whole switch lives in a Settings method, `SS:26513-26623`. It saves `[tldw_api]` URL and token to config.toml (`SS:26529-26537`), rebinds through `app.handle_runtime_backend_changed` (`app.py:1952-1975`), stores the token in the OS keyring with a config.toml fallback (`SS:26584-26597`), and prepares a Sync v2 profile (`SS:26599-26620`). That body moves to one app-level coordinator, which Settings, the Server step, the Ready next step and the plain CLI all call. Same order, same failure copy. ADR-033 already names "one app-level coordinator"; this extends its callers (§12.5, ADR-033 confirmation).
   - *Probe.* `ServerSwitchModal._run_connection_test` (`server_switch_modal.py:211-256`) GETs `/docs` for reachability, then POSTs `/api/v1/sync/send` with the token, and reports "Reachable (HTTP n)" for any server that answers. It cannot say "not a tldw server". The probe moves to one shared owner with five outcomes:

     | Outcome | Copy |
     |---|---|
     | Not reachable | ✗ Can't reach lab.example.org:8000 — nothing answered. Check the address, and that the server is running. |
     | Token rejected (401/403) | ✗ lab.example.org:8000 is a tldw server, but it rejected this token (HTTP 401). Check the API token in your tldw server's settings. |
     | Not a tldw server | ✗ lab.example.org:8000 answered, but it isn't a tldw server. Check the address and port. |
     | Certificate not trusted | ✗ lab.example.org:8000's certificate isn't trusted. Settings ▸ Network can add your organisation's certificate. |
     | Blocked by policy | ✗ chatbook's network policy blocks this address. (The existing egress check, `server_switch_modal.py:212-224`.) |
     | Success | ✓ Connected — lab.example.org:8000 is a tldw server and accepted the token. |
     | Success, no token | ! lab.example.org:8000 is a tldw server. No token was entered, so sign-in wasn't checked. |

     "Not a tldw server" is decided from the identity the server's own health and docs-info endpoints report, the same discovery calls `ActiveServerCapabilityService` uses after binding (`runtime_policy/server_capabilities.py:66-75`), run against the candidate URL before commit. The auth check must be non-mutating; the current POST to a sync endpoint with an empty body (`server_switch_modal.py:229-252`) is kept only if the follow-up confirms that tldw_server offers no authenticated read endpoint (risk K9).
   - *Storage.* The token goes to the OS keyring through the existing server credential store, as Settings does today. With D9 shipped, a secure keyring means the token is **not** also written to config.toml; until then, parity with Settings.

5. **Settings' button leaves the collapsed section.** "Switch Source / Server" moves from inside "Advanced / Diagnostics" (`SS:17648-17681`) into the main body of Settings ▸ Overview, as a Runtime row: "Runtime: This computer only  [ Connect a tldw server… ]", or "Runtime: This computer + tldw server lab.example.org:8000  [ Switch source… ]". It moves rather than being duplicated (one home), and keeps its id `settings-switch-runtime-source`.

**No Welcome router question.** Confirmed rejection (D12.1).

**Why.** PRODUCT.md promises visible source authority. A newcomer with a server should be offered the connection where setup happens. A Runtime row that is always present costs one line and answers "is this local-only?", which today is answered nowhere.

**Rejected alternatives.**
- *Server on Quick.* It taxes the local-first majority, and the Ready next step serves the minority in one action.
- *A provider-list row for tldw server.* A server is a runtime source, not a chat provider; mixing them would blur ADR-033's ownership.
- *A wizard-local copy of the switch logic.* Two writers for the runtime binding is exactly the drift ADR-033 forbids.
- *Show the Runtime row only when a server exists.* Then local-only is never said, and G3 fails.

**Consequence the owner must rule on.** Activation in Settings also prepares a Sync v2 profile for this device (`SS:26599-26620`; the modal says so, `server_switch_modal.py:122-126`). A Server step that shares the coordinator inherits that. §16 Q5 asks whether setup may do so; the recommendation is parity, with the step's copy saying it.

**Preserve.** Secrets never reach disk unless saved (the token goes to the keyring; the template placeholder is never shown as a token); specific connection errors (now including "not a tldw server").

### D4 — The Ready screen

**Decision.** Ready replaces today's Summary. Top to bottom (§3.6 has the copy and mockups):

1. **Verdict region.** One row, computed by Console's own send preflight through the shared readiness verdict that TASK-34100.2 AC#2 creates (today's ingredients: `Chat/console_prepared_request.py:1084` `resolve_request_capacity`, `Chat/provider_readiness.py:730` `get_provider_readiness`, `Chat/console_session_settings.py:1742` `build_console_settings_readiness`). The wizard calls it; it never copies it (TASK-34100.9 AC#10).
   - `✓ Ready to chat — <provider> · <model> (<context>)`, with Say hello (D5) beneath it.
   - `✗ Can't chat yet — <plain cause>`, with **one fix action** (two only where TASK-34100.9 AC#10 already names two: "Choose another model" and "Set context size…").
   - `– Chat not set up — no AI provider connected`, with [ Connect a provider ] (back to Connect).
2. **Read-back rows, only for steps the user saw.** On Quick: Connect (with key location), Model, plus the always-present **Runtime** row (D3) and, when more than one provider has a credential, a **Keys** row. On Full: rows in step order; steps skipped through Finish with defaults collapse into one "Left at their defaults: …" line. Rows are built from TASK-34100.9's outcome record and read back from disk (today's force-reload read-back, `FRSW:6683-6692`, is kept).
3. **"Data:" and "Config:" lines**, each with [ Copy ]. Data is `user_data_dir(config)` (`Backup_Recovery/profile_paths.py:86-90`, default `~/.local/share/tldw_cli/default_user`); Config is the effective config path (`profile_paths.py:25-29`, honouring `TLDW_CONFIG_PATH`). Paths are middle-truncated (existing helper `middle_truncate_path`, `FRSS:609-639`).
4. **What's next** (D6): up to three relevant items and a "More (N)" row, as one focusable list.
5. **Checkboxes:** the model-list consent box, shown only when a cloud provider covered by refresh is configured and naming it (TASK-34100.9 AC#8; offered once, `FRSW:6608-6620`, `:6755-6777`); "Get to know you after setup" (TASK-34100.10 AC#13).
6. **At most three exits**, docked:

   | Situation | Exits, left to right (first is primary) |
   |---|---|
   | First run, verdict ✓ | [ Start chatting ]  [ Add your first document ]  [ Write your first note ] |
   | First run, verdict ✗ or not set up | [ Go to Console ]  [ Add your first document ]  [ Write your first note ]; focus starts on the verdict's fix action |
   | Re-run walkthrough (TASK-34100.10 AC#14) | [ Done ]  [ Go to Console ]  [ Say hello ]; "Add a document" and "New note" move into What's next |

   "Explore Home" and "Open Settings" (today's "Review settings", renamed by TASK-34100.10 AC#12) move into What's next ▸ More. The two Library exits stay visible on first run, because task-32072 and task-32140 made them visible on purpose and TASK-34100.12 forbids hiding them.

**What "at most three visible actions" means here.** At most three exit buttons are docked. The verdict region carries at most one in-place action (Say hello, or the cause's fix), except the two-action case above. Link-outs live in the What's next list, which is one Tab stop whatever its length. No "More ▾" button is added; "More (N)" is the list's last row.

**Row budget at 80x24** (TASK-34100.12's short tier: title and tracker on one row, one nav row, one hint row): verdict 1–3, rows 3–4, Data and Config 2, What's next 5, checkboxes 1–2, exits 1. That is 17 rows at most, within the 19 that TASK-34100.12 AC#1 gives each step. Measured in the mockups (§3.6).

**Why.** The Summary is the peak of the emotional journey, and today it predicts an outcome it never checks (report §3.3 "Peak and end"). A verdict from Console's own preflight, a test reply on request, and three clear exits make the peak honest, and keep the end where the peak points.

**Rejected alternatives.**
- *Five exits, as today* (`FRSW:6640-6658`). Two docked rows at 80x24 cost the read-back its space [a11y-01], and choice overload at the end of setup is the wrong moment.
- *A "More ▾" button for the Library exits.* Rejected by the verifiers (report §5.4 table) and TASK-34100.12.
- *Block "Start chatting" when the verdict is ✗.* It would block the "set it up now, start the server later" path the report says to preserve (report §5.5). On ✗, Start chatting is replaced by [ Go to Console ], and Console's Get started card explains what is missing.

**Preserve.** The Summary reads back from disk; model-list consent asked once; the Library exits stay visible; "Review provider setup" recovers inside the wizard with the staged key intact (it becomes the verdict's fix action, landing on the step that needs it).

### D5 — "Say hello": a real first reply inside setup (E1)

**Decision.**

- **When it can run.** Only when the verdict is ✓. Sending a test that the offline preflight already knows will be refused spends tokens for nothing.
- **What is sent.** A one-line, editable prompt, prefilled with "Say hi in five words." Enter in the field, or the [ Say hello ] button, sends it.
- **Local providers: runs by default.** When the chat provider is a local engine (the catalog's locality fact, TASK-34100.6) **and** its endpoint host is a loopback address (`127.0.0.1`, `localhost`, `::1`), the test starts by itself the first time Ready renders in a run. It does not re-run on Back and Next unless the provider or model changed. The cost is no billing; it may load the model, which can take a while. Ready shows elapsed seconds, "(first reply can be slow while the model loads)", and [ Skip test ] (Esc). A local engine on a LAN address is treated like a cloud provider: no auto-run.
- **Cloud providers: only on an explicit press.** Ready shows the button and one consent line, computed from the actual prepared request: "Sends one message to OpenAI. Uses a few tokens: about 40 in, 256 out max." The words "Uses a few tokens" appear only while the estimate really is small (at most 500 tokens in and, when the catalog knows the price, under one US cent); otherwise the line states the number alone ("Uses about 4,800 tokens in, 256 out max."). When the model catalog carries a price for the model, a money estimate is appended ("≈ $0.0001"); when it does not, the line ends "billed at your usual rate". Pressing the button is the consent for that one message (ADR-012's 2026-09-26 amendment requires the paid test to be "a separate, consented action"). There is no extra dialog, nothing is remembered, and the test never runs automatically.
- **Reply cap.** The test turn caps the reply at 256 output tokens. That is enough for reasoning models, whose reasoning tokens count against the cap, and it bounds the worst case.
- **Path.** The test is a real Console turn. It is submitted through the app-scoped Console runtime (`app.console_runtime`, `app.py:1167`), which can run with no Console view mounted (`Chat/console_runtime.py:4128-4135`, "a runtime can be VIEWLESS FROM BIRTH"), using `ConsoleChatController.submit_draft` (`Chat/console_chat_controller.py:9797`) with origin `MANUAL` and a turn configuration that carries the reply cap. Admission, preflight, trace capture, persistence and dispatch are Console's own. The turn goes to a **saved** conversation, because Moonshot's failure [gap-02] appears only in saved chats.
- **Display.** The reply streams into the verdict region with the model id and latency: `✓ Replied in 0.8 s — gpt-4.1-mini: "Hello there, nice to meet you!"`, truncated to two rows. The full text is in Console.
- **Carried into Console.** The conversation is titled "Setup test · <provider> · <model>". **Start chatting** opens it, with the exchange as its first turn, through the same handoff staging the first-chat path uses today (`FRSW:9327`, PendingHandoffStore under ADR-033). On any other exit it stays in conversation history; it is the user's data and is never deleted silently.
- **On failure:** the plain cause plus one fix action, using TASK-34100.5 AC#4's categories:

  | Category | Ready copy | Fix action |
  |---|---|---|
  | Key rejected (401/403) | ✗ Can't chat yet — the test message failed: OpenRouter said "API key expired." (HTTP 401). Nothing else was sent. | [ Fix key ] (Connect, key field focused) |
  | Model not found (404) | ✗ Can't chat yet — Google doesn't have gemini-1.5-pro any more (HTTP 404). | [ Choose another model ] |
  | Local server not answering | ✗ Can't chat yet — nothing is answering at 127.0.0.1:9099. Start llama.cpp, then check again. | [ Check again ] |
  | First token too slow | ! No reply after 300 s. Large local models can take minutes to load. | [ Wait longer ] |
  | Reply arrived but couldn't be saved | ✗ The reply arrived, but chatbook couldn't save it. This is a chatbook problem, not your setup. | [ Open Logs ] |
  | Anything else | ✗ The test message failed: <provider's own message, capped at 200 characters and scrubbed, per .5 AC#4>. | [ Retry ] |

  A failed test updates the shared readiness through Console's own path (TASK-34100.2 AC#2: readiness leaves Ready after a refused or 401/404 send), so the verdict row flips to ✗ with no wizard-side logic.
- **Skip test.** While a test is running, [ Skip test ] (Esc) cancels it through Console's own stop. Offline users never need it on cloud providers, because nothing runs unless they press.

**Why.** A reply is the only proof that key, model, context budget, streaming and persistence work together. It catches runtime defects no offline check can [gap-02]. Sam's first reply came at 621 s, after 20 recovery actions in Console (report §1); a failed test inside setup points at the fix while the user is still in setup.

**Rejected alternatives.**
- *Auto-run for cloud providers.* It spends money without an action, contradicting ADR-012's explicit-only rule.
- *A consent modal.* The cost line plus an explicit press is the consent; a modal adds a dialog to the busiest end of setup.
- *A Temporary chat.* Temporary chats do not exercise persistence [gap-02].
- *A wizard-built request.* A copy of the request path would drift from Console; it would also not catch the ~20.8 KB agent preamble [new-entry-exit-handoff-01].
- *A fixed "Uses a few tokens" line with no numbers.* The cost depends on the real prepared request, and a model priced at $600 per million output tokens makes "a few" misleading. The phrase stays only when the computed estimate supports it.

**Dependencies.** TASK-34100.5 AC#6 (a minimal prompt for plain chat) must land first; otherwise the consent line would truthfully read thousands of tokens in, and "Uses a few tokens" would be dropped. TASK-34100.2 (readiness) and .3 (Moonshot persistence) also come first.

**Preserve.** Secrets never reach disk (the test carries no key in any persisted field or log; ADR-029); the typed-model "start the server later" path is never blocked (the test is optional and never gates Ready's exits).

### D6 — What's next, the arrival line, and just-in-time setup sheets (E9)

**Decision.**

1. **What's next on Ready.** One list, computed from config each time it renders; nothing about it is persisted. An item shows only when its area is not set up. Up to three show, then "More (N)":

   | Order | Item | Shown when | Opens |
   |---|---|---|---|
   | 1 | Encrypt saved keys with a password… | this run stored a provider key as plain text (D1) | TASK-34100.4's password dialog, in place |
   | 2 | Hear replies aloud… | no voice is configured | the single-step **Spoken replies** sheet (D7) |
   | 3 | Connect a tldw server… | the runtime is local-only (D3) | `ServerSwitchModal` + the shared coordinator |
   | 4 | Add a project folder… | no Workspace folder exists | Settings ▸ Workspaces |
   | 5 | Sync a notes folder… | no notes folder is synced | Library ▸ Notes ▸ Add from files… |
   | 6 | Add a document and ask about it | on re-run Ready, where the Library exits are not docked (on first run the docked "Add your first document" exit is this same route, so it is not repeated) | Library ▸ Import |
   | More | Add another provider · Explore Home · Open Settings · Export these settings (no keys)… · Web search keys · Tool permissions · Startup animation and splash | always | Connect (single step, then back) · Home · Settings · D11 · Settings ▸ Web Search · MCP ▸ Tools · Settings ▸ Splash Screen |

   Each destination comes from TASK-34100.15's registry; an item whose home does not exist yet is hidden (rule S4).

2. **Console arrival line.** TASK-34100.5 AC#11 replaces arrival toasts with one transcript line; this spec fixes its words: `Setup complete — OpenAI · gpt-4.1-mini · streaming on · 1M context. Switch models any time with Alt+M.` For a local model, `llama.cpp on this computer · llama-3.1-8b · streaming on · 8K context`. Every value comes from the resolved request settings, not from the wizard's memory. When the verdict was ✗, there is no line; the Get started card speaks instead.

3. **Just-in-time setup sheets.** Pressing Speak with no voice configured, or Dictate with no dictation engine, opens the single-step sheet for that area over Console (D7, §3.8). **Save** writes through the step's own commit and then does what the user asked (reads the reply aloud, or starts dictation). **Cancel** writes nothing.
   - The sheet **never accepts an API key** when opened from Console. A service that needs a key the user hasn't stored says "Needs an OpenAI key — add it in Settings ▸ Providers & Models…" and uses task-33008's round trip with return. This keeps the model-configuration redesign's owner ruling that no Console surface accepts or displays an API key (task-33008 AC#4).
   - Today's behaviour of Speak and Dictate with nothing configured is not re-checked in this spec. F10's first step records it, so the sheet replaces a known behaviour.

**Why.** Voice and dictation matter the first time a user presses them, not as a detour in a 2-minute track (E9). One list beats three toasts that cover the nav bar [new-entry-exit-handoff-02].

**Rejected alternatives.**
- *A persistent checklist that ticks items off across sessions.* It is state that can go stale; computing the list from config each time cannot.
- *Inline key entry in the Console sheet.* It contradicts that owner ruling.

**Preserve.** The Voice step's strengths (outcome first, plumbing under Advanced, a real verified sample with the existing key) carry into the sheet unchanged, because it is the same step module.

### D7 — Re-run as "Review your setup", with single-step changes (E5)

**Decision.**

- **Which runs see the dashboard.** When setup is opened while `[first_run] setup_completed` is true, or any provider is configured, the first screen is **Review your setup** (§3.7). A never-completed setup with a saved draft gets TASK-34100.10's resume dialog instead; a fresh profile gets Welcome.
- **Rows** are areas, in Full-track order: Provider, Default model, Runtime, Search, Tools, Spoken replies, Dictation, Appearance, Keys. Each shows its current value, read from the real config on every render (force reload, the same reader as Ready), with the outcome glyph from TASK-34100.9. The rows and Ready's rows come from **one row builder** (today `build_summary_rows`, `FRSS:1838`), so the two cannot disagree.
- **Interaction.** The rows are one list (highlight browses, per TASK-34100.11). Enter on a row is **Change**. A detail line under the list explains the highlighted row: its "!" reason, or what Change would do.
- **Change opens that single step and returns.** It runs in the single-step host: the step's own module, its own commit, a [ Cancel ] / [ Save ] pair instead of Back/Next. Changing Provider continues into Model only when the provider changed (the existing provider→model dependency). Save returns to the dashboard with that row refreshed, the highlight on it, and an in-place receipt, "Saved: Default model → gpt-4.1-mini". Cancel writes nothing.
- **Untouched areas are never written.** The dashboard performs no writes of its own; only a step's Save does, under the delta gate.
- **Done** returns to the origin given to TASK-34100.10's entry contract (`open_setup_wizard(origin, resume, start_step)`): Settings, the palette's screen, or Console. It writes nothing; setup is already complete.
- **Secondary actions:** [ Run the full walkthrough ] opens TASK-34100.10's prefilled corridor, with the last-used track preselected. [ Say hello ] runs D5 against the current default.
- **Palette commands per area**, each opening its single step directly and returning to where the palette was used: "Setup: Review your setup", "Setup: Change provider…", "Setup: Change default model…", "Setup: Connect a tldw server…", "Setup: Change document search…", "Setup: Change tools…", "Setup: Set up spoken replies…", "Setup: Set up dictation…", "Setup: Change appearance…", "Setup: Protect keys…", "Setup: Say hello", "Setup: Export these settings (no keys)…". Their names come from TASK-34100.15's registry.
- **Console readiness deep links land on the matching row.** When setup is complete, every Console readiness link (model blocked, key rejected, provider missing) opens the dashboard with the matching row highlighted and its reason expanded; Enter changes it. When setup was never completed, TASK-34100.10 AC#8 stands: "no provider configured" opens setup at Connect. No link opens Welcome.
- **Entry label.** Settings ▸ Overview's setup button reads **Review setup** when setup is complete, **Run setup** when it never ran, and **Resume setup** while a draft exists. "Review setup" is a third verb next to TASK-34100.15's "Run setup" and "Resume setup"; it goes into the same glossary.

**Consistency with TASK-34100.10.** .10 AC#1 opens a re-run on a "Review your setup" header with a short current-state summary and the track choice, in front of the prefilled corridor. This spec upgrades that screen: the summary becomes the actionable row list, and the track choice moves behind "Run the full walkthrough". .10 lands first; F11 (§10) replaces its body. The entry contract, the origin-return rule and the one-wizard guard are .10's and are reused unchanged.

**Why.** A re-run usually means "change one thing" (report §2(b)). A prefilled 11-step corridor still costs Riley about 75 keystrokes. From the palette, "change the default model" takes about five actions: open the palette, type "model", Enter, pick, Save.

**Rejected alternatives.**
- *The hub as first run too.* A newcomer needs a sequence, not a menu (report §4.3).
- *Edit values inline in the dashboard rows.* It duplicates each step's validation and commit; the single-step host reuses them.
- *Deep links straight into the step.* That saves one key, but the user would not see why they were sent there, and could not see the other areas.

**Preserve.** Resume and preview safety (the single-step host uses the same step lifecycle); exit and skip dialogs (Cancel in a single step needs no dialog, because nothing is staged outside the step); the Summary reads back from disk.

### D8 — Documents-first Welcome choice

**Decision.**

- Welcome's third choice: **"Start with my documents or notes — set up AI later"**.
- While it is selected, the tracker reads "Then: Library ▸ Import" instead of a step count (by rule S1, no steps follow), a one-paragraph explanation replaces the time line, and the primary button reads **[ Open Library → ]** (mockup §3.1).
- **Open Library finishes setup through the single finish path** that TASK-34100.10 AC#12 defines for every completing exit (today `_finalize`, `FRSW:9293-9322`). It records `setup_completed = true` and the model-list consent default (consent recorded, automatic refresh off), as TASK-34100.10 AC#7 makes Skip do. So the "Check model lists online?" modal is never raised; today's `_skip_entirely` records only completion (`FRSW:8987-9008`), which is why that modal fires after a skip [entry-exit-handoff-13]. Nothing else is written: no provider, no `chat_defaults`, no draft.
- **It lands on Library's Import canvas** through the existing "Add your first document" route (`FRSW:6880-6885`, task-32072), with **"Write a note"** as the canvas's secondary action. If the Import canvas does not already offer Library's starter "New note" action (ADR-076 starter landing), F7 adds it to the canvas's first-use state, routed like today's notes exit (`EXIT_ROUTE_LIBRARY_NOTES`, `FRSW:407-413`).
- **Console's Get started card is untouched.** With no provider, Console shows it as it does after Skip; its plain-language definition of a provider (`Chat/console_onboarding_state.py:19-22`) and its "Write a note in Library" action (`:48`) stay.
- **Skipping stays one gesture**: Esc on Welcome still raises today's Skip dialog (`FRSW:9744-9760`), with "Keep going" focused. The new choice adds nothing to it.
- **task-28019 is narrowed to its modal-sequencing AC#3** (§0.4). Its AC#1–#2 are delivered by F7; the narrowed task closes when TASK-34100.10 AC#18 lands.

**Why.** Dee came for documents. Setup is about AI providers and offers Dee nothing; a third choice costs one row and gets Dee to value in three keys (Down, Down, Enter) without a dead-end skip dialog [entry-exit-handoff-28].

**Rejected alternatives.**
- *A fourth Summary exit* (task-28019's attempt). It cannot fit at 80x24 (task-28019 notes), and it still makes Dee walk Provider and Model first.
- *A five-way router question.* Rejected (D12.1); this is one extra choice, not a router.
- *Make documents-first the default for profiles with no detected provider.* The default must be predictable; detection belongs in Connect.

**Preserve.** The Get started card catches Skip and Exit; model-list consent asked once (recorded by the finish path, never asked as a modal); Restore a backup reachable from Welcome.

### D9 — Keychain-first key storage (E8)

**Decision.**

- **A storage choice at every key field**: Connect (TASK-34100.6), Settings ▸ Providers & Models (ADR-012's owner), and the inline OpenAI key on Spoken replies (TASK-34100.8 AC#3 reuses Connect's path). One row, a select (DESIGN.md's dense-form one-row convention): **"Keep this key in [ System keychain (macOS Keychain) ▾ ] — recommended"** (mockup §3.2). Options, in order:
  1. **System keychain (<backend name>)** — recommended; default when a secure backend exists.
  2. **Encrypted in config.toml** — "asks for a password at every launch". When encryption is already on, this is the only config.toml option, because the writer encrypts sensitive values automatically (`config.py:7036-7062`).
  3. **Plain text in config.toml** — "only your user account can read the file" (true under ADR-029's owner-only file mode).
  4. **Don't store — read OPENAI_API_KEY at launch** — shown when an environment variable for this provider exists. It merges TASK-34100.6 AC#21's key-source choice into the same select, so the user sees one control, not two.
- **What a secure backend is.** The existing allowlist and detector, reused: macOS, Windows, Secret Service, libsecret and KWallet backends count; fail, null, plaintext and file backends do not (`runtime_policy/server_credentials.py:39-46`, `:423-449`).
- **Fallback when no secure keychain exists** (headless Linux, SSH, a locked or unresponsive Secret Service). The keychain option is shown disabled with its reason: "No system keychain on this computer (common over SSH)." The default becomes **Don't store** when an environment variable for the provider is set, otherwise **Plain text in config.toml**, with the one-line reason above. The probe runs in a worker with a short timeout, never on the UI loop; a timeout reads "System keychain didn't respond" and falls back the same way.
- **How keys resolve.** A new persisted `credential_source = "keychain"` joins `none`, `stored` and `environment` (`Chat/provider_readiness.py:235`). The key itself never appears in config.toml. One credential-store owner writes, reads, moves and deletes keychain keys, in a namespace scoped to the effective config path (the MCP store's pattern, `MCP/credential_bindings.py:144-150`), so two `TLDW_CONFIG_PATH` profiles never share a key by accident. Keychain values are resolved **once, at config load, into the in-memory settings**, the same choke point that decrypts `enc:` values today (`config.py:2159`, strict at `:6869`). Every reader — the spend path (`config.py:1728` `_normalize_legacy_provider_api_key`), `get_api_key` (`config.py:9948`), `resolve_provider_credential` (`provider_readiness.py:610`) and the readiness checks — therefore sees one value. This is the ADR-012 2026-09-19 lesson ("readiness and spend disagree") applied before it happens. Resolution is off the UI loop, bounded by a timeout, cached for the process lifetime and invalidated on write (ADR-097 boot budgets). An unresolvable keychain key reads "key missing — the system keychain didn't return it", never Ready, and nothing else is ever sent in its place.
- **Precedence** is unchanged in shape: an explicit stored credential (keychain, or config.toml stored) outranks the environment variable, which outranks legacy `[API]` (ADR-012 2026-09-19 amendment).
- **Ready and the dashboard name where each key lives**: "OpenAI · key in macOS Keychain", "from OPENAI_API_KEY (not stored)", "plain text in config.toml", "encrypted in config.toml" (TASK-34100.15 AC#5's key-source suffix, extended with keychain).
- **Migration: never automatic.**
  - *Plain-text keys already in config.toml* stay where they are. The dashboard's Keys row says where each key lives ("OpenAI: plain text in config.toml") with ✓, not a warning mark: plain text in an owner-only file is a legitimate choice, and "!" is reserved for states that stop or degrade something (TASK-25818's restraint). Highlighting the row offers the move. Protect keys (Full step and single step) and Settings ▸ Privacy & Security (TASK-34100.4's Encryption card) offer **Move to system keychain**. A move writes to the keychain, reads it back and compares, then removes the key from config.toml and sets `credential_source = "keychain"` in one config write. If any step fails, nothing is removed.
  - *Encrypted keys (`enc:`)* move the same way within an unlocked session; the values are already decrypted in memory. When the last `enc:` value has moved, Protect offers "Nothing is encrypted with your password any more. Turn password encryption off?", which goes through TASK-34100.4's lifecycle owner (`disable_config_encryption`, `config.py:9745`).
  - *Environment-variable users* are untouched.
- **Backups (ADR-126).** Keychain provider keys are Chatbook-owned keyring values, so the credential owner registers a typed adapter: excluded from portable export by default (ADR-126 decision 5), captured in local rollback archives (decision 8). A config.toml carried to another machine says `credential_source = "keychain"` but brings no key; readiness reads "Key not found in this computer's keychain" with [ Fix key ]. TASK-34100.16's "Setting up another machine" section says so.
- **"Remember on this device" is not shipped.** Storing the master password in the keychain makes encryption equivalent to keychain storage, with the password's extra failure modes added. Where a keychain exists, keychain-first already gives "no plain text, no daily password"; where none exists, there is nothing to remember it in. The "type it again" recall check is not shipped either: TASK-34100.4's "[R]eset saved keys" makes a forgotten password recoverable, and the setup dialog already confirms the password once.

**Why.** Most users want "my key isn't in a plain-text file" without a password at every launch, and the password path is the most fragile area in the review (SF6). keyring is already a core dependency (CLAUDE.md "Key Dependencies") and already holds server tokens and MCP bindings.

**Rejected alternatives.**
- *Encrypted-by-default.* A password at every launch is the lock-out path of [protect-summary-01/02].
- *Automatic migration of existing keys.* It writes what the user didn't touch (rule 2), and a half-finished move would split the key across two stores.
- *Keychain as a second lookup at send time.* Every reader would need to learn it, and they would disagree. Resolving at load gives one choke point.
- *Keychain-only, no plain-text fallback.* It strands SSH and headless users (Ash) with no supported store.

**Preserve.** Secrets never reach disk unless saved (keychain writes happen only on Save; drafts still refuse secret-named fields); env keys are named and never stored.

### D10 — Plain-text setup: `tldw-cli setup --plain` (E12)

**Decision.**

- **Entry.** A `setup` subcommand, dispatched before the TUI exactly as `recovery` is today (`cli.py:30-33`). It runs `startup_preflight` first, so an encrypted config unlocks through the one shared prompt (`cli.py:70-86`, TASK-34100.4 AC#3). Plain `tldw-cli setup` without `--plain` opens the TUI straight into setup: the dashboard when setup is complete (D7), otherwise Welcome.
- **Same state machine and commit path.** The plain driver walks the same track definitions (`active_step_ids`), the same pure step-state modules, and the same commit builders through the config owner. `first_run_setup_state.py` is already pure: its imports are stdlib plus one path helper, and it has no Textual import (`FRSS:15-25`). A plain step is a renderer over a step's state, never a second implementation.
- **One prompt per decision.** Numbered choices with a default in brackets; typing letters filters long lists (providers, models); `?` explains the current prompt; `q` quits (exit 1). Every line is a complete sentence. There is no cursor movement, no spinner, no colour-only meaning and no line rewriting, so screen readers read it in order and it is safe with `TERM=dumb`. Result lines start with `OK:` or `FAILED:`; with `--glyphs` they start with ✓ or ✗ instead (some screen readers read ✓ as "check mark", others skip it). Progress reads "Step 2 of 4: Connect".
- **Keys.** Read from a hidden prompt (`getpass`), or from an environment variable named by `--key-env VAR` (applied to the chat provider) or `--key-env PROVIDER=VAR`. Any argument that looks like a key flag (`--key`, `--api-key`, `--token`) is refused with exit 2: "Keys are never read from the command line, because they end up in shell history and process lists. Use --key-env NAME, or type the key when asked." A key is never echoed, logged, or written to a draft.
- **Scope of v1:** Welcome (all three choices), Connect (detection first, then the filterable list; key; storage choice once D9 ships), Model (curated list plus "type part of a model ID"), Server (Full only), Ready (verdict, Runtime and Keys lines, an optional Say hello asked as `[y/N]` for cloud and `[Y/n]` for loopback local, Data and Config paths). The other Full steps print one line each, "Search: not changed. Change it later with: tldw-cli setup (then Review your setup) or Settings ▸ Domain Defaults ▸ RAG", until their step state supports a plain renderer (F13b).
- **Exit codes.**

  | Code | Meaning |
  |---|---|
  | 0 | Setup saved and the verdict is ✓, or documents-first was chosen |
  | 1 | Quit before the provider and model were saved, or a save failed |
  | 2 | Usage error: bad flags, a key on argv, or no terminal (stdin is not a TTY and `--non-interactive` was not given) |
  | 3 | Saved, but the verdict is ✗ "Can't chat yet" |

- **Concurrency.** Writes go through the config owner's atomic path. Running setup while the TUI has the same config open is refused with "chatbook is running with this config. Close it, then run setup again." when the app's single-instance signal is available. Where it is not, the CLI warns and continues (risk K12).

**Transcript (Quick, Anthropic, keychain):** §3.10.

**Why.** Textual exposes no accessibility tree; a line-oriented flow on the same state machine is the cheapest way to make setup usable with a screen reader and over SSH, and it is the carrier for D11.

**Rejected alternatives.**
- *Make the Textual wizard screen-reader friendly.* Not possible without an accessibility tree.
- *A separate "setup script" that edits TOML.* It would be a second implementation that drifts.

**Preserve.** Secrets never reach disk unless saved; env keys never stored; model-list consent asked once (plain mode asks it once, default No, and records the answer); Restore reachable (plain Welcome prints "Moving from another computer? Run: tldw-cli recovery --help").

### D11 — Non-interactive setup and settings export (E6, beyond TASK-34100.16): build it, last

**Decision: build**, as the final slice (F14), on top of D10.

- **"Export these settings (no keys)…"** on Ready (What's next ▸ More), in the palette, and as `tldw-cli setup --export FILE`. It writes a small TOML file of setup choices; the format is in §4.5. It never contains a secret. The exporter refuses any field whose name looks secret, using the same rule drafts use to refuse secret fields.
- **`tldw-cli setup --from FILE`** prefills the plain flow from the file and asks only what is missing.
- **`tldw-cli setup --from FILE --non-interactive [--key-env VAR | --key-env PROVIDER=VAR …] [--key-stdin PROVIDER] [--server-token-env VAR] [--say-hello] [--dry-run]`** applies the file through the same commit builders, records `setup_completed`, runs the same readiness verdict, prints one `OK:` or `FAILED:` line per area, and exits with D10's codes.
  - Keys come only from environment variables or stdin.
  - A file that says a key lives in the keychain needs the key supplied by `--key-env` or `--key-stdin`; it is then stored in **this** machine's keychain, or the run fails with exit 1, naming the provider.
  - `--dry-run` prints which areas would change, by name, and writes nothing.
  - Say hello runs only with `--say-hello`, for cloud and local alike, because a script cannot consent interactively.
  - Unknown fields, a newer schema version, or any secret-looking field fail with exit 2 before anything is written.

**Why build rather than defer.**
- The marginal cost over D10 is small: the same entry point, the same state machine, answers from a file instead of prompts.
- It replaces today's working second-machine route — copying a whole config.toml with `[first_run] setup_completed = true` (report §2(c)) — which may carry plain-text keys and every unrelated setting. The export carries only setup choices and no keys.
- It gives provisioning scripts the readiness verdict and an exit code, which a copied file never had.

**Why last.** It depends on D10 and on the step owners' delta commits. If D10 slips, D11 slips with it, and §16 Q6 lets the owner choose deferral instead. The **reopen condition** in that case: the first user request for scripted setup, or the first support case caused by a copied config.toml carrying keys.

**Rejected alternatives.**
- *`tldw-cli setup --provider … --model … --yes`* (report §2(c) item 4). It is a second flag grammar for the same answers; a file is reviewable, diffable and versionable.
- *Import a full config.toml with a diff preview.* That is a different feature with secret-handling risk; TASK-34100.16 already keeps it out of scope.

**Preserve.** Secrets never reach disk unless saved (keys come from env or stdin and go to the chosen store); Restore a backup is unaffected, because an export is not a backup (ADR-126 confirmation, §12.6).

### D12 — The report's "Rejected, on purpose" list: confirm or overturn

The owner is asked to confirm each item. The recommendation for all seven is **confirm**.

| # | Item (report §4.3) | Recommendation | Why | Owner ruling |
|---|---|---|---|---|
| 1 | A five-way "How will you use chatbook?" router on Welcome | Confirm | It taxes the local-first majority on the first screen. Connect's "Ready on this machine" group detects the same things (TASK-34100.6 AC#29). D8's documents-first choice is one extra row, not a router. | pending |
| 2 | A master tool switch in setup | Confirm | It invites a reflexive "all off" that silently removes web search and Watchlists. TASK-34100.13 AC#2 states the real posture instead. | pending |
| 3 | An embedding-model picker | Confirm | Changing the model clones profiles and rebuilds the index. TASK-34100.13 AC#1 shows it read-only. | pending |
| 4 | A multi-tick provider checklist | Confirm | Env-keyed providers are recorded automatically (TASK-34100.6 AC#6). "Add another provider" is in What's next ▸ More (D6). | pending |
| 5 | Base-URL overrides on every keyed provider | Confirm | Few users run gateways, and the form would grow for everyone. Settings keeps the override (TASK-34100.6 AC#21). | pending |
| 6 | An "Undo this session" ledger before Voice writes only deltas | Confirm, and keep it rejected after the deltas ship | Restoring a snapshot races Settings and Console writers, and it cannot undo encryption or downloads. The Exit dialog's list of saved areas (TASK-34100.9 AC#3) plus D7's single-step changes cover the need. Reopen only on user reports of needing to undo a whole run. | pending |
| 7 | An inline "disable TLS verification" toggle | Confirm | It nudges first-time users toward `ssl_verify = false`. A certificate failure is classified and links to Settings ▸ Network (ADR-079; TASK-34100.6 AC#7; D3's probe). | pending |

---

## 2. Flows per persona

Key counts assume TASK-34100.6/.7/.11 have shipped (Enter selects, the detected row is pre-highlighted, the recommended model is pre-selected).

### 2.1 Sam — cloud, OpenAI key, Quick

1. Welcome: Enter (Quick is the default).
2. Connect: type `open` to filter, Enter on OpenAI, paste the key, Enter (checks it: "✓ Key works — OpenAI returned 137 models"). The storage select already reads "System keychain (macOS Keychain)". Next.
3. Model: the curated recommendation is pre-selected. Enter.
4. Ready: "✓ Ready to chat — OpenAI · gpt-4.1-mini (1M context)". The consent line reads "Sends one message to OpenAI. Uses a few tokens: about 40 in, 256 out max." Sam presses Say hello and sees "✓ Replied in 0.8 s". Start chatting opens Console on that conversation, with the arrival line above the exchange.

If the key is rejected at the test, Ready flips to ✗ with [ Fix key ]. Sam fixes it on Connect and comes back to Ready without re-walking Model.

### 2.2 Jo — private local AI

- **Path B, a server running.** Welcome Enter → Connect: the first row is "llama.cpp on this computer · 127.0.0.1:9099 · 3 models", pre-highlighted; Enter → Model: Enter → Ready. The test starts by itself: "Saying hello… 4 s (first reply can be slow while the model loads)", then "✓ Replied in 6.1 s". The Runtime row says "this computer only". About six keys from Welcome to a reply.
- **Path A, nothing running.** Connect lists what it checked and offers "Use <model> — I'll start it later" (TASK-34100.6 AC#18/#30). Ready says "✗ Can't chat yet — nothing is answering at localhost:11434. Start Ollama, then check again." with [ Check again ]. [ Go to Console ] still finishes setup; the Get started card takes over. No cloud-key form appears anywhere on this path.

### 2.3 Riley — power user

- **First install, env keys.** TASK-34100.2 AC#12's env-aware Get started card covers the zero-wizard path. To get the rest, Riley runs `tldw-cli setup`, chooses Full, keeps the pre-selected rows (OpenAI · "Don't store — read OPENAI_API_KEY"), sets the model, and presses **Finish with defaults** on the Server step. Ready shows "Left at their defaults: Server, Search, Tools, Spoken replies, Dictation, Appearance, Protect keys".
- **Re-run to change one thing.** Ctrl+P, "model", Enter → Model single step → pick → Save → back where Riley was. About five actions. Nothing else is written.
- **Second machine.** On machine 1: Ready ▸ More ▸ Export these settings (no keys) → `chatbook-setup.toml`. On machine 2: `tldw-cli setup --from chatbook-setup.toml --non-interactive --key-env OPENAI_API_KEY` prints one line per area and exits 0.

### 2.4 Dee — documents first

Welcome: Down, Down (the tracker now reads "Then: Library ▸ Import"), Enter → Library Import, with "Write a note" beside it. No modal fires. Console later shows the Get started card when Dee wants AI.

### 2.5 Ash — screen reader or SSH

`tldw-cli setup --plain` (transcript in §3.10). Over SSH with no keychain, the storage prompt offers "1. Plain text in config.toml (only your account can read the file)" and, when the env var is set, "Don't store — read ANTHROPIC_API_KEY at launch" as the default. The run exits 0 when the verdict is ✓.

---

## 3. Screen specs

Mockups assume TASK-34100.12's short tier at 80x24 (title and tracker share one row; no outer border; one nav row; one hint row). Glyphs follow TASK-34100.9's legend (✓ saved or kept current · – skipped · ! needs attention · ✗ failed) and survive NO_COLOR.

### 3.1 Welcome

<!-- mockup welcome 80x24 -->
```text
Welcome to chatbook                                  Step 1 of 4 · next: Connect

Chat with cloud or local AI models, keep notes, and work with your own
documents — all in your terminal. With a model on this computer, your chats
stay on it.

What would you like to do first?
 ● Quick setup — connect AI and pick a model (recommended)
 ○ Full setup — adds server, search, tools, voice, appearance
 ○ Start with my documents or notes — set up AI later

Quick takes about 2 minutes. Full takes about 10, plus any downloads you
choose. You can change any of this later by running setup again.

Moving from another computer?  [ Restore a backup ]







                                                                      [ Next → ]
↑↓ choose · Enter next · Esc skip setup
```

With the documents-first choice highlighted (short radio groups select on highlight, per TASK-21142 and TASK-34100.11):

<!-- mockup welcome-documents-first 80x24 -->
```text
Welcome to chatbook                                       Then: Library ▸ Import

Chat with cloud or local AI models, keep notes, and work with your own
documents — all in your terminal. With a model on this computer, your chats
stay on it.

What would you like to do first?
 ○ Quick setup — connect AI and pick a model (recommended)
 ○ Full setup — adds server, search, tools, voice, appearance
 ● Start with my documents or notes — set up AI later

Setup finishes now and opens Library, where you can import files or write a
note. Nothing else is changed. Connect an AI provider whenever you like from
the Get started card in Console.

Moving from another computer?  [ Restore a backup ]






                                                              [ Open Library → ]
↑↓ choose · Enter open Library · Esc skip setup
```

At 120x40 each choice gets a one-line gloss, and the tracker names the steps:

<!-- mockup welcome 120x40 -->
```text
Welcome to chatbook                                                 Step 1 of 4 · ● Welcome  ○ Connect  ○ Model  ○ Ready

Chat with cloud or local AI models, keep notes, and work with your own documents — all in your terminal.
With a model on this computer, your chats stay on it.

What would you like to do first?

 ● Quick setup — connect AI and pick a model (recommended)
     About 2 minutes. Connect a provider or a server on this computer, then choose the model new chats use.
 ○ Full setup — adds server, search, tools, voice, appearance
     About 10 minutes, plus any downloads you choose. Every extra step is optional and starts on "not now".
 ○ Start with my documents or notes — set up AI later
     Opens Library now. Import files or write a note; connect an AI provider whenever you like.

You can change any of this later by running setup again.

Moving from another computer?  [ Restore a backup ]




















                                                                                                              [ Next → ]
↑↓ choose · Enter next · Esc skip setup

```

- **Focus order:** the track list, then [ Next → ], then [ Restore a backup ] (TASK-34100.13 AC#15).
- **Tracker:** "Step 1 of 4" for Quick and "Step 1 of 11" for Full, updated as the highlight moves (TASK-34100.9 AC#5); "Then: Library ▸ Import" for documents-first.
- **Restore a backup** keeps today's handler (`FRSW:6392-6396`), and TASK-34100.16 makes it open on Inspect.

### 3.2 Connect: the key storage row (D9)

Only the storage row is new; the rest of Connect is TASK-34100.6's. With a secure keychain, the select open:

<!-- mockup connect-key-storage-fragment 80x9 -->
```text
OpenAI — API key
 [ ••••••••••••••••••••••••••••••••••••••abcd ]  [ Show ]  [ Check key ]
 ✓ Key works — OpenAI returned 137 models. Pick one on the next step.
 Keep this key in  [ System keychain (macOS Keychain)          ▾ ]
                     recommended · macOS may ask you to allow access once

   System keychain (macOS Keychain) — recommended
   Encrypted in config.toml — asks for a password at every launch
   Plain text in config.toml — only your user account can read the file
```

Without one:

<!-- mockup connect-key-storage-no-keychain 80x6 -->
```text
OpenAI — API key
 [ ••••••••••••••••••••••••••••••••••••••abcd ]  [ Show ]  [ Check key ]
 Keep this key in  [ Plain text in config.toml                 ▾ ]
                     No system keychain on this computer (common over SSH).
                     The file is readable only by your user account.
```

- Choosing **Encrypted in config.toml** for the first time opens TASK-34100.4's password dialog when the step is saved, not when the option is highlighted.
- Until D9 ships, the row is absent and keys go where they go today; Ready's "Encrypt saved keys…" option covers encryption on Quick.

### 3.3 Model: Finish with defaults (Full only)

On Full steps 4–10 the nav row reads `[ ← Back ]  [ Finish with defaults ]  …  [ Save & continue → ]` ("Continue" when the step will not write, per TASK-34100.9 AC#1). Pressing Finish with defaults on, say, Search goes to Ready, and Search, Tools, Spoken replies, Dictation, Appearance and Protect keys write nothing.

### 3.4 Server (Full, step 4)

<!-- mockup server-step 80x24 -->
```text
tldw server (optional)                               Step 4 of 11 · next: Search

chatbook works fully on this computer. If you run a tldw server, chatbook
can also connect to it and keep this device in sync with it.

 ● This computer only
 ○ Also connect to a tldw server

   Server address  [ https://lab.example.org:8000            ]
   API token       [ ••••••••••••••••••••••••              ]  [ Show ]
   [ Test connection ]
   ✓ Connected — lab.example.org:8000 is a tldw server and accepted
     the token. Connecting also prepares sync for this device.








[ ← Back ]  [ Finish with defaults ]                       [ Save & continue → ]

Enter choose · Ctrl+N next · Ctrl+B back · Esc exit setup
```

| Element | Copy and behaviour |
|---|---|
| Title | tldw server (optional) |
| Body | chatbook works fully on this computer. If you run a tldw server, chatbook can also connect to it and keep this device in sync with it. |
| Choice 1 (default) | This computer only. Writes nothing. Next reads "Continue →". |
| Choice 2 | Also connect to a tldw server. Reveals Server address and API token. |
| Server address | Placeholder `https://your-server:8000`. Prefilled only from a **bound** server (`RuntimePolicyContext`), never from the template's `127.0.0.1:8000`. Validated as a server root with no path, as the modal does today (`server_switch_modal.py:167-193`). |
| API token | Masked, with [ Show ]. Never prefilled with the template placeholder. A saved token shows as "saved in macOS Keychain" with [ Replace ] / [ Clear ] (the wizard spec's Keep/Replace/Clear for secrets). |
| Test connection | Runs D3's shared probe. Results use D3's copy. While it runs: "Checking lab.example.org:8000…". |
| Save & continue | Commits through the shared coordinator. A definitive failure (token rejected, not a tldw server, blocked by policy) refuses the save, with the cause under the field, as TASK-34100.6 AC#13 does for keys. An unreachable or untested server gets one inline line, not a modal: "lab.example.org:8000 wasn't confirmed as a tldw server.  [ Save anyway ]  [ Keep editing ]" ("Keep editing" focused), so "set it up now, start it later" still works. On a commit failure the step stays put with the cause, and nothing is half-applied (ADR-033: a failed commit leaves every observer on the old binding). |
| Re-run | With a server bound, choice 2 is pre-selected with the address and "token saved", and choosing "This computer only" switches back through the same coordinator. |

At 120x40 the layout is the same, with wider fields; the success line fits on one row.

### 3.5 Protect keys (Full, step 10)

State-aware content comes from TASK-34100.4 AC#7/#8. With D9, the step first lists where each key lives, then offers only the actions that apply:

| State | Body | Actions |
|---|---|---|
| Every key in the keychain or the environment | Your keys aren't stored in config.toml: OpenAI is in macOS Keychain; Anthropic comes from ANTHROPIC_API_KEY. Nothing to protect here. | (none; tracker "–") |
| A plain-text key, keychain available | OpenAI's key is plain text in config.toml. Only your user account can read the file. | [ Move to system keychain ]  [ Encrypt with a password… ]  [ Keep as plain text ] |
| A plain-text key, no keychain | (same body) | [ Encrypt with a password… ]  [ Keep as plain text ] |
| Already encrypted | Your saved keys are encrypted with your password. | [ Move to system keychain ]  [ Change password… ] |
| No key stored | Nothing to protect — no API key is saved in config.toml. | (none; tracker "–") |

### 3.6 Ready

**Quick, cloud, before the test (80x24):**

<!-- mockup ready-cloud-untested 80x24 -->
```text
Ready                                  Step 4 of 4 · ✓ Welcome ✓ Connect ✓ Model
✓ Ready to chat — OpenAI · gpt-4.1-mini (1M context)
  [ Say hi in five words.                       ]  [ Say hello ]
  Sends one message to OpenAI. Uses a few tokens: about 40 in, 256 out max.
✓ Connect   OpenAI · key in macOS Keychain
✓ Model     gpt-4.1-mini — new chats start with it (Alt+M switches)
✓ Runtime   this computer only
Data:   ~/.local/share/tldw_cli/default_user                     [ Copy ]
Config: ~/.config/tldw_cli/config.toml                           [ Copy ]

What's next — optional, any time
 ○ Hear replies aloud…
 ○ Connect a tldw server…
 ○ Add a project folder…
 ▸ More (8)
[ ] Refresh OpenAI's model list when chatbook starts
[ ] Get to know you after setup — a short questionnaire, no AI needed




[ Start chatting ]  [ Add your first document ]  [ Write your first note ]
← Back
Tab next action · Enter choose · Ctrl+B back · Esc finish setup
```

**Quick, local, test running by itself:**

<!-- mockup ready-local-testing 80x24 -->
```text
Ready                                  Step 4 of 4 · ✓ Welcome ✓ Connect ✓ Model
✓ Ready to chat — llama.cpp on this computer · llama-3.1-8b (8K context)
  Saying hello… 4 s  (first reply can be slow while the model loads)
  [ Skip test ]
✓ Connect   llama.cpp · on this computer (127.0.0.1:9099) · no key needed
✓ Model     llama-3.1-8b-instruct-q4_k_m.gguf
✓ Runtime   this computer only
Data:   ~/.local/share/tldw_cli/default_user                     [ Copy ]
Config: ~/.config/tldw_cli/config.toml                           [ Copy ]

What's next — optional, any time
 ○ Hear replies aloud…
 ○ Connect a tldw server…
 ○ Add a project folder…
 ▸ More (8)
[ ] Get to know you after setup — a short questionnaire, no AI needed





[ Start chatting ]  [ Add your first document ]  [ Write your first note ]
← Back
Esc skip test · Tab next action · Ctrl+B back
```

**After the reply:**

<!-- mockup ready-replied 80x24 -->
```text
Ready                                  Step 4 of 4 · ✓ Welcome ✓ Connect ✓ Model
✓ Ready to chat — OpenAI · gpt-4.1-mini (1M context)
✓ Replied in 0.8 s — gpt-4.1-mini: "Hello there, nice to meet you!"
  Start chatting continues this conversation.
✓ Connect   OpenAI · key in macOS Keychain
✓ Model     gpt-4.1-mini — new chats start with it (Alt+M switches)
✓ Runtime   this computer only
Data:   ~/.local/share/tldw_cli/default_user                     [ Copy ]
Config: ~/.config/tldw_cli/config.toml                           [ Copy ]

What's next — optional, any time
 ○ Hear replies aloud…
 ○ Connect a tldw server…
 ○ Add a project folder…
 ▸ More (8)
[ ] Refresh OpenAI's model list when chatbook starts
[ ] Get to know you after setup — a short questionnaire, no AI needed




[ Start chatting ]  [ Add your first document ]  [ Write your first note ]
← Back
Tab next action · Enter choose · Ctrl+B back · Esc finish setup
```

**Verdict ✗ from the offline preflight** (for example, a context window that is still unknown):

<!-- mockup ready-cant-chat 80x24 -->
```text
Ready                                  Step 4 of 4 · ✓ Welcome ✓ Connect ! Model
✗ Can't chat yet — gpt-5.6-terra's context size is unknown, so every
  message would be blocked before it is sent.
  [ Choose another model ]  [ Set context size… ]
✓ Connect   OpenAI · key in macOS Keychain
! Model     gpt-5.6-terra — saved, but it can't be used yet (see above)
✓ Runtime   this computer only
Data:   ~/.local/share/tldw_cli/default_user                     [ Copy ]
Config: ~/.config/tldw_cli/config.toml                           [ Copy ]

What's next — optional, any time
 ○ Hear replies aloud…
 ○ Connect a tldw server…
 ○ Add a project folder…
 ▸ More (8)
[ ] Get to know you after setup — a short questionnaire, no AI needed





[ Go to Console ]  [ Add your first document ]  [ Write your first note ]
← Back
Tab next action · Enter choose · Ctrl+B back · Esc finish setup
```

**The test failed** (an expired key on a provider whose model list proves nothing):

<!-- mockup ready-test-failed 80x24 -->
```text
Ready                                  Step 4 of 4 · ✓ Welcome ! Connect ✓ Model
✗ Can't chat yet — the test message failed: OpenRouter said
  "API key expired." (HTTP 401). Nothing else was sent.
  [ Fix key ]
! Connect   OpenRouter · key in system keychain — rejected by OpenRouter
✓ Model     anthropic/claude-sonnet-5-5
✓ Runtime   this computer only
Data:   ~/.local/share/tldw_cli/default_user                     [ Copy ]
Config: ~/.config/tldw_cli/config.toml                           [ Copy ]

What's next — optional, any time
 ○ Hear replies aloud…
 ○ Connect a tldw server…
 ○ Add a project folder…
 ▸ More (8)
[ ] Get to know you after setup — a short questionnaire, no AI needed





[ Go to Console ]  [ Add your first document ]  [ Write your first note ]
← Back
Tab next action · Enter choose · Ctrl+B back · Esc finish setup
```

**Full track at 120x40:**

<!-- mockup ready-full-track 120x40 -->
```text
Ready                                                                                              Step 11 of 11 · Ready

✓ Ready to chat — Anthropic · claude-sonnet-5-5 (200K context)
  [ Say hi in five words.                       ]  [ Say hello ]
  Sends one message to Anthropic. Uses a few tokens: about 40 in, 256 out max, billed at your usual rate.

✓ Connect          Anthropic · key in macOS Keychain · also OpenAI (key from OPENAI_API_KEY, not stored)
✓ Model            claude-sonnet-5-5 — new Console chats start with it (Alt+M switches)
✓ Runtime          this computer + tldw server lab.example.org:8000 · token in macOS Keychain
– Search           your Library is searched only when you ask (all-MiniLM-L6-v2)
✓ Tools            2 of 8 on: Read file, Search in files · web search and Watchlists also available (they ask first)
✓ Spoken replies   OpenAI · tts-1-hd · shimmer (uses your OpenAI key)
– Dictation        not set up · press Dictate in Console when you want it
✓ Appearance       Nord · dark · startup animation short · reduce motion off
✓ Keys             Anthropic: macOS Keychain · OpenAI: OPENAI_API_KEY · tldw server token: macOS Keychain

Data:   ~/.local/share/tldw_cli/default_user                                                   [ Copy ]
Config: ~/.config/tldw_cli/config.toml                                                         [ Copy ]

What's next — optional, any time
 ○ Add a project folder…
 ○ Sync a notes folder…
 ▸ More (7): Add another provider · Explore Home · Open Settings · Export these settings (no keys) · Web search keys · …


[✓] Refresh Anthropic's and OpenAI's model lists when chatbook starts
[ ] Get to know you after setup — a short questionnaire on this computer, no AI needed








[ Start chatting ]  [ Add your first document ]  [ Write your first note ]

← Back
Tab next action · Enter choose · Ctrl+B or Alt+← back · Esc finish setup

```

**Copy rules for Ready.**
- The verdict names provider and model by display name (TASK-34100.9 AC#6) and context in K or M tokens.
- The Say hello prompt is an editable one-line field prefilled with "Say hi in five words."; Enter in the field and the [ Say hello ] button both send it. The consent line under it is recomputed when the prompt changes.
- "Left at their defaults: …" appears only after Finish with defaults, and names the skipped steps in order.
- What's next shows only relevant items (D6). When none is relevant, the list holds only its "More (N)" row, under the heading "What's next — everything is set up; more options:".
- The footer hint is generated for the focused control (TASK-34100.11).
- **Esc** finishes setup with no dialog and lands wherever TASK-34100.10 sends every other completing exit (.10 AC#12/#17). Setup is never left half-finished from Ready.

### 3.7 Review your setup (re-run dashboard)

<!-- mockup review-your-setup 80x24 -->
```text
Review your setup                                           opened from Settings
Change one thing, or press Done. Nothing is saved until you save a change.

 ✓ Provider        OpenAI · key in macOS Keychain
 ✓ Default model   gpt-4.1-mini (1M context) · ready to chat
 ✓ Runtime         this computer only
 – Search          only when you ask
 ✓ Tools           2 of 8 on · web search, Watchlists ask first
 ✓ Spoken replies  OpenAI · tts-1-hd · shimmer
 – Dictation       not set up
 ✓ Appearance      Nord · dark · startup animation short
›✓ Keys            Anthropic: plain text in config.toml

   Keys — chatbook can move Anthropic's key to macOS Keychain. Enter opens
   Protect keys.

Data:   ~/.local/share/tldw_cli/default_user                     [ Copy ]
Config: ~/.config/tldw_cli/config.toml                           [ Copy ]



[ Done ]  [ Run the full walkthrough ]                             [ Say hello ]

↑↓ browse · Enter change · Esc done (returns to Settings)
```

At 120x40:

<!-- mockup review-your-setup 120x40 -->
```text
Review your setup                                                              opened from Settings · Done returns there
Everything setup manages, read from your saved settings. Change one thing, or press Done. Nothing is written unless you
save a change.

 ✓ Provider        OpenAI · key in config.toml (plain text) · also Anthropic (key from ANTHROPIC_API_KEY, not stored)
 ✓ Default model   gpt-4.1-mini (1M context) · ready to chat · last test replied in 0.8 s today
 ✓ Runtime         this computer + tldw server lab.example.org:8000 · reachable
 – Search          your Library is searched only when you ask (all-MiniLM-L6-v2)
 ✓ Tools           2 of 8 on: Read file, Search in files · web search and Watchlists also available (they ask first)
 ✓ Spoken replies  OpenAI · tts-1-hd · shimmer
 – Dictation       not set up
 ✓ Appearance      Nord · dark · startup animation short · reduce motion off
›✓ Keys            OpenAI: plain text in config.toml · Anthropic: ANTHROPIC_API_KEY (not stored)

   Keys — OpenAI's key is stored as plain text in config.toml (only your account can read the file). chatbook can move
   it to macOS Keychain. Enter opens Protect keys.

Data:   ~/.local/share/tldw_cli/default_user                                                                [ Copy ]
Config: ~/.config/tldw_cli/config.toml                                                                      [ Copy ]

















[ Done ]  [ Run the full walkthrough ]                                                                     [ Say hello ]

↑↓ browse · Enter change · Esc done (returns to Settings) · Ctrl+P "Setup: …" jumps straight to one area

```

| Element | Copy |
|---|---|
| Title | Review your setup |
| Origin note (right) | opened from Settings · opened from Console · opened from the command palette |
| Intro | Change one thing, or press Done. Nothing is saved until you save a change. |
| Row value when not set up | not set up |
| Detail line (highlighted row, marked ›) | for a "!" row, its reason; otherwise what Change would do ("Keys — chatbook can move Anthropic's key to macOS Keychain. Enter opens Protect keys.") |
| Receipt after a Save | Saved: <area> → <new value> (in place of the intro line, until the next highlight move) |
| Receipt after a Cancel | Nothing changed. |
| Exits | [ Done ] (primary) · [ Run the full walkthrough ] · [ Say hello ] (shown when a chat provider is configured) |

### 3.8 The single-step sheet

The single-step host renders one step module in a modal sheet, at most 76x20 and centred. At 80x24 it is effectively full width. The step's own body is unchanged; the nav is replaced by [ Cancel ] and a Save button whose label says what happens next ("Save", "Save and read this reply", "Save and start dictating"). Over Console, from Speak:

<!-- mockup spoken-replies-sheet-over-console 80x24 -->
```text
Console                                                       OpenAI · gpt-4.1
│ you  Summarise the attached report in three bullets.
│ ┌──────────────────────────────────────────────────────────────────────┐
│ │ Hear replies aloud                                                   │
│ │ Choose a voice service. You can change it later in Settings ▸        │
│ │ Speech & TTS.                                                        │
│ │                                                                      │
│ │  ● OpenAI — uses your OpenAI key (ready)                             │
│ │  ○ PocketTTS — not running at 127.0.0.1:8765                         │
│ │  ○ OmniVoice — on this computer · 1.1 GB download                    │
│ │  ○ Custom endpoint…                                                  │
│ │                                                                      │
│ │  Sample  [ Hello! This is how replies will sound.           ]        │
│ │  [ Test and Hear ]  ✓ Played the sample — sounds right?              │
│ │                                                                      │
│ │                    [ Cancel ]  [ Save and read this reply ]          │
│ └──────────────────────────────────────────────────────────────────────┘






Enter choose · Esc cancel (nothing is saved)
```

### 3.9 Console arrival line

<!-- mockup console-arrival 80x6 -->
```text
· Setup complete — OpenAI · gpt-4.1-mini · streaming on · 1M context.
  Switch models any time with Alt+M.
│ you  Say hi in five words.
│ gpt-4.1-mini  Hello there, nice to meet you!
```

The line is a dim system row at the top of the conversation; it is not a toast, and it is not repeated on later launches.

### 3.10 Plain-text setup transcript

<!-- mockup plain-cli 80x60 -->
```text
$ tldw-cli setup --plain
chatbook setup, plain text. Type a number or a word and press Enter.
Type ? for help or q to quit. Nothing is saved until a step says Saved.
Config: /home/ash/.config/tldw_cli/config.toml

Step 1 of 4: Welcome
What would you like to do first?
  1. Quick setup: connect AI and pick a model (recommended)
  2. Full setup: adds server, search, tools, voice, appearance
  3. Start with my documents or notes, set up AI later
Choice [1]: 1

Step 2 of 4: Connect
Looking for AI servers on this computer... done.
Ready on this computer:
  1. llama.cpp on this computer, 127.0.0.1:9099, 3 models
  2. OpenAI, key found in OPENAI_API_KEY
Or type part of a name to search all 60 providers.
Choice [1]: anthro
1 match:
  1. Anthropic (cloud, needs an API key)
Choice [1]: 1
Anthropic needs an API key. Get one from your Anthropic account.
API key (typing is hidden; Enter alone skips this provider):
OK: Anthropic accepted this key (13 models listed).
Keep this key in:
  1. System keychain (Secret Service) (recommended)
  2. Encrypted in config.toml (password at every launch)
  3. Plain text in config.toml (only your account can read it)
Choice [1]: 1
Saved: Anthropic key in the system keychain.

Step 3 of 4: Model
Recommended for Anthropic: claude-sonnet-5-5, 200K context.
  1. claude-sonnet-5-5 (recommended)
  2. claude-haiku-5 (fast and inexpensive)
  3. claude-opus-5
Or type part of a model ID to search all 13.
Choice [1]: 1
Saved: new chats use Anthropic, claude-sonnet-5-5.

Step 4 of 4: Ready
OK: Ready to chat: Anthropic, claude-sonnet-5-5, 200K context.
Runtime: this computer only.
Keys: Anthropic in the system keychain.
Send a test message to Anthropic? Uses a few tokens: about 40 in, 256 out max.
Anthropic bills this at your usual rate. Send it? [y/N]: y
OK: claude-sonnet-5-5 replied in 1.2 s: "Hi there, nice to meet you!"
Setup complete. Start chatbook with: tldw-cli
$ echo $?
0
```

- Lines are at most 80 columns. Nothing is redrawn in place.
- With `--glyphs`, "OK:" becomes "✓" and "FAILED:" becomes "✗".
- With `NO_COLOR` or `TERM=dumb`, the output is identical; it carries no colour to begin with.

### 3.11 Non-interactive run

```text
$ tldw-cli setup --from chatbook-setup.toml --non-interactive --key-env OPENAI_API_KEY
chatbook setup from chatbook-setup.toml (schema chatbook-setup/1).
OK: Provider: OpenAI, key from OPENAI_API_KEY (not stored).
OK: Default model: gpt-4.1-mini.
OK: Runtime: this computer only.
OK: Tools: 2 of 8 on.
OK: Appearance: Nord.
OK: Ready to chat: OpenAI, gpt-4.1-mini, 1M context.
Setup complete.
$ echo $?
0
```

---

## 4. State and persistence model

### 4.1 Tracks and the stable-total invariant

```text
QUICK = (welcome, provider, model, summary)
FULL  = (welcome, provider, model, server, rag, tools, voice, speech, appearance, protect-keys, summary)
DOCS  = (welcome)        # finishes on Open Library
```

- Step ids stay as they are, except the new `server`. `notes` leaves (TASK-34100.13). Display titles come from TASK-34100.15's registry: Connect, Model, Server, Search, Tools, Spoken replies, Dictation, Appearance, Protect keys, Ready.
- `active_step_ids(track)` drops its unused `key_entered` parameter once callers are migrated (today it is ignored, `FRSS:1284-1286`).
- **INV-1 (rule S1).** For any sequence of events after the run leaves Welcome, `active_ids` is constant until the user returns to Welcome. A Hypothesis property test generates event sequences (key typed, key cleared, env var present, provider switched, encryption enabled, Finish with defaults, Back) and asserts it (§8.1).

### 4.2 Session modes

| Mode | Entered from | First screen | Exits |
|---|---|---|---|
| `first_run` | boot offer, or `tldw-cli setup` on a fresh profile | Welcome | Ready's exits; Open Library (DOCS); Skip |
| `resume` | boot with a valid draft (TASK-34100.10) | the draft's step | as `first_run` |
| `review` | any re-run entry once setup is complete | Review your setup | Done → origin |
| `walkthrough` | "Run the full walkthrough" from `review` | Welcome, prefilled (TASK-34100.10) | Ready with Done → origin |
| `single_step` | a dashboard row, a palette command, a Ready next step, a Console sheet | that step | Save or Cancel → the caller |

All modes go through TASK-34100.10's `open_setup_wizard(origin, resume, start_step)`, extended with `mode` and `focus_area`. The one-wizard guard covers every mode.

### 4.3 What each surface writes

| Surface | Writes | Never writes |
|---|---|---|
| Welcome | nothing (the track choice is a draft value) | — |
| DOCS finish | `[first_run] setup_completed`, the model-list consent default | provider, `chat_defaults`, draft |
| A step's Save (any mode) | that step's own keys, as a delta | any other step's keys |
| Server Save | `[tldw_api]` URL; the token to the keychain (to config.toml only where no keychain exists, as Settings does); the runtime policy binding through the coordinator | `WIZARD_OWNED_SECTIONS` (`FRSS:1251-1266`) does not gain `tldw_api`: the coordinator owns that write, not the wizard |
| Ready | the consent answer and "Get to know you" through the finish path; Say hello writes a conversation (Console's store), never config | — |
| Dashboard | nothing itself | — |
| Export | a file the user names | config, keychain |
| `--from … --non-interactive` | as the steps' Saves, plus `setup_completed` | anything not in the file |

### 4.4 Draft compatibility (migration)

Drafts are validated against the current track: `_validated_setup_draft` rejects a draft whose `active_step_id`, or any value's step id, is not on the track (`FRSS:1074-1100`), and `read_setup_draft` then returns `None` (`FRSS:1144-1160`). So a draft saved by today's six-step Quick at Voice or Protect would be **silently dropped** after this change, which breaks rule 2. Therefore:

- `SETUP_DRAFT_VERSION` goes from 1 to 2 (`FRSS:33`).
- A one-way migrator reads version-1 drafts **before** validation:
  - It computes the steps completed under the old order (everything before the old `active_step_id`).
  - It resumes at the first step in the new order that is not in that set. Old Quick at Voice or Protect resumes at Ready. Old Full at Voice resumes at Server.
  - It drops values for steps no longer on the track (Voice values on Quick, Notes on Full) and **says so** in the resume dialog: "Voice is now set up from the Ready screen, so your earlier Voice choices weren't kept."
  - It writes back version 2.
- A version-2 draft read by an older build fails closed, as today (a downgrade loses the draft but never corrupts config).

### 4.5 The settings export format (D11)

```toml
schema = "chatbook-setup/1"
exported_at = "2026-10-03T16:00:00Z"
exported_by = "chatbook <version>"

[chat]
provider = "openai"
model = "gpt-4.1-mini"

[credentials.openai]
source = "environment"          # keychain | environment | prompt | none
env_var = "OPENAI_API_KEY"

[credentials.anthropic]
source = "keychain"             # the key itself is never exported

[runtime]
source = "local"                # or "server"
# server_url = "https://lab.example.org:8000"   # the token is never exported

[search]
auto_retrieve = false

[tools]                         # gate keys, as all_tool_gates() names them
read_file_enabled = true
search_files_enabled = true

[spoken_replies]
service = "openai"
model = "tts-1-hd"
voice = "shimmer"

[dictation]
engine = "none"

[appearance]
theme = "nord"
startup_animation = "short"
reduce_motion = false
```

- Strict schema: unknown tables or keys fail (exit 2).
- A field named like a secret (`api_key`, `token`, `password`, `secret`, `auth`) fails even when its value is empty.
- Only setup-owned areas appear. Sampling, prompts, hooks and MCP never do.

### 4.6 Keychain namespace (D9)

- Service: `tldw_chatbook.provider_credentials.<sha256 of the effective config path>`. Username: the provider's canonical key (`normalize_provider_config_key`, `config.py:1580`).
- The config records `credential_source = "keychain"` and no `api_key`. Server tokens keep their existing server-credential namespace.

---

## 5. Accessibility

- **Text carries every state.** Every mark is a glyph plus a word ("✓ Ready to chat", "! Runtime … not reachable now"); colour is decoration (PRODUCT.md "Accessibility & Inclusion"; TASK-34100.12 AC#4).
- **Keyboard.** Every action is reachable by Tab in reading order. Lists follow TASK-34100.11 (highlight browses; Enter, Space or a click selects). No new chords: Finish with defaults is a button and a palette command. Back keeps "← Back or Ctrl+B" and Alt+Left (TASK-34100.12 AC#9).
- **Focus.**
  - Ready focuses the primary exit when the verdict is ✓, and the fix action when it is ✗.
  - A local auto-test never steals focus.
  - The dashboard focuses its list, with the deep-linked row highlighted.
  - The single-step sheet focuses the step's first control and returns focus to the invoking control on close.
- **Motion.** The Say hello wait shows elapsed seconds as text, not a spinner, and honours reduce motion.
- **Sizes.** Every screen in §3 fits 80x24 with TASK-34100.12's short tier; CI checks 80x24, 100x30, 120x40 and 200x60 (TASK-34100.12 AC#16).
- **Screen readers and SSH.** `tldw-cli setup --plain` (D10) is the supported path: line-oriented, no redraws, words before glyphs.
- **Cognitive load.** Quick has three decisions (track, provider with key, model). Ready shows at most three exits. Choice labels lead with their keyword, so truncation never hides it (TASK-34100.12 AC#1).

---

## 6. Migration from today's flow

| Area | Today | After | How |
|---|---|---|---|
| Quick steps | 6: Welcome, Provider, Model, Voice, Protect, Summary (`FRSS:982-989`) | 4 | F5; draft migrator (§4.4) |
| Full steps | 11 with Notes (`FRSS:969-981`) | 11 with Server | F6, after TASK-34100.13 removes Notes |
| Welcome | 2 tracks + Restore (`FRSW:6377-6390`) | 3 choices + Restore | F5 (labels), F7 (third choice) |
| Summary exits | 5 in two docked rows (`FRSW:6640-6658`) | 3 exits + What's next | F8 |
| Protect on Quick | always (TASK-21148) | Ready option when a plain-text key was stored this run | F5 |
| Re-run | Welcome, `rerun=True` (`app_command_providers.py:982`, `app.py:4167`, `SS:31206`) | Review your setup | F11, after TASK-34100.10 |
| Server | Settings only, collapsed (`SS:17648-17681`) | Ready row + next step + Full step + Settings main body | F1–F4, F6 |
| Keys | config.toml plain or encrypted | keychain-first for new keys; existing keys untouched until moved | F12 |
| CLI | `recovery` only (`cli.py:30-33`) | `setup`, `--plain`, `--export`, `--from` | F13, F14 |

- **Users with a completed setup** see no change until they re-run setup (the dashboard) or press Speak or Dictate with nothing set up (the sheet).
- **Users mid-setup when they upgrade** resume through the migrator (§4.4), with the one-line note if any choices were dropped.
- **Tests to rewrite**, deliberately, in the slices named: the six-step pins listed in D1; `test_summary_five_actions_visible_and_focused_on_full_track` (`Tests/UI/test_first_run_wizard_live_contract.py:2426`), which becomes a three-exits test at 80x24 and 120x40 in F8; the Full-order pins in `Tests/Wizards/test_first_run_setup_state.py:476-506`, in F6. Each rewrite is RED-verified against the new behaviour first.
- **User Guide.** `Docs/User_Guide/First_Run_Setup.md` is rewritten across the slices: the two-tracks section (`:51-59`) becomes three choices; the step table (`:61-73`) follows D2; "Running it again" (`:120-127`) describes the dashboard; a new "Setting up from the command line" section covers D10 and D11. The page's "Verified against" header (`:3`) and italic stamps go, per CLAUDE.md (owned by TASK-34100.15 AC#7).
- **Backlog edits after approval:** task-28019 is narrowed to its AC#3 and TASK-34100.10's coordination note corrected (§0.4); TASK-21148 gets a note that AC#5's mechanism is superseded by this spec's rule S1; TASK-34100.13 AC#15 points its time line at D1's copy.

---

## 7. Preserve: what each change must not break (report §5.5)

| Change | Secrets never written unless saved | Model-list consent asked once | Summary read back from disk | Restore a backup reachable from Welcome | Get started card catches Skip and Exit | Other report §5.5 items at risk |
|---|---|---|---|---|---|---|
| D1 Quick 4 steps | Encryption still only on an explicit action | unchanged on Ready | unchanged | kept, with the explanatory line | unchanged exits | Exit and Skip dialogs (Welcome's Skip unchanged) |
| D2 Full order + Finish with defaults | untouched steps write nothing | unchanged | skipped steps read back as "Left at their defaults" | n/a | n/a | delta-aware writes; Parakeet install review |
| D3 Server | the token goes to the keyring; the template placeholder never shows as a token | n/a | Runtime row from `RuntimePolicyContext` | n/a | n/a | specific connection errors; ADR-033's failed commit leaves the old binding |
| D4 Ready | n/a | consent box offered once, cloud only | rows force-reloaded from disk | n/a | ✗ verdict → Go to Console → card | Library exits visible; typed-model "start later" path never blocked |
| D5 Say hello | no key in any persisted field or log | n/a | the verdict reflects Console's readiness after the test | n/a | n/a | streaming end to end; local detection |
| D6 What's next, arrival, sheets | the Console sheet never accepts a key | n/a | the list is computed from config | n/a | n/a | the Voice step's strengths (same module) |
| D7 Dashboard | only a step's Save writes | never re-asked | every row force-reloaded | n/a | Done returns to the origin | resume and preview safety; one-wizard guard |
| D8 Documents first | writes nothing but completion and the consent default | recorded by the finish path, never a modal | n/a | kept on the same screen | the card shows when Dee opens Console | Skip stays one gesture |
| D9 Keychain-first | the keychain write happens only on Save; drafts refuse secret fields | n/a | the Keys row from the resolved source | n/a | n/a | env keys never stored; ADR-029 private files |
| D10 Plain CLI | getpass or env only; never argv; never echoed | asked once, default No | Ready lines re-read from disk | the recovery hint is printed | n/a | robust basics (paths with spaces, bounded waits) |
| D11 Export and `--from` | the export refuses secret fields; keys only from env or stdin | `--from` records the file's answer, default No | `--dry-run` reads current config | n/a (an export is not a backup) | n/a | single sourcing (the same commit builders) |

---

## 8. Test and verification plan

### 8.1 Automated (each RED-verified against the pre-change code)

- **Pure state (Hypothesis where the project already uses it):**
  - track lists for each mode;
  - INV-1 over generated event sequences;
  - the draft migrator, for every old step id on both tracks;
  - the Ready exit set as a function of (mode, verdict), never more than 3;
  - What's next relevance from config;
  - the export round trip and the secret-field refusal over generated field names;
  - CLI argument parsing (key-like flags refused, exit 2);
  - plain-mode transcripts as golden files against a fake terminal.
- **Invariant:** "only Next, Done or Finish with defaults" over populated configs leaves config.toml byte-identical apart from first-run bookkeeping. This is TASK-34100.9 AC#1's test, extended to the dashboard (open, then Done), the DOCS finish (only `[first_run]` and `[model_catalog]` change), and a Cancel in every single step.
- **Widget (Pilot, real stylesheet):**
  - Welcome's three choices and tracker updates;
  - DOCS → Library Import, with no consent modal mounted;
  - Ready in states R-cloud, R-local, replied, ✗ preflight, ✗ test and not set up, at 80x24, 100x30, 120x40 and 200x60: the verdict row, at most three exits, and both Library exits on first run all inside the visible region;
  - the dashboard: Change → Save → returns with a receipt and a refreshed row; Change → Cancel writes nothing; deep link → highlighted row;
  - the Server step against local HTTP fixtures for unreachable, 401, not-tldw, TLS untrusted and success.
- **Say hello:** through the real `ConsoleRuntime` and controller with the provider gateway faked at its protocol boundary (`ConsoleProviderGatewayProtocol`, `console_chat_controller.py:4134`): a saved conversation is created and handed to Console; the reply cap is applied; failures map to D5's table; a cloud provider never auto-sends; a local loopback provider auto-sends once per run.
- **Keychain:** the credential owner against keyring's fail backend (unavailable) and an in-memory test backend; a census test that every provider-credential reader resolves through the load-time path, so no reader can bypass the keychain value.
- **Architecture:** the Server step and Settings import the same coordinator; there is no second writer of `[tldw_api]` (a grep-based guard, like the existing ownership guards).

### 8.2 Ratchets

No module-size or ADR-097 ratchet rises. In particular, `FRSW` stays at or below its row (`Tests/Architecture/test_module_size_ratchet.py:128`); new screens go in their own modules, after TASK-34100.1.

### 8.3 What does not count as evidence

A green Pilot run with a faked gateway is not evidence that Say hello works with a provider. A Summary reading ✓ is not evidence that a chat works. Live runs (§8.4) are required, per `backlog/docs/lessons-testing-evidence.md` and `lessons-live-verification.md`.

### 8.4 Live verification (per slice; fresh isolated profile with `TLDW_CONFIG_PATH` and `HOME`; real providers; real llama.cpp; no mock servers)

| Scenario | Pass when |
|---|---|
| Sam: OpenAI pasted, Quick, Say hello | Ready ✓ within 2 minutes hands-on from Welcome; the reply streams; Console opens on it |
| Jo B: real llama-server on a non-:8080 port | the auto-test replies; Runtime "this computer only"; about 6 keys from Welcome to a reply |
| Jo A: nothing running | ✗ with Check again; start the server; Check again → ✓ |
| An expired OpenRouter key | the test fails with the provider's message and [ Fix key ] |
| Dee | Library Import in 3 keys; no modal; config diff only in `[first_run]` and `[model_catalog]` |
| Riley re-run | palette → change model in ≤ 6 actions (a palette search counts as one); config diff only `chat_defaults.model` (+ provider model) |
| Real tldw_server | Server step success, then 401 with a wrong token, a wrong port (not tldw), and the server stopped (unreachable); Settings shows the same binding |
| macOS Keychain | key stored, relaunch through both entry points, readiness ✓, config.toml has no key |
| Headless Linux (container, no Secret Service) | keychain option disabled with its reason; plain-text fallback; plain mode exits 0 |
| VoiceOver on macOS Terminal with `--plain` | every prompt read in order; the key prompt silent; exit code 0 |

Evidence goes in each follow-up's Implementation Notes, never in the User Guide.

---

## 9. Rollout

- **Order:** cheap and independent first; shape changes after the sibling subtasks they stand on; new surfaces last. The two cheapest slices, F1 and F2, depend on nothing in the programme and can land in Wave A.
- **No feature flags.** Each slice is complete and shippable on its own. Where a later slice changes copy an earlier one shipped, the later slice owns the copy change and its test.
- **Release note per user-visible slice:** F5 ("Quick setup is now four steps"), F7, F11, F12 ("New API keys are stored in your system keychain"), F13 and F14.

---

## 10. Phased implementation: proposed follow-up tasks

Filed under TASK-34100 **only after approval**, with IDs assigned against `origin/dev` at filing time (lessons-backlog-hygiene). Every task carries the programme's ACs:
- verified live on a fresh isolated profile through the real app, with real providers and a real llama.cpp server, never mock servers, and the evidence in Implementation Notes;
- new behaviour covered by tests that fail on the pre-change code (RED verified), with the module-size ratchet not raised;
- the matching `Docs/User_Guide/` page updated, with no "Verified against" stamps.

| # | Slice | Decision | Size | Depends on | Notes |
|---|---|---|---|---|---|
| F1 | Settings ▸ Overview: move "Switch Source / Server" into the main body as a Runtime row | D3.5 | S | — | Cheapest; independent |
| F2 | Summary/Ready: an always-present Runtime row from `RuntimePolicyContext` | D3.1 | S | — | Lands on today's Summary; .9 later re-hosts it |
| F3 | One runtime-source coordinator + a shared candidate probe (unreachable / 401 / not tldw / TLS / policy), used by `ServerSwitchModal` and Settings | D3.4 | M | — | Settings behaviour unchanged except clearer probe copy |
| F4 | Ready next step "Connect a tldw server…" | D3.2 | S | F2, F3 | |
| F7 | Documents-first Welcome choice through the single finish path | D8 | M | .10 (finish path, consent default) | Takes task-28019 AC#1–#2; task-28019 keeps only AC#3 |
| F5 | Quick = 4 steps; Voice → Full; Protect off Quick; Ready "Encrypt saved keys…" and "Hear replies aloud…"; draft migrator v2; rewrite the 6-step pins; Welcome copy | D1 | M | .1, .4, .8, .9 | Supersedes TASK-21148 AC#5 (note added) |
| F8 | Ready layout: verdict region, ≤ 3 exits, What's next list, Data and Config | D4, D6.1 | M | .9 AC#10, .12, F5 | Rewrites the five-actions live-contract test |
| F11 | Single-step host, Review your setup dashboard, palette commands, readiness deep links, "Review setup" label | D7 | L | .10, .9, .15 | Replaces .10's re-run header body |
| F6 | Full = 11 with the Server step; Finish with defaults; final order and titles | D2, D3.3 | L | F3, .13, .15 | |
| F9 | Say hello | D5 | L | F8, .2, .3, .5 (AC#4, AC#6) | Starts with a spike: a viewless runtime submit with a reply cap |
| F10 | Arrival line content; Speak and Dictate setup sheets in Console | D6.2–3 | M | .5 AC#11, .8, .13, F11 | First step records today's Speak and Dictate behaviour |
| F12 | Keychain-first: credential owner, storage select (Connect + Settings), load-time resolution, Move-to-keychain, backup adapter | D9 | L | .4, ADR-012/029 amendments approved | Server token keychain-only when secure (ADR-033 note) |
| F13 | `tldw-cli setup` + `--plain` (Quick + Server + key storage) | D10 | L | .1, F5; F12 optional | F13b later adds plain renderers for the other Full steps |
| F14 | Export (no keys) + `--from FILE [--non-interactive]` | D11 | M | F13 | Owner may defer (§16 Q6) |

Order of work: **F1, F2 → F3 → F4 → F7 → F5 → F8 → F11 → F6 → F9 → F10 → F12 → F13 → F14.** F12 can run in parallel with F8 onward once its ADR amendments are approved.

---

## 11. Risks

| # | Risk | Mitigation |
|---|---|---|
| K1 | Say hello costs more than users expect (reasoning models, expensive models) | The cost line is computed from the prepared request; the reply is capped; cloud never auto-runs; a money estimate shows only from catalog data |
| K2 | Running a Console turn from the wizard couples setup to Console internals | The viewless runtime already exists; F9 starts with a spike; the test uses the public submit path only, never controller internals |
| K3 | Keychain backends prompt, hang or are locked (macOS access prompt, a locked Secret Service, SSH) | Worker-side probe with a timeout; honest fallback; cached resolution; never on the UI loop |
| K4 | A key ends up in two stores, or in none, after a move | Write → read back → compare → remove, in that order; failure removes nothing; the census test |
| K5 | A new credential source reopens "readiness and spend disagree" | Resolution at one load-time choke point; the census test in F12 |
| K6 | Reversing TASK-21148 brings back mid-flight count changes | INV-1 property test; rule S2 (options are not steps) |
| K7 | Test churn from the 6-step and five-exit pins hides real regressions | Rewrites are RED-verified against the new behaviour; pins are rewritten, not deleted |
| K8 | Setup's Server step prepares Sync v2 as a side effect | The copy says so; owner ruling Q5 |
| K9 | The shared probe misreads a reverse proxy as "not a tldw server", or the auth check mutates server state | Identity from health/docs-info only; a misread is fixable with the address (and is covered in live verification against a real server behind a proxy); "Save anyway" stays available for unreachable or untested servers; the POST probe is replaced if a read endpoint exists |
| K10 | Plain mode drifts from the TUI | One state machine; golden transcripts; one commit path |
| K11 | `--from` overwrites a hand-tuned config | Setup-owned areas only; `--dry-run`; a strict schema |
| K12 | CLI setup and a running TUI write the same config | The single-instance check where available; atomic writes; a warning otherwise |
| K13 | Ready or the dashboard overflow 80x24 if TASK-34100.12 slips | F8 and F11 are sequenced after .12; the size matrix in CI |
| K14 | Documents-first users never discover AI setup | The Get started card in Console; Library's assistant affordances; the palette's "Setup" commands |
| K15 | Draft migration bugs strand a mid-setup user | Migrator tests for every old step id; a fail-closed fallback offers "Start over", never a crash |

---

## 12. Proposed ADR changes (drafts; not applied)

ADR numbers collide across branches, so a new ADR's number is assigned at filing and re-verified at merge (`backlog/docs/lessons-backlog-hygiene.md:762-786`). Below, the new ADR is "ADR-NNN".

### 12.1 New: ADR-NNN — First-run setup shape and surfaces

> **Status:** Proposed (TASK-34100.17). **Date:** <approval date>.
>
> **Decision.** One setup owner, the setup state machine in `UI/Wizards/first_run_setup_state.py` and the step modules it drives, serves five surfaces: the first-run corridor, the re-run dashboard ("Review your setup"), single-step sheets (dashboard Change, palette commands, Ready next steps, Console Speak/Dictate), `tldw-cli setup --plain`, and `tldw-cli setup --from FILE [--non-interactive]`. Every surface uses the same track definitions, step modules, commit builders and readiness verdict.
>
> 1. Tracks: Quick = Welcome, Connect, Model, Ready. Full = Welcome, Connect, Model, Server, Search, Tools, Spoken replies, Dictation, Appearance, Protect keys, Ready. Documents-first finishes on Welcome.
> 2. Stable total: once the user leaves Welcome, the step list does not change until they return to it. Conditional offers are options on Ready, never steps. (Supersedes TASK-21148 AC#5's "Protect always on Quick"; keeps its guarantee.)
> 3. Every optional step defaults to "not now" and writes nothing when untouched. "Finish with defaults" is offered on Full once Model has an outcome.
> 4. Ready's verdict is computed by Console's shared send preflight, never by a wizard copy. Ready shows at most three exits; first run keeps both Library exits visible.
> 5. The optional test message goes through Console's own admission and dispatch into a saved conversation. It runs automatically only for local engines on loopback addresses; for every other provider only on an explicit press, with the computed token cost on screen. It is never run by a script without `--say-hello`.
> 6. A re-run of a completed setup opens the dashboard. Only a step's own Save writes. Done returns to the origin.
> 7. Keys are never read from argv by any setup surface. A settings export never contains a secret.
> 8. Setup does not offer: a Welcome router, a master tool switch, an embedding-model picker, a multi-tick provider list, per-provider base-URL overrides, an undo-this-session ledger, or an inline TLS-verification toggle.
>
> **Consequences.** The wizard spec's "re-run and first-run are one code path" (2026-07-28 §1) is replaced by "one state machine, several surfaces". ADR-076's "the only startup/setup owner" now refers to this owner with all five surfaces.

### 12.2 ADR-012 amendment — keychain as a credential store

> **Amendment <date> (TASK-34100.17, D9).** The Consequences sentence "This ADR does not introduce encrypted credential storage, keyring migration…" is narrowed: provider API keys may be stored in the OS keychain through one credential-store owner, recorded as `credential_source = "keychain"` with no `api_key` in config.toml. Settings ▸ Providers & Models and setup offer the same storage choice (keychain, encrypted config, plain config, environment). Keychain is the default where a secure backend exists. Precedence is unchanged in shape: a stored credential (keychain or config) outranks the environment variable, which outranks legacy `[API]`. Keychain values are resolved once at config load into the in-memory settings, so spend and readiness read one value. Existing keys move only on an explicit user action. Console still never accepts or displays a key (the model-configuration redesign's owner ruling, task-33008 AC#4).

### 12.3 ADR-029 amendment — the OS keychain inside the private-data boundary

> **Amendment <date> (TASK-34100.17, D9).** The rejected alternative "Replace config TOML with keyring/encrypted storage" is revisited for provider keys only. The OS keychain becomes a permitted location for provider credentials. Only secure backends count (the `runtime_policy/server_credentials.py` allowlist: macOS, Windows, Secret Service, libsecret, KWallet); fail, null, plaintext and file backends do not, and their absence falls back to the owner-only config.toml. Keychain reads run off the UI loop, with a timeout, cached for the process lifetime. A keychain value is never logged, never written to a draft or diagnostic, and never written back to config.toml by any writer. `config.toml` remains the sole persistence owner for configuration (this ADR's Decision), and the credential-store owner is the sole writer of provider keys in the keychain.

### 12.4 ADR-076 amendment — additional setup surfaces, one owner

> **Amendment <date> (TASK-34100.17).** "The existing application first-run wizard remains the only startup/setup owner" now means the setup owner of ADR-NNN, which presents a first-run corridor, a re-run dashboard, single-step sheets and two CLI surfaces on one state machine. None of these is a second onboarding wizard. Library still reads only the startup admission fact and never writes or reinterprets setup completion. The documents-first exit lands on Library Import through the existing exit route; Library's starter lifecycle is unchanged.

### 12.5 ADR-033 — confirm, with one clarifying amendment

> **Confirmed.** `RuntimePolicyContext` stays the sole authority for the active runtime source.
>
> **Amendment <date> (TASK-34100.17, D3).** "Settings saves the changed URL and token … and passes it to one app-level coordinator" applies to every setup surface: Settings, the setup Server step, Ready's "Connect a tldw server…", and `tldw-cli setup`. They call the same coordinator, and none writes `[tldw_api]` or the binding itself. With a secure OS keychain, the server token is stored only there (no config.toml copy); without one, the existing config.toml fallback stands. Binding a server prepares the Sync v2 profile on every surface alike (owner ruling Q5).

### 12.6 ADR-126 — confirm, with one amendment

> **Confirmed.** Restore stays reachable from first-run setup (Decision 1: "exposed through canonical F9 Settings, first-run setup, and a dependency-light pre-bootstrap recovery launcher"); it stays on Welcome, and plain mode names the recovery command.
>
> **Amendment <date> (TASK-34100.17, D9/D11).** Provider keys held in the OS keychain are Chatbook-owned keyring values under Decisions 5 and 8: excluded from portable export by default and captured in local rollback archives through a typed owner adapter. The setup settings export (`chatbook-setup/1`) is not a backup or a selective content export. It carries setup choices only, never secrets or content, and is out of this ADR's scope.

---

## 13. Current-behaviour citations (`origin/dev @ fcebe51a09`)

| Claim | Where |
|---|---|
| Quick track is Welcome, Provider, Model, Voice, Protect, Summary | `FRSS:982-989` |
| Full track has 11 steps, including Notes | `FRSS:969-981` |
| `active_step_ids` ignores `key_entered`; Protect always present (TASK-21148) | `FRSS:1271-1286` |
| Tracker titles (RAG, Speech, Style, Protect, Summary) | `FRSS:942-954` |
| Step ids that key drafts | `FRSS:991-1002` |
| Draft version is 1; a draft off the current track is rejected | `FRSS:33`; `FRSS:1074-1100`, `:1144-1160` |
| `WIZARD_OWNED_SECTIONS` has no `tldw_api` | `FRSS:1251-1266` |
| Setup state module is pure (no Textual import) | `FRSS:15-25` |
| Summary primary is "start_chatting" when Provider and Default model rows are configured and no probe failed | `FRSS:641-665`; `FRSW:6807-6818` |
| Summary rows builder | `FRSS:1838` |
| Env-key notice; wizard offer rule | `FRSS:873-911`; `FRSS:861-869` |
| Welcome: title, pitch, time copy, two tracks, Restore | `FRSW:6355-6390`; handler `:6392-6396` |
| Protect copy names a Skip and a Settings page; keyless state | `FRSW:6427-6436`; `:6441-6470` |
| Voice preselects PocketTTS at `127.0.0.1:8765` | `UI/Wizards/first_run_voice_step.py:123-127`; `first_run_voice_step_state.py:36` |
| Summary reads config with `force_reload=True` | `FRSW:6683-6692` |
| Summary consent box offered only while unanswered; its commit | `FRSW:6608-6620`, `:6755-6777`, `:6930-6955` |
| Summary's five exits in two docked rows; routing | `FRSW:6640-6658`; `:6871-6899` |
| Quick defaults note | `FRSW:6667-6671` |
| Skip path records completion only (no consent) | `FRSW:8987-9008` |
| Single finish path | `FRSW:9293-9322` |
| First-chat handoff staging | `FRSW:9327` |
| Notes exit sentinel | `FRSW:407-413` |
| Skip dialog copy | `FRSW:9744-9760` |
| Wizard screen; `setup_started` only on first run; key-hint line | `FRSW:9629-9680` |
| Re-run entry points (`rerun=True`) | `app_command_providers.py:975-988`; `app.py:4148-4171`; `SS:31197-31209` |
| No `tldw_api` reference under `UI/Wizards/` | `grep -rln tldw_api tldw_chatbook/UI/Wizards/` → no match |
| "Switch Source / Server" inside the collapsed "Advanced / Diagnostics" | `SS:17648-17681`; collapsed default `SS:3462` |
| Settings switch flow: save, rebind, keyring, Sync v2 | `SS:26477-26623` (save `:26529-26537`; keyring `:26584-26597`; sync `:26599-26620`) |
| `ServerSwitchModal` compose and probe | `Widgets/Settings_Widgets/server_switch_modal.py:93-156`, `:167-193`, `:211-256` |
| `[tldw_api]` template; placeholder token; screening | `config.py:4063-4066`; `config.py:1510`; `config.py:1558-1577` |
| Runtime source state fields | `runtime_policy/types.py:62-70` |
| Server capability discovery (health, readiness, docs info) | `runtime_policy/server_capabilities.py:66-75` |
| `handle_runtime_backend_changed` | `app.py:1952-1975` |
| Persisted credential sources | `Chat/provider_readiness.py:235`; resolver `:598-650` |
| Credential readers | `config.py:1728`; `config.py:9948`; `Chat/provider_readiness.py:610` |
| Secure keyring allowlist and detector | `runtime_policy/server_credentials.py:39-46`, `:423-467` |
| MCP keyring namespace | `MCP/credential_bindings.py:144-176` |
| Encryption: needed check, enable, disable, write-time encrypt, load-time decrypt | `config.py:9673-9688`, `:9718-9742`, `:9745`, `:7036-7062`, `:2159` |
| CLI dispatch and preflight | `cli.py:30-33`, `:35-56`, `:70-86` |
| App-scoped, viewless-capable Console runtime; submit entry; gateway protocol | `app.py:1167`; `Chat/console_runtime.py:4128-4135`; `Chat/console_chat_controller.py:9797`, `:4134` |
| Capacity and readiness owners | `Chat/console_prepared_request.py:1084`; `Chat/provider_readiness.py:730`; `Chat/console_session_settings.py:1742` |
| Get started card copy and notes action | `Chat/console_onboarding_state.py:15-22`, `:48` |
| Data and config paths | `Backup_Recovery/profile_paths.py:21-35`, `:86-90` |
| Wizard module size ratchet row | `Tests/Architecture/test_module_size_ratchet.py:128` (9,866) |
| Six-step and five-exit test pins | listed in D1 and §6 |
| User Guide track section, step table, re-run section, stamp | `Docs/User_Guide/First_Run_Setup.md:51-59`, `:61-73`, `:120-127`, `:3` |

---

## 14. TASK-34100.17 acceptance criteria → sections

| AC | Where |
|---|---|
| #1 tldw server | D3; §3.4; F1–F4, F6 |
| #2 Quick 4 steps; TASK-21148 supersession; Welcome copy; pinned tests and guide | D1; §6 |
| #3 Documents-first; task-28019 | D8; §0.4 |
| #4 Spec exists; claims cited | this file; §13 |
| #5 Full list and order; not-now defaults; Finish with defaults | D2; §3.3 |
| #6 Ready screen | D4; §3.6 |
| #7 Say hello | D5 |
| #8 Re-run dashboard | D7; §3.7 |
| #9 Keychain-first | D9; §3.2; §4.6 |
| #10 What's next, arrival line, sheets | D6; §3.8–3.9 |
| #11 Plain-text setup | D10; §3.10 |
| #12 Non-interactive setup | D11; §3.11; §4.5 |
| #13 Preserve | §7 |
| #14 ADR changes | §12 |
| #15 Rejected-on-purpose rulings | D12 (owner rulings pending) |
| #16 Owner approval recorded | §15 (pending) |
| #17 Follow-up tasks | §10 (to file after approval) |

---

## 15. Approval record

| Field | Value |
|---|---|
| Spec revision approved | — (pending; this is revision 1) |
| Date | — |
| Owner rulings on Q1–Q8 | — |
| Owner rulings on D12 items 1–7 | — |
| ADR drafts approved (§12) | — |

Implementation does not start, and the §10 tasks are not filed, until this table is filled in and TASK-34100.17's notes record the same.

---

## 16. Open questions for the owner

Each has a recommendation; only owner-level calls are listed.

| # | Question | Recommendation |
|---|---|---|
| Q1 | Approve the 4-step Quick track (D1), superseding TASK-21148 AC#5 while keeping its stable-total guarantee as rule S1? | **Approve.** |
| Q2 | Say hello spending policy (D5): auto-run only for local engines on loopback; cloud only on an explicit press with the computed cost shown and no extra dialog; the test kept as a saved "Setup test" conversation that Start chatting continues. | **Approve.** If you prefer not to keep the conversation on non-Console exits, the alternative is to delete it on those exits and say so on screen. |
| Q3 | Make the system keychain the default store for new provider keys (D9), with an honest plain-text fallback where no secure keychain exists, and no automatic migration of existing keys? | **Approve.** It removes the daily-password trade-off for most users without touching anyone's existing setup. |
| Q4 | Leave "Remember on this device" (and the recall check) out of the master-password path? | **Leave out.** The keychain-first default covers the need, and TASK-34100.4's reset makes a forgotten password recoverable. |
| Q5 | Should binding a tldw server from setup also prepare the Sync v2 profile, as Settings does today (D3)? | **Yes, keep parity**, with the step's copy saying so. Different behaviour in setup and Settings would split the runtime-source owner. |
| Q6 | Build non-interactive setup and the settings export now (sequenced last, F14), or defer? | **Build, last.** Small over D10, and it retires copying whole config.toml files between machines. Reopen condition if deferred: the first user request for scripted setup, or the first support case from a copied config carrying keys. |
| Q7 | Plain-text setup v1 scope (D10): Quick + Server + key storage, with the other Full steps following as F13b? | **Approve.** It gets screen-reader and SSH users to a working chat first. |
| Q8 | Confirm the seven "Rejected, on purpose" items (D12)? | **Confirm all seven.** |
