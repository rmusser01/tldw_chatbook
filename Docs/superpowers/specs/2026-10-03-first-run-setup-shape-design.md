# First-run setup shape: design

- **Status:** Draft for owner review, **revision 2**, 2026-10-03. Revision 1 was reviewed by two independent critics, one on HCI and one on engineering. Every point they raised is answered in §17 (Review history), either by a change or by a stated reason for keeping the revision-1 position. **Not approved.** No wizard code changes under TASK-34100.17 until the owner records approval (§15). The follow-up tasks in §10 are filed only after approval.
- **Owner:** the project owner (@rmusser01).
- **Task:** [TASK-34100.17](../../../backlog/tasks/task-34100.17%20-%20Owner-approved-design-spec-for-the-setup-flow-Quick-track-tldw-server-re-run-dashboard-Say-hello.md), under the burn-down programme [TASK-34100](../../../backlog/tasks/task-34100%20-%20First-run-setup-wizard-burn-down-of-the-2026-10-02-UX-review.md).
- **Evidence:** the 2026-10-02 senior design / HCI review, [`Docs/superpowers/qa/first-run-wizard-ux-review-2026-10-02/README.md`](../qa/first-run-wizard-ux-review-2026-10-02/README.md). This spec cites it as "report §N", its structural fixes as SF1–SF10, its enhancements as E1–E12, and its register issues by id in brackets (for example [coverage-06]).
- **Prior designs this builds on:** the [first-run setup wizard design](2026-07-28-first-run-setup-wizard-design.md) (2026-07-28, "the wizard spec"), the [master shell UX design](2026-05-02-new-user-first-run-shell-ux-design.md), [`PRODUCT.md`](../../../PRODUCT.md) and [`DESIGN.md`](../../../DESIGN.md).
- **Base:** `origin/dev @ fcebe51a09` (2026-10-03). Every statement about current behaviour is cited `path:line` against that commit, and §13 lists them in one place. Path shorthand: `FRSW` = `tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py`; `FRSS` = `tldw_chatbook/UI/Wizards/first_run_setup_state.py`; `SS` = `tldw_chatbook/UI/Screens/settings_screen.py`.
  - **Re-anchoring rule.** Line numbers are pinned to `fcebe51a09` and will drift, because TASK-34100.1 moves every step class out of `FRSW`. Each follow-up re-anchors by symbol (`active_step_ids`, `WelcomeStep.compose_step`, `SummaryStep._render_rows`, `handle_switch_runtime_source`, `ServerSwitchModal._run_connection_test`, `_continue_first_run_wizard_result`), never by line.
- **Mockups:** every mockup was measured by script against its stated width and height (80x24, 120x40 or a stated fragment size): no line is wider and no block is taller. Model ids, paths and host names in mockups are illustrative.

---

## Summary for the owner

Setup today is one fixed corridor. This spec gives it a shape that matches what people came to do. It asks you to approve eleven decisions and to rule on the ten questions in §16.

| # | Decision | In one line |
|---|---|---|
| D1 | Quick track | **4 steps: Welcome → Connect → Model → Ready.** Voice and Protect leave Quick. This reverses TASK-21148 AC#5 but keeps its real guarantee, now written as an invariant: once you leave Welcome, the step count never changes. |
| D2 | Full track | **11 steps, in dependency order:** Welcome, Connect, Model, tldw server, Search, Tools, Spoken replies, Dictation, Appearance, Keys, Ready. Untouched optional steps write nothing. **Save and finish with defaults** is offered on Model, and **Finish with defaults** on every later step. |
| D3 | tldw server | An always-shown **tldw server** row on Ready (the AC's "Runtime" row, named for what it measures), a **"Connect a tldw server… (if you run one)"** next step, and an optional Full step. One shared probe says *unreachable / token rejected / not a tldw server*. One coordinator, shared with Settings (ADR-033), rolls back the config write when the bind fails. Sync is disclosed before the choice. Settings' button leaves the collapsed section. |
| D4 | Ready screen | A verdict from Console's own shared readiness, in **four honest states**: Ready to chat · Set up, key not checked yet · Can't chat yet · Chat not set up. The test result sits on its own line and never contradicts the verdict. Rows appear only for steps you saw, plus Data and Config lines. First run docks **three exits on one nav row**. It fits 80x24 on **both** tracks, with the glyph legend. Esc never finishes setup. |
| D5 | Say hello (E1) | One test message through Console's real admission and dispatch, as a **probe turn**: no tools, search, history or persona, a model-aware reply cap, and no hidden-turn toast. **Auto-run only on first run, only for a known local engine on loopback with the model already loaded.** Cloud runs only on a press, with a money-first cost line on screen. **Start chatting opens a new chat**; the test stays in History and the arrival line quotes it (Q2). |
| D6 | What's next (E9) | Up to three relevant next steps (`→`), plus "More (N)". The Console arrival line replaces toasts. Pressing Speak or Dictate with nothing set up opens a setup sheet right there; the sheet never takes an API key. |
| D7 | Re-run (E5) | A re-run of a working setup opens **"Review your setup"**: the verdict with Say hello, then one row per area. Enter changes just that area and comes back. **Done** returns where you started. One predicate decides dashboard or corridor: *is a usable chat provider configured?* Console's readiness links keep going to the control that fixes them (TASK-34100.10 AC#8; Q9). |
| D8 | Documents first | A third Welcome choice, **"Start with my documents or notes — set up AI later"**. It finishes setup at once, opens Library Import and records that AI setup was deferred, so the next "set up AI" opens Quick at Connect, not the dashboard. |
| D9 | Keychain-first keys (E8) | New keys go to the **system keychain** by default, shown as one sentence, "Saved in macOS Keychain · Change where…". **An environment key is never stored**: when one exists, using it is the default. Resolution runs after first paint and uses a provenance overlay, so no writer can copy a keychain key into config.toml. A failed keychain save never silently falls back to plain text. Existing keys move only when you ask. |
| D10 | Plain-text setup (E12) | `tldw-cli setup --plain` runs on one setup core outside `UI/` that never imports Textual. It has one prompt per decision, keys from a hidden prompt or an environment variable, never from argv. **v1:** Welcome, Connect, Model, key storage, the verdict, and a plain review of those areas. No test message and no server step until F13b. |
| D11 | Non-interactive setup (E6) | **Defer** (changed from revision 1), as the report's verifiers advised. TASK-34100.16 documents the copy-config route. Reopen on the first user request for scripted setup, or the first support case caused by a copied config carrying keys. |

You are also asked to confirm the report's seven "Rejected, on purpose" items (D12). §12 drafts one new ADR and five ADR amendments or confirmations, and none is edited until you approve.

**What changed in revision 2** (the full list is in §17):
- **Honesty.** Ready gains a fourth verdict state, a separate test line, quota and rate-limit failures, and a model-aware reply cap (D4, D5).
- **Safety.** Local auto-run is narrowed to an allowlist. Environment keys are never stored. Keychain resolution is moved after first paint, behind a writer guard (D5, D9).
- **Contradictions resolved.** One entry predicate serves D7, D8 and D10. Console links follow TASK-34100.10 AC#8. Re-run exits follow TASK-34100.10 AC#14 (D4, D7, D8).
- **Fit.** Full-track Ready is now mocked at 80x24, with the legend and a one-row nav (§3.6).
- **Engineering seams the shape needs** are named, and an enabler slice, F0, comes before the UI slices. The CLI cannot host Console or the runtime coordinator, so plain mode has no Say hello and no server step in v1 (§4.7, §8.2, §10).
- **D11 is deferred.**

**What this spec does not do.** Sixteen sibling subtasks (TASK-34100.1–.16) fix the defects inside today's steps. This spec covers only the shape decisions they leave open (§0.4). It changes no code.

---

## 0. Problem, goals, personas

### 0.1 Problem

Setup does not adapt to what the user came to do. Five shape problems remain after the sibling subtasks:

1. **Quick carries two steps most people don't need.** The Quick track is Welcome, Provider, Model, Voice, Protect, Summary (`FRSS:982-989`).
   - Voice preselects PocketTTS (`UI/Wizards/first_run_voice_step.py:123-127`), a separate local server at `127.0.0.1:8765` (`first_run_voice_step_state.py:36`) that most newcomers don't run [voice-speech-03].
   - Protect has always been on the track since TASK-21148 (`FRSS:1271-1286`). For keyless, local and env-key users it renders "No API keys saved yet — nothing to protect" (`FRSW:6463-6467`) [coverage-16] [protect-summary-06].
   - Welcome still sells Quick as "Quick setup — provider, model, voice, protection" (`FRSW:6382-6386`).
2. **A tldw server is never offered.** Nothing under `tldw_chatbook/UI/Wizards/` references `tldw_api` (a repository grep on `fcebe51a09` returns no match). The only control is "Switch Source / Server" (`SS:17676-17681`), inside a collapsible titled "Advanced / Diagnostics" (`SS:17648-17651`) that starts collapsed (`SS:3462`). The Summary never says whether chatbook runs local-only [coverage-06].
3. **A documents-first user must get through Provider and Model.** Welcome offers only Quick and Full, plus "Restore a backup" (`FRSW:6377-6390`) [entry-exit-handoff-28].
4. **A re-run replays first run.** Every re-run entry pushes the same wizard with `rerun=True` (`app_command_providers.py:975-988`, `app.py:4148-4171`, `SS:31197-31209`) and lands on Welcome. TASK-34100.10 makes that corridor respect current values, but it remains a corridor [cross-cutting-05].
5. **"Done" means "saved".** The Summary's primary action becomes "Start chatting" when the Provider and Default model rows are configured and no probe failed (`FRSS:641-665`, `FRSW:6807-6818`). Nothing checks that a turn can be sent [cross-cutting-01]. TASK-34100.9 AC#10 adds a "Ready to chat?" row. The screen around it, the optional test message and the hand-off are still undesigned.

Two further gaps belong here because they change who can use setup at all:

6. **Keys live in config.toml or nowhere.** Provider credentials resolve from `stored`, `environment` or `none` (`Chat/provider_readiness.py:235`). The OS keyring already holds tldw server tokens (`SS:26584-26597`, `runtime_policy/server_credentials.py:452-467`) and MCP bindings (`MCP/credential_bindings.py:144-176`), but never a provider key. The only protection offered is a master password typed at every launch, which is the most fragile area of the review (SF6).
7. **Setup exists only as a full-screen Textual app.** Textual exposes no accessibility tree, so screen-reader users cannot use setup. There is no scripted path either: `tldw-cli` parses only recovery selectors (`tldw_chatbook/cli.py:30-56`) [coverage-10].

### 0.2 Goals

- **G1 — Fewest decisions to a reply.** Quick asks only what a first chat needs. It ends on a verdict computed by Console's own preflight, with an optional real reply.
- **G2 — Every step earns its place.** Every Full step changes something the runtime reads, and writes nothing if left untouched. **One accepted exception:** the Full track's Keys step renders "Nothing to change here" when no key is stored in config.toml. It stays on the track so that the total never changes mid-run (rule S1), and it writes nothing.
- **G3 — Local-vs-server status is always visible.** PRODUCT.md asks users to "understand what is local, server-backed…" (`PRODUCT.md:17`). Ready's tldw server row always says which, and the Provider row always says where messages go.
- **G4 — Re-running setup is safe and fast.** Change one thing in a few keys, from wherever you are, and go back there.
- **G5 — Keys are safe without daily friction.** The default store is the OS keychain where one exists. An environment key is never copied anywhere.
- **G6 — Setup is usable without the TUI** by screen-reader users and over SSH, on the same setup core (D10). Scripted setup is deferred (D11).
- **G7 — Keep what works.** Every report §5.5 "Preserve" item survives (§7).

### 0.3 Non-goals

- Fixing defects inside steps. They belong to TASK-34100.1–.16 (§0.4).
- New provider facts, model ranking, encryption mechanics, download coordination, the input policy and the 80x24 frame. This spec states requirements on them and names their owners.
- A setup "hub of cards" for first run. The report rejects it (report §4.3); the hub is the re-run dashboard only.
- Non-interactive setup and a settings export (D11, deferred).
- Mobile, web or server-side setup. `tldw-serve` is out of scope.
- Changing what Settings owns. Settings stays the owner of durable configuration (ADR-012); setup is a guided path into the same owners.

### 0.4 Scope boundary with the sibling subtasks and task-33008

| Sibling | Owns | This spec adds on top |
|---|---|---|
| .1 step extraction | Steps in their own modules, `wizard_worker()` helper, busy line | Requires F0 (§4.7) right after it: a step-host protocol, a pure setup session and pure Provider/Model state, so that the dashboard, sheets and plain mode can host steps |
| .2 first chat works | Catalog repair, **shared readiness verdict with the capacity blocker** (AC#2), real key checks, env-aware Get started card | Ready's four verdict states (D4). The verdict must have an app-free entry for plain mode (D10) and must account for the active runtime source (D3) |
| .3 Moonshot | Continuation persistence | Say hello exercises the saved-chat path that .3 fixes (D5) |
| .4 encryption lifecycle | One unlock path, refuse second enable, Settings Encryption card, state-aware Protect | Protect leaves Quick (D1); Ready's "Encrypt saved keys…" option; keychain-first (D9) |
| .5 handoff | False toast (AC#8), no unavailable tools (AC#2), minimal plain-chat prompt (AC#6), error categories (AC#4), first-send trace failure (AC#7), single arrival line (AC#11) | Arrival line content (D6). Say hello reuses .5's categories and depends on AC#4, #7 and #8 (D5) |
| .6 provider step | Catalog-driven form, "Ready on this machine", filter, key-source choice when an env key exists (AC#21) | Display name "Connect"; the key storage sentence and disclosure (D9). **AC#21 is amended on approval** so that the env choice and the storage choice are one control |
| .7 model step | Curated picker; skip keeps the key | "Save and finish with defaults" on Model (D2) |
| .8 voice step | Untouched Voice writes nothing; "No voice for now"; inline OpenAI key (AC#3) | Voice moves to Full as "Spoken replies" (D1, D2). Hosted as a Console sheet, it never takes a key (D6) |
| .9 honest status | Outcome record, tracker, legend (AC#5), Exit dialog list (AC#3), **"Ready to chat?" row** (AC#10) | Ready composition, exits and What's next (D4, D6). The legend appears on Ready and the dashboard (§3.6) |
| .10 setup session | Entry contract `open_setup_wizard(origin, resume, start_step)`, re-run prefill, "Review your setup" header (AC#1), Console links (AC#8), one finish path (AC#12), re-run exits (AC#14), E10 queue | The dashboard body and single-step Change (D7); documents-first finish (D8). **AC#1 is narrowed on approval** to the entry contract, so that .10 does not build a header that F11 replaces |
| .11 input policy | Highlight browses; Enter/Space/click selects | The dashboard, What's next and storage lists follow it |
| .12 terminal frame | Short tier at 80x24 (AC#1), Back hint and Alt+Left (AC#9), glyph checkboxes, contrast | Ready and dashboard row budgets assume it (§3.6, §3.7) |
| .13 full-track steps | Search, Tools, Notes removal, Speech engine choice, Appearance comfort and ASCII marks (AC#12), Welcome copy (AC#15), Reduce motion on Welcome (AC#16) | Full order (D2). Welcome copy is restated here and supersedes .13 AC#15's time line (D1). The Welcome mockups include .13 AC#16's Reduce motion row (§3.1) |
| .14 downloads | Download coordinator | Places download-bearing steps late (D2) |
| .15 registry | Glossary, area names, "change it later" homes, setup entry points (AC#4), User Guide | One name per area (§3.0, rule S5); exit-route table shared with F0 |
| .16 portable | Second-machine docs, `--config`, `--no-splash`, Restore Inspect | `setup --plain` shares `--config` parsing (D10). Its docs carry the keychain note (D9) and are the whole second-machine story while D11 is deferred |

**task-33008 (Phase 8, "First run connects in place and always returns to Console", high priority, To Do).** It builds the in-Console path: Console's Get started card connects a detected server in two keys, and sends cloud keys through a Settings round trip that returns to the chat. It also rules that no Console surface accepts or displays an API key (AC#4). That path and this spec's wizard are two front doors for two situations:
- the wizard is the **boot offer** on a fresh profile (`FRSS:861-869`);
- the card is the path for everyone who arrives in Console without a working provider: after Skip, after documents-first, or after setup was deferred.

Both read the same shared verdict (TASK-34100.2 AC#2). Neither takes a key inside Console: D6's sheet uses task-33008's round trip, so this spec's sheet relies on task-33008's Settings return. §16 Q10 asks the owner to confirm this split.

**One backlog conflict to resolve.** TASK-34100.10's coordination note says "Narrow task-28019 to its media-first path", while TASK-34100.17 says "Narrow it to its modal-sequencing AC (#3)". Both cannot hold.
- This spec follows TASK-34100.17 AC#3 (D8). task-28019 is narrowed to its modal-sequencing AC#3, and its media-first ACs #1–#2 move into the documents-first follow-up (F7, §10).
- TASK-34100.10's note is corrected to match.
- Because TASK-34100.10 AC#18 (E10) delivers exactly that modal sequencing, the narrowed task-28019 closes when .10 lands.

The edits are made on approval (§6), not now.

### 0.5 Personas

The report's three personas, plus two that the shape decisions serve directly.

| Persona | Situation | What the shape must give them |
|---|---|---|
| **Sam** | First-time user with an OpenAI key, possibly a new account with no credit; fuzzy on jargon | Four steps; a verified "Ready to chat"; a reply before leaving setup if they want one; an honest message when the account can't pay; a key that is safe without a password at every launch |
| **Jo** | First-time user who wants private local AI; may have nothing running | No cloud-key detours. A local test reply that runs on its own only when it costs nothing and disturbs nothing. "Your messages are answered on this computer" said plainly |
| **Riley** | Power user: env keys, local servers, several machines, re-runs to change one thing | Env keys never copied; a dashboard instead of a corridor; Finish with defaults from Model; palette jumps; `--plain`; TASK-34100.16's documented route for the next machine |
| **Dee** (new) | Came for documents and notes; may never use an AI provider | One choice on Welcome that goes straight to Library and writes nothing else, and a later "set up AI" that starts at Connect |
| **Ash** (new) | Uses a screen reader, or works over SSH without a keychain | `tldw-cli setup --plain`, discoverable from `--help` and before the TUI starts; a plain-text key fallback that is named honestly |

### 0.6 Rules

The programme's four rules (report §5) bind every decision:
1. Done means a reply, not a write.
2. Write nothing the user didn't touch, and drop nothing the user typed without saying so.
3. Every mark is computed from persisted, verified state through the runtime's own resolver.
4. One owner per fact.

This spec adds five shape rules:

- **S1 — Stable total (restates TASK-21148's guarantee).** From the moment the user leaves Welcome until they return to it, the run's step list never changes. Only the Welcome choice sets the total, and the tracker shows that total before Next (TASK-34100.9 AC#5).
- **S2 — Options are not steps.** Anything conditional (encrypt now, hear replies aloud, connect a server) is an option on Ready or a next step. It is never a step that joins or leaves the track.
- **S3 — One surface per job.** First run is a corridor, a re-run is a dashboard, and a single change is a single-step sheet. All three, and plain mode, are hosts for the same step state and the same commit path, through one step-host protocol (§4.7).
- **S4 — Every exit says where it goes.** No exit, link or hint names a home that cannot do the job (report §4.4; owned by TASK-34100.15's registry).
- **S5 — One name per area.** Every area has one label on every surface: Ready rows, dashboard rows, palette commands, What's next and plain mode (§3.0). The single exception is the provider step's tracker title, "Connect".

---

## 1. Decisions

Each decision states:
- what is decided, and why;
- what was rejected;
- which report §5.5 "Preserve" items it must not break;
- how it stays consistent with the sibling subtasks and the ADRs.

Screen-level detail is in §3, and the engineering seams each decision needs are in §4.7.

### D1 — Quick track: Welcome → Connect → Model → Ready

**Decision.**
- The Quick track is four steps: `welcome`, `provider` (titled **Connect**), `model` (**Model**) and `summary` (titled **Ready**). Step ids stay as they are, because drafts are keyed by them (`FRSS:991-1002`). Only the display titles change, through TASK-34100.15's registry (§3.0).
- **Voice leaves Quick** and moves to Full as "Spoken replies" (D2). On Quick, Ready offers it as the next step **"Hear replies aloud…"** (D6).
- **Protect leaves Quick.** Ready shows the option **"Encrypt saved keys with a password…"** only when this run stored a provider key as plain text in config.toml. With D9 shipped, that happens only when the user chose plain text, or when no keychain exists.
- `active_step_ids(TRACK_QUICK, …)` returns the four ids for every value of `key_entered`. Today `FRSS:1271-1286` returns six. An unknown track fails closed (§4.4); it no longer falls back to Full.

**This supersedes TASK-21148 AC#5.** That AC reads: "Protect appears in the quick track from the start (marked skipped when keyless); the step total never changes mid-flight." It has two halves:
- *the guarantee*: the total never changes mid-flight. UAT N-6 showed "Step 2 of 5" becoming "Step 3 of 6" when a key was typed (TASK-21148 notes);
- *the mechanism*: Protect is always present, so it can never join.

This spec keeps the guarantee and retires the mechanism. The guarantee now holds by construction: Quick has no conditional step at all, because the only conditional thing (encryption) became an option on Ready (rule S2), and options are not counted. Every keyless, local and env-key user paid for the mechanism by walking an empty step that ticks ✓ ("No API keys saved yet — nothing to protect", `FRSW:6463-6467`).

**Why.**
- **The evidence.** Jo's local path and Riley's env-key path both meet an empty Protect step that ticks ✓ (report §1 Jo Path A; §2(a) stage 11). Sam's Protect step is where the P0 lock-out happens [protect-summary-04]: the step chosen for peace of mind is the one that bites. Voice adds a detour inside a "2-minute" track and, today, a write on Next [voice-speech-01].
- **Relevance at the point of need.** Voice matters the first time a user presses Speak, not as a setup detour (E9). D6 puts it there.
- **Protection without a step.** With D9, Sam's key goes to the keychain by default. That meets the main reason Protect was on Quick, "make my key safe", at the key field and without a password.
- **Stability without padding.** Rule S1 is stronger than TASK-21148's: it also covers the Full track and every future conditional idea.

**Rejected alternatives.**
- *Keep six steps, but make Voice and Protect truly skippable* (TASK-34100.8 and .4 alone). The steps become honest but not relevant: two Nexts on every Quick run, and a Quick label that still has to name "voice, protection".
- *Five steps, keeping a conditional Protect.* This brings back the mid-flight count change that TASK-21148 fixed.
- *Fold encryption into Connect as a checkbox under the key field.* This grows the busiest form in the wizard for every user. The report's verifier preferred the Ready option ("the verifier's lower-risk alternative", report SF10). D9's storage choice is a single sentence for the same reason (§3.2).
- *Three steps (Connect folds Model in).* The provider list and the model list both need the full step height at 80x24 (TASK-34100.6 AC#15, .7 AC#4).

**Welcome copy.** This is restated here and supersedes TASK-34100.13 AC#15's time line, because Quick no longer has optional downloads.

| Element | Copy |
|---|---|
| Title | Welcome to chatbook |
| Pitch | Chat with cloud or local AI models, keep notes, and work with your own documents — all in your terminal. With a model running on this computer, your messages are answered on it. |
| Question | What would you like to do first? |
| Choice 1 (default) | Quick setup — connect AI and pick a model (recommended) |
| Choice 2 | Full setup — adds search, tools, voice, appearance, server |
| Choice 3 | Start with my documents or notes — set up AI later |
| Time line | Quick takes about 2 minutes if you have an API key or a local AI server running. You can change any of this later by running setup again. |
| Reduce motion (TASK-34100.13 AC#16) | [ ] Reduce motion — fewer animations, starting now. (Shown only where it can apply live, as .13 AC#16 says.) |
| Restore line | Moving from another computer?  [ Restore a backup ] |

- The three labels are 55, 58 and 50 characters, inside the 61-character no-wrap budget TASK-21148 set (TASK-21148 notes). Each leads with its keyword (TASK-34100.12 AC#1). Full lists "server" last, because it matters to the fewest people.
- **The pitch is scoped to what is true.** "Your messages are answered on it" is about where the model runs. It does not promise that nothing leaves the computer: web search can send queries when it is on [fulltrack-02]. F5 re-checks the sentence once TASK-34100.5 AC#6 lands, and the live check in §8.4 confirms that a local first chat makes no outbound model request.
- **There is no time claim for Full.** Revision 1 said "about 10 minutes", but nobody has measured it. The Full gloss at 120 columns says what Full adds instead.
- "You can change any of this later by running setup again" replaces "Everything can be changed later in Settings" (`FRSW:6371-6375`). Until TASK-34100.15 fills the missing Settings homes (Dictation, Encryption, Analysis model), the Settings claim is false (report §4.4). The re-run claim is true because of D7.
- **"About 2 minutes" is a measured claim, with its condition stated.** The live verification (§8.4) must reach Ready ✓ in two minutes or less, hands-on, following the recommendations: once with a pasted key and once with a running local server. If it cannot, the copy changes before release. It does not ship as an aspiration.

**Tests and docs that pin six steps.** All are rewritten on purpose in F5 (§10), never deleted:
- `Tests/Wizards/test_first_run_setup_state.py:509-528` (`TestActiveStepIds.test_quick_track`, `test_quick_track_with_key`);
- `Tests/Wizards/test_task_25818_tracker_integration.py:17` (`_TRACK`);
- `Tests/Wizards/test_first_run_setup_wizard.py:11645, 11662, 11679-11681, 11759, 11973, 12847-12872` (`len(…) == 6`, "Step 1 of 6", "Step 2 of 6");
- `Tests/UI/test_first_run_wizard_live_contract.py:86-92, 3170` (imports and uses `STEP_VOICE`/`STEP_PROTECT` on Quick);
- `Tests/integration/test_first_run_pocket_tts_flow.py:140-141` (finds Voice on Quick through `_step_index_for_id(STEP_VOICE)`);
- `Tests/integration/test_pocket_tts_character_roleplay.py:592-603` (walks Quick to Voice);
- `Docs/User_Guide/First_Run_Setup.md:51-59` (the two-tracks section) and `:61-73` (the step table).

The two PocketTTS integration tests move to the Full track, where Spoken replies still offers PocketTTS. They are not weakened.

**Preserve.**
- Restore a backup stays reachable from Welcome, with its explanatory line.
- The Get started card catches Skip and Exit: Quick's exits are unchanged in kind.
- Model-list consent is asked once: unchanged, and it stays on Ready.

### D2 — Full track: eleven steps, in dependency order

**Decision.**

| # | Step (tracker label, §3.0) | Default if untouched | Writes when untouched | Why this position | Content owner |
|---|---|---|---|---|---|
| 1 | Welcome | Quick | nothing | Entry | this spec |
| 2 | Connect | the detected or current provider | nothing new | Start of the only required pair; everything that talks to a model depends on it | .6 |
| 3 | Model | current or recommended model | nothing | Completes the pair. From here, "Save and finish with defaults" is offered | .7 |
| 4 | tldw server (optional) | **Not now — this computer only** | nothing | Decides where Library and sync live, so it comes before the data features | this spec (§3.4) |
| 5 | Search (optional) | **only when you ask** | nothing | "What the assistant can see" … | .13 |
| 6 | Tools | the saved gates | nothing | … "then what it can do" (report §4.3) | .13 |
| 7 | Spoken replies (optional) | **No voice for now** (or the saved voice) | nothing | Late, because it can download (OmniVoice, 1.1 GB) | .8 |
| 8 | Dictation (optional) | **No dictation for now** (or the saved engine) | nothing | Next to Spoken replies, so the two read as input and output. Renamed so the two stop reading as synonyms [cross-cutting-11]. Can download (Parakeet, 633 MiB) | .13, .14 |
| 9 | Appearance | the saved theme and splash | nothing | Nothing depends on it | .13 |
| 10 | Keys | **Keep as is** | nothing | After every step that can store a secret: provider key (2), server token (4), voice key (7) | .4, D9 |
| 11 | Ready | — | — | End | this spec |

- **The count stays 11** (today 11, `FRSS:969-981`). Notes leaves (TASK-34100.13 AC#7) and tldw server joins. Voice moves from position 4 to 7, Speech becomes Dictation at 8, and RAG becomes Search at 5.
- **Step 10 is titled "Keys", not "Protect keys".** With D9, the step mostly reviews where each key lives (keychain, environment, plain text, encrypted), and often has nothing to protect. A page titled "Protect keys" that reads "Nothing to change here" contradicts itself. One name serves the step, the Ready row, the dashboard row and the palette (rule S5). TASK-34100.15 AC#3's example list changes to match (§6).
- **Finish with defaults, offered from Model.** Report §4.3 and E5 offer the exit from Model, so Riley does not pay an extra step.
  - On Full, Model's nav row reads `[ ← Back ]  [ Save and finish with defaults ]  [ Save & continue → ]`. The middle button saves the model, like Save & continue, then goes straight to Ready.
  - Steps 4–10 carry `[ Finish with defaults ]`, which goes to Ready without writing the current step.
  - Every unvisited step stays untouched and nothing is written. Ready shows them as one line: "– Left at their defaults: Search, Tools, Spoken replies, Dictation, Appearance, Keys — change any of them from Review your setup."
  - TASK-34100.7's "Skip — keep the key, no default model" also counts as Model's outcome, so Finish with defaults is offered after it.
  - The buttons are not offered on Quick, where the next step is already Ready.
  - **No new key chord.** Ctrl+Enter, which E5 suggested, reaches the app as plain Enter in most terminals, so a hint teaching it would teach a key that often does nothing. Each button is a Tab stop, and the palette has "Setup: Finish with defaults".
- **Every optional step defaults to "not now" and writes nothing when left untouched.** This spec states the requirement; the step owners implement it (TASK-34100.8 AC#1, .13 AC#1/#8/#12, .4 AC#7). The guard is the byte-identical invariant test from TASK-34100.9 AC#1, extended to both Finish-with-defaults buttons (§8.1).

**Why this order.** It is the report's rationale (report §4.3), accepted with one addition: Keys is last before Ready, because it reviews every secret stored earlier in the run.

**Rejected alternatives.**
- *Spoken replies stays at position 4.* It puts a download-bearing optional step early in the path. A Full user who abandons halfway should leave with the higher-value steps done.
- *tldw server first.* Most users have no server. The chat pair is the only required dependency, so it comes first.
- *Keys right after Connect.* The server token and voice keys come later, and the step would miss them.
- *Appearance first, for reduced motion.* Reduced motion must apply before any animation plays. That means at launch or on Welcome (E7, owned by TASK-34100.13 AC#16, shown in §3.1), not at step 2.
- *A shorter Full track.* Expert speed comes from defaults and the exit ramp, not from fewer steps (report §4.3).
- *Keep the title "Protect keys"* (revision 1). Rejected for the reason above.

**Preserve.**
- Delta-aware writes: Tools, Appearance and Speech already write only what changed.
- The Parakeet install review: unchanged, and it moves with Dictation.
- Scope restraint: no sampling, system prompt, hooks or MCP exposure in setup.

### D3 — tldw server: Ready row, next step, optional step, one owner

**Decision.** Five parts.

1. **Ready always shows a tldw server row**, on both tracks and in the dashboard. This is the row TASK-34100.17 AC#1 calls "Runtime".
   - **It is labelled "tldw server" because the label says what it measures.** "Runtime" is jargon, and "✓ Runtime  this computer only" printed beside "OpenAI" told Sam that everything stays local [entry-exit-handoff-18].
   - The row says where Library, notes and sync live. The Provider row says where messages go (D4).
   - Its value comes only from `RuntimePolicyContext`, ADR-033's sole authority for the active source (`runtime_policy/types.py:62-70`: `active_source`, `active_server_id`, `server_configured`, `server_reachability`, `last_known_server_label`):

   | State | Row |
   |---|---|
   | local, no server configured | `✓ tldw server  none — Library and notes stay on this computer` |
   | server active, reachable | `✓ tldw server  lab.example.org:8000 — connected` |
   | server active, not reachable at the last check | `! tldw server  lab.example.org:8000 — not reachable now` |
   | server configured but not active | `✓ tldw server  none in use — lab.example.org:8000 is set up but off` |

   - **Local-only is a working, verified state**, so it gets ✓, not the "skipped" dash.
   - **The `[tldw_api]` template values never count as a server.** The template's `base_url = "http://127.0.0.1:8000"` (`tldw_chatbook/config.py:4064`) is not a binding. The template token `default-secret-key-for-single-user` (`config.py:1510`, `:4066`) is screened by the existing `resolve_tldw_api_auth_token` (`config.py:1558-1577`, task-31417), which is reused, not copied.
   - **When chat is routed through the server,** the Provider row says so ("via tldw server lab.example.org:8000"), and the verdict must account for the server's reachability. That is a requirement on TASK-34100.2's shared verdict, not wizard logic: an unreachable server that carries chat makes the verdict ✗.

2. **Ready offers "Connect a tldw server… (if you run one)"** in What's next (D6) whenever no server is in use. That means always on Quick, and on Full when the step was left on "Not now". It opens `ServerSwitchModal` (`Widgets/Settings_Widgets/server_switch_modal.py:31`) and commits its result through the shared coordinator (part 4). On success, the row refreshes in place.

3. **An optional tldw server step, on Full only** (§3.4). It offers "Not now — this computer only" (the default, which writes nothing) or "Connect to a tldw server" (address, token, Test connection). The commit runs on **Save & continue**, never while typing. **What sync sends is said before the choice** (part 4), not only after a successful test.

4. **One commit path and one probe, shared with Settings.**
   - **The commit today.** The whole switch lives in a Settings method (`SS:26513-26623`). It:
     - saves the `[tldw_api]` URL and token to config.toml in plain text (`SS:26529-26537`);
     - then rebinds through `app.handle_runtime_backend_changed` (`app.py:1952-1975`);
     - stores the token in the OS keyring, with a config.toml fallback (`SS:26584-26597`);
     - prepares a Sync v2 profile (`SS:26599-26620`).
   - **The commit after this spec.** That body moves to one app-level coordinator (F3). Settings, the tldw server step and Ready's next step all call it; plain mode does not, as below. ADR-033 already names "one app-level coordinator", and this extends its callers (§12.5).
   - **Rollback.** Today a failed bind leaves the new URL and token on disk, because the config write comes first. ADR-033 promises only that in-memory observers stay on the old binding (`backlog/decisions/033-application-session-state-ownership.md:57-63`). The coordinator therefore records the prior `[tldw_api]` values before writing. If the bind fails, it restores them and deletes any keyring token it stored in this attempt. Then "nothing is half-applied" is true on disk as well as in memory.
   - **One client.** The shared probe uses the same HTTP client settings as the bound runtime client (TLS verification, CA bundle, proxy, egress policy). A "certificate not trusted" result therefore always agrees with what the bound client would do.
   - **The Exit dialog lists it.** A bind made during a walkthrough appears in TASK-34100.9 AC#3's Exit dialog ("Saved so far: … tldw server (lab.example.org:8000)"), because it changes the runtime.
   - **Sync disclosure.** Settings' activation prepares a Sync v2 profile with `profile_mode="local_first_sync"` and `display_name=platform.node()` (`SS:26599-26620`). That runs `run_v2_dry_run` against the server (`Sync_Interop/sync_scope_service.py:349-410`), so at least the computer's name reaches the server. The step says so **before** the choice. F3's first task records exactly what preparation sends and stores; the copy names it, and owner question Q5 decides whether setup may do it at all.
   - **The probe.** `ServerSwitchModal._run_connection_test` (`server_switch_modal.py:211-256`) GETs `/docs` for reachability, then POSTs `/api/v1/sync/send` with the token. It reports "Reachable (HTTP n)" for any server that answers, so it cannot say "not a tldw server". The probe moves to one shared, app-free owner with these outcomes:

     | Outcome | Copy |
     |---|---|
     | Not reachable | ✗ Can't reach lab.example.org:8000 — nothing answered. Check the address, and that the server is running. |
     | Token rejected (401/403) | ✗ lab.example.org:8000 is a tldw server, but it rejected this token (HTTP 401). Check the API token in your tldw server's settings. |
     | Not a tldw server | ✗ lab.example.org:8000 answered, but it isn't a tldw server. Check the address and port. |
     | Certificate not trusted | ✗ lab.example.org:8000's certificate isn't trusted. Settings ▸ Network can add your organisation's certificate. |
     | Blocked by policy | ✗ chatbook's network policy blocks this address. (This is the existing egress check, `server_switch_modal.py:212-224`.) |
     | Success | ✓ Connected — lab.example.org:8000 is a tldw server and accepted the token. |
     | Success, no token | ! lab.example.org:8000 is a tldw server. No token was entered, so sign-in wasn't checked. |

     - "Not a tldw server" is decided from the identity the server's own health and docs-info endpoints report. These are the same discovery calls `ActiveServerCapabilityService` makes after binding (`runtime_policy/server_capabilities.py:66-75`), run against the candidate URL before commit.
     - The auth check must not change server state. Today's POST to a sync endpoint with an empty body (`server_switch_modal.py:229-252`) is kept only if the follow-up confirms that tldw_server offers no authenticated read endpoint (risk K9).
   - **Storage.** The token goes to the OS keyring through the existing server credential store, as Settings does today. Once D9 ships, a secure keyring means the token is **not** also written to config.toml. Until then, setup matches Settings.

5. **Settings' button leaves the collapsed section.** "Switch Source / Server" moves out of "Advanced / Diagnostics" (`SS:17648-17681`) into the main body of Settings ▸ Overview, as a tldw server row. The row reads "tldw server: none — Library and notes stay on this computer  [ Connect a tldw server… ]", or "tldw server: lab.example.org:8000 — connected  [ Switch source… ]". The button moves rather than being duplicated, so it has one home, and it keeps its id `settings-switch-runtime-source`.

**No Welcome router question.** Confirmed rejection (D12.1).

**Why.** PRODUCT.md promises visible source authority. A newcomer with a server should be offered the connection where setup happens. A row that is always present costs one line and answers "is anything server-backed?", a question nothing answers today.

**Rejected alternatives.**
- *Server on Quick.* It taxes the local-first majority, and the Ready next step serves the minority in one action.
- *A provider-list row for tldw server.* A server is a runtime source, not a chat provider; mixing the two would blur ADR-033's ownership.
- *A wizard-local copy of the switch logic.* Two writers for the runtime binding is exactly the drift ADR-033 forbids.
- *Show the row only when a server exists.* Then local-only is never said, and G3 fails.
- *Label the row "Library & sync"* (the HCI critic's suggestion). It is accurate, but it names the effect rather than the area. Then the step, the next step, the dashboard row and the palette would need two names for one thing, against rule S5. "tldw server" with the value "none — Library and notes stay on this computer" says both.
- *Label the row "Runtime"* (revision 1). Rejected for the reason above.

**Preserve.**
- Secrets never reach disk unless saved: the token goes to the keyring, a failed bind restores the old values, and the template placeholder is never shown as a token.
- Connection errors stay specific, and now include "not a tldw server".

### D4 — The Ready screen

**Decision.** Ready replaces today's Summary. From top to bottom (§3.6 has the copy and mockups):

1. **Verdict region: one mark, never two.**
   - The verdict line is computed through the shared readiness verdict that TASK-34100.2 AC#2 creates. Its ingredients today are `Chat/console_prepared_request.py:1084` `resolve_request_capacity`, `Chat/provider_readiness.py:730` `get_provider_readiness` and `Chat/console_session_settings.py:1742` `build_console_settings_readiness`.
   - The verdict is then combined with this run's test outcome (D5). The wizard calls the verdict; it never copies it (TASK-34100.9 AC#10).

   | State | Verdict line | When |
   |---|---|---|
   | Ready | `✓ Ready to chat — <provider> · <model> (<context>)` | The shared verdict passes, **and** the key was proven: by a check that proves it (an authenticated endpoint, or a model list that needs the key), by a keyless local engine that answered, or by a test that replied in this run |
   | Set up, key not checked | `✓ Set up — <provider> · <model> (<context>)`, then on the next line "The key isn't checked yet: <provider> lists models for any key. Say hello checks it." | The shared verdict passes, but the key can't be proven offline. This is the [gap-01] class (OpenRouter, Hugging Face, NVIDIA NIM, Novita) until TASK-34100.2/.6 add authenticated probes (report SF1 step 3). Nothing has been tested yet |
   | Test didn't finish | `! Set up, but the test didn't finish — <cause>` | The shared verdict passes, and the test failed for a reason that doesn't prove the setup is broken (rate limited, too slow, network or server error) |
   | Can't chat | `✗ Can't chat yet — <plain cause>` | The shared verdict is blocked, **or** the test failed in a way that proves the saved setup can't work: key rejected, model not found, no credit, local server not answering, or model not downloaded |
   | Not set up | `– Chat not set up — no AI provider connected` | No usable chat provider (§4.2's predicate) |

   - **The test result has its own line, without a glyph.** The verdict carries the only mark, so the two never disagree [cross-cutting-07]. Examples: "Test reply in 0.8 s — gpt-4.1-mini: "Hello there, nice to meet you!"", or "The test message failed: OpenRouter said "API key expired." (HTTP 401)."
   - **Fix actions.** ✗ and ! carry **one fix action**. The exception is the two that TASK-34100.9 AC#10 already names: "Choose another model" and "Set context size…".
   - **Not set up** carries [ Connect a provider ] (back to Connect).
   - **Shared readiness is changed only by Console's own rules** (TASK-34100.2 AC#2: a refused send, 401 or 404). The verdict region combines that with the test outcome for display. It never writes readiness itself, and a failure caused by a test-only parameter never changes it (D5).

2. **Read-back rows, only for steps the user saw.** Labels follow §3.0.
   - **On Quick:** Provider (where messages go, and where the key is kept), Model, the always-present tldw server row (D3) and, when more than one provider has a credential, Keys.
   - **On Full:** rows in step order. Steps skipped through Finish with defaults collapse into one "– Left at their defaults: …" line.
   - Rows are built from TASK-34100.9's outcome record and read back from disk. Today's force-reload read-back (`FRSW:6683-6692`) is kept.
   - **The legend** "✓ saved · – skipped · ! needs attention · ✗ failed" shows under the rows whenever –, ! or ✗ is on screen, including in the tracker (TASK-34100.9 AC#5).

3. **"Data:" and "Config:" lines**, each with [ Copy ].
   - Data is `user_data_dir(config)` (`Backup_Recovery/profile_paths.py:86-90`, default `~/.local/share/tldw_cli/default_user`).
   - Config is the effective config path (`profile_paths.py:25-29`, honouring `TLDW_CONFIG_PATH`).
   - Paths are middle-truncated with the existing helper `middle_truncate_path` (`FRSS:609-639`).

4. **What's next** (D6): one focusable list. At the short tier on Full, it collapses into one row that expands in place (§3.6).

5. **Checkboxes:**
   - the model-list consent box, shown only when a cloud provider covered by refresh is configured, and naming it (TASK-34100.9 AC#8; offered once, `FRSW:6608-6620`, `:6755-6777`);
   - "Get to know you after setup" (TASK-34100.10 AC#13).

6. **Exits, on one nav row with [ ← Back ]** (TASK-34100.12 AC#1 gives the nav one row):

   | Situation | Exits after [ ← Back ] (the first is primary) |
   |---|---|
   | First run; verdict ✓ Ready, ✓ Set up or ! | [ Start chatting ]  [ Add a document ]  [ Write a note ] |
   | First run; verdict ✗ or not set up | [ Go to Console ]  [ Add a document ]  [ Write a note ]. Focus starts on the verdict's fix action |
   | Re-run walkthrough (TASK-34100.10 AC#14) | [ Done ]  [ Go to Console ]  [ Add a document ]  [ Write a note ]. "Go to Console" is hidden when the origin is Console, because Done already goes there |

   - **Short labels below 100 columns.** The Library exits read "Add a document" and "Write a note". From 100 columns they read "Add your first document" and "Write your first note" on first run. Both labels lead with their keyword (TASK-34100.12 AC#1). At 80 columns the full labels plus [ ← Back ] would need 86 columns.
   - **One name for the notes exit.** "Write a note" replaces TASK-34100.10 AC#14's "New note", so Welcome's documents-first copy, D8's Import canvas action, the Get started card ("Write a note in Library") and the exits all say the same thing (§6).
   - "Explore Home" and "Open Settings" (today's "Review settings", renamed by TASK-34100.10 AC#12) move into What's next ▸ More.
   - The two Library exits stay docked on every Ready, on first run and re-run alike. task-32072 and task-32140 made them visible on purpose, TASK-34100.12 forbids hiding them, and TASK-34100.10 AC#14 keeps them on re-run.

**What "at most three exits" means here.** First run docks at most three exits. A re-run walkthrough docks at most four, as TASK-34100.10 AC#14 specifies, and they still fit one row at 80 columns (73 columns with Back). The verdict region carries at most one in-place action (Say hello, or the cause's fix), except the two-action case above. Link-outs live in What's next, which is one Tab stop however long it is. "More (N)" is the list's last row; no "More ▾" button is added.

**Focus, Tab order and keys.**
- **Verdict ✓ Ready:** focus starts on **Start chatting**. **Say hello is next in Tab order**, and the hint reads "Tab: test message first". The hint is generated for the focused control (TASK-34100.11).
- **Verdict ✓ Set up (key not checked):** focus starts on **Say hello**, so the test drives the primary action (report E1). Its cost line sits directly beneath it. For 0.5 s after Ready renders, the button ignores activation, so an Enter carried over from Model cannot send a paid message.
- **After a reply:** focus returns to Start chatting. **After a failure:** focus moves to the fix action.
- **Esc on Ready never finishes setup.** While a test runs, Esc stops it (D5). When the What's next list is expanded, Esc collapses it. Otherwise Esc is inert, which TASK-34100.10 AC#12 allows ("Esc on the Summary is inert, or finishes to Home"), and the hint does not advertise it. Finishing needs an exit button. A reflexive double Esc, or an Esc pressed just as a reply arrives, therefore can never end setup.
- **Back:** [ ← Back ] is a Tab stop. The hint teaches "← Back, Ctrl+B or Alt+←" (TASK-34100.12 AC#9).

**Row budget at 80x24.** TASK-34100.12's short tier puts the title and tracker on one row, the nav on one row and the hint on one row, which leaves 21 rows. TASK-34100.12 AC#1 asks each step to use about 19.
- **Quick:** verdict region 3–4, rows 3–4, legend 0–1, Data and Config 2, What's next 5, checkboxes 1–2. That is 19 at most.
- **Full:** verdict region 3–4, rows 9 (✗ and ! rows first, so that at least 8 read-back rows show, per TASK-34100.12 AC#1), legend 1, Data and Config 2, What's next collapsed to 1, checkboxes 2. That is 19 at most.

Both are measured in the §3.6 mockups.

**Why.** The Summary is the peak of the emotional journey, and today it predicts an outcome it never checks (report §3.3 "Peak and end"). An honest verdict, a test reply on request and three clear exits make the peak honest, and keep the end where the peak points.

**Rejected alternatives.**
- *Five exits, as today* (`FRSW:6640-6658`). Two docked rows at 80x24 take the read-back's space [a11y-01], and the end of setup is the wrong moment for choice overload.
- *A "More ▾" button for the Library exits.* Rejected by the verifiers (report §5.4 table) and by TASK-34100.12.
- *Block "Start chatting" when the verdict is ✗.* That would block the "set it up now, start the server later" path the report preserves (report §5.5). On ✗, Start chatting is replaced by [ Go to Console ], and Console's Get started card explains what is missing.
- *Three verdict states* (revision 1). An OpenRouter key that had expired read "✓ Ready to chat" until the user chose to test it [gap-01], and a failed test sat under a ✓ verdict. Both are fixed by the fourth state and the single mark.
- *Collapse ✓ rows into one line ("✓ 6 more areas set").* TASK-34100.12 AC#1 asks for at least 8 visible read-back rows. Collapsing What's next to one row buys the space instead, and costs less information.
- *Esc finishes setup* (revision 1). A key that also stops a test is too easy to press twice.
- *Move the context size out of the verdict* (the HCI critic's low-severity note). TASK-34100.17 AC#6 specifies "(<context>)". When the verdict is ✗ for capacity, the number is the cause. It stays, as one parenthetical.

**Preserve.**
- The Summary reads back from disk.
- Model-list consent is asked once.
- The Library exits stay visible.
- "Review provider setup" still recovers inside the wizard with the staged key intact; it becomes the verdict's fix action, landing on the step that needs it.

### D5 — "Say hello": a real first reply inside setup (E1)

**Decision.**

- **When it can run.** Only when the verdict is ✓ Ready to chat, ✓ Set up or ! (test didn't finish). On ✗ or not set up, a test the offline preflight already knows will be refused would spend tokens for nothing.
- **What is sent.** A one-line, editable prompt, prefilled with "Say hi in five words.". Enter in the field, or the [ Say hello ] button, sends it.
- **It is a probe turn.** The test is a real Console turn: Console's own admission, preflight, trace capture, persistence and dispatch, with the saved provider and model. It carries a **probe-turn profile**:
  - no tools or agent surface;
  - no retrieval or automatic Library search;
  - no project, workspace or persona instructions;
  - a minimal system prompt;
  - **no prior turns**;
  - a model-aware reply cap (below).

  The profile is **turn-scoped and never persisted** to the session, the conversation or config. So nothing about the test leaks into the user's real chats.
  - **Why a profile is needed.** Without one, the test would inherit session defaults. On Full, Tools (step 6) and "Search: automatically" (TASK-34100.13 AC#1) apply to new conversations. A test could then raise an approval round in a viewless runtime, where view hooks fail closed. Or it could retrieve Library chunks and send them to a cloud provider; on a re-run the Library may be full.
  - **What the test does not prove.** The real first chat's prompt size is not exercised. The offline verdict covers it, because capacity is resolved on the real default prepared request, and TASK-34100.5 AC#6 owns that prompt.
- **Reply cap: model-aware**, taken from the model catalog.
  - **Ordinary models:** 256 output tokens.
  - **Reasoning models:** the test asks for the lowest reasoning effort the provider accepts. Where thinking can be turned off for one request, it is turned off.
  - **Models that need a thinking budget** (for example Anthropic extended thinking, where `max_tokens` must exceed `budget_tokens`, minimum 1,024): the cap is the provider's minimum budget plus 256.
  - **Empty visible text at the length limit** counts as replied: "The model answered, but used its whole test allowance thinking, so there is no text to show."
  - The cost line is computed from the cap actually used.
- **Failures caused by test-only parameters never change shared readiness.** An HTTP 400 that names a parameter only the probe set (the cap, reasoning effort, thinking) is reported as "chatbook's test message didn't suit this model (HTTP 400). This says nothing about your setup." The verdict and readiness are left as they were.
- **Local engines: auto-run only when it costs nothing and disturbs nothing.** The test starts by itself the first time Ready renders in a run only when **every** condition holds:
  1. the session mode is `first_run`, never a re-run, review, walkthrough or single step;
  2. the provider is a known local engine. The allowlist is the provider keys in `LLM_Calls/pricing_catalog.py`'s `LOCAL_PROVIDERS` (llama.cpp, Ollama, vLLM, MLX, koboldcpp, oobabooga, TabbyAPI, Aphrodite and the `local_*` keys). `custom`, `custom_2`, `custom-ep:*` and every other generic OpenAI-compatible endpoint are excluded, because a loopback address can be a LiteLLM-style proxy to a paid API or an SSH tunnel to a remote GPU;
  3. the endpoint host is a loopback address (`127.0.0.1`, `localhost`, `::1`);
  4. no API key is configured for that provider;
  5. the test will not load a model. Either the engine serves exactly the model it was started with (llama.cpp's server, vLLM, MLX), or its loaded-models endpoint lists the chosen model as already loaded (Ollama's `/api/ps`).

  - **Otherwise the local test waits for a press,** with a cost line that names what it does: "Loads llama3.1:8b into memory (about 4.7 GB). Another loaded model may be unloaded." The size appears when the engine reports it.
  - **Once per run.** It does not re-run on Back and Next unless the provider or model changed.
  - **While it runs,** Ready shows elapsed seconds, "(the first reply can be slow while the model loads)", and [ Skip test ] (Esc).
- **Cloud and every other provider: only on an explicit press.** Ready shows the button and one consent line, computed from the actual prepared probe request. It leads with money where the price is known:

  | Estimate | Consent line |
  |---|---|
  | Price known, under one US cent | Sends one short message to OpenAI. Costs under $0.01 (a few tokens). |
  | Price known, one cent or more | Sends one message to OpenAI. Costs about $0.03 (about 4,800 tokens in, at most 1,280 out). |
  | Price unknown, small estimate | Sends one short message to Anthropic: a few tokens, at your usual rate. |
  | Price unknown, large estimate | Sends one message to Anthropic: about 4,800 tokens in, at most 1,280 out, at your usual rate. |
  | Generic endpoint on loopback | Sends one message to the server at 127.0.0.1:4000. If it forwards to a paid service, that service may bill you. |

  - **"A few tokens" is computed, not fixed.** The words appear only when the estimate is at most 500 tokens in, the cap is at most 1,280 tokens out, and the price is unknown or under one cent.
  - **Where prices come from.** Prices come from `LLM_Calls/pricing_catalog.py` through `Chat/console_cost_tracker.py`. The token estimate comes from a **prepare-only estimate** of the probe request (a new seam, F9a). A test asserts that the input tokens actually dispatched never exceed the estimate shown.
  - **The press is the consent.** Pressing the button consents to that one message. ADR-012's 2026-09-26 amendment requires the paid test to be "a separate, consented action". There is no extra dialog, nothing is remembered, and the test never runs automatically.
  - **The line goes with the button.** It appears wherever the button does: on Ready and on the dashboard (D7). The palette's "Setup: Say hello" opens the dashboard with Say hello focused and the cost line on screen; it never sends directly.
- **Path.**
  - The test is submitted through the app-scoped Console runtime (`app.console_runtime`, `app.py:1167`), which can run with no Console view mounted (`Chat/console_runtime.py:4128-4135`, "a runtime can be VIEWLESS FROM BIRTH"). It goes through the controller's submit path (`Chat/console_chat_controller.py:9797`) with a **new origin, `SETUP_PROBE`**. The precedent is `AGENT_WAKE` in `ConsoleSubmissionOrigin` (`Chat/console_chat_models.py:72-84`), a machine origin with its own rules.
  - `SETUP_PROBE` is excluded from the hidden-turn notice and nav badge (`console_runtime.py:1472-1476`), from prompt history and from composer clearing.
  - Streaming into Ready needs a **viewless streaming observer**. Today only `subscribe_message_completed` exists (`console_chat_store.py:2113`). If the F9a spike shows the observer cannot be added safely, Ready shows the reply on completion, with elapsed seconds while it waits. That is the non-streaming fallback CLAUDE.md requires.
  - All new seams go in new modules: `console_chat_controller.py` and `console_chat_store.py` are already over their size budgets on dev (§8.2).
- **Persistence.**
  - The turn goes to a **saved** conversation, because Moonshot's failure [gap-02] appears only in saved chats.
  - It is titled "Setup test · <provider> · <model>". A later test against the same provider and model reuses that conversation. The probe sends no history, so reuse adds no cost and no context.
  - The auto-run local test also writes this conversation without a press. That is data, not config, so rule 2 is not broken, but §16 Q2 asks the owner to confirm it.
- **Display.**
  - The reply streams into the test line with the model id and latency, truncated to two rows. The full text is in the conversation.
  - **Replies and provider error text are rendered as plain text.** Rich markup is escaped, and terminal control sequences (ANSI CSI, OSC and other C0/C1 controls) are stripped before display. This holds in the TUI and in plain mode (CLAUDE.md: sanitize content). A reply cannot restyle the screen or reach the terminal raw.
- **Start chatting opens a new chat.** It goes through the normal first-chat handoff (`FRSW:9327`, PendingHandoffStore under ADR-033), into a new conversation.
  - **The test exchange is not carried into that conversation's context.** As a first turn, "Say hi in five words." would sit in the context of Sam's first real conversation and can make later replies terse. The conversation's title would also read "Setup test".
  - **The proof still reaches Console:** the arrival line quotes the test reply and names the conversation in History (D6).
  - **This narrows TASK-34100.17 AC#7's "carried into Console as the first turn".** §16 Q2 asks the owner to confirm it. The alternative, keeping the turn but excluding it from context, needs a per-message context-exclusion seam that Console does not have.
- **Exits while a test is running.** Every exit, including Start chatting, stops the test through Console's own stop, without any notice. The conversation keeps the stopped turn, as Console keeps any stopped turn. Attaching a view to an in-flight viewless turn is avoided on purpose.
- **Skip test.** While a test runs, [ Skip test ] (Esc) stops it through Console's own stop. Offline users never need it on cloud providers, because nothing runs unless they press.
- **On failure.** The verdict follows D4's states, and the test line gives the detail, using TASK-34100.5 AC#4's categories plus four more:

  | Category | Verdict line | Test line | Fix action |
  |---|---|---|---|
  | Key rejected (401/403) | ✗ Can't chat yet — OpenRouter rejected the key. | The test message failed: OpenRouter said "API key expired." (HTTP 401). Nothing else was sent. | [ Fix key ] (Connect, key field focused) |
  | No credit or spending limit (402; 429 `insufficient_quota`) | ✗ Can't chat yet — OpenAI accepted the key but refused the message. | The account has no credit or hit its spending limit (HTTP 429). Add billing in your OpenAI account, then try again. (The billing page is named when the catalog knows it.) | [ Try again ] |
  | Rate limited (429, other) | ! Set up, but the test didn't finish — OpenAI is limiting requests. | OpenAI asked chatbook to slow down (HTTP 429). Wait a minute, then try again. | [ Try again ] |
  | Model not found (404) | ✗ Can't chat yet — Google doesn't have gemini-1.5-pro any more. | (HTTP 404) | [ Choose another model ] |
  | Local server not answering | ✗ Can't chat yet — nothing is answering at 127.0.0.1:9099. | Start llama.cpp, then check again. | [ Check again ] |
  | Local model not downloaded | ✗ Can't chat yet — Ollama is running but doesn't have llama3.1:8b yet. | To download it, run: `ollama pull llama3.1:8b`  [ Copy command ] | [ Check again ] |
  | First token too slow | ! Set up, but no reply yet. | No reply after 300 s. Large local models can take minutes to load. | [ Wait longer ] |
  | Test didn't suit the model (400 naming a probe-only parameter) | (unchanged) | chatbook's test message didn't suit this model (HTTP 400). This says nothing about your setup. | — |
  | Reply arrived but couldn't be saved | ! The reply arrived, but chatbook couldn't save it. | This is a chatbook problem, not your setup. | [ Open Logs ] |
  | Anything else (5xx, network) | ! Set up, but the test didn't finish. | The test message failed: <provider's own message, capped at 200 characters and scrubbed, per .5 AC#4, rendered as plain text>. | [ Try again ] |

**Why.** A reply is the only proof that key, model, streaming and persistence work together. It catches runtime defects no offline check can [gap-02]. Sam's first reply came at 621 s, after 20 recovery actions in Console (report §1). A failed test inside setup points at the fix while the user is still in setup.

**Rejected alternatives.**
- *Auto-run for cloud providers.* It spends money without an action, which contradicts ADR-012's explicit-only rule.
- *Auto-run on any loopback endpoint* (revision 1). A loopback proxy can bill and can send text off the machine, and an Ollama load can evict the user's loaded model.
- *Auto-run on every Ready, re-runs included* (revision 1). Each re-run would send a message and write a conversation nobody asked for.
- *A consent modal.* The cost line plus an explicit press is the consent; a modal adds a dialog to the busiest end of setup.
- *A Temporary chat.* Temporary chats do not exercise persistence [gap-02].
- *A wizard-built request.* A copy of the request path would drift from Console. The probe profile is a turn-scoped override inside Console's own path, not a second path.
- *Inherit the session's defaults for the test.* This risks approval rounds in a viewless runtime and Library text sent to a cloud provider without the user asking.
- *A fixed 256-token cap* (revision 1). It is unsafe for reasoning models: they can spend the whole cap thinking, and a thinking-budget API rejects a cap below its budget. The test itself would then cause the ✗.
- *A fixed "Uses a few tokens" line with no numbers.* The cost depends on the real prepared request, and a model priced at $600 per million output tokens makes "a few" misleading.
- *Carry the test turn into the conversation Start chatting opens* (revision 1). It primes terse replies and mistitles the user's first conversation (§16 Q2).

**Dependencies.**
- TASK-34100.2: the shared verdict, AC#2.
- TASK-34100.3: Moonshot persistence.
- TASK-34100.5 AC#4: the error categories. The probe adds the four extra categories above.
- TASK-34100.5 AC#7: no `trace_revision_unavailable` on the first send. The probe is often the first send.
- TASK-34100.5 AC#8: the hidden-turn toast fires only when Console is not on screen. `SETUP_PROBE` is excluded either way.

TASK-34100.5 AC#6 (the minimal plain-chat prompt) is no longer a dependency for the cost line, because the probe profile carries its own minimal prompt.

**Preserve.**
- Secrets never reach disk. The test carries no key in any persisted field or log (ADR-029).
- The typed-model "start the server later" path is never blocked: the test is optional, and it never gates Ready's exits.
- Streaming works end to end, with a non-streaming fallback.

### D6 — What's next, the arrival line, and just-in-time setup sheets (E9)

**Decision.**

1. **What's next on Ready.** One list, computed from config each time it renders; nothing about it is persisted. An item shows only when its area is not set up. Up to three show, then "More (N)". Items are marked `→` (ASCII `->`). The empty radio `○` belongs to Welcome's choices, and an action list must not look like one.

   | Order | Item | Shown when | Opens |
   |---|---|---|---|
   | 1 | Encrypt saved keys with a password… | this run stored a provider key as plain text (D1) | TASK-34100.4's password dialog, in place |
   | 2 | Hear replies aloud… | no voice is configured | the single-step **Spoken replies** sheet (D7) |
   | 3 | Connect a tldw server… (if you run one) | no server is in use (D3) | `ServerSwitchModal` and the shared coordinator |
   | 4 | Add a project folder… | no Workspace folder exists **and** a file tool gate is on (Read file or Search in files), since a project folder does nothing without them [fulltrack-06] | Settings ▸ Workspaces |
   | 5 | Sync a notes folder… | no notes folder is synced | Library ▸ Notes ▸ Add from files… |
   | More | Add another provider · Explore Home · Open Settings · Web search keys · Tool permissions · Startup animation and splash | always | Connect in "add" mode (single step, then back; D7) · Home · Settings · Settings ▸ Web Search · MCP ▸ Tools · Settings ▸ Splash Screen |

   - **AC#10's "Add a document and ask about it"** is the docked "Add a document" exit on every Ready (D4). It is not repeated in the list.
   - Each destination, and its exit route, comes from the shared route registry (F0, §4.7) and TASK-34100.15's names. An item whose home does not exist yet is hidden (rule S4).
   - **"More (N)" expands in place.** The list grows over the rows below it and scrolls, with the existing fold cue. Esc or ← collapses it. At the short tier on Full, the whole list starts as one row, "▸ What's next (N) — <first two items> · more", which expands the same way.

2. **Console arrival line.** TASK-34100.5 AC#11 replaces arrival toasts with one transcript line; this spec fixes its words:
   - **Cloud:** `Setup complete — OpenAI · gpt-4.1-mini · streaming on · 1M context. Switch models with Alt+M, or Ctrl+P then "model".`
   - **Local:** `llama.cpp on this computer · llama-3.1-8b · streaming on · 8K context`, with the same second sentence.
   - **After a test:** a second sentence, `Your test reply is in History: "Setup test · OpenAI · gpt-4.1-mini".`
   - **Ctrl+P is taught next to Alt+M.** macOS Terminal sends "µ" for Option+M unless "Use Option as Meta key" is on. That is the same reason D2 rejects teaching Ctrl+Enter.
   - **Every value comes from the resolved request settings,** not from the wizard's memory.
   - **The line is never sent to the model.** It is a transcript system row, not a turn.
   - **When the verdict was ✗,** there is no line, and the Get started card speaks instead.

3. **Just-in-time setup sheets.** Pressing Speak with no voice configured, or Dictate with no dictation engine, opens the single-step sheet for that area over Console (D7, §3.8).
   - **Save** writes through the step's own commit, then does what the user asked: reads the reply aloud, or starts dictation.
   - **Cancel** writes nothing.
   - **The sheet never accepts an API key over Console.** The single-step host tells the step what it may do through host capabilities (§4.7): over Console, `allows_secret_entry` is false. TASK-34100.8 AC#3's inline key field is therefore replaced by the line "Needs an OpenAI key — add it in Settings ▸ Providers & Models…", which uses task-33008's round trip with return. This keeps the owner ruling that no Console surface accepts or displays an API key (task-33008 AC#4). The step's other copy (for example "Setup will pick up at Voice next time") is also host-supplied (`resume_copy`), so a sheet never promises a resume it can't do.
   - **Today's behaviour is not re-checked here.** This spec does not re-check what Speak and Dictate do today with nothing configured. F10's first task records it, so the sheet replaces a known behaviour.

**Why.** Voice and dictation matter the first time a user presses them, not as a detour in a 2-minute track (E9). One list beats three toasts that cover the nav bar [new-entry-exit-handoff-02].

**Rejected alternatives.**
- *A persistent checklist that ticks items off across sessions.* It is state that can go stale; a list computed from config each time cannot.
- *Inline key entry in the Console sheet.* It contradicts the owner ruling (task-33008 AC#4).
- *"○" as the item marker* (revision 1). It reads as an unselected radio.

**Preserve.** The Voice step's strengths carry into the sheet unchanged, because it is the same step state: outcome first, plumbing under Advanced, and a real verified sample with an existing key.

### D7 — Re-run as "Review your setup", with single-step changes (E5)

**Decision.**

- **One predicate decides which surface opens.** `has_usable_chat_provider(config)` is true when at least one chat provider has either:
  - a credential that resolves through the runtime's own resolver (keychain, stored or environment), and is not a placeholder; or
  - for a keyless provider, an endpoint that is not a template value.

  Template values never count (report SF4): for example, the template's OpenAI section with `gpt-5.6-terra` and no key. The same predicate drives D7, D8, D10 and the Settings button label. It lives in the setup core (§4.7), not in a screen.

  | When setup is opened and… | First screen |
  |---|---|
  | a valid draft exists (setup never completed) | TASK-34100.10's resume dialog |
  | setup was finished through documents-first (`ai_setup_deferred`), and no usable provider exists | **Quick at Connect**, ending on Ready (D8) |
  | a usable chat provider exists | **Review your setup** (§3.7) |
  | setup was completed (Skip, or an earlier run), but no usable provider exists now | Welcome, prefilled (TASK-34100.10), run as a first run: first-run exits, and the local auto-test allowed |
  | a fresh profile | Welcome |

- **The dashboard** is one screen (§3.7):
  - **The verdict region,** the same component as Ready's, with [ Say hello ] and its cost line (D5).
  - **The area rows** in Full-track order: Provider, "+ Add another provider…", Model, tldw server, Search, Tools, Spoken replies, Dictation, Appearance, Keys.
  - **A detail line** for the highlighted row.
  - **The legend**, when –, ! or ✗ is on screen.
  - **Data and Config.**
  - **Exits:** [ Done ] and [ Run the full walkthrough ].

  Each row shows its current value, read from the real config on every render (force reload, the same reader as Ready), with the outcome glyph from TASK-34100.9. The rows and Ready's rows come from **one row builder** (today `build_summary_rows`, `FRSS:1838`) and use one set of labels (§3.0), so the two cannot disagree.
- **Interaction.** The rows are one list; highlight browses, per TASK-34100.11. Enter on a row is **Change**. The detail line explains the highlighted row: its "!" reason, or what Change would do.
- **Change opens that single step and returns.**
  - It runs in the single-step host (§4.7): the step's own state and its own commit, with a [ Cancel ] / [ Save ] pair instead of Back/Next.
  - Changing Provider continues into Model only when the provider changed, which is the existing provider→model dependency.
  - **Save returns to the dashboard with the verdict recomputed,** the row refreshed, the highlight on it, and an in-place receipt: "Saved: Model → gpt-4.1-mini". After a Provider or Model change, the verdict region shows Say hello with its cost line, so the user can test the change at once.
  - Cancel writes nothing, and the receipt reads "Nothing changed."
- **Add another provider.** The "+ Add another provider…" row and the palette's "Setup: Add another provider…" open Connect in **add** mode. It saves a credential for another provider, with "Also use it for new chats" **off** by default. So adding a provider never silently changes the default.
- **Untouched areas are never written.** The dashboard makes no writes of its own; only a step's Save writes, under the delta gate.
- **Done** returns to the origin given to TASK-34100.10's entry contract (`open_setup_wizard(origin, resume, start_step)`): Settings, the palette's screen, or Console. It writes nothing, because setup is already complete. Esc is Done on the dashboard, except while a test runs, when Esc stops the test.
- **The secondary action** is [ Run the full walkthrough ]. It opens TASK-34100.10's prefilled corridor with the last-used track preselected (§4.4). Say hello is no longer an exit: it sits in the verdict region with its cost line.
- **Palette commands per area.** Each opens its single step directly and returns to where the palette was used:
  - "Setup: Review your setup", "Setup: Change provider…", "Setup: Add another provider…", "Setup: Change model…";
  - "Setup: Connect a tldw server…" when none is in use, or "Setup: Change tldw server…" when one is;
  - "Setup: Change search…", "Setup: Change tools…", "Setup: Set up spoken replies…", "Setup: Set up dictation…", "Setup: Change appearance…", "Setup: Keys…";
  - "Setup: Say hello", which opens the dashboard with Say hello focused and never sends directly.

  Their names come from §3.0 and TASK-34100.15's registry. They live in a **new command-provider module**, because `app_command_providers.py` has no headroom (1,123 of 1,123 lines, §8.2).
- **Deep links that land on a row.** An entry that names an area opens the dashboard with that row highlighted and its reason expanded, and Enter changes it. Such entries are the palette's area commands when the step can't open directly, and a Settings entry point that passes an area. No link opens Welcome.
- **Console's readiness links are not routed here.** TASK-34100.10 AC#8 stands as written:
  - "no provider configured" opens setup at Connect and returns to the chat;
  - every other reason links to the control that fixes it: the model switcher for a blocked model, Settings ▸ Providers & Models with task-33008's return for a rejected key, and Settings' tldw server row for an unreachable server.

  This narrows TASK-34100.17 AC#8 ("Console readiness deep links landing on the matching row"). §16 Q9 asks the owner to confirm it. The reason: the closest control is one hop, while a dashboard row is two hops with the same result.
- **Entry label.** Both setup buttons follow one rule:
  - today's "Run Setup Wizard" in Settings ▸ Diagnostics (`SS:23301-23305`);
  - the Settings ▸ Overview entry that TASK-34100.15 AC#4 adds.

  They read **Review setup** when a usable chat provider exists, **Resume setup** while a draft exists, and **Run setup** otherwise. "Review setup" is a third verb next to TASK-34100.15's "Run setup" and "Resume setup", and it goes into the same glossary.

**Consistency with TASK-34100.10.** .10 AC#1 opens a re-run on a "Review your setup" header with a short summary of current state and the track choice, in front of the prefilled corridor. This spec replaces that screen with the dashboard. So **.10 AC#1 is narrowed on approval**, before .10 starts (it is To Do): .10 keeps the entry contract, the prefill, "(current)" marks and Cancel-returns-to-origin, and drops the header body that F11 would replace. The entry contract, the origin-return rule and the one-wizard guard are .10's and are reused unchanged.

**Why.** A re-run usually means "change one thing" (report §2(b)). A prefilled 11-step corridor still costs Riley about 75 keystrokes. From the palette, "change the default model" takes about five actions: open the palette, type "model", Enter, pick, Save.

**Rejected alternatives.**
- *The hub as first run too.* A newcomer needs a sequence, not a menu (report §4.3). The predicate keeps it that way: Dee's later "set up AI" opens Quick at Connect, not the dashboard.
- *Choose the dashboard when `setup_completed` is true* (revision 1). Documents-first and Skip both record completion. Dee would have met a menu full of "not set up" rows, with no Ready verdict and no Say hello.
- *Edit values inline in the dashboard rows.* That duplicates each step's validation and commit; the single-step host reuses them.
- *Deep links straight into the step from the dashboard's own entries.* That saves one key, but the user would not see why they were sent there, and could not see the other areas.
- *Route Console readiness links to dashboard rows* (revision 1). It contradicts TASK-34100.10 AC#8, and adds a hop.
- *Say hello as a dashboard exit* (revision 1). An exit should leave the screen; the test belongs with the verdict and its cost line.

**Preserve.**
- Resume and preview safety: the single-step host uses the same step lifecycle.
- Exit and skip dialogs: Cancel in a single step needs no dialog, because nothing is staged outside the step.
- The Summary reads back from disk.

### D8 — Documents-first Welcome choice

**Decision.**

- Welcome's third choice: **"Start with my documents or notes — set up AI later"**.
- While it is selected:
  - the tracker reads "Then: Library ▸ Import" instead of a step count, because by rule S1 no steps follow;
  - a one-paragraph explanation replaces the time line;
  - the primary button reads **[ Open Library → ]** (mockup §3.1).
- **Open Library finishes setup through the single finish path** that TASK-34100.10 AC#12 defines for every completing exit (today `_finalize`, `FRSW:9293-9322`). It records:
  - `setup_completed = true`;
  - the model-list consent default (consent recorded, automatic refresh off), as TASK-34100.10 AC#7 makes Skip do;
  - **`[first_run] ai_setup_deferred = true`**.

  So the "Check model lists online?" modal is never raised. Today `_skip_entirely` records only completion (`FRSW:8987-9008`), which is why that modal fires after a skip [entry-exit-handoff-13]. Nothing else is written: no provider, no `chat_defaults`, no draft. Documents-first never writes a draft, so `_validated_setup_draft` never meets a `docs` track.
- **The next "set up AI" starts at Connect.** While `ai_setup_deferred` is set and no usable chat provider exists (D7's predicate), every setup entry opens **Quick at Connect**, and the walk ends on Ready with its verdict and Say hello. The entries are the Settings button, the palette, and the Get started card's setup link. The finish path clears the flag when a usable provider is saved by any surface.
- **It lands on Library's Import canvas** through the existing "Add your first document" route (`FRSW:6880-6885`, task-32072), with **"Write a note"** as the canvas's secondary action. If the Import canvas does not already offer Library's starter "New note" action (ADR-076 starter landing), F7 adds it to the canvas's first-use state, labelled "Write a note" and routed like today's notes exit (`EXIT_ROUTE_LIBRARY_NOTES`, `FRSW:407-413`). F0's route registry admits the route (§4.7).
- **Console's Get started card is untouched.** With no provider, Console shows the card as it does after Skip. Its plain-language definition of a provider (`Chat/console_onboarding_state.py:19-22`) and its "Write a note in Library" action (`:48`) stay, along with task-33008's in-place connect.
- **Skipping stays one gesture.** Esc on Welcome still raises today's Skip dialog (`FRSW:9744-9760`), with "Keep going" focused. The new choice adds nothing to it.
- **task-28019 is narrowed to its modal-sequencing AC#3** (§0.4). Its AC#1–#2 are delivered by F7, and the narrowed task closes when TASK-34100.10 AC#18 lands.

**Why.** Dee came for documents. Setup is about AI providers and offers Dee nothing. A third choice costs one row and gets Dee to value in three keys (Down, Down, Enter) without a dead-end skip dialog [entry-exit-handoff-28]. Recording the deferral means Dee's later "set up AI" is a sequence that ends in a verdict, not a menu.

**Rejected alternatives.**
- *A fourth Summary exit* (task-28019's attempt). It cannot fit at 80x24 (task-28019 notes), and it still makes Dee walk Provider and Model first.
- *A five-way router question.* Rejected (D12.1). This is one extra choice, not a router.
- *Make documents-first the default for profiles with no detected provider.* The default must be predictable; detection belongs in Connect.
- *Record only completion* (revision 1). Then the next "set up AI" opened the dashboard (H2 in the engineering review).

**Preserve.**
- The Get started card catches Skip and Exit.
- Model-list consent is asked once: the finish path records it, and it is never asked as a modal.
- Restore a backup stays reachable from Welcome.

### D9 — Keychain-first key storage (E8)

**Decision.**

- **The threat model, stated once and used in the copy.** The OS keychain protects a key from being copied, synced, backed up or shared along with config.toml. That covers dotfile repositories, backup archives, a config sent to a colleague, and a config.toml carried to another machine. It does **not** protect against software running as you: such software can usually ask the keychain too, and Secret Service has no per-application access control. Plain text in config.toml is readable by anything running as you, and by every backup or synced copy of the file. The copy never claims more than this.
- **Where the choice appears.** At every key field that stores a provider key:
  - Connect (TASK-34100.6);
  - Settings ▸ Providers & Models (ADR-012's owner);
  - Spoken replies' inline OpenAI key, which TASK-34100.8 AC#3 builds on Connect's path. It is shown only where the host allows secret entry (D6).
- **It is one sentence, not a form row** (§3.2): "Saved in macOS Keychain when you continue.  [ Change where… ]". **Change where…** discloses a short radio list with the options and their consequences. This keeps Quick at three decisions (track, provider with key, model). It is also why D1 rejected an encryption checkbox: neither grows the busiest form.
- **The options, in order:**
  1. **The system keychain, named by platform:** "macOS Keychain", "Windows Credential Manager", "GNOME Keyring" or "KWallet" (both reached through Secret Service). Recommended, and the default when the canary probe below succeeds. On macOS the disclosed text adds: "macOS may ask whether python3 can use your keychain. Choose Always Allow. It can ask again after Python is updated."
  2. **Encrypted in config.toml:** "asks for a password every time chatbook starts". When encryption is already on, this is the only config.toml option, because the writer encrypts sensitive values automatically (`config.py:7036-7062`).
  3. **Plain text in config.toml:** "readable by programs running as you, and by any backup or synced copy of the file".
- **An environment key is never stored.** When an environment variable for this provider exists, TASK-34100.6 AC#21's choice is the control, and it is amended on approval to this shape:
  - **● Use OPENAI_API_KEY from your environment** — the default. The key is not stored, and the line adds: "chatbook reads it each time it starts, so it must be set in the shell that starts chatbook."
  - **○ Store a different key for this app** — reveals the key field. The storage sentence appears only after the user types or pastes a different key, and stores that key, never the environment's.

  A test asserts that an environment value never reaches the keychain, config.toml, a draft or a log, whatever the user selects (§8.1).
- **Is there a usable keychain? A canary probe, not a name check.**
  - In a worker, with a timeout, the credential owner writes, reads back and deletes a canary item in its namespace. Detecting a backend by module name is not enough: on macOS over SSH the login keychain is locked, and writes fail with `errSecInteractionNotAllowed` although the backend name looks secure.
  - Only the existing secure allowlist is tried: macOS, Windows, Secret Service, libsecret and KWallet. The fail, null, plaintext and file backends never count (`runtime_policy/server_credentials.py:39-46`).
  - **A chainer backend with several secure children** makes today's `_resolve_secure_keyring_backend` return None (`server_credentials.py:427-446`). That would quietly mean "no keychain". F12 extends the resolver to pick the highest-priority secure child. Until then, the disabled option says "Several keychains are installed and chatbook couldn't choose one."
- **No keychain.** This covers headless Linux, SSH, a locked or unresponsive Secret Service, and a probe that timed out.
  - The keychain option is shown disabled, with its reason: "No system keychain on this computer (common over SSH)", or "The system keychain didn't respond", or "The keychain is locked".
  - The default becomes plain text in config.toml, and the sentence says so: "Saved as plain text in config.toml when you continue." The environment default above still applies first.
- **Save order at Connect, and failure.**
  1. Write the key to the keychain.
  2. Read it back and compare.
  3. Write config: `credential_source = "keychain"` and no `api_key`, in one config write.
  4. If step 3 fails, delete the keychain item and stay on the step with the cause.

  **A failed or denied keychain write never falls back silently.** The step stays put, says "chatbook couldn't save the key in macOS Keychain (access was denied). Choose another place to keep it.", and discloses the options with focus on them. Plain text is used only when the user picks it.
- **Long credentials on Windows.** Windows Credential Manager holds at most 2,560 bytes per credential (`server_credentials.py:21-23`). Setup accepts credentials up to 8,192 characters (`FRSS:53`, `_MAX_CREDENTIAL_CHARS`). Long JSON credentials are therefore split with the server-credential store's existing part scheme (`_KEYRING_INDEX_PART_CHARACTERS`, `server_credentials.py:24`). Where parts can't be used, the keychain option is disabled for that credential, with the reason.
- **How keys resolve: a provenance overlay, after first paint.**
  - **The persisted marker.** A new persisted `credential_source = "keychain"` joins `none`, `stored` and `environment`. **Both** allowlists gain it: `_PERSISTED_CREDENTIAL_SOURCES` (`Chat/provider_readiness.py:235`) and the setup state's `_CREDENTIAL_SOURCES` (`FRSS:56`). Without that, `configured_provider_credential_source` returns None and readiness drops into legacy mode (`provider_readiness.py:607`). The key itself never appears in config.toml.
  - **No keychain access at config load, and none on the UI loop.** `load_settings` runs synchronously in `TldwCli.__init__`. The config is re-published after every write under `_config_write_lock` (`_publish_runtime_config_unlocked`, `config.py:7455`) and on every `force_reload`. A keychain fetch there could block under the write lock on an access prompt.
  - **Resolution runs in a post-first-paint worker.** The worker lazy-imports `keyring`, and only when some provider has `credential_source = "keychain"`. `keyring` is in neither the boot-import nor the UI-ready census today, and it stays out of both (ADR-097).
  - **One overlay, one choke point.** The resolved values go into a **credential overlay** that records each value's provenance. Every reader sees one value through it: the spend path (`config.py:1728` `_normalize_legacy_provider_api_key`), `get_api_key` (`config.py:9948`), `resolve_provider_credential` (`provider_readiness.py:610`) and the readiness checks. This is the ADR-012 2026-09-19 lesson ("readiness and spend disagree") applied before it happens. The publish path applies the overlay synchronously from its cache and never fetches.
  - **Four states per provider:**
    - **resolving:** readiness reads "Checking the keychain…", never "key missing";
    - **resolved;**
    - **absent:** "Key not found in this computer's keychain", with [ Fix key ];
    - **waiting for you:** a keychain prompt is open, or the keychain is locked. It reads "Waiting for macOS Keychain — answer its prompt, then [ Check again ]".

    A timeout is never cached, and [ Check again ] retries without a restart.
  - **The send path.** A send worker, already off the UI loop, waits for a resolving entry with a bounded timeout, then fails as "key missing — the system keychain didn't return it". Nothing else is ever sent in its place.
  - **Freshness.** Entries carry their read time. A send worker refreshes an entry older than the server-credential store's TTL (`_SECRET_READ_TTL_SECONDS = 5.0`, `server_credentials.py:478`) before dispatch, and keeps the cached value if the keychain doesn't answer. A 401 from that provider drops the entry. So a key rotated in another chatbook instance takes effect at the next send, which matters because the owner runs concurrent instances on purpose (`Utils/instance_lock.py:3-4`).
- **No writer can copy a keychain key into config.toml.**
  - Because the overlay keeps provenance, `save_settings_to_cli_config` refuses to persist an `api_key` for a provider whose `credential_source = "keychain"`, or any value whose provenance is the overlay.
  - The same guard covers every writer that round-trips whole settings: Settings provider forms, the wizard's `_mirror_into_app_config` (`FRSW:9235`), and `app.app_config = override` (`app.py:1985`).
  - Two tests enforce it: an architecture-level **writer census**, and a property test that load → save leaves no secret on disk for keychain-sourced providers (§8.1).
- **Precedence** keeps its shape: an explicit stored credential (keychain, or stored in config.toml) outranks the environment variable, which outranks legacy `[API]` (ADR-012's 2026-09-19 amendment). Because an environment key is never stored, this only matters when the user deliberately stores a *different* key.
- **Namespace: a stable scope id, not a path hash.**
  - Keys live under the service `tldw_chatbook.provider_credentials.<credential_scope_id>`. Its username is the provider's canonical key (`normalize_provider_config_key`, `config.py:1580`).
  - `credential_scope_id` is a random id written once to config, in `[credentials] scope_id`.
  - A hash of the config path, as revision 1 proposed, breaks on symlinks, case changes and moves, and on ADR-126's isolated restore, which creates new credential scopes (ADR-126 decision 9). A persisted id survives all four, and ADR-126's adapter can remap it.
  - Two `TLDW_CONFIG_PATH` profiles never share a key, because each has its own id. Server tokens keep their existing namespace.
- **Ready and the dashboard name where each key lives:** "key in macOS Keychain", "key from OPENAI_API_KEY (not stored)", "key in config.toml (plain text)" and "key in config.toml (encrypted)". This is TASK-34100.15 AC#5's key-source suffix, extended with the keychain.
- **Migration: never automatic.**
  - **Plain-text keys already in config.toml** stay where they are. The Keys row says where each lives with ✓, not a warning mark: plain text in an owner-only file is a legitimate choice, and "!" is reserved for states that stop or degrade something (TASK-25818's restraint). Highlighting the row offers the move. The Keys step (full and single step) and Settings ▸ Privacy & Security (TASK-34100.4's Encryption card) offer **Move to system keychain**. A move writes to the keychain, reads it back and compares, then removes the key from config.toml and sets `credential_source = "keychain"` in one config write. If any step fails, nothing is removed. On success it says: "Moved. Older backups or synced copies of config.toml may still contain the key. If that matters, create a new key at OpenAI and replace this one."
  - **Encrypted keys (`enc:`)** move the same way within an unlocked session, since the values are already decrypted in memory. When the last `enc:` value has moved, the Keys step offers "Nothing is encrypted with your password any more. Turn password encryption off?". That goes through TASK-34100.4's lifecycle owner (`disable_config_encryption`, `config.py:9745`).
  - **Environment-variable users** are untouched.
- **Older builds (config downgrade).** An older build doesn't know `credential_source = "keychain"`. It treats the provider as having no stored key: it uses the environment variable if one is set, and otherwise reports the key missing. It never sends a wrong key. If the user then saves that provider in the older build, the marker is replaced and the keychain item is orphaned. The newer build's Keys step lists orphaned items in its namespace ("A key for OpenAI is in the keychain but not used"), with [ Use it ] and [ Remove it ]. Nothing is deleted automatically.
- **Backups (ADR-126).** Keychain provider keys are Chatbook-owned keyring values. The credential owner therefore registers a typed adapter: they are excluded from portable export by default (ADR-126 decision 5) and captured in local rollback archives (decision 8), and isolated restore remaps the scope id (decision 9). A config.toml carried to another machine says `credential_source = "keychain"` but brings no key, so readiness reads "Key not found in this computer's keychain", with [ Fix key ]. TASK-34100.16's "Setting up another machine" section says so.
- **"Remember on this device" is not shipped.** Storing the master password in the keychain makes encryption equivalent to keychain storage, with the password's extra failure modes added. Where a keychain exists, keychain-first already gives "no plain text, no daily password". Where none exists, there is nothing to remember the password in. The "type it again" recall check is not shipped either: TASK-34100.4's "[R]eset saved keys" makes a forgotten password recoverable, and the setup dialog already confirms the password once.

**Why.** Most users want "my key isn't in a plain-text file" without a password at every launch, and the password path is the most fragile area in the review (SF6). `keyring` is already a core dependency (CLAUDE.md "Key Dependencies") and already holds server tokens and MCP bindings.

**Rejected alternatives.**
- *Encrypted by default.* A password at every launch is the lock-out path of [protect-summary-01/02].
- *Automatic migration of existing keys.* It writes what the user didn't touch (rule 2), and a half-finished move would split the key across two stores.
- *A keychain lookup in each reader at send time.* Every reader would need to learn it, and they would disagree. The overlay gives one choke point.
- *Resolve once at config load, and cache for the process lifetime* (revision 1). It blocks under the config write lock on a keychain prompt, caches a timeout as "missing", and misses a key rotated in another instance.
- *Keychain the default even when an environment key exists* (revision 1). It implies copying an exported value into storage, which breaks "env keys are named and never stored" (report §5.5).
- *Keychain only, with no plain-text fallback.* It strands SSH and headless users (Ash) with no supported store.
- *Silently fall back to plain text when the keychain write fails.* Plain text is a choice the user makes, never a side effect of an error.

**Preserve.**
- Secrets never reach disk unless saved: keychain writes happen only on Save, drafts still refuse secret-named fields, and no writer can persist an overlay value.
- Environment keys are named and never stored.

### D10 — Plain-text setup: `tldw-cli setup --plain` (E12)

**Decision.**

- **Entry: one parser, after the profile is known.**
  - `cli.py` keeps its pre-preflight `recovery` match (`cli.py:30-33`), unchanged. It moves its other arguments to one argparse with subcommands.
  - `setup` is parsed **after** the profile selectors and TASK-34100.16 AC#3's `--config`, so `tldw-cli --config X setup --plain` works. A bare `sys.argv[1:2]` match would not see it.
  - `setup` runs after `startup_preflight` and `admit_startup`, so an encrypted config unlocks through the one shared prompt (`cli.py:70-86`, TASK-34100.4 AC#3).
  - Plain `tldw-cli setup`, without `--plain`, opens the TUI straight into setup, with the first screen chosen by D7's predicate.
- **One setup core that never imports Textual.**
  - The plain driver uses the same track definitions, the same pure setup session (§4.7), the same step-state modules and the same commit builders, all through the config owner. A plain step is a renderer over a step's state, never a second implementation.
  - **Today that core is not import-pure.** `first_run_setup_state.py` imports only stdlib and one path helper at file level (`FRSS:15-25`). But importing it runs `UI/__init__.py`, which installs Textual shims, and `UI/Wizards/__init__.py`, which imports `BaseWizard`. That costs about 0.29 s by `-X importtime` (engineering review, 2026-10-03).
  - So F0 moves the setup core to a new package, `tldw_chatbook/Setup/`. An architecture test asserts that the plain path never imports `textual` (§8.1).
  - **Provider and Model have no pure state today.** Their key checks, detection and mutation building live in Textual classes (`FRSW:1169-3310`). F0 extracts them as pure step-state modules, which plain mode needs (§4.7).
- **No app, so no Console turn and no runtime bind.**
  - `ConsoleRuntime(app)` needs a `TldwCli` (`app.py:1167`). The runtime-source coordinator relies on app objects too: `handle_runtime_backend_changed`, `server_context_provider` and `sync_scope_service` (`SS:26513-26623`).
  - Plain v1 therefore has **no Say hello** and **no tldw server step**. Plain Ready says: "Start chatbook with tldw-cli and send your first message. That is the first real test of this setup."
  - A plain tldw server step needs a launch-time activation seam: save the server and token, then bind and prepare sync at the next launch through the coordinator, reporting the result there. That comes with F13b.
- **The verdict without an app.** Plain Ready prints the shared verdict through its app-free entry. TASK-34100.2's verdict owner must expose that entry: a function over config and the model catalog. If it is missing, F13 adds one in the setup core. It never copies the logic.
- **One prompt per decision.**
  - Numbered choices, with the default in brackets. Typing letters filters long lists (providers, models).
  - `?` explains the current prompt. Only the whole word **`quit`**, or Ctrl+D (end of input), quits, with exit 1. A single `q` is a filter prefix, because Qwen and QwenCloud start with it.
  - Every line is a complete sentence. There is no cursor movement, no spinner, no colour-only meaning and no line rewriting. Screen readers therefore read it in order, and it is safe with `TERM=dumb`.
  - Result lines start with `OK:` or `FAILED:`. With `--glyphs` they start with ✓ or ✗ instead (some screen readers read ✓ as "check mark", others skip it).
  - Progress reads "Step 2 of 4: Connect".
  - Long waits print "Still waiting for Anthropic… 30 s" every 30 seconds. Every wait is bounded.
- **Keys.**
  - Keys are read from a hidden prompt (`getpass`), or from an environment variable named by `--key-env VAR` (applied to the chat provider) or `--key-env PROVIDER=VAR`.
  - **If the terminal can't hide input,** Python's `getpass` would fall back to echoing and only warn (`GetPassWarning`). That warning is turned into an error, and the run exits with code 2: "chatbook can't hide what you type in this terminal, so it won't ask for a key here. Use --key-env NAME."
  - **After a hidden entry,** it confirms what arrived: "Received a key ending abcd (51 characters)."
  - **Argv is refused.** Any argument that looks like a key flag (`--key`, `--api-key`, `--token`) is refused with exit 2: "Keys are never read from the command line, because they end up in shell history and process lists. Use --key-env NAME, or type the key when asked."
  - A key is never echoed, logged or written to a draft.
- **Scope of v1:**
  - Welcome: all three choices. Choosing Full prints the steps v1 can't run, and where to change them.
  - Connect: detection first, then the filterable list; the key; the storage choice once D9 ships, with the environment default.
  - Model: the curated list, plus "type part of a model ID".
  - Ready: the verdict, the Provider, tldw server and Keys lines, and the Data and Config paths.
  - **Plain review:** on a profile where a usable chat provider exists, `tldw-cli setup --plain` prints the areas, numbered, with their values, and asks "Change which area? Type its number, or press Enter when done." In v1, Provider, Model and Keys can be changed. The other areas print their value and "Not changeable in plain mode yet (F13b)".
- **The v1 gap, stated.** Until F13b, a screen-reader user cannot change tldw server, Search, Tools, Spoken replies, Dictation or Appearance without the TUI. That contradicts G6 for those areas, so F13b is filed with F13 rather than left open (§10, Q7).
- **Discoverability.**
  - `tldw-cli --help` lists `setup` and `--plain`.
  - When the first-run offer is about to show, chatbook prints one line before the TUI takes the screen: "Using a screen reader? Quit and run: tldw-cli setup --plain".
  - The README and the User Guide carry the same line.
- **Exit codes.**

  | Code | Meaning |
  |---|---|
  | 0 | Setup saved and the verdict is ✓ (Ready, or Set up with the key not checked yet), or documents-first was chosen |
  | 1 | Quit before the provider and model were saved, or a save failed |
  | 2 | Usage error: bad flags, a key on argv, no terminal (stdin is not a TTY), or a terminal that can't hide input |
  | 3 | Saved, but the verdict is ✗ "Can't chat yet" |

- **Concurrency.** Writes go through the config owner's atomic path. `Utils/instance_lock.py` is "detection only — a second instance gets a warning toast, never a lock-out (the owner runs concurrent instances deliberately)" (`instance_lock.py:3-4`), and it is keyed on the user data directory. So plain setup **warns and asks**, and never refuses: "chatbook may be running with this profile. It could overwrite what you change here. Continue? [y/N]".

**Transcripts:** §3.10 (Quick, Anthropic, keychain) and §3.11 (plain review).

**Why.** Textual exposes no accessibility tree. A line-oriented flow on the same setup core is the cheapest way to make setup usable with a screen reader and over SSH.

**Rejected alternatives.**
- *Make the Textual wizard screen-reader friendly.* Not possible without an accessibility tree.
- *A separate "setup script" that edits TOML.* It would be a second implementation that drifts.
- *Say hello and the server step in plain v1* (revision 1). They need the app, and a second, app-free Console path would be the drift the probe turn avoids.
- *Refuse to run while the TUI is open* (revision 1). It contradicts the owner's concurrent-instances rule.
- *`q` quits* (revision 1). It collides with filtering.

**Preserve.**
- Secrets never reach disk unless saved.
- Environment keys are never stored.
- Model-list consent is asked once: plain mode asks once, default No, and records the answer.
- Restore stays reachable: plain Welcome prints "Moving from another computer? Run: tldw-cli recovery --help".

### D11 — Non-interactive setup and a settings export (E6, beyond TASK-34100.16): defer

**Decision: defer.** This reverses revision 1's "build it, last".

**Why defer.**
- **The report advises it.** Its verifiers say to defer scripted setup "until users ask for it" (report §2(c) item 4). The report records copying config.toml with `setup_completed = true` as a route that **works** (report §2(c) table). Revision 1 called that route "the riskiest habit" without new evidence.
- **The cost is not small.** The CLI has no app, so `--from` could neither bind a server nor run the test. The export needs a strict, allowlisted schema of its own, because the draft's secret-name rule would refuse its own `[credentials.*]` table and Voice's `authentication_mode`. The tool-gate keys also grow with `all_tool_gates()`, so a strict "unknown key fails" would break older exports on newer builds.
- **TASK-34100.16 already covers the second machine.** It documents the copy-config route, `--config`, and the keychain note (D9).

**Reopen condition.** Reopen on whichever comes first:
- the first user request for scripted or unattended setup;
- the first support case caused by a copied config.toml that carried plain-text keys.

**Design notes kept for reopening** (not approved, and not filed):
- **Same core.** Build on D10's setup core and step states, with the same commit builders and verdict.
- **A strict, allowlisted schema.** Use `key_sources.*` tables, not `credentials.*`. Allow `source = keychain | environment | prompt | none` and `env_var`, and never a value.
- **Secret detection on values too.** Check values as well as names: URL userinfo, token query parameters, and known key prefixes.
- **Tool gates are lenient.** Unknown or retired tool gates are warnings, not errors.
- **Keys only from the environment or stdin,** and never from argv.
- **Paired with import.** "Export these settings (no keys)" ships with an import that Welcome's "Moving from another computer?" recognises.

**Rejected alternative.** *Build it now, last* (revision 1). Rejected for the reasons above, so §16 Q6 asks the owner to confirm the deferral.

**Preserve.** Restore a backup is unaffected.

### D12 — The report's "Rejected, on purpose" list: confirm or overturn

The owner is asked to confirm each item. The recommendation for all seven is **confirm**. Nothing in D1–D11, as revised, brings one back.

| # | Item (report §4.3) | Recommendation | Why | Owner ruling |
|---|---|---|---|---|
| 1 | A five-way "How will you use chatbook?" router on Welcome | Confirm | It taxes the local-first majority on the first screen. Connect's "Ready on this machine" group detects the same things (TASK-34100.6 AC#29). D8's documents-first choice is one extra row, not a router. | pending |
| 2 | A master tool switch in setup | Confirm | It invites a reflexive "all off" that silently removes web search and Watchlists. TASK-34100.13 AC#2 states the real posture instead. | pending |
| 3 | An embedding-model picker | Confirm | Changing the model clones profiles and rebuilds the index. TASK-34100.13 AC#1 shows it read-only. | pending |
| 4 | A multi-tick provider checklist | Confirm | Env-keyed providers are recorded automatically (TASK-34100.6 AC#6). "Add another provider" is one action on the dashboard and in the palette (D7), not a checklist. | pending |
| 5 | Base-URL overrides on every keyed provider | Confirm | Few users run gateways, and the form would grow for everyone. Settings keeps the override (TASK-34100.6 AC#21). | pending |
| 6 | An "Undo this session" ledger before Voice writes only deltas | Confirm, and keep it rejected after the deltas ship | Restoring a snapshot races Settings and Console writers, and it cannot undo encryption or downloads. The Exit dialog's list of saved areas (TASK-34100.9 AC#3) plus D7's single-step changes cover the need. Reopen only on user reports of needing to undo a whole run. | pending |
| 7 | An inline "disable TLS verification" toggle | Confirm | It nudges first-time users toward `ssl_verify = false`. A certificate failure is classified and links to Settings ▸ Network (ADR-079; TASK-34100.6 AC#7; D3's probe). | pending |

---

## 2. Flows per persona

Key counts assume TASK-34100.6/.7/.11 have shipped: Enter selects, the detected row is pre-highlighted, and the recommended model is pre-selected.

### 2.1 Sam — cloud, OpenAI key, Quick

1. **Welcome:** Enter (Quick is the default).
2. **Connect:** type `open` to filter, Enter on OpenAI, paste the key, Enter. The key is checked: "✓ Key works — OpenAI returned 137 models". The sentence under the key already reads "Saved in macOS Keychain when you continue." Next.
3. **Model:** the curated recommendation is pre-selected. Enter.
4. **Ready:** "✓ Ready to chat — OpenAI · gpt-4.1-mini (1M context)". Focus is on Start chatting, and the hint says "Tab: test message first". Sam presses Tab, reads "Costs under $0.01 (a few tokens)", and presses Say hello: "Test reply in 0.8 s". Start chatting opens a new chat, and the arrival line quotes the test reply and names the test conversation in History.

If something goes wrong at the test:
- **The account has no credit:** Ready reads "✗ Can't chat yet — OpenAI accepted the key but refused the message", with the billing line and [ Try again ].
- **The key is rejected:** Ready flips to ✗ with [ Fix key ]. Sam fixes it on Connect and comes back to Ready without re-walking Model.

### 2.2 Jo — private local AI

- **Path B, llama.cpp running.**
  1. Welcome: Enter.
  2. Connect: the first row is "llama.cpp on this computer · 127.0.0.1:9099 · 3 models", pre-highlighted. Enter.
  3. Model: Enter.
  4. Ready: the test starts by itself. This is first run, llama.cpp is on the allowlist, the address is loopback, there is no key, and the server serves the model it was started with. Ready shows "Saying hello… 4 s (the first reply can be slow while the model loads)", then "Test reply in 6.1 s". The Provider row reads "llama.cpp — answers on this computer", and the tldw server row reads "none — Library and notes stay on this computer".

  That is about six keys from Welcome to a reply.
- **Path B with Ollama, model not loaded.** Nothing runs by itself. Ready shows [ Say hello ] with "Loads llama3.1:8b into memory (about 4.7 GB). Another loaded model may be unloaded. Nothing is billed." Jo presses it when ready.
- **Path A, nothing running.** Connect lists what it checked and offers "Use <model> — I'll start it later" (TASK-34100.6 AC#18/#30). Ready says "✗ Can't chat yet — nothing is answering at localhost:11434", with "Start Ollama, then check again." and [ Check again ]. [ Go to Console ] still finishes setup, and the Get started card takes over. No cloud-key form appears anywhere on this path.

### 2.3 Riley — power user

- **First install, env keys.**
  - TASK-34100.2 AC#12's env-aware Get started card covers the zero-wizard path.
  - To get the rest, Riley runs `tldw-cli setup` and chooses Full.
  - On Connect, "● Use OPENAI_API_KEY from your environment (found; not stored)" is preselected, and nothing is stored.
  - On Model, Riley picks a model and presses **Save and finish with defaults**.
  - Ready shows "– Left at their defaults: tldw server, Search, Tools, Spoken replies, Dictation, Appearance, Keys".
- **Re-run to change one thing.** Ctrl+P, type "model", Enter: the Model single step opens. Pick, Save, and Riley is back where they were. About five actions, and nothing else is written.
- **Second machine.** Riley follows TASK-34100.16's "Setting up another machine": carry config.toml (it holds no keys, because the keys come from the environment) and export the same variables. A provider whose key was in the keychain reads "Key not found in this computer's keychain" with [ Fix key ]. Scripted setup is deferred (D11).

### 2.4 Dee — documents first

Welcome: Down, Down (the tracker now reads "Then: Library ▸ Import"), Enter. Library Import opens, with "Write a note" beside it, and no modal fires. Weeks later Dee chooses Settings ▸ "Run setup". Because AI setup was deferred and no provider exists, setup opens **Quick at Connect**, and the walk ends on Ready with its verdict and Say hello.

### 2.5 Ash — screen reader or SSH

Ash runs `tldw-cli setup --plain` (transcript in §3.10). `tldw-cli --help` lists it, and so does a line printed before the TUI starts. Over SSH with no keychain:
- the storage prompt offers "1. Plain text in config.toml" as the default;
- when `ANTHROPIC_API_KEY` is set, "Use ANTHROPIC_API_KEY from your environment (not stored)" is the default instead.

The run exits 0 when the verdict is ✓.

---

## 3. Screen specs

Mockups assume TASK-34100.12's short tier at 80x24: title and tracker share one row, there is no outer border, the nav takes one row and the hint takes one row. Glyphs follow §3.0 and survive NO_COLOR.

### 3.0 Area names, glyphs and ASCII marks

**One name per area (rule S5).** Each name enters TASK-34100.15's registry. Ready rows, dashboard rows, What's next, palette commands and plain mode all use the area label.

| Area (step id) | Area label | Step title (tracker) | Palette command |
|---|---|---|---|
| `provider` | Provider | **Connect**, the one exception: it is the AC's verb for what you do on the step, and its page heading reads "Connect an AI provider" | Setup: Change provider… · Setup: Add another provider… |
| `model` | Model | Model | Setup: Change model… |
| `server` | tldw server | tldw server | Setup: Connect a tldw server… (none in use) · Setup: Change tldw server… |
| `rag` | Search | Search | Setup: Change search… |
| `tools` | Tools | Tools | Setup: Change tools… |
| `voice` | Spoken replies | Spoken replies | Setup: Set up spoken replies… |
| `speech` | Dictation | Dictation | Setup: Set up dictation… |
| `appearance` | Appearance | Appearance | Setup: Change appearance… |
| `protect-keys` (id unchanged) | Keys | Keys | Setup: Keys… |
| `summary` | — | Ready | Setup: Review your setup · Setup: Say hello · Setup: Finish with defaults |

- **Entry verbs:** "Run setup", "Resume setup" and "Review setup". The dashboard's title is "Review your setup".
- **Revision 1's drift is gone.** These names replace Connect/Model/Keys on Ready versus Provider/Default model/Keys on the dashboard, "Runtime" versus "Server" versus "Connect a tldw server", "Protect keys" versus "Keys", and "Search" versus "document search".

**Glyphs and their ASCII forms.** The ASCII forms apply with TASK-34100.13 AC#12's "Plain ASCII status marks" (preset from `NO_COLOR` or `TERM=linux`). CI renders every §3 screen in ASCII mode too (§8.1).

| Glyph | Meaning | ASCII |
|---|---|---|
| ✓ | saved, kept current, or working | `+` |
| – | skipped, not set up, or left at its default | `-` |
| ! | needs attention (degrades something) | `!` |
| ✗ | failed (stops something) | `x` |
| ● / ○ | radio selected / not selected | `(*)` / `( )` |
| [ ] / [✓] | checkbox | `[ ]` / `[x]` |
| → | What's next item | `->` |
| ▸ / ▾ | collapsed / expanded, and the breadcrumb separator | `>` / `v` |
| › | highlighted row in a list that has a detail line | `>` |
| · | separator inside a value | `\|` |
| — | dash in copy | `--` |
| … | "opens more" in a label | `...` |
| ← / ↑↓ | Back, browse | `<-` / `up/down` |
| • | masked key character | `*` |

**Legend.** Wherever –, ! or ✗ is on screen, including in the tracker, a dim legend row reads "✓ saved · – skipped · ! needs attention · ✗ failed" (TASK-34100.9 AC#5). In ASCII mode it reads "+ saved | - skipped | ! needs attention | x failed".

### 3.1 Welcome

Quick highlighted (the default):

<!-- mockup welcome 80x24 -->
```text
Welcome to chatbook                                  Step 1 of 4 · next: Connect

Chat with cloud or local AI models, keep notes, and work with your own
documents — all in your terminal. With a model running on this computer,
your messages are answered on it.

What would you like to do first?
 ● Quick setup — connect AI and pick a model (recommended)
 ○ Full setup — adds search, tools, voice, appearance, server
 ○ Start with my documents or notes — set up AI later

Quick takes about 2 minutes if you have an API key or a local AI server
running. You can change any of this later by running setup again.

[ ] Reduce motion — fewer animations, starting now
Moving from another computer?  [ Restore a backup ]






                                                                      [ Next → ]
↑↓ choose · Enter next · Tab more options · Esc skip setup
```

With the documents-first choice highlighted. Short radio groups select on highlight, per TASK-21142 and TASK-34100.11:

<!-- mockup welcome-documents-first 80x24 -->
```text
Welcome to chatbook                                       Then: Library ▸ Import

Chat with cloud or local AI models, keep notes, and work with your own
documents — all in your terminal. With a model running on this computer,
your messages are answered on it.

What would you like to do first?
 ○ Quick setup — connect AI and pick a model (recommended)
 ○ Full setup — adds search, tools, voice, appearance, server
 ● Start with my documents or notes — set up AI later

Setup finishes now and opens Library, where you can import files or write a
note. Nothing else is changed. When you want AI, run setup again: it starts
at Connect.

[ ] Reduce motion — fewer animations, starting now
Moving from another computer?  [ Restore a backup ]





                                                              [ Open Library → ]
↑↓ choose · Enter open Library · Tab more options · Esc skip setup
```

At 120x40, each choice gets a one-line gloss and the tracker names the steps:

<!-- mockup welcome 120x40 -->
```text
Welcome to chatbook                                                 Step 1 of 4 · ● Welcome  ○ Connect  ○ Model  ○ Ready

Chat with cloud or local AI models, keep notes, and work with your own documents — all in your terminal.
With a model running on this computer, your messages are answered on it.

What would you like to do first?

 ● Quick setup — connect AI and pick a model (recommended)
     About 2 minutes if you have an API key or a local AI server running. Connect it, then pick the model for new chats.
 ○ Full setup — adds search, tools, voice, appearance, server
     Every extra step is optional and starts on "not now". Downloads happen only if you choose them.
 ○ Start with my documents or notes — set up AI later
     Opens Library now. Import files or write a note; when you want AI, run setup again and it starts at Connect.

You can change any of this later by running setup again.

[ ] Reduce motion — fewer animations, starting now
Moving from another computer?  [ Restore a backup ]




















                                                                                                              [ Next → ]
↑↓ choose · Enter next · Tab more options · Esc skip setup
```

- **Focus order:** the track list, then [ Next → ], then Reduce motion, then [ Restore a backup ]. TASK-34100.13 AC#15 puts Next before Restore. "Tab more options" in the hint is what tells the user that the two rows above the nav exist.
- **Tracker:** "Step 1 of 4" for Quick and "Step 1 of 11" for Full, updated as the highlight moves (TASK-34100.9 AC#5); "Then: Library ▸ Import" for documents-first.
- **Reduce motion** applies live, and is offered only where it can be (TASK-34100.13 AC#16). It writes only its own appearance key, as a delta, and only when ticked.
- **Restore a backup** keeps today's handler (`FRSW:6392-6396`), and TASK-34100.16 makes it open on Inspect.

### 3.2 Connect: where the key is kept (D9)

Only the storage line is new; the rest of Connect is TASK-34100.6's. With a working keychain, after a key check:

<!-- mockup connect-key-storage 80x4 -->
```text
OpenAI — API key
 [ ••••••••••••••••••••••••••••••••••••••abcd ]  [ Show ]  [ Check key ]
 ✓ Key works — OpenAI returned 137 models. Pick one on the next step.
 Saved in macOS Keychain when you continue.  [ Change where… ]
```

After **Change where…** (a short radio group; the highlight selects):

<!-- mockup connect-key-storage-disclosed 80x11 -->
```text
OpenAI — API key
 [ ••••••••••••••••••••••••••••••••••••••abcd ]  [ Show ]  [ Check key ]
 ✓ Key works — OpenAI returned 137 models. Pick one on the next step.
 Keep this key in:
  ● macOS Keychain (recommended) — not in any file. macOS may ask whether
    python3 can use your keychain: choose Always Allow. It can ask again
    after Python is updated.
  ○ Encrypted in config.toml — asks for a password every time chatbook starts
  ○ Plain text in config.toml — readable by programs running as you, and by
    any backup or synced copy of the file
```

When `OPENAI_API_KEY` is set (TASK-34100.6 AC#21, amended). The key field and the storage line appear only after "Store a different key for this app":

<!-- mockup connect-env-key 80x6 -->
```text
OpenAI — API key
 ● Use OPENAI_API_KEY from your environment (found; not stored)
   chatbook reads it each time it starts, so it must be set in the shell
   that starts chatbook.
 ○ Store a different key for this app
```

Without a usable keychain:

<!-- mockup connect-no-keychain 80x5 -->
```text
OpenAI — API key
 [ ••••••••••••••••••••••••••••••••••••••abcd ]  [ Show ]  [ Check key ]
 Saved as plain text in config.toml when you continue.  [ Change where… ]
 No system keychain on this computer (common over SSH).
```

When the keychain refuses the write. The step stays put and nothing is saved:

<!-- mockup connect-keychain-save-failed 80x9 -->
```text
OpenAI — API key
 [ ••••••••••••••••••••••••••••••••••••••abcd ]  [ Show ]  [ Check key ]
 ✗ chatbook couldn't save the key in macOS Keychain (access was denied).
   Choose another place to keep it:
  ○ macOS Keychain — try again
  ● Encrypted in config.toml — asks for a password every time chatbook starts
  ○ Plain text in config.toml — readable by programs running as you, and by
    any backup or synced copy of the file
```

- **Choosing Encrypted in config.toml** for the first time opens TASK-34100.4's password dialog when the step is saved, not when the option is highlighted.
- **Before D9 ships,** the storage line is absent, and keys go where they go today. Ready's "Encrypt saved keys…" option covers encryption on Quick.

### 3.3 Model: Save and finish with defaults (Full only)

On Full, Model's nav row:

<!-- mockup model-nav-full 80x2 -->
```text
[ ← Back ]  [ Save and finish with defaults ]              [ Save & continue → ]
Enter choose · Tab next action · ← Back, Ctrl+B or Alt+← · Esc exit setup
```

- Steps 4–10 read `[ ← Back ]  [ Finish with defaults ]  …  [ Save & continue → ]`. The right-hand button reads "Continue" when the step will not write (TASK-34100.9 AC#1).
- Say the user presses Finish with defaults on Search. Setup goes to Ready, and Search, Tools, Spoken replies, Dictation, Appearance and Keys write nothing.

### 3.4 tldw server (Full, step 4)

<!-- mockup server-step 80x24 -->
```text
tldw server (optional)                               Step 4 of 11 · next: Search

chatbook works fully on this computer. If you run a tldw server, chatbook
can also connect to it. Connecting also prepares sync: chatbook sends the
server this computer's name (mac-studio) and sets up sync for it there.

 ○ Not now — this computer only
 ● Connect to a tldw server

   Server address  [ https://lab.example.org:8000            ]
   API token       [ ••••••••••••••••••••••••              ]  [ Show ]
   [ Test connection ]
   ✓ Connected — lab.example.org:8000 is a tldw server and accepted
     the token.








[ ← Back ]  [ Finish with defaults ]                       [ Save & continue → ]
Enter choose · Ctrl+N next · ← Back, Ctrl+B or Alt+← · Esc exit setup
```

| Element | Copy and behaviour |
|---|---|
| Title | tldw server (optional) |
| Body | chatbook works fully on this computer. If you run a tldw server, chatbook can also connect to it. Connecting also prepares sync: chatbook sends the server this computer's name (<hostname>) and sets up sync for it there. **F3's first task confirms exactly what preparation sends, and this sentence names it. It is shown before the choice, not after a test.** If the owner rules against sync preparation in setup (Q5), the sentence goes. |
| Choice 1 (default) | Not now — this computer only. Writes nothing. Next reads "Continue →". |
| Choice 2 | Connect to a tldw server. Reveals Server address and API token. |
| Server address | Placeholder `https://your-server:8000`. Prefilled only from a **bound** server (`RuntimePolicyContext`), never from the template's `127.0.0.1:8000`. Validated as a server root with no path, as the modal does today (`server_switch_modal.py:167-193`). |
| API token | Masked, with [ Show ]. Never prefilled with the template placeholder. A saved token shows as "saved in macOS Keychain", with [ Replace ] / [ Clear ] (the wizard spec's Keep/Replace/Clear for secrets). |
| Test connection | Runs D3's shared probe, with the bound client's TLS settings, and shows D3's copy. While it runs: "Checking lab.example.org:8000…". |
| Save & continue | Commits through the shared coordinator.<br>• **A definitive failure refuses the save:** token rejected, not a tldw server, or blocked by policy. The cause shows under the field, as TASK-34100.6 AC#13 does for keys.<br>• **An unreachable or untested server gets one inline line, not a modal:** "lab.example.org:8000 wasn't confirmed as a tldw server.  [ Save anyway ]  [ Keep editing ]", with "Keep editing" focused. So "set it up now, start it later" still works.<br>• **On a commit failure** the coordinator restores the prior `[tldw_api]` values and removes any token it stored. The step stays put with the cause, and nothing is half-applied, on disk or in memory (D3).<br>• A successful bind is listed in the Exit dialog (TASK-34100.9 AC#3). |
| Re-run | With a server bound, choice 2 is pre-selected, with the address and "token saved". Choosing "Not now" switches back through the same coordinator. |

At 120x40 the layout is the same, with wider fields, and the success line fits on one row.

### 3.5 Keys (Full, step 10)

The state-aware content comes from TASK-34100.4 AC#7/#8. With D9, the step first lists where each key lives, then offers only the actions that apply:

| State | Body | Actions |
|---|---|---|
| Every key in the keychain or the environment | Your keys aren't stored in config.toml: OpenAI is in macOS Keychain; Anthropic comes from ANTHROPIC_API_KEY. Nothing to change here. | (none; tracker "–"; G2's accepted exception) |
| A plain-text key, keychain available | OpenAI's key is plain text in config.toml. Programs running as you, and any backup or synced copy of the file, can read it. | [ Move to macOS Keychain ]  [ Encrypt with a password… ]  [ Keep as plain text ] |
| A plain-text key, no keychain | (same body) | [ Encrypt with a password… ]  [ Keep as plain text ] |
| Already encrypted | Your saved keys are encrypted with your password. | [ Move to macOS Keychain ]  [ Change password… ] |
| Orphaned keychain items (D9, older build) | A key for OpenAI is in the keychain but not used. | [ Use it ]  [ Remove it ]  [ Leave it ] |
| No key stored | Nothing to change — no API key is saved in config.toml. | (none; tracker "–") |

After a move, the step shows: "Moved. Older backups or synced copies of config.toml may still contain the key. If that matters, create a new key at OpenAI and replace this one."

### 3.6 Ready

**Quick, cloud, before the test.** Focus is on Start chatting, and Say hello is next in Tab order:

<!-- mockup ready-cloud-untested 80x24 -->
```text
Ready                                  Step 4 of 4 · ✓ Welcome ✓ Connect ✓ Model
✓ Ready to chat — OpenAI · gpt-4.1-mini (1M context)
  Test it  [ Say hi in five words.                     ]  [ Say hello ]
  Sends one short message to OpenAI. Costs under $0.01 (a few tokens).
✓ Provider     OpenAI — your messages go to OpenAI · key in macOS Keychain
✓ Model        gpt-4.1-mini — new chats start with it
✓ tldw server  none — Library and notes stay on this computer
Data:   ~/.local/share/tldw_cli/default_user                     [ Copy ]
Config: ~/.config/tldw_cli/config.toml                           [ Copy ]

What's next — optional, any time
 → Hear replies aloud…
 → Connect a tldw server… (if you run one)
 → Sync a notes folder…
 ▸ More (6)
[ ] Refresh OpenAI's model list when chatbook starts
[ ] Get to know you after setup — a short questionnaire, no AI needed





[ ← Back ]  [ Start chatting ]  [ Add a document ]  [ Write a note ]
Enter start chatting · Tab: test message first · ← Back, Ctrl+B or Alt+←
```

**Quick, OpenRouter: set up, but the key is not checked yet.** Focus starts on Say hello, which ignores activation for 0.5 s:

<!-- mockup ready-key-not-checked 80x24 -->
```text
Ready                                  Step 4 of 4 · ✓ Welcome ✓ Connect ✓ Model
✓ Set up — OpenRouter · anthropic/claude-sonnet-5-5 (200K context)
  The key isn't checked yet: OpenRouter lists models for any key.
  Test it  [ Say hi in five words.                     ]  [ Say hello ]
  Sends one short message to OpenRouter. Costs under $0.01 (a few tokens).
✓ Provider     OpenRouter — messages go to OpenRouter · key in macOS Keychain
✓ Model        anthropic/claude-sonnet-5-5 — new chats start with it
✓ tldw server  none — Library and notes stay on this computer
Data:   ~/.local/share/tldw_cli/default_user                     [ Copy ]
Config: ~/.config/tldw_cli/config.toml                           [ Copy ]

What's next — optional, any time
 → Hear replies aloud…
 → Connect a tldw server… (if you run one)
 → Sync a notes folder…
 ▸ More (6)
[ ] Refresh OpenRouter's model list when chatbook starts
[ ] Get to know you after setup — a short questionnaire, no AI needed




[ ← Back ]  [ Start chatting ]  [ Add a document ]  [ Write a note ]
Enter send the test message · Tab next action · ← Back, Ctrl+B or Alt+←
```

**Quick, local, test running by itself** (all five auto-run conditions hold):

<!-- mockup ready-local-testing 80x24 -->
```text
Ready                                  Step 4 of 4 · ✓ Welcome ✓ Connect ✓ Model
✓ Ready to chat — llama.cpp on this computer · llama-3.1-8b (8K context)
  Saying hello… 4 s (the first reply can be slow while the model loads)
  [ Skip test ]
✓ Provider     llama.cpp — answers on this computer (127.0.0.1:9099), no key
✓ Model        llama-3.1-8b-instruct-q4_k_m.gguf — new chats start with it
✓ tldw server  none — Library and notes stay on this computer
Data:   ~/.local/share/tldw_cli/default_user                     [ Copy ]
Config: ~/.config/tldw_cli/config.toml                           [ Copy ]

What's next — optional, any time
 → Hear replies aloud…
 → Connect a tldw server… (if you run one)
 → Sync a notes folder…
 ▸ More (6)
[ ] Get to know you after setup — a short questionnaire, no AI needed






[ ← Back ]  [ Start chatting ]  [ Add a document ]  [ Write a note ]
Esc skip the test · Tab next action · ← Back, Ctrl+B or Alt+←
```

**Local, where the test would load a model.** Nothing runs until the user presses:

<!-- mockup ready-local-needs-load 80x4 -->
```text
✓ Ready to chat — Ollama on this computer · llama3.1:8b (128K context)
  Test it  [ Say hi in five words.                     ]  [ Say hello ]
  Loads llama3.1:8b into memory (about 4.7 GB). Another loaded model may be
  unloaded. Nothing is billed.
```

**After the reply** (verdict region only; the rest is as above):

<!-- mockup ready-replied 80x3 -->
```text
✓ Ready to chat — OpenAI · gpt-4.1-mini (1M context)
  Test reply in 0.8 s — gpt-4.1-mini: "Hello there, nice to meet you!"
  Start chatting opens a new chat; this test stays in your History.
```

**The test found no credit on the account** (verdict region only):

<!-- mockup ready-no-credit 80x4 -->
```text
✗ Can't chat yet — OpenAI accepted the key but refused the message.
  The account has no credit or hit its spending limit (HTTP 429). Add
  billing in your OpenAI account, then try again.
  [ Try again ]
```

**Verdict ✗ from the offline preflight**, for example a context window that is still unknown:

<!-- mockup ready-cant-chat 80x24 -->
```text
Ready                                  Step 4 of 4 · ✓ Welcome ✓ Connect ! Model
✗ Can't chat yet — gpt-5.6-terra's context size is unknown, so every
  message would be blocked before it is sent.
  [ Choose another model ]  [ Set context size… ]
✓ Provider     OpenAI — your messages go to OpenAI · key in macOS Keychain
! Model        gpt-5.6-terra — saved, but it can't be used yet (see above)
✓ tldw server  none — Library and notes stay on this computer
✓ saved · – skipped · ! needs attention · ✗ failed
Data:   ~/.local/share/tldw_cli/default_user                     [ Copy ]
Config: ~/.config/tldw_cli/config.toml                           [ Copy ]

What's next — optional, any time
 → Hear replies aloud…
 → Connect a tldw server… (if you run one)
 → Sync a notes folder…
 ▸ More (6)
[ ] Refresh OpenAI's model list when chatbook starts
[ ] Get to know you after setup — a short questionnaire, no AI needed




[ ← Back ]  [ Go to Console ]  [ Add a document ]  [ Write a note ]
Enter choose another model · Tab next action · ← Back, Ctrl+B or Alt+←
```

**The test failed** (an expired key on a provider whose model list proves nothing):

<!-- mockup ready-test-failed 80x24 -->
```text
Ready                                  Step 4 of 4 · ✓ Welcome ! Connect ✓ Model
✗ Can't chat yet — OpenRouter rejected the key.
  The test message failed: OpenRouter said "API key expired." (HTTP 401).
  Nothing else was sent.  [ Fix key ]
! Provider     OpenRouter — key in macOS Keychain, rejected by OpenRouter
✓ Model        anthropic/claude-sonnet-5-5 — new chats start with it
✓ tldw server  none — Library and notes stay on this computer
✓ saved · – skipped · ! needs attention · ✗ failed
Data:   ~/.local/share/tldw_cli/default_user                     [ Copy ]
Config: ~/.config/tldw_cli/config.toml                           [ Copy ]

What's next — optional, any time
 → Hear replies aloud…
 → Connect a tldw server… (if you run one)
 → Sync a notes folder…
 ▸ More (6)
[ ] Refresh OpenRouter's model list when chatbook starts
[ ] Get to know you after setup — a short questionnaire, no AI needed




[ ← Back ]  [ Go to Console ]  [ Add a document ]  [ Write a note ]
Enter fix the key · Tab next action · ← Back, Ctrl+B or Alt+←
```

**Full track at 80x24.** All nine read-back rows (✗ and ! rows sort first), the legend, and What's next collapsed to one row:

<!-- mockup ready-full-track 80x24 -->
```text
Ready                                                      Step 11 of 11 · Ready
✓ Ready to chat — Anthropic · claude-sonnet-5-5 (200K context)
  Test it  [ Say hi in five words.                     ]  [ Say hello ]
  Sends one short message to Anthropic: a few tokens, at your usual rate.
✓ Provider        Anthropic — messages go to Anthropic · key in macOS Keychain
✓ Model           claude-sonnet-5-5 — new chats start with it
✓ tldw server     lab.example.org:8000 — connected
– Search          your Library is searched only when you ask
✓ Tools           2 of 8 on · web search and Watchlists ask first
✓ Spoken replies  OpenAI · tts-1-hd · shimmer
– Dictation       not set up · press Dictate in Console when you want it
✓ Appearance      Nord · dark · startup animation short
✓ Keys            Anthropic: macOS Keychain · OpenAI: OPENAI_API_KEY
✓ saved · – skipped · ! needs attention · ✗ failed
Data:   ~/.local/share/tldw_cli/default_user                     [ Copy ]
Config: ~/.config/tldw_cli/config.toml                           [ Copy ]
▸ What's next (8) — Add a project folder… · Sync a notes folder… · more
[✓] Refresh Anthropic's and OpenAI's model lists when chatbook starts
[ ] Get to know you after setup — a short questionnaire, no AI needed



[ ← Back ]  [ Start chatting ]  [ Add a document ]  [ Write a note ]
Enter start chatting · Tab: test message first · ← Back, Ctrl+B or Alt+←
```

**What's next expanded at 80x24.** It grows over the rows below it, and Esc or ← collapses it:

<!-- mockup ready-whats-next-expanded 80x10 -->
```text
▾ What's next (8) — optional, any time                    Esc or ← collapses
 → Add a project folder…
 → Sync a notes folder…
 → Add another provider…
 → Explore Home
 → Open Settings
 → Web search keys…
 → Tool permissions…
   ↓ 1 more
```

**Full track at 120x40:**

<!-- mockup ready-full-track 120x40 -->
```text
Ready                                                                                              Step 11 of 11 · Ready

✓ Ready to chat — Anthropic · claude-sonnet-5-5 (200K context)
  Test it  [ Say hi in five words.                     ]  [ Say hello ]
  Sends one short message to Anthropic: a few tokens (about 40 in, at most 256 out), at your usual rate.

✓ Provider        Anthropic — your messages go to Anthropic · key in macOS Keychain · also OpenAI (from OPENAI_API_KEY)
✓ Model           claude-sonnet-5-5 — new Console chats start with it (Alt+M, or Ctrl+P then "model", switches)
✓ tldw server     lab.example.org:8000 — connected · Library and notes sync with it · token in macOS Keychain
– Search          your Library is searched only when you ask (all-MiniLM-L6-v2)
✓ Tools           2 of 8 on: Read file, Search in files · web search and Watchlists also available (they ask first)
✓ Spoken replies  OpenAI · tts-1-hd · shimmer (uses your OpenAI key)
– Dictation       not set up · press Dictate in Console when you want it
✓ Appearance      Nord · dark · startup animation short · reduce motion off
✓ Keys            Anthropic: macOS Keychain · OpenAI: OPENAI_API_KEY (not stored) · tldw server token: macOS Keychain
✓ saved · – skipped · ! needs attention · ✗ failed

Data:   ~/.local/share/tldw_cli/default_user                                                   [ Copy ]
Config: ~/.config/tldw_cli/config.toml                                                         [ Copy ]

What's next — optional, any time
 → Add a project folder…
 → Sync a notes folder…
 ▸ More (6): Add another provider · Explore Home · Open Settings · Web search keys · Tool permissions · …

[✓] Refresh Anthropic's and OpenAI's model lists when chatbook starts
[ ] Get to know you after setup — a short questionnaire on this computer, no AI needed











[ ← Back ]  [ Start chatting ]  [ Add your first document ]  [ Write your first note ]
Enter start chatting · Tab: send a test message first · ← Back, Ctrl+B or Alt+←
```

**Copy rules for Ready.**
- The verdict names the provider and model by display name (TASK-34100.9 AC#6), and the context in K or M tokens.
- The Say hello prompt is an editable one-line field, prefilled with "Say hi in five words.". Enter in the field and the [ Say hello ] button both send it. The consent line under it is recomputed whenever the prompt changes, and is never hidden while the button is visible.
- "– Left at their defaults: …" appears only after Finish with defaults, and names the skipped steps in order.
- What's next shows only relevant items (D6). When none is relevant, the list holds only its "More (N)" row, under the heading "What's next — everything is set up; more options:".
- The footer hint is generated for the focused control (TASK-34100.11). It always teaches "← Back, Ctrl+B or Alt+←" (TASK-34100.12 AC#9). It mentions Esc only while Esc does something.
- **Esc never finishes setup** (D4).
- **Replies and provider errors render as plain text,** with markup escaped and control sequences stripped (D5).

### 3.7 Review your setup (re-run dashboard)

<!-- mockup review-your-setup 80x24 -->
```text
Review your setup                                           opened from Settings
Change one area, or press Done. Nothing changes until you press Save in one.
✓ Ready to chat — OpenAI · gpt-4.1-mini (1M context)
  Test it  [ Say hi in five words.                     ]  [ Say hello ]
  Sends one short message to OpenAI. Costs under $0.01 (a few tokens).
 ✓ Provider        OpenAI — messages go to OpenAI · key in config.toml
   + Add another provider…
 ✓ Model           gpt-4.1-mini — new chats start with it
 ✓ tldw server     none — Library and notes stay on this computer
 – Search          only when you ask
 ✓ Tools           2 of 8 on · web search, Watchlists ask first
 ✓ Spoken replies  OpenAI · tts-1-hd · shimmer
 – Dictation       not set up
 ✓ Appearance      Nord · dark · startup animation short
›✓ Keys            OpenAI: plain text in config.toml
   Keys — chatbook can move OpenAI's key to macOS Keychain. Enter opens Keys.
✓ saved · – skipped · ! needs attention · ✗ failed
Data:   ~/.local/share/tldw_cli/default_user                     [ Copy ]
Config: ~/.config/tldw_cli/config.toml                           [ Copy ]



[ Done ]  [ Run the full walkthrough ]
↑↓ browse · Enter change · Tab next action · Esc done (back to Settings)
```

At 120x40:

<!-- mockup review-your-setup 120x40 -->
```text
Review your setup                                                              opened from Settings · Done returns there
Everything setup manages, read from your saved settings. Change one area, or press Done.
Nothing changes until you press Save in one.

✓ Ready to chat — OpenAI · gpt-4.1-mini (1M context)
  Test it  [ Say hi in five words.                     ]  [ Say hello ]
  Sends one short message to OpenAI. Costs under $0.01 (a few tokens: about 40 in, at most 256 out).

 ✓ Provider        OpenAI — messages go to OpenAI · key in config.toml (plain text) · also Anthropic (from env)
   + Add another provider…
 ✓ Model           gpt-4.1-mini (1M context) — new chats start with it
 ✓ tldw server     lab.example.org:8000 — connected · Library and notes sync with it
 – Search          your Library is searched only when you ask (all-MiniLM-L6-v2)
 ✓ Tools           2 of 8 on: Read file, Search in files · web search and Watchlists also available (they ask first)
 ✓ Spoken replies  OpenAI · tts-1-hd · shimmer
 – Dictation       not set up
 ✓ Appearance      Nord · dark · startup animation short · reduce motion off
›✓ Keys            OpenAI: plain text in config.toml · Anthropic: ANTHROPIC_API_KEY (not stored)

   Keys — OpenAI's key is plain text in config.toml: programs running as you, and any backup or synced copy of the
   file, can read it. chatbook can move it to macOS Keychain. Enter opens Keys.

✓ saved · – skipped · ! needs attention · ✗ failed

Data:   ~/.local/share/tldw_cli/default_user                                                                [ Copy ]
Config: ~/.config/tldw_cli/config.toml                                                                      [ Copy ]












[ Done ]  [ Run the full walkthrough ]
↑↓ browse · Enter change · Tab next action · Esc done (back to Settings) · Ctrl+P "Setup: …" opens one area
```

| Element | Copy |
|---|---|
| Title | Review your setup |
| Origin note (right) | opened from Settings · opened from Console · opened from the command palette |
| Intro | Change one area, or press Done. Nothing changes until you press Save in one. |
| Verdict region | Ready's, with Say hello and its cost line (D5). After a Provider or Model change it is recomputed, and the highlight returns to the changed row. |
| Row value when not set up | not set up |
| Detail line (highlighted row, marked ›) | For a "!" row, its reason. Otherwise, what Change would do ("Keys — chatbook can move OpenAI's key to macOS Keychain. Enter opens Keys.") |
| Receipt after a Save | Saved: <area> → <new value> (in place of the intro line, until the next highlight move) |
| Receipt after a Cancel | Nothing changed. |
| Exits | [ Done ] (primary) · [ Run the full walkthrough ] |

The dashboard shows no "last test" history. Revision 1's "last test replied in 0.8 s today" needed persisted state that nothing owns, so it is dropped.

### 3.8 The single-step sheet

- **Size.** The single-step host renders one step in a modal sheet, at most 76x20 and centred. At 80x24 it is effectively full width.
- **Body.** The step's own body is used, with **host capabilities** applied (§4.7). Over Console, `allows_secret_entry` is false, so any key field becomes the Settings round-trip line (D6), and `resume_copy` is empty, so the step promises no resume.
- **Nav.** [ Cancel ] and a Save button whose label says what happens next: "Save", "Save and read this reply", or "Save and start dictating".
- **CSS.** The sheet's styles live in its own per-screen sheet. Wizard step selectors are re-scoped so that a step renders the same in the corridor and in the sheet. Today they are scoped to `FirstRunSetupWizard SetupWizardContainer` (`css/features/_wizards.tcss:1315-1412`).

Over Console, from Speak:

<!-- mockup spoken-replies-sheet-over-console 80x24 -->
```text
Console                                                         OpenAI · gpt-4.1
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






Enter choose · Tab next action · Esc cancel (nothing is saved)
```

### 3.9 Console arrival line

<!-- mockup console-arrival 80x6 -->
```text
· Setup complete — OpenAI · gpt-4.1-mini · streaming on · 1M context.
  Switch models with Alt+M, or Ctrl+P then "model". Your test reply is in
  History: "Setup test · OpenAI · gpt-4.1-mini".
│ you  ▏
```

The line is a dim system row at the top of the new conversation. It is not a toast, it is never sent to the model, and it is not repeated on later launches. The last sentence appears only after a test.

### 3.10 Plain-text setup transcript (Quick, Anthropic, Secret Service)

<!-- mockup plain-cli 80x60 -->
```text
$ tldw-cli setup --plain
chatbook setup, plain text. Type a number or a word and press Enter.
Type ? for help, or quit to stop. Nothing is saved until a step says Saved.
Config: /home/ash/.config/tldw_cli/config.toml

Step 1 of 4: Welcome
What would you like to do first?
  1. Quick setup: connect AI and pick a model (recommended)
  2. Full setup: adds search, tools, voice, appearance, server
  3. Start with my documents or notes, set up AI later
Choice [1]: 1

Step 2 of 4: Connect
Looking for AI servers on this computer... done.
Ready on this computer:
  1. llama.cpp on this computer, 127.0.0.1:9099, 3 models
Or type part of a name to search all 60 providers.
Choice [1]: anthro
1 match:
  1. Anthropic (cloud, needs an API key)
Choice [1]: 1
Anthropic needs an API key. Get one from your Anthropic account.
API key (typing is hidden; Enter alone skips this provider):
Received a key ending 7Qx2 (108 characters).
Checking the key with Anthropic...
OK: Anthropic accepted this key (13 models listed).
Keep this key in:
  1. System keychain (GNOME Keyring) (recommended)
  2. Encrypted in config.toml (a password every time chatbook starts)
  3. Plain text in config.toml (readable by programs running as you)
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
Provider: Anthropic. Your messages go to Anthropic.
tldw server: none. Library and notes stay on this computer.
Keys: Anthropic in the system keychain.
Check model lists online when chatbook starts? [y/N]: n
Data: /home/ash/.local/share/tldw_cli/default_user
Setup complete. Start chatbook with: tldw-cli
Your first message there is the first real test of this setup.
$ echo $?
0
```

- Lines are at most 80 columns, and nothing is redrawn in place.
- With `--glyphs`, "OK:" becomes "✓" and "FAILED:" becomes "✗".
- With `NO_COLOR` or `TERM=dumb`, the output is identical, because it carries no colour to begin with.
- Replies and provider errors that plain mode prints (for example a key check's failure) are stripped of control sequences (D5).

### 3.11 Plain review (a profile with a working provider)

<!-- mockup plain-review 80x20 -->
```text
$ tldw-cli setup --plain
chatbook setup, plain text. Setup is complete for this profile.
Config: /home/ash/.config/tldw_cli/config.toml
Your setup:
  1. Provider: Anthropic. Key in the system keychain.
  2. Model: claude-sonnet-5-5, new chats start with it.
  3. Keys: Anthropic in the system keychain.
  4. tldw server: none. Not changeable in plain mode yet.
  5. Search: only when you ask. Not changeable in plain mode yet.
  6. Tools: 2 of 8 on. Not changeable in plain mode yet.
  7. Spoken replies: not set up. Not changeable in plain mode yet.
  8. Dictation: not set up. Not changeable in plain mode yet.
  9. Appearance: Nord, dark. Not changeable in plain mode yet.
OK: Ready to chat: Anthropic, claude-sonnet-5-5, 200K context.
Change which area? Type its number, or press Enter when done: 2
```

---

## 4. State and persistence model

### 4.1 Tracks and the stable-total invariant

```text
QUICK = (welcome, provider, model, summary)
FULL  = (welcome, provider, model, server, rag, tools, voice, speech, appearance, protect-keys, summary)
DOCS  = (welcome)        # finishes on Open Library; never writes a draft
```

- Step ids stay as they are, except for the new `server`. `notes` leaves (TASK-34100.13). Display titles come from §3.0 through TASK-34100.15's registry.
- `active_step_ids(track)` drops its unused `key_entered` parameter once callers are migrated; today it is ignored (`FRSS:1284-1286`). An unknown track is an error that rejects the draft (§4.4). It no longer falls back to Full (`FRSS:1286`).
- **INV-1 (rule S1).** For any sequence of events after the run leaves Welcome, the active step list is constant until the user returns to Welcome.
  - It is asserted as a **Hypothesis state machine over the pure `SetupSession` reducer** (§4.7), not over `active_step_ids`. That function depends only on the track, so testing it alone would prove nothing.
  - The generated events cover:
    - keys typed and cleared, and an environment variable present;
    - the provider switched, and the provider→model dependency;
    - encryption enabled;
    - both Finish-with-defaults buttons, Back, and a resume from a draft.
  - The TUI corridor and plain mode both drive the same reducer, so one property covers both.

### 4.2 Session modes and the entry predicate

| Mode | Entered from | First screen | Exits |
|---|---|---|---|
| `first_run` | the boot offer; `tldw-cli setup` on a fresh profile; or any setup entry after a completed setup that has no usable provider and no deferral (Welcome is then prefilled) | Welcome | Ready's first-run exits; Open Library (DOCS); Skip |
| `resume` | boot with a valid draft (TASK-34100.10) | the draft's step | as `first_run` |
| `deferred_ai` | any setup entry while `ai_setup_deferred` is set and no usable chat provider exists (D8) | Connect, on Quick | Ready's exits |
| `review` | any setup entry while a usable chat provider exists (D7) | Review your setup | Done → origin |
| `walkthrough` | "Run the full walkthrough" from `review` | Welcome, prefilled (TASK-34100.10) | Ready, with TASK-34100.10 AC#14's exits |
| `single_step` | a dashboard row, a palette command, a Ready next step, a Console sheet | that step | Save or Cancel → the caller |

- **One entry point.** All modes go through TASK-34100.10's `open_setup_wizard(origin, resume, start_step)`, extended with `mode` and `focus_area`. The one-wizard guard covers every mode.
- **One predicate.** `has_usable_chat_provider(config)` (D7) is the only input that chooses between `first_run`, `deferred_ai` and `review`. It lives in the setup core, so plain mode and the Settings button label use the same function.
- **Say hello auto-runs only in `first_run`** (D5).

### 4.3 What each surface writes

| Surface | Writes | Never writes |
|---|---|---|
| Welcome | nothing (the track choice is a draft value); Reduce motion only when ticked, as its own delta | — |
| DOCS finish | `[first_run] setup_completed`, `ai_setup_deferred`, and the model-list consent default | provider, `chat_defaults`, draft |
| A step's Save (any mode) | that step's own keys, as a delta | any other step's keys |
| tldw server Save | through the coordinator: the `[tldw_api]` URL; the token to the keychain (to config.toml only where no keychain exists, as Settings does); the runtime binding. On a failed bind, the prior values are restored | `WIZARD_OWNED_SECTIONS` (`FRSS:1251-1266`) does not gain `tldw_api`, because the coordinator owns that write, not the wizard |
| Connect Save with a keychain choice | the keychain item first, then `credential_source = "keychain"` (no `api_key`), and `[credentials] scope_id` on first use | the key in config.toml; an environment key anywhere |
| Ready | through the finish path: the consent answer, "Get to know you", `last_track`, and clearing `ai_setup_deferred` once a usable provider exists. Say hello writes a conversation (Console's store), never config | — |
| Dashboard | nothing itself | — |
| Plain mode | as the steps' Saves, through the same commit builders | the runtime binding (deferred to F13b) |

### 4.4 Draft compatibility (migration)

Drafts are validated against the current track. `_validated_setup_draft` rejects a draft whose `active_step_id`, or any value's step id, is not on the track (`FRSS:1074-1100`), and `read_setup_draft` then returns `None` (`FRSS:1144-1160`). So a draft saved by today's six-step Quick at Voice or Protect would be **silently dropped** after this change, which breaks rule 2. Therefore:

- **Version bump.** `SETUP_DRAFT_VERSION` goes from 1 to 2 (`FRSS:33`).
- **A one-way migrator, in memory.** It reads version-1 drafts **before** validation:
  - It computes the steps completed under the old order: everything before the old `active_step_id`.
  - It resumes at the first step in the new order that is not in that set. Old Quick at Voice or Protect resumes at Ready; old Full at Voice resumes at tldw server.
  - It drops values for steps no longer on the track (Voice values on Quick, Notes on Full) and **says so** in the resume dialog: "Voice is now set up from the Ready screen, so your earlier Voice choices weren't kept."
- **It never writes at boot.** The draft is read at boot by `setup_recovery_action` (`app.py:3582`). Writing there would be a boot-time config write, which TASK-34100.2 AC#7 forbids, and the setup core must stay free of I/O. The migrated draft is persisted as version 2 at the **next normal checkpoint**, the first time the resumed run saves a draft. If the user never resumes, the version-1 draft stays as it is, and is migrated again on the next read.
- **Unknown tracks fail closed.** Documents-first never writes a draft. An unknown track is rejected, not mapped to Full.
- **The last-used track survives a cleared draft.** `[first_run] last_track = "quick" | "full"` is written by the finish path, so "Run the full walkthrough" can preselect it after the draft is gone.
- **Downgrade.** A version-2 draft read by an older build fails closed, as today: a downgrade loses the draft but never corrupts config. Config written by this spec is also safe for older builds: an older build reads `credential_source = "keychain"` as "no stored key" (D9), and ignores `ai_setup_deferred`, `last_track` and `[credentials] scope_id`.

### 4.5 The settings export format

Deferred with D11. The design notes for reopening it are in D11. Revision 1's format is withdrawn: its `[credentials.*]` table would trip the draft secret-name rule (`FRSS:55`, `_SECRET_FIELD_TOKENS` substring-matches "credential" and "token"), and its strict unknown-key rule would break older exports as `all_tool_gates()` grows.

### 4.6 Keychain namespace (D9)

- **Service and username.** The service is `tldw_chatbook.provider_credentials.<credential_scope_id>`. The username is the provider's canonical key (`normalize_provider_config_key`, `config.py:1580`).
- **The scope id.** `credential_scope_id` is a random id, written once to `[credentials] scope_id` the first time a key is saved to the keychain. ADR-126's adapter remaps it on isolated restore. It never derives from a path, so it survives symlinks, case changes and moves.
- **What config records.** Config records `credential_source = "keychain"` and no `api_key`. Server tokens keep their existing server-credential namespace.
- **Long values on Windows.** Values over the Windows blob limit use the server-credential store's part scheme (D9).

### 4.7 Seams the shape needs

The engineering review found that the shape rests on seams that don't exist yet. They are named here, with the slice that builds each (§10). None is a second implementation of something that exists.

1. **Setup core outside `UI/` (F0).**
   - **What moves.** A new package, `tldw_chatbook/Setup/`, holds:
     - the track definitions;
     - the `SetupSession` reducer: pure, so it takes events and returns state, including cross-step state such as the provider outcomes that Model reads, which today live on the container;
     - pure step-state modules for **Provider and Model**: key-check results, detection results and mutation building, extracted from Textual classes (`FRSW:1169-3310`). Voice and Speech already have state modules;
     - the commit builders;
     - the area registry access, and `has_usable_chat_provider`.
   - **Why it moves.** `UI/Wizards/first_run_setup_state.py` is not import-pure in practice: importing it pulls in Textual through `UI/__init__.py` and `UI/Wizards/__init__.py`, about 0.29 s by `-X importtime`. It becomes a re-export shim. An architecture test asserts that importing the setup core and the plain driver never loads `textual`.
2. **A step-host protocol (F0).** `SetupStepHost` is the only thing a step may call. It provides:
   - `commit(mutations)`;
   - `session`, the cross-step state, seeded from config when a step runs alone;
   - a narrow `services` object: the config owner, the readiness verdict, the credential store and the shared probe. No `app_instance`;
   - `capabilities`: `allows_secret_entry`, `resume_copy` and `nav_kind` (corridor / sheet / plain);
   - `finish(route)`.

   Today steps call `self.wizard.commit_config` (6 sites) and `app_instance` (22 sites), plus container methods (`_first_run_selected_provider_outcomes`, `_first_run_selected_provider_models`, `_first_run_provider_config_preconditions`, `review_provider_setup`, `advance_programmatically`, `handle_next`, `compose_failed_steps`, `note_key_entered`). The corridor container, the single-step sheet, the dashboard and the plain driver each implement the protocol. TASK-34100.1 moves classes but defines no host, so F0 follows it directly.
3. **An exit-route registry out of `app.py` (F0).**
   - **Why.** `_continue_first_run_wizard_result` (`app.py:3982-4133`) admits every exit route, and `app.py` has 28 lines of headroom (§8.2). Today it drops any unknown route, and drops a Settings exit that arrives with `completed` not False (`app.py:4034`).
   - **What moves.** The function moves to its own module with a **typed route table** shared with TASK-34100.15's registry.
   - **What the table admits:** documents-first; Done → origin; the single-step return; Open Settings with `completed=True`; Settings ▸ Workspaces; MCP ▸ Tools; Settings ▸ Splash Screen; Settings ▸ Web Search; and Library ▸ Notes ▸ Add from files.
   - **Test.** An admission test covers every route in the table.
4. **A palette command provider for setup (F0).** It is a new module, because `app_command_providers.py` is at its budget (1,123 of 1,123). It registers every "Setup: …" command in §3.0.
5. **Console probe-turn seams (F9a, a spike first).** All of them go in **new modules**, because `console_chat_controller.py` and `console_chat_store.py` are over budget on dev (§8.2). They are:
   - the `SETUP_PROBE` submission origin, with the AGENT_WAKE precedent. It is excluded from attention notices, prompt history and composer clearing, and from readiness changes except under TASK-34100.2 AC#2's own rules;
   - a turn-scoped probe profile that is never persisted;
   - a viewless streaming observer;
   - a prepare-only token estimate.

   If the spike shows that any of these must touch an over-budget module, the slice stops and reports. It does not raise a ratchet.
6. **A runtime-source coordinator and a shared probe (F3).** The coordinator lives in a new module. It saves, binds and prepares sync in Settings' order, with rollback on a failed bind. The probe is app-free, and uses the runtime client's TLS settings.
7. **A credential store and overlay (F12a).** These are:
   - the keychain credential owner, with a canary probe and an injectable secure-backend predicate for tests (today's detector checks module names, so it would reject an in-memory test backend);
   - the provenance overlay;
   - the post-first-paint resolver;
   - the writer guard in `save_settings_to_cli_config`.
8. **CSS outside the boot sheets.** Boot CSS has about 140 bytes of headroom (§8.2). The CSS for Ready, the dashboard, the sheet and the tldw server step therefore loads with their screens as per-screen sheets, never into the five boot sheets. Wizard step selectors are re-scoped so that a step renders the same in every host (§3.8).

---

## 5. Accessibility

- **Text carries every state.** Every mark is a glyph plus a word ("✓ Ready to chat", "! tldw server … not reachable now"); colour is decoration (PRODUCT.md "Accessibility & Inclusion"; TASK-34100.12 AC#4).
- **A legend explains the glyphs** wherever –, ! or ✗ appears, the tracker included (§3.0).
- **ASCII marks** replace every glyph in ASCII mode (§3.0), and CI covers that mode.
- **Keyboard.** Every action is reachable by Tab, in reading order. Lists follow TASK-34100.11: highlight browses; Enter, Space or a click selects.
  - No new chords: Finish with defaults is a button and a palette command.
  - Back keeps "← Back, Ctrl+B or Alt+←" in every hint (TASK-34100.12 AC#9).
  - Wherever a hint or an arrival line teaches Alt+M, it also teaches "Ctrl+P, then "model"", because macOS Terminal sends "µ" for Option+M unless Option is set as Meta.
- **Esc is never destructive on Ready.** It stops a running test or collapses an expanded list; otherwise it is inert (D4). On the dashboard, Esc is Done, which writes nothing.
- **Focus.**
  - **Ready:** focus starts on the primary exit when the verdict is ✓ Ready; on Say hello when the key is not checked, with a 0.5 s activation guard; and on the fix action when the verdict is ✗ or !.
  - **The local auto-test** never steals focus.
  - **The dashboard** focuses its list, with the deep-linked row highlighted.
  - **The single-step sheet** focuses the step's first control, and returns focus to the invoking control when it closes.
- **Motion.** The Say hello wait shows elapsed seconds as text, not a spinner, and honours Reduce motion. Reduce motion is offered on Welcome, before any animation plays (TASK-34100.13 AC#16).
- **Sizes.** Every screen in §3 fits 80x24 with TASK-34100.12's short tier: Quick and Full Ready, the dashboard, the tldw server step and the single-step sheet. CI checks 80x24, 100x30, 120x40 and 200x60 (TASK-34100.12 AC#16), each in glyph mode and ASCII mode.
- **Screen readers and SSH.** `tldw-cli setup --plain` (D10) is the supported path: line-oriented, no redraws, and words before glyphs. It is found from `--help` and from a line printed before the TUI starts. v1's gap (areas not changeable in plain mode) is stated in D10, and F13b closes it.
- **Cognitive load.** Quick has three decisions: track, provider with key, and model. The key's storage place is a sentence, not a fourth decision, unless the user asks to change it. Ready docks at most three exits on first run. Choice labels lead with their keyword, so truncation never hides it (TASK-34100.12 AC#1).

---

## 6. Migration from today's flow

| Area | Today | After | How |
|---|---|---|---|
| Quick steps | 6: Welcome, Provider, Model, Voice, Protect, Summary (`FRSS:982-989`) | 4 | F5; draft migrator (§4.4) |
| Full steps | 11 with Notes (`FRSS:969-981`) | 11 with tldw server | F6, after TASK-34100.13 removes Notes |
| Welcome | 2 tracks + Restore (`FRSW:6377-6390`) | 3 choices + Reduce motion + Restore | F5 (labels), F7 (third choice) |
| Summary exits | 5 in two docked rows (`FRSW:6640-6658`) | 3 exits + Back on one row, + What's next | F8 |
| Protect on Quick | always (TASK-21148) | a Ready option when a plain-text key was stored in this run | F5 |
| Re-run | Welcome, `rerun=True` (`app_command_providers.py:982`, `app.py:4167`, `SS:31206`) | Review your setup, chosen by the usable-provider predicate | F11, after TASK-34100.10 and F0 |
| Server | Settings only, collapsed (`SS:17648-17681`) | Ready row + next step + Full step + Settings main body | F1–F4, F6 |
| Keys | config.toml, plain or encrypted | keychain-first for new keys; env keys never stored; existing keys untouched until moved | F12a, F12b |
| CLI | `recovery` only (`cli.py:30-33`) | `setup`, `--plain` | F13, F13b |

- **Users with a completed setup** see no change until they re-run setup (the dashboard), or press Speak or Dictate with nothing set up (the sheet).
- **Users mid-setup when they upgrade** resume through the migrator (§4.4), with the one-line note if any choices were dropped.
- **Tests to rewrite, on purpose, in the slices named:**
  - the six-step pins listed in D1, including the two PocketTTS integration tests;
  - `test_summary_five_actions_visible_and_focused_on_full_track` (`Tests/UI/test_first_run_wizard_live_contract.py:2426`), which becomes a three-exits-plus-Back test at 80x24 and 120x40 in F8;
  - the Full-order pins in `Tests/Wizards/test_first_run_setup_state.py:476-506`, in F6.

  Each rewritten test is RED-verified: it fails on the pre-change code and passes on the new.
- **User Guide.** `Docs/User_Guide/First_Run_Setup.md` is rewritten across the slices:
  - the two-tracks section (`:51-59`) becomes three choices;
  - the step table (`:61-73`) follows D2;
  - "Running it again" (`:120-127`) describes the dashboard;
  - a new "Setting up from the command line" section covers D10.

  The page's "Verified against" header (`:3`) and its italic stamps go, per CLAUDE.md (owned by TASK-34100.15 AC#7).
- **Backlog edits made on approval, before the affected siblings start (all are To Do):**
  - task-28019 is narrowed to its AC#3, and TASK-34100.10's coordination note is corrected (§0.4);
  - **TASK-34100.10 AC#1** is narrowed to the entry contract, prefill, "(current)" marks and Cancel-to-origin, so .10 does not build a header body that F11 replaces (D7);
  - **TASK-34100.10 AC#14**'s "New note" becomes "Write a note" (D4);
  - **TASK-34100.6 AC#21** is amended so that the environment-key choice and the storage choice are one control, with the environment as the default (D9);
  - **TASK-34100.15 AC#3**'s example tracker labels change "Protect keys" to "Keys", and add "tldw server" (§3.0);
  - TASK-21148 gets a note that AC#5's mechanism is superseded by rule S1;
  - TASK-34100.13 AC#15 points its time line at D1's copy.

---

## 7. Preserve: what each change must not break (report §5.5)

| Change | Secrets never written unless saved | Model-list consent asked once | Summary read back from disk | Restore a backup reachable from Welcome | Get started card catches Skip and Exit | Other report §5.5 items at risk |
|---|---|---|---|---|---|---|
| D1 Quick 4 steps | Encryption still only on an explicit action | unchanged on Ready | unchanged | kept, with the explanatory line | unchanged exits | Exit and Skip dialogs (Welcome's Skip unchanged) |
| D2 Full order + Finish with defaults | untouched steps write nothing | unchanged | skipped steps read back as "Left at their defaults" | n/a | n/a | delta-aware writes; the Parakeet install review |
| D3 tldw server | the token goes to the keyring; a failed bind restores the old values; the template placeholder never shows as a token | n/a | the row comes from `RuntimePolicyContext` | n/a | n/a | specific connection errors; ADR-033's failed commit leaves the old binding |
| D4 Ready | n/a | consent box offered once, cloud only | rows force-reloaded from disk | n/a | ✗ verdict → Go to Console → the card | Library exits visible; the typed-model "start later" path never blocked |
| D5 Say hello | no key in any persisted field or log; the probe profile is never persisted | n/a | the verdict reflects Console's readiness after the test | n/a | n/a | streaming end to end, with a non-streaming fallback; local detection |
| D6 What's next, arrival, sheets | the Console sheet never accepts a key | n/a | the list is computed from config | n/a | n/a | the Voice step's strengths (the same step state) |
| D7 Dashboard | only a step's Save writes | never re-asked | every row force-reloaded | n/a | Done returns to the origin | resume and preview safety; the one-wizard guard |
| D8 Documents first | writes nothing but completion, the deferral flag and the consent default | recorded by the finish path, never a modal | n/a | kept on the same screen | the card shows when Dee opens Console | Skip stays one gesture |
| D9 Keychain-first | the keychain write happens only on Save; no writer persists an overlay value; drafts refuse secret fields | n/a | the Keys row comes from the resolved source | n/a | n/a | env keys never stored; ADR-029 private files |
| D10 Plain CLI | getpass or env only; never argv; never echoed; refuses a terminal that can't hide input | asked once, default No | Ready lines re-read from disk | the recovery hint is printed | n/a | robust basics (paths with spaces, bounded waits) |
| D11 Deferred | n/a | n/a | n/a | unaffected | n/a | none |

---

## 8. Test and verification plan

### 8.1 Automated

Every test below is **RED-verified**: it fails on the pre-change code, then passes on the slice.

- **Pure state, in the setup core.** Use Hypothesis where the project already does. Cover:
  - INV-1 as a Hypothesis **state machine over `SetupSession`**, shared by the corridor and plain mode (§4.1);
  - the track lists for each mode, and the entry predicate's table (§4.2) over generated configs, template values included;
  - the draft migrator for every old step id on both tracks, and that it **never writes during the boot read** (§4.4);
  - the Ready exit set as a function of (mode, verdict, origin, width): never more than three on first run, and never more than four on re-run;
  - the verdict region's state as a function of (shared verdict, key provability, test outcome), so that the region never shows two marks;
  - What's next relevance from config, including the file-tool condition;
  - the consent line's wording against the estimate and price thresholds;
  - the reply cap per catalog model class (ordinary, reasoning effort, thinking budget);
  - CLI argument parsing: key-like flags are refused with exit 2, and `--config X setup` parses.
- **Import purity.** Importing the setup core and the plain driver does not load `textual` (checked through `sys.modules` in a fresh subprocess).
- **Plain mode.**
  - Golden transcripts for line content against a fake terminal.
  - A **pty-based test** that a typed key is not echoed. A golden file cannot prove that.
  - A test that `GetPassWarning` exits 2.
- **Invariant.** Pressing only Next, Done or either Finish-with-defaults button over populated configs leaves config.toml byte-identical, apart from first-run bookkeeping. This is TASK-34100.9 AC#1's test, extended to:
  - the dashboard (open, then Done);
  - the DOCS finish (only `[first_run]` and `[model_catalog]` change);
  - a Cancel in every single step.
- **Widget tests (Pilot, real stylesheet):**
  - Welcome's three choices, tracker updates and the Reduce motion row;
  - DOCS → Library Import, with no consent modal mounted;
  - Ready in states R-cloud, R-key-not-checked, R-local, replied, no credit, ✗ preflight, ✗ test and not set up, on both tracks. Each runs at 80x24, 100x30, 120x40 and 200x60, in glyph and ASCII mode. The checks: the verdict region, the legend whenever –, ! or ✗ shows, at most three exits plus Back on one row, both Library exits and at least 8 read-back rows inside the visible region;
  - Esc on Ready never finishes setup, with and without a running test;
  - the dashboard: Change → Save returns with a receipt, a refreshed row and a recomputed verdict; Change → Cancel writes nothing; a deep link highlights its row;
  - the single-step sheet at 80x24, with `allows_secret_entry` false over Console;
  - the tldw server step against local HTTP fixtures for unreachable, 401, not-tldw, TLS untrusted and success, and a **failed bind that restores `[tldw_api]` on disk**.
- **Exit routes.** Every route in F0's table is admitted, and an unknown route is refused with a diagnostic.
- **Say hello.** Run through the real `ConsoleRuntime` and controller, with the provider gateway faked at its protocol boundary (`ConsoleProviderGatewayProtocol`, `console_chat_controller.py:4134`). Check that:
  - a saved conversation is created and reused for the same provider and model;
  - the probe sends no tools, no retrieval and no history;
  - the cap is applied and **not persisted**, so the next normal turn has the session's own cap;
  - no hidden-turn notice or nav badge fires;
  - dispatched input tokens never exceed the estimate shown;
  - failures map to D5's table, and a probe-only 400 leaves readiness unchanged;
  - a cloud provider never auto-sends;
  - an allowlisted local provider auto-sends once, in `first_run` only;
  - a `custom` loopback endpoint, an endpoint with a key, and an Ollama model that is not loaded never auto-send;
  - every exit stops a running test without a notice;
  - replies with ANSI/OSC sequences and Rich markup render as plain text.
- **Keychain.** Run the credential owner against keyring's fail backend (unavailable), a locked fake (interaction not allowed) and an in-memory backend, which the injectable secure-backend predicate admits. Check that:
  - the save order holds: a config-write failure deletes the item, and a keychain failure never writes plain text;
  - the four resolution states appear, and a timeout is never cached;
  - TTL refresh works across **two processes** (a key rotated in one is used by the other at its next send);
  - a 401 invalidates the entry;
  - long values are split into parts;
  - an environment value never reaches any store, whatever the selection.
- **Census tests.**
  - The **reader census:** every provider-credential reader resolves through the overlay.
  - The **writer census:** no writer persists an `api_key` whose provenance is the overlay. A property test checks that load → save leaves no secret on disk for keychain-sourced providers.
- **Architecture.**
  - The tldw server step and Settings import the same coordinator, and there is no second writer of `[tldw_api]`. This is a grep-based guard, like the existing ownership guards.
  - New setup CSS is not in the boot sheets.

### 8.2 Ratchets

No module-size or ADR-097 ratchet rises. The engineering review measured the starting point in this worktree on 2026-10-03:

| Ratchet | Today | Headroom | What this spec does about it |
|---|---|---|---|
| `FirstRunSetupWizard.py` (`Tests/Architecture/test_module_size_ratchet.py:128`) | 9,866 / 9,866 | 0 | F2 waits for TASK-34100.1. New screens go in their own modules |
| `app_command_providers.py` (`:93`) | 1,123 / 1,123 | 0 | Palette commands go in a new provider module (F0) |
| `app.py` (`:75`) | 5,684 / 5,712 | 28 | Exit-route admission moves out of `app.py` (F0) before F7, F8 and F11 add routes |
| `Chat/console_chat_controller.py` (`:97`) | 31,653 / 29,367 | **over by 2,286 (red on dev)** | F9's seams go in new modules. F9 must not grow it |
| `Chat/console_chat_store.py` (`:98`) | 22,491 / 22,344 | **over by 147 (red on dev)** | as above |
| Boot CSS (`MAX_BOOT_PARSED_CSS_BYTES`, `Tests/Performance/test_boot_css_byte_budget.py:117`) | 607,951 / 608,090 bytes | about 140 bytes | Setup CSS loads per screen, never into the boot sheets (§4.7) |
| Boot-import and UI-ready censuses (`Tests/Performance/boot_budget_snapshots/`) | `keyring` absent from both | — | `keyring` is imported after first paint, and only when a keychain credential exists (D9) |

The two Console rows are already red on dev. This spec does not fix them, but it must not make them worse, so F9a's spike reports and stops if any seam needs to touch them.

### 8.3 What does not count as evidence

- A green Pilot run with a faked gateway is not evidence that Say hello works with a provider.
- A Ready screen reading ✓ is not evidence that a chat works.
- A golden transcript is not evidence that a key is hidden.

Live runs (§8.4) are required, per `backlog/docs/lessons-testing-evidence.md` and `lessons-live-verification.md`.

### 8.4 Live verification

Every slice is verified live on a fresh isolated profile, with `TLDW_CONFIG_PATH` and `HOME` set, real providers and a real llama.cpp, and no mock servers.

| Scenario | Pass when |
|---|---|
| Sam: OpenAI pasted, Quick, Say hello | Ready ✓ within 2 minutes hands-on from Welcome; the reply streams; Start chatting opens a new chat with the arrival line quoting the test |
| An OpenAI account with no credit | the test reads "accepted the key but refused the message" (HTTP 429 `insufficient_quota`, or 402) with billing copy |
| A reasoning model and an Anthropic model with extended thinking | the test replies, or reports "used its whole test allowance thinking"; never a self-caused 400 |
| Jo B: real llama-server on a non-:8080 port | the auto-test replies; tldw server "none"; about 6 keys from Welcome to a reply; no outbound model request is made, checked with a network monitor |
| Jo B with Ollama, model not loaded | no auto-run; the press loads it, and the cost line names the size |
| A LiteLLM proxy on loopback | no auto-run; the cost line names the forwarding risk |
| Jo A: nothing running | ✗ with Check again; start the server; Check again → ✓ |
| An expired OpenRouter key | "✓ Set up — key not checked yet" before the test; the test fails with the provider's message and [ Fix key ] |
| Dee | Library Import in 3 keys; no modal; config diff only in `[first_run]` and `[model_catalog]`; a later "Run setup" opens Quick at Connect |
| Riley re-run | palette → change model in ≤ 6 actions (a palette search counts as one); config diff only `chat_defaults.model` (+ provider model); the env key is never written anywhere |
| Real tldw_server | step success; then 401 with a wrong token, a wrong port (not tldw), the server stopped (unreachable), and a forced bind failure (config.toml restored); Settings shows the same binding |
| macOS Keychain, local session | key stored; relaunch through both entry points; readiness ✓; config.toml has no key; the "python3 wants to use…" prompt answered both ways, including "Deny" (the step stays put, no plain text) |
| macOS over SSH (locked login keychain) | the canary fails; keychain option disabled with its reason; plain-text or env default |
| Windows (Credential Manager) | key stored and resolved; a long JSON credential split into parts, or refused with its reason |
| Desktop Linux, GNOME Keyring unlocked and locked | stored and resolved when unlocked; "Waiting for … — answer its prompt" when locked, and Check again works without a restart |
| Headless Linux (container, no Secret Service) | keychain option disabled with its reason; plain-text fallback; plain mode exits 0 |
| Two chatbook instances, key rotated in one | the other uses the new key at its next send |
| VoiceOver on macOS Terminal, NVDA on Windows Terminal or Orca on GNOME Terminal, with `--plain` | every prompt read in order; the key prompt silent; "Received a key ending …" read; exit code 0 |
| One session with a person who uses a screen reader daily | completes plain Quick unaided; their notes are recorded in the slice's Implementation Notes |

Evidence goes in each follow-up's Implementation Notes, never in the User Guide.

---

## 9. Rollout

- **Order.**
  - Cheap and independent slices come first.
  - Then the enablers (F0) right after TASK-34100.1.
  - Then the shape changes, after the sibling subtasks they stand on.
  - New surfaces come last.

  F1 and F3 depend on nothing in the programme and can land in Wave A.
- **No feature flags.** Each slice is complete and shippable on its own. Where a later slice changes copy that an earlier one shipped, the later slice owns the copy change and its test.
- **A release note for every user-visible slice:**
  - F5: "Quick setup is now four steps";
  - F7;
  - F9b: "Setup can send a test message";
  - F11;
  - F12a: "New API keys are stored in your system keychain";
  - F13.

---

## 10. Phased implementation: proposed follow-up tasks

These are filed under TASK-34100 **only after approval**. Their IDs are assigned against `origin/dev` at filing time (lessons-backlog-hygiene). Every task carries the programme's ACs:
- live verification on a fresh isolated profile through the real app, with real providers and a real llama.cpp server, never mock servers, and the evidence in Implementation Notes;
- new behaviour covered by RED-verified tests (they fail on the pre-change code), with no module-size or ADR-097 ratchet raised;
- the matching `Docs/User_Guide/` page updated, with no "Verified against" stamps.

| # | Slice | Decision | Size | Depends on | Notes |
|---|---|---|---|---|---|
| F1 | Settings ▸ Overview: move "Switch Source / Server" into the main body as a tldw server row | D3.5 | S | — | Cheapest; independent |
| F3 | One runtime-source coordinator, with rollback on a failed bind, and a shared app-free candidate probe (unreachable / 401 / not tldw / TLS / policy) using the bound client's TLS settings. Used by `ServerSwitchModal` and Settings. First task: record exactly what Sync v2 preparation sends | D3.4 | M | — | New module. Settings' behaviour is unchanged except for clearer probe copy and the rollback |
| F0 | Enablers: the setup core in `tldw_chatbook/Setup/` (tracks, `SetupSession` reducer, pure Provider/Model state, commit builders, `has_usable_chat_provider`); the `SetupStepHost` protocol with capabilities; the exit-route registry moved out of `app.py`; the setup palette provider module; the import-purity test | §4.7 (1–4) | L | .1 | No user-visible change. Unblocks F2, F7, F8, F11, F13 |
| F2 | Summary/Ready: an always-present tldw server row from `RuntimePolicyContext` | D3.1 | S | .1 | Lands on the extracted Summary, so no FRSW growth |
| F4 | Ready next step "Connect a tldw server… (if you run one)" | D3.2 | S | F2, F3 | |
| F8 | Ready layout: the four-state verdict region, one-row nav with ≤ 3 exits, What's next with `→` and its short-tier collapse, the legend, Data and Config, Esc inert, short exit labels | D4, D6.1 | M | .9 AC#10, .12, F0 | Comes **before** F5, so Quick never loses its encryption offer. Rewrites the five-actions live-contract test |
| F5 | Quick = 4 steps; Voice → Full; Protect off Quick; Ready's "Encrypt saved keys…" and "Hear replies aloud…"; draft migrator v2 (in memory, persisted at the next checkpoint); `last_track`; rewrite the six-step pins; Welcome copy and the Reduce motion row | D1 | M | .1, .4, .8, .9, F0, F8 | Supersedes TASK-21148 AC#5 (note added) |
| F7 | Documents-first Welcome choice through the single finish path; `ai_setup_deferred` and the `deferred_ai` mode | D8 | M | .10 (finish path, consent default), F0 | Takes task-28019 AC#1–#2; task-28019 keeps only AC#3 |
| F11 | Single-step host, the Review your setup dashboard with the verdict region, the entry predicate, Add another provider, palette commands, row deep links, entry labels | D7 | L | .10, .9, .15, F0, F8 | Replaces the header body that .10 AC#1 no longer builds |
| F6 | Full = 11 with the tldw server step; Save and finish with defaults on Model; Finish with defaults on 4–10; final order; the "Keys" title | D2, D3.3 | L | F3, F0, .13, .15 | |
| F9a | Console probe-turn seams, as a spike first: the `SETUP_PROBE` origin, the turn-scoped probe profile, the viewless streaming observer, the prepare-only estimate. All in new modules | D5 | M | .2, .5 AC#4/#7/#8 | Stops and reports if a seam needs an over-budget Console module |
| F9b | Say hello UI on Ready and the dashboard: consent line, model-aware cap, the auto-run allowlist, the failure table, plain-text rendering, a new chat on Start chatting | D5 | L | F9a, F8, F11, .3 | |
| F10 | Arrival line content; Speak and Dictate setup sheets in Console, with host capabilities | D6.2–3 | M | .5 AC#11, .8, .13, F11 | First task records today's Speak and Dictate behaviour |
| F12a | Keychain credential store: canary probe, scope id, provenance overlay, post-first-paint resolution with four states, writer guard, save order, Connect and Settings storage line, environment default (.6 AC#21 amended) | D9 | L | .4, .6, ADR-012/029 amendments approved | The server token goes keychain-only when the keychain is secure (ADR-033 note) |
| F12b | Move to keychain (from plain and `enc:`), orphan handling, ADR-126 backup adapter and scope-id remap, the Windows part scheme | D9 | M | F12a | |
| F13 | `tldw-cli setup` and `--plain` v1: Welcome, Connect, Model, key storage, the app-free verdict, plain review of Provider/Model/Keys, `quit`, getpass hardening, discoverability | D10 | L | F0, F5; F12a optional | No Say hello, no server step |
| F13b | Plain renderers for tldw server (save now, bind at next launch through the coordinator) and the other Full steps | D10 | M | F13, F3, F6 | Filed together with F13, so v1's accessibility gap has an owner |
| — | Non-interactive setup and settings export | D11 | — | — | **Not filed** (deferred). Reopen condition in D11 |

**Order of work.**
1. **F1, F3.** Independent; Wave A.
2. **TASK-34100.1 lands.**
3. **F0.**
4. **F2, F4.**
5. **F8, then F5.**
6. **F7, F11, F6.**
7. **F9a (spike), then F9b.**
8. **F10.**
9. **F12a, then F12b.** F12a can run in parallel from step 4 onward, once its ADR amendments are approved.
10. **F13, then F13b.**

Each slice can merge on its own once F0 exists, provided it states its CSS bytes and module-size deltas.

---

## 11. Risks

| # | Risk | Mitigation |
|---|---|---|
| K1 | Say hello costs more than users expect (reasoning models, expensive models) | The cost line is computed from the probe request and leads with money; the cap is model-aware; cloud never auto-runs; dispatched tokens are asserted against the estimate |
| K2 | Running a Console turn from the wizard couples setup to Console internals | The viewless runtime exists. F9a is a spike in new modules, through the public submit path and a new origin; it never uses controller internals |
| K3 | Keychain backends prompt, hang or are locked (the macOS access prompt, a locked Secret Service, SSH) | A canary probe in a worker; four resolution states; timeouts never cached; Check again; never on the UI loop or under the config write lock |
| K4 | A key ends up in two stores, or in none | A fixed save order with compensation; a move removes nothing until the read-back matches; the census tests |
| K5 | A new credential source reopens "readiness and spend disagree" | One overlay; the reader census; both allowlists gain "keychain" in one slice |
| K6 | Reversing TASK-21148 brings back mid-flight count changes | INV-1 as a state-machine property over `SetupSession`; rule S2 |
| K7 | Test churn from the six-step and five-exit pins hides real regressions | Rewrites are RED-verified; pins are rewritten, not deleted |
| K8 | Setup's tldw server step prepares Sync v2 as a side effect | Disclosed before the choice; F3 records what is sent; owner ruling Q5 |
| K9 | The shared probe misreads a reverse proxy as "not a tldw server", or the auth check changes server state | Identity from health and docs-info only; live verification behind a proxy; "Save anyway" for unreachable or untested servers; the POST probe is replaced if a read endpoint exists |
| K10 | Plain mode drifts from the TUI | One setup core and reducer; one INV-1 property; golden transcripts; one commit path |
| K11 | A writer copies a keychain key into config.toml | The provenance overlay; the writer guard; the writer census and a load → save property test |
| K12 | CLI setup and a running TUI write the same config | Atomic writes; warn and confirm (the instance check is detection-only by the owner's rule) |
| K13 | Ready or the dashboard overflow 80x24 if TASK-34100.12 slips | F8 and F11 are sequenced after .12; the size matrix in CI, glyph and ASCII |
| K14 | Documents-first users never discover AI setup | `ai_setup_deferred` sends every setup entry to Connect; the Get started card; the palette's Setup commands |
| K15 | Draft migration bugs strand a mid-setup user | Migrator tests for every old step id; a fail-closed fallback that offers "Start over", never a crash; no boot-time write |
| K16 | The Console modules F9 needs are already over budget | Seams in new modules; F9a stops and reports rather than raising a ratchet |
| K17 | Setup CSS breaks the boot CSS budget (about 140 bytes of headroom) | Per-screen sheets only (§4.7) |
| K18 | An older build meets `credential_source = "keychain"` | It reads "no stored key" and fails closed; orphaned items are offered for use or removal (D9) |
| K19 | The local auto-run evicts a loaded model or bills through a proxy | The positive allowlist; no generic endpoints; no keyed endpoints; only when nothing needs loading; first run only |

---

## 12. Proposed ADR changes (drafts; not applied)

ADR numbers collide in this repository. There are three files numbered 029, three numbered 033 and two numbered 097. So every reference below names the file, and a new ADR's number is assigned at filing and re-verified at merge (`backlog/docs/lessons-backlog-hygiene.md:762-786`). The files concerned are:
- `012-provider-credential-settings-boundary.md`;
- `029-local-private-data-boundary.md`;
- `033-application-session-state-ownership.md`;
- `076-library-lifecycle-progressive-disclosure.md`;
- `097-boot-budget-ratchets.md`;
- `126-complete-local-backup-and-recovery.md`.

### 12.1 New: ADR-NNN — First-run setup shape and surfaces

> **Status:** Proposed (TASK-34100.17). **Date:** <approval date>.
>
> **Decision.** One setup owner serves four surfaces: the first-run corridor, the re-run dashboard ("Review your setup"), single-step sheets, and `tldw-cli setup --plain`. The owner is the setup core in `tldw_chatbook/Setup/` (tracks, the `SetupSession` reducer, the step states and the commit builders), hosted through the `SetupStepHost` protocol. Single-step sheets serve dashboard Change, palette commands, Ready next steps and Console Speak/Dictate. Every surface uses the same track definitions, step states, commit builders and readiness verdict, and the same `has_usable_chat_provider` predicate to choose its first screen.
>
> 1. **Tracks.** Quick = Welcome, Connect, Model, Ready. Full = Welcome, Connect, Model, tldw server, Search, Tools, Spoken replies, Dictation, Appearance, Keys, Ready. Documents-first finishes on Welcome and records that AI setup was deferred.
> 2. **Stable total.** Once the user leaves Welcome, the step list does not change until they return to it. Conditional offers are options on Ready, never steps. (This supersedes TASK-21148 AC#5's "Protect always on Quick" and keeps its guarantee.)
> 3. **Optional steps.** Every optional step defaults to "not now" and writes nothing when untouched. "Finish with defaults" is offered on Full from Model onward.
> 4. **The verdict.** Ready's verdict is computed through Console's shared readiness verdict, never a wizard copy, combined with this run's test outcome, with one mark per region. First run docks at most three exits and keeps both Library exits visible.
> 5. **The test message.** The optional test is a Console probe turn: Console's admission and dispatch, a turn-scoped profile with no tools, retrieval, history or persona, and a saved conversation. It runs automatically only on first run, for allowlisted local engines on loopback, with no key and nothing to load. For every other provider it runs only on an explicit press, with the computed cost on screen. Start chatting opens a new conversation.
> 6. **Re-runs.** A setup entry with a usable chat provider opens the dashboard. Only a step's own Save writes. Done returns to the origin. Console readiness links go to the control that fixes each reason.
> 7. **Secrets.** Keys are never read from argv by any setup surface. An environment key is never stored. A keychain-sourced key is never written to config.toml by any writer.
> 8. **Not offered.** Setup does not offer a Welcome router, a master tool switch, an embedding-model picker, a multi-tick provider list, per-provider base-URL overrides, an undo-this-session ledger, or an inline TLS-verification toggle. Non-interactive setup is deferred until users ask.
>
> **Consequences.** The wizard spec's "re-run and first-run are one code path" (2026-07-28 §1) is replaced by "one setup core, several surfaces". ADR-076's "the only startup/setup owner" now refers to this owner and all its surfaces.

### 12.2 `012-provider-credential-settings-boundary.md` — amendment: the keychain as a credential store

> **Amendment <date> (TASK-34100.17, D9).** The Consequences sentence "This ADR does not introduce encrypted credential storage, keyring migration…" is narrowed.
>
> - **Where keys may live.** Provider API keys may be stored in the OS keychain through one credential-store owner. Such a key is recorded as `credential_source = "keychain"`, with no `api_key` in config.toml, under a persisted `credential_scope_id`.
> - **One choice, the environment first.** Settings ▸ Providers & Models and setup offer the same storage choice (keychain, encrypted config, plain config). Where an environment variable exists, using it unstored is the default. The keychain is the default where a canary probe succeeds.
> - **Precedence keeps its shape.** A stored credential (keychain or config) outranks the environment variable, which outranks legacy `[API]`.
> - **Resolution.** Keychain values are resolved after first paint, off the UI loop, into one provenance overlay that every reader uses, so spend and readiness read one value. No writer persists an overlay value.
> - **Moves.** Existing keys move only on an explicit user action.
> - **Unchanged.** Console still never accepts or displays a key (the model-configuration redesign's owner ruling, task-33008 AC#4).

### 12.3 `029-local-private-data-boundary.md` — amendment: the OS keychain inside the private-data boundary

> **Amendment <date> (TASK-34100.17, D9).** The rejected alternative "Replace config TOML with keyring/encrypted storage" is revisited, for provider keys only.
>
> - **Permitted location.** The OS keychain becomes a permitted location for provider credentials.
> - **Which keychains count.** Only secure backends that pass a write/read/delete canary count, from the `runtime_policy/server_credentials.py` allowlist: macOS, Windows, Secret Service, libsecret and KWallet. Fail, null, plaintext and file backends never count. Where no secure backend exists, keys fall back to the owner-only config.toml, as the user's explicit choice.
> - **Reads.** Keychain reads run off the UI loop and outside the config write lock, with a timeout and a short TTL. A timeout is never cached.
> - **Never copied out.** A keychain value is never logged, never written to a draft or diagnostic, and never written back to config.toml by any writer.
> - **Threat model.** The keychain protects against copying, syncing, backing up and sharing config.toml. It does not protect against software running as the user.
> - **Ownership.** `config.toml` remains the sole persistence owner for configuration (this ADR's Decision), and the credential-store owner is the sole writer of provider keys in the keychain.

### 12.4 `076-library-lifecycle-progressive-disclosure.md` — amendment: more setup surfaces, one owner

> **Amendment <date> (TASK-34100.17).**
>
> - **One owner.** "The existing application first-run wizard remains the only startup/setup owner" now means the setup owner of ADR-NNN. That owner presents a first-run corridor, a re-run dashboard, single-step sheets and a plain-text CLI, all on one setup core. None of them is a second onboarding wizard.
> - **Library unchanged.** Library still reads only the startup admission fact, and never writes or reinterprets setup completion. The documents-first exit lands on Library Import through the existing exit route, and Library's starter lifecycle is unchanged.

### 12.5 `033-application-session-state-ownership.md` — confirm, with one clarifying amendment

> **Confirmed.** `RuntimePolicyContext` stays the sole authority for the active runtime source.
>
> **Amendment <date> (TASK-34100.17, D3).**
>
> - **One coordinator.** "Settings saves the changed URL and token … and passes it to one app-level coordinator" applies to every setup surface that binds: Settings, the setup tldw server step, and Ready's "Connect a tldw server…". They call the same coordinator, and none of them writes `[tldw_api]` or the binding itself.
> - **Rollback.** The coordinator restores the prior `[tldw_api]` values, and removes any token it stored, when the bind fails. "A failed commit leaves every observer on the old binding" therefore holds on disk as well.
> - **The token.** With a secure OS keychain, the server token is stored only there, with no config.toml copy. Without one, the existing config.toml fallback stands.
> - **Plain mode.** `tldw-cli setup --plain` has no app, so it saves a server for activation at the next launch through the same coordinator (F13b). It never binds by itself.
> - **Sync.** Binding a server prepares the Sync v2 profile on every surface alike, disclosed before the choice (owner ruling Q5).

### 12.6 `126-complete-local-backup-and-recovery.md` — confirm, with one amendment

> **Confirmed.** Restore stays reachable from first-run setup (Decision 1: "exposed through canonical F9 Settings, first-run setup, and a dependency-light pre-bootstrap recovery launcher"). It stays on Welcome, and plain mode names the recovery command.
>
> **Amendment <date> (TASK-34100.17, D9).** Provider keys held in the OS keychain are Chatbook-owned keyring values under Decisions 5 and 8: excluded from portable export by default, and captured in local rollback archives through a typed owner adapter. Under Decision 9, isolated restore remaps the persisted `credential_scope_id`, so a restored profile never reads or overwrites another profile's keys.

---

## 13. Current-behaviour citations (`origin/dev @ fcebe51a09`)

| Claim | Where |
|---|---|
| Quick track is Welcome, Provider, Model, Voice, Protect, Summary | `FRSS:982-989` |
| Full track has 11 steps, including Notes | `FRSS:969-981` |
| `active_step_ids` ignores `key_entered`; an unknown track falls back to Full; Protect always present (TASK-21148) | `FRSS:1271-1286` |
| Step ids (`protect-keys`, …) and tracker titles (RAG, Speech, Style, Protect, Summary) | `FRSS:930-954` |
| Step ids that key drafts | `FRSS:991-1002` |
| Draft version is 1; a draft off the current track is rejected | `FRSS:33`; `FRSS:1074-1100`, `:1144-1160` |
| Draft secret-name rule (substring match on api_key, credential, password, token, secret); credential sources; credential size cap | `FRSS:53-56`, `:1034-1036` |
| `WIZARD_OWNED_SECTIONS` has no `tldw_api` | `FRSS:1251-1266` |
| Setup state module has no Textual import at file level (but its package `__init__`s load Textual) | `FRSS:15-25`; `UI/__init__.py`; `UI/Wizards/__init__.py` |
| Summary primary is "start_chatting" when Provider and Default model rows are configured and no probe failed | `FRSS:641-665`; `FRSW:6807-6818` |
| Summary rows builder | `FRSS:1838` |
| Env-key notice; wizard offer rule | `FRSS:873-911`; `FRSS:861-869` |
| Welcome: title, pitch, time copy, two tracks, Restore | `FRSW:6355-6390`; handler `:6392-6396` |
| Protect copy names a Skip and a Settings page; keyless state | `FRSW:6427-6436`; `:6441-6470` |
| Provider and Model logic in Textual classes | `FRSW:1169-3310` |
| Voice preselects PocketTTS at `127.0.0.1:8765` | `UI/Wizards/first_run_voice_step.py:123-127`; `first_run_voice_step_state.py:36` |
| Summary reads config with `force_reload=True` | `FRSW:6683-6692` |
| Summary consent box offered only while unanswered; its commit | `FRSW:6608-6620`, `:6755-6777`, `:6930-6955` |
| Summary's five exits in two docked rows; routing | `FRSW:6640-6658`; `:6871-6899` |
| Skip path records completion only (no consent) | `FRSW:8987-9008` |
| The wizard's app-config mirror (a whole-settings writer) | `FRSW:9235` |
| Single finish path | `FRSW:9293-9322` |
| First-chat handoff staging | `FRSW:9327` |
| Notes exit sentinel | `FRSW:407-413` |
| Skip dialog copy | `FRSW:9744-9760` |
| Wizard CSS scoped to the corridor container | `css/features/_wizards.tcss:1315-1412` |
| Re-run entry points (`rerun=True`) | `app_command_providers.py:975-988`; `app.py:4148-4171`; `SS:31197-31209` |
| Exit-result admission; a Settings exit must arrive with `completed is False` | `app.py:3982-4133`; `app.py:4034` |
| Draft read at boot | `app.py:3582` (`setup_recovery_action`) |
| App config override assignment (a whole-settings writer) | `app.py:1985` |
| Settings' setup button lives in Diagnostics ("Run Setup Wizard") | `SS:23301-23305` |
| No `tldw_api` reference under `UI/Wizards/` | `grep -rln tldw_api tldw_chatbook/UI/Wizards/` → no match |
| "Switch Source / Server" inside the collapsed "Advanced / Diagnostics" | `SS:17648-17681`; collapsed default `SS:3462` |
| Settings switch flow: save before bind, rebind, keyring, Sync v2 | `SS:26477-26623` (save `:26529-26537`; keyring `:26584-26597`; sync `:26599-26620`) |
| Sync v2 local-first preparation runs a server dry run with the display name | `Sync_Interop/sync_scope_service.py:349-410` |
| `ServerSwitchModal` compose, the Sync v2 note, and the probe | `Widgets/Settings_Widgets/server_switch_modal.py:93-156`, `:122-126`, `:167-193`, `:211-256` |
| `[tldw_api]` template; placeholder token; screening | `config.py:4063-4066`; `config.py:1510`; `config.py:1558-1577` |
| Runtime source state fields | `runtime_policy/types.py:62-70` |
| Server capability discovery (health, readiness, docs info) | `runtime_policy/server_capabilities.py:66-75` |
| `handle_runtime_backend_changed` | `app.py:1952-1975` |
| Persisted credential sources; unknown source → None | `Chat/provider_readiness.py:235`, `:607`; resolver `:598-650` |
| Credential readers | `config.py:1728`; `config.py:9948`; `Chat/provider_readiness.py:610` |
| Config load; publish after every write | `config.py:1990` (`load_settings`); `config.py:7455` (`_publish_runtime_config_unlocked`); `config.py:9252` (`save_settings_to_cli_config`) |
| Secure keyring allowlist, detector (a chainer with more than one child → None), Windows blob limit and part scheme, read TTL | `runtime_policy/server_credentials.py:39-46`, `:423-449`, `:21-24`, `:478` |
| MCP keyring namespace | `MCP/credential_bindings.py:144-176` |
| Instance check is detection only, never a lock-out | `Utils/instance_lock.py:3-4` |
| Local provider keys (pricing) | `LLM_Calls/pricing_catalog.py:61-79` |
| Encryption: needed check, enable, disable, write-time encrypt, load-time decrypt | `config.py:9673-9688`, `:9718-9742`, `:9745`, `:7036-7062`, `:2159` |
| CLI dispatch and preflight | `cli.py:30-33`, `:35-56`, `:70-86` |
| App-scoped, viewless-capable Console runtime; submit entry; gateway protocol | `app.py:1167`; `Chat/console_runtime.py:4128-4135`; `Chat/console_chat_controller.py:9797`, `:4134` |
| Submission origins (MANUAL, QUEUED, AGENT_WAKE) | `Chat/console_chat_models.py:72-84` |
| Hidden-turn notice copy | `Chat/console_runtime.py:1472-1476` |
| Only a completion subscription exists on the store | `Chat/console_chat_store.py:2113` |
| Capacity and readiness owners | `Chat/console_prepared_request.py:1084`; `Chat/provider_readiness.py:730`; `Chat/console_session_settings.py:1742` |
| Get started card copy and notes action | `Chat/console_onboarding_state.py:15-22`, `:48` |
| Data and config paths | `Backup_Recovery/profile_paths.py:21-35`, `:86-90` |
| Module size ratchet rows | `Tests/Architecture/test_module_size_ratchet.py:75, 93, 97, 98, 128` |
| Boot CSS budget | `Tests/Performance/test_boot_css_byte_budget.py:117` |
| Six-step and five-exit test pins | listed in D1 and §6 |
| User Guide track section, step table, re-run section, stamp | `Docs/User_Guide/First_Run_Setup.md:51-59`, `:61-73`, `:120-127`, `:3` |

---

## 14. TASK-34100.17 acceptance criteria → sections

| AC | Where |
|---|---|
| #1 tldw server | D3; §3.4; F1–F4, F6. The AC's "Runtime" row is labelled "tldw server" (D3.1, rule S5) |
| #2 Quick 4 steps; TASK-21148 supersession; Welcome copy; pinned tests and guide | D1; §6 |
| #3 Documents-first; task-28019 | D8; §0.4 |
| #4 Spec exists; claims cited | this file; §13 |
| #5 Full list and order; not-now defaults; Finish with defaults | D2; §3.3 |
| #6 Ready screen | D4; §3.6 |
| #7 Say hello | D5. "Carried into Console as the first turn" is narrowed to "quoted by the arrival line, kept in History" (Q2) |
| #8 Re-run dashboard | D7; §3.7. "Console readiness deep links landing on the matching row" is narrowed to TASK-34100.10 AC#8's routing (Q9) |
| #9 Keychain-first | D9; §3.2; §4.6 |
| #10 What's next, arrival line, sheets | D6; §3.8–3.9 |
| #11 Plain-text setup | D10; §3.10–3.11 |
| #12 Non-interactive setup | D11: deferred, with its reopen condition |
| #13 Preserve | §7 |
| #14 ADR changes | §12 |
| #15 Rejected-on-purpose rulings | D12 (owner rulings pending) |
| #16 Owner approval recorded | §15 (pending) |
| #17 Follow-up tasks | §10 (to file after approval) |

---

## 15. Approval record

| Field | Value |
|---|---|
| Spec revision approved | — (pending; this is revision 2) |
| Date | — |
| Owner rulings on Q1–Q10 | — |
| Owner rulings on D12 items 1–7 | — |
| ADR drafts approved (§12) | — |

Implementation does not start, and the §10 tasks are not filed, until this table is filled in and TASK-34100.17's notes record the same.

---

## 16. Open questions for the owner

Each question has a recommendation, and only owner-level calls are listed.

| # | Question | Recommendation |
|---|---|---|
| Q1 | Approve the 4-step Quick track (D1), superseding TASK-21148 AC#5 while keeping its stable-total guarantee as rule S1? | **Approve.** |
| Q2 | Approve the Say hello policy (D5)?<br>(a) It auto-runs only on first run, for allowlisted local engines on loopback, with no key and nothing to load.<br>(b) Cloud and every other provider run only on an explicit press, with a money-first cost line and no extra dialog.<br>(c) Start chatting opens a **new** conversation. The test stays in History, and the arrival line quotes it. This narrows AC#7's "carried into Console as the first turn".<br>(d) The local auto-run writes a "Setup test" conversation without a press. | **Approve all four.** For (c), the alternative is to keep the test as the first turn but exclude it from context. That needs a per-message context-exclusion seam in Console's over-budget store, and still titles the user's first chat "Setup test". For (d), the alternative is to auto-run without saving, which loses the Moonshot persistence check [gap-02] on exactly the path that auto-runs. |
| Q3 | Make the system keychain the default store for new provider keys (D9), with an environment key always used unstored when present, an honest plain-text fallback where no keychain works, and no automatic migration of existing keys? | **Approve.** It removes the daily-password trade-off for most users without touching anyone's existing setup. |
| Q4 | Leave "Remember on this device" (and the recall check) out of the master-password path? | **Leave out.** The keychain-first default covers the need, and TASK-34100.4's reset makes a forgotten password recoverable. |
| Q5 | Should binding a tldw server from setup also prepare the Sync v2 profile, as Settings does today (D3)? It sends at least this computer's name to the server. | **Yes, keep parity**, disclosed before the choice. Different behaviour in setup and Settings would split the runtime-source owner. If you'd rather setup didn't, the coordinator gains a "prepare sync" flag, and the step's sentence goes. |
| Q6 | Defer non-interactive setup and the settings export (D11), reversing revision 1's "build it last"? | **Defer.** The report's verifiers advised it, the copy-config route works, and the CLI can't host the bind or the test. Reopen on the first user request for scripted setup, or the first support case from a copied config carrying keys. |
| Q7 | Plain-text setup v1 scope (D10): Welcome, Connect, Model, key storage, the verdict and plain review of those areas; no test message; tldw server and the other Full steps in F13b, filed together with F13? | **Approve.** It gets screen-reader and SSH users to a working configuration first, and filing F13b with F13 gives the remaining gap an owner. |
| Q8 | Confirm the seven "Rejected, on purpose" items (D12)? | **Confirm all seven.** |
| Q9 | Console readiness links (D7): keep TASK-34100.10 AC#8's routing (only "no provider configured" opens setup; other reasons go to their own control) rather than landing on a dashboard row, narrowing TASK-34100.17 AC#8? | **Keep .10 AC#8.** The fixing control is one hop away, and a dashboard row is two hops with the same result. The dashboard keeps row deep links for setup's own entries (palette, Settings). |
| Q10 | Front doors: the setup wizard stays the **boot offer** on a fresh profile, and task-33008's Get started card is the in-Console path for anyone who skipped, deferred AI or arrives without a working provider. Both read the shared verdict, and neither takes a key in Console. Confirm this split? | **Confirm.** The two serve different moments. Merging them would either put a corridor in Console or take the boot offer away from newcomers. |

---

## 17. Review history

### Revision 1 → revision 2 (2026-10-03)

Two independent critiques of revision 1 were received the same day: one HCI review (27 numbered items plus five low-severity notes) and one engineering review (four blocking, eight high, seven medium and three low). Neither critic edited anything. The tables below record what each point changed. Where this revision disagrees with a critic, the reason is also given in the decision named.

**HCI review.**

| Item | Disposition |
|---|---|
| 1 Verdict honesty (unverifiable keys, contradictory marks, quota) | **Changed.** Four verdict states, including "✓ Set up — key not checked yet"; the test on its own line, with one mark per region; rows for no credit (402/429 `insufficient_quota`) and rate limiting (D4, D5, §3.6) |
| 2 256-token cap unsafe for reasoning models | **Changed.** A model-aware cap; empty text at the limit counts as replied; test-only failures never change readiness; the cost line uses the real cap (D5) |
| 3 Loopback is not "costs nothing" | **Changed.** A positive local-engine allowlist; generic endpoints and keyed endpoints excluded; first run only; nothing to load; the memory cost named (D5) |
| 4 D9 default vs §2.3 and env keys | **Changed.** An environment key is always the default and never stored; storage applies only to a different, pasted key; a test that env values reach no store (D9, §2.3) |
| 5 Keychain prompts, write-back, save failure, downgrade | **Changed.** Four resolution states with Check again, timeouts never cached; a provenance overlay and writer guard with a property test; a save failure stays on the step; downgrade behaviour and orphan handling (D9, §4.4) |
| 6 D7/D8 contradiction; template "configured"; D10 rule | **Changed.** One `has_usable_chat_provider` predicate (template values never count); `ai_setup_deferred` sends Dee to Quick at Connect; single-step returns recompute the verdict, with Say hello and its cost line (D7, D8, §4.2) |
| 7 Privacy overclaims | **Changed.** The row is relabelled "tldw server" with "Library and notes stay on this computer"; the Provider row says where messages go; the pitch is scoped to "your messages are answered on it". Kept "tldw server" rather than "Library & sync" for one name per area (D3, D1) |
| 8 80x24 proven only for Quick; no legend | **Changed.** A Full Ready mockup at 80x24 with all nine rows, the legend and What's next collapsed to one row; the legend on every screen that shows –, ! or ✗. Collapsing ✓ rows was not adopted, because TASK-34100.12 AC#1 wants at least 8 read-back rows (D4, §3.6) |
| 9 Esc overloaded on Ready | **Changed.** Esc stops a test or collapses a list; otherwise it is inert, and never finishes setup (D4) |
| 10 In-flight test on other exits | **Changed.** Every exit stops the test silently; `SETUP_PROBE` raises no hidden-turn toast (D5) |
| 11 Say hello consent and discoverability | **Changed.** Next in Tab order with a hint; promoted, with a 0.5 s guard, when the key is not checked; the cost line wherever the button is; out of the exit rows (D4, D5, D7) |
| 12 Test exchange in the first real context | **Changed.** Start chatting opens a new chat; the arrival line quotes the test; owner Q2 (D5, D6) |
| 13 Naming drift | **Changed.** §3.0's one-name table and rule S5; mockups regenerated; one stated exception (the "Connect" step title) |
| 14 Storage selector adds a decision | **Changed.** A read-only sentence with "Change where…" (D9, §3.2) |
| 15 Keychain threat model and copy | **Changed.** The threat model is stated; platform stores are named; the macOS prompt copy is honest; Move warns about backups; the namespace is a persisted scope id (D9, §4.6) |
| 16 What's next issues | **Changed.** `→` marker; in-place expansion; Library exits kept docked on re-run, per .10 AC#14; "Add a project folder…" needs a file tool; "(if you run one)" (D4, D6) |
| 17 Failure gaps and output safety | **Changed.** An Ollama "not downloaded" row with a copyable command; replies and errors rendered as plain text with control sequences stripped, in the TUI and plain mode (D5) |
| 18 .5 AC#6's two branches | **Changed.** The probe profile carries its own minimal prompt, so the cost line no longer depends on .5 AC#6's branch (D5) |
| 19 Plain mode details | **Changed.** `quit`/Ctrl+D; GetPassWarning → exit 2; "Still waiting…" lines; key receipt line; plain review; `--help` and a pre-TUI line; NVDA/Orca and a real AT user in §8.4. The v1 gap is stated, with F13b filed alongside (D10, §3.10–3.11, §8.4) |
| 20 D11 overrides the verifiers | **Changed.** D11 is deferred; the reopen condition and design notes (value-level secret checks, import paired with export) are kept (D11, Q6) |
| 21 Keys taught at first contact | **Changed.** "Ctrl+P, then "model"" beside Alt+M; every hint teaches "← Back, Ctrl+B or Alt+←" (D6, §3, §5) |
| 22 Time claims | **Changed.** Quick's claim states its condition; Full's unmeasured "about 10" is dropped (D1) |
| 23 Finish with defaults one step late | **Changed.** "Save and finish with defaults" on Model (D2, §3.3) |
| 24 Dashboard state never written | **Changed.** "Last test replied…" is dropped (§3.7) |
| 25 No second-provider path on re-run | **Changed.** A "+ Add another provider…" row and a palette command; "also use it for new chats" off by default (D7) |
| 26 Welcome vs TASK-34100.13 (Reduce motion, ASCII) | **Changed.** A Reduce motion row in all Welcome mockups; an ASCII glyph table; ASCII mode in the CI matrix (§3.0, §3.1, §8.1) |
| 27 Sync consent; the active runtime in the verdict | **Changed.** Sync is disclosed before the choice; the verdict must account for a server that carries chat (a requirement on .2) (D3) |
| Low: dashboard intro | **Changed** to "Nothing changes until you press Save in one" |
| Low: cost line leads with jargon | **Changed.** Money first where the price is known (D5) |
| Low: "Runtime" and "(1M context)" | **Changed** for "Runtime". **Kept** "(context)" in the verdict: TASK-34100.17 AC#6 specifies it, and it is the cause when capacity blocks (D4) |
| Low: G2 false for an empty Keys page | **Changed.** Stated as G2's accepted exception (§0.2) |
| Low: Full label leads with "server" | **Changed.** "Full setup — adds search, tools, voice, appearance, server" (D1) |

**Engineering review.**

| Item | Disposition |
|---|---|
| B1 Size and boot ratchets | **Changed.** The measured table in §8.2. F2 depends on .1; the palette commands and exit routes move out of full modules (F0); Console seams in new modules (F9a stops rather than grows them); setup CSS per screen; `keyring` lazy after first paint |
| B2 Steps can't be hosted outside the corridor | **Changed.** F0: `SetupStepHost` with capabilities (`allows_secret_entry`, `resume_copy`), the `SetupSession` reducer, pure Provider/Model state. The sheet-body contradiction is resolved by capabilities (§4.7, D6, §3.8) |
| B3 Keychain resolution model | **Changed.** A post-first-paint worker; a provenance overlay applied synchronously on publish; four states; TTL plus 401 invalidation; a writer guard and census; both allowlists gain "keychain"; the Connect save order with compensation; a persisted scope id instead of a path hash (D9, §4.6, §12.2–12.3) |
| B4 Say hello needs Console seams | **Changed.** `SETUP_PROBE` origin; a turn-scoped probe profile never persisted; a viewless streaming observer with a non-streaming fallback; a prepare-only estimate asserted against dispatch; prices from `pricing_catalog.py`; dependencies on .5 AC#4/#7/#8; local ≠ free (allowlist, no keyed endpoints); Start chatting during a test stops it; no Say hello in plain mode (D5, D10, §4.7) |
| H1 Server commit not atomic | **Changed.** Coordinator rollback restores `[tldw_api]` and removes a new token; the probe uses the bound client's TLS settings; the bind is listed in the Exit dialog (D3) |
| H2 Dashboard chosen on `setup_completed` | **Changed.** The usable-provider predicate (D7) |
| H3 Deep links contradict .10 AC#8 | **Changed.** .10 AC#8 stands; this narrows .17 AC#8; owner Q9 (D7) |
| H4 New exit routes dropped | **Changed.** A typed route registry shared with .15, with an admission test per route (§4.7) |
| H5 Full Ready doesn't fit; two nav rows | **Changed.** One nav row with Back; short exit labels below 100 columns; a Full mockup at 80x24 (§3.6) |
| H6 Welcome missing the Reduce motion row | **Changed** (§3.1) |
| H7 Phasing | **Changed.** F8 before F5; .10 AC#1 narrowed and .6 AC#21 amended on approval, before those tasks start (§6, §10) |
| H8 Draft migration | **Changed.** In-memory migration persisted at the next checkpoint (no boot write); DOCS never drafts; unknown tracks fail closed; `last_track` persisted (§4.4) |
| M1 Setup state not import-pure | **Changed.** The setup core moves to `tldw_chatbook/Setup/`, with an import-purity test (D10, §4.7) |
| M2 CLI grammar | **Changed.** One argparse with subcommands, after the profile selectors and `--config`, and after preflight and admission; `recovery` unchanged (D10) |
| M3 Refusing setup while the TUI runs | **Changed.** Warn and confirm, per the owner's concurrent-instances rule (D10) |
| M4 Export format | **Moot by deferral.** The fixes are recorded in D11's notes for reopening (D11, §4.5) |
| M5 Keychain by platform | **Changed.** A canary probe instead of a name check; the chainer case; the Windows part scheme; the threat model; §8.4 adds Windows, desktop Linux (locked and unlocked) and macOS over SSH (D9, §8.4) |
| M6 Overlaps and citations | **Changed.** task-33008 in §0.4 and Q10; the two PocketTTS pins in D1; the Settings button is in Diagnostics today, and the label rule covers both buttons; ADR references name files (§12) |
| M7 Test strategy | **Changed.** INV-1 over the reducer; writer census, import purity, probe-turn, route admission, migrator-no-boot-write, two-process TTL, pty getpass and a secure-backend seam; "RED-verified" now means failing on the old code; Full Ready and the sheet in the size matrix (§8.1) |
| Low: re-run Say hello creates a conversation each time | **Changed.** It reuses one per provider and model, with no history sent (D5) |
| Low: second Esc on Ready | **Changed.** Esc never finishes setup (D4) |
| Low: the local auto-run writes a conversation unasked | **Folded into Q2(d)** |

**Kept from revision 1, with reasons recorded in the decisions:**
- the context size in the verdict (D4);
- "tldw server" over "Library & sync" as the row label (D3);
- the TASK-34100.10 AC#14 exits, four on re-run, rather than three (D4).

**Section numbering.** §3.11 (the non-interactive run) is replaced by the plain review transcript, and §4.5 is withdrawn with D11.
