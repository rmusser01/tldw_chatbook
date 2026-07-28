# First-Run Setup Wizard — Design

**Date:** 2026-07-28
**Status:** Approved (brainstorm complete; awaiting implementation plan)
**Model:** hermes-agent's setup wizard *process*, rendered in chatbook's native wizard chrome.

## Purpose

Give a brand-new chatbook user a guided, skippable, re-runnable setup experience that
walks through the full configuration surface (providers/keys, default model, RAG,
tools, notes sync, appearance, key encryption) and lands them in a working app —
replacing today's single welcome toast. The UX model is hermes-agent's setup wizard,
whose strengths are: a Quick/Full two-track opener, skip-everywhere with current
values as defaults, live-but-never-blocking validation, and an itemized honest
summary at the end.

## Non-goals (v1)

- Multiple credentials per provider / rotation (hermes has this; Settings covers it).
- Per-provider bespoke key validators beyond the OpenAI-compatible probe.
- Replacing the Console setup card (it remains the in-context post-wizard guide).
- Mimicking hermes's CLI visual style (boxed ANSI banners, curses menus); we reuse
  chatbook's existing wizard chrome.

## Prior art this builds on

- `UI/Wizards/BaseWizard.py` — `WizardScreen` / `WizardStep` / `WizardProgress` /
  `WizardContainer` framework, proven by the Chatbook create/import wizards.
- `Chat/console_onboarding_state.py` — the pure-state-module pattern to mirror.
- `Chat/local_server_discovery.py` (task-188) — localhost-only server discovery
  built for onboarding.
- `UI/Screens/settings_endpoint_probe.py` (task-191) — endpoint probe with
  secret-free summaries.
- `Utils/config_encryption.py` + `Widgets/password_dialog.py` — the existing
  key-protection mechanism; the wizard adds no new crypto.
- Config writers in `config.py` (`save_settings_to_cli_config`,
  `apply_settings_mutation_to_cli_config`) which auto-encrypt sensitive values once
  encryption is enabled.

## 1. User-facing flow

### Entry rules

- **Auto-shown once, ever:** offered when **no wizard state keys exist AND no
  provider is configured** (hermes-style guard via the existing readiness checks) —
  not keyed to the in-memory `_first_run` flag alone, because a first launch that
  dies before the wizard appears would otherwise leave that user permanently
  unoffered (config exists, `_first_run` false, no keys → misread as upgrader).
  Shown after startup completes normally: the initial screen renders first, then
  the wizard is pushed **on top** of it. Both startup paths (splash-enabled and
  splash-disabled) converge on the same single post-startup call. We do not insert
  into the startup critical path.
- **Persisted state** (because `_first_run` never touches disk):
  `first_run.setup_started` written when the wizard first shows;
  `first_run.setup_completed` written on Finish or explicit Skip. Started-but-not-
  completed on a later launch produces only a resume *toast* ("finish setup any time
  from Settings") — never a forced re-push. No "wizard jail."
- **Upgraders:** existing config with any configured provider → never
  auto-offered. A lived-in config with *no* provider configured gets the offer
  once (its readiness state says the app was never set up); skipping writes
  `setup_completed` and it never returns.
- **Re-entry:** "Run setup wizard" in Settings and a command-palette entry, both
  pushing the same screen. Re-run and first-run are one code path; steps prefill
  from current config.
- The existing `_show_first_run_notification` welcome toast is replaced by the
  wizard (no toast when the wizard shows).

### Welcome step

Chatbook branding + one-line pitch, then the two-track choice:

- ● **Quick setup** — provider & model (recommended)
- ○ **Full setup** — configure everything
- ○ **Skip** — explore on my own

Skip lands in a working app; the Console setup card remains as in-context guidance.

### Quick track

Steps 1–2 of the full track (provider & key, default model), then the conditional
"Protect your keys" step if a key was entered, then Summary. On the quick track
the Summary step opens with a **"Left at recommended defaults"** notice listing
exactly what was untouched (tools off, RAG off, default theme, notes sync off)
with pointers to where each lives in Settings.

### Full track (8 steps)

Every step has Back / Next / Skip. Nothing blocks. Esc raises a confirm dialog
("Finish later? Steps you've completed are already saved").

1. **Provider & credentials** — grouped picker (Cloud / Local from
   `Chat/provider_catalog.py`), masked key entry, one-click connect for
   auto-discovered localhost servers (Ollama / llama.cpp), live probe with clear
   pass/fail plus "save anyway". If the provider's key env var is already set:
   "Found in your environment ✓ — nothing to store." One primary provider in v1;
   more via Settings.
2. **Default model** — live-fetched model list where supported (the step fetches
   for the chosen provider itself; it does not rely on the startup catalog-refresh
   worker having finished), curated fallback otherwise, always "enter custom name"
   and Skip.
3. **RAG / embeddings** — checks `embeddings_rag` optional deps via
   `optional_deps.py`; missing → show install command and move on; present →
   embedding provider/model selection.
4. **Tools** — checklist of the gateable built-in tools (`[tools]` gates, all
   default OFF), each row with its risk/approval explanation. Copy makes clear
   enabling a risk-tagged tool still raises per-call approval cards.
5. **Notes sync** — enable toggle + notes directory picker.
6. **Appearance** — theme picker + splash card choice. Depends on task-740 (splash
   config read bug) being fixed first, else the choice silently doesn't take
   effect.
7. **Protect your keys** *(conditional: any key entered during this run, via
   `check_encryption_needed()`)* — offer config encryption through the existing
   `EncryptionSetupDialog` / `enable_config_encryption()`. Copy states plainly:
   "You'll be asked for this password each time chatbook starts."
8. **Summary** — itemized ✓/✗ matrix per area naming the exact missing piece
   ("✗ RAG — embeddings deps not installed"), **read back from the persisted
   config, not in-memory step data**; config file location; how to re-run. Exits
   differ by mode: first-run offers **Start chatting** (→ Chat) or **Explore on
   my own** (→ Home); a re-run launched from Settings offers **Done** (return to
   the launch point) with **Go to Chat** as the secondary action — a re-run must
   not yank the user away from where they were.

### Interplay with existing onboarding

The wizard writes the same provider/model config the Console setup card's readiness
checks read, so the card's steps 1–2 auto-resolve; its step 3 ("send your first
message") remains the natural post-wizard nudge. `console.onboarding.
first_send_completed` is untouched.

## 2. Architecture

### New components

- **`UI/Wizards/FirstRunSetupWizard.py`** — `FirstRunSetupWizard(WizardScreen)`
  with `WizardStep` subclasses: Welcome, Provider, Model, Rag, Tools, NotesSync,
  Appearance, ProtectKeys, Summary. Pushed directly via `push_screen` (not a
  route in `screen_registry.py`; it is not a tab). Contains
  `SetupWizardContainer(WizardContainer)` — see below.
- **`UI/Wizards/first_run_setup_state.py`** — pure-logic module, no Textual
  imports (mirrors `console_onboarding_state.py`). Owns:
  - `should_offer_wizard(config)` — fresh config + not completed;
  - track resolution and conditional step activation (Quick/Full; ProtectKeys
    only when a key was entered);
  - per-step **commit plans** (step data → exact config mutations);
  - **dependency invalidation** — when a step's committed value changes (v1 edge:
    provider → model), dependent steps reset and their prior commit is superseded
    in the same mutation;
  - prefill readers (current config → step defaults). For secrets these return
    **presence metadata only** ("configured ✓" → Keep / Replace / Clear
    affordance), never the value (task-145: secrets masked everywhere);
  - the summary-matrix builder (persisted config → ✓/✗ rows).

### Framework strategy: subclass, don't modify

`BaseWizard.py` stays byte-identical. `SetupWizardContainer(WizardContainer)`
overrides `handle_next()` / `show_step()` / progress accounting to add:

1. **Conditional step activation** — choosing Quick on the Welcome step
   deactivates steps 3–6 (RAG, Tools, Notes sync, Appearance); ProtectKeys
   activates on key entry in either track; progress dots recount.
2. **Commit-on-Next** — after validation passes, `handle_next()` awaits the step's
   commit before advancing.
3. **Async validation affordance** — steps flag "validation in progress" so Next
   shows "Testing…", resolving to pass / fail-with-save-anyway.

Esc → confirmation dialog is an override in our wizard, using the existing
`confirmation_dialog.py`.

### Reused machinery

| Need | Existing piece |
|---|---|
| Provider names/grouping | `Chat/provider_catalog.py` |
| Local server discovery | `Chat/local_server_discovery.py` (localhost-only guarantee) |
| Endpoint/key probe | `UI/Screens/settings_endpoint_probe.py` |
| Model lists | `config.get_cli_providers_and_models()` + `LLM_Provider_Catalog` services |
| Encryption | `check_encryption_needed()`, `enable_config_encryption()`, `EncryptionSetupDialog` |
| Form fields | `Widgets/form_components.py` |
| Confirm dialog | `Widgets/confirmation_dialog.py` |
| Styling | extend `css/features/_wizards.tcss` |

## 3. Persistence

- **Commit-on-Next:** each step's values are committed when the user advances past
  it (hermes saves after each section). A crash loses only the current step;
  Console-card readiness updates live; Skip is trivially safe (skipped steps never
  wrote anything).
- **Serialized writes:** all commits flow through **one exclusive worker**; Next
  awaits the commit, so ordering is inherent and a step you've left is definitely
  on disk. No interleaved read-modify-write of the TOML. This includes the
  ProtectKeys step: `enable_config_encryption()` rewrites the whole config file,
  so it runs inside the same worker, never alongside a queued step commit.
- **Write targets:** only sections/keys the Settings adapters already own —
  `api_settings.<provider>` for credentials, `[chat_defaults]` for provider/model,
  and the existing embeddings/tools/notes/appearance sections. The plan pins exact
  key names by reading each Settings adapter; the wizard invents no parallel keys.
  The single exception is the `[first_run]` wizard-state section
  (`setup_started` / `setup_completed`), which the wizard owns — the invariant
  test's oracle allowlists it explicitly.
- **Wizard state keys:** `first_run.setup_started` / `first_run.setup_completed`,
  persisted off-thread like `console.onboarding.first_send_completed`.
- **Encryption:** once enabled (step 7), the config writers auto-encrypt sensitive
  values (`_maybe_encrypt_setting_value`); keys entered earlier in the run are
  re-encrypted by `enable_config_encryption()`. Steps never handle crypto.

## 4. Validation

- **Scope fence:** live probing uses the OpenAI-compatible `/v1/models` probe and
  local discovery where they apply; all other providers get a format-level sanity
  check + "couldn't verify — saving anyway." No per-provider validators in v1.
- **Two timeout budgets:** ~2.5s for localhost discovery (existing), ~8s for cloud
  validation. Both worker-side, both skippable mid-probe.
- **Generation tokens:** every probe carries a token from the launching step;
  results apply only if the step is still current with unchanged inputs; in-flight
  workers are cancelled on Back/Esc. No stale "verified ✓" landing on a different
  provider's screen.
- **Injected probes:** steps receive probe/discovery callables as constructor deps
  (defaulting to the real ones), so tests pass instant fakes instead of
  monkeypatching `httpx`. The state module performs no I/O and never holds these.

## 5. Error handling

Governing rule (from hermes): **never hard-fail, never block — note it, offer
skip, continue.**

- **Probes/network:** failures are explicit but non-blocking ("Couldn't verify —
  save anyway?"). Model-fetch failure falls back to curated list + custom entry.
  Fully offline first runs work via the local-provider path.
- **Two failure classes, two policies:** step `compose()` crash → step auto-skipped
  with a one-line notice and a ✗-with-reason summary row (no surface to retry on);
  commit/validation failure → inline Retry / Skip, user stays on the step (Next
  awaits commit, so no silent advance past unpersisted data).
- **Encryption:** dialog handles password entry/strength/mismatch; on
  `enable_config_encryption` failure keys stay plaintext, the step reports it,
  summary shows "Encryption: off". No partial-encrypted state.
- **Expected non-errors:** missing RAG deps (install command + move on), zero
  discovered local servers (affordance doesn't render), key already in env
  ("found in environment ✓").
- **Crash/abandon:** completed steps are on disk; later launches get only the
  resume toast.

## 6. Testing

1. **Unit (pure state module — bulk of coverage):** `should_offer_wizard` across
   fresh / upgrader / started-not-finished / completed; track resolution and
   conditional activation; commit-plan builders; dependency invalidation
   (provider change resets model and supersedes its commit); prefill readers
   (secrets → masked presence only); summary-matrix builder. Hypothesis property
   tests where the project already uses them (arbitrary config states → consistent
   matrix).
2. **Widget (Textual Pilot):** Quick track end-to-end with injected fake probes;
   Full track skip-everything; Back/Next around provider→model dependency; Esc →
   confirm dialog; conditional progress recount.
3. **Integration (real config file via `TLDW_CONFIG_PATH` temp dir):** commits
   produce expected TOML; encryption path → values encrypted at rest +
   `first_run.setup_completed` present; **re-run against populated config** →
   correct prefills, masked secrets, unchanged steps produce no-op mutations (no
   config churn); **upgrader config** → never auto-offers. The "only keys Settings
   owns" invariant is asserted against a key set **derived from the Settings
   adapters' own constants** (if not exported today, extracting them is a plan
   item), not a hardcoded copy.
4. **Regression:** existing Chatbook create/import wizard tests pass untouched
   (guaranteed cheap by subclass-not-modify).

**Live verification** (per `backlog/docs/lessons-live-verification.md`): walk both
tracks in the real app with a fresh config dir, on **both** startup paths (splash
on and off — the hook convergence is where a path gets missed), plus one
small-terminal pass (80×24). Per CLAUDE.md, the matching `Docs/User_Guide/` page is
written/stamped (new user-facing screen).

## 7. Dependencies & open items for the implementation plan

- **task-740** (splash config ignored by `get_cli_setting`) is a prerequisite for
  the Appearance step's splash-card choice; the fix is a call-signature correction.
- Verify whether `probe_settings_endpoint` can send an auth header for cloud-key
  validation or needs a small extension.
- Verify whether `apply_settings_mutation_to_cli_config` is coupled to the
  Settings screen's domain-category contracts; if its mutation shapes don't fit,
  fall back to the plain `save_settings_to_cli_config` batch writer.
- Pin exact config key names per step by reading each Settings adapter.
- If Settings adapters don't export their owned-key constants, extract them so the
  invariant test has a real oracle.
- RAG step install copy: the install command differs by install method (pip vs
  uv-managed venv — a documented papercut in this repo), so show the generic
  extras syntax (`tldw_chatbook[embeddings_rag]`) rather than guessing the user's
  package manager.

## 8. Hermes affordances deliberately ported

Quick/Full two-track opener · skip-everywhere with skip as a safe default ·
current values prefill on re-run · Keep/Replace/Clear for existing secrets ·
live-but-never-blocking validation ("couldn't verify — saving anyway") ·
"left at defaults" honesty notice on the quick track · security as a first-class
step (tools off by default, approval-card explanation) · itemized ✓/✗ summary
naming the exact missing piece · "Start chatting now?" exit · one-shot
auto-offer with graceful re-entry instead of nagging.
