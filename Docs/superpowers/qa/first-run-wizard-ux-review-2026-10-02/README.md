# First-run setup wizard: senior design / HCI review

**Reviewed:** `origin/dev` @ `2d34cbf80d`, 2026-10-02/03 · both tracks (Quick, 6 steps; Full, 11 steps) · live on fresh isolated profiles with real provider keys, plus a full code read.
**Personas:** Sam, a first-time user with an OpenAI key · Jo, a first-time user who wants private local AI · Riley, an experienced power user with env keys, local servers and several machines.

## The short answer

**Does it properly help a first-time user? Not reliably.**
- The frame is good:
  - honest Welcome copy with time estimates;
  - a stepper with "Step N of M";
  - masked key entry that picks up exported keys;
  - forgiving endpoint entry with a live "Chat URL" preview;
  - specific connection errors;
  - safe exit dialogs.
- But "done" means "saved", not "works":
  - Of the five cloud providers walked through to a first message with real keys, **only Anthropic produced a reply**.
  - OpenAI is the first Popular row and the template default. Following the wizard's own recommendation, it ends at a ✓ Summary and a first message the app refuses to send.
- Heuristic score: **16/40 (Poor)**.

**Does it serve an experienced power user? Not yet.** The foundations are expert-grade:
- Env keys are detected and never written to disk.
- Provider and model are written in one atomic step.
- Back and Forward keep every choice.
- The speech-model install review is exemplary.

The surface undoes that work:
- Only one provider per run.
- A 60-row provider list with no filter.
- A model list capped at 20 of up to 466, with no search.
- A re-run that ignores current values, and an untouched Voice step that overwrites working TTS config.
- No scripted or second-machine path.
- A Protect step that can strand encrypted keys.

**Does it expose all configurable options? No, but adding more options is the wrong fix.**
- The wizard writes about 25 of the template's 1,234 config keys. That is the right order of magnitude for onboarding.
- The real problems are three:
  1. **Five of the seven optional Full-track steps don't do what they say.**
     - Voice writes config the user never chose.
     - Search & RAG saves a key nothing reads.
     - Notes writes nothing.
     - Tools says "Everything is off by default" while three hidden gates are on.
     - Protect turns on encryption that no reachable screen can turn off.
  2. **A handful of day-one needs are missing:**
     - an honest path for exported API keys;
     - a tldw server connection;
     - a way to turn off the splash and reduce motion;
     - a choice of key source when an env key exists;
     - showing where data lives;
     - more than one provider per run.
  3. **"Change it later" pointers lead to homes that are wrong or don't exist.**

## By the numbers

| | |
|---|---|
| Discovery | 7 agents (3 static, 4 live persona walkthroughs) → 320 observations |
| Consolidation | 124 canonical issues in 14 root-cause clusters |
| Adversarial verification | 9 skeptics, every P0/P1 re-run live → 120 confirmed (63 with corrections), 3 partly confirmed, 1 refuted |
| Gap hunt + new findings | 18 untested paths checked; 29 new issues; the 8 high-severity ones re-verified by a second skeptic |
| **Final register** | **151 issues: 5 P0 · 23 P1 · 71 P2 · 52 P3**, plus 80 verified strengths and 63 improvement ideas |
| Backlog coverage | **122 of 151 have no open task**, including 4 of the 5 P0s and 18 of the 23 P1s |

## The five blocking (P0) issues

| ID | What happens | Root cause (verified) |
|---|---|---|
| [cross-cutting-01] | Setup ends with ✓ and "Start chatting", but the first message is refused with "This request cannot fit the selected model…". This happened with both the "(recommended)" model and the shipped default `gpt-5.6-terra`. | OpenAI ids missing from the catalog fall back to a stale 4,096-token "API default" context window, and the 4,096-token reply reservation uses all of it. Nothing checks end to end before the handoff. |
| [model-02] | "(recommended)" is simply the first id the provider's API returned. It was once `tts-1-hd-1106`, a text-to-speech model, and four fresh runs recommended four different models. | No curation, no chat-only filter, and an unstable API order, sliced to 20 rows before any ranking. |
| [model-01] | Pressing Next or Enter on Model without clicking a row silently discards the provider and the key just entered, while the tracker still shows Provider ✓. | Provider only stages the connection in memory, and only a committed model writes it to disk. |
| [protect-summary-04] | After encrypting, Enter reopens the password dialog. A different second password re-keys the verifier over keys encrypted with the first, so **both launch paths lock the user out**. The step says "your keys are unchanged (plain text)" (false), and Next and Exit setup both fail for the rest of the session. | `enable_config_encryption` writes before it validates and never rolls back; Protect ignores existing encryption. |
| [gap-02] *(newly found)* | Moonshot (Kimi) setup ends with ✓, but every reply in a saved chat is generated and billed, then thrown away with "Provider continuation could not be persisted", behind a false "may send a duplicate request" warning. | All four models Moonshot currently lists use the reasoning-content continuation path, and its persistence step fails. Only Temporary chats work. |

## Selected major (P1) issues

- **Developer-launch lock-out** [protect-summary-01]. After Protect, `python -m tldw_chatbook.app` crashes at the unlock prompt with `NoActiveWorker` (`app_entry.py:501` calls `push_screen(wait_for_dismiss=True)` from `on_mount`). The documented `tldw-cli` launch unlocks through its pre-TUI prompt and works.
- **No unlock recovery** [protect-summary-02]. A forgotten password means hand-editing `config.toml`. No reachable screen can disable encryption or change the password [protect-summary-03].
- **Voice writes on Next** [voice-speech-01]. Next on an untouched Voice step writes TTS config, and on a re-run it overwrites a working endpoint. Its subtitle says "skip with Next".
- **Keyed provider with no key** [provider-01]. Next loops on "Retry with Next", which can never succeed. The only skip is a hidden Enter in the empty key field, so mouse users are stuck.
- **Detected local server forgotten** [provider-02]. A detected server is lost on the user's next action: Enter selects the highlighted OpenAI, and clicking llama.cpp fills the default `:8080` instead of the port that was found.
- **Cloud keys never really checked** [gap-01] [gap-03].
  - OpenRouter and Hugging Face accept any key, because the "check" is a public model listing.
  - Gemini offers no model list and never checks the key, and **all 8 shipped Google catalog models are retired (404)**.
- **False warning on every finish** [entry-exit-handoff-04]. Every successful "Start chatting" shows "Provider settings changed before Console opened. Review setup and try again." (8 of 8 runs; tracked as task-33001.10).
- **Re-run ignores current setup** [cross-cutting-05]. A re-run ignores the current provider, model, voice, RAG model and track, and cancelling a re-run started from Settings lands on Console.
- **Next and Enter are inconsistent** [cross-cutting-03] [cross-cutting-04]. Next means skip, save defaults or refuse depending on the step, and Enter is advertised as "next" while doing five different things.
- **Tools copy is false** [fulltrack-02]. Tools says "Everything is off by default" while 4 gates are hidden and 3 of them are on.
- **RAG step is a no-op** [fulltrack-01]. RAG saves an embedding key the pipeline never reads, cannot turn RAG on, and then reports "✓ RAG".
- **80x24 barely works** [a11y-01]. The frame takes 14 of 24 rows, and the Summary's read-back shrinks to a 3-row window.

## What to do first: about one sprint (detail in §5)

1. **Make the first chat work.**
   - Repair the OpenAI context-window fallback and cap the reply reservation; test every shipped default model.
   - Rank the model list: chat models only, a deterministic order and a curated recommendation.
   - End the false handoff toast.
   - Give honest key verdicts for listing-only providers.
   - Replace the dead Gemini catalog.
   - Fix Moonshot continuation persistence.
2. **Stop data loss and lock-outs.**
   - Skipping Model keeps the provider key.
   - Protect refuses a second enable and validates before writing.
   - Both launch paths share one unlock routine, with retry and reset.
   - An untouched step writes nothing.
   - Cancelling a re-run returns to where it started.
   - Ctrl+Q honours the exit guard.
3. **Remove dead ends.**
   - A real Skip on Provider, by keyboard and mouse.
   - Resume shows the key field.
   - An endpoint field for Azure, Databricks and Cloudflare.
   - A detected server stays chosen.
4. **Tell the truth.**
   - Tools copy derived from real state.
   - Correct "change it later" homes.
   - Fix the User Guide's contradictions.
   - A splash skip hint.

**Done when:** on a fresh profile, following the wizard's own recommendations reaches a reply for OpenAI, Anthropic, llama.cpp and Ollama. Setting a password and relaunching through either entry point opens the app. No Next is a dead end.

After that, ten structural fixes (SF1–SF10) retire the mechanisms that keep regenerating these bugs:
- "first chat works" is the definition of done;
- a curated, searchable model picker;
- a session draft instead of commit-on-Next;
- marks computed from persisted, verified state;
- a catalog-driven provider form with detection as the first row;
- an encryption lifecycle;
- a setup-session model for re-run and resume;
- one input policy;
- a terminal-native frame;
- steps that earn their place.

The review's history shows why point fixes aren't enough. Each honesty fix has been layered on the last, and `FirstRunSetupWizard.py` is now 10,854 lines against a 10,404-line size limit.

## Recommended shape (detail in §4)

- **Quick: 4 steps.**
  1. **Welcome:** adds a documents-first choice.
  2. **Connect:** a "Ready on this machine" group built from env keys, stored keys and running local servers, a filter, and a real key check.
  3. **Model:** chat models only, curated, with the recommended row highlighted.
  4. **Ready:** a verified "✓ Ready to chat — OpenAI · gpt-4.1-mini" row, an optional "Say hello" test, and named link-outs.

  Voice and Protect leave Quick; Protect becomes an option on Ready when a plaintext key exists.
- **Full: 11 real steps.** Every optional step defaults to "not now" and writes nothing if untouched.
  - New optional **tldw server** step.
  - Voice and Speech renamed **Spoken replies** and **Dictation**.
  - **Appearance** gains startup animation Off/Short/Full and Reduce motion.
  - **Protect** becomes state-aware (encrypt / change password / nothing to protect).
- **A re-run is a dashboard** of current values with a Change action per row, not a replay of first run.

## What works: keep it

Detail in §1 and §2 and §5.5.
- **Keys stay safe.** Keys are masked; secrets never reach disk unless the user saves them; exported keys are named and never stored.
- **Writes are careful.** Provider and model are written in one atomic step; Back and Forward keep every choice; resume restores the step and tracker.
- **Local detection is fast.** It finds a running server in under a second, correctly ignored an unrelated web server on :8080, and adopts it in one click.
- **Errors and recovery are well written.** Connection errors are specific ("Start it (ollama serve), then Retry"). The Summary swaps "Start chatting" for "Review provider setup" when the provider is missing.
- **Installs and previews are trustworthy.** The speech install review shows repo, revision, licence, size, destination, free space and SHA-256. The theme preview is live and reverts. Exit dialogs default to "Keep going" and absorb a double Esc.

## Method and limits

- **Isolation:** every live run used a fresh scratch `HOME` with the keyring backend disabled. The real `~/.config/tldw_cli` was never touched (verified by mtime).
- **Real providers:** OpenAI, Anthropic, Gemini, OpenRouter and Moonshot, using the repo-root key files.
- **Expired key:** the OpenRouter key file is expired (`401 API key expired`), which is how [gap-01] surfaced. You may want to rotate it.
- **Stub server for local paths:** no llama.cpp, Ollama or LM Studio is installed on this machine, so the local paths ran against a stub OpenAI-compatible server. The 9 findings that rest mainly on it are tagged *stub* in the register, and so are the local time-to-first-reply figures. Re-confirm them on a real local runtime before acting on them.
- **Contamination:** another session's Next.js server held `:8080` (the llama.cpp discovery port) throughout. Verifiers accounted for this, and the wizard correctly ignored it.
- **Review only:** no code changed and no backlog tasks filed. §5.6 lists the tasks to file and the existing tasks to update.
- **Evidence paths** (`evidence/…`) are relative to this folder; only captures the report cites are committed. File:line references are against `2d34cbf80d`; `group-assignment.json` maps every issue to its TASK-34100 subtask, and `register.json` carries current-dev pointers re-checked on `3c439d606e`.

## Contents

1. First-time user journey (Sam, Jo, and the early-skip user)
2. Power-user journey (Riley)
3. Heuristic evaluation (Nielsen 10, cognitive load, emotional journey, per-step scorecard)
4. Configuration coverage and information architecture
5. Solutions, structural fixes and improvement roadmap
6. Issue register (P0/P1 cards, P2/P3 tables)

---


---

## 1. First-time user journey

Two newcomers were walked live on fresh profiles: a 200×50 tmux window, a scratch HOME, keyring off, no provider environment variables. The bar is the wizard's own. The spec's first goal is "a guided, skippable, re-runnable setup that lands the user in a working app", and Welcome promises "Quick takes about 2 minutes."

**Friction scale.** **Low**: the user moves on without thinking. **Med**: the user hesitates, re-reads or takes a detour. **High**: the user is stuck or misled, or ends with a broken setup. Where the user's feeling and the real outcome differ, the row shows both.

---

### Persona 1: Sam (cloud, has an OpenAI key, fuzzy on LLM jargon)

**Goal:** "be chatting with a good model within a few minutes, and feel sure the key is safe."
**Run:** OpenAI on the Quick track, following every on-screen recommendation, with a password set on Protect.

| Stage | What Sam is trying to do | What actually happens | Feeling / friction | Issues |
|---|---|---|---|---|
| **Launch (splash)** | Open the app they just installed | Raw Python warnings print first: "RequestsDependencyWarning: urllib3 (2.6.3) or chardet … doesn't match a supported version!" and "python-frontmatter not installed. Markdown import will not be available." Then a random retro card: "A JULES PRODUCTION / PRESENTING / TLDW CHATBOOK". Any key skips it, but no hint says so. The wizard appears at about 11 s. | **Med**: "Who is Jules? Is it broken?" | [entry-exit-handoff-17] [new-coverage-power-01] (newly found) |
| **Welcome** | Get started, fast | "Quick takes about 2 minutes; Full about 10. Everything can be changed later in Settings, and most steps can be skipped with Next — Esc exits setup." Quick is preselected, so one Enter starts it. "Restore a backup" sits below as an unexplained bare line of text. | **Low**: reassured. This is the best-written screen in the flow. | [entry-exit-handoff-18] (minor) |
| **Provider: choose** | Find OpenAI | The provider box shows five rows ("Popular: OpenAI, Anthropic, Ollama, llama.cpp"). The other 56 providers hide behind a faint scrollbar thumb, and no row has a description. As soon as Sam selects OpenAI, before they have typed anything: "Couldn't discover models for OpenAI. Set OPENAI_API_KEY or add api_key under [api_settings.openai]. Or go Back." | **Med**: an error they didn't cause, worded in environment-variable and TOML terms | [provider-11] [cross-cutting-07] |
| **Provider: paste key** | Paste the key and know it went in | Tab stops on "▼ Authentication" before reaching the field. The pasted key shows as about 100 dots, with no reveal and no last-4. The help line is good: "New keys: platform.openai.com/api-keys. (Already exported OPENAI_API_KEY? It's picked up automatically.)" The status then reads "Provider settings changed since test; test again.", but OpenAI has no Test control on screen. | **Med**: "Did it paste? Test *what*?" | [provider-23] [provider-17] [provider-06] |
| **Provider: continue** | Move on; the footer says "Enter / Ctrl+N next" | Enter runs a probe instead: "✓ Reached the server, but your chosen model was not in its list. Pick one on the next step." Sam never chose a model, and the claim is false because the verdict is hard-coded. A second Enter re-tests, and so does every one after it. They have to find and click Next. | **High**: the keyboard contract breaks on step 2 | [cross-cutting-04] [new-cross-cutting-02] (newly found) [new-model-01] (newly found) |
| **Model** | Pick "a good model" | The top row is "tts-1-hd-1106 (recommended)", a text-to-speech model. 10 of the 20 rows cannot chat at all. The box shows 20 of OpenAI's 137 ids in raw API order, with no count, search or descriptions. "Or enter a model name" is prefilled with `gpt-5.6-terra`, and a single arrow press silently wipes it. Sam takes the recommendation, and the step ticks ✓. | **Low** felt / **High** actual: confident and already doomed. Next on the untouched prefill would have saved `gpt-5.6-terra`, which is also blocked at send. | [model-02] [model-04] [model-05] [cross-cutting-06] [cross-cutting-01] |
| **Voice** | Skip it, or try it quickly | The copy says "PocketTTS or OmniVoice run locally, no account needed", and PocketTTS is preselected. "Test and Hear" fails with "Not tested yet — the sample failed. Check the service, then retry." The cause is that PocketTTS is a separate server Sam doesn't have. Switching to OpenAI works and reuses their key: "Verified. The sample is ready to hear." Had they just pressed Next, the untouched step would have pointed their working OpenAI TTS at `127.0.0.1:8765` with auth set to none. | **Med**: a detour inside a "2-minute" track | [voice-speech-03] [voice-speech-01] [voice-speech-06] [coverage-16] |
| **Protect** | Make the key safe | The copy reads "…Skip to leave keys as plain text", but no Skip button exists. "Set up master password" opens with no field focused, so their first 12 typed characters vanish. After a Tab: "Password strength: Strong password", then "✓ Encryption enabled." Encryption is real: the keys become `enc:` values. Focus stays on "Set a password", so their next Enter reopens the dialog. Submitting a second, *different* password strands the keys and traps the wizard. | **High**: the step they chose for peace of mind is the one that bites | [protect-summary-07] [protect-summary-05] [protect-summary-04] |
| **Summary** | Confirm they're done | "✓ Provider — openai", "✓ Default model — tts-1-hd-1106", "✓ Voice — OpenAI (default voice)", "✓ Key encryption", "– Theme — textual-dark", "– Tools — all off; turn them on under MCP ▸ Servers ▸ built-in row ▸ Tool gates". "▐X▌ Get to know you after setup" looks ticked although it is off. There are five exits, and "Start chatting" has focus. | **Low** felt: every ✓ is green, but nothing on this screen checks that a chat can actually be sent | [cross-cutting-01] [protect-summary-12] [protect-summary-11] |
| **Handoff** | Land in a chat | A toast says "Provider settings changed before Console opened. Review setup and try again." It is false: the status bar shows "Provider: OpenAI  Model: tts-1-hd-1106". The screen also shows 14 nav tabs, "Agent blocked" and "Context unknown". The composer has focus: "Ready — type a message to begin." | **Med**: a warning on the success path ("did I do something wrong?") | [entry-exit-handoff-04] [entry-exit-handoff-22] |
| **First message** | "Say hi in five words." | The reply is a system row: "This request cannot fit the selected model. Response reservation and safety margin leave no model input capacity. Summarizing older turns cannot make enough room. Repair the model limit, reduce mandatory context or the response maximum, or allow older turns to be omitted." The recovery panel says "Response accepted; waiting for dispatch."; the composer says "Send blocked — resolve response recovery first". Retry only duplicates the error. The Alt+M switcher lists `gpt-5.6-terra/sol/luna` as "~4k" and "Ready · not tested", and terra blocks the same way. A reply ("Hello there! Hope you're well.") arrived only after Sam opened a new tab and hand-picked `gpt-4.1-mini`: about 20 extra actions, 621 s after launch. | **High**: the most likely point to quit. It reads as "my key or the app is broken." | [cross-cutting-01] [entry-exit-handoff-03] [new-cross-cutting-01] (newly found) [entry-exit-handoff-23] |
| **Next launch** | Open the app again tomorrow | **Developer launch** (`python -m tldw_chatbook.app`): a traceback ending "NoActiveWorker: push_screen must be run from a worker when `wait_for_dismiss` is True", then "Cannot proceed without decryption password." and exit 1, in 2 of 2 runs. **Documented launch** (`tldw-cli`): a bare pre-TUI "Configuration password: " prompt, which works if typed correctly. One typo drops them onto the full Backup & Restore screen with no retry, and the reason ("Recovery required: configuration_unlock_failed") appears only after Esc. One password goes by three names: "master password", "Unlock Configuration" and "Configuration password". | **High**: locked out on the developer path, and one typo from a recovery screen on the documented one | [protect-summary-01] [protect-summary-02] |

**Variant: Sam with an Anthropic key.**
- **What works.** "claude-sonnet-5-5 (recommended)" heads a sensible list, and the first reply works. This is the only cloud path that does.
- **A typo'd key still advances.** Provider shows ✓, and the rejection appears only on Model, as a greyed radio row: "Authentication failed — this API key was rejected. Go Back to fix it, or enter a model ID below." [provider-03] [model-07]
- **Enter on Model throws the connection away.** Pressing Enter without clicking a row discards both the provider and the key [model-01].
- **The Summary is honest, but its recovery path loops.** It shows "✗ Provider — no credentials or saved endpoint" and makes "Review provider setup" the primary button. That button lands on Provider rather than Model, and nothing says a model is missing, so the same Enter drops the pair again.
- **Time:** 223 s to the first reply, including both detours.

**Other cloud providers** were walked through to a first reply, and three more ended the same way: Gemini [gap-03], OpenRouter with an expired key [gap-01] and Moonshot [gap-02], all newly found. Each finishes setup with ✓ and fails on the first reply. The error copy also drops the provider's own reason [gap-06] (newly found).

---

### Persona 2: Jo (wants private, local AI; may have nothing running)

**Goal:** chat with a model that never leaves the machine.
**Paths:** two, because the experience depends on whether a server is already running.

#### Path A: nothing running yet

| Stage | What Jo is trying to do | What actually happens | Feeling / friction | Issues |
|---|---|---|---|---|
| **Launch (splash)** | Open the app | The same warnings and random card. The wizard appears at about 14 s from a cold start. | **Med** | [entry-exit-handoff-17] |
| **Welcome** | Check that this app can stay private | "Chat with cloud or local AI models, keep notes, and work with your own documents — all in your terminal." Nothing says that a local model needs no account or that nothing leaves the computer. | **Low–Med**: no reassurance, but nothing blocks them | [entry-exit-handoff-18] |
| **Provider: browse** | Find the local option | "Cloud providers need an API key. Local servers just need to be running — we'll look for them." The copy assumes they already have a server. Five rows are visible. The Local group comes after 42 cloud rows, about 50 key presses away; PageDown works but nothing hints at it. There is no type-to-filter and no LM Studio row. Arrowing *selects* each row it passes and runs discovery, flashing "Couldn't discover models for Baidu Qianfan. Set QIANFAN_API_KEY or add api_key under [api_settings.qianfan]. Or go Back." | **High**: errors about providers they never chose | [provider-11] [cross-cutting-06] [provider-13] [provider-12] |
| **Provider: Ollama** | Connect Ollama | "Couldn't discover models for Ollama. You can continue anyway.", with no reason given. "Find local servers" returns "Detected endpoints / No local endpoints found." It doesn't say what was checked or what to do next. "Test connection" is the bright spot: "✗ The connection was refused - nothing is listening at that address. Start the server, or check the endpoint and port." The wizard never explains what a local server is or how to get one. | **High**: a dead end | [provider-14] |
| **Model** | Pick a model, or work out what to do | The error is drawn as a greyed radio row and cut off: "Ollama isn't running at http://localhost:11434/v1/chat/completions. Start it (ollama serve), then Retry —…". The hidden half ("or enter a model ID below") is the rescue. Next opens "Continue anyway? / The server couldn't be reached, so this model setup is unverified. Continue anyway?" with "Keep editing" safely focused. It does not say that continuing without a model saves nothing. | **High**: the rescue that works is invisible | [model-07] [model-01] |
| **Voice** | Skip it | The copy says "skip with Next if you don't want voice." Next writes TTS config pointing at the PocketTTS port anyway. | **Low** (they don't notice) | [voice-speech-01] |
| **Protect** | Move on | The encryption pitch is followed by "No API keys saved yet — nothing to protect. This step matters once a key is stored; Next continues." The tracker then ticks ✓. | **Low**: a no-op page inside a six-step "quick" track | [coverage-16] [protect-summary-06] |
| **Summary** | See where they stand | "✗ Provider — no credentials or saved endpoint", "– Default model — not selected", "✓ Voice — PocketTTS (default voice)". "Review provider setup" replaces "Start chatting", which is honest. But the config still says OpenAI / `gpt-5.6-terra`: their local choice is gone. "No credentials" is also the wrong reason for a server that needs no key. | **Med–High**: told they failed at something they never attempted | [model-01] [cross-cutting-02] [protect-summary-12] |
| **Handoff (Explore Home)** | Get help connecting | Home shows "Home \| Blocked · Local". Its "Set up Console model" opens Settings ▸ Providers & Models preset to "OpenAI / gpt-5.6-terra / https://api.openai.com/v1". Console's "Get started" card explains the idea well: "A provider is the AI service that answers your messages — for example OpenAI, Anthropic, or a server running on this computer." But its "Set up provider" opens the same OpenAI form, with the API-key field focused. | **High**: every "set up" button points at a cloud key | [entry-exit-handoff-12] |
| **First message** | n/a | Not possible: "Composer unlocks after setup". | n/a | n/a |
| **Next launch (Ollama now running)** | Expect the app to notice | Console still says "Not ready · no key". The card has a detected-server path, but it never probes. The palette's "Setup: Run setup wizard…" does find Ollama, but it reopens at the first-run Welcome with no memory of their earlier attempt. | **Med**: recoverable, but only if they know about the palette | [entry-exit-handoff-21] [cross-cutting-05] [cross-cutting-12] |

**A rescue exists, but nothing on screen suggests it** [provider-14].
1. Choose Ollama.
2. Type a model id.
3. Choose "Continue anyway".

The Summary then honestly says "✗ Provider — saved, but the server couldn't be reached when models were checked". Once Ollama is started, Home shows "Ready · Local".

#### Path B: llama.cpp already running on :9099

Launch, Welcome, Voice and Protect are as in Path A.

| Stage | What Jo is trying to do | What actually happens | Feeling / friction | Issues |
|---|---|---|---|---|
| **Provider: detection** | Use the server the app found | In under 1 s: "Found a local endpoint: http://127.0.0.1:9099." with a "Use this server" button. Detection is fast, and it correctly ignored an unrelated web server on :8080. But focus stays on the list, with OpenAI highlighted. Enter (the advertised "next") selects OpenAI, the banner disappears, and "Couldn't discover models for OpenAI. Set OPENAI_API_KEY…" takes its place. Clicking llama.cpp instead fills the default `http://localhost:8080`, not the :9099 server just found. "Use this server" can only be reached with Shift+Tab or the mouse. | **High**: the app found their server, then forgot it | [provider-02] [cross-cutting-04] |
| **Provider: recover** | Get the server back | "Find local servers" brings back the banner and a "Detected endpoints" row. "Couldn't discover models for llama.cpp. You can continue anyway." stays on screen alongside them: three signals that contradict each other. Picking the detected row gives "Found 3 model(s) for llama.cpp." | **Med** | [cross-cutting-07] |
| **Model** | Accept the recommendation | Three models; the first is tagged "(recommended)", and none is selected. Ctrl+N advances silently with an unexplained amber "!", and the provider is never saved. | **High** (unseen) | [model-01] |
| **Summary** | Finish | "✗ Provider — no credentials or saved endpoint", while the tracker shows Provider ✓. "Review provider setup" keeps their settings but lands on Provider, not Model. After they click a model: "✓ Provider — llama_cpp", "✓ Default model — llama-3.1-8b-instruct-q4_k_m.gguf", "Start chatting". | **Med**: recoverable, but rows show raw ids and never say "on this computer" | [model-01] [cross-cutting-02] [protect-summary-12] |
| **Handoff** | Land in a chat | The same false toast: "Provider settings changed before Console opened. Review setup and try again." It appeared in 3 of 3 local runs. | **Med** | [entry-exit-handoff-04] |
| **First message** | Say hello | Streaming works end to end, with about 2 s to the first token (measured against a stub OpenAI-compatible server — no real llama.cpp/Ollama was installed on the review machine). Then "A Console turn completed while hidden. Return to Console to review." appears, although Console is on screen. More problems sit underneath this path: <br>• The "plain" chat sent a ~20.8 KB system prompt listing 16 agent tools, though the Summary said "Tools — all off". <br>• A slow cold model load shows 90 s of unchanging "Generating…", then fails with no guidance. <br>• In 2 of 5 runs, the first send hit "Trace capture blocked … Impact: The provider was not contacted." (trigger unidentified). <br>• LM Studio users, set up through Custom OpenAI-compatible, get no streaming at all. | **Low–Med**: it works, with noise | [entry-exit-handoff-05] [new-entry-exit-handoff-01] (newly found) [gap-07] (newly found) [entry-exit-handoff-06] [entry-exit-handoff-25] |
| **Next launch** | Come back | Calm: Console in about 12 s, no nag. The splash replays on every launch, still with no skip hint. | **Low** | [coverage-15] [new-coverage-power-01] (newly found) |

---

### Time to first chat (measured)

| Scenario | Measured time | Reached a reply? | Source |
|---|---|---|---|
| Launch to Welcome (splash) | 10–16 s across five cold starts (about 12 s lightly loaded, with a 7.0 s splash); 25 s or more with several apps running. Any key cuts it to about 2.5–2.8 s, but nothing says so. | n/a | [entry-exit-handoff-17] [new-coverage-power-01] (newly found) |
| Jo, server running, best case (scripted driver) | About 79 s from Welcome to the first streamed reply; **about 100 s** including the splash (stub server) | Yes | Gap check "Full llama.cpp setup" |
| Jo, server running, clean mouse path | About 8 actions; under 1 min after the splash | Yes | live-firsttime-local |
| Jo, server running, following on-screen hints | **About 4 min**, with two dead ends (the found server is lost on Provider; Model is skipped silently) | Yes, after "Review provider setup" | [provider-02] [model-01] |
| Jo, nothing running | n/a | **No.** Setup ends with their local choice discarded. A rescue exists but is never suggested. | [model-01] [provider-14] |
| Sam, OpenAI, following every recommendation | Summary at 467 s; reply at **621 s** (includes capture pauses; about 4–5 min hands-on) | Only after abandoning the wizard's model in Console (about 20 actions) | [cross-cutting-01] |
| Sam, Anthropic variant | **223 s**, including a typo'd-key detour and a silent provider drop | Yes, on the second pass through Model | [model-01] [provider-03] |
| Gemini / OpenRouter (expired key) / Moonshot | n/a | **No.** Setup ends with ✓; the first reply fails. | [gap-03] [gap-01] [gap-02] (all newly found) |

The "about 2 minutes" promise holds in two cases only:
- a local user whose server is already running and who clicks the right controls;
- Anthropic.

It does not hold for OpenAI, the provider most cloud newcomers arrive with and the one the shipped config itself defaults to.

---

### Where newcomers give up, or leave with a broken setup

1. **The first message after a ✓ Summary (Sam).**
   - **What happens:** every step is green and "Start chatting" is focused, yet the first send is refused in context-budget jargon, with no "Switch model" action [cross-cutting-01] [entry-exit-handoff-03].
   - **Root cause:** every OpenAI id missing from the capability tables resolves to a 4,096-token window, and the template reserves 4,096 tokens for the reply.
   - **The rest of the app agrees with the wrong answer:** the switcher, the Console header and Settings all say the blocked model is "Ready · not tested" [new-cross-cutting-01] (newly found).
   - **Why it is the worst:** the user did everything right, and the product tells them otherwise. Of five cloud providers walked to a first reply, only Anthropic got there.
2. **The Model step's silent discard (Jo; Sam on Anthropic).**
   - **What happens:** the Provider step only *stages* the connection; it is written to config only when a model row is pressed. Enter or Next with no row pressed advances with an amber "!" and saves nothing [model-01].
   - **What the user sees:** the Summary's "✗ Provider — no credentials or saved endpoint" blames the user. "Review provider setup" sends them back to Provider, not Model. Next then shows the same untouched model list, and the same Enter drops the pair again.
   - **On OpenAI the trap flips:** the prefill saves a model that is blocked at send instead [model-04].
3. **The Provider step's keyboard contract (everyone).**
   - **Enter does three different things on one screen:** it selects OpenAI over a server the app just found [provider-02], re-probes forever in a filled key field, and skips the whole provider when the field is empty [cross-cutting-04].
   - **No key means no way forward:** a newcomer without a key who presses Next gets "API key required. Set OPENAI_API_KEY or add api_key under [api_settings.openai].  Retry with Next, or go Back." Retrying never works, and a mouse user has no way past [provider-01].
4. **"No local endpoints found." followed by a cloud-only way back (Jo).**
   - **At the dead end:** nothing says what was checked or how to get a local server [provider-14].
   - **Every recovery button points at the cloud:** each "set up" button after setup opens an expert Settings form preset to OpenAI [entry-exit-handoff-12].
   - **Starting the server later doesn't help:** the Console card never notices [entry-exit-handoff-21].
   - **The result:** a privacy-first user is pushed toward a cloud key by the product itself.
5. **The next launch after Protect (Sam).**
   - **Developer launch:** it crashes (`NoActiveWorker`) and exits 1 [protect-summary-01].
   - **Documented `tldw-cli` launch:** it asks once, before the TUI. One typo opens Backup & Restore with no retry [protect-summary-02].
   - **Inside the wizard:** after success, the next Enter reopens the password dialog. A second, different password leaves the keys unreadable, and both Next and Exit setup fail [protect-summary-04].

---

### Persona red flags

**Sam (cloud): elements that fail them**

| Element | Why it fails Sam | Issue |
|---|---|---|
| "(recommended)" on the first model row | It marks list position, not curation. This run it tagged a TTS model, and four runs gave four different picks. | [model-02] |
| `gpt-5.6-terra` prefilled in "Or enter a model name" | The template's default looks like their own choice, disagrees with "(recommended)", and is blocked at send | [model-04] [cross-cutting-01] |
| Footer hint "Enter / Ctrl+N next" | On Provider, Enter selects, probes or skips. It never means "next". | [cross-cutting-04] |
| "✓ Reached the server, but your chosen model was not in its list." | False and hard-coded. It makes them doubt a model they never picked. | [new-cross-cutting-02] (newly found) |
| "Set OPENAI_API_KEY or add api_key under [api_settings.openai]" | Environment-variable and TOML jargon, shown before they act, with the key field right there | [cross-cutting-07] |
| Masked key with no reveal and no Clear | They can't verify a paste. Fixing a typo took 40 Backspaces. | [provider-17] |
| "Provider settings changed since test; test again." | OpenAI has no Test control | [provider-06] |
| PocketTTS preselected on Voice | Needs a server they don't have, and the failure gives no cause | [voice-speech-03] |
| "Set a password" keeps focus after success | Their next Enter reopens the dialog, and a second password strands the keys | [protect-summary-04] |
| "Skip to leave keys as plain text" | Names a button that doesn't exist | [protect-summary-07] |
| Summary ✓ rows plus "Start chatting" | "Saved" is presented as "works" | [cross-cutting-01] [cross-cutting-02] |
| "Provider settings changed before Console opened." toast | A false warning on every successful handoff | [entry-exit-handoff-04] |
| Blocked-send system row | Jargon with no "Switch model" action, contradicted by "Response accepted" | [entry-exit-handoff-03] |
| "~4k" and "Ready · not tested" in the switcher | Tells them blocked models are fine | [new-cross-cutting-01] (newly found) |
| Approval card for an internal `chat_with_llm` tool | Asks permission on the first chat, although the Summary said "Tools — all off" (2 of 4 sends on `gpt-4.1-nano`) | [entry-exit-handoff-24] |

**Jo (local): elements that fail them**

| Element | Why it fails Jo | Issue |
|---|---|---|
| Welcome copy | Never says "no account needed" or "stays on this computer" | [entry-exit-handoff-18] |
| Five-row provider box; Local group after 42 cloud rows; no filter | Hides the local options they came for | [provider-11] |
| Arrowing a list selects each row | Browsing runs discovery and shows cloud-key errors | [cross-cutting-06] |
| "Found a local endpoint" banner while focus stays on the list | Enter picks OpenAI and erases the find. The banner is muted grey and says "endpoint". | [provider-02] [provider-21] |
| Picking llama.cpp by hand fills :8080 | Ignores the server found seconds earlier | [provider-02] |
| No LM Studio row; discovery probes only ports 8080, 9099 and 11434 | LM Studio (1234), Jan and KoboldCpp are never found or named. Custom OpenAI-compatible defaults to :1234 without saying it is LM Studio's port. | [provider-13] |
| "No local endpoints found." | No list of what was checked and no how-to | [provider-14] |
| Model errors drawn as greyed, truncated radio rows | The rescue clause is cut off | [model-07] |
| "Continue anyway?" dialog | Doesn't say that nothing will be saved | [model-01] |
| "✗ Provider — no credentials or saved endpoint" | The wrong cause for a keyless local server | [protect-summary-12] |
| "Keep model lists fresh — checks your configured providers online at startup" | Alarming to a privacy-first user, and silent on what leaving it off means | [protect-summary-17] |
| Home "Set up Console model" and Console "Set up provider" | Open an OpenAI-preset Settings form, not local discovery | [entry-exit-handoff-12] |
| Console "Not ready · no key" after their server starts | The card never re-probes | [entry-exit-handoff-21] |
| ~20.8 KB agent prompt on a plain chat | Likely to overflow a small-context local server | [new-entry-exit-handoff-01] (newly found) |

---

### Third newcomer: the user who presses Skip or Esc early

| How they leave | What they see immediately | Next launch | Way back to setup |
|---|---|---|---|
| **Esc or "Skip setup" on Welcome** | "Skip setup and stop showing it at launch? You can rerun setup from Settings ▸ Diagnostics." "Keep going" has focus, and a reflexive double Esc is absorbed. Confirming lands on Home ("Blocked · Local"), and a "Check model lists online?" modal opens at once. It claims "your configured cloud providers (OpenAI, Anthropic, MistralAI, Moonshot, OpenRouter, QwenCloud, ZAI) … using your configured API keys", though nothing is configured. | No wizard. Console opens, a different landing screen from the first one, with the "Get started" card: "1. ● Connect a provider (API key or local server) / Not ready · no key" and "Composer unlocks after setup". | The card's "Set up provider" and the footer's "Enter continue setup" both open Settings ▸ Providers & Models preset to OpenAI, not the wizard. [entry-exit-handoff-13] [entry-exit-handoff-15] [entry-exit-handoff-12] |
| **Esc mid-setup** (e.g. on Provider with a key typed) | "Exit setup? This provider connection is staged only in this wizard and has not been saved. Your non-secret setup progress will resume at Provider." Accurate but full of jargon. It lands on Console. | "Continue setup? A previous setup stopped before it finished. Resume from the last completed step, start over, or continue later. Credentials are not retained in setup recovery and may need to be re-entered." The key is gone. "Later" writes nothing, so this dialog returns on every launch. | Resume works. The exit copy's "You can continue setup any time from Settings ▸ Diagnostics" is false: that button restarts at Welcome. [entry-exit-handoff-08] [entry-exit-handoff-07] [cross-cutting-08] |
| **Ctrl+Q** (which Ctrl+C's toast teaches) | Instant exit with no confirmation; a typed key is discarded | If they quit on Welcome, setup is never offered again automatically. Instead every launch shows "Setup isn't finished — run it any time from Settings ▸ Diagnostics ▸ Run setup wizard." | Only through the palette or Settings. [gap-05] (newly found) [entry-exit-handoff-07] |
| **Esc on Summary** after completing every step | "Exit setup? …" | Setup is never marked complete, so the "Continue setup?" nag appears; "Resume" lands back on Summary. | [protect-summary-08] |

**Finding setup again** is a recall task:
- **Command palette:** "Setup: Run setup wizard…" matches "setup", but not "api key" or "getting started".
- **Settings:** Settings ▸ Troubleshooting ▸ Diagnostics ▸ Run Setup Wizard takes about 11 keys. At 80×24 it sits below the fold, under Validate and Reload Config [cross-cutting-12].
- **The rerun itself:** it reopens with first-run copy ("Skip setup and stop showing it at launch?") and ignores the current provider and model [cross-cutting-05].
- **Console's "setup-blocked" links:** they open the wizard at Welcome whatever the cause [entry-exit-handoff-27].

The early skipper is not stranded: the Console card catches them and explains "provider" well. But every way back steers them away from the wizard, which has the local discovery and the key help, and into an expert form preset to OpenAI.

---

### Verdict

**Does the wizard help a first-time user get to value? Not reliably, and least of all for the most common newcomer.**

- **The core defect:** the wizard's completion contract is "saved", not "works". Every ✓ reflects what was written to config, and nothing between Summary and Send checks that a turn can actually go out [cross-cutting-01].
- **The experience runs backwards.** The path that feels smoothest, Sam accepting every recommendation, ends broken. The honest, recoverable states ("✗ Provider", "Review provider setup") are the ones that feel like failure.
- **Cloud:** of five cloud providers walked to a first reply, only Anthropic got there. OpenAI did not, and it is the provider most newcomers bring and the one the shipped config defaults to. It can work only when the arbitrary "(recommended)" row happens to land on a working chat model, as `o3` did in one verification run [model-02].
- **Local:** it works, and fast (about 100 s best case (measured against a stub OpenAI-compatible server — no real llama.cpp/Ollama was installed on the review machine)), but only for a user who already runs a server, clicks instead of pressing Enter, and clicks a model row. A user with nothing running hits a dead end and is then routed, repeatedly, to a cloud-key form.
- **Protect:** the step meant to build trust locks out the developer launch path and leaves the documented path one typo from a recovery screen.

**What it does well, and must keep:**
- **Welcome:** honest expectations, explicit reversibility and a safe default.
- **Orientation:** a stepper, "Step N of 6" and per-step key hints. Sam was never lost about *where* they were, only about *what had happened*.
- **Local auto-detect:** under 1 s, with no false positive on an unrelated :8080 web server, plus one-click "Use this server".
- **Endpoint entry:** forgiving (eight spellings resolve correctly) with a live "Chat URL:" preview.
- **Connection errors:** specific copy, plus engine-specific rescue ("Start it (ollama serve)").
- **Key entry:** masked, with a where-to-get-one line and automatic pickup of exported keys.
- **OpenAI voice:** reuses the key and verifies a real sample.
- **Encryption:** real, and the dialog states what forgetting the password costs.
- **Summary:** reads back from disk and swaps "Start chatting" for "Review provider setup" when the provider is missing.
- **"Review provider setup":** returns into the wizard with the user's settings intact.
- **Exit dialogs:** default to "Keep going" and absorb a double Esc.
- **Console "Get started" card:** the clearest plain-language explanation of "provider" anywhere in the product.
- **Streaming:** works end to end against llama.cpp- and Ollama-shaped endpoints (measured against a stub OpenAI-compatible server — no real llama.cpp/Ollama was installed on the review machine).
- **Second launch:** calm.

The parts are mostly right. What's missing is the contract that connects them:
- a final "can this chat actually send?" check before "Start chatting";
- a Model step that never silently drops what Provider staged;
- recovery paths that lead back into setup, not into a Settings form preset to OpenAI.


---

## 2. Power-user journey ("Riley")

**Riley** works keyboard-first. Keys for OpenAI, Anthropic and OpenRouter live in environment variables. Riley runs local model servers, reads `config.toml` directly, sets up more than one machine, and re-runs setup to change one thing at a time. Riley asks three things of any setup tool: **is there a fast path, do I get full control, and will it leave my working config alone?**

**Sources.** A live walkthrough in isolated tmux servers with a scratch HOME, keyboard only, with a `config.toml` snapshot after every change (`reports/live-power-user.md`, `evidence/live-power-user/`). The coverage audit (`reports/static-config-coverage.md`). The verified register. Where verification corrected the live report, this section follows the correction and says so.

### The short answer

The engineering under this wizard is what an expert wants. Secrets never reach disk. Provider and model go to disk in one atomic write. Tools, Appearance and Speech write only what changed. Back and Forward keep every choice. The surface on top undoes that work:

- The Summary's ✓ does not mean chat works [cross-cutting-01].
- Pressing Next on a Voice step Riley never touched rewrites a working TTS endpoint [voice-speech-01].
- A re-run promises "current values" and shows almost none of them [cross-cutting-05].
- A second password on Protect strands the encrypted keys and traps the wizard [protect-summary-04].

Riley finishes setup by diffing `config.toml` to find out what it did. Showing what setup did is the wizard's own job.

| Riley's test | Grade | Why |
|---|---|---|
| Fast path | **Partial** | With env keys the wizard is skipped (0 keys), and Quick takes 7 keys from the palette. Both end on a default model whose every send is blocked [coverage-04] [cross-cutting-01]. |
| Full control | **Fail** | One provider per run, no base-URL override, no splash switch, no cloud speech-to-text, a model list capped at 20, and 4 tool gates hidden [coverage-08] [provider-19] [coverage-15] [voice-speech-15] [model-05] [fulltrack-02]. |
| No clobbering | **Fail** | Next on an untouched Voice step overwrites config. Exit keeps every write. Nothing lists what changed [voice-speech-01] [cross-cutting-09]. |
| Portability | **Fail** | Restore accepts only a `.tldw-backup.zip`. There are no setup flags. The route that works (copy `config.toml`) is undocumented [entry-exit-handoff-16] [coverage-10]. |
| Honesty of status | **Fail** | In the tracker, the Summary and Console, ✓ means "saved", not "works" [cross-cutting-02] [fulltrack-01] [new-cross-cutting-01] (newly found). |

---

### (a) First install, Full track, keyboard only

On a fresh install Riley never sees the wizard. Any provider key found in the environment suppresses the offer [coverage-04]. Riley lands in Console under a 10-second toast: "Provider key detected · Found OPENAI_API_KEY, ANTHROPIC_API_KEY (and more) — you're ready to chat. Run setup any time: Settings ▸ Diagnostics ▸ Run setup wizard." It appears at the same moment as the "Check model lists online?" modal [new-coverage-power-02] (newly found).

"Ready to chat" is never checked. The template default, OpenAI · gpt-5.6-terra, has every send blocked. To get any control, Riley opens setup from the command palette, and the code treats that as a re-run (`rerun=True`).

| Stage | Goal | What happens | Keystrokes / efficiency | Issues |
|---|---|---|---|---|
| **0 · Launch** | Get to something useful | A random splash card ("A JULES PRODUCTION / PRESENTING / TLDW CHATBOOK") plays. Before it, third-party warnings print to the terminal. Launch to first screen was about 12 s lightly loaded and 25 s or more under load. Any key skips the splash, but nothing says so. | 0 keys. About 7 s of splash, or 2.5–2.8 s for someone who knows to press a key | [entry-exit-handoff-17] [new-coverage-power-01] (newly found) [coverage-15] |
| **1 · Open setup** | Reach the wizard | The palette finds "Setup: Run setup wizard…" for `setup`. "api key" and "getting started" match nothing. Through Settings, `Tab` walks the category list instead of the content. A second palette launch can stack a second wizard. | Palette: 7 keys. Settings: about 11 (`F4` `/` `diag` `Enter` `F6` `Tab` `Tab` `Enter`) | [cross-cutting-12] [entry-exit-handoff-11] |
| **2 · Welcome** | Choose Full | Focus starts on the track choice; `Down` selects Full and `Enter` advances. The tracker still reads "Step 1 of 6" with the Quick labels until Next. On the 11-step track it then drops every step name and shows numbers 1–11. The copy is first-run copy ("Welcome to tldw chatbook", "Skip setup") on what is a re-run. "Full setup — configure everything" oversells it. | 2 keys | [cross-cutting-10] [cross-cutting-05] [entry-exit-handoff-18] |
| **3 · Provider** | Connect OpenAI, Anthropic, OpenRouter | **Env-key handling is right.** The step says "Found OPENAI_API_KEY in your environment; nothing to store.", discovery runs ("Found 137 model(s) for OpenAI."), and the key is never written: config records only `credential_source = "environment"`. **The rest works against Riley:** <br>• 60 rows plus 4 headings in a 5-row box. Typing "openr" does nothing, and env-ready rows carry no badge. <br>• Moving the highlight selects the row, so each arrow press over a row whose key is present starts discovery. Keyless rows leave errors behind ("Couldn't discover models for Baidu Qianfan. Set QIANFAN_API_KEY or add api_key under [api_settings.qianfan]. Or go Back."). <br>• One provider per run. Picking Anthropic replaces OpenAI. <br>• No base-URL field for OpenAI, Anthropic or OpenRouter. An env key hides the key field and its Keep / Replace / Clear actions. <br>• "Found 13 model(s) for Anthropic." sits beside "Connection testing is unavailable for this provider." <br>• Five legacy-alias rows look like duplicates. <br>• OpenRouter accepts any key, because its "check" is a public model listing. | OpenRouter is 32 `Down` presses away, or about 7 with `PageDown`, which nothing advertises. The first highlight is not a selection, so `Space` is required. `Enter` on the list selects but never advances. The first `Tab` stops on the "▼ Authentication" header. | [provider-11] [cross-cutting-06] [cross-cutting-04] [coverage-08] [provider-19] [provider-06] [provider-12] [provider-23] [gap-01] (newly found) |
| **4 · Model** | Pick a chat model | "(recommended)" is simply the first id the API returned: "tts-1-hd-1106   (recommended)" for OpenAI and "inclusionai/ling-3.1-flash   (recommended)" for OpenRouter. 10 of the 20 OpenAI rows cannot chat. OpenRouter's 466 ids become 20 rows in a 5-row box with no count. The typed id `anthropic/claude-opus`, which OpenRouter does not list, was accepted and written to both `[chat_defaults].model` and `[api_settings.openrouter].model`. The prefilled `gpt-5.6-terra` is wiped by a single `Down`. Separately, the Provider step's probe line always says "your chosen model was not in its list". That verdict is hard-coded, and it is false for gpt-5.6-terra, which OpenAI does list. | The list wraps silently from row 20 to row 1. No filter. | [model-02] [model-05] [model-06] [model-04] [new-cross-cutting-02] (newly found) |
| **5 · Voice** | Use OpenAI TTS (key already in env) | PocketTTS is preselected despite the OpenAI key. Its test fails with "Not tested yet — the sample failed. Check the service, then retry." The OpenAI path verifies ("Verified. The sample is ready to hear."). After the test, focus jumps back to the Service radio, and the next `Tab`+`Space` wipes Sample text to "0 / 500" and resets the result. The Advanced disclosure exposes endpoint, auth, model, voice, format and speed. | The focus wipe reproduced 2 of 2 times | [voice-speech-03] [voice-speech-05] [voice-speech-06] |
| **6 · RAG** | Set the default embedding model | The list is raw `[embedding_config.models]`. It includes the broken TOML-split ids "bge-base-en-v1" and "bge-small-en-v1" and template placeholders, with no metadata and no "(current)" mark. **The pick writes a key the RAG pipeline never reads.** RAG keeps all-MiniLM-L6-v2, yet the Summary says "✓ RAG — embedding model: openai-text-embedding-3-small". No control turns retrieval on. | `Up` wraps to the OpenAI rows, which is quick | [fulltrack-01] [coverage-13] |
| **7 · Speech** | Use cloud speech-to-text | The only options are the 632.8 MiB Parakeet download, a model folder, or a transcribe.cpp GGUF. There is no Whisper API option despite the OpenAI key. The install review is exemplary. The download has no Cancel, speed or ETA, and "Not installed." stays above the bar. Next leaves silently while the download keeps going (disk 105M → 121M → 169M). The finished model never becomes the default. A killed download restarts from zero. | The actions sit above the Language and Precision choices that decide what gets installed. Faint button focus made one `Shift+Tab` + `Enter` open the file picker instead of install. | [voice-speech-15] [voice-speech-09] [voice-speech-10] [voice-speech-11] [voice-speech-12] [a11y-11] |
| **8 · Tools** | Turn on read tools | Eight switches with plain-language blurbs and ⚠ on write tools, which is good. The subtitle "Everything is off by default. Tools that read or change your files still show an approval card every time they run." is false: 4 gates are hidden and 3 of them ship on. The app's built-in MCP server also gives the agent `create_note` while "Create note" is off. On/off shows only by knob position and colour. The list overflows at 120×40 with no count. No web-search keys and no permission mode. | 1 key per switch. No "read-only tools" preset. | [fulltrack-02] [new-full-track-steps-02] (newly found) [a11y-07] [fulltrack-12] [fulltrack-06] |
| **9 · Notes** | (none) | No controls: "Nothing is activated during first-run setup." | 1 Next that does nothing | [fulltrack-05] |
| **10 · Style** | Pick a theme; turn off the splash | Live theme preview and a "(current)" marker, which is good. There is no splash off or duration setting and no reduce-motion. Half the step is a splash-card gallery that expands to 78 cards. | (none) | [coverage-15] [fulltrack-09] |
| **11 · Protect** | (none for env keys) | "No API keys saved yet — nothing to protect. This step matters once a key is stored; Next continues." It is accurate, but it never says the keys come from the environment and nothing is stored. The tracker ticks it ✓. | 1 Next that does nothing | [protect-summary-06] [coverage-16] |
| **12 · Summary** | Confirm what changed | It reads back from disk, which is good: "✓ Provider — openai", "✓ Default model — gpt-5.4-mini", "✓ Voice — OpenAI (default voice)", "✓ Tools — 5 enabled". **Not shown:** `[api_settings.openai].model` was overwritten, `[tts_settings]` was duplicated, and "OpenAI" became "openai". No key source, tool names or data folder. Rows can't be acted on and there is no Splash row. "▐X▌ Get to know you after setup" draws an X when unchecked. | Focus lands on "Start chatting" | [cross-cutting-09] [protect-summary-13] [protect-summary-16] [protect-summary-11] [cross-cutting-02] |
| **13 · First chat** | "Reply with just: OK" | **Blocked:** "This request cannot fit the selected model. Response reservation and safety margin leave no model input capacity…". The composer shows "Send blocked — resolve response recovery first" and the status bar "Context unknown". The default gpt-5.6-terra fails the same way. On arrival a false toast also reads "Provider settings changed before Console opened. Review setup and try again." With some models the first reply asks to approve an internal `chat_with_llm` tool, although the Summary said "Tools — all off". | About **75–90 keys** for the whole Full run, ending in a blocked composer | [cross-cutting-01] [entry-exit-handoff-03] [entry-exit-handoff-04] [entry-exit-handoff-24] |

**The pattern.** Riley can't see the best-engineered parts: env detection, the atomic provider write, Back/Forward. Riley can see the lists, the ticks and the copy, and that is where trust is lost. Selection, recommendation and status all show the app's internal state ("the highlighted row", "API row 0", "a key was written"). None of them shows what Riley wants to know.

---

### (b) Re-running setup over an existing config

The Settings button promises "Re-run the guided first-run setup with current values." The spec promised more: a re-run that ends with **Done**, returns to where Riley started, and "must not yank the user away" (spec D7 and D8, both unmet). Here is what a re-run actually does:

| Area | Riley expects | What happens | Issues |
|---|---|---|---|
| Framing | "Review your setup" | First-run Welcome. Quick is preselected even when the last run was Full. A "Skip setup" button, and Esc asks "Skip setup and stop showing it at launch?" | [cross-cutting-05] |
| Provider | `openai` marked current | Nothing is selected or marked. The tracker shows "!" on Provider and Model for a healthy config. Next with nothing selected keeps `chat_defaults`, which is good. | [cross-cutting-05] [cross-cutting-02] |
| Model | Current model pressed | "Models for your provider." / "Pick a provider first — or type a model name below". `read_wizard_prefill()` loads the provider and model, but the Provider step never reads them. | [cross-cutting-05] [model-04] |
| Voice | Current voice shown, untouched | PocketTTS is shown. **Next rewrites the endpoint** (diff below). | [voice-speech-01] |
| RAG | Current default marked | Not marked. Next with nothing selected writes nothing, which is good. | [coverage-13] |
| Speech | Saved language and engine preselected | The step reads a config key that does not exist at startup. There is no "Currently configured" line, a saved German default shows as English, and a configured GGUF is reported as missing. Installing silently replaces the default. | [new-voice-speech-01] (newly found) |
| Tools | Gates as saved | Prefilled within the same app session. **After a restart every switch shows OFF**, because prefill reads `load_settings()`, which omits `[tools]`. A gate shown OFF can't then be switched off, and the Summary contradicts the step. | [new-full-track-steps-01] (newly found) |
| Style | "(current)" | Works for the theme. The splash card always shows "Surprise me". | [fulltrack-09] |
| Summary | **Done**, back to where I was | The same five first-run exits. No Done. | [protect-summary-14] |

**The clobber, from Riley's own config** (`configs/power1-02-…` → `power1-03-…`, after one Next on an untouched Voice step):

```diff
 [app_tts]
-OPENAI_BASE_URL = "https://api.openai.com/v1/audio/speech"
-OPENAI_AUTH_MODE = "api_key"
+OPENAI_BASE_URL = "http://127.0.0.1:8765/v1/audio/speech"
+OPENAI_AUTH_MODE = "none"
```

`default_model = "tts-1-hd"` and `default_voice = "shimmer"` stay as they were. OpenAI voice names are now pointed at the PocketTTS port with no auth, a mix that cannot work. Meanwhile the draft records `pocket-tts` / `alba` / `wav`. Verification adds that this breaks working TTS on the **first** run too, for any user with an OpenAI key [voice-speech-01]. The Voice subtitle says "skip with Next if you don't want voice". The only way to re-run setup without this write is to exit before reaching Voice.

**Cancel semantics: every Next is a commit, and no exit rolls back.**

- The Esc dialog reads "Exit setup? Steps you've already completed are saved. You can continue setup any time from Settings ▸ Diagnostics." It is honest about saving. It has no undo and no list of what changed [cross-cutting-09].
- "Continue" is false. A re-run always restarts at Welcome. A killed Full re-run leaves `draft_track = "full"` and `active_step_id = "tools"` in `config.toml`, and the next open ignores them and starts on Quick [cross-cutting-08]. It also leaves the partial 169 MB speech download in staging [voice-speech-11].
- Cancelling a re-run started from Settings lands on Console. That path was never wired, though TASK-31813 AC#2 was ticked on the strength of the palette path. The palette re-run stays in place [cross-cutting-05].
- `Ctrl+Q` quits at once and skips the guard that Esc shows. `Ctrl+C`'s stock toast, "Do you want to quit? Press ctrl+q to quit the app", teaches exactly that bypass [gap-05] (newly found).
- The always-visible hint teaches `Ctrl+B` for Back, which is the default tmux prefix. tmux swallows it unless it is pressed twice [cross-cutting-15].

**Encryption re-run: the most dangerous path in the wizard.**

1. On a re-run, Protect decides whether keys exist through a check that returns False whenever encryption is on. So it says "No API keys saved yet — nothing to protect." about keys that are saved and encrypted (verified in code) [protect-summary-04].
2. After "✓ Encryption enabled.", focus stays on "Set a password" while the hint says `Enter` means next. `Enter` reopens "Set up master password" (live) [protect-summary-04].
3. Entering a second, different password writes the new verifier to disk, and then the runtime reload fails. The step shows "✓ Encryption enabled." next to "Enabling encryption failed — your keys are unchanged (plain text)." The second line is false: the keys are encrypted under the *first* password.
   - From then on, Next says "Saving setup progress failed. Retry before continuing." and Exit says "Setup progress could not be saved. Retry Exit setup." Only `Ctrl+Q` gets out.
   - On the next launch both passwords fail and the app opens in recovery mode [protect-summary-04].
4. No screen can change the password or turn encryption off. Protect's "you can enable this later in Settings ▸ Privacy & Security" points at a read-only page that says "Credential mutation: not available yet - password-gated flow required" [protect-summary-03].
5. With encryption on, Riley's developer launch `python -m tldw_chatbook.app` exits with code 1 before any UI, on 2 of 2 relaunches (``NoActiveWorker: push_screen must be run from a worker when `wait_for_dismiss` is True``). Only the packaged `tldw-cli` works; it asks "Configuration password: " before the TUI starts [protect-summary-01].
6. A wrong or forgotten password drops into recovery mode on the Backup & Restore screen. The only retry is a relaunch, and there is no "start without saved keys" [protect-summary-02].

---

### (c) Second machine and scripted setup

| Route | Result | Issues |
|---|---|---|
| **Welcome ▸ "Restore a backup"** | Easy to find, and Esc returns to the wizard intact. But it opens on "Create backup". The Inspect pane never names the format; only the Create form says "Use .tldw-backup.zip, or .tldw-backup.zip.age when encrypted." Riley's `config.toml` got "Failed: inspecting — Review the selected file and local folders, then try again. Open the detailed recovery evidence if the problem continues. (backup_operation_failed)". "Create backup" stayed disabled after Review with no reason given, and Riley could not produce an archive in the time allowed. | [entry-exit-handoff-16] |
| **Copy `config.toml` (with `[first_run] setup_completed = true`) and export env keys** | **Works.** It boots straight to Console in 7–8 s. With no keys it shows Console's "Get started" card. Nothing documents it and the wizard never offers it. | [coverage-10] |
| **Env keys only, fresh machine** | The wizard is skipped and the toast says "you're ready to chat", but the OpenAI default is blocked. With only `ANTHROPIC_API_KEY`, Console's "Set up provider" opens Settings preselected on OpenAI and never mentions the Anthropic key it found. | [coverage-04] [new-coverage-power-03] (newly found) |
| **`TLDW_CONFIG_PATH`** | Works. It is documented under README "Advanced profiles" but absent from `--help`. | [coverage-10] |
| **CLI flags** | `tldw-cli --help` lists `--serve --host --port --web-title --debug --focus`, plus `recovery` subcommands. There is no `--config`, `--no-splash`, `--setup` / `--no-setup` or `--non-interactive`. | [coverage-10] |
| **Carrying an encrypted config** | `tldw-cli` asks once, before the TUI starts. A miss drops into recovery. `python -m` crashes. | [protect-summary-01] [protect-summary-02] |
| **Same providers on every machine** | One provider per run, so each machine needs N re-runs or hand edits. | [coverage-08] |

**The right-sized contract,** in the verifiers' order:

1. Ship docs first: a "Setting up another machine" section in the User Guide (copy `config.toml`, `TLDW_CONFIG_PATH`, the `[first_run]` keys, env-key behaviour) and a `--help` epilog naming `TLDW_CONFIG_PATH`.
2. Add `--config PATH` and `--no-splash`. Any `--no-setup` applies to one launch and never writes `setup_completed` silently.
3. Make Restore open on Inspect, name the archive format, recognise a `.toml` and say "That's a settings file, not a backup archive", and show why Create is disabled.
4. Defer `tldw-cli setup --provider … --yes` until users ask for it. If it is built, it must go through `persist_provider_setup` and take keys from env or stdin only, never from the command line. "Import settings (config.toml)" is its own task, because of the secret-handling risk.

---

### (d) Efficiency metrics

| Metric | Measured | Read | Issues |
|---|---|---|---|
| Cold launch to first screen | About 12 s lightly loaded (splash logs `duration: 7.0s`). 10–16 s across five cold starts. 25 s or more under load. | Heavy for a tool relaunched daily | [entry-exit-handoff-17] |
| Splash cost per launch | **About 7 s.** Any key cuts it to 2.5–2.8 s, but no hint says so; the live Riley run concluded it could not be skipped. Setup cannot turn it off. | The skip exists but nobody can find it | [new-coverage-power-01] (newly found) [coverage-15] |
| Second launch after setup | About 12 s to Console, no nag. 7–8 s with a seeded config. | Calm | (none) |
| Opening the wizard | Palette: 7 keys. Settings: about 11 keys. | Palette is fine; Settings is buried | [cross-cutting-12] |
| Fresh install with env keys | **0 keys**: the wizard is skipped | Fast, to a blocked default | [coverage-04] [cross-cutting-01] |
| **Quick with an env key, minimum** | **7 keys** after the palette (14 from Console): `Enter` `Space` `Ctrl+N` `Enter` `Ctrl+N` `Ctrl+N` `Enter`. This path accepts the prefilled gpt-5.6-terra (blocked on every send) and writes the PocketTTS endpoint on Voice. The Summary then claims "✓ Voice — PocketTTS (default voice)". | The minimum path ends in two defects | [cross-cutting-01] [voice-speech-01] [cross-cutting-02] |
| Full, keyboard only | About 75–90 keys, including the provider hunt | (none) | (none) |
| Provider list navigation | 60 rows + 4 headings, 5 visible. OpenRouter = 32 `Down` or about 7 `PageDown`. No type-ahead. With keys exported, each arrow press over a row whose key is present starts discovery. | The costliest step per keystroke | [provider-11] [cross-cutting-06] |
| Model picker limits | The first 20 of 466 (OpenRouter) or 137 (OpenAI), in raw API order, in a 5-row box. Silent wrap, no count or filter. 10 of the 20 OpenAI rows can't chat. | The model can't be found and the recommendation can't be trusted | [model-05] [model-02] |
| Wait per Next | 2.5–4 s in the review run (about 20 s per Quick run). Verification attributes most of that to machine load; the measured post-Next tracker update on dev is about 0.25 s. The long waits are by design: discovery up to about 8 s and the Voice save up to 30 s, **with no busy cue**. | Needs a busy line, not more speed | [cross-cutting-14] [cross-cutting-10] |
| Steps that do nothing | Notes (always), plus Protect for env-key users: 2 of 11 Full steps | Overhead | [fulltrack-05] [coverage-16] |
| **Time to first reply** | Local, with the server running: about 100 s best case, including the splash (stub OpenAI-compatible server; no real local runtime was installed). Cloud: of five providers walked, **only Anthropic** produced a reply. OpenAI, Gemini, OpenRouter and Moonshot all finished setup with ✓ and failed on the first reply. | The metric that matters fails for 4 of 5 cloud providers | [cross-cutting-01] [gap-01] [gap-02] [gap-03] (all three newly found) |

---

### (e) Persona red flags

Each item is what Riley concludes, ranked by how fast it ends trust.

1. **"✓ means saved, not works."** The Summary ticks a model that is blocked on every send. Console says "Ready · not tested" for it. OpenRouter accepts an expired key. RAG ticks a model RAG never uses. [cross-cutting-01] [fulltrack-01] [new-cross-cutting-01] [gap-01] [gap-02] [gap-03] (last four newly found)
2. **"Next writes even when I touch nothing."** The Voice step overwrites a working endpoint, and the copy calls that skipping. [voice-speech-01] [cross-cutting-03]
3. **"A second password breaks my keys and locks the wizard."** [protect-summary-04]
4. **"My dev launch dies after Protect."** `python -m tldw_chatbook.app` exits with code 1 once encryption is on. [protect-summary-01]
5. **"Re-run doesn't know my config."** Provider, Model, Voice, RAG and the track are not prefilled. Speech never is, and Tools isn't after a restart. [cross-cutting-05] [new-voice-speech-01] [new-full-track-steps-01] (last two newly found)
6. **"Browsing is committing."** Each arrow press selects, starts discovery and wipes a prefilled model. [cross-cutting-06] [model-04]
7. **"'(recommended)' is noise."** It is API row 0, sometimes a text-to-speech model. [model-02]
8. **"I can't see what changed, and I can't undo it."** [cross-cutting-09] [protect-summary-13]
9. **"The tool posture is misstated."** The step says "Everything is off by default", while three hidden gates and a built-in MCP server are on. [fulltrack-02] [new-full-track-steps-02] (newly found) [entry-exit-handoff-24]
10. **"Exits don't do what they say."** "Continue" restarts at Welcome. A Settings cancel lands on Console. `Ctrl+Q` skips the guard. `Ctrl+B` is the tmux prefix. [cross-cutting-08] [cross-cutting-05] [gap-05] (newly found) [cross-cutting-15]
11. **"One provider at a time, no endpoint override, and an env key hides the key field."** [coverage-08] [provider-19]
12. **"The only restore path takes an archive I couldn't create."** [entry-exit-handoff-16] [coverage-10]

---

### (f) Verdict: does the wizard serve an expert?

**Not yet.** On the fast path, the wizard is fast but ends at a blocked first chat. Full control is not offered: one provider per run, a model list capped at 20, and no splash, base-URL or cloud speech-to-text choices. Re-running clobbers: the Voice step writes on Next, Exit keeps every write, and encryption can be corrupted. Riley can still get a working setup, but by learning which parts of the wizard to avoid.

The foundations are why this is a fixable problem rather than a rebuild. The no-clobber model already exists in three steps. It has not been applied to the step that needs it most, and the re-run path does not use it at all.

**What works well. Keep it.**

- **Env-key detection without writing keys.** "Found OPENAI_API_KEY in your environment; nothing to store." For the credential, config records only `credential_source = "environment"`. A grep of every home, config and log found no key.
- **Back and Forward keep every choice.** The OpenAI voice, gpt-5.4-mini and the embedding pick all survive. Back from Model keeps the staged key. A crash at Voice resumes at Voice with ✓✓✓ restored.
- **The speech install review.** Before a byte moves it shows the source repo, pinned revision, licence, "Contents: 4 files, 630.6 MiB", destination, free space, and SHA-256 verification.
- **The no-clobber model, where it exists.** Tools, Appearance and Speech write deltas only. Provider and model are one atomic, compare-and-swap-guarded write. `commit_config` touches only wizard-owned sections.
- **Live theme preview.** It reverts on Exit, Skip or a crash, and marks the saved theme "(current)".
- **Voice Advanced disclosure.** Endpoint, auth, model, voice, format and speed. The OpenAI path verifies with a real sample using the env key.
- **The Summary reads back from disk.** When Provider is ✗, it swaps "Start chatting" for "Review provider setup", which returns inside the wizard with state intact.
- **Predictable chrome.** The palette entry, an accurate key-hint line, the double-Esc guard, and an Exit dialog that is honest about what is already saved.
- **Disciplined leave-outs.** Sampling, system prompt, budgets, hooks, MCP exposure and `users_name` correctly stay out of setup.
- **A route that already works.** A seeded `config.toml` plus env keys gives a working second machine today.

**What it takes to earn Riley,** using the verifiers' smaller fixes where they proposed them:

| # | Fix | Closes | Effort |
|---|---|---|---|
| 1 | Delta-commit Voice: an untouched Next writes nothing. Add a required invariant test: Next through every step on a populated config leaves `config.toml` byte-identical apart from `[first_run]`. | [voice-speech-01] [cross-cutting-03] | M (ship the untouched-step gate first) |
| 2 | Fix the model catalog: drop the stale OpenAI 4096 "API default", add `gpt-5*` context windows, and cap the reservation for estimated windows. Then add a Summary "Ready to chat?" row computed offline. | [cross-cutting-01] [new-cross-cutting-01] (newly found) | Small catalog fix, then M |
| 3 | Refuse `enable_config_encryption` when encryption is already on, and validate before writing. Move focus to Next after success. Route `python -m` through `startup_preflight`. | [protect-summary-04] [protect-summary-01] | S |
| 4 | Re-run: pass `cancel_to_console=False` from Settings, read prefill from the full TOML, mark "(current)" on Provider, Model, Voice and RAG, and make **Done** the primary exit. | [cross-cutting-05] [new-full-track-steps-01] (newly found) [protect-summary-14] | S → L |
| 5 | Provider list: arrows only move the highlight (`Enter`/`Space` select), discovery is debounced, a filter input sits above the list, and a "Ready on this machine" group shows "key in OPENAI_API_KEY". Record every env-keyed provider automatically. | [cross-cutting-06] [provider-11] [coverage-08] | M–L |
| 6 | Model: turn "Or enter a model name" into a filter over all ids, show "Showing 20 of 466 — type to filter", drop non-chat ids before the 20-row cut, and give a soft "not in list" warning. | [model-05] [model-02] [model-06] | M |
| 7 | When an env key exists, offer a key-source choice: "Use OPENAI_API_KEY from your environment / Store a different key for this app". Leave the base URL to Settings. | [provider-19] | M |
| 8 | Second machine: the docs section, a `--help` epilog, `--config PATH` and `--no-splash`. Restore opens on Inspect and recognises a `.toml`. | [coverage-10] [entry-exit-handoff-16] | M (docs first) |
| 9 | Show "Press any key to skip" on the splash. Add "Startup animation: Off / Short / Full" to Style. | [new-coverage-power-01] (newly found) [coverage-15] | S hint, M step |
| 10 | Add `confirm_quit` to the wizard. Teach "← Back or Ctrl+B" plus `Alt+←`. | [gap-05] (newly found) [cross-cutting-15] | S |

Fixes 1–4 are the trust floor: without them, Riley cannot safely run setup a second time. Fixes 5–10 make it fast and complete.


---

## 3. Heuristic evaluation

**Scope.** Both tracks: Quick (Welcome, Provider, Model, Voice, Protect, Summary) and Full (adds RAG, Speech, Tools, Notes, Style). The scope also covers the handoff into the first chat, because that is where the wizard's promise is either kept or broken. **Basis.** The 151-entry verified register, the four persona walkthroughs (Sam: cloud, first-time; Jo: local, first-time; Riley: power user; plus the resilience run), the 18 gap checks and the 80 verified positives. Ids in brackets point to the register. Ids marked *(newly found)* surfaced during verification and are still being re-checked.

**Verdict: 16 / 40, Poor.** The foundation is sound. Secrets never reach disk unless the user saves them, the provider and model are written atomically, resume works, waits are bounded, and the exit dialogs are safe. The outcome is not sound. The wizard has one job, a working first chat. Of the five cloud providers walked end to end, only Anthropic produced one. OpenAI, Gemini, OpenRouter (with an expired key) and Moonshot all finished setup with ✓ and then failed on the first reply [cross-cutting-01], [gap-01] *(newly found)*, [gap-02] *(newly found)*, [gap-03] *(newly found)*.

---

### 3.1 Nielsen's 10 heuristics

| # | Heuristic | Score | Strongest evidence | Issues |
|---|---|:-:|---|---|
| 1 | Visibility of system status | **1** | Every status that carries weight is wrong. The Summary shows "✓ Provider — openai" and "✓ Default model — tts-1-hd-1106" with "Start chatting" focused, and the first send is then blocked. Provider turns green ✓ right after "✗ The connection was refused - nothing is listening at that address." Every successful handoff opens with the false toast "Provider settings changed before Console opened. Review setup and try again." Console then calls a model it refuses to send to "Ready · not tested". What does work is orientation: with "Step N of M" and the labelled Quick tracker, Sam was never lost about *where* they were, only about *what had happened*. | [cross-cutting-01] [provider-03] [cross-cutting-02] [entry-exit-handoff-04] [new-cross-cutting-01] *(newly found)* [coverage-04] |
| 2 | Match between system and real world | **2** | "(recommended)" is just the first id the API returned. On OpenAI that was a text-to-speech model, and users read the label as the app's judgement. The Provider step answers a paste box that is right there on screen with "Set OPENAI_API_KEY or add api_key under [api_settings.openai]." The language is plain where it works: the Welcome copy, the key help line, and "Start it (ollama serve), then Retry". | [model-02] [cross-cutting-07] [entry-exit-handoff-03] [protect-summary-12] [voice-speech-12] |
| 3 | User control and freedom | **2** | Back and Forward keep every choice, and the exit dialogs focus "Keep going". But there is no Skip control: a mouse user who picks a keyed provider without a key is stuck, because "Retry with Next" can never succeed. Every Next commits and Exit undoes nothing, so an untouched Voice step overwrites a working TTS config with no rollback. No reachable UI can turn encryption off or change the password, and a forgotten password locks the user out of the whole app. A failed re-encryption makes both Next and Exit setup fail. | [provider-01] [cross-cutting-03] [cross-cutting-09] [voice-speech-01] [protect-summary-03] [protect-summary-02] [gap-05] *(newly found)* |
| 4 | Consistency and standards | **1** | The chrome is consistent; the verbs are not. Next skips, saves defaults or refuses, depending on the step. Enter is advertised as "next" on every step, but it selects, tests, skips the provider or saves instead. The provider list and the model list on adjacent steps use different selection models. The tracker and the Summary show different glyphs for the same state. In a wizard, Next and Enter are the product. | [cross-cutting-03] [cross-cutting-04] [cross-cutting-06] [cross-cutting-02] [cross-cutting-11] [provider-06] |
| 5 | Error prevention | **1** | The default keystroke destroys work: Next or Enter on an untouched Model list silently discards the provider and key just entered. Non-chat models are offered as chat defaults and accepted. Typed model ids are never checked, and typos save silently. OpenRouter, Hugging Face, NVIDIA NIM and Novita accept any key. Setting a second password strands keys that are already encrypted. What works: the "Continue anyway?" gate with "Keep editing" focused, and the atomic provider+model write. | [model-01] [model-02] [model-06] [gap-01] *(newly found)* [protect-summary-04] [cross-cutting-01] |
| 6 | Recognition rather than recall | **2** | The provider list puts 60 providers in a 5-row box with no search and no "key found" badge. The model list shows 20 of up to 466 ids with no count or search. The only skip on a keyed provider (Enter in the empty key field), and the only keyboard route to "Use this server" (Shift+Tab), have to be known in advance. Re-runs don't show the current provider, model, voice or RAG choice. What helps: the labelled Quick tracker and the key help line. | [provider-11] [model-05] [provider-01] [provider-02] [cross-cutting-05] [cross-cutting-10] |
| 7 | Flexibility and efficiency of use | **2** | Quick and Full tracks, keyboard–mouse parity, an env-key path that needs zero keystrokes, and a 7-key palette entry. But there is no type-to-filter anywhere and only one provider per run. The big three providers have no base-URL override. There is no CLI or headless setup, and a re-run cannot jump to a single step. | [provider-11] [coverage-08] [provider-19] [coverage-10] [entry-exit-handoff-27] [cross-cutting-12] |
| 8 | Aesthetic and minimalist design | **2** | On large terminals the lists stay 5 rows tall above 30–40 empty rows, while at 80×24 the frame takes 14 of 24 rows. Some steps do nothing (Notes always, Protect for keyless users). 32 presets show a Test button that never enables, legacy-alias duplicates sit in the main list, and Provider stacks up to three contradictory status lines. Voice's Advanced disclosure is the pattern to copy. | [a11y-04] [a11y-01] [fulltrack-05] [protect-summary-06] [provider-06] [provider-12] [cross-cutting-07] |
| 9 | Recognize, diagnose, recover from errors | **2** | The local endpoint errors are the best copy in the product: "✗ The connection was refused - nothing is listening at that address. Start the server, or check the endpoint and port." "Review provider setup" recovers inside the wizard. The errors that matter most fall short. The blocked first send is jargon with no "Switch model" action. A retired Gemini model shows as "provider unavailable. Status: 404." Azure users are told "API key required" after pasting a key. The Summary blames "no credentials or saved endpoint" for a key the wizard itself dropped. | [entry-exit-handoff-03] [gap-06] *(newly found)* [provider-05] [model-01] [model-07] [voice-speech-03] |
| 10 | Help and documentation | **1** | The "change it later" pointers are wrong more often than right. "you can enable this later in Settings ▸ Privacy & Security" points to a read-only page. "You can continue setup any time from Settings ▸ Diagnostics" leads to a restart at Welcome. Speech has three different homes. The User Guide contradicts the shipped wizard in about seven places. Only 8 of 45 keyed providers say where to get a key, and nothing explains what a local server is. | [protect-summary-03] [cross-cutting-08] [voice-speech-16] [cross-cutting-16] [provider-25] [provider-14] |
| | **Total** | **16 / 40** | **Poor (12–19): "Major UX overhaul required; core experience broken."** | |

**Reading the score.**

- **Why it lands below the usual 20–32.** The rubric's definition of Poor is "core experience broken", and on the default path that is the measured result. OpenAI is the first row under Popular and the template default. Following the wizard's own recommendation on that path ends in a blocked first message. Sam's first working reply came 621 s after launch (wall clock, including capture pauses), and only after about 20 extra actions to pick an older model by hand in Console [cross-cutting-01]. The local path (about 100 s to a first reply, best case, including the splash) and the Anthropic path would score higher. The score reflects the path most users are steered onto.
- **What keeps every heuristic off zero.** The 80 verified positives are real:
  - Keys are masked and never written to disk unless saved.
  - Env keys are named and never stored.
  - Writes are atomic and use compare-and-swap.
  - Tools, Appearance and Speech write only what changed.
  - Resume lands on the right step.
  - Waits are bounded and honest.
  - The Summary reads back from disk.
- **Where the points are.** Three root-cause clusters account for most of the four 1s (H1, H4, H5, H10):
  - "Commit-on-Next overloads 'Next' and keeps no draft ledger"
  - "Status indicators not grounded in verified, in-effect state"
  - "No end-to-end 'first chat works' contract across Summary and handoff"

  Fixing those three moves H1, H4 and H5 together. H10 needs a separate pass on destination strings and the guide (cluster: "Names, destinations and 'where it lives' are hard-coded at each call site").

---

### 3.2 Cognitive load

#### Checklist

| Check | Result | Evidence |
|---|:-:|---|
| Single focus | ✗ | Provider shows up to three contradictory statuses at once: "Couldn't discover models for Anthropic. You can continue anyway.", "Key staged — it will be checked when you continue. Connection testing is unavailable for this provider." and the pinned "The last connection check failed: this API key was rejected. Update it, then continue." [cross-cutting-07]. On the local path, the found-server banner, "Detected endpoints" and "Couldn't discover models for llama.cpp" appeared together (live-firsttime-local, capture 31). |
| Chunking (≤4 per group) | ✗ | "Popular" holds 4, which is right. "Cloud" then holds 42 [provider-11]. The model list holds 20 [model-05] and Speech › Language holds 25 [voice-speech-12]. |
| Grouping | ~ | The provider list is grouped into Popular / Cloud / Local / Other. Speech puts its action buttons above the Language and Precision choices that decide what gets installed [voice-speech-12]. The Summary rows are one undifferentiated block above five equal-weight exits [protect-summary-16]. |
| Visual hierarchy | ✗ | The most important message, the pinned error, measures 2.74:1, sits flush-left and can be up to 16 rows from the field. ✓ and ✗ lines share the helper-text grey [provider-16]. Secondary buttons render as bare bold text [a11y-10]. "Test and Hear" uses the same primary blue as Next [voice-speech-06]. |
| One thing at a time | ✓ | The step sequence does break setup into decisions. One exception: Model shows two competing defaults, a "(recommended)" row and a pre-typed `gpt-5.6-terra` [model-04]. |
| Minimal choices (≤4) | ✗ | Nine decision points exceed four options (table below). |
| Working memory | ✗ | Ten memory bridges (list below). |
| Progressive disclosure | ~ | Done right: Voice's "Advanced — endpoint, model & output" and the collapsed Authentication section. Done wrong: Speech shows three expert setup paths as peers [voice-speech-12], and legacy aliases sit in the main provider list [provider-12]. |

**Result: 5 fail, 2 partial, 1 pass. High cognitive load (4+ failures is the critical band).**

#### Decision points with more than four visible options

| Decision | Step | On screen at once | Behind it | What makes it heavy | Issues |
|---|---|---|---|---|---|
| Which provider | Provider | 5 rows: "Popular / OpenAI / Anthropic / Ollama / llama.cpp" | 60 providers and 4 headings. The Local group comes after 42 cloud rows. 5 legacy aliases and "Custom OpenAI-compatible #2" | No type-to-filter, no descriptions, and no "key found" or "server running" badge. Each arrow press selects the row and starts discovery. OpenRouter is 32 Down presses away, or about 7 with PageDown, which nothing advertises | [provider-11] [provider-12] [coverage-08] [cross-cutting-06] |
| Which model | Model | 5 rows | The first 20 of 137 (OpenAI) or 466 (OpenRouter) ids, in raw API order | 10 of OpenAI's 20 rows cannot chat. No count, search or metadata, and the list wraps silently. A second, different default is pre-typed in the field below | [model-05] [model-02] [model-04] |
| How to set up dictation | Speech | 3 setup paths, a 5-row Language list and Precision | 25 languages, INT8/F32, a 632.8 MiB download | The actions sit above the choices they act on. ONNX, INT8, GGUF and transcribe.cpp are never explained | [voice-speech-12] [voice-speech-15] |
| Which embedding model | RAG | 5 rows | 14 template rows, including a placeholder and the malformed "bge-\*-en-v1" ids | Raw ids, nothing distinguishes a local download from a paid API, and the RAG pipeline never reads the pick | [coverage-13] [fulltrack-01] |
| Which tools | Tools | 5 of 8 rows at 120×40 | 8 shown, plus 4 hidden gates (3 of them on) | Rows are 3 lines tall and overflow. There is no "n of 8 on" count and no preset | [fulltrack-12] [fulltrack-02] |
| Theme and splash | Style | 5 of 6 shortlisted themes, plus 10 splash cards | The full theme list and 78 cards | A cosmetic gallery with no preview takes half the step | [fulltrack-09] [fulltrack-13] |
| How to finish | Summary | 5 exits and 2 checkboxes | — | The exits have equal weight and read as a line of prose. "Get to know you after setup" is never explained | [protect-summary-16] [a11y-10] [protect-summary-11] |
| How to resume | "Continue setup?" | 4 stacked, full-width buttons | — | All have equal weight. "Start over" and "Later" render as plain bold text. The dialog never names the step | [entry-exit-handoff-08] |
| Where to go first | Console arrival | 14 nav tabs and an expanded left rail | — | "Agent blocked" and "Context unknown" chips appear right after a successful setup | [entry-exit-handoff-22] |

#### Jargon load per step

| Step | Terms the user has to translate (quoted from screen) | Load | Issues |
|---|---|:-:|---|
| Welcome | "Restore a backup" (no explanation) | Low | [entry-exit-handoff-18] |
| Provider | "OPENAI_API_KEY", "[api_settings.openai]", "endpoint", "Chat URL", "staged", "legacy alias", "Connection testing is unavailable for this provider." | **High**, and it appears *before* the user has done anything | [cross-cutting-07] [provider-12] [provider-21] |
| Model | Raw ids ("tts-1-hd-1106", "gpt-4o-transcribe-diarize"); "model name" / "model ID" / "model-id" for one thing; "/v1/chat/completions" | **High** | [model-02] [model-07] [model-09] |
| Voice | "PocketTTS", "OmniVoice", "Use as default"; under Advanced: endpoint, authentication, format | Medium | [voice-speech-06] |
| RAG | "RAG", "Embedding dependencies", "e5-small-v2", a `pip install` command | **High** | [fulltrack-11] [coverage-13] |
| Speech | "Parakeet v2", "INT8", "F32", "onnx-asr", "transcribe.cpp GGUF"; the install review opens with "TASK-593" and SHA-256 digests | **Very high** | [voice-speech-12] [voice-speech-18] |
| Tools | "approval card", "longer scope"; on Summary, "MCP ▸ Servers ▸ built-in row ▸ Tool gates" | Medium | [fulltrack-02] [protect-summary-12] |
| Notes | "Library → Notes → Add from files…" | Low | [fulltrack-05] |
| Style | "textual-dark", "textual" | Low–medium | [fulltrack-13] |
| Protect | "master password". The same password is called "Unlock Configuration" and "Configuration password" at launch | Low on the step. The gap is missing facts (where the key lives, who can read it), not jargon | [protect-summary-07] [protect-summary-02] |
| Summary | "openai", "llama_cpp", "RAG — off by default — embedding model e5-small-v2" on a track that never showed RAG, "Keep model lists fresh", "Get to know you after setup" | **High** | [protect-summary-12] [protect-summary-17] |
| First chat | "Response reservation and safety margin leave no model input capacity", "mandatory context", "Trace capture blocked", "source device", "chat_with_llm" | **Very high** | [entry-exit-handoff-03] [entry-exit-handoff-06] [entry-exit-handoff-24] |

The jargon peaks at the first real step (Provider) and at the end (the first chat), where users are least able to absorb it. It is lowest on the steps that barely matter (Welcome, Notes). The clearest definition of "provider" anywhere in onboarding appears only *after* the wizard, on Console's Get started card: "A provider is the AI service that answers your messages — for example OpenAI, Anthropic, or a server running on this computer."

#### Working-memory demands

1. **The hidden skip.** On a keyed provider, the only skip is Enter in the empty key field, mentioned at the end of a help line. Mouse users have no skip at all [provider-01].
2. **The hidden primary action.** "Use this server" is reachable only with Shift+Tab or a click [provider-02].
3. **An unexplained glyph.** The amber '!' has no legend, and no step gives its reason [cross-cutting-10]. In the light theme it measures 1.38:1 [gap-09] *(newly found)*.
4. **A numbered-only Full track.** The 11-step tracker shows numbers 1–11 with no names at every width, because the titled version needs 137 columns and the panel is capped at 120. "Step 7 of 11" means nothing [cross-cutting-10] [fulltrack-13].
5. **Invisible staged state.** The provider and key live only in memory until Model commits them, and nothing on screen says so [model-01].
6. **Exact recall of model ids.** When the 20-row cut hides the model the user wants, they must type its id exactly, and a typo saves silently [model-05] [model-06].
7. **Re-runs that forget.** A re-run shows no current provider, model, voice or RAG choice, so users must remember their own configuration [cross-cutting-05].
8. **A read-back the user can't see.** At 80×24 the Summary's read-back window shows zero summary rows [a11y-01].
9. **A buried way back.** Reaching setup from Settings takes about 11 keys, through Troubleshooting ▸ Diagnostics [cross-cutting-12].
10. **Context switches mid-decision.** Voice's "Add API key in Settings" silently ends the wizard [voice-speech-04]. After setup, "Set up provider" opens an expert Settings form preset to OpenAI instead of the wizard [entry-exit-handoff-12].

---

### 3.3 Emotional journey

#### The curve: Sam (first-time, OpenAI, Quick track, following every recommendation)

| # | Moment | What Sam sees | Feeling | Issues |
|---|---|---|:-:|---|
| 0 | Launch | Raw Python warnings, then a random splash ("A JULES PRODUCTION / PRESENTING / TLDW CHATBOOK") with no skip hint. The wizard appears 10–25 s after launch, depending on machine load | ▼ "Is it broken? Who is Jules?" | [entry-exit-handoff-17] [new-coverage-power-01] *(newly found)* |
| 1 | Welcome | "Quick takes about 2 minutes; Full about 10. Everything can be changed later in Settings, and most steps can be skipped with Next — Esc exits setup." | ▲ Calm and oriented | Positive. At 80×24 it opens mid-sentence [a11y-02] |
| 2 | Picks OpenAI | Before typing anything: "Couldn't discover models for OpenAI. Set OPENAI_API_KEY or add api_key under [api_settings.openai]. Or go Back." | ▼ "Did I do something wrong?" | [cross-cutting-07] |
| 3 | Pastes the key | About 100 masked dots with no echo. "Provider settings changed since test; test again." with no Test button on screen | ▼ Doubt at the highest-stakes input | [provider-17] [provider-06] |
| 4 | Presses Enter, as the hint says | "✓ Reached the server, but your chosen model was not in its list. Pick one on the next step." Sam has chosen no model. A second Enter re-tests | ▼ A success that reads as a failure | [new-cross-cutting-02] *(newly found)* [cross-cutting-04] |
| 5 | Model | "tts-1-hd-1106 (recommended)" | ▲ False confidence: the trap looks like help | [model-02] |
| 6 | Voice | PocketTTS is preselected and fails with "Not tested yet — the sample failed. Check the service, then retry." OpenAI then returns "Verified." | ▼ then ▲ | [voice-speech-03] |
| 7 | Protect | The typed password goes nowhere. Then "✓ Encryption enabled." The next Enter reopens "Set up master password" | ▼▲▼ "Did it fail?" | [protect-summary-05] [protect-summary-04] |
| 8 | **Summary: the designed peak** | Tracker ✓ ✓ ✓ ✓ ✓; "✓ Default model — tts-1-hd-1106"; "Start chatting" focused | ▲▲ "Done." | [cross-cutting-01] |
| 9 | Console arrival | Toast: "Provider settings changed before Console opened. Review setup and try again." | ▼ The first thing success produces is an error | [entry-exit-handoff-04] |
| 10 | **First message: the end** | "This request cannot fit the selected model. Response reservation and safety margin leave no model input capacity…", next to "Response accepted; waiting for dispatch." and "Send blocked — resolve response recovery first" | ▼▼ Lowest point: "my key or the app is broken" | [cross-cutting-01] [entry-exit-handoff-03] |
| 11 | Next launch, after Protect | With `python -m tldw_chatbook.app`: a traceback ending in `NoActiveWorker` and exit code 1. With `tldw-cli`: a one-shot "Configuration password: ", where one typo lands in Backup & Restore | ▼▼ An aftermath that rewrites the memory of the whole setup | [protect-summary-01] [protect-summary-02] |

**Jo (local, private) gets a different curve with the same shape.**

- **Peak, then drop.** The strongest *genuine* peak in the product is Jo's: "Found a local endpoint: http://127.0.0.1:9099." appears in under a second. One Enter, as the footer advises, then selects the pre-highlighted OpenAI, and the banner disappears [provider-02].
- **Silent loss.** Next on Model without a pick drops the local provider [model-01].
- **The end, when Jo gets there.** A reply streams within about 2 s. Best case is about 100 s from launch, splash included (stub server). It is a good ending, but it arrives under two false toasts [entry-exit-handoff-04] [entry-exit-handoff-05].
- **Riley's version.** Riley never sees the curve. An exported key skips the wizard, and the toast says "you're ready to chat" one send before the block [coverage-04].

#### Peak and end

The wizard places its peak deliberately at the Summary: a full row of ✓ and a focused "Start chatting". The peak-end rule says users judge the whole experience by its most intense moment and its last one. On the default path those two moments sit two screens apart and say opposite things. The peak promises a working setup; the end refutes it, first with a false warning and then with a jargon block. So the ✓ does more than fail to help: it turns a configuration problem into a trust problem. The register records the cost directly: after a "recommended" model fails, "trust in every later recommendation drops" [model-02]. For users who set a password, the experience has a further aftermath: the next launch [protect-summary-01].

**Fix the end before polishing the peak.** A ✓ is only worth the outcome it predicts. On the Anthropic and local paths the end is already good, a real reply, so stripping the two false toasts from that moment is the cheapest emotional win in the product: both fixes are rated S effort [entry-exit-handoff-04] [entry-exit-handoff-05].

#### Valleys

The valleys cluster in the two places users can least afford them:

- **The credential** (moments 2–4): an error before acting, a paste that can't be verified, and a success line that reads as failure.
- **The finish** (moments 9–11): a false toast, a blocked send, and a lock-out on relaunch.

The middle of the journey (Voice, Protect) is bumpy but recoverable.

#### Reassurance missing at high-stakes moments

The wizard does the hard trust work under the hood and never says so:

- Keys are never written to disk unless saved.
- Env keys are never stored.
- A grep of every profile and log found no key.
- config.toml is mode 0600.

At the moments that need reassurance, the screen says something weaker, something vaguer, or something false.

| Moment | What the user needs to hear | What the wizard says | Gap | Issues |
|---|---|---|---|---|
| Pasting an API key | That the paste is complete, where the key will be stored, and who can read it | "Key staged — it will be checked when you continue." The key is masked with no length or last-4 echo. The plain-text location appears only on Protect and the Summary | Good custody, said nowhere | [provider-17] [protect-summary-07] |
| Checking the key | "Your key works." | A hard-coded "✓ Reached the server, but your chosen model was not in its list." OpenRouter, HF, NVIDIA and Novita get ✓ for any key. Anthropic shows "Found 13 model(s)" next to "Connection testing is unavailable for this provider." | A real check reads as a partial failure; a non-check reads as success | [new-cross-cutting-02] *(newly found)* [gap-01] *(newly found)* [provider-06] |
| Accepting "(recommended)" | "This is a good chat model for you." | The first id the API returned: a TTS model in the cloud walkthrough, and four different picks across four OpenAI runs | Fake reassurance | [model-02] |
| Setting a master password | What it costs, what happens if I forget it, and how to undo it | Dialog: "If you forget it, the encrypted keys cannot be recovered — you'll need to re-enter them." Step: "you can enable this later in Settings ▸ Privacy & Security" | The dialog's warning is honest but understates the cost: a forgotten password blocks the whole app, not just the keys. No UI can turn encryption off or change the password | [protect-summary-02] [protect-summary-03] |
| First launch after encrypting | "Enter your password to unlock." | Module path: a traceback and exit code 1. Packaged path: a one-shot "Configuration password: ". A failure opens Backup & Restore; "Recovery required: configuration_unlock_failed" appears only after Esc | No "wrong password", no retry, no "start without saved keys" | [protect-summary-01] [protect-summary-02] |
| Starting a 632.8 MiB (Parakeet) or 1.1 GB (OmniVoice) download | The size, the time, that I can cancel, and what happens if I leave | The install review lists size, destination and free space, but leads with "TASK-593", hashes and revisions. There is no Cancel, speed or ETA. "Not installed." sits above the progress bar. Leaving mid-download is silent | The facts are there but in the wrong order; there is no way out; the promise ("installed and activated") is broken | [voice-speech-18] [voice-speech-09] [voice-speech-10] |
| Leaving setup midway | What was saved and how to come back | "Steps you've already completed are saved. You can continue setup any time from Settings ▸ Diagnostics." | No list of what was written, and "continue" restarts at Welcome | [cross-cutting-09] [cross-cutting-08] |
| Deciding what the assistant can touch | The real tool posture | "Everything is off by default." The Summary says "Tools — all off" | 3 hidden gates are on, and the first chat can raise an approval card for "chat_with_llm" | [fulltrack-02] [entry-exit-handoff-24] |
| Choosing local for privacy | "No account needed; nothing leaves this computer." | Welcome says only "Chat with cloud or local AI models…" | The privacy story exists elsewhere (the Voice subtitle, consent off by default) but not where the decision is made | [entry-exit-handoff-18] |

---

### 3.4 Per-step scorecard

Legend: ✓ good · ~ mixed · ✗ failing. **Effort** ✓ means low effort for the user. **Honesty of status** asks whether what the step and the tracker claim matches what was actually saved and what will actually work.

| Step | Track | Purpose clarity | Effort | Honesty of status | Recovery | Worst issue |
|---|---|:-:|:-:|:-:|:-:|---|
| 1 Welcome | Both | ✓ | ✓ | ~ | ✓ | [a11y-02] P2: opens scrolled past its own heading at 80×24 and 100×30 |
| 2 Provider | Both | ~ | ✗ | ✗ | ✗ | [provider-01] P1: a keyed provider with no key traps Next and mouse users; only Enter in the empty field skips |
| 3 Model | Both | ✓ | ✗ | ✗ | ~ | [model-01] P0: Next without a pick silently discards the provider and key (close second: [model-02] P0, "(recommended)" is the API's first id) |
| 4 Voice | Both | ~ | ~ | ✗ | ✗ | [voice-speech-01] P1: an untouched Next writes a hybrid PocketTTS/OpenAI config and overwrites a working setup |
| 5 RAG | Full | ~ | ✓ | ✗ | ~ | [fulltrack-01] P1: saves an embedding key RAG never reads, then the Summary reports "✓ RAG" |
| 6 Speech | Full | ~ | ✗ | ✗ | ✗ | [voice-speech-10] P2: leaving mid-download is silent, and the finished model never becomes the default |
| 7 Tools | Full | ✓ | ~ | ✗ | ~ | [fulltrack-02] P1: "Everything is off by default" while 3 hidden gates are on |
| 8 Notes | Full | ✗ | ✓ | ✓ | ✓ | [fulltrack-05] P3: a whole step with no control that configures nothing |
| 9 Style | Full | ~ | ~ | ~ | ✓ | [fulltrack-09] P3: a cosmetic splash gallery with no preview that misreports the current card on re-runs |
| 10 Protect | Both | ~ | ~ | ✗ | ✗ | [protect-summary-04] P0: ignores existing encryption; Enter reopens setup; a second password strands keys |
| 11 Summary | Both | ✓ | ~ | ✗ | ~ | [cross-cutting-01] P0: ✓ and "Start chatting", then the first chat is blocked |
| (Handoff → first chat) | Both | ✓ | ✗ | ✗ | ✗ | [entry-exit-handoff-03] P1: the blocked send is jargon with no fix action, and its panel says "accepted" |

**Reading the columns.**

- **Honesty of status is the story.** It fails on 8 of 11 steps. The only step rated fully honest is Notes, whose entire content is "Nothing is activated during first-run setup."; the one step that does nothing is the one that tells the truth.
- **Purpose clarity is mostly fine.** No step is mysterious except Notes, which has no reason to exist. The problem is not that users don't understand what a step is for, but that they cannot trust what it reports.
- **Recovery fails at the ends of the flow and wherever writes escape the audited commit path.** It fails on Provider (the entrance), on Protect and the handoff (the exit), and on Voice and Speech, whose writes bypass the wizard's audited commit and survive Exit [cross-cutting-09]. The other steps recover acceptably because Back preserves state.
- **Effort concentrates on two lists.** The provider and model pickers are the two highest-effort controls, and both are the same 5-row, unsearchable box [provider-11] [model-05].


---

## 4. Configuration coverage and information architecture

**The owner asked: does the wizard expose every configurable option a user needs?**

No, but adding more options is the wrong fix. The shipped template has 1,234 leaf keys in 59 tables, and the wizard writes about 25 of them (static-config-coverage §2.4). That ratio is right for onboarding. The real problems are these three:

1. **Most of the optional steps don't do what they say.** Five of the seven optional Full-track steps fail in one of these ways:
   - **Voice** writes a config the user never chose [voice-speech-01].
   - **Search & RAG** writes a key that nothing at runtime reads [fulltrack-01].
   - **Notes** writes nothing at all [fulltrack-05].
   - **Tools** describes a tool posture that isn't the real one [fulltrack-02].
   - **Protect** turns on encryption that nothing else can turn off [protect-summary-03].

   Only Speech (Parakeet install) and Appearance write what the user picked, to keys the runtime reads, and nothing more.
2. **A few day-one needs are missing:**
   - an honest path for exported API keys. This is the most common developer setup, and today it skips the wizard and says "you're ready to chat" [coverage-04];
   - a tldw server connection [coverage-06];
   - a way to turn off the 7 s splash or reduce motion [coverage-15];
   - a choice of where the key comes from [provider-19];
   - the location of the data folder [protect-summary-13].
3. **"Change it later" points at homes that are wrong or don't exist.** Encryption is read-only in Settings. Dictation has no Settings home. The analysis model has no editor. "Settings ▸ RAG" is really Settings ▸ Domain Defaults ▸ RAG [protect-summary-03] [voice-speech-16] [coverage-17] [fulltrack-01].

What I recommend:

- **Quick** becomes four steps and ends with a first chat the wizard has actually checked.
- **Full** covers every area it touches. Each optional step defaults to "not now", and an untouched step writes nothing.
- **Everything else** becomes a named link-out on the final screen, and each link lands on a real home.
- **A re-run of setup** is a dashboard, not a replay of the first run.

### 4.1 Coverage matrix

Columns:

- **In wizard?** Which track shows it today.
- **Should it be in setup?**
  - **Quick**: in the four-step Quick track.
  - **Full**: on the Full track only.
  - **Link-out**: a named link from the final screen, or at the point where the user needs it.
  - **No**: leave it out.

**Connect and chat**

| Config area | In wizard? | Where it lives after setup | Should it be in setup? | Why |
|---|---|---|---|---|
| First chat provider and credential | Quick + Full (Provider) | Settings ▸ Providers & Models | **Quick** | This one decision makes or breaks the first run. The foundations hold: Settings uses the same 60-row catalog, env keys are named and never written to disk, and provider and model are saved together as one atomic write. The list is the problem. It has 60 rows in a 5-row box with no filter and no ready badges [provider-11]. It also has five "legacy alias" rows plus "#2" that look like duplicates, and one of them points Ollama at :5678 [provider-12]. |
| Where the key comes from (env var, stored, encrypted) | Partial: the env var is detected only after a row is selected. When one is found, the key field and Keep / Replace / Clear disappear | Settings ▸ Privacy & Security (read-out only) | **Quick**, inline at the key field | When an env key exists, the only input is hidden. An invalid exported key is then a dead end: "this API key was rejected. Update it", with nowhere to update it [provider-19]. Offer "Use OPENAI_API_KEY from your environment" or "Store a different key for this app". Config precedence already lets a stored key outrank the env var. |
| Additional providers | None: one provider per run | Settings ▸ Providers & Models | **Full**, recorded automatically | A power user with three exported keys needs three passes [coverage-08]. Don't build a multi-tick list. Instead, record every env-keyed provider with `credential_source = environment` (no secret is written), keep single-select for the default, and add "Add another provider" to the Summary. |
| Exported `*_API_KEY` on first launch | None: any env key that resolves suppresses the wizard. A 10 s toast says "you're ready to chat" | Console "Get started" card, whose button opens Settings preselected on OpenAI | **Quick-equivalent fast path**, outside the wizard | The toast claims readiness without checking it. With only ANTHROPIC_API_KEY set, Console stays on an unkeyed OpenAI, and "Set up provider" opens "API key source: missing; set OPENAI_API_KEY or paste a local key" [coverage-04] [new-coverage-power-03] (newly found). The toast also lands on top of the model-list consent modal [new-coverage-power-02] (newly found). Make the Get started card env-aware ("Found ANTHROPIC_API_KEY · [Use Anthropic]"). Say "ready" only when the readiness check passes. Never write chat_defaults silently at boot. |
| Default chat model | Quick + Full (Model) | Settings ▸ Providers & Models; Console switcher (Alt+M) | **Quick** | It stays, but it has to be honest. "(recommended)" is just the first id the API returns, often a TTS or realtime model [model-02]. Typed ids are never checked [model-06]. Unknown OpenAI ids resolve to a 4,096-token window, so every send is blocked before it goes out [cross-cutting-01]. |
| Secondary default models (`[analysis_defaults]`, embedding contextualization, MCP chat default) | None | **No live editor**; raw TOML via Advanced Config only | **No control; seed them safely** | Library "Analyze after import" stays on OpenAI · gpt-4o whatever provider setup connects. Its hint then tells an Anthropic user "Set OPENAI_API_KEY or add api_key under [api_settings.openai]" [coverage-17]. In the wizard's atomic commit, seed these only while they still equal the template values. Add one Summary row: "Analysis & summaries — use your default model". |
| System prompt | None | Nowhere: Console never reads `chat_defaults.system_prompt` | **No** | It's a dead key, so a control for it would be a dead control (static-config-coverage row 9, evidence 15). |
| Temperature, sampling, max tokens | None | Settings ▸ Console Behavior | **No** | A newcomer can't improve on the defaults, and experts already have a home for these. |
| Streaming | None | Settings ▸ Console Behavior (`streaming`); per provider in `api_settings.<p>.streaming` | **No control, but the commit should set it** | LM Studio and other Custom OpenAI-compatible users get `stream=False`, because the template ships `streaming = false` [entry-exit-handoff-25]. Write `streaming = true` when committing a local provider. Cloud providers follow task-33620.10. |
| Online model-list refresh | Quick + Full (Summary checkbox) | Settings ▸ Providers & Models | **Quick, conditional** | Keep it: it's asked once and defaults to off. But it refreshes only cloud providers, so for a local-only user the box does nothing [protect-summary-17]. Hide it in that case, and name the providers otherwise. Skip must record the default answer too. Otherwise a modal about "your configured cloud providers" fires right after a skip [entry-exit-handoff-13]. |
| tldw_server connection and runtime source | **None** | Settings ▸ Overview ▸ "Advanced / Diagnostics" (collapsed by default) ▸ "Switch Source / Server" | **Full (optional step), plus a Summary link-out** | This is the tldw client, and PRODUCT.md promises clear local-vs-server status. Yet no step, provider row or Summary row mentions it [coverage-06]. Add a Summary row that is always present ("Runtime — this computer only · connect a server…"). Add a Full-track Server step that defaults to "This computer only". Move the Settings button out of the collapsed section. Don't make it a Welcome question. |

**Identity, data and files**

| Config area | In wizard? | Where it lives after setup | Should it be in setup? | Why |
|---|---|---|---|---|
| User display name (`{{user}}`) | None. The "Get to know you" interview writes a separate record | Settings ▸ Console Behavior | **Full**, one optional field | `user_display_name` drives `{{user}}`, and today only Console Behavior sets it (improvement: optional "What should replies call you?" field). The template default "User" works, so keep it out of Quick. |
| Data folder | None. The Summary shows only the config path | Settings ▸ Storage | **Link-out** (read-only line on the Summary) | Every database lives there, and setup never says where [protect-summary-13]. Show "Data: <dir>" with Copy and a link to Settings ▸ Storage. Never make `users_name` editable in setup, because changing it moves the data folder (`profile_paths.py:86-90`). |
| Config file path | Summary footer | Settings ▸ Diagnostics (full path) | **Keep on the Summary** | Add a Copy action, and name `TLDW_CONFIG_PATH` in `--help` [coverage-10]. |
| Personal profile | Summary checkbox "Get to know you after setup" | My Profile | **Keep on the Summary** | It's a stock checkbox that draws an X in both states, and it is silently dropped on both Library exits [protect-summary-11]. |

**Voice and speech**

| Config area | In wizard? | Where it lives after setup | Should it be in setup? | Why |
|---|---|---|---|---|
| Spoken replies (TTS) | Quick + Full (Voice) | Settings ▸ Speech & TTS | **Full only**, defaulting to "No voice for now". On Quick, a Summary link-out | Next on an untouched step writes `[app_tts]`: a PocketTTS URL, auth "none", and tts-1-hd/shimmer. For users with an OpenAI key, that breaks working TTS on the first run, and the Summary still reads "✓ Voice — PocketTTS (default voice)" [voice-speech-01]. PocketTTS is preselected but needs a server that nobody has running [voice-speech-03]. Voice doesn't belong in a "2-minute" track [coverage-16]. |
| Dictation (STT) | Full (Speech): 633 MiB Parakeet download, a model folder from disk, or a transcribe.cpp GGUF | **No Settings home.** Lab ▸ Models can install and delete; language and provider exist only in raw `[transcription]` | **Full, as an engine choice** | The step has no cloud Whisper, even when an OpenAI key exists, and it can't pick an engine that is already installed [voice-speech-15]. The GGUF picker writes as soon as a file is chosen, and dictation never uses it [voice-speech-13]. Re-run prefill reads a key that doesn't exist, so a saved German default is silently replaced [new-voice-speech-01] (newly found). Start the step on "No dictation for now", preselected. |

**Library, search and notes**

| Config area | In wizard? | Where it lives after setup | Should it be in setup? | Why |
|---|---|---|---|---|
| Document search on/off (`rag_auto_retrieve_on_send`) | **None**, even though the step is titled "Search & RAG" | Settings ▸ Domain Defaults ▸ RAG ▸ "Automatic retrieval: Never / Automatic" | **Full** | This is the one control in the area that actually changes behaviour, and the step never offers it. Even "✓ RAG" never means "my documents are used when I chat" [fulltrack-01]. Keyword retrieval works without the optional extras, so keep the step when they're missing [fulltrack-11]. |
| Embedding model | Full (RAG step) | Settings ▸ Domain Defaults ▸ RAG (the active profile; the built-in "Hybrid Basic" is read-only) | **No. Show the current model read-only, with a link** | The step writes `embedding_config.default_model_id`, which only the wizard and a deprecated window read. The runtime uses all-MiniLM-L6-v2 whatever the user picks [fulltrack-01]. The list includes broken `bge-*-en-v1` rows (an unquoted TOML key), two template samples, and placeholder keys [coverage-13]. Changing the real model means cloning a profile and rebuilding an index, which is not a first-run decision. |
| Embedding API keys | None. The OpenAI rows ship the placeholder "YOUR_OPENAI_API_KEY_OR_LEAVE_BLANK_IF_ENV_VAR_SET" | RAG profile settings | **No** | This goes away with the picker. Fix the template anyway [coverage-13]. |
| Notes folder sync | Full (Notes step, which has no controls and writes nothing) | Library ▸ Notes ▸ Add from files… | **Link-out** | Lasting sync moved to a reviewed Library flow, and the step was left behind as a pointer [fulltrack-05]. |
| Web search engine and keys | None. DuckDuckGo needs no key, and `web_search` is already on | Settings ▸ Web Search | **Link-out from Tools** | The default works. The power-user persona looked for Serper, Exa and Brave keys next to tools (live-power-user §3). One line pointing to Settings ▸ Web Search answers that. |

**Tools and agent**

| Config area | In wizard? | Where it lives after setup | Should it be in setup? | Why |
|---|---|---|---|---|
| Built-in tool gates (8) | Full (Tools) | MCP ▸ Servers ▸ built-in row ▸ Tool gates | **Full** | Keep what works: the copy is shared with the MCP pane and the commit writes only changes. Group the switches as "Look things up" and "Make changes ⚠", with counts [fulltrack-12]. After an app restart, a re-run shows every switch OFF [new-full-track-steps-01] (newly found). |
| Default-ON and ungated tool surfaces | **None**, while the step says "Everything is off by default" | MCP Tool gates covers 4 of them. Nothing covers the always-on tools or the six tools from the app's own built-in MCP server | **Full, as a read-only line, not as switches** | `console.local_tools_enabled`, `ask_user_enabled` and `character_tools_enabled` all ship ON [fulltrack-02]. The built-in MCP server exposes create_note even when "Create note" is off [new-full-track-steps-02] (newly found). The first chat raises an approval card for an internal `chat_with_llm` tool [entry-exit-handoff-24]. A master switch in first-run setup invites a reflexive "all off", which silently removes web search and Watchlists. State the real posture instead of offering the switch. |
| Workspace folders for file tools | None | Settings ▸ Workspaces | **Full, as one hint** | File tools only reach the chat scratch area and Workspace folders [fulltrack-06]. Add one static line under the file rows. No new flow. |
| Per-tool Allow/Ask/Off, MCP servers, agent budgets, hooks | None | MCP ▸ Tools/Permissions; Console Behavior; Hooks; Agents | **No** | These are expert surfaces that already have good homes. Add one link from Tools. |

**Look and feel**

| Config area | In wizard? | Where it lives after setup | Should it be in setup? | Why |
|---|---|---|---|---|
| Theme | Full (Appearance; the tracker calls it "Style") | Settings ▸ Theme (also the Theme row in Appearance) | **Full** | Works well: live preview, an honest revert, and a "(current)" mark. Show display names with a tone tag ("Nord · dark") instead of ids like "textual-dark" [fulltrack-13]. |
| Splash on/off and duration | **None.** 7.0 s on every launch | Settings ▸ Splash Screen | **Full** ("Startup animation: Off / Short / Full") | This is the setting people actually want, and the step offers the card picker instead [coverage-15]. Outside setup, draw "Press any key to skip" [new-coverage-power-01] (newly found), and skip or shorten the splash on the very first launch [entry-exit-handoff-17]. |
| Splash card | Full (10 cards; "Show all cards…" expands to 78) | Settings ▸ Splash Screen | **No. Link out** | It's cosmetic and can't be previewed, and a re-run shows the wrong current card [fulltrack-09]. |
| Reduce motion, ASCII marks | None | Settings ▸ Appearance | **Full.** Add "Reduce motion" to Welcome only once it can apply live | These accessibility settings need to take effect before the animated Console backdrop appears [coverage-15]. Preselect ASCII marks when NO_COLOR or TERM=linux is set. |

**Security, portability, operations**

| Config area | In wizard? | Where it lives after setup | Should it be in setup? | Why |
|---|---|---|---|---|
| Config encryption | Quick + Full (Protect, always shown) | Settings ▸ Privacy & Security, read-only: "Credential mutation: not available yet - password-gated flow required" | **A Full step, plus a Quick Summary checkbox, both shown only when a key is stored** | Setup is the only door in, and there is no door out [protect-summary-03]. Fix the lifecycle before widening the offer: (1) After encrypting, relaunching through the module path crashes with NoActiveWorker and exits 1 [protect-summary-01]. (2) A forgotten password has no recovery path [protect-summary-02]. (3) Setting a second password strands the keys [protect-summary-04]. |
| Backup and restore | Welcome: "Restore a backup" | Backup & Restore screen | **Keep on Welcome, demoted** | This is a real migration path, and Esc returns cleanly to the wizard. But it opens on "Create backup", never names `.tldw-backup.zip`, and fails opaquely when given a config.toml [entry-exit-handoff-16]. Open it on Inspect / restore, and move it after Next in the focus order [entry-exit-handoff-18]. |
| Settings-only import (config.toml), second machine | None | Nowhere. Copying a config.toml with `setup_completed = true` works but isn't documented | **Link-out (docs and CLI), not a step** | Write a "Moving from another computer" doc section. Add `--config PATH`, `--no-splash`, and a `--help` epilog that names TLDW_CONFIG_PATH [coverage-10]. A TOML import with a diff preview is a separate feature with secret-handling risk. |
| Non-interactive setup | None | Nowhere | **No (CLI later)** | Defer `tldw-cli setup --provider … --yes` until there is demand. If it's built, keys must never be passed on argv [coverage-10]. |
| TLS / corporate CA | None | Settings ▸ Network | **Link-out at the point of failure** | A certificate failure reads "could not reach the server" [coverage-19]. Classify these failures and point to Settings ▸ Network. Never offer an inline "disable verification" toggle. |
| Logging and diagnostics | None | Settings ▸ Diagnostics | **No** | There's no decision to make at setup time. The first-run fix runs the other way: send `RequestsDependencyWarning` and optional-dependency notices to the log instead of printing them to the terminal before the TUI starts [entry-exit-handoff-17]. |
| Telemetry and privacy posture | None. `[metrics]` is off and no telemetry exists | — | **No control; one sentence on Welcome** | Scope the promise to local models. Don't claim "nothing leaves your machine" while the wizard still offers cloud TTS and a cloud model-list refresh [entry-exit-handoff-18]. |

**Net change**

- **Into Quick:**
  - a key-source choice at the key field;
  - an honest env-key fast path;
  - a check that the first chat can actually be sent.
- **Out of Quick:**
  - Voice;
  - the Protect step, which becomes a conditional checkbox on the Summary.
- **Into Full:**
  - tldw server;
  - document search on/off;
  - splash off, reduce motion, ASCII marks;
  - dictation engine choice, including cloud;
  - display name;
  - a truthful tool posture.
- **Out of Full:**
  - the Notes step;
  - the splash gallery;
  - the embedding picker;
  - the GGUF picker.
- **New link-outs:**
  - data folder, notes sync, web-search keys, splash card and Workspaces;
  - TLS settings, shown when a check fails;
  - tldw server, from Quick.

### 4.2 Over-reach: what setup holds that it shouldn't, or only half-builds

| Step (track) | What the screen says | What actually happens | Verdict |
|---|---|---|---|
| Voice (Quick + Full) | "Hear replies read aloud — optional. PocketTTS or OmniVoice run locally, no account needed; skip with Next if you don't want voice." | Next writes `OPENAI_BASE_URL = "http://127.0.0.1:8765/v1/audio/speech"` and `OPENAI_AUTH_MODE = "none"` with tts-1-hd/shimmer, and copies them into `[tts_settings]`. That redirects a working OpenAI TTS on the first run. The Summary then reads "✓ Voice — PocketTTS (default voice)" [voice-speech-01] [cross-cutting-02]. | **Move to Full.** Preselect "No voice for now", and write only what changed. |
| Search & RAG (Full) | "Embedding dependencies are installed. Pick a default model, or skip." | Picking a model writes `embedding_config.default_model_id`. The runtime resolves the active profile (hybrid_basic → all-MiniLM-L6-v2) and ignores it. The Summary says "✓ RAG — embedding model: openai-text-embedding-3-small". Auto-retrieve stays off [fulltrack-01]. | **Rebuild as "Search your documents".** Offer the auto-retrieve radio, and show the current model read-only. |
| Tools (Full) | "Everything is off by default. Tools that read or change your files still show an approval card every time they run." | The step shows 8 of 12 gates, and 3 of the hidden ones are ON [fulltrack-02]. On top of that come always-on tools and the built-in MCP server's create_note, chat_with_llm and character tools [new-full-track-steps-02] (newly found). A plain first chat to a local model ships a ~20.8 KB agent system prompt [new-entry-exit-handoff-01] (newly found). | **Keep the switches and replace the claim.** Derive the posture copy from state, add an "Also available (asks first)" line, and build the Summary row from the real catalog. |
| Notes folder sync (Full) | "Nothing is activated during first-run setup." The guide promises "Folder + on/off toggle". | The step has no controls, and `commit()` writes nothing [fulltrack-05]. | **Remove it.** Add a "Sync a notes folder…" link on the Summary. |
| Appearance splash card (Full) | A splash card list: 10 cards, expandable to 78 with "Show all cards…" | The choice is cosmetic and can't be previewed. Every run shows "Surprise me" as the current choice [fulltrack-09]. | **Replace it** with "Startup animation: Off / Short / Full". |
| Protect (Quick + Full) | "Skip to leave keys as plain text (you can enable this later in Settings ▸ Privacy & Security)." | That page is read-only. There is no Skip control. The step shows the encryption pitch to users with no stored key, then "No API keys saved yet — nothing to protect" [protect-summary-03] [protect-summary-06] [protect-summary-07]. | **Make it conditional and state-aware.** Use two explicit buttons, and fix the lifecycle first (see 4.1). |
| Speech GGUF picker (Full) | "Optional; Next skips it." | Choosing a file writes `[transcription.transcribe_cpp] model_path` immediately, and dictation never routes to it [voice-speech-13]. | **Move it** to Library ingest settings or Lab ▸ Models. |
| Welcome (both) | "Full setup — configure everything" | Full can't add a second provider or turn off the splash, and setup offers no way to turn encryption off [entry-exit-handoff-18]. | **Rewrite** as "Full setup — also transcription, search, tools and appearance". |
| Provider list (both) | Five "(legacy alias)" / "(legacy generic)" rows plus "Custom OpenAI-compatible #2" | They look like duplicates. "Ollama (legacy alias)" points at :5678 [provider-12]. | **Hide them** behind "Show legacy entries (5)". Settings keeps them. |

**The rule every step must pass.** A step must:

1. write what the user chose;
2. write it to the key the runtime reads;
3. write nothing when left untouched;
4. derive its Summary row from the runtime's own resolver, not from "visited".

Every failure in the table above breaks at least one of these. They share one root cause, the cluster "Status indicators not grounded in verified, in-effect state". Make the rule enforceable with the proposed invariant test: pressing Next through every step on a populated config leaves config.toml byte-identical apart from first-run bookkeeping (improvement). The Tools, Appearance and Speech commits already behave this way; Voice slipped through.

**Keep what works:**

- one provider catalog, shared with Settings;
- the atomic provider-and-model write;
- env keys that are never written to disk;
- model-list consent asked once, default off;
- a Summary read back from disk;
- the Parakeet install review;
- the deliberate omissions: system prompt, sampling, agent budgets, hooks, MCP exposure, image and video generation, and `users_name`.

### 4.3 Recommended information architecture

Four rules:

1. **Quick has the fewest decisions that lead to a reply the user has seen.**
2. **Full is complete for every area it touches.** Every optional step defaults to "not now" or "keep current" and writes nothing if untouched. A "Finish with defaults" action is available once Model is saved.
3. **Everything else is a named link-out to a real home.** No step should exist only to point somewhere else.
4. **A re-run is a different product.** It's a dashboard, not a corridor.

#### Quick track: 4 steps (today 6)

| # | Step | Contains | Change vs today | Why |
|---|---|---|---|---|
| 1 | Welcome | (1) Quick / Full. (2) A third choice: "Start with my documents or notes — set up AI later". (3) A quiet line after the footer: "Moving from another computer? [Restore a backup]". | The Quick label drops "voice, protection". Adds a documents-first path. Restore moves after Next in focus order. | A documents-first user today has to walk through Provider and Model [entry-exit-handoff-28] (task-28019). The Welcome copy oversells [entry-exit-handoff-18]. |
| 2 | Connect | (1) A "Ready on this machine" group pinned above Popular, built from env keys, stored keys and running local servers, each with a text status ("OpenAI · key in OPENAI_API_KEY", "Ollama · running on :11434 · 3 models"). (2) A filter field and hidden legacy rows. (3) A key-source choice when an env key exists. (4) A key field and a key check. | Today's Provider step, reorganised. | The app knows what is ready before the user acts, but today shows it only after a row is selected [coverage-08] [provider-11] [provider-19]. |
| 3 | Model | (1) Chat-capable rows only, ranked with curated models first. (2) The recommended or current row highlighted. (3) Caption: "This is the model new Console chats start with. Switch any time with Alt+M." | Filter and rank before the 20-row cut. | "(recommended)" is currently API row 0 [model-02]. Typed ids are never checked [model-06]. |
| 4 | Ready (Summary) | (1) Top row: "✓ Ready to chat — OpenAI · gpt-4.1-mini (1M context)" or "✗ Can't chat yet — …" with [Choose another model]. (2) An optional "Say hello" test: on by default for local providers, behind consent for cloud ("Send a one-word test message to OpenAI? Uses a few tokens."). (3) Rows only for steps the user saw. (4) "Data:" and "Config:" lines. (5) The model-list box only when a cloud provider is configured. (6) "Encrypt saved keys with a password…" only when a key was stored in this run. (7) Next-step links. | Voice and Protect leave Quick. The step count stays stable at 4. | See below. |

**Why Quick must end with a first chat the wizard has checked.**

- **Local:** the best case to a first reply is about 100 s, including the splash (stub server; re-confirm on real llama.cpp/Ollama).
- **Cloud:** of five providers walked end to end, only Anthropic reached a first reply. Each of the others finished setup with ✓ and then failed:
  - OpenAI: context window [cross-cutting-01].
  - Gemini: retired default model [gap-03] (newly found).
  - OpenRouter: any key accepted [gap-01] (newly found).
  - Moonshot: "Provider continuation could not be persisted" [gap-02] (newly found).

A Quick track that ends at "saved" measures the wrong thing. The root fixes sit upstream, in the model catalog and the context-window defaults. The Ready row is defence in depth, and it must call Console's own send preflight (context budget plus chat-capability) rather than a wizard copy of it [cross-cutting-01].

**Why Voice and Protect leave Quick.**

- Voice is optional, rarely works on the first try, and writes config on Next [coverage-16].
- Protect was made permanent deliberately ("A stable total beats a shorter one", TASK-21148). Turning it into a Summary checkbox keeps that stable total while removing an empty step for keyless and env-key users [protect-summary-06].

#### Full track: 11 steps (same count, every step real)

| # | Tracker label | Contains | Default if untouched | Change vs today | Issues |
|---|---|---|---|---|---|
| 1 | Welcome | As Quick | — | — | [entry-exit-handoff-18] |
| 2 | Connect | Quick's step. Also records every env-keyed provider automatically; "Add another provider" on the Summary loops back here | Nothing new written | Multi-provider without a checklist | [coverage-08] |
| 3 | Model | As Quick | — | — | [model-02] |
| 4 | Server (optional) | "This computer only" / "Also connect to a tldw server": URL, token, [Test connection] using ServerSwitchModal's probe. The template placeholder token counts as "not configured" | Writes nothing | **New** | [coverage-06] |
| 5 | Search | Title "Search your documents (optional)". "When you chat: Search my Library only when I ask (default) / automatically". The current embedding model, read-only, with a link. The missing-extras line becomes secondary | Writes nothing | **Rebuilt** from "Search & RAG" | [fulltrack-01] [fulltrack-11] [coverage-13] |
| 6 | Tools | The 8 switches in two counted groups. A truthful posture sentence. "Also available (asks first): …", built from `all_tool_gates()`. One Workspace hint. Links to Settings ▸ Web Search and MCP ▸ Tools | Writes nothing | Copy and layout | [fulltrack-02] [fulltrack-06] [fulltrack-12] |
| 7 | Spoken replies (optional) | "No voice for now" preselected. Services labelled by readiness ("OpenAI — uses your OpenAI key"). OpenAI reuses the provider key | Writes nothing | Moved from step 4; writes only what changed | [voice-speech-01] [voice-speech-03] [voice-speech-04] |
| 8 | Dictation (optional) | "No dictation for now" (preselected), "Already available — <engine>", "On this computer — Parakeet (633 MiB download)", "Cloud — OpenAI Whisper (audio is sent to OpenAI)". Advanced: precision, model folder | Writes nothing | Engine choice. The GGUF option moves out | [voice-speech-15] [voice-speech-12] [voice-speech-13] |
| 9 | Appearance | Theme (display names), "Startup animation: Off / Short / Full", "Reduce motion", "Plain ASCII status marks", optional "What should replies call you?" | Writes nothing | Splash gallery out; comfort settings in | [coverage-15] [fulltrack-09] [fulltrack-13] |
| 10 | Protect keys | State-aware. With a stored plaintext key: "Encrypt with a password…" / "Keep as plain text". When already encrypted: "Change password…". With nothing stored: one line ("Nothing to protect — no API key is saved in config.toml"), and the tracker shows "–" | Writes nothing | Conditional copy; slot kept so the count stays stable | [protect-summary-04] [protect-summary-06] [protect-summary-07] |
| 11 | Ready | As Quick, with rows in step order | — | — | [protect-summary-16] |

**Why this order.**

- **Connect and Model come first.** They are the dependency chain and the only required pair.
- **Server comes next.** It decides where the runtime and data live, so it precedes the data features.
- **Search then Tools** reads as "what the assistant can see, then what it can do".
- **Spoken replies and Dictation sit together and are renamed**, so they stop reading as synonyms [cross-cutting-11]. They sit late because both carry downloads. Until a shared download coordinator exists, leaving a step mid-download is silent, and the finished model never becomes the default [voice-speech-10].
- **Appearance comes late** because nothing depends on it.
- **Protect comes last before Ready.** It must follow every step that can store a secret: provider key, server token, voice key.

**Expert speed comes from defaults and an exit ramp, not from fewer steps.** Today a Full run takes about 75–90 keys, with 2.5–4 s of waiting per Next (live-power-user §2). Untouched-means-no-write makes Next safe, and "Finish with defaults" after Model takes an expert from step 3 straight to Ready (improvement). I'm deliberately not turning first-run Full into a hub of cards (improvement "Make the Full track a hub"). That model belongs to the re-run, where the user wants to change one thing (4.4).

**Rejected, on purpose:**

- **A five-way "How will you use chatbook?" router on Welcome.** It taxes the local-first majority, and Connect's "Ready on this machine" group detects the same things without asking [coverage-06].
- **A master tool switch in setup**, for the reason above [fulltrack-02].
- **An embedding-model picker.** Changing the model clones profiles and rebuilds the index [fulltrack-01].
- **A multi-tick provider checklist** [coverage-08].
- **Base-URL overrides on every keyed provider.** Few users run gateways, and the form grows for everyone [provider-19].
- **An "Undo this session" ledger** before Voice writes only what changed [cross-cutting-09].
- **An inline "disable TLS verification" toggle** [coverage-19].

#### What the Ready screen links out to

At 80×24 the Summary's read-back already shrinks to a 3-row window [a11y-01]. Show at most three actions; put the rest under "More ▾" [protect-summary-16].

| Link | Lands on | Shown |
|---|---|---|
| Start chatting (primary only when Ready passes; otherwise "Fix model") | Console | Always |
| Add your first document / Write your first note | Library (these already exist and need no provider) | Always |
| Hear replies aloud… | The Voice step on its own, or Settings ▸ Speech & TTS | Quick |
| Connect a tldw server… | ServerSwitchModal | Quick, and Full when Server was skipped |
| Add another provider | The Connect step, then back to Ready | More ▾ |
| Sync a notes folder… | Library ▸ Notes ▸ Add from files… | More ▾ |
| Web search keys · Tool permissions · Workspaces · Splash card | Settings ▸ Web Search · MCP ▸ Tools · Settings ▸ Workspaces · Settings ▸ Splash Screen | More ▾ |
| Data: <dir> [Copy] · Config: <path> [Copy] | Settings ▸ Storage | Footer |

The env-key persona never enters the wizard. Their "Quick" is the env-aware Get started card in Console ([coverage-04], where the verifier prefers it to a new dialog). It runs the same readiness check and offers "Full setup…" as a secondary.

### 4.4 The "where it lives later" problem

There are three layers. Setup itself has no stable name or home. The Settings homes it names are wrong or missing. And every surface hard-codes its own strings.

**Setup has no stable name or entry point.**

- **Seven names** [cross-cutting-11]. The captured strings are:
  - "Connect a provider", "Set up provider", "Set up Console model";
  - "Enter continue setup" (which opens Settings, not the wizard), "continue setup";
  - "rerun setup", "Re-run setup";
  - "Review setup and try again".

  The Settings button says "Run Setup Wizard"; the Summary and toasts say "Run setup wizard".
- **It's buried.** The re-run lives under Settings ▸ Troubleshooting ▸ Diagnostics, at the bottom of the pane below Validate and Reload Config. Reaching it by keyboard takes about 11 keys (F4, /, diag, Enter, F6, Tab, Tab, Enter). The palette matches "setup", but "api key" and "getting started" return "No matches found" [cross-cutting-12].
- **Three promises don't hold:**
  - The exit dialog says "Steps you've already completed are saved. You can continue setup any time from Settings ▸ Diagnostics." Re-runs actually restart at Welcome on Quick and ignore the saved draft [cross-cutting-08].
  - The Settings hint says "Re-run the guided first-run setup with current values." Provider, Model, Voice, RAG and the track are not prefilled [cross-cutting-05].
  - Console's "Set up provider" and "press Enter to continue setup" open a Settings form preset to OpenAI, not setup [entry-exit-handoff-12].
- **Wrong destination for blockers.** Any "setup-blocked" reason in Console opens the whole wizard at Welcome, "Step 1 of 6", even for a configured user [entry-exit-handoff-27].

**Settings homes that are wrong or don't exist**

| Setting | What setup or the guide says | What's actually there | Fix |
|---|---|---|---|
| Encryption | "you can enable this later in Settings ▸ Privacy & Security" | A read-only page: "Config encryption: disabled", "Credential mutation: not available yet - password-gated flow required". The only other caller is the deprecated, unreachable Tools_Settings_Window | Add an Encryption card with Encrypt / Change password / Turn off. Change password needs a current-password field. Interim copy: "To encrypt later, re-run setup" [protect-summary-03] |
| Dictation | The step and Summary say "Lab ▸ Models". The guide says "`[transcription]` in config.toml — no Settings category owns it yet". The manual-setup route opens Settings ▸ Speech & TTS | Speech & TTS has no transcription controls. Lab ▸ Models only installs and deletes | Add a "Dictation (speech-to-text)" section to Settings ▸ Speech & TTS [voice-speech-16] |
| Analysis model | Nothing, and the not-ready hint asks for an OpenAI key | No Settings editor. The Media viewer has a per-analysis Select; Library import has none | Add an "Analysis model: Same as chat" row in Providers & Models [coverage-17] |
| Document search | "Settings ▸ RAG" (the guide and the missing-extras copy) | Settings ▸ Domain Defaults ▸ RAG, a collapsed group. The active profile is the read-only "Hybrid Basic" | Name the real path [fulltrack-01] [fulltrack-11] |
| Tools | Quick note: "Left at recommended defaults: tools off, RAG off, default theme, notes sync off — each lives in Settings when you want it." Manual setup opens Advanced Config (raw TOML). Guide line 83: "MCP ▸ Servers ▸ Tool gates" | MCP ▸ Servers ▸ built-in row ▸ Tool gates | Use `TOOL_GATES_PANE_PATH` everywhere, and generate the note from the rows [protect-summary-12] [cross-cutting-13] [cross-cutting-16] |
| Notes sync | Manual setup opens Advanced Config. The step's breadcrumb is "Library → Notes" | Library ▸ Notes ▸ Add from files… | Hide manual setup where no page owns the setting. Use one breadcrumb glyph, "▸" [cross-cutting-13] [fulltrack-05] |
| tldw server | Nothing | Overview ▸ "Advanced / Diagnostics", collapsed by default | Move the button into the main Overview body [coverage-06] |
| Splash card | The guide says "Settings ▸ Appearance" | Settings ▸ Splash Screen | Fix the guide row [fulltrack-13] |
| "Review settings" (Summary) | The label implies Settings in general | It opens Providers & Models only and leaves setup unfinished, so the next launch asks "Continue setup?" | Finish setup first, then open; rename it "Open Settings" [protect-summary-08] |
| Welcome | "Everything can be changed later in Settings" | False for dictation, the analysis model, and turning encryption off or changing its password | True once the three missing homes above exist; scope the claim until then |

Several fixes proposed in the register point at "Settings ▸ General", which doesn't exist (verifier notes on [cross-cutting-08] and [cross-cutting-12]). Use Settings ▸ Overview. Fixing the problem must not invent a new home that doesn't exist.

**Fixes, in order**

1. **One setup registry.** Map each setting to its display name, its owning destination (a Settings category, MCP, Library, or a CLI flag) and its route. Step copy, Summary rows (as actionable links), manual-setup routes, palette aliases and the User Guide step table all read from it. A doc-drift test fails when First_Run_Setup.md disagrees. This is the cluster "Names, destinations and 'where it lives' are hard-coded at each call site". Start from [voice-speech-16]'s single `STEP_LATER_HOME` mapping, fold it into TASK-33623, and gate it with TASK-32589's claim check. It also closes the guide's roughly seven solid contradictions with the shipped wizard [cross-cutting-16].
2. **Fill the three missing homes:** the Encryption card, the Dictation section and the Analysis model row. Until each ships, setup copy names "re-run setup" or the raw path, not a Settings page that can't do the job. Where no page owns a setting, hide "Use manual setup" instead of opening raw TOML [cross-cutting-13].
3. **One name.**
   - The task is "Setup", with the verbs "Run setup" and "Resume setup".
   - Tracker labels are the stems of the step titles: "Spoken replies", "Dictation", "Search", "Appearance", "Protect keys".
   - Every optional step gets the same suffix.
   - Breadcrumbs use only "▸", and body copy uses "chatbook" [cross-cutting-11].

   On the Full track, tracker titles never render at the 120-column panel cap, so the renaming has to ship with the tracker fix to be visible [fulltrack-13] [cross-cutting-10].
4. **One entry contract**, `open_setup_wizard(origin, resume, start_step)`:
   - Every surface calls it: boot, Settings, palette, the Console card, the Home card and the composer.
   - It resumes a valid draft, deep-links to a named step, and returns to the origin on cancel or Done (improvement "One setup entry contract") [entry-exit-handoff-12] [cross-cutting-05].
   - Console's "no provider" blocker opens Connect. Every other blocker links to its own control [entry-exit-handoff-27].
5. **Make it findable.**
   - Put "Run setup" (or "Resume setup" while a draft exists) on Settings ▸ Overview and in the Providers & Models empty state. Diagnostics becomes the secondary home.
   - Add the palette aliases "api key", "provider setup", "getting started", "onboarding" and "first run" [cross-cutting-12].
6. **Turn the re-run into "Review your setup".**
   - One screen built from the existing Summary rows, showing each area's current value with a Change action. Change opens that single step and returns.
   - Untouched areas are never written.
   - The primary action is "Done", which returns to wherever setup was opened from [protect-summary-14] [cross-cutting-05] (improvement "Re-run as a 'Review your setup' dashboard").
   - Console's readiness deep links land here too, so "fix my model" opens Model, not Welcome.


---

## 5. Solutions, structural fixes and improvement roadmap

The register lists 151 verified defects. They have far fewer causes. Every issue maps to one of fourteen root-cause clusters, and ten structural fixes plus three enabling refactors cover all of them. Each issue is assigned to exactly one primary fix below, so nothing is double-counted or left out.

The wizard got here by patching symptoms. The review's own history shows it: each honesty fix (TASK-2724, 25818, 21143, 32959) added one more overlay on top of the last, 14 Done wizard tasks still have unticked acceptance criteria, and `FirstRunSetupWizard.py` is at 10,854 lines against a 10,404-line ratchet [cross-cutting-17]. Another round of point fixes will just produce another register like this one.

The metric that matters is **time to a first reply**. Today it looks like this:

| Path | Result |
|---|---|
| Local, server already running (Jo) | About 100 s at best: 79 s from Welcome to the first streamed reply for a scripted driver, plus about 20 s of splash (stub OpenAI-compatible server) |
| Anthropic, following the recommendation | Works. This is the only cloud path that met 'Quick takes about 2 minutes' |
| OpenAI, following the recommendation (Sam) | Summary at 467 s. The first reply came at 621 s, and only after hand-switching to gpt-4.1-mini in Console [cross-cutting-01] |
| Gemini, OpenRouter, Moonshot | Setup ends with ✓, then the first reply fails: a retired default model, any key accepted, a continuation-persistence error (newly found: [gap-03], [gap-01], [gap-02]) |

The backlog barely covers this. **122 of 151 issues have no open task, including 4 of the 5 P0s and 18 of the 23 P1s** (see §6).

### Four rules every fix enforces

1. **Done means a reply, not a write.** Setup is complete when a first turn can be sent to the chosen model. A provider and a model id sitting in config is not enough.
2. **Write nothing the user didn't touch; drop nothing the user typed without saying so.** Next on an untouched step writes nothing. A skip never silently discards a key.
3. **Every mark is computed, not inferred.** ✓, ✗, ! and "Ready" come from persisted, verified state, read through the same resolver the runtime uses.
4. **One owner per fact.** Provider facts, step names, "change it later" destinations, download jobs and encryption state each live in one place, and every surface reads from there.

---

### 5.1 The ten structural fixes

| # | Structural fix | What it removes | Issues | P0 / P1 | Effort | Risk |
|---|---|---|---|---|---|---|
| SF1 | "First chat works" is the definition of done | ✓ Summary followed by a blocked first send | 18 | 1 / 7 | S → L, staged | Medium |
| SF2 | Curated, searchable model picker with capability data | '(recommended)' meaning "API row 0" | 9 | 1 / 1 | M, in 3 increments | Medium |
| SF3 | A session draft replaces commit-on-Next | Silent data loss, dead ends, overwrites | 12 | 1 / 5 | M; L for the full ledger | High |
| SF4 | One outcome record per step drives every mark | False ✓, stale and contradictory errors | 15 | 0 / 2 | M | Low |
| SF5 | Provider catalog drives the form; detection is row one | Hand-coded per-provider rules | 20 | 0 / 2 | L, shipped in slices | Medium |
| SF6 | Encryption lifecycle with one owner | Lock-outs, stranded keys | 10 | 2 / 2 | S → M | High |
| SF7 | Setup-session model and one entry contract | Re-run, resume and exit drift | 17 | 0 / 3 | L; S slices first | Medium |
| SF8 | One input policy: highlight browses, Enter/Space/click selects | Browsing that commits; Enter with five meanings | 6 | 0 / 1 | M | Medium |
| SF9 | Terminal-native frame: adaptive height, state shown as text, safe keys | Unusable 80x24, colour-only state | 15 | 0 / 1 | M | Low–Medium |
| SF10 | Steps earn their place: tracks scoped by relevance | Padded Quick track, no-op Full steps | 13 | 0 / 0 | M → L | Medium |

The three enablers in §2 cover the remaining 17 issues.

#### SF1 — "First chat works" is the definition of done

**Flaw.** Each step validates one local fact: the key was accepted, or the id is in the list. The Summary then ticks rows because config contains values. Nothing asks Console's send preflight whether a turn can actually go out. The OpenAI path ends on '✓ Default model — tts-1-hd-1106' and 'Start chatting'. The first send then answers 'This request cannot fit the selected model…', because `PROVIDER_CONTEXT_WINDOWS['openai'] = 4096` minus `max_tokens = 4096` leaves zero input capacity [cross-cutting-01]. The env-key fast path skips the wizard and says "you're ready to chat" without checking anything [coverage-04]. Every successful handoff raises 'Provider settings changed before Console opened. Review setup and try again.' [entry-exit-handoff-04]. The Summary says '– Tools — all off', yet the very first chat asks approval for an internal `chat_with_llm` tool [entry-exit-handoff-24].

**Fix.** Follow the verifier's order: fix the catalog first, add the wizard gate second.

1. **Repair the catalog (S).** Drop or raise the stale OpenAI 4096 "API default", so unknown OpenAI ids fall back to the estimated window. Add `gpt-5*` and `chatgpt*` patterns and a context window for gpt-5.6-terra. When a window is only estimated, cap the response reservation, for example at min(max_tokens, window/4). Add a test that every shipped default resolves to a window larger than its provider's max_tokens. Replace the retired Gemini template default, and add a nightly check that template defaults still appear in the live lists [gap-03] (newly found). This step alone fixes the env-key path, which never sees the wizard.
2. **Add a "Ready to chat?" row to the Summary (M).** Run Console's own preflight offline against the committed config: context-window and request-capacity resolution, plus chat-capability classification. Call the shared code; never copy it into the wizard. The row reads either '✓ Ready to chat — OpenAI · gpt-4.1-mini (1M context)' or '✗ Can't chat yet — gpt-5.6-terra's context size is unknown, so every message would be blocked.' with [Choose another model] and [Set context size…]. 'Start chatting' is the primary action only when the check passes.
3. **Check keys where a model list proves nothing (M).** OpenRouter, Hugging Face, NVIDIA NIM and Novita serve their model list to any key. Probe an authenticated endpoint instead (OpenRouter `/api/v1/key`, HF whoami-v2). Until that ships, say that the key is checked on the first chat, and show '– Provider — openrouter (key not verified)' [gap-01] (newly found).
4. **One finish routine for every exit, including the env-key path (M).** It:
   - fences the handoff on values, not on the config generation [entry-exit-handoff-04];
   - turns streaming on for local OpenAI-compatible providers [entry-exit-handoff-25];
   - points analysis defaults at the chat model only while they still equal the shipped template [coverage-17];
   - drops `_UNAVAILABLE_DIRECT_TOOLS` from the agent catalog [entry-exit-handoff-24];
   - says "ready" only when readiness agrees [coverage-04].
5. **First-reply failures name the cause and offer the fix.**
   - Show the provider's own error message, capped and scrubbed [gap-06] (newly found).
   - For a refusal before dispatch, say '…context size isn't known…' with [Set context size] [Switch model], and don't mount the dispatch-recovery panel [entry-exit-handoff-03].
   - Give a cold local model a longer first-token window, and show the elapsed time [gap-07] (newly found).
6. **Calm arrival.**
   - Replace arrival toasts with one receipt line (E9) [entry-exit-handoff-05] [new-entry-exit-handoff-02] (newly found).
   - Say 'Tools off' instead of 'Agent blocked' [entry-exit-handoff-22].
   - Backfill or wait for the trace revision instead of failing the first send [entry-exit-handoff-06].
   - Give plain chat a minimal prompt until tools are enabled, instead of a ~20.8 KB agent preamble [new-entry-exit-handoff-01] (newly found).
   - Show one first-launch notice at a time [new-coverage-power-02] (newly found).
7. **Optional, consented test message (E1).** It catches what an offline preflight cannot, such as Moonshot's continuation failure [gap-02] (newly found).

**Resolves (18):** [cross-cutting-01] [coverage-04] [coverage-17] [entry-exit-handoff-03] [entry-exit-handoff-04] [entry-exit-handoff-05] [entry-exit-handoff-06] [entry-exit-handoff-22] [entry-exit-handoff-24] [entry-exit-handoff-25]. Newly found: [new-cross-cutting-01] [new-coverage-power-02] [new-entry-exit-handoff-01] [new-entry-exit-handoff-02] [gap-01] [gap-02] [gap-06] [gap-07].

**Effort and risk.** S for the catalog, M for the Summary row, M for the finish routine; L once the test message ships. Risk is Medium:
- A wizard-local copy of the preflight would drift from Console, so call the shared code.
- The check must never block the typed-model "set up now, start the server later" path.
- The test message must state its cost and offer 'Skip test' for offline users.

**Backlog.** No task covers the end-to-end check. [entry-exit-handoff-04] → task-33001.10 (To Do, medium); raise it to high, since it reproduced 8 of 8 times. Partial coverage: task-33621.4 ([entry-exit-handoff-03]), task-33621.6 ([entry-exit-handoff-06]), task-33620.6 ([entry-exit-handoff-05]), task-33620.10 ([entry-exit-handoff-25]), task-28018 ([coverage-17]).

#### SF2 — A curated, searchable model picker that knows what each model can do

**Flaw.**
- '(recommended)' is attached to `models[0]` of the raw discovery list, which is cut to 20 entries before anything else happens. OpenAI's order varies between calls, so four fresh runs recommended four different models, one of them a text-to-speech model. 10 of the 20 OpenAI rows cannot chat [model-02].
- The box shows 20 of up to 466 ids in 5 rows, with no count, search or metadata [model-05].
- Typed ids are never checked against the list [model-06].
- A fresh install pre-types the template's gpt-5.6-terra, which competes with the recommendation and is wiped by browsing [model-04].
- Errors render as greyed radio rows, and cloud users are told to start a server [model-07].
- Gemini gets no model list at all [gap-03] (newly found).

**Fix.**

1. **One pure ranking helper (M).** `rank_setup_models(provider, discovered, curated, capabilities) → (recommended | None, ordered, hidden)`.
   - Rank and filter *before* the 20-row slice.
   - Use metadata the app already has first: OpenRouter output modalities, models.dev capability data, Ollama `/api/show`. Fall back to an OpenAI-family denylist (tts, whisper, transcribe, realtime, audio, image, sora, embedding, moderation, babbage, davinci…) only as a last resort.
   - Recommend only the first curated id that discovery returned and that resolves a context window. Local servers and gateways usually have no curated match, so they show no tag.
   - Give the recommendation a reason: 'Recommended for OpenAI: gpt-4.1-mini — fast and inexpensive. You can switch any time in Console (Alt+M).'
2. **Make the free-text field the filter (M).** This is the verifier's simpler alternative to adding a second search box.
   - Typing in 'Or enter a model name' filters all discovered ids. Enter on a value that matches nothing accepts it as a typed id.
   - A non-focusable count reads 'Showing 20 of 466 — type to filter'.
   - Add inline completion, and a soft 'not in OpenRouter's list' warning. Normalise variant suffixes such as `:nitro` and `:latest` before comparing.
3. **Show the current model as a pressed row (S).** On re-runs the list shows 'gpt-5.4-mini   (current)', matching Appearance's existing '(current)'. Prefill only when setup has completed or chat_defaults differ from the template.
4. **Status as text, not as list rows (M).** Put a wrapping status line above the list, with a real Retry button and copy that fits the provider class (local or hosted). This is task-33008.5. Use one term, 'model ID', everywhere [model-09].
5. **Gemini discovery** via `v1beta/models`, filtered to generateContent models. The same call doubles as the key check.
6. **Later:** embed the Console switcher's row renderer (task-14812), so context size and readiness read the same in the wizard, Settings and Console.

**Resolves (9):** [model-02] [model-04] [model-05] [model-06] [model-07] [model-09]. Newly found: [new-model-01] [new-cross-cutting-02] [gap-03].

**Effort and risk.** Three M increments. Risk is Medium.
- Ship this after SF1's catalog repair. gpt-5.6-terra is the curated OpenAI head and *is* in OpenAI's live listing, so ranking first would simply move the failure onto the recommended row.
- Keep validation soft. The typed-model rescue for a server that isn't running yet must keep working.
- Keep two existing guarantees: the list and the free-text field share one source of truth, and saved ids never carry label decoration.

**Backlog.** [model-07] → task-33008.5 (To Do); widen it to cover hosted-provider copy, wrapping and Retry placement. Everything else in SF2 is untracked.

#### SF3 — A session draft replaces commit-on-Next

**Flaw.** Each step's `commit()` decides by itself whether Next means skip, write defaults or refuse.
- Provider persistence hides inside the Model commit. Next on Model without a pick therefore drops the provider and key just entered, and the Summary blames the user: '✗ Provider — no credentials or saved endpoint' [model-01].
- An untouched Voice step writes a hybrid PocketTTS/OpenAI table. That breaks working OpenAI TTS and shows '✓ Voice — PocketTTS (default voice)' [voice-speech-01].
- A keyed provider with an empty key refuses forever with 'Retry with Next, or go Back.', although Welcome promised that 'most steps can be skipped with Next' [provider-01].
- Exit keeps every write, and some writes bypass the audited path [cross-cutting-09].

**Fix.** Ship in stages, in the verifier's order.

1. **Skipping Model keeps the key (M).** Write the staged credential and endpoint to `api_settings.<p>` without touching chat_defaults. The atomic provider+model write exists to keep that pair consistent, and a credential-only write doesn't break it. Confirm with the existing two-button guarded dialog: 'Choose a model' (default focus) and 'Skip — keep the key, no default model'. When only the model is missing, 'Review provider setup' lands on Model.
2. **Untouched means unwritten (M).** Voice commits nothing when the draft matches what is saved and the user neither tested nor ticked 'Use as default'. This copies the delta gate that Tools, Appearance and Speech already use. Then:
   - add 'No voice for now' as the first service, so a PocketTTS server that isn't running is never the preselection [voice-speech-03];
   - prefill re-runs from the raw `[app_tts]` table;
   - make the status copy stop promising a Save button that doesn't exist [voice-speech-06].
3. **A skip wherever Next refuses (M).**
   - Next with an empty key behaves exactly like today's Enter-in-field skip.
   - A visible 'Skip — connect later' button serves mouse users.
   - Refusal copy names something that can work: 'Paste your OpenAI API key above, or choose Skip — connect later.'
   - 'Retry with Next' appears only when a retry can succeed.
   - A skipped provider stays selected with a "skipped" flag, so a key typed after Back is staged normally.
   - OpenAI voice without a key names its in-wizard escapes: pick another service, or 'No voice for now' [voice-speech-04].
   - Speech's GGUF picker holds its path in the draft instead of saving it on selection [voice-speech-13].
4. **Say what Next will do (S).** Label it 'Save & continue' when the step will write and 'Continue' when it won't.
5. **Then the ledger (L).** First, the Exit dialog lists the saved areas from wizard_data. Later, one container-owned draft is persisted in a single audited transaction at Finish.

Leave out three ideas:
- **A 'Skip step' footer button on every step.** It adds chrome everywhere to solve one real problem.
- **A two-press Next.** It is hidden modal state, and an impatient user will trip it.
- **'Undo this session'.** It races writers in Settings and Console, and it cannot undo encryption or downloads.

**Resolves (12):** [model-01] [provider-01] [cross-cutting-03] [cross-cutting-09] [voice-speech-01] [voice-speech-03] [voice-speech-04] [voice-speech-06] [voice-speech-13]. Newly found: [new-provider-01] [new-a11y-terminal-visual-02] [new-voice-speech-02].

**Effort and risk.** M for steps 1–4; L for the ledger. Risk is High.
- This deliberately rewrites the 34 skip-safe tests pinned by task-25820.
- It changes the voice save path. A `persist_default_preferences` flag is needed, or the Summary keeps its false '(default voice)'.
- Guard it with an invariant test: pressing only Next through both tracks, on randomly populated configs, leaves config.toml byte-identical apart from first-run bookkeeping.

**Backlog.** [model-01] → task-26837 (To Do, high); its notes say the commit path "was never examined". Everything else in SF3 is untracked. [voice-speech-01] contradicts Done TASK-32959 AC#2.

#### SF4 — One outcome record per step drives every mark

**Flaw.** ✓ and ✗ come from step position, from whether config holds a value, or from whichever status widget rendered last.
- Provider shows ✓ after a refused or rejected check [provider-03].
- '✓ RAG' reports an embedding key that the RAG pipeline never reads [fulltrack-01].
- 'Everything is off by default' hides four gates, three of which are on [fulltrack-02]. Leaving 'Create note' off doesn't stop the agent creating notes (newly found [new-full-track-steps-02]).
- A working model-from-disk Parakeet reads 'configured but not installed' [voice-speech-14].
- Errors appear before the user acts, and stay after the cause is fixed [cross-cutting-07].

**Fix.** Each step returns `StepOutcome {status: saved | kept_current | skipped | staged | failed, summary, writes, reason}`. It is computed by the runtime's own resolvers:
- provider readiness plus probe evidence, through the shared evidence owner (task-33005.9);
- `resolve_active_rag_config`;
- `all_tool_gates()` plus the active MCP sources;
- the STT source resolver.

The tracker, the step gate, one status line per step, the Summary and the Exit dialog read only that record.

- **Glyphs.** ✓ saved or kept current · – skipped (neutral, keeping TASK-25818's "don't cry wolf" restraint) · … staged · ! needs attention, always with its reason. A legend line appears whenever – or ! is on screen [cross-cutting-10].
- **Status line rules.**
  - Field errors appear under the field.
  - The pinned strip is reserved for commit errors, and is cleared on any change to provider, endpoint or key.
  - Copy leads with the on-screen fix and keeps env/TOML routes on a dim second line.
  - Only actions that can succeed are named.
  - Outcome colours reach at least 4.5:1 [provider-16].
- **Tools, without exposing the master switch.** Use copy derived from state: 'These switches add file and note tools — off until you turn them on. The assistant also has web search, Watchlists, character-card and question tools; anything that reaches the web or changes your data asks you first.' Add a read-only 'Also available (asks first)' line built from all_tool_gates(). File-tool rows say they only reach Workspace folders and the chat scratch area [fulltrack-06].
- **RAG.** Drop the embedding picker. Show the effective model read-only, and keep one real control: automatic retrieval, off by default, written to `chat_defaults.rag_auto_retrieve_on_send`.
- **Template values.** Endpoints and models still at their shipped sample values read 'Not set up', never 'Ready' [entry-exit-handoff-23].
- **Summary.**
  - Use display names, not raw ids, and list rows in step order [protect-summary-12] [protect-summary-16].
  - Each row that needs attention names its fix.
  - The model-list consent appears only when a cloud provider is configured, and names that provider [protect-summary-17].
- **Smallest first slice.** Derive the tracker glyphs for Provider, Model, Voice and Protect from `build_summary_rows`, and stop recomputing state from step position on Back.

**Resolves (15):** [cross-cutting-02] [cross-cutting-07] [cross-cutting-10] [provider-03] [provider-16] [fulltrack-01] [fulltrack-02] [fulltrack-06] [voice-speech-14] [entry-exit-handoff-23] [protect-summary-12] [protect-summary-16] [protect-summary-17]. Newly found: [new-voice-speech-03] [new-full-track-steps-02].

**Effort and risk.** M. Risk is Low, with one trap: don't answer a false ✓ with more confirm dialogs.
- On Provider, refuse only on a definitive "unauthorized" verdict.
- Otherwise, switch the tracker to ! as soon as a failure is known, and keep a single 'Continue anyway?' gate on Model.
- Don't hold Next for up to 8 s on every keyed provider.

**Backlog.** Partial: task-33005.9 (shared evidence) and task-33005.7 (readiness vocabulary; the wizard isn't listed there, so add it). One product decision is needed: whether the built-in tldw_chatbook MCP server belongs in plain Console chat at all.

#### SF5 — The provider catalog drives the form; detection is the first row

**Flaw.** The fields, checks, links and discovery ports that the Provider step offers come from scattered, hand-maintained rules: a 20-key probe whitelist, a 6-entry base-URL table, an 8-entry key-URL map and hard-coded ports. Each one drifts on its own.
- Azure OpenAI, Databricks and Cloudflare dead-end with no endpoint field [provider-05].
- 32 presets show a Test button that never enables [provider-06].
- LM Studio, Jan and KoboldCpp are never detected [provider-13]. The KoboldCpp row can't be set up at all (newly found [new-provider-02]).
- Z.ai previews the wrong chat URL [provider-20].
- 'Where do I get a key' pointers exist for only 8 of 45 keyed providers [provider-25].
- 60 providers sit in an unsearchable 5-row box [provider-11].
- A server found seconds earlier is lost when the user presses Enter or picks its engine by hand [provider-02].

**Fix.** Extend the shared catalog record that Settings already uses. Each record gains:
- locality, key_url and env_var;
- requires_base_url, with a label and an example;
- default local ports and the route suffix;
- the check kind (model list / authenticated endpoint / none), and whether listing requires auth;
- a legacy flag.

Add a parity test that fails when a keyed row lacks a required fact. Keep an explicit allowlist for unknown key URLs rather than inventing them. Then render the step from the catalog:

- **Detection is the first, pre-highlighted row.** For example: 'llama.cpp on this computer · 127.0.0.1:9099 · 3 models'.
  - Enter means "use it", and the arrows still browse. This replaces the separate button that Tab can't reach.
  - Detection stays sticky when the user switches providers, and a manual pick of that engine uses the port that was found.
  - When nothing is found, say which addresses were checked and offer a collapsed 'Don't have a local server yet?' with 'Check again' [provider-14].
- **A filter** that matches like Settings' provider search, plus a count line, and a list that grows into free height.
  - Legacy aliases go behind 'Show legacy entries' [provider-12].
  - Env-keyed rows say so ('OpenAI · key in OPENAI_API_KEY') [coverage-08]. These badges are their own increment: 60 readiness lookups on entry have a performance cost under ADR-097.
- **One 'Check key' button,** backed by the non-billable check Settings already uses (TASK-33005.4). Hide 'Find local servers' and the dead Test button on cloud presets.
- **The endpoint field goes above the key** for Azure, Databricks and Cloudflare, labelled per provider. Endpoint errors name the real cause: a bad scheme, https against an http server, a stray /api, or the HTTP status [provider-15].
- **Discovery** adds :1234 (with an LM Studio preset that Settings also gets) and :1337. It maps :5001 to Custom until KoboldCpp is fixed. It probes both loopback spellings and merges the results into one row [provider-22]. The Console 'Get started' card uses the same candidates, plus an explicit 'Look again' button [entry-exit-handoff-21].
- **Paste hygiene.**
  - Strip invisible characters, wrapping quotes and 'Bearer ' or 'NAME=' prefixes, and say so: 'Removed 1 invisible character and surrounding quotes from the pasted key.' [gap-04] (newly found)
  - Advertise Ctrl+U, and add a Show toggle [provider-17].
  - When an env key exists, offer 'Use OPENAI_API_KEY from your environment / Store a different key for this app' [provider-19].
  - Console's 'Set up provider' preselects the provider whose key was found (newly found [new-coverage-power-03]).
- **Copy.** '✓ Connected — OpenAI returned 137 models. Pick one on the next step.' The found-server banner always names the address it found [provider-21].
- **TLS interception** gets its own error category, pointing to Settings ▸ Network. Never offer an inline "disable verification" toggle [coverage-19].

**Resolves (20):** [provider-02] [provider-05] [provider-06] [provider-11] [provider-12] [provider-13] [provider-14] [provider-15] [provider-17] [provider-19] [provider-20] [provider-21] [provider-22] [provider-25] [coverage-08] [coverage-19] [entry-exit-handoff-21]. Newly found: [new-provider-02] [new-coverage-power-03] [gap-04].

**Effort and risk.** L as a whole, but don't block on the registry. Three slices ship on their own: the fix for Azure, Databricks and Cloudflare (M), sticky detection (part of an M), and the filter (M). Risk is Medium:
- Invented signup URLs go stale.
- Readiness lookups cost time on entry.
- Redirecting typed characters into the filter must not swallow Space, which selects.

**Backlog.** [provider-20] → task-33621.32. [provider-06] is partly covered by task-33005.9. Everything else in SF5 is untracked.

#### SF6 — An encryption lifecycle with one owner

**Flaw.** Setup can turn encryption on. Nothing else in the lifecycle works.
- On the module entry point, the next launch crashes in the unlock prompt (NoActiveWorker) and exits with code 1 [protect-summary-01].
- The pre-TUI prompt has no retry and no reset [protect-summary-02].
- No reachable UI can turn encryption off or change the password, although Protect promises 'you can enable this later in Settings ▸ Privacy & Security' [protect-summary-03].
- Protect ignores existing encryption. Enter after success reopens the dialog, and a second password strands the keys [protect-summary-04]. That state also traps the wizard: Next and Exit setup both fail [protect-summary-04].
- Ciphertext passes the key-validity check, so a skipped decrypt reads as Ready (newly found [new-protect-summary-03]).

**Fix.** One service owns the four states (off / on / locked / unlocked). Three surfaces use it.

- **Startup: one unlock implementation.**
  - Delete the module path's private `PasswordPromptApp`. Route `python -m tldw_chatbook.app` through the same `startup_preflight` that `tldw-cli` uses, with strict decryption.
  - The first step is a retry loop: 'That password didn't match. Try again.'
  - On final failure, offer '[R]eset saved keys / [Q]uit'. Reset strips `enc:` values from every section and drops `[encryption]`.
  - Recovery mode stops opening Backup & Restore automatically for `configuration_unlock_failed`.
  - A shared Textual unlock screen can come later.
- **Settings ▸ Privacy & Security.** An Encryption card with a state line and three actions: Encrypt, Change password, Turn off. Change password needs a current-password field; today's change mode has only one password field plus a confirm.
- **Protect.**
  - Its state comes from persisted config: encrypted, plain text on disk, env keys only, or nothing saved [protect-summary-06].
  - `enable_config_encryption` refuses when encryption is already on, and validates before it writes (or restores the previous bytes).
  - After success the button changes its label and focus moves to Next.
  - Two honest buttons: 'Encrypt with a password…' and 'Keep as plain text'. They also remove the Skip that the copy names but that doesn't exist [protect-summary-07].
  - The password field takes focus when the dialog opens [protect-summary-05].
- **Everywhere.** `enc:` values read as absent until they are decrypted.

**Resolves (10):** [protect-summary-01] [protect-summary-02] [protect-summary-03] [protect-summary-04] [protect-summary-05] [protect-summary-06] [protect-summary-07]. Newly found: [new-protect-summary-02] [new-protect-summary-03].

**Effort and risk.** S for the refusal, focus, strict decrypt and absent-ciphertext changes; M for retry/reset and the Settings card. Risk is High: this is the one area where a mistake locks people out of their own app.
- Keep `startup_preflight` pre-TUI and isolated, as the Backup_Recovery design intends.
- Never ship a 'Start without saved keys' option before ciphertext reads as absent. Otherwise the ciphertext is sent as a bearer token.

**Backlog.** Only [protect-summary-01] is tracked, and only loosely: task-33621.25 lists the push among 26 unreviewed wait-for-dismiss calls, not as a lock-out. Reword it. Everything else in SF6 is untracked.

#### SF7 — A setup-session model and one entry contract

**Flaw.** Setup state is a handful of loose first_run flags. Five hand-wired entry points each implement cancel, guard and resume differently.
- A re-run ignores the current provider, model, voice, RAG choice and track, and uses first-run copy. Cancelling a re-run started from Settings lands on Console [cross-cutting-05].
- The exit dialog promises 'You can continue setup any time from Settings ▸ Diagnostics.', but re-runs restart at Welcome [cross-cutting-08].
- Resume hides the key field [provider-04] and restores a preset voice as 'Custom' [cross-cutting-19].
- 'Later' saves nothing, so setup nags at every launch [entry-exit-handoff-07].
- Every 'Set up provider' or 'continue setup' button opens an OpenAI-preset Settings form [entry-exit-handoff-12].
- Ctrl+Q skips the guard that Esc shows (newly found [gap-05]).
- Prefill reads `load_settings()`, which omits `[tools]` and `[transcription]`, so a re-run shows every tool gate off (newly found [new-full-track-steps-01], [new-voice-speech-01]).

**Fix.**

- **Explicit states.** Not started / in progress / deferred / completed, persisted with the draft.
  - `setup_started` is written only after the first Next on Welcome.
  - 'Not now' writes a snooze.
  - A deferred setup surfaces as one non-modal 'Continue setup (Provider)' row in Console's 'Get started' card.
- **One entry point.** `app.open_setup_wizard(origin, resume_draft, start_step)`:
  - owns the duplicate-wizard guard [entry-exit-handoff-11];
  - returns cancel to the screen it came from (Console only when the origin is boot);
  - gives re-runs a primary 'Done' button [protect-summary-14].

  Console's 'no provider' blocker opens setup at Provider, and every other blocker links to its own control [entry-exit-handoff-27]. Every abandon lands on the Console 'Get started' card [entry-exit-handoff-15]. Test the matrix of origin × {cancel, finish, skip}.
- **Prefill from the real file.** Load the full TOML once (force-reload) and refresh it after each commit. Every list marks its saved value '(current)', the way Style already shows 'nord (current)'.
- **One finish path for every completing exit.** It records completion and consent, so Skip never triggers the consent modal [entry-exit-handoff-13]. It honours 'Get to know you after setup' on the Library exits [protect-summary-11]. Esc on the Summary becomes inert or finishes setup [protect-summary-08].
- **A recovery dialog built from the draft.** Its primary button names the step: 'Resume at Model' [entry-exit-handoff-08].
- **`confirm_quit`** reuses the existing exit dialog.

**Resolves (17):** [cross-cutting-05] [cross-cutting-08] [cross-cutting-19] [provider-04] [entry-exit-handoff-07] [entry-exit-handoff-08] [entry-exit-handoff-11] [entry-exit-handoff-12] [entry-exit-handoff-13] [entry-exit-handoff-15] [entry-exit-handoff-27] [protect-summary-08] [protect-summary-11] [protect-summary-14]. Newly found: [new-full-track-steps-01] [new-voice-speech-01] [gap-05].

**Effort and risk.** L overall. The slices that stop damage are S and go first: the cancel route from Settings, untouched steps writing nothing, the palette guard and truthful copy. Risk is Medium:
- The palette deliberately passes `cancel_to_console=False` (TASK-31813), so a naive reroute regresses it.
- The draft doesn't yet hold a provider that was highlighted but not committed, so the Provider step must checkpoint its selection on exit.

**Backlog.** [entry-exit-handoff-27] → task-33620.2, task-33620.3, task-33620 and task-33008. Partial: [entry-exit-handoff-12] → task-33008 (high) and task-33008.3; [entry-exit-handoff-13] → task-28019 AC#3. TASK-31813 AC#2 was ticked on the palette path only and never wired for Settings, so correct it.

#### SF8 — One input policy

**Flaw.** Choice lists treat a highlight as a selection, and several handlers dispatch the same pick.
- Browsing commits rows, fires network discovery, wipes typed values and erases errors [cross-cutting-06]. A failure line is wiped before it can be read [provider-26].
- Focused lists and inputs consume Enter before the wizard's 'Enter = next' binding sees it. So Enter selects, tests, skips or saves depending on focus [cross-cutting-04], and picks OpenAI over a local server that was just detected [provider-02].
- At 100x30 and below, arrowing through the Provider list scrolls the list off screen (newly found [new-a11y-terminal-visual-01]).

**Fix.** Highlight moves the cursor. Enter, Space or a click selects, through one dispatch path. Enter on a settled control advances. The footer hint is generated from this policy for whichever control has focus.
- **Provider list:** highlight-only (OptionList's native behaviour), with discovery debounced on selection.
- **Short radio groups (6 options or fewer):** keep selection-follows-highlight. TASK-21142's decision stands.
- **Model, RAG and theme lists:** render with the current or recommended row already selected. Only an explicit pick clears a typed model id; arrowing never does.
- **Key field:** Enter means "check, then continue on ✓".
- **Focus:**
  - Move to the key input only after an explicit selection, with no extra Tab stop on the Authentication header [provider-23].
  - Programmatic collapse changes don't auto-scroll.
  - After Test and Hear, focus returns to the button [voice-speech-05].

Leave out two ideas: making Enter in free-text fields only move focus (it reverses UAT decision N-1), and a two-press Enter to skip (TASK-32555 just shipped the one-press skip).

**Resolves (6):** [cross-cutting-04] [cross-cutting-06] [provider-23] [provider-26] [voice-speech-05]. Newly found: [new-a11y-terminal-visual-01]. Together with SF5 it also completes [provider-02].

**Effort and risk.** M. Risk is Medium: overcorrecting would reverse accepted UAT decisions. Any per-field Enter exception must be signposted in the hint line.

**Backlog.** No open task. TASK-21142 and TASK-32555 (both Done) created today's behaviour. [provider-26] → task-33621.37.

#### SF9 — A terminal-native frame

**Flaw.** The frame and lists use fixed sizes tuned for 120x40.
- At 80x24, the stock terminal size, the chrome takes 14 of 24 rows and the Summary's read-back shows no status rows [a11y-01].
- The size warning covers the key hints and is itself cut off [a11y-03]. Welcome opens scrolled past its own heading [a11y-02].
- Large terminals leave 30–40 empty rows [a11y-04].
- Hidden content is signalled only by a 1.55:1 scrollbar thumb [a11y-08].
- On/off is shown only by colour or knob position, and stock checkboxes draw 'X' in both states [a11y-07].
- Contrast is low: primary labels measure 3.78:1 and the unselected '○' 1.42:1 [a11y-09].
- Secondary buttons render as bare bold text [a11y-10], and keyboard focus is invisible on primary buttons [a11y-11].
- Back is bound to Ctrl+B, the default tmux prefix [cross-cutting-15].
- In the light theme the tracker's '!' measures 1.38:1 (newly found [gap-09]).

**Fix.**

- **Layout tiers.**
  - A '-short' class below 30 rows, following `_sync_compact_mode`. Title and tracker share one row that keeps the '!' state and the step name. Drop the outer border. Use one bottom row, where the size note truncates before any key hint does.
  - On the Summary, remove row margins and list ✗ and ! rows first, which gets about 8 rows visible. Don't hide 'Add your first document' and 'Write your first note' behind a 'More ▾'; those exits were made visible on purpose.
  - A '-tall' class (50 rows or more) lets lists grow only into leftover space. A flat 30vh would push the key field below the fold at 120x40.
- **Focus.** Call `focus(scroll_visible=False)`, then make the smallest scroll that shows the target. After every resize, scroll the focused field back into view (newly found [gap-08]).
- **Overflow.** Reuse the app's existing fold cue ('▼ more — scroll…'). The step container has no border, so a border subtitle won't render. The Tools list groups its rows and shows counts [fulltrack-12].
- **State as text.**
  - Promote the existing glyph checkbox that honours ascii_glyph_mode, rather than adding a fourth convention.
  - Put an On/Off label and a count beside the tool switches.
  - Show toast severity as a glyph plus a word [a11y-13].
  - Delete the side-stripe selection border that DESIGN.md forbids [a11y-14].
- **Buttons and focus at the theme layer.** Add one 1-row bracketed secondary-button tier and a 'bold reverse' focus style to the shared button styles, by widening task-33626 beyond Console. A wizard-only override would diverge from the rest of the app and would only fix textual-dark.
- **Keys.** Teach the button ('← Back or Ctrl+B'), add alt+left, show a one-time tip when `$TMUX` is set, and keep Ctrl+B as an alias.
- **CI.** Add a size matrix (80x24, 100x30, 120x40 and 200x60, both tracks) and a theme contrast matrix (textual-dark, textual-light, NO_COLOR).

**Resolves (15):** [a11y-01] [a11y-02] [a11y-03] [a11y-04] [a11y-07] [a11y-08] [a11y-09] [a11y-10] [a11y-11] [a11y-13] [a11y-14] [cross-cutting-15] [fulltrack-12]. Newly found: [gap-08] [gap-09].

**Effort and risk.** M for the tiers and the theme-layer tokens; S for each component fix. Risk is Low–Medium:
- Theme-layer changes touch every screen.
- The proposed 1-row nav and the proposed bordered 3-row nav conflict; pick one secondary-button treatment.
- Don't always scroll Welcome to the top. At 80x24 that puts the focused control below the fold, which fails WCAG 2.4.11.

**Backlog.** No task for the wizard. Widen task-33626 (currently Console-only) and task-32465 (currently outside the wizard).

#### SF10 — Steps earn their place

**Flaw.** Steps were added for parity or a stable step count, not because they were relevant.
- Quick ('about 2 minutes') carries an optional Voice step that defaults to a server most newcomers don't run, and a Protect step that does nothing whenever no key is stored [coverage-16].
- Full carries a Notes step that configures nothing [fulltrack-05], a dead page of pip instructions when the RAG extras are missing [fulltrack-11], and a 78-card splash gallery [fulltrack-09].
- Yet Full offers no way to turn the splash off or reduce motion [coverage-15], no tldw server connection [coverage-06], and no cloud or already-installed dictation engine [voice-speech-15].
- Welcome promises 'Full setup — configure everything' [entry-exit-handoff-18] and has no documents-first path [entry-exit-handoff-28].
- First launch shows 10–16 s of random, off-brand splash with no skip hint [entry-exit-handoff-17].

**Fix.**

- **Quick = Welcome, Provider, Model, Summary.** This is close to the spec's Quick track (Provider, Model, conditional Protect, Summary).
  - Voice moves to Full and becomes a Summary next step, 'Hear replies aloud…'.
  - Protect leaves Quick. The Summary offers 'Encrypt saved keys with a password…' only when a key was stored this run; that is the verifier's lower-risk alternative to growing the Provider form.
  - Full keeps its Protect step, always present so the step count never changes mid-run, with state-aware content (SF6).
- **Full.**
  - Remove Notes; a Summary next step, 'Sync a notes folder…', replaces it.
  - Replace the splash gallery with 'Startup animation: Off / Short / Full' plus 'Reduce motion'.
  - Keep the RAG step for its auto-retrieve control even without the extras, because keyword retrieval works without them. Fix the malformed sample rows in the template [coverage-13].
  - Speech opens with an engine choice, with 'No dictation for now' preselected. Precision, from-disk models and GGUF move under Advanced [voice-speech-12].
- **Server.** A Summary row, 'Runtime — this computer only · connect a server in Settings ▸ Overview', a next step, 'Connect a tldw server…', and an optional Server step on Full only.
- **Welcome.**
  - Time copy: 'Quick: about 2 minutes (plus any optional downloads)'.
  - A third choice: 'Start with my documents or notes — set up AI later'.
  - Scope the privacy copy to local models. Don't promise 'nothing leaves your machine' while cloud TTS and model-list refresh are on offer.
- **First launch.** Skip the splash on first run, or show a fixed wordmark with 'Press any key to skip'. Send third-party warnings to the log.

**Resolves (13):** [coverage-06] [coverage-13] [coverage-15] [coverage-16] [fulltrack-05] [fulltrack-09] [fulltrack-11] [entry-exit-handoff-17] [entry-exit-handoff-18] [entry-exit-handoff-28] [voice-speech-12] [voice-speech-15]. Newly found: [new-coverage-power-01].

**Effort and risk.** M for the Quick rescope; L for the Speech engine choice and the Server step. Risk is Medium: tests pin the 6-step Quick track and the guide's track table, and step count must still never change mid-run.

**Backlog.** [entry-exit-handoff-28] → task-28019 (low). [entry-exit-handoff-17] is partly covered by task-24306. Everything else in SF10 is untracked.

---

### 5.2 Enablers

**E-a — Step-lifecycle controller and module split.** Effort L.
- **Why.** A 10.8k-line module mounts and recomposes widgets without restoring focus, and treats `is_mounted` as a liveness check. It awaits long commits with no busy state: each Next stalls 2.5–4 s, up to 30 s on Voice, with no cue.
- **Plan.**
  1. Make the provider suite green first (41 failures).
  2. Extract steps behind the existing SetupStep seam, a few at a time.
  3. Sweep `is_mounted` to `is_attached`, guarded by an architecture test.
  4. Add one `wizard_worker()` helper, and show a busy line after about 400 ms.
- **Do not** hold the P0 and P1 fixes for the split.
- **Issues:** [cross-cutting-17] [cross-cutting-14].
- **Backlog:** TASK-32809.2 (In Progress; the module's ratchet row is red at 10,854 > 10,404), TASK-33621.36, TASK-32892, TASK-32800.

**E-b — Setup registry for names and destinations.** Effort M; L if it also grows the CLI.
- **Why.** Setup has seven names, Speech points to three different homes, manual setup opens the wrong pages, and the guide contradicts the wizard in ten places.
- **Plan.** One module maps each setting to its display name, owning destination and route. Step copy, Summary rows, manual-setup routes, palette aliases, `--help` and a guide claim-check all read from it.
- **Issues:** [voice-speech-16] [cross-cutting-11] [cross-cutting-12] [cross-cutting-13] [cross-cutting-16] [cross-cutting-18] [coverage-10] [entry-exit-handoff-16] [protect-summary-13] [fulltrack-13].
- **Backlog:** fold it into task-33623; the claim gate is task-32589.

**E-c — Shared download coordinator.** Effort L.
- **Why.** Voice and Speech each run their own install worker. There is no cancel, speed or ETA; leaving the step is silent; a finished model never becomes the default; and an interrupted download orphans its partial file.
- **Plan.** One job owner, shared with Lab ▸ Models, with:
  - resumable staging and a reconcile on start;
  - progress, cancel and ETA, plus the combined download size;
  - a guard when the user leaves mid-download;
  - completion applied through the setup draft.
- **Issues:** [voice-speech-08] [voice-speech-09] [voice-speech-10] [voice-speech-11] [voice-speech-18].
- **Backlog:** task-33079. Update its premise: concurrent installs serialize through the session lease.

---

### 5.3 Roadmap

#### NOW — stop the bleeding (about one sprint)

**Rules for this phase.**
- Every item ships with a fresh-profile test on an isolated config (TLDW_CONFIG_PATH plus HOME). The test must end on a reply or on the expected landing screen.
- No item waits for the module split.
- **Prerequisite:** the wizard provider suite goes green first (TASK-33621.36), so these fixes land under working tests.

| # | Item | Issues | Effort | Backlog |
|---|---|---|---|---|
| | **Make the first chat work** | | | |
| N1 | Repair OpenAI context windows, cap the reservation for estimated windows, test every shipped default | [cross-cutting-01]; newly found [new-cross-cutting-01] | S | none |
| N2 | `rank_setup_models`: chat-only, deterministic order, curated recommendation, ranking before the slice. Ship after N1 | [model-02] | M | none |
| N3 | Fence the handoff on values, ending the false 'Provider settings changed' toast | [entry-exit-handoff-04] | S | task-33001.10 (raise to high) |
| N4 | Keep `chat_with_llm` out of the agent catalog | [entry-exit-handoff-24] (point 1) | Slice of M | none |
| N5 | Honest cloud verdicts: "key not verified" for OpenRouter-class providers; replace the retired Gemini default; the env-key notice says "ready" only when it is | Newly found [gap-01] [gap-03]; [coverage-04] | S slices | none |
| N6 | Moonshot: reproduce, log the swallowed persistence error, keep the delivered reply | Newly found [gap-02] | Unsized | none |
| | **Stop data loss and lock-outs** | | | |
| N7 | Skipping Model keeps the key; two-button confirm; 'Review provider setup' lands on Model | [model-01] | M | task-26837 (rewrite task-25820's pins) |
| N8 | Protect refuses a second enable, validates before writing and moves focus to Next; add 'Exit without saving progress' | [protect-summary-04] | S | none |
| N9 | One unlock path through `startup_preflight`; ciphertext reads as absent; the password field takes focus | [protect-summary-01] [protect-summary-05]; newly found [new-protect-summary-02] [new-protect-summary-03] | S | task-33621.25 (reword) |
| N10 | Unlock retry and '[R]eset saved keys / [Q]uit'; truthful Protect copy | [protect-summary-02] [protect-summary-03] (copy) | M | none |
| N11 | An untouched Voice step writes nothing | [voice-speech-01] (delta-gate slice) | Slice of M | none |
| N12 | Re-run safety: cancelling a re-run from Settings returns to Settings; untouched steps write nothing | [cross-cutting-05] (items 3–4) | S slices | none (correct TASK-31813's AC tick) |
| N13 | Ctrl+Q honours the exit guard | Newly found [gap-05] | Unsized | none |
| | **Remove dead ends** | | | |
| N14 | A Provider skip that works by keyboard and mouse; the error strip clears on provider change; a skipped provider stays selected | [provider-01] [cross-cutting-03]; newly found [new-provider-01] [new-a11y-terminal-visual-02] | M | none |
| N15 | Resume shows the key field | [provider-04] | S | none |
| N16 | Endpoint field above the key for Azure, Databricks and Cloudflare | [provider-05] | M | none |
| N17 | Sticky detection: picking the detected engine uses the port that was found | [provider-02] (core slice) | Slice of M | none |
| | **Tell the truth (near-free)** | | | |
| N18 | Tools copy derived from state; optional steps set to required=False with real homes; Skip records the consent default; fix the guide's ten contradictions and remove its stamps; splash skip hint | [fulltrack-02] (copy) [cross-cutting-13] [entry-exit-handoff-13] [cross-cutting-16] [cross-cutting-18]; newly found [new-coverage-power-01] | S each | task-28019 AC#3 (partial); task-32589 |

Five P1s move to NEXT, because their honest fix is structural:
- [a11y-01] → SF9
- [fulltrack-01] → SF4 and SF10
- [entry-exit-handoff-03] → SF1; N1 already makes it rare
- [entry-exit-handoff-12] → SF7
- newly found [new-a11y-terminal-visual-01] → SF8

**NOW is done when:**
- On a fresh profile, following the wizard's own recommendations reaches a reply for OpenAI, Anthropic, llama.cpp and Ollama.
- Setting a password and relaunching through either entry point opens the app.
- Skipping Model never discards a typed key.
- No step's Next is a dead end.

#### NEXT — replace the mechanisms

| Wave | Work | Key issues | Effort | Depends on |
|---|---|---|---|---|
| **1 · Foundations** | E-a extraction (steps behind SetupStep, `is_attached`, busy line) | [cross-cutting-17] [cross-cutting-14] | L | Green suite |
| | SF4 outcome record, starting with the tracker read from Summary statuses | [cross-cutting-02] [provider-03] [fulltrack-01] [fulltrack-02] | M | — |
| | SF7 entry contract and prefill from the raw TOML | [cross-cutting-05] [entry-exit-handoff-12] [entry-exit-handoff-27]; newly found [new-full-track-steps-01] | L | — |
| | SF8 input policy | [cross-cutting-04] [cross-cutting-06]; newly found [new-a11y-terminal-visual-01] | M | — |
| **2 · First-chat path** | SF1 'Ready to chat?' row, single finish routine, first-reply error copy | [entry-exit-handoff-03]; newly found [gap-06] [gap-07] | M | N1, SF4 |
| | SF2 combobox picker, '(current)' row, status line, Gemini discovery | [model-04] [model-05] [model-06] [model-07]; newly found [gap-03] | M ×3 | N2 |
| | SF3 'No voice for now', 'Save & continue' label, Exit dialog that lists saved areas | [voice-speech-03] [voice-speech-04] [cross-cutting-09] | M | SF4 |
| | SF9 short tier first, then components, theme-layer tokens, CI matrices | [a11y-01] first | M | — |
| **3 · Breadth** | SF5 provider catalog: filter, detection row, Check key, discovery ports | [provider-06] [provider-11] [provider-13] [coverage-08] | L | SF8 |
| | SF6 Settings encryption card and state-aware Protect | [protect-summary-03] [protect-summary-06] [protect-summary-07] | M | N8–N10 |
| | SF10 Quick rescope, Full clean-up, Speech engine choice, Server row | [coverage-16] [fulltrack-05] [voice-speech-15] [coverage-06] | M → L | SF3, SF7 |
| | E-b setup registry; E-c download coordinator | [cross-cutting-11] [voice-speech-16]; [voice-speech-09] [voice-speech-10] | M; L | E-a |

**Guardrails that land with NEXT.** Each comes from the review's improvement ideas:
- an invariant test that pressing only Next writes nothing;
- a cold-start regression walk: Quick and Full, Enter-only and mouse-only, kill-and-resume at every step, Protect → relaunch, re-run → cancel. It asserts the landing screen, that the tracker matches the Summary, a golden config diff, and that focus is never None;
- the size matrix and the theme contrast matrix (SF9);
- a provider smoke matrix that includes kimi and checks template defaults against the live model lists.

#### LATER — redesign and delight

| Order | Enhancement (§4) | Effort | Builds on |
|---|---|---|---|
| 1 | E1 'Say hello': the first reply inside setup | M–L | SF1 |
| 2 | E2 'Ready now' first, and a 'Keys found' path for env installs | M | SF5 |
| 3 | E9 A 'what's next' checklist and handoff receipt | S–M | SF1 finish routine |
| 4 | E7 A quiet, comfortable first launch (the S parts can go in NOW) | S–M | SF10 |
| 5 | E4 Curated starting choices | M | SF2 |
| 6 | E3 Local coaching and an LM Studio preset | M | SF5 |
| 7 | E5 'Review your setup' for re-runs | M–L | SF4, SF7 |
| 8 | E10 One startup attention queue | M | SF7 |
| 9 | E11 Paste-time key help | S | SF5 paste hygiene |
| 10 | E6 Portable, scriptable setup | S → L | E-b |
| 11 | E8 Keychain-first key storage | L | SF6 |
| 12 | E12 Plain-text setup for screen readers and SSH | L | E6, E-a |

---

### 5.4 Enhancements beyond the defects

The 63 improvement ideas reduce to these twelve once duplicates are merged and the ideas already absorbed into SF1–SF10 are set aside. Four ideas proposed the same "test drive", four proposed the same re-run hub, and three proposed the same portable setup. Personas: **Sam** is a first-time user with a cloud key, **Jo** is a first-timer who wants local, private AI, and **Riley** is a power user with env keys and several machines.

| | Enhancement | Why (evidence) | Personas | Effort |
|---|---|---|---|---|
| **E1** | **'Say hello': the first reply happens inside setup.** The Summary's primary action becomes a test drive. The prompt is prefilled ('Say hi in five words.'), sent on Enter through Console's real admission and dispatch path, and the reply streams inline ('gpt-4.1-mini: Hello! · 0.8 s'). The exchange carries into Console as the first turn. On failure it shows the plain cause with one action: 'Switch to gpt-4.1-mini', 'Start Ollama' or 'Fix key'. Cloud providers ask consent first ('Send a one-word test message to OpenAI? Uses a few tokens.'); local providers run it by default, because it costs nothing. 'Skip test' serves offline users. | Every P0 in the model area ends with ✓ and then a failed first send. Sam's first reply came at 621 s, after 20 recovery actions in Console. A reply is the only proof that key, model, context budget and streaming work together, and it catches runtime defects such as [gap-02] (newly found) that an offline preflight can't. | Sam, Jo, Riley | M–L |
| **E2** | **'Ready now' first.** Above the provider list: 'llama.cpp on this computer — 3 models' and 'OpenAI — key found in OPENAI_API_KEY', each one Enter away. When env keys bypass the wizard, Console's 'Get started' card says 'Found ANTHROPIC_API_KEY · [Use Anthropic]' instead of showing a toast. Every env-keyed provider is recorded with `credential_source = environment` automatically; the user picks only the default. | Detected servers and exported keys are known before the user does anything, yet the step opens on 60 rows with OpenAI highlighted. Riley's env keys skip the wizard entirely [coverage-04]. | Riley, Jo | M |
| **E3** | **Local coaching.** An LM Studio preset, plus 'Check again' and polling while the Provider step is visible (localhost only, cancelled on leave). A 'Popular models to run' list with a copyable `ollama pull llama3.1:8b`, and 'Use llama3.1:8b — I'll start it later'. Embedding models stay out of the chat list, using Ollama's capability data. | Jo installs the server during setup, and nothing notices. LM Studio is never detected [provider-13], and gets no streaming after setup [entry-exit-handoff-25]. | Jo | M |
| **E4** | **Curated starting choices.** 2–3 rows per provider with plain trade-offs ('Fast & inexpensive — gpt-4.1-mini · 1M context'). They come from one table shared by the wizard, Settings and the Console switcher, tested in CI against model_capabilities. Add a caption: 'This is the model new Console chats start with. Switch any time with Alt+M — nothing here is permanent.' | Sam can't tell which model is good. The Console switcher already knows context sizes. | Sam | M (caption S) |
| **E5** | **'Review your setup' for re-runs.** One screen of current values, with a Change action per row that opens just that step and returns. 'Done' goes back to the origin, and 'Run the full walkthrough' is the secondary action. Add palette commands ('Setup: Change default model…'), and a 'Finish with defaults' action (Ctrl+Enter) once Provider and Model are saved. | A re-run usually means "change one thing". Even with prefill, a linear walk through all 11 steps costs Riley about 75 keystrokes. With an env key they spend about 20 s and 14 keystrokes just confirming defaults. | Riley | M–L |
| **E6** | **Portable, scriptable setup.** Step 1 (S): a 'Setting up another machine' guide section, a `--help` epilog that names TLDW_CONFIG_PATH, plus `--config PATH` and `--no-splash` flags. Step 2: 'Export these settings (no keys)' on the Summary. Step 3 (L): `tldw-cli setup --from FILE --non-interactive [--key-env VAR]`, using the same commit path and readiness check, and exiting non-zero on failure. | A seeded config with `[first_run] setup_completed = true` plus env keys already boots straight to a working Console; it just isn't documented [coverage-10]. 'Restore a backup' is the only import, and it fails opaquely on a config.toml [entry-exit-handoff-16]. | Riley | S → L |
| **E7** | **A quiet, comfortable first launch.** No splash on a profile's first launch, or a fixed wordmark for at most 1.5 s with 'Press any key to skip'. Third-party warnings go to the log. Honour TLDW_REDUCE_MOTION and NO_MOTION. A 'Display' row on Welcome offers Standard / High contrast / Plain ASCII glyphs / Reduce motion, applied live. Track 'time to Welcome' as a perf-guard ratchet. | reduce_motion helps only if it is set before animations play. Today a user who needs it must get through 9 Full-track steps before reaching Style, and the splash can't be turned off. | Everyone, especially low-vision and SSH users | S–M |
| **E8** | **Keychain-first keys.** At key entry, offer: 'In the system keychain (recommended)', 'In config.toml as plain text', 'Encrypted with a master password', or 'From an environment variable'. The Summary says 'key in macOS Keychain'. If the master password stays, add 'Remember on this device' and a skippable "type it again" recall check before finishing. | Most users want "my key isn't in a plain-text file" without typing a password at every launch, and the password path is the most fragile area in this review (SF6). keyring is already a core dependency and already holds server credentials and MCP bindings. | Sam, Riley | L (+M for 'Remember') |
| **E9** | **'What's next', not toasts.** Replace arrival toasts with one dim transcript line: 'Setup complete — OpenAI · gpt-4.1-mini · streaming on · context 1M. Change any time with Alt+M.' The Summary carries a next-steps checklist: 'Hear replies aloud…', 'Add a document and ask about it', 'Sync a notes folder…', 'Add a project folder…', 'Connect a tldw server…'. Pressing Speak or Dictate with nothing configured opens a compact setup sheet at that moment. | Arrival produces a false toast [entry-exit-handoff-04], a 'completed while hidden' toast while Console is on screen [entry-exit-handoff-05], and toasts over the nav bar (newly found [new-entry-exit-handoff-02]). Success is stated nowhere. Voice and dictation matter the first time a user presses them, not as a detour in the wizard. | Sam, Jo | S–M |
| **E10** | **One startup attention queue.** At most one modal per launch. Everything else becomes a non-modal line in the 'Get started' card. | Modals stacked on or after the wizard keep recurring: the consent modal after Skip [entry-exit-handoff-13], a toast on top of the consent modal (newly found [new-coverage-power-02]), and the recovery prompt at every launch [entry-exit-handoff-07]. | Everyone | M |
| **E11** | **Paste-time key help.** Recognise well-known key prefixes locally (sk-ant-, sk-or-, gsk_, AIza, sk-proj-) and suggest 'This looks like an Anthropic key — switch to Anthropic?'. The key is never logged or sent, and the suggestion never blocks. | Newcomers paste a key under the wrong provider and get an opaque rejection. Invisible characters already crash the step (newly found [gap-04]). | Sam | S |
| **E12** | **Plain-text setup for screen readers and SSH.** `tldw-cli setup --plain` runs the same state machine as a sequence of prompts, with plain ✓/✗ lines and the same commit path. | Textual exposes no accessibility tree, so the full-screen wizard can't serve screen-reader users. `first_run_setup_state.py` is already pure logic. The same entry point carries E6's non-interactive mode. | Screen-reader users; Riley over SSH | L |

**Also worth doing, at lower priority:**
- auditioning voices, and a Voice picker that shows each service's live status;
- an end-to-end microphone test in Speech;
- tool posture presets with a live "can / cannot" preview, built on `all_tool_gates()` and without exposing the `[console]` master switch;
- "model roles", so analysis and summaries follow the chat model by default;
- an interactive tracker (click or Alt+number to revisit a step);
- prefetching discovery while the user reads;
- an optional "What should replies call you?" field;
- 'Spoken replies' and 'Dictation' as plain-language step names.

**Ideas to adopt only in modified form.** In each case the verifier's critique wins.

| Proposed | Adopt instead | Why |
|---|---|---|
| A 'Skip step' button in every step's footer | Skip inside each optional step ('No voice for now', 'No dictation for now'), plus a link wherever Next refuses | A third footer button adds chrome everywhere to solve one problem [cross-cutting-03] |
| Two-press Next or Enter to skip | One press, as the hint already promises | Hidden modal state; TASK-32555 just shipped the one-press skip [provider-01] [cross-cutting-04] |
| 'Undo this setup run' snapshot | The Exit dialog lists saved areas; a ledger comes later | Restoring a snapshot races Settings and Console writers, and can't undo encryption or downloads [cross-cutting-09] |
| Welcome as a mandatory 'How will you use chatbook?' router | Detection-driven Welcome plus Summary next steps; a Server step on Full only | A mandatory question taxes the local-first majority on the very first screen [coverage-06] |
| Ticking multiple providers inside the list | Single-select default; env-keyed providers are recorded automatically | Env keys write no secret, so ticking them adds nothing, and it is a heavier interaction than anything else in the wizard [coverage-08] |
| A 30 s idle poll on Console's 'Get started' card | Fix the one-shot probe and add an explicit 'Look again' | Polling contradicts the card's "a quiet network stays quiet" design [entry-exit-handoff-21] |
| An inline "trust this certificate" toggle | Classify the TLS failure and point to Settings ▸ Network | It nudges first-time users toward `ssl_verify=false` [coverage-19] |
| 'More ▾' for the Summary's Library exits at 80x24 | Remove margins and the border instead | It undoes exits made visible on purpose (task-32072, task-32140) [a11y-01] |
| 'Turn all on' for every tool group | Offer it for the read-only group only | One keystroke enabling Write file, Create note and Update note contradicts the step's own ⚠ warning [fulltrack-12] |
| The `[console]` master switch in setup | Truthful copy plus an 'Also available (asks first)' line | It invites a reflexive "turn everything off" that silently removes web search and Watchlists tools [fulltrack-02] |
| A Provider-step 'Continue anyway?' on top of Model's | One gate on Model, and ! on Provider | Confirm fatigue; "set up now, start the server later" is a legitimate path [provider-03] |

---

### 5.5 Preserve: what works, and which fix could break it

| Keep | Evidence | Watch out during |
|---|---|---|
| **Secrets never reach disk unless saved.** Keys are masked and never prefilled. Env keys are recorded only as `credential_source = environment`. Drafts refuse secret-named fields. A grep of every scratch home, config and log found 0 key matches, and config.toml is mode 0600. | live-power-user; live-resilience §5 | SF3 draft/ledger, SF5 key-source choice, E8 |
| **Local detection.** It finds a server in under 1 s, correctly ignored a Next.js server on :8080, and adopts it in one click: '✓ Using http://127.0.0.1:9099.' | live-firsttime-local 27, 45 | SF5: probe new ports in parallel within the same short timeout |
| **Forgiving endpoint entry** with a live 'Chat URL:' preview. Every URL form tested resolved correctly. | live-firsttime-local 53 | SF5 route-suffix move |
| **Specific connection errors.** '✗ The connection was refused - nothing is listening at that address. Start the server, or check the endpoint and port.' Ollama copy names `ollama serve`, and Retry is hidden after auth failures. | live-firsttime-local 14, 15 | SF2 status-line move, SF4 copy rules |
| **The 'Continue anyway?' gate** with 'Keep editing' focused, and the typed-model rescue: a typed model with a down server later showed 'Ready · Local' on Home. | live-resilience 62; live-firsttime-local 59–61 | SF1 and SF2: validation must never block this path |
| **The Summary reads back from disk.** 'Review provider setup' recovers inside the wizard with the staged key intact. | live-firsttime-cloud cloud2/15–16 | SF1 and SF4: keep the force-reload read-back |
| **Delta-aware writes.** Tools, Appearance and Speech write only what changed. Provider+model is one atomic, compare-and-swap mutation. | live-power-user 85; static-code-map §1.7 | SF3: extend this model to Voice; don't replace it |
| **Resume and preview safety.** Resume lands on the right step. A crash during theme preview leaks nothing. The live preview reverts honestly and marks '(current)'. | live-resilience 101, 139; live-power-user 86 | SF7 |
| **Exit and Skip dialogs.** 'Keep going' has focus, the destructive button is red, the copy is step-aware, and a double Esc 150 ms apart is absorbed. | live-resilience 74, 160 | SF7 / [gap-05] (newly found): `confirm_quit` should reuse this dialog |
| **Glyph-based state.** ●/○ and ✓/blank on the wizard's own widgets, and ✓ / ! / bold-underline in the tracker, all survive NO_COLOR. | live-resilience 141, 143 | SF9: promote these, don't add a fourth convention. SF4 must keep '!' |
| **Navigation is never lost** from 80x24 to 260x70, and the Summary actions are docked and focused. | live-resilience §3 | SF9 short tier |
| **Single sourcing.** One provider catalog is shared with Settings, and the Tools copy comes from GateableTool. | static-config-coverage 01 | SF5: add facts to that record, never a second list |
| **Install transparency.** Source, pinned revision, licence, 632.8 MiB, free space and SHA-256 verification are shown before download, and progress is shown in bytes. | live-power-user 35; live-resilience 131 | E-c: restructure the panel, don't strip it |
| **The Voice step.** It leads with the outcome and keeps the plumbing behind Advanced. OpenAI voice reuses the existing key and verifies a real sample ('Verified.'). | live-firsttime-cloud 19–20 | SF3, SF10 |
| **Model-list consent** is asked once, off by default, inside setup. | FirstRunSetupWizard.py:7943-7978 | SF7 finish path |
| **The 'Get started' card catches Skip and Exit.** Its sentence 'A provider is the AI service that answers your messages — for example OpenAI, Anthropic, or a server running on this computer.' is the best definition of "provider" anywhere in onboarding. Reuse it on the Provider step. | live-resilience 25, 163 | SF7 landing changes |
| **Scope restraint.** Sampling, system prompt, budgets, hooks, MCP and image/video generation are left out, and `users_name` isn't editable. | static-config-coverage §3 | E2, E5 and SF10 coverage additions |
| **Robust basics.** Waits are bounded. A HOME path with spaces and non-ASCII characters works. Every cell has an explicit background, so light terminal profiles are safe. A second launch is calm (about 12 s, no nag). | verify-gaps | E-a extraction |

---

### 5.6 Backlog: what is tracked, what isn't

**Coverage.**
- 16 issues link to an open task, and several of those links are loose.
- 10 are partially covered.
- **122 have no open task.** That includes 4 of the 5 P0s and 18 of the 23 P1s.

**File these now: untracked P0s and P1s.**
- **P0:** [cross-cutting-01] [model-02] [protect-summary-04] [gap-02] (newly found; re-rated P0 on second verification).
- **P1:** [provider-01] [provider-02] [provider-04] [provider-05] [voice-speech-01] [fulltrack-01] [fulltrack-02] [protect-summary-02] [entry-exit-handoff-24] [cross-cutting-03] [cross-cutting-05] [a11y-01] [coverage-04].
- **P1, newly found:** [new-provider-01] [new-cross-cutting-01] [new-a11y-terminal-visual-01] [new-a11y-terminal-visual-02] [gap-01] [gap-03].

**Existing tasks to update.**

| Task | Change | Issues |
|---|---|---|
| task-26837 (To Do, high) | Re-scope to a credential-only save on Model skip, and deliberately rewrite task-25820's 34 pins | [model-01] |
| task-33001.10 (To Do, medium) | Raise to high (reproduced 8 of 8 times) | [entry-exit-handoff-04] |
| task-33621.25 (To Do) | Call out the module-path unlock as a user lock-out, not one of 26 W003 rows | [protect-summary-01] |
| task-33008 (To Do, high) | Extend to the 'Continue setup' contract; amend AC#8, which asks for 'Verified against' stamps that CLAUDE.md forbids | [entry-exit-handoff-12] [entry-exit-handoff-27] [cross-cutting-18] |
| task-33008.5 (To Do) | Widen to hosted-provider copy, wrapping and Retry placement | [model-07] |
| task-33005.9 / task-33005.7 (To Do) | Add the wizard as a consumer of the shared evidence and the readiness vocabulary | [provider-03] [provider-06] [cross-cutting-07] [entry-exit-handoff-23] |
| task-33626 (To Do) | Widen the primary-button contrast and focus treatment beyond Console | [a11y-09] [a11y-11] |
| task-33620.10 (To Do) | Name the local Custom section, not just the cloud templates | [entry-exit-handoff-25] |
| task-33079 (To Do) | Update the premise: concurrent installs serialize through the session lease | [voice-speech-08] |
| task-33623 (To Do) | Absorb the setup registry (E-b) with wizard-specific acceptance criteria | [cross-cutting-11] [cross-cutting-12] |
| task-28019 (To Do) | Narrow it to its modal-sequencing AC, and move the documents-first Welcome choice into its own task | [entry-exit-handoff-13] [entry-exit-handoff-28] |
| TASK-31813 (Done) | Correct AC#2: it was ticked on the palette path only and never wired for Settings | [cross-cutting-05] |

**Hygiene.** These come from the review's remediation-prerequisites idea:
- Close the stale TASK-33002.13 and the archived TASK-187.
- Tick or strike the acceptance criteria on the 14 Done wizard tasks that still have unticked boxes. Until then, a Done wizard task is not evidence that anything works.


---

## 6. Issue register
151 verified issues: **5 P0 · 23 P1 · 71 P2 · 52 P3**. Each was found by a discovery agent, merged, and then independently re-checked by an adversarial verifier (live re-run for every P0/P1; code trace for the rest). Issues marked *newly found* were discovered during verification and re-checked by a second skeptic. Corrections from verification are folded in; where the verifier proposed a better fix, it is given as the recommended fix.
Merged during verification: `new-protect-summary-01` (the re-encryption mechanism) is folded into [protect-summary-04]. Items tagged *stub* rest mainly on a stub OpenAI-compatible server (no real local LLM runtime on the review machine).
Severity: **P0** blocks setup or a working first chat, or loses data / locks the user out · **P1** major confusion or a silent wrong outcome · **P2** annoyance with a workaround · **P3** polish.

### P0 — blocking (5)

#### [cross-cutting-01] Setup finishes with ✓ and 'Start chatting', but the first chat is blocked (no context window, non-chat models, no end-to-end readiness check)
**Area:** Cross-cutting · **Who:** first-time, power-user · **Verification:** confirmed-with-correction · **Effort:** L · **Backlog:** —

**What happens.** The wizard declares success while the first message cannot be sent. Summary: '✓ Provider — openai', '✓ Default model — tts-1-hd-1106', with 'Start chatting' as the primary action. The first send returns 'System: This request cannot fit the selected model. Response reservation and safety margin leave no model input capacity. Summarizing older turns cannot make enough room. Repair the model limit, reduce mandatory context or the response maximum, or allow older turns to be omitted.' The composer then shows 'Send blocked — resolve response recovery first'. The prefilled default gpt-5.6-terra fails identically; the switcher shows it and its siblings (gpt-5.6-sol, gpt-5.6-luna) as '~4k'. A clean Full run with gpt-5.4-mini was also blocked. Only gpt-4.1-mini, picked by hand in Console, replied ('Hello there! Hope you're well.'), 621 s after launch.

**Verified detail.** Root cause is upstream of the wizard and more specific than stated. Utils/token_counter.py:497-503 has PROVIDER_CONTEXT_WINDOWS['openai'] = 4096, the 'API default' tier of resolve_context_window (:527-564). Every OpenAI id missing from MODEL_TOKEN_LIMITS and DEFAULT_MODEL_CAPABILITIES (all gpt-5*, chatgpt-4o-latest, tts-*) resolves to exactly 4096. [api_settings.openai] max_tokens = 4096 (config.py:4546), so console_context_policy.py:293-302 computes window − reservation − margin ≤ 0. This is NOT a TASK-32709 regression.

**Why it matters.** First-time (Sam) follows every recommendation and gets a ✓ Summary. Then comes a jargon error and a stuck composer. Recovery needed Retry, Discard, Ctrl+T, Alt+M, a search and 6 arrow presses, and most users would conclude the key or the app is broken. Power user (Riley): a Summary ✓ is no evidence of anything, so they must debug context budgets in the first minute.

**Fix.** Add a final 'Ready to chat?' check to the Summary. Compute it with the same preflight Console runs before a send (context-budget resolution plus model capability classification), not a wizard copy. UI: a top Summary row reading either '✓ Ready to chat — OpenAI · gpt-4.1-mini (1M context)' or '✗ Can't chat yet — gpt-5.6-terra's context size is unknown, so every message would be blocked.', with [Choose another model] (returns to Model with the reason pinned) and [Set context size…]. Optionally, behind the consent copy 'Send a one-word test message? Uses a few tokens.', run a 1-token completion and show the reply latency.

**Designer's note on the fix.** Split the fix. The P0 fix is small and belongs in the catalog, not in an L-sized wizard gate. (1) Drop or raise the stale openai 4096 'API default' so unknown OpenAI models fall to the 32000 estimated fallback, or 128k. (2) Add ^gpt-5/^gpt-6/^chatgpt patterns and a context_window for gpt-5.6-terra (model_capabilities.py:94). (3) When the window is only estimated, cap the response reservation, e.g. min(max_tokens, window/4), so no unknown model is ever pre-blocked to zero.

<sub>Evidence: evidence/live-firsttime-cloud/12-model-step.txt; evidence/live-firsttime-cloud/30-summary.txt; evidence/live-firsttime-cloud/36-console-sent-t8s.txt; evidence/live-firsttime-cloud/39-switcher-terra-highlighted.txt</sub>

#### [gap-02] Moonshot (Kimi) setup ends with ✓, but every reply is blocked with 'Provider continuation could not be persisted' *(newly found)*
**Area:** Gap checks · **Who:** first-time, power-user · **Verification:** confirmed-with-correction · **Backlog:** —

**What happens.** With a valid key, Summary shows '✓ Provider — moonshot / ✓ Default model — kimi-k3'. The first message shows '⚠ Provider continuation could not be persisted; retry or recover the interrupted run.' plus 'Response delivery status is unknown on the source device… [Retry anyway] [Discard]', and the composer then shows 'Send blocked — resolve response recovery first'. kimi-k2.6 fails the same way. The provider call succeeds: the log shows 'agent step error … step=0' about 4 s after routing. Every kimi reply carries reasoning_content, so moonshot.py builds a 'complete' continuation checkpoint even on a one-turn plain reply. persist_continuation then swallows the exception (agent_runtime.py:1454-1455), so nothing names the failed precondition. Anthropic and llama.cpp work. The root cause is not pinned beyond this chain, and no backlog task covers it.

**Verified detail.** Core claim reproduced independently, and two parts are corrected. 1. Live run (vn4-b, real key, kimi-k3 typed). Provider showed 'Key staged — it will be checked when you continue. Connection testing is unavailable for this provider.' Model listed 'kimi-k2.7-code (recommended)', kimi-k3, kimi-k2.6, kimi-k2.7-code-highspeed. The recommended row differs from the prior run's kimi-k3 (see model-02). Summary showed '✓ Provider — moonshot' and '✓ Default model — kimi-k3'.

**Fix.** Reproduce in a pilot test with a stubbed tool-free kimi turn that carries reasoning_content, then fix the failing precondition. Log the store's RuntimeError message in persist_continuation. When preserving reasoning on a final turn fails, keep the delivered reply, note 'reasoning not preserved' in the trace, and don't raise the delivery-unknown panel. Add kimi-k3 to the provider smoke matrix.

**Designer's note on the fix.** Right priorities. Re-order them and sharpen one. (a) Ship step 3 first and on its own: it is the user-facing fix. The app knows the reply arrived (the persisted 'Model response completed' step), so 'Response delivery status is unknown on the source device' and its duplicate-send warning are false here. When the optional reasoning-replay checkpoint can't be stored on a tool-free final turn, persist the reply as an ordinary assistant message without the continuation. Replay is 'accepted and never required' per model_capabilities.py:659-663.

<sub>Evidence: evidence/verify-gaps/69-gap5c-model.txt; evidence/verify-gaps/6a-gap5c-summary.txt; evidence/verify-gaps/6b-gap5c-first-reply.txt; evidence/verify-gaps/6c-gap5c-error-expanded.txt</sub>

#### [model-01] Next/Enter on Model without a pick silently throws away the provider and key just entered
**Area:** Model · **Who:** first-time, power-user · **Verification:** confirmed-with-correction · **Effort:** M · **Backlog:** task-26837 (To Do, high)

**What happens.** The Provider step only stages the connection in memory. The provider, key and endpoint are written only when the Model step commits a model. With no row pressed, Next/Enter on Model returns 'skip-safe' and writes nothing, so the staged key is dropped. The UI never says so: the tracker keeps Provider ✓ and gives Model an unexplained amber '!'. The Summary then blames the user with '✗ Provider — no credentials or saved endpoint', although they had entered credentials. Protect still offers 'Set a password' for a key that was never saved. On the server-down path the only warning is 'Continue anyway? / The server couldn't be reached, so this model setup is unverified. Continue anyway?'. It does not say that continuing without a model saves nothing. Nothing is preselected, not even the '(recommended)' row.

**Verified detail.** The core claim holds, but the impact narrative is partly wrong. (1) The user is not 'landed in Console believing setup succeeded'. With this state the Summary hides 'Start chatting', and its focused primary is 'Review provider setup' (FRW:7826-7835). (2) The key is not lost at the Model step. It stays staged in memory: after 'Review provider setup' the Provider step still showed the masked key and 'Found 13 model(s) for Anthropic.' It is lost only when the wizard is exited (Explore Home, Add your first document, Write your first note, Review settings, Esc, or a quit), because the checkpoint s…

**Why it matters.** First-time (Sam, Jo): believes setup succeeded, lands in Console with chat blocked and the key gone. The draft scrubs secrets, so Resume cannot restore it, and the key must be found and pasted again. Privacy-minded local user: an explicit local choice is replaced by a leftover cloud default, and Home's 'Set up Console model' opens Settings preset to OpenAI. Power-user: the same trap whenever they accept the visible recommendation with Enter.

**Fix.** Fix the collision explicitly, without silently auto-committing a model (this respects task-25820's ruling). (1) After _render_models mounts real rows, seat the highlight, not the press, on the recommended or current row (radio_set._selected = idx). Enter on the focused list then presses that visible row through the existing _select_highlighted: one deliberate keystroke accepts the recommendation. (2) Extend ModelStep.confirm_before_advance with a dedicated three-way dialog whenever a provider is staged and no model is chosen. Title: 'No model chosen'.

**Designer's note on the fix.** Direction is right, but the order and one sub-fix need changing. (a) The root fix should be the 'optional owner call', not an afterthought. When Model is skipped, persist the staged credential and endpoint into api_settings.<p> without touching chat_defaults. The 08-12 atomic write protects the consistency of the chat_defaults (provider, model) pair; a credential-only write does not break that. Skip-safe would then mean 'don't change my default model' rather than 'discard what I typed', and the P0 data loss disappears whatever the user presses.

<sub>Evidence: evidence/live-firsttime-cloud/cloud2/12-anthropic-model-step.txt; evidence/live-firsttime-cloud/cloud2/13-model-enter-nothing-selected.txt; evidence/live-firsttime-cloud/cloud2/15-summary-no-model.txt; evidence/live-firsttime-local/33-local2-model-step.txt</sub>

#### [model-02] '(recommended)' is just the API's first id: often a TTS/realtime model, different each run
**Area:** Model · **Who:** first-time, power-user · **Verification:** confirmed-with-correction · **Effort:** M · **Backlog:** —

**What happens.** The '(recommended)' tag goes on whatever id the discovery endpoint returned first. Nothing is curated, filtered or sorted. OpenAI's /v1/models order is not stable, so four fresh runs recommended four different models, one of them a text-to-speech model. 10 of the 20 OpenAI rows cannot chat at all (tts-1, gpt-realtime-*, gpt-4o-transcribe-diarize, omni-moderation, gpt-image-1-mini, gpt-audio-1.5, sora-2, babbage-002). On Ollama, embedding models such as 'nomic-embed-text:latest' are offered as chat defaults. The wizard accepts any of them and the Summary ticks it ('✓ Default model — tts-1-hd-1106').

**Verified detail.** (1) The order is not random per call. OpenAI serves a few distinct orderings: 4 distinct heads in 8 calls, each repeated. The `created` fields in those heads are unsorted, which falsifies the FRW:1002-1005 comment ('roughly chronological'). That comment should be removed, because the [:20] slice was justified by it. (2) When the recommended row happens to be a chat model, the first chat works: o3-2025-04-16 was recommended, picked and sent, and the reply was 'hello there'. So the harm is probabilistic, though frequent.

**Why it matters.** First-time: reads 'recommended' as the app's evaluated advice, picks a speech model, and the first message fails with a jargon block ('This request cannot fit the selected model…'). Trust in every later recommendation drops. Power-user: noisy, unordered list. Advice that changes per run reads as broken.

**Fix.** Add one pure ranking helper in first_run_setup_state, e.g. rank_setup_models(provider, discovered, curated, capabilities) -> (recommended_id | None, ordered_ids, hidden_ids), and render from it. (1) Filter non-chat ids. Use explicit metadata first (task/type fields, OpenRouter output modalities, and for Ollama the /api/show capabilities lacking 'completion'), then a conservative denylist for OpenAI-family ids: tts, whisper, transcribe, realtime, audio, image, dall-e, sora, embed(ding), moderation, babbage, davinci, search-preview, deep-research. Hidden ids stay reachable through 'Show all models' and the free-text field.

**Designer's note on the fix.** Sound and proportionate, with three adjustments. (a) Rank and filter before the [:20] slice, not after; otherwise the filter only thins an already random 20. (b) Prefer metadata the app already has before a hand denylist: OpenRouter architecture.output_modalities, and the models.dev capability data already wired via model_capabilities._models_dev_capabilities. Keep the OpenAI-family denylist as the last resort, because OpenAI's listing has no modality fields. (c) The dependency is sharper than stated.

<sub>Evidence: evidence/live-firsttime-cloud/12-model-step.txt; evidence/live-firsttime-cloud/13b-model-picker-full-enumeration.txt; evidence/live-firsttime-cloud/30-summary.txt ('✓ Default model — tts-1-hd-1106'); evidence/live-power-user/54-run2-model-openai.txt</sub>

#### [protect-summary-04] Protect ignores existing encryption: Enter reopens setup; a 2nd password strands keys
**Area:** Protect / Summary · **Who:** first-time, power-user · **Verification:** confirmed-with-correction · **Effort:** S · **Backlog:** —

**What happens.** Success only updates the status line. The button keeps its 'Set a password' label and focus while the hint reads 'Enter / Ctrl+N next · Ctrl+B back · Esc exit setup', so Enter, meant as 'continue', reopens 'Set up master password' (live). Submitting again calls enable_config_encryption a second time. encrypt_api_keys_in_config regenerates password_verifier for the new password (config.py:1101-1104) but leaves values that are already enc: untouched (1124-1132), so with a different password the keys become unreadable while the new password 'works'. On a re-run, on_show decides through stored_plaintext_key_present, which returns False whenever encryption is enabled (first_run_setup_state.py:1392-1393). The step therefore says 'No API keys saved yet — nothing to protect.' about keys that are saved and encrypted, or offers 'Set a password' as if none existed.

**Verified detail.** The consequence is worse than the issue claims, and some details differ. The second enable_config_encryption writes the new verifier to disk and only then fails to publish the runtime config ('Configuration runtime reload failed'), so it returns False. The step then shows both '✓ Encryption enabled.' and 'Enabling encryption failed — your keys are unchanged (plain text).' The second message is false: the key on disk is enc:, encrypted under the first password. After that, Next shows 'Saving setup progress failed.

**Why it matters.** First-time: reads the reopened dialog as 'did it fail?'. Typing a password again, especially a different one, silently breaks the keys they just protected, and the next launch accepts the new password but the providers fail. Power user re-running setup: is told nothing is saved. Violates Nielsen #1 (status), #5 (error prevention) and #4 (the hint line contradicts what Enter does on this focus).

**Fix.** Derive three Protect states from persisted config, both on show and after success. (1) Encryption on: status '✓ Your saved API keys are encrypted. You'll enter your master password when chatbook starts.'; the button becomes 'Change password…' (PasswordDialog mode=change, then change_encryption_password); focus moves to Next. (2) Plaintext key on disk: the current offer. (3) Nothing on disk: see protect-summary-06. Guard enable_config_encryption so it returns False (logging 'already enabled') when [encryption].enabled is true; then no caller can rotate the verifier without re-encrypting. While 'Set a password' has focus, show the hint 'Enter set password · Ctrl+N next · Esc exit setup'.

**Designer's note on the fix.** Refusing enable_config_encryption when [encryption].enabled is already true is the right minimum. Also fix the write-then-validate ordering: validate (strict-decrypt the existing enc: values) before _write_raw_cli_config_unlocked, or restore the previous bytes when publish fails, so a failure never leaves the disk changed while the UI says 'unchanged'. On success, hide or relabel the button and move focus to Next. That is simpler and more robust than a hint line that depends on focus. 'Change password…' needs two passwords (see 03).

<sub>Evidence: evidence/live-firsttime-cloud/27-password-enter.txt; evidence/live-firsttime-cloud/28-protect-enter-again.txt; evidence/live-firsttime-cloud/29-protect-modal-escape.txt; tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py:7452-7483 (on_show), 7485-7504 (button always reopens setup mode), 7540-7550 (success only…</sub>

### P1 — major (23)

#### [a11y-01] At 80x24 the chrome eats 14 of 24 rows; Summary's read-back shrinks to a 3-row window
**Area:** Accessibility & terminal · **Who:** first-time, power-user · **Verification:** confirmed-with-correction · **Effort:** M · **Backlog:** —

**What happens.** At 80x24 the fixed frame takes about 14 of the 24 rows: the border, the 'Set up tldw chatbook' title (2 rows plus a separator), the 3-row tracker plus a separator, the nav bar ('Step 1 of 6 ← Back Next → Skip setup', with a blank row above and below it), and the bottom hint row. The step itself gets about 10 rows. - **Summary:** only 'Setup summary' and one line show: 'Left at recommended defaults: tools off, RAG off, default theme, notes'. Below that sit the docked buttons 'Review provider setup / Add your first document / Write your first note / Explore Home / Review settings', followed by 3 wasted blank rows. Every ✓/✗ row, including '✗ Provider — no credentials or saved endpoint', is off screen. - **Provider:** after a rejected key, the first visible step line is 'Authentication section on this step.', the tail of '✗ Unauthorized - the server rejected this API key.

**Verified detail.** (1) 'padding: 2' (BaseWizard.py:62-69) is not in effect. The rendered content touches the border at every size. The 14 chrome rows are: border 2 + title 3 + tracker 3 + separator 1 + nav 4 + key-hint row 1. The count is right, but that part of the cause is wrong. (2) The Provider symptom did not reproduce as written. Typing 'hello' and pressing Ctrl+N advanced to Model with Provider ✓ in 2 of 2 of my runs (that is provider-03). What I saw instead is worse: after Back, the pinned error plus the frame leave 6 body rows, and the key field is cut off below them.

**Why it matters.** First-time user on a default terminal: Summary offers a primary 'Review provider setup' with no visible reason, and the key error reads as a sentence fragment. The ✓/✗ rows can only be found by scrolling, and the scroll thumb is nearly invisible (a11y-08). Power user in an 80-column tmux split: every step scrolls, and 58% of the screen is frame.

**Fix.** Add a 'short' layout tier, switched on in on_resize when the screen is under 30 rows. Do this the way _sync_compact_mode already works: add a '-short' class to SetupWizardContainer. 1. Merge the title and tracker into one row: 'Set up tldw chatbook · Step 2 of 6 · Provider ✓ ● ○ ○ ○ ○'. 2. Make the nav bar 1 row (height 1, no blank padding rows). In short mode drop the outer border and padding ('border: none; padding: 0 1'). 3. Merge the key-hint row and the size-hint row into one (a11y-03). That gives back about 9 rows, so the step gets about 19 of 24 rows, roughly twice what it has now.

**Designer's note on the fix.** The direction is right: a '-short' class set on resize, following the _sync_compact_mode precedent. Four concerns. (a) A one-row merged title+tracker must keep the amber '!' attention state from TASK-21143 and the current step's name. The proposed '✓ ● ○' glyph row drops '!'. (b) Hiding 'Add your first document / Write your first note' behind 'More ▾' undoes the deliberate visible exits from task-32072 and task-32140. Instead, remove the row margins and padding-bottom (about 3 rows) and drop the outer border in short mode (2 rows). That reaches about 8 read-back rows without a popover.

<sub>Evidence: /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/setup-wizard-ux-qa/evidence/live-resilience/01-80x24-welcome.txt; /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/setup-wizard-ux-qa/evidence/live-resilience/08-80x24-provider-hello-next.txt; /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/setup-wizard-ux-qa/evidence/live-resilience/17-80x24-summary.txt; /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/setup-wizard-ux-qa/evidence/live-resilience/19-80x24-summary-scroll-sequence.t…</sub>

#### [new-a11y-terminal-visual-01] At 100x30 and below, arrowing the Provider list scrolls the focused list off screen; the user picks a provider without seeing the list *(newly found)*
**Area:** Accessibility & terminal · **Verification:** confirmed-with-correction · **Backlog:** —

**What happens.** On a fresh profile at 100x30 (the wizard's own recommended minimum) and at 80x24, one Down press in the provider list scrolls the step until the list is entirely out of view. The list keeps focus, and each further Up or Down silently re-selects a provider, because selection follows highlight. The only feedback is the status line ('Couldn't discover models for Anthropic…'). After Back from Model, focus lands on the same off-screen list, and End silently switched the provider to 'vLLM (legacy alias)'. At 120x40 the list stays visible.

**Verified detail.** Reproduced on fresh profiles with no found-server banner at 100x30 (vn3-a) and 80x24 (vn3-c). At 120x40 (vn3-d) the list stayed visible. Every claim held: one Down hid the list, End selected 'vLLM (legacy alias)', Back from Model left focus on the off-screen list, and End then switched to 'vLLM (legacy alias)' again. Typing 'x' never reached the key field, which confirms the hidden list kept focus. The trigger is narrower than 'every arrow press'.

**Fix.** Suppress Textual's auto-scroll for programmatic collapse changes: a SetupCollapsible subclass that skips scroll_visible while a flag is set, set around FirstRunSetupWizard.py:2891. After a provider change, call call_after_refresh(choices.scroll_visible) so the highlighted row stays in view. Combine with the minimal-scroll focus fix from a11y-02. Add pilot tests at 80x24 and 100x30: after Down and End in the list, the highlighted option is inside the step's visible region.

**Designer's note on the fix.** The direction is right: programmatic collapse changes must not scroll. The mechanism as written fails, though. Collapsible defers scroll_visible through call_after_refresh, so a flag set and cleared around `auth.collapsed = …` at FRW:2891 is already cleared when the deferred call runs. Invert it instead. Use a SetupCollapsible whose _on_collapsible_title_toggle sets a one-shot `_user_toggled` flag and whose _watch_collapsed scrolls only when it consumes that flag. Every programmatic change is then silent by default, with no call-site discipline needed.

<sub>Evidence: /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/setup-wizard-ux-qa/evidence/verify-a11y-terminal-visual/05-80x24-provider-arro…; /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/setup-wizard-ux-qa/evidence/verify-a11y-terminal-visual/35-100x30-provider-aft…; /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/setup-wizard-ux-qa/evidence/verify-a11y-terminal-visual/36-100x30-provider-aft…; /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/setup-wizard-ux-qa/evidence/verify-a11y-terminal-visual/37-100x30-provider-aft…</sub>

#### [new-a11y-terminal-visual-02] Skipping the key after a rejected key traps Model in a 'Saving the provider and model setup failed' loop *(newly found)*
**Area:** Accessibility & terminal · **Verification:** confirmed-with-correction · **Backlog:** —

**What happens.** Quick track at 120x40 (also seen at 80x24). Pick OpenAI, type a wrong key, Ctrl+N: the wizard advances to Model with Provider ✓. Go Back, clear the key, and press Enter in the empty field, the documented skip. Model still shows 'Models for OpenAI.' with the pre-typed 'gpt-5.6-terra'. Next opens 'Continue anyway?', and Continue shows 'Saving the provider and model setup failed. Retry with Next, or go Back.' Retrying loops every time. The only escape I found is not obvious: clear the pre-typed model ID, then Continue anyway (that reached Voice). Exit setup also works but abandons setup.

**Verified detail.** Reproduced live at 120x40 on a fresh profile (vn3-b): OpenAI, 'hello', Ctrl+N gave Model with Provider ✓ and 'Authentication failed — this API key was rejected…' ([provider-03]). Ctrl+B, clearing the key and Enter gave Model 'Models for OpenAI.' with 'gpt-5.6-terra', tracker Provider '!', and the row 'Couldn't reach the server (request failed). Check it's running, then Retry — or enter a model ID below.' Ctrl+N opened 'Continue anyway? / The server couldn't be reached, so this model setup is unverified.', and Continue gave 'Saving the provider and model setup failed.

**Fix.** When Provider takes its skip path, clear the Model step's provider context and the template-prefilled model ID, or make Model's commit treat 'nothing staged' as skip-safe. Replace the generic 'Saving … failed' with the real reason and a working action. Add a pilot test for: rejected key, Back, Enter-skip, Next, ending on Voice with Provider and Model shown as skipped. This is adjacent to provider-01, provider-03 and model-04 but not covered by them.

**Designer's note on the fix.** Fix the source, not Model. The real defect is that 'skip the provider' clears ProviderStep's selection but not the wizard-level staged draft. The reset already exists: call clear_provider_setup_sensitive_state(clear_widgets=False) (FRW:8233) from the skip path, so Model honestly renders 'Pick a provider first'. Better still, fold it into one unselect_provider() that also resets the Authentication form, key field and status line, which fixes [new-provider-01] in the same change. Reject the proposal's alternative of making Model's commit treat 'nothing staged' as skip-safe.

<sub>Evidence: /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/setup-wizard-ux-qa/evidence/verify-a11y-terminal-visual/41-120x40-after-ctrl-n…; /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/setup-wizard-ux-qa/evidence/verify-a11y-terminal-visual/44-120x40-after-enter-…; /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/setup-wizard-ux-qa/evidence/verify-a11y-terminal-visual/46-120x40-after-contin…; /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/setup-wizard-ux-qa/evidence/verify-a11y-terminal-visual/48-120x40-retry-contin…</sub>

#### [coverage-04] An exported API key skips setup and the toast says 'you're ready to chat' without any readiness check; a non-OpenAI-only key leaves Console unready
**Area:** Coverage & power use · **Who:** power-user, first-time · **Verification:** confirmed-with-correction · **Effort:** M · **Backlog:** —

**What happens.** should_offer_wizard() returns False as soon as any provider env key resolves, so the typical developer setup never sees the wizard (provider and model choice, tools posture, voice, encryption, model-list consent). The one-time toast hard-codes “you're ready to chat” without checking that chat_defaults.provider is the provider whose key was found, or that its default model can accept a turn. Live, the template default (OpenAI / gpt-5.6-terra) blocked the first send; with only a non-OpenAI key, Console would default to an unkeyed OpenAI.

**Verified detail.** (1) 'The only route back is a vanishing toast' is overstated. In the non-OpenAI case Console shows its own 'Get started' card: '1. ● Connect a provider (API key or local server) / Not ready · no key ... Composer unlocks after setup / [Set up provider]'. The palette also has 'Setup: Run setup wizard…'. But that card's Set up provider opens Settings ▸ Providers & Models preselected on OpenAI, with 'API key source: missing; set OPENAI_API_KEY or paste a local key'. It never mentions the ANTHROPIC_API_KEY that was found.

**Why it matters.** Power user: the fastest path ends in a blocked first message right after being told they're ready; the way back is a toast that disappears after 10 s and a Settings path about 11 keys deep. First-time user with a key exported for another tool: the same false claim, and no idea that setup exists. Violates #1 Visibility of system status (false readiness), #9 Help users recognize, diagnose and recover from errors, #3 User control and freedom (the toast has no action).

**Fix.** Keep env keys as a fast path, but make it honest and actionable. Behaviour: on a fresh profile with env keys and an untouched template chat_defaults, (a) if the template default provider has no key and exactly one env-keyed provider exists, set chat_defaults to that provider and its curated default model; (b) run the shared readiness check (provider readiness plus model context resolution) before claiming anything. UI: replace the toast with a compact one-screen dialog, “We found your keys”: “Found keys for OpenAI, Anthropic, OpenRouter. New chats use: [OpenAI ▾] [model ▾] [Start chatting] [Full setup…] [Not now]”.

**Designer's note on the fix.** The direction is right. Order it differently: (a) the cheap, immediate fix is the toast copy. Compute get_provider_readiness(chat_defaults.provider) and say 'ready' only when it is ready; otherwise say 'Found ANTHROPIC_API_KEY, but new chats use OpenAI, which has no key' with a 'Set up' action. (b) Don't write chat_defaults silently at boot. A config mutation keyed off env presence outlives the env var and surprises dotfile users. Make the existing Console 'Get started' card env-aware instead: 'Found ANTHROPIC_API_KEY · [Use Anthropic]'.

<sub>Evidence: evidence/live-power-user/02-envkeys-wizard-skipped-toast.txt; evidence/live-power-user/75-first-chat-after-setup.txt, 77-blocked-chat-tab.txt, 107-power2-default-model-chat.txt; evidence/static-config-coverage/07-env-key-path.txt; tldw_chatbook/UI/Wizards/first_run_setup_state.py:704-746, 861-916 (offer suppression; env_keys_that_silenced_first_run)</sub>

#### [cross-cutting-03] Next silently means skip, save defaults, or refuse depending on the step; no explicit Skip
**Area:** Cross-cutting · **Who:** first-time, power-user · **Verification:** confirmed-with-correction · **Effort:** M · **Backlog:** —

**What happens.** Welcome promises '…most steps can be skipped with Next'. On Provider, Next with no key shows 'API key required. Set OPENAI_API_KEY or add api_key under [api_settings.openai]. Retry with Next, or go Back.' indefinitely. Only Enter inside the empty key field skips, and a mouse user has no skip at all. On Model, Next with no model chosen skips the step and silently discards the staged provider and key: the Summary reads '✗ Provider — no credentials or saved endpoint' and chat_defaults stays the template OpenAI / gpt-5.6-terra, even for a working local Ollama/llama.cpp. Voice says 'skip with Next if you don't want voice', but Next writes [app_tts] OPENAI_BASE_URL = http://127.0.0.1:8765/v1/audio/speech, and the Summary then claims '✓ Voice — PocketTTS (default voice)'. Protect says 'Skip to leave keys as plain text', but there is no Skip control.

**Verified detail.** (1) Provider is not a dead end. The key field's own hint says 'No key yet? Enter skips this step — you can add a provider later in Settings.', and Exit setup works. The gap is the missing clickable skip; every terminal user has an Enter key. (2) 'Retry with Next' can succeed once a key is pasted. (3) The Model skip discards the staged provider, but the Summary shows it ('✗ Provider' plus 'Review provider setup'), and the key is still staged if the user goes Back. (4) The Voice write is worse than described because it is a hybrid.

**Why it matters.** First-time users cannot predict whether Next will save, skip or block. A mouse-only user is stuck on Provider, and a local user's working connection is lost on Next. Power users find that an untouched Voice step on a re-run breaks a working OpenAI TTS endpoint, and cancelling does not undo it. Violates H5 Error prevention, H3 User control and freedom (no explicit skip, no undo), H4 Consistency and standards and H1 Visibility of system status.

**Fix.** Separate the two intents in the footer. Add a 'Skip step' button between Back and Next, on every step except Welcome and Summary, bound to Alt+S and listed in the hint line. Its contract: nothing from this step is saved, and the outcome is recorded as 'skipped'. Next then means 'save what you chose and continue'; label it 'Save & continue' when the step will write and 'Continue' when it won't (from a step.will_write() query). Per step: - Provider: Skip clears the selection (today's skip_without_key) and is reachable by mouse. Next with no key shows the inline 'Paste a key above, or choose Skip step.' - Model: when a provider is staged, Skip asks 'Skip choosing a model?

**Designer's note on the fix.** Right direction, but a third footer button on every step adds chrome for one real problem. More proportionate: (a) delta-commit Voice so an untouched step writes nothing, as Tools and Appearance already do; (b) on Model with a staged provider, pre-select the recommended chat row so Next saves it, and if the user clears it, refuse inline with 'Pick a model to save your Anthropic key — or skip'; (c) add a visible 'Skip this step' link only where Next refuses, in the Provider key panel; (d) fix the Welcome and Protect copy ('Next without a password keeps keys as plain text').

<sub>Evidence: tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py:7384-7388 (Welcome copy); tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py:4316-4322 (Voice 'skip with Next') vs :5127-5167 (commit always saves); tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py:3248-3263 (Provider refuses Next), :9874 (suffix); tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py:4194-4198 (Model skip-safe drops the staged provider)</sub>

#### [cross-cutting-05] Re-running setup ignores current provider, model, voice, RAG and track, uses first-run copy, and cancelling from Settings lands on Console
**Area:** Cross-cutting · **Who:** power-user, first-time · **Verification:** confirmed-with-correction · **Effort:** L · **Backlog:** —

**What happens.** Settings promises 'Re-run the guided first-run setup with current values.' The re-run shows the first-run Welcome: 'Welcome to tldw chatbook', Quick preselected even though the last run was Full, a 'Skip setup' button, and an Esc dialog asking 'Skip setup and stop showing it at launch?'. Step by step: - Provider: nothing is selected or marked, although openai is current. - Model: 'Models for your provider.' and 'Pick a provider first — or type a model name below'. - Voice: PocketTTS is preselected although OpenAI/shimmer is saved. Next rewrote OPENAI_BASE_URL to http://127.0.0.1:8765/v1/audio/speech and OPENAI_AUTH_MODE to 'none'. - RAG: the current default is not marked. - Only Tools (5 gates back on) and Style ('nord (current)') prefill. The tracker then shows '!' on Provider and Model for a healthy config.

**Verified detail.** TASK-31813 AC#2 did not regress for the Settings entry; it was never wired there. Commit df645d94c1 (TASK-31813) touched no settings_screen.py. The handler at settings_screen.py:31172-31183 still passes the public alias handle_first_run_wizard_result, which drops cancel_to_console, so the default True routes to TAB_CHAT (app.py:3921-3923, 3996-4006). The AC was ticked on the palette path only. read_wizard_prefill already exposes provider_value and model_id (first_run_setup_state.py:1765-1778), but ProviderStep never reads them.

**Why it matters.** Power user (Riley): re-running to change one setting costs a full walkthrough and silently breaks TTS, and the '!' glyphs suggest the config is broken. A returning first-time user reads 'Welcome to tldw chatbook' as 'my setup was lost'. Violates H1 Visibility of system status (current state not shown), H3 User control and freedom (cancel jumps to Console), H4 Consistency (Style marks '(current)', Provider does not) and H6 Recognition rather than recall.

**Fix.** Build a RerunContext once from read_wizard_prefill() and pass it to every step. 1. Welcome on a re-run: title 'Review your setup', plus a 3-line current-state summary, e.g. 'Provider: OpenAI (key from OPENAI_API_KEY) · Model: gpt-5.4-mini · Voice: OpenAI shimmer'. Preselect the last-used track. The third button becomes 'Close', and its dialog reads 'Close setup? Nothing has changed.' 2. Every list preselects the current value and marks it '(current)': the Provider row, Model (as a pressed radio, not the custom input), the Voice preset with endpoint/model/voice from [app_tts], and the RAG default. The tracker shows ✓ 'kept current' for untouched steps. 3.

**Designer's note on the fix.** Sound. Ship the destructive-path fixes first; both are small. Item 3: delta commit for untouched steps. Item 4: pass cancel_to_console=False from Settings, add a Settings-origin case to Tests/UI/test_first_run_wizard_cancel_route.py, and correct the false AC tick. Items 1-2 ('Review your setup' Welcome, '(current)' marks on Provider, Model, Voice and RAG) are the larger UX lift and match Style's existing '(current)'. The single open_setup_wizard(origin=…) entry point from the merged proposal is the cleanest way to stop the four call sites drifting again.

<sub>Evidence: evidence/live-power-user/80-rerun3-from-settings-welcome.txt; evidence/live-power-user/81-rerun3-provider.txt; evidence/live-power-user/82-rerun3-model.txt; evidence/live-power-user/83-rerun3-voice.txt</sub>

#### [new-cross-cutting-01] Readiness surfaces say 'Ready · not tested' for a model every send is pre-blocked on *(newly found)*
**Area:** Cross-cutting · **Verification:** confirmed-with-correction · **Backlog:** —

**What happens.** On a fresh OpenAI profile with the shipped gpt-5.6-terra, three surfaces report the model as usable: the Console header ('Ready · not tested'), Settings ▸ Overview ('Configuration: OpenAI / gpt-5.6-terra; Status: Ready · not tested') and the model switcher ('~4k Ready · not tested'). Yet the context-capacity preflight refuses every send, and the status bar says 'Context unknown'. The readiness verdict covers credentials and connection only, never capacity, so the app tells the user a blocked model is fine everywhere except at Send. This is the same root as cross-cutting-01, but it lives outside the wizard and would survive a wizard-only fix.

**Verified detail.** (1) The false verdict is not limited to 'not tested'. A successful Settings test shows 'Ready · verified HH:MM' for the blocked model, although that test 'lists models without generating', so 'verified' covers only the key and endpoint. (2) Code trace confirms the mechanism. ConsoleSettingsBlockerCode (Chat/console_session_settings.py:216-228) has no capacity code; its blockers are provider, endpoint, credential, model_missing, endpoint_unreachable and active_run only.

**Fix.** Fix capacity at the source (cross-cutting-01's catalog fix). Then include a capacity facet in the shared readiness verdict, for example 'Ready · context size estimated' or 'Can't send · model context unknown', so the header, Settings, the switcher and the wizard Summary all use the same words.

**Designer's note on the fix.** The order is right: fix capacity at the source first. Better still, make the state impossible. Cap the response reservation whenever the window is only estimated (cross-cutting-01's review suggests min(max_tokens, window/4)), so an unknown model can never reach zero input capacity; the capacity facet then becomes defence in depth, not the main fix. Build the facet as a real blocker code (e.g. 'capacity_exhausted', recovery 'adjust_model_limits').

<sub>Evidence: evidence/verify-cross-cutting/15-settings.txt; evidence/verify-cross-cutting/14-console-first-send.txt (header 'Ready · not tested' + blocked send + 'Context unknown'); evidence/live-firsttime-cloud/44-switcher-search-gpt.txt; tldw_chatbook/Chat/console_session_settings.py:2713-2742 (readiness_words)</sub>

#### [entry-exit-handoff-03] Blocked first send is jargon with no fix action, and its recovery panel says 'accepted'
**Area:** Entry, exits & handoff · **Who:** first-time, power-user · **Verification:** confirmed · **Effort:** M · **Backlog:** task-33621.4 (To Do)

**What happens.** The system row reads: 'This request cannot fit the selected model. Response reservation and safety margin leave no model input capacity. Summarizing older turns cannot make enough room. Repair the model limit, reduce mandatory context or the response maximum, or allow older turns to be omitted.' At the same time the recovery panel says 'Response accepted; waiting for dispatch.' with [Retry] [Discard], and the composer says 'Send blocked — resolve response recovery first'. Retry only adds 'Response failed.' and a second copy of the same system error. There is no 'Switch model' or 'Set context size' action.

**Why it matters.** First-time: 'response reservation', 'mandatory context' and 'model limit' mean nothing, and 'accepted' suggests a delay rather than a block. Power user: has to find the per-provider max_tokens or context settings unaided. Violates Match between system and the real world (#2), Help users recover from errors (#9), Visibility of system status (#1, contradictory states), and Consistency (#4).

**Fix.** Rewrite the system row as 'gpt-5.6-terra's context size isn't known, so chatbook can't fit a reply. [Set context size] [Switch model] [Shorten replies]'. 'Set context size' opens a one-field popover that writes context_window for this provider and model. 'Switch model' opens the Alt+M switcher filtered to models with a known window. 'Shorten replies' lowers max_tokens for this chat. Do not mount the 'Response accepted; waiting for dispatch' panel for a pre-dispatch block. Disable Retry with the hint 'Change a setting above, then retry', and never append a duplicate system row. Composer strip: 'Send blocked — model size unknown ›', linking to the same popover.

**Designer's note on the fix.** The direction is right: name the model and the cause, add [Switch model] and [Set context size], and do not mount the dispatch-recovery panel for a pre-dispatch refusal. Key the copy off the limiting reason. The same blocked() string covers NON_COMPACTABLE and genuine overflow, so 'context size isn't known' must be emitted only for the UNKNOWN_WINDOW plus fallback-ceiling case. The larger lever is -02: stop giving known cloud families a ~4k fallback, so this surface becomes rare. 'Shorten replies' (lower max_tokens) is a reasonable third action, but should say what it changes.

<sub>Evidence: evidence/live-firsttime-cloud/36-console-sent-t8s.txt; evidence/live-firsttime-cloud/37-console-retry.txt; evidence/live-power-user/77-blocked-chat-tab.txt; evidence/live-power-user/107-power2-default-model-chat.txt</sub>

#### [entry-exit-handoff-04] Every successful 'Start chatting' shows a false 'Provider settings changed' warning
**Area:** Entry, exits & handoff · **Who:** first-time, power-user · **Verification:** confirmed · **Effort:** S · **Backlog:** task-33001.10 (To Do, priority medium)

**What happens.** On a fresh profile the first-run handoff toast reads 'Provider settings changed before Console opened. Review setup and try again.' At the same time the status bar shows the right provider and model ('Provider: OpenAI Model: gpt-5.2-codex') and the canvas says 'Ready — type a message to begin.' The released handoff then stays pending for the rest of the session.

**Verified detail.** none of substance. Timing: the toast appeared about 3-6 s after Enter (Console mount lags the click), not 'within about 1 s'. My run 1 was inconclusive because the capture started after the toast had expired.

**Why it matters.** First-time: the first thing a successful setup produces is an error. Users may re-run setup, and pressing Next through Voice can clobber TTS config (cross-area), or they stop trusting later warnings. Power user: learns to ignore toasts. Violates Visibility of system status (#1, false status), Help users recognize errors (#9, an error message for a non-error), and Consistency (#4).

**Fix.** Fence on values, not on the config generation, as option 2 of task-33001.10 suggests. When the generation differs, re-read the saved chat_defaults and accept the handoff if provider_config_key and model still equal the intent. AC#2's stale-handoff protection still holds, because a changed provider or model fails the comparison. Never show a warning when values match. If they genuinely changed, say 'Your default model changed to X after setup — using X.', not 'Review setup'. Add an end-to-end test that boots a fresh scratch profile, finishes the wizard via Start chatting, lets Console mount for the first time, and asserts there is no warning toast and no pending handoff.

**Designer's note on the fix.** Fencing on values is the right, proportionate fix, and it keeps AC#2 of task-33001.10. Alternatives worth weighing: seed the Console rail scope as part of the wizard's completion commit (before staging), or compare a chat_defaults-scoped revision instead of the global config generation. Both remove the race at its source rather than relaxing the fence. Agree on the end-to-end test through Console's real first mount. Also raise task-33001.10's priority.

<sub>Evidence: evidence/live-firsttime-cloud/32-console-arrival.txt; evidence/live-firsttime-cloud/cloud2/20-console-arrival.txt; evidence/live-firsttime-local/40-local2-console-landing.txt; evidence/live-firsttime-local/57-local3-console-landing.txt</sub>

#### [entry-exit-handoff-12] 'Set up provider' and 'continue setup' open an OpenAI-preset Settings form, not setup
**Area:** Entry, exits & handoff · **Who:** first-time, power-user · **Verification:** confirmed · **Effort:** M · **Backlog:** task-33008 (To Do, high), task-33008.3 (To Do), task-33623 (To Do)

**What happens.** All post-setup 'continue' and 'set up' calls to action go to the expert Settings editor, preselected with the template default (OpenAI), instead of returning to setup. The Anthropic choice the user was making is gone, and so is the wizard's local-server discovery. The wizard's own exit copy promises 'You can continue setup any time from Settings ▸ Diagnostics', but a Settings re-run starts again at Welcome and never reads the saved draft. The Console's typing-locked hint says 'press Enter to continue setup', and Enter opens Settings.

**Why it matters.** First-time (Jo, local-first): faces a three-pane editor asking for an OpenAI key with no discovery, and is likely to give up. A cloud user loses the provider they had chosen. Power user: 'continue' restarts from Welcome. Violates Consistency (#4: the same words lead to different places), Match with the real world (#2), User control and freedom (#3), and Recognition rather than recall (#6).

**Fix.** Define one 'Continue setup' contract, built on the -10 entry point. 1) If a valid draft exists, every continue or set-up control (the Console card primary, Enter on the card, the Home card, Settings ▸ Diagnostics, the palette) opens the wizard with resume_draft at the saved step, with the staged provider preselected. Label: 'Continue setup (Provider)'. 2) If there is no draft and no provider, the card primary 'Set up a provider' opens the wizard at the Provider step (new start_step), where discovery runs. 'Providers & Models (advanced)' stays as a secondary link.

**Designer's note on the fix.** One 'Continue setup' contract with resume_draft and a start_step deep link is the right fix and consistent with task-33008. Two refinements. (a) The draft does not hold the highlighted-but-uncommitted provider (my Anthropic exit stored no draft_values), so 'preselect the staged provider' needs the Provider step to checkpoint its selection on exit. (b) Keep the Settings deep link for 'provider exists but not ready', always with the actual provider and never the template default. Fix the exit-dialog and card copy in the same change.

<sub>Evidence: evidence/live-resilience/75-120x40-after-exit-provider.txt; evidence/live-resilience/76-120x40-console-enter-continue-setup.txt; evidence/live-firsttime-local/20-local1-after-explore-home.txt; evidence/live-firsttime-local/21-local1-home-setup-console-model.txt</sub>

#### [entry-exit-handoff-24] First chat asks approval for an internal 'chat_with_llm' tool despite 'Tools — all off'
**Area:** Entry, exits & handoff · **Who:** first-time, power-user · **Verification:** confirmed · **Effort:** M · **Backlog:** —

**What happens.** The Console agent's tool catalog includes the app's own built-in MCP source. mcp_execution_log.jsonl records server_key 'builtin:tldw_chatbook', tool 'chat_with_llm', decision 'denied-unresolved'. The model chose to call it on the very first message. Meanwhile Summary had just said '– Tools — all off; turn them on under MCP ▸ Servers ▸ built-in row ▸ Tool gates'. The direct local runtime marks chat_with_llm as unavailable (_UNAVAILABLE_DIRECT_TOOLS).

**Verified detail.** Frequency is model-dependent: 2 of 4 'Reply with just: hi' sends with gpt-4.1-nano raised the approval (the first send was clean). Approve once -> 'mcp__tldw_chatbook__chat_with_llm · Failed · 0.4s / ERROR: mcp_execution_failed', then the reply 'hi'. On the non-native (fenced) path, llama.cpp and Ollama, chat_with_llm is not in the 16 tools listed in the system prompt, though find_tools and load_tools exist. The root-cause gap is code-confirmed: the built-in inventory lists every manifest tool (local_control_service get_inventory), while the direct runtime marks chat_with_llm unavailable.

**Why it matters.** First-time: is asked to approve something called chat_with_llm with no idea what it means; denying leaves a stuck run, and approving can recurse or cost money. Power user: Summary's tool claims aren't true of the agent. Violates Match with the real world (#2), Visibility of system status (#1), and Error prevention (#5).

**Fix.** 1) Never expose a tool listed in _UNAVAILABLE_DIRECT_TOOLS (chat_with_llm) to the agent catalog. 2) Keep the built-in MCP source out of plain Console chat until the user enables it, either through [mcp] enabled or an explicit switch in MCP ▸ Tool gates via all_tool_gates(). 3) Make the Summary's Tools row enumerate the agent's real catalog (all_tool_gates plus active MCP sources), so 'all off' is true when shown. 4) Add a regression test: on a fresh profile the Console agent's tool list contains no builtin:tldw_chatbook entries.

**Designer's note on the fix.** Point 1 (filter _UNAVAILABLE_DIRECT_TOOLS out of builtin_tools_from_inventory) is the minimal, uncontroversial fix and should ship alone first. Point 2 (keep the whole built-in MCP source out of plain chat until enabled) is a product decision, since the built-in notes and library tools may be intended Console features. Decide it explicitly, not as part of a bug fix. Point 3 (make the Summary's Tools row describe the real agent surface) is right, and is also needed because the status chip shows 'Tools: 6 ready'.

<sub>Evidence: evidence/live-resilience/110-120x40-first-message.txt; evidence/live-resilience/111-120x40-after-stop.txt; homes/res5/.local/share/tldw_cli/default_user/mcp_execution_log.jsonl (server_key 'builtin:tldw_chatbook', tool_name 'chat_with_llm'); tldw_chatbook/MCP/local_runtime_delegate.py:163 (_UNAVAILABLE_DIRECT_TOOLS = {"chat_with_llm"})</sub>

#### [fulltrack-01] RAG step saves an embedding key the RAG pipeline never reads, can't turn RAG on, then the Summary reports '✓ RAG'
**Area:** RAG / Tools / Notes / Style · **Who:** first-time, power-user · **Verification:** confirmed-with-correction · **Effort:** M · **Backlog:** —

**What happens.** Step 5 "Search & RAG" says "Embedding dependencies are installed. Pick a default model, or skip." A pick writes only [embedding_config] default_model_id. The RAG search and ingestion path ignores that key. It resolves its embedding model from the active RAG profile (hybrid_basic → "all-MiniLM-L6-v2"). Settings ▸ RAG's "Embedding model" field edits that same profile value, so it disagrees with the wizard. An isolated probe confirmed it: with default_model_id = openai-text-embedding-3-small written, resolve_active_rag_config().embedding.model still returns all-MiniLM-L6-v2. The Summary nonetheless shows "✓ RAG — embedding model: openai-text-embedding-3-small" (live, power-user 73). Left untouched, it shows "– RAG — off by default — embedding model e5-small-v2" (live, cloud 30 / power2 105), which names a model RAG does not use. The step also offers no control that turns RAG on.

**Verified detail.** Core claim holds live. Three details are overstated or imprecise. (1) Library documents are NOT silently embedded with all-MiniLM-L6-v2 on add. Settings shows "Semantic index not built — Hybrid search is keyword-only until you Backfill" plus a "Review local model" gate, so a local download needs an explicit Backfill or review that names the model. The cost is a dead choice plus a false ✓, not a surprise download. (2) "Settings ▸ RAG" is actually Settings ▸ Domain Defaults ▸ RAG, a collapsed group.

**Why it matters.** First-time: reads "✓ RAG" as "the assistant will now use my documents". It won't: auto-retrieve is off, and the model shown is not the one in use. Power-user: a deliberate choice (API vs local, multilingual) is silently dropped. They discover it only through the Settings ▸ RAG mismatch, or after a local model download they meant to avoid.

**Fix.** Rebuild the step on the seams Settings ▸ RAG already uses. (1) Retitle it "Search your documents". Subtitle: "chatbook can look up passages from your Library (documents, notes, media) and give them to the model while you chat. Nothing is indexed until you add something." (2) Primary control: a SetupRadioSet "When you chat: ( ) Search my Library only when I ask (default) / ( ) Search my Library automatically". Write chat_defaults.rag_auto_retrieve_on_send through settings_library_rag_defaults (chat_defaults is already wizard-owned).

**Designer's note on the fix.** The direction is right: stop writing embedding_config.default_model_id, derive the Summary from the live owners, and offer auto-retrieve. The model half is heavier than the proposal admits. The active profile is a read-only built-in, so "write it through settings_rag_profile_adapter" really means clone hybrid_basic into a custom profile, set_active_profile, and accept a new fingerprinted collection. Those profile JSON writes also sit outside config.toml, WIZARD_OWNED_SECTIONS and the commit snapshot/rollback path. A more proportionate fix for first run: drop the embedding picker entirely.

<sub>Evidence: tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py:5355-5366 (RagStep.commit → build_rag_commit); tldw_chatbook/UI/Wizards/first_run_setup_state.py:1474 (writes only {'embedding_config': {'default_model_id': ...}}); tldw_chatbook/UI/Wizards/first_run_setup_state.py:1886-1903 (Summary RAG row derived from default_model_id); tldw_chatbook/RAG_Search/simplified/active_config.py:298-323 (runtime model = active profile rag_config.embedding.model)</sub>

#### [fulltrack-02] Tools step says 'Everything is off by default' but hides 4 gates, 3 of them on (local master switch, ask_user, character tools)
**Area:** RAG / Tools / Notes / Style · **Who:** first-time, power-user · **Verification:** confirmed-with-correction · **Effort:** M · **Backlog:** —

**What happens.** The subtitle reads "Everything is off by default. Tools that read or change your files still show an approval card every time they run." The step renders only the 8 _GATEABLE_BUILTINS rows. all_tool_gates(), which drives the MCP Tool gates pane, has 12 gates. Of the four hidden ones, three default to ON: [console] local_tools_enabled (web_search / web_fetch / web_crawl, Watchlists tools, and workspace-scoped fs_* / git_* tools), [tools] ask_user_enabled, and [tools] character_tools_enabled. The fourth, web_deep_search_enabled, defaults to off. The Summary repeats the claim ("– Tools — all off…"), and so does the Quick-track note ("Left at recommended defaults: tools off…"). The Summary counts only truthy [tools] keys ("✓ Tools — 5 enabled"), so the default-ON groups are never counted.

**Verified detail.** Understated in scope. The hidden default-on gates are real: the probe lists console.local_tools_enabled, ask_user_enabled and character_tools_enabled as True. But the agent also gets tools outside all_tool_gates() entirely. Live catalog with all 8 wizard switches off: ask_*(1), calculator(1), canvas_*(5), character-creator(1), character_*(3), get_*(1), mcp_*(6), todo_*(4), watchlists_*(14), web_*(3). Directly listed are spawn_subagent, install_skill, run_skill_script, fork_chat, new_chat and others.

**Why it matters.** First-time: forms a false mental model ("the assistant can't touch the web or my characters"), then hits tool cards it was told were off. Trust in every other setup claim drops. Power-user: can't audit the real agent posture during setup. Leaving "Write file" off doesn't mean the agent has no write path, because workspace fs_write sits under the hidden master switch.

**Fix.** Render the step from all_tool_gates() in two labelled groups. "On by default — the assistant asks before each use" holds the local master switch, ask_user and character tools, using their existing titles and descriptions. "Off until you turn them on" holds the 8 built-ins plus web_deep_search. Derive the subtitle from state. First run: "Tools let the assistant act for you. 3 groups are on by default; anything that reads, changes or reaches out asks you first, unless you allow it for longer." Re-run: "5 of 12 tools on — change any switch below." Add the master key to WIZARD_OWNED_SECTIONS (or a narrow 'console.local_tools_enabled' entry) and keep the delta-aware commit.

**Designer's note on the fix.** Rendering all 12 gates in setup still would not make any 'off' claim true. Always-on tools (sub-agents, skills, canvas, todo, calculator, datetime) and the built-in MCP server's 6 tools sit outside all_tool_gates(). Putting the [console] master switch in first-run setup invites a reflexive 'turn everything off' that silently removes web search and Watchlists agent tools. It also needs 'console' added to WIZARD_OWNED_SECTIONS. A better and smaller fix: keep the 8 opt-in switches and replace the posture sentence with truthful copy derived from state.

<sub>Evidence: tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py:7001-7040 (renders gateable_builtin_tools(); hard-coded subtitle at 7015-7019); tldw_chatbook/Agents/builtin_tool_gate.py:799-800,883-1000 (all_tool_gates(): local master default True, ask_user/character default True); tldw_chatbook/Agents/local_tool_provider.py:132-138 (ASK_USER/CHARACTER_TOOLS_DEFAULT_ENABLED = True); tldw_chatbook/Agents/local_tool_provider.py:3891-4420 (fs_*, git_*, web_fetch, web_search, web_crawl, watchlists_* specs under the master s…</sub>

#### [gap-01] OpenRouter, Hugging Face, NVIDIA NIM and Novita accept any API key in setup because the 'check' is a public model listing *(newly found)*
**Area:** Gap checks · **Who:** first-time, power-user · **Verification:** confirmed-with-correction · **Backlog:** —

**What happens.** Provider says 'Key staged — it will be checked when you continue.' That check is model discovery. /models on OpenRouter, the HF router, NVIDIA and Novita answers 200 to any key, so an expired or made-up key gets Provider ✓, the full 466-model list, and Summary '✓ Provider — openrouter' with 'Start chatting'. In the live run, the expired OpenRouter key (/api/v1/key → 401 'API key expired.') completed setup with ✓, and the first message failed with 'provider returned HTTP 401 (… authentication failed. Status: 401.)'. This is distinct from #85 and #2: here the check runs and reports success.

**Verified detail.** Holds for OpenRouter and Hugging Face. Does not hold in the wizard for NVIDIA NIM and Novita, which fail differently. 1. OpenRouter (live, vn4-a, random fake key): before any key is typed, Provider already says 'Found 466 model(s) for OpenRouter.' After the fake key it says 'Key staged — it will be checked when you continue.' Model lists the ids with no auth warning. No 'Continue anyway?' appears. Summary shows '✓ Provider — openrouter' and '✓ Default model — openai/gpt-4.1-mini' with 'Start chatting'.

**Fix.** Probe authenticated endpoints: OpenRouter GET /api/v1/key and HF whoami-v2. For NVIDIA and Novita, send a 1-token completion behind the consent #85 proposes. Mark these presets listing_requires_auth=False in the registry and pin that with a test. Success copy: '✓ Key accepted by OpenRouter'. Failure copy: '✗ OpenRouter rejected this key: API key expired. Create a new one at openrouter.ai/keys.' Until a probe exists, change the copy to say the key is checked on the first chat, and have Summary show '– Provider — openrouter (key not verified)'.

**Designer's note on the fix.** The direction is right: a public listing must never stand in for a key check, and the copy must stop promising a check that can't fail. Changes: (1) Drop the new listing_requires_auth registry flag. The rule already exists as PUBLIC_MODEL_LISTING_PROVIDER_KEYS, and Settings and Console readiness already obey it. Move that set onto ProviderRecord as the single source, add huggingface and novita (both 200 on a bogus key), pin it with a test that probes nothing live, and make the wizard read it. One rule then gives one verdict on every surface.

<sub>Evidence: evidence/verify-gaps/5a-gap5d-openrouter-fake-key.txt; evidence/verify-gaps/5b-gap5d-openrouter-fake-key-model.txt; evidence/verify-gaps/62-gap5b-summary.txt; evidence/verify-gaps/63-gap5b-first-reply.txt</sub>

#### [gap-03] Gemini setup offers no model list and never checks the key, and the obvious model (also the shipped default) is retired, so the first chat 404s *(newly found)*
**Area:** Gap checks · **Who:** first-time, power-user · **Verification:** confirmed-with-correction · **Backlog:** —

**What happens.** After pasting a Gemini key, Provider says 'Key staged — it will be checked when you continue. Connection testing is unavailable for this provider.' The two sentences contradict each other, and nothing is checked. The Model step shows only 'Model listing unavailable; enter the model ID used by this endpoint.' and an empty field, although Google's models.list works with the same key (61 models). The id people type, gemini-2.5-flash, is also the template default (config.py:4612). Google refuses it for new users with 404 '…no longer available to new users. Please update your code to use models/gemini-3.8-flash'. Setup still shows ✓ and 'Start chatting'. The first message shows 'provider returned HTTP 404 (… provider unavailable. Status: 404.)'; Google's message is only in the log.

**Verified detail.** (1) The wizard never prefills or suggests gemini-2.5-flash for Google. The field is blank (only OpenAI is prefilled, model-04), and a blank Next throws away the provider and key: '✗ Provider — no credentials or saved endpoint', '– Default model — not selected' (model-01). The retired id reaches users through the app's own catalog instead: config.py:4444 Google = ["gemini-2.5-flash", …] (the first Google row in the Alt+M switcher, shown as 'Ready · not tested'), [api_settings.google] model (config.py:4613) and the code fallback (config.py:2677).

**Fix.** Implement Gemini discovery via v1beta/models with x-goog-api-key, filtered to generateContent models; it doubles as the key check ('✓ Google accepted this key — 41 chat models'). Replace the retired template default, and add a nightly smoke test that fails when a template default is missing from the live list. Drop 'Connection testing is unavailable' where discovery can validate. On first chat, show Google's reason with a [Switch model] action (see gaps-06).

**Designer's note on the fix.** Native discovery is the right core, and Google's listing does double as a key check: a bad key returns 400 INVALID_ARGUMENT 'API key not valid' and a good one returns 200 with 61 models (both verified). That makes 'it will be checked when you continue' true. Filter on supportedGenerationMethods containing generateContent, and also drop -tts, -image, embedding and preview ids. Do not sort 'newest flash first' by id: that would surface gemini-3.8-flash-tts or -lite-tts.

<sub>Evidence: evidence/verify-gaps/52-gap5a-gemini-key.txt; evidence/verify-gaps/53-gap5a-gemini-model.txt; evidence/verify-gaps/55-gap5a-summary.txt; evidence/verify-gaps/57-gap5a-first-reply.txt</sub>

#### [protect-summary-01] Setting a password on Protect bricks the next launch: the unlock prompt crashes (NoActiveWorker) and the app exits 1
**Area:** Protect / Summary · **Who:** first-time, power-user · **Verification:** confirmed-with-correction · **Effort:** S · **Backlog:** task-33621.25 (To Do)

**What happens.** After 'Set a password' and '✓ Encryption enabled.', every relaunch through the module entry point dies before any UI. The startup unlock app awaits `push_screen(PasswordDialog(mode='unlock', title='Unlock Configuration', ...), wait_for_dismiss=True)` directly in `on_mount`. Textual 8 raises `NoActiveWorker: push_screen must be run from a worker when wait_for_dismiss is True`, and the code then logs 'Cannot proceed without decryption password.' and calls sys.exit(1). The 'Unlock Configuration' prompt that the Protect copy promises ('You'll be asked for this password each time chatbook starts') never appears. The wrong-password branch would also recurse into `await self.on_mount()` and hit the same crash. The packaged `tldw-cli` command uses a different, pre-TUI prompt with its own problems; see protect-summary-02.

**Verified detail.** The core claim holds exactly as written for the module entry point. Live, on 2 of 2 relaunches: two 'Encryption is enabled but no password is set' warnings, a traceback ending 'NoActiveWorker: push_screen must be run from a worker when `wait_for_dismiss` is True', then 'Cannot proceed without decryption password.' and EXITED:1. The overstatements are 'the app is impossible to open' and 'makes the whole app unusable': the packaged `tldw-cli` prompts 'Configuration password: ' before the TUI and works.

**Why it matters.** First-time (Sam): the security step the wizard recommended makes the whole app unusable, including notes, chats and settings, and all they see is a Python traceback. Power user: has to reverse-engineer config.toml (the [encryption] table and the enc: values) to get back in. Violates Nielsen #9 (help users recognize, diagnose and recover from errors), #5 (error prevention) and #1 (visibility of system status).

**Fix.** Retire the module path's private PasswordPromptApp and call the same unlock implementation as the packaged command; protect-summary-02 proposes a single TUI unlock screen for both. Minimum immediate fix: push the dialog with a callback, `self.push_screen(PasswordDialog(mode='unlock', ...), self._on_unlock_result)`, or from an @work method. Verify the password inside the callback. On a wrong password keep the dialog open and call PasswordDialog.show_error('That password didn't match. Try again.') instead of notify plus a recursive on_mount. Delete census row 91.

**Designer's note on the fix.** Better than the 'minimum' callback fix: delete PasswordPromptApp and route `_run_module_main` through the same `Backup_Recovery.launcher.startup_preflight` that cli.py uses, or just call cli.main_cli_runner, so only one unlock implementation exists. The proposed minimal fix would keep a second unlock that checks only the verifier and then relies on the app's non-strict decrypt (config.py:1056-1064). If a password passes the verifier but cannot decrypt the keys (exactly the state protect-summary-04 produces), that path would start the app with 'enc:…' ciphertext as API keys.

<sub>Evidence: evidence/live-firsttime-cloud/46-relaunch-encrypted-CRASH-full-scrollback.txt; evidence/live-firsttime-cloud/47-relaunch-encrypted-CRASH-repro2.txt; evidence/live-firsttime-cloud/27-password-enter.txt; tldw_chatbook/app_entry.py:480-554 (PasswordPromptApp 492-536; awaited push 501-510; recursion 530-531; exit 549-550)</sub>

#### [protect-summary-02] Startup unlock has no recovery: a wrong or forgotten password leaves no way into the app
**Area:** Protect / Summary · **Who:** first-time, power-user · **Verification:** confirmed-with-correction · **Effort:** M · **Backlog:** —

**What happens.** The packaged path asks once, before the TUI, with getpass: 'Configuration password: ' (launcher.py:55). A wrong password, an empty entry, Ctrl+C or a missing private terminal all return 'configuration_unlock_failed'. cli.py then runs minimal_recovery, whose RecoveryApp composes 'Recovery required: configuration_unlock_failed' and immediately pushes BackupRestoreScreen over it (launcher.py:120-130). Nothing says the password was wrong, nothing offers another try, and nothing lets the user start without the encrypted keys. The setup dialog promises 'If you forget it, the encrypted keys cannot be recovered — you'll need to re-enter them.' In practice a forgotten password blocks the whole app, not just the keys, and the app offers no way to re-enter them.

**Verified detail.** 'One typo ... leaves no way into the app' is overstated. A wrong password, an empty entry and Ctrl+C all return 'configuration_unlock_failed' (probe, 3 of 3 failure kinds), and recovery mode opens. Escape then shows 'Recovery required: configuration_unlock_failed' with an Exit button, and a relaunch asks for the password again. One detail the issue missed: the first screen is the full Backup & Restore screen (Create backup, Inspect / restore, Recovery copies, ...), and the raw reason code appears only after pressing Escape.

**Why it matters.** First-time: one typo at launch lands them in Backup & Restore with no hint that the password was the problem, and a forgotten password looks like data loss. Power user: there is no scripted or non-TTY unlock; getpass refuses without a private terminal and that also routes to recovery mode. Violates Nielsen #9 (recover from errors), #1 (status), #2 (raw error code instead of plain language) and #4 (three names for one password).

**Fix.** Build one unlock screen and use it from both `tldw-cli` and `python -m tldw_chatbook.app`. It is a minimal Textual app, pushed with a callback (see 01), titled 'Unlock tldw chatbook', with the body 'Your saved API keys are encrypted. Enter the master password you set during setup.' On a wrong password the dialog stays open with the inline error 'That password didn't match. Try again.'; there is no attempt cap, and a 1-second delay starts after 5 failures. Add two secondary buttons. (1) 'Start without saved keys': this session only; the keys stay encrypted, providers that need a stored key show Not ready, and environment-variable keys still work.

**Designer's note on the fix.** The direction is right: retry in place, a plain-language error, a reset path and one name ('master password'). Four cautions. (1) startup_preflight is deliberately pre-TUI and independent of the normal startup composition (Backup_Recovery isolation), so the proportionate first step is a getpass retry loop ('That password didn't match. Try again.'). On final failure, print a plain sentence with '[R]eset saved keys / [Q]uit'. Also stop RecoveryApp from auto-pushing BackupRestoreScreen when the reason is configuration_unlock_failed.

<sub>Evidence: tldw_chatbook/cli.py:77-86; tldw_chatbook/Backup_Recovery/launcher.py:15-23 (_secret), 25-74 (startup_preflight: one-shot prompt at 55, failure at 57-58), 120-130 (rea…; tldw_chatbook/Widgets/password_dialog.py:146-164 (setup and unlock copy); tldw_chatbook/config.py:1050-1051 (with no password set, decryption only warns and startup continues)</sub>

#### [new-provider-01] After 'Enter skips', going Back shows the skipped provider's form; a key typed there is silently discarded and the step is skipped again *(newly found)*
**Area:** Provider · **Verification:** confirmed-with-correction · **Backlog:** —

**What happens.** Enter in the empty key field calls skip_without_key(), which clears selected_provider_key and _provider_choice_interacted but leaves OpenAI's Authentication form, key field and status on screen. When the user comes Back to add their key, the step looks exactly like 'OpenAI selected, paste key'. They paste a key and press Next, and commit() takes the skip path (_effective_provider_key returns '' because nothing is selected and the list was not touched). The Model step says 'Pick a provider first — or type a model name below' and the tracker keeps Provider '!'. The typed key is never staged. Typing in the key field doesn't re-select the provider (_on_key_changed only clears the strip). This is the natural 'skip now, come back with my key' path that the help line itself suggests.

**Verified detail.** The core claim reproduced exactly, live on a fresh profile at 120x40 (Quick track, OpenAI, Enter-skip, Ctrl+B, type key, Ctrl+N gives Step 3 'Pick a provider first — or type a model name below' with Provider '!'). One precision fix: _on_key_changed does more than clear the strip. It bumps the credential revision and shows 'Provider settings changed since test; test again.', but it never re-selects the provider. The key line is a visible tell the original missed.

**Fix.** Make the visible form and the selection agree. Either (a) on skip, also reset the step's UI to the 'nothing selected' state (hide Authentication, clear status, clear the list highlight), so Back shows an honest blank step; or, better, (b) keep the provider selected and record a 'skipped' flag, so typing a key (Input.Changed with non-empty value) clears the flag and Next stages it normally. (b) preserves the user's choice and matches 'you can add it later'. Add a regression test: skip → Back → type key → Next stages the key for that provider.

**Designer's note on the fix.** Option (b), keeping the choice, is the right direction and (a) is wrong: resetting the step to blank on skip throws away the user's provider choice and turns Back into a dead end. Implement (b) as a one-shot skip, not a sticky 'skipped' flag. A sticky flag has to be cleared on typing, re-pick, Test, env-var detection and resume, and every missed event becomes this bug again. Instead, have the Enter-skip (and the explicit Skip control that provider-01 calls for) set a flag that commit() consumes for that single advance.

<sub>Evidence: evidence/verify-provider/49-v8-back-after-enter-skip.txt (Back after skip: OpenAI form shown); evidence/verify-provider/50-v8-key-typed-after-skip.txt (key typed into the field); evidence/verify-provider/51-v8-next-after-key-typed.txt (Next → Step 3 'Pick a provider first', Provider '!'); FRW:2952-2968 skip_without_key; FRW:2969-2984 _effective_provider_key; FRW:3037-3059 _on_key_changed (no re-selection)</sub>

#### [provider-01] Picking a cloud provider with no key traps Next and mouse users; only Enter-in-field skips
**Area:** Provider · **Who:** first-time, power-user · **Verification:** confirmed · **Effort:** M · **Backlog:** —

**What happens.** After choosing a keyed provider with an empty key, Next/Ctrl+N shows the pinned red strip 'API key required. Set OPENAI_API_KEY or add api_key under [api_settings.openai]. Retry with Next, or go Back.' Retrying yields the same refusal every time; 'go Back' goes to Welcome. There is no Skip control and the list cannot be unselected. The only way past is Enter inside the empty password field (skip_without_key), mentioned only at the end of the small help line: 'No key yet? Enter skips this step — you can add a provider later in Settings.' Welcome promised 'most steps can be skipped with Next', and the footer says 'Enter / Ctrl+N next', implying Enter and Ctrl+N are the same action. In the mouse-only run, two clicks on Next changed nothing.

**Why it matters.** First-time (no key yet, or exploring): stuck on step 2 and told to edit environment variables or a TOML table; mouse users must abandon the provider choice or exit setup. Power user: learns an undocumented exception to 'Next skips'. Violates H3 User control and freedom (no visible skip/emergency exit), H9 Help users recognize, diagnose and recover from errors (the suggested remedy 'Retry with Next' can never succeed), H4 Consistency and standards (Next skips on every other step; Enter and Ctrl+…

**Fix.** UI: add a secondary button 'Skip — connect later' on the key row (beside the 'Check key' button from provider-06), shown whenever a keyed provider is selected and not ready; it calls skip_without_key() then action_next(). Behaviour: make Next on a keyed provider with an empty key a two-press skip — the first press shows an in-step note (not the red strip): 'No key yet. Press Next again to skip for now — you can connect a provider later from Console or Settings ▸ Providers & Models.'; a second Next (or the button) skips; any edit to the key cancels the pending skip. Let commit() return whether a retry can succeed so _advance only appends 'Retry with Next' when it can (e.g.

**Designer's note on the fix.** Direction is right: a visible skip control plus refusal copy that names something that can actually work. Drop the two-press Next. It is hidden modal state, and an impatient double-press would skip without the user meaning to. Simpler and consistent with Welcome's promise: make Next with an EMPTY key field behave exactly like Enter-in-field already does (skip, tracker '!', Summary 'not connected'), since nothing is staged and nothing is lost. Keep the explicit 'Skip — connect later' button for mouse users. The commit()-returns-retryable idea is good.

<sub>Evidence: evidence/live-resilience/11-80x24-provider-empty-key-next.txt; evidence/live-resilience/13-80x24-provider-empty-enter.txt; evidence/live-resilience/123-120x40-mouse-next-empty-key-blocked.txt; evidence/static-code-map/02-readiness-copy-and-template-defaults.txt</sub>

#### [provider-02] Detected local server gets lost: Enter picks OpenAI, a manual pick ignores the found port *(stub-server evidence)*
**Area:** Provider · **Who:** first-time, power-user · **Verification:** confirmed-with-correction · **Effort:** M · **Backlog:** task-33008.3 (To Do — Console-side detected-server switcher, different surface)

**What happens.** Auto-detect shows 'Found a local endpoint: http://127.0.0.1:9099.' with a 'Use this server' button, but focus stays in the provider list with OpenAI highlighted. Pressing Enter selects OpenAI: the banner and button vanish and are replaced by 'Couldn't discover models for OpenAI. Set OPENAI_API_KEY or add api_key under [api_settings.openai]. Or go Back.' and an expanded API key field. Clicking llama.cpp afterwards fills Endpoint 'http://localhost:8080' (Chat URL …:8080) with 'Couldn't discover models for llama.cpp. You can continue anyway.' — the server found seconds earlier is forgotten until the user presses 'Find local servers' and picks the detected row. 'Use this server' is reachable only by Shift+Tab or mouse; the code comment says it is 'only reachable backwards or by click'.

**Verified detail.** Enter selecting the highlighted OpenAI row is normal OptionList behaviour, and the User Guide documents it, so on its own it is weak evidence. The stronger defect is that ANY provider pick drops the detection. Clicking llama.cpp directly while the banner says 'Found a local endpoint: http://127.0.0.1:9099.' (no Enter misstep) clears the banner, fills Endpoint 'http://localhost:8080' and shows 'Couldn't discover models for llama.cpp. You can continue anyway.' Tab from the list goes to '← Back' (verified by ANSI focus style), so 'Use this server' is only reachable with Shift+Tab or a click.

**Why it matters.** First-time local user (Jo): loses the found server with one keystroke and is pushed toward a cloud key; picking the right engine still fails. Keyboard power user: the primary local CTA is outside the Tab order. Violates H1 Visibility of system status (detection result silently discarded), H5 Error prevention (default focus makes the wrong choice one keystroke away), H6 Recognition rather than recall (must remember 'Find local servers' / Shift+Tab), H7 Flexibility (no accelerator for the primary…

**Fix.** Let detection drive the default. When discovery finds a server and nothing is selected: move the provider-list highlight to the detected engine's row and return the 'Use this server' button from preferred_focus(), so Enter means 'use it'; relabel the button with the engine ('Use llama.cpp on this computer'). Keep detection sticky: _clear_detected_provider_state() should only hide the banner while a non-matching provider is selected, and restore banner and button when the user returns to a matching provider or to no selection; never drop _detected_servers on a switch.

**Designer's note on the fix.** Sticky detection, and making _initial_endpoint_for prefer a detected server for that engine, are the core, low-risk fixes. Do them first. Moving focus to the 'Use this server' button above the list fights the established Tab order (TASK-1496). A cleaner pattern: render the detection as the first row of the provider list ('llama.cpp on this computer · 127.0.0.1:9099 · 3 models'), pre-highlighted, so Enter means 'use it' and the arrows still browse. That also removes the separate button and the duplicate-affordance clutter.

<sub>Evidence: evidence/live-firsttime-local/27-local2-provider-detected.txt; evidence/live-firsttime-local/28-local2-enter-on-list.txt; evidence/live-firsttime-local/30-local2-llamacpp-clicked.txt; evidence/live-firsttime-local/31-local2-find-local-found.txt</sub>

#### [provider-04] After resuming interrupted setup, the API-key field is hidden and re-picking does nothing
**Area:** Provider · **Who:** first-time, power-user · **Verification:** confirmed · **Effort:** S · **Backlog:** —

**What happens.** Resume lands on Provider with OpenAI restored, but the Authentication section is hidden and not in the Tab order. The only text is 'Couldn't discover models for OpenAI. Set OPENAI_API_KEY or add api_key under [api_settings.openai]. Or go Back.' Clicking OpenAI again does nothing. Getting the field back took arrowing to Anthropic and back, then expanding a now-collapsed '▶ Authentication'. The recovery dialog had said 'Credentials are not retained in setup recovery and may need to be re-entered.'

**Why it matters.** First-time: after a crash there is no visible key field and the guidance is env-var/TOML only — likely abandonment. Power user: needs the switch-away-and-back trick. Violates H1 (the required input is not shown), H9 (recovery instructions don't match the UI), H3 (re-selecting the item is a dead control), H6 Recognition rather than recall.

**Fix.** In _restore_resume_controls call provider_step.select_provider(provider_key) inside the step guard (and set _provider_choice_interacted = True) instead of assigning the attribute. When the restored provider needs a key and none is stored or exported, expand Authentication, make the key Input the step's preferred focus, and set the key line to 'Re-enter your OpenAI API key — keys aren't kept when setup is interrupted.' Make _select_provider_option re-apply select_provider() when the selected provider's Authentication panel is hidden, as a self-heal for any other path that sets the attribute directly. Have the recovery dialog name the step ('Resume at Provider — step 2 of 6').

**Designer's note on the fix.** Right fix, correctly sized (S). Run select_provider() inside step_guard.provider_switch so a failing read during resume is contained. Also make the auth section's collapsed state depend on requires_api_key whenever the key input is visible and empty: the switch-back path collapsing a required key section is a second, independent bug. The self-heal in _select_provider_option is a good defensive addition.

<sub>Evidence: evidence/live-resilience/93-120x40-resumed-after-crash.txt; evidence/live-resilience/94-120x40-resumed-provider-tabbed.txt; evidence/live-resilience/95-120x40-resumed-reselect-openai.txt; evidence/live-resilience/96-120x40-resumed-reselect-via-arrows.txt</sub>

#### [provider-05] Azure OpenAI, Databricks and Cloudflare dead-end: no endpoint field, Next refused with 'API key required' after a key is pasted
**Area:** Provider · **Who:** first-time, power-user · **Verification:** confirmed · **Effort:** M · **Backlog:** —

**What happens.** These providers need a per-account URL (Azure resource host, Databricks workspace host, Cloudflare account URL). The template ships no api_base_url for them, so the wizard hides the Endpoint panel. With a key pasted, readiness stays not ready ('Missing resource URL'), so the key line keeps saying 'An API key is needed — paste it above. (Already exported AZURE_OPENAI_API_KEY? It's picked up automatically.) No key yet? Enter skips this step…', and Next refuses with 'API key required. Set api_base_url to your … resource host under [api_settings.azure] (for example https://my-resource.openai.azure.com); the /openai/v1 path is appended automatically. Retry with Next, or go Back.' There is no field for that URL, and the Enter skip only works after deleting the key. The User Guide says 'Every row can be picked with the arrow keys and set up here'.

**Verified detail.** none on substance. The live copy uses the raw key, not the display name: 'API key required. Set api_base_url to your azure resource host under [api_settings.azure] (for example https://my-resource.openai.azure.com); the /openai/v1 path is appended automatically. Retry with Next, or go Back.' The key help line still says 'An API key is needed — paste it above.' with a key in the field, and 'Provider settings changed since test; test again.' appears with no test control.

**Why it matters.** First-time Azure/Databricks/Cloudflare user: told the key is missing after pasting it; can only finish by editing config.toml. Power user: forced out of the wizard. Violates H9 (wrong diagnosis), H2 Match between system and the real world (TOML instructions), H5, H3.

**Fix.** Show the Endpoint panel for every provider in PROVIDERS_REQUIRING_BASE_URL_KEYS, labelled from _BASE_URL_RECOVERY: 'Azure resource URL' (placeholder 'https://my-resource.openai.azure.com'), 'Databricks workspace URL', 'Cloudflare account URL (…/accounts/<account-id>/ai/v1)'; hide 'Find local servers' for them. Map readiness reasons to wizard copy instead of prefixing 'API key required.': 'Missing resource URL' → field-level 'Enter your Azure resource URL above — the /openai/v1 path is added for you.' and refusal 'Add your Azure resource URL to continue.' Root fix: carry requires_base_url, label and example in the provider catalog record that both readiness and the wizard read (improvement '…

**Designer's note on the fix.** Showing the Endpoint field, labelled per provider for PROVIDERS_REQUIRING_BASE_URL_KEYS, is the right minimal fix, and mapping readiness reasons to copy (instead of a blanket 'API key required.') is overdue. Put the resource-URL field ABOVE the key for these three (it is the missing piece), and fix the raw-key 'azure' in the readiness copy. The 'provider facts registry' root fix is right but large. Ship the three-provider fix and its tests first, and do not block on the registry.

<sub>Evidence: consolidated/provider-evidence/01-keyed-provider-test-affordance-and-account-endpoint-readiness.txt (readiness ready=False, 'Missing resour…; tldw_chatbook/Chat/provider_readiness.py:165-187 (PROVIDERS_REQUIRING_BASE_URL_KEYS, _BASE_URL_RECOVERY); tldw_chatbook/Chat/provider_readiness.py:863-887 (key present but no base URL → not ready); tldw_chatbook/config.py:4701-4712, 4960-4968, 4981-4990 (no api_base_url shipped for databricks/azure/cloudflare)</sub>

#### [voice-speech-01] Next on an untouched Voice step writes TTS config and overwrites a working setup on re-run
**Area:** Voice / Speech · **Who:** first-time, power-user · **Verification:** confirmed-with-correction · **Effort:** M · **Backlog:** —

**What happens.** The subtitle reads "Hear replies read aloud — optional. PocketTTS or OmniVoice run locally, no account needed; skip with Next if you don't want voice." Next always saves instead. In every live profile that passed Voice untouched (local1-6, res1/2/3/5/6, cloud2, power2), Next wrote [app_tts] OPENAI_BASE_URL = "http://127.0.0.1:8765/v1/audio/speech", OPENAI_AUTH_MODE = "none", default_provider = "openai", default_model = "tts-1-hd", default_voice = "shimmer", default_format = "mp3", and duplicated these into [tts_settings]. Only profiles that never reached Voice (local2b, res4, probe) have no [app_tts]. With 'Use as default' unticked, the PocketTTS preset's own model, voice and format (pocket-tts / alba / wav) are dropped, and runtime fallbacks are written as the default. The result points the OpenAI TTS backend at the PocketTTS port with OpenAI voice names.

**Verified detail.** The first-run harm is described wrongly in both directions. (a) Overstated for keyless users. With no [app_tts], the effective TTS default was already openai/tts-1-hd/shimmer against https://api.openai.com/v1/audio/speech with API-key auth (TTS/backends/openai.py:31,67-71; openai_compatible_config.py:200-212). A keyless user could not hear speech before the step either, so nothing that worked is lost; what they gain is a false '✓ Voice — PocketTTS (default voice)' and a Console 'Ready · not tested'. (b) Understated for users who do have an OpenAI key, e.g.

**Why it matters.** First-time: a user who skipped voice now has a default TTS provider that cannot speak. The first 'Speak replies' fails, with nothing linking the failure to the wizard. Power user: re-running setup to change one thing silently breaks a working OpenAI TTS setup, with no undo, and the prior endpoint is lost.

**Fix.** One root-cause fix in three parts. It also clears voice-speech-02's false check mark, static-code-map-38's resume-as-Custom and the User Guide's skipped-voice drift. (1) Skip option: make the first Service radio 'No voice for now' (id setup-voice-preset-none), preselected on a first run when the raw [app_tts] table is absent. Choosing it hides sample, test, default and Advanced and shows 'Nothing is saved. Set up a voice any time in Settings > Speech & TTS.' commit() then returns (True, '') and posts nothing. (2) Re-run prefill: add a pure voice_draft_from_config(raw_app_tts) to first_run_voice_step_state.py.

**Designer's note on the fix.** Sound, and it fixes the root cause. Three notes. (1) Ship order: the smallest safe slice is the delta/untouched gate, i.e. commit() posts nothing when the draft equals what is saved (or equals the initial draft on a profile with no raw [app_tts]) and the user neither tested nor ticked default. That slice alone stops the clobbering. 'No voice for now' and the re-run prefill are the right follow-ups for clarity. (2) The prefill must read the RAW [app_tts] from COMPREHENSIVE_CONFIG_RAW or load_cli_config_and_ensure_existence.

<sub>Evidence: evidence/live-firsttime-local/17-local1-voice-step.txt; evidence/live-power-user/83-rerun3-voice.txt; evidence/live-power-user/configs/power1-run2-07-before-rerun3.toml vs power1-run2-08-after-rerun3-cancel.toml; evidence/live-power-user/configs/power1-02-rerun2-at-model-step.toml vs power1-03-rerun2-after-voice-next.toml</sub>

### P2 — minor (71)

**Accessibility & terminal**

| ID | Issue | Fix | Backlog |
|---|---|---|---|
| a11y-02 | Welcome opens scrolled past its own heading at 80x24 and 100x30 | In show_step(), call `target.focus(scroll_visible=False)`, then `current_step.scroll_home(animate=False)`. Only if the target is still entirely outside the visible region, call `scroll_to_widget(target, top=False)` for… | — |
| a11y-03 | Below 100x30 the size warning covers the key-hint line, and is itself cut off | Compose one bottom status row and fill it in _sync_size_hint(). At small sizes put the keys first, then a short size note: 'Enter/Ctrl+N next · Ctrl+B back · Esc exit · ⚠ 80×24: 100×30+ recommended'. | — |
| a11y-07 | On/off state shown by colour or knob position alone: tool switches, stock checkboxes that draw 'X' both ways, Show password | Promote one `StateCheckbox` (✓ checked, blank unchecked) into tldw_chatbook/Widgets/ and replace the three local copies with it. Use it for 'Get to know you after setup' and 'Show password'. | task-32465 (To Do) |
| a11y-08 | Hidden content is signalled only by a 1.55:1 scrollbar thumb | 1. Apply the app scrollbar tokens to the wizard's scroll containers: 'FirstRunSetupWizard .setup-step, FirstRunSetupWizard .setup-choice-list { scrollbar-color: $primary; scrollbar-color-hover: $primary-lighten-1; scrol… | task-33626 (To Do) |
| a11y-09 | Contrast bundle: primary labels 3.78:1, unselected '○' 1.42:1, list headers 3.05:1 | Do a wizard-scoped token pass in _wizards.tcss: 1. 'FirstRunSetupWizard Button.-primary { background: $primary-darken-2; color: $text; }'. White on #0053aa measures 7.45:1. | task-33626 (To Do) |
| a11y-10 | Secondary buttons render as bare bold text with no visible shape | Define three wizard button tiers: - **Primary:** filled, as today. - **Secondary:** - In the nav bar, which already reserves 4 rows for a 1-row button, use 'SetupWizardNavigation Button:not(.-primary) { border: round $p… | — |
| a11y-11 | Keyboard focus on buttons is a faint cue (invisible on primary), so Shift+Tab+Enter fires the wrong action (e.g. the Speech file picker) | Give the wizard a focus treatment that changes shape: - 1-row buttons: 'FirstRunSetupWizard Button:focus { text-style: bold reverse; }'. Reverse video survives NO_COLOR and monochrome. | task-33626 (To Do) |

**Coverage & power use**

| ID | Issue | Fix | Backlog |
|---|---|---|---|
| coverage-06 | Setup never offers a tldw server connection; the only path is buried in Settings | Welcome: add “How will you use chatbook?” with “● On this computer (local or cloud models)” / “○ Also connect to a tldw server”. | — |
| coverage-08 | Setup handles one provider per run: env keys aren't marked in the list, and a key typed for another provider is silently dropped | Add a “Ready on this machine” group pinned above Popular, built from env presence, stored keys and local discovery, with text status on each row: “OpenAI · key in OPENAI_API_KEY”, “Ollama · running on :11434 · 3 models”… | — |
| coverage-16 | The '2-minute' Quick track carries an optional Voice step that rarely works first try and a Protect step that is often empty | Quick = Welcome, Provider, Model, Summary. Move Voice to Full, and add a Summary next-step button “Hear replies aloud…” that opens the Voice step on its own (or Settings ▸ Speech & TTS). | — |
| coverage-17 | Document analysis stays on OpenAI/gpt-4o whatever provider setup connects, and no Settings screen edits [analysis_defaults] | Product rule: secondary defaults inherit chat_defaults unless the user set them. Implementation: when the wizard commits provider/model, also write analysis_defaults.provider/model if they still equal the template value… | task-28018 (To Do) |
| coverage-19 | Provider checks behind a TLS-inspecting proxy fail with generic copy and no pointer to certificates or Settings ▸ Network | Classify connection errors whose cause is certificate verification into a “tls_untrusted” category. Copy: “Your network is intercepting secure connections (certificate not trusted). | — |
| new-coverage-power-03 | Console 'Set up provider' ignores detected env keys and preselects the template OpenAI *(new)* | When chat_defaults.provider is unready and an env key resolves for another provider, preselect that provider in Settings ▸ Providers & Models, or offer 'Use Anthropic — key found in ANTHROPIC_API_KEY' on the card. | — |

**Cross-cutting**

| ID | Issue | Fix | Backlog |
|---|---|---|---|
| cross-cutting-02 | Tracker and Summary tick ✓ for skipped, failed or discarded steps (including '✓ Voice — PocketTTS'), and the two disagree | Record an explicit outcome for each step when the user leaves it: saved, kept current, skipped, staged (not yet saved), or failed/unverified. | — |
| cross-cutting-04 | Enter is advertised as 'next' but selects, tests, skips the provider or saves instead, depending on the focused widget | Define one Enter contract and make the hint describe the focused control. - Provider list: Enter selects the row, then advances if the provider is ready. | — |
| cross-cutting-06 | Browsing a list selects every highlighted row, fires network discovery, leaves errors behind and wipes typed values | Use the right pattern for each list size. Short radio groups (6 options or fewer: track, voice service, language, precision) keep selection-follows-highlight. | — |
| cross-cutting-07 | Provider errors appear before the user acts, lead with env/TOML jargon, offer an impossible 'Retry with Next', and stay stale and contradic… | Give each step one StepStatus owner holding {severity, message, fix_action}, rendered in one region directly under the control it concerns. | — |
| cross-cutting-08 | 'Continue setup any time from Settings ▸ Diagnostics' is false: re-runs restart at Welcome | Make the promise true. When a valid draft exists, every re-run entry (Settings button, palette, Console link) first shows a small chooser: 'Pick up where you left off? Full setup — step 7 of 11 (Tools). | — |
| cross-cutting-09 | No record of what setup wrote; cancel keeps every write; some writes skip the audited path | Add a setup-session change ledger. Snapshot the config when the wizard mounts; every writer (commit_config and the four out-of-band paths) appends {section.key: old → new}. In the UI: 1. | — |
| cross-cutting-13 | Optional steps that fail to render block as 'required'; manual setup opens wrong pages | Mark Voice, RAG, Speech, Tools, Notes, Appearance and Protect required=False. A compose failure then auto-skips with 'This optional step couldn't be shown and was skipped.' and adds a Summary row '✗ {title} — couldn't b… | — |
| cross-cutting-15 | Back is bound to Ctrl+B, the default tmux prefix, and the always-on hint teaches it | Teach a multiplexer-safe Back key. Add alt+left (the browser and OS 'back' convention; the app already teaches Alt chords such as Alt+M) as the documented Back binding, and keep ctrl+b as an untaught alias. | — |
| cross-cutting-16 | User Guide contradicts the shipped wizard in at least ten places | Fix the factual errors now, because they are wrong today whatever the flow fixes do: the Full list, the table order, the Notes row, the tool-gates path, the Voice-skip line, the exits (including Start chatting), the re-… | — |
| cross-cutting-17 | Structural debt keeps producing focus-loss, soft-lock and crash defects in setup | 1. Move each step class into UI/Wizards/steps/<step>.py behind the existing SetupStep seam, bringing FirstRunSetupWizard.py under its ratchet before the UX fixes land. 2. | — |
| new-cross-cutting-02 | Wizard's provider probe always claims the chosen model is missing ('model="unconfirmed"' is hard-coded) *(new)* | Set the model facet from the probe's listed ids: 'confirmed' when the staged or prefilled model is listed, 'missing' when none is chosen. | — |

**Entry, exits & handoff**

| ID | Issue | Fix | Backlog |
|---|---|---|---|
| entry-exit-handoff-06 | First message after finishing via 'Explore Home' fails with 'Trace capture blocked' *(stub)* | 1) Make the first send robust. If the reservation finds the active session's revision row missing, create or backfill it in the same transaction instead of failing, or await the session's revision write before enabling… | task-33621.6 (To Do) |
| entry-exit-handoff-07 | Unfinished setup nags every launch: 'Later' saves nothing, other states loop or go silent | Add one persisted snooze, [first_run] resume_deferred = true plus resume_reminders_shown = N. 1) 'Later' (and Esc) writes resume_deferred. Later launches show no modal. | — |
| entry-exit-handoff-08 | Recovery and exit dialogs are generic: no step named, and a crash reads like an exit | Pass the validated SetupDraft into the dialog, and add a draft field exit_kind ('exit' when written by _finish_later, absent when interrupted). | — |
| entry-exit-handoff-13 | Skipping setup pops 'Check model lists online?' claiming configured providers and keys | 1) Skip records the Summary's default answer in the same commit as setup_completed: [model_catalog] refresh_consent_recorded = true, auto_refresh_enabled = false. | task-28019 (To Do) |
| entry-exit-handoff-16 | 'Restore a backup' is the only import: it opens create-first, never names the archive format, and fails opaquely on a config.toml | 1) Add initial_mode='inspect' to BackupRestoreScreen and a setup-specific action that opens directly on Inspect / restore, titled 'Restore from a backup', with the line 'Choose a .tldw-backup.zip (or .tldw-backup.zip.ag… | task-32562 (In Progress) |
| entry-exit-handoff-17 | First launch: 10–16 s of random off-brand splash, no skip hint, raw warnings first | 1) First run (no [first_run] flags): skip the splash, or show a fixed neutral wordmark for at most 1.5 s ('tldw chatbook — getting things ready…') with 'Press any key to skip' in its footer. | task-24306 (To Do) |
| entry-exit-handoff-21 | Console's 'Get started' card doesn't offer a local server started after setup *(stub)* | When the card mounts, and again every 30 s of idle while it is visible, run local_server_discovery off-thread on 8080, 9099, 11434 and LM Studio's 1234. | task-33008.3 (To Do), task-33008 (To Do) |
| entry-exit-handoff-23 | Model switcher lists never-configured placeholder models as 'Ready · not tested' | Treat endpoint and model values that equal the shipped template defaults as 'Not set up' until the user saves that provider or a probe succeeds. | task-33005.7 (To Do) |
| entry-exit-handoff-25 | LM Studio users (Custom OpenAI-compatible) get no streaming after setup *(stub)* | Set streaming = true for local OpenAI-compatible template sections (custom, custom_2, lmstudio-style), keeping the non-streaming fallback. When the wizard commits a local provider, write streaming = true for it. | task-33620.10 (To Do) |
| entry-exit-handoff-27 | Console sends any 'setup-blocked' reason to the full wizard at Welcome, even if configured | Only 'no provider configured' links to setup, and it opens the wizard at Provider (start_step via the -10 entry point) with a return to the Console chat on finish. | task-33620.2 (To Do), task-33620.3 (To Do), task-33620 (To Do), task-33008 (To Do) |
| new-entry-exit-handoff-01 | First plain chat ships a ~20.8 KB agent system prompt (16 agent tools) to local models while Summary says 'Tools — all off' *(new)* *(stub)* | Give plain Console chat a minimal prompt profile until the user enables an agent or tool surface, or at least budget the agent preamble against the model's known or unknown window, and show it in the context meter. | — |

**RAG / Tools / Notes / Style**

| ID | Issue | Fix | Backlog |
|---|---|---|---|
| new-full-track-steps-01 | After an app restart, re-running setup shows every Tools gate OFF (and drops the saved splash card) because step prefill reads load_setting… *(new)* | Read wizard prefill and delta baselines from the same source the Summary uses: load the full TOML once when the wizard opens (load_cli_config_and_ensure_existence(force_reload=True)), and refresh it after each commit_co… | — |
| new-full-track-steps-02 | Leaving 'Create note' off in setup does not remove note creation: the built-in tldw_chatbook MCP server exposes create_note (and character… *(new)* | Decide whether the Console agent should see the app's own MCP server at all, given that native equivalents exist in-process. | — |

**Gap checks**

| ID | Issue | Fix | Backlog |
|---|---|---|---|
| gap-04 | A pasted key with an invisible character either crashes the Provider step, or (with smart quotes) is reported as 'couldn't reach the server' *(new)* | Normalise pasted keys at the input: strip Cf/Zs/control characters, one pair of wrapping quotes, and a 'Bearer ' or 'NAME=' prefix. | — |
| gap-05 | Ctrl+Q quits setup instantly and skips the exit guard that Esc shows, while Ctrl+C's toast points users to it *(new)* | Add confirm_quit to FirstRunSetupWizard. It should reuse the exit dialog via await_quit_prompt ('Quit chatbook? Your OpenAI key hasn't been saved yet. | — |
| gap-06 | First-reply failures after setup drop the provider's own reason and label stalls 'unexpected provider error' *(new)* | Pass through one allowlisted field, error.message from a JSON error body: capped at 200 characters, secret-scrubbed, and prefixed with the provider name. Give StreamStallError its own category and copy. | — |
| gap-07 | A slow first local reply shows 90 s of unchanging 'Generating…', then fails with no elapsed time, cold-load allowance or guidance *(new)* *(stub)* | Give the first token a longer window (about 300 s for local providers, configurable) and keep 90 s for gaps between tokens. | — |

**Model**

| ID | Issue | Fix | Backlog |
|---|---|---|---|
| model-04 | Template model pre-typed on fresh installs fights '(recommended)' and is wiped by browsing | (1) Gate the prefill on first_run.setup_completed or a wizard-written provider, the same test provider_summary_configured uses, so a fresh install has no pre-typed model and the list's recommendation is the only default… | — |
| model-05 | Model picker shows 20 of up to 466 ids in a 5-row box, with no count, search or metadata | Turn the step into a filterable picker. (1) Add a filter Input above the list ('Filter 137 models…') that matches over all of _discovered_model_ids; non-chat ids stay hidden unless 'Show all models' is toggled. | — |
| model-06 | Typed model ids are never checked against the provider's list; typos save silently | (1) Attach Textual's SuggestFromList(self._discovered_model_ids, case_sensitive=False) to #setup-model-custom, so inline completion appears and Right accepts it. | — |
| model-07 | Model-step errors show as greyed radio rows, cut off; cloud users told to start a server | Implement task-33008.5 and widen it slightly. (1) Add Static#setup-model-status above the list. It wraps, uses the warning or error token colour plus a glyph, and is never focusable. | task-33008.5 (To Do, medium) |

**Protect / Summary**

| ID | Issue | Fix | Backlog |
|---|---|---|---|
| protect-summary-03 | 'Enable later in Settings' is false: no reachable UI can enable or disable encryption or change the password | Add an 'Encryption' card to Settings ▸ Privacy & Security. Show a state line: 'On — you'll enter your master password when chatbook starts' or 'Off — API keys are stored as plain text in config.toml'. | — |
| protect-summary-05 | Password dialog opens focused on its scroll box: typed password lost, stray inner frame | In PasswordDialog set AUTO_FOCUS = '#password-input' (or focus it in on_mount), and build the wrapper as VerticalScroll(can_focus=False); focus-follow still scrolls fields into view. | — |
| protect-summary-06 | Protect step misreads keyless, env-key and unsaved-key users, then ticks it ✓ | Keep the step slot so the count stays stable, but compute its state from persisted config only, at on_show: (a) no stored key and no env key: hide the pitch and show 'Nothing to protect — no API key is saved in config.t… | — |
| protect-summary-08 | Esc or 'Review settings' on Summary leaves setup unfinished; next launch nags to resume | Send every Summary exit through one finish path that writes setup_completed and the model-list consent. Esc on the Summary opens 'Finish setup? Your choices are saved. | — |
| protect-summary-11 | 'Get to know you after setup' looks checked when off, is unexplained, dropped on Library | Swap in SetupCheckbox. Label it 'After setup, answer a few quick questions so replies fit you' and add a dim description line: 'A short questionnaire on this computer — no AI provider needed. | — |
| protect-summary-14 | Re-run Summary offers only first-run exits; no 'Done' back to where you started | When rerun=True, make the primary 'Done'. It dismisses with completed=True and a return route that keeps the caller screen (the cancel_to_console=False semantics). | — |

**Provider**

| ID | Issue | Fix | Backlog |
|---|---|---|---|
| new-provider-02 | KoboldCpp row can't be set up: its shipped endpoint is rejected as 'ambiguous API suffixes', and any accepted URL is saved in a form the na… *(new)* | Decide KoboldCpp's contract in one place. Either make the row OpenAI-compatible (KoboldCpp serves /v1): template api_url = http://localhost:5001, the chat handler routed through the OpenAI-compatible path, and an engine… | — |
| provider-03 | Provider step ignores its own failed checks: refused or rejected setups advance with ✓ | Put the trust gate where the fault is. In ProviderStep.commit: (1) if the latest evidence or discovery for this exact draft identity is unauthorized/forbidden, refuse with a field-level error under the key: '✗ Anthropic… | task-33005.9 (To Do — wizard publishes settled test results to the shared evidence owner), task-33008.5 (To Do — Model-step status as text, not radio) |
| provider-06 | No visible way to check a cloud key; 32 presets show a Test button that never enables | One 'Check key' button per keyed provider, beside the key input (revive .setup-key-row: Horizontal(Input, Button 'Check key')). | task-33005.9 (To Do — shared connection evidence) |
| provider-11 | 60-provider list is unsearchable in a 5-row box with no descriptions or ready badges | Add a filter Input above the list ('Filter providers — name or ID', same matching as #settings-provider-search) and redirect printable keys typed in the list into it; show a count line ('60 providers · type to filter'). | — |
| provider-13 | LM Studio and other common local servers are never detected or named *(stub)* | Add localhost candidates 1234 (LM Studio), 1337 (Jan) and 5001 (KoboldCpp) with the existing short timeout and /v1/models shape check, each mapped to its catalog row (custom for LM Studio and Jan until a preset exists);… | — |
| provider-14 | No local server found is a dead end: no reason, no list of what was checked, no how-to *(stub)* | On the Provider step reuse the Model step's categorized copy: 'Nothing is answering at http://localhost:11434 — start Ollama (ollama serve), then Check again.' On no detection result: 'Looked on this computer at 127.0.0… | — |
| provider-15 | Endpoint errors misdiagnose or hide the cause (bad scheme, https-on-http, /api, HTTP 500) | Before prefixing, detect scheme typos (':/' without '//', or an unknown scheme before ':') → 'Start the address with http:// or https:// (e.g. | — |
| provider-16 | Provider errors hard to see: 2.74:1 strip far from the field, ✓/✗ lines in the same grey | Add outcome classes to the status region (-ok uses $success, -error uses $error, -warn uses $warning, each verified ≥4.5:1 on the panel background) and keep the ✓/✗/! glyphs. | — |
| provider-19 | No base-URL override or app-specific key for OpenAI, Anthropic or OpenRouter; an env key hides the key field | Add an 'Advanced' Collapsible inside Authentication for every keyed provider: 'Base URL (optional)' with the built-in endpoint as placeholder (persisted through provider_setup_persistence's endpoint precedence only when… | — |
| provider-20 | Z.ai's Chat URL preview is wrong (…/paas/v4/v1/chat/completions) and may be saved | Per task-33621.32: let the provider/preset record own its route suffix (as engine presets already do through _ENGINE_PRESETS) and add Z.ai's; preview and persist the real chat route. | task-33621.32 (To Do) |

**Voice / Speech**

| ID | Issue | Fix | Backlog |
|---|---|---|---|
| new-voice-speech-01 | Speech step reads [transcription] from a config key that does not exist at startup, so re-run prefill, the saved-language preselection and… *(new)* | Read the raw section the way the Summary does: from app_config['COMPREHENSIVE_CONFIG_RAW'] when present, else from the top level. | — |
| voice-speech-03 | Preselected PocketTTS needs a server that isn't running; test fails with no cause or fix | (1) When the step shows and when the preset changes, run a worker probe of at most 1 s: TCP connect to 127.0.0.1:8765, check _existing_openai_credential(), and reuse the OmniVoice state check that already exists. | — |
| voice-speech-04 | OpenAI voice without a key blocks Next, and 'Add API key in Settings' silently ends setup | When auth is 'API key' and no credential is found, show an inline masked Input 'OpenAI API key' (password=True) in the Voice step. | — |
| voice-speech-05 | After Test and Hear focus jumps to the top; the next Tab+Space wipes the sample and result | Keep 'Test and Hear' enabled during a test and relabel it 'Testing… (press to cancel)' so focus never drops. Otherwise, remember app.focused before disabling and call query_one('#setup-voice-test').focus() in the test's… | — |
| voice-speech-09 | Speech download has no Cancel, speed or ETA, and its status still reads 'Not installed.' | While _operation == 'install', show the status 'Downloading Parakeet v2 (English, INT8) — 65.0 of 632.8 MiB · 4.1 MiB/s · about 2 min left', with speed and ETA computed in ModelInstallProgress from successive Acquisitio… | — |
| voice-speech-10 | Leaving Speech mid-download is silent, and the finished model never becomes the default | If _operation == 'install' when the user presses Next or Exit, open a SetupModal: 'The speech model is still downloading (65 of 633 MiB). [Keep downloading in background] [Cancel download] [Stay here]'. | — |
| voice-speech-12 | Speech step shows three expert setup paths and jargon, with actions above their choices | Restructure: (1) Start with an engine/recommendation card (engine choice: voice-speech-15): 'Recommended: Parakeet v2 — English speech-to-text that runs on this computer. | — |
| voice-speech-13 | transcribe.cpp GGUF picker saves at once, isn't used for dictation, and Summary ignores it | Decide where the option belongs. Dictation never routes to transcribe.cpp, so move 'Use an existing transcribe.cpp GGUF…' to Library ingest settings or Lab > Models (task-1915/2808), or relabel it under Advanced as 'For… | — |
| voice-speech-15 | Speech offers only a 633 MiB Parakeet download or a GGUF; no cloud Whisper or already-installed engine | Start the step with an 'Engine' SetupRadioSet built from detected capabilities: 'On this computer — Parakeet (recommended, 633 MiB download)'; 'Already available — <detected engine> (no download)' when its runtime impor… | — |
| voice-speech-16 | Speech has no Settings home, and setup names three different places to change it later | Add a 'Dictation (speech-to-text)' section to Settings > Speech & TTS that owns [transcription] default_provider, model and language, with a link 'Manage downloaded models in Lab ▸ Models'. | — |

### P3 — polish (52)

**Accessibility & terminal**

| ID | Issue | Fix | Backlog |
|---|---|---|---|
| a11y-04 | Large terminals: choice lists stay 5 rows and the panel leaves 30–40 empty rows | 1. Make the lists size to the viewport: '.setup-choice-list { min-height: 5; max-height: 30vh; }'. That is about 7 rows at 24 lines, 12 at 40 and 21 at 70, so small terminals are unchanged and large ones use their space… | — |
| a11y-13 | Toast severity is shown only by border colour | Add an app-level notify wrapper that adds a severity title when none is given: 'Warning', 'Error' or 'Done'. A glyph prefix ('⚠ ', '✗ ', '✓ ') is an alternative. | — |
| a11y-14 | Selected radio rows use a side-stripe accent border that DESIGN.md forbids | Delete the border-left line. Keep the tint, bold-underline and the structural ● glyph, which together already satisfy WCAG 1.4.1. | — |

**Coverage & power use**

| ID | Issue | Fix | Backlog |
|---|---|---|---|
| coverage-10 | No scripted or documented way to skip or pre-seed setup on another machine (no CLI flag; TLDW_CONFIG_PATH and [first_run] contract undocume… | CLI: `--config PATH` (sets TLDW_CONFIG_PATH for the run), `--no-setup` (never auto-offer this launch), `--setup` (open the wizard at launch), `--no-splash`. | — |
| coverage-13 | RAG model list shows malformed ('bge-*-en-v1'), placeholder and mis-specified rows as raw ids, with no metadata, recommendation or current… | If coverage-01 lands, the wizard no longer renders this table. Fix the template regardless: quote keys (`[embedding_config.models."bge-small-en-v1.5"]`), move the two samples into comments, correct dimensions (3-small 1… | — |
| coverage-15 | Style offers a 78-card splash gallery but no way to turn off or shorten the 7 s splash, reduce motion or use ASCII marks | Replace the card gallery with a “Startup and motion” group: radio “Startup animation: ○ Off ● Short (2 s, any key skips) ○ Full (7 s)” → splash_screen.enabled/duration/skip_on_keypress; SetupCheckbox “Reduce motion (sti… | — |
| new-coverage-power-01 | Splash skip works but is undiscoverable (no 'press any key' hint) *(new)* | Render a dim 'Press any key to skip' line under the progress bar when skip_on_keypress is true. | — |
| new-coverage-power-02 | Env-key first launch stacks the 'ready to chat' toast on top of the model-list consent modal (and the animated setup backdrop) *(new)* | Sequence first-launch notices: show the consent modal first, then an honest, action-bearing key-detected notice, or merge both into the env-aware Get started card proposed under coverage-04. | — |

**Cross-cutting**

| ID | Issue | Fix | Backlog |
|---|---|---|---|
| cross-cutting-10 | Tracker drops step names on the Full track and at 80 columns, lags the track choice, and has no legend | In compact mode, add a one-line caption under the boxes, e.g. 'Step 7 of 11 · Tools — next: Notes'. Re-project the tracker on the Welcome track's RadioSet.Changed, so choosing Full previews '11 steps' straight away. | — |
| cross-cutting-11 | Setup has seven names; tracker labels differ from step titles; Voice and Speech collide | Create one glossary module for setup copy and feed it into TASK-33623's registry: - The task is called 'Setup', with the verbs 'Run setup' and 'Resume setup'. | — |
| cross-cutting-12 | Setup is hard to find again: buried under Settings ▸ Troubleshooting ▸ Diagnostics (~11 keys), and the palette only matches 'setup' | Put 'Run setup' (or 'Resume setup' while a draft exists) at the top of Settings ▸ General and in the Providers & Models empty state; keep Diagnostics as a secondary home. | — |
| cross-cutting-14 | Every Next blocks 2.5–4 s (3–6 s for Full, up to 30 s on Voice) with no busy cue, and choosing Full gives no feedback until Next | When _set_advancing(True) fires, write a neutral busy line to the pinned strip: 'Saving voice settings…', 'Checking OpenAI models…' or 'Preparing 11 steps…'. | — |
| cross-cutting-18 | Guide carries 'Verified against' stamps CLAUDE.md forbids; TASK-33008 AC#8 asks for more | Move the evidence from the six stamps into the notes of the named tasks and delete the stamps from the page. Amend TASK-33008 AC#8 to 'pages updated (no Verified-against stamp, per CLAUDE.md)'. | — |
| cross-cutting-19 | Resume restores a saved preset voice as 'Custom' and a list-picked model as typed text | Restore the preset radio from the draft's 'preset' value for all four services. Store the model's provenance ('list' or 'typed') in the draft: press the matching radio when the id is in the rendered list, otherwise fill… | — |

**Entry, exits & handoff**

| ID | Issue | Fix | Backlog |
|---|---|---|---|
| entry-exit-handoff-05 | 'A Console turn completed while hidden' toast fires while Console is on screen | As in task-33620.6, decide 'hidden' at completion time from (app.screen is the Console screen) AND (the turn's tab is the active tab), and suppress both the toast and the nav badge otherwise. | task-33620.6 (To Do) |
| entry-exit-handoff-11 | Command-palette re-run has no guard against opening a second wizard | Route the palette through the single app.open_setup_wizard(origin='palette') from -10, which carries the guard. If the wizard is already open, focus it and show a toast: 'Setup is already open'. | — |
| entry-exit-handoff-15 | Skip, Exit and the next launch land on three different screens | Use one landing for every abandon: Console with its Get started card, the designed catch from ADR-210. Skip returns exit_route = TAB_CHAT with completed = True, and without the consent modal (-13). | — |
| entry-exit-handoff-18 | Welcome copy oversells: 'Full setup — configure everything' and time estimates that exclude downloads, plus an empty red error bar | One WelcomeStep rewrite. Delete line 7404. Subtitle: 'Chat with AI models — on this computer or in the cloud — keep notes, and work with your own documents, all in your terminal. | — |
| entry-exit-handoff-22 | Console arrival overwhelms: 14 tabs, late tabs shift the layout, 'Agent blocked' in status | Add a first-run arrival profile, used when the first-chat intent is consumed: collapse the rail, hide the Speak/Hands-free cluster until voice is configured, and mount the late nav tabs before first paint (or reserve th… | task-33620 (To Do, one run-state and readiness truth), task-33005.7 (To Do, readiness vocabulary) |
| entry-exit-handoff-28 | No 'start with my documents or notes' path on Welcome | Add a third Welcome choice: 'Start with my documents or notes — set up AI later'. It writes setup_completed (no consent modal), opens Library Import with a 'Write a note' secondary, and leaves the Console Get started ca… | task-28019 (To Do, low) |
| new-entry-exit-handoff-02 | First-run toasts overlay the nav tab bar and status chips at top-right for about 5 s *(new)* | Anchor Console toasts below the header and status rows, as the other Console notices are, or replace arrival toasts with a one-line transcript receipt. | — |

**RAG / Tools / Notes / Style**

| ID | Issue | Fix | Backlog |
|---|---|---|---|
| fulltrack-05 | Notes folder sync is a whole Full-track step with no control that configures nothing | Remove NotesSyncStep from _FULL_TRACK (Full becomes 10 steps) and drop 'notes' from WIZARD_OWNED_SECTIONS. Add a Summary next-step action "Sync a notes folder…" beside "Write your first note". | — |
| fulltrack-06 | File tools can be switched on with no hint that they only reach the Chat scratch area and Workspace folders, and no pointer to web keys or… | Reword the shared blurbs, which also fixes the MCP pane: "Read a file in your Workspace folders or the chat scratch area. Asks first unless you allow longer." When any of Read file / List directory / Find files / Search… | — |
| fulltrack-09 | Splash-card gallery is cosmetic, unpreviewable, and shows the wrong current choice on re-runs | Remove the card radio from setup and replace it with the "Startup animation" control in fulltrack-07. At most, keep one read-only line: "Animation: Surprise me — preview and pick in Settings ▸ Splash Screen". | — |
| fulltrack-11 | Without the RAG extras, the step is a dead page of pip instructions | When embeddings_rag_deps_installed() is False, either skip the step (active_step_ids filters it) and move the hint to the Summary row ("– Documents in chat — needs the optional search extras · Copy install command"), or… | — |
| fulltrack-12 | Tools list overflows at 120×40 with no "more below" cue, count or presets | Group the rows. "Look things up" holds Read file, List directory, Find files, Search in files and Expand document. "Make changes ⚠" holds Write file, Create note and Update note. | — |
| fulltrack-13 | Appearance polish: raw theme ids, hidden shortlist entry, Style/Appearance naming, wrong Settings pointer | Label themes with display names plus a tone tag, keeping the raw id on _theme_name: "Default dark", "Default light", "Nord · dark", "Gruvbox · dark", "Tokyo Night · dark", "Catppuccin Mocha · dark". | task-33623 (To Do) |

**Gap checks**

| ID | Issue | Fix | Backlog |
|---|---|---|---|
| gap-08 | After a terminal resize the focused field stays off-screen, so typing goes in blind *(new)* | In the wizard's on_resize, call call_after_refresh(focused.scroll_visible(animate=False)). Add a pilot test that resizes 200x50 → 80x24 and asserts the focused field's region is still visible. | — |
| gap-09 | In the light theme the tracker's '!' needs-attention marker measures 1.38:1, effectively invisible *(new)* | Use a theme warning token that resolves per theme (≥3:1 for the glyph, ≥4.5:1 preferred), and add the label '! Provider — not set' where width allows. Extend #110's pilot contrast test to run under textual-light. | — |

**Model**

| ID | Issue | Fix | Backlog |
|---|---|---|---|
| model-09 | Model copy polish: dead 'Refresh models', mixed terms, dropped typed model, echo dialog | (a) Replace with 'Connection settings changed since the list loaded. Pick a model from the updated list or type its ID again.' Or add a visible 'Refresh list' button and keep the wording. (b) Use 'model ID' throughout. | — |
| new-model-01 | Provider step's success line always claims 'your chosen model was not in its list' (hard-coded verdict) *(new)* | Merge into provider-21 with a corrected root cause. In the wizard, build the snapshot with model='missing' (no model chosen yet) and print '✓ Connected — OpenAI returned 137 models. | — |

**Protect / Summary**

| ID | Issue | Fix | Backlog |
|---|---|---|---|
| new-protect-summary-02 | The module-path crash traceback prints config frame locals, including part of the password verifier *(new)* | This goes away with the 01 fix if PasswordPromptApp is deleted. More generally, run any pre-main Textual app that handles secrets with locals display disabled, and keep secrets out of frame locals (pass them through a c… | — |
| new-protect-summary-03 | 'enc:' ciphertext passes the provider-key validity check, so any non-strict or skipped decrypt reads as Ready *(new)* | Treat values with the encryption prefix (ConfigEncryption.is_encrypted) as absent in resolve_provider_api_key and readiness. Add a test asserting that a locked config reads providers as 'key missing', not Ready. | — |
| protect-summary-07 | Nothing says where the key is stored or that it is plain text in config.toml; Protect's copy names a Skip that isn't there | Rewrite the stored-key state. Title: 'Protect your API key'. Body: 'Your OpenAI key is saved as plain text in <full config path, wrapped>. Any program or person using your account can read it. | — |
| protect-summary-12 | Summary copy shows raw ids, wrong homes and wrong failure causes | Render display names from settings_provider_catalog ('OpenAI', 'llama.cpp', 'Ollama', 'Custom (OpenAI-compatible)') and from the theme titles. | — |
| protect-summary-13 | Summary hides what power users need: full config path, data folder, key source, tool and voice names, and out-of-band writes | Expand the footer: 'Config: <full path, wrapped>' and 'Data: <resolved data dir>', each with a 'Copy' button (app.copy_to_clipboard, toast 'Path copied') and an 'Open Storage settings' link. | — |
| protect-summary-16 | Summary rows can't be acted on; order differs from steps; splash choice missing | Render the rows as focusable list items in step order, grouped as 'Needs attention', 'Ready' and 'Left at defaults'. Give each attention row an inline action that jumps to its step (generalise review_provider_setup into… | — |
| protect-summary-17 | 'Keep model lists fresh' alarms local users and hides what leaving it off means | Label: 'Refresh model lists when chatbook starts'. Description: 'Contacts only the providers you set up. Off: lists update when you press Refresh in the model picker.' Record the answer on every completing exit; this fo… | — |

**Provider**

| ID | Issue | Fix | Backlog |
|---|---|---|---|
| provider-12 | Legacy alias and '#2' rows look like duplicates; 'Ollama (legacy alias)' points at :5678 | Keep the single catalog and add a presentation filter in ProviderStep._grouped_sections: omit entries flagged legacy (and the '#2' slot) unless the current config already uses that key, and end the list with 'Show legac… | — |
| provider-17 | Pasted key can't be verified or quickly cleared (no reveal or echo, no Clear for a draft) | Add a 'Show' toggle on the key row (and a Ctrl+R binding) that flips Input.password; after a paste show a non-secret echo under the field, 'Pasted 51 characters, ending …abcd' (last 4 only, never logged, cleared on blur… | — |
| provider-21 | Provider copy polish: 'your chosen model', vague found-server banner, stray local buttons | Probe without a model from the wizard (verdict 'model missing') and map it to '✓ Connected — OpenAI returned 137 models. Pick one on the next step.' Banner: 'Found Ollama running on this computer — 3 models (http://127.… | — |
| provider-22 | The same local server is listed twice (127.0.0.1 and localhost) in Detected endpoints *(stub)* | Canonicalize loopback hosts (localhost, 127.0.0.1, ::1) in the dedupe key while keeping the configured spelling for display; probe each port once. | — |
| provider-23 | Extra Tab stop on the 'Authentication' header before the key field | For providers that require a key, render the key row without a Collapsible (or make the title non-focusable while it's required and expanded). | — |
| provider-25 | 'Where do I get a key' pointer exists for only 8 of 45 keyed providers | Add key_url to the provider catalog/preset record for every keyed provider and read it in the wizard (and Settings); fallback copy 'Get a key from your {display_name} account dashboard.' Add a test asserting every requi… | — |
| provider-26 | A provider-switch failure line is wiped before the user can read it | Resolved by provider-07's single dispatch (select only on OptionSelected). Independently, never clear a contained error on a no-op re-pick of the same provider. | task-33621.37 (To Do) |

**Voice / Speech**

| ID | Issue | Fix | Backlog |
|---|---|---|---|
| new-voice-speech-02 | Editing Advanced fields leaves the 'PocketTTS' service radio selected, and switching service then discards the edits *(new)* | When any Advanced field diverges from the selected preset's values, select 'Custom' (with RadioSet.prevent so the preset handler does not re-apply) and store the draft as _custom_draft. | — |
| new-voice-speech-03 | Voice step error stays visible after the cause is fixed *(new)* | Clear the step error (show_step_error('') or the wizard's equivalent) on any input or radio change in the step. Better, do it in SetupStep for all steps, so a fixed field never leaves a stale error behind. | — |
| voice-speech-06 | Voice copy polish: phantom Save, unexplained 'Use as default', wrong-tense success line | Default status: 'Optional — press Test and Hear to play a short sample.' Success: 'Played the sample — sounds right? Continue with Next, or press Test and Hear to replay.' Playback failure: 'The service answered, but th… | — |
| voice-speech-08 | Parallel OmniVoice and Parakeet installs are untested; total download size is never shown | Do TASK-33079 (fake-source concurrent provisioning test, plus a live run). Show in-flight downloads on the tracker ('Voice ↓34%', 'Speech ↓12%') and on Summary ('Downloads in progress: 2 — 1.1 GB + 633 MiB'). | task-33079 (To Do) |
| voice-speech-11 | An interrupted speech download leaves an orphaned partial file and restarts from zero | When the Speech step shows, run reconcile() for this artifact in a worker (or a scoped resume probe). If a resumable partial exists, show 'A previous download stopped at 129 of 633 MiB. | — |
| voice-speech-14 | Summary flags a working model-from-disk Parakeet setup as 'configured but not installed' | Resolve readiness through the same source resolver dictation uses (STT/parakeet_sources.py, parakeet_external.py), and accept an active external source. | — |
| voice-speech-18 | Install review shows an internal ticket ID and hashes before the facts users need | Start with a three-line summary: 'Download 633 MiB once · Installs to ~/.local/share/tldw_cli/…/models (110 GB free) · Licences: CC-BY-4.0 (speech model), MIT (voice-activity detector)'. | — |
