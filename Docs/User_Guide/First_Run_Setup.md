# First-Run Setup

> Verified against: fix/task-32959-summary-voice-line, 2026-09-27 (task-32959: the Summary gains a Voice line read back from the saved `[app_tts]` table). Previously: feat/wizard-omnivoice-tts, 2026-09-26 (task-32958: OmniVoice as a fourth Voice service).

On your first launch, chatbook offers a guided setup. It is entirely optional —
most steps can be skipped (Next moves on without configuring it; the one
exception is a cloud provider you've picked, which needs its API key before
Next continues — press **Enter** in the empty key field instead and the
provider step is skipped outright, which the field's own hint says), Escape
asks before closing, and anything you configure (or
don't) can be changed later in Settings.

If a step can't save what you entered, the reason appears just above the
navigation buttons — fix it and press Next again, or go Back. The same line
reports an unexpected error on a step: the step stays on screen, the keyboard
keeps working, and **Esc** (exit setup) and, after the first step, **← Back**
stay available, from the keys (**Ctrl+B**, **Ctrl+N**, **Enter**, **Esc**)
as well as the buttons. That includes a run you resumed from "Continue
setup?", and a step that fails while it opens: setup lands on that step with
the error line. If picking a provider fails this way, the provider you had
picked before stays selected, together with any key you typed for it; a key
is never carried over to a different provider. The highlight stays on the row
you tried, so the line names the provider that is still selected ("Couldn't
switch to Anthropic — OpenAI is still selected"), and it goes away once you
pick a provider that works. If you had not picked one yet, none is selected,
and Next goes on without a provider, as if you had skipped the step. A
failure after the new provider is already selected leaves it selected, and
the line then names no other provider.

The Provider step lists the same providers as Settings ▸ Providers & Models:
Popular first, then Cloud, Local and Other. Every row can be picked with the
arrow keys and set up here, hosted presets included (Together, Fireworks,
NVIDIA NIM, BytePlus, DeepInfra and the rest; see
[Settings ▸ Inference clouds](settings.md#inference-clouds)). Setup saves each
preset's key, model and documented base URL in its own
`[api_settings.<provider>]` table. **Custom Hosted** is not listed: it is the
internal engine route for custom endpoints, which you set up as **Custom
OpenAI-compatible**.

## Keyboard

- **Enter** continues to the next step (from a choice list or a text field;
  in the API-key field it first tests a key you typed, and with the field
  empty it skips the provider step). **Ctrl+N** / **Ctrl+B** also
  move next/back, and **Escape** asks before leaving setup.
- **Arrow keys select** as they move through a choice list — what you land on
  is what you get; no extra keypress needed.
- **Tab** moves from a step's content to **Next** first; the footer runs
  progress · Back · Next · Skip/Exit, left to right.

## The two tracks

- **Quick setup (recommended)** — connect one provider, pick a default model,
  optionally try a voice and protect your keys, done. Everything else stays at
  recommended defaults (tools off, RAG off, default theme, notes sync off).
  The step count never changes mid-run — the key-protection step is always
  shown, and simply says so when there is nothing to protect yet.
- **Full setup** — also walks through RAG/embeddings, built-in tools, notes
  sync, appearance, and key encryption.

## What each step does

| Step | What it configures | Where to change it later |
|---|---|---|
| Provider | API key or local server endpoint | Settings ▸ Providers & Models |
| Model | Default chat model | Settings ▸ Providers & Models |
| RAG | Embedding model (needs the `embeddings_rag` extras) | Settings ▸ RAG |
| Speech (full track) | Voice-input transcription language and precision | `[transcription]` in config.toml — no Settings category owns it yet |
| Tools | Built-in tool gates (all off by default) | MCP ▸ Servers ▸ built-in row ▸ **Tool gates**, or `[tools]` in config.toml — no Settings category owns them |
| Notes sync | Folder + on/off toggle | [Library ▸ Notes](library/notes.md), the toolbar's Sync panel — not in Settings |
| Appearance | Theme and splash screen card | Settings ▸ Appearance |
| Voice | Spoken replies — PocketTTS, OmniVoice (local, installs its model here), Official OpenAI or a compatible endpoint; sample + "Test and Hear" (endpoint/model under Advanced) | Settings ▸ Speech & TTS |
| Protect keys | Config encryption (password at startup) | Settings ▸ Privacy & Security ▸ **Encryption**: Encrypt keys, Change password, Turn off encryption (each asks for the master password) |

The Tools step is the only place in setup that turns a tool on, and it says
so up front: "Everything is off by default. Tools that read or change your
files still show an approval card every time they run." Each row carries the
tool's plain-language name and one line about what it does — the read-class
ones (Read file, List directory, Find files, Search in files, Expand
document) add that they ask before running unless you approve a longer
scope, and the ones that write are marked with ⚠. Leaving every switch off
is a supported outcome: the summary then reads "all off; turn them on under
MCP ▸ Servers ▸ Tool gates", which is where the same switches live after
setup.

The Voice step leads with a sample text and **Test and Hear**; the endpoint,
model, and output settings sit under its "Advanced" section. Advancing saves
the voice settings; the step reports the result itself and refuses to move on
if the save failed, so setup never raises a pop-up notification over a later
step's buttons. On terminals smaller than about 100×30 the wizard shows a
one-line nudge — everything still works, steps just scroll.

Choosing **OmniVoice** shows a local panel instead of the endpoint fields: it
tells you if the `omnivoice_tts` engine is missing, installs the 1.1 GB model
through the same consent dialog as the model browser, and tests a sample on
this computer. Saving it as the default also saves a fixed voice seed, but the
default voice can still vary between replies — create a voice profile in Voice
Cloning for a consistent voice.

The final summary shows a ✓/✗ line per area, read back from what was actually
saved — and if the connection check failed while you were setting up (a
rejected API key, an unreachable local server), the summary says so instead
of showing a ✓, the progress tracker marks those steps with !, and moving
past the model step asks for an explicit "Continue anyway". The Voice line names
the saved service (for example "✓ Voice — OmniVoice (default voice)"), says
when a voice was saved without being made the default, and reads "not set up
(optional)" when you skipped the step.

The Summary's exits are **Review provider setup**, **Add your first document**
(lands on Library's Import canvas — this is where your content lives),
**Write your first note** (lands on Library's New note view — no provider
needed), **Explore Home**, and **Review settings**.

The Summary also asks — once, default off — whether chatbook may check your
configured providers' model lists online at startup. Whatever you choose is
final until you change it in Settings; finishing setup never hands you a
separate consent pop-up afterwards. Local servers (Ollama, llama.cpp) are auto-detected on localhost; no
probe traffic leaves your machine without your action.

## Starting chatbook when your keys are encrypted

If you set a master password (Protect keys, or Settings ▸ Privacy &
Security ▸ Encryption), chatbook asks for it in the terminal before the app
opens. Both ways of starting it — `tldw-cli` and
`python -m tldw_chatbook.app` — use the same prompt:

```
Your saved API keys are encrypted. Enter the master password you set during setup.
Master password (leave empty if you forgot it):
```

- **A wrong password** prints "That password didn't match. Try again." and
  asks again. There is no limit on attempts.
- **The right password, but a key encrypted with a different one** (left
  behind by an older version that let you set a second password) prints
  "That password is right, but some saved keys were encrypted with a
  different password and can't be read with it." The app does not open with
  keys it cannot read. If you remember the earlier master password, enter
  it; otherwise leave the prompt empty to reset the keys.
- **An earlier password that still reads every saved key** is accepted even
  though it is not the latest one you set. chatbook says "That password
  reads every saved key, so it is your master password again", re-encrypts
  the keys under it and waits for Enter before opening the app. The later
  password stops working.
- **Forgot it?** Press Enter on an empty prompt. chatbook explains what a
  reset does and offers `[R]eset saved keys or [Q]uit`. **Reset** removes
  every encrypted value from config.toml and turns encryption off, then
  waits for Enter so you can read what happened, and opens the app. Chats,
  notes and documents are not touched; you re-enter your API keys in
  Settings ▸ Providers & Models. **Quit** leaves everything as it was. The
  choice is read from the terminal even when standard input is redirected.
- **Ctrl+C or Ctrl+D** at the prompt, or while the password is being
  checked, quits without changing anything.
- **config.toml says encryption is on but its password check is missing**:
  chatbook says so. If saved keys are encrypted, it asks for the master
  password: one that reads every key unlocks them and repairs the check, and
  an empty entry offers the reset. If nothing is encrypted, it offers the
  `[R]eset saved keys or [Q]uit` choice straight away.
- If chatbook cannot ask privately (no terminal attached), it opens a small
  recovery window that says so in plain words; relaunch from a terminal to try
  again. Backup & Restore stays one button away there but no longer opens on
  its own.
- **Browser sessions (`tldw-cli --serve`) cannot ask for the master
  password.** Each browser session shows that recovery window instead of
  prompting on the server's terminal. To serve chatbook in a browser, turn
  encryption off first (from a terminal launch, in Settings ▸ Privacy &
  Security ▸ Encryption). In that window, opening another profile from
  Backup & Restore says it needs a terminal instead of ending the session.

A saved key that is still encrypted is never treated as a usable key. If a
key cannot be decrypted, its provider shows as not ready, never ready. A
provider that needs a key reads as missing its key ("API key missing", and
"API key source: missing" in Providers & Models). A local provider that
needs no key still says why: "Saved API key is still encrypted". Either
way, re-enter or clear the key in Settings ▸ Providers & Models.

## Running it again

- **Settings ▸ Diagnostics ▸ Run Setup Wizard**, or
- Command palette: "Setup: Run setup wizard…"

On a re-run, current values are prefilled and stored API keys are shown only
as "configured" — never displayed.

*Verified against fix/library-crit8-polish-shell — 2026-09-08 (task-32072: the
Summary now offers "Add your first document", which finishes setup on Library's
Import canvas; the wizard previously never mentioned Library at all.)*

*Verified against fix/library-notes-onboarding — 2026-09-09 (task-32140: the
Summary offered no path into Notes for a local-first user without a
provider — every exit pointed at provider setup or the generic Import
canvas. Added "Write your first note", which finishes setup on Library's
New note view directly. The Console's post-setup "Get started" card gained
the matching "Write a note in Library" action, which needs no provider and
stays available for the whole time the card blocks the composer.)*

*Tools step verified against `fix/approval-wave-b-card` @ e7409210cc and
`fix/approval-wave-c-hub` @ a999fcf6e6 — 2026-09-10 (task-32290, against code
and tests, not a live screen): the step's own copy, the read-class "Asks
before running unless you approve a longer scope." descriptions, and the
summary's "all off; turn them on under MCP ▸ Servers ▸ Tool gates"
destination, which replaces this page's older "there is no Tools category"
pointer. Read-class wording updated 2026-09-11 (task-32290 part 2, Qodo
follow-up to task-32284/32289): the blurb's old closing sentence is gone,
replaced by the "approve a longer scope" wording quoted above, once the
wizard's copy moved onto `GateableTool.blurb` alongside the rest of the
tool row.*

*Verified against fix/library-notes-w3-wizard-toast — 2026-09-11 (task-32266:
the Voice step's save raised the global "Settings saved successfully!" toast,
which Textual docks bottom-right of the current screen — by the time the write
settled the wizard had advanced, so the toast landed over the Protect step's
buttons or the Summary's exit actions, "Write your first note" among them. The
wizard's save no longer announces itself; the step that made it still reports
every outcome in place.)*

*Verified against fix/library-notes-w4-console-handoff — 2026-09-14 (task-32555
AC#3, at 235x52): on the Provider step with OpenAI picked and no key, the key
field's hint ends "No key yet? Enter skips this step — you can add a provider
later in Settings." and Enter in the empty field moves from Step 2 of 6 to Step
3 of 6; the Summary then reads "✗ Provider — no credentials or saved endpoint"
(`wave4-caps/console-handoff/handoff-12a-wizard-key-hint`, `12-wizard-skip`,
`12b-wizard-summary`). Enter used to do nothing there — no probe, no advance,
no message. A provider that is already ready without a typed key (an exported
environment variable, a local server) is not cleared by that Enter; the hint
does not offer the skip in that state.*
