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

While a Next is saving, **← Back**, **Next →** and **Skip setup** / **Exit
setup** are disabled and drawn dimmed. A Next that is still working after
about half a second also shows a line just above them that names the work:
"Preparing the Full setup…" after Welcome, "Saving the OpenAI connection…"
after Provider, "Saving the provider and model…" after Model, "Saving voice
settings…" after Voice (that save can take up to 30 seconds), "Finishing
setup…" after a Summary button, and "Saving *step* settings…" elsewhere. After
two seconds the line also counts the seconds ("Saving voice settings… 4 s").
It goes away as soon as the next step opens, setup closes, or the step
explains why it can't move on. A quick Next shows no line at all. When a step does refuse to move on (a missing API key, say), the keyboard
stays where it was, on **Next →** if you pressed it, so once you have fixed
the problem, Enter there (or Ctrl+N anywhere) tries again.
The search for local servers that the Provider step starts runs in the
background, so the screen keeps responding while it looks.

The model list that the Provider step fetches is kept for that provider, key
and address, so going Back to Provider and forward again reuses it. Only a
list that arrived is kept. After a check that failed (the server wasn't
running yet, say), setup asks the server again in the background the next
time it needs the list: when you go Back to Provider, when you press **Next →**
there, when you come Back to the Model step from a later step, and when you
press **Retry** on the Model step. A model list that arrives after a failed
**Test connection** replaces that test's result. A different key or address
always counts as a new check.

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
| Voice | Spoken replies — "No voice for now" (the default), PocketTTS (its own local server), OpenAI (your OpenAI key), a Custom endpoint, or OmniVoice (local, installs its model here); sample + "Test and Hear" (endpoint, model, voice and format under Advanced) | Settings ▸ Speech & TTS |
| Protect keys | Config encryption (password at startup) | Settings ▸ Privacy & Security ▸ **Encryption**: Encrypt keys, Change password, Turn off encryption (each asks for the master password) |

The Tools step is the only place in setup that turns a tool on, and it says
so up front: "Everything is off by default. Tools that read or change your
files still show an approval card every time they run." Each row carries the
tool's plain-language name and one line about what it does — the read-class
ones (Read file, List directory, Find files, Search in files, Expand
document) add that they ask before running unless you approve a longer
scope, and the ones that write are marked with ⚠. Leaving every switch off
is a supported outcome: the summary then reads "gates off; the assistant
may still use chatbook's own tools (sub-agents, web search, notes, Watchlists)
when the model's context window has room. Turn gates on under MCP ▸ Servers ▸
built-in row ▸ Tool gates", which is
where the same switches live after setup. The first half is there because
the switches are not the whole story: Console's assistant also gets the agent
runtime (sub-agents, skills), the local web and Watchlists tools
(`[console] local_tools_enabled`) and chatbook's built-in notes and character
tools, every call still subject to its MCP Ask/Allow/Off permission. The
app's internal `chat_with_llm` tool, which the in-process server cannot run,
is never offered. A self-hosted model whose context window the app has not
read yet gets a plain request with no tools until it can be sized.

The Voice step's first choice is **No voice for now**, and it is selected
unless a voice is already saved: Next then writes nothing, and the step says
so ("Nothing is saved. Set up a voice any time in Settings ▸ Speech & TTS.").
On a re-run the step starts from the voice you saved, for example "Current
voice: OpenAI · tts-1-hd · shimmer — unchanged unless you edit it.", and Next
leaves it exactly as it was unless you change something, test a sample, or
change **Use this voice when Chatbook reads replies aloud**. Choosing **No
voice for now** over a saved voice keeps that voice and says so ("Keeps your
current voice (OpenAI · tts-1-hd · shimmer); Next changes nothing. Replies are
read aloud only while Speak replies is on in Console."). When another provider
reads your replies (for example kokoro, set up in Settings), the step starts on
**No voice for now** and names it ("Current voice: kokoro — kept as it is").
A profile that still holds the PocketTTS address an earlier version of setup
wrote (`127.0.0.1:8765/v1/audio/speech`, which pocket-tts never serves) also
starts on **No voice for now**, with a line saying that voice can't speak; pick
a service to replace it. Resuming a setup that an earlier version left
unfinished doesn't bring that address back either.

A line under the service choice says whether it will work: "PocketTTS — not
running at 127.0.0.1:8000" (one quick connection check), "OpenAI — uses your
OpenAI key (key found)", and so on. That check only shows that something is
listening at the address, so when it is, the line says "a server is listening"
and leaves it to **Test and Hear** to confirm the server is PocketTTS (port
8000 is a common default for other local servers). It only connects to an IP
address or `localhost`; for a host name the line says **Test and Hear** checks
it, so typing an address never waits on a name lookup. **Test and Hear** plays a
short sample; Enter in **Sample text** runs it too (the hint line says so). A
failed test starts "Test failed —" and names the cause: the server isn't
running at that address, the key was rejected, there is no speech endpoint at
that host, it timed out, or the reply wasn't audio. A PocketTTS address on
another port counts as Custom, but its failures still name PocketTTS and say
how to start it there (`pocket-tts serve --port 8766`). While **Test and
Hear** can't run, the status line under it says why ("Type some sample text
above to test the voice.", or "To test, fix this under Advanced: …", for
example a PocketTTS address with a format other than `wav`). After a test,
focus goes back to **Test and Hear**, unless you moved it while the test ran.

PocketTTS, OpenAI and Custom share one OpenAI-compatible voice slot. While no
other provider reads your replies, a voice saved there *is* the one replies
use, so **Use this voice when Chatbook reads replies aloud** is ticked and
can't be unticked, and the line under it says so ("Replies will use this voice
— no other voice is set up.", or "This becomes the voice replies use — it
replaces OpenAI · tts-1-hd · shimmer."). When another provider reads replies,
the box is yours: a successful test ticks it, and unticked the service is
saved for later while replies keep using that provider ("Saved for later;
replies keep using kokoro."). When OmniVoice reads your replies, the step
starts on **OmniVoice** with the box ticked and locked ("Replies use OmniVoice
now — pick another service to change that."), and the other services name it
("Saved for later; replies keep using OmniVoice."). The
endpoint, authentication, model, voice and format sit under "Advanced" (the
**API key** option uses your OpenAI key, from the Provider step, this step,
Settings or `OPENAI_API_KEY`); Voice and Format are pickers with an "Other…"
choice, and editing one of these away from the selected service's own values
switches the service to **Custom** (your edits are kept if you switch away and
back).

**PocketTTS** is a separate local server, not part of Chatbook: install
`pocket-tts` (from PyPI) and start it with `pocket-tts serve`. Chatbook talks
to its own API at `http://127.0.0.1:8000/tts` (its default port; change the
endpoint under Advanced if you started it on another port). It returns WAV
audio only.

**OpenAI** needs an OpenAI API key. If none is found, paste one into the
masked **OpenAI API key** field that appears: it is kept in setup until you
press Next, then saved where Settings ▸ Speech & TTS keeps it, so **Protect
keys** can still encrypt it. You can also pick another service or choose "No
voice for now". **Leave setup and add key in Settings…** asks first — setup
picks up at Voice next time.

When Voice does save, the step reports the result itself and refuses to move
on if the save failed, so setup never raises a pop-up notification over a
later step's buttons. On terminals smaller than about 100×30 the wizard shows
a one-line nudge — everything still works, steps just scroll. Where one row is
too narrow for every service name (80 columns, for example), the Voice
service choice wraps to two rows so no name is cut off.

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
the voice replies use — service, model and voice (for example "✓ Voice —
OpenAI · tts-1-hd · shimmer") — or another provider that reads them, adding a
voice saved beside it ("kokoro (default voice); PocketTTS also saved"). It
reads "not set up (optional)" when no voice is saved, and flags the old
`127.0.0.1:8765` PocketTTS address as unable to speak instead of showing a ✓.

The Summary's exits are **Review provider setup**, **Add your first document**
(lands on Library's Import canvas — this is where your content lives),
**Write your first note** (lands on Library's New note view — no provider
needed), **Explore Home**, and **Review settings**.

Finishing with **Start chatting** opens Console on the provider and model you
just saved, in one chat tab, with no warning. The empty transcript says what
setup connected, for example "Setup complete — OpenAI · gpt-4.1-mini. Ready —
type a message to begin." If the saved default changed between setup and
Console opening, a single notice names the model Console is using instead.

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

## Setting up another machine

A second computer does not need setup again. There are three routes:
carry your config.toml, export the same environment keys, or restore a
backup. The first two carry settings only; to bring your chats, notes and
documents with them, also copy the data folder (below). A backup can carry
both, but Create backup does not yet work on every profile; see "Restore a
backup" below before relying on it.

### Carry your config.toml

Setup saves what it configures in one file, `config.toml`, by default
`~/.config/tldw_cli/config.toml`. Copy it to the new machine, then either
put it at the same path and start `tldw-cli` as usual, or keep it anywhere
and point chatbook at it:

```
tldw-cli --config /path/to/config.toml
python -m tldw_chatbook.app --config /path/to/config.toml
TLDW_CONFIG_PATH=/path/to/config.toml tldw-cli
```

`--config` and the `TLDW_CONFIG_PATH` environment variable do the same
thing: `--config` for one launch, the variable for every launch that sees
it. When both are set, `--config` wins. If the file does not exist yet,
chatbook creates a new config there and offers setup as on a first launch;
its folder must already exist. Before anything starts, `--config` refuses
a folder ("give the config.toml file inside it"), a file in a folder that
does not exist ("… does not exist; create that folder or check the path"),
a new file in a folder you cannot write to ("cannot create …"), and a file
or folder you cannot read ("cannot read …"). With either, Settings names
the file an override config.

The `[first_run]` table decides whether setup is offered:

| In the copied file | On the new machine |
|---|---|
| `setup_completed = true` | Setup is not offered; chatbook opens as it did on the old machine. |
| `setup_started = true`, without `setup_completed` | The setup was left unfinished: chatbook asks "Continue setup?" when the file still holds that run's progress, and otherwise says setup isn't finished. It never restarts setup by itself. |
| No `[first_run]` table | Setup is offered once, unless a provider is already configured (a key saved in the file, or a provider key in the environment). |

So copy the config of a finished setup. You can run setup again any time
from Settings ▸ Diagnostics ▸ Run Setup Wizard.

**Your keys travel with the file.** A copied config.toml carries every
provider API key saved in it, in plain text unless password encryption is on
(Protect keys, or Settings ▸ Privacy & Security ▸ Encryption). Handle the
copy like a password: don't commit it to a dotfiles repository, put it in a
shared or synced folder, or send it unencrypted. If encryption is on, the new
machine asks for the same master password before the app opens, through
both `tldw-cli` and `python -m tldw_chatbook.app`, including with
`--config` (see "Starting chatbook when your keys are encrypted" above).

What config.toml does not carry: your chats, notes, characters and documents
live in database files under `~/.local/share/tldw_cli/`, not in the config.
To bring them, quit chatbook on both machines and copy that whole folder,
every file in it, to the same place on the new machine. If chatbook has
already run there, move its folder aside first rather than copying into it;
the copy replaces anything chatbook saved there. A backup (below) carries
them too, once Create backup works on your profile. Paths written in the
file are used as written: `~/...` paths follow the new machine's home
folder, while an absolute path (a notes sync folder, a custom database
location) must exist there. `--config` and `TLDW_CONFIG_PATH` choose the
config file only; they do not move the data folder.

### Environment keys

chatbook never writes a key it read from an environment variable
(`OPENAI_API_KEY`, `ANTHROPIC_API_KEY` and so on) into config.toml. A setup
that used an exported key therefore carries no key in the file: export the
same variables on the new machine. When the file also holds a saved key for
the same provider, the saved key is used. On a machine with no config at
all, an exported provider key is enough for chatbook to skip the setup
offer; it says once which variable it found ("Found OPENAI_API_KEY — you're
ready to chat. Run setup any time: Settings ▸ Diagnostics ▸ Run setup
wizard.").

### Launch flags

| Flag | What it does |
|---|---|
| `--config PATH` | Uses this config.toml for this launch (the same as `TLDW_CONFIG_PATH`; the flag wins when both are set). |
| `--no-splash` | Skips the splash screen for this launch only. `[splash_screen] enabled` in config is not changed. A `--serve` browser session is started without it, so it still shows the splash when config enables it. |

Neither flag writes to config.toml, and no launch flag marks setup as
completed. `tldw-cli --help` lists both and ends with a note that names
`TLDW_CONFIG_PATH` and this section, with a link to this page. A recovery
profile opened from Backup & Restore selects its own config, so `--config`
cannot be combined with it.

### Restore a backup

To bring your data as well as your settings, create a backup on the old
machine (Settings ▸ Overview ▸ **Backup & Restore** ▸ **Create backup**). It
writes a `.tldw-backup.zip` file, or `.tldw-backup.zip.age` when encrypted.
Backup coverage is still growing: the Review lists each store, and one
marked `unsupported` (on some profiles that includes the config and the main
databases) is not in the archive. Until the Review shows what you need as
included, carry config.toml and the data folder as described above.

**Known problem:** on some profiles Create backup currently stops with
"Failed: capturing …" (ending in `backup_operation_failed` or
`admission_timeout`) and writes no file, even after a successful Review. If
that happens, carry config.toml and the data folder as described above
instead.

In the Create backup form, the line just above the buttons says why
**Create backup** is disabled: "Create backup unlocks after a successful
Review." until **Review** succeeds (and again after any change to the form),
or the review's own reason, such as a Partial backup that needs "Acknowledge
Partial archive…" ticked, or not enough free space (at the destination, or
in the temporary folder where the backup is staged). Pressing **Create
backup** uses up that review, so the line then reads "Backup started;
progress is shown at the top. Press Review to create another."

On the new machine, setup's Welcome step (and the "Continue setup?" dialog)
has **Restore a backup**. It opens "Restore from a backup" straight on the
Inspect / restore pane, with the cursor in the archive field and the format
named under it; **Inspect / restore** is the first of the actions above it.
Choose the archive and press **Inspect** (or Enter in the field); the
restore controls appear once the archive is verified. **Esc** returns to
setup with your choices intact.

- A config.toml there gets "That's a settings file, not a backup archive."
  with the `tldw-cli --config <this file>` command and this section's name.
  Carry it as described above instead.
- A folder gets "Choose the archive file, not a folder."
- Changing the archive path clears either message.

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
