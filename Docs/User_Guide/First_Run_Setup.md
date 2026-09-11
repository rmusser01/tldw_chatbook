# First-Run Setup

> Verified against: first-run wizard implementation, 2026-09 (task-31820: honest skip/continue copy — a keyed provider without a credential refuses Next until a key is supplied or you go Back; Escape remains the universal out).

On your first launch, chatbook offers a guided setup. It is entirely optional —
most steps can be skipped (Next moves on without configuring it; the one
exception is a cloud provider you've picked, which needs its API key before
Next continues), Escape asks before closing, and anything you configure (or
don't) can be changed later in Settings.

If a step can't save what you entered, the reason appears just above the
navigation buttons — fix it and press Next again, or go Back.

## Keyboard

- **Enter** continues to the next step (from a choice list or a text field;
  in the API-key field it first tests the key). **Ctrl+N** / **Ctrl+B** also
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
| Voice | Spoken replies (sample + "Test and Hear"; endpoint/model under Advanced) | Settings ▸ Speech & TTS |
| Protect keys | Config encryption (password at startup) | Settings ▸ Privacy & Security is a read-out; encryption changes are password-gated and not editable there |

The Tools step is the only place in setup that turns a tool on, and it says
so up front: "Everything is off by default. Tools that read or change your
files still show an approval card every time they run." Each row carries the
tool's plain-language name and one line about what it does — the read-class
ones (Read file, List directory, Find files, Search in files, Expand
document) add that they ask you each time before running, and the ones that
write are marked with ⚠. Leaving every switch off is a supported outcome: the
summary then reads "all off; turn them on under MCP ▸ Servers ▸ Tool gates",
which is where the same switches live after setup.

The Voice step leads with a sample text and **Test and Hear**; the endpoint,
model, and output settings sit under its "Advanced" section. Advancing saves
the voice settings; the step reports the result itself and refuses to move on
if the save failed, so setup never raises a pop-up notification over a later
step's buttons. On terminals smaller than about 100×30 the wizard shows a
one-line nudge — everything still works, steps just scroll.

The final summary shows a ✓/✗ line per area, read back from what was actually
saved — and if the connection check failed while you were setting up (a
rejected API key, an unreachable local server), the summary says so instead
of showing a ✓, the progress tracker marks those steps with !, and moving
past the model step asks for an explicit "Continue anyway".

The Summary's exits are **Review provider setup**, **Add your first document**
(lands on Library's Import canvas — this is where your content lives),
**Write your first note** (lands on Library's New note view — no provider
needed), **Explore Home**, and **Review settings**.

The Summary also asks — once, default off — whether chatbook may check your
configured providers' model lists online at startup. Whatever you choose is
final until you change it in Settings; finishing setup never hands you a
separate consent pop-up afterwards. Local servers (Ollama, llama.cpp) are auto-detected on localhost; no
probe traffic leaves your machine without your action.

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
and tests, not a live screen): the step's own copy, the read-class "Asks you
each time before running." descriptions, and the summary's "all off; turn them
on under MCP ▸ Servers ▸ Tool gates" destination, which replaces this page's
older "there is no Tools category" pointer.*

*Verified against fix/library-notes-w3-wizard-toast — 2026-09-11 (task-32266:
the Voice step's save raised the global "Settings saved successfully!" toast,
which Textual docks bottom-right of the current screen — by the time the write
settled the wizard had advanced, so the toast landed over the Protect step's
buttons or the Summary's exit actions, "Write your first note" among them. The
wizard's save no longer announces itself; the step that made it still reports
every outcome in place.)*
