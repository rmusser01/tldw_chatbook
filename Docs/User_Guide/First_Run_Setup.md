# First-Run Setup

> Verified against: first-run wizard implementation, 2026-08 (TASK-21148: layout/density — stable step totals, stacked tracker titles, outcome-first Voice step, small-terminal hint).

On your first launch, chatbook offers a guided setup. It is entirely optional —
every step can be skipped (Next moves on without configuring it), Escape asks
before closing, and anything you configure (or don't) can be changed later in
Settings.

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
| Tools | Built-in tool gates (all off by default) | `[tools]` in config.toml (Settings ▸ Advanced Config) — there is no Tools category |
| Notes sync | Folder + on/off toggle | [Library ▸ Notes](library/notes.md), the toolbar's Sync panel — not in Settings |
| Appearance | Theme and splash screen card | Settings ▸ Appearance |
| Voice | Spoken replies (sample + "Test and Hear"; endpoint/model under Advanced) | Settings ▸ Speech & TTS |
| Protect keys | Config encryption (password at startup) | Settings ▸ Privacy & Security is a read-out; encryption changes are password-gated and not editable there |

The Voice step leads with a sample text and **Test and Hear**; the endpoint,
model, and output settings sit under its "Advanced" section. On terminals
smaller than about 100×30 the wizard shows a one-line nudge — everything
still works, steps just scroll.

The final summary shows a ✓/✗ line per area, read back from what was actually
saved — and if the connection check failed while you were setting up (a
rejected API key, an unreachable local server), the summary says so instead
of showing a ✓, the progress tracker marks those steps with !, and moving
past the model step asks for an explicit "Continue anyway".

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
