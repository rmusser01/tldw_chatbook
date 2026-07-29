# First-Run Setup

> Verified against: first-run wizard implementation, 2026-07 (this page ships with it).

On your first launch, chatbook offers a guided setup. It is entirely optional —
every step has a Skip, Escape asks before closing, and anything you configure
(or don't) can be changed later in Settings.

## The two tracks

- **Quick setup (recommended)** — connect one provider, pick a default model,
  done. Everything else stays at recommended defaults (tools off, RAG off,
  default theme, notes sync off).
- **Full setup** — also walks through RAG/embeddings, built-in tools, notes
  sync, appearance, and key encryption.

## What each step does

| Step | What it configures | Where it lives in Settings |
|---|---|---|
| Provider | API key or local server endpoint | Providers & Models |
| Model | Default chat model | Providers & Models |
| RAG | Embedding model (needs the `embeddings_rag` extras) | RAG |
| Tools | Built-in tool gates (all off by default) | Tools |
| Notes sync | Folder + on/off toggle | Notes |
| Appearance | Theme and splash screen card | Appearance |
| Protect keys | Config encryption (password at startup) | Privacy & Security |

The final summary shows a ✓/✗ line per area, read back from what was actually
saved. Local servers (Ollama, llama.cpp) are auto-detected on localhost; no
probe traffic leaves your machine without your action.

## Running it again

- **Settings ▸ Diagnostics ▸ Run Setup Wizard**, or
- Command palette: "Setup: Run setup wizard…"

On a re-run, current values are prefilled and stored API keys are shown only
as "configured" — never displayed.
