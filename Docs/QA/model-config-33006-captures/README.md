# Phase 6 (TASK-33006) full-screen captures (2026-10-03)

Every surface Phase 6 changed, captured from the real app at branch head `5210a4e981` (`git status -- tldw_chatbook` clean). Each moment was captured at 211x44. The tmux window was then resized to 235x52, captured again, and resized back. `.txt` is `tmux capture-pane -p`; `.ansi.txt` is the same moment with `-e` colour escapes. Focus on an Input or Select shows in the `.txt` as the █ cursor; focus on a Button shows only in the `.ansi.txt`, as its label on the focus fill (`48;2;25;68;102`).

## Setup

- **Isolation.** The app ran from this worktree in its own tmux server (`-L cap33006`) under `env -i`. `HOME`, `XDG_*` and `TLDW_CONFIG_PATH` pointed into a scratch directory, the keyring backend was the null one, and `PYTHONPATH` was this worktree.
- **Scratch profile.** users_name `verify_cap33006_5210a4e981`. `[chat_defaults]` is Anthropic `claude-sonnet-4-5`, Temperature 0.7, Max tokens 2048. It also sets `[first_run] setup_completed = true`, `[console.onboarding] first_send_completed = true`, `[model_catalog] auto_refresh_enabled = false`, and the splash off.
- **Two runs on that profile.**
  - Run A (01-14): a dummy `ANTHROPIC_API_KEY` in the environment, so the chat is ready.
  - Run B (15-17): no key anywhere, so the chat is Not ready (no key). Capture 16 pastes a fake key into Settings and saves it to the scratch config.
- **No provider contacted.** No Test connection, no paid test, no send. No key appears in any capture.
- **Input.** Keys sent with `tmux send-keys` (Ctrl+O, Tab, Shift+Tab, Enter, Esc, `d`, End, Backspace, Ctrl+T, typed text; Ctrl+Enter as its CSI u sequence `\e[13;5u`), plus SGR mouse input: clicks on Save as model default, Use saved defaults, a view tab, a Chat tab and Return to Chat settings, and wheel events to scroll the body.
- **Real profile unchanged.** Before and after, `shasum -a 256 ~/.config/tldw_cli/config.toml` began 15c6cb224a6a51c7 (mtime Sep 26), and `ls ~/.local/share/tldw_cli | shasum -a 256` began db7e7faf5bff92d2.
- **Cleanup.** tmux was killed and the scratch home deleted. The driver is not committed.

## Captures

Run A (ready Anthropic chat):

1. `01-model-view-anthropic` (Ctrl+O on the untouched chat). Title "Chat settings · Chat 1" and the scope line "Applies to this chat only · saved with the conversation · defaults live in Settings ▸ Providers & Models (F4)". The MODEL row reads "claude-sonnet-4-5 · Anthropic", "this chat", "Ready · not tested · ~200k context" and **Change Alt+M**. The CORE rows are Temperature (focused), Max tokens, Streaming, Thinking and Thinking budget, each with a Source word and a help line. The four closed disclosures are one row each. The footer reads Esc close, **Matches saved defaults** (dimmed), Default for new chats (Ctrl+N), Apply to this chat (Ctrl+Enter), without scrolling. The frame is 150x22 at both sizes.
2. `02-sampling-open-anthropic` (Tab x5, Enter). Sampling opens with "Anthropic does not accept: Min P, Seed, Presence penalty, Frequency penalty, Reasoning effort, Reasoning summary, Verbosity.", then Top P and Top K only.
3. `03-connection-open-anthropic` (Enter closes Sampling, Tab, Enter). Connection opens in place under "Connection · api.anthropic.com · key from env ANTHROPIC_API_KEY · change it in Settings ▸ Providers & Models". No Endpoint label shows, because Anthropic takes no server address.
4. `04-connection-open-scrolled-anthropic` (six wheel steps). The rest of Connection: the readiness detail and "Generation test unavailable for this provider.", then the closed Request estimate and name rows.
5. `05-estimate-and-name-open` (Esc, Ctrl+O, Tab x8, Enter, Shift+Tab, Enter, Tab x2, type "Ada"). Request estimate is open, and the name title reads "Your name in this chat · Ada" while typing. The title counts "1 unsaved edit". Esc then asked about the name, and `d` discarded it.
6. `06-change-opens-pick-mode` (Ctrl+O, Shift+Tab to Change, Enter). Switch model opens over Chat settings in pick mode ("Enter picks · Esc cancel"), with the chat's pair marked ● CURRENT. In pick mode the arrow keys move only through pickable pairs; the NEEDS SETUP "(any model)" rows are listed but cannot be picked.
7. `07-picked-pair-staged` (type `opus-5`, Enter). The draft is on "claude-opus-5 · Anthropic" with "edited *". Nothing is applied: the title reads "3 unsaved edits". The closed Sampling title counts the 8 fields claude-opus-5 does not accept.
8. `08-esc-asks-after-pick` (Esc). The unsaved prompt reads "3 unsaved edits to this chat: Model, Min P, Streaming." Min P is hidden for Anthropic, and Streaming still reads Off; rider TASK-33006.16 owns this. `d` then discarded the draft.
9. `09-context-view` (Ctrl+O, Shift+Tab x3, Tab, Enter on the Context and memory tab). The Context view opens at Model capacity. Every label ends at least one cell before its control ("Conversation max tokens │ if Custom"). The footer is Esc close, Cancel, Apply to this chat, with no defaults line.
10. `10-model-view-scrolled-to-end` (click the Model and generation tab; Tab x5, Enter opens Sampling; Tab x3, Enter opens Connection; 30 wheel steps). Top P is the first body row.
11. `11-context-view-after-scrolled-model-view` (30 more wheel steps, click Context and memory). The Context view opens at its top. All four files (`.txt` and `.ansi.txt`, both sizes) are byte-identical to `09`'s.
12. `12-chat-with-work-offers-use-saved-defaults`. Before it: Chat 1 applied Temperature 0.9 (Ctrl+Enter). In a new Chat 2 (Ctrl+T, Ctrl+O), Temperature 0.3 and Max tokens 4096 were saved with **Save as model default**, which wrote the scratch config. Then Chat 1's tab was clicked and Ctrl+O pressed. Chat 1 keeps Temperature 0.9 and Max tokens 2048 ("this chat"). **Use saved defaults** and **Save as model default** are offered, with "Used by future conversations for Anthropic." above them.
13. `13-use-saved-defaults-staged` (click Use saved defaults). Temperature 0.3 and Max tokens 4096 read "edited *", and the title counts "2 unsaved edits". The pair is unchanged. The button reads **Matches saved defaults**. The scratch config hash was the same before and after the click.
14. `14-reopened-applied-values` (Ctrl+Enter, then Ctrl+O). Temperature 0.3 and Max tokens 4096 are the chat's values and read "model default". The scratch config hash was the same before and after Apply.

Run B (no key):

15. `15-not-ready-opens-connection` (Ctrl+O). The MODEL row reads "Not ready · no key". Connection opens by itself ("key missing"), and **Configure credential…** has focus (`.ansi.txt`). The blocker line reads "Add or verify the Anthropic API key to continue."
16. `16-settings-credential-saved-return-to-chat-settings` (Enter on Configure credential…, a fake key typed into the focused API key field, Esc, `s`). Settings ▸ Providers & Models shows "API key source: local config key saved" and the **Return to Chat settings** button.
17. `17-returned-to-chat-settings-key-saved` (click Return to Chat settings). Chat settings comes back on Chat 1 with Connection open: "key saved", "Credential saved — checking readiness", then "Ready · not tested".
