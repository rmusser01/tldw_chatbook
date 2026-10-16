# Phase 8 (TASK-33008) full-screen captures

Live captures of the surfaces Phase 8 changed, keyboard only. The set was retaken
2026-10-13 at `4f2bffa20f` (after the capture fix 2 wave and its two fix rounds); the
captures the capture fix 3 wave changed were retaken the same day at `8b97eeb05f`
(after that wave and its six fix rounds; see "Retaken at 8b97eeb05f" below); the
captures the capture fix 4 wave changed were retaken the same day at `e5e5855ab5`
(after that wave and its four fix rounds; see "Retaken at e5e5855ab5" below); the
captures the capture fix 5 wave changed were retaken the same day at `7c260226ec`
(after that wave and its one fix round; see "Retaken at 7c260226ec" below); `06d`
was retaken 2026-10-16 at `f54e5fa863` (after its copy fix; see "Retaken at
f54e5fa863" below).
Each moment was captured at 211x44, the tmux window was resized to 235x52 and
captured again, then resized back. `.txt` is `tmux capture-pane -p`; `.ansi.txt` is
the same moment with `-e` colour escapes (focus shows there: a focused button is bold
and underlined on the accent colour).

## Key counts

Re-measured at `4f2bffa20f`, `8b97eeb05f`, `e5e5855ab5` and `7c260226ec`; both are
unchanged from the first set.

- **Parent AC#1, local server: 2 key presses**, measured at 211x44. From the card
  with "Use detected llama.cpp (127.0.0.1:8080)" focused (`02a`): **Enter** opens
  Switch model on llama.cpp (`02b`), **Enter** applies the highlighted model. The
  card is gone and the composer has focus (`02c`; a typed character landed in the
  composer and was deleted). Saved: `chat_defaults` llama_cpp · qwen3-8b-instruct-q4_k_m
  and `api_settings.llama_cpp.api_url = "http://127.0.0.1:8080"`.
- **Parent AC#2, cloud key: 6 key actions** (budget 12), measured at 211x44, the
  key check and the return included. From the card (`03a`): **Enter** on Set up
  provider (Settings opens on OpenAI with the API key field focused, `03b`), one
  **paste** of the key (`03c`), **Esc**, **s** (saved, focus on Return to Console,
  `03d`), **t** (key check: "Ready · verified", focus stays on Return to Console,
  `03e`), **Enter** (back in "Chat 1", card gone, composer focused, `03f`; a typed
  character landed in the composer and was deleted). No key navigated back to
  Console by hand.

## Captures

| File prefix | What it shows |
|---|---|
| `01a-card-at-rest-set-up-provider-focused` | Get started card, no provider, no server found: Tab stop 1 (Set up provider). Step 1 reads "OpenAI: Not ready · no key". The footer reads "Enter continue setup \| F1 help \| Ctrl+P palette \| Ctrl+Q quit": F6 is left out while the card blocks. |
| `01b-card-tab-write-a-note-focused` | Same card after **Tab**: Tab stop 2 (Write a note in Library); one more **Tab** wraps to Set up provider. |
| `01c-card-footer-after-alt-m-ctrl-o-f6` | The card after **Alt+M**, **Ctrl+O** and **F6**: none of them did anything (no dialog, no notice, focus still on Set up provider; text and colours equal `01a`). The footer names Enter continue setup, F1 help and Ctrl+P palette, and the app-wide cluster only Ctrl+Q quit. |
| `02a-card-detected-local-server-focused` | Card with a loopback server found: "Use detected llama.cpp (127.0.0.1:8080)" added and focused; step 1 names the chat's own pair ("OpenAI: Not ready · no key"). |
| `02b-switch-model-from-card-filtered-to-llama-cpp` | AC#1 key 1: Switch model with Find "llama.cpp", the server's three models, the first highlighted; the key line reads "Enter apply to this chat and new chats · Tab edit values · Ctrl+O chat settings · Esc cancel" (no Ctrl+N hint, since Enter already sets the new-chat default), matching "Applies to: this chat · also the default for new chats"; the Enter button is 40 cells, its label plus one each side. The footer under the dialog reads "Ctrl+P palette \| Ctrl+Q quit" (no "Enter continue setup"). |
| `02c-console-ready-after-enter-enter` | AC#1 key 2: card gone, "Setup complete — llama.cpp · qwen3-8b-instruct-q4_k_m", "New conversation" in the Conversations tray, composer focused. "This chat updated" showed 1.5 s after the Enter and had timed out by the shot. The status row's cost chip reads "Ctx 0%" at 211x44, whole, in a shot taken before any resize (it read "Ctx 0% · No", cut mid-figure, until `e5e5855ab5`), and "Ctx 0% · Now $0.00 · Next —" at 235x52, whole. |
| `02d-composer-with-text-typed` | 211x44 only: the same Console with "What does a context window measure?" typed in the composer; the send button reads "Send \| $"; the cost chip reads "Ctx 0%" (taken after the 235x52 round trip of `02c`). Nothing was sent; the text was deleted. |
| `03a-cloud-card-set-up-provider` | AC#2 start: card on OpenAI with no key ("OpenAI: Not ready · no key"). |
| `03b-cloud-settings-api-key-focused-return-to-console` | AC#2 key 1: Settings ▸ Providers & Models on OpenAI, API key field focused, Return to Console shown on arrival. |
| `03c-cloud-key-pasted-return-without-saving` | AC#2 key 2: key pasted (masked, "edited *"), "Unsaved changes: API key."; Return to Console replaced by Return without saving. |
| `03d-cloud-saved-focus-on-return-to-console` | AC#2 keys 3-4 (**Esc**, **s**): "Credential saved. Return to Console to continue in “Chat 1”; provider acceptance is not yet verified.", focus on Return to Console. |
| `03e-cloud-key-check-verified` | AC#2 key 5 (**t**): Key check "Ready · verified"; the status line now reads "…; the provider accepted the key."; "key accepted (3 models listed)" in the inspector; focus still on Return to Console. |
| `03f-cloud-back-in-console-composer-ready` | AC#2 key 6 (**Enter**): back in "Chat 1", "Setup complete — OpenAI · gpt-5.6-terra", "Ready · verified" badge, "New conversation" in the tray, composer focused. The cost chip reads "Ctx 0% · Now $0.00" at 211x44 and "Context 0% · Current $0.00 · On next send —" at 235x52, whole at both. |
| `04a-return-without-saving-focused-key-unsaved` | A key pasted and not saved; Return without saving focused (6 **Tab**s from the API key field: Env var, Endpoint, Model, Custom ID, Model, Return without saving). |
| `04b-return-without-saving-confirmation` | **Enter** on it: "Return without saving / Discard all unsaved changes to Providers & Models and return to Console?", Keep editing focused by default. |
| `04c-discard-returns-to-card` | **Tab**, **Enter** on Discard changes: back on the Get started card, Set up provider focused, nothing saved; "Settings category changes reverted." at 211x44 (timed out by the 235x52 capture). |
| `05a-switch-model-needs-setup-row-deepseek` | Configured Console (OpenAI ready), **Alt+M**, "deepseek" typed, **Down** ×3 from Find's best match (Command Code's row) to DeepSeek's own NEEDS SETUP row; the key line reads "Enter set up in Settings" (a 26-cell button), the overflow line "… 1 more provider needs setup", the footer "Ctrl+P palette \| Ctrl+Q quit", and the cost chip "Ctx 0% · Now $0.00" (211x44) / "Context 0% · Current $0.00 · On next send —" (235x52). |
| `05b-needs-setup-route-settings-deepseek` | **Enter** on that row: Settings on DeepSeek, API key focused, model deepseek-v4-flash, Return to Console shown. |
| `05c-reopened-switch-model-after-save-deepseek` | Paste, **Esc**, **s**, **Enter** on Return to Console: Switch model reopened with Find "DeepSeek", deepseek-v4-flash highlighted and ● CURRENT, key line "Enter apply to this chat" in a 26-cell button (its label plus one each side), "… 1 more provider needs setup", footer "Ctrl+P palette \| Ctrl+Q quit". At 211x44 two notices are stacked at the bottom right: Settings' "Provider and model settings saved: …" (raised by **s**, still live) over the dialog's right edge, and "Console provider changed OpenAI → DeepSeek … (Settings ▸ Providers & Models)."; at 235x52 only the second. Both notices cover the cost chip (ruling B: notices stay at the default place); at 211x44 nothing of the chip shows past the notices' right edge (the `8b97eeb05f` shot showed a stale "0" there), and once they timed out it read "Context 0%", whole. |
| `05d-switch-model-plain-alt-m-configured` | Configured Console, plain **Alt+M**: Find counts "14 models in 1 provider"; OpenAI under READY PROVIDERS, NEEDS SETUP, and NOT RUNNING on two rows, the second "MLX-LM (Apple silicon) (not checked)" uncut; "Enter apply to this chat" in a 26-cell button, footer "Ctrl+P palette \| Ctrl+Q quit", cost chip whole at both sizes. |
| `05e-live-switch-deepseek-to-openai` | 211x44 only: after `05c`, **Enter** applied DeepSeek · deepseek-v4-flash (chip "Context 0%"), then **Alt+M**, "gpt-5.6-terra" typed, **Enter** on OpenAI's row. No resize before the shot: the cost chip reads "Ctx 0% · Now $0.00", whole, the same as after a 235x52 round trip. |
| `06a-wizard-model-step-key-required` | First-run wizard, llama.cpp against a server that answered 401 to a request that sent no key, Model step: "llama.cpp needs an API key and none was set. Go Back to add one, or enter a model ID below." in a one-line status box. |
| `06b-wizard-provider-step-test-key-required` | **Ctrl+B** back to Provider, **Tab** ×3 to Test connection, the step scrolled to its top with the mouse wheel (Authentication below the fold), **Enter** on Test connection: "✗ llama.cpp needs an API key and none was set. Add one under Authentication, or continue anyway.", with the inset pinned notice "The last connection check failed: … Add one under Authentication, or continue anyway.". The step scrolled so the whole Authentication section shows, titled "Authentication" (no "(optional)"), at both sizes; focus stays on Test connection. |
| `06c-wizard-model-step-key-refused` | First-run wizard, llama.cpp against a server that answered 401 to a request that sent a key: the endpoint set to the stub, the made-up key typed under Authentication, **Enter** on Test connection (`06d`), then **Ctrl+N** to the Model step: "Authentication failed — this API key was rejected. Go Back to fix it, or enter a model ID below." in a one-line status box, no Retry button, focus on the model name field. |
| `06d-wizard-provider-step-test-key-refused` | The Provider step after that Test: "✗ Unauthorized - the server rejected this API key. Check the Authentication section on this step.", Authentication open with the masked key and "Key staged — the server rejected it; update it above.", focus on Test connection. The section is titled "Authentication" (no "(optional)"), as in `06b`. At 211x44 the step is scrolled so the provider list's top rows are above the fold; the 235x52 shot shows the whole step. |

`02d` is new in the `4f2bffa20f` set, `05e` in the `e5e5855ab5` set, and `06c` and
`06d` at `f6db56bb4a`; every other file is retaken under its old name.

## Retaken at 8b97eeb05f

Retaken: `02b`, `02c`, `02d`, `03f`, `05a`, `05c`, `05d`, `06b` (the Switch model
key row and overflow line, the footer under a dialog, the status row's cost chip and
the wizard's Authentication scroll). Driven again and compared with the committed
files, unchanged and left as they were: `02a` and `03a` (text and colours equal),
`03b`-`03e` (only clock times differ), `05b`, and `06a` at both sizes. Not retaken:
`01a`-`01c` and `04a`-`04c` (the card with no dialog over it, and Settings with its
own footer, whose confirmation the Console's covered footer does not reach).

Seen: the status row's cost chip is cut mid-figure in `02c` at 211x44 ("Ctx 0% ·
No"). The chip is fitted to the room left by the chips before it; a model applied
from Switch model changes the Provider and Model chips' widths, and the cost chip
keeps the fit for the old widths until a resize refits it (read from the screen, not
traced: the refit seems to run before the Console, under the dialog, lays out the new
widths). Measured
live: after the 235x52 round trip the same row reads "Ctx 0%"; switching a chat from
DeepSeek · deepseek-v4-flash to OpenAI · gpt-5.6-terra left "Context 0%" with room
for "Ctx 0% · Now $0.00" until a resize. `05c`'s cost chip sits under the notices.
(Fixed by the capture fix 4 wave; see the next section.)

## Retaken at e5e5855ab5

The capture fix 4 wave changed only the status row's cost chip (it now refits after a
layout moves it). Retaken: `02c`, `03f`, `05c`; new: `05e`. Driven again and compared
with the committed files, unchanged and left as they were: `02a` and `03a` (text and
colours equal), `02b`, `02d` and `03c`-`03e` (only clock times differ), `03b`, `05a`
and `05b` (also an input's blinking cursor), and `05d` (see below). Not retaken: `01a`-`01c`,
`04a`-`04c`, `06a` and `06b` (no status row). `03f` differs from its old files only in
clock times and the composer's blinking cursor; it is retaken because the chip is its
subject.

Seen, all at 211x44 with no resize before the read:

- `02c`: "Ctx 0%", whole, in place of "Ctx 0% · No"; the same after a 235x52 round
  trip.
- `05c`: no stale "0" right of the notices; "Context 0%" once they timed out.
- Live, Switch model, DeepSeek · deepseek-v4-flash -> OpenAI · gpt-5.6-terra (`05e`):
  "Ctx 0% · Now $0.00" from the first read 0.3 s after the **Enter**, unchanged by a
  235x52 round trip. The reverse, OpenAI -> DeepSeek: "Context 0%", unchanged by a
  round trip. Read every 0.1 s, the old fit showed for the first one or two reads
  ("Context 0%" for OpenAI; "Ctx 0% · Now $0.00", whole, for DeepSeek), then the
  refit.
- `05d`'s first retake at 211x44 had one blank row under "MLX-LM (Apple silicon) (not
  checked)" in Switch model's list. It was gone after the 235x52 round trip, and two
  more fresh launches (read at 4 s and 9 s) matched the committed `05d`. That list is
  not part of this wave's change. (The capture fix 5 wave changed it; see the next
  section.)

## Retaken at 7c260226ec

The capture fix 5 wave changed only Switch model's list: each rebuild of its rows now
makes the list measure itself again. Retaken: `02b`, `05a`, `05c` and `05d`, the
captures that show Switch model. Compared with the old files, `05a` is byte-identical,
`05c` and `05d` differ only in the Find input's blinking cursor, and `02b` only in clock
times. These were driven again, compared with the committed files, found unchanged and
left as they were: `02a`, `03a`-`03d` and `05b` (text equal), `03e` (clock times),
`02c` and `03f` (clock times and the composer's blinking cursor), and `05e` (the
composer's cursor). Not retaken: `01a`-`01c`, `04a`-`04c`, `06a` and `06b` (no Switch
model).

Seen:

- `05d` on 16 fresh launches of the `05` profile: 13 opened Switch model at 211x44
  (Alt+M 0, 1, 3 or 6 s after the composer drew) and 3 at 235x52. Each was read once
  the probes had answered (2.5 to 4.5 s after the Alt+M), then at the other size, then
  back at the first size. No read had a blank row under "MLX-LM (Apple silicon) (not
  checked)", and every read's text equals the committed `05d` at its size.
- The same steps on a `git archive` of `db641a08cc`, the wave's base, 18 launches at
  211x44: no blank row there either. This live check cannot tell the two trees
  apart; the wave's test, red at the base, can.
- Read every 0.2 s or 0.04 s while the probes answer, both trees sometimes show one
  read, about 0.4 s after the Alt+M, with two blank rows under the NOT RUNNING row
  (head: 2 of 16 launches; base: 3 of 18). In that read one of the two custom
  endpoints has answered and the other has not. The same two reads, one for each
  endpoint answering first, appear on both trees, and the next read has no blank row
  (0.04 s later where the reads were that close). It is not the resting list, and the
  wave did not change it.

## Taken at f6db56bb4a

New: `06c` and `06d`, the key-sent-and-refused state (TASK-33008.5 AC#5), taken after
`efe420252d` made a Test's 401 read as a key problem after Next. Nothing else was
retaken. This one launch differed from the procedure below:

- **Launch.** `python -m tldw_chatbook` (the package entry point). `python -m
  tldw_chatbook.app` exited at startup, about 2 s after the wizard mounted (its log),
  on an unhandled `AttributeError` from `_capture_deferred_collections_setup`
  (`app_collections_wiring.py:188`): app.py defines `_CONSOLE_SKILL_APP_SOURCE` after
  its `if __name__ == "__main__":` block, so that name does not exist yet while the
  module runs as `__main__`.
- **Isolation.** A fresh scratch `HOME` and `XDG_*` directories, `TLDW_CONFIG_PATH`,
  `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` unset, and a `sandbox-exec` profile denying
  writes to the real profile directories. No profile template: the app wrote its own
  default config, so the wizard opened and the splash showed.
- **Stub.** 127.0.0.1:18432, answering 401 with a JSON error body to every request.
  Its log shows two requests, both `GET /v1/models` with an Authorization header: one
  from Test connection and one from **Ctrl+N**.

## Retaken at f54e5fa863

`7f2e67d95c` changed two lines of `06d`: the key line under a key a Test saw refused
read "Key staged — it will be checked when you continue.", and the section kept
"(optional)". Retaken: `06d` only, the same way as at `f6db56bb4a` (fresh scratch
`HOME`/`XDG_*`, the same sandbox, the stub on 127.0.0.1:18432, `python -m
tldw_chatbook`): llama.cpp, the endpoint set to the stub, a made-up key typed under
Authentication, **Enter** on Test connection. Compared with the old files,
each size differs in exactly those two lines: "▼ Authentication" and "Key staged — the
server rejected it; update it above."; the Test line and the scroll are unchanged. The
stub's log shows one request, `GET /v1/models` with an Authorization header, from
Test connection.

Checked and not retaken: `06b`, the only other Provider-step capture. Its 401 is to a
check that sent no key, so its Test line ("needs an API key and none was set") and its
"Authentication" title come from the key-required path, whose copy neither commit
changed, and no key is typed, so it has no key line. `c933bd0ad4`'s "Model listing unavailable; enter
the model ID used by this endpoint." is for a server with no model list or a Claude
subscription, and no capture shows either. `06a` and `06c` are the Model step, whose
lines are unchanged (`c933bd0ad4` moved its unavailable line into a constant, same text).

## Procedure

- **Isolation.** The app ran from this worktree (`PYTHONPATH` = the worktree) in its
  own tmux server (`-L cap33008r`) under `env -i`, with `HOME`, `XDG_CONFIG_HOME`,
  `XDG_DATA_HOME`, `XDG_CACHE_HOME` and `TLDW_CONFIG_PATH` in a fresh scratch
  directory per launch, the null keyring backend, `TERM=xterm-256color`,
  `COLORTERM=truecolor`, `LANG=en_US.UTF-8`, and a `sandbox-exec` profile denying
  reads and writes of the real profile directories (in the `7c260226ec` retake it also
  denied every outbound connection except to loopback; none of its 37 launch logs
  shows a denial).
  The `7c260226ec` retake's base launches ran a `git archive` of `db641a08cc` the same
  way.
- **Profiles.** Each one is the shipped template (`CONFIG_TOML_CONTENT`, dumped by a
  process whose own `HOME`/`XDG_*` were scratch, under the same sandbox) with
  `users_name = "verify_cap33008r_<profile>"` (`verify_cap33008r4_<profile>` in the
  `e5e5855ab5` retake, `verify_cap33008r5_<profile>` in the `7c260226ec` one), the
  splash off, `[model_catalog] auto_refresh_enabled = false` and `[first_run]
  setup_completed = true`. No provider key, except: `05` adds a made-up OpenAI key
  and `[console.onboarding] first_send_completed = true`; `06` keeps the template's first-run state, so the
  wizard opens. `05d` is a fresh launch of the `05` profile.
- **Stubs.** A loopback HTTP server answering `GET /v1/models`: on 127.0.0.1:8080
  with three model ids (qwen3-8b-instruct-q4_k_m, phi-4-mini-q8, gemma-3-4b-it-q4)
  for `02` (13 requests over the `e5e5855ab5` retake's three launches, none with an
  Authorization header; 4 over the `7c260226ec` retake's one launch, none with one);
  on 127.0.0.1:8080 answering 401 for `06` (its log shows six requests, none with an
  Authorization header; eight, none with one, in the `8b97eeb05f` retake); on
  127.0.0.1:18431 for `03`-`05`, where
  `[api_settings.openai] api_base_url` points at it so the key check (`t`) lists
  models without contacting OpenAI. That is why the Endpoint row reads
  `http://127.0.0.1:18431/v1` in `03` and `04`.
- **Input.** `tmux send-keys`; the key in `03c`, `04a` and `05c` was pasted with
  `tmux paste-buffer -p` (bracketed paste, one action). Keys are made-up
  placeholders and never appear in a capture.
- **Timing.** Notices time out: `05c` was taken about 2.5 s after the Enter and
  about 4-5 s after the **s**. In the `e5e5855ab5` retake `02c` and `03f` were taken
  6.5 s after the last **Enter** (an earlier `02c` at 3 s still showed "This chat
  updated"), and their 235x52 shot waited for the composer's blinking cursor to be
  drawn. For `01c` the three keys were sent 1.5 s apart; on the
  first try the "m" of Alt+M reached the card as a typed key and it showed its
  "Typing is locked…" notice (not seen again in five more tries), so `01c` is from a
  later try.
- **No key check or send reached a real provider.** No message was sent. The only
  key check went to the 18431 stub (one authenticated `GET /v1/models`; the same in
  the `e5e5855ab5` and `7c260226ec` retakes; none from the `05` launches, 28 of
  which followed `03` in the `7c260226ec` retake).
