# Phase 5 (TASK-33005) full-screen captures (2026-10-02)

Every changed Phase 5 surface, captured at 211x44 and at 235x52 at branch head `ad6d39998d`. Each moment was captured at 211x44. The tmux window was then resized to 235x52, captured again, and resized back. `.txt` is `tmux capture-pane -p`. `.ansi.txt` is the same moment with `-e` colour escapes.

## Setup

- The real app ran from this worktree in an isolated tmux server (`-L cap33005`) under `env -i`. `HOME`, `XDG_*` and `TLDW_CONFIG_PATH` pointed into a scratch directory, and the keyring backend was the null one.
- There were two scratch profiles. Profile A (`users_name = "verify_cap33005_a"`) set `first_send_completed = true`. Profile B (`verify_cap33005_b`) set it to false, for the Get started card.
- Both profiles set `[first_run] setup_completed = true`, `[model_catalog] auto_refresh_enabled = false` and the splash off.
- Both profiles saved llama.cpp at `http://127.0.0.1:9199` as the chat default (model `model-a`), Ollama at `http://127.0.0.1:11435`, and a fake OpenAI key.
- The real `~/.config/tldw_cli/config.toml` (sha256 prefix 15c6cb224a6a51c7) and the `~/.local/share/tldw_cli` name listing (db7e7faf5bff92d2) were the same before and after.
- Local servers were throwaway stdlib stand-ins that answered `GET …/models`. Their request log is `stub-request-log.txt`.
  - On 11435, an Ollama stand-in listed `qwen2.5:0.5b`.
  - On 9199, a llama.cpp stand-in listed `model-a`. It was down for captures 01-06 and 16, and up from 07 on.
  - On 9201, an OpenAI stand-in listed `gpt-4o` and `gpt-4o-mini`. It answered only for `Bearer <the saved fake key>` until 07:36, then only for another key.
- The driver and stand-ins are not committed.

## Captures

1. `01-console-ready-not-tested`: Console before any test. The header badge and the rail Model line read "Ready · not tested".
2. `02-settings-t-refused`: Settings ▸ Providers & Models, llama.cpp, **t**. "Readiness   Not ready · refused :9199" leads the rows, and Endpoint reads "model listing failed (connection refused)".
3. `03-settings-overview-refused`: Settings ▸ Overview right after. "Last connection test: Readiness: Not ready · refused :9199" sits under "Configuration: llama.cpp / model-a; Status: Ready": the Overview status line is still config-only, a second framing not covered by rider TASK-33005.7.
4. `04-console-refused`: Console. The badge and rail line read "Not ready · refused :9199", with the rail line in the error colour (`38;2;209;126;146` in the `.ansi.txt`). **Retry connection** appears, and the composer reads "Send blocked — retry the connection to continue".
5. `05-chat-settings-refused`: Chat settings (Ctrl+O) ran no test of its own. It reads "Not ready · refused :9199", "Endpoint · Unreachable — connection refused", "Generation · Not tested".
6. `06-switcher-local-probe`: Switch model (Alt+M), one probe per local server (log 07:29:11).
   - llama.cpp reads "Not ready · refused :9199" on CURRENT and in NEEDS SETUP.
   - Ollama reads "Ready · reachable 07:29".
   - vLLM reads "Ready · not tested": another process on this machine answers 404 at `localhost:8000`.
   - The shipped localhost defaults read "refused :PORT".
   - Cloud rows read "no key" or "not tested".
7. `07-console-reachable`: llama.cpp stand-in started, one click on **Retry connection**. The badge and rail read "Ready · reachable 07:29", and Send is unblocked.
8. `08-settings-t-key-rejected`: Settings, OpenAI (draft) on its shipped endpoint, **t**: a real `api.openai.com/v1/models` request with the fake key, answered 401. "Readiness   Not ready · key rejected", while the Key row still reads "saved in config · present, not verified". Two requests were sent, because a click on the **Test Provider** button also ran the test.
9. `09-switcher-key-rejected`: Switch model. OpenAI moves to NEEDS SETUP with "Not ready · key rejected · Enter: add key in Settings". The header (now "reachable 07:31") agrees with llama.cpp's re-probed row.
10. `10-settings-t-key-verified`: Endpoint typed (draft) as `http://127.0.0.1:9201/v1`, **t**. Readiness reads "Ready · verified 07:33" and Key "key accepted (2 models listed)". Model reads "gpt-5.6-terra (draft) · not listed by the server".
11. `11-console-verified`: after `s` (Save) the unused chat follows the default. The badge and rail read "Ready · verified 07:33".
12. `12-switcher-verified`: every OpenAI row reads "Ready · verified 07:33". Opening the switcher sent nothing to 9201 (log: no request at 07:35:41). Only the local servers were re-probed.
13. `13-console-key-rejected`: the 9201 stand-in now requires another key, Settings **t** (log 07:36:15 `auth=other`), then the Console. The badge and rail read "Not ready · key rejected", with **Set up provider**. The composer reads "Send blocked — add an API key to continue" (rider TASK-33005.7 #3).
14. `14-settings-opened-from-key-rejected`: **Set up provider** opens Settings ▸ Providers & Models at OpenAI with the API key field focused.
15. `15-settings-t-no-model-listing`: llama.cpp with the Model cleared, **t**. The listing still runs ("model listing reached"), and Readiness reads "Not ready · no model". The draft was then discarded.
16. `16-console-get-started-refused`: profile B, Alt+M (probe) then Esc. The Get started card's step 1 reads "Reconnect the provider server" with "Not ready · refused :9199" and **Retry connection**.
17. `17-console-after-card-retry`: llama.cpp stand-in started, **Retry connection** on the card. The card closes, and the badge reads "Ready · reachable 07:40". The transcript still says "Ready — type a message to begin." (rider TASK-33005.7 #4).

Log note: Ollama 07:28:42 is a `curl` check before the app started. On 9201, 07:32:37 is a click on the **Test Provider** button and 07:32:38 is **t**, both for capture 10. 07:33:06 (**t**) and 07:33:27 (the button) are re-checks before capture 10 was taken. Save, the Console and the switcher sent nothing to 9201.

Reading note: "Agent blocked" in the status strip is the Library access policy chip, unrelated to provider readiness.

## Addendum: fix wave (`fix/`, 2026-10-02)

The three surfaces the checkpoint fix changed, re-captured at 211x44 and 235x52 with the same procedure: tmux socket `-L cap33005fix`, `env -i` with `HOME`/`XDG_*`/`TLDW_CONFIG_PATH` in a scratch directory, the null keyring, `[first_run] setup_completed = true`, `[console.onboarding] first_send_completed = true`, `[model_catalog] auto_refresh_enabled = false`, splash off. The real `~/.config/tldw_cli/config.toml` (sha256 prefix 15c6cb224a6a51c7) and the `~/.local/share/tldw_cli` name listing (db7e7faf5bff92d2) were the same before and after. Stand-ins (not committed): an Ollama listing `qwen2.5:0.5b` on 11435, and on 9201 an OpenAI-compatible server that answers 401 to every key. Nothing listened on 2242, 8080, 1234, 5678 or 9199; another process answers 404 on `localhost:8000`. `fix/stub-request-log.txt` is their log. No real provider was contacted.

Profile A (`verify_cap33005fix_a`): llama.cpp saved at `http://127.0.0.1:9199` as the chat default (model `model-a`), Ollama saved at `http://127.0.0.1:11435`, a fake OpenAI key. Every other provider keeps the shipped template's table.

1. `fix/01-switcher-untouched-defaults-not-running`: Console, Alt+M, about 7 s. llama.cpp (configured: the chat default, endpoint edited) reads "Not ready · refused :9199" on CURRENT and leads NEEDS SETUP with "start it; rechecked on open". Ollama reads "Ready · reachable 08:24" (log 08:24:49). Aphrodite Engine, TabbyAPI and both custom slots were probed, refused, and are listed once under "NOT RUNNING · local servers never set up · rechecked on open", not as "refused :PORT" rows (TASK-33005.6 AC#3). NEEDS SETUP's overflow line now reads "… 39 more" (43 in capture 06). The header badge agrees: "Not ready · refused :9199".
2. `fix/02-settings-overview-refused`: Esc, F4, Providers & Models, **t**, then Overview. "Configuration: llama.cpp / model-a; Status: Not ready · refused :9199" sits directly above "Last connection test: Readiness: Not ready · refused :9199" (capture 03 read "Status: Ready" there). The refusal evidence is shared: the switcher's probe in capture 1 had already recorded it.

Profile B (`verify_cap33005fix_b`): OpenAI as the chat default (model `gpt-4o`) with the fake key and `api_base_url = "http://127.0.0.1:9201/v1"` saved.

3. `fix/03-settings-t-key-rejected-key-row`: F4, Providers & Models (OpenAI), **t**: one listing request with the key (log 08:26:55), answered 401. "Readiness   Not ready · key rejected" and "Key   saved in config · key rejected" (capture 08 read "present, not verified" there).
