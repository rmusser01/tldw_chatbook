# Live captures of engine presets

Raw responses from each engine preset's real API, captured by `capture.py`
and replayed offline by `Tests/LLM_Calls/test_live_capture_replay.py`
(TASK-33640). Most preset records were built from public documentation, and
their allowances stay provisional until a capture here confirms or amends
them.

## Keys

Put keys in `~/.config/tldw-live/keys.env`, outside any repository, and make
it private with `chmod 600`. Use one `NAME=value` per line, with the env var
names the Settings guide lists:

```
TOGETHER_API_KEY=...
NVIDIA_API_KEY=...
```

Environment variables with the same names take precedence. Only providers
whose key is present are captured; the rest skip.

The key is never written to a fixture or printed. Before committing, check
that `git diff --cached | grep -c '<first 8 chars of a key>'` prints `0`. If
a key was ever pasted into a chat or a log, rotate it afterwards.

Some providers need more than a key. `<KEY>` below is the preset key in upper
case, for example `AZURE` or `OLLAMA_CLOUD`:

| Variable | Needed for |
|---|---|
| `TLDW_LIVE_<KEY>_BASE_URL` | Azure (your resource host), Cloudflare (`https://api.cloudflare.com/client/v4/accounts/<id>/ai/v1`), Databricks (your workspace host). Its value is stored as `<per-account>`. |
| `TLDW_LIVE_<KEY>_MODEL` | Azure (your deployment name). Optional for any other provider, to pick the model. |
| `TLDW_LIVE_WANDB_PROJECT`, `TLDW_LIVE_CLOUDFLARE_GATEWAY_ID` | The optional W&B project and Cloudflare gateway headers. |

## Run

```bash
.venv/bin/python Tests/fixtures/cloud_live/capture.py --list     # who is ready, and why not
.venv/bin/python Tests/fixtures/cloud_live/capture.py            # every provider with a key
.venv/bin/python Tests/fixtures/cloud_live/capture.py nvidia kilo
```

Each provider takes up to four small requests: the model listing, a plain
chat, a tool call and a stream, at about 200 output tokens each. The script
prints the chosen model, the HTTP status of each round, and any response key
names that the record's allowances do not cover. It never prints response
text.

## Replay and amend

```bash
.venv/bin/python -m pytest Tests/LLM_Calls/test_live_capture_replay.py -q
```

A capture that does not parse fails with the key names its record does not
allow. Amend the record's allowances in `tldw_chatbook/provider_registry.py`
for exactly those keys, and cite the fixture in the registry comment. Then
rerun the replay. Rounds the provider refused (non-200) stay in the fixture
as evidence of the error shape, but they are not replayed.

Review a fixture before committing it. It holds model output (a short "ok"
and one tool call) and, for per-account providers, a redacted URL.
