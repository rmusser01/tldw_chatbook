# Lessons: verifying against the real thing

Traps found by running the app and talking to a real server, which the test suite
structurally could not surface. Every entry states the incident that produced it.

---

## The suite cannot see a contract you guessed

**What happened.** The Library ingest UAT (tasks 673–702) found **seven** defects
through live contact that a green suite never would:

| Defect | Why tests missed it |
|---|---|
| Runtime policy refuses `…launch.server` outside server mode | 941 unit tests passed; direct API probes bypass the app's enforcer |
| Server accepts only `video\|audio\|document\|pdf\|ebook` | The accepted set is **not in its OpenAPI spec** — `media_type` is a bare string with a runtime validator |
| `result` typed as another domain's model | Only a *completed* job triggers it; cancel-only testing never does |
| `offset` never sent | The fake declared it |
| `/ingest-web-content` needs a lowercase `token` header | Not exercised by any test |
| YouTube URL grouped as an unsupported *file* | Two classifiers disagreed; nothing compared them |
| Pre-flight vetoed pages the backend could fetch | Its own 403 became the user's error |

**What to do.** Before building on a remote contract, **ask the service**, not the
spec. Submit one real request. A spec that types a field as `string` may still have a
runtime validator behind it.

---

## This server reports denials as 429, not 401

**What happened.** `token` alone on `/ingest-web-content` returned
`429 {"error": "rate_limited"}` with `retry_after: 1`. That reads as throttling, so the
obvious response is to back off — and backing off never helps, because it is a
**denial**. ~80 seconds were lost to retries before a control request against a
route known to accept `X-API-KEY` returned 200 and distinguished the two.

**What to do.** On a 429 from this server, send a **control request** to a route you
know works. If that succeeds, you are being denied, not throttled. Unauthenticated
calls to media routes return 429 `policy_id: media.access` — not 401.

---

## Verify at the surface the user touches

**What happened.** A feature was fully unit-tested and unusable: the switch was
offered, the user could select it, and every submission failed with
*"requires server mode"*. Only driving the real TUI exposed it. Separately, an action
button rendered *above* its own status line, so the screen read as a contradiction
top-to-bottom — no unit test would ever say so.

**What to do.** For anything user-facing, run the app and look at it. Ask: does the
screen read correctly top-to-bottom, and does the affordance actually lead somewhere?

Headless recipe (no repo tooling required):

```bash
tmux -L verify new-session -d -x 235 -y 52 '.venv/bin/python -m tldw_chatbook.app'
sleep 12                                    # cold start is ~10s, import-dominated
tmux -L verify capture-pane -p | head -8    # pane as text
tmux -L verify send-keys C-p                # command palette
tmux -L verify kill-server                  # done
```

Use `TLDW_CONFIG_PATH=<scratch>/config.toml` so the run cannot touch real state (see
the profile-isolation entry below). Ctrl+digit hotkeys cannot be sent through tmux --
verify those bindings by reading `BINDINGS` in the code instead.

---

## "It resolves" is not "it resolves the right thing"

**What happened.** Verifying an "open the item the server created" action, the fetch
returned without error but with `title=None`. Accepting that would have been a false
positive. Inspecting the payload confirmed a real `MediaDetailResponse(media_id=1125)`
— the exact row the ingest reported.

**What to do.** Assert on **identifying content**, not on the absence of an exception.
The chain that matters is: real submit → real completion → real reconcile → the
affordance → *the thing it opens*.

---

## Isolate the profile before touching runtime state

**What happened.** The runtime-policy state file (which records local vs. server mode)
resolved from a hardcoded home path, ignoring `TLDW_CONFIG_PATH`. Putting a *scratch*
profile into server mode would have left the **real** profile in server mode — and the
self-healing demote would not fire, because a server was configured. This blocked
server-mode verification entirely until fixed (task-701).

**What to do.** Launch with `TLDW_CONFIG_PATH=<scratch>/config.toml`, then **verify the
isolation actually held**: back up the real file, and diff it afterwards. Never assume
an env var isolates everything — check which paths derive from it.

---

## Credentials in a live run

A live credential pasted into a session is a real secret. Keep it in an env var for the
run; never write it to a config file that could be committed; and before committing,
confirm `git diff | grep -c "<key-fragment>"` is `0`. Advise rotation afterwards.

---

## Related

- `lessons-testing-evidence.md` — why the green suite was not evidence
