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

**A second incident, same rule (task-842).** `NLTK_DATA` *appends* to `nltk.data.path`;
it does not replace it. A run that set `NLTK_DATA` to an empty directory therefore kept
finding the machine's real corpus at `~/nltk_data`, exercised the working tokeniser, and
was reported as proof that the *fallback* worked — a fallback that had never run. That
false positive reached a PR description. Isolation held only after
`nltk.data.path[:] = [scratch]`, and the very first honest run failed immediately.

The general form: **an env var that adds a search path is not isolation.** Before
trusting a negative-condition run, confirm the resource is genuinely unreachable — the
setup should fail the way the real broken environment fails.

---

## A bare interpreter call is not an isolated test

**What happened (TASK-1264/Task 5, first-run wizard skeleton).** After the
required Pilot tests passed, extra ad hoc verification of the real
Next-button click path was run via `python3 -c "..."` directly against the
venv interpreter — not through `pytest`. `Tests/conftest.py`'s
`isolate_test_environment` autouse fixture (which redirects `HOME`,
`XDG_CONFIG_HOME`, `XDG_DATA_HOME`, `TLDW_CONFIG_PATH` to a temp directory)
only runs inside a pytest session; a bare script gets none of it. The
wizard's `on_mount()` fired a `@work(thread=True)` worker that called the
real `save_settings_to_cli_config()`, which wrote
`[first_run]\nsetup_started = true` into the actual
`~/.config/tldw_cli/config.toml` — a real user's live config file, mutated
by a "just checking" verification script. It was caught immediately from
the write in the log output, diffed, and the exact three added lines were
removed to restore the file; no other content had changed.

**What to do.** Before running any ad hoc script (not just `pytest`) against
code that can reach `save_settings_to_cli_config()` or any other real I/O
path, set the same isolation env vars the test suite's autouse fixture
uses — `TLDW_TEST_MODE=1`, `XDG_DATA_HOME`, `XDG_CONFIG_HOME`, `HOME`,
`TLDW_CONFIG_PATH`, all pointed at a scratch directory — *before* importing
anything from `tldw_chatbook`. "It's just a quick check outside the test
suite" is exactly when the suite's isolation fixture is absent.

---

## Credentials in a live run

A live credential pasted into a session is a real secret. Keep it in an env var for the
run; never write it to a config file that could be committed; and before committing,
confirm `git diff | grep -c "<key-fragment>"` is `0`. Advise rotation afterwards.

---

## SGR click columns from `awk index()` are byte offsets, not terminal columns

**What happened.** A Watchlists UAT drove the TUI with injected SGR mouse clicks,
computing each target column with `awk '{print index($0, "Items")}'` over
`tmux capture-pane` output. Clicking "Items" activated "Runs"; clicking
"New Source" toggled "Filters"; a modal's "Create" button ignored clicks that
`Enter` accepted. All three were filed or half-filed as hit-region defects.

None of them were real. `awk`'s `index()` counts **bytes**. Every line of this
app contains box-drawing and arrow glyphs — `▊`, `▔`, `▼`, `╭` — which are three
bytes each in UTF-8, so the byte offset runs ahead of the true column by three
per glyph already on the line. On one measured row:

```
New Source   char-col=169   byte-offset=181
Filters      char-col=186   byte-offset=198
```

A click computed at 185 "inside New Source" actually lands on Filters. The
error grows left-to-right, so the further right a control sits, the more
confidently you will click the wrong one.

**What to do.** Compute the column by **character** position, not bytes:

```python
line = capture.splitlines()[row - 1]
col = line.find(label) + 1          # 1-based, character-accurate
```

Then verify the click did what you expected before concluding anything about
the app. A click that activates the *neighbouring* control is the signature of
this bug, not of a broken hit region — the app was fine every time.

---

## A terminal capture is not evidence about what rendered

**TASK-1210, 2026-07-27.** A new cadence dropdown on the Watchlists create form
appeared, in `tmux capture-pane`, to open with **no options at all** — just an
empty bordered box:

```
┌──────────────┐
│▊▔▔▔▔▔▔▔▔▔▔▔▔▎│
│▊  Every  ▼  ▎│
```

The pre-existing `All statuses` Select in the same pane looked identical, at
both 160x42 and 235x52. Two controls, two terminal sizes, same symptom: it read
as a screen-wide defect, and the next step was going to be replacing the Select
with a cycling Button to route around it.

It was not real. `Screen._compositor.render_strips()` shows all four options
painted:

```
PAINTROW 37: │  Every 15m   │
PAINTROW 38: │  Every 1h    │
PAINTROW 39: │  Every 6h    │
PAINTROW 40: │  Every 24h   │
```

**Widget state does not settle this either.** `select.region`, `overlay.visible`
and `overlay.option_count` are all *pre-paint* facts — they describe what the
layout engine decided, not what reached the screen. They were correct here, but
they would have been correct for a genuinely clipped overlay too. Only the
compositor answers the question actually being asked.

**What to do.** When a live capture suggests something did not render, confirm
with `Screen._compositor.render_strips()` before believing it, and certainly
before redesigning around it:

```python
strips = screen._compositor.render_strips()
row_text = "".join(seg.text for seg in strips[y])
```

This is the **fourth** capture-harness artifact on this programme — after the
byte-offset click bug above, a stale screenshot that produced a whole spec, and
a defect filed against a control that was working. The pattern is consistent
enough to state as a rule: **the harness is wrong more often than the app is.**
Terminal art is a hint about where to look, never a finding.

---

## `pilot.click()` can silently miss and no one tells you

**TASK-1264/Task 12, 2026-07-29.** An app-level Pilot test drove the
first-run wizard's Summary step to completion after a burst of rapid,
unsettled Back/Next clicks, then clicked its "Start chatting" exit button.
The click reported no error, but the wizard never dismissed — `current_step`
was unchanged 0.3s and then 10s later. `container._advancing` was `False`
and `can_proceed` was `True` the whole time, so nothing about the
container's own state explained it.

`Button.query_one(...).region` reported a plausible, non-empty, on-screen
rectangle, and `button.visible` was `True` — every pre-paint fact available
said the click should have landed. It was resolved only by asking the
compositor directly what widget actually owns that pixel:

```python
widget_at, _ = app.get_widget_at(*button.region.center)
assert widget_at is button  # was SummaryStep (the parent), not the Button
```

`pilot.click(selector)` computes its target coordinate from the selected
widget's own **cached** `region` attribute, then dispatches a mouse event at
that screen position — it does **not** verify the region still matches
reality. `Pilot.click()`'s own docstring says exactly this and is easy to
miss: it returns `True` only "if ... the selected widget was under the
mouse when the click was initiated" and `False` otherwise, with **no
exception** either way. A test that does not check the return value (every
test in this suite up to this point did not) cannot tell "clicked
successfully" from "silently missed."

**What to do.** For any Pilot test that drives a control through a **state
machine** (a wizard step, not a one-shot visual check), prefer driving the
widget directly over a pixel-coordinate click: `Button.press()` posts the
identical `Button.Pressed` message a click ultimately posts, and setting
`RadioButton.value = True` posts the identical `Changed` message a click's
`toggle()` would. Both still honor `disabled`/`display` correctly, so they
do not mask a genuinely un-interactable control — they only remove the
irrelevant risk of stale-region hit-testing. Reserve pixel-coordinate
`pilot.click()` for tests that are actually verifying the click surface
itself (hit-region size/placement, obscured-widget detection); check its
return value there, and cross-check with `app.get_widget_at()` or
`Screen._compositor.render_strips()` (see the entry above) rather than
trusting `region` alone.

---

## Related

- `lessons-testing-evidence.md` — why the green suite was not evidence

---

## When live behavior diverges from your probes on the same machine, diff the INTERPRETERS first (2026-07-31)

**Incident.** Four live-gate rounds of Console voice commands failed with "broken" /
"nothing", while every probe — headless controller, full-app harness, real-microphone
rig — passed on the same machine, same branch, same config. Three real defects were
found and fixed along the way (loop starvation, slow model, missing feedback), but the
final "nothing." had a one-line cause: the user launches with `python3`, which resolves
to `/usr/bin/python3` (system Python 3.9.6) — it has enough of the stack to run the app
and record audio (textual, sounddevice, faster-whisper) but no `webrtcvad`, so the
recorder delivers every frame, the silence gate never fires, and no segment, final, or
command can ever exist. Every probe ran on the repo `.venv` (Python 3.12, full stack).
Two interpreters, one machine, opposite behavior — and the degraded-VAD toast that
should have flagged it claimed "dictation still works; commands execute when you stop",
which was no longer true under the current architecture.

**What to do.** The first question when a user's live run contradicts your own live
verification is "which interpreter/environment ran it?" — `which python3` costs one
second and would have cut three rounds of debugging to zero. Give run instructions with
the ABSOLUTE venv python path, never bare `python3`. And when a feature degrades
without a dependency, the warning must state the actual current consequences — copy
written for an old architecture ("commands execute when you stop") becomes actively
misleading after a rework and nobody re-reads it unless a review targets it.
