# Lessons: verifying against the real thing

Traps found by running the app and talking to a real server, which the test suite
structurally could not surface. Every entry states the incident that produced it.

---

## Shell width equality does not prove pane containment

**TASK-18919, 2026-09-01.** The Collections reader's production-shaped 120×35 and
100×30 walkthroughs proved that every shell child width added up exactly to the
shell width, yet visible controls still escaped their panes. The Items toolbar
put Quick Capture, Filters, and Sort on one row; the Work toolbar did the same
with four primary actions. Both parents were geometrically correct while their
own descendants overflowed.

**What to do.** At every compact breakpoint, check every visible descendant's
left and right edge against its owning pane, not only the top-level shell sum.
Split action groups into additional rows when the controls are semantically
distinct; do not shrink labels until the full control hierarchy has been
measured.

---

## A manually pinned tmux window can make a correct TUI look as if its right rail vanished

**TASK-20937.6, 2026-08-24.** A Console QA session was created at 235x52 and
then forced to `window-size manual` before an iTerm2 operator attached. The
server-side `tmux capture-pane` showed the complete `<-Inspect` handle at the
right application edge. Both operator screenshots omitted it, and were initially
misclassified as partial acceptance evidence. The screenshots also truncated the
header's Hands-free control and tmux's own right-side date/status text, proving
that the whole tmux canvas—not only the Inspector—extended past the client
viewport. The fixed manual canvas was wider than the actual iTerm2 client, so the
client clipped its rightmost columns.

**What to do.** For operator captures, use tmux `window-size latest` (or the
normal client-following policy), then record the resulting client/window/pane
cells after attachment. A detached `capture-pane` proves the server buffer, not
that the terminal client can see all of it. Reject any screenshot whose terminal
status line or app header is cut at the same edge as the feature under test; do
not diagnose product geometry until the client and tmux window sizes agree.

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

## A route-activated geometry test does not prove the untouched startup state

**TASK-22857, 2026-08-26.** The production-styled Library width matrix passed at
235, 170, 120, 100, 80, and 60 columns, but its ordinary-route setup selected
Prompts before measuring geometry. Detached tmux UAT entered a fresh Library
landing without selecting a row and found that the legacy graduated landing
clause still forced rail-only mode at 100 and 80 columns. The selected-route
matrix had changed the exact state that triggered the defect. The same live run
also found that the visible emergency return could not be reached by keyboard
and that Settings reported ASCII glyphs Disabled while the runtime rendered the
saved ASCII form; mounted tests had verified the underlying actions and config
values without exercising those complete user journeys.

**What to do.** For responsive shells, verify at least three distinct lifecycle
entries: untouched startup/landing, route activation, and resize restoration.
Do not let a helper select a route before the startup assertion. For a visible
keyboard affordance, drive focus to it through the advertised production keys
and activate it; calling its action directly proves behavior, not reachability.
When Settings owns a runtime preference, restart a scratch-profile process and
compare the visible Settings value with the behavior rendered by that same
process. Keep the pre-fix capture as failed evidence instead of replacing it
with the post-fix pass.

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

**A third incident, same rule (TASK-15482).** A post-run payload-invariant probe
set a scratch `TLDW_CONFIG_PATH` before importing the visual evaluator. The import
created the scratch config as intended, but `load_settings()` still logged that it
ensured `chat_dicts` beneath the normal `~/.local/share/tldw_cli/default_user`
profile. The directory already existed and before/after fingerprints proved no
file changed, but the log exposed that the probe was not actually data-isolated.

**What to do.** For an ad hoc run that imports application config, create the
scratch config before the import and set its `[paths].data_dir` to a scratch
directory. `TLDW_CONFIG_PATH` controls the config file only; it does not relocate
data paths selected from that config. Keep before/after fingerprints as the final
backstop because path isolation and proof of non-mutation are separate claims.

**A fourth incident, same rule (TASK-3401.14 / TASK-15674).** After a long real
Textual UAT session configured with a disposable profile, the unrelated default
config's post-run byte fingerprint differed from its validated pre-run snapshot.
The byte delta consisted of built-in default keys appearing while existing values
remained unchanged.
Restoring that exact snapshot before scratch cleanup was the right containment
action. It was not, however, proof that the isolated app lifecycle caused the
mutation: unrelated concurrent activity existed, so the fingerprint established
that bytes changed, not which actor changed them.

TASK-15674 tested attribution under controlled current-development conditions. It
used a disposable `HOME`, the relevant XDG config, data, and cache directories, an
effective scratch `TLDW_CONFIG_PATH`, a distinct decoy default config, a scratch
`[paths].data_dir`,
and disabled model-catalog networking. Through the real mounted app's
startup-to-approved-quit lifecycle, persistence ran and selected only the exact
effective profile path; the decoy remained byte-identical.

**What to do.** Keep a validated recovery snapshot and compare before deleting it;
restore on unexplained drift. To identify the writer, separately reproduce with a
distinct decoy default and effective profile under an isolated lifecycle. A
fingerprint difference proves mutation, not actor identity. Track sensitive
unexplained mutation as an investigation or provisional defect, but do not label a
confirmed cross-profile writer or actor until causality is demonstrated.

---

## A configured data root is not necessarily the database's final parent (TASK-22453, 2026-08-27)

**What happened.** Roleplay pagination UAT set `[paths].data_dir` to a disposable
root and seeded `tldw_chatbook_ChaChaNotes.db` directly beneath it. The app opened
`<data_dir>/default_user/tldw_chatbook_ChaChaNotes.db` instead, populated its normal
starter characters, and honestly reported no saved conversations for the selected
card. Both databases were scratch-only, so the isolation boundary held, but the
fixture targeted a plausible path rather than the production-resolved path and
produced a false product failure.

**What to do.** Before seeding a live scratch database, launch once or call the
same profile-aware path resolver production uses, then verify the exact open file
with `lsof` (or an equivalent read-only probe). A configured root proves the
storage boundary; it does not prove the final filename or profile namespace.

---

## A schema bump is a one-way door for every OTHER worktree on this machine (2026-08-04)

**What happened.** task-2364 added `messages.metadata_json` and moved
ChaChaNotes from v30 to v31. Every checkout on this machine shares ONE real
database (`~/.local/share/tldw_cli/`), and `CharactersRAGDB._initialize_schema`
refuses — by design — to open a database whose version is newer than the code:

```
SchemaError: Database schema 'rag_char_chat_schema' version (31) is newer than
supported by code (30). Aborting.
```

So a single live launch of a schema-bumping branch migrates the shared database
and every concurrent worktree still on the old version stops opening it. There
is no downgrade path; the fix is restoring a backup or moving every other branch
forward. The migration itself is additive and idempotent, which is exactly what
makes this easy to under-rate: the DANGER is not a broken migration, it is a
correct one applied to a database other work is still reading.

**What to do.** While a schema bump is in flight, verify it with in-memory or
`tmp_path` databases only — `CharactersRAGDB(":memory:")` and the seeded-old-version
pattern in `Tests/DB/test_chachanotes_*_migration.py` cover the migration, the
version bump, the sync-trigger exclusion and the idempotence guard without
touching a real file. Do not launch the app. `TLDW_CONFIG_PATH` alone does NOT
protect you here: it redirects the config FILE only, and the database path comes
from `[paths] data_dir` *inside* that file (`config.get_user_data_dir`), so a
scratch config that omits `data_dir` still opens the real
`tldw_chatbook_ChaChaNotes.db` — the same shape as the profile-isolation lesson
above. Live verification of a schema-bumping branch waits until the branch is
the one everything else is on.

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

## Redirecting the TUI's stderr blanks the pane — the app renders there, and an empty `capture-pane` looks like a hung app (TASK-15700, 2026-08-13)

**What happened.** A live check launched the app in tmux with the usual
recipe plus `2>$SCRATCH/app.log`, to keep the log out of the pane.
`capture-pane -p` then returned **52 blank lines** while the process sat at
~10% CPU — which reads exactly like "the app hung during startup". Two
launches and a round of log-archaeology were spent on that: the app's own
file log ended at `scheduler_configured`, and the redirected `app.log` turned
out to contain 16,537 lines of **ANSI render output** — the splash screen's
starfield. The render goes to **stderr**; sending stderr to a file (or
`/dev/null`) leaves the pane genuinely empty and there is nothing to capture.
Relaunching with no redirect at all painted the nav bar immediately.

**What to do.** Launch with stderr left attached to the pane. If a log is in
the way, read the app's own file sink under the profile's data directory
(`<data_dir>/<user>/tldw_cli_app.log`) instead of redirecting. And before
concluding a launch hung, check that a *process* is alive AND that something
was written where the render should be going — an empty pane plus a live PID
is more often a redirect than a hang.

---

## A green process test does not prove multiprocessing starts under Textual (TASK-18926, 2026-08-27)

**What happened.** The raw CLI executor's focused process suite passed, including
real spawned shells and process-tree cleanup. The first mounted Console run still
failed every command before launch. Textual had replaced `sys.stderr` with a
capture whose `fileno()` returns `-1`; the first spawn-context `Event()` started
CPython's process-global resource tracker, which blindly passed that sentinel to
`fork_exec` and raised `ValueError: bad value(s) in fds_to_keep`. The controller
correctly rendered its generic local `spawn_failed` marker, so only capturing the
underlying executor exception revealed the cause. Library ingest had already met
the same CPython/Textual interaction, but the separately built raw executor had
not inherited that launch guard.

**What to do.** Any feature that first constructs spawn-context primitives from a
mounted Textual worker must construct them under an fd-backed stderr (and serialize
the brief redirect because `sys.stderr` is process-global). Keep a fresh-process
regression with `fileno() == -1`; an ordinary executor test may reuse an already
running resource tracker and cannot prove first-launch behavior. Finish with a
mounted command that reaches the actual subprocess, not only a synthetic executor.

## A spawned worker replays package initializers in a fresh import order (TASK-22510, 2026-08-28)

**What happened.** The model raw-shell suites passed because their pytest process
had already imported Chatbook's Chat package. The first real command from a mounted
Console still failed inside the spawned worker before shell launch. In the fresh
interpreter, `raw_cli_executor` imported the general `input_validation` module;
that module eagerly imported `Chat.console_chat_models`, which executed the eager
`Chat/__init__.py` and `Library/__init__.py` chains. Library then imported
`sanitize_string` from the still-partially initialized validation module and raised
a circular-import `ImportError`. Preloading Chat made the same executor import pass,
which is why ordinary in-process coverage hid the defect.

**What to do.** Treat every spawn target's transitive imports as a fresh-interpreter
contract. Keep a subprocess regression that imports the target with the repository
root explicitly first on `sys.path`, and verify one mounted first launch as well as
the in-process suite. General boundary modules should not eagerly import feature
packages with nontrivial `__init__.py` files; defer the feature-specific dependency
behind the narrow function that needs it while preserving any established patch
seam.

## Credentials in a live run

A live credential pasted into a session is a real secret. Keep it in an env var for the
run; never write it to a config file that could be committed; and before committing,
confirm `git diff | grep -c "<key-fragment>"` is `0`. Advise rotation afterwards.

---

## Validate the multimodal fixture before blaming the request path (2026-08-21)

**What happened.** Console AGENTS.md native-provider UAT reached OpenAI with the
expected model, message roles, native tool schemas, and redacted credential logging,
but the first request returned HTTP 400 before any model output. The application
surface intentionally converted that into content-free provider failure copy, so the
response body was not available there. A minimal authenticated control request using
the same inline one-pixel PNG exposed the actual provider error:
`image_parse_error` / "unsupported image." Replacing only that fixture with a
synthetic 32x32 checkerboard PNG made the control return HTTP 200 and the unchanged
full native tool UAT pass. The original feature path was correct; the supposedly
convenient test image was not accepted by the real provider.

**What to do.** Before diagnosing a multimodal live failure as an adapter or message
conversion defect, send the exact image through a minimal provider control request.
Use a synthetic scratch image rather than repository/user content, require an HTTP
success from that control, and only then add tools, project context, and persistence
assertions. Tiny base64 fixtures that satisfy local shape tests are not proof that a
provider will decode them.

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

`awk index()` is not the only source: `grep -bo` and `wc -c` are byte counters too,
and a 2026-08-03 Library ingest round re-derived this same bug through them —
mis-clicking a Clear button by ~12 columns and filing it as "dead to mouse", then
spending a diagnosis round proving the button was fine.

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

---

## Scratch-profile live launches: copy `chromadb/` too, expect a config rewrite, and the real provider lever is `[API] default_api` (PR-3 Task 8, 2026-08-03)

**Incident.** Live-verifying the Library's honest RAG answering (PR-3 Task 8) needed a
scratch profile pointed at a copy of the real Library DBs, with a real provider
configured, launched via `TLDW_CONFIG_PATH`. Three things about that recipe were wrong
on the first attempt, each costing a full relaunch to diagnose.

**1. The DB-copy recipe must copy `chromadb/` too, BEFORE first launch.** The plan
copied only the SQLite DB files. The app auto-creates an empty `chromadb/` directory
the moment it boots against a scratch profile with none present. Copying the real
`chromadb/` directory in AFTER that first boot nests it under the auto-created one
instead of replacing it, so embeddings resolve against the empty nested copy and
retrieval looks broken for a reason that has nothing to do with the code under test.
This is exactly what happened on the first attempt. Copy `chromadb/` in the same
pre-launch pass as the DB files, never after.

**2. The app rewrites its scratch config on boot.** A scratch `config.toml` written by
hand with an `[API]` table already present gets a SECOND `[API]` table appended by the
app's own startup config-write path. A duplicate table is enough to make the first-run
wizard reappear on the next launch, overriding the scratch config's intent (skip
first-run, use a real provider). Do not hand-author a config and trust it to survive
untouched — expect the app to rewrite it, and diff the file AFTER first boot, not just
before.

**3. The real-provider lever is the legacy `[API] default_api` key, not
`[llm_api_settings] default_api_endpoint`.** The verification plan pointed a scratch
config at the latter, which the running app never reads for this purpose, so the
provider stayed unconfigured despite the file looking correct.
`resolve_library_rag_answer_provider()` (`Library/library_rag_answer_service.py`) reads
`config.default_api_endpoint`, which `config.py` resolves from `[API] default_api` — the
legacy key, still the one a live launch actually honors.

**What to do.** For any scratch-profile live TUI verification against real Library data
and a real provider: copy `chromadb/` alongside the SQLite DBs in the SAME pre-launch
pass, before the scratch profile is ever launched; treat the scratch `config.toml` as
something the app WILL rewrite on boot and diff it AFTER first launch rather than
trusting the hand-authored version; and set the provider via `[API] default_api`, not
`[llm_api_settings] default_api_endpoint`.

---

## A control that moves when the form changes produces phantom "dead click" bugs (2026-08-04)

**What happened.** Two consecutive Library ingest critiques reported an intermittent
**dead Start button**: a click at the commit moment that produced no job, no toast, and
no queue row, while the identical click worked moments later. It was filed as a
suspected event-handling defect and cost a round of investigation plus a regression
test that could never fail, because the harness reproduces nothing.

The mechanism turned out to be geometric. Typing a valid path replaces the gate line
("Enter a file path to start.") with a forecast line and a commit summary — **the Start
button moves down three rows**. A driver that reads the button's coordinates, types a
path, and then clicks is clicking where the button *was*. The evidence arm hit the same
trap in its own probe run, re-located the button in a fresh capture, and the first
click submitted in 0.18s.

**What to do.** Re-locate every control **in the same capture you click from**, never
from one taken before the last state-changing keystroke:

```bash
ROW=$(tmux -L "$SOCK" capture-pane -p | grep -n "Start ingest" | head -1 | cut -d: -f1)
COL=$(tmux -L "$SOCK" capture-pane -p | sed -n "${ROW}p" \
      | python3 -c "import sys; print(sys.stdin.readline().find('Start ingest')+5)")
```

And treat "the same click works sometimes" as a **layout-shift** signature, not a
hit-region defect — the two look identical from outside and only one of them is real.

---

## Critique scores from different agent instances are not like-for-like (2026-08-04)

**What happened.** Seven rounds of dual-agent design critique on one surface scored
21 → 24 → 29 → 25 → 26 → 31 → 22 out of 40. Read as a time series, round 7 looks like a
nine-point collapse immediately after four approved improvements shipped. It was the
opposite: the round-7 reviewer drove paths no previous round had touched (typo paths,
404 URLs, Retry behaviour, measured WCAG contrast, cross-run counter reconciliation)
and graded harder — while the same round's **mechanical arm ran 14 probes against every
shipped behaviour and all 14 passed**, the first fully clean run of the arc. The dips at
rounds 4 and 7 were both new coverage; the two genuine regressions in the whole arc were
each caught by mechanical probes, not by the score.

**What to do.** Treat the score as a **prompt for reading the findings**, never as the
finding. When it moves, say why in the same breath, and cite the comparable signal:

- **Comparable:** deterministic probes over fixed behaviours, pass/fail, same script.
- **Not comparable:** a judgement score across agent instances, prompts, or coverage depth.

Keep a mechanical arm in every round precisely so there is something to compare when the
judgement number swings, and never report a delta without stating whether coverage
changed underneath it.

---

## A backend fix proven by a unit test can still be dead code from the only real UI entry point (2026-08-04)

**What happened (TASK-2450, voice-profiles slice 1).** Two review-and-test-passed fixes
turned out to be unreachable from the app they shipped in, and only driving the real TUI
against a real OpenAI account found either one.

1. A slice fixed `_generate_legacy`'s provenance construction so a legacy-provider TTS
   generation would carry a `TTSRequestedSelectionSnapshot`, making "Save result as
   profile" eligible — reviewed, unit-tested, green. Live-clicking Generate in the real
   Playground never made the Save button appear. The Playground pane is *always*
   constructed with a non-`None` `studio_preferences` snapshot (`UI/STTS_Window.py`), so
   every real click's request carries `studio_preferences != None` and the dispatcher
   (`_generate_tts_worker`) always routes to `_generate_studio_effective`, never to
   `_generate_legacy` — which is only reachable when `studio_preferences is None`, a state
   the live UI can never produce. `_generate_studio_effective` had its own, separate,
   untouched provenance construction that only fires for `provider_id == "audio_cpp"`.
   The fixed function was correct and completely unreachable.
2. In the same slice, the backend's assignment path (`TTSProfileService.set_assignment`,
   the character resolver) was correctly extended to accept the new provider set, and a
   profile classified `"unverified"` accordingly. The **client-side** Select widget that
   is the only way to create an assignment (`personas_character_tts_widget.py`,
   `_profile_changed`) still read `if option.availability != "available":
   self._restore_selected_value()` — written when `"available"`/`"unavailable"` was the
   whole vocabulary, never taught the new third state. Every attempt to assign a legacy
   profile silently reverted with no error. The backend was correct; the one widget that
   calls it was stale.

Both were found by clicking the real button in a real running app, not by reading either
function in isolation — a code reviewer reading `_generate_legacy`'s diff has no reason
to go looking for a sibling function with the same job, and a reviewer of the assignment
service has no reason to open an unrelated, untouched widget file three directories away.

**What to do.** When a fix changes what one function accepts, ask a second question
before calling it shippable: **is this the function the real UI actually calls on the
path a user takes?** `grep` for the dispatcher/conditional that chooses between it and
any sibling implementing the "same" behavior, and check what determines the choice at
the actual call site — not what the test harness passes. Separately, `grep` for every
manual string comparison against a status/enum field the change touches
(`!= "available"`, `== "unavailable"`, etc.) outside the files the diff itself modified;
a new state must be taught to every comparison, not just the ones in the changed files.
Live-clicking the real affordance is the cheapest thing that reliably catches both
shapes — a unit test that constructs the request/selection by hand cannot, because it
never asks who else was supposed to construct it that way.

**Addendum (2026-08-05, TASK-2453 fix): the same stale check can exist at two layers, and fixing the one you found does not prove you found all of them.** Shape (2) above (the Roleplay assignment Select silently refusing anything not exactly `"available"`) turned out to have a **second, independent copy** one layer deeper: `personas_screen.py`'s `_character_tts_assignment_worker` had its own `if tokens[1].state != "available": return`, entirely separate from the widget's own gate, with no log line and no error on the silent early return. Fixing only the widget (its own `Select.Changed` handler correctly stopped reverting the value) produced a **convincing false positive**: the Select's displayed value updated immediately, matching what a successful assignment should look like — but navigating away and back showed it had reverted, because the message posted correctly and then hit the second gate on the way to the database. The paragraph above already prescribed the right process (grep every manual comparison against the touched enum, everywhere, before calling a fix complete) — and it was not followed mechanically the first time around; the second gate was found only because live re-verification checked **persistence across a remount**, not just the widget's own immediate-post-click state. The generalizable rule, sharpened: after fixing a UI-level stale-enum-comparison bug, grep the *whole codebase* for the same literal comparison pattern (`!= "available"` was one string away from the actual second occurrence) before declaring the feature reachable, and verify any "it looks assigned now" result survives a full remount/reload — an optimistic local render and a persisted round trip look identical for exactly as long as you don't check.

**Second addendum (2026-08-05, TASK-2450 AC#8/#9, found by review not live verification): there was a third instance, one layer further out.** `TTSProfileService.commit_portable_profile_import` — the character-*import* path, not the Roleplay assignment path either of the first two instances lived in — had the identical `current.availability == "available"` gate blocking auto-apply of an imported legacy voice, plus three toast strings inheriting the same "not currently available" mislabel. This one was **not** caught by the live-verification sessions that found and fixed the first two; it was caught by a reviewer explicitly re-running the "grep for every occurrence of this comparison" step this same entry already prescribes, against the full diff, as its own dedicated pass — confirming the remedy works when actually performed as a full-codebase sweep rather than scoped to the one feature just fixed. Three independent occurrences of one enum being taught in exactly two of its three call sites is not a coincidence of this feature; it is what happens whenever a state gains a new value after several call sites already hard-coded the old two. **The actionable form going forward:** when a value's vocabulary grows (two states becoming three, in this repo's case), grep is cheap enough to run as its own numbered step *before* any live verification, not as a thing live verification occasionally happens to surface — treat "how many places compare against the old vocabulary" as a question with a countable, checkable answer, not a hunch.

**Third addendum (2026-08-06, slice-3 TASK-4 review): the sweep the two addenda above prescribe is real and it works — and it structurally cannot see the fourth shape.** Slice 3 added a *second failure domain* to an existing bounded-code table: alongside the four character-voice resolution codes, two new `default_profile_*` codes, each with copy naming the app-wide default voice instead of a character's. The implementer ran exactly the full-codebase identifier sweep this entry prescribes (`_RESOLUTION_COPY`, `_GLOBAL_OVERRIDE_CODES`, `CharacterTTSResolutionError`, `CharacterTTSResolutionCode`), and the sweep was **accurate and complete for every surface that references those identifiers**. It still missed the defect, because the defect was not a reference to any of them: `app.py::_offer_tts_global_override` built its `ConfirmationDialog` from a **hardcoded literal** — "The assigned character voice could not be resolved" — shown for *every* override offer. So a user with no character in play, whose app-wide default voice had been deleted, was asked to consent to a voice substitution by a dialog describing a character assignment that did not exist. The toast (built from `event.error`, i.e. the bounded copy) was correct; the prompt the user must read *in order to consent* was not.

**Why the grep could not find it.** An identifier sweep finds code that *reads* the state. It cannot find code that *restates the state in prose* — a string literal that duplicates, in English, an assumption the code no longer guarantees. The dialog was correct when exactly one failure domain existed; adding the second falsified a sentence in a different file that nothing links to the code table.

**What to do.** When a bounded vocabulary gains a new *domain* (not merely a new value), run a second sweep the identifier grep cannot substitute for: grep the **old copy's distinctive phrases** across every user-facing string in the repo — here, "character voice" / "assigned character" — and for each hit ask whether it is still true for every path that now reaches it. Then apply the structural remedy so the next domain cannot reintroduce it: the fix threaded a `domain` field derived *from the code itself* (frozenset membership, not a per-call-site argument) through to the UI, which selects from a bounded copy dict with an honest neutral fallback for an unknown domain — making desync impossible by construction rather than by discipline. **The general principle: a hardcoded sentence is an untested assertion about program state. Audit user-facing copy by asking "what does this screen claim, and is that true on every path that reaches it," not by grepping for symbols the copy never mentions.**

---

## `save_screenshot` does not render the toast rack — probe `app._notifications`

**What happened.** TASK-2154.16 (FB-05) added an error toast on Console stream
failure. The notification fired correctly — visible in `app._notifications` with
`severity="error"`, alive 2.5s after posting — yet the UAT SVG capture
(`app.save_screenshot`, Textual 8.2.8) showed no trace of it, and neither did a
control probe that called `notify()` manually and screenshotted 0.3s later. The
toast rack is simply absent from SVG exports in this Textual version, so "toast
visible in capture" is unprovable by screenshot and a fix can look broken when it
is not.

**What to do.** To verify toast behavior in a pilot session, assert on
`app._notifications` (message text + severity + that it is still alive after the
expected interval), not on the SVG/PNG capture. Use captures for transcript/row
content only.

---

## A child return code does not prove `asyncio.Process.wait()` can finish when descendants inherit its pipes (TASK-3792, 2026-08-08)

**What happened.** The managed audio.cpp supervisor's injected-process suite was
green, but the first real subprocess fixture on macOS exposed a cleanup hang. The
exact child exited and its `returncode` became available; a test-only descendant
kept the child's inherited stdout/stderr descriptors open. The supervisor's sole
`Process.wait()` call nevertheless remained pending, so cleanup could not reach
the output-drain join or generation retirement. Waiting harder, cancelling drain
readers, or signaling the descendant would either remain unbounded or violate
the exact-child ownership contract.

**What to do.** Characterize subprocess ownership with a real executable fixture,
including a descendant that holds inherited stdout/stderr after the owned child
exits. Keep exactly one reaper task. Once that task observes the exact child's
return code, immediately invalidate public endpoint, readiness, health, and
catalog evidence even if `wait()` and pipe cleanup are still pending. Close only
the parent's pipe transports so the existing wait can finish; independently
bound and join/cancel the drain tasks. Assert cleanup finishes while the
descendant is still alive, then let the test finalizer kill only its captured
fixture PID. Injected launchers remain useful for races, but they cannot prove
event-loop child-watcher and pipe-transport behavior on the host platform.

---

## `.value = "x"` in a Pilot test cannot see a widget that never paints its own text (TASK-13154.1, 2026-08-09)

**What happened.** Live-verifying the new Settings ▸ Agents CRUD editor (fleet
PR-1), clicking into the Name field and typing produced no visible change at
all — not the typed text, not even the field's own placeholder. Tab-cycling
focus between fields *did* move a visible border, and blindly typing anyway,
then pressing Save, produced `Saved 'researcher'.` and a correct row in the
list — so the value was reaching `Input.value` and the DB write was correct.
The bug was 100% cosmetic and 100% severe: a real user typing into this form
sees nothing happen, no matter how many characters they type, with no error.

Root cause: `AgentsSettingsPanel`'s four `Input(...)` calls omitted
`classes="settings-compact-input"` — the one class every other Settings Input
in this screen carries. `.settings-input-row { height: 1 }` (task-1586's
one-row control convention) gives the row exactly one line; `.settings-compact
-input` is what turns off Textual's default 3-row bordered `Input` chrome
(`border: none; border-left: solid …`) so the single row is spent on text, not
on a border. Without the class, Textual's default border ate the only
available row, so the field painted a slice of box-drawing characters (a
"┌────" top edge when focused, a squashed "▔"-style artifact unfocused) and
**never painted the placeholder or the value on any state** — not empty, not
typed, not populated via a ListView selection round-trip.

**Why every existing test passed anyway.** `Tests/UI/test_settings_agents_category.py`'s
four tests all write `panel.query_one("#agents-name-input").value = "researcher"`
directly and assert against `.value` afterward — never a real keypress, never
a render check. That is the only way this class of tests can exercise the
widget, and it is structurally blind to "does this widget paint what it
holds" — the exact shape the "geometry assertions, not just display/text"
entry in `lessons-testing-evidence.md` already names, now with a `Switch`
widget confirmed as a second instance in the same panel (the Enabled switch's
render is a narrow fixed-width glyph, not a crushed border, but it is the
*only* raw `Switch` in the whole Settings screen — every sibling category uses
a button-as-toggle instead — so it too had zero prior test coverage of "does
a click on it actually change `.value`").

**What to do.** When a new Settings (or any `.settings-input-row`-family)
field uses a bare `Input`/`Switch`/similar without copying the class list of a
working sibling field verbatim, grep the screen for the compact/style class
every other instance of that widget carries and add it explicitly — do not
assume default widget styling is safe inside a height-constrained row. And
before trusting a form's Pilot-test coverage, check whether any of its
assertions actually simulate a keypress or a real click-and-observe against
the compositor; a suite built entirely from `.value = "x"` assignments proves
the data model works and says nothing about whether a human can see or drive
the same field.

---

## The app's own config-rewrite-on-boot can corrupt its own file into invalid TOML, and the failure mode is a silent profile swap, not an error (TASK-13154.1, 2026-08-09)

**What happened.** Mid live-verification of the same fleet PR-1 task, a
scratch profile launched via `TLDW_CONFIG_PATH=<scratch>/config.toml` (the
standard recipe: `[general] users_name = "verify_x"`) started opening real
`~/.local/share/tldw_cli/default_user/*.db` file handles — the actual live
user's profile — confirmed via `lsof -p <pid>`. This was caught, the process
was killed immediately, and a direct check of `default_user`'s tables
(`agent_definitions`, recent `conversations`, `agent_runs` for today's date)
confirmed nothing had actually been written there; only benign WAL-mode mtime
touches occurred. No real damage, but the near-miss is the finding.

Root cause: across two boots of the same scratch config, the app's own
config-normalization write path appended a *second* `api_key = "…"` line into
an `[api_settings.openrouter]` table that already had one (each boot rewrites
the file — see the existing "app rewrites its scratch config on boot" entry
above) — producing a table with the same key defined twice, which is invalid
TOML (`tomllib` raises `"Cannot overwrite a value"`). `TLDW_CONFIG_PATH` was
confirmed correctly set in the process's own environment
(`ps -wwE -p <pid>`) and the *config file itself* was never touched (its mtime
never moved) — the corruption was entirely inside the scratch file the app was
supposed to read from, not a config-path/env-var isolation failure. With the
file unparseable, every setting inside it — including `users_name` — silently
had nowhere to come from, and `get_user_folder_name()`'s own fallback
(`"default_user"`) took over with no error, toast, or log line naming the
parse failure. The first-run wizard also re-offered itself on that same boot,
consistent with the loader seeing none of the `[first_run]` flags that were,
in fact, sitting right there in the (invalid) file.

**What to do.** Never hand-edit a scratch `config.toml` a running app has
already rewritten without validating it parses (`python3 -c "import tomllib;
tomllib.load(open(path,'rb'))"`) immediately after the edit AND immediately
after the next boot — a `printf >> file` append that looked fine in isolation
combined with the app's own next rewrite pass to produce the duplicate key;
neither edit alone was the problem. After any multi-boot scratch-profile
session, confirm isolation held by `lsof`-checking the actual running PID for
open file handles under the real profile directory (`grep -o
"default_user\|<scratch_profile_name>"`), not just by trusting the launch
command's env var — a `ps -wwE` showing the right `TLDW_CONFIG_PATH` is
necessary but, as this incident shows, not sufficient evidence that the
*content* of that file is still valid. This is also a real product defect
independent of the verification recipe: a config loader whose own
normalization pass can write itself into unparseable TOML, with no visible
failure and a silent fallback to the default profile, is a data-integrity and
privacy-boundary risk for real users, not just scratch-profile verification —
filed as task-13157.

## Your own library may not contain the precondition the defect needs — check it before the live check counts (TASK-4110, 2026-08-10)

**Incident.** TASK-4110's failure mode is structural: a keyword-only document
cannot enter hybrid's fused top-k *when the vector leg returns k or more
distinct documents*. The closing task's live check was to run a keyword-only
query against a scratch profile holding a copy of the real Library DBs and
vector index, per the usual recipe. One `sqlite3` count over the copied
`chromadb/chroma.sqlite3` first, purely to pick a target, showed what the
recipe would have hidden: **453 embeddings but only four distinct documents**
(450 of them chunks of one media item). The vector leg on that library can
never fill a top-5, so a keyword-only row was going to appear no matter what
the fusion weighting was — the very sighting the task's own description warns
is not evidence. The check would have rendered a green screenshot of the
intended behaviour while proving nothing about the fix.

**What to do.** Before running a live check, write down the *precondition* the
defect requires and query the live environment for it — row counts, distinct
ids, index population — rather than assuming a copy of real data is
representative. Real personal data is usually *smaller and lumpier* than the
test fixtures, so the shape a bug needs is often exactly what it lacks. If the
precondition is absent, build it through the app's own paths (here: 60 real
documentation files ingested via `index_entries`, the seam both production
indexing routes converge on, giving a 64-document library) and say in the
write-up that you did, and why. Then run **both arms in the same UI on the same
data** — ship value, then the previous value, one constant apart. That pair is
what turns "the row I hoped for appeared" into evidence: here the shipped value
put the keyword-only document in the last visible slot and the reverted value
put an ordinary vector row there, with the four rows above it byte-identical in
both runs — which also proves the change did not disturb the ranking it was not
supposed to touch.

## Importing the test harness outside pytest is NOT config-isolated (2026-08-10, task-4023 Task 3)

**Incident.** While root-causing a scroll race, three quick probes ran
`python - <<EOF` scripts that imported `Tests.UI.test_library_shell`'s harness
helpers directly and drove a real `LibraryScreen` (typed a query, pressed Run).
The probes "used the same harness as the green tests", so they looked safe —
but the `TLDW_CONFIG_PATH` bootstrap that isolates the suite lives in
`Tests/conftest.py`, which only runs UNDER PYTEST. The probe's app therefore
resolved the REAL user config, and the query it executed ("tides") was
persisted into the live `~/.config/tldw_cli/config.toml`
`[library.search] history` by `_save_library_search_history`. Found only
because the close-out self-review greped the live config; repaired by hand.

**Rule.** The harness is not the isolation — the conftest is. Any ad-hoc
`python` invocation that imports app or test modules must set
`TLDW_CONFIG_PATH` to a scratch config itself, or be written as a real
(possibly throwaway) pytest test. After ANY non-pytest probe that could have
driven app behavior, grep the live config for the probe's own inputs before
declaring the session clean.

## A refusal gate after imports is not a refusal gate (TASK-15262, 2026-08-11)

**Incident.** The first visual-compaction evaluation CLI checked
`--confirm-billable` before its explicit `load_settings()` call, which looked
like a safe charge boundary. A dry `--help` run still imported
`tldw_chatbook.Chat.console_provider_gateway` at module load. Importing the Chat
package transitively initialized `config.py` and attempted to create or update
the normal profile before argument parsing; on the verification machine it
failed against the real profile path. The command made no provider request, but
its supposed refusal/help path had already crossed the profile boundary.

**Rule.** For confirmation-gated or profile-isolated CLIs, keep every
application import that can initialize config behind the validated boundary.
Prove both `--help` and the unconfirmed refusal path in a subprocess whose
`TLDW_CONFIG_PATH` points to a nonexistent file, then assert that file was not
created. A guard is only real if importing the command cannot bypass it.

## Detecting Textual focus from a captured pane: diff the STYLE, not the line — a ticking counter is not focus (supervisor-fleet PR 3a-1 Task 7, 2026-08-11)

**Incident.** Live-verifying the Console fleet panel's per-row cancel (focus a
sub-agent row, press `Delete`) needed a way to know *when* the row had focus, since
`tmux` gives no focus readout. The first attempt tabbed one key at a time and compared
the row's captured line by `md5`, treating any change as "focus moved here". It fired on
Tab #2 — and it was wrong twice over:

- The row's text contains a **live elapsed segment** (`· 48s`). It ticks. Any
  line-content hash changes on its own, with no focus anywhere near it.
- Tab **scrolls the rail** as focus walks widgets above the panel, so the row's line can
  move or disappear entirely; `grep | head -1` then returns empty, whose hash also
  differs.

Both false positives were "confirmed" by a capture that looked right. Meanwhile the real
signal is one CSS rule — `.console-inspector-section-row:focus { background: ... }` — so
the honest probe is to extract just the background SGR codes from the row's line
(`grep -o '48;2;[0-9;]*'`) and watch for the focus tint appearing. With that, the row was
found at Shift-Tab #7 and `Delete` cancelled the child for real (`running` → `cancelled`).

Three live sub-agents were spent chasing this: each check needed a child still running,
and the diagnosis cost longer than the children lived.

**What to do.** To prove a Textual widget has focus from a terminal capture, find its
`:focus` rule in the TCSS first and assert on **that specific style token**, not on the
line's bytes. Never treat "the captured line changed" as evidence about focus when the
widget renders anything time-varying. And prefer `Shift+Tab` when the target sits near
the end of a scrollable container — walking forward from the top scrolls the target out
of the very capture you are reading.
## PNG compression is not image-token compression (TASK-15482 / TASK-15505, 2026-08-11)

**Incident.** The first valid raw-context visual-compaction run sent two
deterministic PNG pages to GPT-5.6 Terra and used 2,909 input tokens versus
1,060 for the text control. The PNGs were compact on disk, but the provider
charged 174.4% more input. TASK-15505 then traced the geometry: the renderer
drew each page on a 512x512 logical canvas and mechanically enlarged it to
1024x1024 before dispatch. Current OpenAI image-input documentation says
GPT-5.6 omitted/auto detail preserves original dimensions and meters 32x32
patches, so the enlargement changed each page from 256 raw patches to 1,024
without adding transcript content.

**What to do.** For image-context optimizations, record the exact dispatched
dimensions, detail setting, model family, and provider-reported usage. Treat
local byte size and raw patch geometry as diagnostics only, never as measured
token savings. Remove redundant geometry in an evaluation-only renderer first,
then run the same downstream quality gate before changing production; smaller
text can reduce recall even when its patch count is lower.
## `extra="allow"` on a request model says nothing about the server (task-3309)

**Incident.** The Library forwarded per-type ingest options to the server by
name, relying on `MediaIngestJobSubmitRequest`'s `extra="allow"`. That was read,
for months, as "the server accepts these". It does not mean that — it only means
the *client* will serialize them onto the wire. The receiving endpoint binds
each form field with an explicit `Form(...)` and never reads `request.form()`,
so FastAPI discarded every undeclared field silently and answered `200`.
Nineteen fields were in that state: a user could set OCR language, speaker
diarization, timestamps or VAD in server mode and nothing at all happened, with
a successful-looking job to show for it.

**Why nothing caught it.** Two of the repo's own tests asserted the broken
names travelled verbatim (`assert kwargs["pdf_engine"] == "docling"`), which
converted the silent drop into a requirement. Every one of them passed.

**The check that works.** Ask the *running server* what it declares —
`/openapi.json` enumerates the endpoint's form fields — and assert that every
field the client puts on the wire is in that set. Capture the list as a fixture
with its provenance so the assertion is against a real server of a known
version, not against a hand-written list that drifts the same way the code did.

**Widen it past the obvious loop.** The first version of the guard checked only
the per-type options loop and passed. The nineteenth field,
`force_regenerate_embeddings`, was named in the *service method's own signature*
and sent on every submission. Check what reaches the request, not what one
code path contributes to it.

**"No server equivalent" is a claim about the whole server, and it needs the
whole server to back it.** The first version of this fix labelled eleven fields
"no server equivalent" on the strength of one endpoint's schema. The owner
pushed back -- "the server should have full support" -- and was right. Checked
against the server *source*, two of them (`transcription_provider`,
`translate_to_english`) are real capabilities of the transcription core that its
HTTP API simply does not expose; two more are server-side config rather than
request fields; one is accepted on a different endpoint; and two were not
missing at all -- they are accepted by the web endpoint that the client already
routes to correctly. Only four were genuine absences. The behaviour (do not send
them here) was right either way, but the *reason* attached to each one is what a
reader acts on later, and four of the seven wrong reasons pointed at the wrong
repo to fix. Compare endpoint surfaces before concluding a capability is
missing: `/media/add` turned out to be a strict subset of
`/media/ingest/jobs`, so "the client is on the wrong endpoint" was also wrong.

**A blocked live call is not a blocked verification.** Real submissions were
impossible here (the instance rejected the configured API key, and its key is
env-only on the server process). That did not weaken the finding: a field the
endpoint never binds cannot take effect, whatever a submission would have
shown. Reach for the server's own contract before concluding a live check is
unavailable.

**Postscript: the key existed.** The live check was called impossible because
`~/.config/tldw_cli/config.toml` held a stale key the server rejected. The real
one was in the server repo's own `Config_Files/.env` all along. Before
recording a live check as blocked on credentials, look in the server repo -- the
running process was started from it.

---

## Trace the whole stream sequence, not only the first failing event (TASK-16074, 2026-08-13)

**Incident.** Moonshot's paid Kimi K3 native-tool UAT failed behind Console's
safe synthetic 502. A keys-only trace of the first rejected SSE event isolated
`system_fingerprint`; the parser fix passed every deterministic and joined
fixture, but the paid UAT still failed. The next structural trace found terminal
`choices[0].usage`; supporting that also passed the suite, but the paid UAT still
failed because Moonshot repeated the identical usage mapping in the following
top-level empty-choice event. The first probe was correct but incomplete: three
valid shapes appeared at different points in one live stream.

**What to do.** When a strict streaming parser rejects a real provider, trace
the complete event sequence using only key names, container types/counts,
bounded enums, and parser state transitions. Do not stop after the first
unexpected field, and do not log raw values or bodies. Turn every newly observed
sequence rule into RED/GREEN coverage, including negative controls for repeated,
conflicting, misplaced, and JSON-type-distinct data. A first-event trace proves
one incompatibility; only consuming the whole live stream proves the contract.

## Pilot-mouse tests cannot certify real-terminal mouse flows (console text selection, 2026-08-15)

**Incident.** The Console drag-text-selection feature passed 400+ pilot-driven
widget tests (including a real-ChatScreen smoke test) and still did nothing in
kitty, iTerm2, and Terminal.app. Three successive live-spike rounds each found a
defect invisible to the synthetic suite: (1) real-terminal `MouseDown`/`MouseUp`
events arrive with `event.widget` unset -- Textual's screen forwarding only
assigns it on the translated `MouseMove` path -- so arming logic keyed on
`event.control` no-oped while pilot events (which carry the widget) stayed
green; (2) a menu anchored via `dock: top` + `styles.offset` painted translated
by the offset but CLIPPED to the un-translated dock slot, and hit-tested at the
un-translated region, so most buttons were invisible AND unclickable; (3) a
drag's synthesized release Click can dispatch LATE, after an intervening press
already consumed the suppression flag it relied on. Every fix shipped with a
regression test that drives the REAL event shape (`widget=None`), the real
anchor geometry, or the real message ordering.

**What to do.** A mouse-interaction suite is not certified until the flow has
run in a real terminal. When pilot tests pass but the live terminal disagrees,
instrument the widget's event handlers to a scratch log and diff the real event
shapes against the synthetic ones before touching logic. Anything anchored or
positioned in Textual must use `position: absolute` + `absolute_offset` (the
tooltip mechanism, where region/paint/clip agree) -- never `dock` plus
`styles.offset`, whose paint and hit regions diverge. Interaction suppression
meant for a synthesized follow-up event needs its own one-shot token, not a
shared flag that other handlers consume.

## `position: absolute` still eats 1fr budget in textual 8.2.8 — screen-mounted overlays need `overlay: screen` (console "black bar" spike, 2026-08-16)

The user's screenshots of the Console all showed a black bar of dead rows under
the composer whenever a selection was active. Three vision passes analyzed the
selection menu (my most recent work) and called it correct — which it was; the
menu was also the *cause*. Textual 8.2.8's vertical layout excludes
`position: absolute` children from sibling stacking
(`layouts/vertical.py`: `if not overlay and not absolute: y = next_y`) but
still passes **every** child's height into `resolve_box_models`, the fr
denominator — so the 9-row menu mounted on the screen silently subtracted 9
rows from `#screen-content`'s `1fr`, floating the composer above dead rows.
`overlay: screen` is the style that removes an overlay from the container's
flow math entirely; `absolute + overlay: screen` keeps the anchor and frees
the budget. Fixed in d2b4d2630.

Two process failures made this cost hours: (1) every reproduction attempt
measured layouts **without the menu open**, because the harness flows that
mount the menu didn't have a 1fr sibling for it to rob — the defect's
precondition (screen-mounted overlay + 1fr sibling) never existed in one
place until the real app was booted with the menu mounted; the "Your own
library may not contain the precondition" lesson above is the same trap.
(2) I analyzed what I had most recently changed instead of what the user was
pointing at. A temporary keybinding that dumps the live geometry of the whole
bottom chain (F12 → regions/display/dock/height of every screen child) turned
one angry screenshot into an exact attribution — screen-children dumps beat
fixed-widget dumps because they can name a consumer you didn't think to ask
about.

## Mouse capture reroutes the synthesized Click to the capturer — pilot clicks bypass capture entirely (console "dead buttons"/"can't select messages" spike, 2026-08-16)

Two live-only mouse defects, both invisible to pilot tests, both traced only
after a user-run F12-style event dump:

1. **The drag's synthesized release Click can arrive after `just_finished`
   was already consumed.** The row guard's `or` chain short-circuited past
   `consume_release_click()` and did not stop the event, so the artifact
   click bubbled to the transcript's `on_click`, whose dismissal cleanup
   (`_remove_selection_menu` → `row.clear_selection()`) erased the selection
   the just-opened menu existed to act on. Every selection-dependent menu
   action then read an empty quote and hit the silent blank-selection guard
   — "buttons only work once", because the first menu of a session usually
   won the message-queue race and later ones (after a modal round-trip
   reordered the queues) reliably lost. One-shot suppression tokens must be
   consumed at EVERY layer that can see the artifact, and the artifact must
   be `stop()`ped where identified.
2. **A plain click's synthesized Click routes to the mouse CAPTURER, not the
   widget under the pointer.** Arming the selection drag captures the mouse
   on press; the capture only releases when the MouseUp is *processed*, which
   happens after the Click was already forwarded — so the click landed on
   the transcript while the row the user clicked never saw it, and mouse
   click-to-select a message never toggled in any real terminal. Pilot
   `click()` computes the target itself and delivers directly, bypassing
   capture — the same pilot-vs-real divergence class as the phase-1 arming
   bug. Widgets that capture on press must re-dispatch clicks themselves:
   the transcript's `on_click` walks `event.control` (the true target,
   preserved on the synthesized event) up to the row and applies the row's
   click semantics.

Fixed in 78cd9aeba and 86f5807c9. The diagnostic pattern that cracked both:
log events at the WIDGET level (menu received down/up, hit-test result,
app-level `_mouse_down_widget`) and — when the chain completes but nothing
happens — at the app's raw-event boundary. A fixed widget list cannot name a
consumer you did not think to ask about; a screen-children dump can.

---

## A pytest probe outside the repo tree runs with NO conftest — no sandbox, no egress guard (TASK-16198, 2026-08-15)

**Incident.** While tracing the knowledge_entry teardown egress, a scratch
probe test was written to the session scratchpad (`/private/tmp/...`) and run
with `pytest $SCRATCH/test_probe.py` from the worktree. pytest's conftest
discovery walks the TEST FILE's ancestors, not the invoking directory's — so
`Tests/conftest.py` never loaded: no `TLDW_CONFIG_PATH`/HOME sandbox, no
network-guard install, no autouse isolation. The probe booted the full
`TldwCli` against the LIVE `~/.config/tldw_cli/config.toml` and
`~/.local/share/tldw_cli/default_user/` — opening the user's real databases
and overwriting `model_catalog_cache.json` (two entries dropped). Only the
app's own consent gate happened to prevent real network egress. The run
LOOKED sandboxed: it printed `blocked_attempts=()` from an imported-but-never-
installed guard module — a green that measured nothing.

**What to do.** Scratch pytest probes that import the app go INSIDE the
worktree's `Tests/` tree (delete after), never in /tmp or the scratchpad —
placement is what activates the sandbox and the guard. If a probe must live
outside, it does not get to import `tldw_chatbook` without first setting
HOME/XDG_*/TLDW_CONFIG_PATH to a throwaway root by hand (and asserting the
config path took). Treat an imported guard whose `install()` conftest never
ran as adversarial: `blocked_attempts=()` from an uninstalled guard is not
evidence of no egress.

---

## The venv's editable install points `tldw_chatbook` at a FOREIGN worktree; you win only by import ordering (task-15860, 2026-08-14 → 2026-08-17)

**What happened.** Every headless-wake landing ran its tests from a
worktree under `.worktrees/`, using the repo-root `.venv`. That venv's
editable install resolves the package elsewhere entirely:

```
.venv/lib/python3.12/site-packages/__editable___tldw_chatbook_0_1_8_0_finder.py
    tldw_chatbook -> .worktrees/task-2512-mcp-unified/tldw_chatbook
```

Every result in this arc would have been a statement about *someone
else's branch* if that finder had won. It does not win — setuptools'
editable finder is **appended** to `sys.meta_path`, so the stdlib
`PathFinder` searching the rootdir pytest prepends (because `Tests/` is a
package) resolves first. That is the entire margin: an ordering detail in
a third-party install hook, with nothing in the repo pinning it.

It is silent in both directions. A run against the foreign worktree
raises nothing, prints nothing, and produces plausible passes and
plausible failures — the arc's own baseline "measure the failure at the
merge-base" discipline would have compared two branches neither of which
was yours.

**What to do.** Keep an executable assertion of import provenance in the
gate, not a habit: `Tests/test_probe_import_provenance.py` asserts
`tldw_chatbook.__file__` lives under the worktree the test file is in.
Run it first, read its printed path, and treat a green suite whose
provenance probe was not in the run as unproven. The same shape applies
to any machine with several checkouts sharing one venv — which, on this
repo, is every machine.

## A DB append is invisible to a live Console *and* to the next mount — the STORE is what the transcript and the payload are built from (task-15860, Task 0 probe P1)

**What happened.** Two of the three designs for headless wake rested on
one assumption: write the wake's rows straight to ChaChaNotes while
Console is down, and the user sees them when they come back. Executed
through the production `ChatPersistenceService.create_message` and the
real navigation API, rows written with Console unmounted were, at the
next mount: **absent from the transcript, absent from the rendered
widgets, absent from the next send's provider payload** — and the next
persisted append *forked the tree*, parenting itself to the pre-nav
assistant and stranding the headless rows on a dead branch. Maintaining
the durable active-leaf pointer as well changed none of it. Two runs,
identical. The composed case was worse: a wake turn that genuinely ran,
spent real tokens and stamped its ledger persisted four rows, and the
returning user saw two.

The mechanism is ordinary once seen and invisible until then: Console
history travelled across a navigation as an in-memory `ScreenStateStore`
snapshot, and the restore path rebuilt the store from that payload
**without ever re-reading the database**. The DB was a write-only mirror
for this purpose.

**What to do.** Before treating a database as the place a background
writer can deposit user-visible state, find out what the READER is
actually built from. Grep the restore/hydrate path for the DB read and,
if there isn't one, say so out loud — "it's persisted" is not "it will be
shown". The general form: a durable write is evidence about durability
and about nothing else. Two live sessions of this programme were spent on
designs that a single executed probe retired in an afternoon.
## A worktree's live probe imports the MAIN checkout, and a redirected log hides its own progress (task-17370/17380, 2026-08-17)

Two mechanical traps cost a cycle each while measuring the research pipeline
from a worktree.

The venv here is an editable install pointing at
`/Users/.../tldw_chatbook/tldw_chatbook`, i.e. the MAIN checkout. A probe run
from a worktree with the venv's python imported the main checkout's package,
which was on an unrelated branch — the first attempt died with
`ModuleNotFoundError: No module named
'tldw_chatbook.Research_Interop.local_research_engine'` on a module that exists
in the worktree. Fix: `PYTHONPATH=<worktree>` (or run the script from the
worktree root, so `sys.path[0]` wins).

Second: the baseline recorder prints progress with `print()`, and Python
block-buffers stdout when it is redirected to a file — so a 50-minute run's
per-question lines all appeared at the very end, while loguru's stderr flowed
continuously. A monitor watching for `Running:` saw nothing for the whole run
and a `grep -c` for it returned 0 on a live, healthy process. Fix:
`PYTHONUNBUFFERED=1` for anything whose stdout you intend to watch.

**Also worth stating**: verifying a UI launch path means launching it the way
the UI does. A window-created research run is CHECKPOINTED (the scope service
never passes `autonomy_mode`, so the service default applies), so it parks at
`plan_review` and never reaches `collecting` until the checkpoint is approved.
A harness that used `autonomy_mode="autonomous"` would have "verified" a path
no user takes.

## tmux synthetic mouse (`send-keys -l` SGR bytes) reaches some widgets and not others — a dead click is not a dead control (task-17500, 2026-08-17)

**What happened.** Live-verifying the fixed approval card, clicks synthesized as
raw SGR sequences (`\e[<0;COL;ROWM` / `…m` via `tmux send-keys -l`) worked on
the Console session TAB STRIP (they created a tab and switched sessions, with
one-shot coordinates computed from a single capture) but never registered on
anything inside the transcript region — the approval card's fast Deny, Deny
all, and its decision Select all ignored identical sequences at verified
coordinates, across press/release timing variants and a hover-first attempt.
Real user mouse input is not this selective; ~15 minutes went into suspecting
the buttons before the pattern (tab strip yes, transcript region no) showed it
was the harness. Blind Tab-walking (25 presses with probes) never reached the
card either.

**What to do.** When a synthetic click does not land, test a KNOWN-clickable
control elsewhere on screen before concluding anything about the target — and
compute coordinates from ONE capture in the same shell invocation as the click
(captures taken across separate invocations race the UI and land clicks on
stale coordinates). If the control stays unreachable, do not force it: prove
press-resolution with the automated widget test (a mounted-widget `press()`
drives the same production seam) and use a documented keyboard path (here:
quit-denies) to end the live round. Record which regions accepted synthetic
mouse so the next rig starts there.

## The Console's provider-error text replaces the provider's own 400 body — go direct with curl to get it back (TASK-18414, 2026-08-18)

**What happened.** TASK-18414 was filed off a live-observed failure: a scratch-profile
Console on `claude-opus-5` failed every send, and the preserved pane read

    Agent run failed: provider returned HTTP 400 (Provider error from anthropic: bad
    request. Status: 400. Selected model: claude-opus-5. The provider rejected this
    request. Confirm the model is still available, or choose another model from the
    model picker.)

That message is entirely the app's own text. Anthropic had actually answered
``` `temperature` is deprecated for this model. ``` — naming the offending parameter —
and the mapping layer discarded it in favour of advice ("confirm the model is still
available") that points at the wrong cause: the model was fine, the payload was not.
So the filed task could say a 400 happened but not *which* of two candidate parameters
caused it, and it explicitly left that as owed work. Recovering it took one `curl`.
The second failure shape was more valuable still: for `budget_tokens` the provider
replies ``` "thinking.type.enabled" is not supported for this model. Use
"thinking.type.adaptive" and "output_config.effort" to control thinking behavior. ``` —
i.e. the provider hands you the exact remediation, and the app throws it away.

**What to do.** A provider 400 seen through this app is a *report that* a 400 happened,
never *why*. Before theorising, re-issue the minimal failing request straight at the
provider with `curl` and the repo-root key file, and paste the verbatim body into the
task. Two cheap habits that paid off here: probe **every** shape the builder can emit
(the two shapes had different causes and different fixes), and probe the **controls**
that must keep working in the same batch — Opus 4.6, Sonnet 4.5 and Haiku 4.5 returning
200 for the identical payload is what turned "don't break older models" from an
intention into evidence. Keep this out of the app's config path entirely: a standalone
`curl` imports no `tldw_chatbook` module and cannot touch the live config.
---

## An "unchanged behavior" AC can faithfully pin behavior that is already broken — run the control leg live

**TASK-18802, 2026-08-20.** The summarization fix gated sampling params on the
modern-Anthropic predicate and pinned AC #4 ("models that still accept these
parameters are unchanged") with payload tests: legacy models keep receiving
`temperature=0.1, top_k=0, top_p=1.0` byte-for-byte. Those pins were green,
mutation-hardened — and pinning a payload that no served model accepts. Only
running the *legacy* leg of the live pass (claude-haiku-4-5, expected to just
work) surfaced it: HTTP 400 ``` `temperature` and `top_p` cannot both be
specified for this model. Please use only one. ``` (req_011CeEDXPHNyF7apkaZepbTN).
Follow-up probes showed every currently-served Claude 4.x rejects the
temperature+top_p combination, and the function's own fallback default
(`claude-3-haiku-20240307`) now 404s as retired — so the "preserved" legacy
path had no live model it worked on. Filed as TASK-19020 rather than silently
widening the fix, since changing those payloads is exactly what the AC forbade.

**What to do.** When an AC says "X is unchanged", the payload pin proves only
*unchanged*, not *working*. Give the control leg one live request in the same
pass as the fixed leg — it is one extra call, and it is the only thing that can
distinguish "preserved" from "preserved a fossil". If the control fails live,
probe the minimal shapes standalone, and file the discovery against its own
task instead of mutating the payloads your current AC pins.

---

## A Textual live harness needs screen, event, and paint readiness (TASK-18913, 2026-08-16)

**Incident.** The first Prompt-pagination live attempts failed even though the
mounted product state was correct. The harness treated app-level `_ui_ready` as
Library readiness, called Textual 8's async `Input.action_submit()` without
awaiting it, and sampled the compositor in the same cycle that focus/scroll was
scheduled. Those three harness errors respectively lost the initial rail action,
never posted `Input.Submitted`, and reported a reachable row as unpainted. A later
real run exposed the inverse product race: loading recomposed disabled pager
buttons and moved focus before the ready page could restore the invoking button.

**What to do.** A live Textual check must wait at each boundary for all three
layers it claims: authoritative screen state, the freshly mounted DOM, and the
compositor text/geometry. Read the installed Textual method contract before
driving it (`Button.press()` and `focus()` are synchronous schedulers;
`Input.action_submit()` is async), re-query widgets after recomposition, and
settle then re-check side-effect-free predicates. App startup readiness does not
prove a destination screen has loaded. A correct harness can then reveal a real
focus race instead of manufacturing one.

---

## Four Console-inspector traps from one programme (task-18300, 2026-08-18 → 2026-08-20)

**A migration `.sql` file must be registered in FOUR packaging registries or
an installed build cannot open the DB.** `aae305cbf` found that the v40→v41
`message_exchanges` migration was missing from `pyproject.toml`
`[tool.setuptools.package-data]`, `MANIFEST.in`, both sdist/wheel lists in
`Packaging/check_manifest.py`, and `RUNTIME_MIGRATION_PATHS` in
`Tests/Packaging/test_installed_distribution.py` — and, digging further,
`46945ebbe`'s earlier v39→v40 `transcript_annotations` migration had the
identical gap already sitting on `dev`, undiscovered, meaning any packaged
wheel/sdist already could not open a pre-v39 database before this task even
started (`422534a5f` proved it: reverting one registry line turned a green
release-checker probe red through the real migration chain). The release
checker validates against this same enumerated list, so a registry that is
itself incomplete cannot catch its own gap — only a `uv build` against the
real artifact plus a direct tar/zip listing does. **What to do:** after
adding any migration `.sql` file, grep for every existing migration's
filename across the repo (`pyproject.toml`, `MANIFEST.in`,
`check_manifest.py`, `test_installed_distribution.py`) and confirm the new
file appears in the same four places — then build a real sdist+wheel and
list their contents, don't trust the checker alone to catch its own
enumeration gap.

**A Textual `Collapsible(title=...)` is markup-parsed even when sibling
`Static`s pass `markup=False`.** `1850ea3dc`: a bracketed model id such as
`[test]` was silently eaten from a Collapsible title, and one containing
`[/]` raised `MarkupError` *inside* `compose()`, taking the whole modal down
— reproduced directly against the installed Textual both before and after
the fix. `Static(..., markup=False)` protects only that one widget class;
`Collapsible`'s title has no equivalent constructor flag and needs
`Content.from_text(title_text, markup=False)` built explicitly. **What to
do:** when any user- or model-supplied string (an id, a filename, a free-text
label) becomes a `Collapsible` title, assume it can contain `[` and build the
title via `Content.from_text(..., markup=False)` — grep for other
`Collapsible(title=` construction sites using an f-string or raw variable
directly and check each one.

**Naming a Textual reactive `loading` shadows `Widget`'s own built-in
loading-overlay reactive — and the collision surfaces as a `NoMatches`, not
an obvious name clash.** While porting `ConsoleConversationInspector`'s
Next-Send-tab loading flag, a bare `loading = reactive(False)` collided with
`Widget`/`ModalScreen`'s own `loading` (whole-widget loading-overlay
semantics): Textual's internal `loading` reads (e.g.
`Screen.update_pointer_shape`) walk the ancestor chain and invoke the new
reactive's `init=True` watcher before the pane's own DOM subtree exists to
query, producing a real `NoMatches`. The fix (still in the shipped code as a
comment at the reactive's declaration) was simply renaming it to
`next_send_loading`. **What to do:** never name a widget-local reactive
`loading` (or any other name that shadows a `Widget`/`Screen` base
attribute — `disabled`, `visible`, `styles` are the same trap); grep
`class Widget` / `class Screen` reactive declarations in the installed
Textual before picking a name for anything state-flag-shaped, and prefer a
prefixed name (`next_send_loading`, not `loading`) by default.

**Stale comments asserting `diagnose=True` overstated a security finding's
severity — verify sink config by grepping `logger.add(`, not by reading
comments.** `cee88d074`'s task-9 review had two prior rounds disagreeing
about whether the app's file log sink ran `diagnose=True` (which would dump
frame-locals — i.e. secrets — into the log on any traceback). Three `app.py`
comments asserted it did. Empirically, every live `logger.add()` call in the
codebase is `diagnose=False`; the comments were stale and had led a reviewer
to treat them as ground truth, overstating the finding's blast radius.
**What to do:** when a security or logging-config claim rests on what a sink
"does," grep every `logger.add(` call site directly and read the actual
keyword arguments — treat a comment describing sink behavior as an
unverified claim, not a source of truth, especially in a codebase old enough
for the comment to have drifted out of sync with a later refactor.

---

## A model's response can reveal in the transcript UI as one late batch, not incrementally — "stop while nothing is visible yet" is not proof the click missed (task-18300, 2026-08-20)

**What happened.** Live-verifying Console's Stop-mid-stream capture (a real
OpenAI `gpt-4o-mini` session, `~$0.03` total spend across the whole
programme), the most reliable way to *see* partial content before stopping
was expected to be watching the transcript pane fill in gradually. It did
not, for several prompt shapes tried in sequence: a 1200-word essay, a
2000-word essay, and "count 1 to 300" all showed a bare `Assistant
Generating…` label for anywhere from 3 to 20+ seconds with zero visible
characters, then — for the one case left to run to completion instead of
being stopped — painted nearly the entire response in one step once it
finished (reached "241, 242, … 267" out of a 300-target count, cut off by
`max_tokens`, appearing all at once around the 13s mark with nothing visible
before it). By contrast, short repeated-word prompts ("write 'apple' 100
times") revealed content within ~1s and then completed within another
1-2s — too fast to reliably win a tmux-round-trip race to click Stop in the
middle. Two *genuinely* early Stop clicks (landing within ~1-4.5s of send,
before the essay-style prompts had revealed anything) produced real
`call 0 [stopped]` entries in the Exchange tab with `Response (~0 tokens
est.)` — a legitimate empty-partial-content capture, not a harness failure,
confirmed by checking the *capture*, not just the transcript pane. Several
further attempts to reproduce a **non-empty** stopped capture all instead
raced past natural completion (`call 0 [complete]`) because the fast
word-repeat prompts finish before a tmux-driven click lands, while the
slow-reveal essay prompts show nothing to click "during."

**What to do.** Do not infer "the click missed" or "nothing streamed yet"
from an unchanging transcript pane — Console's transcript rendering can
legitimately hold back all visible content until a late point (observed:
right around natural completion) regardless of whether the underlying
provider stream has been delivering deltas the whole time. To confirm
whether a Stop-click genuinely raced ahead of the first token (true empty
capture) versus arrived after the response had already finished (a
`complete` call wearing a truncated look), check the *Exchange tab's own
call status and Response section* for that specific turn, never the
transcript pane's rendered text or a "Stopped" label's mere presence. When a
manual UI repro like this consumes several real provider round-trips without
converging (this one used ~10), that is itself the signal to stop chasing a
single visual confirmation and report exactly what was and was not observed,
per the honesty rule already in this file's header — a slow repro can
consume the very verification budget it was meant to spend on other
scenarios.

---

## A streaming generator must not yield from `finally` when Stop closes it (TASK-18300, 2026-08-26)

**What happened.** A direct real-OpenAI lifecycle replay finally produced the
non-empty stopped capture that the earlier tmux session could not time. The
capture itself was correct, but Python emitted `RuntimeError: generator ignored
GeneratorExit` after Stop. A focused regression reproduced the same failure
without the network: `chat_with_openai()` was suspended at a content yield,
`generator.close()` injected `GeneratorExit`, and the generator's `finally`
block tried to yield its synthetic SSE `[DONE]` sentinel before closing the
response/session. Yielding while handling `GeneratorExit` converted normal
cancellation into a runtime error and postponed cleanup.

**What to do.** Reserve `finally` for cleanup that cannot suspend. If a stream
needs a terminal sentinel on normal or handled-error exhaustion, emit it after
the `try`/`except`/`finally` block. Then an early `close()` still runs cleanup
and skips the sentinel naturally. Pin both halves: a unit regression that
closes after the first chunk and asserts transport cleanup, plus a real-provider
Stop replay that confirms the warning is gone and the partial capture remains
durable.

## A broad screen needle can certify the wrong subview (TASK-19012, 2026-08-21)

**What happened.** The isolated Notes journey sent the Library's advertised
`n` shortcut and waited for `Library notes`. The check passed at wide and
60-column sizes, and the helper reported that it had captured the Database
Notes route. Reading the frames top-to-bottom showed the shortcut had opened
**New note**: the heading, authority strip, and template list all contained the
broad `Library notes` needle, but the Add-from-files entry the run meant to
verify was not present. The process was healthy, the route was real, and the
assertion was still false evidence.

**What to do.** After every synthetic navigation input, assert a marker unique
to the intended subview and inspect the capture itself. Here the verifier now
captures New note honestly, sends Escape, and requires `Add from files` before
recording the Notes-list frames. A parent-surface title proves only that the
parent mounted; it cannot certify which retained child view is active.

---

## A live app launch can rewrite a tracked generated artifact (TASK-21161, 2026-08-23)

**What happened.** The isolated Console/DeepSeek UAT changed only the generated
timestamp in tracked `tldw_chatbook/css/tldw_cli_modular.tcss`. No source CSS
had changed; app startup had rebuilt the consolidated file. Left in the
working tree, that runtime side effect would have looked like an intentional
implementation change and polluted the review diff.

**What to do.** Capture `git status --short` before a live app launch and again
after clean exit. For tracked generated artifacts, compare the body as well as
the header before deciding whether a delta belongs to the task. Restore a
timestamp-only runtime rebuild to the pre-UAT content; regenerate and commit it
only when its source modules actually changed.

## A recovery modal can complete while the owning run stays non-terminal (TASK-20941, 2026-08-22)

**What happened.** Persona Buddy full-app UAT sent a real Console prompt through
a disposable loopback provider. Console first opened the project-instructions
folder recovery modal because the scratch profile had no eligible workspace
binding. Choosing **Disable** dismissed the modal and updated the session, but
the controller returned a raw rejected result without replacing its earlier
`VALIDATING` run state. The transcript, Console header, composer, and Buddy then
remained stuck at `Running` / `thinking`; no provider request was made. Unit
coverage had verified the setup decision and session mutation separately, so it
never observed the owning run after the modal completed.

**What to do.** For every modal or async recovery branch that rejects an active
run, assert both the domain mutation and the owner-facing terminal state, then
retry through the same controller. A dismissed modal or a returned failure value
is not sufficient evidence: the run-state ledger, composer admission, dependent
UI state, and retry path must all be terminal/current together.

---

## Restart the partial-review state, not only each isolated choice (TASK-97, 2026-08-23)

**What happened.** Real-authority integration tests proved Keep file, Keep
note, Keep both, Skip, receipts, restart history, and Undo one choice at a time.
The isolated live run then applied three choices and skipped the fourth. After
restart, the fresh plan correctly contained three `NO_CHANGE` rows plus the
remaining conflict, but Apply rejected the reviewed no-change IDs as an invalid
review. The single-choice cases never produced that mixed plan shape.

**What to do.** For a reviewed batch that permits partial completion, restart
after a genuinely mixed partial Apply and resolve the remainder through the
same public boundary. Per-choice restart tests prove each operation is durable;
they do not prove the next review accepts terminal no-op rows alongside work
that still mutates.

---

**A green seam is not a green flow; a bound can exist at two layers.**
TASK-23089 / PR #2158, 2026-08-27. First-run setup was verified against a real
OpenAI key for the first time and failed three ways in a row. Each fix was
verified *at the seam it changed* and each time the flow was still broken:

1. `settings_endpoint_probe` never sent `Accept-Encoding: identity`, so a valid
   key read as "connection error". Fixed; probe returned `reachable`, 100 models.
2. Discovery's `DISCOVERED_MODEL_MAX_COUNT` was 100 while api.openai.com returns
   128, so discovery failed closed. Raised to 512; discovery returned 128 models.
3. That success then hit the wizard's *own* still-100 bound in
   `_model_ids_from_discovery_result`, which raised, and the caller folds any
   raise into a failed discovery -- so a successful 128-model discovery rendered
   as "Couldn't reach the server (request failed)". Fix #2 had moved the failure
   up a layer, and only the live walk revealed it.

Two traps generalize. **Mocked peers never compress, throttle, or return
production-sized payloads** -- every test here used `MockTransport` with a
handful of uncompressed models, and a 401 returns before the body read, so no
existing test could see any of this. **When you relax a bound, grep for every
layer that re-checks the same quantity**; here `MODEL_IDS_MAX_COUNT` was
imported by four call sites with two different meanings (a probe's truncation
sample vs. a fail-closed catalog limit), and aliasing them is what put the
fail-closed bound below reality.

**What to do.** For provider-facing work, verify against a real credential and
walk the actual UI to the end state a user reaches -- not just the function you
changed. Two intermediate diagnoses in this task ("the key never reaches the
request", "the stale keyless failure is reused") were both wrong and both came
from repros built by hand; the malformed one passed staged settings as a flat
dict where the caller wraps them under `api_settings`. When a headless repro
disagrees with the running app, suspect the repro, and instrument the app --
a file-append probe reads out where `logger` does not, because the persistent
sink only records structured `diagnostics.*` events.

---

## A compact pane can pass one frame and still collapse mid-flow (TASK-18915, 2026-08-29)

**What happened.** The 80-column Skills reader correctly started with Items
collapsed and exposed its five-cell expansion grip. Expanding Items made the
pager visible, and the first paging assertions passed, but a later ordinary
layout sync discarded the inherited Items priority and collapsed the pane
again before filtering. Point-in-time geometry tests and wider terminal runs
all passed; only the multi-step 80x24 walkthrough observed the state transition.

**What to do.** Compact reader verification must exercise an explicit pane
toggle followed by at least one normal refresh-producing action, then recheck
the pane, focus target, and controls. Testing only the frame immediately after
the toggle proves that expansion is possible, not that the user's pane choice
survives the next resize or data refresh.

---

## A `1fr` child with `min_width: 0` starves silently; a static `min_width` "fix" clips the trailing controls instead (TASK-24415, 2026-08-29)

**What happened.** A user reported the Console `/` command trigger as "funky
in a bad way". The popup logic was fine — the live tmux run traced it to the
composer row: `#console-send-disabled-reason` toggled to `width: auto`
(capped 52) while the `1fr` visible draft carried an explicit
`min_width: 0`, so at ≤90 columns the advisory strip consumed the row and the
draft laid out at ZERO columns — no text, no caret, no placeholder, with the
suggestion popup filtering invisible input above. A constants comment
(TASK-2154.14) had promised "the draft keeps its 32-cell floor" in arithmetic
only; nothing in layout derived from it. The whole suite missed it because no
test asserted a draft REGION width while an advisory strip was visible.

The naive fix is wrong in a non-obvious way: giving the draft a static
`min_width: 32` makes Textual clamp the fr child and OVERFLOW the row — the
children painted last (Send/Dictate) get clipped off-screen, trading an
invisible draft for unreachable buttons. The working pattern is the one the
composer already used for its actions row: compute the advisory element's
cap in Python from the live row width (`row − fixed furniture − draft
floor`), hide it below a legibility budget, and re-derive on resize.

**What to do.** For any Textual row mixing `1fr` content with advisory
strips: assert the fr child's `region.width` at your narrowest supported
width with every advisory visible, and budget the advisory elements
dynamically in Python rather than trusting `max_width` to yield — an `auto`
element never shrinks below its content width, and `min_width` on the fr
child overflows rather than protects. Same-class suspects here:
`#console-voice-status` (dictation chip, up to 53 cells) during dictation.

---

## Provider seams differ per consumer — probe resolution before headless runs (TASK-21513, 2026-08-29)

**Incident.** The Daily-Reports live run needed three scratch-config rewrites before the demo saw the provider. First config used `[llm_api_settings] default = ...` per config.py's reader (line ~6946 reads `settings["llm_api_settings"]["default_api"]`): the brief still went to `openai` and failed "OpenAI API Key is required" — silent fallback, no error pointing at the key name. Second rewrite used `[API] default_api` + `anthropic_api_key` (the legacy bridge, `_LEGACY_PROVIDER_API_KEY_BRIDGE`), which the endpoint resolver honored — but the stored Anthropic key was dead at the provider, and the pivot to DeepSeek then failed "returned an empty response" because `chat_with_deepseek` reads its model from `[api_settings.deepseek]`, not `[API] deepseek_model`: the config-default `deepseek-v4-flash` spent all 2000 `BRIEFING_MAX_TOKENS` on `reasoning_content` (finish=length, content empty). Each trap was invisible from the TUI (a generic failed row) and only diagnosable by replaying `generate_briefing` with a spy on the chat seam.

**What to do.** Before a headless provider-dependent run, resolve the seam in one python -c (`config.default_api_endpoint`, key non-empty, and the handler's own `[api_settings.<provider>]` model). Never trust one section name because a different consumer reads a different one. When a row says "empty response" from a reasoning-capable model, suspect `finish_reason=length` with reasoning consuming the budget before assuming an extraction bug.

## The TUI renders on stderr — redirecting it blinds tmux capture (TASK-21513, 2026-08-29)

**Incident.** The first tmux launch used `2>stderr.log` to keep diagnostics; the pane went visually blank while `capture-pane` returned empty lines, yet the app was alive and painting — into the log. The render stream is stderr, so redirecting it for "clean logs" removes the very surface the verification is supposed to observe.

**What to do.** In tmux live runs, leave both stdout and stderr attached to the pane (the pane IS the capture artifact); take diagnostics from the app's own file log under the profile's data dir. Also note `tmux send-keys -H` can emit SGR mouse sequences (`ESC [ < 0 ; col ; row M/m`) to click Textual buttons that Tab/Enter cannot reliably reach, but the row/col must be recomputed from a fresh capture after every repaint.

## Three scratch-profile traps that masquerade as app bugs (Home recents UAT, 2026-08-30)

**What happened.** Live-UATing the Home recents PR against a copy of the real
profile failed four consecutive times with the same symptom — Home empty, no
errors — each for a DIFFERENT environmental reason that read like a product bug:

1. **`[database]` absolute `*_db_path` pins override `[paths] data_dir`.** The
   copied real config pins `chachanotes_db_path` etc. to absolute home paths;
   a scratch `data_dir` does not relocate them. The app silently opened
   nonexistent pinned paths (services wired as `None`, never re-wired) while
   unrelated DBs (workspaces) resolved through the scratch — so Console's
   workspace rail restored while every conversation/notes seam returned empty.
   Redirect EVERY `*_db_path` (and `USER_DB_BASE_DIR`) in the scratch config.
2. **`verify_trusted_directory` rejects `/tmp` scratch data dirs**
   (`unsafe_parent: shared_sticky_directory_not_allowed`). A disposable HOME
   nested under /tmp fails the same guard. Scratch profiles must live under
   the real `$HOME` (a private subdir is fine).
3. **`rsync`ing a live WAL-mode SQLite file produces a torn snapshot.** The
   copied DB later failed its v51→v52 migration with "database disk image is
   malformed" — while the app swallowed the error and booted with no
   ChaChaNotes at all. Copy live DBs with the online-backup API
   (`sqlite3.connect(src).backup(dst_con)`), and run `PRAGMA integrity_check`
   on the RESULT before blaming the migration: here the source DB itself
   carried a corrupt index (`idx_sync_log_entity` — wrong entry count), which
   `REINDEX` on the copy repaired before the migration could complete.

**What to do.** For a scratch-profile run against real data: build it under
`$HOME`, redirect every `[database]` pin, copy DBs via the backup API, and
integrity-check the copies. When a scratch boot shows "everything restored
except one subsystem", suspect per-service path resolution before the code.

## A green header test hid a feature that never worked end to end (TASK-26022 AC#7, 2026-09-02)

**What happened.** The Claude-subscription credential borrow shipped with six
green ACs and 16 passing tests — including one asserting the subscription path
sends `authorization: Bearer …` and no `x-api-key`. AC#7 ("verified against a
real account") was the only one left. Running a single real send against the
owner's Max account exposed that the feature was **non-functional end to end**
for two reasons no header-shape test could catch:

1. **The credential wasn't where the code read it.** On macOS Claude Code stores
   its OAuth credential in the login Keychain (`security find-generic-password
   -s "Claude Code-credentials"`), NOT in `~/.claude/.credentials.json`. The
   file-only reader returned `None`, so the subscription path never even
   engaged on the primary dev platform.
2. **The token is gated to the Claude Code identity.** With the credential
   supplied by hand, the send still failed: the OAuth token is rejected — as a
   *misleading* `429 rate_limit_error`, not an auth error — unless the request's
   `system` leads with "You are Claude Code, Anthropic's official CLI for
   Claude." Deterministic: that identity → 200; no system or any other system →
   429 (a Claude-Code request immediately after succeeded, ruling out real rate
   limiting). The shipped code set headers but never sent the identity.

**What to do.** For any feature that borrows a real, externally-minted
credential, the header-shape unit test is necessary but proves almost nothing —
credentials carry constraints (where they're stored per-OS, what identity/scopes
the token is bound to, what the server demands beyond auth) that only a live
send reveals. Do the one real call before closing the task, and read the failure
body literally: a `rate_limit_error` here was really "this token is Claude-Code
only." Corollary: a misleading upstream error code (429 for an identity
rejection) will send you chasing the wrong fix unless you test the discriminating
cases (identity present vs absent vs other) rather than trusting the label.

---

## `TLDW_CONFIG_PATH` does not isolate the OS keyring, and the app writes your config token into it on first use (schedules-handoff PR-6 task 6, 2026-09-02)

**What happened.** The PR-6 live gate launched a scratch profile
(`TLDW_CONFIG_PATH`, `users_name = "verify_pr6"`) against a local tldw_server and
could not authenticate: every scheduling call returned 401 and the Automations tab
said only *"Could not load server automations — see the log."* The scratch config
carried the real `SINGLE_USER_API_KEY`. A direct `httpx` probe with that key
returned 200. A header-dumping listener proved the app was putting the **correct**
`X-API-KEY` on the wire when pointed at a different port. It still 401'd on the
original one. Roughly 25 minutes went into this before the cause was read out of
`RuntimeServerContextProvider._resolve_auth_token`: credentials resolve from
`KeyringServerCredentialStore` — the **machine-global** OS keyring, service
`tldw_chatbook.server_credentials`, keyed by `server_id` — *before* the
`[tldw_api]` config fallback, and `_import_legacy_token` writes the config token
into that keyring the first time it is used. The scratch profile's very first boot
had happened with a placeholder token (see the sibling trap below), which was
imported under `server_id = "http://127.0.0.1:8000"` and from then on outranked
every corrected config value. Moving the server to an unused port (`:8010`, a
`server_id` with no keyring entry) made the identical client and identical config
authenticate on the first request.

**What to do.** `TLDW_CONFIG_PATH` isolates the config *file*; it does not isolate
the keyring, and neither does `users_name`. For a live server run: pick a
`server_id` (host:port) no other profile on the machine has used, or clear the
entry first — `KeyringServerCredentialStore().clear_server("<base_url>")`. Clear
the entries you created when you are done; they hold a live credential. And treat
"the wire carries the right key but the server says 401" as evidence that the app
is not reading the credential you edited, not as a server problem.

**The sibling trap that seeded it.** A scratch `[tldw_api]` written with only
`api_key = <real>` comes back from the first boot with the app's own
`auth_token = "default-secret-key-for-single-user"` added beside it — and
`build_runtime_api_client` resolves `auth_token or api_key or bearer_token`, so the
placeholder wins. `config.py` screens *provider* keys for placeholder values
(`resolve_provider_api_key`); the `[tldw_api]` token gets no such check. Write
`auth_token`, not `api_key`, in a scratch profile, and re-read the file after the
first boot to see what the app made of it.

---

## A captured server fixture that drifted from the real response hides a whole class of routing defect (schedules-handoff PR-6 task 6, 2026-09-02)

**What happened.** The Schedules workbench decides whether an automation is
server-executed with `is_server_scoped_owner(row["owner_id"])`, which requires a
`server:` prefix. `Tests/Scheduling/fixtures/server_responses/automation_definition_list.json`
supplies `"owner_id": "server:42"` for every item, and every test passed. The live
server sends `"owner_id": "1"` — a bare user id. `_load_server_automations` stamps
an owner only when the field is **absent** (`if not item.get("owner_id")`), which a
present-but-wrong-shaped value defeats, so against a real server *every* server
automation is classified local. Live consequences, all in one pane: rows render
`[This device] …` while the pane's own notice says "2 automations on the server";
`r` (run now) refuses with the **local** health message instead of dispatching
server-side; `m` (move to this device) refuses with "This automation no longer
exists." Leg (b) of the live plan could not be started at all. The same session
found the sibling of this bug — a synced result carries the server's
`definition_id` while the mark-solved eligibility check looks it up among **local**
ids, so `o` always refuses on exactly the rows it exists for.

**What to do.** A fixture is a *recording*, not a contract: re-capture it from a
running server whenever the code that reads it changes shape, and diff the field
values, not just the field names. When a guard has the form "fill this in if the
server omitted it", ask what happens when the server sends something present but
different — that is the case a fixture written by hand never covers. And whenever
an id crosses the client/server boundary, check which id space every lookup keyed
on it lives in.

---

## An attribute assignment a test can read back is not a render (schedules-handoff PR-6 task 6, 2026-09-02)

**What happened.** The Results tab's unread badge is written as
`pane.label = f"Results ({unread})"` on a `TabPane`, mirroring the Conflicts tab.
`Tests/UI/test_schedules_results_tab.py:408` asserts `str(pane.label) == "Results (2)"`
and passes. Live, with two unread results in the database, the tab bar reads plain
`Results`. Textual 8.2.8's `TabPane` has no `label` attribute or reactive at all —
it stores `self._title`, and the tab text is rendered by a separate `Tab` widget —
so the assignment creates an inert Python attribute on the widget and nothing
repaints. The user-guide had already been written promising the badge. The bug was
pre-existing in the Conflicts badge and was copied verbatim into the new code
because "the Conflicts tab does it this way" was taken as proof it works.

**What to do.** For any widget property you assign to drive a visual, confirm the
framework actually declares it (`'x' in Widget._reactives`, or read the class) —
Python will happily accept an assignment to a name the widget has never heard of.
A test that asserts on the same attribute it set proves the assignment ran, nothing
more; the assertion has to read the rendered surface (or at minimum the widget the
framework really renders from). And "the neighbouring feature does it this way" is
a precedent for the *shape* of the code, never evidence that the shape works.

---

## A detached tmux session renders the app but never feeds it input — and that reads exactly like a hung app (schedules-handoff PR-6 task 6 round 2, 2026-09-02)

**What happened.** Round 1 of the PR-6 live gate ended with a defect filed as "the UI froze after
saving a reminder": the screen stopped repainting, no injected key or click had any effect, the process
sat at 0% CPU while its background DB threads kept logging. It was recorded as uncharacterised because
`py-spy dump` needs root on this machine. Round 2 reproduced it **at boot**, before any interaction,
which made it diagnosable:

1. **The app was never hung.** This app registers `SIGUSR2` for an on-demand all-threads dump
   (`Logging_Config.py`, `faulthandler.register(..., all_threads=True)`) — non-destructive and needing
   no privileges, unlike `SIGABRT` or `py-spy`. The dump showed the main thread idle in
   `asyncio.base_events._run_once` → `selectors.select`, Textual's input thread parked in `select`
   inside `linux_driver.run_input_thread`, and the `WriterThread` idle in `queue.get`. A healthy,
   *idle* loop — not the output-queue deadlock that `Utils/fd_protection.py` documents.
2. **The harness was the cause.** `tmux -L x new-session -d` with no client attached rendered the app
   fine but delivered it nothing: `Ctrl+Q` was ignored, and writing SGR bytes straight to the pane's
   `/dev/ttys018` was ignored too. A stock 15-line Textual app in the *same* tmux server, venv and
   terminal answered an injected key immediately (`GOT KEY: z`), proving tmux, Textual and the OS were
   fine. Starting a client — `tmux -L outer new-session -d 'tmux -L verify6b attach'` — made the very
   next keypress land, and everything worked for the rest of the session. (Round 1 was also detached
   and did work for ~25 minutes, so the delivery is flaky rather than absolutely broken, which is worse:
   it fails partway through a session and looks like a regression in whatever you just did.)

**What to do.** Attach a client before driving the TUI: run `tmux -L <inner> attach` inside a second
tmux server. Before concluding "the app froze", send `SIGUSR2` and read
`<profile>/faulthandler.log` — an idle `run_forever` frame means the app is waiting for input it never
got, not deadlocked. And keep one trivial Textual app around as a harness control; it separates "our app
is broken" from "the harness is not delivering" in under a minute.

---

## A placeholder that reads like a value turns a working button into a phantom defect (schedules-handoff PR-6 task 6 round 2, 2026-09-02)

**What happened.** The reminder create form's Save button appeared inert across two rounds and roughly
a dozen attempts — recomputed SGR coordinates, a preceding mouse-move event, `Enter` on the focused
button, both "Runs on" values, all fields filled. It was nearly filed as a defect. It was working the
whole time: the form was rejecting the submission with `Run At is required for one-time tasks`, because
the `Run at` field *displays* `2026-08-28 09:00` as its **placeholder** — a string identical to the help
text underneath it — so it looks populated while `.value` is empty. Typing a real value made the same
click save instantly. The error line sits directly above the button row and was never in the
`capture-pane | cut -c110-235` windows being used, so two rounds of captures showed a blank modal.

**What to do.** When a control "does nothing", capture the WHOLE pane before concluding anything — an
inline validation message is the single most likely explanation and it is usually rendered next to the
control you are blaming. Read the handler (`_save`, `on_button_pressed`) and check what it validates
before theorising about click delivery. And treat a placeholder that is indistinguishable from a real
value as a finding in its own right: in a terminal there is no greyed-out affordance to tell them apart.

---

## The right escape function depends on the surface, not the app — and one widget can undo another's fix (schedules-handoff PR-6 task 6 round 2, 2026-09-02)

**What happened.** A fix round replaced `rich.markup.escape` with an escape-every-`[` helper because the
detail pane renders through `Content.from_markup`, which consumes ANY `[...]` token — that fix verified
live. The same round explicitly scoped out the results **table**, whose cells go through
`rich.text.Text.from_markup` (lowercase-initial tags only), on the grounds that the bracketed content it
had seen (`[PR-6]`) survives there. Round 2 then found the Automations table silently eating the owner
prefix it had just fixed the routing for: `automation_name_cell` emits
`[http://127.0.0.1:8020] <name>`, and `http` starts with a lowercase letter, so `Text.from_markup`
swallows the whole prefix. Local rows keep `[This device]` (capital T) and server rows show nothing —
one pane where the count line says "1 automation on the server" and every row claims to be local. The
pre-fix fixture value `server:42` would have been swallowed identically, so no fixture shape could have
caught it either.

**What to do.** Enumerate every widget a string reaches and the parser each one uses before deciding a
string is safe; "this particular example survives" is not the same as "this surface is escaped". Be
especially suspicious of generated prefixes — URLs, ids, scope labels — because whether they trip a
markup parser depends on their first character, which is data, not code.

## tmux SGR click columns must be counted in CELLS — BSD `awk index()`/`cut -c` count BYTES (review-set picker, task-28243, 2026-09-02)

**What happened.** Live-verifying the review-set picker, clicks computed from
`capture-pane` output via `awk '{ print index($0, "Dismiss") }'` (and slices
via `cut -c`) reliably "missed": the modal closed but the DB showed the set
never dismissed, three rounds in a row, while the same button worked via
keyboard and `pilot.click`. That read as an app-side hit-region bug and burned
a debugging round with region probes and app-log archaeology. The real cause:
macOS/BSD `awk index()` and `cut -c` count **bytes**, and every multi-byte
glyph earlier in the captured line (`▊` `✓` `—` `·` = 2–3 bytes each) inflated
the computed column, so the SGR click (which addresses screen **cells**)
landed ~6 cells right of the target — on the neighbouring 1fr button, whose
"open" action closed the modal and made the state look untouched. A DB-oracle
column bisect (`col=154` no, `col=150` fired) pinned it.

**What to do.** Compute click columns with a character-aware tool — e.g. pipe
the captured line through python `line.find(needle)` (see
`scratchpad/click.sh` pattern: find by char index, then emit the SGR press +
release from that) — and treat "the click closes the dialog but the action
didn't happen" as a coordinates smell, not an app bug, whenever the row
contains any non-ASCII glyph. Verify a suspected hit-region bug with the
widget's own oracle (DB row, `pilot.click`) before touching app code.

## Region assertions are blind to paint-over — the `*:focus` outline family (media type chooser, task-31221, 2026-09-03)

**What happened.** The Library media type chooser rendered live as an empty
bordered band with its options invisible, while every harness test on it
passed. Two hypotheses died on evidence (the OptionList height math; a
compact-variant focus border) before compositor painted-text probes showed
the truth: the app-global `*:focus { outline: solid }` fallback
(core/_reset.tcss) PAINTS OVER a widget's outermost rows without costing
geometry, and the screen focuses the option-count-height chooser on open —
with a two-option catalogue the outline's top and bottom rows covered every
option. Every geometry assertion (`region`, `scrollable_content_region`)
measured *correct* the whole time, because outlines do not participate in
layout. This was the THIRD widget bitten by that reset (TASK-1160 DataTable,
TASK-2300 SelectOverlay — whose block literally predicts "a bare box with
nothing in it" for two-option compacts).

**What to do.** When a widget looks empty or clipped live but its regions
measure fine, suspect paint-over (outline, overlay, ANSI layer) before
layout, and assert on **compositor painted text**
(`screen._compositor.render_strips()` cropped to the region — see
`_painted_text_in_region` in test_library_media_reader_shell.py) rather
than on geometry. Check the TASK-1160/TASK-2300 blocks in
components/_lists.tcss first: any focusable widget whose outermost rows
carry content needs the same `outline: none` + content-safe-cue opt-out.
Two auxiliary traps from the same hunt: widget-tier CSS (BUNDLED_CSS /
DEFAULT_CSS) loses to app-tier rules regardless of specificity, so an
opt-out fighting an app-tier reset must live at app tier; and a sentinel
edit used to test code provenance MUST be verified to have landed
(`assert old in s` before replace) — an unverified no-op replace produced a
false "app runs foreign code" scare mid-hunt.

## A live report names a SYMPTOM — re-derive the mechanism from code before you fix it (schedules-handoff PR-6 rounds 1-2, 2026-09-02)

**What happened.** A live run is expensive, so its findings arrive with authority: you
watched the thing fail. That authority attaches to the *symptom*, and PR-6's two fix
rounds each shipped a wrong mechanism inferred from a correct symptom.

**Mis-diagnosis 1 — results sorted wrong (D4).** The live observation was exact: the
server emits `2026-09-02T23:25:45.681750Z` while local rows write `+00:00`, and
`list_automation_results` ordered on the raw string, so `Z` (0x5A) vs `.` (0x2E) inverted
neighbouring rows. The obvious reading — "the string comparison is the bug, sort by a real
instant instead" — produced `ORDER BY datetime(created_at)`. That is worse than it looks:
SQLite's `datetime()` **truncates to whole seconds**, and whole seconds is *precisely* the
resolution at which a `Z` and a `+00:00` stamp of the same instant can disagree. The fix
threw away the only information that distinguished the rows, leaving them tied and ordered
arbitrarily by the UUID tiebreak — and a sync pull mirroring a page of results all stamped
in the same second is the normal case, not a corner one. The real fix needed sub-second
precision (`strftime('%Y-%m-%dT%H:%M:%f', …)`, and the docstring records that even `%f` is
only milliseconds against microsecond local writes). Reading the comparison semantics of
the function being reached for would have caught it before the round shipped.

**Mis-diagnosis 2 — bracketed content eaten.** Covered in detail in *"The right escape
function depends on the surface, not the app"* above. Same shape: a correct symptom
(brackets disappearing), a mechanism generalised from one surface (`Content.from_markup`)
to a second one it did not describe (`rich.text.Text.from_markup`, lowercase-tag regex),
and a scope-out justified by "the example I have survives there" rather than by reading
the second parser. Round 2 found the same bug in the pane the round-1 fix had explicitly
excluded.

**Why live findings are especially prone to this.** The symptom is vivid and the run was
costly, so there is real pressure to fix on the spot with the terminal still open. Both
mis-diagnoses were plausible readings of the evidence actually in hand — the evidence just
did not reach as far as the mechanism did.

**What to do.**

- Write the symptom and the mechanism as **separate claims**. The live run establishes the
  symptom; only code establishes the mechanism. A fix round justified by "we saw it fail"
  has skipped a step.
- Before reaching for a stdlib/SQL/framework function as the fix, **read what it actually
  does to your data** — precision, truncation, escaping, ordering. A function that
  *sounds* like the right abstraction is where these mis-diagnoses land.
- When a fix is deliberately **scoped out** of a second surface, state the mechanism-level
  reason ("this parser does not consume that token"), never the example-level one ("the
  string I tried survives"). An example-level scope-out is exactly how round 1 shipped a
  fix that round 2 had to make again.
- Re-derive before re-driving. The cheapest step in a fix round is grepping the call sites
  of the function you are about to change; the most expensive is a second live gate.

## The theme palette matches "Theme: Switch to Textual Light", not the theme id (TASK-31429, 2026-09-04)

**What happened.** Driving a live theme switch through the command palette
for the Console-rail colour check: typing `textual-light` (the id every
config file and test uses) returned zero hits, and the one hit for the
generic query, "Theme: Change Theme", only posts a notification telling
you to search — activating it looked like the palette had silently
dismissed itself (the TASK-397 fast-Down+Enter trap was the first, wrong,
suspect). `ThemeCommandProvider.search` (app.py) builds each hit as
`"Theme: Switch to " + name.replace("_"," ").replace("-"," ").title()` and
fuzzy-matches the QUERY against that Title-Cased string, so the hyphen in
the id is what killed the match.

**What to do.** Query the palette with the rendered command text — `Switch
to Textual Light`, `Switch to Apricot` — narrow to one hit, then Down,
Enter. Two smaller traps from the same session: SGR clicks on the rail's
section-header toggles registered only on the NEXT input event (the header
looked untoggled in the capture taken right after the click, then opened
when the following key arrived), so capture again before concluding a
click was dead; and Ctrl+Shift+Right (expand every rail section) did
nothing when sent through tmux — open sections one at a time instead.

## A host out of POSIX semaphores fails every multiprocessing pool with "No space left on device" — and the disk is fine (media wave 4 PR D, 2026-09-04)

**The incident.** The Task 3 implementer's live import run (two text files through the
Library Import canvas) died with `OSError(28, 'No space left on device')` from
`multiprocessing.Pool`. The data volume was 84% used with 300 GB free. Reproduced outside
the app with nothing but `python -c "import multiprocessing as mp; mp.Lock()"` — same
error. Root cause: macOS's named-semaphore limit was exhausted by 38-40 `pytest`
processes that four other Claude sessions were running in parallel on the same machine;
it stayed exhausted for the rest of the evening. Every live import for PR #2400 was
blocked; the gate, the id set and the per-row receipts were verified in real-screen
app-tests with a stubbed resolver and generator instead, and the PR body says so.

**What to do.**
- When a live step dies with ENOSPC on a disk that is not full, run the one-line
  `mp.Lock()` probe before touching the app; count `pytest` processes with
  `ps -axo command | grep -c '[p]ytest'`.
- Do not kill other sessions' runs to free the semaphores. Either wait for them to finish
  or reboot; until then, substitute app-tests and state the gap in the PR — never report a
  live verification you could not run.

## Two live assessments on one profile reproduce a wedge the app warned you about (media critique #5, 2026-09-04)

**The incident.** Critique #5 ran its two assessment agents in parallel, each launching the real app under its own tmux socket against the same real profile and media DB. The app's startup guard — "Another copy of tldw is already using this profile" — fired in both and both continued. Both then hit the same P0 within minutes: a bulk delete painted `✓ deleted` while the DB row stayed untouched, and the bulk-mutation interlock left Undo, Retry, every row and `s` inert until the process was killed. That is task-31220's storage wedge, which a 24-round single-instance repro had never triggered. Concurrent writers made the failed-write path reachable; the product's dishonest presentation of it is what the critique scored.

**What to do.** Never run two live-app assessments concurrently on one profile: serialize their live phases, or give the second a scratch profile (`TLDW_CONFIG_PATH`, and note the keyring caveat above). When the app's own guard fires, treat it as a real signal and stop. And when a wedge is only reachable under contention, say so in the finding — the trigger is the environment, the presentation is the product's — and file the mechanism, not just the symptom.
