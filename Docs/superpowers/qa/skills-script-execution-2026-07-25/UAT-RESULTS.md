# UAT — skill script execution (task-579)

Date: 2026-07-25 · Under test: `dev` @ `1e706169f` (after #871/#873/#875/#883)
Driver: real app in tmux, isolated `qa579` profile, live llama.cpp (Gemma) on `:9099`.
Only the model is "simulated" — every gate, card and subprocess is real.

## Verified

### AC#1 — the confirm card describes what will run ✅

Raised by a real agent tool call, rendered:

```
An agent wants to run a script from a skill:
demo-runner — scripts/hello.py (runs with /usr/bin/python3)
It runs with a scrubbed environment in a temporary folder (not the skill's own
folder); only its output comes back.
   Allow once    Always allow this skill       Deny
```

Skill name, script path and the resolved mechanism are all correct and match what
the service would actually execute. Evidence: `uat-ac1-confirm-card.txt`.

### AC#2 — Allow runs it, Deny does not ✅

**Allow once** → `⚙ run_skill_script → exit_code: 0`, and the assistant replied with
the script's genuine stdout, `QA579 HELLO FROM THE SCRIPT`. That string exists
nowhere but the script, so the subprocess really ran. Evidence: `uat-ac2-allow-ran.txt`.

**Deny** → a second invocation raised a fresh card; clicking Deny left the
transcript with still exactly **one** occurrence of the script's output — i.e. the
denied run never executed. Evidence: `uat-ac2-deny.txt`.

### Incidental positives

- The `request_id` handshake works: two sequential rounds each resolved their own
  card, and clicking a button had the intended effect rather than being silently
  dropped.
- `run_skill_script` is genuinely reachable by a **spawned sub-agent**, not just the
  primary agent — the model's first attempt delegated via `spawn_subagent`. That is
  the all-agents caller scope working as designed.

### AC#3 — "Always allow this skill" suppresses the next prompt ✅

Clicking **Always allow this skill** ran the script AND persisted the grant, pinned
to the skill's fingerprint digest:

```json
{"demo-runner": "866a623398fac6f26c63b8ee1ab5b42fac0a60b1850d4a75ca8e9f2f9965489d"}
```

The next invocation then ran with **zero** cards shown — two script outputs in the
transcript, the second at `exit_code: 0`, no prompt. Evidence: `uat-ac3-always-allow.txt`.

### AC#4 — a content change invalidates the grant and re-prompts ✅

The strongest result of the pass, because it exercises the whole point of pinning the
grant to a digest:

| Step | trust status | grant |
|---|---|---|
| after "Always allow" | `trusted` | **True** |
| after editing `scripts/hello.py` | `quarantined_modified` | **False** |
| after the user re-reviews and re-approves | `trusted` | **still False** |

Re-approving trust does **not** silently restore the standing permission. The next run
then raised a fresh card, and allowing it executed the **new** content
(`QA579 MUTATED SCRIPT v2`), confirming the run is not serving a stale copy.
Evidence: `uat-ac4-mutation-reprompt.txt`.

### AC#5 — Library grant line and Revoke ✅

Opening a skill in the in-canvas editor (clicking the skill row button) and scrolling
to the trust panel shows, with a grant in force:

> Scripts: this skill may run its bundled scripts without asking. Any change to its
> files cancels this automatically.

Clicking **Revoke script access** flipped it to:

> Scripts: you are asked to confirm each time this skill runs a script.

and the on-disk grant store went to `{}`. The button → handler → panel-refresh path is
therefore proven end to end, which was the single most valuable untested path in this
feature: it is the user's only way to withdraw a standing permission.
Evidence: `uat-ac5-grant-line.txt`, `uat-ac5-after-revoke.txt`.

### AC#6 — a context switch does not leave the run blocked ✅

With a confirm card pending, switching session cleared the card and released the run
within ~45s — well inside the 120s confirm timeout, so the worker is not left blocked.
Evidence: `uat-ac6-context-switch.txt`.

Two observations worth recording rather than filing:
- The clear is not instantaneous. At 24s the card was still rendered; by 45s it was
  gone. Anyone re-testing should allow for that rather than concluding it failed.
- Opening a **new tab** (Ctrl+T) with a card pending did *not* clear it — the card
  followed into the new tab and remained actionable, and clicking Deny there resolved
  the original run correctly. That is consistent with the code, which wires the
  deny-on-context-change into `switch_session` specifically. It is defensible (the
  decision still belongs to the user and still fails closed), but a card that follows
  you into a different conversation is worth a UX look.

## All acceptance criteria verified

Every behavioural criterion (#1-#6) is now verified against the running application
with real subprocesses, and #7 (evidence) is this directory.

## Superseded — earlier "not verified" section

| AC | Status |
|---|---|
| #5 Library grant line + Revoke | **Partial** — the Skills list, trust header and on-disk grant are verified, but the per-skill grant line and the **Revoke script access** button were not reached. Both live inside `#library-skill-trust-panel`, which only renders once a skill is opened in the editor; neither an injected mouse click on the skill row nor Tab-then-Enter got there (Tab moved focus out of the Library screen entirely, back to Console). Finding the actual affordance that opens a skill for editing is the first thing to solve on resume |
| #6 context switch does not leave the run blocked | **Not run** |

