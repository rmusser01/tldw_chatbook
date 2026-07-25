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

## Not verified

| AC | Status |
|---|---|
| #5 Library grant line + Revoke | **Partial** — the Skills list, trust header and on-disk grant are verified, but the per-skill grant line and the **Revoke script access** button were not reached. Both live inside `#library-skill-trust-panel`, which only renders once a skill is opened in the editor; neither an injected mouse click on the skill row nor Tab-then-Enter got there (Tab moved focus out of the Library screen entirely, back to Console). Finding the actual affordance that opens a skill for editing is the first thing to solve on resume |
| #6 context switch does not leave the run blocked | **Not run** |

The Revoke button remains the single most valuable thing left to verify: it is the
user's only way to withdraw a standing permission, and task-579 exists partly because
nothing tests its wire-up. Its *service* half is covered by unit tests and was
exercised here indirectly (grants written and cleared on disk), but the button →
handler → panel-refresh path is still unproven in a running app.

## Findings

### F1 — keyring convenience never auto-unlocked → **fixed** (task-624, PR #883 merged)

### F2 — `local-llm` provider is unusable from its documented config → filed as task-625

`chat_with_local_llm` reads a **top-level** `local-llm` settings key; the provider's
config lives at `api_settings.local-llm`, which is where the app's own documented
example puts it. Every sibling local-provider function in the same file resolves via
`settings["api_settings"]` correctly. There is no config workaround — `load_settings()`
does not preserve arbitrary top-level sections, verified. Surfaced in the Console as
*"Agent run failed: provider returned HTTP 502 … configuration error"*. Blocked this
UAT until it was switched to the `llama_cpp` provider.

### F3 — the trust posture header is accurate and discoverable (positive)

Exercised across marker/manifest mismatch, manifest-without-keys, and ready.

## Driving notes for whoever resumes

- Clicking a card button moves focus off the composer; click the composer line
  before typing the next prompt or the keystrokes go nowhere.
- Give the model room: this Gemma build spends heavily on `reasoning_content`, and
  at `max_tokens = 512` it produced "No response was generated." 3000 works.
- A vague skill body makes the model delegate via `spawn_subagent`, which then got
  stuck. A body that names the tool and its exact arguments — and says not to spawn —
  produces a direct call reliably.
