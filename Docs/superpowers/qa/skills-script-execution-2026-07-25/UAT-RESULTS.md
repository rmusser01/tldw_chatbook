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

## Not verified

| AC | Status |
|---|---|
| #3 "Always allow" suppresses the second prompt | **Not run** |
| #4 content change re-prompts | **Not run** |
| #5 Library grant line + Revoke | **Partial** — list and trust header verified; the per-skill grant line and Revoke button not reached |
| #6 context switch does not leave the run blocked | **Not run** |

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
