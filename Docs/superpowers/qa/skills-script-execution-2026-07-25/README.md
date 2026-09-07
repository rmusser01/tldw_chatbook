# Live TUI verification — skill script execution (task-579)

Date: 2026-07-25 · Branch under test: `dev` @ `4e5e78278` (after #871, #873, #875)

## Why this pass exists

Trust-gated skill script execution shipped with unit, widget and end-to-end
coverage, but **nothing had ever been driven in a running application**. Two of
its surfaces are security controls whose failure mode is silence:

- the in-chat confirm card, including the per-round `request_id` handshake — if
  the id is not echoed correctly the resolve is *silently dropped by design*,
  the card appears inert, and the agent's worker thread blocks for 120s;
- the Library ▸ Skills "Revoke script access" button, the user's only way to
  withdraw a skill's standing permission.

A broken wire-up on either would leave every automated test passing.

## Harness

Isolated profile so QA never touches the live database, per the repo's `verify`
skill:

```bash
TLDW_CONFIG_PATH=<scratch>/config.toml   # [general] users_name = "qa579"
PYTHONPATH=<worktree>                    # the venv's editable install points at
                                         # the MAIN checkout, which sits on an
                                         # unrelated branch
tmux -L qa579 new-session -d -x 235 -y 52 '.venv/bin/python -m tldw_chatbook.app'
```

Scripts in this directory's sibling scratch (`seed3.py`) build the profile:
two trusted skills, each with a bundled script, plus a bootstrapped trust store.

### Harness lessons (cost real time — write them down)

1. **Seed through the service API, not the filesystem.** Writing skill
   directories by hand produces `Skills (0)`: the Library list is driven by the
   index file (`tldw_chatbook_skills.json`), which only `create_skill` /
   `import_skill_*` maintain.
2. **Match the app's marker-store construction exactly.** `app.py` builds the
   generation marker via `build_skill_trust_marker_store_with_fallback(...)`,
   which is **keychain-first**. Seeding with a `FileSkillTrustGenerationMarkerStore`
   leaves the app reading a different marker and reporting
   *"Skill trust needs to be set up again after an update"* — a confusing
   dead-end that is an artefact of the harness, not a product bug.
3. **Redirecting stderr blanks the pane.** Launch without `2>file`; capture with
   `tmux capture-pane -p`.
4. **`PYTHONPATH` must point at the worktree.** Otherwise `python -m` resolves
   `tldw_chatbook` from the main checkout and you test the wrong code — the
   symptom is `AttributeError: no attribute 'describe_skill_script'`.

## Findings

### F1 — Keyring convenience never auto-unlocks (filed as task-624)

**Confirmed defect.** `SkillTrustService.unlock_from_keyring_convenience()`
exists and is unit-tested, but has **zero callers in the application**
(`grep -rn 'unlock_from_keyring_convenience' tldw_chatbook/` → only its own
definition; the only other hits are two tests calling it directly).

`app.py:4636-4646` wires a `key_cache` **and** hardcodes
`keyring_convenience_enabled=False`, then never attempts the cached-key unlock.

User-visible effect: a user who enabled keyring convenience — whose derived keys
are sitting in the OS keychain for exactly this purpose — still sees *"Skill
trust is locked for this session"* on every launch and must unlock manually.

This is the archetypal defect unit tests cannot catch: the method passes its own
tests **because those tests supply the call site that production is missing**.

### F2 — Trust posture messaging is accurate and discoverable (positive)

The adaptive trust header behaved correctly across three distinct states induced
during setup, each with the right remediation offered inline:

| Induced state | Header shown | Action offered |
|---|---|---|
| Marker/manifest mismatch | "Skill trust needs to be set up again after an update." | Set up skill trust / Reset |
| Manifest present, no keys | "Skill trust is locked for this session." | Unlock / Reset |
| Fully bootstrapped | *(no warning; skills listed)* | — |

This is the Spec-1 discoverability goal working in a real app, not a mock.

## Status of the acceptance criteria

| AC | Status |
|---|---|
| #1 confirm card shows skill/script/mechanism/args | **Not verified** — blocked behind F1 (trust must be unlocked before a run can reach the card) |
| #2 Allow once runs / Deny does not | **Not verified** |
| #3 Always allow suppresses the second prompt | **Not verified** |
| #4 content change re-prompts | **Not verified** |
| #5 Library grant line + Revoke | **Partially** — the Skills list and trust header render correctly; the per-skill grant line and Revoke button sit inside the skill editor panel, not yet reached |
| #6 context switch does not leave the run blocked | **Not verified** |
| #7 evidence captured | This document + the harness scripts |

### Update — F1 is fixed (task-624, PR #883)

The blocker below is resolved. `_try_cached_unlock()` now sits behind every
lockedness decision, so a profile with cached keys comes up **"Skill trust:
ready."** with no unlock click. Re-verified live in the same profile; capture in
`after-fix-trust-ready.txt`.

**A resuming session starts here.** The harness is built and the trust friction
is gone; what remains is driving the agent-initiated card flows.

#### Resume recipe

```bash
SCRATCH=<scratchpad>/qa579           # config.toml + seed3.py live here
WT=<worktree on the branch under test>
tmux -L qa579 new-session -d -x 235 -y 52 \
  "cd $WT && TLDW_CONFIG_PATH=$SCRATCH/config.toml PYTHONPATH=$WT \
   /path/to/.venv/bin/python -m tldw_chatbook.app"
```

Profile `qa579` (at `~/.local/share/tldw_cli/qa579`) already holds two trusted
skills — `demo-runner` (`scripts/hello.py`, prints a recognisable marker) and
`grant-demo` (`scripts/mark.py`, for the grant/revoke flows). Re-run `seed3.py`
to rebuild from scratch. A local llama.cpp server on `:9099` (Gemma) is
confirmed emitting `finish_reason: tool_calls`, so the agent path is drivable;
point the Console at the `local-llm` provider already in the QA config.

To reach the card, ask the Console agent to run `demo-runner`'s script. Then
work AC#1-#4 and #6 in order — each needs one agent turn plus a capture.

**This pass is incomplete and should not be read as a sign-off.** It produced a
reusable harness and one confirmed defect; the agent-driven card flows (#1-#4,
#6) remain to be driven, and are the reason task-579 stays open.

A local llama.cpp server on `:9099` (Gemma, `finish_reason: tool_calls`
verified) is available to drive the agent path when this resumes.
