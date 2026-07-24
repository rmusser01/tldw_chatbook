# Console skills + agent-runtime — Phase-2 live gate (Task 14)

## Scope

- **Branch:** `claude/skills-spec`, on top of Tasks 7–13 (slash resolver, picker, dispatch wiring,
  render-vs-persist substitution, `SkillToolProvider`, spawn-wired executor + allow-list intersection,
  catalog dedupe + per-run owner-map cache).
- **Target:** the native Console's TWO skill invocation surfaces — `/skill-name` slash dispatch
  (Tasks 7–10) and agent-tool discovery via `find_tools`/`load_tools` (Tasks 11–13) — driven live
  against a real local model (`llama.cpp` @ `127.0.0.1:9099`,
  `Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf`), with `[console] agent_runtime = true`.
- **Recipe:** textual-serve, playwright bundled Chromium, viewport 2050×1240, `https://**` route-abort,
  `body.-first-byte` gate. Scripts: scratchpad `serve_qa_autounlock.py` (see "Live-gate finding" below
  for why this replaces the plain `serve_qa.py` + Library-click unlock recipe) + `cap.py` (patched this
  session to move-then-click with 15 interpolation steps, kept even after the finding above turned out
  to be unrelated to mouse-move fidelity). CSS prebuilt (`tldw_chatbook/css/build_css.py`) before capture.
  Port 9093.
- **Profile:** fresh `qa-home-skills-agent14` (scratchpad), combining the Task 6 gate's real-skill trust
  seeding recipe with the Task 8 agent-runtime provider wiring: `PYTHON_KEYRING_BACKEND=
  keyring.backends.fail.Keyring` (both seed script and served app), `[console] agent_runtime = true`,
  `provider = "Llama_cpp"` pointed at `127.0.0.1:9099`, splash off, onboarding `first_send_completed`.

## Corpus (17 skills — real + synthetic, labeled)

- **14 real `obra/superpowers` 6.1.1 skills** (unmodified upstream content, imported via
  `LocalSkillsService.import_skill` — the real directory-name import path): `brainstorming`,
  `dispatching-parallel-agents`, `executing-plans`, `finishing-a-development-branch`,
  `receiving-code-review`, `requesting-code-review`, `subagent-driven-development`,
  `systematic-debugging`, `test-driven-development`, `using-git-worktrees`, `using-superpowers`,
  `verification-before-completion`, `writing-plans`, `writing-skills`.
- **2 SYNTHETIC demo skills** (labeled — not from the real corpus), built for fast, deterministic live
  scenarios: `shout` (`context: inline` — "Reply to the following in ALL CAPS…") and `fork-demo`
  (`context: fork` — "You have no memory of any earlier conversation…").
- **1 SYNTHETIC quarantine skill**: `quarantine-demo`, imported but deliberately never approved
  (stays `quarantined_added`/needs-review) — exists solely so the needs-review-hint scenario (6) has a
  real blocked skill to prefix-match against.
- **17 > `DIRECT_DISCLOSE_THRESHOLD` (8)** — progressive disclosure (`find_tools`/`load_tools`) engages
  for the agent-tool surface; confirmed live in scenario 5's own step log (see below).

Seeding sequence (scratchpad `seed_home_skills_agent14.py`): `bootstrap_trust` on an empty store first
(generation 1) → import all 17 → approve all 16 real+synthetic-demo skills headlessly (`unlock_with_passphrase`
→ `capture_review` → `trust_reviewed_snapshot`) → **post-approval direct file mutation** of `brainstorming`'s
`SKILL.md` (append one comment line, bypassing `update_skill` entirely) → live re-verified:
`trust_status=quarantined_modified`, `trust_reason_code=skill_modified`, `trust_blocked=True`. Final state:
**15 trusted / 2 blocked (`brainstorming` modified, `quarantine-demo` added) / 17 total managed.**

## Live-gate finding: Library Skills detail editor breaks the app's tab bar (workaround shipped, not a Skills-code bug)

**FIXED (commit `81700962`)** — see "Gate fix 2" in `.superpowers/sdd/task-14-report.md` for the full
root-cause writeup. Summary: not a client-side/textual-serve quirk and not scoped to the `Body`
`TextArea` as originally suspected — root cause was `Input` (any of the editor's Inputs, or the list
view's Filter input) leaking `App.mouse_captured` when the Library screen recomposes
(`self.refresh(recompose=True)`) while a click/selection is still in flight on it. Fixed generically in
`BaseAppScreen.refresh` (`tldw_chatbook/UI/Navigation/base_app_screen.py`), so it also covers the
identically-shaped, pre-existing Prompts/Notes in-canvas editors. The workaround below (kept verbatim
as this gate's historical record) is now obsolete — a plain Library-UI unlock-and-click-back-to-Console
flow works post-fix — but re-running it was not required since the regression is covered by a headless
test (`Tests/UI/test_library_skills_canvas.py::test_opening_skill_editor_does_not_break_tab_bar_click_activation`)
and the fix touches no visual layout.

While driving the intended recipe (unlock trust via the Library Skills editor's passphrase modal, then
click back to Console), the served app's **top-level tab bar stopped responding to click activation for
the remainder of that process** — hover/hbox focus (an underline) still rendered on Home/Console/Library,
but no screen switch ever happened, for any tab, from that point on. Isolated via 8+ targeted probes:
- Reproduces with **zero** modal/passphrase interaction — merely opening ANY skill's detail view (tested
  `brainstorming` and the tiny synthetic `shout`) and then clicking a tab bar item reproduces it.
- Does **not** reproduce from the Skills **list** view (confirmed clean nav Library→Console with no
  detail visited).
- Not fixed by: longer settles, `Escape`, mouse-move interpolation before click, clicking a different tab
  first, clicking "‹ Back to list", or repeating the click up to 8×.
This is very likely a genuine textual-serve/client-side focus-capture bug scoped to the Skill detail
screen's `Body` `TextArea` (out of scope for this Skills feature to fix) — flagged here as a real product
finding, not fabricated around. ~~(Superseded — see "FIXED" note above: the actual root cause was
`Input`'s missing mouse-capture-release-on-removal, not the `TextArea`, and it was fixed in this
branch's reachable code.)~~ **Workaround used for this gate**: `serve_qa_autounlock.py` monkeypatches
`TldwCli.on_mount` to call the real `SkillTrustService.unlock_with_passphrase` directly on the same
`app_instance.local_skill_trust_service` the Console's `get_context`/`execute_skill`/`SkillToolProvider`
paths already read — exercises the identical trust-gated code paths without ever touching the buggy
Library click path. Confirmed via `_autounlock_debug.log`: `patched on_mount CALLED` →
`AUTOUNLOCK: skill trust unlocked at on_mount`.

## Captures

1. **`skills-bare-list-2026-07-14.png`** — bare `/skills` → real, sorted skill names (15 trusted; both
   blocked skills correctly absent): `dispatching-parallel-agents`, `executing-plans`,
   `finishing-a-development-branch`, `fork-demo`, `receiving-code-review`, `requesting-code-review`,
   `shout`, `subagent-driven-development`, `systematic-debugging`, `test-driven-development`,
   `using-git-worktrees`, `using-superpowers`, `verification-before-completion`, `writing-plans`,
   `writing-skills`.
2. **`shout-inline-command-2026-07-14.png`** — `/shout hello world` (trusted, `context: inline`):
   `User /shout hello world` → `Tool skill shout → driving this turn` → `Assistant HELLO WORLD`.
   **DB evidence** (`tldw_chatbook_ChaChaNotes.db`, `messages` table): the persisted user row is the
   **literal raw command** `/shout hello world` — never the rendered body — confirming render-vs-persist
   (Task 10): the store keeps what the user typed; only the ephemeral provider payload for that turn was
   ever substituted. The assistant's `HELLO WORLD` reply is itself indirect proof the **on-wire** payload
   *was* the rendered instruction ("Reply to the following in ALL CAPS…: hello world") — a raw slash
   command sent verbatim to the model would not produce that completion.
3. **`fork-demo-fact-established-2026-07-14.png`** + **`fork-demo-clean-context-2026-07-14.png`** —
   real conversational clean-context proof, not a synthetic no-history case: turn 1 states "My favorite
   color is teal…", the model correctly answers "Noted! Your favorite color is teal." (proving it DOES
   have that history normally); turn 2, `/fork-demo What is my favorite color?` (trusted, `context: fork`)
   → `Tool skill fork-demo → driving this turn` → `Assistant Your favorite color is unknown.` — the fork
   skill's rendered turn genuinely drops all prior history (leading `system` message only, per
   `_apply_skill_substitution`'s fork branch). **DB evidence**: `messages` table shows the raw
   `/fork-demo What is my favorite color?` persisted (not the rendered "you have no memory…" body) with
   the assistant's real reply directly beneath it.
4. **`edited-skill-refuse-2026-07-14.png`** — `/skills brainstorming …` on the post-approval-mutated
   skill → **exact** `SKILL_UNTRUSTED_REFUSE` transcript row: `Skill "brainstorming" isn't trusted
   (skill_modified) — review and approve it in Library ▸ Skills before running it.` — no Tool marker, no
   Assistant reply, draft left untouched in the composer. Reason code `skill_modified` matches the live
   `SkillTrustService.status_for_skill` re-derivation exactly (verified via the seed script's own printed
   evidence — see corpus section above).
5. **`needs-review-hint-2026-07-14.png`** — same screenshot, second row: `/skills quara …` (a prefix that
   matches **only** the never-approved `quarantine-demo`, not any trusted skill) →
   `1 matching skill(s) need review in Library ▸ Skills before running.` — the
   `CONSOLE_SKILL_NEEDS_REVIEW_HINT_TEMPLATE` copy, correctly distinct from capture 4's exact-match refuse
   row directly above it in the same transcript.
6. **`agent-tool-discovery-2026-07-14.png`** + **`agent-tool-discovery-resumed-2026-07-14.png`** — a
   plain natural-language message ("Find a skill that can shout, load it, and use it on: hello") with
   >8 catalog engaged progressive disclosure. Live TOOL markers: `⚙ shout → HELLO` then
   `⚠ step budget exhausted`; `System Agent run stuck: step budget exhausted.` The `find_tools`/
   `load_tools` steps themselves are correctly quiet (no marker — by design, see
   `_QUIET_STEP_TOOLS`) but are fully visible in the raw step log (below). The resumed capture (a fresh
   process reconnecting to the same persisted conversation) shows the exact same TOOL markers
   re-derived from `AgentRunsDB` via `resume_marker_messages` — byte-identical live vs. resumed.
   **FIXED (commit `6736835d`)** — see "Gate finding 1: FIXED" below;
   `agent-tool-discovery-fixed-2026-07-14.png` + `agent-tool-discovery-fixed-resumed-2026-07-14.png`
   are the post-fix re-captures of the same prompt, now ending in a real assistant answer.

## Gate finding 1: FIXED — discovery-run step budget (commit `6736835d`)

Scenario 6's `stuck` outcome was a budget-shape bug at the Console bridge's config-assembly site, not
a Skills-code defect (as the original caveat suspected): a discovery-heavy round trip (`find_tools` →
`load_tools` → skill call → final answer) needs exactly 10 primary-loop steps at the floor
(`STEP_MODEL`+`STEP_TOOL_CALL`+`STEP_TOOL_RESULT` per round × 3 rounds + 1 final model turn), but
`ConsoleAgentBridge.run_reply` built its `AgentConfig` with a bare engine-default `RunBudget()`
(`max_steps=8`) — one round short by construction whenever the >8-skill catalog forces the find/load
path. Fix: a Console-site `CONSOLE_RUN_BUDGET` override (`max_steps=16`, `max_wall_seconds=480.0` —
2× the engine defaults: ~60% headroom over the counted 10-step floor, plus matching wall-clock for
the slow local model). The engine's own `RunBudget` defaults are unchanged and stay pinned at 8 by
`Tests/Agents/test_agent_models.py::test_budget_defaults`. Regression tests
(`Tests/Chat/test_console_agent_bridge.py`): a scripted run with the gate's exact 5-turn shape
completes `done` (RED-verified `stuck` before the fix), and the persisted per-run budget is pinned
`>8 / >=16 / >240s`.

**Post-fix live re-capture** (same recipe verbatim — `serve_qa_autounlock.py` port 9093,
`qa-home-skills-agent14` profile, llama.cpp @9099, generous 60s/turn waits):
`agent-tool-discovery-fixed-2026-07-14.png` shows the same prompt now ending
`Assistant HELLO` + `Tool ⚙ shout → HELLO` — no budget-exhausted marker, no stuck banner.
**AgentRunsDB evidence**: primary run `7f87d5545c6341c09f7796edb2831446` — persisted budget
`{"max_steps": 16, "max_wall_seconds": 480.0, ...}`, **`status=done`**, `result=HELLO`, steps
`find_tools({"query":"shout"})` → `load_tools({"ids":["skill:shout"]})` → `shout({"args":"hello"})`
→ `tool_result HELLO` → final model turn `HELLO` (10 steps total — exactly the counted floor);
child run `511a5dfabaac450b9e34edf79de448e6` (`parent_run_id` = the primary, `status=done`,
`result=HELLO`). **ChaChaNotes evidence**: conversation `96c88262…` persists the raw user prompt
plus a real `assistant` row `HELLO`. `agent-tool-discovery-fixed-resumed-2026-07-14.png` (fresh
process, same persisted conversation) re-derives the identical transcript + TOOL marker from
`AgentRunsDB`. The pre-fix captures above are kept as the finding's historical evidence.

## Service/DB evidence (not pixels alone)

**AgentRunsDB** (`agent_runs.db`) — 9 total runs across this gate: 7 primary + 2 subagent. The two
subagent runs are both the live discovery scenario's own skill-tool spawn (ran twice across two attempts
of scenario 5, both reproduced identically):

```
id            = 14fababce0564d4a8a5307c1c975019e   (primary, conversation "Find a skill that…")
status        = stuck
steps (raw)   =
  0 model      {"name":"find_tools","arguments":{"query":"shout"}}
  1 tool_call  find_tools({"query":"shout"})
  2 tool_result find_tools -> "skill:shout — shout: Reply to the user's message in ALL CAPS. …"
  3 model      {"name":"load_tools","arguments":{"ids":["skill:shout"]}}
  4 tool_call  load_tools({"ids":["skill:shout"]})
  5 tool_result load_tools -> "loaded: shout"
  6 model      {"name":"shout","arguments":{"args":"hello"}}
  7 tool_call  shout({"args":"hello"})
  8 tool_result shout -> "HELLO"
  9 error      "step budget exhausted"

id            = e7512215e2e04596a2f330b5fc4be6d6   (subagent)
parent_run_id = 14fababce0564d4a8a5307c1c975019e   <- budget-counted, parent-linked
agent_kind    = subagent
status        = done
task          = "Reply to the following in ALL CAPS and nothing else, no explanation: hello"
result        = "HELLO"
```

This is the complete, real live pipeline: `find_tools` → `load_tools` → skill-tool invocation → spawn a
budget-counted, `parent_run_id`-linked sub-agent run (`_BridgeSkillRunner.run`, Task 12) → sub-agent
result folded back into the parent's step log — all exactly as designed, with the parent run only
failing to compose a final natural-language wrap-up before its step budget ran out (see caveat below).

**Skills index + trust manifest** (`tldw_chatbook_skills.json` / `trust/`): 17 skills on disk, matching
the corpus list above exactly; live-read post-mutation trust status for `brainstorming`:
`trust_status=quarantined_modified trust_reason_code=skill_modified trust_blocked=True`.

**ChaChaNotes messages table**: cross-checked for scenarios 2 and 3 above — every persisted user row is
the literal raw slash command, never a rendered body, for every skill-triggered turn in this gate.

## Caveats (honest, not glossed over)

- ~~**Library Skills detail editor tab-bar-click bug**~~ **FIXED (commit `81700962`)** — see
  "Live-gate finding" above and "Gate fix 2" in `.superpowers/sdd/task-14-report.md` for the full
  root-cause writeup (an `Input` leaking `App.mouse_captured` across a same-screen recompose, fixed
  generically in `BaseAppScreen.refresh`). The original observation (workaround shipped, out of scope,
  suspected `TextArea`/client-side) stands as the finding's historical record; the suspicion turned out
  to be off by one widget (`Input`, not `TextArea`) but the finding itself was real and reproducible.
- ~~**Scenario 5's primary run ends `stuck` (step budget exhausted), not a clean final answer.**~~
  **FIXED (commit `6736835d`)** — root-caused to the Console bridge assembling its `AgentConfig` with
  the bare engine-default `RunBudget()` (`max_steps=8`), one round short of the 10-step floor a
  discovery run needs by construction. See "Gate finding 1: FIXED" above for the fix, regression
  tests, and the post-fix live re-capture (`status=done`, real assistant answer, DB-verified). The
  original observation (pipeline correct, only the wrap-up turn missing; reproduced identically across
  two attempts, second sub-agent run `b6bd1c56…`) stands as the finding's historical record.
- **`t14-01`/`sess1b` naming**: the composer draft is intentionally left un-cleared after the bare
  `/skills` list and after a refusal (matches the documented "draft untouched" behavior — confirmed live,
  not a capture artifact).
- Live captures used `Control+u` to clear the Console composer between turns (discovered mid-session that
  Enter-to-submit does **not** clear prior text — a UI-harness detail, not a product bug: a real user
  would see their own typed text and clear it manually or overtype).

## Verification

- **Sweep**: `Tests/Skills Tests/Agents Tests/Chat Tests/Library Tests/UI/test_console_skill_commands.py
  Tests/UI/test_console_skill_picker.py Tests/UI/test_library_skills_canvas.py
  Tests/UI/test_console_native_chat_flow.py Tests/UI/test_library_shell.py` = **2000 passed, 1 failed,
  69 skipped** (1516.79s). The 1 failure —
  `test_library_shell_export_registry_failure_warns_it_wont_appear_in_artifacts` — is the exact
  order/global-state-dependent flake already documented in the Task 6 gate's own README
  (`Docs/superpowers/qa/skills-library-2026-07/README.md`, "Verification" section): re-confirmed here
  passing in isolation (`pytest <test> -q` → 1 passed). Net: **no NEW failures**, gate is green.
- Every population/count/status claim above cross-checked against the real on-disk skills index + trust
  manifest + `AgentRunsDB` + `ChaChaNotes` messages table, not screenshots alone.
- No PR opened — this is the Phase-2 gate checkpoint only, per the program's approval-gate convention.
