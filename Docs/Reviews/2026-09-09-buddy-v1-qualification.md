# Chatbook Buddy v1 qualification — 2026-09-09

TASK-32108 remains **In Progress**, pending root-owned native integration. This report qualifies the merged implementation through isolated database/service tests and headless production Textual application journeys. It does not claim physical native-terminal, microphone, audible playback or real provider verification.

## Source and isolation

- Branch: `codex/buddy-v1-live-qualification`.
- Production commit: `5655c4820733d24754f872c21cc6196d83bf1504`; no production files changed for this qualification.
- Contract: [ADR-139](../../backlog/decisions/139-independent-buddy-conversation-and-workspace-bindings.md). The Backlog plan's incorrect ADR filename was corrected with the CLI.
- Environment: macOS 26.5.2 arm64, Python 3.12.11, Textual 8.2.8, main repository `.venv`, worktree `PYTHONPATH=.:packages/tldw_profile_core/src`.
- Repository pytest bootstrap redirects HOME, USERPROFILE, XDG data/config and TLDW_CONFIG_PATH before application imports, forces the null keyring, and disables model downloads. Mounted journeys additionally create fresh per-test scratch HOME/data/config directories. The capture harness asserts both null-keyring selection and that the actual ChaChaNotes path lies inside its scratch profile.
- Provider-dependent existing tests use controlled gateways and barriers. The new captures leave the provider unconfigured, honestly displaying the setup-required state. No provider keys, live network model, microphone recording or audio playback were used. No GUI automation was performed by this qualification worker.
- [Capture harness](../../Tests/UI/test_buddy_v1_qualification_capture.py) uses real `TldwCli.run_test`, production stylesheet, local services, SQLite, built-in artwork and Textual controls at 150×45. `NO_COLOR` is removed for readable color exports. This is a headless pilot journey; direct Select assignment/Button.press is not evidence of physical hit testing.

## Verification results

The initial scoped gate passed **142 tests, 3 warnings in 85.81s**:

```sh
PYTHONPATH=.:packages/tldw_profile_core/src PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring .venv/bin/pytest \
  Tests/Persona_Buddy/test_buddy_library.py \
  Tests/Persona_Buddy/test_buddy_management_coordinator.py \
  Tests/Persona_Buddy/test_buddy_interaction.py \
  Tests/Persona_Buddy/test_buddy_workspace_coordinator.py \
  Tests/Persona_Buddy/test_buddy_inbox.py \
  Tests/Workspaces/test_workspace_assistant_defaults.py \
  Tests/Workspaces/test_agent_provisioning.py \
  Tests/UI/test_buddy_management_journey.py \
  Tests/UI/test_buddy_management_modal.py \
  Tests/UI/test_buddy_entry_points.py \
  Tests/UI/test_buddy_workspace_modal.py \
  Tests/UI/test_buddy_conversation_modal.py \
  Tests/UI/test_buddy_conversation_layout.py -q
```

The atomic Persona/default assignment gate passed **33 tests, 1 warning in 13.38s**:

```sh
PYTHONPATH=.:packages/tldw_profile_core/src .venv/bin/pytest Tests/Chat/test_console_persona_assignment.py -q
```

Actual commands use `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/pytest`; `.venv/bin/pytest` above is shorthand for that interpreter. Logs and complete capture set are under `/private/tmp/chatbook-buddy-v1-qualification-32108/` (`targeted-tests.log`, `persona-assignment-tests.log`, `capture-tests-final.log`). The final rendered two-profile gate passed **2 tests, 3 warnings in37.30s**. It runs the capture harness with `TLDW_BUDDY_QUALIFICATION_OUTPUT=/private/tmp/chatbook-buddy-v1-qualification-32108`; the environment variable is optional and ordinary test runs write only under their temporary fixture. Total scoped qualification: **177 passed**, with no skipped or failed cases in the final runs. Warnings are the installed requests dependency-version warning and existing `datetime.utcnow` deprecation.

The new harness passes Ruff and formatter checks, compiles, and adds no whitespace errors. Selected production source files are byte-identical to HEAD; the [source manifest](artifacts/buddy-v1-32108/source-manifest.json) binds their SHA-256 and the final harness to the evidence.

### Refresh after dev advanced

The qualification branch was rebased onto `2e3389e694e93592a1c66e5c3416bf29a1057d6c`, which adds the merged TTS runtime/recovery work. The two rendered fresh/upgrade journeys plus `Tests/TTS/test_buddy_guarded_speech.py` passed **13 tests, 3 warnings in 39.04s** on that updated source. [Refresh evidence](artifacts/buddy-v1-32108/after-rebase-verification.json) records the exact command, source hashes and log. Those two repeated journeys are not counted again as unique coverage in the earlier 177-test total. The original screenshots remain tied to the earlier source snapshot; refreshed raw captures are retained separately in `/private/tmp/chatbook-buddy-v1-qualification-after-rebase`.

## Coverage and limits

| Outcome | Evidence | What this proves / limit |
| --- | --- | --- |
| Buddy without Persona | Fresh app management picks built-in Pixel Migu; library tests resolve independent owner without Persona service; capture asserts graph Persona ID null and unchanged Persona list | Actual installed artwork with independent ownership. Built-in character seeding is separate from creating an assistant Persona. |
| Fresh setup | Scratch DB does not exist before constructing app; real schema70 creation, Console management, Home overlay and exact conversation interaction | Fresh profile with first-run wizard already marked complete and splash disabled; not a walkthrough of the first-run wizard itself. |
| Upgrade | Capture prepares a real schema69 DB with legacy Persona visual graph, closes it, then boots current production app; schema70 and graph equality are checked before the same mounted journey | Synthetic predecessor-schema fixture using current migration code, not an archived old executable or a real user's historical installation. Legacy fixture graph preservation is checked; actual bytes/copy/notices/restart behavior is additionally covered by library tests. |
| Static/Dynamic | Actual bundled frames sampled40times at0.1s; Static stays0, Dynamic reaches0→1→2→3→0 | Visible idle-state loop works in both profiles. No assertion that every listening/thinking/speaking/error artwork sequence was exercised. |
| Explicit conversation follow | Management commits exact target; Home opener exposes that target without changing its identity | Headless app navigation and modal targeting. |
| Conversation drafts and running work | `test_real_console_buddy_send_preserves_both_durable_console_drafts[False/True]` uses real app/store and barrier gateway: reply goes to bound owner; another selected session/composer keeps its unrelated draft; closing modal leaves one run active; completion doesn't write sibling; Open Console restores correct draft | Actual application ownership and persistence with simulated provider output. |
| Truly cold Home entry | `test_cold_home_saved_buddy_bootstraps_only_after_explicit_open` asserts runtime store/controller absent before open, loads exact saved transcript, retains decisions after modal close | No prior Console visit hides bootstrap behavior; generated responses and decision requests are test-controlled. |
| Late authority changes | Saved-loader barrier cases and headless decision tests | Asynchronous reads allow loop progress and reject stale/replaced owners; they do not grant approvals. |
| Workspace and explicit None | Both captures create local named workspace with explicitNone, select its target and None Persona in real manager, then open settled inbox; registry default remains absent and Console active session unchanged | Actual empty-workspace projection and persistence. No existing chat is moved simply by switching the Buddy's follow target. |
| Workspace results | Existing projection/coordinator/modal tests cover frozen receipt acknowledgement, later unseen receipt preservation, moved/deleted membership refusal, failed refresh blocking actions | Result rows and receipt sources are controlled fixtures; populated live-provider workspace inbox remains native/integration follow-up. |
| Persona assignment/defaults |33atomic tests plus defaults/provisioning tests: exact identity/version, explicitNone, policy preservation, stale refusal, no hidden durable chat creation, pending backfill/restart respects absent default | Real local transactional behavior; mounted capture explicitly exercises None, not every Persona selection option. |
| Compact UI | Existing layout tests cover60×20 reply/recovery controls and workspace modal sizes | Headless geometry/control evidence; native glyph rendering and physical mouse/key mapping remain outstanding. |

## Rendered evidence

The original SVGs are exported directly by Textual/Rich, with source/capture SHA-256 in the associated evidence files. Four representative SVGs plus metadata are retained beside this report; all 14 raw views are in the temporary artifact directory. Repository copies normalize one whitespace-only line per SVG so the staged whitespace check passes. The source manifest records both raw and repository SHA-256 values; the per-profile evidence retains the original export hashes.

- [Fresh management, Static, no Persona](artifacts/buddy-v1-32108/fresh-static-management.svg)
- [Fresh Home, Dynamic Buddy](artifacts/buddy-v1-32108/fresh-dynamic-home.svg)
- [Upgraded profile, exact conversation](artifacts/buddy-v1-32108/upgrade-conversation.svg)
- [Upgraded profile, settled workspace inbox](artifacts/buddy-v1-32108/upgrade-workspace-inbox.svg)

Visual inspection uses derived PNGs rendered locally with CairoSVG and installed Menlo substituted for the SVG's Fira Code font declaration; raw SVG exports remain unchanged in scratch storage. This corrects local font fallback/tofu and does not alter application state or layout data. The Home export shows the cyan Buddy inside the bottom-right viewport, while management visibly states Current Persona:None and fixed-target behavior. Conversation reply/recovery actions are present; empty workspace state is captured only after its real asynchronous receipt bootstrap finishes.

The first capture run failed because the new harness queried `app.query`, which Textual deliberately resolves against the default screen; the real Buddy was on the active Home screen. Controller/overlay diagnostics confirmed the mounted view. Changing the harness to `app.screen.query` resolved this without production changes. An early inbox export caught the honest loading state; final harness waits `_loaded` and `_fresh` rather than presenting the transient view as settled evidence.

## Remaining work

No production defect was reproduced; no production fixes or full suite were attempted. Native Terminal selection returned **"Computer Use is not allowed to use the app 'com.apple.Terminal' for safety reasons."** The packaged Chrome surface separately returned **"Computer Use permissions are not granted."** No alternate automation mechanism was used to bypass either restriction. Root completed the server's fresh WebUI conversation-navigation and populated workspace reply/acknowledgement checks in the permitted in-app browser; those do not supply native Textual evidence.

The task remains In Progress pending native Chatbook integration coverage. Real speech input/output and subjective artwork quality across all authored states are not qualified by these runs. The source-bound headless evidence and remaining gaps are recorded; native acceptance is still outstanding.
