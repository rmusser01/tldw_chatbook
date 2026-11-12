---
id: TASK-38970
title: "File Notes and the Watchlists quickstart have no screenshot, and no capture driver can reach their states"
status: Done
assignee:
- '@claude'
created_date: 2026-11-06 01:21
updated_date: 2026-11-10 18:28
labels:
- follow-up
- docs
- screenshots
priority: low
references:
- Docs/User_Guide/library/file-notes.md
- Docs/User_Guide/watchlists-quickstart.md
- Docs/User_Guide/_template.md
- scripts/regen_approval_card_svg.py
dependencies:
- TASK-36521
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down; checked on dev 29592ecc9b and the TASK-36521 branch, 2026-11-05.

TASK-36521's branch merges before this one and deletes two images that no longer matched the app:
- `Docs/User_Guide/images/library/file-notes.svg`, referenced from `library/file-notes.md`;
- `images/watchlists/console-workflow-complete.svg`, referenced with its caption from `watchlists-quickstart.md`.

The branch removes those references too. Only frozen QA evidence JSON under `Docs/superpowers/qa/` still names the files.

Neither picture can be retaken yet because there is no driver for either state:
- File Notes needs a folder-picker flow that chooses a notes folder.
- The Watchlists image needs a completed Console workflow run.

Capture today is the hand recipe in `Docs/User_Guide/_template.md:55-134`. The repo's only capture script is `scripts/regen_approval_card_svg.py`, which captures the approval card through a harness. TASK-36521's capture scripts stayed in a scratchpad. TASK-24632's workflow image was a copied UAT capture, not a driver.

Merge state: the TASK-36521 branch is now stacked on the TASK-38757 branch, and its tip (cc9647f8cd) has neither image nor any reference to them, so the removal survives both merges.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 No User Guide page references a missing image; a Docs test resolves every image link under Docs/User_Guide and fails when one link is pointed at a missing file (negative control)
- [x] #2 A checked-in capture driver reaches the File Notes workspace with a folder chosen and the Watchlists Console workflow in its completed state, using a scratch profile and no real provider
- [x] #3 File Notes and the Watchlists quickstart each show a screenshot again, taken from the current app at 235x52 with that driver and with cdnjs font URLs stripped; if a state cannot be reached, the notes say which and why
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a Docs test resolving every image link under Docs/User_Guide (with a temporary broken link as negative control).
2. Add scripts/capture_user_guide.py: boots the real TldwCli at 235x52 on a fresh temporary profile (null keyring, loopback-only sockets, real-profile guard), drives a named target through the UI, writes the SVG with cdnjs font URLs stripped.
3. Targets: file-notes (Library > Notes > Folder files > Choose folder... picker > Select) and watchlists-workflow (the scripted Console Watchlists workflow from the mounted UAT, run to completion with no real provider).
4. Capture both images and restore their references in file-notes.md and watchlists-quickstart.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a checked-in capture driver, `scripts/capture_user_guide.py`, and retook both images with it.

- **AC#1** `Tests/Docs/test_user_guide_contract.py::test_every_user_guide_image_link_resolves` resolves every `![…](…)` link in every `Docs/User_Guide` page through the link resolver it shares with the Console Library contract test (TASK-38974). Negative control: `home.md`'s `images/home/idle.svg` link temporarily pointed at `idle-missing.svg` failed with `{'home.md': ['images/home/idle-missing.svg']}`; reverted. Green on dev before the images came back (TASK-36521 had removed the references), green after.
- **AC#2** `scripts/capture_user_guide.py <target>`, one target per process. It makes a fresh temporary profile (HOME, XDG_*, TLDW_CONFIG_PATH) before importing the app, sets the null keyring, installs `Tests/real_profile_guard`, allows loopback `socket.connect` only, boots a real `TldwCli` in `run_test(size=(235, 52))`, writes the SVG through `write_guide_svg`, deletes the profile and ends with `os._exit` (lessons-live-verification, TASK-36521).
  - `file-notes`: Library ▸ Notes ▸ **Folder files** ▸ **Choose folder…** by keyboard (Enter, Ctrl+A, the demo folder's path, Enter, Tab, Enter in the real `SelectDirectory` picker), then `welcome.md` opened from the Files tree.
  - `watchlists-workflow`: the provider is saved through the first-run commit (`llama_cpp` at a loopback URL that is never called). The model's turns come from the mounted UAT's scripted gateway (`Tests/QA/test_console_watchlists_workflow_uat.py`), subclassed to answer `cached_context_window` (the real app's context meter asks the gateway; the factory app never did) and to drop the UAT marker from the final reply. Feed fetches and the briefing's model call return canned text. Each approval round is approved once its **Allow once** is enabled: pressing it while disabled is a no-op, and the first loop stalled on that.
- **AC#3** Both states were reached; captured at 235x52, cdnjs-free, no temp paths or account names in the SVG text. `library/file-notes.md` shows `../images/library/file-notes.svg` in its Layout tour; `watchlists-quickstart.md` shows `images/watchlists/console-workflow-complete.svg` with a new caption (three source checks and the briefing COMPLETE, the summary reply below). Final run of all three targets after the last driver edit: rc=0 each.
- Deviation, docs correctness: the workflow only completed with `[tools] full_agent_tools_enabled = true`; plain chat's light tool surface (TASK-34100.5.3) offers no Watchlists tools, so `load_tools` failed and the approved call came back "Tool not permitted". The quickstart's "Before you start" now lists **Full agent tools** with where to turn it on.
- Files: `scripts/capture_user_guide.py` (new), `Docs/User_Guide/images/library/file-notes.svg`, `Docs/User_Guide/images/watchlists/console-workflow-complete.svg` (both new), `Docs/User_Guide/library/file-notes.md`, `Docs/User_Guide/watchlists-quickstart.md`, `Docs/User_Guide/_template.md` (points the capture recipe at the two drivers).
- **Review round 1 (r13h-rev1).**
  - I-2: the workflow target could export a turn that was still running (Stop and Redirect visible). After the reply appears it now waits, bounded, until `#console-stop-generation` is hidden. The recapture's composer row reads "Send disabled: type a message Send Dictate".
  - M-1: `os._exit` skipped the real-profile guard's atexit report. The driver now reads `real_profile_guard.take_violations()` before exiting, prints any refusal and exits 1.
  - M-2: `_isolate` drops `HTTP_PROXY`/`HTTPS_PROXY`/`ALL_PROXY` (any case) and every inherited `*API_KEY`, `*_TOKEN` and `*_SECRET` variable before the app is imported.
  - M-4: `console/agent-runs-and-tools.md`'s "Running the Watchlists workflow from Console" now says to turn on Full agent tools, linking the light-surface section.
  - Smoke test (reviewer recommendation): the gateway subclass moved to `guide_gateway()` in the driver. `Tests/Docs/test_user_guide_contract.py::test_capture_driver_builds_its_gateway_on_the_uat_planner` builds it and runs its first planner turn. Red under a temporary rename of the UAT's `_ScriptedWatchlistsGateway` (`ImportError: cannot import name '_ScriptedWatchlistsGateway'`), the break it exists to catch; reverted, green.
  - All three targets were rerun with the fixed driver: rc=0 each. `search-rag.svg` and `file-notes.svg` came out byte-identical to the previous run.
- **Review round 2 (r13h-rev2).** The environment scrub missed credentials the app reads (`FAL_KEY`, `CONFLUENCE_PASSWORD`, `DOCEXT_PASSWORD`). It now drops every inherited variable ending in `KEY`, `_TOKEN`, `_SECRET`, `_PASSWORD` or `_USERNAME`, plus the proxies and `SSH_AUTH_SOCK`, and the comment says exactly that. `PYTHON_KEYRING_BACKEND`, `TLDW_*` paths and `TIKTOKEN_CACHE_DIR` do not match. `search-rag` rerun with it: rc=0 and the same painted rows. The file differed only in Rich's per-export class hash, so the committed image is kept.
<!-- SECTION:NOTES:END -->
