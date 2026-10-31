---
id: TASK-36521
title: 'User Guide: the Ctrl+6 row, the nav-bar screenshots, the Domain Defaults count and a duplicated approval paragraph describe retired layouts'
status: To Do
created_date: 2026-10-24 00:26
labels:
- follow-up
- docs
priority: low
references:
- Docs/User_Guide/index.md
- Docs/User_Guide/settings.md
- Docs/User_Guide/console/agent-runs-and-tools.md
- Docs/User_Guide/images
dependencies:
- TASK-36387
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-36387's review (PR #3226, 2026-10-23), checked on dev 5b5920b8e2 and at PR #3226's head:
- **The Ctrl+6 row.** `index.md:40` lists "Ctrl+6 | [Artifacts](artifacts.md) | Generated outputs…" as a screen. ADR-172 retired the top-level Artifacts destination. Ctrl+6 is now a compatibility shortcut into Library ▸ Artifacts (`ARTIFACTS_COMPATIBILITY_SHORTCUT`, shell_destinations.py:212), and `artifacts.md` already says so. PR #3226 rewrote the neighbouring Ctrl+9 row in that form.
- **Screenshots.** All 11 User Guide SVGs that show the nav bar show labels the app no longer has: ⌃9 MCP, ⌃4 or ⌃6 Artifacts, F7 Lab, F8 Logs and F9 Settings. The shipped bar is ⌃1-⌃5, ⌃7, ⌃8, ⌃0, then F2 Lab, F3 Logs, F4 Settings, F5 Research and F7 Meetings (`SHELL_DESTINATION_SHORTCUTS`). The files: `images/console/{action-row,attachment-staged,overview}.svg`, `images/home/{details,idle,overview}.svg`, `images/library/prompts-editor.svg`, `images/settings/{console-draft,overview,rag}.svg`, `images/watchlists/console-workflow-complete.svg`.
- **The Domain Defaults count.** The Layout tour (`settings.md:69`) says the toggle reads "Domain Defaults ▸ (10)" and shows "ten rows". The group has 12 categories (`DOMAIN_SETTINGS_CATEGORY_IDS`; Video Gen, Dreams and Guardian joined later), and the label counts them.
- **A duplicated paragraph.** In `console/agent-runs-and-tools.md:551-565`, a paragraph that says **Always** covers "MCP tools *and* local workspace tools" is followed by one that says **Always allow** is for "MCP tools only". Between them is a stray fragment, "install and run-script confirm cards follow the same rule).", which repeats the end of a sentence at :486. The fragment and the second paragraph came in with 6a6242ce3a, a rebased feature commit.

Not in scope: the badge table and the category map in settings.md are TASK-35013 AC#7 and TASK-35016 AC#12. TASK-35016 (To Do) also replaces the Domain Defaults group; if it lands first, the Layout tour follows its group labels instead. TASK-36519 rewrites the approval-card lines this guide quotes in the same agent-runs-and-tools.md section, so whichever lands second rebases onto the other.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 index.md's Ctrl+6 row says Ctrl+6 opens Library ▸ Artifacts and that Artifacts is not a screen of its own
- [ ] #2 No User Guide image shows a nav-bar label the app no longer has; each listed image is recaptured from the current app at a full-screen size, or removed with its reference
- [ ] #3 settings.md's Layout tour names the collapsed settings group and the count exactly as the app shows them
- [ ] #4 agent-runs-and-tools.md says once which tools Always covers, with no contradicting paragraph and no sentence fragment
<!-- AC:END -->
