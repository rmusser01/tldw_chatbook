---
id: TASK-41050
title: "Settings Network: s with no edit rewrites the config, and an invalid trust mode shows as Verify"
status: To Do
created_date: 2026-11-15 02:03
labels:
- follow-up
- architecture-review-2026-10-25
- settings
- ux
priority: low
references:
- tldw_chatbook/UI/Settings_Modules/category_panels.py
- Docs/User_Guide/settings.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Three Network problems, re-read on dev ecad404bd4 (2026-11-15):
1. **`s` with no edit still writes the config.** `NetworkPanel.request_save` (`UI/Settings_Modules/category_panels.py`) writes `[network] ssl_verify` on every **s**, even with no staged edit. The other draft categories answer "No Settings changes to save." instead. Found by TASK-1378.1, where it predates the task.
2. **An invalid trust mode shows as Verify.** When `ssl_verify` in config is invalid, the trust-mode Select is seeded with "Verify". Choosing Verify then stages nothing, and **s** refuses, so the user must pick another mode and then Verify. TASK-1378.1's reviewers ruled to keep the refusal and left this behaviour as a follow-up.
3. **The guide's troubleshooting bullet is stale.** The bullet "s did nothing at all. That category is not one of the seven draft categories" (`Docs/User_Guide/settings.md:2028`) predates Network saving with **s**.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Pressing s on an unedited Network category writes nothing and says there is nothing to save
- [ ] #2 With an invalid ssl_verify, the Select shows no valid mode as chosen, and choosing any mode once stages it
- [ ] #3 The guide's troubleshooting bullet matches which categories s saves
<!-- AC:END -->
