---
id: TASK-35578
title: Chat settings and the first-run wizard say a chosen model is not added to the saved model list
status: To Do
created_date: 2026-10-19 21:50
labels:
- console
- first-run
- model-discovery
- model-config
- copy
- ux-review-2026-10-14
dependencies:
- TASK-33641
- TASK-35011
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Finding C6 of the 2026-10-14 Settings UX review: Console's Chat settings and the first-run wizard's model step never persist a provider's saved model list. A user can choose a model a server listed there, but neither surface adds it to that provider's list in the config file's [providers] section, which Settings ▸ Providers & Models shows as the Saved model list. The review rated this BY-DESIGN at P3, and its critique snapshot (.impeccable/critique/2026-10-14T15-26-24Z__tldw-chatbook-ui-screens-settings-screen-py.md) lists C6 among the findings retired as by design. The behaviour stays. What is left is copy: these surfaces should say that their choice does not join the saved list, and where a model can be added to it.

Why it was deferred. The S1 plan (backlog/docs/plan-2026-10-14-s1-discovered-model-default-saved-lists.md, Out of scope) kept C6 out because other streams owned both surfaces:
- console_settings_modal.py had no size headroom, and Phase 9's TASK-33006.19 was reworking its served listings.
- Phase 8 was rewriting the wizard's model step.

The plan said to revisit C6 after Phase 8 and Phase 9 land. Phase 8 has now merged (#3144), and Phase 9 (#3169) is armed in the merge queue. On 2026-10-19 the owner ruled that C6 stays out of S1 and gets this follow-up.

The gap now matters more. After TASK-35011, choosing a discovered model in Settings' Default model does add it to the saved list, so the two surfaces that do not add it need to say so.

Both files are still exactly on their module size ratchet rows at Phase 9's head: console_settings_modal.py is 6,166 of 6,166, and first_run_model_step.py is 997 of 997. The copy has to fit inside those rows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Where Chat settings offers models from a server's listing, it says that choosing one does not add it to the provider's saved model list, and names where in Settings a model can be added to that list
- [ ] #2 The first-run wizard's model step says the same for a model chosen from a server's listing
- [ ] #3 Choosing a model in Chat settings or in the wizard still leaves the provider's saved model list in the config file unchanged
- [ ] #4 console_settings_modal.py and first_run_model_step.py do not grow past their module size ratchet rows
- [ ] #5 The Console and First-Run Setup User Guide pages say that a model chosen in Chat settings or in the wizard is not added to the saved model list, and where to add one
<!-- AC:END -->
