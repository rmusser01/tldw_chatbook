---
id: TASK-42478
title: "The Console, Settings and Personas credential polls read the Claude Code Keychain credential every 5 s even when Anthropic subscription auth is not selected"
status: To Do
assignee: []
created_date: 2026-11-23 11:13
updated_date: 2026-11-23 11:13
labels:
- follow-up
- console
- security
- privacy
priority: high
references:
- tldw_chatbook/LLM_Calls/anthropic_subscription.py
- tldw_chatbook/UI/Screens/chat_screen.py
- tldw_chatbook/UI/Screens/settings_screen.py
- tldw_chatbook/UI/Screens/personas_screen.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 22 of the follow-up burn-down (2026-11-23), by TASK-42477's review. Checked on `fu/r22-keychain` (the TASK-42477 fix on 62531d037a); paths are under `tldw_chatbook/`.

**Why.** Three UI polls ask for the subscription credential status, whichever provider and auth source are selected:
- `UI/Screens/chat_screen.py:10621`, the Console's credential poll (`_poll_console_credential_readiness`, 4 Hz);
- `UI/Screens/settings_screen.py:5310`, the Settings provider status gate;
- `UI/Screens/personas_screen.py:7356`, the Personas handoff readiness gate.

Each calls `subscription_credential_status(background=True)` unconditionally, as one element of its change-detection tuple. `_SubscriptionReadinessCache.status()` (`LLM_Calls/anthropic_subscription.py`) starts a background `read_claude_code_credential()` whenever its 5 s snapshot has lapsed. That read opens `~/.claude/.credentials.json`. On macOS, when the file is absent, it spawns `/usr/bin/security find-generic-password -s "Claude Code-credentials" -w`. TASK-26022's live verification (AC#7, 2026-09-02) found Claude Code's credential in exactly that Keychain item, and this path returned it.

So a macOS user logged in to Claude Code, with no credential file, has Chatbook spawn `security` and take their Claude Code OAuth token into process memory about every 5 s while any of those screens is mounted. That holds even with `auth_source = "api_key"` or another provider selected. TASK-26022 framed the borrow as explicit opt-in: the module docstring says only `auth_source = "claude_subscription"` activates it. TASK-42477's moved-HOME check does not change this for a real user, whose HOME is the account's own. In TASK-42477's measurement, every mounted Console test spawned the read with no Anthropic subscription configured.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 While the provider a screen reports on is not Anthropic, or is Anthropic with an auth source other than `claude_subscription`, keeping the Console, Settings or Personas screen mounted causes no credential read: no read of the credential file and no `security` spawn. A test per screen pins it
- [ ] #2 With Anthropic and `auth_source = "claude_subscription"`, each screen still shows the background readiness states (checking, ready, expired, missing) and refreshes when a read completes, as today
- [ ] #3 Switching to Anthropic subscription auth starts the read without a restart; switching away stops further reads
- [ ] #4 The send-time and provider-readiness paths for subscription users behave as before
<!-- AC:END -->
