---
id: TASK-41133
title: "Trace credential masking leftovers: a credentials-v3 projection for whsec_ and gsk_, and compound labels such as client_secret= in shared bundle text"
status: To Do
created_date: 2026-11-15 02:20
labels:
- follow-up
- security
- trace
priority: low
references:
- tldw_chatbook/Utils/log_sanitizer.py
- tldw_chatbook/Chat/trajectory_export.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 endpoint stream fu/r14-endpoint 6cb8e34461 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** TASK-40301 masked `whsec_` (Stripe/Svix webhook secrets) and `gsk_` (Groq) in logs and shared bundles, but deliberately not in the stored trace projections, because changing those outputs needs a new credentials version. Stored trace history therefore still holds them. Separately, shared bundle text masks `api_key=`/`token=`/`password=`/`secret=` but not compound labels such as `client_secret=`, which logs do mask.

**Evidence.** TASK-40301's notes (endpoint branch): AC#2 "They are deliberately in neither stored trace projection (`sanitize_trace_credentials_v1`/`_v2`) … (proposed follow-up)"; Scope: "logs also mask compound labels such as `client_secret=` in text (proposed follow-up)".
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A credentials-v3 stored-trace projection masks whsec_ and gsk_ values; existing v1/v2 history still verifies, and new traces use v3
- [ ] #2 Shared bundle text masks compound credential labels the log sanitizer masks (client_secret=, refresh_token= and the like), with tests red first
<!-- AC:END -->
