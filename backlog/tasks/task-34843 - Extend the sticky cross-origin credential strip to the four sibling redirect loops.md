---
id: TASK-34843
title: >-
  Extend the sticky cross-origin credential strip to the four sibling redirect
  loops (image-gen http_client x2, image_format_utils, Model_Artifacts fetch)
status: To Do
assignee: []
created_date: '2026-10-12'
labels:
  - security
  - egress
  - follow-up
dependencies:
  - TASK-19863
priority: medium
---

## Description

Found by PR #3082's post-merge code review (2026-10-12). TASK-19863 made the
cross-origin credential strip sticky in the four \`Utils/egress.py\` guarded-fetch
loops via \`_StickyOriginPolicy\`, but four sibling redirect loops outside egress.py
still decide per hop with \`egress.same_origin(url, current)\` — the exact
non-sticky comparison 19863 fixed. They carry real credentials (adapters pass
\`Authorization: Bearer\` keys) and cannot inherit the policy because it is
underscore-private:

- \`tldw_chatbook/Image_Generation/http_client.py:191\` (\`fetch_json\`)
- \`tldw_chatGeneration/http_client.py:350\` (\`fetch_bytes_via_post\`)
- \`tldw_chatbook/Image_Generation/adapters/image_format_utils.py:268\` (\`fetch_image_bytes\`)
- \`tldw_chatbook/Model_Artifacts/fetch.py:174\`

Severity class (per 19863's own filing): low — credentials return only to their
own origin with attacker-chosen path/query. Not an AC violation of 19863 (AC#4
scoped to egress.py's loops), but the hazard class is open.

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] \`_StickyOriginPolicy\` (or an equivalent per-chain helper) is exported publicly from \`Utils/egress.py\`
- [ ] All four sibling loops adopt the sticky policy; per-hop \`same_origin\` credential decisions are gone
- [ ] A→B→A tests per converted loop assert the credential is NOT re-sent to the origin after the cross-origin hop
- [ ] \`skill_remote_fetch.py\` is NOT converted (its GitHub-host allowlist is a different, sound design — the reviewer verified)
<!-- AC:END -->
