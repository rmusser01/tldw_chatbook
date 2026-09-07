---
id: TASK-2261
title: Document pointing chatbook at OpenAI-compatible TTS servers
status: Done
updated_date: '2026-08-04 19:30'
assignee:
  - '@claude'
created_date: '2026-08-04 12:00'
labels:
  - tts
  - docs
dependencies:
  - task-2260
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The complaint that produced TASK-2260 was a discoverability failure as much as a behavior one:
a user wanting to use a local OpenAI-compatible TTS server (pocket-tts) had no documentation
telling them the OpenAI provider's Base URL setting is the way to do it. There is no Speech/TTS
page under `Docs/User_Guide/` at all (the User Guide programme G1-G5 has not reached Speech).
Write a short user-facing doc covering: choosing the OpenAI provider with a custom Base URL for
any OpenAI-compatible server (keyless local servers included, with custom model/voice names
passed through — behavior shipped in TASK-2260), and the AllTalk provider as the alternative
for AllTalk servers specifically. Include a worked pocket-tts example.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A user-guide page explains how to point TTS at an OpenAI-compatible server via the OpenAI provider's Base URL setting, including a keyless local-server example
- [x] #2 The page states that custom model and voice names are passed through unmodified when a custom Base URL is set
- [x] #3 The page is reachable from wherever the User Guide indexes speech/TTS topics
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Map the CURRENT Speech/TTS settings UI on dev (exact navigation path, section titles,
   field labels) so the page documents reality, not memory.
2. Write `Docs/User_Guide/` page covering: reaching TTS settings; OpenAI provider with a
   custom Base URL for any OpenAI-compatible server (keyless local servers, custom
   model/voice passthrough per TASK-2260); worked pocket-tts example; AllTalk provider as
   the AllTalk-specific alternative; match house style incl. "Verified against" stamp.
3. Link the page from the User Guide index/nav location the other pages use.
4. Verify statements against the live code paths (labels quoted from source), PR, merge.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added `Docs/User_Guide/openai-compatible-tts.md`: a how-to for pointing the OpenAI TTS
provider at any OpenAI-compatible server via Settings ▸ Speech & TTS (Provider setup ▸
Configure Provider ▸ OpenAI ▸ Base URL), with the keyless-credential path, the
model/voice exact-passthrough behavior from TASK-2260, a pocket-tts worked example
(values kept honest — port/model/voice deferred to the server's own docs rather than
invented), the AllTalk provider as the AllTalk-specific alternative, and troubleshooting
(full endpoint path required — requests go to the Base URL as written; org ID never sent
to custom URLs; OPENAI_API_KEY env caveat). Linked from a new "How-to guides" section in
`Docs/User_Guide/index.md`. Every quoted label was verified against
`Widgets/Settings_Widgets/speech_tts_settings_panel.py` on dev 265dbd687 (the settings
UI moved from STTS_Window to the Settings screen since the task was filed); stamp per
`_template.md` convention. Docs-only change — no tests to run; no link-checker exists.
<!-- SECTION:NOTES:END -->
