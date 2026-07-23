# Collapsible Console composer live QA

Approval status: approved

The user explicitly approved all six rendered screenshots on 2026-07-23 for
tested commit `20aba85893a4bd06e8091389b2f57161877b154c`.

## Revision and isolation

- Branch: `chore/harness-review-tasks-320-334`
- Tested commit:
  `20aba85893a4bd06e8091389b2f57161877b154c`
- Fresh synthetic profile:
  `/private/tmp/tldw-qa-composer-r3-3U326Q`
- Synthetic pending image:
  `/private/tmp/tldw-qa-composer-r3-3U326Q/synthetic-attachment.png`

The profile contained only a local-provider configuration, empty app state, a
deterministic QA stub, synthetic draft text, and the synthetic PNG copied from
an already-inspected QA screenshot. It contained no API key, personal
conversation, or personal file-picker history.

## Runtime recipe

The deterministic OpenAI-compatible stub listened only on
`127.0.0.1:8898`. It streamed 28 synthetic response lines, then delayed
completion for 60 seconds so the real Console run/Stop path remained active.

The real app ran through:

```text
HOME=/private/tmp/tldw-qa-composer-r3-3U326Q
XDG_CONFIG_HOME=/private/tmp/tldw-qa-composer-r3-3U326Q/.config
XDG_DATA_HOME=/private/tmp/tldw-qa-composer-r3-3U326Q/.local/share
PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook
.venv/bin/tldw-serve --host 127.0.0.1 --port 9198
```

The normal Playwright wrapper could not bootstrap its npm package in this
environment. The locally cached `playwright-cli 0.1.17` binary and its bundled
Chromium were used directly; system Chrome and `@playwright/test` were not
used. The browser aborted only external `https://**` requests, navigated to
`http://127.0.0.1:9198` with `waitUntil: "commit"`, and waited successfully
for `body.-first-byte`.

## Geometry

| Layout | Browser pixels | xterm pixels | Exact cells | Grid cell | Expanded height | Collapsed height | Transcript rows reclaimed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Wide | 1013×630 | 1008×630 | 140×42 | 7.2×15 px | 5 rows | 1 row | 4 |
| Compact | 725×480 | 720×480 | 100×32 | 7.2×15 px | 8 rows | 1 row | 7 |

The live measurements therefore cover the approved five-to-eight-row expanded
range, exact one-row collapsed geometry, and four-to-seven reclaimed transcript
rows.

## Evidence

| Screenshot | Live behavior verified | SHA-256 |
| --- | --- | --- |
| `wide-expanded.png` | Full `Composer ▾`; synthetic draft and pending PNG visible; five-row composer; fixed action group does not overlap. | `4ffc8cda822d83ed93e88dcab0697504b48e8a3c509d854973972c41e1a05f1a` |
| `wide-collapsed-draft.png` | Exact row 41; `Composer hidden · Draft retained · Attachment retained`; full `Expand ▴`; no filename or draft content. | `65f5753c27cd61cfebeff80ba0bb297ea27b2427184f7379532a406f74e4b9e6` |
| `wide-collapsed-generating.png` | Exact row 41; `Generating` at the left, `Stop` at column 123, and `Expand ▴` at column 131 without overlap. | `9f40c964466c7ee0a2e0f8ac5811dcffe836d26553549f5b10f46869ff59a893` |
| `compact-expanded.png` | Full `Composer ▾`; synthetic pending PNG and all composer actions remain visible at exact 100×32. | `7d29a4fd617ba69c5522ea1fa10119c042c52199c23d955504677924a6b5cdb4` |
| `compact-collapsed-draft.png` | Exact row 31; `Attachment retained` starts at column 36 and full `Expand ▴` starts at column 91; no overlap or filename/content disclosure. | `c65023ade7647e8a18b905c1066f51f7ad883e0b9a4c30b3a50c65628f7264a6` |
| `compact-collapsed-generating.png` | Exact row 31; status at column 1, full `Stop` at column 83, and full `Expand ▴` at column 91 without overlap. | `9d1e4e97ea8602b168738199ba7ee1834a472de5f49c6dd6e942d72ea97a940a` |

Every PNG was inspected individually at original resolution with
`functions.view_image`.

## Interaction checks

### Pending attachment

Selecting the PNG through the real picker produced the pending-image controls
and label in expanded mode. Collapsing then produced the exact presence-only
status:

```text
Composer hidden · Draft retained · Attachment retained
```

The collapsed status exposed neither `synthetic-attachment.png` nor image
content. This confirms the PNG exercised the pending-attachment store path,
unlike the earlier inline `.txt` setup.

The synthetic model was intentionally not marked vision-capable. Attempting to
send with the PNG was correctly blocked by normal model-capability validation,
leaving the draft and pending image intact. After the retention captures, the
PNG was cleared through the real `✕` control and the retained synthetic draft
was sent to exercise generation.

### Escape, Stop, selection, and reading position

- One Escape from collapsed pending-work mode expanded the composer and
  restored its visible caret.
- A real delayed run was collapsed while streaming. `Stop` was clicked in the
  collapsed row; `Generating` and `Stop` disappeared while the composer stayed
  collapsed with `Expand ▴`. The assistant message remained and was marked
  `[stopped]`.
- The same collapsed-Stop result was repeated at compact geometry.
- On the real stopped assistant message, the transcript was scrolled upward
  from its tail so synthetic lines 06–27 were visible, then line 15 was clicked
  to select the message. The selected background remained visible.
- One Escape expanded and focused the composer while keeping the selection.
  The manually chosen semantic top line remained exactly line 06; the expanded
  viewport ended at line 23 instead of 27, matching the four rows returned to
  the composer rather than jumping to the tail.

## Automated verification already completed

- Original static gate: Ruff passed, product compilation passed, and
  `git diff --check` passed.
- Mandatory broad command:
  `12 failed, 2023 passed, 69 skipped, 4 warnings, 2 errors in 1109.71s`.
  The 12 failures reproduced identically at the pre-feature revision, and the
  two loopback-denied setup nodes passed when rerun with loopback permission.
  No feature-touched test file failed.
- Composer-toggle fix final gate:
  `260 passed, 1 warning in 234.69s`.
- Fresh post-fix live-QA sanity:
  `8 passed, 26 deselected, 1 warning in 4.24s`.

The 18-minute broad suite was not rerun after the narrowly scoped label fix,
per the QA controller.

## Privacy inspection

All six screenshots contain only synthetic UI state and synthetic response
text. Original-resolution inspection found no API key, credential, personal
conversation, personal filename, file-picker contents, absolute path, or
repository path.

The user explicitly approved this rendered evidence on 2026-07-23. TASK-398
was then eligible for Backlog closeout; no merge or push was performed as part
of this evidence update.
