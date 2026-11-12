# Combined existing-control CI preparation wave

Date: 2026-10-29 UTC. Permanent source edits are confined to the three allowed
test/helper files; no product, Git, task, fixture/service/getter, stylesheet,
workflow, census or global wait-helper edits. Root owns integration
and the single scoped review. Verification is targeted; no full-suite sweep.

Final status: the three Reader journeys, full original link journey, amended
extended keyboard journey and amended Preview sibling have qualified native
GREEN outcomes across the bounded runs below. Held-folder and held-Preview
old/new boundaries and the Reader wrong-write negative are qualified. Scoped
Ruff and AST conservation pass; exact pre-existing formatter debt is retained.
All failed/unqualified attempts remain separately recorded. No hosted-CI or
product typing/focus/action-loss repair is claimed.

## Scope and changes

- `Tests/UI/test_library_media_reader_controls_in_list_view.py`: only the
  dirty/off-page Reader case's filter preparation now clicks/focuses the real
  filter and posts native App `Paste("Product")`, then awaits the unchanged
  authoritative browse helper and unchanged expected row. Removed unused
  `_query_settled` and `_type` import; qualified the module/comment text.
  The native paced-key query loss remains product debt TASK-37360.
- `Tests/UI/library_note_links_support.py`: local `_open_notes` waits for the
  original rail action's navigation-context acknowledgement, list/cleared
  session, the appropriate fresh/current tree visit, every expected root/open
  folders/placements slice, and actual DOM publication. Local
  `_notes_tree_publication` reads existing immutable branch owners, epoch,
  generations, freshness, native `_recompose_required`, existing
  `recompose_in_flight`, ordered `(placement_id, note_id)` rows and live
  nonpruning ancestors. `_open_note` shares one 30-second setup deadline,
  reacquires the exact title/note row over a native turn, and presses once;
  `_armed_editor` stays unchanged. Reentry requests a fresh visit; retained
  Create-to-Browse uses the acknowledged current visit, matching the actual
  route owner. The broader lost-gesture product obligation remains existing
  TASK-34000.85.
  Root-approved Ruling13/Task5 amendment also extends only `_preview`: retain
  the original show/paint waits inside one overall30s bound; require current
  live focused Preview region/body, original painted text, both native
  canvas pending/inflight witnesses clear, nonpruning ancestors, and unchanged
  current identities across a native layout/message turn. The old fixed0.1s
  pause is replaced by that publication condition. No caller oracle changes.
- `Tests/UI/test_library_note_links_picker_and_preview.py`: only
  `_blank_note_with_body_focused` adds an actual title click and focus readiness
  before the original `_type(VERIFIER)`. No product new-note focus repair is
  claimed.

ADR required: no. Existing ADR-027/033/086/103 boundaries remain; this is native
test preparation under TASK-34000.11/18 and Task 5 of the approved plan.

## Findings addressed

1. Existing UI shard 12 failed the Reader case at original line 192 with query
   `roduct` never settled, applied query `t`, rows 2. The supplied native report
   proves fresh focused Input select-all on debounce recomposition can replace
   prefixes even with native spaced keys. Paste prepares the Reader identity
   regression atomically; every Reader/write/no-toast oracle remains.
2. Existing UI shard 19 failed the full link case's first Navigation reopen at
   original line 245 after real insertion/autosave. The supplied native report
   proves an attached row's queued Pressed can lose its parent during the
   outstanding sibling folder publication, with no note load beginning. The
   new setup barrier waits for complete branch and DOM publication before its
   one original row press.

## Native verification and evidence

All valid native calls use project Python
`C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Scripts/python.exe -X utf8`,
the unchanged approved `.test-tmp/textual-37266-37268/run-targeted.py`, native
0700 external profile/TEMP roots, the existing bootstrap-profile UI collection
path and unchanged `--timeout=180`. One App batch runs at a time. Exact argv,
profile/source roots, raw logs, XML and verification JSON live in each named
run directory under `.test-tmp/textual-37266-37268/ci-fix-wave/`.

`owned-source-01` ran the actual changed sources with observation-only
wrappers: three Reader controls, the full original link journey and the
Preview-opens-in-app sibling all passed. The selected extended keyboard
Preview consumer failed (see below), so the honest batch result is **5 passed,
1 failed**, zero errors/skips, 262.54 seconds total. XML SHA256:
`f84f46aa343fb6afde8de8c115b78cf5f165353f4345d3b25be4ac117354d668`.

`held-full-new-02` ran the actual full original link journey, with the real
root folders answer held and an independent native message/layout witness
releasing it after placements DOM publication. **1 passed**, zero
errors/skips, 81.42 seconds total / 59.16 seconds call. XML SHA256:
`4ad38ab8ff760bb391694957e743653f90d8f3cec28650d79f3418bbeb36f654`.
The witness at monotonic 634883.828 saw both actual canvases published,
placements present and the folder branch still held; helper returned=False,
target presses=0, loads=0. The real folder answer released at the same time.
After current DOM publication the helper returned and exactly one target row
press delivered exactly one real note load; watchdog=False. All holds drained
inside App at 634885.203. Every original picker/Escape/latch/body/caret/focus/
autosave/navigation/backlink/no-OS-URL assertion passed. The observer also
retained initial title-route/focus state; no wider title-prep change was made.

`reader-wrong-target-01` is a deliberate private negative mutation ONLY.
`wave_probe.py` arms the mutation at the test's original Save Button press,
then returns `local:media:2` from the action-target seam where the real owner
answered `local:media:1`. The actual service recorder and original test
assertions remain. The test fails at the original write-ID assertion,
`assert [2] == [1]`, recording title `Interview Recording!` against item 2.
**1 intended failure**, zero XML errors/skips, 48.15 seconds total / 20.02
seconds call. XML SHA256:
`83251bb88be295199ae3ea581b37ede7a69ea714b984a39ccd8696c4696159ef`.
A secondary logged missing trash-canvas traceback occurred during failure
cleanup; it is retained in the raw log and was not counted as an XML error.

Executed phase-one after-source SHA256 (before the root-approved Preview
amendment below):

| File | SHA256 |
| --- | --- |
| `library_note_links_support.py` | `d8ab593517307b98ced0a008a1f1e10bbc6a05aea05af3b434ca3a8f0db3e7ce` |
| `test_library_media_reader_controls_in_list_view.py` | `6ae9d9d8da88cd1d755ec07dfd079ea8f67b953df8db75b7df09d8c35ebe37e7` |
| `test_library_note_links_picker_and_preview.py` | `e65c84282dc92f64d4d676e0ebef5974809d00c790e5be623ad433911bfe9f08` |

Raw commands, XML outcomes, source/trace/probe hashes are consolidated by
`receipts.py` in `evidence-receipt.json`; each test's raw observation trace
captures the actual loaded source paths and hashes.

`held-new-already-list-01` passed the same local held-answer oracle against
the actual new helper and then exercised an already-list caller: **1 passed**,
zero errors/skips, 69.73 seconds total / 47.93 seconds call. XML SHA256:
`3242c7dca9b18d973cf2d3c35098cf791ad88ef7cf4dc5d89ce0a15f8ab2d75d`.
While folders were held the helper had not returned and presses/loads were
both zero. After release it delivered one press/load and drained the hold in
App. A subsequent actual Notes rail return published epoch3 list; invoking
`_open_note` from that list retained its original rail action, acknowledged
epoch4 and opened Index. This verifies the actual already-list refresh rather
than skipping the rail gesture or assuming it is a no-op.

## Selected extended consumer

`test_enter_on_the_focused_preview_follows_the_only_link_and_lists_several`
failed in `owned-source-01` at original line87, waiting for Lisbon to arm
after the first native Preview Enter. Its first `_open_note(INDEX)` did
deliver a real note load, and the original test had already waited for Preview
focus and asserted the implemented follow-link footer entry. No Lisbon load
began in the trace. This alone does not distinguish a helper-timing regression
from existing Preview publication/action loss. Per root direction, the exact
old-helper baselines below used the preserved support snapshot privately,
with no worktree checkout or product getter replacement. No optional test or
product edits were made in response.

`extended-old-helper-01` did not qualify that comparison: the exact old helper
failed at initial Index setup (line80), with the actual product source
five-second deadline and visible source recovery callout. No note load began.
**1 failure**, zero errors/skips, 85.51 seconds total. XML SHA256:
`6c806a6f7a8fec55fb731778dd4a84b7aff7513b56e0acbc532c1b56dc8baaea`.
One further bounded baseline, explicitly authorized by root, uses only actual
source readiness and its visible supported Retry if needed before the unchanged
Index/Preview journey; real five-second/180-second guards remain. It observes
native Enter dispatch, follow-link gates, focus identity, pending/inflight
publication and pruning. No later assertion/action is retried.

`extended-old-helper-ready-01`: actual source readiness succeeded without
needing Retry (`_library_loaded=True`, lookup_error=None). The exact compiled
old helper opened Index, and the first Enter reached the focused Preview,
passed the real follow-link gate, invoked the action and began the real Lisbon
load. The original assertions for that step passed. The same run later failed
line105, where the two-link picker never painted Index, despite a focused
Preview, a native Enter delivered through App, a true follow-link gate and the
action being invoked. **1 failure**, zero errors/skips, 79.09 seconds total /
56.24 seconds call. XML SHA256:
`55e62d015e60daf0f359f25b7a7ead5a9ff9167d7a4a62bde14f388cfb065a60`.
This is a different failure; it is not an exact baseline reproduction of the
current-helper line87 failure.

`extended-new-helper-ready-01`: the explicitly requested final comparison
retained the actual current helper (no old-helper replacement), with identical
actual source readiness and observation-only Preview tracing. No Retry was
needed. It failed the first Enter at original line87: **1 failure**, zero
errors/skips, 85.90 seconds total / 63.17 seconds call. XML SHA256:
`f15d1b2856e71278837fadf2f5aa2c2188bedc32783b922f1e532c577b74baa0`.
The native trace distinguishes this from a missing initial focus assumption:

- Enter first reached the current focused Preview at 635627.265 and bubbled
  through the screen while that focus still held.
- Real Work-pane recomposition started at 635627.328; Items recomposition
  started at 635627.343. App received the same Enter at 635627.343 with both
  existing `recompose_in_flight` witnesses true, the old Preview still attached
  but pruning, and focus reset to the Items pane grip.
- The follow-link gate/action and Lisbon load never ran. The native Enter
  activated that grip instead. Replacement Work published and restored Preview
  focus at 635627.671, after the key had been consumed. Route remained Notes,
  view remained editor, and the Index session identity stayed current.

The raw `preview-old-helper-ready.trace.json`,
`preview-new-helper-ready.trace.json` and `preview-comparison-summary.json`
retain the complete native focus, pending/inflight, pruning, dispatch and load
receipts. The App slot was released after that comparison; the subsequent
root-approved amendment is recorded below.
At that point the optional consumer remained **red**, and a blanket
no-regression claim was unsupported. The changed helper affects timing;
the different old failure does not prove the new first-Enter failure
pre-existing. Root then approved the bounded `_preview` preparation amendment
below before implementation. No optional-test assertion, focus getter or
product behavior was changed.

## Root-approved Preview preparation amendment

Ruling13 and the appended Task5/brief amendment were read before editing.
Only existing `_preview` plus its stdlib `asyncio` import changed. The
already-verified `_notes_tree_publication`, `_open_notes` and `_open_note`
functions compare AST-identical against phase-one source SHA d8ab…; shared
publication logic was not changed. Therefore the approved rule does not call
for repeating Reader, full-link or held-folder checks.

`held-preview-old-01` failed before Preview, during initial note arming; it
provides no boundary evidence. XML SHA256:
`d3db202cbd33fcce8d88f2121d024b9d693372b15a2c41505f8226467545331f`.
The next private probe used the proven actual New Blank/title gestures before
the unchanged Index open.

`held-preview-old-02` qualified the old boundary RED: after the original paint
wait, the probe requested the same native Work refresh used by publication,
then deferred its actual queued `_check_recompose` callback without clearing
the real native pending mark. App turns stayed available; no flag/data/getter
was substituted. The old `_preview` returned while that publication remained
pending, failing `Preview helper returned before native pending publication`.
The native callback was then released and the real remount drained inside
App; watchdog=False. **1 intended failure**, zero errors/skips, 66.83 seconds
total / 42.44 seconds call. XML SHA256:
`df551338e49bee163c56e12170303f49f324f04760d572924986786395a51c33`.

`held-preview-new-01` ran that boundary against actual amended source and
passed: the original text painted at636523.687; native pending publication was
held at636523.703; independent native witness at636524.437 observed
returned=False/watchdog=False. The real callback resumed at636524.484, actual
remount completed at636524.859, and the helper returned at636525.046. All holds
drained inside App. Native Enter then opened the real Lisbon target with its
original title/armed assertions. **1 passed**, zero errors/skips, 74.35 seconds
total / 50.37 seconds call. XML SHA256:
`be56f712f4955d00648c1af9408dbc173120797de60ff649a366bfc1381ddf9c`.

Final amended support source SHA256:
`5f530bfa6324c6d45cf72ea916632b3a0760b1c692d385fee03bb85d5a77c9da`.
Reader/title sources retain their phase-one pins.

`preview-amend-owned-01`: the complete actual-source extended keyboard
consumer now passed all its unchanged single-link, no-link, several-link
picker, focus, Enter, target-title and no-OS-URL assertions (43.74 seconds
call). The Preview sibling stopped before `_preview`, at its first Index
`_open_notes` publication setup wait (56.47 seconds call). Thus this batch is
honestly **1 passed / 1 failed**, zero errors/skips, 125.86 seconds total;
the sibling is unqualified for the amended Preview helper. XML SHA256:
`ee2ddbd6c163dd8490171be537c2fba0f69862c7a542d7b7ac4a163260a2f26f`.
The private actual source-readiness fixture at that point wrapped only the
extended module's `_library`, not the sibling's. One bounded sibling-only
qualification applies that same already-approved native readiness/visible
Retry preparation to its `_library`; no permanent helper, fixture/service,
assertion, later click or action retry is introduced.

`preview-amend-sibling-ready-01` passed the actual amended helper and every
original Preview-sibling in-app click, native link open, browser count/notice,
deleted-target refusal, unchanged note identity and no-OS-URL assertion.
Actual source readiness was loaded=True/error=None; no Retry was necessary.
**1 passed**, zero errors/skips, 81.78 seconds total / 50.62 seconds call.
XML SHA256:
`6834bb68f68bc59660f920e50b33eb4f2a20c291b8a08db7e84bf5016a68a97e`.
No permanent source edits followed the qualified extended/Preview positives.

## Covering commands

All run-directory `command.json` files record exact argv. The common launcher
was the project Python `-X utf8 .test-tmp/textual-37266-37268/run-targeted.py`
from this worktree; `PYTHONPATH` selected the private `ci-fix-wave` plugins.

| Run | Plugin/target arguments after run directory | Mode |
| --- | --- | --- |
| `owned-source-01` | `-p wave_probe Tests/UI/test_library_media_reader_controls_in_list_view.py Tests/UI/test_library_note_links_picker_and_preview.py Tests/UI/test_library_note_links_preview_opens_in_app.py Tests/UI/test_library_note_links_picker_and_preview_extended.py::test_enter_on_the_focused_preview_follows_the_only_link_and_lists_several` | `CI_FIX_MODE=observe` |
| `held-full-new-02` | `-p wave_probe -p title_observer Tests/UI/test_library_note_links_picker_and_preview.py::test_typing_two_brackets_opens_a_keyboard_picker_that_inserts_a_real_link` | `CI_FIX_MODE=held` |
| `reader-wrong-target-01` | `-p wave_probe Tests/UI/test_library_media_reader_controls_in_list_view.py::test_a_dirty_form_filtered_off_the_page_saves_to_the_reader_item_after_trash_back` | `CI_FIX_MODE=wrongreader` |
| `held-old-03`, `held-new-already-list-01` | `-p test_held_publication Tests/UI/test_library_note_links_picker_and_preview.py::test_typing_two_brackets_opens_a_keyboard_picker_that_inserts_a_real_link` | Private helper oracle; actual original test node supplies unchanged bootstrap/autouse setup. The latter adds already-list coverage; the former executed before helper edits. |
| `extended-old-helper-01` | `-p wave_probe -p old_helper_baseline Tests/UI/test_library_note_links_picker_and_preview_extended.py::test_enter_on_the_focused_preview_follows_the_only_link_and_lists_several` | `CI_FIX_MODE=oldextended`; exact compiled old helper, no readiness prep |
| `extended-old-helper-ready-01` | `-p wave_probe -p old_helper_baseline -p preview_observer Tests/UI/test_library_note_links_picker_and_preview_extended.py::test_enter_on_the_focused_preview_follows_the_only_link_and_lists_several` | `CI_FIX_MODE=oldextendedready`; exact old helper, actual source readiness |
| `extended-new-helper-ready-01` | Same target/plugin arguments as prior row | `CI_FIX_MODE=newextendedready`, `CI_USE_OLD_HELPER=0`; actual current helper, same readiness |
| `held-preview-old-02`, `held-preview-new-01` | `-p held_preview Tests/UI/test_library_note_links_picker_and_preview.py::test_typing_two_brackets_opens_a_keyboard_picker_that_inserts_a_real_link` | `PREVIEW_HOLD_MODE=old` / `new`; private native publication boundary control, actual original UI bootstrap |
| `preview-amend-owned-01` | `-p wave_probe -p old_helper_baseline -p preview_observer Tests/UI/test_library_note_links_picker_and_preview_extended.py::test_enter_on_the_focused_preview_follows_the_only_link_and_lists_several Tests/UI/test_library_note_links_preview_opens_in_app.py` | `CI_FIX_MODE=previewamend`, `CI_USE_OLD_HELPER=0`; actual amended helper; private readiness only applied to extended in this batch |
| `preview-amend-sibling-ready-01` | `-p wave_probe -p old_helper_baseline -p preview_observer Tests/UI/test_library_note_links_preview_opens_in_app.py` | `CI_FIX_MODE=previewamendsibling`, `CI_USE_OLD_HELPER=0`; same actual native source readiness applied to sibling |

Executed probe versions are retained, including the exact old-helper ready
plugin and the recovered first invalid full-held plugin whose SHA256 was
independently checked against its recorded digest. The first failed scratch
collection command/log remain private; no real-config contents are copied
into this report.

## Invalid and negative attempts

- `held-old-01`: collection error, bootstrap marker not registered because a
  scratch test was collected outside `Tests/UI`. Importing its support ahead
  of UI conftest also read the real config; no App ran. Excluded entirely from
  mechanism evidence; subsequent probes use the original UI node/plugin path.
- `held-old-02`: entered App but failed the first Index open because startup
  sources did not answer within the existing product five-second admission.
  No hold ran; excluded from publication RED evidence.
- `held-old-03`: qualified old-helper pending/publication RED. Actual New Blank
  and native title gestures preceded reopen. After actual placements DOM
  mounted with the real folders answer still held, `_open_notes` had returned
  and the target row had already been pressed. Assertion:
  `old helper returned before the held folder publication`. Holds released and
  drained inside App; no watchdog. Original helper SHA256
  `b153b5720194669f9f73a8829988baab83851b2821c2b8a794d357072e2d736f`.
  XML SHA256 `b74651cf3ba7f0e47687edd4c064e1042f6c9f31fc82e078ff25bc65c3f3610f`.
  Raw `held-old-qualified.trace.json` and `held-old-qualified-probe.py` retained.
- `held-full-new-01`: full original owned-source journey with permitted title
  prep failed before any reopen or hold: title disappeared during original
  `_type(VERIFIER)` and original title equality raised NoMatches. Excluded
  from new-helper publication qualification, not described as CI-line245 RED.
  XML SHA256 `b07942c95ec9c376d509cd3fb495bae876bc81d19f427334c5763ee9a47c12e4`.
  NoMatches alone does not distinguish a route change from a remove/mount gap;
  that earlier failure remains unclassified. The later original title setup
  passed in both the owned-source batch and qualified full-held run, so no
  broader title preparation or product focus change was introduced.

## Static conservation and self-review

`static_check.py` captures baseline and after-source Ruff `check --no-cache`
and formatter `--no-cache --diff`, plus structural comparisons. All three
scoped Ruff checks pass. Formatter debt existed in all three original files;
the exact normalized formatter diff regions are conserved rather than changing
unrelated owners. The full original link-test function, every unaffected
function/class owner, Reader statements after `_browse_settled`, and title
typing plus all following statements compare AST-identical.
The final scoped `git diff --check -- <the three allowed files>` returned zero.
The Preview amendment's tree/open publication owners compare AST-identical
against phase-one source, explicitly supporting the approved decision not to
repeat those already-qualified controls.

The setup uses no Input/controller assignment, fake positive target, service
return replacement, global worker drain, fixed sleep, row-action retry,
assertion deletion, broadened condition, skipped target or timeout increase.
Negative Reader mutation is explicitly confined to its private negative run.

Final receipt SHA256:
`0d985c83528e6437644c814a80b2dab56740a8d54138f7b3fd89f5f6a730c236`.
Selected raw artifact pins (all remaining source/XML/probe/trace pins are in
that receipt):

| Artifact | SHA256 |
| --- | --- |
| `held-old-qualified.trace.json` | `edc2eee54ade66914bc8624420706e3b11250f91ec46b6acfb24b81b78371d20` |
| `held-full-new-qualified.trace.json` | `fc2908d9602c509179189c23192463d59fe712f7a8bbad77f5a049f6b9649ebb` |
| `held-new-already-list.trace.json` | `6a55674e00ceea2d71f12d753ca8194c3b3b242b001300ebe8f68f6ed13834d0` |
| `preview-old-helper-ready.trace.json` | `7dbb8c171d0a94a93c69cc2d2c6d46ae509f25e34c021583b20183b9ebec17cb` |
| `preview-new-helper-ready.trace.json` | `09fbeb9623a0fb568b4a83bf839249470e0c711bf81fcd36efa9d40be350a9bf` |
| `preview-comparison-summary.json` | `dbe7ba0fd45d0d30588cc337a21a611e3bad99fca7a5d027e6e4c2a4311c5191` |
| `held-preview-old-qualified.trace.json` | `ba73a94db066be460405e16a78a41ebd596e3e51431d1f8e8bb7c6fef8b58dbe` |
| `held-preview-new-qualified.trace.json` | `f99c36fcb544dbde361f4ee26694a5d967ede347c4e2fb3c7c90b91cb2fbfad1` |
| `preview-amend-extended-green.trace.json` | `4cf0e756324a65a320dc4255c9c0b47bbb41faf26e19ef07c819bc6a2f546d0b` |
| `preview-amend-sibling-green.trace.json` | `c6f95ac11243670c6026dad7d361b12fd76ca95d9ddb75bb7aafdcc15dcc2849` |

Limits: these checks establish native preparation and preservation of existing
oracles, not a product filter/focus/click-race fix or hosted CI success. Root
must still push and obtain actual green CI on the final committed head.
