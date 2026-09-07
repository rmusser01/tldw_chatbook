---
id: TASK-2300
title: Watchlists Selects render empty option lists
status: Done
assignee: []
created_date: '2026-08-04'
labels:
  - watchlists
  - bug
  - uat-2026-08-04
dependencies: []
priority: high
---

## Description (the why)

UAT (2026-08-04, live tmux, fresh profile): two Selects on the Watchlists
screen open with no options at all. The Items tab's status filter opens an
empty floating overlay (a bare border, nothing selectable), and the New Rule
form's condition Select displays "No items". Both are dead controls; the
items filter one compounds TASK-2301 into ingested/ignored items being
unreachable. Likely one root cause (option population), possibly interacting
with the recently-added PruneSafeSelect guard — diagnose before fixing.

UAT findings F30 (critical), F36.

## Acceptance Criteria (the what)

- [x] The Items status filter opens a populated option list covering every
      item status the backend can produce, and picking one filters the list.
- [x] The New Rule condition Select offers the real condition vocabulary.
- [x] The root cause is identified and recorded in the task notes (including
      whether PruneSafeSelect was involved), with a regression test that
      fails when option population breaks again.
- [x] Verified live in a real terminal, not only under pytest.

## Implementation Plan

1. Diagnose empirically before touching anything: mount the production
   Watchlists screen with the production stylesheet, expand
   `#items-status-select`, and read the compositor's painted rows (not the
   widget's `option_count`, which can be right while the screen is wrong).
2. Establish whether `PruneSafeSelect` is involved by measuring `_pruning` /
   `_closing` and the overlay's option count at the moment of expansion, and
   record the answer in the notes either way.
3. Fix the mechanism that actually destroys the options, at the layer it
   lives in, following the TASK-1160 precedent for the same app-wide rule.
4. Regression test that reads the RENDERED rows through the real compositor,
   so it fails again the moment options stop reaching the screen -- an
   `option_count` assertion would have stayed green through this defect.
5. Verify live in a real terminal at 235x52.

## Implementation Notes

Option population was never broken. `SelectOverlay.option_count` was **6**
throughout, the right six `Option` objects were in the list, and
`Select.value` was `"all"`. **`PruneSafeSelect` was NOT involved**: measured
at the moment of expansion, `_pruning=False`, `_closing=False`, and
`_setup_options_renderables` had run normally. The guard is untouched by this
task and remains in place.

The options were destroyed on their way to the screen, by CSS. Three
app-wide rules, all written for a widget shape that has chrome rows to
spare, applied to widgets that have none:

| Rule | Where | Victim |
|---|---|---|
| `*:focus { outline: solid $ds-focus-accent; }` | core/_reset.tcss | the compact `SelectOverlay` |
| `Input:focus, TextArea:focus, Select:focus { border: solid ... }` | components/_forms.tcss | every `Select` |
| `Select:hover { border: solid ... }` | features/_embeddings.tcss | every `Select` |

**Why they destroy content rather than decorate it.** An `outline` is
painted OVER a widget's outermost rendered lines and costs no geometry, so
those lines must be expendable. A `border` costs geometry -- unless the box
model has already been sized without room for it, in which case the content
is pushed out. A COMPACT `Select` is ONE row (`OptionList.-textual-compact`
sets `border: none !important; padding: 0`, and this app's filter strips pin
`height: 1`), and its overlay likewise reserves no perimeter at all, so
every cell of both is content. A NON-compact `Select` is three rows and
draws its border on its child `SelectCurrent`, so a border on the `Select`
itself is a *second* one with nowhere to go.

Measured at 235x52 against the production stylesheet:

```
#items-status-select overlay        #items-status-select control
before        after                 before (focused)   after
┌──────────────┐  All statuses      ┌──────────────┐   All statuses  ▼
│ew            │  New               (value gone)
│ead           │  Read
│ngested       │  Ingested          #rules-create-condition (focused)
│gnored        │  Ignored           ┌──────────────────────┐
└──────────────┘  Error             │▊▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▎│  ->  No items
                                    └──────────────────────┘
```

Two of six options gone and four mangled; and for any compact Select with two
options or fewer (`#watchlists-backend-select`) the overlay renders as a bare
box with nothing in it -- which is what the UAT reported. The `:hover` half is
the one that best explains the report: it fires on the way TO clicking, so the
control was already a bare border before anything had been chosen.

### The fixes, all three following TASK-1160's precedent

* `SelectOverlay:focus { outline: none; }` -- the overlay is only ever on
  screen while it holds focus, so a focus ring conveys nothing there; the
  highlighted-option recolour is the real cue.
* `Select.-textual-compact:focus/:hover { border: none; outline: none; }`
  plus the same exemption on its `> SelectCurrent`. Focus keeps
  `_forms.tcss`'s own `background`/`color` recolour; hover takes the
  `$surface-lighten-1` this file already gives `OptionList` rows.
* `Select:focus`/`Select:hover` borders move to `> SelectCurrent`, onto the
  `tall` border Textual already reserves space for -- geometry byte-identical
  to Textual's own focused state, cue unchanged.

The inner exemption is load-bearing, not belt-and-braces: without it the
third fix moves the empty box inward instead of removing it (measured -- the
compact filter went straight back to `▊▔▔▔▔▔▔▔▔▔▔▔▔▔▔▎`).

### F36 was a misreading, and it is left alone

The New Rule condition Select "displaying No items" is correct: `No items` is
the first entry of `RulesPane._CONDITION_OPTIONS`, a real alert condition, not
an empty-list message. Its overlay offers all five conditions. The genuine
defect on that control was the focus/hover one above.

### Verification

* New file `Tests/UI/test_watchlists_select_option_overlays.py` (7 tests),
  every assertion read off the rows the **compositor painted**, through the
  production stylesheet in the full shell. That shape is the point: an
  `option_count == 6` assertion stayed green through the entire defect.
* Mutation-verified: 4 mutations, each reverted individually -> RED ->
  restored byte-exact (md5).
* Gates: `test_destination_visual_parity_correction.py` +
  `test_watchlists_source_create_form.py` + `test_destination_shells.py` +
  `test_destination_headers.py` **252 passed, 1 skipped**;
  `test_console_session_settings.py` + `test_console_scope_picker_modal.py` +
  `test_evals_bench_editor.py` + `test_evals_card_picker.py` +
  `test_console_workbench_contract.py` + `Tests/Widgets/test_prune_safe_select.py`
  **330 passed**; `--collect-only Tests/UI Tests/Watchlists` **8639
  collected**, no errors.

### Live verification (235x52, fresh profile, tmux)

Two of the three defects here were found live with the suite already green,
after the first fix had shipped. Final state:

```
=== filter at rest ===        All statuses  ▼
=== filter OPEN ===           All statuses  ▲
                              All statuses / New / Read / Ingested / Ignored / Error
=== condition Select ===      ▊  No items          (focused, value intact)
                              No items / Error rate above / Items below /
                              Items above / Run failed
```

### Files

* `tldw_chatbook/css/components/_lists.tcss`,
  `tldw_chatbook/css/components/_forms.tcss`,
  `tldw_chatbook/css/features/_embeddings.tcss` (+ the regenerated bundle).
* `Tests/UI/test_watchlists_select_option_overlays.py` (new).
