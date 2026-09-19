# Shared forms, buttons and dialogs audit

Baseline: component-pattern-library e35e6bea4634e27e8620f0c40e72c45ae7a0050d. Read-only production audit; no implementation or Backlog edits. Adapted Impeccable checks to Textual cells, compositor paint and keyboard focus. No web/ARIA/touch-target claims. Upstream comparison uses origin/dev fd30614.

## Verified findings

### P2 — Gallery's inline form exemplar clips its own input

Location: `tldw_chatbook/Widgets/pattern_gallery.py:60-62`; shared width declaration `tldw_chatbook/css/components/_forms.tcss:283`; catalog inline skeleton `backlog/docs/component-patterns.md:123-125`.

The gallery places the Temperature label beside a `.form-input`, but that input requests 100% of the entire row. At 80 columns the parent is x=1,width=77 and the input is x=12,width=77 (right edge89 versus parent78); at120 columns this remains an11-cell overrun. Both dark/light, rest/focus paint a missing right border. The catalog's plain Container sample also calls the same label/control combination a horizontal row although `.form-row` itself does not establish horizontal layout.

Repro: open Design System: Pattern Gallery, scroll to Temperature, Tab into the input; right edge is clipped at both tested widths. Evidence `gallery-textual-dark-80-3-{rest,focus}.{json,txt,svg}`, matching120/light captures and `gallery-overflow-*` show geometry and actual clipped border. Sample value remains short and readable; this report does not claim a reproduced lost suffix or broad production form failure. Impact: the canonical reference teaches an overflowing composition, so downstream implementations cannot safely copy it.

Recommendation: make this exemplar and catalog skeleton agree on a contained label+remaining-width input pattern, or demonstrate the existing form-col stacked field composition; pin child containment and painted edges, not just tall screenshots. Scope the fix to examples unless production consumers are separately verified.

Branch-specific: pattern_gallery.py/catalog do not exist in origin/dev. Shared full-width `.form-input` declaration is inherited; this is the new exemplar's misuse of it. Existing TASK-32532.4 is Done and records gallery creation, without this defect; no exact corrective task found.

### P2 — Shared Settings label columns squeeze compact controls at80 columns

Location: `tldw_chatbook/UI/Screens/settings_screen.py:23271-23295`; geometry owner `tldw_chatbook/css/features/_settings.tcss:352-357` and compact input sizing in `_forms.tcss`.

At80×24 the Network card's inner row is35 columns. The fixed24-column label consumes most of it, leaving the TLS Select11 columns (9 content columns, with control chrome further reducing label space). The actual closed Select paints `Verif` for `Verify certificates`, and `Cust` for `Custom CA bundle`. The path input has8 content columns and shows only `ca.pem` after typing `/missing/ca.pem`. At120×45 the same Select paints `Verify certificates` completely. All four dark/light×width runs reproduce the distinction. No field disappears and path input still accepts full text.

Repro: Settings → Network at80×24, inspect the closed Certificate verification selector; choose Custom CA bundle and type a path. Evidence `network-textual-{dark,light}-80.{json,txt,svg}` and `network-*-80-typed.*`;120 counterparts are the control leg. The same probe records the Providers model field at x=65,width=10,content width7, painting only `-terra` (`settings-textual-dark-80-4-focus.*`). Providers uses the same label/row/input classes at settings_screen.py:15690-15708 (Model/Endpoint) and15761-15768 (API key). Parent independently observed the same pressure in native Console setup→Providers at80×24; combine both consumers into one shared Settings compact-row task, with the parent’s native artifact citation alongside these mounted Network captures.

Recommendation: let shared compact Settings rows stack label and control when its usable width cannot show the value, preserving full closed policy names and a practical editable path width at80 columns. Retain the one-row compact field border/focus contract when appropriate.

Inherited, not migration regression: `_render_network_detail` is byte-identical to origin/dev (saved branch-network.txt/upstream-network.txt); origin/dev's `_agentic_terminal.tcss:8609` still sets24-column labels, and compact inputs remain1fr. TASK-24653 created Network and is Done; TASK-1342 addresses sidebar/outer layout, and TASK-1713 widened labels for other settings. No exact existing fix task was found by filename/content search.

## Positive checks and coverage

| Surface/state |120×45 dark/light|80×24 dark/light|Evidence/limit|
|---|---|---|---|
| Gallery standard Input, Select, Checkbox rest/focus|Paint preserved|Paint preserved except Temperature right edge|Glyphs/values and borders inspected in compositor text/SVG; hidden Advanced Input excluded|
| Gallery form/action/sidebar Buttons focus|Visible bold underline|Visible bold underline|Primary action retains primary fill but gains underline; not an invisible-focus finding|
| Disabled gallery Button|Readable|Readable|Painted label contrast7.25:1 dark,4.72:1 light; opacity1; focus attempt stays unfocused|
| ConfirmationDialog title/message/actions|Complete|Complete|Cancel and Discard changes labels render; Escape exercised|
| Canonical Settings provider/model compact inputs|Paint preserved|Paint preserved but narrow model suffix|Focus gets thick left edge and visible text|
| Network typed path and rejected invalid save|Passed|Passed|Real keypresses produce exact `/missing/ca.pem`; validation yields `CA bundle path invalid: Path does not exist`|
| Invalid compact-input specimen|Text preserved|Text preserved|`settings-invalid-input` applied to real mounted model field; pink/tint computed state retained; no claim that this state was reached via production model validation|
| Disabled hover|Not completed|Not completed|Probe's dynamic ID was not query-visible; no application failure inferred|

Round1 command selected provenance, temporary mounted matrix and existing disabled-contrast tests:12 passed. This includes eight matrix cases, three existing contrast tests, and provenance. Round2 selected four cases and captured all above network/invalid/gallery/disabled data, then each stopped at the final hover step with a harness `NoMatches('#audit-disabled')`. First Round2 attempt had stopped earlier because direct Save was invoked while Input retained focus; clearing focus reached the real validation notification on the corrected attempt. Both logs are retained. No green full-round2 claim. No whole-UI or full suite run. No third evidence expansion.

The invalid-input `*-contrast.json` files are explicitly NOT text contrast evidence: the reused first-glyph helper sampled the left border, not the input text. Actual SVG text shows `-terra` in #e0e0e0 (dark) / #1f1f1f (light); no contrast finding is made from that helper.

No native terminal screenshot, mouse hardware test, performance profiling, filepicker modality, multi-line TextArea interaction or complete Settings category sweep was performed. Findings rely on production stylesheet mounted compositor output, and are limited to the stated paths. No P0/P1 identified in this bounded sample.

Temporary test source is preserved as `probe.py` in this evidence directory and removed from Tests/UI. Production files unchanged by this agent.
