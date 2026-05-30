# Non-Obscuring Focus Widget Exception Matrix

Date: 2026-05-28

| Widget/control | Supported cues | PR 1 behavior | Exception/fallback | Test |
| --- | --- | --- | --- | --- |
| `Button` | background, foreground, text-style, outline | underline plus subtle background, no heavy outline | none | `Tests/UI/test_non_obscuring_focus_contract.py` |
| `Input` | border, background, foreground, outline | thin border plus subtle bottom emphasis | no full-field heavy frame | `Tests/UI/test_focus_accessibility.py` and `Tests/UI/test_non_obscuring_focus_contract.py` |
| `TextArea` | border, background, foreground, outline | thin border plus subtle bottom emphasis | no full-field heavy frame | `Tests/UI/test_focus_accessibility.py` and `Tests/UI/test_non_obscuring_focus_contract.py` |
| `Select` | border/background varies by Textual widget | thin border plus subtle bottom emphasis where supported | if native styling resists bottom emphasis, document verified fallback here and in visual QA | `Tests/UI/test_focus_accessibility.py` and visual/mounted exception note |
| `Checkbox` | native check mark, foreground, outline | global fallback only in PR 1 | migrated after screen-by-screen review | deferred |
| `RadioButton` | native check mark, foreground, outline | global fallback only in PR 1 | migrated after screen-by-screen review | deferred |
| `DataTable` | cursor row and selected row styling | shared neutral row background, readable text, bold underline for cursor/selected rows | feature styles must not reintroduce primary/accent row fills | `Tests/UI/test_non_obscuring_focus_contract.py` |
| `Tree` | cursor/selected row styling | deferred | row highlight fallback allowed if readable | deferred |
| `ListView` | item hover and selected/highlight classes | shared neutral hover background, readable text, bold underline for highlighted rows | feature styles may scope local rows but must keep the shared row-state contract | `Tests/UI/test_non_obscuring_focus_contract.py` |
| `SelectionList` | selected/highlight option styling | deferred | row highlight fallback allowed if readable | deferred |
| `Tabs` | tab active/focus classes | top nav only in PR 1 | native `Tabs` deferred | deferred |
| Custom button/list rows | background, marker text, classes, text-style | PR 1 for named Console/Library rows only | each custom row needs explicit selector audit | source contract or mounted test |
