# Non-Obscuring Focus Widget Exception Matrix

Date: 2026-05-28

| Widget/control | Supported cues | PR 1 behavior | Exception/fallback | Test |
| --- | --- | --- | --- | --- |
| `Button` | background, foreground, text-style, outline | underline plus subtle background, no heavy outline | none | `Tests/UI/test_non_obscuring_focus_contract.py` |
| `Input` | border, background, foreground, outline | thin border plus subtle bottom emphasis | no full-field heavy frame | `Tests/UI/test_focus_accessibility.py` and `Tests/UI/test_non_obscuring_focus_contract.py` |
| `TextArea` | border, background, foreground, outline | thin border plus subtle bottom emphasis | no full-field heavy frame | `Tests/UI/test_focus_accessibility.py` and `Tests/UI/test_non_obscuring_focus_contract.py` |
| `Select` | border/background varies by Textual widget | thin border plus subtle bottom emphasis where supported | if native styling resists bottom emphasis, document verified fallback here and in visual QA | `Tests/UI/test_focus_accessibility.py` and visual/mounted exception note |
| `Checkbox` | native check mark, foreground, outline | shared toggle focus keeps labels readable with subtle background and underline | native check mark remains semantic; label focus must not use block cursor fills | `Tests/UI/test_non_obscuring_focus_contract.py` |
| `RadioButton` | native check mark, foreground, outline | shared toggle focus keeps labels readable with subtle background and underline | native selected mark remains semantic; label focus must not use block cursor fills | `Tests/UI/test_non_obscuring_focus_contract.py` |
| `DataTable` | cursor row and selected row styling | shared neutral row background, readable text, bold underline for cursor/selected rows | feature styles must not reintroduce primary/accent row fills | `Tests/UI/test_non_obscuring_focus_contract.py` |
| `Tree` | cursor/selected row styling | shared neutral cursor and hover backgrounds, readable text, bold underline for cursor rows | feature styles may scope local tree rows but must keep the shared row-state contract | `Tests/UI/test_non_obscuring_focus_contract.py` |
| `ListView` | item hover and selected/highlight classes | shared neutral hover background, readable text, bold underline for highlighted rows | feature styles may scope local rows but must keep the shared row-state contract | `Tests/UI/test_non_obscuring_focus_contract.py` |
| `SelectionList` | selected/highlight option styling | shared neutral selected and highlighted button backgrounds, readable text, bold underline | feature styles may scope local choices but must keep the shared row-state contract | `Tests/UI/test_non_obscuring_focus_contract.py` |
| `Tabs` | tab active/focus classes | shared native active/focused-active tabs use subtle background, readable text, and underline | feature styles may scope native tab active states but must not reintroduce block-cursor fills | `Tests/UI/test_non_obscuring_focus_contract.py` |
| Custom button/list rows | background, marker text, classes, text-style | named Console/Library rows plus Console session tabs use readable focus/selected cues | each custom row needs explicit selector audit | source contract or mounted test |
