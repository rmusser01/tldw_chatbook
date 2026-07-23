# RAG Scope Narrowing — Phase 2 QA captures (2026-07-21)

Live app via textual-serve, real production CSS bundle, isolated HOME seeded with 5 tagged media items.

- **01-inspector-unscoped-row.png** — Inspector "Retrieval scope" row below the Sources tray, unscoped state ("Scope: everything · Narrow…"); header scope chip correctly hidden.
- **02-scope-picker-modal.png** — ConsoleScopePickerModal: title names target ("Narrow RAG scope — Chat 1"), All/Media/Notes tabs, title filter, tag chips with counts (sales(4)/q3(2)/hr(1)/research(1)), Sort: Recent, All/Selected toggle, item list, pagination, Select-all-matching/Clear-shown, "N selected of M · Save · Clear scope · Cancel".
- **03-modal-two-selected.png** — Two Q3 Sales items checked (green), footer "2 selected of 5".
- **04-scoped-row-chip-recipe.png** — After Save: header chip "Scope: 2"; Inspector row shows Edit/Clear; run-recipe line reads "… / scope 2 items".
