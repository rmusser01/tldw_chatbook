- **Query embedding before compatible Ready** — ADDRESSED. `tldw_chatbook/Workspaces/project_index/service.py:640-705` issues the current artifact carrier, checks `semantic_ready_generation` under a fenced SQLite read, retires registry ownership, and closes the carrier on refusal before constructing the wrapper. Post-embed catalog/artifact, Ready and live-file validation remains at `service.py:797-845`. The final-source incompatible-artifact test at `Tests/Workspaces/test_project_semantic_search.py:388-577` asserts a keyword hit with zero wrapper constructions.
- **Between-page/final-CAS mutation and cancelled query-owner controls** — ADDRESSED. `Tests/Workspaces/test_project_semantic_generations.py:589-656` uses 33 real SQLite chunks and changes the committed keyword manifest revision after page one or after the final append; both paths refuse publication and settle Error. `Tests/Workspaces/test_project_semantic_search.py:579-870` invokes the real tool on a worker, cancels two observers while the original owner waits on failed native close, checks concurrent keyword results, then proves two same-thread retries, carrier/factory retirement and no pending operation.
- **New generation test formatting** — ADDRESSED. The focused `ruff format --check --no-cache` on both changed semantic test files returned `2 files already formatted`; `Tests/Workspaces/test_project_semantic_generations.py:1` is the final formatted file.

### New Breakage in the Fix Diff

None. Critical 0; Important 0; Minor 0.

### Out-of-Scope Observations

None.

### Checks

- Read the complete 2,342-line packaged fix diff in bounded chunks and inspected the changed service path and test cases. Current source SHA256 values match the final run receipts: service `6BDED02A5BB7`, generations `A6990D3C47DF`, search `FE9121B57849`.
- Inspected retained final-source parent outputs: `task3-r1-mutations-01` reports 2 passed; `task3-r1-final-preflight-01`, `task3-r1-final-query-owner-01`, `task3-r1-existing-query-close-01` and `task3-r1-synonym-01` each report 1 passed. The final preflight and query-owner child XML each record one test with zero failures, errors or skips. No test was replayed.
- Ran focused `ruff format --check --no-cache` on the two changed test files: exit 0, two files formatted.

### Verdict

**Fix round:** All findings addressed, no new Critical/Important breakage.
