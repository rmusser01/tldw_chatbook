# Library + Library ▸ Notes — Sr Designer / HCI review (2026-10-02)

Target: origin/dev `2d34cbf80d`. Scope: the Library destination (Media, Ingest, Search/RAG, Export,
Collections, Conversations, Prompts, Skills, Artifacts) and its Notes sub-destination (editor, tree,
folders, links, import once, lasting sync, Folder files), walked as first-time and power users.

| File | What |
|---|---|
| `report.md` | The review: Nielsen scores (Library 17/40, Notes 16/40), P0–P3 issues with fixes, roadmap, persona red flags, appendix of all findings |
| `findings.md` / `findings.json` | Final verified ledger (103 findings; severities, tracking, fixes) |
| `consolidated-findings.md` | Pre-verification consolidated list with full repro steps and sources |
| `journeys/` | Seven live persona walkthroughs (step logs, task outcomes, scores) |
| `assessment-b/` | Detector result, route × size mechanical metrics, static code sweeps |
| `maps/` | Code-derived UX maps of each surface (flows, states, bindings) and the known-context pack |
| `improvements/` | Three ideation lenses and the ranked roadmap |
| `verify/` | Text/ANSI captures and logs cited by findings (only the 150 cited files are committed) |

Method: dual-assessment impeccable critique — A (7 live persona journeys) and B (detector, mechanical
matrix, static sweeps) run in isolation; consolidated; every finding re-reproduced by an independent
refute-first verifier and triaged against backlog/ADRs. App driven in isolated tmux profiles with a
mock OpenAI-compatible LLM. Local scratch paths in captures are normalized to `$HARNESS` /
`$TMPROOT/…` / `~`. Filed as TASK-34000 and its subtasks.
