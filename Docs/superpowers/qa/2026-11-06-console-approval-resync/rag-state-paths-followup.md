# Lightweight protected RAG state paths

ADR-355 moves the existing RAG state-directory selection into one lightweight owner. Console protection and both legacy public getters share it. The change preserves environment/current/legacy priority, value coercion, home expansion, relative paths, live configuration and user-data reads. Sensitive-path memoization, failed-read handling, permission authority and recovery are unchanged. Configuration-read errors use constant text and a controlled exception type rather than raw exception values.

A guarded original-helper probe on `f3cfffd9` took 1.5782705 seconds and loaded 570 additional dependency names. This is one diagnostic sample, with its observer overhead included. It does not explain the separate full-app 60-second workspace-binding wait. The optional dependency edge was independently reproduced by invoking the original protected-directory getter with RAG/NumPy imports blocked; it failed before extraction and passes afterward. No after-change runtime probe or Console performance claim is made here.

Targeted verification on the four production and two test files:

| Check | Result |
| --- | --- |
| Lightweight path/import/currentness/privacy regressions | 16 passed, no skips |
| Existing sensitive-path and RAG compatibility cases | 72 passed; eight failures reproduced on exact dev8b; one Windows symlink-privilege limitation |
| Optional Chroma cases | Two existing optional skips; one unrelated runtime case excluded |
| Required-lane enrollment contracts | Six passed; 167 ordered fast targets, existing admission limits unchanged |
| Canonical embedded worker, stamp, ping and real fs_list | 20 passed, one existing POSIX-only skip |
| Production diagnostic inventory | Canonical rebuild and verification pass; 730 owners, unchanged classification/privacy/sinks and total calls |
| UI-ready / eager modules | 1038/1038 passes, six approval helpers absent; strict eager limit fails at 688 identically on exact dev8b |
| Two file-tool protection nodes | Open: both stop at the same config-selection fixture refusal on exact dev8b, before denial assertions |
| Changed-source static checks | 91 Python files parse; Ruff 185 versus 190 baseline diagnostics, zero new fingerprints; both new files formatted |

The eight compatibility failures are seven existing raw-source-selection refusals and one Windows HOME-only expansion expectation. They were not converted to passes. No denial assertion, bootstrap rule, timing budget, worker protocol, policy constant or CI resource limit was relaxed. The UI census remains 1816; the non-UI regression is enrolled in the fast owner lane. The public guarded launcher retains all 64 prior targets and adds eight finite targets without duplicates.

An independent review of the four-file production delta found no material findings. The canonical worker diff contains only the corresponding unavailable-import stub and final identity stamp. All 15 worker inputs and its 13-module standard-library closure remain intact.

Source-bound receipts remain private under `.superpowers/pr3036-resync-20261106/`: `rag-state-paths-20261107T061500Z-2d6f`, `rag-state-ci-20261107T061300Z-d4e`, `bundle-rag355-20261107T062303Z-8685a6bf`, `provider-rag355-finalchecks-20261107T062739Z-bf0bbc30`, `diagnostic-rag355-20261107T063300Z-e82`, and `static-rag355-final-20261107T065000Z-82c`. The adjacent manifest binds current implementation and verification artifacts. Historical failed attempts are retained. No full suite, model calls or external benchmark traffic was used.

Fresh source freezing, the same isolated after-probe and genuine native/browser Console qualification follow this commit. Product timing samples remain zero. TASK-34564 and TASK-34569 remain In Progress; the PR remains draft with auto-merge off until qualification and current-head hosted review/CI pass.
