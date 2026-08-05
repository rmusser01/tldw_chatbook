# Phase-to-evidence matrix

| Phase | Lane | Evidence | Verdict |
|---|---|---|---|
| Existing root, tree, editor, autosave | Production app PTY | `actions.txt`, `wide-phases.txt` | Pass |
| Exact frontmatter preservation | Production app PTY + digest | `evidence.json` | Pass |
| Session stage/unstage and promise/count | Production app PTY | `actions.txt`, `wide-phases.txt` | Pass |
| Session-only reviewed commit | Production app PTY | `actions.txt`, `evidence.json` | Pass |
| Authorization cancel: zero network | Production app PTY + fixture counter | `evidence.json` | Pass |
| Authorization: one read-only check | Production app PTY + fixture counter | `evidence.json` | Pass |
| Frozen SSH review facts | Production app PTY | `wide-phases.txt`, `evidence.json` | Pass |
| Trust drift before push | Production app PTY + fixture counter | `evidence.json` | Pass |
| Exact successful ref transition | Production app PTY + refs/counters | `evidence.json` | Pass |
| Divergent destination refusal | Production app PTY + refs/counters | `evidence.json` | Pass |
| Original uncertain outcome | Production app PTY, induced bounded delay | `wide-phases.txt`, `evidence.json` | Pass |
| Dismissed result survives ordinary refresh | Production app PTY + mounted regression | `wide-phases.txt`, `evidence.json` | Pass |
| Explicit reopen restores query-only recovery | Production app PTY + mounted regression | `wide-phases.txt`, `evidence.json` | Pass |
| 40x20 keyboard/accessibility behavior | Production app PTY | `compact-viewport.txt` | Pass |
| POSIX descendant settlement | Focused native automated lane | `process-tree.txt` | Pass |
| HTTPS transport | Focused automated transport lane | `evidence.json` | Pass (automated only) |
| Windows Job Object containment | Focused suite on macOS | `process-tree.txt` | Skipped (not native) |
