# Core-loop upgrade wave — QA evidence (2026-07-12)

Branch: `claude/uat-upgrade-wave-2026-07` (worktree at dev @ 89de6554, post PR #606).
Implements the UAT upgrade tasks 188/189/190/191 and completes task-185 (keyword stemming).
Captured live from textual-serve (real CSS, worktree code), playwright chromium 2050×1240,
fresh isolated profile, live llama.cpp (Qwen3.6-27B) reachable on the llama.cpp DEFAULT port via a
local 8080→9099 TCP forward so default-port detection is exercised honestly.

- `setup-card-detected-local-server-2026-07-12.png` — task-188: on a VIRGIN profile the blocked
  setup card detects the local server and offers **"Use detected llama.cpp (127.0.0.1:8080)"**
  under "Set up provider" (probes are strictly localhost, 2.5s, llama.cpp + Ollama defaults plus
  configured local endpoints).
- `one-click-connect-console-ready-2026-07-12.png` — pressing it wrote provider/endpoint/model
  (model auto-picked from `/v1/models`) and unlocked Console in the same session: chip bar
  `Provider: llama_cpp · Model: Qwen3.6-27B-…`, rail `Streaming: on`, "Ready — type a message to
  begin." **Steps-to-first-chat: one click** (was ~9 manual steps in the 2026-07-11 UAT).
- `home-real-state-ready-counts-resume-2026-07-12.png` — task-190: Home canvas shows
  "Start a conversation / Console is ready for a task. / Conversations: 1", a resume row for the
  newest conversation, and the primary Start control (fresh-config readiness; counts via the same
  seams as the Library rail incl. `scope_type="all"`).
- `search-plural-matches-singular-2026-07-12.png` — task-185: query **"feedback loops"** matches a
  conversation containing "feedback loop" (safe FTS5 variant expansion), secondary line pluralizes
  correctly ("1 message").
- `settings-connect-block-first-2026-07-12.png` — task-189: Providers & Models leads with the
  Connect block (provider/model/endpoint/credentials/Test Provider + model discovery) and folds
  sampling into a collapsed **Generation defaults** disclosure; the eight "Unavailable for X" rows
  are gone (one summary line when applicable).
- `settings-test-honest-failure-toast-2026-07-12.png` — task-191: Test toast reports the actual
  outcome ("Provider test failed: openai is not ready: Missing API key…"); for URL-based providers
  a passing test also live-probes the endpoint and reports reachable(model count)/refused/timeout.

## Residuals / follow-ups
- Settings "Provider source: Current app selection" still reflects the boot reactive after the
  one-click Console connect (Console runs llama.cpp; Settings displays OpenAI until reselected) —
  same boot-echo family as task-177, Settings-side. Follow-up candidate.
- The Console "Generating…" placeholder is covered by a transcript unit test but was not visible
  in served captures during the reasoning phase — verify the live render path.
- `console_model_popover` still renders raw provider keys (minor; modal + Settings share the
  catalog now).
- Two first-run smoke suites were failing on dev itself (stale pins from the July shell renames,
  masked by cancelled CI) — repaired in this branch.
