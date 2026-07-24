# Skills Runtime — Remote Fetch (install a skill from a GitHub link)

**Status:** Approved design (2026-07-24), amended after the pre-spec review round.
**Program:** Skills-install program, the critical-path layer for the north star
("a user asks an agent to install a skill/pack from a GitHub link"). Follows
trust (#762), bundle fidelity (#784), `$`-mention invocation (#801), and
reference-file reachability (#814) — all merged. The agent-callable install
tool is the NEXT layer and wraps this one.

## Problem

The import machinery is done and hardened (zip + folder → trust-pending →
review → `$use`), and installed skills are usable — but nothing can fetch
bytes from a URL. "Install `github.com/obra/superpowers/tree/main/skills/
brainstorming`" dead-ends at a copy-paste-clone detour. This layer adds the
fetch: URL → hardened download → the EXISTING import pipeline. It adds **zero
new import logic** — every import guarantee (caps, junk pruning, symlink/
case-fold/zip-slip rejection, bounded decompression, trust-pending landing)
stays owned by `import_skill_file`.

## Decisions (user-approved)

- **URL surface:** GitHub forms (repo root, `/tree/{ref}[/{subdir}]`, release
  assets) normalized to the codeload zipball, PLUS any direct `https://…zip`.
- **Private repos:** the existing `[github]` token attaches when configured —
  ONLY to GitHub-family hosts, stripped on any cross-host redirect.
- **Multi-skill repo roots:** error + candidate listing (single-skill install
  this layer; pack import remains its own planned layer).
- Caps/policy: **30 MB compressed** download cap (streamed, aborts mid-stream),
  ≤3 manual redirect hops, connect/total timeouts 10 s/60 s.

## Design

### 1. URL classification (pure) — `classify_skill_source_url(url)`

Recognizes and normalizes, rejecting everything else BEFORE any network I/O:

| Input | Result |
|---|---|
| `github.com/{owner}/{repo}` | GitHub ref: `https://api.github.com/repos/{owner}/{repo}/zipball/HEAD`, no subdir |
| `github.com/{owner}/{repo}/tree/{tail…}` | GitHub ref + subdir (tail split per §2) → `…/zipball/{ref}` |
| `github.com/{owner}/{repo}/releases/download/{tag}/{asset}` | Direct zip with GitHub auth eligibility |
| any other `https://…zip` | Direct zip, no auth |
| anything else (http:, non-zip non-GitHub, git@, etc.) | Rejected with a clear message |

**Normalization targets the documented API zipball endpoint**
(`api.github.com/repos/{o}/{r}/zipball/{ref}`, `HEAD` for the default
branch), NOT codeload directly: the API endpoint honors the `Authorization`
header for public AND private repos and 302s to codeload — an in-family hop
the manual-redirect policy (§3) already handles. One uniform URL shape,
correct private-repo auth (codeload-direct with a header is a known 403
trap).

Skill-name suggestion: subdir basename → else repo name → else asset/zip
basename (all through the existing `_derive_name_from_filename`
normalization).

### 2. Ref-vs-subdir disambiguation (slash-refs are real)

`/tree/{tail}` cannot be split from the URL alone when branch names contain
slashes (`feature/foo`). Rules:
- Tail has ONE segment → it is the ref, no subdir.
- Tail has MULTIPLE segments → list branches via the **existing**
  `Utils/github_api_client.GitHubAPIClient.get_branches` and take the
  **longest branch name that prefix-matches** the tail; the remainder is the
  subdir. Verified constraint: `get_branches` currently issues ONE unpaged
  request (GitHub defaults to 30/page — silent truncation) and caches for
  5 minutes, so this layer makes a **one-line edit** to it: request
  `per_page=100` (additive, backward-compatible for existing callers).
  Repos with >100 branches can still truncate: therefore "no branch in the
  (possibly-capped) list prefix-matches the tail" is treated EXACTLY like
  "API unavailable" → fall back to the first-segment heuristic, and the
  eventual 404's error copy says a slash-containing branch may need the
  plain repo URL + subdir-less form. A truncated list can only cause a
  clear failure, never a wrong silent match.

`GitHubAPIClient` is also the ONLY token source (its param→env→config
resolution with placeholder guards is reused, never reimplemented). Its API
calls touch only `api.github.com` (safe surface); it is NOT used for byte
downloads.

### 3. Hardened fetcher — new `Skills_Interop/skill_remote_fetch.py` (httpx)

Owns ALL byte downloads. Per request:
1. **https-only**; `input_validation.validate_url` first (parse-level:
   whitespace/backslash/credential/malformed-host rejection — verified it does
   NOT do IP-range checks, hence:)
2. **Resolve-and-reject before connect:** resolve the host (or take the
   IP-literal directly) and check **EVERY resolved address** (A and AAAA —
   a malicious resolver can pair a public v4 with a private v6); reject if
   ANY address is private, loopback, link-local, reserved, multicast, or
   unspecified (covers 169.254.169.254 metadata).
3. **Manual redirects:** `follow_redirects=False`; loop ≤3 hops; every hop
   re-runs steps 1–2; scheme must remain https.
4. **Auth scoping:** the token header attaches only while the CURRENT hop's
   host ∈ {`github.com`, `api.github.com`, `codeload.github.com`,
   `objects.githubusercontent.com`} (release assets 302 to the last one);
   any hop outside the family proceeds WITHOUT the header.
5. **Streaming cap:** iterate the body in chunks, abort with a clean error
   the moment cumulative bytes exceed **30 MB** (compressed). Timeouts
   10 s connect / 60 s total.
6. Content-type is advisory only (GitHub serves zips as `application/zip`;
   octet-stream accepted); the zip's own magic/parse failure is the real
   gate downstream.

**Documented residual:** DNS rebinding between the resolve-check and the
connect is mitigated, not eliminated (single-resolve pinning via a custom
transport is the named follow-up if this ever matters; this is a local
single-user TUI, not a multi-tenant server).

### 4. Extraction bridge — bounded re-rooting (bomb-class guard)

The fetched zip is re-rooted to a single-skill zip and handed to
`import_skill_file`. **The bridge is itself a decompression surface and MUST
be bounded** — the 30 MB cap bounds only the compressed download:

- Candidate discovery is **decompression-free** (central-directory namelist
  only).
- Root normalization: root `SKILL.md` → the archive IS the skill; else a
  single top-level wrapper dir (the `{repo}-{sha}/` zipball shape — also
  covers wrapped release assets; unwrapped assets hit the first rule) →
  descend once and re-test; else scan (depth-limited) for `*/SKILL.md`.
- Subdir install: members under `{root}/{subdir}/` only.
- Exactly ONE candidate → re-root; MANY → clean error listing up to 20
  discovered skill paths ("paste a subdirectory URL"); ZERO → "no SKILL.md
  found".
- **Re-root synthesis reuses `_read_zip_member_bounded`** (the existing
  bounded streaming member reader) and enforces the import caps (5 MB/file,
  25 MB total, 500 files) DURING synthesis — a lying-header bomb aborts here
  exactly as it would in the importer. Junk/symlink members are simply
  copied through or dropped cheaply; the importer remains the authority on
  every import rule (no duplicated validation beyond the caps needed to
  bound the synthesis itself).
- Output: an in-memory zip named `{suggested-name}.zip` →
  `import_skill_file(bytes, filename=…)` → trust-pending, existing review
  flow.

### 5. Seam, policy, UI

- `install_skill_from_url(url, *, scope_service, overwrite=False)` in
  `skill_remote_fetch.py` (network stays OUT of `LocalSkillsService`; the
  function calls INTO the existing import seam via the scope service, which
  the UI already holds).
- **Required registry entry** (fail-closed lesson):
  `_resource("skills.install_remote", actions=(LAUNCH,))` in the
  `server_skills` capability block → `skills.install_remote.launch.local`,
  enforced BEFORE any network I/O via a **new small PUBLIC passthrough on
  `SkillsScopeService`** (e.g. `enforce_install_remote()` wrapping its
  private `_enforce_policy`) — verified: `_enforce_policy` is private and
  never called across the class boundary; the module must not reach past
  the underscore. The existing import path's own
  `skills.import.launch.local` gate still fires inside `import_skill_file`
  — intentional defense-in-depth (double-gated), not a bug.
- UI: the URL branch goes at the TOP of `_run_library_skills_import`,
  BEFORE the `validate_path_simple(..., require_exists=True)` call
  (verified: a URL otherwise dies there as "Could not find that file or
  folder"); the three existing outcome helpers
  (`_apply_library_skills_import_status` / `_success` /
  `_outcome_from_exception`) are reused unchanged; placeholder copy gains
  "or GitHub/zip URL". No new widgets.
- Always lands trust-pending. Never `trust_approved`.

## Components & boundaries

| Unit | File | Responsibility |
|---|---|---|
| Classifier + ref-split (pure) | `Skills_Interop/skill_remote_fetch.py` | URL → GitHubRef/DirectZip/reject; name suggestion; §2 tail split (branch list injected as a callable) |
| Hardened fetcher | same module | §3: https-only, resolve-and-reject, manual hops, auth scoping, streamed cap |
| Extraction bridge | same module | §4: bounded re-rooting → in-memory skill zip |
| Install seam | same module | policy gate → fetch → bridge → `import_skill_file` |
| Token/branches reuse | `Utils/github_api_client.py` | token resolution; `get_branches` for slash-ref disambiguation (ONE additive edit: `per_page=100` on the branches request) |
| Public policy passthrough | `Skills_Interop/skills_scope_service.py` | `enforce_install_remote()` — public wrapper so the module never calls the private `_enforce_policy` |
| Policy entry | `runtime_policy/registry.py` | `skills.install_remote` (REQUIRED — engine fails closed) |
| UI routing | `UI/Screens/library_screen.py` | URL detection in `_run_library_skills_import`; worker + outcome copy |

## Testing strategy

- **Classifier (pure table tests):** all five URL classes; rejects http:,
  git@, non-zip; name suggestions; single- vs multi-segment tails (branch
  list injected as a fake callable — longest-prefix win; fallback heuristic).
- **Fetcher (httpx.MockTransport):** private/loopback/metadata IP rejection
  (direct AND on a redirect hop; a host resolving to a MIXED public+private
  address set is rejected); >3 hops rejected; https→http hop rejected;
  auth present on api.github.com/codeload/objects.githubusercontent hops,
  STRIPPED on a cross-host hop; mid-stream cap abort; timeout mapping to a
  clean error.
- **Branches edit:** `get_branches` requests `per_page=100` (pinned); a
  multi-segment tail with NO prefix match in the returned list falls back
  to the heuristic (never a wrong silent match).
- **Bridge:** zipball-wrapper descent; unwrapped asset; subdir re-root;
  one/many(≤20 listed)/zero candidates; **lying-header bomb inside the
  fetched zip aborts during synthesis** (bounded reader reused); caps
  enforced during synthesis.
- **Seam/policy:** `skills.install_remote.launch.local` registered (engine
  fails closed — pinned); install lands trust-pending; disabled policy denies
  cleanly before network.
- **UI:** URL routes to the fetch worker; error copy surfaces (multi-skill
  listing, over-cap, bad URL); success primes the same Review button.
- **E2E (MockTransport):** a real zipball-shaped archive (wrapper dir +
  `skills/demo/SKILL.md` + references) installs via a `/tree/…` URL,
  trust-pending, then approvable.

## Out of scope (deliberate)

Pack/multi-skill install (next layer, unchanged plan); the agent-callable
install tool (wraps this next); other forges' URL normalization (direct .zip
covers them); `git` protocol/clone; provenance signatures; auto-update
subscriptions; DNS-rebinding single-resolve pinning (named follow-up);
private-repo UI for entering a token (config/env only, as the existing
`[github]` section documents).
