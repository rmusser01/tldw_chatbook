# Petdex import implementation plan

> **For agentic workers:** Use subagent-driven-development with bounded ownership and integrated review. Execute continuously.

**Goal:** Import a downloaded or public Petdex pet into a reviewed local Buddy draft.
**Architecture:** Immutable bounded sources feed atlas mapping and a native archive; existing Persona Visual import/save owns publication. A dedicated standard-library transport pins validated IPs without altering general egress.
**Tech Stack:** Python, Pillow, stdlib HTTP/TLS/ZIP, Textual, existing SQLite/native pack services.
**Spec:** [Petdex design](../specs/2026-09-07-petdex-buddy-import-design.md).

ADR required: yes
ADR path: backlog/decisions/134-reviewed-petdex-import-and-pinned-https.md
Reason: external network trust boundary, native mapping and carried source terms.

## Global constraints and interfaces

- All work stays in the existing isolated branch. No new dependencies; no executing source commands. No application defaults or server mutation.
- Metadata 2 MiB, registry 10 MiB decoded, image 25 MiB, edge4096; native image/frame/total limits apply. Local ZIP max128 members, 32MiB total, no links/traversal/duplicates; only declared sprite and bounded notices are consumed. Reject ambiguity. Folder reads pin non-symlink files and revalidate bytes.
- Package shape: pet.json plus its declared spritesheet (or exactly one spritesheet.png/webp); bounded LICENSE/NOTICE/COPYING files retained. Metadata version must agree with exact integer-cell geometry. Optional explicit states is an array of {name,row,frames,duration_ms,loop}; manual review uses the same shape. v2 without declarations starts with no guessed state map.
- Petdex/sources.py: frozen PetdexSource(title, description, metadata_json, image_bytes, image_name, source_sha256, artwork, _guard) with is_current(); source_from_bytes(metadata:bytes,image:bytes,image_name:str,*,registry_entry=None,notices="",guard=lambda:True); read_local_package(path).
- Petdex/network.py: fetch_bytes(url, *, max_bytes, cancel_requested=lambda:False) -> bytes validates allowlisted HTTPS, DNS, connection peer, SNI, redirects and streaming caps; no proxy trust. Endpoint errors expose safe categories, not credential-bearing URLs.
- Petdex/registry.py: fetch_petdex_source(value, *, cancel_requested=lambda:False) -> PetdexSource; parse exact public object or compact manifest, unique slug, allowed asset URLs, then bounded pet.json/image fetch into source_from_bytes. No authenticated endpoint required; fallback to legacy only for unsupported/absent compact endpoint.
- Petdex/conversion.py: PetdexState(name,row,frames,duration_ms,loop=True); PetdexInspection(version,rows,cell_width,cell_height,states,mapping_source,warnings); inspect_petdex(source) returns pinned classic states, declared states or empty manual-required states. build_petdex_archive(source, *, states=None, mappings=None) -> bytes revalidates, emits one native sprite_sheet with regions and exact per-frame timings. Native mappings dict has required idle/thinking/error/listening/speaking values referencing source names; default absent states fall back explicitly to idle. Preserve every source state as custom catalog.
- Attribution: dedicated canonical string artwork validated through existing tldw/artwork contract; mapping_source token and source_id digest. Generic source-context fields remain restricted. Add native Buddy export preserving this context and adapt existing authoring/import/snapshot projection; no silent notice loss.
- Petdex/review.py and a focused dialog: URL/slug or local package/folder source, source credits, editable state JSON when needed and required-state selection, preview each state, accept into existing unsaved native draft. Never auto-save or change Buddy/Console preferences. Capture destination/source across every await. URL and package acquisition run in workers; cancellation drains owned work and removes only owned staging.

## Task 1: Pinned network and public registry (agent)
Files: Petdex/network.py, Petdex/registry.py, dedicated tests.
- [x] Write failing mocked DNS/socket/TLS tests for private/mixed DNS, pinned numeric connect, correct SNI, peer mismatch, redirects, body limits and cancellation.
- [x] Implement bounded fetch using stdlib sockets/TLS/http.client and egress IP classification, ignoring proxies.
- [x] Write object/compact selection tests including duplicates, unknown versions and malicious URLs; implement registry adapter using the source interface above.
- [x] Run targeted tests and report whether actual upstream fetch is possible separately.

## Task 2: Portable native artwork (agent)
Files: Persona_Visual authoring/repository/importer/snapshot plus focused native exporter and tests; coordinate Actor Pack additions if required.
- [x] Prove current notices loss with import/save/reopen/export fixture.
- [x] Add narrow canonical artwork source-context validation and retain it through the native draft pipeline.
- [x] Make native export/reimport and saved Buddy conversion preserve exact creator/source/terms; verify file identity and native checksums.
- [x] Keep older context keys and native archives compatible; test invalid carrier and stale bytes.

## Task 3: Sources, atlas mapping and native generation (parent)
Files: Petdex/__init__.py, sources.py, conversion.py, tests.
- [x] Test folder/ZIP pinning, symlinks, duplicate/path/size/JSON ambiguity and notice retention; implement source boundary.
- [x] Test classic/scaled geometry and colored row/cell sentinels. v2 declarations/manual maps must specify exact frame counts and timing; unknown/conflicting layouts reject.
- [x] Build a real native archive using sprite regions, exact loop durations and explicit required-state fallbacks. Consume the real native validator, never bypass it.
- [x] Exercise native import/save/offline snapshot/character conversion and injected stale/failure cases.

## Task 4: User-facing review (agent)
Files: focused Petdex dialog/controller, Persona Visual widget/screen hooks, tests/CSS.
- [x] Add Petdex… source action for eligible local Personas; retain draft guard.
- [x] Let user fetch URL/slug or choose downloaded ZIP/pet.json folder; show terms and manual v2 mapping when required.
- [x] Prepare native preview and edit required mappings before accepting as an unpublished draft through existing import service.
- [x] Test mounted entry, actual imported draft, cancellation, stale authority, mapping errors, credits and explicit Save. Provide native export control if no existing user path exposes it.

## Integration
- [x] Review source, transport and UI contracts independently; fix findings.
- [x] Run focused regressions, actual permitted public-source probe, native export/reimport and character conversion.
- [x] Record evidence and remaining external limitations, update user guide/programme/task and commit.

## Ledger
Ruling: Existing approved Petdex design authorizes implementation; no repeated design confirmation. Standard-library HTTPS gives an inspectable connection-pinning seam with no new dependency.
Ruling: Local states declarations and manual mappings use one explicit schema; unknown upstream variants require manual review rather than inferred v2 rows.


Result: Implemented and reviewed. The combined targeted run passed 782 tests; the
final Petdex run passed 103 after four extra source-admission cases. Native export
and independent animated character publication also passed against a real downloaded
pet in a disposable profile. See ../reviews/2026-09-07-petdex-import-verification.md.

Review corrections: preserve differing registry/source credit statements and nested
notices; reject ZIP orig_filename truncation; keep CDN URL separate from the logical
sprite filename; release modal resources after early failures; use atomic native
export and an explicit static preview fallback when WebP encoding is unavailable.
