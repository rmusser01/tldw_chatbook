# Artwork attribution implementation plan

> **For agentic workers:** Use executing-plans inline. Steps use checkbox syntax.

**Goal:** Preserve reviewed artwork credits and notices through character expression editing and portable Actor Packs, as the first conversion prerequisite.

**Architecture:** Keep public attribution in the known `tldw/artwork` source-context namespace. Carry it in a checksummed optional Shared Visual Identity sidecar, with a required feature flag so older importers reject rather than silently discard it. Native visual manifests and runtime authority remain unchanged.

**Tech Stack:** Python, stdlib JSON/SHA-256, existing SQLite repositories and Actor Pack services.

**Spec:** [Buddy conversion](../specs/2026-09-07-buddy-to-character-design.md), specifically publication and portability.

ADR required: yes
ADR path: backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md
Reason: additive portable data contract and attribution ownership. Amend the existing decision.

## Global constraints and exact contract

- No new dependencies, schema migration, server endpoint, or runtime cross-binding.
- Original collection creator remains `tldw-project`; third-party artwork retains its original creator.
- Optional member `shared-visual-identity/attribution.json`, required feature `visual-artwork-attribution/v1`; either both are present or neither. It requires a Shared Visual Identity section. Legacy exports with no records stay byte-compatible.
- Sidecar is canonical UTF-8 JSON `{ "version": 1, "pack": record-or-null, "assets": {expression_key: record} }`. It is covered by the existing file inventory and content digest. Maximum 512 KiB before JSON decoding; at most 128 asset records, all keys present in the section manifest.
- Record has exactly `version`, `creator`, `license`, `source_url`, `notices` plus `output_sha256` only for asset records. Version must be integer 1. Creator is null or up to 512 UTF-8 bytes; license null or up to 4096; notices is a string up to 64 KiB, unmodified including newlines. Source URL is null or up to 2048 bytes of HTTPS URL, no credentials, query, fragment, IP literal, local hostname, port, or backslash. No URL is fetched.
- Asset records must match the image's SHA-256. Unknown keys/version and malformed known namespaces fail closed. Unrelated local source context is never copied into the carrier. Pack records describe source attribution, not a grant of terms to replacement artwork.
- `tldw/buddy_conversion` remains a separate future conversion record; this slice establishes the notice carrier, not conversion lineage or a server guarantee.
- Retain manifest license from the existing graph on edit. A legacy graph without a license uses `unspecified`, never an invented builtin license. Replaced assets drop their old artwork record even when their bytes happen to match. Retained assets and profile forks preserve valid records.

## Task 1: Contract, archive round trip, and review (TASK-32024)

Files: create `Character_Chat/artwork_attribution.py`, `Tests/Character_Chat/test_artwork_attribution.py`, `Tests/Actor_Packs/test_actor_pack_attribution.py`; modify Actor Pack contracts/export/importer/activation and `Widgets/Persona_Widgets/actor_pack_import_review.py`.

Interfaces: `artwork_context(context, expected_sha256=None) -> dict` returns only the validated namespace; `encode_artwork_attribution(pack_context, assets) -> bytes | None`; `decode_artwork_attribution(data, asset_hashes) -> dict`. Asset inputs map expression keys to `(sha256, context)`.

- [x] Write red tests for records, malformed URLs/limits/hash mismatch, and a real independent archive → review → activation → reopen → export → import cycle. Assert a notice longer than 4096 characters is exact and no local paths/IDs leak.
  ```python
  assert json.loads(exported_sidecar)["assets"]["neutral"] == incoming_record
  assert review.artwork_attribution == incoming_sidecar
  ```
- [x] Implement the typed record validator and bounded canonical sidecar; add paired feature/member validation before staging. Read and validate it against section image hashes, carry immutable bytes on the review and section material, and re-read it under the existing staging lease at activation.
- [x] Restore only the namespace into pack/asset contexts. Show notices as plain text in the existing scrollable review. Keep the sidecar separate from native visual manifests.
- [x] Run the new tests and existing Actor Pack contract/export/import/activation/round-trip tests, including unchanged golden fixtures.

## Task 2: Editing and fork preservation (same atomic deliverable)

Files: modify `Character_Chat/visual_identity.py`, extend `Tests/Character_Chat/test_visual_identity_publication.py`.

- [x] Write failing real SQLite/filesystem tests for retained/replaced assets and profile-copy publication.
  ```python
  assert updated_manifest["license"] == "MIT"
  assert retained_context["tldw/artwork"] == source_record
  assert "tldw/artwork" not in replaced_context
  ```
- [x] Snapshot the source manifest license in `VisualIdentityCandidate`; retain public records beside local publication bookkeeping for copied assets/packs. Validate output digests before publication; use `unspecified` for absent legacy terms.
- [x] Run the whole publication module and attribution round trip with an intervening edit; verify other actors' shared bindings remain unchanged.
- [x] Review the diff and run focused lint/format checks, update evidence and task status, and commit the complete slice.

## Evidence boundaries

This deliverable does not add Create character from Buddy or claim server preservation. It completes the attribution prerequisite and is independently useful for existing characters. Conversion's validated snapshot, timeline encoder, review UI, atomic creation, and server-compatible lineage still require their own implementation.

## Implementation notes

Imported packs also needed a private publication identifier to be editable; activation now assigns it. Same-actor review needed exact JSON hashing independent of the actor payload string limit to accept long notices and detect notice changes. Both repairs were discovered by the real round-trip test and are covered by regressions. See [verification](../reviews/2026-09-07-artwork-attribution-verification.md) for results and reproduced baseline failures.
