---
id: DOC-1
title: Backup recovery archive inspection
---

# Backup recovery archive inspection

TASK-31995 implements original plan Task 12 under ADR-126. This component inspects recovery containers; it does not capture live profiles, extract files, migrate databases, or enable replacement.

## API and ownership

`Backup_Recovery.archive_reader.acquire(source, work_root, limits, password, cancel)` returns `SealedArchive(path, digest, manifest_bytes)`. It copies into a newly allocated private operation directory, checks source stability and actual byte quotas, authenticates encrypted input with the bundled age helper, then validates the ZIP and manifest. The original source is no longer used. Failed operations remove their own staging; successful operations retain it for the caller. `verify_sealed` checks the digest and canonical embedded manifest immediately before later use. Private permissions are not a claim of filesystem-enforced immutability.

`ArchiveLimits` contains the original independent input, decrypted, expanded, member, count, manifest and path budgets. Space is checked before and during staging. `CompressionReviewRequired` exposes the copied plaintext digest and declared expanded size. After explicit local review, retry with those values in `reviewed_digest` and `reviewed_expanded_bytes`; acquisition repeats copying and space admission, and unchanged hard budgets still apply.

## Version-one writer contract

The container consists of `manifest.json` and exactly the payload names referenced by its files. Use a seekable ZIP64 writer, stored or deflated regular files, UTF-8 names, and no comments, ZIP directories, data descriptors or unrelated extra fields. Folder topology, including empty directories, lives in the manifest. Local and central structure, paths, duplicates, parent relationships, dependency groups, sizes and SHA-256 digests are validated.

The strict frozen `ArchiveManifest` model in `tldw_chatbook/Backup_Recovery/archive_models.py` defines the schema. Serialize its JSON-mode representation canonically with sorted keys and compact separators. Owner/schema metadata remains inert: installed schema qualification and migration belong to original Task 13. Partial coverage cannot authorize replacement. Managed relocation records and included/rollback credentials require encryption.

## Qualification boundary

Real age round trips use the pinned helper and its existing package qualification checks. There is no PATH helper or plaintext fallback. Unsupported native storage/helper capabilities remain unavailable. Destination, migration and rollback volume admission belong to restore execution, not archive inspection. Focused regression evidence is recorded in TASK-31995; the complete backup/restore product remains subject to the remaining original plan tasks.
