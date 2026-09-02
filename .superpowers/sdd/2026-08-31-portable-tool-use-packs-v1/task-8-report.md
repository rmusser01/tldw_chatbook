# Task 8 report — side-effect-free Tool Pack inspection

## Outcome

Implemented `ToolPackImportService.inspect_archive()` as a side-effect-free,
bounded admission and review boundary. It reads a regular source through a
no-follow descriptor, validates the exact two-member canonical ZIP encoding from
raw local/central/EOCD headers, admits the canonical manifest before the payload,
uses the strict permission-store snapshot and sealed V1 inventory once each, and
returns a deeply immutable review expiring exactly fifteen minutes after capture.

Exact matching uses authority, mapped or original server key, raw tool name, and
portable contract fingerprint. Explicit mappings are external-MCP-only, capped at
256, Unicode-case-fold one-to-one, and collision checked across resulting fallback
and tool identities. Changed or missing Ask/Allow rules are omitted; unresolved
Deny rules remain restrictive pending Denies.

ADR required: no new ADR

ADR path: `backlog/decisions/107-portable-tool-use-packs.md`

Reason: ADR-107 already fixes the archive, strict-read, mapping, review, and later
activation boundaries implemented here.

## Files

- `tldw_chatbook/Tool_Packs/importer.py`
- `Tests/Tool_Packs/test_importer.py`
- `Tests/Tool_Packs/test_import_safety.py`

## TDD evidence

### Missing module RED

Command:

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Tool_Packs/test_importer.py Tests/Tool_Packs/test_import_safety.py -q
```

Output:

```text
ERROR Tests/Tool_Packs/test_importer.py
E   ModuleNotFoundError: No module named 'tldw_chatbook.Tool_Packs.importer'
ERROR Tests/Tool_Packs/test_import_safety.py
E   ModuleNotFoundError: No module named 'tldw_chatbook.Tool_Packs.importer'
!!!!!!!!!!!!!!!!!!! Interrupted: 2 errors during collection !!!!!!!!!!!!!!!!!!!!
2 errors in 0.14s
```

Minimum public-type GREEN:

```text
..                                                                       [100%]
2 passed, 1 warning in 0.58s
```

### Canonical archive review RED/GREEN

Command:

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Tool_Packs/test_importer.py::test_inspects_canonical_export_with_one_strict_snapshot_and_inventory -q
```

RED output before the sealed inventory seam existed in the importer:

```text
E   AttributeError: <module 'tldw_chatbook.Tool_Packs.importer' ...> has no attribute 'capture_v1_inventory'
FAILED Tests/Tool_Packs/test_importer.py::test_inspects_canonical_export_with_one_strict_snapshot_and_inventory
1 failed, 1 warning in 0.55s
```

GREEN output after the minimum archive/store/inventory review path:

```text
.                                                                        [100%]
1 passed, 1 warning in 0.58s
```

### Exact ZIP header admission RED/GREEN

Command:

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Tool_Packs/test_import_safety.py::test_rejects_noncanonical_member_timestamp -q
```

RED output:

```text
E       Failed: DID NOT RAISE <class 'tldw_chatbook.Tool_Packs.contracts.ToolPackError'>
FAILED Tests/Tool_Packs/test_import_safety.py::test_rejects_noncanonical_member_timestamp
1 failed, 1 warning in 0.60s
```

GREEN output after raw/header validation:

```text
.                                                                        [100%]
1 passed, 1 warning in 0.55s
```

The first attempt at this RED had a missing `fixed_now` fixture and errored during
setup. That test-harness error was corrected and rerun to obtain the behavior RED
above before production changed.

### External MCP mapping RED/GREEN

Command:

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Tool_Packs/test_importer.py::test_explicit_external_mapping_matches_exact_contract_and_maps_pending_deny -q
```

RED output:

```text
E   tldw_chatbook.Tool_Packs.contracts.ToolPackError: tool_pack.import.mapping_invalid
FAILED Tests/Tool_Packs/test_importer.py::test_explicit_external_mapping_matches_exact_contract_and_maps_pending_deny
1 failed, 1 warning in 0.58s
```

GREEN output after exact mapping and mapped pending-Deny classification:

```text
.                                                                        [100%]
1 passed, 1 warning in 0.80s
```

### Admission regression expansion

The safety suite was expanded across traversal, absolute and Windows-style names,
backslashes, NUL, dot segments, Windows devices, nested/extra/duplicate members,
linked and nonregular modes, hardlink metadata, directories, encryption, data
descriptors, compression, comments, malformed/duplicate/noncanonical JSON,
digest/size mismatch, archive/member/depth/node limits, no-follow sources, archive
substitution, and corrupt/unknown strict stores. A node-limit fixture initially
called the production canonical encoder, which correctly rejected the oversized
tree before the importer ran; it was replaced with independently constructed raw
JSON, then the importer suite was rerun.

## Final verification

Focused importer and safety command:

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Tool_Packs/test_importer.py Tests/Tool_Packs/test_import_safety.py -q
```

Output:

```text
.....................................................                    [100%]
53 passed, 1 warning in 0.73s
```

Contracts, catalog, importer, and safety matrix:

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Tool_Packs/test_contracts.py Tests/Tool_Packs/test_catalog_snapshot.py Tests/Tool_Packs/test_importer.py Tests/Tool_Packs/test_import_safety.py -q
```

Output:

```text
........................................................................ [ 48%]
........................................................................ [ 96%]
.....                                                                    [100%]
149 passed, 1 warning in 1.24s
```

Scoped formatting and Ruff:

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format tldw_chatbook/Tool_Packs/importer.py Tests/Tool_Packs/test_importer.py Tests/Tool_Packs/test_import_safety.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check tldw_chatbook/Tool_Packs/importer.py Tests/Tool_Packs/test_importer.py Tests/Tool_Packs/test_import_safety.py
```

Output:

```text
3 files left unchanged
All checks passed!
```

Staged diff hygiene:

```text
git diff --cached --check
```

Output: empty, exit status 0.

The single pytest warning is the repository/environment-level
`RequestsDependencyWarning` for the installed urllib3/charset packages; the new
tests emit no feature-specific warning.

## Self-review

- **Parser bounds:** admission checks 5 MiB archive bytes before allocation growth,
  exact two members, per-member caps before JSON decoding, canonical strict JSON,
  and shared depth/node/schema limits from `contracts.py`.
- **TOCTOU and no-follow:** the source is opened with `O_NOFOLLOW`, proven regular by
  `fstat`, read in bounded chunks, and compared before/after against the current
  no-follow path identity. Rename substitution is covered.
- **Byte preservation:** inspection calls only `read_snapshot_strict()`; tests
  monkeypatch legacy `load()` to raise, compare corrupt/unknown bytes exactly, and
  prove no backup appears.
- **Stable errors:** untrusted parser/filesystem exceptions collapse to path-free
  `ToolPackError` codes. Reviews may retain the selected path for later revalidation;
  error strings never do.
- **Mapping collisions:** mappings are external-only, exact, maximum 256, one-to-one
  under Unicode case folding, destination-inventory backed, and checked for duplicate
  resulting fallback/tool identities.
- **Immutability/privacy:** review types use frozen slotted dataclasses and recursively
  immutable contract objects/tuples; no callback, permission payload, inventory,
  connection configuration, endpoint, credential, secret, workspace, or Persona data
  is retained.
- **YAGNI:** implementation is one independent module using stdlib `os`, `struct`, and
  `zipfile`, plus existing canonical contracts and sealed inventory capture. It does
  not extract archives, stage files, mutate authority, or anticipate activation.

No generalizable new repository lesson was surfaced; existing strict-read and
testing-evidence lessons were applied directly.
