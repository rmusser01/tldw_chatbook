# Tests/Chunking/test_descope_ledger.py
"""Pins for the permanent descope rulings (2026-08-23 propositions-vendoring
spec §4): the manifest's ``excluded`` comments are the ledger, the synced
skip-table reasons carry the same rulings into the ported test files, and no
"deferred to" + "#6" residue survives anywhere in the synced surfaces.

These are raw-text assertions on purpose: TOML comments are free-form and
tomllib drops them, so the ledger is pinned by parsing the file's text."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ENGINE = REPO / "tldw_chatbook" / "Chunking" / "engine"
MANIFEST = ENGINE / "VENDOR_MANIFEST.toml"
SYNC = REPO / "Helper_Scripts" / "sync_chunking_engine.py"

# The spec §4 ruling each not-vendored entry must carry in its preceding
# comment block. Markers are lowercase; matching is case-insensitive.
RULINGS = {
    "async_chunker.py": [
        "not vendored",          # the ruling itself
        "http_client",           # the blocking dependency (RetryPolicy/afetch)
        "in-process",            # chatbook chunks in-process — no consumer
        "no chatbook consumer",
        "revisit only if a consumer appears",  # the recorded revisit condition
    ],
    "auto_boundary_assistant.py": [
        "not vendored",
        "server-stack shims",    # AuthNZ ×2, Chat ×2, LLM_Calls registry, schemas
        "no chatbook consumer",
        "#3",                    # capability covered by auto-selection
        "#4",                    # ...and the agent surface
        "revisit only if a consumer appears",
    ],
}


def _ledger_block(manifest_text: str, entry: str) -> str:
    """Return the comment block immediately preceding the entry's list line,
    flattened to a single space-separated string (comments wrap freely, so
    a marker may span lines)."""
    lines = manifest_text.splitlines()
    for i, line in enumerate(lines):
        if f'"{entry}"' in line:
            block = []
            for prev in reversed(lines[max(0, i - 14):i]):
                if prev.strip().startswith("#"):
                    block.append(prev.strip().lstrip("#").strip())
                else:
                    break
            return " ".join(reversed(block))
    raise AssertionError(f"{entry} not found in the manifest excluded list")


def test_manifest_excluded_carries_not_vendored_rulings():
    text = MANIFEST.read_text()
    for entry, markers in RULINGS.items():
        block = _ledger_block(text, entry).lower()
        assert block, f"no comment block above {entry} in the manifest"
        missing = [m for m in markers if m.lower() not in block]
        assert not missing, \
            f"{entry} ruling block is missing {missing}; block was:\n{block}"


def test_descoped_files_absent_from_engine_tree():
    # The rulings are enforced by absence: these never land in the tree.
    for rel in ("async_chunker.py", "auto_boundary_assistant.py"):
        assert not (ENGINE / rel).exists(), f"descoped file vendored: {rel}"


def test_telemetry_noop_reaffirmed_in_skip_table():
    # Spec §4.3 rides the sync script's TESTS_MODULE_SKIPPED reason for the
    # metrics suite (the third recording place) and regenerates into the
    # ported file on sync.
    ported = (REPO / "Tests" / "Chunking" / "test_chunker_process_metrics.py")
    reason = ported.read_text().lower()
    assert "no-op" in reason
    assert "reaffirm" in reason


def test_no_deferred_to_6_residue():
    """Spec §2 goal 2 / plan Task 1: the program closed — the deferral phrase
    (see `needle` below) may not survive in the synced surfaces (specs/plans
    are exempt as history; .superpowers planning dirs are not synced surfaces
    either). The needle is assembled at runtime so this file cannot match
    itself."""
    needle = "deferred to " + "#6"
    offenders = []
    for root in ("Helper_Scripts", "tldw_chatbook", "Tests"):
        for py in (REPO / root).rglob("*.py"):
            if needle in py.read_text().lower():
                offenders.append(str(py.relative_to(REPO)))
    assert not offenders, f"'{needle}' residue: {offenders}"


def test_regenerated_tests_carry_terminal_dispositions():
    # The un-skipped propositions suite: no module-skip block at all.
    props = (REPO / "Tests" / "Chunking" / "test_propositions_strategy.py")
    src = props.read_text()
    assert "NoSuchDeferredModule" not in src
    assert "importorskip" not in src
    # The templates suite stays module-skipped, with the terminal wording
    # (its FastAPI/fixture blockers stand; the descope is cited).
    templates = (REPO / "Tests" / "Chunking" / "test_upstream_chunking_templates.py")
    tsrc = templates.read_text().lower()
    assert "importorskip" in tsrc  # still module-skipped
    assert "terminal" in tsrc
    assert "server-side" in tsrc
    # And the formerly propositions-blocked hierarchical suite revived with
    # the module: no skip block there either.
    hier = (REPO / "Tests" / "Chunking" / "test_hierarchical_rewrite_offsets.py")
    assert "importorskip" not in hier.read_text()
