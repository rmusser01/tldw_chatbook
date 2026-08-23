# Tests/Architecture/test_vendor_pin_consistency.py
"""Cross-checks the upstream ``tldw_server`` provenance pins (TASK-19574).

Two INDEPENDENT provenance pins live in this repo and currently happen to
share the same commit SHA (``385afa951922c8a9dc2002c675bb6cad65e4ac23``):

  * the Chunking-engine VENDORING pin -- the commit actual vendored *code*
    under ``tldw_chatbook/Chunking/engine`` is copied from. Source of truth:
    the ``PIN`` constant in ``Helper_Scripts/sync_chunking_engine.py`` (the
    script that performs the sync and fails loudly, via
    ``verify_clean()``/``git_show()``, on a mismatch).
  * the Samira visual-identity COMPATIBILITY pin -- the commit the local
    expression-normalization contract (byte-for-byte copied constants) and
    the bundled ``visual_identity_pack.json`` asset are verified against.
    Source of truth: the ``SAMIRA_SERVER_COMMIT`` constant in
    ``tldw_chatbook/Character_Chat/visual_identity.py``.

They pin different upstream subsystems (``app/core/Chunking`` vs.
``app/core/Visual_Identities``) for different reasons and are **not**
required to move together -- bumping one does not imply bumping the other,
and this test does not assert they equal each other. What it DOES assert:
every other hand-maintained copy of EACH pin agrees with that pin's own
designated source of truth, so a hand-edit to one copy that misses a sibling
copy fails loudly here instead of silently drifting (2026-08-21 holistic
review, Lane 7 F4: "the pin is duplicated in six authoritative places with
nothing cross-checking them").

Upgrade path (bumping the Chunking vendoring pin to a new upstream commit):
  1. Update ``PIN`` in ``Helper_Scripts/sync_chunking_engine.py`` (the source
     of truth) to the new 40-char commit SHA.
  2. Update ``commit =`` in ``tldw_chatbook/Chunking/engine/VENDOR_MANIFEST.toml``
     to match -- the sync script does NOT regenerate the manifest file.
  3. Update ``PIN`` in ``Tests/Chunking/test_sync_script.py`` to match.
  4. Run ``python Helper_Scripts/sync_chunking_engine.py --source <local
     tldw_server worktree checked out at the new pin>`` to re-vendor the 38
     engine files + re-port the test suite.
  5. Run ``Tests/Chunking/`` (with ``TLDW_SERVER_SYNC_SOURCE`` set to that
     worktree) and this file to confirm everything lines up.

Upgrade path (bumping the Samira compatibility pin):
  1. Update ``SAMIRA_SERVER_COMMIT`` (and the "Begin pinned server
     normalization block" docstring comment a few lines above it) in
     ``tldw_chatbook/Character_Chat/visual_identity.py`` -- the source of
     truth.
  2. Update the hardcoded literal in
     ``Tests/Character_Chat/test_visual_identity_contract.py``'s
     ``test_samira_inventory_mapping_and_contract_constants_are_exact``.
  3. Regenerate ``tldw_chatbook/assets/characters/samira/visual_identity_pack.json``
     (its ``normalization_contract.source_commit`` and top-level
     ``source_server_commit`` fields, and ``pack_content_sha256``) against the
     new pin.
  4. Run ``Tests/Character_Chat/`` and this file to confirm everything lines
     up.

Out of scope by design: ~19 further copies of this SHA across ``backlog/``
decision records and ``Docs/superpowers/`` planning docs are historical,
point-in-time snapshots of past decisions, not live configuration -- rewriting
them on every future pin bump would misrepresent history. Only the six
authoritative, behavior-affecting copies enumerated above are cross-checked.

These are raw-text/JSON assertions on purpose (matching
``Tests/Chunking/test_descope_ledger.py``'s convention): a couple of these
files are read as source text rather than imported, so drift is caught
without importing modules for their side effects alone.
"""
from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

SYNC_SCRIPT = REPO / "Helper_Scripts" / "sync_chunking_engine.py"
VENDOR_MANIFEST = REPO / "tldw_chatbook" / "Chunking" / "engine" / "VENDOR_MANIFEST.toml"
SYNC_TEST = REPO / "Tests" / "Chunking" / "test_sync_script.py"

VISUAL_IDENTITY = REPO / "tldw_chatbook" / "Character_Chat" / "visual_identity.py"
SAMIRA_PACK = (
    REPO / "tldw_chatbook" / "assets" / "characters" / "samira" / "visual_identity_pack.json"
)
VISUAL_IDENTITY_CONTRACT_TEST = (
    REPO / "Tests" / "Character_Chat" / "test_visual_identity_contract.py"
)

_SHA_RE = re.compile(r"[0-9a-f]{40}")


def _extract_one(text: str, pattern: str, *, where: str) -> str:
    match = re.search(pattern, text, re.MULTILINE)
    assert match, f"pin pattern {pattern!r} not found in {where}"
    shas = _SHA_RE.findall(match.group(0))
    assert len(shas) == 1, (
        f"expected exactly one SHA in the matched text from {where}: {match.group(0)!r}"
    )
    return shas[0]


def _chunking_pin() -> str:
    text = SYNC_SCRIPT.read_text()
    return _extract_one(text, r'^PIN = "[0-9a-f]{40}"', where=str(SYNC_SCRIPT))


def _samira_pin() -> str:
    from tldw_chatbook.Character_Chat.visual_identity import SAMIRA_SERVER_COMMIT

    assert _SHA_RE.fullmatch(SAMIRA_SERVER_COMMIT), (
        f"SAMIRA_SERVER_COMMIT is not a 40-char hex SHA: {SAMIRA_SERVER_COMMIT!r}"
    )
    return SAMIRA_SERVER_COMMIT


# ---------------------------------------------------------------------------
# Chunking-engine vendoring pin: 3 authoritative copies + the source of truth.
# ---------------------------------------------------------------------------

def test_chunking_pin_is_a_valid_sha() -> None:
    """The Chunking-engine vendoring pin itself is a 40-char hex SHA."""
    assert _SHA_RE.fullmatch(_chunking_pin())


def test_vendor_manifest_matches_sync_script_pin() -> None:
    """VENDOR_MANIFEST.toml's upstream.commit agrees with the sync script's
    PIN, the source of truth (see this file's docstring upgrade path)."""
    manifest = tomllib.loads(VENDOR_MANIFEST.read_text())
    assert manifest["upstream"]["commit"] == _chunking_pin(), (
        f"{VENDOR_MANIFEST} upstream.commit is out of sync with the source of "
        f"truth PIN in {SYNC_SCRIPT} -- see this file's docstring for the "
        "upgrade path"
    )


def test_sync_test_pin_matches_sync_script_pin() -> None:
    """test_sync_script.py's hardcoded PIN agrees with the sync script's own
    PIN, the source of truth (see this file's docstring upgrade path)."""
    text = SYNC_TEST.read_text()
    test_pin = _extract_one(text, r'^PIN = "[0-9a-f]{40}"', where=str(SYNC_TEST))
    assert test_pin == _chunking_pin(), (
        f"{SYNC_TEST} PIN is out of sync with the source of truth PIN in "
        f"{SYNC_SCRIPT} -- see this file's docstring for the upgrade path"
    )


# ---------------------------------------------------------------------------
# Samira visual-identity compatibility pin: 3 authoritative copies + the
# source of truth (SAMIRA_SERVER_COMMIT in visual_identity.py itself).
# ---------------------------------------------------------------------------

def test_samira_pin_is_a_valid_sha() -> None:
    """The Samira compatibility pin itself is a 40-char hex SHA."""
    assert _SHA_RE.fullmatch(_samira_pin())


def test_visual_identity_docstring_matches_its_own_constant() -> None:
    """Bonus check beyond the task's six: visual_identity.py carries the pin
    TWICE in one file (the "pinned server normalization block" docstring
    comment, and the SAMIRA_SERVER_COMMIT constant a few lines below it) --
    cheap to keep those two from drifting from each other too."""
    text = VISUAL_IDENTITY.read_text()
    docstring_pin = _extract_one(
        text, r"commit [0-9a-f]{40}\.", where=f"{VISUAL_IDENTITY} docstring comment"
    )
    assert docstring_pin == _samira_pin(), (
        f"{VISUAL_IDENTITY}'s docstring comment SHA is out of sync with its "
        "own SAMIRA_SERVER_COMMIT constant"
    )


def test_samira_pack_matches_visual_identity_pin() -> None:
    """visual_identity_pack.json's two commit fields agree with
    SAMIRA_SERVER_COMMIT, the source of truth (see this file's docstring
    upgrade path)."""
    data = json.loads(SAMIRA_PACK.read_text())
    pin = _samira_pin()
    assert data["normalization_contract"]["source_commit"] == pin, (
        f"{SAMIRA_PACK} normalization_contract.source_commit is out of sync "
        f"with the source of truth SAMIRA_SERVER_COMMIT in {VISUAL_IDENTITY} "
        "-- see this file's docstring for the upgrade path"
    )
    assert data["source_server_commit"] == pin, (
        f"{SAMIRA_PACK} source_server_commit is out of sync with the source "
        f"of truth SAMIRA_SERVER_COMMIT in {VISUAL_IDENTITY} -- see this "
        "file's docstring for the upgrade path"
    )


def test_visual_identity_contract_test_matches_visual_identity_pin() -> None:
    """test_visual_identity_contract.py's hardcoded literal agrees with
    SAMIRA_SERVER_COMMIT, the source of truth (see this file's docstring
    upgrade path)."""
    text = VISUAL_IDENTITY_CONTRACT_TEST.read_text()
    literal_pin = _extract_one(
        text,
        r"assert SAMIRA_SERVER_COMMIT == \"[0-9a-f]{40}\"",
        where=str(VISUAL_IDENTITY_CONTRACT_TEST),
    )
    assert literal_pin == _samira_pin(), (
        f"{VISUAL_IDENTITY_CONTRACT_TEST}'s hardcoded literal is out of sync "
        f"with the source of truth SAMIRA_SERVER_COMMIT in {VISUAL_IDENTITY} "
        "-- see this file's docstring for the upgrade path"
    )
