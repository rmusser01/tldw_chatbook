from tldw_chatbook.Sync_Interop.hashing import canonical_payload_hash, HASH_VERSION


def test_hash_is_deterministic_and_key_order_independent():
    a = canonical_payload_hash({"title": "T", "content": "B"})
    b = canonical_payload_hash({"content": "B", "title": "T"})
    assert a == b
    assert a.startswith("sha256:")


def test_hash_changes_with_content():
    assert canonical_payload_hash({"title": "T"}) != canonical_payload_hash({"title": "U"})


def test_hash_version_constant_exists():
    assert isinstance(HASH_VERSION, int) and HASH_VERSION >= 1
