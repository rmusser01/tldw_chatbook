"""The vendored tiktoken cache is complete, and tokenizing needs no network.

`Utils/token_counter` and the chunking engine obtain encodings from tiktoken,
which downloads its tables on first use and caches them outside anything the
test sandbox controls. That cache is warm on any machine that has run the suite
before and cold on every CI run, so the download only fails where nobody looks:
one core shard recorded 1,156 blocked attempts. The caller wraps the lookup in a
broad `except` and returns None, so the refusal is swallowed -- the only reason
it surfaces at all is the egress guard recording the attempt, and the test then
fails at teardown pointing at a network address rather than at tokenizing.

`Tests/fixtures/tiktoken_cache/` holds the tables and `Tests/conftest.py` points
`TIKTOKEN_CACHE_DIR` at it. These tests pin that arrangement, because its two
failure modes are both quiet: a cache that is present but incomplete falls back
to downloading, and a filename that no longer matches tiktoken's key does the
same.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

CACHE_DIR = Path(__file__).resolve().parent / "fixtures" / "tiktoken_cache"
BASE = "https://openaipublic.blob.core.windows.net"

#: Every table the suite is known to request, and the URL tiktoken fetches it
#: from. The filename is `sha1(url)` -- tiktoken's own key, recomputed here so a
#: rename cannot pass unnoticed.
VENDORED = {
    "gpt2 vocab": f"{BASE}/gpt-2/encodings/main/vocab.bpe",
    "gpt2 encoder": f"{BASE}/gpt-2/encodings/main/encoder.json",
    "cl100k_base": f"{BASE}/encodings/cl100k_base.tiktoken",
}


def test_the_conftest_points_at_the_vendored_cache() -> None:
    """The mechanism. Without this the tables are present but unused."""
    assert os.environ.get("TIKTOKEN_CACHE_DIR") == str(CACHE_DIR)


@pytest.mark.parametrize(("label", "url"), sorted(VENDORED.items()))
def test_each_vendored_table_is_present_under_tiktokens_own_key(
    label: str, url: str
) -> None:
    """A renamed file is indistinguishable from a missing one, to tiktoken.

    It would simply download again -- which is the failure this directory
    exists to prevent, and which is invisible on a machine with a warm cache
    or network access.
    """
    expected = CACHE_DIR / hashlib.sha1(url.encode()).hexdigest()
    assert expected.is_file(), (
        f"{label} is missing from the vendored cache. tiktoken keys entries by "
        f"sha1 of the download URL, so it must be named {expected.name}. See "
        f"{CACHE_DIR.name}/README.md for how to refresh."
    )
    assert expected.stat().st_size > 100_000, f"{label} looks truncated"


def test_encodings_load_with_the_network_guard_active() -> None:
    """The property, not the mechanism: this runs under the egress guard.

    If either encoding were fetched rather than read from disk, the guard would
    record the attempt and fail this test at teardown. Passing is therefore
    evidence that tokenizing needs no network, which asserting on file presence
    alone would not give.
    """
    tiktoken = pytest.importorskip("tiktoken")

    assert tiktoken.get_encoding("gpt2").encode("hello world")
    assert tiktoken.get_encoding("cl100k_base").encode("hello world")
