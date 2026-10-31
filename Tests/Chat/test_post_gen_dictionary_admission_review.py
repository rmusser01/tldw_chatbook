"""Warm replacement dictionaries retain the real parser's admission checks."""

from contextlib import contextmanager
from pathlib import Path

import pytest

from Tests.Backup_Recovery.test_compound_chat_source_lifetimes import (
    fresh_dictionary_library,
)
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
from tldw_chatbook.Chat import Chat_Functions as chat_functions

pytestmark = pytest.mark.bootstrap_profile


@pytest.fixture
def dictionary_file(tmp_path, monkeypatch):
    library = fresh_dictionary_library(monkeypatch)
    monkeypatch.setattr(
        chat_functions,
        "parse_user_dict_markdown_file",
        library.parse_user_dict_markdown_file,
    )
    directory = Path(library._default_dictionary_import_directory())
    path = directory / f"{tmp_path.name}-replacements.md"
    path.write_text("alpha: beta\n", encoding="utf-8")
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def test_warm_post_gen_dictionary_refuses_while_storage_is_paused(dictionary_file):
    assert chat_functions._load_post_gen_dict_entries(str(dictionary_file)) == {
        "alpha": "beta"
    }
    pause = storage._begin_local_pause()
    try:
        with pytest.raises(RecoveryRequired, match="storage_locally_paused"):
            chat_functions._load_post_gen_dict_entries(str(dictionary_file))
    finally:
        pause.resume()


def test_warm_post_gen_dictionary_revalidates_symlink_containment(
    dictionary_file, tmp_path
):
    assert chat_functions._load_post_gen_dict_entries(str(dictionary_file)) == {
        "alpha": "beta"
    }
    # Moving the same file preserves the old size/mtime signature. A warm
    # cache must still reject the new target outside the active config root.
    moved = tmp_path / "outside-config.md"
    dictionary_file.rename(moved)
    dictionary_file.symlink_to(moved)

    assert chat_functions._load_post_gen_dict_entries(str(dictionary_file)) == {}


def test_warm_post_gen_dictionary_rejects_a_replaced_parser_source(
    dictionary_file, monkeypatch
):
    assert chat_functions._load_post_gen_dict_entries(str(dictionary_file)) == {
        "alpha": "beta"
    }
    # The imported Chat function still references the departed installed parser.
    fresh_dictionary_library(monkeypatch)
    with pytest.raises(RecoveryRequired, match="dictionary_file_source_invalid"):
        chat_functions._load_post_gen_dict_entries(str(dictionary_file))


def test_parsed_dictionary_cache_retains_only_the_current_file(
    dictionary_file, monkeypatch
):
    from tldw_chatbook.Character_Chat import Chat_Dictionary_Lib as library

    reads = []
    original = library._dictionary_files.opened

    @contextmanager
    def observed_open(path, *args, **kwargs):
        reads.append(Path(path))
        with original(path, *args, **kwargs) as source:
            yield source

    monkeypatch.setattr(library._dictionary_files, "opened", observed_open)
    other = dictionary_file.with_name(dictionary_file.stem + "-other.md")
    other.write_text("gamma: delta\n", encoding="utf-8")
    try:
        first = chat_functions._load_post_gen_dict_entries(str(dictionary_file))
        first["alpha"] = "caller mutation"
        assert chat_functions._load_post_gen_dict_entries(str(dictionary_file)) == {
            "alpha": "beta"
        }
        assert chat_functions._load_post_gen_dict_entries(str(other)) == {
            "gamma": "delta"
        }
        assert chat_functions._load_post_gen_dict_entries(str(dictionary_file)) == {
            "alpha": "beta"
        }
        assert reads == [dictionary_file, other, dictionary_file]
    finally:
        other.unlink(missing_ok=True)
