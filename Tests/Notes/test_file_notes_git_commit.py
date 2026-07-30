from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass
from time import perf_counter

import pytest

from tldw_chatbook.Notes.file_notes_git_commit import (
    CommitContractError,
    CommitIncludedNote,
    CommitOutcome,
    CommitRecoveryProjection,
    CommitReviewHandle,
    CommitReviewProjection,
    CommitReviewResult,
    GitIdentity,
    RawCommitObject,
    RawStagedDeltaEntry,
    format_git_identity_display,
    normalize_commit_message,
    parse_git_identity,
    parse_raw_commit_object,
    parse_raw_staged_delta,
)


_ZERO_OID = "0" * 40
_OLD_OID = "1" * 40
_NEW_OID = "2" * 40
_TREE_OID = "a" * 40
_PARENT_OID = "b" * 40


@pytest.mark.parametrize(
    "model_type",
    [
        GitIdentity,
        CommitIncludedNote,
        CommitReviewProjection,
        CommitReviewHandle,
        CommitReviewResult,
        CommitOutcome,
        CommitRecoveryProjection,
    ],
)
def test_commit_message_public_contract_models_are_frozen(
    model_type: type[object],
) -> None:
    assert is_dataclass(model_type)
    assert model_type.__dataclass_params__.frozen is True  # type: ignore[attr-defined]


def _raw_delta_record(
    *,
    old_mode: str = "100644",
    new_mode: str = "100644",
    old_oid: str = _OLD_OID,
    new_oid: str = _NEW_OID,
    status: str = "M",
    path: bytes = b"note.md",
) -> bytes:
    header = f":{old_mode} {new_mode} {old_oid} {new_oid} {status}".encode()
    return header + b"\0" + path + b"\0"


def _raw_commit(
    *,
    tree: str = _TREE_OID,
    parents: tuple[str, ...] = (_PARENT_OID,),
    author: bytes = b"Ada Lovelace <ada@example.test> 1700000000 +0000",
    committer: bytes = b"Grace Hopper <grace@example.test> 1700000001 -0700",
    extra_headers: tuple[bytes, ...] = (),
    message: bytes = b"Subject\n\nBody\n",
) -> bytes:
    headers = [f"tree {tree}".encode()]
    headers.extend(f"parent {parent}".encode() for parent in parents)
    headers.extend((b"author " + author, b"committer " + committer))
    headers.extend(extra_headers)
    return b"\n".join(headers) + b"\n\n" + message


def test_commit_message_trims_subject_and_emits_exact_subject_bytes() -> None:
    assert normalize_commit_message(" \t  Subject \t ", "") == b"Subject\n"


@pytest.mark.parametrize("subject", ["", " ", "\t \t"])
def test_commit_message_requires_a_nonempty_subject(subject: str) -> None:
    with pytest.raises(CommitContractError) as error:
        normalize_commit_message(subject, "")

    assert error.value.code == "subject_required"


def test_commit_message_accepts_a_512_character_subject() -> None:
    subject = "s" * 512

    assert normalize_commit_message(subject, "") == subject.encode() + b"\n"


def test_commit_message_rejects_a_subject_longer_than_512_characters() -> None:
    with pytest.raises(CommitContractError) as error:
        normalize_commit_message("s" * 513, "")

    assert error.value.code == "subject_too_long"


def test_commit_message_accepts_512_multibyte_unicode_characters() -> None:
    subject = "🙂" * 512

    assert len(subject) == 512
    assert normalize_commit_message(subject, "") == subject.encode() + b"\n"


def test_commit_message_rejects_513_multibyte_unicode_characters() -> None:
    with pytest.raises(CommitContractError) as error:
        normalize_commit_message("🙂" * 513, "")

    assert error.value.code == "subject_too_long"


@pytest.mark.parametrize("separator", ["\n", "\r", "\r\n"])
def test_commit_message_requires_a_single_line_subject(separator: str) -> None:
    with pytest.raises(CommitContractError) as error:
        normalize_commit_message(f"first{separator}second", "")

    assert error.value.code == "subject_multiline"


def test_commit_message_normalizes_crlf_and_cr_in_the_body() -> None:
    result = normalize_commit_message(
        "Subject",
        "first\r\nsecond\rthird\nfourth",
    )

    assert result == b"Subject\n\nfirst\nsecond\nthird\nfourth\n"


def test_commit_message_removes_only_surrounding_blank_body_lines() -> None:
    result = normalize_commit_message(
        "Subject",
        " \r\n\t\r\n  first  \r\n \t\r\nlast\t\r\n \r\n\t",
    )

    assert result == b"Subject\n\n  first  \n \t\nlast\t\n"


def test_commit_message_trims_a_large_blank_prefix_in_linear_time() -> None:
    body = (" \n" * 250_000) + "Body"

    started = perf_counter()
    result = normalize_commit_message("Subject", body)
    elapsed = perf_counter() - started

    assert result == b"Subject\n\nBody\n"
    assert elapsed < 2.0


def test_commit_message_preserves_internal_body_whitespace_and_newlines() -> None:
    body = "first  \t\n\n \t\n  last  "

    assert normalize_commit_message("Subject", body) == (
        b"Subject\n\nfirst  \t\n\n \t\n  last  \n"
    )


def test_commit_message_emits_exact_subject_or_subject_body_shapes() -> None:
    assert normalize_commit_message("Subject", " \n\t\n") == b"Subject\n"
    assert normalize_commit_message("Subject", "Body") == (b"Subject\n\nBody\n")


def test_commit_message_accepts_exactly_64_kib_of_utf8() -> None:
    body = "é" * 32_766
    result = normalize_commit_message("s", body)

    assert len(result) == 64 * 1024
    assert result == b"s\n\n" + body.encode("utf-8") + b"\n"


def test_commit_message_rejects_more_than_64_kib_of_utf8() -> None:
    with pytest.raises(CommitContractError) as error:
        normalize_commit_message("s", "é" * 32_767)

    assert error.value.code == "message_too_large"


def test_commit_message_accepts_emoji_and_ordinary_rtl_text() -> None:
    subject = "✨ שלום"
    body = "مرحبا بالعالم 👋"

    assert normalize_commit_message(subject, body) == (
        f"{subject}\n\n{body}\n".encode()
    )


@pytest.mark.parametrize(
    ("unsafe", "location"),
    [
        ("\0", "subject"),
        ("\x07", "body"),
        ("\x1b", "subject"),
        ("\x7f", "body"),
        ("\x85", "subject"),
        ("\x9b", "body"),
        ("\ud800", "subject"),
        ("\udfff", "body"),
        ("\u202d", "subject"),
        ("\u202e", "body"),
        ("\u2066", "subject"),
        ("\u2069", "body"),
    ],
)
def test_commit_message_rejects_unsafe_or_unpreviewable_text(
    unsafe: str,
    location: str,
) -> None:
    subject = f"safe{unsafe}secret-subject" if location == "subject" else "safe"
    body = f"safe{unsafe}secret-body" if location == "body" else ""

    with pytest.raises(CommitContractError) as error:
        normalize_commit_message(subject, body)

    assert error.value.code == "unsafe_text"
    assert "secret" not in str(error.value)
    assert len(str(error.value).encode()) <= 80


def test_git_identity_parses_the_effective_ident_from_the_right() -> None:
    raw = b"Dr 1700000000 +0000 Ada Lovelace <ada@example.test> 1700000001 -0730\n"

    assert parse_git_identity(raw) == GitIdentity(
        name="Dr 1700000000 +0000 Ada Lovelace",
        email="ada@example.test",
    )


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"<ada@example.test> 1700000000 +0000\n",
        b"  <ada@example.test> 1700000000 +0000\n",
        b"Ada <> 1700000000 +0000\n",
        b"Ada <   > 1700000000 +0000\n",
        b"Ada <ada@example.test> +0000\n",
        b"Ada <ada@example.test> 1700000000\n",
        b"Ada <ada@example.test> when +0000\n",
        b"Ada <ada@example.test> 1700000000 UTC\n",
        b"Ada <ada@example.test> 1700000000 +2460\n",
    ],
)
def test_git_identity_rejects_missing_or_malformed_fields(raw: bytes) -> None:
    with pytest.raises(CommitContractError) as error:
        parse_git_identity(raw)

    assert error.value.code == "invalid_identity"
    assert str(error.value) == "Git identity is missing or invalid."


def test_git_identity_display_collapses_equal_author_and_committer() -> None:
    author = parse_git_identity(b"[ops] Ada <ada@example.test> 1700000000 +0000\n")
    committer = parse_git_identity(b"[ops] Ada <ada@example.test> 1800000000 -0700\n")

    assert format_git_identity_display(author, committer) == (
        ("Identity", "[ops] Ada <ada@example.test>"),
    )


def test_git_identity_display_separates_different_author_and_committer() -> None:
    author = GitIdentity("Ada", "ada@example.test")
    committer = GitIdentity("Grace", "grace@example.test")

    assert format_git_identity_display(author, committer) == (
        ("Author", "Ada <ada@example.test>"),
        ("Committer", "Grace <grace@example.test>"),
    )


@pytest.mark.parametrize(
    "raw",
    [
        b"Evil\0Name <evil@example.test> 1700000000 +0000\n",
        b"Evil\x1b[2J <evil@example.test> 1700000000 +0000\n",
        b"Evil <evil\xc2\x9b@example.test> 1700000000 +0000\n",
        "Evil\u202eName <evil@example.test> 1700000000 +0000\n".encode(),
        "Evil <evil\u2066@example.test> 1700000000 +0000\n".encode(),
        b"\xff <evil@example.test> 1700000000 +0000\n",
    ],
)
def test_git_identity_rejects_hostile_terminal_text(raw: bytes) -> None:
    with pytest.raises(CommitContractError) as error:
        parse_git_identity(raw)

    assert error.value.code == "invalid_identity"
    assert len(str(error.value).encode()) <= 80


def test_git_identity_diagnostics_suppress_hostile_decode_details() -> None:
    with pytest.raises(CommitContractError) as error:
        parse_git_identity(b"\xff <evil@example.test> 1700000000 +0000\n")

    assert error.value.__cause__ is None
    assert error.value.__suppress_context__ is True


def test_git_identity_accepts_markup_looking_printable_text_literally() -> None:
    identity = parse_git_identity(
        "[bold] שלום [/bold] <[link]@example.test> 1700000000 +0000\n".encode()
    )

    assert identity == GitIdentity(
        "[bold] שלום [/bold]",
        "[link]@example.test",
    )
    assert identity.display == "[bold] שלום [/bold] <[link]@example.test>"


def test_git_identity_is_frozen() -> None:
    identity = GitIdentity("Ada", "ada@example.test")

    with pytest.raises(FrozenInstanceError):
        identity.name = "Grace"  # type: ignore[misc]


def test_raw_staged_delta_parses_additions_deletions_and_mode_changes() -> None:
    payload = b"".join(
        (
            _raw_delta_record(
                old_mode="000000",
                new_mode="100644",
                old_oid=_ZERO_OID,
                status="A",
                path=b"added.md",
            ),
            _raw_delta_record(
                new_mode="000000",
                new_oid=_ZERO_OID,
                status="D",
                path=b"deleted.md",
            ),
            _raw_delta_record(
                old_mode="100644",
                new_mode="100755",
                old_oid=_NEW_OID,
                new_oid=_NEW_OID,
                status="M",
                path=b"mode.md",
            ),
        )
    )

    assert parse_raw_staged_delta(payload) == (
        RawStagedDeltaEntry(
            old_mode="000000",
            new_mode="100644",
            old_object_id=_ZERO_OID,
            new_object_id=_NEW_OID,
            status="A",
            path=b"added.md",
        ),
        RawStagedDeltaEntry(
            old_mode="100644",
            new_mode="000000",
            old_object_id=_OLD_OID,
            new_object_id=_ZERO_OID,
            status="D",
            path=b"deleted.md",
        ),
        RawStagedDeltaEntry(
            old_mode="100644",
            new_mode="100755",
            old_object_id=_NEW_OID,
            new_object_id=_NEW_OID,
            status="M",
            path=b"mode.md",
        ),
    )


def test_raw_staged_delta_preserves_filename_bytes_for_proof_comparison() -> None:
    path = b"-hostile-\xff\n[bold].md"

    (entry,) = parse_raw_staged_delta(_raw_delta_record(path=path))

    assert entry.path == path


def test_raw_staged_delta_accepts_an_empty_delta() -> None:
    assert parse_raw_staged_delta(b"") == ()


@pytest.mark.parametrize(
    "payload",
    [
        _raw_delta_record()[:-1],
        _raw_delta_record().replace(b":100644", b"100644", 1),
        b":100644 100644\0note.md\0",
        _raw_delta_record(old_mode="10064x"),
        _raw_delta_record(old_oid="1" * 39),
        _raw_delta_record(new_oid=("2" * 39) + "z"),
        _raw_delta_record(status="R100"),
        _raw_delta_record(path=b""),
        _raw_delta_record() + b"\0",
    ],
)
def test_raw_staged_delta_rejects_malformed_or_truncated_records(
    payload: bytes,
) -> None:
    with pytest.raises(CommitContractError) as error:
        parse_raw_staged_delta(payload)

    assert error.value.code == "malformed_staged_delta"
    assert str(error.value) == "Staged Git data is malformed."


def test_raw_staged_delta_diagnostics_never_disclose_filename_bytes() -> None:
    payload = _raw_delta_record(path=b"secret-unrelated-note.md")[:-1]

    with pytest.raises(CommitContractError) as error:
        parse_raw_staged_delta(payload)

    assert "secret" not in str(error.value)
    assert len(str(error.value).encode()) <= 80


def test_raw_commit_object_parses_exact_proof_fields_and_message_bytes() -> None:
    payload = _raw_commit(message=b"Subject\n\nBody \xff bytes\n")

    assert parse_raw_commit_object(payload) == RawCommitObject(
        tree_object_id=_TREE_OID,
        parent_object_id=_PARENT_OID,
        author=GitIdentity("Ada Lovelace", "ada@example.test"),
        committer=GitIdentity("Grace Hopper", "grace@example.test"),
        message=b"Subject\n\nBody \xff bytes\n",
        signature_headers=(),
    )


def test_raw_commit_object_accepts_multiline_headers() -> None:
    payload = _raw_commit(
        extra_headers=(
            b"mergetag object " + (b"c" * 40),
            b" type commit",
            b" tag release",
        )
    )

    parsed = parse_raw_commit_object(payload)

    assert parsed.message == b"Subject\n\nBody\n"
    assert parsed.signature_headers == ()


@pytest.mark.parametrize("signature_name", [b"gpgsig", b"gpgsig-sha256"])
def test_raw_commit_object_detects_multiline_signature_headers(
    signature_name: bytes,
) -> None:
    payload = _raw_commit(
        extra_headers=(
            signature_name + b" -----BEGIN SIGNATURE-----",
            b" abcdef",
            b" -----END SIGNATURE-----",
        )
    )

    parsed = parse_raw_commit_object(payload)

    assert parsed.signature_headers == (signature_name.decode(),)
    assert parsed.has_signature is True


def test_raw_commit_object_treats_header_looking_message_bytes_as_message() -> None:
    message = b"tree not-a-header\nparent still-message\n\0\xff"

    parsed = parse_raw_commit_object(_raw_commit(message=message))

    assert parsed.message == message


def test_raw_commit_object_rejects_a_duplicate_tree_header() -> None:
    payload = _raw_commit(extra_headers=(b"tree " + (b"c" * 40),))

    with pytest.raises(CommitContractError) as error:
        parse_raw_commit_object(payload)

    assert error.value.code == "malformed_commit_object"


def test_raw_commit_object_rejects_a_duplicate_author_header() -> None:
    duplicate = b"author Other <other@example.test> 1700000002 +0000"
    payload = _raw_commit(extra_headers=(duplicate,))

    with pytest.raises(CommitContractError) as error:
        parse_raw_commit_object(payload)

    assert error.value.code == "malformed_commit_object"


def test_raw_commit_object_rejects_a_duplicate_committer_header() -> None:
    duplicate = b"committer Other <other@example.test> 1700000002 +0000"
    payload = _raw_commit(extra_headers=(duplicate,))

    with pytest.raises(CommitContractError) as error:
        parse_raw_commit_object(payload)

    assert error.value.code == "malformed_commit_object"


@pytest.mark.parametrize(
    "payload",
    [
        _raw_commit(parents=()),
        _raw_commit(parents=(_PARENT_OID, "c" * 40)),
        _raw_commit(tree="a" * 39),
        _raw_commit(tree="a" * 64),
        _raw_commit(tree=_ZERO_OID),
        _raw_commit(parents=(_ZERO_OID,)),
        _raw_commit().replace(b"tree ", b"treeX ", 1),
        _raw_commit().replace(b"\nauthor ", b"\nauthorX ", 1),
        _raw_commit().replace(b"\ncommitter ", b"\ncommitterX ", 1),
        _raw_commit().replace(b"\n\n", b"\n", 1),
        b" continuation-without-header\n" + _raw_commit(),
        _raw_commit(extra_headers=(b"malformed-header",)),
        _raw_commit(author=b"\xff <ada@example.test> 1 +0000"),
    ],
)
def test_raw_commit_object_rejects_malformed_or_incomplete_objects(
    payload: bytes,
) -> None:
    with pytest.raises(CommitContractError) as error:
        parse_raw_commit_object(payload)

    assert error.value.code == "malformed_commit_object"
    assert str(error.value) == "Commit object data is malformed."
    assert len(str(error.value).encode()) <= 80
