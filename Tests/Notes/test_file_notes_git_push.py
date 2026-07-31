from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError, asdict, fields, is_dataclass

import pytest

import tldw_chatbook.Notes.file_notes_git_push as push_contracts
from tldw_chatbook.Notes.file_notes_git_push import (
    PushAuthorizationHandle,
    PushAuthorizationProjection,
    PushCandidateProjection,
    PushContractError,
    PushDestinationProjection,
    PushDiagnostic,
    PushDiagnosticCategory,
    PushIncludedNote,
    PushOutcomeProjection,
    PushRecoveryHandle,
    PushRecoveryProjection,
    PushReviewHandle,
    PushReviewProjection,
    RemoteRefObservation,
    build_push_argv,
    build_push_query_argv,
    classify_push_diagnostic,
    parse_ls_remote_refs,
    parse_push_endpoint,
    parse_push_porcelain,
    push_outcome_copy,
    push_recovery_copy,
    validate_destination_ref,
)


_PARENT_OID = "1" * 40
_CANDIDATE_OID = "2" * 40
_DIVERGENT_OID = "3" * 40
_SHA256_PARENT = "a" * 64
_SHA256_CANDIDATE = "b" * 64
_DESTINATION_REF = "refs/heads/main"


def _candidate() -> PushCandidateProjection:
    return PushCandidateProjection(
        local_branch_ref=_DESTINATION_REF,
        parent_oid=_PARENT_OID,
        candidate_oid=_CANDIDATE_OID,
        subject="Publish guarded note",
        included_notes=(PushIncludedNote(7, "Meeting notes"),),
    )


def _destination() -> PushDestinationProjection:
    return parse_push_endpoint(
        "https://example.com/team/notes.git",
        _DESTINATION_REF,
    )


def test_production_transport_admission_rejects_local_and_file_endpoints() -> None:
    """A future refactor must not make local transports user-configurable."""
    admission = push_contracts.TransportAdmission()

    for endpoint in ("/tmp/remote.git", "file:///tmp/remote.git"):
        with pytest.raises(PushContractError) as error:
            push_contracts._admit_push_transport(
                admission,
                endpoint,
                _DESTINATION_REF,
            )

        assert error.value.code == "invalid_endpoint"


def test_private_local_bare_transport_admission_is_explicit_and_nonproduction() -> None:
    """Deleting the private issuer must leave no way to test local proof safely."""
    admission = push_contracts._local_bare_transport_admission_for_tests()

    admitted = push_contracts._admit_push_transport(
        admission,
        "/tmp/disposable-remote.git",
        _DESTINATION_REF,
    )

    assert admitted.test_local_bare is True
    assert admitted.configured_identity
    assert admitted.endpoint is None
    assert admitted.destination.host == "local-test.invalid"


@pytest.mark.parametrize(
    "model_type",
    [
        PushIncludedNote,
        PushCandidateProjection,
        PushDestinationProjection,
        PushAuthorizationProjection,
        PushReviewProjection,
        PushOutcomeProjection,
        PushRecoveryProjection,
        PushAuthorizationHandle,
        PushReviewHandle,
        PushRecoveryHandle,
        RemoteRefObservation,
    ],
)
def test_push_public_contract_models_are_frozen(model_type: type[object]) -> None:
    assert is_dataclass(model_type)
    assert model_type.__dataclass_params__.frozen is True  # type: ignore[attr-defined]


def test_candidate_projection_is_an_immutable_sanitized_snapshot() -> None:
    candidate = _candidate()

    assert candidate.included_note_count == 1
    assert candidate.transition == f"{_PARENT_OID} → {_CANDIDATE_OID}"
    with pytest.raises(FrozenInstanceError):
        candidate.subject = "changed"  # type: ignore[misc]


def test_candidate_markup_looking_text_is_preserved_for_literal_rendering() -> None:
    subject = "[WIP] publish [bold]literal[/bold] notes"
    note_labels = ("[WIP] note", "notes/[draft].md")

    candidate = PushCandidateProjection(
        local_branch_ref=_DESTINATION_REF,
        parent_oid=_PARENT_OID,
        candidate_oid=_CANDIDATE_OID,
        subject=subject,
        included_notes=tuple(
            PushIncludedNote(group_id, label)
            for group_id, label in enumerate(note_labels, start=1)
        ),
    )

    assert candidate.subject == subject
    assert tuple(note.display_text for note in candidate.included_notes) == note_labels


def test_destination_projection_exposes_only_selectable_safe_details() -> None:
    destination = parse_push_endpoint(
        "ssh://git@例え.テスト:2222/srv/notes.git",
        "refs/heads/release",
    )

    assert destination == PushDestinationProjection(
        scheme="ssh",
        host="xn--r8jz45g.xn--zckzah",
        port=2222,
        repository_path="/srv/notes.git",
        destination_ref="refs/heads/release",
        ssh_user="git",
    )
    assert destination.selectable_details == (
        ("Scheme", "ssh"),
        ("Host", "xn--r8jz45g.xn--zckzah"),
        ("Port", "2222"),
        ("SSH user", "git"),
        ("Repository path", "/srv/notes.git"),
        ("Destination ref", "refs/heads/release"),
    )


def test_authorization_and_review_project_fixed_policy_disclosures() -> None:
    candidate = _candidate()
    destination = _destination()
    authorization = PushAuthorizationProjection(destination)
    review = PushReviewProjection(candidate, destination)

    assert authorization.action_label == "Authorize and check"
    assert authorization.terminal_prompts_disabled is True
    assert authorization.helper_contact_possible is True
    assert authorization.trusts_remote_content is False
    assert review.exact_lease == f"{_DESTINATION_REF}:{_PARENT_OID}"
    assert review.exact_refspec == f"{_CANDIDATE_OID}:{_DESTINATION_REF}"
    assert review.hooks_bypassed is True
    assert review.later_note_edits_remain_local is True


@pytest.mark.parametrize(
    "factory",
    [
        lambda: PushAuthorizationProjection(  # type: ignore[arg-type]
            {"effective_endpoint": "https://user:secret@example.test/repo.git"}
        ),
        lambda: PushReviewProjection(  # type: ignore[arg-type]
            {"token": "secret"},
            _destination(),
        ),
        lambda: PushReviewProjection(  # type: ignore[arg-type]
            _candidate(),
            {"effective_endpoint": "ext::hostile"},
        ),
        lambda: PushOutcomeProjection(  # type: ignore[arg-type]
            "uncertain",
            "Uncertain",
            "Currently unknown.",
            recovery_available=1,
        ),
        lambda: PushRecoveryProjection(  # type: ignore[arg-type]
            {"effective_endpoint": "ext::hostile"},
            "uncertain",
            "Uncertain",
            "Currently unknown.",
            True,
        ),
        lambda: PushRecoveryProjection(
            _destination(),
            "uncertain",
            "Uncertain",
            "Currently unknown.",
            1,  # type: ignore[arg-type]
        ),
    ],
)
def test_public_projections_reject_mutable_or_nonboolean_authority(
    factory: Callable[[], object],
) -> None:
    with pytest.raises(PushContractError) as error:
        factory()

    assert error.value.code == "unsafe_text"
    assert "secret" not in str(error.value)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: PushDiagnostic(
            PushDiagnosticCategory.UNKNOWN_FAILURE,
            "fatal: https://user:super-secret-token@example.test/repo.git",
        ),
        lambda: PushOutcomeProjection(
            "uncertain",
            "Uncertain",
            "Raw SSH failure for /private/path-canary/repo.git",
            recovery_available=True,
        ),
        lambda: PushRecoveryProjection(
            _destination(),
            "uncertain",
            "Uncertain",
            "Raw helper token: super-secret-token",
            can_check_again=True,
        ),
    ],
)
def test_public_copy_contracts_reject_caller_supplied_raw_copy(
    factory: Callable[[], object],
) -> None:
    with pytest.raises(PushContractError) as error:
        factory()

    rendered = repr(error.value)
    assert error.value.code == "unsafe_text"
    assert "super-secret-token" not in rendered
    assert "path-canary" not in rendered


def test_outcome_projection_rejects_inconsistent_canonical_flags() -> None:
    with pytest.raises(PushContractError) as error:
        PushOutcomeProjection(
            "already_published",
            "Already published",
            (
                "The configured destination currently points to this commit. "
                "No push was started by Chatbook."
            ),
            recovery_available=True,
        )

    assert error.value.code == "unsafe_text"


def test_recovery_projection_rejects_inconsistent_canonical_flags() -> None:
    with pytest.raises(PushContractError) as error:
        PushRecoveryProjection(
            _destination(),
            "succeeded",
            "Succeeded",
            (
                "A query-only check currently observes the candidate at the "
                "original destination. The observation does not establish the "
                "cause of the update. No push was sent by this check."
            ),
            can_check_again=True,
        )

    assert error.value.code == "unsafe_text"


@pytest.mark.parametrize(
    "model",
    [
        _candidate(),
        _destination(),
        PushAuthorizationProjection(_destination()),
        PushReviewProjection(_candidate(), _destination()),
        push_outcome_copy("uncertain"),
        push_recovery_copy(
            _destination(),
            RemoteRefObservation("parent", _PARENT_OID),
        ),
    ],
)
def test_public_projections_contain_no_private_authority_fields(model: object) -> None:
    forbidden_fragments = {
        "effective_endpoint",
        "endpoint",
        "url",
        "token",
        "capability",
        "handle",
        "authorization_epoch",
        "candidate_generation",
    }
    public_field_names = {
        field.name for field in fields(model) if not field.name.startswith("_")
    }
    serialized = repr(asdict(model))

    assert not any(
        fragment in field_name
        for field_name in public_field_names
        for fragment in forbidden_fragments
    )
    assert "super-secret-token" not in serialized
    assert "https://example.com" not in serialized


@pytest.mark.parametrize(
    "handle_type",
    [PushAuthorizationHandle, PushReviewHandle, PushRecoveryHandle],
)
def test_opaque_push_handles_have_no_public_constructor(
    handle_type: type[object],
) -> None:
    with pytest.raises(TypeError):
        handle_type()


@pytest.mark.parametrize(
    "destination_ref",
    [
        "refs/heads/main",
        "refs/heads/feature/exact-push",
        "refs/heads/release-2026.07",
        "refs/heads/éclair",
    ],
)
def test_destination_ref_accepts_only_exact_heads_refs(destination_ref: str) -> None:
    assert validate_destination_ref(destination_ref) == destination_ref


@pytest.mark.parametrize(
    "destination_ref",
    [
        "",
        "main",
        "heads/main",
        "refs/main",
        "refs/tags/main",
        "refs/heads/",
        "refs/heads//main",
        "refs/heads/./main",
        "refs/heads/.hidden",
        "refs/heads/main.",
        "refs/heads/main.lock",
        "refs/heads/-main",
        "refs/heads/main..next",
        "refs/heads/main@{next",
        "refs/heads/main next",
        "refs/heads/main~1",
        "refs/heads/main^",
        "refs/heads/main:next",
        "refs/heads/main?next",
        "refs/heads/main*",
        "refs/heads/main[next",
        "refs/heads/main\\next",
        "refs/heads/main\x00secret",
        "refs/heads/main\x85secret",
        "refs/heads/main\u202esecret",
        "refs/heads/main\ud800secret",
    ],
)
def test_destination_ref_rejects_relative_malformed_or_hostile_refnames(
    destination_ref: str,
) -> None:
    with pytest.raises(PushContractError) as error:
        validate_destination_ref(destination_ref)

    assert error.value.code == "invalid_destination_ref"
    assert "secret" not in str(error.value)


@pytest.mark.parametrize("format_character", ["\u200b", "\u00ad"])
def test_destination_ref_rejects_default_ignorable_format_characters(
    format_character: str,
) -> None:
    with pytest.raises(PushContractError) as error:
        validate_destination_ref(f"refs/heads/release{format_character}candidate")

    assert error.value.code == "invalid_destination_ref"


@pytest.mark.parametrize(
    ("endpoint", "expected"),
    [
        (
            "https://example.com/team/notes.git",
            PushDestinationProjection(
                "https",
                "example.com",
                443,
                "/team/notes.git",
                _DESTINATION_REF,
            ),
        ),
        (
            "https://例え.テスト/team/notes.git",
            PushDestinationProjection(
                "https",
                "xn--r8jz45g.xn--zckzah",
                443,
                "/team/notes.git",
                _DESTINATION_REF,
            ),
        ),
        (
            "https://example.com:8443/team/notes.git",
            PushDestinationProjection(
                "https",
                "example.com",
                8443,
                "/team/notes.git",
                _DESTINATION_REF,
            ),
        ),
        (
            "ssh://git@example.com:2222/srv/notes.git",
            PushDestinationProjection(
                "ssh",
                "example.com",
                2222,
                "/srv/notes.git",
                _DESTINATION_REF,
                "git",
            ),
        ),
        (
            "git@example.com:team/notes.git",
            PushDestinationProjection(
                "ssh",
                "example.com",
                22,
                "~/team/notes.git",
                _DESTINATION_REF,
                "git",
            ),
        ),
        (
            "git@[2001:db8::1]:team/notes.git",
            PushDestinationProjection(
                "ssh",
                "2001:db8::1",
                22,
                "~/team/notes.git",
                _DESTINATION_REF,
                "git",
            ),
        ),
        (
            "https://[2001:db8::2]/team/notes.git",
            PushDestinationProjection(
                "https",
                "2001:db8::2",
                443,
                "/team/notes.git",
                _DESTINATION_REF,
            ),
        ),
        (
            "ssh://git@[2001:db8::3]/srv/notes.git",
            PushDestinationProjection(
                "ssh",
                "2001:db8::3",
                22,
                "/srv/notes.git",
                _DESTINATION_REF,
                "git",
            ),
        ),
    ],
)
def test_endpoint_parser_accepts_only_verified_https_and_literal_ssh_forms(
    endpoint: str,
    expected: PushDestinationProjection,
) -> None:
    destination = parse_push_endpoint(endpoint, _DESTINATION_REF)

    assert destination == expected
    assert destination.certificate_verification_required is (
        destination.scheme == "https"
    )
    assert destination.host_key_verification_required is (destination.scheme == "ssh")


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://[fe80::1%ETH0]/team/notes.git",
        "https://[fe80::1%25ETH0]/team/notes.git",
        "ssh://git@[fe80::1%Scope With Space]/srv/notes.git",
        "ssh://git@[fe80::1%25En0]/srv/notes.git",
        "git@[fe80::1%en0]:team/notes.git",
        "git@[fe80::1%25ETH0]:team/notes.git",
    ],
)
def test_endpoint_parser_rejects_scoped_ipv6_hosts(endpoint: str) -> None:
    with pytest.raises(PushContractError) as error:
        parse_push_endpoint(endpoint, _DESTINATION_REF)

    assert error.value.code == "invalid_endpoint"


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://user@example.com/team/notes.git",
        "https://user:password@example.com/team/notes.git",
        "ssh://git:password@example.com/team/notes.git",
        "ssh://-oProxyCommand@example.com/team/notes.git",
        "-F@example.com:team/notes.git",
        "https://example.com/team/notes.git?token=super-secret-token",
        "https://example.com/team/notes.git#super-secret-token",
        "https://example.com/team/notes.git?",
        "https://example.com/team/notes.git#",
        "https://example.com:/team/notes.git",
        "ssh://git@example.com:/team/notes.git",
        "http://example.com/team/notes.git",
        "git://example.com/team/notes.git",
        "file:///private/secret/repo.git",
        "ssh://example.com/team/notes.git",
        "/private/secret/repo.git",
        "./repo.git",
        "../repo.git",
        "repo.git",
        "C:\\private\\secret\\repo.git",
        "C:/private/secret/repo.git",
        "\\\\server\\share\\repo.git",
        "ext::ssh -oProxyCommand=hostile example.com",
        "hg::https://example.com/repo",
        "git+ssh://git@example.com/team/notes.git",
        "example.com:team/notes.git",
        "git@example.com:team/notes.git:other",
        "git@example.com:team/[bold]notes.git",
        "git@example.com:team/../notes.git",
        "git@example.com:~root/notes.git",
        "git@host name:team/notes.git",
        "https://exa\u202emple.com/team/notes.git",
        "https://exa\u200fmple.com/team/notes.git",
        "https://example.com/team/\x85notes.git",
        "https://\ud800.example/team/notes.git",
    ],
)
def test_endpoint_parser_rejects_credentials_ambiguity_and_unsafe_transports(
    endpoint: str,
) -> None:
    with pytest.raises(PushContractError) as error:
        parse_push_endpoint(endpoint, _DESTINATION_REF)

    assert error.value.code == "invalid_endpoint"
    assert "super-secret-token" not in str(error.value)
    assert "/private/secret" not in str(error.value)
    assert len(str(error.value).encode()) <= 96


def test_endpoint_parser_normalizes_ssh_identity_without_exposing_a_url() -> None:
    destination = parse_push_endpoint(
        "ssh://release-bot@EXAMPLE.COM/srv/releases.git",
        "refs/heads/stable",
    )

    assert destination.ssh_user == "release-bot"
    assert destination.host == "example.com"
    assert destination.port == 22
    assert destination.repository_path == "/srv/releases.git"
    assert all("://" not in value for _, value in destination.selectable_details)


@pytest.mark.parametrize("format_character", ["\u200b", "\u00ad"])
def test_endpoint_parser_rejects_default_ignorable_hostname_characters(
    format_character: str,
) -> None:
    with pytest.raises(PushContractError) as error:
        parse_push_endpoint(
            f"https://exa{format_character}mple.com/team/notes.git",
            _DESTINATION_REF,
        )

    assert error.value.code == "invalid_endpoint"


@pytest.mark.parametrize(
    ("field_name", "unsafe"),
    [
        ("subject", "safe\x00secret"),
        ("subject", "safe\x9bsecret"),
        ("subject", "safe\u202esecret"),
        ("subject", "safe\ud800secret"),
        ("note", "safe\x1b[2Jsecret"),
        ("note", "safe\u2066secret"),
    ],
)
def test_candidate_projection_rejects_unpreviewable_text(
    field_name: str,
    unsafe: str,
) -> None:
    kwargs: dict[str, object] = {
        "local_branch_ref": _DESTINATION_REF,
        "parent_oid": _PARENT_OID,
        "candidate_oid": _CANDIDATE_OID,
        "subject": "Safe subject",
        "included_notes": (PushIncludedNote(1, "Safe note"),),
    }
    if field_name == "subject":
        kwargs["subject"] = unsafe

    with pytest.raises(PushContractError) as error:
        if field_name == "note":
            kwargs["included_notes"] = (PushIncludedNote(1, unsafe),)
        PushCandidateProjection(**kwargs)  # type: ignore[arg-type]

    assert error.value.code == "unsafe_text"
    assert "secret" not in str(error.value)


@pytest.mark.parametrize(
    ("parent_oid", "candidate_oid"),
    [
        ("1" * 39, _CANDIDATE_OID),
        ("1" * 41, _CANDIDATE_OID),
        ("A" * 40, _CANDIDATE_OID),
        (_PARENT_OID, "2" * 12),
        (_PARENT_OID, "g" * 40),
        ("0" * 40, _CANDIDATE_OID),
        (_PARENT_OID, _PARENT_OID),
        (_SHA256_PARENT, _CANDIDATE_OID),
    ],
)
def test_candidate_projection_requires_full_lowercase_same_algorithm_oids(
    parent_oid: str,
    candidate_oid: str,
) -> None:
    with pytest.raises(PushContractError) as error:
        PushCandidateProjection(
            _DESTINATION_REF,
            parent_oid,
            candidate_oid,
            "Subject",
            (),
        )

    assert error.value.code == "invalid_object_id"


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            f"{_PARENT_OID}\t{_DESTINATION_REF}\n".encode(),
            RemoteRefObservation("parent", _PARENT_OID),
        ),
        (
            f"{_CANDIDATE_OID}\t{_DESTINATION_REF}\n".encode(),
            RemoteRefObservation("candidate", _CANDIDATE_OID),
        ),
        (b"", RemoteRefObservation("missing")),
        (
            f"{_DIVERGENT_OID}\t{_DESTINATION_REF}\n".encode(),
            RemoteRefObservation("divergent", _DIVERGENT_OID),
        ),
    ],
)
def test_ls_remote_parser_classifies_one_exact_destination_record(
    payload: bytes,
    expected: RemoteRefObservation,
) -> None:
    assert (
        parse_ls_remote_refs(
            payload,
            _DESTINATION_REF,
            _PARENT_OID,
            _CANDIDATE_OID,
        )
        == expected
    )


def test_ls_remote_parser_supports_full_sha256_object_ids() -> None:
    payload = f"{_SHA256_CANDIDATE}\t{_DESTINATION_REF}\n".encode()

    assert parse_ls_remote_refs(
        payload,
        _DESTINATION_REF,
        _SHA256_PARENT,
        _SHA256_CANDIDATE,
    ) == RemoteRefObservation("candidate", _SHA256_CANDIDATE)


@pytest.mark.parametrize(
    "payload",
    [
        f"{_PARENT_OID}\t{_DESTINATION_REF}".encode(),
        f"{_PARENT_OID} {_DESTINATION_REF}\n".encode(),
        f"{'A' * 40}\t{_DESTINATION_REF}\n".encode(),
        f"{_PARENT_OID[:-1]}\t{_DESTINATION_REF}\n".encode(),
        f"{_PARENT_OID}\trefs/heads/other\n".encode(),
        (
            f"{_PARENT_OID}\t{_DESTINATION_REF}\n{_PARENT_OID}\t{_DESTINATION_REF}\n"
        ).encode(),
        b"\xff\trefs/heads/main\n",
        b"hostile-unstructured-response\n",
        b"x" * 70_000,
    ],
)
def test_ls_remote_parser_closes_malformed_or_ambiguous_output(
    payload: bytes,
) -> None:
    observation = parse_ls_remote_refs(
        payload,
        _DESTINATION_REF,
        _PARENT_OID,
        _CANDIDATE_OID,
    )

    assert observation == RemoteRefObservation("malformed")
    assert "hostile" not in repr(observation)


def test_push_porcelain_parser_accepts_one_exact_fast_forward_result() -> None:
    payload = (
        b"To https://super-secret-token@example.test/private/repo.git\n"
        + f" \t{_CANDIDATE_OID}:{_DESTINATION_REF}\t".encode()
        + f"{_PARENT_OID[:7]}..{_CANDIDATE_OID[:7]}\n".encode()
        + b"Done\n"
    )

    result = parse_push_porcelain(
        payload,
        _CANDIDATE_OID,
        _DESTINATION_REF,
    )

    assert result.state == "accepted"
    assert "super-secret-token" not in repr(result)
    assert "/private/repo.git" not in repr(result)


def test_push_porcelain_parser_accepts_an_exact_safe_unicode_destination() -> None:
    destination_ref = "refs/heads/éclair"
    payload = (
        f" \t{_CANDIDATE_OID}:{destination_ref}\t"
        f"{_PARENT_OID[:7]}..{_CANDIDATE_OID[:7]}\n"
    ).encode()

    result = parse_push_porcelain(
        payload,
        _CANDIDATE_OID,
        destination_ref,
    )

    assert result.state == "accepted"


def test_push_porcelain_parser_classifies_one_exact_rejection() -> None:
    payload = (
        b"To ssh://git@example.test/repo.git\n"
        + f"!\t{_CANDIDATE_OID}:{_DESTINATION_REF}\t".encode()
        + b"[rejected] (stale info)\n"
        + b"Done\n"
    )

    assert (
        parse_push_porcelain(
            payload,
            _CANDIDATE_OID,
            _DESTINATION_REF,
        ).state
        == "rejected"
    )


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"Done\n",
        (f" \t{_PARENT_OID}:{_DESTINATION_REF}\twrong source\n").encode(),
        (f" \t{_CANDIDATE_OID}:refs/heads/other\twrong destination\n").encode(),
        (f"*\t{_CANDIDATE_OID}:{_DESTINATION_REF}\t[new branch]\n").encode(),
        (
            f" \t{_CANDIDATE_OID}:{_DESTINATION_REF}\tone\n"
            f" \t{_CANDIDATE_OID}:{_DESTINATION_REF}\ttwo\n"
        ).encode(),
        (
            f"?\t{_CANDIDATE_OID}:refs/heads/other\tunknown flag\n"
            f" \t{_CANDIDATE_OID}:{_DESTINATION_REF}\texact line\n"
        ).encode(),
        (f" \t{_CANDIDATE_OID}:{_DESTINATION_REF}\tmissing newline").encode(),
        b"\xff\tbad\tbytes\n",
        b"x" * 70_000,
    ],
)
def test_push_porcelain_parser_rejects_nonexact_or_ambiguous_results(
    payload: bytes,
) -> None:
    assert (
        parse_push_porcelain(
            payload,
            _CANDIDATE_OID,
            _DESTINATION_REF,
        ).state
        == "malformed"
    )


@pytest.mark.parametrize(
    ("payload", "category"),
    [
        (
            b"Permission denied (publickey).",
            PushDiagnosticCategory.AUTHENTICATION_FAILED,
        ),
        (
            b"Host key verification failed.",
            PushDiagnosticCategory.HOST_VERIFICATION_FAILED,
        ),
        (
            b"remote: pre-receive hook declined",
            PushDiagnosticCategory.REMOTE_REJECTED,
        ),
        (
            b"Could not resolve host: example.test",
            PushDiagnosticCategory.TRANSPORT_FAILED,
        ),
        (
            b"fatal: protocol error: bad line length",
            PushDiagnosticCategory.INVALID_RESPONSE,
        ),
        (
            b"arbitrary failure",
            PushDiagnosticCategory.UNKNOWN_FAILURE,
        ),
    ],
)
def test_diagnostic_classifier_returns_only_closed_bounded_categories(
    payload: bytes,
    category: PushDiagnosticCategory,
) -> None:
    diagnostic = classify_push_diagnostic(payload)

    assert diagnostic.category is category
    assert len(diagnostic.message.encode()) <= 160


def test_diagnostic_categories_are_closed_and_explicit() -> None:
    assert set(PushDiagnosticCategory) == {
        PushDiagnosticCategory.AUTHENTICATION_FAILED,
        PushDiagnosticCategory.HOST_VERIFICATION_FAILED,
        PushDiagnosticCategory.REMOTE_REJECTED,
        PushDiagnosticCategory.TRANSPORT_FAILED,
        PushDiagnosticCategory.INVALID_RESPONSE,
        PushDiagnosticCategory.UNKNOWN_FAILURE,
    }


@pytest.mark.parametrize(
    "payload",
    [
        b"\x00\x1b\x7f\x85\x9b",
        "\u202e[bold]secret-markup[/bold]".encode(),
        b"\xff\xfeinvalid-encoding",
        b"https://user:super-secret-token@example.test/repo.git",
        b"/private/path-canary/session/repo.git",
    ],
)
def test_diagnostic_classifier_discards_raw_bytes_and_canaries(
    payload: bytes,
) -> None:
    diagnostic = classify_push_diagnostic(payload)
    rendered = repr(diagnostic) + diagnostic.message

    assert "secret" not in rendered
    assert "bold" not in rendered
    assert "private/path-canary" not in rendered
    assert "\x1b" not in rendered
    assert "\u202e" not in rendered


def test_query_argv_is_the_exact_frozen_destination_contract() -> None:
    frozen = push_contracts._freeze_push_endpoint(
        "https://例え.テスト/team/notes.git",
        _DESTINATION_REF,
    )

    assert build_push_query_argv(
        "git",
        "/private/network.git",
        frozen,
    ) == (
        "git",
        "--git-dir=/private/network.git",
        "--no-replace-objects",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "maintenance.auto=false",
        "-c",
        "gc.auto=0",
        "ls-remote",
        "--refs",
        "--",
        "https://xn--r8jz45g.xn--zckzah/team/notes.git",
        _DESTINATION_REF,
    )


def test_push_argv_is_the_exact_one_commit_compare_and_swap_contract() -> None:
    frozen = push_contracts._freeze_push_endpoint(
        "git@example.com:team/notes.git",
        _DESTINATION_REF,
    )

    argv = build_push_argv(
        "git",
        "/private/network.git",
        frozen,
        _PARENT_OID,
        _CANDIDATE_OID,
    )

    assert argv == (
        "git",
        "--git-dir=/private/network.git",
        "--no-replace-objects",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "maintenance.auto=false",
        "-c",
        "gc.auto=0",
        "push",
        "--porcelain",
        "--no-verify",
        "--no-follow-tags",
        "--recurse-submodules=no",
        f"--force-with-lease={_DESTINATION_REF}:{_PARENT_OID}",
        "--",
        "git@example.com:team/notes.git",
        f"{_CANDIDATE_OID}:{_DESTINATION_REF}",
    )


def test_push_argv_excludes_all_broadening_and_implicit_behavior() -> None:
    frozen = push_contracts._freeze_push_endpoint(
        "https://example.com/team/notes.git",
        _DESTINATION_REF,
    )
    argv = build_push_argv(
        "git",
        "/private/network.git",
        frozen,
        _PARENT_OID,
        _CANDIDATE_OID,
    )

    forbidden = {
        "origin",
        "--tags",
        "--push-option",
        "-o",
        "--delete",
        "--mirror",
        "--set-upstream",
        "-u",
        "--recurse-submodules=on-demand",
        "--recurse-submodules=check",
        "--force",
        "--all",
        "--atomic",
        "--follow-tags",
        "--retry",
    }
    assert forbidden.isdisjoint(argv)
    assert argv.count(f"{_CANDIDATE_OID}:{_DESTINATION_REF}") == 1
    assert argv.count(f"--force-with-lease={_DESTINATION_REF}:{_PARENT_OID}") == 1


def test_private_endpoint_has_no_public_constructor_or_revealing_repr() -> None:
    with pytest.raises(TypeError):
        push_contracts._FrozenPushEndpoint()

    frozen = push_contracts._freeze_push_endpoint(
        "https://example.com/team/notes.git",
        _DESTINATION_REF,
    )

    assert "https://example.com" not in repr(frozen)
    assert frozen.projection == _destination()


def test_push_builder_rejects_a_forged_private_endpoint_instance() -> None:
    forged = object.__new__(push_contracts._FrozenPushEndpoint)
    object.__setattr__(forged, "projection", _destination())

    with pytest.raises(PushContractError) as error:
        build_push_query_argv("git", "/private/network.git", forged)

    assert not hasattr(push_contracts, "_ENDPOINT_SEAL")
    assert error.value.code == "invalid_endpoint"
    assert "super-secret-token" not in str(error.value)


@pytest.mark.parametrize(
    ("parent_oid", "candidate_oid"),
    [
        ("1" * 12, _CANDIDATE_OID),
        (_PARENT_OID, "2" * 12),
        ("A" * 40, _CANDIDATE_OID),
        (_PARENT_OID, "B" * 40),
    ],
)
def test_push_builder_rejects_non_authoritative_object_ids(
    parent_oid: str,
    candidate_oid: str,
) -> None:
    frozen = push_contracts._freeze_push_endpoint(
        "https://example.com/team/notes.git",
        _DESTINATION_REF,
    )

    with pytest.raises(PushContractError) as error:
        build_push_argv(
            "git",
            "/private/network.git",
            frozen,
            parent_oid,
            candidate_oid,
        )

    assert error.value.code == "invalid_object_id"


@pytest.mark.parametrize(
    ("state", "title"),
    [
        ("already_published", "Already published"),
        ("succeeded", "Succeeded"),
        (
            "failed_no_update_observed",
            "Failed with no update currently observed",
        ),
        ("uncertain", "Uncertain"),
    ],
)
def test_outcome_copy_is_bounded_honest_and_point_in_time(
    state: str,
    title: str,
) -> None:
    outcome = push_outcome_copy(state)  # type: ignore[arg-type]
    copy = f"{outcome.title} {outcome.message}".lower()

    assert outcome.title == title
    assert outcome.point_in_time is True
    assert "currently" in copy
    assert "no server work occurred" not in copy
    assert "nothing happened" not in copy
    assert "never reached the server" not in copy
    assert len(outcome.message.encode()) <= 320


def test_already_published_copy_says_no_push_was_started() -> None:
    outcome = push_outcome_copy("already_published")

    assert "No push was started by Chatbook." in outcome.message


def test_uncertain_copy_forbids_retry_and_offers_query_only_recovery() -> None:
    outcome = push_outcome_copy("uncertain")

    assert outcome.recovery_available is True
    assert "Do not push again automatically." in outcome.message
    assert "Check the original destination again without pushing." in outcome.message


@pytest.mark.parametrize(
    ("observation", "state", "title"),
    [
        (
            RemoteRefObservation("candidate", _CANDIDATE_OID),
            "succeeded",
            "Succeeded",
        ),
        (
            RemoteRefObservation("parent", _PARENT_OID),
            "uncertain",
            "Uncertain",
        ),
        (
            RemoteRefObservation("divergent", _DIVERGENT_OID),
            "needs_attention",
            "Needs attention",
        ),
        (
            RemoteRefObservation("missing"),
            "needs_attention",
            "Needs attention",
        ),
        (
            RemoteRefObservation("malformed"),
            "uncertain",
            "Uncertain",
        ),
    ],
)
def test_recovery_copy_is_query_only_and_never_claims_causation(
    observation: RemoteRefObservation,
    state: str,
    title: str,
) -> None:
    recovery = push_recovery_copy(_destination(), observation)
    copy = f"{recovery.title} {recovery.message}".lower()

    assert recovery.state == state
    assert recovery.title == title
    assert recovery.query_only is True
    assert "currently" in copy
    assert "no push was sent by this check" in copy
    assert "chatbook caused" not in copy
    assert "no server work occurred" not in copy


def test_recovery_parent_observation_remains_uncertain() -> None:
    recovery = push_recovery_copy(
        _destination(),
        RemoteRefObservation("parent", _PARENT_OID),
    )

    assert recovery.state == "uncertain"
    assert recovery.can_check_again is True
    assert "Remote-side work may still be pending" in recovery.message


def test_recovery_malformed_observation_remains_uncertain() -> None:
    recovery = push_recovery_copy(
        _destination(),
        RemoteRefObservation("malformed"),
    )

    assert recovery.state == "uncertain"
    assert recovery.can_check_again is True
    assert "prior attempt remains uncertain" in recovery.message
