"""Preserve existing recovery authority across an installed config-owner write."""

import hashlib
import json
import secrets
from contextlib import contextmanager

from tldw_chatbook.Utils.platform_files import os
from tldw_chatbook.Utils.private_paths import PrivateFileWritePrecondition

from . import bootstrap, raw_participants
from .activation import _private
from .admission import Admission, fcntl
from .control_records import _activation_record_identity
from .native_files import flush_directory


@contextmanager
def preserve_owned_binding(source, selected, serialized):
    """Advance only a valid bound fingerprint, never enroll or choose storage."""
    operation = getattr(raw_participants._local, "operation", None)
    state = raw_participants._check(operation, selected, writing=True)
    if (
        state.source is not source
        or state.route != "config"
        or state.selected != selected
    ):
        raise bootstrap.RecoveryRequired("config_binding_source_changed")
    root = bootstrap.default_bootstrap_root()
    pending, profiles, associations = bootstrap._control_records(root)
    registry = bootstrap._registry(root)
    previous = bootstrap._binding(selected, profiles, registry)
    if previous is None:
        # Ordinary/unbound or externally edited selectors gain no new authority.
        yield None
        return
    association = next(
        (row for row in associations if row["selector"] == str(selected)), None
    )
    if pending or (association["activation"] if association else None) != previous.get(
        "activation"
    ):
        raise bootstrap.RecoveryRequired("config_binding_authority_changed")
    hold = next(
        (
            held
            for lease, held in zip(state.leases, state.holds)
            if held is not None
            and lease.execution_scope()[0] == root
            and set(previous["namespaces"]) <= set(lease.execution_scope()[1])
        ),
        None,
    )
    if hold is None:
        raise bootstrap.RecoveryRequired("config_binding_scope_unheld")
    authority = hold.authority
    name = "profile-" + bootstrap._key(str(selected)) + ".json"
    association_name = "activation-" + bootstrap._key(str(selected)) + ".json"
    expected_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    after = dict(previous, fingerprint=expected_hash)
    with (
        _private(root) as parent,
        authority._directory() as staging,
        authority._lock(staging, "registry.lock", fcntl.LOCK_SH),
    ):
        identity = _activation_record_identity(parent, name, previous)
        association_identity = _activation_record_identity(
            parent, association_name, association
        )
        selected_before = os.stat(selected, follow_symlinks=False)

        def check_authority():
            raw_participants._check(operation, selected, writing=True)
            current_pending, _, _ = bootstrap._control_records(root)
            if (
                current_pending != pending
                or bootstrap._registry(root) != registry
                or _activation_record_identity(parent, name, previous) != identity
                or _activation_record_identity(parent, association_name, association)
                != association_identity
            ):
                raise bootstrap.RecoveryRequired("config_binding_authority_changed")

        check_authority()
        if (
            bootstrap._fingerprint(selected) != previous["fingerprint"]
            or os.stat(selected, follow_symlinks=False) != selected_before
        ):
            raise bootstrap.RecoveryRequired("config_binding_preimage_changed")
        state.config_publication = None
        yield PrivateFileWritePrecondition(
            (selected_before.st_dev, selected_before.st_ino)
        )

        def check_publication():
            raw_participants._check(operation, selected, writing=True)
            info = os.stat(selected, follow_symlinks=False)
            if (
                state.config_publication != (selected, (info.st_dev, info.st_ino))
                or bootstrap._fingerprint(selected) != expected_hash
                or os.stat(selected, follow_symlinks=False) != info
            ):
                raise bootstrap.RecoveryRequired("config_binding_publication_changed")

        check_publication()
        check_authority()
        temporary = "config-binding-" + secrets.token_hex(16) + ".json"
        Admission._write_new_record(staging, temporary, json.dumps(after).encode())
        temporary_identity = os.stat(temporary, dir_fd=staging, follow_symlinks=False)
        try:
            flush_directory(staging)
            check_publication()
            check_authority()
            if (
                _activation_record_identity(staging, temporary, after)
                != temporary_identity
            ):
                raise bootstrap.RecoveryRequired("config_binding_stage_changed")
            os.replace(temporary, name, src_dir_fd=staging, dst_dir_fd=parent)
            flush_directory(parent)
            flush_directory(staging)
            check_publication()
            if bootstrap._read(parent, name) != after:
                raise bootstrap.RecoveryRequired("config_binding_publication_changed")
        finally:
            try:
                remaining = os.stat(temporary, dir_fd=staging, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                if (
                    remaining == temporary_identity
                    and _activation_record_identity(staging, temporary, after)
                    == temporary_identity
                ):
                    os.unlink(temporary, dir_fd=staging)
                    flush_directory(staging)
