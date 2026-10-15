"""Bounded immutable acquisition, owned by the existing profile storage worker."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from urllib.parse import urlencode, urlsplit
from uuid import uuid4

from tldw_chatbook.Notes.git_process_containment import (
    ProcessTreeAdmissionError,
    ProcessTreeController,
)
from tldw_chatbook.Utils.private_paths import (
    PrivateFileWritePrecondition,
    open_private_binary,
    unlink_private_file,
)

from . import package_files, retention
from .authority_store import _directory, _write
from .data_cleanup import boot_identity, physical_identity
from .git_source import SourceLocator, parse_source_locator, select_local_source
from .models import AcquiredSnapshot

MAX_ACQUISITIONS = 2
ACQUISITION_SECONDS = 120
MAX_STAGING_BYTES = 500 * 1024 * 1024
MAX_CUSTODY_RECORDS = retention.TERMINAL_RECEIPTS
POLL_SECONDS = 0.05
HOST_GIT_SYSTEM = Path("/etc/gitconfig")
HOST_SSH_SYSTEM = Path("/etc/ssh/ssh_config")


class AcquisitionError(package_files.PackageFileError):
    """Metadata-only failure; external command output never reaches diagnostics."""


def _physical(path):
    with package_files._directory_fd(path) as fd:
        return physical_identity(fd)


def _record_path(owner, operation_id):
    name = hashlib.sha256(operation_id.encode()).hexdigest()
    return owner.root / "acquisition-custody" / (name + ".json")


def _save(owner, authority, record):
    owner.require_owner(owner.root)
    path = _record_path(owner, record["operation_id"])
    condition = PrivateFileWritePrecondition.missing()
    if os.path.lexists(path):
        with open_private_binary(path) as opened:
            data = opened.stream.read(65537)
            if not opened.result.verified_private or len(data) > 65536:
                raise AcquisitionError("acquisition_custody_invalid")
            previous = json.loads(data)
            authority.verify_acquisition_custody(previous["body"], previous["mac"])
            old = previous["body"]
            if any(
                old[key] != value
                for key, value in record.items()
                if key not in {"phase", "recovery_referenced"}
            ):
                raise AcquisitionError("acquisition_custody_changed")
            if (old["phase"], record["phase"]) not in {
                ("pending", "terminal"),
                ("terminal", "terminal"),
                ("terminal", "cleaned"),
            }:
                raise AcquisitionError("acquisition_custody_changed")
            condition = PrivateFileWritePrecondition.from_opened(opened)
    _write(
        path,
        {
            "body": record,
            "mac": authority.sign_acquisition_custody(record),
        },
        precondition=condition,
    )


def _inventory(owner, authority):
    owner.require_owner(owner.root)
    records = []
    directory = owner.root / "acquisition-custody"
    if directory.exists():
        with package_files._directory_fd(directory) as fd, os.scandir(fd) as entries:
            for entry in entries:
                if len(records) >= MAX_CUSTODY_RECORDS or not re.fullmatch(
                    r"[0-9a-f]{64}\.json", entry.name
                ):
                    raise AcquisitionError("acquisition_custody_limit")
                with open_private_binary(directory / entry.name) as opened:
                    data = opened.stream.read(65537)
                    if not opened.result.verified_private or len(data) > 65536:
                        raise AcquisitionError("acquisition_custody_invalid")
                    envelope = json.loads(data)
                if set(envelope) != {"body", "mac"}:
                    raise AcquisitionError("acquisition_custody_invalid")
                body = envelope["body"]
                authority.verify_acquisition_custody(body, envelope["mac"])
                if _record_path(owner, body["operation_id"]).name != entry.name:
                    raise AcquisitionError("acquisition_custody_invalid")
                records.append(body)
    return records


def _qualify(owner, record, *, leaf=True):
    owner.require_owner(owner.root)
    if record["boot"] != boot_identity() or record["root"] != _physical(owner.root):
        raise AcquisitionError("acquisition_owner_changed")
    anchor = owner.root / "staging"
    if record["anchor"] != _physical(anchor):
        raise AcquisitionError("acquisition_anchor_changed")
    destination = owner.acquisition_destination(record["operation_id"])
    if destination.parent.name != record["name"] or record["output"] != "snapshot":
        raise AcquisitionError("acquisition_destination_invalid")
    if leaf and record["leaf"] != _physical(destination.parent):
        raise AcquisitionError("acquisition_staging_changed")
    return destination


def _retire_cleaned(owner, authority, records, now, *, make_room=False):
    """Drop only qualified absent cleaned history, preserving unknown custody."""
    retained = list(records)
    for record in sorted(records, key=lambda row: row["created_at"]):
        if record["phase"] != "cleaned" or record["recovery_referenced"]:
            continue
        expired = (
            now - datetime.fromisoformat(record["created_at"]) > retention.RECEIPT_AGE
        )
        if not expired and not (make_room and len(retained) >= MAX_CUSTODY_RECORDS):
            continue
        destination = _qualify(owner, record, leaf=False)
        with package_files._directory_fd(destination.parent.parent) as anchor_fd:
            try:
                os.stat(
                    destination.parent.name, dir_fd=anchor_fd, follow_symlinks=False
                )
            except FileNotFoundError:
                pass
            else:
                continue
            path = _record_path(owner, record["operation_id"])
            with open_private_binary(path) as opened:
                raw = opened.stream.read(65537)
                if not opened.result.verified_private or len(raw) > 65536:
                    raise AcquisitionError("acquisition_custody_invalid")
                current = json.loads(raw)
                authority.verify_acquisition_custody(current["body"], current["mac"])
                if current["body"] != record:
                    raise AcquisitionError("acquisition_custody_changed")
                unlink_private_file(path)
            retained.remove(record)
    return retained


def _reserve(owner, authority, operation_id, destination):
    expected = owner.acquisition_destination(operation_id)
    if destination != expected:
        raise AcquisitionError("acquisition_destination_invalid")
    authority.verify_current()
    records = _retire_cleaned(
        owner,
        authority,
        _inventory(owner, authority),
        datetime.now(UTC),
        make_room=True,
    )
    if len(records) >= MAX_CUSTODY_RECORDS:
        raise AcquisitionError("acquisition_custody_limit")
    if any(row["operation_id"] == operation_id for row in records):
        raise AcquisitionError("acquisition_operation_reused")
    anchor = owner.root / "staging"
    _directory(anchor)
    # A replayed terminal receipt cannot hide a different live incarnation.
    for row in records:
        destination_for_row = _qualify(owner, row, leaf=row["phase"] != "cleaned")
        if row["phase"] == "cleaned" and os.path.lexists(destination_for_row.parent):
            raise AcquisitionError("acquisition_custody_changed")
    known = {row["name"] for row in records}
    unsettled = sum(row["phase"] == "pending" for row in records)
    with package_files._directory_fd(anchor) as fd, os.scandir(fd) as entries:
        for index, entry in enumerate(entries):
            if index >= MAX_CUSTODY_RECORDS:
                raise AcquisitionError("acquisition_custody_limit")
            if entry.name not in known:
                unsettled += 1
    if unsettled >= MAX_ACQUISITIONS:
        raise AcquisitionError("acquisition_concurrency_limit")
    retention.require_capacity(owner.root, 0)
    # Never replace a leaf even when its journal is missing.
    destination.parent.mkdir(mode=0o700)
    record = {
        "version": 1,
        "namespace_id": authority.namespace_id(),
        "operation_id": operation_id,
        "nonce": uuid4().hex,
        "session": owner.session_id,
        "name": destination.parent.name,
        "output": "snapshot",
        "created_at": datetime.now(UTC).isoformat(),
        "boot": boot_identity(),
        "root": _physical(owner.root),
        "anchor": _physical(anchor),
        "leaf": _physical(destination.parent),
        "phase": "pending",
        "recovery_referenced": False,
    }
    _save(owner, authority, record)
    return record


def release_snapshot(owner, authority, snapshot: AcquiredSnapshot) -> None:
    """Release a consumed review candidate; age/terminal custody still gate deletion."""
    records = _inventory(owner, authority)
    record = next(
        (row for row in records if row["operation_id"] == snapshot.operation_id), None
    )
    if (
        record is None
        or record["nonce"] != snapshot.acquisition_nonce
        or record["phase"] != "terminal"
    ):
        raise AcquisitionError("acquisition_release_mismatch")
    if str(_qualify(owner, record)) != snapshot.materialized_identity:
        raise AcquisitionError("acquisition_release_mismatch")
    record["recovery_referenced"] = False
    _save(owner, authority, record)


def cleanup_acquisitions(
    owner, authority, *, now: datetime | None = None
) -> tuple[str, ...]:
    """Reconcile exact authenticated terminal receipts before using the F7 seam."""
    now = now or datetime.now(UTC)
    removed = []
    for record in _inventory(owner, authority):
        if record["phase"] != "terminal" or record["recovery_referenced"]:
            continue
        _qualify(owner, record)
        custody = retention.StagingCustody(
            name=record["name"],
            owner_id=record["session"],
            created_at=datetime.fromisoformat(record["created_at"]),
            device=record["leaf"]["device"],
            inode=record["leaf"]["inode"],
            reconciled=True,
            terminal=True,
            recovery_referenced=False,
        )
        if retention.cleanup_staging(owner, custody, now):
            record["phase"] = "cleaned"
            _save(owner, authority, record)
            removed.append(record["operation_id"])
    _retire_cleaned(owner, authority, _inventory(owner, authority), now)
    return tuple(removed)


class _Producer:
    def __init__(self, owner, authority, record, cancelled):
        self.owner, self.authority, self.record = owner, authority, record
        self.cancelled = cancelled
        self.deadline = time.monotonic() + ACQUISITION_SECONDS
        self.staging = owner.acquisition_destination(record["operation_id"]).parent
        self.controller = ProcessTreeController()
        self.last_quota = 0.0
        self.workers = set()
        self.git = shutil.which("git")
        self.environment = {
            key: value
            for key, value in os.environ.items()
            if key
            in {
                "PATH",
                "HOME",
                "XDG_CONFIG_HOME",
                "SSH_AUTH_SOCK",
                "TMPDIR",
                "SYSTEMROOT",
            }
        }
        self.environment.update(
            {
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_CONFIG_SYSTEM": os.devnull,
                "GIT_TERMINAL_PROMPT": "0",
                "GIT_ALLOW_PROTOCOL": "https:ssh:file",
                "GIT_LFS_SKIP_SMUDGE": "1",
                "GIT_NO_REPLACE_OBJECTS": "1",
                "GIT_OPTIONAL_LOCKS": "0",
                "LC_ALL": "C",
            }
        )

    async def worker(self, callback, *args, **kwargs):
        task = asyncio.create_task(asyncio.to_thread(callback, *args, **kwargs))
        self.workers.add(task)
        try:
            return await asyncio.shield(task)
        finally:
            if task.done():
                self.workers.discard(task)

    def write_blob(self, output, data, mode):
        output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with output.open("xb") as stream:
            for offset in range(0, len(data), package_files.CHUNK_BYTES):
                self.quota()
                stream.write(data[offset : offset + package_files.CHUNK_BYTES])
        output.chmod(0o700 if mode == "100755" else 0o600)

    def checkpoint(self):
        if self.cancelled.is_set():
            raise AcquisitionError("acquisition_cancelled_limit")
        if time.monotonic() >= self.deadline:
            raise AcquisitionError("acquisition_deadline_limit")

    def quota(self, *, force=False):
        self.checkpoint()
        now = time.monotonic()
        if not force and now - self.last_quota < POLL_SECONDS:
            return
        self.last_quota = now
        total = 0
        for directory, dirs, files in os.walk(self.staging, followlinks=False):
            self.checkpoint()
            for name in dirs + files:
                info = (Path(directory) / name).lstat()
                if stat.S_ISREG(info.st_mode):
                    total += info.st_size
                elif not stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode):
                    raise AcquisitionError("acquisition_staging_type")
                if total > MAX_STAGING_BYTES:
                    raise AcquisitionError("acquisition_staging_limit")
        retention.require_capacity(self.owner.root, 0)

    async def command(
        self,
        *args,
        limit=package_files.MAX_FILE_BYTES,
        executable=None,
        env=None,
        stdin=None,
    ):
        self.checkpoint()
        executable = executable or self.git
        if executable is None:
            raise AcquisitionError("plugin_git_missing")
        tree = None
        readers = []

        async def read(stream, maximum):
            output = bytearray()
            while block := await stream.read(package_files.CHUNK_BYTES):
                output.extend(block)
                if len(output) > maximum:
                    raise AcquisitionError("acquisition_output_limit")
            return bytes(output)

        try:
            try:
                tree = await self.controller.spawn(
                    executable,
                    *args,
                    cwd=str(self.staging),
                    environment=env or self.environment,
                    stdin=stdin is not None,
                )
            except ProcessTreeAdmissionError as error:
                tree = error.tree
                raise AcquisitionError("acquisition_process_admission") from None
            readers = [
                asyncio.create_task(read(tree.process.stdout, limit)),
                asyncio.create_task(read(tree.process.stderr, 65536)),
            ]
            if stdin is not None:
                tree.process.stdin.write(stdin)
                await tree.process.stdin.drain()
                tree.process.stdin.close()
            while tree.process.returncode is None or not all(
                task.done() for task in readers
            ):
                self.checkpoint()
                for task in readers:
                    if task.done() and task.exception() is not None:
                        raise task.exception()
                await self.worker(self.quota)
                await asyncio.sleep(POLL_SECONDS)
            output, _ = await asyncio.gather(*readers)
            if tree.process.returncode:
                raise AcquisitionError("acquisition_git_failed")
            return output
        except OSError:
            raise AcquisitionError("acquisition_process_unavailable") from None
        finally:
            if tree is not None:

                async def settle():
                    while True:
                        try:
                            if await self.controller.wait(tree, timeout=0):
                                self.controller.close(tree)
                                return
                            self.controller.terminate(tree)
                            if await self.controller.wait(tree, timeout=0.2):
                                self.controller.close(tree)
                                return
                            self.controller.kill(tree)
                        # Unknown failures preserve custody; exception text may be sensitive.
                        except Exception:  # noqa: BLE001, S110
                            pass
                        await asyncio.sleep(POLL_SECONDS)

                settling = asyncio.create_task(settle())
                while not settling.done():
                    try:
                        await asyncio.shield(settling)
                    except asyncio.CancelledError:
                        # Even repeated cancellation cannot manufacture terminal proof.
                        self.cancelled.set()
                settling.result()
            for task in readers:
                if not task.done():
                    task.cancel()
            if readers:
                await asyncio.gather(*readers, return_exceptions=True)

    async def git_command(self, *args, **kwargs):
        return await self.command(
            "-c",
            "core.hooksPath=" + os.devnull,
            "-c",
            "core.fsmonitor=false",
            "-c",
            "maintenance.auto=false",
            "-c",
            "gc.auto=0",
            "-c",
            "fetch.recurseSubmodules=false",
            "-c",
            "http.followRedirects=false",
            "-c",
            "protocol.ext.allow=never",
            *args,
            **kwargs,
        )

    async def host_authentication(self, source):
        """Project explicit host settings without reading source repository config."""
        if any(
            os.environ.get(key)
            for key in (
                "GIT_SSH",
                "GIT_SSH_COMMAND",
                "GIT_ASKPASS",
                "SSH_ASKPASS",
                "SSH_SK_PROVIDER",
                "HTTPS_PROXY",
                "https_proxy",
                "ALL_PROXY",
                "all_proxy",
            )
        ):
            raise AcquisitionError("acquisition_host_auth_unsupported")
        home = Path(self.environment["HOME"])
        global_files = [
            HOST_GIT_SYSTEM,
            Path(self.environment.get("XDG_CONFIG_HOME", str(home / ".config")))
            / "git/config",
            home / ".gitconfig",
        ]
        selected = []
        for path in global_files:
            if not path.exists():
                continue
            raw = await self.command(
                "config",
                "--file",
                str(path),
                "--includes",
                "--null",
                "--list",
                limit=65536,
            )
            for entry in raw.rstrip(b"\0").split(b"\0"):
                key, _, value = entry.decode("utf-8").partition("\n")
                lower = key.lower()
                if lower.startswith("credential."):
                    if lower.rsplit(".", 1)[-1] not in {
                        "helper",
                        "username",
                        "usehttppath",
                    }:
                        raise AcquisitionError("acquisition_host_auth_unsupported")
                    selected.append((key, value))
                elif lower.startswith("http."):
                    if lower.rsplit(".", 1)[-1] not in {
                        "sslverify",
                        "sslcainfo",
                        "sslcapath",
                        "sslcert",
                        "sslkey",
                        "sslcertpasswordprotected",
                        "sslversion",
                        "sslcipherlist",
                        "pinnedpubkey",
                    }:
                        raise AcquisitionError("acquisition_host_auth_unsupported")
                    selected.append((key, value))
                elif lower.startswith("url.") or lower in {
                    "core.sshcommand",
                    "core.askpass",
                }:
                    raise AcquisitionError("acquisition_host_auth_unsupported")
                if len(selected) > 256:
                    raise AcquisitionError("acquisition_host_auth_limit")
        config = self.staging / "host-git-config"
        config.touch(mode=0o600)
        for key, value in selected:
            await self.command(
                "config", "--file", str(config), "--add", key, value, limit=65536
            )
        self.environment["GIT_CONFIG_GLOBAL"] = str(config)
        self.environment["GIT_ALLOW_PROTOCOL"] = source.transport
        if source.transport == "ssh":
            await self.ssh_authentication(source, home)

    async def ssh_authentication(self, source, home):
        ssh = shutil.which("ssh")
        if ssh is None:
            raise AcquisitionError("plugin_ssh_missing")
        config_files = [home / ".ssh/config", HOST_SSH_SYSTEM]
        contents = []
        supported = {
            "host",
            "hostname",
            "user",
            "port",
            "identityfile",
            "certificatefile",
            "identitiesonly",
            "identityagent",
            "pubkeyauthentication",
            "preferredauthentications",
            "userknownhostsfile",
            "globalknownhostsfile",
            "stricthostkeychecking",
            "hostkeyalgorithms",
            "pubkeyacceptedalgorithms",
            "casignaturealgorithms",
            "hostkeyalias",
            "verifyhostkeydns",
            "updatehostkeys",
            "securitykeyprovider",
            "pkcs11provider",
            "proxycommand",
            "proxyjump",
            "canonicalizehostname",
            "forwardagent",
            "forwardx11",
            "localcommand",
            "remotecommand",
            "permitlocalcommand",
            "localforward",
            "remoteforward",
            "dynamicforward",
        }
        for path in config_files:
            if not path.exists():
                continue
            with path.open("rb") as stream:
                data = stream.read(65537)
            if len(data) > 65536:
                raise AcquisitionError("acquisition_host_auth_limit")
            content = data.decode("utf-8")
            for line in content.splitlines():
                words = shlex.split(line, comments=True)
                if words and words[0].lower().split("=", 1)[0] not in supported:
                    raise AcquisitionError("acquisition_ssh_config_unsupported")
                if (
                    words
                    and words[0].lower()
                    in {"userknownhostsfile", "globalknownhostsfile"}
                    and any(any(c.isspace() for c in value) for value in words[1:])
                ):
                    raise AcquisitionError("acquisition_ssh_config_unsupported")
            contents.append(content)
        captured = self.staging / "host-ssh-input"
        captured.write_text("\nHost *\n".join(contents))
        captured.chmod(0o600)
        parsed = urlsplit(source.url)
        target = parsed.hostname
        if parsed.username:
            target = parsed.username + "@" + target
        arguments = ["-G", "-F", str(captured)]
        if parsed.port:
            arguments += ["-p", str(parsed.port)]
        raw = await self.command(*arguments, target, executable=ssh, limit=65536)
        settings = {}
        for line in raw.decode().splitlines():
            key, _, value = line.partition(" ")
            settings.setdefault(key, []).append(value)

        def one(key, default="none"):
            return settings.get(key, [default])[-1]

        if (
            one("hostname") != parsed.hostname
            or one("port") != str(parsed.port or 22)
            or (parsed.username is not None and one("user") != parsed.username)
            or any(
                one(key) not in {"none", "no", "false"}
                for key in (
                    "proxycommand",
                    "proxyjump",
                    "canonicalizehostname",
                    "forwardagent",
                    "forwardx11",
                    "localcommand",
                    "remotecommand",
                    "permitlocalcommand",
                    "localforward",
                    "remoteforward",
                    "dynamicforward",
                    "pkcs11provider",
                )
            )
            or one("securitykeyprovider", "internal")
            not in {"internal", "none", "$SSH_SK_PROVIDER"}
        ):
            raise AcquisitionError("acquisition_ssh_config_unsupported")
        retained = {
            "identityfile",
            "certificatefile",
            "identitiesonly",
            "identityagent",
            "pubkeyauthentication",
            "preferredauthentications",
            "userknownhostsfile",
            "globalknownhostsfile",
            "stricthostkeychecking",
            "hostkeyalgorithms",
            "pubkeyacceptedalgorithms",
            "casignaturealgorithms",
            "hostkeyalias",
            "verifyhostkeydns",
            "updatehostkeys",
            "user",
            "hostname",
            "port",
        }
        output = [
            "Host *",
            "BatchMode yes",
            "ProxyCommand none",
            "ProxyJump none",
            "CanonicalizeHostname no",
            "ForwardAgent no",
            "PermitLocalCommand no",
            "SecurityKeyProvider internal",
        ]
        for key in sorted(retained):
            for value in settings.get(key, []):
                if key in {
                    "identityfile",
                    "certificatefile",
                    "identityagent",
                    "hostkeyalias",
                }:
                    value = json.dumps(value)
                output.append(key + " " + value)
        private = self.staging / "host-ssh-config"
        private.write_text("\n".join(output) + "\n")
        private.chmod(0o600)
        self.environment["GIT_SSH_COMMAND"] = shlex.join([ssh, "-F", str(private)])
        self.environment["GIT_SSH_VARIANT"] = "ssh"

    def copy_local_git(self, source, bare):
        """Copy only regular Git storage, never its config, hooks or alternates."""
        source = source / ".git" if os.path.lexists(source / ".git") else source
        if not source.is_dir() or (source / "commondir").exists():
            raise AcquisitionError("acquisition_local_git_unsupported")
        count = total = 0
        with package_files._directory_fd(source) as root_fd:

            def copy(member, destination):
                nonlocal count, total
                count += 1
                if count > package_files.MAX_FILES * 4:
                    raise AcquisitionError("acquisition_local_git_limit")
                self.quota()
                path = package_files.validate_relative_member(member)
                info = os.stat(member, dir_fd=root_fd, follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    destination.mkdir(mode=0o700, exist_ok=True)
                    with (
                        package_files._directory_fd(source / member) as fd,
                        os.scandir(fd) as entries,
                    ):
                        before = os.fstat(fd)
                        if package_files._unchanged(info) != package_files._unchanged(
                            before
                        ):
                            raise AcquisitionError("acquisition_source_changed")
                        for entry in entries:
                            copy(member + "/" + entry.name, destination / entry.name)
                        if package_files._unchanged(before) != package_files._unchanged(
                            os.fstat(fd)
                        ):
                            raise AcquisitionError("acquisition_source_changed")
                elif stat.S_ISREG(info.st_mode):
                    if member in {
                        "objects/info/alternates",
                        "objects/info/http-alternates",
                    }:
                        raise AcquisitionError("acquisition_local_git_unsupported")
                    with package_files._member_fd(root_fd, path) as fd:
                        before = os.fstat(fd)
                        if package_files._unchanged(info) != package_files._unchanged(
                            before
                        ):
                            raise AcquisitionError("acquisition_source_changed")
                        with destination.open("wb") as output:
                            while block := os.read(fd, package_files.CHUNK_BYTES):
                                total += len(block)
                                if total > MAX_STAGING_BYTES:
                                    raise AcquisitionError("acquisition_staging_limit")
                                self.quota()
                                output.write(block)
                        if package_files._unchanged(before) != package_files._unchanged(
                            os.fstat(fd)
                        ):
                            raise AcquisitionError("acquisition_source_changed")
                else:
                    raise AcquisitionError("acquisition_local_git_unsupported")

            for member in ("objects", "refs", "HEAD", "packed-refs"):
                if os.path.lexists(source / member):
                    copy(member, bare / member)

    async def git_tree(self, source):
        bare = self.staging / "repository.git"
        await self.git_command("init", "--bare", "--template=", str(bare), limit=65536)
        if source.transport == "local":
            await self.worker(self.copy_local_git, Path(source.repository), bare)
            ref = source.ref
        else:
            await self.host_authentication(source)
            await self.git_command(
                "--git-dir=" + str(bare),
                "fetch",
                "--no-tags",
                "--no-recurse-submodules",
                "--depth=1",
                "--",
                source.url,
                source.ref,
                limit=65536,
            )
            ref = "FETCH_HEAD"
        prefix = "--git-dir=" + str(bare)
        commit = (
            (
                await self.git_command(
                    prefix,
                    "rev-parse",
                    "--verify",
                    "--end-of-options",
                    ref + "^{commit}",
                    limit=256,
                )
            )
            .decode()
            .strip()
        )
        if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit):
            raise AcquisitionError("acquisition_commit_invalid")

        async def verified_object(kind, oid):
            data = await self.git_command(
                prefix, "cat-file", kind, oid, limit=package_files.MAX_FILE_BYTES
            )
            digest = hashlib.new("sha1" if len(oid) == 40 else "sha256")
            digest.update(f"{kind} {len(data)}\0".encode())
            digest.update(data)
            if digest.hexdigest() != oid:
                raise AcquisitionError("acquisition_object_identity")
            return data

        commit_data = await verified_object("commit", commit)
        tree_line = commit_data.split(b"\n", 1)[0]
        if not re.fullmatch(rb"tree [0-9a-f]{40}|tree [0-9a-f]{64}", tree_line):
            raise AcquisitionError("acquisition_commit_invalid")
        tree = tree_line[5:].decode("ascii")
        tree_data = await verified_object("tree", tree)
        # Resolve each path component only inside a content-address verified parent.
        for component in (source.subdirectory or "").split("/"):
            if not component:
                continue
            offset, selected = 0, None
            while offset < len(tree_data):
                self.checkpoint()
                end = tree_data.find(b"\0", offset)
                width = len(tree) // 2
                if end < 0 or end + 1 + width > len(tree_data):
                    raise AcquisitionError("acquisition_tree_invalid")
                mode, separator, name = tree_data[offset:end].partition(b" ")
                oid = tree_data[end + 1 : end + 1 + width].hex()
                if not separator:
                    raise AcquisitionError("acquisition_tree_invalid")
                if name == component.encode("utf-8"):
                    if selected is not None or mode not in {b"40000", b"040000"}:
                        raise AcquisitionError("acquisition_subdirectory_invalid")
                    selected = oid
                offset = end + 1 + width
            if selected is None:
                raise AcquisitionError("acquisition_subdirectory_invalid")
            tree = selected
            tree_data = await verified_object("tree", tree)
        listing = await self.git_command(
            prefix, "ls-tree", "-r", "-t", "-z", "-l", tree, limit=16 * 1024 * 1024
        )
        members, seen, total = {}, {}, 0
        entries_seen = set()
        for row in listing.split(b"\0"):
            self.checkpoint()
            if not row:
                continue
            header, raw_name = row.split(b"\t", 1)
            mode, kind, oid, size = header.decode().split()
            name = raw_name.decode("utf-8")
            if name in entries_seen:
                raise AcquisitionError("package_path_collision")
            entries_seen.add(name)
            member = package_files.validate_relative_member(name)
            if member.as_posix() != name or len(members) >= package_files.MAX_FILES:
                raise AcquisitionError("package_file_count_limit")
            for path in [member, *member.parents[:-1]]:
                key = package_files.collision_key(path.as_posix())
                existing = seen.get(key)
                if existing is not None and existing != path.as_posix():
                    raise AcquisitionError("package_path_collision")
                seen[key] = path.as_posix()
            if kind == "tree" and mode == "040000":
                await verified_object("tree", oid)
                continue
            if mode == "160000" or kind != "blob":
                raise AcquisitionError("acquisition_submodule_unsupported")
            if mode not in {"100644", "100755", "120000"}:
                raise AcquisitionError("package_special_file")
            size = int(size)
            total += size
            if (
                size > package_files.MAX_FILE_BYTES
                or total > package_files.MAX_PACKAGE_BYTES
            ):
                raise AcquisitionError("package_bytes_limit")
            members[name] = (mode, oid, size)
        extracted = self.staging / "content"
        extracted.mkdir(mode=0o700)
        links = {}
        for name, (mode, oid, size) in members.items():
            data = await verified_object("blob", oid)
            if len(data) != size:
                raise AcquisitionError("acquisition_object_changed")
            if data.startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise AcquisitionError("acquisition_lfs_unsupported")
            if mode == "120000":
                links[name] = data.decode("utf-8")
                continue
            output = extracted / name
            await self.worker(self.write_blob, output, data, mode)
        resolved = {}
        expanded = sum(size for mode, _, size in members.values() if mode != "120000")
        for name in links:
            target, visited = name, set()
            while target in links:
                self.checkpoint()
                if target in visited or len(visited) >= package_files.MAX_PATH_DEPTH:
                    raise AcquisitionError("package_link_target_type")
                visited.add(target)
                parts = list(Path(target).parent.parts)
                value = links[target]
                if value.startswith("/") or "\\" in value:
                    raise AcquisitionError("package_path_escape")
                for part in value.split("/"):
                    if part == "..":
                        if not parts:
                            raise AcquisitionError("package_path_escape")
                        parts.pop()
                    elif part not in {"", "."}:
                        parts.append(part)
                target = "/".join(parts)
            if target not in members:
                raise AcquisitionError("package_link_target_type")
            expanded += members[target][2]
            if expanded > package_files.MAX_PACKAGE_BYTES:
                raise AcquisitionError("package_bytes_limit")
            output = extracted / name
            data = await self.worker((extracted / target).read_bytes)
            await self.worker(self.write_blob, output, data, members[target][0])
            resolved[name] = target
        return extracted, commit, tree, resolved

    async def acquire(self, source, destination):
        commit = tree = None
        accepted = False
        try:
            self.checkpoint()
            root = Path(source.repository)
            links = {}
            if (
                source.transport != "local"
                or os.path.lexists(root / ".git")
                or ((root / "HEAD").is_file() and (root / "objects").is_dir())
            ):
                root, commit, tree, links = await self.git_tree(source)
            elif source.subdirectory is not None:
                root /= source.subdirectory
            inspection = await self.worker(
                package_files.materialize_package,
                root,
                destination,
                checkpoint=self.quota,
            )
            await self.worker(self.quota, force=True)
            self.checkpoint()
            if links:
                inspection = inspection.model_copy(
                    update={
                        "link_targets": MappingProxyType(links),
                        "source_digest": hashlib.sha256(
                            package_files.canonical_json(
                                {
                                    "content": inspection.content_digest,
                                    "links": links,
                                }
                            ).encode()
                        ).hexdigest(),
                    }
                )
            snapshot = AcquiredSnapshot(
                operation_id=self.record["operation_id"],
                acquisition_nonce=self.record["nonce"],
                transport=source.transport,
                origin=source.origin,
                repository=source.repository,
                ref=source.ref,
                subdirectory=source.subdirectory,
                commit=commit,
                tree=tree,
                materialized_identity=str(destination),
                inspection=inspection,
                diagnostics=inspection.diagnostics,
            )
            accepted = True
            return snapshot
        finally:
            if self.workers:
                self.cancelled.set()
                for worker in tuple(self.workers):
                    while not worker.done():
                        try:
                            await asyncio.shield(worker)
                        except asyncio.CancelledError:
                            continue
                        # Joining preserves the original cancellation.
                        except Exception:  # noqa: BLE001
                            break
                    if not worker.cancelled():
                        worker.exception()
                    self.workers.discard(worker)
            _qualify(self.owner, self.record)
            self.record["phase"] = "terminal"
            self.record["recovery_referenced"] = accepted
            _save(self.owner, self.authority, self.record)


async def acquire_snapshot(
    source: SourceLocator, operation_id: str, destination: Path, *, owner, authority
) -> AcquiredSnapshot:
    """Acquire on the existing storage worker, retaining cleanup after cancellation."""
    started = time.monotonic()
    if not isinstance(source, SourceLocator):
        raise AcquisitionError("acquisition_source_invalid")
    if source.transport == "local":
        validated = select_local_source(
            Path(source.repository), ref=source.ref, subdirectory=source.subdirectory
        )
    else:
        selectors = {"ref": source.ref}
        if source.subdirectory is not None:
            selectors["subdir"] = source.subdirectory
        validated = parse_source_locator(source.url + "#" + urlencode(selectors))
    if source != validated:
        raise AcquisitionError("acquisition_source_invalid")
    owner.require_owner(owner.root)
    record = _reserve(owner, authority, operation_id, destination)
    cancelled = threading.Event()
    producer = _Producer(owner, authority, record, cancelled)
    producer.deadline = started + ACQUISITION_SECONDS
    task = asyncio.create_task(producer.acquire(source, destination))
    owner.acquisition_tasks.add(task)

    def completed(done):
        owner.acquisition_tasks.discard(done)
        if not done.cancelled():
            done.exception()

    task.add_done_callback(completed)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        cancelled.set()
        raise
