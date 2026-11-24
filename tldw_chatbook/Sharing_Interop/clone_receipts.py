"""Retain private lookup references for the Server's native clone operations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
from uuid import UUID, uuid4

from ..Utils.private_paths import (
    PrivateFileWritePrecondition,
    PrivatePathError,
    atomic_private_write_text,
    lexical_path,
    open_private_binary,
    secure_private_directory,
)


@dataclass(frozen=True, slots=True)
class SharingCloneReceipt:
    share_id: int
    idempotency_key: str
    operation_id: str = ""

    def __post_init__(self) -> None:
        if type(self.share_id) is not int or not 1 <= self.share_id < 2**63:
            raise ValueError("Choose a valid recipient share")
        if self.idempotency_key != str(UUID(self.idempotency_key)) or (
            self.operation_id and self.operation_id != str(UUID(self.operation_id))
        ):
            raise ValueError("The clone receipt has an invalid operation identity")


class SharingCloneReceipts:
    """Keep one caller key and owner operation ID per authenticated recipient share."""

    def __init__(
        self, directory: str | Path, profile_id: str, principal_id: str
    ) -> None:
        if not profile_id or not principal_id:
            raise ValueError("Clone receipts require a Server profile and account")
        scope = json.dumps((profile_id, principal_id), separators=(",", ":"))
        self.directory = (
            lexical_path(directory)
            / "sharing-clones"
            / sha256(scope.encode()).hexdigest()
        )

    def _file(self, share_id: int) -> Path:
        if type(share_id) is not int or not 1 <= share_id < 2**63:
            raise ValueError("Choose a valid recipient share")
        return self.directory / f"{share_id}.json"

    def _read(self, share_id: int):
        target = self._file(share_id)
        try:
            self.directory.lstat()
            with open_private_binary(target) as opened:
                expected = PrivateFileWritePrecondition.from_opened(opened)
                raw = opened.stream.read(1025)
        except FileNotFoundError:
            return None, PrivateFileWritePrecondition.missing()
        if len(raw) > 1024:
            raise ValueError("The saved clone receipt exceeds its size limit")
        try:
            receipt = SharingCloneReceipt(**json.loads(raw))
        except (TypeError, ValueError, AttributeError):
            raise ValueError("The saved clone receipt is invalid") from None
        if receipt.share_id != share_id:
            raise ValueError("The saved clone receipt belongs to a different share")
        return receipt, expected

    def get(self, share_id: int) -> SharingCloneReceipt | None:
        return self._read(share_id)[0]

    def list_receipts(self) -> dict[int, SharingCloneReceipt]:
        try:
            names = tuple(self.directory.iterdir())
        except FileNotFoundError:
            return {}
        receipts = {}
        for path in names:
            if path.suffix == ".json" and path.stem.isascii() and path.stem.isdecimal():
                share_id = int(path.stem)
                if 1 <= share_id < 2**63 and (receipt := self.get(share_id)):
                    receipts[share_id] = receipt
        return receipts

    def prepare(self, share_id: int) -> SharingCloneReceipt:
        return self._save(share_id)

    def remember(
        self, share_id: int, key: str, operation_id: str
    ) -> SharingCloneReceipt:
        if not key or not operation_id:
            raise ValueError("The Server did not confirm the clone operation")
        return self._save(share_id, key=key, operation_id=operation_id)

    def _save(
        self, share_id: int, *, key: str = "", operation_id: str = ""
    ) -> SharingCloneReceipt:
        target = self._file(share_id)
        secure_private_directory(self.directory, create=True, application_owned=True)
        for _ in range(3):
            previous, expected = self._read(share_id)
            if key:
                if previous is None or previous.idempotency_key != key:
                    raise ValueError("The receipt belongs to a different clone request")
                if previous.operation_id and previous.operation_id != operation_id:
                    raise ValueError("The Server returned a different clone operation")
                saved = replace(previous, operation_id=operation_id)
            else:
                saved = previous or SharingCloneReceipt(share_id, str(uuid4()))
            if saved == previous:
                return saved
            try:
                atomic_private_write_text(
                    target,
                    json.dumps(asdict(saved), separators=(",", ":")),
                    application_owned_directory=self.directory,
                    target_precondition=expected,
                )
            except PrivatePathError as exc:
                if exc.result.reason in {"target_appeared", "target_replaced"}:
                    continue
                raise
            return saved
        raise ValueError("The clone receipt changed. Reopen Sharing and retry")
