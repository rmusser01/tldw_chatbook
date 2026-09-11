"""Exact optional ConfigFileStorage state; dormant custom stores have no locator."""

import hashlib
import os
import re
from dataclasses import replace
from pathlib import Path

from tldw_chatbook.Backup_Recovery.models import (
    OwnerAdapter,
    StorageItem,
    discovery_context,
    storage_logical_id,
)
from tldw_chatbook.Backup_Recovery.recovery_files import _RawDeclaration


class _ConfiguredPets(_RawDeclaration):
    def discover(self, config):
        # ConfigFileStorage's actual platform/default app_name path. No override
        # is invented for dormant arbitrary-path JSONStorage/SQLiteStorage APIs.
        parent = (
            Path(os.environ.get("APPDATA", "~"))
            if os.name == "nt"
            else Path("~/.config")
        ).expanduser() / "tldw_chatbook"
        path = parent / "tamagotchi_pets.json"
        from tldw_chatbook.Backup_Recovery.file_inventory import _inventory_root

        # A checked absent canonical parent also proves its JSON/backups absent.
        # Inspect the parent itself first so links and unsafe traversal refuse.
        parent_item = _inventory_root(parent, owner=self.owner_id, external=False)
        if parent_item.status == "unavailable":
            container = _inventory_root(
                parent.parent, owner=self.owner_id, external=False
            )
            if container.status == "unused":
                parent_item = replace(parent_item, status="unused")
        if parent_item.status != "included_directory":
            context = discovery_context(config)
            return (
                replace(
                    parent_item,
                    path=path,
                    logical_id=storage_logical_id(context, self.owner_id),
                    status="unused"
                    if parent_item.status == "unused"
                    else "unavailable",
                    metadata=None,
                ),
            )
        items = [self._item(config, path)]
        if not parent.exists():
            return tuple(items)
        try:
            names = list(parent.iterdir())
            if len(names) > 100000:
                raise ValueError("pet_inventory_limit")
            for candidate in sorted(names):
                if re.fullmatch(
                    r"tamagotchi_pets\.backup_\d{8}_\d{6}\.json", candidate.name
                ):
                    items.append(self._item(config, candidate, candidate.name))
                elif candidate != path:
                    context = discovery_context(config)
                    items.append(
                        StorageItem(
                            self.owner_id,
                            storage_logical_id(
                                context,
                                self.owner_id,
                                "unknown-"
                                + hashlib.sha256(
                                    os.fsencode(candidate.name)
                                ).hexdigest(),
                            ),
                            candidate,
                            "unsupported",
                            (),
                        )
                    )
        except (OSError, ValueError):
            context = discovery_context(config)
            items.append(
                StorageItem(
                    self.owner_id,
                    storage_logical_id(context, self.owner_id, "unavailable"),
                    None,
                    "unavailable",
                    (),
                )
            )
        return tuple(items)


def recovery_adapters() -> tuple[OwnerAdapter, ...]:
    return (_ConfiguredPets("tamagotchi.config", "opaque", 16 * 1024**2),)
