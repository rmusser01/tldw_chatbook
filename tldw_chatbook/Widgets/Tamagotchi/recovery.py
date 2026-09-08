"""Exact optional ConfigFileStorage state; dormant custom stores have no locator."""

import os
import hashlib
from pathlib import Path
import re

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
