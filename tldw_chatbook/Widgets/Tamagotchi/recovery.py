"""The profile's ConfigFileStorage pet files; dormant custom stores have no locator.

TASK-42000: pets live beside the profile's config (``tamagotchi_pets.json``, its
dated copies and the seed-once marker), so they are captured and restored like
the profile's other settings files. The archive rows are unchanged, so a backup
taken while pets sat in the old shared folder restores into the profile too.
TASK-42114: the dated copies are the store's own rotating backups; they are
listed as intentionally excluded, so a backup neither captures them nor
registers them as roots. A restore skips the dated rows that archives made
before TASK-42114 hold (``rotating_copy``), so it never recreates them and
leaves a target's own copies as they are; the current pet file still restores.
"""

import re
from dataclasses import replace

from tldw_chatbook.Backup_Recovery.models import (
    OwnerAdapter,
    StorageItem,
    discovery_context,
    storage_logical_id,
)
from tldw_chatbook.Backup_Recovery.recovery_files import _RawDeclaration

_CURRENT = "tamagotchi_pets.json"
_DATED = re.compile(r"tamagotchi_pets\.backup_\d{8}_\d{6}\.json")
#: ConfigFileStorage's seed-once marker; its local ID is ``seeded``.
_MARKER = "tamagotchi_pets.seeded"


class _ConfiguredPets(_RawDeclaration):
    @staticmethod
    def rotating_copy(relative_path: str) -> bool:
        """The store's own dated copies, which a restore never recreates.

        TASK-42114 review: archives made before TASK-42114 hold them; a restore
        skips those rows, so it neither creates them nor collides with a
        target's own copies, which discovery lists as excluded.
        """
        return _DATED.fullmatch(relative_path) is not None

    def _restore_path(self, profile, payload, root, config_target):
        """Map only native pet files in their own top-level synthetic root."""
        key = f"profile:{profile}:tamagotchi.config"
        leaf = payload.relative_path
        canonical = leaf == _CURRENT and payload.logical_id == key
        backup = _DATED.fullmatch(leaf) and payload.logical_id == key + ":" + leaf
        marker = leaf == _MARKER and payload.logical_id == key + ":seeded"
        if not (
            self.owner_id == payload.owner_id == "tamagotchi.config"
            and (canonical or backup or marker)
            and root.synthetic
            and root.parent_id is None
            and root.relative_path == ""
            and root.logical_id == root.root_id == payload.root_id
            and payload.parent_id == payload.root_id
        ):
            raise ValueError("owner_relocation_unverified:tamagotchi.config")
        return config_target.parent / leaf

    def discover(self, config):
        from tldw_chatbook.Backup_Recovery.bootstrap import pinned_directory
        from tldw_chatbook.Backup_Recovery.mobile_preview_budget import read_options
        from tldw_chatbook.Utils.platform_files import os

        budget, custody = read_options()
        context = discovery_context(config)
        parent = context.config_path.parent
        if not os.path.lexists(parent):
            # No config folder yet, so no pet files: the same unused row an
            # absent pet file beside an existing config gets.
            return (
                StorageItem(
                    self.owner_id,
                    storage_logical_id(context, self.owner_id),
                    parent / _CURRENT,
                    "unused",
                    (storage_logical_id(context, "config"),),
                ),
            )
        items = [self._item(config, parent / _CURRENT)]
        try:
            pin = pinned_directory if custody is None else custody.pinned_directory
            with pin(parent) as fd:
                if budget is None:
                    names = os.listdir(fd)
                    if len(names) > 100000:
                        raise ValueError("pet_inventory_limit")
                else:
                    from tldw_chatbook.Backup_Recovery.bootstrap import _bounded_names

                    names = _bounded_names(fd, 100000, budget, custody)
                    budget.container(len(names))
                names.sort()
        except (OSError, ValueError, RuntimeError):
            return (
                StorageItem(
                    self.owner_id,
                    storage_logical_id(context, self.owner_id),
                    parent,
                    "unavailable",
                    (storage_logical_id(context, "config"),),
                ),
            )
        # Other files beside the config belong to other owners.
        for name in names:
            if _DATED.fullmatch(name):
                # TASK-42114: the store's own rotating copies are never
                # captured, so a backup never binds a root the store deletes.
                item = self._item(config, parent / name, name)
                items.append(
                    replace(item, status="intentionally_excluded")
                    if item.status == "included"
                    else item
                )
            elif name == _MARKER:
                items.append(self._item(config, parent / name, "seeded"))
        return tuple(items)


def recovery_adapters() -> tuple[OwnerAdapter, ...]:
    return (_ConfiguredPets("tamagotchi.config", "opaque", 16 * 1024**2),)
