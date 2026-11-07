"""Exact installed workflow identities shared by local and recovery readers."""

from typing import NamedTuple

from . import recovery_workflow_catalog_data as data


class _Profile(NamedTuple):
    version: int
    schema: tuple
    metadata: tuple
    legacy: bool
    stamps: tuple | None = None


_RECEIPTS = "console_activity_receipts"
_PROFILES = {}


def _install(family, *, legacy=False, full_stamps=False, degraded=()):
    schemas = []
    for number, (version, indices) in enumerate(family):
        objects = tuple(data._OBJECTS[index] for index in indices)
        rows = tuple(row for row, _ in objects)
        schema = tuple(row[3] for row in rows if row[3] is not None)
        schemas.append((version, schema))
        metadata = tuple(meta for _, meta in objects if meta is not None)
        stamps = tuple(range(4, 31)) if full_stamps else None
        profile = _Profile(version, schema, metadata, legacy, stamps)
        _PROFILES[rows] = profile
        if number in degraded:
            retained = tuple(
                (row, meta) for row, meta in objects if row[2] != _RECEIPTS
            )
            stamps = tuple(v for v in range(4, version + 1) if v != 15)
            _PROFILES[tuple(row for row, _ in retained)] = profile._replace(
                metadata=tuple(meta for _, meta in retained if meta is not None),
                stamps=stamps,
            )
    return tuple(schemas)


REVIEW_SCHEMAS = _install(data.REVIEW)
# Original v15 savepoint failure in the fresh v30 constructor (61) and
# the local v29 constructor (24), followed by its installed v30 bridge (32).
CANONICAL_SCHEMAS = _install(data.CANONICAL, degraded=(61, 32))
_install(data.LEGACY, legacy=True, degraded=(24,))
HISTORICAL_SCHEMAS = _install(data.HISTORICAL, full_stamps=True)


# Each pair records the actual SQLite catalog row and PRAGMA metadata. These
# installed descriptors can be checked even on a reader without ENABLE_STAT4.
def _stat_descriptor(name, columns):
    row = ("table", name, name, f"CREATE TABLE {name}({','.join(columns)})")
    metadata = (
        name,
        tuple((i, col, "", 0, None, 0, 0) for i, col in enumerate(columns)),
        (),
    )
    return row, metadata


_STATS = dict(
    (
        _stat_descriptor("sqlite_stat1", ("tbl", "idx", "stat")),
        _stat_descriptor(
            "sqlite_stat4", ("tbl", "idx", "neq", "nlt", "ndlt", "sample")
        ),
    )
)


def split_statistics(rows):
    """Separate only the complete engine-produced catalog forms."""
    stats = tuple(row for row in rows if row in _STATS)
    if len(stats) == 1 and stats[0][1] == "sqlite_stat4":
        raise ValueError("unsupported workflow schema catalog")
    return tuple(row for row in rows if row not in _STATS), stats


def expected_metadata(base, stats):
    if not stats:
        return base
    return tuple(sorted((*base, *(_STATS[row] for row in stats)), key=_metadata_order))


def _metadata_order(item):
    # _metadata follows catalog order (index, table, view), then object name.
    return len(item), item[0]


def identify(connection, base):
    """Read stamps only after the complete physical catalog has matched."""
    profile = _PROFILES.get(base)
    if profile is None:
        return None
    if profile.stamps is None:
        version = connection.execute(
            "SELECT MAX(version) FROM schema_version"
        ).fetchone()[0]
        valid = type(version) is int and version == profile.version
    else:
        rows = connection.execute(
            "SELECT version FROM schema_version ORDER BY version LIMIT ?",
            (len(profile.stamps) + 1,),
        )
        valid = tuple(row[0] for row in rows) == profile.stamps
    if not valid:
        raise ValueError("unsupported workflow schema version")
    return profile
