"""Installed v1 recovered-media catalog migration and exact schema policy."""

from .models import SchemaPolicy

VERSION = 1
MIGRATION_1 = (
    "CREATE TABLE assets (asset_id TEXT PRIMARY KEY, digest TEXT NOT NULL, size INTEGER NOT NULL CHECK(size >= 0), media_type TEXT NOT NULL, state TEXT NOT NULL CHECK(state IN ('pending','ready','deleted')))",
    "CREATE TABLE refs (profile TEXT NOT NULL, message TEXT NOT NULL, slug TEXT NOT NULL, media_type TEXT NOT NULL, asset_id TEXT NOT NULL REFERENCES assets(asset_id), PRIMARY KEY(profile,message,slug,media_type))",
    "CREATE TABLE tombstones (asset_id TEXT PRIMARY KEY REFERENCES assets(asset_id), version INTEGER NOT NULL CHECK(version=1), references_json TEXT NOT NULL)",
    "CREATE TABLE operations (asset_id TEXT PRIMARY KEY REFERENCES assets(asset_id), kind TEXT NOT NULL CHECK(kind IN ('retain','delete')))",
    "CREATE TABLE recovery_holds (asset_id TEXT NOT NULL REFERENCES assets(asset_id), hold_id TEXT NOT NULL, PRIMARY KEY(asset_id,hold_id))",
)
SCHEMAS = ((1, tuple(sorted(MIGRATION_1, key=lambda sql: sql.split()[2]))),)


def schema_policy():
    """Return installed SQL, never SQL supplied by an imported catalog."""
    return SchemaPolicy("recovered.media", (VERSION,), SCHEMAS, ((0, 1, MIGRATION_1),))


def migrate(connection):
    """Create only the known empty-to-v1 layout; refuse unknown versions."""
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if version == 0:
        if connection.execute(
            "SELECT 1 FROM sqlite_schema WHERE sql IS NOT NULL"
        ).fetchone():
            raise ValueError("unsupported_recovered_schema")
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            for statement in MIGRATION_1:
                connection.execute(statement)
            connection.execute("PRAGMA user_version=1")
    elif version != VERSION:
        raise ValueError("unsupported_recovered_schema")
