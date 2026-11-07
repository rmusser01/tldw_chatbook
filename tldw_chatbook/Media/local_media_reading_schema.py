"""Local reading auxiliary SQLite declarations."""

_LOCAL_READING_AUX_SCHEMA: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS local_reading_saved_searches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        query_json TEXT NOT NULL DEFAULT '{}',
        sort TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS local_reading_note_links (
        item_id INTEGER NOT NULL,
        note_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (item_id, note_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS local_reading_archives (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        item_id INTEGER NOT NULL,
        title TEXT NOT NULL,
        format TEXT NOT NULL,
        source TEXT NOT NULL,
        storage_path TEXT NOT NULL,
        content TEXT NOT NULL,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        retention_until TEXT,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS local_reading_highlights (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        item_id INTEGER NOT NULL,
        quote TEXT NOT NULL,
        start_offset INTEGER,
        end_offset INTEGER,
        color TEXT,
        note TEXT,
        anchor_strategy TEXT NOT NULL DEFAULT 'fuzzy_quote',
        state TEXT NOT NULL DEFAULT 'active',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_local_reading_highlights_item_id
        ON local_reading_highlights(item_id)
    """,
    """
    CREATE TABLE IF NOT EXISTS local_document_annotations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        media_id INTEGER NOT NULL,
        location TEXT NOT NULL,
        text TEXT NOT NULL,
        color TEXT NOT NULL DEFAULT 'yellow',
        note TEXT,
        annotation_type TEXT NOT NULL DEFAULT 'highlight',
        chapter_title TEXT,
        percentage REAL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_local_document_annotations_media_id
        ON local_document_annotations(media_id)
    """,
    """
    CREATE TABLE IF NOT EXISTS local_reading_digest_schedules (
        id TEXT PRIMARY KEY,
        name TEXT,
        cron TEXT NOT NULL,
        timezone TEXT NOT NULL DEFAULT 'UTC',
        enabled INTEGER NOT NULL DEFAULT 1,
        require_online INTEGER NOT NULL DEFAULT 0,
        format TEXT NOT NULL DEFAULT 'md',
        template_id INTEGER,
        template_name TEXT,
        retention_days INTEGER,
        filters_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS local_reading_digest_outputs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        schedule_id TEXT,
        title TEXT NOT NULL,
        format TEXT NOT NULL DEFAULT 'md',
        storage_path TEXT,
        content TEXT,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        FOREIGN KEY (schedule_id) REFERENCES local_reading_digest_schedules(id) ON DELETE SET NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_local_reading_digest_outputs_schedule_id
        ON local_reading_digest_outputs(schedule_id)
    """,
    """
    CREATE TABLE IF NOT EXISTS local_file_artifacts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_type TEXT NOT NULL,
        title TEXT,
        payload_json TEXT NOT NULL DEFAULT '{}',
        validation_json TEXT NOT NULL DEFAULT '{}',
        export_json TEXT NOT NULL DEFAULT '{}',
        options_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        deleted INTEGER NOT NULL DEFAULT 0,
        deleted_at TEXT
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_local_file_artifacts_type_deleted
        ON local_file_artifacts(file_type, deleted)
    """,
)
