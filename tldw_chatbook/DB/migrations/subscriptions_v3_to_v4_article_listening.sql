CREATE TABLE listening_session (
    id INTEGER PRIMARY KEY,
    revision INTEGER NOT NULL CHECK (revision >= 0),
    incarnation TEXT NOT NULL,
    profile TEXT NOT NULL,
    policy_epoch INTEGER NOT NULL,
    default_voice TEXT,
    brief_id INTEGER REFERENCES briefings(id),
    cursor INTEGER CHECK (cursor IS NULL OR (typeof(cursor)='integer' AND cursor BETWEEN 0 AND 255)),
    resume_queue INTEGER NOT NULL DEFAULT 0 CHECK (resume_queue IN (0,1)),
    transport TEXT NOT NULL CHECK (transport IN ('empty','paused','playing','preparing','completed')),
    playback_generation INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL DEFAULT 0,
    updated_at INTEGER NOT NULL DEFAULT 0,
    UNIQUE(profile)
);

CREATE TABLE article_audio (
    generation INTEGER PRIMARY KEY AUTOINCREMENT,
    file_key TEXT NOT NULL UNIQUE CHECK (
        length(file_key) = 36 AND substr(file_key,33) = '.wav'
        AND substr(file_key,1,32) NOT GLOB '*[^0-9a-f]*'
    ),
    item_id INTEGER REFERENCES subscription_items(id),
    profile TEXT NOT NULL,
    speech_hash TEXT,
    converter TEXT,
    voice_hash TEXT,
    dependency_hash TEXT,
    status TEXT NOT NULL CHECK (status IN ('pending','ready','missing','pending_delete')),
    sha256 TEXT,
    sample_rate INTEGER,
    content_frames INTEGER,
    file_frames INTEGER,
    byte_size INTEGER,
    last_used INTEGER,
    created_at INTEGER,
    target_entry TEXT,
    target_revision INTEGER,
    regenerate INTEGER
);

CREATE TABLE listening_entries (
    slot INTEGER NOT NULL CHECK (typeof(slot)='integer' AND slot BETWEEN -1 AND 255),
    session_id INTEGER NOT NULL REFERENCES listening_session(id),
    entry_id TEXT NOT NULL UNIQUE,
    item_id INTEGER REFERENCES subscription_items(id),
    source_id INTEGER REFERENCES subscriptions(id),
    title TEXT,
    text TEXT,
    speech_hash TEXT,
    converter TEXT,
    voice TEXT,
    voice_hash TEXT,
    dependency_hash TEXT,
    artifact INTEGER REFERENCES article_audio(generation),
    bookmark_hash TEXT,
    bookmark_rate INTEGER CHECK (bookmark_rate IS NULL OR typeof(bookmark_rate)='integer'),
    bookmark_frame INTEGER CHECK (bookmark_frame IS NULL OR (typeof(bookmark_frame)='integer' AND bookmark_frame >= 0)),
    unavailable_reason TEXT,
    CHECK ((item_id IS NULL AND unavailable_reason IS NOT NULL AND unavailable_reason IN ('missing_item','unsupported_content','empty_body'))
        OR (item_id IS NOT NULL AND unavailable_reason IS NULL)),
    UNIQUE(session_id,slot)
);

CREATE INDEX idx_article_audio_identity ON article_audio(item_id,profile,speech_hash,converter,voice_hash,dependency_hash,status);

CREATE INDEX idx_article_audio_lru ON article_audio(status,last_used,generation);

CREATE INDEX idx_listening_entries_artifact ON listening_entries(artifact);

CREATE INDEX idx_listening_entries_item ON listening_entries(item_id);

CREATE TRIGGER listening_item_delete BEFORE DELETE ON subscription_items
WHEN EXISTS (SELECT 1 FROM listening_entries WHERE item_id=OLD.id)
    OR EXISTS (SELECT 1 FROM article_audio WHERE item_id=OLD.id)
BEGIN
    SELECT listening_delete_guard();
    UPDATE listening_session SET revision=revision+1, playback_generation=playback_generation+1,
        updated_at=unixepoch(), transport=CASE WHEN transport='completed' THEN transport ELSE 'paused' END
        WHERE id IN (SELECT session_id FROM listening_entries WHERE item_id=OLD.id);
    UPDATE listening_entries SET item_id=NULL,source_id=NULL,title=NULL,text=NULL,speech_hash=NULL,
        converter=NULL,voice=NULL,voice_hash=NULL,dependency_hash=NULL,artifact=NULL,
        bookmark_hash=NULL,bookmark_rate=NULL,bookmark_frame=NULL,unavailable_reason='missing_item' WHERE item_id=OLD.id;
    UPDATE article_audio SET item_id=NULL,speech_hash=NULL,converter=NULL,voice_hash=NULL,
        dependency_hash=NULL,status='pending_delete',sha256=NULL,sample_rate=NULL,content_frames=NULL,
        file_frames=NULL,byte_size=NULL,last_used=NULL,created_at=NULL,target_entry=NULL,target_revision=NULL,
        regenerate=NULL WHERE item_id=OLD.id;
END;

CREATE TRIGGER listening_brief_delete BEFORE DELETE ON briefings
WHEN EXISTS (SELECT 1 FROM listening_session WHERE brief_id=OLD.id)
BEGIN
    SELECT listening_delete_guard();
    DELETE FROM listening_entries WHERE (slot >= 0 OR item_id IS NULL)
        AND session_id IN (SELECT id FROM listening_session WHERE brief_id=OLD.id);
    UPDATE listening_session SET brief_id=NULL,cursor=NULL,resume_queue=0,revision=revision+1,
        updated_at=unixepoch(), playback_generation=playback_generation+1,
        transport=CASE WHEN EXISTS (SELECT 1 FROM listening_entries WHERE session_id=listening_session.id) THEN 'paused' ELSE 'empty' END,
        default_voice=(SELECT voice FROM listening_entries WHERE slot=-1 AND session_id=listening_session.id)
        WHERE brief_id=OLD.id;
END;

DELETE FROM schema_version;

INSERT INTO schema_version(version) VALUES (4);
