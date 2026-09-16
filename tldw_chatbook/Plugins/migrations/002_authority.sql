CREATE TABLE revision_trust (
    installation_id TEXT NOT NULL,
    revision_digest TEXT NOT NULL,
    reviewed INTEGER NOT NULL CHECK (reviewed IN (0, 1)),
    PRIMARY KEY (installation_id, revision_digest),
    FOREIGN KEY (installation_id, revision_digest) REFERENCES revisions
);
CREATE TABLE tombstones (
    installation_id TEXT PRIMARY KEY NOT NULL,
    generation INTEGER NOT NULL CHECK (generation >= 0),
    operation_id TEXT NOT NULL
);
