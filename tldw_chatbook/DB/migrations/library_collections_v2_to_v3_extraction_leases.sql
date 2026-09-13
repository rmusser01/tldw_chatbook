ALTER TABLE collection_capture_items
ADD COLUMN extraction_owner_token TEXT;

ALTER TABLE collection_capture_items
ADD COLUMN extraction_lease_expires_at TEXT;
