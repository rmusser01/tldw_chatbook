-- Legacy captures have no evidence of explicit user entry. Retry stays guarded.
ALTER TABLE collection_capture_items
ADD COLUMN url_provenance TEXT NOT NULL DEFAULT 'unknown'
CHECK(url_provenance IN ('unknown', 'user_entered'));
