module tldw.chatbook/backup-age

go 1.26.0

require filippo.io/age v1.3.2

require (
	filippo.io/edwards25519 v1.2.0 // indirect
	filippo.io/hpke v0.4.0 // indirect
	filippo.io/nistec v0.0.4 // indirect
	golang.org/x/crypto v0.55.0 // indirect
	golang.org/x/sys v0.47.0 // indirect
	golang.org/x/term v0.45.0 // indirect
)

// Qualification-only interoperability command, pinned with the library.
tool filippo.io/age/cmd/age
