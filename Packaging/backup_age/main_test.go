package main

import (
	"bytes"
	"encoding/binary"
	"io"
	"strings"
	"testing"
)

func TestPasswordAdmission(t *testing.T) {
	for _, n := range []uint32{0, 4097, 0xffffffff} {
		var in bytes.Buffer
		binary.Write(&in, binary.BigEndian, n)
		if _, err := readPassword(&in); err == nil {
			t.Fatalf("accepted length %d", n)
		}
	}
	if _, err := readPassword(bytes.NewReader([]byte{0, 0, 0, 3, 'a'})); err == nil {
		t.Fatal("accepted truncated password")
	}
}

func TestHeaderAdmission(t *testing.T) {
	prefix := "age-encryption.org/v1\n-> scrypt AAAAAAAAAAAAAAAAAAAAAA "
	for _, input := range []string{
		prefix + "19\nAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n--- AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n",
		prefix + "18\nAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n-> scrypt AAAAAAAAAAAAAAAAAAAAAA 18\nAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n--- AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n",
		"age-encryption.org/v1\n-> plugin bad\n--- bad\n",
		"age-encryption.org/v1\n" + strings.Repeat("a", 65536),
		"bad header\n", prefix + "018\n", prefix + "0\n",
	} {
		if _, err := admitHeader(strings.NewReader(input)); err == nil {
			t.Fatal("accepted hostile header")
		}
	}
	header := prefix + "18\nAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n--- AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n"
	r, err := admitHeader(strings.NewReader(header + "payload"))
	if err != nil {
		t.Fatal(err)
	}
	got, _ := io.ReadAll(r)
	if string(got) != header+"payload" {
		t.Fatal("header gate lost stream bytes")
	}
}

func TestByteBudgets(t *testing.T) {
	r := &budgetReader{strings.NewReader("12345"), 4}
	if _, err := io.ReadAll(r); err == nil {
		t.Fatal("accepted excessive input")
	}
	var out bytes.Buffer
	w := &budgetWriter{&out, 4}
	if _, err := w.Write([]byte("12345")); err == nil {
		t.Fatal("accepted excessive output")
	}
	if out.Len() != 0 {
		t.Fatal("published over-budget bytes")
	}
}

type failedWriter struct {
	prefix    bytes.Buffer
	nonceSeen bool
}

func (w *failedWriter) Write(p []byte) (int, error) {
	if w.nonceSeen {
		return 0, io.ErrClosedPipe
	}
	if bytes.Contains(w.prefix.Bytes(), []byte("\n--- ")) && len(p) == 16 {
		w.nonceSeen = true
	}
	return w.prefix.Write(p)
}
func TestWriterFinalizationFailure(t *testing.T) {
	var password bytes.Buffer
	binary.Write(&password, binary.BigEndian, uint32(4))
	password.WriteString("test")
	writer := &failedWriter{}
	if err := run("encrypt", &password, writer); err == nil {
		t.Fatal("ignored finalization failure")
	}
	if !writer.nonceSeen {
		t.Fatal("failure occurred before finalization")
	}
}
