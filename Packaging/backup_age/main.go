// Command backup-age implements the private, pipe-only recovery protocol.
package main

import (
	"bufio"
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"runtime"
	"strconv"
	"strings"

	"filippo.io/age"
)

const maxHeader = 64 * 1024
const maxPassword = 4096
const maxContainer int64 = 2 * 1024 * 1024 * 1024 * 1024

var errProtocol = errors.New("protocol_error")
var errHeader = errors.New("invalid_header")
var errLimit = errors.New("byte_limit")

func readPassword(in io.Reader) (string, error) {
	var size uint32
	if binary.Read(in, binary.BigEndian, &size) != nil || size == 0 || size > maxPassword {
		return "", errProtocol
	}
	password := make([]byte, size)
	if _, err := io.ReadFull(in, password); err != nil {
		return "", errProtocol
	}
	return string(password), nil
}

// Admit the complete canonical single-scrypt header before any KDF allocation.
// The library still verifies header authentication and the complete payload.
func admitHeader(in io.Reader) (io.Reader, error) {
	buffered := bufio.NewReaderSize(in, 4096)
	var header bytes.Buffer
	line := func() (string, error) {
		var result []byte
		for {
			b, err := buffered.ReadByte()
			if err != nil || header.Len() >= maxHeader {
				return "", errHeader
			}
			header.WriteByte(b)
			result = append(result, b)
			if b == '\n' {
				return string(result), nil
			}
		}
	}
	magic, err := line()
	if err != nil || magic != "age-encryption.org/v1\n" {
		return nil, errHeader
	}
	stanza, err := line()
	if err != nil {
		return nil, errHeader
	}
	fields := strings.Split(strings.TrimSuffix(stanza, "\n"), " ")
	if len(fields) != 4 || fields[0] != "->" || fields[1] != "scrypt" {
		return nil, errHeader
	}
	salt, err := base64.RawStdEncoding.Strict().DecodeString(fields[2])
	if err != nil || len(salt) != 16 {
		return nil, errHeader
	}
	factor, err := strconv.Atoi(fields[3])
	if err != nil || factor < 1 || factor > 18 || strconv.Itoa(factor) != fields[3] {
		return nil, errHeader
	}
	body, err := line()
	if err != nil {
		return nil, errHeader
	}
	wrapped, err := base64.RawStdEncoding.Strict().DecodeString(strings.TrimSuffix(body, "\n"))
	if err != nil || len(wrapped) != 32 {
		return nil, errHeader
	}
	footer, err := line()
	if err != nil || !strings.HasPrefix(footer, "--- ") {
		return nil, errHeader
	}
	mac, err := base64.RawStdEncoding.Strict().DecodeString(strings.TrimSuffix(strings.TrimPrefix(footer, "--- "), "\n"))
	if err != nil || len(mac) != 32 {
		return nil, errHeader
	}
	return io.MultiReader(bytes.NewReader(header.Bytes()), buffered), nil
}

// Budget actual streamed bytes, including the byte probing for over-budget input.
type budgetReader struct {
	reader    io.Reader
	remaining int64
}

func (r *budgetReader) Read(p []byte) (int, error) {
	if int64(len(p)) > r.remaining+1 {
		p = p[:r.remaining+1]
	}
	n, err := r.reader.Read(p)
	if int64(n) > r.remaining {
		return 0, errLimit
	}
	r.remaining -= int64(n)
	return n, err
}

type budgetWriter struct {
	writer    io.Writer
	remaining int64
}

func (w *budgetWriter) Write(p []byte) (int, error) {
	if int64(len(p)) > w.remaining {
		return 0, errLimit
	}
	n, err := w.writer.Write(p)
	w.remaining -= int64(n)
	return n, err
}

func run(mode string, in io.Reader, out io.Writer) error {
	if mode == "info" {
		return json.NewEncoder(out).Encode(map[string]any{"protocol": 1, "helper_version": "1", "age_version": "v1.3.2", "os": runtime.GOOS, "arch": runtime.GOARCH})
	}
	if mode != "encrypt" && mode != "decrypt" {
		return errProtocol
	}
	password, err := readPassword(in)
	if err != nil {
		return err
	}
	input := &budgetReader{in, maxContainer}
	output := &budgetWriter{out, maxContainer}
	if mode == "decrypt" {
		admitted, err := admitHeader(input)
		if err != nil {
			return err
		}
		identity, err := age.NewScryptIdentity(password)
		if err != nil {
			return errProtocol
		}
		identity.SetMaxWorkFactor(18)
		reader, err := age.Decrypt(admitted, identity)
		if err != nil {
			return err
		}
		_, err = io.CopyBuffer(output, reader, make([]byte, 64*1024))
		return err // EOF authentication is mandatory, including the final chunk.
	}
	recipient, err := age.NewScryptRecipient(password)
	if err != nil {
		return errProtocol
	}
	recipient.SetWorkFactor(18)
	writer, err := age.Encrypt(output, recipient)
	if err != nil {
		return err
	}
	if _, err := io.CopyBuffer(writer, input, make([]byte, 64*1024)); err != nil {
		return err
	}
	return writer.Close()
}

func main() {
	// Panics must not print runtime diagnostics containing input or process state.
	defer func() {
		if recover() != nil {
			fmt.Fprintln(os.Stderr, "helper_failed")
			os.Exit(1)
		}
	}()
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "protocol_error")
		os.Exit(1)
	}
	if err := run(os.Args[1], os.Stdin, os.Stdout); err != nil {
		code := "transform_failed"
		for _, fixed := range []error{errProtocol, errHeader, errLimit} {
			if errors.Is(err, fixed) {
				code = fixed.Error()
			}
		}
		fmt.Fprintln(os.Stderr, code)
		os.Exit(1)
	}
}
