# Logs — Application logs and diagnostics

> 🚧 **This page is a stub.** The full write-up is planned; the sections
> below cover orientation only. See the [guide index](index.md).

## What this screen is for

Logs shows application logs and diagnostics (on-screen subtitle:
"Application logs and diagnostics.").

## Getting there

- Press **F3**, click **F3 Logs** in the nav bar, or press **Ctrl+P** →
  "Tab Navigation: Switch to Logs".
  There is no hotkey digit for Logs.

## Sharing logs safely

Both copy actions preserve diagnostic text, with credentials and recognizable
personally identifiable information (PII) masked:

- **Copy visible logs** (`y`) copies the lines the current filter matches —
  including message text and exception details.
- **Copy all (redacted)** copies all retained session logs, including messages,
  exception details, timestamps, logger names and levels.

What is removed, on screen and on the clipboard alike:

- credentials in **recognised** formats — `Bearer` prefixes, URL userinfo,
  `api_key=`/`x-auth-token:`-style labelled values, and the standalone key
  shapes the redactor knows (OpenAI, Anthropic, OpenRouter, Google, GitHub,
  Hugging Face, AWS, Slack, JWTs);
- your operating-system account name — home paths are shown as `~/…`;
- recognizable email addresses, phone numbers and SSNs, plus labelled personal
  fields such as names, usernames and postal addresses;
- private-key blocks, authentication cookies and credential fields in database
  connection strings. Server names and database names remain readable.

This is a denylist, so it is **not** a promise that every secret is caught: a
credential in a format it does not recognise, with no `key=`-style label
beside it, will pass through. Treat the list above as "the common shapes are
handled", not "nothing can leak".

Lines longer than 2,000 characters are shortened before anything is stored,
so a dumped response body is never retained whole. The cut lands on a word
boundary and the redactor sees everything that survives it, so a key sitting
across the limit is dropped rather than half-shown. A single unbroken run of
more than 2,000 characters — one enormous token with no spaces in it — is
withheld entirely for the same reason.

Ordinary diagnostic text is retained, including provider/model names, versions,
phases, correlation IDs, file names and non-secret keys. Message text is not
removed wholesale just because it contains a title, search term, prompt or
provider response. Pattern matching cannot identify every personal detail in
prose, so review what you copied before posting a bug report.

The rotating application log file uses the same credential/PII redaction policy
under [ADR-029](../../backlog/decisions/029-local-private-data-boundary.md).
Existing retention limits and private-file permission checks still apply.
