# Logs — Application logs and diagnostics

> 🚧 **This page is a stub.** The full write-up is planned; the sections
> below cover orientation only. See the [guide index](index.md).

## What this screen is for

Logs shows application logs and diagnostics (on-screen subtitle:
"Application logs and diagnostics.").

## Getting there

- Press **F8**, click **F8 Logs** in the nav bar, or press **Ctrl+P** →
  "Tab Navigation: Switch to Logs".
  There is no hotkey digit for Logs.

## Sharing logs safely

Two copy actions sit at the bottom of the screen, and they hand you
deliberately different things (TASK-19555):

- **Copy visible logs** (`y`) copies the lines the current filter matches —
  the real log text, so it is what you want when someone is helping you
  debug. You can read it before you send it, which is the point.
- **Copy all (redacted)** copies the whole session as timestamps, logger
  names, levels and exception types, with the message bodies removed. It
  exports thousands of lines you have never read, so it deliberately carries
  no log text.

What is removed, on screen and on the clipboard alike:

- credentials in **recognised** formats — `Bearer` prefixes, URL userinfo,
  `api_key=`/`x-auth-token:`-style labelled values, and the standalone key
  shapes the redactor knows (OpenAI, Anthropic, OpenRouter, Google, GitHub,
  Hugging Face, AWS, Slack, JWTs);
- your operating-system account name — home paths are shown as `~/…`.

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

What is **not** removed from the visible log, and therefore not from
**Copy visible logs**: file names, note titles, keywords, search terms,
prompts, tool arguments, and provider response text. Read what you copied
before you post it in a bug report.

The rotating log file on disk (see the paths in the feature docs) is
narrower still: under [ADR-029](../../backlog/decisions/029-local-private-data-boundary.md)
it is metadata-only, so it holds operational events rather than log text.
