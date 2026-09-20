# Qodo follow-up on PR2713

Qodo reviewed `8f481108897a899a5ab60639df0dfeb20b5633ea` and posted five
findings, all in QA runners or test contracts. Production/UI source is unchanged.

| Finding | Resolution |
| --- | --- |
| [Historical runner API contract](https://github.com/rmusser01/tldw_chatbook/pull/2713#discussion_r4057926535) | Retired the obsolete executable copy. Its immutable historical source remains linked beside the original evidence; the supported current runner has a typed, documented entry point. |
| [Lifecycle test contracts](https://github.com/rmusser01/tldw_chatbook/pull/2713#discussion_r4057926541) | Documented the service, lifecycle helpers and test cases; annotated parameters, return values and asynchronous test doubles. |
| [Fixture collision](https://github.com/rmusser01/tldw_chatbook/pull/2713#discussion_r4057926548) | Check the real store before saving. An occupied fixture ID raises before mutation. Cleanup uses only the successfully created ID and preserves other profiles. Three real-store cases cover collision, another profile and an empty store. |
| [Historical runner path validation](https://github.com/rmusser01/tldw_chatbook/pull/2713#discussion_r4057926521) | Retired that executable. The supported runner already uses the shared canonical-path and allowed-root validator before app imports or output creation. |
| [Historical runner tmux arguments](https://github.com/rmusser01/tldw_chatbook/pull/2713#discussion_r4057926528) | Retired that executable. The supported runner already uses centralized input validation; seven additional CLI matrix cases verify invalid input exits without touching the profile. |

The [collision baseline](qodo-fixture-red.txt) fails before the guard and the
[three store cases](qodo-fixture-green.txt) pass afterward. The complete
[targeted follow-up](qodo-targeted-green.txt) passes 35 cases: eleven repeated
lifecycle cases and 24 additional runner/input cases. Together with the existing
72 component cases, this is 96 distinct passes. No full suite ran.

Independent review found no remaining issues. The corrected runner then passed
the full four-cell native journey; all twelve captures were inspected again.
The final native source, lifecycle and export receipts live beside this file.
The previous native evidence remains available at commit `8f48110889`.

The first preflight retry could not fetch the pinned Mermaid input in the sandbox;
the same guard passed with authorized network access. No generated assets changed.
A native profile-preparation attempt omitted a required database directory and
was rejected before app imports; that failure is preserved separately and does
not count as successful qualification.

Owner visual approval is recorded. Current-head remote review/CI remain merge gates.
