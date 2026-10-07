# Buddy qualification after Console hook dev rebase

Sanitized source-bound receipt. Raw logs and captures remain local. Existing native and physical voice acceptance remains open.

```json
{
  "tested_source": "21eadf3b34e346b610ea1b44a868c5b2bc458bd3",
  "dev_base": "75c06af39a07154560ab1db49561a620a9689e72",
  "previous_head": "6540139b1693f73285869e62452215d09d9269af",
  "previous_dev_base": "dcd9f4e0223b739c18cf66a815472a481403757a",
  "incoming_pr": 2922,
  "range_diff": "Ten prior patches are identical. The final metadata patch retains all prior content, with append context moved after three new upstream lessons. All four prior metadata contents and all incoming lessons were verified unchanged.",
  "distinct_tests_passed": 227,
  "incoming_hook_runtime_config_ui_cases_passed": 211,
  "buddy_cases_passed": 16,
  "combined_run": {
    "passed": 225,
    "failed": 2,
    "elapsed_seconds": 127.57,
    "failure_reason": "Invocation used /private/tmp captures while inherited macOS TMPDIR selected another allowed root. Both confinement failures occurred before app construction; no code change.",
    "warning": "Session-level file-descriptor growth warning: 239 above start; no additional leak investigation claimed."
  },
  "corrected_buddy_run": {
    "passed": 16,
    "elapsed_seconds": 74.99,
    "invocation_change": "TMPDIR=/private/tmp; original timeout=120 and capture confinement unchanged."
  },
  "preflight_guards_passed": 11,
  "scoped_ruff_and_format": "Three retained Buddy/private-profile/coverage test files pass.",
  "python_version": "3.12.11",
  "pytest_version": "8.4.2",
  "full_suite_run": false,
  "paid_provider_calls": false,
  "native_or_physical_voice_acceptance": false,
  "prior_evidence_attribution": "b007 coverage and all earlier receipts retain original tested-source attribution. Passing 6540 hosted gates remain old-head results.",
  "log_sha256": "58439d89b98486b3681148e6ab4ebb5b5b9d827f7578ee14b27e3b1ddcec20f8",
  "corrected_profile_log_sha256": "528e5b3bb0349c05bc704ace5567e0a8a966077c9d4d0461839992420fb8b88f",
  "preflight_log_sha256": "108d52836e34466f6cf4787547261d9ce6f7a5ad491c2115ce6f5394be7a4ed6",
  "helper_sha256": "0e67da6536e5faf07a0ac4e2e33a2ab20107e59fd7bb65fc9a484e5a67f688e1",
  "regression_sha256": "da316bf1979e72ff53d906e9478e7e09964ef694e98337664b5435d28eccddaa",
  "profiles": {
    "fresh": {
      "profile": "fresh",
      "headless": true,
      "frames": {
        "static": {
          "frame_count": 1,
          "observed_indices": [
            0
          ],
          "sequence": [
            0
          ],
          "animate": false,
          "loop": true,
          "buddy_owner": "5cbabba9-cb30-4170-83d2-2013dbd2cc5f",
          "persona_owner": null
        },
        "dynamic": {
          "frame_count": 4,
          "observed_indices": [
            0,
            1,
            2,
            3
          ],
          "sequence": [
            0,
            1,
            2,
            3,
            0
          ],
          "animate": true,
          "loop": true,
          "buddy_owner": "5cbabba9-cb30-4170-83d2-2013dbd2cc5f",
          "persona_owner": null
        }
      },
      "workspace_explicit_none": true,
      "workspace_persona_before_clear": "qualification-workspace-persona",
      "workspace_persona_after_clear": null,
      "workspace_explicit_none_before": false,
      "workspace_explicit_none_after": true,
      "conversation_id": "3e863bb5-ec66-4a65-a17f-49f6e5997cc4",
      "schema_version": 73,
      "persona_count_unchanged_during_artwork_selection": true,
      "source_commit": "21eadf3b34e346b610ea1b44a868c5b2bc458bd3",
      "harness_sha256": "667ccdeeee287325c41318bdbd46fb60b4a66b3e103487ef3b0190471c30cad1",
      "captures": {
        "fresh-static-management.svg": "0378212904210f86b335574b1b4ddff7d4b0309446424bb35e7dfe9190647abc",
        "fresh-static-home.svg": "37d7d673d76f86b40c5cabb71fc67243f63eae135919524d3bb84db7e50ca957",
        "fresh-dynamic-management.svg": "ce87756bcb84093a0a76bd0a536a8f268afd5ae5350d740c6350f43e801150f7",
        "fresh-dynamic-home.svg": "37d7d673d76f86b40c5cabb71fc67243f63eae135919524d3bb84db7e50ca957",
        "fresh-conversation.svg": "d742184e75b3673d6965b9784ec3c151ac16b4901c3e2005a5becae03d55a375",
        "fresh-workspace-management.svg": "e19dea269b437681fc979e0b58b5c2b92ecc9f280bde236c695c2f7375f79263",
        "fresh-workspace-inbox.svg": "4c7ee28e461fcaec0b0150ac0b7b708d4462a89d8338803dc36d7eab49af5192"
      }
    },
    "upgrade": {
      "profile": "upgrade",
      "headless": true,
      "frames": {
        "static": {
          "frame_count": 1,
          "observed_indices": [
            0
          ],
          "sequence": [
            0
          ],
          "animate": false,
          "loop": true,
          "buddy_owner": "2bea661a-f9b9-4e47-95c3-964782f6f86d",
          "persona_owner": null
        },
        "dynamic": {
          "frame_count": 4,
          "observed_indices": [
            0,
            1,
            2,
            3
          ],
          "sequence": [
            0,
            1,
            2,
            3,
            0
          ],
          "animate": true,
          "loop": true,
          "buddy_owner": "2bea661a-f9b9-4e47-95c3-964782f6f86d",
          "persona_owner": null
        }
      },
      "workspace_explicit_none": true,
      "workspace_persona_before_clear": "qualification-workspace-persona",
      "workspace_persona_after_clear": null,
      "workspace_explicit_none_before": false,
      "workspace_explicit_none_after": true,
      "conversation_id": "b3064b8c-fd4f-43ee-b580-3812e26fb9b2",
      "schema_version": 73,
      "persona_count_unchanged_during_artwork_selection": true,
      "source_commit": "21eadf3b34e346b610ea1b44a868c5b2bc458bd3",
      "harness_sha256": "667ccdeeee287325c41318bdbd46fb60b4a66b3e103487ef3b0190471c30cad1",
      "captures": {
        "upgrade-static-management.svg": "c7ccf7cb9d23875a2ca490786fad70468ffdaf1409f3ec1afd9db6df3fe13cf8",
        "upgrade-static-home.svg": "37d7d673d76f86b40c5cabb71fc67243f63eae135919524d3bb84db7e50ca957",
        "upgrade-dynamic-management.svg": "d4b3e179c20b180315986be47e7edff9b6ba8876fd04af0e2c0f117c121605a2",
        "upgrade-dynamic-home.svg": "37d7d673d76f86b40c5cabb71fc67243f63eae135919524d3bb84db7e50ca957",
        "upgrade-conversation.svg": "d742184e75b3673d6965b9784ec3c151ac16b4901c3e2005a5becae03d55a375",
        "upgrade-workspace-management.svg": "e19dea269b437681fc979e0b58b5c2b92ecc9f280bde236c695c2f7375f79263",
        "upgrade-workspace-inbox.svg": "4c7ee28e461fcaec0b0150ac0b7b708d4462a89d8338803dc36d7eab49af5192"
      }
    }
  }
}
```
