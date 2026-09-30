# Buddy qualification after SSH session dev rebase

Sanitized source-bound receipt. Raw logs and captures remain local. Native terminal and physical voice acceptance remains open.

```json
{
  "tested_source": "a242b1b82d69c0eb1aeb74a9ab85583ddffc8bfc",
  "dev_base": "856ffd9962253743173e4ec92c4cdbc609473bda",
  "previous_head": "3d53e78ff01d0bd4ff44094988afba7783ce738b",
  "previous_dev_base": "75c06af39a07154560ab1db49561a620a9689e72",
  "incoming_pr": 2907,
  "range_diff": "All twelve preceding PR commits remain patch-identical; no conflicts.",
  "distinct_tests_passed": 125,
  "incoming_ssh_cases_passed": 109,
  "buddy_cases_passed": 16,
  "elapsed_seconds": 145.97,
  "test_files": [
    "Tests/UI/test_buddy_v1_qualification_capture.py",
    "Tests/Tools/test_remote_executor_ssh.py",
    "Tests/Tools/test_remote_session_registry.py",
    "Tests/Tools/test_remote_session_worker.py"
  ],
  "ssh_evidence_limit": "Real bootstrap, loader and bundle run locally through existing spawn/fake-transport seams; no live remote-host acceptance claimed.",
  "timeout_seconds": 120,
  "invocation": "TMPDIR=/private/tmp, PYTHONDONTWRITEBYTECODE=1, PYTHONPATH=., existing Python 3.12.11 venv.",
  "scoped_ruff_and_format": "Three retained Buddy/private-profile/coverage files pass with --no-cache; initial sandbox cache write was denied before checks ran.",
  "initial_preflight": "Default selection chose Python 3.14.7; only Canvas Mermaid failed its pinned-build-version guard. Explicit Python 3.12.11 Mermaid check passes; no artifact or source regeneration.",
  "full_suite_run": false,
  "paid_provider_calls": false,
  "native_or_physical_voice_acceptance": false,
  "prior_evidence_attribution": "Original b007 coverage, MCP repetition/negative control and all earlier receipts retain original tested-source attribution. Passing 3d53 hosted gates are old-head results after rebase.",
  "log_sha256": "b170b6c7a784ee4d90f97dbe0daeaa158cb58987befb077265cbaac2acf0f0c6",
  "initial_preflight_log_sha256": "aee2825c049eaaf2cd0b04a1539334cad6fddbd42ebf54225478ab8e8dc9ac5f",
  "pinned_mermaid_log_sha256": "7d735cc6cda40478416e3df9eda1dc7919845f1109841377210d3f7eda661b73",
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
          "buddy_owner": "b341fa49-c552-4d92-9391-2c93e3bfe339",
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
          "buddy_owner": "b341fa49-c552-4d92-9391-2c93e3bfe339",
          "persona_owner": null
        }
      },
      "workspace_explicit_none": true,
      "workspace_persona_before_clear": "qualification-workspace-persona",
      "workspace_persona_after_clear": null,
      "workspace_explicit_none_before": false,
      "workspace_explicit_none_after": true,
      "conversation_id": "88f59a5d-45c8-4048-a4d5-476d65369ae2",
      "schema_version": 73,
      "persona_count_unchanged_during_artwork_selection": true,
      "source_commit": "a242b1b82d69c0eb1aeb74a9ab85583ddffc8bfc",
      "harness_sha256": "667ccdeeee287325c41318bdbd46fb60b4a66b3e103487ef3b0190471c30cad1",
      "captures": {
        "fresh-static-management.svg": "21037f69a638b74ca2dc9e3068965e8b737032a07e79c1064c1f285c0a7dfef7",
        "fresh-static-home.svg": "37d7d673d76f86b40c5cabb71fc67243f63eae135919524d3bb84db7e50ca957",
        "fresh-dynamic-management.svg": "439876168f9cb6452f7bdcdda378e35a313eef2395d127f49ec25f67be5b6bc6",
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
          "buddy_owner": "024e8bc5-73fc-49e1-957e-d73f8a88654f",
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
          "buddy_owner": "024e8bc5-73fc-49e1-957e-d73f8a88654f",
          "persona_owner": null
        }
      },
      "workspace_explicit_none": true,
      "workspace_persona_before_clear": "qualification-workspace-persona",
      "workspace_persona_after_clear": null,
      "workspace_explicit_none_before": false,
      "workspace_explicit_none_after": true,
      "conversation_id": "f197b738-7d91-4324-be52-0888acadb949",
      "schema_version": 73,
      "persona_count_unchanged_during_artwork_selection": true,
      "source_commit": "a242b1b82d69c0eb1aeb74a9ab85583ddffc8bfc",
      "harness_sha256": "667ccdeeee287325c41318bdbd46fb60b4a66b3e103487ef3b0190471c30cad1",
      "captures": {
        "upgrade-static-management.svg": "b6934a0b7d5e1814a2a95208a4978f9c553d4cb5d51ec32347bc690caf4576b4",
        "upgrade-static-home.svg": "37d7d673d76f86b40c5cabb71fc67243f63eae135919524d3bb84db7e50ca957",
        "upgrade-dynamic-management.svg": "8e5cc2a6a1669cdcb90fc681ec74d6e7ebd30ff87854f8c03b810e05418d5e36",
        "upgrade-dynamic-home.svg": "37d7d673d76f86b40c5cabb71fc67243f63eae135919524d3bb84db7e50ca957",
        "upgrade-conversation.svg": "d742184e75b3673d6965b9784ec3c151ac16b4901c3e2005a5becae03d55a375",
        "upgrade-workspace-management.svg": "e19dea269b437681fc979e0b58b5c2b92ecc9f280bde236c695c2f7375f79263",
        "upgrade-workspace-inbox.svg": "4c7ee28e461fcaec0b0150ac0b7b708d4462a89d8338803dc36d7eab49af5192"
      }
    }
  },
  "preflight_guards_passed": 11,
  "preflight_python_version": "3.12.11",
  "preflight_log_sha256": "108d52836e34466f6cf4787547261d9ce6f7a5ad491c2115ce6f5394be7a4ed6"
}
```
