# Prepared UniDic asset — venv installation deferred

Full UniDic `3.1.0+2021-08-31` is prepared in this task's `dicdir` directory. The archive is 525,943,040 bytes, SHA256 `638718c4c63625ab300de4c92c67925d54c0e9e3830009eaa992f29819d59c43`. The prepared dictionary is 811,662,881 bytes. No tokenizer, TTS model, playback, package installation, or venv link was run during preparation.

The installed `unidic` 1.1.0 downloader declares the package-maintainer [dictionary metadata](https://raw.githubusercontent.com/polm/unidic-py/31ba63c3ba2a1b95652bf322d182fbc0bc9f3c2b/dicts.json). That pinned commit selects this full version from the package's published AWS source. This is the dictionary supplied by the normal `python -m unidic download` route, with the package's documented changes from the original NINJAL release; it is not `unidic-lite`. The selected metadata, upstream README, license, commit response, response headers, exact download command, and local source-file hashes are retained here.

The download required the observed ETag `2a6b618247cacd507cd28e9c9d1254fe-32`; its response retained S3 object version `s1n.ag7lT3TyclL14vs8oJI7AMtkO.GQ`. Archive member paths, sizes, and CRCs are in `archive-inventory.json`. Extraction consumed each member completely, verified its CRC and size, and recorded its SHA256. Only the top-level directory name changed, and the exact `version` and dummy `mecabrc` contents from installed `unidic.download.download_and_clean` were added. Every original dictionary file remains; the archive itself is preserved.

`asset-provenance.json` contains hashes for the archive, upstream source records, every original and prepared dictionary file, and the inspected installed package files. `dicdir/licenses/` retains COPYING, AUTHORS, BSD, GPL, and LGPL; `upstream-LICENSE.unidic` is also retained. The BSD conditions are identical; the archive's copyright line ends in 2021 while the repository copy ends in 2017. Both originals remain unchanged. The public package distributes the dictionary under that BSD license.

Inspection found that Misaki 0.9.4's default `JAG2P` uses its Cutlet implementation, which calls `fugashi.Tagger()`; this path selects installed `unidic.DICDIR`. That directory was absent in the isolated Python 3.12 venv. The separately provisioned OpenJTalk dictionary does not supply this default Cutlet path. Preparation leaves the venv path absent and all inspected package hashes unchanged.

After the parent has joined the current matrix processes, run this exact deferred step:

```sh
/private/tmp/tts-macos-burndown/runtime-python312/bin/python -B /private/tmp/tts-macos-burndown/languages/unidic-3.1.0+2021-08-31/link_after_matrix.py
```

It creates only a new symlink at `/private/tmp/tts-macos-burndown/runtime-python312/lib/python3.12/site-packages/unidic/dicdir` pointing to this task's prepared `dicdir`. It refuses to replace any existing path and records `link-receipt.json`. It changes no global config and performs no inference. The link script has been parsed, but deliberately has not been executed.

Then a new process can check the actual default tokenizer path before a separately authorized Japanese synthesis retry:

```sh
/private/tmp/tts-macos-burndown/runtime-python312/bin/python -B - <<'PY'
from pathlib import Path
import unidic
from fugashi import Tagger

expected = Path('/private/tmp/tts-macos-burndown/languages/unidic-3.1.0+2021-08-31/dicdir')
assert Path(unidic.DICDIR).resolve() == expected
assert unidic.VERSION == 'unidic-3.1.0+2021-08-31'
tagger = Tagger()
words = list(tagger('静かな庭に朝の光が差します。'))
assert words
print(unidic.VERSION, unidic.DICDIR)
print([(word.surface, word.feature.pron) for word in words])
PY
```

This tokenizer check is a deferred command, not an observed success. `verification.json` records only archive, file, license, package-immutability, and absent-venv-link checks completed during preparation.
