"""The bake-off corpus: a fixed, listed subset of VoxConverse dev + AMI dev.

Both are CC-BY-4.0 with RTTM labels (spec §7). The ids below are FIXED -- the
report's numbers only mean something against a named corpus, and a changed
list silently invalidates every comparison in it.

Nothing here is downloaded at import. `materialise()` links (or copies) an
already-prepared corpus directory when one is given, and only downloads what
is missing otherwise. Files land under `Tests/Audio/bakeoff/data/`, which is
git-ignored: the audio is ~370 MB and is not ours to redistribute.

Every WAV must be mono 16 kHz 16-bit PCM -- what the meeting recorder writes
and the only thing `diarizer_worker.read_pcm16_span` accepts. Both
sources already are; `check_wav` refuses anything else rather than letting a
resampling bug show up as a diarization result.
"""
from __future__ import annotations

import os
import shutil
import wave
from dataclasses import dataclass
from pathlib import Path

#: VoxConverse dev ids: five files each at 2, 3, 4 and 5 speakers, spread over
#: 2-10 minutes. (id, speakers, seconds) as published:
#: wmori 2 120.1 - ngyrk 2 151.3 - zajzs 2 191.1 - qvtia 2 364.6 - zrlyl 2 567.2
#: jnivh 3 139.3 - ezsgk 3 208.1 - iwdjy 3 284.3 - mvjuk 3 375.0 - yrsve 3 587.0
#: ycxxe 4 142.4 - uvnmy 4 173.6 - azisu 4 194.1 - qfdpp 4 324.6 - mgpok 4 564.3
#: nnqfq 5 145.3 - kszpd 5 160.5 - vbjlx 5 323.7 - cmfyw 5 483.9 - zcdsd 5 577.8
VOXCONVERSE = [
    "wmori", "ngyrk", "zajzs", "qvtia", "zrlyl",
    "jnivh", "ezsgk", "iwdjy", "mvjuk", "yrsve",
    "ycxxe", "uvnmy", "azisu", "qfdpp", "mgpok",
    "nnqfq", "kszpd", "vbjlx", "cmfyw", "zcdsd",
]

#: AMI dev meetings (headset mix), for meeting acoustics. The pyannote dev
#: split is ES2011a-d / IB4001-4011 / IS1008a-d / TS3004a-d; these four are one
#: from each series.
AMI = ["ES2011a", "IS1008a", "TS3004a", "IB4001"]

_VOX_RTTM_URL = "https://raw.githubusercontent.com/joonson/voxconverse/master/dev/{id}.rttm"
#: One 1.9 GB zip for the whole dev set; only the listed ids are extracted.
_VOX_AUDIO_ZIP = "https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_dev_wav.zip"
_AMI_AUDIO_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{id}/audio/{id}.Mix-Headset.wav"
_AMI_RTTM_URL = "https://raw.githubusercontent.com/pyannote/AMI-diarization-setup/main/only_words/rttms/dev/{id}.rttm"

DATA_DIR = Path(__file__).resolve().parent / "data"
#: A directory holding `{voxconverse,ami}/<id>.wav|.rttm` already prepared
#: (16 kHz mono PCM16). Set it and nothing is downloaded.
SCRATCH_ENV = "TLDW_BAKEOFF_CORPUS"


@dataclass(frozen=True)
class CorpusFile:
    id: str
    source: str          # "voxconverse" | "ami"
    wav: Path
    rttm: Path

    @property
    def key(self) -> str:
        return f"{self.source}/{self.id}"


def listing() -> list[tuple[str, str]]:
    """`[(source, id), ...]` for the whole fixed corpus, VoxConverse first."""
    return [("voxconverse", i) for i in VOXCONVERSE] + [("ami", i) for i in AMI]


def read_rttm(path: Path) -> list[tuple[float, float, str]]:
    """RTTM -> `(start_s, end_s, speaker)`, SPEAKER lines only.

    Fields are `SPEAKER <file> <chan> <start> <dur> <NA> <NA> <spk> ...`; both
    corpora write turns, not frames, and both label overlapping speech.
    """
    out: list[tuple[float, float, str]] = []
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if len(fields) < 8 or fields[0] != "SPEAKER":
            continue
        start, duration = float(fields[3]), float(fields[4])
        if duration > 0:
            out.append((start, start + duration, fields[7]))
    return out


def check_wav(path: Path) -> float:
    """Duration in seconds; raises unless the file is mono 16 kHz 16-bit PCM."""
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2 or handle.getframerate() != 16000:
            raise ValueError(f"{path.name}: need mono 16 kHz 16-bit PCM (convert with ffmpeg)")
        return handle.getnframes() / 16000.0


def read_pcm(path: Path, start_s: float, end_s: float) -> bytes:
    """Raw PCM16 bytes for `[start_s, end_s)` of a checked WAV."""
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        total = handle.getnframes()
        a = min(total, max(0, round(start_s * rate)))
        b = min(total, max(a, round(end_s * rate)))
        handle.setpos(a)
        return handle.readframes(b - a)


def _link_or_copy(src: Path, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        dest.symlink_to(src)
    except OSError:  # Windows without developer mode, or a cross-device link
        shutil.copy2(src, dest)


def _download(url: str, dest: Path) -> None:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, open(tmp, "wb") as out:
        shutil.copyfileobj(response, out, 1 << 20)
    tmp.replace(dest)


def _fetch_voxconverse_audio(ids: list[str], dest_dir: Path) -> None:
    """Download the dev zip once and extract only `ids` (spec §7's fixed list)."""
    import zipfile

    zip_path = dest_dir / "voxconverse_dev_wav.zip"
    if not zip_path.is_file():
        _download(_VOX_AUDIO_ZIP, zip_path)
    wanted = {f"{i}.wav": i for i in ids}
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            name = os.path.basename(member)
            if name not in wanted:
                continue
            with archive.open(member) as src, open(dest_dir / name, "wb") as out:
                shutil.copyfileobj(src, out, 1 << 20)


def materialise(
    dest: Path = DATA_DIR,
    scratch: Path | None = None,
    subset: list[str] | None = None,
) -> list[CorpusFile]:
    """Make the corpus present under `dest` and return it.

    Args:
        dest: Where the harness reads the corpus from (git-ignored).
        scratch: An already-prepared `{voxconverse,ami}/<id>.wav|.rttm`
            directory to link from; `$TLDW_BAKEOFF_CORPUS` when omitted.
            Anything missing there is downloaded from the published sources.
        subset: Only these ids (any source), for a narrowed sweep.

    Returns:
        One `CorpusFile` per id, VoxConverse first, in the fixed listed order.
    """
    if scratch is None:
        env = os.environ.get(SCRATCH_ENV)
        scratch = Path(env) if env else None

    wanted = listing()
    if subset:
        keep = set(subset)
        wanted = [(source, i) for source, i in wanted if i in keep]

    files: list[CorpusFile] = []
    missing_vox_audio: list[str] = []
    for source, file_id in wanted:
        wav = dest / source / f"{file_id}.wav"
        rttm = dest / source / f"{file_id}.rttm"
        if scratch is not None:
            for suffix, target in ((".wav", wav), (".rttm", rttm)):
                candidate = Path(scratch) / source / f"{file_id}{suffix}"
                if candidate.is_file():
                    _link_or_copy(candidate, target)
        if not rttm.is_file():
            url = (_VOX_RTTM_URL if source == "voxconverse" else _AMI_RTTM_URL).format(id=file_id)
            _download(url, rttm)
        if not wav.is_file():
            if source == "ami":
                _download(_AMI_AUDIO_URL.format(id=file_id), wav)
            else:
                missing_vox_audio.append(file_id)
        files.append(CorpusFile(file_id, source, wav, rttm))

    if missing_vox_audio:
        _fetch_voxconverse_audio(missing_vox_audio, dest / "voxconverse")

    for entry in files:
        check_wav(entry.wav)
    return files
