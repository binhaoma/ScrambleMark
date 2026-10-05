#!/usr/bin/env python3
"""Download the checkpoints and audio used by the ScrambleMark pipelines."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import os
from pathlib import Path
import re
import subprocess
import sys
from dataclasses import dataclass
from tqdm import tqdm

MIN_GDOWN_VERSION = (6, 0, 0)


@dataclass(frozen=True)
class Checkpoint:
    name: str
    file_id: str
    relative_path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class AudioFolder:
    name: str
    folder_id: str
    relative_path: str
    expected_wavs: int


CHECKPOINTS = (
    Checkpoint(
        name="AudioMarkNet",
        file_id="1DDvfY2DPWnO1wBLQfLMaWwJUwS6fvIGx",
        relative_path="audiomark/out/ExpEmbedWatermark/ckpt/saved_epoch-169.tar",
        size=4_358_912_497,
        sha256="2bd91a931cb3413ee1b90d7a0aebe921cac7fea446f1a75c23b604587f397f94",
    ),
    Checkpoint(
        name="ScrambleMark DiffWave",
        file_id="1oubz7nmcFX0dk-ZXObkeGqRbUJSUEpca",
        relative_path="diffwave/model/weights.pt",
        size=31_742_130,
        sha256="6eef39959b95968e3dec288bcdeeb5792575ac861e9d7104b2ab698396da8f68",
    ),
    Checkpoint(
        name="Timbre watermark",
        file_id="1uZCDm5THkLHTGOkOaMBjjDcS7yB56dSk",
        relative_path=(
            "timbre_watermarking/results/ckpt/pth/"
            "compressed_none-conv2_ep_20_2023-01-17_23_01_01.pth.tar"
        ),
        size=33_778_945,
        sha256="5a52dad52607ca7e00c6498c142fbeb3d450acb3dbcbe5fba08962fd2114eaf4",
    ),
    Checkpoint(
        name="Timbre HiFi-GAN",
        file_id="1nCLdccr7XaHIJn0fcWzaJdfk3Jh216J2",
        relative_path="timbre_watermarking/hifigan/model/VCTK_V1/generator_v1",
        size=55_788_858,
        sha256="27cdeb835874516f9404d6c1a9ea229b092fd98c329b8444f5955c24cf7b29a1",
    ),
)


AUDIO_FOLDERS = (
    AudioFolder(
        name="Clean audio",
        folder_id="1qvjV9zdYNzMUV5XSXQ_YshkJQFo9OvUH",
        relative_path="clean_audio",
        expected_wavs=102,
    ),
    AudioFolder(
        name="AudioMarkNet watermarked audio",
        folder_id="1bBMVtZIEAL1TqW9Xv41RotrgsiyGfVFn",
        relative_path="audiomarknet_audio",
        expected_wavs=102,
    ),
)


def parse_version(version: str) -> tuple[int, int, int]:
    numbers = [int(value) for value in re.findall(r"\d+", version)[:3]]
    return tuple((numbers + [0, 0, 0])[:3])  # type: ignore[return-value]


def load_gdown(no_install_deps: bool):
    try:
        gdown = importlib.import_module("gdown")
        installed_version = parse_version(getattr(gdown, "__version__", "0"))
    except ImportError:
        gdown = None
        installed_version = (0, 0, 0)

    if gdown is not None and installed_version >= MIN_GDOWN_VERSION:
        return gdown

    requirement = "gdown>=6.0.0,<7"
    if no_install_deps:
        raise RuntimeError(
            f"This downloader requires {requirement}. Install it with:\n"
            f"  {sys.executable} -m pip install --upgrade '{requirement}'"
        )

    current = getattr(gdown, "__version__", "not installed")
    print(f"Installing required downloader ({requirement}); current version: {current}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", requirement],
        check=True,
    )

    # Restart so Python imports the newly installed package instead of a module
    # cached from the old version.
    os.execv(
        sys.executable,
        [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--no-install-deps"],
    )
    raise AssertionError("unreachable")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        while chunk := file_obj.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_is_valid(path: Path, checkpoint: Checkpoint) -> bool:
    if not path.is_file() or path.stat().st_size != checkpoint.size:
        return False

    print(f"  Verifying existing file: {path}")
    return sha256_file(path) == checkpoint.sha256


def download_checkpoint(gdown, root: Path, checkpoint: Checkpoint) -> None:
    destination = root / checkpoint.relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nCheckpoint: {checkpoint.name}")
    if checkpoint_is_valid(destination, checkpoint):
        print(f"  OK, already present: {destination}")
        return

    partial = destination.with_name(destination.name + ".part")
    if partial.exists() and partial.stat().st_size > checkpoint.size:
        partial.unlink()

    print(f"  Downloading to: {partial}")
    result = gdown.download(
        id=checkpoint.file_id,
        output=str(partial),
        quiet=False,
        resume=True,
    )
    if result is None or not partial.is_file():
        raise RuntimeError(f"Download failed: {checkpoint.name}")

    actual_size = partial.stat().st_size
    if actual_size != checkpoint.size:
        raise RuntimeError(
            f"Size mismatch for {checkpoint.name}: "
            f"expected {checkpoint.size}, got {actual_size}. "
            f"Partial file kept at {partial}"
        )

    print("  Verifying SHA256...")
    actual_sha256 = sha256_file(partial)
    if actual_sha256 != checkpoint.sha256:
        raise RuntimeError(
            f"SHA256 mismatch for {checkpoint.name}. "
            f"Partial file kept at {partial}"
        )

    os.replace(partial, destination)
    print(f"  Installed: {destination}")


def download_audio_folder(gdown, root: Path, audio_folder: AudioFolder) -> None:
    destination = root / audio_folder.relative_path
    destination.mkdir(parents=True, exist_ok=True)

    print(f"\nAudio: {audio_folder.name}")
    print(f"  Destination: {destination}")
    remote_files = gdown.download_folder(
        id=audio_folder.folder_id,
        output=str(destination),
        quiet=True,
        skip_download=True,
    )
    if remote_files is None:
        raise RuntimeError(f"Could not list Google Drive folder: {audio_folder.name}")

    remote_wavs = [item for item in remote_files if item.path.lower().endswith(".wav")]
    if len(remote_wavs) != audio_folder.expected_wavs:
        raise RuntimeError(
            f"Unexpected remote file count for {audio_folder.name}: expected "
            f"{audio_folder.expected_wavs} WAV files, found {len(remote_wavs)}."
        )

    # gdown can enumerate the folder, but Google Drive currently serves these
    # small WAV files as inline audio responses that gdown does not recognize.
    # Download each public file through its normal HTTP endpoint instead.
    import requests

    with requests.Session() as session:
        session.headers.update({"User-Agent": "Mozilla/5.0"})
        for remote_file in tqdm(
            remote_wavs,
            desc=audio_folder.name,
            unit="file",
            leave=False,
        ):
            relative_path = Path(remote_file.path)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise RuntimeError(f"Unsafe remote path: {remote_file.path}")

            local_path = destination / relative_path
            download_public_drive_wav(session, remote_file.id, local_path)

    wav_count = sum(1 for path in destination.rglob("*.wav") if path.is_file())
    if wav_count < audio_folder.expected_wavs:
        raise RuntimeError(
            f"{audio_folder.name} is incomplete: expected at least "
            f"{audio_folder.expected_wavs} WAV files, found {wav_count}."
        )

    print(f"  Ready: {wav_count} WAV files")


def is_valid_wav(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 44:
        return False
    with path.open("rb") as file_obj:
        header = file_obj.read(12)
    return header[:4] in (b"RIFF", b"RF64") and header[8:12] == b"WAVE"


def download_public_drive_wav(session, file_id: str, destination: Path) -> None:
    if is_valid_wav(destination):
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    url = f"https://drive.google.com/uc?id={file_id}&export=download"

    with session.get(
        url,
        stream=True,
        allow_redirects=True,
        timeout=(20, 120),
    ) as response:
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower()
        if "text/html" in content_type:
            raise RuntimeError(
                f"Google Drive returned HTML instead of WAV for {destination.name}. "
                "Check sharing permissions or try again later."
            )

        expected_size = response.headers.get("Content-Length")
        with partial.open("wb") as file_obj:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file_obj.write(chunk)

    if expected_size is not None and partial.stat().st_size != int(expected_size):
        raise RuntimeError(
            f"Incomplete download for {destination.name}: expected "
            f"{expected_size} bytes, got {partial.stat().st_size}."
        )
    if not is_valid_wav(partial):
        raise RuntimeError(f"Downloaded file is not a valid WAV: {destination.name}")

    os.replace(partial, destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download all checkpoints and audio required by the ScrambleMark "
            "watermark pipelines."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project root directory (default: directory containing this script).",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--checkpoints-only",
        action="store_true",
        help="Download and install only checkpoints.",
    )
    group.add_argument(
        "--audio-only",
        action="store_true",
        help="Download only clean and AudioMarkNet audio.",
    )
    parser.add_argument(
        "--no-install-deps",
        action="store_true",
        help="Do not automatically install or upgrade gdown.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.root.expanduser().resolve()

    if not root.is_dir():
        raise FileNotFoundError(f"Project root does not exist: {root}")

    gdown = load_gdown(args.no_install_deps)

    download_checkpoints = not args.audio_only
    download_audio = not args.checkpoints_only

    print(f"Project root: {root}")

    if download_checkpoints:
        print("\n========== Downloading checkpoints ==========")

        for checkpoint in tqdm(
            CHECKPOINTS,
            desc="Checkpoints",
            unit="checkpoint",
        ):
            download_checkpoint(gdown, root, checkpoint)

    if download_audio:
        print("\n========== Downloading audio ==========")

        for audio_folder in tqdm(
            AUDIO_FOLDERS,
            desc="Audio folders",
            unit="folder",
        ):
            download_audio_folder(gdown, root, audio_folder)

    print("\nAll requested project assets are ready.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nDownload interrupted. Run the same command to resume.", file=sys.stderr)
        raise SystemExit(130)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"\nError: {error}", file=sys.stderr)
        raise SystemExit(1)
