"""
HeavyBuilder2's model weights: download and checking.

The weights are published on Zenodo (https://doi.org/10.5281/zenodo.24115156).
Each file is downloaded on first use into the weights directory and checked
against its SHA-256.  A file already in the directory is used only if its
SHA-256 is right; otherwise it is downloaded again.
"""

import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

RECORD = "https://zenodo.org/records/24115156/files"
DOI = "https://doi.org/10.5281/zenodo.24115156"

# Seconds to wait for a response, or for more data, before giving up.
TIMEOUT = 60

# The SHA-256 of each weights file in the Zenodo record.
SHA256 = {
    "heavy_chain_model_1": "ca8d974f0c43ad80463847898d7cc31ee3cc3d4ca31854ce2b6f722321888a77",
    "heavy_chain_model_2": "a85496b331f4249ae24b91177a51c510be207f2ec80b55594dfd43d5bff64c9e",
    "heavy_chain_model_3": "98164168764f53c81b499574afdcd40f09d90a77aab07bcd9d46e8c04f5ee6d2",
}


def default_directory():
    """The `trained_model` directory inside the installed package."""

    return Path(__file__).resolve().parent / "trained_model"


def fetch(name, directory=None):
    """
    Make sure the weights file `name` is in `directory`, downloading it if it
    is missing or its SHA-256 is wrong, and return its path.
    """

    sha256 = SHA256[name]
    directory = default_directory() if directory is None else Path(directory)
    path = directory / name
    if path.exists() and _sha256(path) == sha256:
        return path

    directory.mkdir(parents=True, exist_ok=True)
    _download(name, path, sha256)
    return path


def _url(name):
    return f"{RECORD}/{name}?download=1"


def _download(name, path, sha256):
    url = _url(name)
    print(f"Downloading weights for {name} from {DOI}", flush=True)
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as response, _replacing(path) as f:
            while block := response.read(1 << 20):
                digest.update(block)
                f.write(block)
            if digest.hexdigest() != sha256:
                raise ValueError(f"downloaded file has the wrong SHA-256 ({digest.hexdigest()})")
    except (OSError, ValueError) as e:
        raise RuntimeError(
            f"HeavyBuilder2 could not download the weights {name} into {path.parent}: {e}.  "
            f"Download {name} from {DOI} into that directory, or choose a directory "
            "that has it with --weights_dir (weights_dir in Python)."
        ) from e


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


class _replacing:
    """
    Write to a temporary file beside `path`, and move it into place only if
    the writing succeeds, so that `path` is never left incomplete.
    """

    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        self.file = tempfile.NamedTemporaryFile(
            dir=self.path.parent, prefix=f".{self.path.name}.", delete=False)
        return self.file

    def __exit__(self, exc_type, exc, traceback):
        self.file.close()
        if exc_type is None:
            # Temporary files are private, but weights may be shared.
            os.chmod(self.file.name, 0o644)
            os.replace(self.file.name, self.path)
        else:
            os.unlink(self.file.name)
