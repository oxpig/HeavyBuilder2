"""Tests of downloading and checking the model weights."""

import hashlib
import os

import pytest
import torch

from HeavyBuilder import weights
from HeavyBuilder.HeavyBuilder2 import HeavyBuilder2, embed_dim
from HeavyBuilder.models import StructureModule


@pytest.fixture
def record(tmp_path, monkeypatch):
    """A stand-in for the Zenodo record, holding one small file, `w`."""

    record = tmp_path / "record"
    record.mkdir()
    (record / "w").write_bytes(b"some weights")
    monkeypatch.setattr(weights, "SHA256", {"w": hashlib.sha256(b"some weights").hexdigest()})
    monkeypatch.setattr(weights, "_url", lambda name: (record / name).as_uri())
    return record


def test_downloads_and_reuses(record, tmp_path):
    directory = tmp_path / "weights"

    path = weights.fetch("w", directory)

    assert path == directory / "w"
    assert path.read_bytes() == b"some weights"
    assert os.listdir(directory) == ["w"]
    assert path.stat().st_mode & 0o777 == 0o644

    # Once downloaded, the file is used without the record.
    (record / "w").unlink()
    assert weights.fetch("w", directory) == path


@pytest.mark.parametrize("contents", [b"", b"EMPTY", b"other weights"])
def test_replaces_a_wrong_file(record, tmp_path, contents):
    directory = tmp_path / "weights"
    directory.mkdir()
    (directory / "w").write_bytes(contents)

    weights.fetch("w", directory)

    assert (directory / "w").read_bytes() == b"some weights"


def test_rejects_a_corrupt_download(record, tmp_path, monkeypatch):
    monkeypatch.setattr(weights, "SHA256", {"w": "0" * 64})
    directory = tmp_path / "weights"

    with pytest.raises(RuntimeError, match="wrong SHA-256"):
        weights.fetch("w", directory)
    assert os.listdir(directory) == []


def test_keeps_the_old_file_if_a_download_fails(record, tmp_path):
    directory = tmp_path / "weights"
    directory.mkdir()
    (directory / "w").write_bytes(b"other weights")
    (record / "w").unlink()

    with pytest.raises(RuntimeError, match="could not download the weights w") as e:
        weights.fetch("w", directory)
    assert "https://doi.org/10.5281/zenodo.24115156" in str(e.value)
    assert os.listdir(directory) == ["w"]
    assert (directory / "w").read_bytes() == b"other weights"


def test_default_directory():
    assert weights.default_directory() == weights.Path(weights.__file__).resolve().parent / "trained_model"


def test_checksums_cover_the_models():
    assert set(weights.SHA256) == set(embed_dim)


def test_predictor_loads_the_downloaded_weights(tmp_path, monkeypatch):
    name = "heavy_chain_model_1"
    record = tmp_path / "record"
    record.mkdir()
    torch.manual_seed(0)
    state = StructureModule(rel_pos_dim=64, embed_dim=embed_dim[name]).state_dict()
    torch.save(state, record / name)
    sha256 = hashlib.sha256((record / name).read_bytes()).hexdigest()
    monkeypatch.setattr(weights, "SHA256", {name: sha256})
    monkeypatch.setattr(weights, "_url", lambda name: (record / name).as_uri())

    predictor = HeavyBuilder2(model_ids=[1], weights_dir=tmp_path / "weights")

    loaded = predictor.models[name].state_dict()
    assert all(torch.equal(loaded[k].cpu(), state[k]) for k in state)
