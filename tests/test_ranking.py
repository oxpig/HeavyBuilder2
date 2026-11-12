"""Tests that saving uses the models in the order of their ranking."""

import importlib

import pytest
import torch

from HeavyBuilder.HeavyBuilder2 import HeavyBuilder

# `HeavyBuilder.HeavyBuilder2` is also the name of the predictor class, which
# the package exports in place of the module.
hb2 = importlib.import_module("HeavyBuilder.HeavyBuilder2")

N_RESIDUES = 10


@pytest.fixture
def prediction():
    """
    Three predictions of a short heavy chain, of which model 0 is furthest
    from the others, so that it is ranked last.
    """

    torch.manual_seed(0)
    numbered = {"H": [((i + 1, " "), "A") for i in range(N_RESIDUES)], "L": []}
    # A spread-out cloud of atoms, so that the alignment is well defined.
    base = 10 * torch.randn(N_RESIDUES, 14, 3)
    atoms = [
        base + 2.0 * torch.randn(N_RESIDUES, 14, 3),
        base + 0.1 * torch.randn(N_RESIDUES, 14, 3),
        base + 0.2 * torch.randn(N_RESIDUES, 14, 3),
    ]
    model = HeavyBuilder(numbered, [(a, None) for a in atoms])
    assert model.ranking[-1] == 0
    return model


@pytest.fixture
def refined(monkeypatch):
    """Replace refinement with a copy, recording the input it was given."""

    inputs = []

    def refine(input_file, output_file, check_for_strained_bonds=True):
        with open(input_file) as f:
            text = f.read()
        inputs.append(text)
        with open(output_file, "w") as f:
            f.write(text)
        return True

    monkeypatch.setattr(hb2, "refine", refine)
    return inputs


def unrefined(model, index, path):
    model.save_single_unrefined(path, index=index)
    return path.read_text()


def test_save_writes_the_best_model(prediction, refined, tmp_path):
    prediction.save(str(tmp_path / "out.pdb"))

    assert refined == [unrefined(prediction, prediction.ranking[0], tmp_path / "best.pdb")]


def test_save_all_writes_the_models_in_rank_order(prediction, refined, tmp_path):
    directory = tmp_path / "out"

    prediction.save_all(str(directory))

    for rank, index in enumerate(prediction.ranking):
        expected = unrefined(prediction, index, tmp_path / "expected.pdb")
        assert (directory / f"rank{rank}_unrefined.pdb").read_text() == expected
    assert refined == [unrefined(prediction, prediction.ranking[0], tmp_path / "best.pdb")]
