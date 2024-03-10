import numpy as np
from syntheseus import Molecule
from syntheseus.search.graph.and_or import OrNode

from syntheseus_retro_star_benchmark import RetroStarValueMLP


def test_values_match() -> None:
    """Test the retro-star value function."""
    mols = [
        Molecule("CCC"),
        Molecule("c1ccccc1"),
        Molecule("FCCCCCCF"),
    ]

    val_fn = RetroStarValueMLP()

    output = val_fn([OrNode(mol=m) for m in mols])
    assert np.allclose(output, [1.284, 1.392, 2.399], atol=1e-3)
