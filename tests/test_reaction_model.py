import pickle

import pytest
from syntheseus import Bag, Molecule, SingleProductReaction

from syntheseus_retro_star_benchmark import RetroStarReactionModel
from syntheseus_retro_star_benchmark.files import get_test_routes_file


@pytest.fixture
def test_routes() -> list[list[str]]:
    with open(get_test_routes_file(), "rb") as f:
        test_routes = pickle.load(f)
    return test_routes


def test_retro_star_reaction_model(test_routes):
    """Test that the retro-star reaction model seems to work."""
    rxn_model = RetroStarReactionModel()
    for route in test_routes[:10]:  # don't test everything

        # Which reaction is expected to occur
        first_rxn = route[0]
        prod, _, reactants = first_rxn.split(">")
        expected_rxn = SingleProductReaction(
            product=Molecule(prod),
            reactants=Bag([Molecule(r) for r in reactants.split(".")]),
        )

        # Does this reaction actually appear in the outputs?
        outputs = rxn_model([expected_rxn.product])[0]
        assert expected_rxn in outputs
