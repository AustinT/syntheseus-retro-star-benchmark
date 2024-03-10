from syntheseus import Molecule

from syntheseus_retro_star_benchmark import RetroStarInventory


def test_retro_star_inventory():
    """Test that the retro-star inventory is correct."""
    test_tuples = [
        # First some molecules near the start of the inventory
        (Molecule("Br"), True),
        (Molecule("CC12CC(O)C(CC1=O)C2(C)C"), True),
        #
        # Next some molecules near end of inventory
        (Molecule("O=C(O)c1cc(-c2cccc([N+](=O)O)c2)[nH]n1"), True),
        (Molecule("COc1ncnc(NS(=O)(=O)/C=C/C=C2/N=C2C)c1OC"), True),
        #
        # Next some non-canonical SMILES which should have been canonicalized
        (
            Molecule("O=C(Cc1csc(-c2ccoc2)n1)N[C@H]1C[C@H](O)[C@H](O)C1", canonicalize=False),
            False,
        ),
        (Molecule("O[C@H]1CN(c2nn[nH]n2)C[C@H]1O", canonicalize=False), False),
        #
        # Next some random molcules which should definitely not be in the inventory
        (
            Molecule(
                "C" * 41,
            ),
            False,
        ),  # huge alkane with non-standard number of carbons
        (
            Molecule("hello", make_rdkit_mol=False, canonicalize=False),
            False,
        ),  # not a SMILES
    ]
    inventory = RetroStarInventory()
    for mol, expected_purchasable in test_tuples:
        assert inventory.is_purchasable(mol) == expected_purchasable
