from __future__ import annotations

import pickle

from rdkit import Chem

from .files import get_test_routes_file


def get_190_hard_test_smiles() -> list[str]:
    with open(get_test_routes_file(), "rb") as f:
        test_routes = pickle.load(f)
    output = [r[0].split(">")[0] for r in test_routes]
    output = [Chem.CanonSmiles(s) for s in output]  # canonicalize
    assert len(output) == 190
    return output
