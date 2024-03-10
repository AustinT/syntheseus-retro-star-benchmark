from __future__ import annotations

import csv
import gzip
from pathlib import Path
from typing import Optional

import rdkit
from rdkit import Chem
from syntheseus import Molecule
from syntheseus.search.mol_inventory import BaseMolInventory

from .files import get_original_inventory_csv


def get_canonical_inventory_csv() -> Path:
    """Return the path to the canonicalized inventory."""
    original_csv_path = get_original_inventory_csv()
    rdkit_version = rdkit.__version__
    output_path = get_original_inventory_csv().with_stem(f"rdkit-v{rdkit_version}-canonical-inventory")
    if not output_path.exists():
        print(f"Canonical inventory does not exist for rdkit version {rdkit_version}. Will now make it.", flush=True)
        from tqdm.auto import tqdm

        # Read in entire inventory
        with gzip.open(original_csv_path, "rt") as f_in:
            reader = csv.reader(f_in)
            with gzip.open(output_path, "wt") as f:
                writer = csv.writer(f)

                writer.writerow(next(reader))  # write header
                for row in tqdm(reader):
                    row[1] = Chem.CanonSmiles(row[1])
                    writer.writerow(row)

    return output_path


class RetroStarInventory(BaseMolInventory):
    """
    A molecule inventory for the RetroStar task.
    This is a simple wrapper around a set of SMILES strings which will save the
    canonical SMILES to disk if it does not exist.
    """

    def __init__(
        self,
        inventory_csv: Optional[str] = None,
        canonicalize: bool = False,
        **kwargs,
    ) -> None:

        super().__init__(**kwargs)
        if inventory_csv is None:
            inventory_csv = get_canonical_inventory_csv()
        with gzip.open(inventory_csv, "rt") as f:
            reader = csv.reader(f)
            next(reader)  # discard header
            smiles_list = [row[1].strip() for row in reader]

        if canonicalize:
            raise NotImplementedError

        self._smiles_set = set(smiles_list)

    def is_purchasable(self, mol: Molecule) -> bool:
        return mol.smiles in self._smiles_set
