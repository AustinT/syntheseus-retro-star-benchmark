from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch
from rdkit import RDLogger
from syntheseus import BackwardReactionModel, Bag, Molecule, SingleProductReaction

from .files import get_rxn_model_checkpoint, get_template_file
from .original_code.mlp_inference import MLPModel

DEFAULT_RETROSTAR_EXPANSION_TOPk = 50

# Turn off rdkit logger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


class RetroStarReactionModel(BackwardReactionModel):
    def __init__(
        self,
        model_checkpoint: Optional[str] = None,
        template_file: Optional[str] = None,
        expansion_topk: int = 50,
        device: Optional[int] = None,
        shuffle: bool = False,
        **kwargs,
    ) -> None:

        super().__init__(**kwargs)

        if model_checkpoint is None:
            model_checkpoint = str(get_rxn_model_checkpoint())
        if template_file is None:
            template_file = str(get_template_file())

        self.expansion_topk = expansion_topk
        if device is None:
            # Smart default: CUDA if it is available
            device = 0 if torch.cuda.is_available() else -1
        self.model = MLPModel(model_checkpoint, template_file, device=device)
        self.model.net.eval()  # ensure eval mode
        self.random_state = random.Random()
        self._shuffle = shuffle  # whether to shuffle reactions before outputting them

    def _get_reactions(self, inputs: list[Molecule], num_results: int) -> list[list[SingleProductReaction]]:
        output = []
        for mol in inputs:
            curr_output = []

            # Call model
            output_dict = self.model.run(mol.smiles, topk=self.expansion_topk)
            if output_dict is not None:  # could be None if no reactions are possible
                reactants = output_dict["reactants"]
                scores = output_dict["scores"]
                templates = output_dict["template"]
                if len(reactants) > 0:
                    priors = np.clip(np.asarray(scores), 1e-3, 1.0)  # done by original paper
                    costs = -np.log(priors)

                    for j in range(len(reactants)):
                        rxn = SingleProductReaction(
                            reactants=Bag(
                                [Molecule(s, canonicalize=True, make_rdkit_mol=False) for s in reactants[j].split(".")]
                            ),
                            product=mol,
                            metadata=dict(
                                cost=float(costs[j]),
                                probability=float(priors[j]),
                                template=templates[j],
                                rank=j,
                            ),
                        )
                        curr_output.append(rxn)

            # Add to cumulative output list, optionally shuffling
            if self._shuffle:
                self.random_state.shuffle(curr_output)
            output.append(curr_output[:num_results])
        return output
