from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import torch
from syntheseus.search.graph.and_or import AndOrGraph, OrNode
from syntheseus.search.node_evaluation.base import NoCacheNodeEvaluator

from .files import get_value_function_checkpoint
from .original_code.smiles_to_fp import batch_smiles_to_fp
from .original_code.value_mlp import ValueMLP


class RetroStarValueMLP(NoCacheNodeEvaluator[OrNode]):
    """Wrapper for retro*'s pre-trained value function."""

    def __init__(
        self,
        model_checkpoint: Optional[str] = None,
        device: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if model_checkpoint is None:
            model_checkpoint = str(get_value_function_checkpoint())

        if device is None:
            # Smart default: CUDA if it is available
            device = 0 if torch.cuda.is_available() else -1

        # Default values taken from:
        # https://github.com/binghong-ml/retro_star/blob/master/retro_star/common/parse_args.py
        self._fp_dim = 2048
        self._value_mlp = ValueMLP(
            n_layers=1,
            fp_dim=self._fp_dim,
            latent_dim=128,
            dropout_rate=0.1,
            device=device,
        )
        self._value_mlp.load_state_dict(torch.load(model_checkpoint))
        self._value_mlp.eval()

    @property
    def _mlp_device(self):
        return self._value_mlp.layers[0].weight.device

    @property
    def _mlp_dtype(self):
        return self._value_mlp.layers[0].weight.dtype

    def _evaluate_nodes(self, nodes: Sequence[OrNode], graph: Optional[AndOrGraph] = None) -> list[float]:
        # Edge case: no input mols
        if len(nodes) == 0:
            return []

        fps = batch_smiles_to_fp([node.mol.smiles for node in nodes], fp_dim=self._fp_dim)
        fp_tensor = torch.as_tensor(fps, dtype=self._mlp_dtype).to(self._mlp_device)
        with torch.no_grad():
            values = self._value_mlp(fp_tensor).detach().cpu().numpy().flatten()
        assert len(values) == len(nodes)
        return [float(v) for v in values.flatten()]
