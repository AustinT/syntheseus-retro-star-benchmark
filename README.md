# Syntheseus retro star benchmark

A wrapper for using the benchmark from retro*
([Chen et al 2020](http://proceedings.mlr.press/v119/chen20k.html))
in [syntheseus](https://github.com/microsoft/syntheseus/).

Usage:

```python
from syntheseus_retro_star_benchmark import RetroStarReactionModel
model = RetroStarReactionModel()  # a syntheseus BackwardReactionModel object wrapping the pre-trained template classifier

from syntheseus_retro_star_benchmark import RetroStarInventory
inventory = RetroStarInventory()  # their inventory of ~23M purchasable molecules

from syntheseus_retro_star_benchmark import get_190_hard_test_smiles
test_smiles = get_190_hard_test_smiles()  # their recommended 190 test SMILES

from syntheseus_retro_star_benchmark import RetroStarValueMLP
value_fn = RetroStarValueMLP()  # their pre-trained search heuristic
```

Code was based on open source code from [here](https://github.com/binghong-ml/retro_star/tree/master/retro_star/packages/mlp_retrosyn/mlp_retrosyn).
Some data was uploaded to
[figshare](https://figshare.com/articles/dataset/Syntheseus_retro_star_benchmark_data/25376728)
to ensure stable, consistent access.

## Installation

Install either by cloning and using pip or running
`pip install syntheseus-retro-star-benchmark`.

## Development

Ensure to install all pre-commit hooks and run unit tests (provided by pytest).
