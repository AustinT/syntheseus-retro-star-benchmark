"""Code to handle retrieving all files."""

from pathlib import Path
from urllib.request import urlretrieve

from syntheseus.reaction_prediction.utils.downloading import get_cache_dir

BASE_PATH = get_cache_dir("syntheseus_retro_star_benchmark")


def get_test_routes_file() -> Path:
    expected_path = BASE_PATH / "190-hard-test-routes.pkl"
    if not expected_path.exists():
        urlretrieve(
            url="https://figshare.com/ndownloader/files/44956066",
            filename=expected_path,
        )
    return expected_path


def get_value_function_checkpoint() -> Path:
    expected_path = BASE_PATH / "value-function-checkpoint.pt"
    if not expected_path.exists():
        urlretrieve(
            url="https://figshare.com/ndownloader/files/44956063",
            filename=expected_path,
        )
    return expected_path


def get_rxn_model_checkpoint() -> Path:
    expected_path = BASE_PATH / "pretrained-reaction-model.ckpt"
    if not expected_path.exists():
        urlretrieve(
            url="https://figshare.com/ndownloader/files/44956078",
            filename=expected_path,
        )
    return expected_path


def get_template_file() -> Path:
    expected_path = BASE_PATH / "template_rules_1.dat"
    if not expected_path.exists():
        urlretrieve(
            url="https://figshare.com/ndownloader/files/44956072",
            filename=expected_path,
        )
    return expected_path


def get_original_inventory_csv() -> Path:
    expected_path = BASE_PATH / "ORIGINAL-inventory.csv.gz"
    if not expected_path.exists():
        urlretrieve(
            url="https://figshare.com/ndownloader/files/44956075",
            filename=expected_path,
        )
    return expected_path
