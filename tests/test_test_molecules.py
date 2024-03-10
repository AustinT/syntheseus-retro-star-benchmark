from syntheseus_retro_star_benchmark import get_190_hard_test_smiles


def test_test_smiles():
    test_smiles_list = get_190_hard_test_smiles()
    assert len(test_smiles_list) == 190
    assert test_smiles_list[0] == "C[C@H](c1ccccc1)N1C[C@]2(C(=O)OC(C)(C)C)C=CC[C@@H]2C1=S"
