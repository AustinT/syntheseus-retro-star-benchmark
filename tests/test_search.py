"""Test that retro* search works as expected in the paper."""

from __future__ import annotations

import math

import pytest
from syntheseus import Bag, Molecule, SingleProductReaction
from syntheseus.search.algorithms.best_first.retro_star import RetroStarSearch
from syntheseus.search.analysis.route_extraction import iter_routes_cost_order
from syntheseus.search.node_evaluation.common import ConstantNodeEvaluator, ReactionModelLogProbCost

from syntheseus_retro_star_benchmark import RetroStarInventory, RetroStarReactionModel, get_190_hard_test_smiles

index_to_route_plan = {
    1: "CCCC[C@@H](C(=O)N1CCC[C@H]1C(=O)O)[C@@H](F)C(=O)OC>0.0910>CCCC[C@@H](C(=O)N1CCC[C@H]1C(=O)OC)[C@@H](F)C(=O)OC|CCCC[C@@H](C(=O)N1CCC[C@H]1C(=O)OC)[C@@H](F)C(=O)OC>0.6925>O=S(=O)(OS(=O)(=O)C(F)(F)F)C(F)(F)F.CCCC[C@@H](C(=O)N1CCC[C@H]1C(=O)OC)[C@H](O)C(=O)OC|CCCC[C@@H](C(=O)N1CCC[C@H]1C(=O)OC)[C@H](O)C(=O)OC>0.1714>CCCC[C@@H](C(=O)N1CCC[C@H]1C(=O)OC)[C@H](O)C(=O)O.CO|CCCC[C@@H](C(=O)N1CCC[C@H]1C(=O)OC)[C@H](O)C(=O)O>0.3300>COC(=O)[C@@H]1CCCN1.CCCCC(C(=O)O)C1OC(C)(C)OC1=O|CCCCC(C(=O)O)C1OC(C)(C)OC1=O>0.0010>COC(C)(C)OC.CCCCC(C(=O)O)C(O)C(=O)O",
    21: "COCCCc1cc(CN(C(=O)[C@H]2CNCC[C@@H]2c2ccc(OCCOc3c(Cl)cc(C)cc3Cl)cc2)C2CC2)cc(OCCOC)c1>0.9074>COCCCc1cc(CN(C(=O)[C@H]2CN(C(=O)OC(C)(C)C)CC[C@@H]2c2ccc(OCCOc3c(Cl)cc(C)cc3Cl)cc2)C2CC2)cc(OCCOC)c1|COCCCc1cc(CN(C(=O)[C@H]2CN(C(=O)OC(C)(C)C)CC[C@@H]2c2ccc(OCCOc3c(Cl)cc(C)cc3Cl)cc2)C2CC2)cc(OCCOC)c1>0.2336>COCCCc1cc(CN(C(=O)[C@H]2CN(C(=O)OC(C)(C)C)CC[C@@H]2c2ccc(O)cc2)C2CC2)cc(OCCOC)c1.Cc1cc(Cl)c(OCCO)c(Cl)c1|COCCCc1cc(CN(C(=O)[C@H]2CN(C(=O)OC(C)(C)C)CC[C@@H]2c2ccc(O)cc2)C2CC2)cc(OCCOC)c1>0.1910>COCCCc1cc(CN(C(=O)[C@H]2CN(C(=O)OC(C)(C)C)CC[C@@H]2c2ccc(OCc3ccccc3)cc2)C2CC2)cc(OCCOC)c1|Cc1cc(Cl)c(OCCO)c(Cl)c1>0.8220>Cc1cc(Cl)c(O)c(Cl)c1.O=C1OCCO1|COCCCc1cc(CN(C(=O)[C@H]2CN(C(=O)OC(C)(C)C)CC[C@@H]2c2ccc(OCc3ccccc3)cc2)C2CC2)cc(OCCOC)c1>0.2292>CC(C)(C)OC(=O)N1CC[C@H](c2ccc(OCc3ccccc3)cc2)[C@@H](C(=O)O)C1.COCCCc1cc(CNC2CC2)cc(OCCOC)c1|CC(C)(C)OC(=O)N1CC[C@H](c2ccc(OCc3ccccc3)cc2)[C@@H](C(=O)O)C1>0.9394>CCOC(=O)[C@H]1CN(C(=O)OC(C)(C)C)CC[C@@H]1c1ccc(OCc2ccccc2)cc1|COCCCc1cc(CNC2CC2)cc(OCCOC)c1>0.9296>NC1CC1.COCCCc1cc(C=O)cc(OCCOC)c1|CCOC(=O)[C@H]1CN(C(=O)OC(C)(C)C)CC[C@@H]1c1ccc(OCc2ccccc2)cc1>0.9006>CCOC(=O)[C@H]1CN(C(=O)OC(C)(C)C)CC[C@H]1c1ccc(OCc2ccccc2)cc1|COCCCc1cc(C=O)cc(OCCOC)c1>0.3680>COCCOc1cc(Br)cc(C=O)c1.C=CCOC|CCOC(=O)[C@H]1CN(C(=O)OC(C)(C)C)CC[C@H]1c1ccc(OCc2ccccc2)cc1>0.8218>CCOC(=O)[C@H]1CNCC[C@H]1c1ccc(OCc2ccccc2)cc1.CC(C)(C)OC(=O)OC(=O)OC(C)(C)C|CCOC(=O)[C@H]1CNCC[C@H]1c1ccc(OCc2ccccc2)cc1>0.8101>CCOC(=O)C1=C(c2ccc(OCc3ccccc3)cc2)CCN(Cc2ccccc2)C1|CCOC(=O)C1=C(c2ccc(OCc3ccccc3)cc2)CCN(Cc2ccccc2)C1>0.9994>CCOC(=O)C1=C(OS(=O)(=O)C(F)(F)F)CCN(Cc2ccccc2)C1.Brc1ccc(OCc2ccccc2)cc1|CCOC(=O)C1=C(OS(=O)(=O)C(F)(F)F)CCN(Cc2ccccc2)C1>0.9950>CCOC(=O)C1CN(Cc2ccccc2)CCC1=O.O=S(=O)(OS(=O)(=O)C(F)(F)F)C(F)(F)F",
    41: "CN1CCN(c2ccc3c(c2)[nH]c2c(C(N)=O)cc(-c4ccc(O)c(Cl)c4)nc23)CC1>0.7738>COC(=O)c1cc(-c2ccc(O)c(Cl)c2)nc2c1[nH]c1cc(N3CCN(C)CC3)ccc12.N|COC(=O)c1cc(-c2ccc(O)c(Cl)c2)nc2c1[nH]c1cc(N3CCN(C)CC3)ccc12>0.3775>OB(O)c1ccc(O)c(Cl)c1.COC(=O)c1cc(Br)nc2c1[nH]c1cc(N3CCN(C)CC3)ccc12|COC(=O)c1cc(Br)nc2c1[nH]c1cc(N3CCN(C)CC3)ccc12>0.4225>COC(=O)c1cc(Br)nc(-c2ccc(N3CCN(C)CC3)cc2)c1N=[N+]=[N-]|COC(=O)c1cc(Br)nc(-c2ccc(N3CCN(C)CC3)cc2)c1N=[N+]=[N-]>1.0000>COC(=O)c1cc(Br)nc(-c2ccc(N3CCN(C)CC3)cc2)c1N.[N-]=[N+]=[N-]|COC(=O)c1cc(Br)nc(-c2ccc(N3CCN(C)CC3)cc2)c1N>0.5712>CN1CCN(c2ccc(B(O)O)cc2)CC1.COC(=O)c1cc(Br)nc(Br)c1N",
    61: "COCCCc1cc(O)cc(CN(C(=O)[C@H]2CN(C(=O)OC(C)(C)C)CC[C@@H]2c2ccc(OCCOc3c(Cl)cc(C)cc3Cl)cc2)C2CC2)c1>0.3498>COCCCc1cc(CN(C(=O)[C@H]2CN(C(=O)OC(C)(C)C)CC[C@@H]2c2ccc(OCCOc3c(Cl)cc(C)cc3Cl)cc2)C2CC2)cc(O[Si](C)(C)C(C)(C)C)c1|COCCCc1cc(CN(C(=O)[C@H]2CN(C(=O)OC(C)(C)C)CC[C@@H]2c2ccc(OCCOc3c(Cl)cc(C)cc3Cl)cc2)C2CC2)cc(O[Si](C)(C)C(C)(C)C)c1>0.3548>Cc1cc(Cl)c(OCCO)c(Cl)c1.COCCCc1cc(CN(C(=O)[C@H]2CN(C(=O)OC(C)(C)C)CC[C@@H]2c2ccc(O)cc2)C2CC2)cc(O[Si](C)(C)C(C)(C)C)c1|Cc1cc(Cl)c(OCCO)c(Cl)c1>0.8220>Cc1cc(Cl)c(O)c(Cl)c1.O=C1OCCO1|COCCCc1cc(CN(C(=O)[C@H]2CN(C(=O)OC(C)(C)C)CC[C@@H]2c2ccc(O)cc2)C2CC2)cc(O[Si](C)(C)C(C)(C)C)c1>0.8995>COCCCc1cc(CN(C(=O)[C@H]2CN(C(=O)OC(C)(C)C)CC[C@@H]2c2ccc(O[Si](C)(C)C(C)(C)C)cc2)C2CC2)cc(O[Si](C)(C)C(C)(C)C)c1|COCCCc1cc(CN(C(=O)[C@H]2CN(C(=O)OC(C)(C)C)CC[C@@H]2c2ccc(O[Si](C)(C)C(C)(C)C)cc2)C2CC2)cc(O[Si](C)(C)C(C)(C)C)c1>0.0622>COCCCc1cc(CNC2CC2)cc(O[Si](C)(C)C(C)(C)C)c1.CC(C)(C)OC(=O)N1CC[C@H](c2ccc(O[Si](C)(C)C(C)(C)C)cc2)[C@@H](C(=O)O)C1|COCCCc1cc(CNC2CC2)cc(O[Si](C)(C)C(C)(C)C)c1>0.9228>COC/C=C/c1cc(CNC2CC2)cc(O[Si](C)(C)C(C)(C)C)c1|CC(C)(C)OC(=O)N1CC[C@H](c2ccc(O[Si](C)(C)C(C)(C)C)cc2)[C@@H](C(=O)O)C1>0.9797>CCOC(=O)[C@H]1CN(C(=O)OC(C)(C)C)CC[C@@H]1c1ccc(O[Si](C)(C)C(C)(C)C)cc1|COC/C=C/c1cc(CNC2CC2)cc(O[Si](C)(C)C(C)(C)C)c1>0.9782>NC1CC1.COC/C=C/c1cc(C=O)cc(O[Si](C)(C)C(C)(C)C)c1|CCOC(=O)[C@H]1CN(C(=O)OC(C)(C)C)CC[C@@H]1c1ccc(O[Si](C)(C)C(C)(C)C)cc1>0.7162>CCOC(=O)[C@H]1CN(C(=O)OC(C)(C)C)CC[C@H]1c1ccc(O[Si](C)(C)C(C)(C)C)cc1|COC/C=C/c1cc(C=O)cc(O[Si](C)(C)C(C)(C)C)c1>0.9507>CC(C)(C)[Si](C)(C)Cl.COC/C=C/c1cc(O)cc(C=O)c1|CCOC(=O)[C@H]1CN(C(=O)OC(C)(C)C)CC[C@H]1c1ccc(O[Si](C)(C)C(C)(C)C)cc1>0.6680>CCOC(=O)[C@H]1CNCC[C@H]1c1ccc(O[Si](C)(C)C(C)(C)C)cc1.CC(C)(C)OC(=O)OC(=O)OC(C)(C)C|COC/C=C/c1cc(O)cc(C=O)c1>0.9994>COC/C=C/B1OC(C)(C)C(C)(C)O1.O=Cc1cc(O)cc(Br)c1|CCOC(=O)[C@H]1CNCC[C@H]1c1ccc(O[Si](C)(C)C(C)(C)C)cc1>0.8311>CCOC(=O)C1=C(c2ccc(O[Si](C)(C)C(C)(C)C)cc2)CCN(Cc2ccccc2)C1|CCOC(=O)C1=C(c2ccc(O[Si](C)(C)C(C)(C)C)cc2)CCN(Cc2ccccc2)C1>0.9999>CC(C)(C)[Si](C)(C)Oc1ccc(Br)cc1.CCOC(=O)C1=C(OS(=O)(=O)C(F)(F)F)CCN(Cc2ccccc2)C1|CCOC(=O)C1=C(OS(=O)(=O)C(F)(F)F)CCN(Cc2ccccc2)C1>0.9950>CCOC(=O)C1CN(Cc2ccccc2)CCC1=O.O=S(=O)(OS(=O)(=O)C(F)(F)F)C(F)(F)F",
    82: "O[C@H]1C[C@H](c2cnn3c(N[C@H]4CCc5ccccc54)ncnc23)C=C1COCc1ccccc1>0.0010>O[C@@H]1C[C@H](c2cnn3c(N[C@H]4CCc5ccccc54)ncnc23)C=C1COCc1ccccc1|O[C@@H]1C[C@H](c2cnn3c(N[C@H]4CCc5ccccc54)ncnc23)C=C1COCc1ccccc1>0.9999>Brc1cnn2c(N[C@H]3CCc4ccccc43)ncnc12.O[C@@H]1CC=C[C@H]1COCc1ccccc1|Brc1cnn2c(N[C@H]3CCc4ccccc43)ncnc12>1.0000>O=c1[nH]cnc2c(Br)cnn12.N[C@H]1CCc2ccccc21",
    100: "COC(=O)CC12CCC(c3ccc(-c4ccc(NC(=O)c5nc(C)oc5C(F)(F)F)cc4)cc3)(CC1)CO2>0.3965>Cc1nc(C(=O)Nc2ccc(B3OC(C)(C)C(C)(C)O3)cc2)c(C(F)(F)F)o1.COC(=O)CC12CCC(c3ccc(Br)cc3)(CC1)CO2|Cc1nc(C(=O)Nc2ccc(B3OC(C)(C)C(C)(C)O3)cc2)c(C(F)(F)F)o1>0.9999>CC1(C)OB(c2ccc(N)cc2)OC1(C)C.Cc1nc(C(=O)O)c(C(F)(F)F)o1|COC(=O)CC12CCC(c3ccc(Br)cc3)(CC1)CO2>0.3752>O=C(O)CC12CCC(c3ccc(Br)cc3)(CC1)CO2.CO|O=C(O)CC12CCC(c3ccc(Br)cc3)(CC1)CO2>0.0666>N#CCC12CCC(c3ccc(Br)cc3)(CC1)CO2.[OH-].O|N#CCC12CCC(c3ccc(Br)cc3)(CC1)CO2>0.7003>O=C1CCC(=O)N1Br.N#CCC12CCC(c3ccccc3)(CC1)CO2|N#CCC12CCC(c3ccccc3)(CC1)CO2>0.3187>CS(=O)(=O)OCC12CCC(c3ccccc3)(CC1)CO2.[C-]#N|CS(=O)(=O)OCC12CCC(c3ccccc3)(CC1)CO2>0.9998>OCC12CCC(c3ccccc3)(CC1)CO2.CS(=O)(=O)Cl|OCC12CCC(c3ccccc3)(CC1)CO2>0.9755>OCC1(c2ccccc2)CCC2(CC1)CO2|OCC1(c2ccccc2)CCC2(CC1)CO2>0.0199>C[S+](C)(C)=O.O=C1CCC(CO)(c2ccccc2)CC1",
    121: "COC(=O)[C@@H]1CCCC2(CCCCC2)[C@H]1O>0.1263>O=C(O)[C@@H]1CCCC2(CCCCC2)[C@H]1O.CO|O=C(O)[C@@H]1CCCC2(CCCCC2)[C@H]1O>0.2352>CCOC(=O)[C@@H]1CCCC2(CCCCC2)[C@H]1O|CCOC(=O)[C@@H]1CCCC2(CCCCC2)[C@H]1O>0.1138>CCOC(=O)C1CCCC2(CCCCC2)C1=O|CCOC(=O)C1CCCC2(CCCCC2)C1=O>0.2719>O=C1CCCCC12CCCCC2.CCOC(=O)OCC",
    141: "COC(=O)CCc1cc2cc(-c3noc(-c4ccc(OC(C)C)c(Cl)c4)n3)ccc2n1C>0.9999>CCOC(=O)CCc1cc2cc(-c3noc(-c4ccc(OC(C)C)c(Cl)c4)n3)ccc2[nH]1.C1CN2CCN1CC2|CCOC(=O)CCc1cc2cc(-c3noc(-c4ccc(OC(C)C)c(Cl)c4)n3)ccc2[nH]1>0.0010>CCOC(=O)/C=C/c1cc2cc(-c3noc(-c4ccc(OC(C)C)c(Cl)c4)n3)ccc2[nH]1|CCOC(=O)/C=C/c1cc2cc(-c3noc(-c4ccc(OC(C)C)c(Cl)c4)n3)ccc2[nH]1>0.9893>CCOC(=O)C=P(c1ccccc1)(c1ccccc1)c1ccccc1.ClCCl.CC(C)Oc1ccc(-c2nc(-c3ccc4[nH]c(CO)cc4c3)no2)cc1Cl|CC(C)Oc1ccc(-c2nc(-c3ccc4[nH]c(CO)cc4c3)no2)cc1Cl>0.9315>N=C(NO)c1ccc2[nH]c(CO)cc2c1.CC(C)Oc1ccc(C(=O)O)cc1Cl|N=C(NO)c1ccc2[nH]c(CO)cc2c1>0.9983>NO.N#Cc1ccc2[nH]c(CO)cc2c1",
    161: "CCc1[nH]c(C(=O)N[C@H]2CCN(c3cccc(C(=O)OC(C)(C)C)c3)C[C@H]2OC)nc1C(F)(F)F>0.9988>CCc1[nH]c(C(=O)O)nc1C(F)(F)F.CO[C@@H]1CN(c2cccc(C(=O)OC(C)(C)C)c2)CC[C@@H]1N|CCc1[nH]c(C(=O)O)nc1C(F)(F)F>0.9916>CCc1[nH]c(C=O)nc1C(F)(F)F.[O-][Cl+][O-]|CO[C@@H]1CN(c2cccc(C(=O)OC(C)(C)C)c2)CC[C@@H]1N>0.5394>CO[C@@H]1CN(c2cccc(C(=O)OC(C)(C)C)c2)CC[C@@H]1NC(=O)OCc1ccccc1|CCc1[nH]c(C=O)nc1C(F)(F)F>0.9994>CCc1[nH]c(C(OC)OC)nc1C(F)(F)F|CO[C@@H]1CN(c2cccc(C(=O)OC(C)(C)C)c2)CC[C@@H]1NC(=O)OCc1ccccc1>0.9433>CO[C@@H]1CNCC[C@@H]1NC(=O)OCc1ccccc1.CC(C)(C)OC(=O)c1cccc(Br)c1|CCc1[nH]c(C(OC)OC)nc1C(F)(F)F>1.0000>N.COC(C=O)OC.CCC1(C(=O)C(F)(F)F)SCCCS1.O=C1CCC(=O)N1Cl|CO[C@@H]1CNCC[C@@H]1NC(=O)OCc1ccccc1>0.0010>O=C(N[C@H]1CCNC[C@H]1O)OCc1ccccc1.CI|CCC1(C(=O)C(F)(F)F)SCCCS1>1.0000>CCC1SCCCS1.CCOC(=O)C(F)(F)F",
    181: "CCCCOC(=O)N1CCN(C(=O)[C@H](CCCO[Si](c2ccccc2)(c2ccccc2)C(C)(C)C)NC(=O)OC(C)(C)C)CC1>0.0017>CCCCOC(=O)N1CCNCC1.CC(C)(C)OC(=O)N[C@@H](CCCO[Si](c1ccccc1)(c1ccccc1)C(C)(C)C)C(=O)O|CC(C)(C)OC(=O)N[C@@H](CCCO[Si](c1ccccc1)(c1ccccc1)C(C)(C)C)C(=O)O>0.0489>CC(C)(C)OC(=O)N[C@@H](CCCO)C(=O)O.CC(C)(C)[Si](Cl)(c1ccccc1)c1ccccc1",
}

index_to_route_cost = {
    1: 12.544282470771117,
    21: 6.528483654170535,
    41: 2.652369639708481,
    61: 6.267220671310183,
    82: 6.9079095775164,
    100: 10.057556142256182,
    121: 6.992401007072727,
    141: 6.991289102156023,
    161: 7.593806950657397,
    181: 9.380352818909243,
}


class RetroStarStopOnFirstSolution(RetroStarSearch):
    """Retro-star from the original paper which terminates on the first solution."""

    def setup(self, *args, **kwargs):
        self._found_solution = False
        super().setup(*args, **kwargs)

    def time_limit_reached(self) -> bool:
        return super().time_limit_reached() or self._found_solution

    def teardown(self, *args, **kwargs):
        del self._found_solution
        super().teardown(*args, **kwargs)

    def set_node_values(self, nodes, graph, **kwargs):
        output = super().set_node_values(nodes, graph, **kwargs)
        if graph.root_node.has_solution:
            self._found_solution = True
        return output


rxn_tuple_set = set[tuple[frozenset[str], str]]


def rxn_string_to_reactions(rxn_str: str) -> set[SingleProductReaction]:
    """Splits a solution string into a set of (reactant, product) tuples."""
    rxns = rxn_str.split("|")
    output: set[SingleProductReaction] = set()
    for rxn_str in rxns:
        prod, cost, reactants = rxn_str.split(">")
        rxn = SingleProductReaction(
            product=Molecule(prod),
            reactants=Bag([Molecule(r) for r in reactants.split(".")]),
            metadata={"cost": float(cost)},
        )
        output.add(rxn)
    return output


@pytest.mark.parametrize("test_idx", sorted(index_to_route_plan.keys()))
def test_found_retro_star0_route(test_idx: int) -> None:
    """
    Test that retro*-0 finds exact route from the paper.

    NOTE: there appears to be some randomness in these tests: sometimes they fail,
    sometimes they don't. If it fails I suggest re-running it.
    """

    # Load molecules and inventory
    rxn_model = RetroStarReactionModel()
    inventory = RetroStarInventory()
    test_smiles_list = get_190_hard_test_smiles()

    # Run retro-star search
    alg = RetroStarSearch(
        reaction_model=rxn_model,
        mol_inventory=inventory,
        limit_reaction_model_calls=500,
        and_node_cost_fn=ReactionModelLogProbCost(normalize=False),
        value_function=ConstantNodeEvaluator(0.0),
        time_limit_s=300,
        stop_on_first_solution=True,
    )
    output_graph, _ = alg.run_from_mol(Molecule(test_smiles_list[test_idx]))
    assert output_graph.root_node.has_solution

    # Extract routes
    expected_reaction_set = rxn_string_to_reactions(index_to_route_plan[test_idx])
    all_routes = list(iter_routes_cost_order(output_graph, max_routes=100))
    assert len(all_routes) > 0
    synthesis_routes = [output_graph.to_synthesis_graph(nodes=r) for r in all_routes]
    synthesis_route_reaction_sets = [frozenset(r.nodes()) for r in synthesis_routes]
    synthesis_route_costs = [sum(r.metadata["cost"] for r in rxn_set) for rxn_set in synthesis_route_reaction_sets]

    # Test that expected route is found (and has the correct cost)
    assert expected_reaction_set in synthesis_route_reaction_sets
    route_idx = synthesis_route_reaction_sets.index(expected_reaction_set)
    assert math.isclose(synthesis_route_costs[route_idx], index_to_route_cost[test_idx], rel_tol=1e-3)
