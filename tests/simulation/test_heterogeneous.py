from economic_engine.simulation.heterogeneous import (
    ARCHETYPES,
    make_archetype,
    run_heterogeneous_experiment,
)


def test_all_archetypes_constructible():
    for a in ARCHETYPES:
        opp = make_archetype(a, base=100.0, rng_seed=1)
        assert opp.reservation > 0


def test_counterfactual_report_shape():
    res = run_heterogeneous_experiment(per_archetype=3)
    assert "overall" in res and "by_archetype" in res
    assert set(res["overall"].keys()) == {"fixed", "nash", "chotu"}
    for a in ARCHETYPES:
        assert a in res["by_archetype"]
        v = res["by_archetype"][a]
        assert "delta_chotu_minus_fixed" in v
        assert "delta_chotu_minus_nash" in v
    assert res["router_ev"] >= res["overall"]["fixed"]["ev"] - 1e-9
    total = res["outcomes"]
    assert total["chotu_wins"] + total["fixed_wins"] + total["ties"] == len(ARCHETYPES)
