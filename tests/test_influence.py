import networkx as nx
import numpy as np
import pytest

from graph_tiger.influence import Influence


def test_independent_cascade_spreads_one_hop_per_round():
    simulation = Influence(nx.path_graph(4), model='independent_cascade',
                           seeds={0}, probability=1, runs=1, steps=3, seed=17)

    assert simulation.run_single_sim() == [1, 2, 3, 4]
    assert simulation.sim_info[1]['changed'] == {1}
    assert simulation.sim_info[2]['changed'] == {2}


def test_independent_cascade_zero_probability_stops():
    simulation = Influence(nx.path_graph(4), model='independent_cascade',
                           seeds={0}, probability=0, runs=1, steps=3)

    assert simulation.run_single_sim() == [1, 1, 1, 1]
    assert simulation.sim_info[3]['frontier'] == set()


def test_independent_cascade_does_not_retry_failed_edge():
    class SequenceRandom:
        def __init__(self):
            self.values = iter([0.9, 0.0])

        def random(self):
            return next(self.values)

    simulation = Influence(nx.path_graph(2), model='independent_cascade',
                           seeds={0}, probability=0.5, runs=1, steps=2)
    simulation.random = SequenceRandom()

    assert simulation.run_single_sim() == [1, 1, 1]


def test_independent_cascade_reads_edge_probabilities():
    graph = nx.path_graph(3)
    nx.set_edge_attributes(graph, {(0, 1): 1, (1, 2): 0}, 'probability')
    simulation = Influence(graph, model='independent_cascade', seeds={0},
                           probability='probability', runs=1, steps=2)

    assert simulation.run_single_sim() == [1, 2, 2]


def test_linear_threshold_combines_neighbor_influence():
    graph = nx.Graph([('u1', 'v', {'weight': 0.3}),
                      ('u2', 'v', {'weight': 0.4})])
    thresholds = {'u1': 1, 'u2': 1, 'v': 0.6}
    simulation = Influence(graph, model='linear_threshold', seeds={'u1', 'u2'},
                           threshold=thresholds, runs=1, steps=1)

    assert simulation.run_single_sim() == [2, 3]
    assert np.isclose(simulation.influence['v'], 0.7)


def test_linear_threshold_does_not_activate_from_one_weak_neighbor():
    graph = nx.Graph([('u', 'v', {'weight': 0.4})])
    thresholds = {'u': 1, 'v': 0.6}
    simulation = Influence(graph, model='linear_threshold', seeds={'u'},
                           threshold=thresholds, runs=1, steps=2)

    assert simulation.run_single_sim() == [1, 1, 1]


def test_directed_influence_uses_successors():
    graph = nx.DiGraph([(0, 1)])
    forward = Influence(graph, model='independent_cascade', seeds={0},
                        probability=1, runs=1, steps=1)
    backward = Influence(graph, model='independent_cascade', seeds={1},
                         probability=1, runs=1, steps=1)

    assert forward.run_single_sim() == [1, 2]
    assert backward.run_single_sim() == [1, 1]


def test_voter_update_changes_at_most_one_node():
    simulation = Influence(nx.path_graph(2), model='voter',
                           initial_state={0: 0, 1: 1}, tracked_state=1,
                           runs=1, steps=1, seed=17)

    simulation.run_single_sim()

    assert len(simulation.sim_info[1]['changed']) == 1
    assert simulation.sim_info[1]['tracked'] in {0, 2}


def test_voter_leaves_isolated_nodes_unchanged():
    simulation = Influence(nx.empty_graph(2), model='voter',
                           initial_state={0: 0, 1: 1}, tracked_state=1,
                           runs=1, steps=4)

    assert simulation.run_single_sim() == [1, 1, 1, 1, 1]
    assert simulation.state == {0: 0, 1: 1}


def test_competitive_cascade_resolves_collision_after_round():
    graph = nx.Graph([('claim', 'target'), ('correction', 'target')])
    simulation = Influence(
        graph,
        model='competitive_cascade',
        message_seeds={'claim': {'claim'}, 'correction': {'correction'}},
        probability=1,
        tracked_state='correction',
        tie_break='priority',
        priority=['correction', 'claim'],
        runs=1,
        steps=1
    )

    assert simulation.run_single_sim() == [1, 2]
    assert simulation.state['target'] == 'correction'
    assert simulation.sim_info[1]['counts'] == {'claim': 1, 'correction': 2}


def test_competitive_cascade_uses_message_probabilities():
    graph = nx.Graph([('a', 'x'), ('b', 'y')])
    simulation = Influence(
        graph,
        model='competitive_cascade',
        message_seeds={'a': {'a'}, 'b': {'b'}},
        probability={'a': 1, 'b': 0},
        tracked_state='a',
        runs=1,
        steps=1
    )

    assert simulation.run_single_sim() == [1, 2]
    assert simulation.state['x'] == 'a'
    assert simulation.state['y'] is None


def test_influence_reset_is_reproducible_and_preserves_inputs():
    graph = nx.cycle_graph(8)
    edges = set(graph.edges)
    first = Influence(graph, model='independent_cascade', seeds={0},
                      probability=0.4, runs=1, steps=4, seed=17)
    repeated = Influence(graph, model='independent_cascade', seeds={0},
                         probability=0.4, runs=1, steps=4, seed=17)

    assert first.run_single_sim() == repeated.run_single_sim()
    first.reset_simulation()
    assert first.state[0] == 1
    assert sum(first.state.values()) == 1
    assert set(graph.edges) == edges


def test_influence_run_simulation_returns_fixed_mean_timeline():
    simulation = Influence(nx.path_graph(4), model='independent_cascade',
                           seeds={0}, probability=1, runs=3, steps=5)

    assert simulation.run_simulation() == [1, 2, 3, 4, 4, 4]


@pytest.mark.parametrize('kwargs', [
    {'model': 'unknown'},
    {'model': 'independent_cascade', 'seeds': {9}},
    {'model': 'independent_cascade', 'probability': -0.1},
    {'model': 'independent_cascade', 'probability': 1.1},
    {'model': 'independent_cascade', 'probability': np.nan},
    {'model': 'linear_threshold', 'seeds': {0}},
    {'model': 'linear_threshold', 'seeds': {0}, 'threshold': 0},
    {'model': 'linear_threshold', 'seeds': {0}, 'weight': 1},
    {'model': 'voter', 'initial_state': {0: 0}},
    {'model': 'voter', 'initial_state': {0: [], 1: [], 2: []}},
    {'model': 'competitive_cascade',
     'message_seeds': {'a': {0}, 'b': {0}}, 'tracked_state': 'a'},
    {'model': 'competitive_cascade',
     'message_seeds': {'a': {0}, 'b': {1}}, 'tracked_state': 'missing'},
    {'model': 'competitive_cascade',
     'message_seeds': {'a': {0}, 'b': {1}}, 'tracked_state': 'a',
     'tie_break': 'priority', 'priority': ['a']}
])
def test_influence_validates_model_inputs(kwargs):
    with pytest.raises(ValueError):
        Influence(nx.path_graph(3), runs=1, steps=1, **kwargs)


def test_influence_rejects_multigraph_and_directed_voter():
    with pytest.raises(ValueError):
        Influence(nx.MultiGraph([(0, 1)]), runs=1, steps=1)
    with pytest.raises(ValueError):
        Influence(nx.DiGraph([(0, 1)]), model='voter',
                  initial_state={0: 0, 1: 1}, tracked_state=1,
                  runs=1, steps=1)


def test_influence_plotting_uses_public_simulation_interface(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    simulation = Influence(nx.path_graph(3), model='independent_cascade',
                           seeds={0}, probability=1, runs=1, steps=2,
                           plot_transition=True)

    results = simulation.run_simulation()
    simulation.plot_results(results)

    assert len(list(tmp_path.glob('plots/**/*.pdf'))) == 4
