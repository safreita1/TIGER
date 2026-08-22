import json

import networkx as nx
import numpy as np

from graph_tiger.attacks import get_node_ns, run_attack_method
from graph_tiger.cascading import Cascading
from graph_tiger.defenses import Defense, run_defense_method
from graph_tiger.diffusion import Diffusion
from graph_tiger.graphs import get_graph_options, graph_loader, p4_graph
from graph_tiger.measures import algebraic_connectivity, largest_connected_component, run_measure


def get_simulation_params():
    return {
        'runs': 1,
        'steps': 1,
        'seed': 1,
        'plot_transition': False,
        'gif_animation': False
    }


def test_netshield_preserves_node_labels():
    graph = nx.path_graph(['left', 'middle', 'right'])

    nodes = get_node_ns(graph, k=1)

    assert nodes == ['middle']


def test_attack_and_defense_validate_budgets():
    invalid = [
        lambda: run_attack_method(p4_graph(), method='unknown', k=1),
        lambda: run_attack_method(p4_graph(), method='id_node', k=-1),
        lambda: run_attack_method(p4_graph(), method='id_node', k=5),
        lambda: run_defense_method(p4_graph(), method='unknown', k=1),
        lambda: run_defense_method(p4_graph(), method='id_node', k=-1),
        lambda: run_defense_method(p4_graph(), method='id_node', k=5)
    ]

    for call in invalid:
        raised = False
        try:
            call()
        except ValueError:
            raised = True

        assert raised


def test_random_methods_use_local_seed():
    graph = nx.path_graph(10)

    first_attack = run_attack_method(graph, method='rnd_node', k=3, seed=1)
    repeated_attack = run_attack_method(graph, method='rnd_node', k=3, seed=1)
    second_attack = run_attack_method(graph, method='rnd_node', k=3, seed=2)

    first_defense = run_defense_method(graph, method='add_edge_random', k=3, seed=1)
    repeated_defense = run_defense_method(graph, method='add_edge_random', k=3, seed=1)
    second_defense = run_defense_method(graph, method='add_edge_random', k=3, seed=2)

    assert first_attack == repeated_attack
    assert first_attack != second_attack
    assert first_defense['added'] == repeated_defense['added']
    assert first_defense['added'] != second_defense['added']


def test_random_edge_addition_rejects_complete_graph():
    raised = False
    try:
        run_defense_method(nx.complete_graph(4), method='add_edge_random', k=1, seed=1)
    except ValueError:
        raised = True

    assert raised

def test_defense_removes_unprotected_attacked_nodes():
    params = get_simulation_params()
    params.update({
        'k_a': 2,
        'attack': 'id_node',
        'k_d': 1,
        'defense': 'id_node'
    })

    graph = p4_graph()
    df = Defense(graph, **params)

    expected = set(graph.nodes).difference(set(df.attacked).difference(df.protected))

    assert set(df.graph_.nodes) == expected
    assert set(graph.nodes) == {0, 1, 2, 3}


def test_defense_uses_requested_budget():
    params = get_simulation_params()
    params.update({
        'k_a': 0,
        'attack': None,
        'k_d': 2,
        'defense': 'id_node'
    })

    df = Defense(p4_graph(), **params)

    assert len(df.protected) == params['k_d']


def test_defense_edge_attack_removes_edges():
    params = get_simulation_params()
    params.update({
        'k_a': 1,
        'attack': 'id_edge',
        'k_d': 0,
        'defense': None
    })

    df = Defense(p4_graph(), **params)

    for u, v in df.attacked:
        assert not df.graph_.has_edge(u, v)


def test_defense_rewiring_applies_to_simulation_copy():
    params = get_simulation_params()
    params.update({
        'k_a': 0,
        'attack': None,
        'k_d': 1,
        'defense': 'rewire_edge_random'
    })

    graph = nx.cycle_graph(6)
    edges = set(graph.edges)
    df = Defense(graph, **params)
    df.run_single_sim()

    for u, v in df.protected['added']:
        assert df.graph_.has_edge(u, v)

    for u, v in df.protected['removed']:
        assert not df.graph_.has_edge(u, v)

    assert set(graph.edges) == edges


def test_preferential_addition_uses_nonedges():
    graph = p4_graph()
    info = run_defense_method(graph, method='add_edge_preferential', k=2, seed=1)

    assert len(info['added']) == 2

    for u, v in info['added']:
        assert u != v
        assert not graph.has_edge(u, v)


def test_preferential_rewiring_avoids_self_loops():
    graph = nx.star_graph(3)
    info = run_defense_method(graph, method='rewire_edge_preferential', k=1, seed=1)

    for u, v in info['added']:
        assert u != v


def test_sis_state_transition():
    params = get_simulation_params()
    params.update({
        'model': 'SIS',
        'b': 1,
        'd': 0,
        'c': 0,
        'diffusion': None,
        'method': None,
        'k': 0
    })

    ds = Diffusion(nx.path_graph(3), **params)
    ds.infected = {1}
    ds.run_single_sim()

    assert ds.infected == {0, 1, 2}
    assert ds.vaccinated == set()


def test_sir_state_transition():
    params = get_simulation_params()
    params.update({
        'model': 'SIR',
        'b': 1,
        'd': 1,
        'c': 0,
        'diffusion': None,
        'method': None,
        'k': 0
    })

    ds = Diffusion(nx.path_graph(3), **params)
    ds.infected = {1}
    ds.run_single_sim()

    assert ds.infected == {0, 2}
    assert ds.vaccinated == {1}


def test_diffusion_validates_parameters():
    invalid = [
        {'model': 'SEIR'},
        {'b': -0.1},
        {'b': 1.1},
        {'d': -0.1},
        {'d': 1.1},
        {'c': -0.1},
        {'c': 1.1}
    ]

    for update in invalid:
        params = get_simulation_params()
        params.update({
            'model': 'SIS',
            'b': 0.1,
            'd': 0.1,
            'c': 0.1,
            'diffusion': None,
            'method': None,
            'k': 0
        })
        params.update(update)

        raised = False
        try:
            Diffusion(nx.path_graph(10), **params)
        except ValueError:
            raised = True

        assert raised


def test_diffusion_uses_requested_seed():
    params = get_simulation_params()
    params.update({
        'model': 'SIS',
        'b': 0.1,
        'd': 0.1,
        'c': 0.25,
        'diffusion': None,
        'method': None,
        'k': 0
    })

    first = Diffusion(nx.path_graph(20), **params)
    repeated = Diffusion(nx.path_graph(20), **params)

    params['seed'] = 2
    second = Diffusion(nx.path_graph(20), **params)

    assert first.infected == repeated.infected
    assert first.infected != second.infected


def test_diffusion_timeline_length():
    params = get_simulation_params()
    params.update({
        'steps': 3,
        'model': 'SIS',
        'b': 0.1,
        'd': 0.1,
        'c': 0.25,
        'diffusion': None,
        'method': None,
        'k': 0
    })

    ds = Diffusion(nx.path_graph(20), **params)
    results = ds.run_single_sim()

    assert len(results) == params['steps'] + 1

def test_cascading_uses_requested_seed():
    params = get_simulation_params()
    params.update({
        'l': 0.8,
        'r': 0.2,
        'k_a': 0,
        'attack': None,
        'k_d': 0,
        'defense': None,
        'robust_measure': 'largest_connected_component'
    })

    params['model'] = 'legacy_redistribution'
    first = Cascading(nx.path_graph(5), **params)
    repeated = Cascading(nx.path_graph(5), **params)

    params['seed'] = 2
    second = Cascading(nx.path_graph(5), **params)

    assert dict(first.load) == dict(repeated.load)
    assert dict(first.load) != dict(second.load)


def test_cascading_edge_attack_preserves_input_graph():
    params = get_simulation_params()
    params.update({
        'l': 0.8,
        'r': 0.2,
        'k_a': 1,
        'attack': 'id_edge',
        'k_d': 0,
        'defense': None,
        'robust_measure': 'largest_connected_component'
    })

    graph = nx.cycle_graph(5)
    edges = set(graph.edges)
    cf = Cascading(graph, **params)

    assert set(graph.edges) == edges
    assert all(n in graph.nodes for n in cf.failed)

    cf.run_single_sim()


def test_motter_lai_initial_load_and_capacity():
    params = get_simulation_params()
    params.update({
        'model': 'motter_lai',
        'l': 0.8,
        'r': 0.2,
        'k_a': 0,
        'attack': None,
        'k_d': 0,
        'defense': None,
        'robust_measure': 'largest_connected_component'
    })

    graph = nx.path_graph(4)
    expected = nx.betweenness_centrality(graph, normalized=False, endpoints=False)
    cf = Cascading(graph, **params)

    for n in graph.nodes:
        assert cf.capacity_og[n] == expected[n]
        assert cf.capacity[n] == (1 + params['r']) * expected[n]


def test_legacy_cascading_redistributes_failed_load_once():
    params = get_simulation_params()
    params.update({
        'steps': 2,
        'model': 'legacy_redistribution',
        'l': 0.8,
        'r': 0.2,
        'k_a': 0,
        'attack': None,
        'k_d': 0,
        'defense': None,
        'robust_measure': 'largest_connected_component'
    })

    cf = Cascading(nx.path_graph(3), **params)
    cf.load = {0: 0, 1: 2, 2: 0}
    cf.capacity = {0: 2, 1: 1, 2: 2}
    cf.failed = {1}
    cf.sim_info = {}
    cf.run_single_sim()

    assert cf.load[0] == 1
    assert cf.load[2] == 1


def test_legacy_cascading_redistributes_to_functioning_neighbors():
    params = get_simulation_params()
    params.update({
        'model': 'legacy_redistribution',
        'l': 0.8,
        'r': 0.2,
        'k_a': 0,
        'attack': None,
        'k_d': 0,
        'defense': None,
        'robust_measure': 'largest_connected_component'
    })

    cf = Cascading(nx.path_graph(3), **params)
    cf.load = {0: 0, 1: 2, 2: 0}
    cf.capacity = {0: 1, 1: 1, 2: 3}
    cf.failed = {0, 1}
    cf.sim_info = {}
    cf.run_single_sim()

    assert cf.load[2] == 2


def test_cascading_timeline_length():
    params = get_simulation_params()
    params.update({
        'steps': 3,
        'l': 0.8,
        'r': 0.2,
        'k_a': 0,
        'attack': None,
        'k_d': 0,
        'defense': None,
        'robust_measure': 'largest_connected_component'
    })

    cf = Cascading(nx.path_graph(4), **params)
    results = cf.run_single_sim()

    assert len(results) == params['steps'] + 1


def test_standard_average_vertex_betweenness():
    value = run_measure(p4_graph(), 'average_vertex_betweenness')

    assert value == 1


def test_large_path_exact_laplacian_measures():
    for n in [99, 100, 101]:
        graph = nx.path_graph(n)
        expected_resistance = (n ** 3 - n) / 6

        num_trees = run_measure(graph, 'number_spanning_trees')
        resistance = run_measure(graph, 'effective_resistance')

        assert num_trees == 1
        assert resistance == expected_resistance


def test_measure_direct_calls_and_empty_graph():
    assert algebraic_connectivity(p4_graph()) == 0.59
    assert largest_connected_component(nx.Graph()) == 0


def test_unknown_measure_raises_value_error():
    raised = False
    try:
        run_measure(p4_graph(), 'unknown_measure')
    except ValueError:
        raised = True

    assert raised

def test_natural_connectivity_uses_graph_order():
    graph = nx.complete_graph(100)
    expected = round(np.log((np.exp(99) + 99 * np.exp(-1)) / 100), 2)

    value = run_measure(graph, 'natural_connectivity', k=1)

    assert value == expected


def test_graph_options_are_json_serializable():
    options = json.loads(get_graph_options())

    assert set(options) == {'models', 'datasets', 'custom'}
    assert 'karate' in options['datasets']


def test_graph_loader_loads_karate_offline():
    graph = graph_loader('karate')

    assert len(graph) == 34
    assert len(graph.edges) == 78


def main():
    test_netshield_preserves_node_labels()
    test_attack_and_defense_validate_budgets()
    test_random_methods_use_local_seed()
    test_random_edge_addition_rejects_complete_graph()
    test_defense_removes_unprotected_attacked_nodes()
    test_defense_uses_requested_budget()
    test_defense_edge_attack_removes_edges()
    test_defense_rewiring_applies_to_simulation_copy()
    test_preferential_addition_uses_nonedges()
    test_preferential_rewiring_avoids_self_loops()
    test_sis_state_transition()
    test_sir_state_transition()
    test_diffusion_validates_parameters()
    test_diffusion_uses_requested_seed()
    test_diffusion_timeline_length()
    test_cascading_uses_requested_seed()
    test_cascading_edge_attack_preserves_input_graph()
    test_motter_lai_initial_load_and_capacity()
    test_legacy_cascading_redistributes_failed_load_once()
    test_legacy_cascading_redistributes_to_functioning_neighbors()
    test_cascading_timeline_length()
    test_standard_average_vertex_betweenness()
    test_large_path_exact_laplacian_measures()
    test_measure_direct_calls_and_empty_graph()
    test_unknown_measure_raises_value_error()
    test_natural_connectivity_uses_graph_order()
    test_graph_options_are_json_serializable()
    test_graph_loader_loads_karate_offline()


if __name__ == '__main__':
    main()
