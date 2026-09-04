import json
import time

import networkx as nx
import numpy as np

import graph_tiger.measures as tiger_measures

from graph_tiger.attacks import Attack, get_node_ns, run_attack_method
from graph_tiger.cascading import Cascading
from graph_tiger.defenses import Defense, run_defense_method
from graph_tiger.diffusion import Diffusion
from graph_tiger.graphs import get_graph_options, get_graph_urls, graph_loader, p4_graph
from graph_tiger.measures import algebraic_connectivity, largest_connected_component, run_measure
from graph_tiger.simulations import Simulation


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



def test_simulation_resets_advance_reproducible_random_streams():
    params = get_simulation_params()
    params.update({
        'steps': 3,
        'attack': 'rnd_node',
        'defense': None,
        'k_d': 0,
        'robust_measure': 'largest_connected_component'
    })

    first_attack = Attack(nx.path_graph(20), **params)
    first_attack_sample = first_attack.attacked.copy()
    first_attack.reset_simulation()
    second_attack_sample = first_attack.attacked.copy()

    repeated_attack = Attack(nx.path_graph(20), **params)
    repeated_attack_sample = repeated_attack.attacked.copy()
    repeated_attack.reset_simulation()
    repeated_second_attack_sample = repeated_attack.attacked.copy()

    assert first_attack_sample == repeated_attack_sample
    assert second_attack_sample == repeated_second_attack_sample
    assert first_attack_sample != second_attack_sample

    params.update({
        'model': 'SIS',
        'b': 0.1,
        'd': 0.1,
        'c': 0.25,
        'diffusion': None,
        'method': None,
        'k': 0
    })

    first_diffusion = Diffusion(nx.path_graph(20), **params)
    first_infected = first_diffusion.infected.copy()
    first_diffusion.reset_simulation()
    second_infected = first_diffusion.infected.copy()

    repeated_diffusion = Diffusion(nx.path_graph(20), **params)
    repeated_first_infected = repeated_diffusion.infected.copy()
    repeated_diffusion.reset_simulation()
    repeated_second_infected = repeated_diffusion.infected.copy()

    assert first_infected == repeated_first_infected
    assert second_infected == repeated_second_infected
    assert first_infected != second_infected

def test_random_edge_addition_rejects_complete_graph():
    raised = False
    try:
        run_defense_method(nx.complete_graph(4), method='add_edge_random', k=1, seed=1)
    except ValueError:
        raised = True

    assert raised


def test_defense_tracks_fully_failed_graph():
    params = get_simulation_params()
    params.update({
        'steps': 1,
        'k_a': 2,
        'attack': 'id_node',
        'k_d': 0,
        'defense': None
    })

    df = Defense(nx.path_graph(2), **params)

    assert df.sim_info[0]['failed'] == 2
    assert df.sim_info[0]['measure'] == 0

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



def test_motter_lai_recomputes_and_fails_synchronously():
    params = get_simulation_params()
    params.update({
        'model': 'motter_lai',
        'r': 0.1,
        'k_a': 0,
        'attack': None,
        'k_d': 0,
        'defense': None
    })

    cf = Cascading(nx.cycle_graph(5), **params)
    cf.failed = {0}

    failed_new = cf.run_motter_lai_step()

    assert failed_new == {2, 3}


def test_crucitti_reference_transition():
    params = get_simulation_params()
    params.update({
        'steps': 1,
        'model': 'crucitti',
        'l': 0.8,
        'r': 0.2,
        'k_a': 1,
        'attack': 'id_node',
        'k_d': 0,
        'defense': None
    })

    graph = nx.cycle_graph(4)
    edges = set(graph.edges)
    cf = Cascading(graph, **params)
    results = cf.run_single_sim()

    functioning = set(graph.nodes).difference(cf.failed)
    graph_functioning = cf.graph.subgraph(functioning)
    center = max(graph_functioning.nodes, key=graph_functioning.degree)

    assert len(cf.failed) == 1
    assert cf.sim_info[1]['failed'] == 1
    assert cf.sim_info[1]['overloaded'] == {center}
    assert np.isclose(results[0], 5 / 6)
    assert np.isclose(results[1], 1 / 2)

    for u, v in graph_functioning.edges:
        assert np.isclose(cf.graph[u][v]['efficiency'], 0.6)

    assert set(graph.edges) == edges
    assert all('efficiency' not in data for _, _, data in graph.edges(data=True))


def test_crucitti_restores_edge_efficiency():
    params = get_simulation_params()
    params.update({
        'model': 'crucitti',
        'l': 0.8,
        'r': 0.2,
        'k_a': 0,
        'attack': None,
        'k_d': 0,
        'defense': None
    })

    cf = Cascading(nx.path_graph(3), **params)
    cf.load = {0: 0, 1: 2, 2: 0}
    cf.capacity = {0: 1, 1: 1, 2: 1}

    changed = cf.run_crucitti_step()

    assert changed
    assert cf.failed == set()
    assert all(cf.graph[u][v]['efficiency'] == 0.5 for u, v in cf.graph.edges)

    changed = cf.run_crucitti_step()

    assert changed
    assert all(cf.graph[u][v]['efficiency'] == 1 for u, v in cf.graph.edges)
    assert not cf.run_crucitti_step()


def test_crucitti_uses_weighted_efficient_paths():
    params = get_simulation_params()
    params.update({
        'model': 'crucitti',
        'l': 0.8,
        'r': 0.2,
        'k_a': 0,
        'attack': None,
        'k_d': 0,
        'defense': None
    })

    cf = Cascading(nx.complete_graph(3), **params)
    cf.graph[0][2]['efficiency'] = 0.25

    graph = cf.get_efficiency_graph(cf.graph)
    load = cf.get_load(graph, weight='distance')
    efficiency = cf.get_efficiency(cf.graph)

    assert load[1] == 1
    assert np.isclose(efficiency, 5 / 6)


def test_crucitti_uses_most_congested_endpoint():
    params = get_simulation_params()
    params.update({
        'model': 'crucitti',
        'r': 0.2,
        'k_a': 0,
        'attack': None,
        'k_d': 0,
        'defense': None
    })

    cf = Cascading(nx.path_graph(2), **params)
    cf.load = {0: 2, 1: 4}
    cf.capacity = {0: 1, 1: 1}

    cf.run_crucitti_step()

    assert cf.graph[0][1]['efficiency'] == 0.25


def test_crucitti_validates_parameters():
    invalid = [
        {'r': 0, 'attack': None, 'k_a': 0},
        {'r': 0.2, 'attack': 'id_edge', 'k_a': 1}
    ]

    for update in invalid:
        params = get_simulation_params()
        params.update({
            'model': 'crucitti',
            'l': 0.8,
            'k_d': 0,
            'defense': None
        })
        params.update(update)

        raised = False
        try:
            Cascading(nx.cycle_graph(4), **params)
        except ValueError:
            raised = True

        assert raised

def test_local_load_sharing_initial_load_and_capacity():
    params = get_simulation_params()
    params.update({
        'model': 'local_load_sharing',
        'beta': 0,
        'r': 0.5,
        'k_a': 0,
        'attack': None,
        'k_d': 0,
        'defense': None
    })

    graph = nx.path_graph(3)
    cf = Cascading(graph, **params)

    assert cf.load == dict(graph.degree())
    assert cf.capacity == {0: 1.5, 1: 3, 2: 1.5}


def test_local_load_sharing_equal_redistribution():
    params = get_simulation_params()
    params.update({
        'model': 'local_load_sharing',
        'beta': 0,
        'r': 0.2,
        'k_a': 0,
        'attack': None,
        'k_d': 0,
        'defense': None
    })

    cf = Cascading(nx.star_graph(3), **params)
    cf.load = {0: 12, 1: 0, 2: 0, 3: 0}
    cf.capacity = {0: 0, 1: 10, 2: 10, 3: 10}
    cf.failed = {0}
    cf.processed = set()

    failed_new = cf.run_local_load_sharing_step()

    assert failed_new == set()
    assert cf.load == {0: 0, 1: 4, 2: 4, 3: 4}
    assert cf.shed_load == 0


def test_local_load_sharing_degree_preference():
    params = get_simulation_params()
    params.update({
        'model': 'local_load_sharing',
        'beta': 1,
        'r': 0.2,
        'k_a': 0,
        'attack': None,
        'k_d': 0,
        'defense': None
    })

    graph = nx.Graph([(0, 1), (0, 2), (0, 3), (2, 4), (3, 5), (3, 6)])
    cf = Cascading(graph, **params)
    cf.load = {n: 0 for n in graph.nodes}
    cf.load[0] = 12
    cf.capacity = {n: 20 for n in graph.nodes}
    cf.failed = {0}
    cf.processed = set()

    cf.run_local_load_sharing_step()

    assert cf.load[1] == 2
    assert cf.load[2] == 4
    assert cf.load[3] == 6


def test_local_load_sharing_updates_synchronously():
    params = get_simulation_params()
    params.update({
        'model': 'local_load_sharing',
        'beta': 0,
        'r': 0.2,
        'k_a': 0,
        'attack': None,
        'k_d': 0,
        'defense': None
    })

    graph = nx.Graph([(0, 2), (1, 2)])
    cf = Cascading(graph, **params)
    cf.load = {0: 4, 1: 6, 2: 0}
    cf.capacity = {0: 0, 1: 0, 2: 9}
    cf.failed = {0, 1}
    cf.processed = set()

    failed_new = cf.run_local_load_sharing_step()

    assert failed_new == {2}
    assert cf.load == {0: 0, 1: 0, 2: 10}


def test_local_load_sharing_records_shed_load():
    params = get_simulation_params()
    params.update({
        'model': 'local_load_sharing',
        'beta': 0,
        'r': 0.2,
        'k_a': 0,
        'attack': None,
        'k_d': 0,
        'defense': None
    })

    graph = nx.empty_graph(1)
    cf = Cascading(graph, **params)
    cf.load = {0: 5}
    cf.capacity = {0: 0}
    cf.failed = {0}
    cf.processed = set()

    cf.run_local_load_sharing_step()

    assert cf.load[0] == 0
    assert cf.shed_load == 5


def test_local_load_sharing_validates_parameters():
    params = get_simulation_params()
    params.update({
        'model': 'local_load_sharing',
        'beta': -1,
        'r': 0.2,
        'k_a': 0,
        'attack': None,
        'k_d': 0,
        'defense': None
    })

    raised = False
    try:
        Cascading(nx.path_graph(3), **params)
    except ValueError:
        raised = True

    assert raised


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



def test_measure_timeout_uses_standard_library():
    def slow_measure(graph, **kwargs):
        time.sleep(0.1)
        return len(graph)

    tiger_measures.measures['slow_measure'] = slow_measure

    try:
        value = tiger_measures.run_measure(nx.path_graph(3), 'slow_measure', timeout=0.01)
    finally:
        del tiger_measures.measures['slow_measure']

    assert value is None

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


def test_graph_coordinates_preserve_node_labels():
    graph = nx.Graph()
    graph.add_node('left', pos=[0, 0])
    graph.add_node('right', pos=[1, 0])
    graph.add_edge('left', 'right')

    simulation = Simulation(graph, runs=1, steps=1)
    node_pos, _ = simulation.get_graph_coordinates()

    assert set(node_pos) == {'left', 'right'}

def test_graph_dataset_sources_match_names():
    urls = get_graph_urls()

    assert urls['ca_hep_th'][0].endswith('datasets/ca-HepTh.txt')
    assert urls['cit_hep_th'][0].endswith('datasets/cit-HepTh.txt')


def test_watts_strogatz_uses_standard_generator():
    expected = nx.watts_strogatz_graph(n=20, k=4, p=0.2, seed=1)
    graph = graph_loader('WS', n=20, m=4, p=0.2, seed=1)

    assert set(graph.edges) == set(expected.edges)


def test_unknown_graph_raises_value_error():
    raised = False
    try:
        graph_loader('unknown_graph')
    except ValueError:
        raised = True

    assert raised

def test_graph_loader_loads_karate_offline():
    graph = graph_loader('karate')

    assert len(graph) == 34
    assert len(graph.edges) == 78


def main():
    test_netshield_preserves_node_labels()
    test_attack_and_defense_validate_budgets()
    test_random_methods_use_local_seed()
    test_simulation_resets_advance_reproducible_random_streams()
    test_random_edge_addition_rejects_complete_graph()
    test_defense_tracks_fully_failed_graph()
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
    test_motter_lai_recomputes_and_fails_synchronously()
    test_crucitti_reference_transition()
    test_crucitti_restores_edge_efficiency()
    test_crucitti_uses_weighted_efficient_paths()
    test_crucitti_uses_most_congested_endpoint()
    test_crucitti_validates_parameters()
    test_local_load_sharing_initial_load_and_capacity()
    test_local_load_sharing_equal_redistribution()
    test_local_load_sharing_degree_preference()
    test_local_load_sharing_updates_synchronously()
    test_local_load_sharing_records_shed_load()
    test_local_load_sharing_validates_parameters()
    test_legacy_cascading_redistributes_failed_load_once()
    test_legacy_cascading_redistributes_to_functioning_neighbors()
    test_cascading_timeline_length()
    test_measure_timeout_uses_standard_library()
    test_standard_average_vertex_betweenness()
    test_large_path_exact_laplacian_measures()
    test_measure_direct_calls_and_empty_graph()
    test_unknown_measure_raises_value_error()
    test_natural_connectivity_uses_graph_order()
    test_graph_options_are_json_serializable()
    test_graph_coordinates_preserve_node_labels()
    test_graph_dataset_sources_match_names()
    test_watts_strogatz_uses_standard_generator()
    test_unknown_graph_raises_value_error()
    test_graph_loader_loads_karate_offline()


if __name__ == '__main__':
    main()
